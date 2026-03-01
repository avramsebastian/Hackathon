#!/usr/bin/env python3
"""
sim/world.py
============
Entity-based intersection world.

This module manages a flat list of :class:`Car` entities.  Every car can
be used as the ego vehicle for ML inference, and every car can publish
its own V2X payload.  The :class:`World` class owns the physics loop,
stop-sign enforcement, optional collision guard, and spawn logic.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sim.traffic_policy import (
    SafetyPolicy,
    danger_score,
    pair_safe_distance_m,
)
from sim.network import IntersectionNode, RoadNetwork, default_network

log = logging.getLogger("world")

_ML_DIRECTIONS: Tuple[str, ...] = ("FORWARD", "LEFT", "RIGHT")
_APPROACHES: Tuple[str, ...] = ("W", "N", "E", "S")

# Semaphore phase names
_PHASE_GREEN  = "GREEN"
_PHASE_YELLOW = "YELLOW"
_PHASE_RED    = "RED"

# Which approaches share a green phase
_AXIS_APPROACHES: Dict[str, Tuple[str, ...]] = {
    "EW": ("E", "W"),
    "NS": ("N", "S"),
}
_OTHER_AXIS: Dict[str, str] = {"EW": "NS", "NS": "EW"}

# ── Turn waypoints: (approach, ml_direction) → list of (x, y) waypoints ──────
# Cars follow these waypoints through the intersection for smooth turns.
# Waypoints are sampled along circular arcs so heading changes gradually.
_L: float = 7.0   # Must stay in sync with SafetyPolicy.lane_offset_m

_ARC_RIGHT_R: float = 2.0   # turn radius for right turns (metres)
_ARC_LEFT_R:  float = 10.0  # turn radius for left turns  (metres)
_ARC_N: int = 8             # sample points per arc


def _arc(cx: float, cy: float, r: float,
         t0: float, t1: float, n: int) -> List[Tuple[float, float]]:
    """Return *n* evenly-spaced points on a circular arc."""
    return [(cx + r * math.cos(t0 + (t1 - t0) * i / (n - 1)),
             cy + r * math.sin(t0 + (t1 - t0) * i / (n - 1)))
            for i in range(n)]


_TURN_WAYPOINTS: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
    # ── RIGHT turns (tight 90° CW arcs, r = _ARC_RIGHT_R) ────────────────
    # W RIGHT: east → south  (centre at (-L-r, -L-r))
    ("W", "RIGHT"): _arc(-_L - _ARC_RIGHT_R, -_L - _ARC_RIGHT_R,
                         _ARC_RIGHT_R, math.pi / 2, 0, _ARC_N)
                     + [(-_L, -_L - 15)],
    # E RIGHT: west → north  (centre at (L+r, L+r))
    ("E", "RIGHT"): _arc(_L + _ARC_RIGHT_R, _L + _ARC_RIGHT_R,
                         _ARC_RIGHT_R, -math.pi / 2, -math.pi, _ARC_N)
                     + [(_L, _L + 15)],
    # N RIGHT: south → west  (centre at (-L-r, L+r))
    ("N", "RIGHT"): _arc(-_L - _ARC_RIGHT_R, _L + _ARC_RIGHT_R,
                         _ARC_RIGHT_R, 0, -math.pi / 2, _ARC_N)
                     + [(-_L - 15, _L)],
    # S RIGHT: north → east  (centre at (L+r, -L-r))
    ("S", "RIGHT"): _arc(_L + _ARC_RIGHT_R, -_L - _ARC_RIGHT_R,
                         _ARC_RIGHT_R, math.pi, math.pi / 2, _ARC_N)
                     + [(_L + 15, -_L)],

    # ── LEFT turns (wide 90° CCW arcs, r = _ARC_LEFT_R) ──────────────────
    # W LEFT: east → north  (centre at (L-r, -L+r))
    ("W", "LEFT"): _arc(_L - _ARC_LEFT_R, -_L + _ARC_LEFT_R,
                        _ARC_LEFT_R, -math.pi / 2, 0, _ARC_N)
                    + [(_L, 15.0)],
    # E LEFT: west → south  (centre at (-L+r, L-r))
    ("E", "LEFT"): _arc(-_L + _ARC_LEFT_R, _L - _ARC_LEFT_R,
                        _ARC_LEFT_R, math.pi / 2, math.pi, _ARC_N)
                    + [(-_L, -15.0)],
    # N LEFT: south → east  (centre at (-L+r, -L+r))
    ("N", "LEFT"): _arc(-_L + _ARC_LEFT_R, -_L + _ARC_LEFT_R,
                        _ARC_LEFT_R, math.pi, 3 * math.pi / 2, _ARC_N)
                    + [(15.0, -_L)],
    # S LEFT: north → west  (centre at (L-r, L-r))
    ("S", "LEFT"): _arc(_L - _ARC_LEFT_R, _L - _ARC_LEFT_R,
                        _ARC_LEFT_R, 0, math.pi / 2, _ARC_N)
                    + [(-15.0, _L)],
}

# How far from centre (along travel axis) the car begins following waypoints.
# Right turns start earlier to accommodate the arc onset; lefts closer in.
_TURN_TRIGGER: Dict[str, float] = {
    "RIGHT": _L + _ARC_RIGHT_R + 1.0,  # ~10 m — arc entry + 1 m run-in
    "LEFT":  4.0,                       # ~4 m — closer to the centre
}


@dataclass
class Car:
    """A standalone vehicle entity.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. ``CAR_000``).
    x, y : float
        World-space position (metres from intersection centre).
    speed : float
        Current speed in km/h.
    ml_direction : str
        Intended manoeuvre: ``FORWARD``, ``LEFT``, or ``RIGHT``.
    approach : str
        Approach arm the car entered from: ``W``, ``N``, ``E``, or ``S``.
    vx, vy : float
        Unit velocity vector (+1 / −1 / 0).
    stop_wait_s : float
        Seconds the car has been stopped at a STOP sign.
    stop_completed : bool
        True once the mandatory STOP wait has been fulfilled.
    """

    id: str
    x: float
    y: float
    speed: float            # km/h
    ml_direction: str       # FORWARD | LEFT | RIGHT  (for ML model)
    approach: str           # W | N | E | S            (which side car enters from)
    role: str = "civilian"
    cruise_speed: float = 0.0
    speed_limit_kmh: float = 42.0
    wait_s: float = 0.0
    vx: float = 0.0        # unit velocity component  (+1 / -1 / 0)
    vy: float = 0.0
    current_int_id: str = ""   # intersection this car is approaching / in
    passed: bool = field(default=False, repr=False)
    stopped: bool = field(default=False, repr=False)
    stop_wait_s: float = field(default=0.0, repr=False)
    stop_completed: bool = field(default=False, repr=False)
    has_turned: bool = field(default=False, repr=False)
    _turn_waypoints: List[Tuple[float, float]] = field(
        default_factory=list, repr=False,
    )
    """Remaining waypoints the car must pass through during a turn."""

    def __post_init__(self) -> None:
        if self.cruise_speed <= 0.0:
            self.cruise_speed = self.speed

    @property
    def is_turning(self) -> bool:
        """True while the car is actively following turn waypoints."""
        return len(self._turn_waypoints) > 0

    # ── movement ──────────────────────────────────────────────────────────
    def move(self, dt: float) -> None:
        """Advance the car — follow turn waypoints if active, else straight."""
        if self.stopped:
            return
        speed_mps = self.speed / 3.6
        if self._turn_waypoints:
            self._move_along_waypoints(speed_mps, dt)
        else:
            self.x += self.vx * speed_mps * dt
            self.y += self.vy * speed_mps * dt

    def _move_along_waypoints(self, speed_mps: float, dt: float) -> None:
        """Steer toward the next turn waypoint, consuming it when reached."""
        remaining = speed_mps * dt
        while remaining > 1e-6 and self._turn_waypoints:
            tx, ty = self._turn_waypoints[0]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                self._turn_waypoints.pop(0)
                continue
            if remaining >= dist:
                # Reach this waypoint and continue to the next.
                self.x = tx
                self.y = ty
                remaining -= dist
                self._turn_waypoints.pop(0)
            else:
                # Move toward waypoint but don't reach it yet.
                frac = remaining / dist
                self.x += dx * frac
                self.y += dy * frac
                remaining = 0.0
        # Update vx/vy to reflect current heading (for collision guard, UI, etc.)
        if self._turn_waypoints:
            tx, ty = self._turn_waypoints[0]
            dx = tx - self.x
            dy = ty - self.y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                self.vx = dx / d
                self.vy = dy / d
        elif self.has_turned:
            # Finished all waypoints — snap vx/vy to the nearest cardinal.
            if abs(self.vx) >= abs(self.vy):
                self.vx = 1.0 if self.vx > 0 else -1.0
                self.vy = 0.0
            else:
                self.vy = 1.0 if self.vy > 0 else -1.0
                self.vx = 0.0

    # ── helpers ───────────────────────────────────────────────────────────
    def has_passed(self, threshold: float = 10.0,
                   cx: float = 0.0, cy: float = 0.0) -> bool:
        """True once the car is *threshold* metres past *cx, cy*
        along its travel axis.
        """
        rx, ry = self.x - cx, self.y - cy
        if self.vx > 0 and rx > threshold:
            return True
        if self.vx < 0 and rx < -threshold:
            return True
        if self.vy > 0 and ry > threshold:
            return True
        if self.vy < 0 and ry < -threshold:
            return True
        return False

    def as_dict(self) -> dict:
        """Serialisable mapping of ``id``, ``x``, ``y``, ``speed``,
        ``direction`` (ML) and ``role``.
        """
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "direction": self.ml_direction,
            "role": self.role,
        }

    def ml_payload(self, sign: str, others: Sequence["Car"],
                   traffic_light: str = "NONE") -> Dict[str, Any]:
        """Build the JSON-compatible dict expected by
        :func:`ml.comunication.Inference.fa_inferenta_din_json`.
        """
        return {
            "my_car": self.as_dict(),
            "sign": sign,
            "traffic_light": traffic_light,
            "traffic": [car.as_dict() for car in others if car.id != self.id],
        }

    def state_payload(self, sign: str, others: Sequence["Car"]) -> Dict[str, Any]:
        """
        Build a V2X *state-only* payload (no decision).

        This is what a real car would broadcast on the V2V channel:
        its own position/speed plus the sign it sees and neighbours
        it detects locally.  Decisions travel on a separate topic.
        """
        return {
            "position": self.as_dict(),
            "sign": sign,
            "traffic": [car.as_dict() for car in others if car.id != self.id],
        }

    def v2x_payload(
        self,
        sign: str,
        others: Sequence["Car"],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build V2X payload emitted by this car's communication unit."""
        payload = dict(decision)
        payload.update(
            {
                "position": self.as_dict(),
                "traffic": [car.as_dict() for car in others if car.id != self.id],
                "sign": sign,
            }
        )
        return payload


class World:
    """Entity-based multi-intersection scenario.

    Cars spawn on the terminal arms of a :class:`~sim.network.RoadNetwork`
    and drive through one or more intersections.  Each intersection has its
    own semaphore or sign configuration.

    Parameters
    ----------
    num_cars : int
        Number of vehicles to spawn.
    seed : int or None
        Random seed for reproducibility.
    policy : SafetyPolicy or None
        Tunable constants; uses defaults when *None*.
    priority_axis : str
        ``'EW'`` or ``'NS'`` — default sign layout (used when creating
        the default network).
    network : RoadNetwork or None
        The road layout.  Uses :func:`default_network` when *None*.
    """

    def __init__(
        self,
        num_cars: int = 6,
        seed: Optional[int] = None,
        policy: Optional[SafetyPolicy] = None,
        priority_axis: str = "EW",
        network: Optional[RoadNetwork] = None,
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.priority_axis = self._normalize_priority_axis(priority_axis)
        self.network = network or default_network()
        # Per-intersection sign tables
        self._signs: Dict[str, Dict[str, str]] = {}
        for node in self.network.intersections.values():
            self._signs[node.id] = self._build_signs_by_approach(node.priority_axis)
        # Legacy single-intersection shims
        self.signs_by_approach: Dict[str, str] = self._build_signs_by_approach(self.priority_axis)
        self.current_sign: str = "YIELD"
        self.num_cars = max(1, int(num_cars))
        self._rng = random.Random(seed)
        self.cars: List[Car] = []
        self.safety_interventions: int = 0
        self.collision_resolutions: int = 0
        self.green_approach: str = "W"
        self._green_ttl_s: float = 0.0
        self._tick_count: int = 0

        # ── Semaphore state (per intersection) ──────────────────────────
        # Stagger timers so each semaphore intersection runs independently.
        sem_offset = 0.0
        for node in self.network.intersections.values():
            if node.has_semaphore:
                node.sem_green_axis = node.priority_axis
                node.sem_phase = _PHASE_GREEN
                # Stagger by half-cycle so intersections are out of phase
                node.sem_timer = self.policy.semaphore_green_s + sem_offset
                sem_offset += self.policy.semaphore_green_s / 2.0
            else:
                # Sign-only intersection: no semaphore state
                node.sem_green_axis = ""
                node.sem_phase = ""
                node.sem_timer = 0.0
        # Legacy single-intersection fields (kept for compatibility)
        self._sem_green_axis: str = "EW"
        self._sem_phase: str = _PHASE_GREEN
        self._sem_timer: float = self.policy.semaphore_green_s

        self._init_cars()

    # ── initialisation / reset ────────────────────────────────────────────

    def _init_cars(self) -> None:
        self.cars = []
        for idx in range(self.num_cars):
            self.cars.append(self._make_random_car(idx))
        self.safety_interventions = 0
        self.collision_resolutions = 0
        self.green_approach = "W"
        self._green_ttl_s = 0.0
        # Reset semaphore state (staggered timers)
        sem_offset = 0.0
        for node in self.network.intersections.values():
            if node.has_semaphore:
                node.sem_green_axis = node.priority_axis
                node.sem_phase = _PHASE_GREEN
                node.sem_timer = self.policy.semaphore_green_s + sem_offset
                sem_offset += self.policy.semaphore_green_s / 2.0
            else:
                node.sem_green_axis = ""
                node.sem_phase = ""
                node.sem_timer = 0.0
        self._sem_green_axis = "EW"
        self._sem_phase = _PHASE_GREEN
        self._sem_timer = self.policy.semaphore_green_s
        self._finished = False

    def reset(self) -> None:
        """Re-initialise all cars so the scenario can be replayed."""
        self._init_cars()

    # ── queries ───────────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        return self._finished

    def all_cars(self) -> List[Car]:
        return list(self.cars)

    def get_signs(self) -> Dict[str, str]:
        return dict(self.signs_by_approach)

    def get_signs_for(self, int_id: str) -> Dict[str, str]:
        """Return the sign table for intersection *int_id*."""
        return dict(self._signs.get(int_id, self.signs_by_approach))

    def sign_for_approach(self, approach: str) -> str:
        return self.signs_by_approach.get(str(approach).upper(), "NO_SIGN")

    def sign_for_car(self, car: Car) -> str:
        table = self._signs.get(car.current_int_id, self.signs_by_approach)
        return table.get(str(car.approach).upper(), "NO_SIGN")

    # ── physics tick ──────────────────────────────────────────────────────

    def update_physics(
        self,
        dt: float = 0.1,
        decisions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Advance every car using ML decisions as the primary control signal.

        Optional world-level heuristic layers (virtual signal / collision guard)
        can be enabled through ``SafetyPolicy`` for backward compatibility.
        """
        decisions = decisions or {}
        self._tick_count += 1
        tick = self._tick_count
        if self.policy.world_signal_scheduler_enabled:
            self._update_virtual_signal(dt)
        if self.policy.semaphore_enabled:
            self._update_semaphore(dt)

        targets = self._build_target_speeds(decisions, dt)

        # ── Debug: log car state before collision guard ───────────────
        if tick % 10 == 1:
            log.debug("=== TICK %d ===", tick)
            for car in self.cars:
                sign = self.sign_for_car(car)
                dist = self._distance_to_stop_line(car)
                dec = decisions.get(car.id, {})
                ml_dec = dec.get("decision", "?")
                node = self.network.intersections.get(car.current_int_id)
                cx = node.cx if node else 0.0
                cy = node.cy if node else 0.0
                rx, ry = car.x - cx, car.y - cy
                log.debug(
                    "  %s  pos=(%.1f,%.1f) v=(%.1f,%.1f) int=%s ctr=(%.0f,%.0f) rel=(%.1f,%.1f) "
                    "app=%s ml_dir=%s spd=%.1f cruise=%.1f "
                    "sign=%s dist_line=%.1f passed=%s turning=%s has_turned=%s "
                    "stopped=%s stop_wait=%.2f stop_done=%s ml=%s target=%.1f",
                    car.id, car.x, car.y, car.vx, car.vy,
                    car.current_int_id, cx, cy, rx, ry,
                    car.approach, car.ml_direction, car.speed, car.cruise_speed,
                    sign, dist, car.passed, car.is_turning, car.has_turned,
                    car.stopped, car.stop_wait_s, car.stop_completed, ml_dec,
                    targets.get(car.id, -1),
                )

        if self.policy.world_collision_guard_enabled:
            interventions = self._apply_collision_guard(targets, dt, tick)
            self.safety_interventions += interventions

        self._apply_speed_targets(targets, dt)

        for car in self.cars:
            car.move(dt)

        self._apply_turns()

        for car in self.cars:
            # Don't mark a car as "passed" while it's still following
            # turn waypoints — the intermediate velocity isn't cardinal
            # and the car hasn't reached the exit lane yet.
            node = self.network.intersections.get(car.current_int_id)
            cx = node.cx if node else 0.0
            cy = node.cy if node else 0.0
            threshold = self.policy.pass_threshold_m
            actually_passed = car.has_passed(threshold, cx, cy)
            if not car.passed and not car.is_turning and actually_passed:
                # Check if car should transition to next intersection
                next_int = self._next_intersection_for(car)
                exit_arm = self._exit_arm(car)
                log.debug(
                    "  TRANSITION CHECK %s: exit_arm=%s next_int=%s",
                    car.id, exit_arm, next_int.id if next_int else None,
                )
                if next_int is not None:
                    # Transition: reset per-intersection state for the new target
                    old_int = car.current_int_id
                    car.current_int_id = next_int.id
                    # Determine the arrival arm from the road connection
                    new_approach = self._arrival_arm(car, next_int)
                    if new_approach:
                        car.approach = new_approach
                    car.has_turned = False
                    car.stop_completed = False
                    car.stop_wait_s = 0.0
                    # Pick a new random direction for the next intersection
                    car.ml_direction = self._rng.choice(_ML_DIRECTIONS)
                    log.debug(
                        "  TRANSITION %s: %s -> %s new_app=%s ml_dir=%s",
                        car.id, old_int, next_int.id, new_approach, car.ml_direction,
                    )
                else:
                    car.passed = True
            # Clear leftover waypoints once a car has passed through so
            # it doesn't stay stuck in "turning" state while coasting.
            if car.passed and car._turn_waypoints:
                car._turn_waypoints.clear()
                if abs(car.vx) >= abs(car.vy):
                    car.vx = 1.0 if car.vx > 0 else -1.0
                    car.vy = 0.0
                else:
                    car.vy = 1.0 if car.vy > 0 else -1.0
                    car.vx = 0.0

        if self.policy.world_overlap_resolver_enabled:
            self.collision_resolutions += self._resolve_overlaps()

        # Mark simulation finished once every car has passed through.
        if all(c.passed for c in self.cars):
            self._finished = True

    # ── Legacy compatibility ──────────────────────────────────────────────

    def get_ml_input(self) -> dict:
        """
        Compatibility shim for older call sites expecting ``my_car + traffic``.
        Uses the first car as ego.
        """
        if not self.cars:
            return {
                "my_car": {"x": 0.0, "y": 0.0, "speed": 0.0, "direction": "FORWARD"},
                "sign": self.current_sign,
                "traffic": [],
            }
        ego = self.cars[0]
        return {
            "my_car": ego.as_dict(),
            "sign": self.sign_for_car(ego),
            "traffic": [car.as_dict() for car in self.cars[1:]],
        }

    # ── spawning ──────────────────────────────────────────────────────────

    def _make_random_car(self, idx: int) -> Car:
        """
        Spawn one car on a random terminal arm of the road network.
        """
        terminal_arms = self.network.terminal_arms()
        int_id, arm = self._rng.choice(terminal_arms)

        speed = self._rng.uniform(60.0, self.policy.speed_limit_kmh + 10.0)
        ml_direction = self._rng.choice(_ML_DIRECTIONS)
        role = self._rng.choices(
            ("civilian", "ambulance", "police"),
            weights=(0.94, 0.03, 0.03),
            k=1,
        )[0]

        if role in ("ambulance", "police"):
            speed *= 1.1

        pos: Optional[Tuple[float, float, float, float]] = None
        for _ in range(self.policy.spawn_max_attempts):
            distance = self._rng.uniform(self.policy.spawn_min_radius_m, self.policy.spawn_max_radius_m)
            candidate = self.network.arm_spawn_position(
                int_id, arm, distance, self.policy.lane_offset_m,
            )
            if self._spawn_is_clear(candidate[0], candidate[1]):
                pos = candidate
                break

            # Try another terminal arm if this lane is crowded.
            int_id, arm = self._rng.choice(terminal_arms)

        if pos is None:
            # Deterministic fallback: cycle through terminal arms.
            int_id, arm = terminal_arms[idx % len(terminal_arms)]
            distance = self.policy.spawn_min_radius_m + idx * 10.0
            pos = self.network.arm_spawn_position(
                int_id, arm, distance, self.policy.lane_offset_m,
            )

        x, y, vx, vy = pos
        return Car(
            id=f"CAR_{idx:03d}",
            x=x,
            y=y,
            speed=speed,
            ml_direction=ml_direction,
            approach=arm,
            role=role,
            cruise_speed=speed,
            speed_limit_kmh=self.policy.speed_limit_kmh,
            vx=vx,
            vy=vy,
            current_int_id=int_id,
        )

    def _spawn_position(self, approach: str, distance: float) -> Tuple[float, float, float, float]:
        """
        Return spawn (x, y) and heading vector (vx, vy) for one approach arm.
        Legacy — used only by old call-sites; new code uses network arms.
        """
        L = self.policy.lane_offset_m
        arm = approach.upper()
        if arm == "W":
            return (-distance, -L, 1.0, 0.0)   # west -> east
        if arm == "E":
            return (distance, L, -1.0, 0.0)    # east -> west
        if arm == "N":
            return (-L, distance, 0.0, -1.0)   # north -> south
        return (L, -distance, 0.0, 1.0)        # south -> north

    # ── turning ────────────────────────────────────────────────────────────

    def _int_center(self, car: Car) -> Tuple[float, float]:
        """Return the centre of the car's current intersection."""
        node = self.network.intersections.get(car.current_int_id)
        return (node.cx, node.cy) if node else (0.0, 0.0)

    @staticmethod
    def _distance_to_center_xy(car: Car, cx: float, cy: float) -> float:
        """Signed distance from *car* to *(cx, cy)* along its travel axis."""
        rx, ry = car.x - cx, car.y - cy
        if car.vx > 0:  return -rx
        if car.vx < 0:  return  rx
        if car.vy < 0:  return  ry
        if car.vy > 0:  return -ry
        return math.hypot(rx, ry)

    def _distance_to_center(self, car: Car) -> float:
        cx, cy = self._int_center(car)
        return self._distance_to_center_xy(car, cx, cy)

    def _apply_turns(self) -> None:
        """Assign turn waypoints to cars entering their turn zone.

        Once assigned, the car's ``move()`` follows the waypoints
        automatically — no position snap is needed.
        """
        for car in self.cars:
            if car.has_turned or car.is_turning or car.ml_direction == "FORWARD":
                continue
            wps = _TURN_WAYPOINTS.get((car.approach, car.ml_direction))
            if wps is None:
                continue
            trigger = _TURN_TRIGGER.get(car.ml_direction, 5.0)
            dist = self._distance_to_center(car)
            if dist > trigger:
                continue
            # Offset waypoints by the intersection centre.
            cx, cy = self._int_center(car)
            car._turn_waypoints = [(wx + cx, wy + cy) for wx, wy in wps]
            car.has_turned = True
            log.debug(
                "TURN START: %s approach=%s dir=%s  trigger_dist=%.1f  "
                "waypoints=%s",
                car.id, car.approach, car.ml_direction, dist,
                car._turn_waypoints,
            )

    def _spawn_is_clear(self, x: float, y: float) -> bool:
        for car in self.cars:
            if math.hypot(car.x - x, car.y - y) < self.policy.spawn_min_gap_m:
                return False
        return True

    def _next_intersection_for(self, car: Car) -> Optional[IntersectionNode]:
        """If the car just passed its current intersection and is heading
        toward a connected one, return the next IntersectionNode."""
        # Determine what arm the car is exiting through.
        exit_arm = self._exit_arm(car)
        if exit_arm is None:
            return None
        conn = self.network.connected_arm(car.current_int_id, exit_arm)
        if conn is None:
            return None
        next_id, _entry_arm = conn
        return self.network.intersections.get(next_id)

    def _exit_arm(self, car: Car) -> Optional[str]:
        """Determine which arm a car is exiting through based on its velocity."""
        if car.vx > 0.5:   return "E"
        if car.vx < -0.5:  return "W"
        if car.vy > 0.5:   return "N"
        if car.vy < -0.5:  return "S"
        return None

    def _arrival_arm(self, car: Car, target_node: IntersectionNode) -> Optional[str]:
        """Determine the approach arm at the target intersection."""
        exit_arm = self._exit_arm(car)
        if exit_arm is None:
            return None
        conn = self.network.connected_arm(car.current_int_id, exit_arm)
        if conn is None:
            return None
        _next_id, entry_arm = conn
        return entry_arm

    def _car_in_intersection(self, car: Car) -> bool:
        """True when the car is physically inside its target intersection box."""
        half = self.policy.intersection_box_half_m
        cx, cy = self._int_center(car)
        return abs(car.x - cx) < half and abs(car.y - cy) < half

    def _distance_to_stop_line(self, car: Car) -> float:
        """Distance from car to its stop line along the travel axis.

        The stop line sits at ``policy.stop_line_offset_m`` from the
        car's current intersection centre.  Returns a positive value
        when the car hasn't reached the line yet, zero/negative when
        it has passed.
        """
        cx, cy = self._int_center(car)
        rx, ry = car.x - cx, car.y - cy
        edge = self.policy.stop_line_offset_m
        if car.vx > 0:       # W → E
            return -rx - edge
        elif car.vx < 0:     # E → W
            return rx - edge
        elif car.vy < 0:     # N → S
            return ry - edge
        elif car.vy > 0:     # S → N
            return -ry - edge
        return math.hypot(rx, ry)

    # ── safety / control ─────────────────────────────────────────────────

    def _build_target_speeds(self, decisions: Dict[str, Dict[str, Any]], dt: float = 0.1) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        for car in self.cars:
            raw_decision = decisions.get(car.id, {})
            if isinstance(raw_decision, dict):
                decision_payload = raw_decision
            else:
                decision_payload = {"decision": raw_decision}
            target = self._target_speed_from_decision(car, decision_payload)
            if self.policy.world_signal_scheduler_enabled and self._must_yield_to_signal(car):
                cx, cy = self._int_center(car)
                dist = math.hypot(car.x - cx, car.y - cy)
                if dist <= self.policy.red_hard_radius_m:
                    target = 0.0
                else:
                    target = min(target, self.policy.red_soft_speed_kmh)

            # ── Per-intersection control type ────────────────────────────
            car_node = self.network.intersections.get(car.current_int_id)
            car_has_sem = car_node.has_semaphore if car_node else False

            # ── Semaphore enforcement (overrides signs when enabled) ─────
            if car_has_sem and not car.passed:
                sem_color = self.semaphore_color_for_car(car)
                if sem_color in (_PHASE_RED, _PHASE_YELLOW):
                    dist_to_line = self._distance_to_stop_line(car)
                    if dist_to_line > self.policy.semaphore_brake_zone_m:
                        # Far away — cruise normally
                        target = max(target, car.cruise_speed)
                    elif dist_to_line > 0.0:
                        ratio = dist_to_line / self.policy.semaphore_brake_zone_m
                        approach_target = ratio * car.cruise_speed
                        target = min(target, approach_target)
                    elif dist_to_line > -2.0:
                        # Just reached the stop line — hold here
                        target = 0.0
                    else:
                        # Already well past the line (committed) — let it through
                        target = max(target, car.cruise_speed)
                elif sem_color == _PHASE_GREEN:
                    # Green → drive through, override any lingering ML STOP
                    target = max(target, car.cruise_speed)
                    # Also mark stop_completed so sign logic doesn't re-apply
                    car.stop_completed = True

            # ── Stop / Yield enforcement (signs – only when no semaphore) ─
            elif not car_has_sem:
                sign = self.sign_for_car(car)
                required_wait = 0.0
                if sign == "STOP":
                    required_wait = self.policy.stop_sign_wait_s     # 0.2 s
                elif sign == "YIELD":
                    required_wait = self.policy.stop_sign_wait_s * 0.5  # 0.1 s

                if required_wait > 0.0 and not car.passed:
                    if not car.stop_completed:
                        dist_to_line = self._distance_to_stop_line(car)

                        if dist_to_line > self.policy.stop_brake_zone_m:
                            target = max(target, car.cruise_speed)

                        elif dist_to_line > 0.0:
                            ratio = dist_to_line / self.policy.stop_brake_zone_m
                            approach_target = ratio * car.cruise_speed
                            target = min(target, approach_target)

                            if car.speed < 1.0:
                                car.stop_wait_s += dt
                            if car.stop_wait_s >= required_wait:
                                car.stop_completed = True

                        else:
                            target = 0.0
                            if car.speed < 1.0:
                                car.stop_wait_s += dt
                            if car.stop_wait_s >= required_wait:
                                car.stop_completed = True
                    else:
                        target = max(target, car.cruise_speed)

            # ── Intersection box speed cap ───────────────────────────────
            # Lift the cap ONLY during yellow so committed cars clear
            # the box quickly, then re-apply once yellow is over.
            if self._car_in_intersection(car) and not car.passed:
                sem_color = self.semaphore_color_for_car(car)
                committed = self._distance_to_stop_line(car) < -2.0
                if committed and sem_color == _PHASE_YELLOW:
                    target = max(target, car.cruise_speed)
                else:
                    target = min(target, self.policy.intersection_speed_cap_kmh)

            targets[car.id] = target
        return targets

    def _target_speed_from_decision(self, car: Car, decision_payload: Dict[str, Any]) -> float:
        """
        Convert one vehicle decision payload into a speed target.

        Contract (scalable):
        - ``target_speed_kmh`` (numeric) has highest priority.
        - ``decision == STOP`` maps to ``policy.ml_stop_target_speed_kmh``.
        - otherwise defaults to cruise speed.
        """
        raw_target = decision_payload.get("target_speed_kmh")
        if raw_target is not None:
            try:
                explicit = float(raw_target)
            except (TypeError, ValueError):
                explicit = car.cruise_speed
            return max(0.0, min(explicit, self.policy.ml_max_target_speed_kmh))

        decision = str(decision_payload.get("decision", "none")).upper()
        if decision == "STOP":
            return max(0.0, self.policy.ml_stop_target_speed_kmh)
        return car.cruise_speed

    def _apply_speed_targets(self, targets: Dict[str, float], dt: float) -> None:
        for car in self.cars:
            target = max(0.0, float(targets.get(car.id, car.cruise_speed)))

            # Choose acceleration rate: use green-launch boost when
            # accelerating from near-standstill toward a high target.
            accel = self.policy.max_accel_kmh_s
            if (target >= car.cruise_speed * 0.8
                    and car.speed < self.policy.creep_filter_kmh * 2):
                accel = self.policy.green_launch_accel_kmh_s

            if car.speed > target:
                car.speed = max(target, car.speed - self.policy.max_brake_kmh_s * dt)
            elif car.speed < target:
                car.speed = min(target, car.speed + accel * dt)

            # Anti-creep filter: snap very low speeds to zero
            if car.speed < self.policy.creep_filter_kmh and target < self.policy.creep_filter_kmh:
                car.speed = 0.0

            if car.speed < 1.0:
                car.wait_s += dt
            else:
                car.wait_s = max(0.0, car.wait_s - 0.25 * dt)

    def _apply_collision_guard(self, targets: Dict[str, float], dt: float, tick: int = 0) -> int:
        interventions = 0
        n = len(self.cars)
        if n < 2:
            return interventions

        # ── 1.  Cross-approach / general pair-wise guard ──────────────
        # Runs FIRST so that hard-stops propagate to the following-
        # distance guard below (followers must see the leader's
        # reduced target, not its stale cruise speed).
        for _ in range(3):
            for i in range(n):
                a = self.cars[i]
                for j in range(i + 1, n):
                    b = self.cars[j]

                    # Skip pairs where both cars already passed the intersection.
                    if a.passed and b.passed:
                        continue

                    if not self._pair_needs_guard(a, b, targets, dt):
                        continue

                    safe_dist = pair_safe_distance_m(a, b, self.policy)
                    now_dist = math.hypot(a.x - b.x, a.y - b.y)
                    yielder = self._pick_yielder(a, b)
                    current = targets.get(yielder.id, yielder.cruise_speed)

                    # If the yielder hasn't entered the intersection yet,
                    # hold it at the stop line instead of letting it creep in.
                    dist_to_line = self._distance_to_stop_line(yielder)
                    yielder_outside = dist_to_line > 0.0 and not yielder.passed

                    if now_dist <= safe_dist * 0.78:
                        # Very close — hard stop.
                        new_target = 0.0
                        xreason = "VERY CLOSE hard stop"

                        # ── Backtrack: nudge the yielder backward when
                        #    dangerously close inside the intersection.
                        yielder_in_box = self._car_in_intersection(yielder)
                        if yielder_in_box and now_dist < safe_dist * 0.5:
                            bt = self.policy.intersection_backtrack_m
                            yielder.x -= yielder.vx * bt
                            yielder.y -= yielder.vy * bt
                            xreason = f"BACKTRACK {bt:.1f}m"
                    elif yielder_outside and dist_to_line < 3.0:
                        # Near the stop line — hard stop to prevent creeping in.
                        new_target = 0.0
                        xreason = "NEAR LINE hard stop"
                    elif yielder_outside:
                        # Approaching — ramp speed down toward the line.
                        ratio = min(1.0, dist_to_line / self.policy.stop_brake_zone_m)
                        new_target = min(current, ratio * yielder.cruise_speed)
                        xreason = f"RAMP ratio={ratio:.2f}"
                    else:
                        # Yielder already inside the intersection.
                        other = b if yielder is a else a
                        both_in_box = (self._car_in_intersection(yielder)
                                       and self._car_in_intersection(other))
                        if both_in_box and now_dist < safe_dist * 1.2:
                            # Both in the box and close — hard stop yielder
                            new_target = 0.0
                            xreason = "IN-BOX CONFLICT hard stop"
                        else:
                            new_target = min(current, self.policy.collision_guard_soft_speed_kmh)
                            xreason = "SOFT CAP"

                    if new_target < current:
                        targets[yielder.id] = new_target
                        interventions += 1
                        if tick % 10 == 1:
                            other = b if yielder is a else a
                            log.debug(
                                "  CROSS: %s yields to %s  dist=%.1f safe=%.1f  "
                                "d2line=%.1f outside=%s  %s  %.1f->%.1f",
                                yielder.id, other.id, now_dist, safe_dist,
                                dist_to_line, yielder_outside, xreason,
                                current, new_target,
                            )

        # ── 2.  Same-lane following-distance guard ────────────────────
        # Runs AFTER cross-guard so followers inherit any hard-stop
        # that was applied to their leader above.
        lane_groups: Dict[Tuple[float, float], List[Car]] = {}
        for c in self.cars:
            key = (c.vx, c.vy)
            lane_groups.setdefault(key, []).append(c)

        for (vx, vy), group in lane_groups.items():
            if len(group) < 2:
                continue
            # Sort so the car closest to the intersection is first.
            if vx > 0:
                group.sort(key=lambda c: -c.x)   # W→E: largest x first
            elif vx < 0:
                group.sort(key=lambda c: c.x)     # E→W: smallest x first
            elif vy < 0:
                group.sort(key=lambda c: c.y)     # N→S: smallest y first
            elif vy > 0:
                group.sort(key=lambda c: -c.y)    # S→N: largest y first

            for idx in range(1, len(group)):
                leader = group[idx - 1]
                follower = group[idx]
                gap = self._following_gap(leader, follower)
                safe = pair_safe_distance_m(leader, follower, self.policy)

                if gap < safe * 1.5:
                    leader_target = targets.get(leader.id, leader.cruise_speed)
                    # Also consider the leader's actual speed — if it is
                    # physically much slower than its target (e.g. just
                    # stopped by cross-guard or waiting at a sign), the
                    # follower must match the real speed, not the wish.
                    effective_leader = min(leader_target, leader.speed)
                    follower_current = targets.get(follower.id, follower.cruise_speed)

                    if gap < safe * 0.8:
                        new_target = 0.0
                        reason = "gap<0.8*safe HARD STOP"
                    elif gap < safe:
                        new_target = min(follower_current, max(0.0, effective_leader * 0.5))
                        reason = "gap<safe HALF"
                    else:
                        new_target = min(follower_current, effective_leader)
                        reason = "gap<1.5*safe MATCH"

                    if new_target < follower_current:
                        targets[follower.id] = new_target
                        interventions += 1
                        if tick % 10 == 1:
                            log.debug(
                                "  FOLLOW: %s behind %s  gap=%.1f safe=%.1f  "
                                "%s  leader_eff=%.1f(tgt=%.1f spd=%.1f)  %.1f->%.1f",
                                follower.id, leader.id, gap, safe,
                                reason, effective_leader, leader_target, leader.speed,
                                follower_current, new_target,
                            )
        return interventions

    @staticmethod
    def _following_gap(leader: Car, follower: Car) -> float:
        """Axial distance between two same-direction cars."""
        if leader.vx != 0:
            return abs(leader.x - follower.x)
        return abs(leader.y - follower.y)

    def _pair_needs_guard(
        self,
        a: Car,
        b: Car,
        targets: Dict[str, float],
        dt: float,
    ) -> bool:
        safe_dist = pair_safe_distance_m(a, b, self.policy)
        now = math.hypot(a.x - b.x, a.y - b.y)
        if now <= safe_dist:
            return True

        # ── Skip cross-intersection projection for distant cars ──────
        # Cars at different intersections can't conflict unless they
        # are on a connecting road segment (same direction, close).
        # Skip expensive projection for cars at different intersections
        # that are far apart. This prevents false conflicts between
        # cars at INT_A and INT_B for example.
        same_int = (a.current_int_id == b.current_int_id)
        same_direction = (a.vx == b.vx and a.vy == b.vy)
        if not same_int and not same_direction:
            # Different intersections, different directions → no conflict
            # unless they're already very close (covered by check above).
            return False
        if not same_int and same_direction and now > 50.0:
            # Same direction (possibly connecting road) but far apart.
            return False

        # ── Time-swept projection ────────────────────────────────────
        # Check multiple time steps so perpendicular approaches that
        # converge on the intersection centre don't slip through a
        # single-snapshot check.
        # When the semaphore is active, skip projection for
        # perpendicular pairs — the traffic light already separates
        # conflicting phases and the projection would wrongly block
        # green-phase cars from launching.
        perpendicular = (a.vx != 0) != (b.vx != 0)
        a_node = self.network.intersections.get(a.current_int_id)
        b_node = self.network.intersections.get(b.current_int_id)
        both_sem = (a_node and a_node.has_semaphore) and (b_node and b_node.has_semaphore)
        if not (perpendicular and both_sem):
            va = max(0.0, targets.get(a.id, a.speed)) / 3.6
            vb = max(0.0, targets.get(b.id, b.speed)) / 3.6
            horizon = max(dt, self.policy.horizon_s)
            steps = 4
            step_dt = horizon / steps
            for k in range(1, steps + 1):
                t = step_dt * k
                ax = a.x + a.vx * va * t
                ay = a.y + a.vy * va * t
                bx = b.x + b.vx * vb * t
                by = b.y + b.vy * vb * t
                if math.hypot(ax - bx, ay - by) <= safe_dist:
                    return True

        # ── Intersection-zone conflict for perpendicular approaches ──
        # If both cars are heading into the SAME intersection from different
        # directions and both will reach the stop line within the
        # horizon, they *will* conflict at the centre even if the
        # swept check above missed it (e.g. they arrive at slightly
        # different times and the step granularity skipped the overlap).
        # SKIP this check when the semaphore is active — the traffic
        # light already separates the conflicting phases.
        # Also SKIP if cars are at different intersections.
        if same_int and not (a.passed or b.passed) and not both_sem:
            da = self._distance_to_stop_line(a)
            db = self._distance_to_stop_line(b)
            perpendicular = (a.vx != 0) != (b.vx != 0)  # one horizontal, one vertical
            if perpendicular and da >= 0 and db >= 0:
                # Time each car needs to reach the stop line.
                eta_a = da / va if va > 0.5 else 999.0
                eta_b = db / vb if vb > 0.5 else 999.0
                # Both arrive within a wide window → conflict.
                window = max(2.5, horizon)
                if eta_a < window and eta_b < window:
                    return True

        return False

    @staticmethod
    def _project(car: Car, speed_kmh: float, dt: float) -> Tuple[float, float]:
        speed_mps = max(0.0, speed_kmh) / 3.6
        return (
            car.x + car.vx * speed_mps * dt,
            car.y + car.vy * speed_mps * dt,
        )

    def _pick_yielder(self, a: Car, b: Car) -> Car:
        # A car already inside the intersection (crossed stop line but not yet
        # through) has effective right-of-way — never make it yield.
        a_in = self._distance_to_stop_line(a) <= 0.0 and not a.passed
        b_in = self._distance_to_stop_line(b) <= 0.0 and not b.passed
        if a_in and not b_in:
            return b
        if b_in and not a_in:
            return a

        # A car that already passed should not be slowed down.
        if a.passed and not b.passed:
            return b
        if b.passed and not a.passed:
            return a

        # Same-lane rear-end guard: trailing car yields.
        if a.vx == b.vx and a.vy == b.vy:
            if a.vx != 0 and abs(a.y - b.y) <= 1.0:
                a_ahead = (a.x - b.x) * a.vx > 0
                return b if a_ahead else a
            if a.vy != 0 and abs(a.x - b.x) <= 1.0:
                a_ahead = (a.y - b.y) * a.vy > 0
                return b if a_ahead else a

        # ── Traffic-sign priority (dominant rule) ─────────────────────
        # PRIORITY > NO_SIGN > YIELD > STOP.
        # The car with the weaker sign must always yield.
        _SIGN_RANK = {"PRIORITY": 3, "NO_SIGN": 2, "YIELD": 1, "STOP": 0}
        sa = _SIGN_RANK.get(self.sign_for_car(a), 2)
        sb = _SIGN_RANK.get(self.sign_for_car(b), 2)
        if sa != sb:
            return a if sa < sb else b

        # ── Right-of-way inside the intersection ─────────────────────
        # (a) Left-turning car yields to oncoming straight/right car.
        a_turning_left = (a.ml_direction == "LEFT" and a.is_turning)
        b_turning_left = (b.ml_direction == "LEFT" and b.is_turning)
        if a_turning_left and not b_turning_left:
            return a
        if b_turning_left and not a_turning_left:
            return b
        # (b) Right-hand priority: the car that has the other on its
        #     right side must yield.  We use the cross product of the
        #     two velocity vectors to determine relative side.
        if a.vx != b.vx or a.vy != b.vy:
            cross = a.vx * b.vy - a.vy * b.vx
            if cross > 0:      # b is to a's right → a yields
                return a
            elif cross < 0:    # a is to b's right → b yields
                return b

        # Priority scheduler: lower-priority car yields.
        pa = danger_score(a, self.policy)
        pb = danger_score(b, self.policy)
        if abs(pa - pb) > 0.1:
            return a if pa < pb else b

        # Fallback intersection guard: farther from its intersection centre yields.
        cxa, cya = self._int_center(a)
        cxb, cyb = self._int_center(b)
        da = math.hypot(a.x - cxa, a.y - cya)
        db = math.hypot(b.x - cxb, b.y - cyb)
        if abs(da - db) > 0.1:
            return a if da > db else b

        # Stable tie-breaker.
        return a if a.id > b.id else b

    # ── Semaphore (fixed-cycle traffic lights) ──────────────────────────

    def _update_semaphore(self, dt: float) -> None:
        """Advance per-intersection semaphore state machines."""
        for node in self.network.intersections.values():
            if not node.has_semaphore:
                continue
            node.sem_timer -= dt
            if node.sem_timer > 0.0:
                continue
            if node.sem_phase == _PHASE_GREEN:
                node.sem_phase = _PHASE_YELLOW
                node.sem_timer = self.policy.semaphore_yellow_s
            elif node.sem_phase == _PHASE_YELLOW:
                node.sem_phase = _PHASE_RED
                node.sem_timer = self.policy.semaphore_red_clearance_s
            else:
                other = "NS" if node.sem_green_axis == "EW" else "EW"
                node.sem_green_axis = other
                node.sem_phase = _PHASE_GREEN
                node.sem_timer = self.policy.semaphore_green_s
        # Keep legacy fields in sync with first semaphore intersection.
        for node in self.network.intersections.values():
            if node.has_semaphore:
                self._sem_green_axis = node.sem_green_axis
                self._sem_phase = node.sem_phase
                self._sem_timer = node.sem_timer
                break

    def semaphore_color_for_car(self, car: Car) -> str:
        """Return 'GREEN', 'YELLOW', or 'RED' for *car*'s current intersection."""
        node = self.network.intersections.get(car.current_int_id)
        if node is None or not node.has_semaphore:
            return "GREEN"
        axis_for_approach = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}
        car_axis = axis_for_approach.get(car.approach, "EW")
        if car_axis == node.sem_green_axis:
            return node.sem_phase
        return _PHASE_RED

    def semaphore_color_for_approach(self, approach: str) -> str:
        """Legacy method — uses first semaphore intersection."""
        if not self.policy.semaphore_enabled:
            return "GREEN"
        axis_for_approach = {
            "N": "NS", "S": "NS",
            "E": "EW", "W": "EW",
        }
        car_axis = axis_for_approach.get(approach, "EW")
        if car_axis == self._sem_green_axis:
            return self._sem_phase
        return _PHASE_RED

    def semaphore_state(self) -> Dict[str, Any]:
        """Return full semaphore state dict (for UI / bus)."""
        return {
            "enabled": self.policy.semaphore_enabled,
            "green_axis": self._sem_green_axis,
            "phase": self._sem_phase,
            "timer": round(self._sem_timer, 1),
            "colors": {
                a: self.semaphore_color_for_approach(a) for a in _APPROACHES
            },
        }

    def semaphore_state_for(self, int_id: str) -> Dict[str, Any]:
        """Return semaphore state for a specific intersection."""
        node = self.network.intersections.get(int_id)
        if node is None or not node.has_semaphore:
            return {"enabled": False}
        axis_for_approach = {"N": "NS", "S": "NS", "E": "EW", "W": "EW"}
        colors = {}
        for a in _APPROACHES:
            car_axis = axis_for_approach.get(a, "EW")
            if car_axis == node.sem_green_axis:
                colors[a] = node.sem_phase
            else:
                colors[a] = _PHASE_RED
        return {
            "enabled": True,
            "green_axis": node.sem_green_axis,
            "phase": node.sem_phase,
            "timer": round(node.sem_timer, 1),
            "colors": colors,
        }

    def _must_yield_to_signal(self, car: Car) -> bool:
        if car.passed:
            return False
        if self._is_emergency(car):
            return False
        cx, cy = self._int_center(car)
        dist = math.hypot(car.x - cx, car.y - cy)
        if dist > self.policy.signal_control_radius_m:
            return False
        return car.approach != self.green_approach

    def _is_emergency(self, car: Car) -> bool:
        return car.role in ("ambulance", "police", "fire")

    def _update_virtual_signal(self, dt: float) -> None:
        self._green_ttl_s = max(0.0, self._green_ttl_s - dt)
        approach_scores: Dict[str, float] = {a: 0.0 for a in _APPROACHES}

        for car in self.cars:
            if car.passed:
                continue
            cx, cy = self._int_center(car)
            dist = math.hypot(car.x - cx, car.y - cy)
            if dist > self.policy.signal_control_radius_m:
                continue
            score = danger_score(car, self.policy)
            if self._is_emergency(car):
                score += 3.0
            approach_scores[car.approach] += score

        best_approach = max(approach_scores, key=approach_scores.get)
        best_score = approach_scores[best_approach]
        current_score = approach_scores.get(self.green_approach, 0.0)

        should_switch = (
            self._green_ttl_s <= 0.0
            or (
                best_approach != self.green_approach
                and best_score > current_score + self.policy.signal_switch_margin
            )
        )
        if should_switch and best_score > 0.0:
            self.green_approach = best_approach
            self._green_ttl_s = self.policy.signal_window_s

    @staticmethod
    def _normalize_priority_axis(priority_axis: str) -> str:
        axis = str(priority_axis).upper()
        if axis in ("NS", "SN"):
            return "NS"
        return "EW"

    @staticmethod
    def _build_signs_by_approach(priority_axis: str) -> Dict[str, str]:
        axis = World._normalize_priority_axis(priority_axis)
        if axis == "NS":
            return {
                "N": "PRIORITY",
                "S": "PRIORITY",
                "E": "STOP",
                "W": "YIELD",
            }
        return {
            "E": "PRIORITY",
            "W": "PRIORITY",
            "N": "STOP",
            "S": "YIELD",
        }

    def _resolve_overlaps(self) -> int:
        """
        Last-resort overlap resolver to prevent cars rendering inside each other.
        """
        count = 0
        tick = self._tick_count
        n = len(self.cars)
        # Use a fixed hard radius for overlap detection so speed changes
        # don't create a feedback loop (pair_safe_distance_m shrinks when
        # we cap speed, causing repeated triggers).
        hard_radius = self.policy.min_pair_distance_m
        for _ in range(3):
            changed = False
            for i in range(n):
                a = self.cars[i]
                for j in range(i + 1, n):
                    b = self.cars[j]
                    dx = a.x - b.x
                    dy = a.y - b.y
                    dist = math.hypot(dx, dy)
                    if dist >= hard_radius:
                        continue

                    count += 1
                    changed = True

                    a_spd_before = a.speed
                    b_spd_before = b.speed

                    if dist < 1e-6:
                        b.x += 0.5
                        b.y += 0.5
                    else:
                        overlap = (hard_radius - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        a.x += ux * overlap
                        a.y += uy * overlap
                        b.x -= ux * overlap
                        b.y -= uy * overlap

                    # Only slow the faster car instead of capping both.
                    if a.speed > b.speed:
                        a.speed = min(a.speed, max(b.speed, 3.0))
                    else:
                        b.speed = min(b.speed, max(a.speed, 3.0))

                    if tick % 10 == 1:
                        log.debug(
                            "  OVERLAP: %s & %s  dist=%.1f hard_r=%.1f  "
                            "nudge=%.2f  %s spd %.1f->%.1f  %s spd %.1f->%.1f",
                            a.id, b.id, dist, hard_radius,
                            (hard_radius - dist) / 2.0,
                            a.id, a_spd_before, a.speed,
                            b.id, b_spd_before, b.speed,
                        )
            if not changed:
                break
        return count


if __name__ == "__main__":
    world = World()
    world.update_physics()
    print(world.get_ml_input())
