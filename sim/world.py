#!/usr/bin/env python3
"""
sim/world.py
============
Entity-based intersection world.

This module no longer models one ``my_car`` plus ``traffic``. Instead it keeps
only a scalable list of standalone car entities (`cars[]`). Every car can be
used as ego for ML inference, and every car can publish its own V2X payload.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sim.traffic_policy import (
    SafetyPolicy,
    danger_score,
    pair_safe_distance_m,
)

_ML_DIRECTIONS: Tuple[str, ...] = ("FORWARD", "LEFT", "RIGHT")
_APPROACHES: Tuple[str, ...] = ("W", "N", "E", "S")


@dataclass
class Car:
    """A standalone vehicle entity."""

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
    passed: bool = field(default=False, repr=False)
    stopped: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.cruise_speed <= 0.0:
            self.cruise_speed = self.speed

    # ── movement ──────────────────────────────────────────────────────────
    def move(self, dt: float) -> None:
        if self.stopped:
            return
        speed_mps = self.speed / 3.6
        self.x += self.vx * speed_mps * dt
        self.y += self.vy * speed_mps * dt

    # ── helpers ───────────────────────────────────────────────────────────
    def has_passed(self, threshold: float = 10.0) -> bool:
        """True once the car is *threshold* metres past the origin."""
        if self.vx > 0 and self.x > threshold:
            return True
        if self.vx < 0 and self.x < -threshold:
            return True
        if self.vy > 0 and self.y > threshold:
            return True
        if self.vy < 0 and self.y < -threshold:
            return True
        return False

    def as_dict(self) -> dict:
        """Return a serializable car payload."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "direction": self.ml_direction,
            "role": self.role,
        }

    def ml_payload(self, sign: str, others: Sequence["Car"]) -> Dict[str, Any]:
        """Build ML input using this car as ego and all other cars as traffic."""
        return {
            "my_car": self.as_dict(),
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
    """
    Entity-based intersection scenario.

    Cars spawn randomly on approach arms and move through the intersection.
    After passing the centre, they decelerate and eventually stop.
    """

    def __init__(
        self,
        num_cars: int = 6,
        seed: Optional[int] = None,
        policy: Optional[SafetyPolicy] = None,
        priority_axis: str = "EW",
    ) -> None:
        self.policy = policy or SafetyPolicy()
        self.priority_axis = self._normalize_priority_axis(priority_axis)
        self.signs_by_approach: Dict[str, str] = self._build_signs_by_approach(self.priority_axis)
        self.current_sign: str = "YIELD"  # legacy fallback for old call-sites
        self.num_cars = max(1, int(num_cars))
        self._rng = random.Random(seed)
        self.cars: List[Car] = []
        self.safety_interventions: int = 0
        self.collision_resolutions: int = 0
        self.green_approach: str = "W"
        self._green_ttl_s: float = 0.0
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

    def sign_for_approach(self, approach: str) -> str:
        return self.signs_by_approach.get(str(approach).upper(), "NO_SIGN")

    def sign_for_car(self, car: Car) -> str:
        return self.sign_for_approach(car.approach)

    # ── physics tick ──────────────────────────────────────────────────────

    def update_physics(
        self,
        dt: float = 0.1,
        decisions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Advance every car using ML decisions + safety guard.

        The safety layer is authoritative and can override ML "GO" decisions to
        prevent overlaps in the simulation.
        """
        decisions = decisions or {}
        self._update_virtual_signal(dt)

        targets = self._build_target_speeds(decisions)
        interventions = self._apply_collision_guard(targets, dt)
        self.safety_interventions += interventions

        self._apply_speed_targets(targets, dt)

        for car in self.cars:
            car.move(dt)
            if not car.passed and car.has_passed(self.policy.pass_threshold_m):
                car.passed = True

        self.collision_resolutions += self._resolve_overlaps()

        all_passed = all(c.passed for c in self.cars)
        if all_passed:
            for car in self.cars:
                if car.stopped:
                    continue
                car.speed = max(0.0, car.speed - self.policy.coast_decel_kmh_s * dt)
                if car.speed < 0.5:
                    car.speed = 0.0
                    car.stopped = True

        if all(c.stopped for c in self.cars):
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
        Spawn one car on a random approach arm, while keeping a minimum
        distance from already spawned cars.
        """
        approach = self._rng.choice(_APPROACHES)
        speed = self._rng.uniform(24.0, self.policy.speed_limit_kmh + 8.0)
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
            candidate = self._spawn_position(approach, distance)
            if self._spawn_is_clear(candidate[0], candidate[1]):
                pos = candidate
                break

            # Try another approach if this lane is crowded.
            approach = self._rng.choice(_APPROACHES)

        if pos is None:
            # Deterministic fallback: radial spread by index.
            fallback_approach = _APPROACHES[idx % len(_APPROACHES)]
            distance = self.policy.spawn_min_radius_m + idx * 10.0
            pos = self._spawn_position(fallback_approach, distance)
            approach = fallback_approach

        x, y, vx, vy = pos
        return Car(
            id=f"CAR_{idx:03d}",
            x=x,
            y=y,
            speed=speed,
            ml_direction=ml_direction,
            approach=approach,
            role=role,
            cruise_speed=speed,
            speed_limit_kmh=self.policy.speed_limit_kmh,
            vx=vx,
            vy=vy,
        )

    def _spawn_position(self, approach: str, distance: float) -> Tuple[float, float, float, float]:
        """
        Return spawn (x, y) and heading vector (vx, vy) for one approach arm.
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

    def _spawn_is_clear(self, x: float, y: float) -> bool:
        for car in self.cars:
            if math.hypot(car.x - x, car.y - y) < self.policy.spawn_min_gap_m:
                return False
        return True

    # ── safety / control ─────────────────────────────────────────────────

    def _build_target_speeds(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        for car in self.cars:
            decision = decisions.get(car.id, {}).get("decision", "none")
            target = self._target_speed_from_decision(car, decision)
            if self._must_yield_to_signal(car):
                dist = math.hypot(car.x, car.y)
                if dist <= self.policy.red_hard_radius_m:
                    target = 0.0
                else:
                    target = min(target, self.policy.red_soft_speed_kmh)
            targets[car.id] = target
        return targets

    def _target_speed_from_decision(self, car: Car, decision: str) -> float:
        if decision != "STOP":
            return car.cruise_speed

        if car.passed:
            return car.cruise_speed

        dist_center = math.hypot(car.x, car.y)
        if dist_center <= self.policy.ml_stop_hard_radius_m:
            return min(car.cruise_speed, self.policy.ml_stop_soft_speed_kmh)
        if dist_center <= self.policy.ml_stop_soft_radius_m:
            return min(car.cruise_speed, self.policy.ml_stop_soft_speed_kmh)
        return car.cruise_speed

    def _apply_speed_targets(self, targets: Dict[str, float], dt: float) -> None:
        for car in self.cars:
            target = max(0.0, float(targets.get(car.id, car.cruise_speed)))
            if car.speed > target:
                car.speed = max(target, car.speed - self.policy.max_brake_kmh_s * dt)
            elif car.speed < target:
                car.speed = min(target, car.speed + self.policy.max_accel_kmh_s * dt)
            if car.speed < 0.5:
                car.speed = 0.0

            if car.speed < 1.0:
                car.wait_s += dt
            else:
                car.wait_s = max(0.0, car.wait_s - 0.25 * dt)

    def _apply_collision_guard(self, targets: Dict[str, float], dt: float) -> int:
        interventions = 0
        n = len(self.cars)
        if n < 2:
            return interventions

        # Two short passes let one pair-wise intervention propagate to others.
        for _ in range(2):
            for i in range(n):
                a = self.cars[i]
                for j in range(i + 1, n):
                    b = self.cars[j]
                    if not self._pair_needs_guard(a, b, targets, dt):
                        continue

                    safe_dist = pair_safe_distance_m(a, b, self.policy)
                    now_dist = math.hypot(a.x - b.x, a.y - b.y)
                    yielder = self._pick_yielder(a, b)
                    current = targets.get(yielder.id, yielder.cruise_speed)

                    # Far conflicts slow down; only close conflicts force a full stop.
                    if now_dist <= safe_dist * 0.78:
                        new_target = 0.0
                    else:
                        new_target = min(current, self.policy.ml_stop_soft_speed_kmh)

                    if new_target < current:
                        targets[yielder.id] = new_target
                        interventions += 1
        return interventions

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

        horizon = max(dt, self.policy.horizon_s)
        ax, ay = self._project(a, targets.get(a.id, a.speed), horizon)
        bx, by = self._project(b, targets.get(b.id, b.speed), horizon)
        nxt = math.hypot(ax - bx, ay - by)
        return nxt <= safe_dist

    @staticmethod
    def _project(car: Car, speed_kmh: float, dt: float) -> Tuple[float, float]:
        speed_mps = max(0.0, speed_kmh) / 3.6
        return (
            car.x + car.vx * speed_mps * dt,
            car.y + car.vy * speed_mps * dt,
        )

    def _pick_yielder(self, a: Car, b: Car) -> Car:
        # Same-lane rear-end guard: trailing car yields.
        if a.vx == b.vx and a.vy == b.vy:
            if a.vx != 0 and abs(a.y - b.y) <= 1.0:
                a_ahead = (a.x - b.x) * a.vx > 0
                return b if a_ahead else a
            if a.vy != 0 and abs(a.x - b.x) <= 1.0:
                a_ahead = (a.y - b.y) * a.vy > 0
                return b if a_ahead else a

        # Priority scheduler: lower-priority car yields.
        pa = danger_score(a, self.policy)
        pb = danger_score(b, self.policy)
        if abs(pa - pb) > 0.1:
            return a if pa < pb else b

        # Fallback intersection guard: farther from centre yields.
        da = math.hypot(a.x, a.y)
        db = math.hypot(b.x, b.y)
        if abs(da - db) > 0.1:
            return a if da > db else b

        # Stable tie-breaker.
        return a if a.id > b.id else b

    def _must_yield_to_signal(self, car: Car) -> bool:
        if car.passed:
            return False
        if self._is_emergency(car):
            return False
        dist = math.hypot(car.x, car.y)
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
            dist = math.hypot(car.x, car.y)
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
        n = len(self.cars)
        for _ in range(3):
            changed = False
            for i in range(n):
                a = self.cars[i]
                for j in range(i + 1, n):
                    b = self.cars[j]
                    dx = a.x - b.x
                    dy = a.y - b.y
                    dist = math.hypot(dx, dy)
                    safe_dist = pair_safe_distance_m(a, b, self.policy)
                    if dist >= safe_dist:
                        continue

                    count += 1
                    changed = True
                    if dist < 1e-6:
                        # Degenerate case: nudge deterministically by id ordering.
                        b.x += 0.5
                        b.y += 0.5
                    else:
                        overlap = (safe_dist - dist) / 2.0
                        ux, uy = dx / dist, dy / dist
                        a.x += ux * overlap
                        a.y += uy * overlap
                        b.x -= ux * overlap
                        b.y -= uy * overlap

                    a.speed = min(a.speed, 5.0)
                    b.speed = min(b.speed, 5.0)
            if not changed:
                break
        return count


if __name__ == "__main__":
    world = World()
    world.update_physics()
    print(world.get_ml_input())
