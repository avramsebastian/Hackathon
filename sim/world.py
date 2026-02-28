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


# Lane-centre offset in world metres.
LANE_OFFSET_M = 7.0

# How far past the centre a car must travel before we consider it "through".
_PASS_THRESHOLD_M = 10.0

# Deceleration once a car is past (applied to km/h value per second).
_DECEL_KMH_PER_S = 60.0
_MAX_ACCEL_KMH_PER_S = 18.0
_MAX_BRAKE_KMH_PER_S = 90.0
_CAR_COLLISION_RADIUS_M = 2.6
_CAR_PAIR_SAFE_DIST_M = _CAR_COLLISION_RADIUS_M * 2.0
_ML_STOP_SOFT_RADIUS_M = 30.0
_ML_STOP_HARD_RADIUS_M = 14.0
_ML_STOP_SOFT_SPEED_KMH = 24.0

_ML_DIRECTIONS: Tuple[str, ...] = ("FORWARD", "LEFT", "RIGHT")
_APPROACHES: Tuple[str, ...] = ("W", "N", "E", "S")
_SPAWN_MIN_RADIUS_M = 70.0
_SPAWN_MAX_RADIUS_M = 140.0
_SPAWN_MIN_GAP_M = 18.0
_SPAWN_MAX_ATTEMPTS = 300


@dataclass
class Car:
    """A standalone vehicle entity."""

    id: str
    x: float
    y: float
    speed: float            # km/h
    ml_direction: str       # FORWARD | LEFT | RIGHT  (for ML model)
    approach: str           # W | N | E | S            (which side car enters from)
    cruise_speed: float = 0.0
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
    def has_passed(self, threshold: float = _PASS_THRESHOLD_M) -> bool:
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

    def __init__(self, num_cars: int = 6, seed: Optional[int] = None) -> None:
        self.current_sign: str = "YIELD"
        self.num_cars = max(1, int(num_cars))
        self._rng = random.Random(seed)
        self.cars: List[Car] = []
        self.safety_interventions: int = 0
        self.collision_resolutions: int = 0
        self._init_cars()

    # ── initialisation / reset ────────────────────────────────────────────

    def _init_cars(self) -> None:
        self.cars = []
        for idx in range(self.num_cars):
            self.cars.append(self._make_random_car(idx))
        self.safety_interventions = 0
        self.collision_resolutions = 0
        self._finished = False

    def reset(self) -> None:
        """Re-initialise all cars so the scenario can be replayed."""
        self._init_cars()

    # ── queries ───────────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        return self._finished

    def all_cars(self) -> List[Car]:
        return list(self.cars)

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

        targets = self._build_target_speeds(decisions)
        interventions = self._apply_collision_guard(targets, dt)
        self.safety_interventions += interventions

        self._apply_speed_targets(targets, dt)

        for car in self.cars:
            car.move(dt)
            if not car.passed and car.has_passed():
                car.passed = True

        self.collision_resolutions += self._resolve_overlaps()

        all_passed = all(c.passed for c in self.cars)
        if all_passed:
            for car in self.cars:
                if car.stopped:
                    continue
                car.speed = max(0.0, car.speed - _DECEL_KMH_PER_S * dt)
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
            "sign": self.current_sign,
            "traffic": [car.as_dict() for car in self.cars[1:]],
        }

    # ── spawning ──────────────────────────────────────────────────────────

    def _make_random_car(self, idx: int) -> Car:
        """
        Spawn one car on a random approach arm, while keeping a minimum
        distance from already spawned cars.
        """
        approach = self._rng.choice(_APPROACHES)
        speed = self._rng.uniform(24.0, 45.0)
        ml_direction = self._rng.choice(_ML_DIRECTIONS)

        pos: Optional[Tuple[float, float, float, float]] = None
        for _ in range(_SPAWN_MAX_ATTEMPTS):
            distance = self._rng.uniform(_SPAWN_MIN_RADIUS_M, _SPAWN_MAX_RADIUS_M)
            candidate = self._spawn_position(approach, distance)
            if self._spawn_is_clear(candidate[0], candidate[1]):
                pos = candidate
                break

            # Try another approach if this lane is crowded.
            approach = self._rng.choice(_APPROACHES)

        if pos is None:
            # Deterministic fallback: radial spread by index.
            fallback_approach = _APPROACHES[idx % len(_APPROACHES)]
            distance = _SPAWN_MIN_RADIUS_M + idx * 10.0
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
            cruise_speed=speed,
            vx=vx,
            vy=vy,
        )

    def _spawn_position(self, approach: str, distance: float) -> Tuple[float, float, float, float]:
        """
        Return spawn (x, y) and heading vector (vx, vy) for one approach arm.
        """
        L = LANE_OFFSET_M
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
            if math.hypot(car.x - x, car.y - y) < _SPAWN_MIN_GAP_M:
                return False
        return True

    # ── safety / control ─────────────────────────────────────────────────

    def _build_target_speeds(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        targets: Dict[str, float] = {}
        for car in self.cars:
            decision = decisions.get(car.id, {}).get("decision", "none")
            targets[car.id] = self._target_speed_from_decision(car, decision)
        return targets

    def _target_speed_from_decision(self, car: Car, decision: str) -> float:
        if decision != "STOP":
            return car.cruise_speed

        if car.passed:
            return car.cruise_speed

        dist_center = math.hypot(car.x, car.y)
        if dist_center <= _ML_STOP_HARD_RADIUS_M:
            return 0.0
        if dist_center <= _ML_STOP_SOFT_RADIUS_M:
            return min(car.cruise_speed, _ML_STOP_SOFT_SPEED_KMH)
        return car.cruise_speed

    def _apply_speed_targets(self, targets: Dict[str, float], dt: float) -> None:
        for car in self.cars:
            target = max(0.0, float(targets.get(car.id, car.cruise_speed)))
            if car.speed > target:
                car.speed = max(target, car.speed - _MAX_BRAKE_KMH_PER_S * dt)
            elif car.speed < target:
                car.speed = min(target, car.speed + _MAX_ACCEL_KMH_PER_S * dt)
            if car.speed < 0.5:
                car.speed = 0.0

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
                    yielder = self._pick_yielder(a, b)
                    if targets.get(yielder.id, 0.0) > 0.0:
                        targets[yielder.id] = 0.0
                        interventions += 1
        return interventions

    def _pair_needs_guard(
        self,
        a: Car,
        b: Car,
        targets: Dict[str, float],
        dt: float,
    ) -> bool:
        now = math.hypot(a.x - b.x, a.y - b.y)
        if now <= _CAR_PAIR_SAFE_DIST_M:
            return True

        ax, ay = self._project(a, targets.get(a.id, a.speed), dt)
        bx, by = self._project(b, targets.get(b.id, b.speed), dt)
        nxt = math.hypot(ax - bx, ay - by)
        return nxt <= _CAR_PAIR_SAFE_DIST_M

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

        # Intersection guard: car farther from centre yields.
        da = math.hypot(a.x, a.y)
        db = math.hypot(b.x, b.y)
        if abs(da - db) > 0.1:
            return a if da > db else b

        # Stable tie-breaker.
        return a if a.id > b.id else b

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
                    if dist >= _CAR_PAIR_SAFE_DIST_M:
                        continue

                    count += 1
                    changed = True
                    if dist < 1e-6:
                        # Degenerate case: nudge deterministically by id ordering.
                        b.x += 0.5
                        b.y += 0.5
                    else:
                        overlap = (_CAR_PAIR_SAFE_DIST_M - dist) / 2.0
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
