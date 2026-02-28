#!/usr/bin/env python3
"""
sim/world.py
============
Lightweight intersection world.

Each :class:`Car` carries **two** direction concepts:

*   ``ml_direction`` – one of ``"FORWARD"`` / ``"LEFT"`` / ``"RIGHT"``
    (what the ML model expects via :pymeth:`get_ml_input`).
*   ``approach`` – the cardinal side the car enters from
    (``"W"``, ``"N"``, ``"E"``, ``"S"``).  Together with the velocity
    vector (``vx``, ``vy``) this lets the bridge compute a UI-friendly
    cardinal ``direction`` and a ``road_line``.

Cars start ≈100 m away from the intersection centre, drive through it,
then decelerate and stop — exactly like the demo scenario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Lane-centre offset in world metres (matches the demo value).
LANE_OFFSET_M = 7.0

# How far past the centre a car must travel before we consider it "through".
_PASS_THRESHOLD_M = 10.0

# Deceleration once a car is past (applied to km/h value per second).
_DECEL_KMH_PER_S = 60.0


@dataclass
class Car:
    """A single vehicle in the world."""

    x: float
    y: float
    speed: float            # km/h
    ml_direction: str       # FORWARD | LEFT | RIGHT  (for ML model)
    approach: str           # W | N | E | S            (which side car enters from)
    vx: float = 0.0        # unit velocity component  (+1 / -1 / 0)
    vy: float = 0.0
    passed: bool = field(default=False, repr=False)
    stopped: bool = field(default=False, repr=False)

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
        """Return ML-compatible dict (uses ``ml_direction`` as ``direction``)."""
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "direction": self.ml_direction,
        }


class World:
    """
    Simple intersection scenario with one player car and traffic.

    Mirrors the demo: cars approach from different sides, pass through
    the intersection, then decelerate to a stop.
    """

    def __init__(self) -> None:
        self.current_sign: str = "YIELD"
        self._init_cars()

    # ── initialisation / reset ────────────────────────────────────────────

    def _init_cars(self) -> None:
        L = LANE_OFFSET_M

        # Player car — approaching from the West
        self.my_car = Car(
            x=-100.0, y=-L, speed=35.0,
            ml_direction="LEFT", approach="W",
            vx=1, vy=0,
        )

        # Traffic cars
        self.traffic: List[Car] = [
            Car(
                x=-L, y=100.0, speed=28.0,
                ml_direction="LEFT", approach="N",
                vx=0, vy=-1,
            ),
            Car(
                x=100.0, y=L, speed=40.0,
                ml_direction="RIGHT", approach="E",
                vx=-1, vy=0,
            ),
        ]

        self._finished = False

    def reset(self) -> None:
        """Re-initialise all cars so the scenario can be replayed."""
        self._init_cars()

    # ── queries ───────────────────────────────────────────────────────────

    def is_finished(self) -> bool:
        return self._finished

    def all_cars(self) -> List[Car]:
        return [self.my_car] + list(self.traffic)

    # ── physics tick ──────────────────────────────────────────────────────

    def update_physics(self, dt: float = 0.1) -> None:
        """Advance every car and handle pass-through / deceleration."""
        for car in self.all_cars():
            car.move(dt)
            if not car.passed and car.has_passed():
                car.passed = True

        all_passed = all(c.passed for c in self.all_cars())
        if all_passed:
            for car in self.all_cars():
                if car.stopped:
                    continue
                car.speed = max(0.0, car.speed - _DECEL_KMH_PER_S * dt)
                if car.speed < 0.5:
                    car.speed = 0.0
                    car.stopped = True

        if all(c.stopped for c in self.all_cars()):
            self._finished = True

    # ── ML interface ──────────────────────────────────────────────────────

    def get_ml_input(self) -> dict:
        """Return the dict expected by the ML model (uses FORWARD/LEFT/RIGHT)."""
        return {
            "my_car": self.my_car.as_dict(),
            "sign": self.current_sign,
            "traffic": [car.as_dict() for car in self.traffic],
        }


if __name__ == "__main__":
    world = World()
    world.update_physics()
    print(world.get_ml_input())
