#!/usr/bin/env python3
"""
Safety tests for world-level collision prevention and control application.
"""

from __future__ import annotations

import math
import unittest

from sim.world import Car, World, _CAR_PAIR_SAFE_DIST_M


class WorldSafetyTests(unittest.TestCase):
    def test_collision_guard_prevents_overlap(self) -> None:
        world = World(num_cars=1, seed=1)
        world.cars = [
            Car(
                id="CAR_A",
                x=-7.6,
                y=-7.0,
                speed=40.0,
                ml_direction="FORWARD",
                approach="W",
                cruise_speed=40.0,
                vx=1.0,
                vy=0.0,
            ),
            Car(
                id="CAR_B",
                x=-7.0,
                y=-6.5,
                speed=38.0,
                ml_direction="FORWARD",
                approach="N",
                cruise_speed=38.0,
                vx=0.0,
                vy=-1.0,
            ),
        ]

        decisions = {
            "CAR_A": {"decision": "GO"},
            "CAR_B": {"decision": "GO"},
        }

        min_dist = float("inf")
        for _ in range(60):
            world.update_physics(dt=0.1, decisions=decisions)
            d = math.hypot(
                world.cars[0].x - world.cars[1].x,
                world.cars[0].y - world.cars[1].y,
            )
            min_dist = min(min_dist, d)

        self.assertGreaterEqual(min_dist, _CAR_PAIR_SAFE_DIST_M - 0.01)
        self.assertGreater(world.safety_interventions + world.collision_resolutions, 0)

    def test_stop_decision_reduces_speed(self) -> None:
        world = World(num_cars=1, seed=7)
        world.cars = [
            Car(
                id="CAR_STOP",
                x=-10.0,
                y=-7.0,
                speed=36.0,
                ml_direction="FORWARD",
                approach="W",
                cruise_speed=36.0,
                vx=1.0,
                vy=0.0,
            )
        ]
        car = world.cars[0]
        initial = car.speed
        decisions = {car.id: {"decision": "STOP"}}

        for _ in range(20):
            world.update_physics(dt=0.1, decisions=decisions)

        self.assertLess(car.speed, initial * 0.35)


if __name__ == "__main__":
    unittest.main()
