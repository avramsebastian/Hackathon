#!/usr/bin/env python3
"""
Safety tests for world-level collision prevention and control application.
"""

from __future__ import annotations

import math
import unittest

from sim.network import IntersectionNode, RoadNetwork, RoadSegment
from sim.traffic_policy import SafetyPolicy
from sim.world import Car, World

_EMERGENCY_ROLES = {"ambulance", "police", "fire"}


class WorldSafetyTests(unittest.TestCase):
    def _assert_priority_equals_emergency_role(self, world: World) -> None:
        for car in world.cars:
            is_emergency_role = str(car.role).lower() in _EMERGENCY_ROLES
            self.assertEqual(
                is_emergency_role,
                bool(car.priority),
                msg=f"{car.id}: role={car.role} priority={car.priority}",
            )

    def test_collision_guard_prevents_overlap(self) -> None:
        world = World(
            num_cars=1,
            seed=1,
            policy=SafetyPolicy(
                world_collision_guard_enabled=True,
                world_overlap_resolver_enabled=True,
            ),
        )
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

        self.assertGreaterEqual(min_dist, world.policy.min_pair_distance_m - 0.01)
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

        self.assertLess(car.speed, initial)
        self.assertLessEqual(car.speed, world.policy.ml_stop_target_speed_kmh + 0.2)

    def test_transition_updates_arrival_approach(self) -> None:
        network = RoadNetwork(
            intersections=[
                IntersectionNode(id="INT_A", cx=0.0, cy=0.0),
                IntersectionNode(id="INT_B", cx=150.0, cy=0.0),
            ],
            roads=[
                RoadSegment(
                    from_id="INT_A", from_arm="E",
                    to_id="INT_B", to_arm="N",
                )
            ],
        )
        world = World(num_cars=1, seed=11, network=network)
        world.cars = [
            Car(
                id="CAR_TRANS",
                x=11.0,
                y=-7.0,
                speed=0.0,
                ml_direction="FORWARD",
                approach="W",
                cruise_speed=0.0,
                vx=1.0,
                vy=0.0,
                current_int_id="INT_A",
            )
        ]

        world.update_physics(dt=0.1, decisions={"CAR_TRANS": {"decision": "GO"}})
        car = world.cars[0]
        self.assertEqual(car.current_int_id, "INT_B")
        self.assertEqual(car.approach, "N")

    def test_priority_cadence_and_unique_intersections(self) -> None:
        old_counter = World._scenario_counter
        try:
            World._scenario_counter = 0
            world = World(num_cars=6, seed=12345)

            # Scenario 1 (odd): no priority cars
            c1 = [car for car in world.cars if car.priority]
            self.assertEqual(len(c1), 0)
            self._assert_priority_equals_emergency_role(world)

            # Scenario 2 (even): at least one
            world.reset()
            c2 = [car for car in world.cars if car.priority]
            self.assertGreaterEqual(len(c2), 1)
            self._assert_priority_equals_emergency_role(world)

            # Scenario 3 (odd): no priority cars
            world.reset()
            c3 = [car for car in world.cars if car.priority]
            self.assertEqual(len(c3), 0)
            self._assert_priority_equals_emergency_role(world)

            # Scenario 4 (multiple): at least two when enough intersections exist
            world.reset()
            c4 = [car for car in world.cars if car.priority]
            unique_ints = {car.current_int_id for car in world.cars}
            if len(unique_ints) >= 2:
                self.assertGreaterEqual(len(c4), 2)
            else:
                self.assertLessEqual(len(c4), 1)

            # Never more than one priority car in the same intersection
            prio_ints = [car.current_int_id for car in c4]
            self.assertEqual(len(prio_ints), len(set(prio_ints)))
            self._assert_priority_equals_emergency_role(world)
        finally:
            World._scenario_counter = old_counter

    def test_runtime_enforces_one_priority_per_intersection(self) -> None:
        world = World(num_cars=1, seed=77)
        int_id = next(iter(world.network.intersections.keys()))
        x1, y1, vx, vy = world.network.arm_spawn_position(
            int_id, "W", 40.0, world.policy.lane_offset_m
        )
        x2, y2, _, _ = world.network.arm_spawn_position(
            int_id, "W", 55.0, world.policy.lane_offset_m
        )
        world.cars = [
            Car(
                id="CAR_P1",
                x=x1,
                y=y1,
                speed=30.0,
                ml_direction="FORWARD",
                approach="W",
                cruise_speed=30.0,
                vx=vx,
                vy=vy,
                current_int_id=int_id,
                priority=True,
            ),
            Car(
                id="CAR_P2",
                x=x2,
                y=y2,
                speed=25.0,
                ml_direction="FORWARD",
                approach="W",
                cruise_speed=25.0,
                vx=vx,
                vy=vy,
                current_int_id=int_id,
                priority=True,
            ),
        ]

        world.update_physics(dt=0.1, decisions={})
        prio_same_int = [
            car for car in world.cars if car.priority and car.current_int_id == int_id
        ]
        self.assertEqual(len(prio_same_int), 1)
        self._assert_priority_equals_emergency_role(world)

    def test_passed_priority_car_does_not_demote_incoming_priority(self) -> None:
        world = World(num_cars=1, seed=88)
        int_id = next(iter(world.network.intersections.keys()))
        node = world.network.intersections[int_id]

        # Car that already cleared this intersection and is far away.
        exited = Car(
            id="CAR_EXITED",
            x=node.cx + 220.0,
            y=node.cy + 7.0,
            speed=65.0,
            ml_direction="FORWARD",
            approach="W",
            role="ambulance",
            priority=True,
            cruise_speed=65.0,
            vx=1.0,
            vy=0.0,
            current_int_id=int_id,
            passed=True,
        )

        # Priority car currently approaching the same intersection.
        incoming = Car(
            id="CAR_INCOMING",
            x=node.cx - 45.0,
            y=node.cy - 7.0,
            speed=40.0,
            ml_direction="FORWARD",
            approach="W",
            role="police",
            priority=True,
            cruise_speed=40.0,
            vx=1.0,
            vy=0.0,
            current_int_id=int_id,
            passed=False,
        )

        world.cars = [exited, incoming]
        world.update_physics(dt=0.1, decisions={})

        self.assertTrue(incoming.priority)
        self._assert_priority_equals_emergency_role(world)

    def test_priority_vehicle_does_not_brake_for_red_semaphore(self) -> None:
        network = RoadNetwork(
            intersections=[
                IntersectionNode(
                    id="INT_A", cx=0.0, cy=0.0, has_semaphore=True, priority_axis="EW"
                ),
            ],
            roads=[],
        )
        world = World(num_cars=1, seed=999, network=network)
        node = world.network.intersections["INT_A"]
        node.sem_green_axis = "EW"
        node.sem_phase = "GREEN"  # NS approaches are RED in this state
        node.sem_timer = 10.0

        world.cars = [
            Car(
                id="CAR_EM",
                x=-7.0,
                y=20.0,
                speed=40.0,
                ml_direction="FORWARD",
                approach="N",
                role="ambulance",
                priority=True,
                cruise_speed=40.0,
                vx=0.0,
                vy=-1.0,
                current_int_id="INT_A",
            )
        ]

        world.update_physics(dt=0.1, decisions={"CAR_EM": {"decision": "GO"}})
        car = world.cars[0]
        self.assertGreaterEqual(car.speed, 39.9)
        self._assert_priority_equals_emergency_role(world)

    def test_priority_vehicle_ignores_stop_sign_and_ml_stop(self) -> None:
        network = RoadNetwork(
            intersections=[
                IntersectionNode(
                    id="INT_A", cx=0.0, cy=0.0, has_semaphore=False, priority_axis="EW"
                ),
            ],
            roads=[],
        )
        world = World(num_cars=1, seed=1001, network=network)
        world.cars = [
            Car(
                id="CAR_EM_STOP",
                x=-7.0,
                y=20.0,
                speed=40.0,
                ml_direction="FORWARD",
                approach="N",  # STOP on EW-priority sign layout
                role="ambulance",
                priority=True,
                cruise_speed=40.0,
                vx=0.0,
                vy=-1.0,
                current_int_id="INT_A",
            )
        ]

        world.update_physics(dt=0.1, decisions={"CAR_EM_STOP": {"decision": "STOP"}})
        car = world.cars[0]
        self.assertGreaterEqual(car.speed, 39.9)
        self.assertTrue(car.stop_completed)
        self._assert_priority_equals_emergency_role(world)

    def test_priority_vehicle_ignores_yield_sign(self) -> None:
        network = RoadNetwork(
            intersections=[
                IntersectionNode(
                    id="INT_A", cx=0.0, cy=0.0, has_semaphore=False, priority_axis="EW"
                ),
            ],
            roads=[],
        )
        world = World(num_cars=1, seed=1003, network=network)
        world.cars = [
            Car(
                id="CAR_EM_YIELD",
                x=7.0,
                y=-20.0,
                speed=40.0,
                ml_direction="FORWARD",
                approach="S",  # YIELD on EW-priority sign layout
                role="ambulance",
                priority=True,
                cruise_speed=40.0,
                vx=0.0,
                vy=1.0,
                current_int_id="INT_A",
            )
        ]

        world.update_physics(dt=0.1, decisions={"CAR_EM_YIELD": {"decision": "GO"}})
        car = world.cars[0]
        self.assertGreaterEqual(car.speed, 39.9)
        self.assertTrue(car.stop_completed)
        self._assert_priority_equals_emergency_role(world)

    def test_pick_yielder_prefers_non_emergency_for_crossing_conflict(self) -> None:
        network = RoadNetwork(
            intersections=[
                IntersectionNode(
                    id="INT_A", cx=0.0, cy=0.0, has_semaphore=False, priority_axis="EW"
                ),
            ],
            roads=[],
        )
        world = World(num_cars=1, seed=1002, network=network)

        emergency = Car(
            id="CAR_EM",
            x=-30.0,
            y=-7.0,
            speed=35.0,
            ml_direction="FORWARD",
            approach="W",
            role="police",
            priority=True,
            cruise_speed=35.0,
            vx=1.0,
            vy=0.0,
            current_int_id="INT_A",
        )
        civilian = Car(
            id="CAR_CIV",
            x=-7.0,
            y=30.0,
            speed=35.0,
            ml_direction="FORWARD",
            approach="N",
            role="civilian",
            priority=False,
            cruise_speed=35.0,
            vx=0.0,
            vy=-1.0,
            current_int_id="INT_A",
        )
        world.cars = [emergency, civilian]

        yielder = world._pick_yielder(emergency, civilian)
        self.assertIs(yielder, civilian)


if __name__ == "__main__":
    unittest.main()
