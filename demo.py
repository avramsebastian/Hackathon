#!/usr/bin/env python3
"""
Quick demo — runs the Pygame view with fake vehicles so you can
see the UI without needing the backend ready.

Usage:
    python3 Hackathon/demo.py
"""

import random
import time


class DemoBus:
    """Fake bus that returns hardcoded vehicles and random ML decisions."""

    # How far past the centre (in metres) a vehicle must travel
    # before we consider it "through" the intersection.
    _PASS_THRESHOLD_M = 10.0
    # Deceleration applied once a vehicle is past (m/s²-ish, applied to km/h).
    _DECEL_KMH_PER_S = 60.0
    def __init__(self):
        self._init_vehicles()

    def _init_vehicles(self):
        self._last_time = time.time()

        # Each vehicle carries public fields (forwarded to the UI)
        # and private bookkeeping fields (prefixed with '_').
        # Lane centre offset in world metres (LANE_WIDTH_PX / 2 / PIXELS_PER_METER).
        # With LANE_WIDTH=28 and PPM=2.0 this is 7 m.
        L = 7.0

        self._vehicles = [
            {
                "id": "CAR-1",
                "x": -100.0,
                "y": -L,
                "speed": 35.0,
                "speed_unit": "kmh",
                "direction": "NORTH",
                "approach": "W",
                "color": (86, 168, 255),
                "road_line": [(-100, -L), (0, -L), (100, -L)],
                # internal
                "_vx": 1,
                "_vy": 0,
                "_passed": False,
                "_stopped": False,
            },
            {
                "id": "CAR-2",
                "x": -L,
                "y": 100.0,
                "speed": 28.0,
                "speed_unit": "kmh",
                "direction": "SOUTH",
                "approach": "N",
                "color": (255, 88, 88),
                "road_line": [(-L, 100), (-L, 0), (-L, -100)],
                "_vx": 0,
                "_vy": -1,
                "_passed": False,
                "_stopped": False,
            },
            {
                "id": "CAR-3",
                "x": 100.0,
                "y": L,
                "speed": 40.0,
                "speed_unit": "kmh",
                "direction": "WEST",
                "approach": "E",
                "color": (100, 226, 170),
                "road_line": [(100, L), (0, L), (-100, L)],
                "_vx": -1,
                "_vy": 0,
                "_passed": False,
                "_stopped": False,
            },
        ]
        self._decisions = {}
        self._finished = False

    # ----- helpers ---------------------------------------------------- #

    @staticmethod
    def _has_passed(v, threshold):
        """True once the vehicle is *threshold* metres past the origin."""
        if v["_vx"] > 0 and v["x"] > threshold:
            return True
        if v["_vx"] < 0 and v["x"] < -threshold:
            return True
        if v["_vy"] > 0 and v["y"] > threshold:
            return True
        if v["_vy"] < 0 and v["y"] < -threshold:
            return True
        return False

    # ----- bus interface ---------------------------------------------- #

    def get_vehicles(self):
        now = time.time()
        dt = min(now - self._last_time, 0.1)  # cap to avoid big jumps
        self._last_time = now

        for v in self._vehicles:
            if v["_stopped"]:
                v["speed"] = 0.0
                continue

            # Move according to declared speed
            speed_mps = v["speed"] / 3.6
            v["x"] += v["_vx"] * speed_mps * dt
            v["y"] += v["_vy"] * speed_mps * dt

            # Detect crossing the intersection
            if not v["_passed"] and self._has_passed(v, self._PASS_THRESHOLD_M):
                v["_passed"] = True

        # Wait until the LAST vehicle has passed before decelerating all
        all_passed = all(v["_passed"] for v in self._vehicles)

        if all_passed:
            for v in self._vehicles:
                if v["_stopped"]:
                    continue
                v["speed"] = max(0.0, v["speed"] - self._DECEL_KMH_PER_S * dt)
                if v["speed"] < 0.5:
                    v["speed"] = 0.0
                    v["_stopped"] = True

        # Mark finished once every vehicle has stopped
        if all(v["_stopped"] for v in self._vehicles):
            self._finished = True

        # Return only the public keys (strip '_'-prefixed bookkeeping)
        return [
            {k: val for k, val in veh.items() if not k.startswith("_")}
            for veh in self._vehicles
        ]

    def get_ml_decision(self, vehicle_id: str):
        # Simulate ML decisions with confidence values
        if vehicle_id not in self._decisions or random.random() < 0.02:
            go = random.uniform(0.3, 0.95)
            decision = "GO" if go > 0.5 else "STOP"
            self._decisions[vehicle_id] = {
                "decision": decision,
                "confidence_go": round(go, 2),
                "confidence_stop": round(1.0 - go, 2),
            }
        return self._decisions[vehicle_id]

    def get_intersection(self):
        return {
            "lanes": 2,
            "signs": {"N": "STOP", "S": "STOP", "E": "YIELD", "W": "YIELD"},
        }

    def is_finished(self) -> bool:
        return self._finished

    def reset(self):
        """Re-initialise all vehicles so the demo can be replayed."""
        self._init_vehicles()


if __name__ == "__main__":
    from ui import run_pygame_view

    print("Starting demo with fake data...")
    print("Controls: SPACE=pause  +/-=zoom  F3=debug  L=legend  F12=screenshot  R=reset")
    run_pygame_view(DemoBus())
