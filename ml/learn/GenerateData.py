"""
ml/learn/GenerateData.py
========================
Synthetic dataset generator for the intersection collision-risk model.

Each row is a 59-feature vector (see :class:`ml.entities.Intersections`)
plus a binary label (0 = STOP, 1 = GO) determined by hand-crafted
traffic rules.

Usage::

    python ml/learn/GenerateData.py
"""

import csv
import random
import sys
import os

# ── path setup ────────────────────────────────────────────────────────────────
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path:
        sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

# ── constants ─────────────────────────────────────────────────────────────────
LANE_OFFSET = 7.0       # metres from road axis to lane centre
MAX_CARS = 6             # max neighbour cars encoded in the feature vector
TOTAL_FEATURES = 59      # length of the feature vector
ACTION_ZONE = 65.0       # metres — decisions only matter inside this radius


class TrafficDataGenerator:
    """Generate labelled CSV datasets of intersection scenarios."""

    @staticmethod
    def _spawn_random_car() -> Car:
        """Create a car on a random lane at a random distance."""
        lane = random.choice(["EB", "WB", "NB", "SB"])
        is_approaching = random.random() < 0.7
        dist = random.uniform(1, 120)
        speed = random.uniform(10, 50)
        direction = random.choice(list(Directions))

        if lane == "EB":
            x = -dist if is_approaching else dist
            return Car(x=x, y=-LANE_OFFSET, speed=speed, direction=direction)
        elif lane == "WB":
            x = dist if is_approaching else -dist
            return Car(x=x, y=LANE_OFFSET, speed=speed, direction=direction)
        elif lane == "NB":
            y = -dist if is_approaching else dist
            return Car(x=LANE_OFFSET, y=y, speed=speed, direction=direction)
        else:  # SB
            y = dist if is_approaching else -dist
            return Car(x=-LANE_OFFSET, y=y, speed=speed, direction=direction)

    @staticmethod
    def _axial_distance_to_centre(car: Car) -> float:
        """Signed axial distance to the intersection centre.

        Positive = approaching, negative = already past.
        """
        if abs(car.y + LANE_OFFSET) < 0.1:
            return -car.x   # eastbound
        if abs(car.y - LANE_OFFSET) < 0.1:
            return car.x    # westbound
        if abs(car.x - LANE_OFFSET) < 0.1:
            return -car.y   # northbound
        if abs(car.x + LANE_OFFSET) < 0.1:
            return car.y    # southbound
        return -100.0       # off-lane fallback

    def generate(self, file_path: str, num_scenarios: int) -> None:
        """Write *num_scenarios* labelled rows to *file_path* (CSV).

        Label rules:
        - If the car is already inside the intersection (< 10 m) → GO.
        - If within the action zone and facing a STOP sign → STOP until
          close (< 12 m) and no dangerous traffic.
        - YIELD / NO_SIGN → STOP if another car is closer or on the right.
        """
        print(f"Generating {num_scenarios} scenarios …")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode="w", newline="") as fh:
            writer = csv.writer(fh)
            header = [f"feature_{i + 1}" for i in range(TOTAL_FEATURES)] + ["label"]
            writer.writerow(header)

            for _ in range(num_scenarios):
                my_car = self._spawn_random_car()
                sign = random.choice(list(Sign))

                num_other_cars = random.randint(1, MAX_CARS)
                traffic = [self._spawn_random_car() for _ in range(num_other_cars)]

                state = Intersections(my_car, traffic, sign, max_tracked_cars=MAX_CARS)
                features = state.get_feature_vector()

                label = 1  # default: GO

                ego_dist = self._axial_distance_to_centre(my_car)

                # Core rule: if the car has entered the intersection (< 10 m),
                # it must keep going — label stays 1 (GO).
                if ego_dist > 10.0 and ego_dist <= ACTION_ZONE:
                    dangerous = [
                        c for c in traffic
                        if self._axial_distance_to_centre(c) > -10.0
                    ]

                    if sign == Sign.STOP:
                        if ego_dist > 12.0:
                            label = 0  # must stop before the line
                        else:
                            if any(self._axial_distance_to_centre(c) < 40.0 for c in dangerous):
                                label = 0

                    elif sign in (Sign.YIELD, Sign.NO_SIGN):
                        for c in dangerous:
                            dist_c = self._axial_distance_to_centre(c)
                            if dist_c < ACTION_ZONE + 20.0:
                                if dist_c < ego_dist - 5.0:
                                    label = 0
                                    break
                                elif dist_c <= ego_dist + 5.0:
                                    cross = my_car.x * c.y - my_car.y * c.x
                                    if cross > 0:  # other car is on the right
                                        label = 0
                                        break

                writer.writerow(features + [label])

        print(f"  → Saved to '{os.path.basename(file_path)}'\n")


if __name__ == "__main__":
    generator = TrafficDataGenerator()
    folder_generated = os.path.join(_ML_ROOT, "generated")
    generator.generate(os.path.join(folder_generated, "train_dataset.csv"), 8000)
    generator.generate(os.path.join(folder_generated, "val_dataset.csv"), 1500)