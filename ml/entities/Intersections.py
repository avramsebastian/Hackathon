"""
ml/entities/Intersections.py
============================
Feature extraction for the intersection collision-risk model.

The :class:`Intersections` class converts an ego car, a list of nearby
traffic and a sign type into a fixed-length numeric feature vector
consumed by the Random Forest classifier.
"""

import math
from typing import List

from entities.Car import Car
from entities.Sign import Sign
from entities.Directions import Directions

# Lane offset (metres) — must stay in sync with sim/traffic_policy.py
_LANE_OFFSET = 7.0


class Intersections:
    """Build a feature vector from an ego car, traffic and sign.

    Parameters
    ----------
    initial_car : Car
        The ego vehicle.
    other_cars : list[Car]
        All other vehicles in the neighbourhood.
    sign : Sign
        Traffic sign visible to the ego car.
    max_tracked_cars : int
        Maximum number of neighbour cars to encode (zero-padded).
    """

    def __init__(
        self,
        initial_car: Car,
        other_cars: List[Car],
        sign: Sign,
        max_tracked_cars: int = 6,
    ):
        self.initial_car = initial_car
        self.other_cars = other_cars
        self.sign = sign
        self.max_tracked_cars = max_tracked_cars

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _one_hot_encode(index: int, length: int) -> List[float]:
        """Return a one-hot vector of *length* with a 1 at *index*."""
        vector = [0.0] * length
        vector[index] = 1.0
        return vector

    @staticmethod
    def _get_linear_dist(car: Car) -> float:
        """Axial (lane-aligned) distance to the intersection centre.

        Positive = approaching, negative = already past.
        Falls back to Euclidean distance when the car is not on a
        recognised lane.
        """
        if abs(car.y + _LANE_OFFSET) < 0.1:
            return -car.x   # eastbound
        if abs(car.y - _LANE_OFFSET) < 0.1:
            return car.x    # westbound
        if abs(car.x - _LANE_OFFSET) < 0.1:
            return -car.y   # northbound
        if abs(car.x + _LANE_OFFSET) < 0.1:
            return car.y    # southbound
        return math.hypot(car.x, car.y)  # fallback

    # ── public API ────────────────────────────────────────────────────────

    def get_feature_vector(self) -> List[float]:
        """Return the 59-element feature vector for the ML model.

        Layout (59 floats total):
            [0..6]   ego car  — x, y, linear_dist, speed, direction (one-hot 3)
            [7..10]  sign     — one-hot (4)
            [11..58] traffic  — up to 6 cars × 8 features each
                     per car: x, y, linear_dist, speed, cross_product,
                              direction (one-hot 3)
        """
        features: List[float] = []

        # A. Ego car
        ego_dist = self._get_linear_dist(self.initial_car)
        features.extend([
            self.initial_car.x,
            self.initial_car.y,
            ego_dist,
            self.initial_car.speed,
        ])
        features.extend(self._one_hot_encode(self.initial_car.direction.value, 3))

        # B. Traffic sign
        features.extend(self._one_hot_encode(self.sign.value, 4))

        # C. Sort traffic by axial distance, keep closest N
        sorted_cars = sorted(self.other_cars, key=self._get_linear_dist)
        closest_cars = sorted_cars[: self.max_tracked_cars]

        # D. Neighbour features
        for car in closest_cars:
            dist_c = self._get_linear_dist(car)
            cross_product = (
                self.initial_car.x * car.y - self.initial_car.y * car.x
            )
            features.extend([car.x, car.y, dist_c, car.speed, cross_product])
            features.extend(self._one_hot_encode(car.direction.value, 3))

        # E. Zero-padding for missing cars
        cars_missing = self.max_tracked_cars - len(closest_cars)
        features.extend([0.0] * (8 * cars_missing))

        return features