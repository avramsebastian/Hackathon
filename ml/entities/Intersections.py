import math
from typing import List

from entities.Car import Car
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions

_LANE_OFFSET = 7.0

class Intersections:
    def __init__(
        self,
        initial_car: Car,
        other_cars: List[Car],
        sign: Sign,
        traffic_light: TrafficLight, # <--- Parametru NOU
        max_tracked_cars: int = 6,
    ):
        self.initial_car = initial_car
        self.other_cars = other_cars
        self.sign = sign
        self.traffic_light = traffic_light
        self.max_tracked_cars = max_tracked_cars

    @staticmethod
    def _one_hot_encode(index: int, length: int) -> List[float]:
        vector = [0.0] * length
        vector[index] = 1.0
        return vector

    @staticmethod
    def _get_linear_dist(car: Car) -> float:
        if abs(car.y + _LANE_OFFSET) < 0.1: return -car.x   
        if abs(car.y - _LANE_OFFSET) < 0.1: return car.x    
        if abs(car.x - _LANE_OFFSET) < 0.1: return -car.y   
        if abs(car.x + _LANE_OFFSET) < 0.1: return car.y    
        return math.hypot(car.x, car.y)  

    def get_feature_vector(self) -> List[float]:
        features: List[float] = []

        # A. Ego car (7 parametri)
        ego_dist = self._get_linear_dist(self.initial_car)
        features.extend([
            self.initial_car.x, self.initial_car.y, ego_dist, self.initial_car.speed,
        ])
        features.extend(self._one_hot_encode(self.initial_car.direction.value, 3))

        # B. Traffic sign (4 parametri)
        features.extend(self._one_hot_encode(self.sign.value, 4))
        
        # C. Traffic Light / Semafor (4 parametri NOI)
        features.extend(self._one_hot_encode(self.traffic_light.value, 4))

        # D. Sort traffic by axial distance
        sorted_cars = sorted(self.other_cars, key=self._get_linear_dist)
        closest_cars = sorted_cars[: self.max_tracked_cars]

        # E. Neighbour features (48 parametri)
        for car in closest_cars:
            dist_c = self._get_linear_dist(car)
            cross_product = self.initial_car.x * car.y - self.initial_car.y * car.x
            features.extend([car.x, car.y, dist_c, car.speed, cross_product])
            features.extend(self._one_hot_encode(car.direction.value, 3))

        cars_missing = self.max_tracked_cars - len(closest_cars)
        features.extend([0.0] * (8 * cars_missing))

        return features