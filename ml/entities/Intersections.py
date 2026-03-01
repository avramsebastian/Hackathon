import math
from typing import List
from entities.Car import Car
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions

class Intersection:
    def __init__(self, initial_car: Car, other_cars: List[Car], sign: Sign, traffic_light: TrafficLight, max_tracked_cars: int = 6):
        self.initial_car = initial_car
        self.other_cars = other_cars
        self.sign = sign
        self.traffic_light = traffic_light
        self.max_tracked_cars = max_tracked_cars

    def _one_hot_encode(self, index: int, length: int) -> List[float]:
        vector = [0.0] * length
        vector[index] = 1.0
        return vector

    def _is_horizontal(self, car: Car) -> bool:
        dy = min(abs(car.y - 7.0), abs(car.y + 7.0))
        dx = min(abs(car.x - 7.0), abs(car.x + 7.0))
        return dy <= dx
        
    def _get_linear_dist(self, car: Car) -> float:
        if self._is_horizontal(car):
            return -car.x if car.y < 0 else car.x
        else:
            return -car.y if car.x > 0 else car.y
    
    def get_feature_vector(self) -> List[float]:
        features = []
        dist_mea = self._get_linear_dist(self.initial_car)
        
        # Ego Car (adaugam role.value)
        features.extend([
            self.initial_car.x, self.initial_car.y, dist_mea, 
            self.initial_car.speed, float(self.initial_car.role.value)
        ])
        features.extend(self._one_hot_encode(self.initial_car.direction.value, 3))
        
        features.extend(self._one_hot_encode(self.sign.value, 4))
        features.extend(self._one_hot_encode(self.traffic_light.value, 4))

        sorted_cars = sorted(self.other_cars, key=lambda c: abs(self._get_linear_dist(c)))
        closest_cars = sorted_cars[:self.max_tracked_cars]

        for car in closest_cars:
            dist_c = self._get_linear_dist(car)
            cross_product = (self.initial_car.x * car.y) - (self.initial_car.y * car.x)
            # Trafic: adaugam role.value la final
            features.extend([
                car.x, car.y, dist_c, car.speed, 
                cross_product, float(car.role.value)
            ])
            features.extend(self._one_hot_encode(car.direction.value, 3))

        # Padding (9 feature-uri per mașină lipsă)
        for _ in range(self.max_tracked_cars - len(closest_cars)):
            features.extend([0.0] * 9)
            
        return features