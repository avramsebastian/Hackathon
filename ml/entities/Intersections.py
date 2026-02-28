import math
from typing import List
from entities.Car import Car
from entities.Sign import Sign
from entities.Directions import Directions

class Intersections:
    def __init__(self, initial_car: Car, other_cars: List[Car], sign: Sign, max_tracked_cars: int = 6):
        self.initial_car = initial_car
        self.other_cars = other_cars
        self.sign = sign
        self.max_tracked_cars = max_tracked_cars

    def _one_hot_encode(self, index: int, length: int) -> List[float]:
        vector = [0.0] * length
        vector[index] = 1.0
        return vector
        
    def _get_linear_dist(self, car: Car) -> float:
        """Calculează distanța LINIARĂ inteligentă. 
           Valori pozitive = se apropie. Valori negative = a trecut deja."""
        if abs(car.y + 7.0) < 0.1: return -car.x   # EB 
        if abs(car.y - 7.0) < 0.1: return car.x    # WB 
        if abs(car.x - 7.0) < 0.1: return -car.y   # NB 
        if abs(car.x + 7.0) < 0.1: return car.y    # SB 
        return math.hypot(car.x, car.y) # Fallback 
    
    def get_feature_vector(self) -> List[float]:
        features = []

        # A. Ego Car (Aici punem distanța liniară în loc de hypot!)
        dist_mea = self._get_linear_dist(self.initial_car)
        features.extend([self.initial_car.x, self.initial_car.y, dist_mea, self.initial_car.speed])
        features.extend(self._one_hot_encode(self.initial_car.direction.value, 3))

        # B. Semnul de circulație
        features.extend(self._one_hot_encode(self.sign.value, 4))

        # C. Sortăm traficul după distanța liniară
        sorted_cars = sorted(self.other_cars, key=lambda c: self._get_linear_dist(c))
        closest_cars = sorted_cars[:self.max_tracked_cars]

        # D. Trafic 
        for car in closest_cars:
            dist_c = self._get_linear_dist(car)
            cross_product = (self.initial_car.x * car.y) - (self.initial_car.y * car.x)
            
            features.extend([car.x, car.y, dist_c, car.speed, cross_product])
            features.extend(self._one_hot_encode(car.direction.value, 3))

        # E. Zero-Padding 
        cars_missing = self.max_tracked_cars - len(closest_cars)
        for _ in range(cars_missing):
            features.extend([0.0] * 8)

        return features