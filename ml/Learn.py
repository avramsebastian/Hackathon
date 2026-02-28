import math
from enum import Enum
from dataclasses import dataclass
from typing import List
from Directions import Directions
from Sign import Sign
from Car import Car

# 3. Clasa care procesează situația și o pregătește pentru Modelul de AI
class IntersectionState:
    def __init__(self, initial_car: Car, other_cars: List[Car], sign: Sign, max_tracked_cars: int = 2):
        self.initial_car = initial_car
        self.other_cars = other_cars
        self.sign = sign
        self.max_tracked_cars = max_tracked_cars # Câte alte mașini urmărim maxim

    def _one_hot_encode(self, index: int, length: int) -> List[float]:
        """Transformă un index într-un vector one-hot (ex: index 1 din 3 -> [0.0, 1.0, 0.0])"""
        vector = [0.0] * length
        vector[index] = 1.0
        return vector

    def get_feature_vector(self) -> List[float]:
        """Aplatizează toate datele într-un singur șir de numere pentru rețeaua neurală"""
        features = []

        # --- A. Adăugăm mașina noastră (InitialCar) ---
        features.extend([self.initial_car.x, self.initial_car.y, self.initial_car.speed])
        features.extend(self._one_hot_encode(self.initial_car.direction.value, 3))

        # --- B. Adăugăm Semnul de circulație ---
        features.extend(self._one_hot_encode(self.sign.value, 4))

        # --- C. Sortăm celelalte mașini ca să le luăm pe cele mai apropiate ---
        # Le ordonăm crescător după distanța față de centru
        sorted_cars = sorted(self.other_cars, key=lambda c: c.distance_to_center())
        
        # Păstrăm doar numărul maxim setat (ex: primele 2)
        closest_cars = sorted_cars[:self.max_tracked_cars]

        # --- D. Adăugăm celelalte mașini în vector ---
        for car in closest_cars:
            features.extend([car.x, car.y, car.speed])
            features.extend(self._one_hot_encode(car.direction.value, 3))

        # --- E. Zero-Padding ---
        # Dacă sunt mai puține mașini decât maximul admis, completăm cu zerouri
        cars_missing = self.max_tracked_cars - len(closest_cars)
        for _ in range(cars_missing):
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, speed + 3 dir

        return features

# ==========================================
# EXEMPLU DE UTILIZARE
# ==========================================
if __name__ == "__main__":
    # Mașina ta se apropie de intersecție
    my_car = Car(x=0.0, y=-15.0, speed=10.0, direction=Directions.FORWARD)

    # Alte mașini din trafic
    traffic = [
        Car(x=5.0, y=2.0, speed=8.0, direction=Directions.LEFT),    # Aproape
        Car(x=-30.0, y=0.0, speed=15.0, direction=Directions.RIGHT), # Departe
        Car(x=10.0, y=-5.0, speed=5.0, direction=Directions.FORWARD) # Destul de aproape
    ]

    # Intersecție cu Cedează Trecerea
    current_sign = Sign.YIELD

    # Creăm starea
    state = IntersectionState(initial_car=my_car, other_cars=traffic, sign=current_sign, max_tracked_cars=2)

    # Generăm datele pentru model
    model_input = state.get_feature_vector()

    print(f"Numărul total de parametri de intrare: {len(model_input)}")
    print(f"Vectorul pentru model:\n{model_input}")