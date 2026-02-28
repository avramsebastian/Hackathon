import csv
import random
import sys
import os

cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if cale_ml not in sys.path:
    sys.path.append(cale_ml)
    sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

LANE_OFFSET = 7.0
MAX_CARS = 6
TOTAL_FEATURES = 59
ACTION_ZONE = 65.0      

class TrafficDataGenerator:
    @staticmethod
    def _spawn_random_car() -> Car:
        lane = random.choice(['EB', 'WB', 'NB', 'SB'])
        is_approaching = random.random() < 0.7 
        dist = random.uniform(1, 120) 
        speed = random.uniform(10, 50)
        direction = random.choice(list(Directions)) 
        
        if lane == 'EB':   
            x = -dist if is_approaching else dist
            return Car(x=x, y=-LANE_OFFSET, speed=speed, direction=direction)
        elif lane == 'WB': 
            x = dist if is_approaching else -dist
            return Car(x=x, y=LANE_OFFSET, speed=speed, direction=direction)
        elif lane == 'NB': 
            y = -dist if is_approaching else dist
            return Car(x=LANE_OFFSET, y=y, speed=speed, direction=direction)
        else:              
            y = dist if is_approaching else -dist
            return Car(x=-LANE_OFFSET, y=y, speed=speed, direction=direction)

    @staticmethod
    def _dist_pina_la_centru(car: Car) -> float:
        if abs(car.y + LANE_OFFSET) < 0.1: return -car.x   
        if abs(car.y - LANE_OFFSET) < 0.1: return car.x    
        if abs(car.x - LANE_OFFSET) < 0.1: return -car.y   
        if abs(car.x + LANE_OFFSET) < 0.1: return car.y    
        return -100.0 

    def generate(self, file_path: str, num_scenarios: int):
        print(f"Generăm {num_scenarios} scenarii (Vectorizat Liniar)...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = [f"feature_{i+1}" for i in range(TOTAL_FEATURES)] + ["label"]
            writer.writerow(header)
            
            for _ in range(num_scenarios):
                my_car = self._spawn_random_car()
                sign = random.choice(list(Sign))
                
                num_other_cars = random.randint(1, MAX_CARS) 
                traffic = [self._spawn_random_car() for _ in range(num_other_cars)]
                
                state = Intersections(my_car, traffic, sign, max_tracked_cars=MAX_CARS)
                features = state.get_feature_vector()
                
                label = 1 # Default: GO!
                
                d_centru = self._dist_pina_la_centru(my_car)
                
                # REGULA SUPREMĂ: Dacă am depășit 10m (adică am intrat în intersecție), label rămâne 1 (GO) obligatoriu.
                if d_centru > 10.0:
                    
                    if d_centru <= ACTION_ZONE:
                        trafic_periculos = [c for c in traffic if self._dist_pina_la_centru(c) > -10.0]
                        
                        if sign == Sign.STOP:
                            if d_centru > 12.0:
                                label = 0 
                            else:
                                if any(self._dist_pina_la_centru(c) < 40.0 for c in trafic_periculos):
                                    label = 0
                        
                        elif sign == Sign.YIELD or sign == Sign.NO_SIGN:
                            for c in trafic_periculos:
                                dist_c = self._dist_pina_la_centru(c)
                                if dist_c < ACTION_ZONE + 20.0:
                                    if dist_c < d_centru - 5.0:
                                        label = 0
                                        break
                                    elif dist_c <= d_centru + 5.0:
                                        cross_product = (my_car.x * c.y) - (my_car.y * c.x)
                                        if cross_product > 0: 
                                            label = 0 
                                            break
                
                writer.writerow(features + [label])
                
        print(f"-> Salvat cu succes în '{os.path.basename(file_path)}'!\n")

if __name__ == "__main__":
    generator = TrafficDataGenerator()
    folder_generated = os.path.join(cale_ml, "generated")
    generator.generate(os.path.join(folder_generated, "train_dataset.csv"), 8000)
    generator.generate(os.path.join(folder_generated, "val_dataset.csv"), 1500)