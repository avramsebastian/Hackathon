import csv, random, sys, os
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path: sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions

LANE_OFFSET = 7.0       
MAX_CARS = 6             
TOTAL_FEATURES = 63      # <--- Actualizat la 63!
ACTION_ZONE = 65.0       

class TrafficDataGenerator:
    @staticmethod
    def _spawn_random_car() -> Car:
        lane = random.choice(["EB", "WB", "NB", "SB"])
        is_approaching = random.random() < 0.7
        dist = random.uniform(1, 120)
        speed = random.uniform(10, 50)
        direction = random.choice(list(Directions))
        if lane == "EB": return Car(x=-dist if is_approaching else dist, y=-LANE_OFFSET, speed=speed, direction=direction)
        elif lane == "WB": return Car(x=dist if is_approaching else -dist, y=LANE_OFFSET, speed=speed, direction=direction)
        elif lane == "NB": return Car(x=LANE_OFFSET, y=-dist if is_approaching else dist, speed=speed, direction=direction)
        else: return Car(x=-LANE_OFFSET, y=dist if is_approaching else -dist, speed=speed, direction=direction)

    @staticmethod
    def _axial_distance_to_centre(car: Car) -> float:
        if abs(car.y + LANE_OFFSET) < 0.1: return -car.x   
        if abs(car.y - LANE_OFFSET) < 0.1: return car.x    
        if abs(car.x - LANE_OFFSET) < 0.1: return -car.y   
        if abs(car.x + LANE_OFFSET) < 0.1: return car.y    
        return -100.0       

    def generate(self, file_path: str, num_scenarios: int) -> None:
        print(f"Generăm {num_scenarios} scenarii cu SEMAFOARE ...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode="w", newline="") as fh:
            writer = csv.writer(fh)
            header = [f"feature_{i + 1}" for i in range(TOTAL_FEATURES)] + ["label"]
            writer.writerow(header)

            for _ in range(num_scenarios):
                my_car = self._spawn_random_car()
                sign = random.choice(list(Sign))
                # Am adăugat șansa de a genera un semafor aleatoriu
                traffic_light = random.choice(list(TrafficLight)) 
                
                num_other_cars = random.randint(1, MAX_CARS)
                traffic = [self._spawn_random_car() for _ in range(num_other_cars)]

                state = Intersections(my_car, traffic, sign, traffic_light, max_tracked_cars=MAX_CARS)
                features = state.get_feature_vector()
                label = 1  
                ego_dist = self._axial_distance_to_centre(my_car)

                if ego_dist > 10.0 and ego_dist <= ACTION_ZONE:
                    dangerous = [c for c in traffic if self._axial_distance_to_centre(c) > -10.0]

                    # 1. LOGICA PENTRU SEMAFOR (Are prioritate absolută)
                    if traffic_light == TrafficLight.RED:
                        label = 0 # La roșu stai
                    
                    elif traffic_light == TrafficLight.YELLOW:
                        if ego_dist > 25.0: 
                            label = 0 # Dacă ești departe, oprești la galben
                    
                    elif traffic_light == TrafficLight.GREEN:
                        # La verde treci, OPREȘTI DOAR dacă cineva a blocat fizic intersecția
                        for c in dangerous:
                            d_c = self._axial_distance_to_centre(c)
                            if -5.0 < d_c < 10.0:
                                label = 0
                                break

                    # 2. Dacă NU ESTE SEMAFOR, respectăm semnele
                    elif traffic_light == TrafficLight.NONE:
                        if sign == Sign.STOP:
                            if ego_dist > 12.0:
                                label = 0  
                            else:
                                if any(self._axial_distance_to_centre(c) < 40.0 for c in dangerous):
                                    label = 0
                        elif sign in (Sign.YIELD, Sign.NO_SIGN):
                            for c in dangerous:
                                dist_c = self._axial_distance_to_centre(c)
                                if dist_c < ACTION_ZONE + 20.0:
                                    if dist_c < ego_dist - 5.0:
                                        label = 0; break
                                    elif dist_c <= ego_dist + 5.0:
                                        cross = my_car.x * c.y - my_car.y * c.x
                                        if cross > 0:  
                                            label = 0; break

                writer.writerow(features + [label])
        print(f"  → Salvat în '{os.path.basename(file_path)}'\n")

if __name__ == "__main__":
    generator = TrafficDataGenerator()
    folder_generated = os.path.join(_ML_ROOT, "generated")
    # Generăm mai multe ca să învețe bine noul parametru
    generator.generate(os.path.join(folder_generated, "train_dataset.csv"), 10000)
    generator.generate(os.path.join(folder_generated, "val_dataset.csv"), 1500)