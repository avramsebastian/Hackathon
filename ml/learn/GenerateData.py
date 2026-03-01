import csv, random, sys, os
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path: sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersection
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions
from entities.Role import Role

LANE_OFFSET = 7.0       
MAX_CARS = 6             
TOTAL_FEATURES = 70      # Actualizat pentru Role!
ACTION_ZONE = 65.0       

class TrafficDataGenerator:
    @staticmethod
    def _spawn_random_car() -> Car:
        lane = random.choice(["EB", "WB", "NB", "SB"])
        is_approaching = random.random() < 0.7
        dist = random.uniform(1, 120)
        speed = 0.0 if random.random() < 0.2 else random.uniform(5, 50)
        direction = random.choice(list(Directions))
        
        # Șansă de 5% ca mașina să fie Ambulanță/Poliție
        role = Role.EMERGENCY if random.random() < 0.05 else Role.CIVILIAN
        
        if lane == "EB": return Car(x=-dist if is_approaching else dist, y=-LANE_OFFSET, speed=speed, direction=direction, role=role)
        elif lane == "WB": return Car(x=dist if is_approaching else -dist, y=LANE_OFFSET, speed=speed, direction=direction, role=role)
        elif lane == "NB": return Car(x=LANE_OFFSET, y=-dist if is_approaching else dist, speed=speed, direction=direction, role=role)
        else: return Car(x=-LANE_OFFSET, y=dist if is_approaching else -dist, speed=speed, direction=direction, role=role)

    @staticmethod
    def _is_horizontal(car: Car) -> bool:
        return min(abs(car.y - 7.0), abs(car.y + 7.0)) <= min(abs(car.x - 7.0), abs(car.x + 7.0))

    @staticmethod
    def _dist_pina_la_centru(car: Car) -> float:
        if TrafficDataGenerator._is_horizontal(car): return -car.x if car.y < 0 else car.x
        return -car.y if car.x > 0 else car.y

    @staticmethod
    def _is_oncoming(my_car: Car, c: Car) -> bool:
        if TrafficDataGenerator._is_horizontal(my_car) != TrafficDataGenerator._is_horizontal(c): return False 
        if TrafficDataGenerator._is_horizontal(my_car): return (my_car.y * c.y) < 0 
        else: return (my_car.x * c.x) < 0

    def _este_pericol(self, my_car: Car, c: Car) -> bool:
        d_c = self._dist_pina_la_centru(c)
        if d_c < -5.0: return False 
        
        # Ambulanța/Poliția este MEREU un pericol, indiferent de pe ce bandă vine!
        if c.role == Role.EMERGENCY: return True
            
        if self._is_horizontal(my_car) != self._is_horizontal(c): return True
        if my_car.direction == Directions.LEFT and self._is_oncoming(my_car, c):
            if c.direction in [Directions.FORWARD, Directions.RIGHT]: return True
        return False

    def generate(self, file_path: str, num_scenarios: int) -> None:
        print(f"Generăm {num_scenarios} scenarii (EMERGENCY ROLE)...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, mode="w", newline="") as fh:
            writer = csv.writer(fh)
            header = [f"feature_{i + 1}" for i in range(TOTAL_FEATURES)] + ["label"]
            writer.writerow(header)

            for _ in range(num_scenarios):
                my_car = self._spawn_random_car()
                sign = random.choice(list(Sign))
                traffic_light = random.choice(list(TrafficLight)) 
                
                num_other_cars = random.randint(1, MAX_CARS)
                traffic = [self._spawn_random_car() for _ in range(num_other_cars)]

                state = Intersection(my_car, traffic, sign, traffic_light, max_tracked_cars=MAX_CARS)
                features = state.get_feature_vector()
                
                label = 1  
                ego_dist = self._dist_pina_la_centru(my_car)

                if ego_dist > 11.0:
                    if ego_dist <= ACTION_ZONE:
                        trafic_periculos = [c for c in traffic if self._este_pericol(my_car, c)]
                        urgente = [c for c in trafic_periculos if c.role == Role.EMERGENCY]

                        # =======================================================
                        # IERARHIA 0: VEHICULE DE URGENȚĂ (AMBULANȚĂ/POLIȚIE)
                        # =======================================================
                        if my_car.role == Role.EMERGENCY:
                            # Suntem ambulanța! Trecem pe roșu, ignorăm STOP.
                            # Oprim DOAR dacă intersecția e complet blocată fizic în fața noastră.
                            for c in trafic_periculos:
                                if -5.0 < self._dist_pina_la_centru(c) < 10.0:
                                    label = 0; break
                        
                        elif len(urgente) > 0:
                            # Vine o ambulanță în intersecție! Noi suntem civili.
                            # OPRIM OBLIGATORIU indiferent că avem verde sau prioritate!
                            label = 0

                        # =======================================================
                        # IERARHIA 1: SEMAFOR
                        # =======================================================
                        elif traffic_light == TrafficLight.RED:
                            label = 0  
                            
                        elif traffic_light == TrafficLight.YELLOW:
                            if ego_dist > 25.0: label = 0  
                                
                        elif traffic_light == TrafficLight.GREEN:
                            for c in trafic_periculos:
                                if -5.0 < self._dist_pina_la_centru(c) < 15.0:
                                    label = 0; break
                                    
                        # =======================================================
                        # IERARHIA 2: SEMNELE DE CIRCULAȚIE
                        # =======================================================
                        elif traffic_light == TrafficLight.NONE:
                            if sign == Sign.PRIORITY:
                                for c in trafic_periculos:
                                    if -5.0 < self._dist_pina_la_centru(c) < 15.0:
                                        label = 0; break
                                        
                            elif sign == Sign.STOP:
                                if ego_dist > 12.0: label = 0 
                                else:
                                    for c in trafic_periculos:
                                        if self._dist_pina_la_centru(c) < 65.0:
                                            label = 0; break
                                            
                            elif sign in (Sign.YIELD, Sign.NO_SIGN):
                                for c in trafic_periculos:
                                    d_c = self._dist_pina_la_centru(c)
                                    if d_c < ACTION_ZONE + 15.0:
                                        if d_c < ego_dist - 3.0: label = 0; break
                                        elif d_c <= ego_dist + 5.0:
                                            if self._is_oncoming(my_car, c): label = 0; break
                                            else:
                                                if my_car.x * c.y - my_car.y * c.x > 0: label = 0; break

                writer.writerow(features + [label])
                
        print(f"  → Set de date salvat cu succes în '{os.path.basename(file_path)}'\n")

if __name__ == "__main__":
    generator = TrafficDataGenerator()
    folder_generated = os.path.join(_ML_ROOT, "generated")
    # 15000 scenarii sunt necesare pt ca Ambulanțele apar rar (5%)
    generator.generate(os.path.join(folder_generated, "train_dataset.csv"), 15000)
    generator.generate(os.path.join(folder_generated, "val_dataset.csv"), 2000)