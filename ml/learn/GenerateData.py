import csv
import random
import sys
import os
import math

cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if cale_ml not in sys.path:
    sys.path.append(cale_ml)
    sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

LANE_OFFSET = 7.0

def get_random_lane_car():
    lane = random.choice(['EB', 'WB', 'NB', 'SB'])
    is_approaching = random.random() < 0.7 
    
    # 1. FIX: Am mărit distanța la 120 de metri ca AI-ul să vadă tot ecranul din world.py!
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

def se_apropie_de_centru(car):
    if abs(car.y + LANE_OFFSET) < 0.1: return car.x < 8.0   
    if abs(car.y - LANE_OFFSET) < 0.1: return car.x > -8.0  
    if abs(car.x - LANE_OFFSET) < 0.1: return car.y < 8.0   
    if abs(car.x + LANE_OFFSET) < 0.1: return car.y > -8.0  
    return True 

def este_periculos(car):
    if math.hypot(car.x, car.y) < 12.0: return True 
    if abs(car.y + LANE_OFFSET) < 0.1: return car.x < 0.0   
    if abs(car.y - LANE_OFFSET) < 0.1: return car.x > 0.0   
    if abs(car.x - LANE_OFFSET) < 0.1: return car.y < 0.0   
    if abs(car.x + LANE_OFFSET) < 0.1: return car.y > 0.0   
    return False

def genereaza_dataset(nume_fisier, numar_scenarii):
    print(f"Generăm {numar_scenarii} scenarii perfect calibrate pe geometria jocului...")
    os.makedirs(os.path.dirname(nume_fisier), exist_ok=True)
    
    with open(nume_fisier, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [f"feature_{i+1}" for i in range(22)] + ["label"]
        writer.writerow(header)
        
        for _ in range(numar_scenarii):
            my_car = get_random_lane_car()
            sign = random.choice(list(Sign))
            
            num_other_cars = random.randint(1, 3) 
            traffic = [get_random_lane_car() for _ in range(num_other_cars)]
            
            state = Intersections(my_car, traffic, sign)
            features = state.get_feature_vector()
            
            label = 1 # Presupunem drum liber (GO)
            
            if se_apropie_de_centru(my_car):
                trafic_periculos = [c for c in traffic if este_periculos(c)]
                dist_mea = math.hypot(my_car.x, my_car.y)
                
                if sign == Sign.STOP:
                    if dist_mea > 6.0:
                        label = 0 
                
                elif sign == Sign.YIELD:
                    # 2. FIX: Cedează doar dacă cealaltă mașină e la sub 90m ȘI
                    # este mai aproape de intersecție decât mine (sau la o distanță comparabilă)
                    for c in trafic_periculos:
                        dist_c = math.hypot(c.x, c.y)
                        if dist_c < 90.0 and dist_c < dist_mea + 15.0:
                            label = 0
                            break
                        
                elif sign == Sign.NO_SIGN:
                    for c in trafic_periculos:
                        dist_c = math.hypot(c.x, c.y)
                        if dist_c < 90.0 and dist_c < dist_mea + 15.0:
                            cross_product = (my_car.x * c.y) - (my_car.y * c.x)
                            if cross_product > 0:
                                label = 0 # E in DREAPTA mea! Opresc.
                                break
            
            rand_date = features + [label]
            writer.writerow(rand_date)
            
    print(f"-> Salvat cu succes în {nume_fisier}!\n")

if __name__ == "__main__":
    folder_generated = os.path.join(cale_ml, "generated")
    genereaza_dataset(os.path.join(folder_generated, "train_dataset.csv"), 4000)
    genereaza_dataset(os.path.join(folder_generated, "val_dataset.csv"), 500)
    print("Gata! Acum AI-ul vede până la 120m distanță și judecă corect cine trece primul.")