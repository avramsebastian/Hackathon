import csv
import random
import sys
import os

# 1. HACK PENTRU CĂI: Îi spunem lui Python să caute module și în folderele "ML" și "entities"
cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(cale_ml)
sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

def genereaza_dataset(nume_fisier, numar_scenarii):
    print(f"Generăm {numar_scenarii} scenarii pentru {nume_fisier}...")
    
    with open(nume_fisier, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Generăm Header-ul (Numele coloanelor: f1, f2 ... f22, label)
        header = [f"feature_{i+1}" for i in range(22)] + ["label"]
        writer.writerow(header)
        
        for _ in range(numar_scenarii):
            # 1. Mașina principală
            my_car = Car(
                x=random.uniform(-20, 20), 
                y=random.uniform(-20, 20), 
                speed=random.uniform(10, 50), 
                direction=random.choice(list(Directions))
            )
            
            # 2. Semn de circulație
            sign = random.choice(list(Sign))
            
            # 3. Trafic (0 până la 3 mașini)
            num_other_cars = random.randint(0, 3)
            traffic = []
            for _ in range(num_other_cars):
                traffic.append(Car(
                    x=random.uniform(-30, 30),
                    y=random.uniform(-30, 30),
                    speed=random.uniform(0, 50),
                    direction=random.choice(list(Directions))
                ))
            
            # 4. Creăm starea și extragem cei 22 de parametri
            state = Intersections(my_car, traffic, sign)
            features = state.get_feature_vector()
            
            # 5. Stabilim eticheta corectă (0 = Stop, 1 = Go)
            label = 1
            if sign == Sign.STOP:
                label = 0
            elif sign == Sign.YIELD:
                if any(c.distance_to_center() < 15 for c in traffic):
                    label = 0
            
            # 6. Salvăm rândul în CSV
            rand_date = features + [label]
            writer.writerow(rand_date)
            
    print(f"-> Salvat cu succes în {nume_fisier}!\n")

if __name__ == "__main__":
    # Generăm 150 de rânduri pentru Antrenament/Testare
    genereaza_dataset("ML/generated/train_dataset.csv", 2000)
    
    # Generăm 30 de rânduri complet noi pentru Validare
    genereaza_dataset("ML/generated/val_dataset.csv", 500)
    
    print("Gata! Acum ai fișierele CSV în folderul tău.")