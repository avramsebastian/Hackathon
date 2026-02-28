import csv
import random
import sys
import os

cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(cale_ml)
sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

def genereaza_dataset(nume_fisier, numar_scenarii):
    print(f"Generăm {numar_scenarii} scenarii pentru {nume_fisier}...")
    
    # Ne asigurăm că folderul există
    os.makedirs(os.path.dirname(nume_fisier), exist_ok=True)
    
    with open(nume_fisier, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Generăm Header-ul
        header = [f"feature_{i+1}" for i in range(22)] + ["label"]
        writer.writerow(header)
        
        for _ in range(numar_scenarii):
            my_car = Car(
                x=random.uniform(-20, 20), 
                y=random.uniform(-20, 20), 
                speed=random.uniform(10, 50), 
                direction=random.choice(list(Directions))
            )
            
            sign = random.choice(list(Sign))
            
            num_other_cars = random.randint(0, 3)
            traffic = []
            for _ in range(num_other_cars):
                traffic.append(Car(
                    x=random.uniform(-30, 30),
                    y=random.uniform(-30, 30),
                    speed=random.uniform(0, 50),
                    direction=random.choice(list(Directions))
                ))
            
            state = Intersections(my_car, traffic, sign)
            features = state.get_feature_vector()
            
            # ========================================================
            # LOGICA DE ETICHETARE (PROFESORUL VIRTUAL ÎMBUNĂTĂȚIT)
            # ========================================================
            label = 1 # Presupunem implicit că e liber (GO)
            
            # 1. Verificăm dacă NU a intrat încă în intersecție. 
            # Dacă distanța e sub 4 metri, mașina e deja în mijlocul acțiunii, deci continuă (nu mai frânează aiurea).
            if my_car.distance_to_center() > 4.0:
                
                if sign == Sign.STOP:
                    # La STOP oprim mereu înainte de intersecție
                    label = 0
                    
                elif sign == Sign.YIELD:
                    # La Cedează Trecerea, oprim dacă orice altă mașină e mai aproape de 15 metri
                    if any(c.distance_to_center() < 15.0 for c in traffic):
                        label = 0
                        
                elif sign == Sign.NO_SIGN:
                    # PRIORITATE DE DREAPTA
                    for c in traffic:
                        # Ne interesează doar mașinile care reprezintă un pericol (sunt aproape)
                        if c.distance_to_center() < 15.0:
                            # Calculăm produsul vectorial 2D (Cross Product)
                            # Un rezultat > 0 indică faptul că mașina 'c' se află în cadranul din DREAPTA
                            cross_product = (my_car.x * c.y) - (my_car.y * c.x)
                            if cross_product > 0:
                                label = 0 # Oprim pentru că mașina 'c' are prioritate de dreapta!
                                break
                                
            # ========================================================
            
            rand_date = features + [label]
            writer.writerow(rand_date)
            
    print(f"-> Salvat cu succes în {nume_fisier}!\n")

if __name__ == "__main__":
    folder_generated = os.path.join(cale_ml, "generated")
    
    # Re-generăm datele cu noile reguli inteligente de circulație
    genereaza_dataset(os.path.join(folder_generated, "train_dataset.csv"), 2000)
    genereaza_dataset(os.path.join(folder_generated, "val_dataset.csv"), 500)
    
    print("Gata! Noile date au fost generate și iau în calcul prioritatea de dreapta și distanța.")