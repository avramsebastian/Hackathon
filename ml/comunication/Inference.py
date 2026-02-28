import sys
import os
<<<<<<< HEAD
=======
import json
import threading
>>>>>>> 4d385f6 (sim stuff)
import joblib
import numpy as np

# Hack-ul pentru căi
cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(cale_ml)
sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

<<<<<<< HEAD
_MODEL_CACHE = None

def get_model(model_path):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE
=======
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()

# --- Funcții de "Traducere" din JSON (String) în Enum-urile tale ---
>>>>>>> 4d385f6 (sim stuff)

def parse_direction(dir_str):
    mapping = {
        "LEFT": Directions.LEFT,
        "RIGHT": Directions.RIGHT,
        "FORWARD": Directions.FORWARD
    }
    return mapping.get(dir_str.upper(), Directions.FORWARD) # Default: FORWARD

def parse_sign(sign_str):
    mapping = {
        "STOP": Sign.STOP,
        "YIELD": Sign.YIELD,
        "PRIORITY": Sign.PRIORITY,
        "NO_SIGN": Sign.NO_SIGN
    }
    return mapping.get(sign_str.upper(), Sign.NO_SIGN) # Default: NO_SIGN

# --- Funcția Principală de Inferență ---

def _load_model_cached(model_path):
    abs_path = os.path.abspath(model_path)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(abs_path)
        if model is None:
            model = joblib.load(abs_path)
            _MODEL_CACHE[abs_path] = model
    return model


def fa_inferenta_din_json(
    date_json,
    model_path="ML/generated/traffic_model.pkl",
    verbose=False,
):
    # 1. Încărcăm modelul (cache-uit în memorie)
    try:
<<<<<<< HEAD
        model = get_model(model_path)
=======
        model = _load_model_cached(model_path)
>>>>>>> 4d385f6 (sim stuff)
    except FileNotFoundError:
        if verbose:
            print("Eroare: Nu găsesc modelul! Asigură-te că ai rulat Train.py înainte.")
        return {"error": "Model not found"}

    # 2. Parsăm datele din dicționarul primit din JSON
    # Extragem mașina principală
    mc_data = date_json.get("my_car", {})
    my_car = Car(
        x=float(mc_data.get("x", 0.0)),
        y=float(mc_data.get("y", 0.0)),
        speed=float(mc_data.get("speed", 0.0)),
        direction=parse_direction(mc_data.get("direction", "FORWARD"))
    )
    
    # Extragem semnul de circulație
    semn = parse_sign(date_json.get("sign", "NO_SIGN"))
    
    # Extragem traficul (lista de alte mașini)
    traffic = []
    for t_data in date_json.get("traffic", []):
        traffic.append(Car(
            x=float(t_data.get("x", 0.0)),
            y=float(t_data.get("y", 0.0)),
            speed=float(t_data.get("speed", 0.0)),
            direction=parse_direction(t_data.get("direction", "FORWARD"))
        ))
        
<<<<<<< HEAD
    stare_intersectie = Intersections(my_car, traffic, semn, max_tracked_cars=6)
=======
    # 3. Creăm starea și extragem vectorul (cei 22 de parametri)
    stare_intersectie = Intersections(my_car, traffic, semn)
>>>>>>> 4d385f6 (sim stuff)
    vector_features = stare_intersectie.get_feature_vector()
    
    # 4. Formatăm pentru model și facem predicția
    input_formatat = np.array(vector_features).reshape(1, -1)
    probabilitati = model.predict_proba(input_formatat)[0]
    
    prob_stop = probabilitati[0]
    prob_go = probabilitati[1]
    
<<<<<<< HEAD
    decizie = "GO" if prob_go > 0.5 else "STOP"
=======
    # 5. Returnăm rezultatul (ca să-l poți trimite înapoi prin API)
    decizie = "GO" if prob_go > 0.5 else "STOP"
    if verbose:
        print("=========================================")
        print("           REZULTAT AI LIVE              ")
        print("=========================================")
        print(f"Șanse să fie SIGUR (Accelerează): {prob_go * 100:.1f}%")
        print(f"Șanse de COLIZIUNE (Frânează):    {prob_stop * 100:.1f}%")
        print(f"-> DECIZIE: {decizie}")
>>>>>>> 4d385f6 (sim stuff)
    
    return {
        "status": "success",
        "decision": decizie,
        "confidence_go": float(prob_go),
        "confidence_stop": float(prob_stop)
    }

# ==========================================
# TESTĂM CU UN JSON SIMULAT
# ==========================================
if __name__ == "__main__":
    
    # Acesta este exact formatul JSON pe care front-end-ul ar trebui să ți-l trimită
    json_input_string = """
    {
        "my_car": {
            "x": 0.0,
            "y": -10.0,
            "speed": 15.0,
            "direction": "FORWARD"
        },
        "sign": "YIELD",
        "traffic": [
            {
                "x": -12.0,
                "y": 2.0,
                "speed": 30.0,
                "direction": "RIGHT"
            },
            {
                "x": 40.0,
                "y": -5.0,
                "speed": 10.0,
                "direction": "LEFT"
            }
        ]
    }
    """
    
    # 1. Transformăm String-ul JSON primit într-un Dicționar Python
    date_parsite = json.loads(json_input_string)
    
    # 2. Trimitem dicționarul către funcția noastră de inferență
    rezultat = fa_inferenta_din_json(date_parsite, verbose=True)
    
    # (Opțional) Afișăm rezultatul final tot sub formă de JSON, cum ar pleca spre frontend
    print("\n--- Răspunsul care se întoarce la Frontend/Simulator ---")
    print(json.dumps(rezultat, indent=4))
