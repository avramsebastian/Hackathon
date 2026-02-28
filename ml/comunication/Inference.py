import sys
import os
import json
import joblib
import numpy as np

cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if cale_ml not in sys.path:
    sys.path.append(cale_ml)
    sys.path.append(os.path.join(cale_ml, 'entities'))

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

# ==========================================
# MAGIA AICI: Păstrăm modelul în RAM!
# ==========================================
_MODEL_CACHE = None

def get_model(model_path):
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print(f"[AI] Se încarcă modelul în memoria RAM din '{model_path}' o singură dată...")
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE

def parse_direction(dir_str):
    mapping = {
        "LEFT": Directions.LEFT,
        "RIGHT": Directions.RIGHT,
        "FORWARD": Directions.FORWARD
    }
    return mapping.get(dir_str.upper(), Directions.FORWARD)

def parse_sign(sign_str):
    mapping = {
        "STOP": Sign.STOP,
        "YIELD": Sign.YIELD,
        "PRIORITY": Sign.PRIORITY,
        "NO_SIGN": Sign.NO_SIGN
    }
    return mapping.get(sign_str.upper(), Sign.NO_SIGN)

def fa_inferenta_din_json(date_json, model_path="traffic_model.pkl"):
    try:
        # Folosim modelul din RAM, durează 0.0001 secunde!
        model = get_model(model_path)
    except FileNotFoundError:
        return {"error": "Model not found"}

    mc_data = date_json.get("my_car", {})
    my_car = Car(
        x=float(mc_data.get("x", 0.0)),
        y=float(mc_data.get("y", 0.0)),
        speed=float(mc_data.get("speed", 0.0)),
        direction=parse_direction(mc_data.get("direction", "FORWARD"))
    )
    
    semn = parse_sign(date_json.get("sign", "NO_SIGN"))
    
    traffic = []
    for t_data in date_json.get("traffic", []):
        traffic.append(Car(
            x=float(t_data.get("x", 0.0)),
            y=float(t_data.get("y", 0.0)),
            speed=float(t_data.get("speed", 0.0)),
            direction=parse_direction(t_data.get("direction", "FORWARD"))
        ))
        
    stare_intersectie = Intersections(my_car, traffic, semn)
    vector_features = stare_intersectie.get_feature_vector()
    
    input_formatat = np.array(vector_features).reshape(1, -1)
    
    # Facem predicția instant
    probabilitati = model.predict_proba(input_formatat)[0]
    
    prob_stop = probabilitati[0]
    prob_go = probabilitati[1]
    
    decizie = "GO" if prob_go > 0.4 else "STOP"
    
    return {
        "status": "success",
        "decision": decizie,
        "confidence_go": float(prob_go),
        "confidence_stop": float(prob_stop)
    }