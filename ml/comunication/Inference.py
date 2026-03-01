import sys, os, joblib
import numpy as np

_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path: sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions

_MODEL_CACHE = None

def get_model(model_path: str):
    global _MODEL_CACHE
    if _MODEL_CACHE is None: _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE

def parse_direction(dir_str: str) -> Directions:
    return {"LEFT": Directions.LEFT, "RIGHT": Directions.RIGHT, "FORWARD": Directions.FORWARD}.get(dir_str.upper(), Directions.FORWARD)

def parse_sign(sign_str: str) -> Sign:
    return {"STOP": Sign.STOP, "YIELD": Sign.YIELD, "PRIORITY": Sign.PRIORITY, "NO_SIGN": Sign.NO_SIGN}.get(sign_str.upper(), Sign.NO_SIGN)

# FUNCȚIE NOUĂ
def parse_traffic_light(tl_str: str) -> TrafficLight:
    return {"RED": TrafficLight.RED, "YELLOW": TrafficLight.YELLOW, "GREEN": TrafficLight.GREEN, "NONE": TrafficLight.NONE}.get(tl_str.upper(), TrafficLight.NONE)

def fa_inferenta_din_json(data_json: dict, model_path: str = "traffic_model.pkl") -> dict:
    try: 
        model = get_model(model_path)
    except FileNotFoundError: 
        return {"error": "Model not found"}

    mc_data = data_json.get("my_car", {})
    my_car = Car(
        x=float(mc_data.get("x", 0.0)), y=float(mc_data.get("y", 0.0)),
        speed=float(mc_data.get("speed", 0.0)), 
        direction=parse_direction(mc_data.get("direction", "FORWARD")),
    )

    traffic = [
        Car(x=float(t.get("x", 0.0)), y=float(t.get("y", 0.0)), speed=float(t.get("speed", 0.0)), 
            direction=parse_direction(t.get("direction", "FORWARD"))) 
        for t in data_json.get("traffic", [])
    ]

    # =========================================================
    # SMART MAPPING (Autocorectare date de la utilizator)
    # =========================================================
    raw_sign = data_json.get("sign", "NO_SIGN").upper()
    raw_tl = data_json.get("traffic_light", "NONE").upper()

    # Dacă ai trimis o culoare în câmpul "sign"
    if raw_sign in ["RED", "YELLOW", "GREEN"]:
        raw_tl = raw_sign
        raw_sign = "NO_SIGN"
        
    # Dacă ai trimis un semn în câmpul "traffic_light"
    if raw_tl in ["STOP", "YIELD", "PRIORITY"]:
        raw_sign = raw_tl
        raw_tl = "NONE"

    sign = parse_sign(raw_sign)
    traffic_light = parse_traffic_light(raw_tl)

    # =========================================================

    intersection = Intersections(my_car, traffic, sign, traffic_light, max_tracked_cars=6)
    
    # Siguranță: Dacă am depășit linia, dăm forțat GO
    if intersection._get_linear_dist(my_car) < 8.0:
        return {"status": "success", "decision": "GO", "confidence_go": 1.0, "confidence_stop": 0.0}

    features = np.array(intersection.get_feature_vector()).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]

    prob_stop = probabilities[0]
    prob_go = probabilities[1]
    decision = "GO" if prob_go > 0.5 else "STOP"

    return {"status": "success", "decision": decision, "confidence_go": float(prob_go), "confidence_stop": float(prob_stop)}