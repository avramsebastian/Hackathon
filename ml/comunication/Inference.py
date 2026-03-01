import sys, os, joblib, numpy as np
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path: sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersection
from entities.Sign import Sign
from entities.TrafficLight import TrafficLight
from entities.Directions import Directions
from entities.Role import Role

_MODEL_CACHE = None

def get_model(model_path: str):
    global _MODEL_CACHE
    if _MODEL_CACHE is None: _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE

def parse_direction(dir_str: str) -> Directions:
    return {"LEFT": Directions.LEFT, "RIGHT": Directions.RIGHT, "FORWARD": Directions.FORWARD}.get(dir_str.upper(), Directions.FORWARD)

def parse_sign(sign_str: str) -> Sign:
    return {"STOP": Sign.STOP, "YIELD": Sign.YIELD, "PRIORITY": Sign.PRIORITY, "NO_SIGN": Sign.NO_SIGN}.get(sign_str.upper(), Sign.NO_SIGN)

def parse_traffic_light(tl_str: str) -> TrafficLight:
    return {"RED": TrafficLight.RED, "YELLOW": TrafficLight.YELLOW, "GREEN": TrafficLight.GREEN, "NONE": TrafficLight.NONE}.get(tl_str.upper(), TrafficLight.NONE)

# FUNCTIE NOUA
def parse_role(role_str: str) -> Role:
    if role_str.lower() in ["ambulance", "police", "fire"]:
        return Role.EMERGENCY
    return Role.CIVILIAN

def fa_inferenta_din_json(data_json: dict, model_path: str = "traffic_model.pkl") -> dict:
    try: model = get_model(model_path)
    except FileNotFoundError: return {"error": "Model not found"}

    mc = data_json.get("my_car", {})
    my_car = Car(
        x=float(mc.get("x", 0.0)), y=float(mc.get("y", 0.0)),
        speed=float(mc.get("speed", 0.0)), 
        direction=parse_direction(mc.get("direction", "FORWARD")),
        role=parse_role(mc.get("role", "civilian"))
    )

    traffic = [
        Car(x=float(t.get("x", 0.0)), y=float(t.get("y", 0.0)), speed=float(t.get("speed", 0.0)), 
            direction=parse_direction(t.get("direction", "FORWARD")), role=parse_role(t.get("role", "civilian"))) 
        for t in data_json.get("traffic", [])
    ]

    raw_sign = data_json.get("sign", "NO_SIGN").upper()
    raw_tl = data_json.get("traffic_light", "NONE").upper()
    if raw_sign in ["RED", "YELLOW", "GREEN"]: raw_tl = raw_sign; raw_sign = "NO_SIGN"
    if raw_tl in ["STOP", "YIELD", "PRIORITY"]: raw_sign = raw_tl; raw_tl = "NONE"

    intersection = Intersection(my_car, traffic, parse_sign(raw_sign), parse_traffic_light(raw_tl), max_tracked_cars=6)
    
    if intersection._get_linear_dist(my_car) < 8.0:
        return {"status": "success", "decision": "GO", "confidence_go": 1.0, "confidence_stop": 0.0}

    features = np.array(intersection.get_feature_vector()).reshape(1, -1)
    probs = model.predict_proba(features)[0]

    return {"status": "success", "decision": "GO" if probs[1] > 0.5 else "STOP", "confidence_go": float(probs[1]), "confidence_stop": float(probs[0])}