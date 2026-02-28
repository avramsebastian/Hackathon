"""
ml/comunication/Inference.py
============================
ML inference entry point.

:func:`fa_inferenta_din_json` (kept for backward compatibility) accepts a
JSON-like dict with ``my_car``, ``sign`` and ``traffic`` keys, runs the
pre-trained Random Forest model, and returns a decision dict.
"""

import sys
import os
import joblib
import numpy as np

# ── path setup (allows ``from entities.X import …``) ─────────────────────────
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path:
        sys.path.append(_p)

from entities.Car import Car
from entities.Intersections import Intersections
from entities.Sign import Sign
from entities.Directions import Directions

# ── model cache (loaded once per process) ─────────────────────────────────────
_MODEL_CACHE = None


def get_model(model_path: str):
    """Load (and cache) the scikit-learn model from *model_path*."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def parse_direction(dir_str: str) -> Directions:
    """Map a direction string to a :class:`Directions` enum member."""
    mapping = {
        "LEFT": Directions.LEFT,
        "RIGHT": Directions.RIGHT,
        "FORWARD": Directions.FORWARD,
    }
    return mapping.get(dir_str.upper(), Directions.FORWARD)


def parse_sign(sign_str: str) -> Sign:
    """Map a sign string to a :class:`Sign` enum member."""
    mapping = {
        "STOP": Sign.STOP,
        "YIELD": Sign.YIELD,
        "PRIORITY": Sign.PRIORITY,
        "NO_SIGN": Sign.NO_SIGN,
    }
    return mapping.get(sign_str.upper(), Sign.NO_SIGN)


def fa_inferenta_din_json(
    data_json: dict,
    model_path: str = "traffic_model.pkl",
) -> dict:
    """Run ML inference on a traffic-state dictionary.

    Parameters
    ----------
    data_json : dict
        Must contain ``my_car`` (dict with x, y, speed, direction),
        ``sign`` (str) and ``traffic`` (list of car dicts).
    model_path : str
        Filesystem path to the ``.pkl`` model file.

    Returns
    -------
    dict
        ``{status, decision, confidence_go, confidence_stop}`` on success,
        or ``{error: …}`` on failure.
    """
    try:
        model = get_model(model_path)
    except FileNotFoundError:
        return {"error": "Model not found"}

    # ── build entities ────────────────────────────────────────────────────
    mc_data = data_json.get("my_car", {})
    my_car = Car(
        x=float(mc_data.get("x", 0.0)),
        y=float(mc_data.get("y", 0.0)),
        speed=float(mc_data.get("speed", 0.0)),
        direction=parse_direction(mc_data.get("direction", "FORWARD")),
    )

    sign = parse_sign(data_json.get("sign", "NO_SIGN"))

    traffic = [
        Car(
            x=float(t.get("x", 0.0)),
            y=float(t.get("y", 0.0)),
            speed=float(t.get("speed", 0.0)),
            direction=parse_direction(t.get("direction", "FORWARD")),
        )
        for t in data_json.get("traffic", [])
    ]

    # ── feature extraction & prediction ───────────────────────────────────
    intersection = Intersections(my_car, traffic, sign, max_tracked_cars=6)
    features = np.array(intersection.get_feature_vector()).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]

    prob_stop = probabilities[0]
    prob_go = probabilities[1]
    decision = "GO" if prob_go > 0.5 else "STOP"

    return {
        "status": "success",
        "decision": decision,
        "confidence_go": float(prob_go),
        "confidence_stop": float(prob_stop),
    }