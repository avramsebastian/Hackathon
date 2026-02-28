"""
ml/comunication/api.py
======================
Optional FastAPI server that exposes the ML model as a REST endpoint.

Start the server::

    python ml/comunication/api.py          # → http://localhost:8000/predict

The ``/predict`` endpoint accepts a JSON body with ``my_car``, ``sign``
and ``traffic`` fields and returns ``{decision, confidence_go, confidence_stop}``.

.. note::

   This server is **not** required to run the Pygame simulation.
   It exists for external integrations and testing.
"""

import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── path setup ────────────────────────────────────────────────────────────────
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path:
        sys.path.append(_p)

from Inference import fa_inferenta_din_json

# ── Pydantic request schemas ─────────────────────────────────────────────────


class CarModel(BaseModel):
    """Single vehicle in the request payload."""
    x: float
    y: float
    speed: float
    direction: str


class TrafficStateRequest(BaseModel):
    """Intersection state submitted to ``/predict``."""
    my_car: CarModel
    sign: str
    traffic: List[CarModel]


# ── FastAPI application ──────────────────────────────────────────────────────

app = FastAPI(
    title="V2X AI Inference API",
    description="Predicts GO / STOP collision risk at intersections.",
    version="1.0",
)


@app.post("/predict")
def predict_action(state: TrafficStateRequest):
    """Run ML inference on the provided traffic state."""
    try:
        data = state.dict()
        result = fa_inferenta_din_json(
            data, model_path="ml/generated/traffic_model.pkl",
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting V2X AI server on http://0.0.0.0:8000 …")
    uvicorn.run(app, host="0.0.0.0", port=8000)