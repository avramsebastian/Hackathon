import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Setup căi
_ML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ML_ROOT, os.path.join(_ML_ROOT, "entities")):
    if _p not in sys.path:
        sys.path.append(_p)

from Inference import fa_inferenta_din_json

class CarModel(BaseModel):
    x: float
    y: float
    speed: float
    direction: str
    role: str = "civilian"

class TrafficStateRequest(BaseModel):
    my_car: CarModel
    # FACEM PARAMETRII OPȚIONALI CU VALORI IMPLICITE
    sign: str = "NO_SIGN"           
    traffic_light: str = "NONE"     
    traffic: List[CarModel] = []    

app = FastAPI(title="V2X AI Inference API", version="1.0")

@app.post("/predict")
def predict_action(state: TrafficStateRequest):
    try:
        data = state.dict()
        result = fa_inferenta_din_json(data, model_path=os.path.join(_ML_ROOT, "generated", "traffic_model.pkl"))
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    print("Starting V2X AI server on http://0.0.0.0:8000 …")
    uvicorn.run(app, host="0.0.0.0", port=8000)