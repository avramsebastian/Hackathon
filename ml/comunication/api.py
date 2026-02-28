import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Hack-ul pentru căi, la fel ca în celelalte fișiere
cale_ml = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if cale_ml not in sys.path:
    sys.path.append(cale_ml)
    sys.path.append(os.path.join(cale_ml, 'entities'))

# Importăm funcția ta de inferență pe care tocmai am creat-o
# Asigură-te că fișierul tău se numește exact Inference.py
from Inference import fa_inferenta_din_json

# ==========================================
# 1. DEFINIM STRUCTURA JSON-ULUI (Modele Pydantic)
# ==========================================
class CarModel(BaseModel):
    x: float
    y: float
    speed: float
    direction: str

class TrafficStateRequest(BaseModel):
    my_car: CarModel
    sign: str
    traffic: List[CarModel]

# ==========================================
# 2. INIȚIALIZĂM APLICAȚIA FASTAPI
# ==========================================
app = FastAPI(
    title="BEST Hackathon - V2X AI API",
    description="API pentru agentul autonom V2X care prezice riscul de coliziune la intersecții.",
    version="1.0"
)

# ==========================================
# 3. DEFINIM RUTA PENTRU INFERENȚĂ
# ==========================================
@app.post("/predict")
def predict_action(state: TrafficStateRequest):
    try:
        # Pydantic transformă automat JSON-ul valid într-un dicționar Python (.model_dump() sau .dict())
        date_json = state.dict()
        
        # Apelăm modelul tău de ML
        # NOTĂ: Asigură-te că traffic_model.pkl se află în același folder cu api.py
        rezultat = fa_inferenta_din_json(date_json, model_path="ML/generated/traffic_model.pkl")
        
        # Dacă funcția de inferență a returnat o eroare (ex: modelul nu a fost găsit)
        if "error" in rezultat:
            raise HTTPException(status_code=500, detail=rezultat["error"])
            
        return rezultat
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==========================================
# 4. RULAREA SERVERULUI
# ==========================================
if __name__ == "__main__":
    print("Pornește serverul V2X AI...")
    # Rulăm serverul pe portul 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)