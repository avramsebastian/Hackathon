#!/usr/bin/env python3

import sys
import os

# Add the 'comunication' folder to the Python path
project_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(project_root, "ml", "comunication"))

from Inference import fa_inferenta_din_json

# Example usage:
test_json = {
    "my_car": {"x":0, "y":-10, "speed":15, "direction":"FORWARD"},
    "sign": "YIELD",
    "traffic": [
        {"x":-12, "y":2, "speed":30, "direction":"RIGHT"},
        {"x":40, "y":-5, "speed":10, "direction":"LEFT"}
    ]
}

rezultat = fa_inferenta_din_json(test_json, model_path="ml/generated/traffic_model.pkl")
print(rezultat)
