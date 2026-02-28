#!/usr/bin/env python3
"""
test_ml.py
==========
Quick smoke-test for the ML inference pipeline.

Runs a single scenario through the model and prints the result.
Usage::

    python test_ml.py
"""

import sys
import os

# Ensure the ML inference module is importable
project_root = os.path.abspath(".")
sys.path.insert(0, os.path.join(project_root, "ml", "comunication"))

from Inference import fa_inferenta_din_json


def main() -> None:
    test_payload = {
        "my_car": {"x": 0, "y": -10, "speed": 15, "direction": "FORWARD"},
        "sign": "YIELD",
        "traffic": [
            {"x": -12, "y": 2, "speed": 30, "direction": "RIGHT"},
            {"x": 40, "y": -5, "speed": 10, "direction": "LEFT"},
        ],
    }

    result = fa_inferenta_din_json(
        test_payload, model_path="ml/generated/traffic_model.pkl",
    )
    print(result)


if __name__ == "__main__":
    main()
