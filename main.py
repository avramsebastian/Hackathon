#!/usr/bin/env python3

import sys
import os
import time
import logging

# Make project modules discoverable
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.extend([
    project_root,
    os.path.join(project_root, "ml"),
    os.path.join(project_root, "ml", "comunication"),
    os.path.join(project_root, "ml", "entities"),
    os.path.join(project_root, "sim"),
    os.path.join(project_root, "bus")
])

# Logging
from logging_setup import setup_logging
# World simulation
from sim.world import World
# ML inference
from comunication.Inference import fa_inferenta_din_json
# V2X bus
from bus.v2x_bus import V2XBus

def main():
    setup_logging(logging.INFO)
    log = logging.getLogger("main")
    log.info("Starting main loop...")

    # Initialize world
    world = World()

    # ML model path
    model_path = os.path.join(project_root, "ml", "generated", "traffic_model.pkl")

    # Initialize V2X bus
    v2x = V2XBus(drop_rate=0.0, latency_ms=0)

    try:
        while True:
            # 1. Get world state for ML
            ml_input = world.get_ml_input()
            log.debug(f"World state: {ml_input}")

            # 2. Run ML inference
            result = fa_inferenta_din_json(ml_input, model_path=model_path)
            log.info(f"ML decision: {result}")

            # 3. Take action based on decision
            if result['decision'] == "STOP":
                log.info("Car should stop!")
            else:
                log.info("Car can go!")

            # 4. Publish to V2X bus for other agents / UI
            v2x.publish(
                topic="v2v.state",
                sender="my_car",
                payload={
                    "decision": result['decision'],
                    "confidence_go": result.get("confidence_go", 0.0),
                    "confidence_stop": result.get("confidence_stop", 0.0),
                    "position": world.my_car.as_dict(),
                    "traffic": [c.as_dict() for c in world.traffic],
                    "sign": world.current_sign,
                }
            )

            # 5. Advance world simulation
            world.update_physics(dt=0.5)  # half-second tick

            time.sleep(0.5)

    except KeyboardInterrupt:
        log.info("Shutting down...")

if __name__ == "__main__":
    main()
