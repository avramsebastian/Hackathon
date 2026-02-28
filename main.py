#!/usr/bin/env python3

"""
main.py
=======
Entry point for the V2X Intersection Safety simulator.

Starts the SimBridge (simulation + ML + V2X bus) in a background thread,
then hands the bridge to the PyGame UI as its data source.

Run from the project root:
    python main.py
"""

import sys
import os
import logging

from logging_setup import setup_logging
from sim.sim_bridge import SimBridge
from ui.pygame_view import run_pygame_view


def main() -> None:
    setup_logging(logging.INFO)
    log = logging.getLogger("main")
    log.info("Starting V2X Intersection Safety Simulator")

    bridge = SimBridge(
        tick_rate_hz=10.0,
        drop_rate=0.0,
        latency_ms=0,
    )
    bridge.start()

    try:
        # Blocks until the PyGame window is closed
        run_pygame_view(bridge, width=1000, height=700, fps=60)
    finally:
        bridge.stop()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
