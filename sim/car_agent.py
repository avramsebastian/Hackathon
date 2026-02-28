"""
sim/car_agent.py
================
A single autonomous car agent. Each agent:
  - owns its position / speed / direction
  - runs its own ML inference against all other agents
  - exposes a rich vehicle dict (with road_line, approach, color, speed_unit)
    compatible with PygameIntersectionView
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    _ROOT,
    os.path.join(_ROOT, "ml"),
    os.path.join(_ROOT, "ml", "comunication"),
    os.path.join(_ROOT, "ml", "entities"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comunication.Inference import fa_inferenta_din_json

log = logging.getLogger("car_agent")

# direction string → (dx, dy) unit vector
_DIR_VECTORS: Dict[str, Tuple[float, float]] = {
    "FORWARD":  ( 0.0,  1.0),
    "BACKWARD": ( 0.0, -1.0),
    "LEFT":     (-1.0,  0.0),
    "RIGHT":    ( 1.0,  0.0),
}

# UI colour palette — mirrors ViewConstants.DEFAULT_VEHICLE_COLORS
VEHICLE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    ( 86, 168, 255),
    (255,  88,  88),
    (100, 226, 170),
    (246, 191,  90),
    (180, 120, 255),
    (255, 160, 100),
)

# How far past the intersection centre before a car is "done" (world units)
FINISHED_DIST = 60.0


class CarAgent:
    """
    One autonomous car agent.

    Parameters
    ----------
    agent_id : str
        Unique identifier, e.g. "CAR_0".
    x, y : float
        Starting world-space position in metres (origin = intersection centre).
    speed : float
        Initial speed in metres per second.
    direction : str
        Travel direction: FORWARD | BACKWARD | LEFT | RIGHT.
    approach : str
        Cardinal arm this car approaches from: N | S | E | W.
        Used by the UI to place road signs on the correct side.
    color_index : int
        Index into VEHICLE_COLORS palette.
    model_path : str
        Path to the shared .pkl ML model file.
    """

    def __init__(
        self,
        agent_id: str,
        x: float,
        y: float,
        speed: float,
        direction: str,
        approach: str,
        color_index: int,
        model_path: str,
    ) -> None:
        self.id        = agent_id.upper()
        self.x         = x
        self.y         = y
        self.speed     = speed
        self.direction = direction.upper()
        self.approach  = approach.upper()
        self.color     = VEHICLE_COLORS[color_index % len(VEHICLE_COLORS)]
        self._model_path = model_path

        # Latest ML result
        self.decision:        str   = "none"
        self.confidence_go:   float = 0.0
        self.confidence_stop: float = 0.0
        self.ml_ok:           bool  = False

    # ── Physics ───────────────────────────────────────────────────────────────

    def move(self, dt: float) -> None:
        """Advance position, slowed to 35 % when ML says STOP."""
        dx, dy = _DIR_VECTORS.get(self.direction, (0.0, 1.0))
        effective = self.speed * (0.35 if self.decision == "STOP" else 1.0)
        self.x += dx * effective * dt
        self.y += dy * effective * dt

    def is_finished(self) -> bool:
        """True once the car has travelled past the far side of the intersection."""
        return (self.x ** 2 + self.y ** 2) ** 0.5 > FINISHED_DIST

    # ── ML inference ──────────────────────────────────────────────────────────

    def run_inference(self, other_agents: List[CarAgent], current_sign: str) -> None:
        """Run ML inference from this agent's own perspective."""
        ml_input = {
            "my_car": {
                "x": self.x, "y": self.y,
                "speed": self.speed, "direction": self.direction,
            },
            "sign": current_sign,
            "traffic": [
                {"x": a.x, "y": a.y, "speed": a.speed, "direction": a.direction}
                for a in other_agents if a.id != self.id
            ],
        }
        raw = fa_inferenta_din_json(ml_input, model_path=self._model_path)

        if raw.get("status") == "success":
            self.decision        = raw["decision"]
            self.confidence_go   = float(raw.get("confidence_go",   0.0))
            self.confidence_stop = float(raw.get("confidence_stop", 0.0))
            self.ml_ok           = True
        else:
            self.decision        = "none"
            self.confidence_go   = 0.0
            self.confidence_stop = 0.0
            self.ml_ok           = False

        log.debug("%s → %s  go=%.2f  stop=%.2f",
                  self.id, self.decision, self.confidence_go, self.confidence_stop)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def as_dict(self) -> Dict[str, Any]:
        """Rich vehicle dict for the UI and V2X bus payloads."""
        return {
            "id":         self.id,
            "x":          self.x,
            "y":          self.y,
            "speed":      self.speed,
            "speed_unit": "kmh",
            "direction":  self.direction,   # raw ML direction: LEFT|RIGHT|FORWARD|BACKWARD
            "approach":   self.approach,
            "color":      self.color,
        }

    def decision_dict(self) -> Dict[str, Any]:
        """Decision payload for get_ml_decision()."""
        return {
            "decision":        self.decision,
            "confidence_go":   self.confidence_go,
            "confidence_stop": self.confidence_stop,
        }
