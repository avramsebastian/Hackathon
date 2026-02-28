"""
sim/sim_bridge.py
=================
Runs the simulation loop in a background thread and exposes the three
methods that PygameIntersectionView._poll_bus() expects:

    bridge.get_vehicles()               → List[dict]
    bridge.get_ml_decision(vehicle_id)  → dict  {"decision", "confidence_go", "confidence_stop"}
    bridge.get_intersection()           → dict  {"signs", "lane_count", "box_size"}

Usage in main.py (or anywhere):

    from sim.sim_bridge import SimBridge
    from ui.pygame_view import run_pygame_view

    bridge = SimBridge()
    bridge.start()
    run_pygame_view(bridge)          # blocks until window is closed
    bridge.stop()
"""

from __future__ import annotations

import os
import sys
import threading
import time
import logging
from typing import Any, Dict, List, Optional

# ── path setup ────────────────────────────────────────────────────────────────
# sim_bridge.py lives in sim/; project root is one level up
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    _ROOT,
    os.path.join(_ROOT, "ml"),
    os.path.join(_ROOT, "ml", "comunication"),
    os.path.join(_ROOT, "ml", "entities"),
    os.path.join(_ROOT, "sim"),
    os.path.join(_ROOT, "bus"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.world import World
from bus.v2x_bus import V2XBus
from comunication.Inference import fa_inferenta_din_json

log = logging.getLogger("sim_bridge")


class SimBridge:
    """
    Simulation orchestrator + bus adapter.

    The background thread ticks World → ML → V2XBus every ``1/tick_rate_hz``
    seconds and caches the results.  The UI thread calls the three ``get_*``
    methods to read the latest snapshot without ever blocking.

    Parameters
    ----------
    tick_rate_hz : float
        How many simulation ticks per second (default 10).
    drop_rate : float
        V2X packet-drop probability 0.0–1.0 (default 0 = perfect channel).
    latency_ms : int
        Simulated V2X latency in milliseconds (default 0).
    model_path : str | None
        Explicit path to the .pkl model file.
        Defaults to <project_root>/ml/generated/traffic_model.pkl.
    """

    def __init__(
        self,
        tick_rate_hz: float = 10.0,
        drop_rate: float = 0.0,
        latency_ms: int = 0,
        model_path: Optional[str] = None,
    ) -> None:
        self._tick_rate_hz = tick_rate_hz
        self._model_path   = model_path or os.path.join(
            _ROOT, "ml", "generated", "traffic_model.pkl"
        )
        self._world = World()
        self._bus   = V2XBus(drop_rate=drop_rate, latency_ms=latency_ms)

        self._lock = threading.Lock()

        # Cached state — written by sim thread, read by UI thread
        self._vehicles:    List[Dict[str, Any]] = []
        self._decisions:   Dict[str, Any]       = {}
        self._intersection: Dict[str, Any]      = {}

        self._thread:   Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background simulation loop."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="SimBridge"
        )
        self._thread.start()
        log.info("SimBridge started at %.1f Hz", self._tick_rate_hz)

    def stop(self) -> None:
        """Stop the background loop and join the thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("SimBridge stopped")

    # ── Bus adapter API (called by PygameIntersectionView._poll_bus) ──────────

    def get_vehicles(self) -> List[Dict[str, Any]]:
        """
        Return a list of vehicle dicts, one per car in the simulation.

        Each dict contains:
            id          – unique string ID
            x, y        – world-space position (metres, origin = intersection centre)
            speed       – metres per second
            direction   – "FORWARD" | "LEFT" | "RIGHT"
            is_player   – bool
        """
        with self._lock:
            return list(self._vehicles)

    def get_ml_decision(self, vehicle_id: str) -> Dict[str, Any]:
        """
        Return the latest ML decision for the given vehicle ID.

        Returns a dict with:
            decision        – "GO" | "STOP"
            confidence_go   – float 0–1
            confidence_stop – float 0–1

        Returns {"decision": "none"} if the vehicle is unknown or no
        inference has run yet.
        """
        with self._lock:
            return dict(self._decisions.get(vehicle_id.upper(), {"decision": "none"}))

    def get_intersection(self) -> Dict[str, Any]:
        """
        Return static/dynamic intersection metadata.

        Keys used by the UI:
            signs       – dict mapping approach cardinal ("N","S","E","W") → sign type string
            lane_count  – int (uniform) or {"horizontal": n, "vertical": n}
            box_size    – int, intersection box size in pixels (optional)
        """
        with self._lock:
            return dict(self._intersection)

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        dt = 1.0 / self._tick_rate_hz
        while self._running:
            t0 = time.perf_counter()
            try:
                self._tick(dt)
            except Exception:
                log.exception("SimBridge tick error")
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    def _tick(self, dt: float) -> None:
        # 1. Snapshot world → ML input
        ml_input = self._world.get_ml_input()

        # 2. Run ML inference for the player car
        raw = fa_inferenta_din_json(ml_input, model_path=self._model_path)

        decision_payload: Dict[str, Any]
        if raw.get("status") == "success":
            decision_payload = {
                "decision":        raw["decision"],
                "confidence_go":   float(raw.get("confidence_go",   0.0)),
                "confidence_stop": float(raw.get("confidence_stop", 0.0)),
            }
        else:
            decision_payload = {"decision": "none"}

        # 3. Build vehicle list
        player_dict: Dict[str, Any] = {
            "id":        "PLAYER",
            "x":         self._world.my_car.x,
            "y":         self._world.my_car.y,
            "speed":     self._world.my_car.speed,
            "direction": self._world.my_car.direction,
            "is_player": True,
        }
        traffic_dicts: List[Dict[str, Any]] = [
            {
                "id":        f"TRAFFIC_{i}",
                "x":         car.x,
                "y":         car.y,
                "speed":     car.speed,
                "direction": car.direction,
                "is_player": False,
            }
            for i, car in enumerate(self._world.traffic)
        ]
        all_vehicles = [player_dict] + traffic_dicts

        # 4. Publish to real V2X bus (keeps bus metrics alive)
        self._bus.publish(
            topic="v2v.state",
            sender="PLAYER",
            payload={
                **decision_payload,
                "position": self._world.my_car.as_dict(),
                "traffic":  [c.as_dict() for c in self._world.traffic],
                "sign":     self._world.current_sign,
            },
        )

        # 5. Build intersection metadata
        # The UI reads "signs" as {"N": "STOP", "S": "YIELD", ...}
        # We broadcast the same sign on all four approaches for now.
        # If the world gains per-approach signs, update this mapping.
        sign = self._world.current_sign
        intersection_meta: Dict[str, Any] = {
            "signs":      {"N": sign, "S": sign, "E": sign, "W": sign},
            "lane_count": 2,
            "box_size":   100,
        }

        # 6. Build decisions dict keyed by UPPER vehicle ID
        decisions: Dict[str, Any] = {"PLAYER": decision_payload}
        # Traffic cars use the same world-level ML input for now.
        # Extend here when each car gets its own inference call.
        for i in range(len(self._world.traffic)):
            decisions[f"TRAFFIC_{i}"] = {"decision": "none"}

        # 7. Advance physics
        self._world.update_physics(dt=dt)

        # 8. Atomic swap
        with self._lock:
            self._vehicles     = all_vehicles
            self._decisions    = decisions
            self._intersection = intersection_meta
