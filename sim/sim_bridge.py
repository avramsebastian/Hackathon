"""
sim/sim_bridge.py
=================
Runs the simulation loop in a background thread and exposes the methods
that PygameIntersectionView._poll_bus() expects:

    bridge.get_vehicles()               → List[dict]
    bridge.get_ml_decision(vehicle_id)  → dict  {"decision", "confidence_go", "confidence_stop"}
    bridge.get_intersection()           → dict  {"signs", "lane_count", "box_size"}
    bridge.is_finished()                → bool
    bridge.reset()                      → None

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
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

from sim.world import World, Car
from bus.v2x_bus import V2XBus
from comunication.Inference import fa_inferenta_din_json

log = logging.getLogger("sim_bridge")

# ── constants shared with UI ──────────────────────────────────────────────────
_SPEED_UNIT = "kmh"

# Vehicle palette (same order as ViewConstants.DEFAULT_VEHICLE_COLORS)
_VEHICLE_COLORS: Sequence[Tuple[int, int, int]] = (
    (86, 168, 255),
    (255, 88, 88),
    (100, 226, 170),
    (246, 191, 90),
    (180, 120, 255),
    (255, 160, 100),
)

# Cardinal direction the car is *heading towards* given its velocity vector.
_VEC_TO_CARDINAL: Dict[Tuple[int, int], str] = {
    (1, 0): "EAST",
    (-1, 0): "WEST",
    (0, 1): "NORTH",
    (0, -1): "SOUTH",
}


def _cardinal_direction(car: Car) -> str:
    """Map a car's velocity vector to a cardinal direction string for the UI."""
    return _VEC_TO_CARDINAL.get((int(car.vx), int(car.vy)), "NORTH")


def _road_line(car: Car) -> List[Tuple[float, float]]:
    """
    Compute a three-point road_line (start → centre → end) from the car's
    approach side and velocity.  The line is 200 m long, centred on the
    intersection, offset to the correct lane.
    """
    # The car's current lane offset (perpendicular to travel axis)
    if car.vx != 0:
        # Horizontal travel → lane offset is in Y
        lane_y = car.y if abs(car.y) > 0.1 else 0.0
        return [
            (-100.0 * car.vx, lane_y),
            (0.0, lane_y),
            (100.0 * car.vx, lane_y),
        ]
    else:
        # Vertical travel → lane offset is in X
        lane_x = car.x if abs(car.x) > 0.1 else 0.0
        return [
            (lane_x, -100.0 * car.vy),
            (lane_x, 0.0),
            (lane_x, 100.0 * car.vy),
        ]


class SimBridge:
    """
    Simulation orchestrator + bus adapter.

    The background thread ticks World → ML → V2XBus every ``1/tick_rate_hz``
    seconds and caches the results.  The UI thread calls the public methods
    to read the latest snapshot without blocking.
    """

    def __init__(
        self,
        tick_rate_hz: float = 10.0,
        drop_rate: float = 0.0,
        latency_ms: int = 0,
        model_path: Optional[str] = None,
        vehicle_count: int = 6,
        random_seed: Optional[int] = None,
    ) -> None:
        self._tick_rate_hz = tick_rate_hz
        self._model_path = model_path or os.path.join(
            _ROOT, "ml", "generated", "traffic_model.pkl"
        )
        self._drop_rate = drop_rate
        self._latency_ms = latency_ms

        self._world = World(num_cars=vehicle_count, seed=random_seed)
        self._bus = V2XBus(drop_rate=drop_rate, latency_ms=latency_ms)

        self._lock = threading.Lock()

        # Cached state — written by sim thread, read by UI thread
        self._vehicles: List[Dict[str, Any]] = []
        self._decisions: Dict[str, Any] = {}
        self._intersection: Dict[str, Any] = {}
        self._color_by_id: Dict[str, Tuple[int, int, int]] = {}

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background simulation loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
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

    # ── Bus adapter API ───────────────────────────────────────────────────────

    def get_vehicles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._vehicles)

    def get_ml_decision(self, vehicle_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._decisions.get(vehicle_id.upper(), {"decision": "none"}))

    def get_all_ml_decisions(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {vid: dict(dec) for vid, dec in self._decisions.items()}

    def get_intersection(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._intersection)

    def is_finished(self) -> bool:
        """True when all cars have stopped after passing through."""
        return self._world.is_finished()

    def reset(self) -> None:
        """Re-initialise the world so the scenario replays."""
        self._world.reset()
        with self._lock:
            self._vehicles = []
            self._decisions = {}
            self._color_by_id = {}
        log.info("SimBridge reset")

    def set_paused(self, paused: bool) -> None:
        """Pause / unpause the simulation tick."""
        self._paused = paused

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        dt = 1.0 / self._tick_rate_hz
        while self._running:
            t0 = time.perf_counter()
            if not self._paused and not self._world.is_finished():
                try:
                    self._tick(dt)
                except Exception:
                    log.exception("SimBridge tick error")
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    # ── helpers: build rich vehicle dict ──────────────────────────────────────

    @staticmethod
    def _make_vehicle_dict(
        car: Car, color: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        return {
            "id": car.id,
            "x": car.x,
            "y": car.y,
            "speed": car.speed,
            "speed_unit": _SPEED_UNIT,
            "direction": _cardinal_direction(car),
            "approach": car.approach,
            "color": color,
            "road_line": _road_line(car),
        }

    # ── tick ──────────────────────────────────────────────────────────────────

    def _infer_for_car(self, ego_car: Car, others: Sequence[Car]) -> Dict[str, Any]:
        """
        Run ML inference for one standalone vehicle entity.
        """
        raw = fa_inferenta_din_json(
            ego_car.ml_payload(self._world.current_sign, others),
            model_path=self._model_path,
        )
        if raw.get("status") == "success":
            return {
                "decision": raw["decision"],
                "confidence_go": float(raw.get("confidence_go", 0.0)),
                "confidence_stop": float(raw.get("confidence_stop", 0.0)),
            }
        return {"decision": "none"}

    def _color_for_car(self, car_id: str) -> Tuple[int, int, int]:
        existing = self._color_by_id.get(car_id)
        if existing:
            return existing
        color = _VEHICLE_COLORS[len(self._color_by_id) % len(_VEHICLE_COLORS)]
        self._color_by_id[car_id] = color
        return color

    def _effective_decisions_from_bus(
        self,
        defaults: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resolve decisions consumed by world physics from V2X bus messages.

        This keeps a concrete flow: ML -> bus publish -> world control.
        If a message is dropped/missing, fallback remains the ML decision.
        """
        effective = {vid: dict(dec) for vid, dec in defaults.items()}
        for msg in self._bus.poll("v2v.state"):
            sender = str(msg.sender).upper()
            if sender not in effective:
                continue
            decision = str(msg.payload.get("decision", "none")).upper()
            if decision in ("GO", "STOP"):
                effective[sender]["decision"] = decision
        return effective

    def _tick(self, dt: float) -> None:
        all_cars = self._world.all_cars()

        # 1. Build local neighborhood view for each car
        others_by_id: Dict[str, List[Car]] = {}
        for ego in all_cars:
            others_by_id[ego.id] = [other for other in all_cars if other is not ego]

        # 2. Compute an ML decision for each car instance
        decisions: Dict[str, Any] = {}
        for car in all_cars:
            decisions[car.id] = self._infer_for_car(car, others_by_id[car.id])

        # 3. Publish V2X state from each standalone car entity
        for car in all_cars:
            self._bus.publish(
                topic="v2v.state",
                sender=car.id,
                payload=car.v2x_payload(
                    sign=self._world.current_sign,
                    others=others_by_id[car.id],
                    decision=decisions.get(car.id, {"decision": "none"}),
                ),
            )

        # 4. Feed physics through bus-resolved decisions + safety guard.
        effective_decisions = self._effective_decisions_from_bus(decisions)
        self._world.update_physics(dt=dt, decisions=effective_decisions)

        moved_cars = self._world.all_cars()

        # 5. Build vehicle list with rich UI fields from updated world state.
        all_vehicles: List[Dict[str, Any]] = [
            self._make_vehicle_dict(car, self._color_for_car(car.id))
            for car in moved_cars
        ]

        # 6. Build intersection metadata (UI-compatible + extensible)
        sign = self._world.current_sign
        intersection_meta: Dict[str, Any] = {
            "signs": {"N": sign, "S": sign, "E": sign, "W": sign},
            "lane_count": 2,
            "box_size": 100,
            "safety_interventions": self._world.safety_interventions,
            "collision_resolutions": self._world.collision_resolutions,
            "intersections": [
                {
                    "id": "INT_000",
                    "center": [0.0, 0.0],
                    "box_size": 100,
                    "signs": {"N": sign, "S": sign, "E": sign, "W": sign},
                }
            ],
        }

        # 7. Atomic swap
        with self._lock:
            self._vehicles = all_vehicles
            self._decisions = effective_decisions
            self._intersection = intersection_meta
