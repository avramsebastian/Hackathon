"""
sim/sim_bridge.py
=================
Background-thread orchestrator tying :mod:`sim.world`, ML inference and
the :class:`bus.v2x_bus.V2XBus` together.  The UI polls the bridge for
the latest snapshot without blocking.

Public API consumed by :mod:`ui.pygame_view`
--------------------------------------------
* ``get_vehicles()``          → ``List[dict]``
* ``get_ml_decision(id)``     → ``dict``
* ``get_all_ml_decisions()``  → ``Dict[str, dict]``
* ``get_intersection()``      → ``dict``
* ``is_finished()``           → ``bool``
* ``reset()``                 → ``None``
* ``set_paused(bool)``        → ``None``
"""

from __future__ import annotations

import os
import sys
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
# The ML package uses relative imports that assume its own sub-packages
# are on sys.path.  We add them here once so the rest of the codebase
# can import ``from comunication.Inference import …`` without per-file hacks.
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
from sim.traffic_policy import SafetyPolicy, danger_score
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
    Compute a road_line polyline showing the car's planned route through
    the intersection based on its approach arm and intended manoeuvre
    (FORWARD / LEFT / RIGHT).

    Right turns follow a tight arc near the intersection corner.
    Left turns cross through the intersection centre with a wider arc.
    """
    L = 7.0  # Must stay in sync with SafetyPolicy.lane_offset_m
    R = 3.0  # Curve smoothing offset for turn waypoints

    approach = getattr(car, "approach", "")
    ml_dir  = getattr(car, "ml_direction", "FORWARD")

    _lines: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
        # ── W approach (eastbound at y = -L) ──────────────────────────
        ("W", "FORWARD"): [(-100, -L), (0, -L), (100, -L)],
        ("W", "RIGHT"):   [(-100, -L), (-L - R, -L), (-L, -L - R), (-L, -100)],
        ("W", "LEFT"):    [(-100, -L), (-R, -L), (R, -R), (L, R), (L, 100)],
        # ── E approach (westbound at y = +L) ──────────────────────────
        ("E", "FORWARD"): [(100, L), (0, L), (-100, L)],
        ("E", "RIGHT"):   [(100, L), (L + R, L), (L, L + R), (L, 100)],
        ("E", "LEFT"):    [(100, L), (R, L), (-R, R), (-L, -R), (-L, -100)],
        # ── N approach (southbound at x = -L) ────────────────────────
        ("N", "FORWARD"): [(-L, 100), (-L, 0), (-L, -100)],
        ("N", "RIGHT"):   [(-L, 100), (-L, L + R), (-L - R, L), (-100, L)],
        ("N", "LEFT"):    [(-L, 100), (-L, R), (-R, -R), (R, -L), (100, -L)],
        # ── S approach (northbound at x = +L) ────────────────────────
        ("S", "FORWARD"): [(L, -100), (L, 0), (L, 100)],
        ("S", "RIGHT"):   [(L, -100), (L, -L - R), (L + R, -L), (100, -L)],
        ("S", "LEFT"):    [(L, -100), (L, -R), (R, R), (-R, L), (-100, L)],
    }

    line = _lines.get((approach, ml_dir))
    if line:
        return line

    # Fallback: straight line using velocity (legacy / unknown approach)
    if car.vx != 0:
        lane_y = car.y if abs(car.y) > 0.1 else 0.0
        return [
            (-100.0 * car.vx, lane_y),
            (0.0, lane_y),
            (100.0 * car.vx, lane_y),
        ]
    else:
        lane_x = car.x if abs(car.x) > 0.1 else 0.0
        return [
            (lane_x, -100.0 * car.vy),
            (lane_x, 0.0),
            (lane_x, 100.0 * car.vy),
        ]


class SimBridge:
    """Simulation orchestrator running in a background thread.

    The thread calls :meth:`_tick` at ``tick_rate_hz``, advancing
    :class:`~sim.world.World`, running ML inference for every car,
    routing decisions through the :class:`~bus.v2x_bus.V2XBus`, and
    caching the results for the UI thread.

    Parameters
    ----------
    tick_rate_hz : float
        Simulation ticks per second.
    drop_rate : float
        V2X bus packet drop probability (0.0–1.0).
    latency_ms : int
        Simulated bus latency in milliseconds.
    model_path : str or None
        Path to the ML model ``.pkl`` file.
    vehicle_count : int
        Number of cars to spawn.
    random_seed : int or None
        Seed for reproducibility.
    priority_axis : str
        ``'EW'`` or ``'NS'``.
    policy : SafetyPolicy or None
        Tunable constants.
    """

    def __init__(
        self,
        tick_rate_hz: float = 10.0,
        drop_rate: float = 0.0,
        latency_ms: int = 0,
        model_path: Optional[str] = None,
        vehicle_count: int = 6,
        random_seed: Optional[int] = None,
        priority_axis: str = "EW",
        policy: Optional[SafetyPolicy] = None,
    ) -> None:
        self._tick_rate_hz = tick_rate_hz
        self._model_path = model_path or os.path.join(
            _ROOT, "ml", "generated", "traffic_model.pkl"
        )
        self._drop_rate = drop_rate
        self._latency_ms = latency_ms

        self._world = World(
            num_cars=vehicle_count,
            seed=random_seed,
            policy=policy,
            priority_axis=priority_axis,
        )
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
        """Spawn the background simulation thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="SimBridge"
        )
        self._thread.start()
        log.info("SimBridge started at %.1f Hz", self._tick_rate_hz)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to join."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("SimBridge stopped")

    # ── Bus adapter API ───────────────────────────────────────────────────────

    def get_vehicles(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._vehicles)

    def get_ml_decision(self, vehicle_id: str) -> Dict[str, Any]:
        """Return the latest ML decision for one vehicle."""
        with self._lock:
            return dict(self._decisions.get(vehicle_id.upper(), {"decision": "none"}))

    def get_all_ml_decisions(self) -> Dict[str, Dict[str, Any]]:
        """Return all ML decisions keyed by vehicle ID."""
        with self._lock:
            return {vid: dict(dec) for vid, dec in self._decisions.items()}

    def get_intersection(self) -> Dict[str, Any]:
        """Return intersection metadata (signs, metrics, etc.)."""
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

    def _make_vehicle_dict(
        self,
        car: Car,
        color: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        return {
            "id": car.id,
            "x": car.x,
            "y": car.y,
            "speed": car.speed,
            "speed_unit": _SPEED_UNIT,
            "direction": _cardinal_direction(car),
            "approach": car.approach,
            "role": car.role,
            "priority_score": danger_score(car, self._world.policy),
            "color": color,
            "road_line": _road_line(car),
        }

    # ── tick ──────────────────────────────────────────────────────────────────

    def _infer_for_car(self, ego_car: Car, others: Sequence[Car]) -> Dict[str, Any]:
        """
        Run ML inference for one standalone vehicle entity.
        """
        raw = fa_inferenta_din_json(
            ego_car.ml_payload(self._world.sign_for_car(ego_car), others),
            model_path=self._model_path,
        )
        if raw.get("status") == "success":
            result: Dict[str, Any] = {
                "decision": str(raw.get("decision", "none")).upper(),
                "confidence_go": float(raw.get("confidence_go", 0.0)),
                "confidence_stop": float(raw.get("confidence_stop", 0.0)),
            }
            # Pass through extra ML outputs (e.g., target_speed_kmh) for
            # scalable control contracts without changing world internals.
            for key, value in raw.items():
                if key in {"status", "decision", "confidence_go", "confidence_stop"}:
                    continue
                result[key] = value
            return result
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
        all_cars: Sequence[Car],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Poll decisions delivered via the ``i2v.command`` bus topic.

        Flow: ML inference → bus.publish("i2v.command") → bus.poll → here.

        If a decision message was *dropped* or *delayed* by the bus
        (simulated packet loss / latency), the safe fallback for that
        vehicle is **STOP** — modelling what a real V2X-equipped car
        should do when it loses communication with infrastructure.
        """
        # Safe default: every car STOPs unless a decision message arrives.
        effective: Dict[str, Dict[str, Any]] = {
            car.id: {
                "decision": "STOP",
                "confidence_go": 0.0,
                "confidence_stop": 1.0,
            }
            for car in all_cars
        }

        for msg in self._bus.poll("i2v.command"):
            payload = msg.payload if isinstance(msg.payload, dict) else {}
            vehicle_id = str(payload.get("vehicle_id", "")).upper()
            if not vehicle_id or vehicle_id not in effective:
                continue

            decision_data: Dict[str, Any] = {}
            for key in ("decision", "confidence_go", "confidence_stop", "target_speed_kmh"):
                if key not in payload:
                    continue
                val = payload[key]
                if key == "decision":
                    decision_data[key] = str(val).upper()
                else:
                    try:
                        decision_data[key] = float(val)
                    except (TypeError, ValueError):
                        pass

            if decision_data:
                effective[vehicle_id] = decision_data

        return effective

    def _tick(self, dt: float) -> None:
        all_cars = self._world.all_cars()

        # 1. Build local neighbourhood view for each car.
        others_by_id: Dict[str, List[Car]] = {}
        for ego in all_cars:
            others_by_id[ego.id] = [other for other in all_cars if other is not ego]

        # 2. Each car broadcasts its *state* on the V2V channel.
        #    (No decision here — just position, speed, sign, neighbours.)
        for car in all_cars:
            self._bus.publish(
                topic="v2v.state",
                sender=car.id,
                payload=car.state_payload(
                    sign=self._world.sign_for_car(car),
                    others=others_by_id[car.id],
                ),
            )

        # 3. Infrastructure / edge ML: compute a decision for every car.
        raw_decisions: Dict[str, Dict[str, Any]] = {}
        for car in all_cars:
            raw_decisions[car.id] = self._infer_for_car(car, others_by_id[car.id])

        # 4. Publish each ML decision on the I2V command channel.
        #    These messages are subject to bus drop_rate & latency_ms,
        #    so the car may never receive them → safe fallback = STOP.
        for car in all_cars:
            dec = raw_decisions.get(car.id, {"decision": "none"})
            self._bus.publish(
                topic="i2v.command",
                sender="INFRA",
                payload={"vehicle_id": car.id, **dec},
            )

        # 5. Resolve effective decisions from bus-delivered messages.
        effective_decisions = self._effective_decisions_from_bus(all_cars)

        # 6. Feed physics through bus-resolved decisions + safety guard.
        self._world.update_physics(dt=dt, decisions=effective_decisions)

        moved_cars = self._world.all_cars()

        # 7. Build vehicle list with rich UI fields from updated world state.
        all_vehicles: List[Dict[str, Any]] = [
            self._make_vehicle_dict(car, self._color_for_car(car.id))
            for car in moved_cars
        ]

        # 8. Build intersection metadata (UI-compatible + extensible).
        signs = self._world.get_signs()
        intersection_meta: Dict[str, Any] = {
            "signs": signs,
            "lane_count": 2,
            "box_size": 100,
            "green_approach": self._world.green_approach,
            "safety_interventions": self._world.safety_interventions,
            "collision_resolutions": self._world.collision_resolutions,
            "bus_metrics": self._bus.metrics.report(),
            "intersections": [
                {
                    "id": "INT_000",
                    "center": [0.0, 0.0],
                    "box_size": 100,
                    "signs": signs,
                }
            ],
        }

        # 9. Atomic swap — UI thread reads these via public methods.
        with self._lock:
            self._vehicles = all_vehicles
            self._decisions = effective_decisions
            self._intersection = intersection_meta
