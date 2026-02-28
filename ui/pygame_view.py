#!/usr/bin/env python3
"""
Main view class — combines all UI mixins into one runnable Pygame window.

Module layout
─────────────
    ui/
    ├── types.py           – ColorRGB, ColorRGBA, VehicleRenderState
    ├── constants.py       – ViewConstants mixin (all class-level constants)
    ├── helpers.py         – ViewHelpers mixin  (static utilities)
    ├── draw_road.py       – RoadRenderer mixin (roads, lanes, signs)
    ├── draw_vehicles.py   – VehicleRenderer mixin (sprites, zones, animation)
    ├── hud.py             – HudRenderer mixin  (HUD, legend, debug, splash)
    └── pygame_view.py     – PygameIntersectionView (this file – main loop)
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pygame

from .constants import ViewConstants
from .draw_road import RoadRenderer
from .draw_vehicles import VehicleRenderer
from .helpers import ViewHelpers
from .hud import HudRenderer
from .types import VehicleRenderState


class PygameIntersectionView(
    ViewConstants,
    ViewHelpers,
    RoadRenderer,
    VehicleRenderer,
    HudRenderer,
):
    """Intersection-safety visualiser powered by Pygame.

    Inherits drawing logic from focused mixin modules so each file
    stays small and single-purpose.
    """

    # Direction → one-hot index (matches ml/entities/Directions.py enum order)
    _DIR_ONEHOT = {"LEFT": [1,0,0], "RIGHT": [0,1,0], "FORWARD": [0,0,1]}
    # Sign → one-hot index (matches ml/entities/Sign.py enum order)
    _SIGN_ONEHOT = {"STOP": [1,0,0,0], "YIELD": [0,1,0,0], "PRIORITY": [0,0,1,0], "NO_SIGN": [0,0,0,1]}

    def __init__(self, bus: Any, width: int = 1000, height: int = 700, fps: int = 60):
        self.bus = bus
        self.width = width
        self.height = height
        self.fps = fps

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_tiny: Optional[pygame.font.Font] = None
        self.font_title: Optional[pygame.font.Font] = None

        self.center = pygame.Vector2(width * 0.5, height * 0.5)
        self.time_seconds = 0.0
        self.vehicle_states: Dict[str, VehicleRenderState] = {}

        # UI state
        self.paused = False
        self.show_debug = False
        self.show_legend = True
        self.show_splash = True
        self.zoom = 1.0
        self._hud_scroll_offset = 0
        self._conflict_total = 0
        self._screenshot_flash_until = 0.0

        # Pre-load ML model once so we can run inference for every car
        self._ml_model = None
        try:
            import joblib
            _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            model_path = os.path.join(_root, "ml", "generated", "traffic_model.pkl")
            self._ml_model = joblib.load(model_path)
        except Exception:
            pass  # ML unavailable — decisions will stay as-is

    # ------------------------------------------------------------------ #
    #  Resize                                                              #
    # ------------------------------------------------------------------ #
    def _handle_resize(self, new_w: int, new_h: int) -> None:
        self.width = max(400, new_w)
        self.height = max(300, new_h)
        self.center = pygame.Vector2(self.width * 0.5, self.height * 0.5)
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )

    # ------------------------------------------------------------------ #
    #  Screenshot                                                          #
    # ------------------------------------------------------------------ #
    def _take_screenshot(self) -> None:
        if self.screen is None:
            return
        os.makedirs(self.SCREENSHOT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.SCREENSHOT_DIR, f"sim_{stamp}.png")
        pygame.image.save(self.screen, path)
        self._screenshot_flash_until = self.time_seconds + 0.35

    # ------------------------------------------------------------------ #
    #  Per-vehicle ML inference                                            #
    # ------------------------------------------------------------------ #
    def _infer_for_vehicle(
        self,
        vehicle: Any,
        all_vehicles: Sequence[Any],
        sign: str,
    ) -> Dict[str, Any]:
        """Run ML inference from *vehicle*'s perspective.

        Builds a 22-feature vector identical to
        ``ml.entities.Intersections.get_feature_vector`` and predicts
        GO / STOP using the cached model.  Returns a decision dict
        compatible with what the bridge produces.
        """
        if self._ml_model is None:
            return {"decision": "none"}

        # ── my_car features (6): x, y, speed, direction_onehot(3) ────────
        mx = float(self._get(vehicle, "x", default=0.0))
        my = float(self._get(vehicle, "y", default=0.0))
        ms = float(self._get(vehicle, "speed", default=0.0))
        md = str(self._get(vehicle, "direction", default="FORWARD")).strip().upper()
        features: List[float] = [mx, my, ms]
        features.extend(self._DIR_ONEHOT.get(md, [0, 0, 1]))  # default FORWARD

        # ── sign features (4): onehot ────────────────────────────────────
        features.extend(self._SIGN_ONEHOT.get(sign, [0, 0, 0, 1]))

        # ── 2 closest other cars (12): (x, y, speed, dir_onehot) × 2 ────
        my_id = self._vehicle_id(vehicle)
        others = [v for v in all_vehicles if self._vehicle_id(v) != my_id]
        others.sort(
            key=lambda v: math.hypot(
                float(self._get(v, "x", default=0.0)),
                float(self._get(v, "y", default=0.0)),
            )
        )
        for i in range(2):
            if i < len(others):
                o = others[i]
                features.extend([
                    float(self._get(o, "x", default=0.0)),
                    float(self._get(o, "y", default=0.0)),
                    float(self._get(o, "speed", default=0.0)),
                ])
                od = str(self._get(o, "direction", default="FORWARD")).strip().upper()
                features.extend(self._DIR_ONEHOT.get(od, [0, 0, 1]))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        try:
            inp = np.array(features, dtype=float).reshape(1, -1)
            probs = self._ml_model.predict_proba(inp)[0]
            prob_stop, prob_go = float(probs[0]), float(probs[1])
            return {
                "decision": "GO" if prob_go > 0.5 else "STOP",
                "confidence_go": prob_go,
                "confidence_stop": prob_stop,
            }
        except Exception:
            return {"decision": "none"}

    # ------------------------------------------------------------------ #
    #  Bus polling                                                         #
    # ------------------------------------------------------------------ #
    def _poll_bus(self) -> Tuple[List[Any], Dict[str, Any], Any]:
        vehicles_payload = []
        if hasattr(self.bus, "get_vehicles"):
            vehicles_payload = self.bus.get_vehicles() or []

        vehicles: List[Any]
        intersection: Any = {}

        # ── New ML-input format: {"my_car": {...}, "sign": ..., "traffic": [...]}
        if isinstance(vehicles_payload, Mapping) and "my_car" in vehicles_payload:
            vehicles = []
            my_car = dict(vehicles_payload["my_car"])
            my_car.setdefault("id", "PLAYER")
            vehicles.append(my_car)
            for i, t in enumerate(vehicles_payload.get("traffic", [])):
                td = dict(t)
                td.setdefault("id", f"TRAFFIC_{i}")
                vehicles.append(td)
            sign = str(vehicles_payload.get("sign", "YIELD")).strip().upper()
            intersection = {
                "signs": {"N": sign, "S": sign, "E": sign, "W": sign},
                "lane_count": 2,
                "box_size": 100,
            }
        elif isinstance(vehicles_payload, Mapping):
            vehicles = []
            for key, raw_vehicle in vehicles_payload.items():
                if isinstance(raw_vehicle, Mapping):
                    merged = dict(raw_vehicle)
                    merged.setdefault("id", key)
                    vehicles.append(merged)
                else:
                    vehicles.append(raw_vehicle)
        else:
            vehicles = list(vehicles_payload)

        decisions: Dict[str, Any] = {}
        if hasattr(self.bus, "get_ml_decision"):
            for vehicle in vehicles:
                vehicle_id = self._vehicle_id(vehicle)
                try:
                    decisions[vehicle_id] = self.bus.get_ml_decision(vehicle_id)
                except Exception:
                    decisions[vehicle_id] = "none"

        if not intersection:
            if hasattr(self.bus, "get_intersection"):
                try:
                    intersection = self.bus.get_intersection() or {}
                except Exception:
                    intersection = {}

        # ── Run ML inference for every vehicle that lacks a real decision ─
        if self._ml_model is not None and vehicles:
            # Determine each vehicle's approach-side sign
            signs_map = self._extract_signs(intersection) if intersection else {}
            default_sign = (
                str(intersection.get("sign", "YIELD")).upper()
                if isinstance(intersection, Mapping)
                else "YIELD"
            )
            for vehicle in vehicles:
                vid = self._vehicle_id(vehicle)
                existing = decisions.get(vid)
                if self._normalize_decision(existing) not in ("go", "stop"):
                    # Pick sign for this vehicle's approach
                    approach = str(self._get(vehicle, "approach", default="")).strip().upper()[:1]
                    sign = signs_map.get(approach, default_sign)
                    decisions[vid] = self._infer_for_vehicle(vehicle, vehicles, sign)

        return vehicles, decisions, intersection

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("INTERSECTION SAFETY SIM")
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.font_small = self._load_font(13, bold=False)
        self.font_tiny = self._load_font(11, bold=False)
        self.font_title = self._load_font(28, bold=True)

        running = True
        while running:
            delta_time = self.clock.tick(self.fps) / 1000.0
            self.time_seconds += delta_time

            # ---- events ------------------------------------------------- #
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    if self.show_splash:
                        self.show_splash = False
                        continue
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        if hasattr(self.bus, 'set_paused'):
                            self.bus.set_paused(self.paused)
                    elif event.key == pygame.K_F3:
                        self.show_debug = not self.show_debug
                    elif event.key == pygame.K_l:
                        self.show_legend = not self.show_legend
                    elif event.key == pygame.K_r:
                        self.zoom = 1.0
                        self.vehicle_states.clear()
                        self._hud_scroll_offset = 0
                        self._conflict_total = 0
                        self.paused = False
                        if hasattr(self.bus, 'reset'):
                            self.bus.reset()
                        if hasattr(self.bus, 'set_paused'):
                            self.bus.set_paused(False)
                    elif event.key == pygame.K_F12:
                        self._take_screenshot()
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.zoom = min(3.0, self.zoom + 0.1)
                    elif event.key == pygame.K_MINUS:
                        self.zoom = max(0.3, self.zoom - 0.1)
                    elif event.key == pygame.K_UP:
                        self._hud_scroll_offset = max(0, self._hud_scroll_offset - 1)
                    elif event.key == pygame.K_DOWN:
                        self._hud_scroll_offset += 1

            # ---- splash ------------------------------------------------- #
            if self.show_splash:
                self.screen.fill(self.BG_COLOR)
                self._draw_splash(self.screen, self.time_seconds)
                pygame.display.flip()
                continue

            # ---- auto-pause when simulation finishes -------------------- #
            if (not self.paused
                    and hasattr(self.bus, 'is_finished')
                    and self.bus.is_finished()):
                self.paused = True
                if hasattr(self.bus, 'set_paused'):
                    self.bus.set_paused(True)

            # ---- simulation tick ---------------------------------------- #
            if not self.paused:
                vehicles, ml_decisions, intersection = self._poll_bus()
                self._last_vehicles = vehicles
                self._last_ml_decisions = ml_decisions
                self._last_intersection = intersection
            else:
                vehicles = getattr(self, '_last_vehicles', [])
                ml_decisions = getattr(self, '_last_ml_decisions', {})
                intersection = getattr(self, '_last_intersection', {})
            if not self.paused:
                self._sync_vehicle_states(vehicles)
                for vehicle in vehicles:
                    vehicle_id = self._vehicle_id(vehicle)
                    self.animate_vehicle(
                        vehicle,
                        ml_decisions.get(vehicle_id, "none"),
                        delta_time,
                    )

            # ---- render ------------------------------------------------- #
            self.screen.fill(self.BG_COLOR)

            # Apply zoom via a transform surface
            if abs(self.zoom - 1.0) > 0.01:
                world_surf = pygame.Surface(
                    (self.width, self.height), pygame.SRCALPHA
                )
            else:
                world_surf = self.screen

            self.draw_road(world_surf, intersection)
            self.draw_lane_markings(world_surf, intersection)
            self.draw_signs(world_surf, intersection)

            for vehicle in vehicles:
                self.draw_road_line(world_surf, vehicle)

            for vehicle in vehicles:
                vehicle_id = self._vehicle_id(vehicle)
                self.draw_detection_zone(
                    world_surf,
                    vehicle,
                    ml_decisions.get(vehicle_id, "none"),
                )

            vehicles_in_detection = sum(
                1 for state in self.vehicle_states.values() if state.in_detection
            )
            if vehicles_in_detection >= 2:
                self.draw_conflict_highlight(
                    world_surf, intersection, self.time_seconds
                )
                self._conflict_total += 1

            for vehicle in vehicles:
                self.draw_vehicle(world_surf, vehicle)

            # Blit zoomed world
            if world_surf is not self.screen:
                scaled_w = int(self.width * self.zoom)
                scaled_h = int(self.height * self.zoom)
                scaled = pygame.transform.smoothscale(
                    world_surf, (scaled_w, scaled_h)
                )
                self.screen.blit(
                    scaled,
                    (
                        (self.width - scaled_w) // 2,
                        (self.height - scaled_h) // 2,
                    ),
                )

            # HUD layers (drawn on top, unzoomed)
            self.draw_hud(self.screen, vehicles, ml_decisions, self.time_seconds)
            if self.show_legend:
                self._draw_legend(self.screen)
            if self.show_debug:
                self._draw_debug_overlay(self.screen, vehicles, delta_time)
            if self.paused:
                self._draw_pause_banner(self.screen)
            if self.time_seconds < self._screenshot_flash_until:
                flash = pygame.Surface(
                    (self.width, self.height), pygame.SRCALPHA
                )
                flash.fill((255, 255, 255, 40))
                self.screen.blit(flash, (0, 0))

            pygame.display.flip()

        pygame.quit()


# ---------------------------------------------------------------------- #
#  Convenience entry point                                                 #
# ---------------------------------------------------------------------- #
def run_pygame_view(
    bus: Any, width: int = 1000, height: int = 700, fps: int = 60
) -> None:
    view = PygameIntersectionView(bus=bus, width=width, height=height, fps=fps)
    view.run()


if __name__ == "__main__":
    raise SystemExit(
        "pygame_view.py needs a bus object. Run `python Hackathon/main.py` "
        "or call run_pygame_view(your_bus)."
    )
