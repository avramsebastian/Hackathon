"""
ui/types.py
===========
Lightweight data containers used across every UI module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class Camera:
    """Viewport mapping world coordinates to screen pixels."""
    screen_w: int
    screen_h: int
    world_x: float = 0.0
    world_y: float = 0.0
    zoom: float = 3.0

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        cx = self.screen_w / 2
        cy = self.screen_h / 2
        sx = cx + (wx - self.world_x) * self.zoom
        sy = cy - (wy - self.world_y) * self.zoom
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        cx = self.screen_w / 2
        cy = self.screen_h / 2
        wx = (sx - cx) / self.zoom + self.world_x
        wy = -((sy - cy) / self.zoom) + self.world_y
        return wx, wy


@dataclass
class VehicleSnapshot:
    """Smoothed vehicle state for interpolation between bridge ticks."""
    x: float
    y: float
    speed: float
    direction: str
    heading_deg: float


@dataclass
class ButtonRect:
    """Stores a button's screen rect and label for click detection."""
    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, mx: int, my: int) -> bool:
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h
