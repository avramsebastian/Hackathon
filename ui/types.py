#!/usr/bin/env python3
"""Shared type aliases and data structures for the UI layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pygame

ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]


@dataclass
class VehicleRenderState:
    """Per-vehicle visual state tracked across frames."""

    color: ColorRGB
    render_pos: pygame.Vector2
    path_points: List[pygame.Vector2] = field(default_factory=list)
    path_index: int = 1
    travel_vec: pygame.Vector2 = field(default_factory=lambda: pygame.Vector2(1.0, 0.0))
    approach_vec: pygame.Vector2 = field(default_factory=pygame.Vector2)
    zone_from: ColorRGBA = (0, 0, 0, 0)
    zone_target: ColorRGBA = (0, 0, 0, 0)
    zone_current: ColorRGBA = (0, 0, 0, 0)
    zone_frame: int = 20
    in_detection: bool = False
    speed_scale_current: float = 1.0
