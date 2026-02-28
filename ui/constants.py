#!/usr/bin/env python3
"""Visual constants shared across all renderers."""

from __future__ import annotations

from typing import Sequence, Tuple

from .types import ColorRGB


class ViewConstants:
    """Mixin providing every visual / layout constant."""

    BG_COLOR: ColorRGB = (15, 15, 15)
    ROAD_COLOR: ColorRGB = (30, 30, 30)
    LANE_DASH_COLOR: ColorRGB = (58, 58, 58)
    LANE_EDGE_COLOR: ColorRGB = (42, 42, 42)
    HUD_BG_COLOR: ColorRGB = (22, 22, 22)
    HUD_BORDER_COLOR: ColorRGB = (42, 42, 42)
    WARNING_COLOR: ColorRGB = (255, 60, 60)
    GO_COLOR: ColorRGB = (0, 255, 127)
    STOP_COLOR: ColorRGB = (255, 60, 60)
    CONFLICT_COLOR: ColorRGB = (255, 136, 0)

    ROAD_LINE_ALPHA = 72
    DETECTION_OUTLINE_ALPHA = 80
    ZONE_GO_ALPHA = 84
    ZONE_STOP_ALPHA = 84
    ZONE_TRANSITION_FRAMES = 20
    CONFLICT_ALPHA_MIN = 60
    CONFLICT_ALPHA_MAX = 120
    HUD_BLINK_MS = 500

    PIXELS_PER_METER = 2.0  # 1 px = 0.5 m
    INTERSECTION_BOX_SIZE = 100
    LANE_WIDTH = 14
    DETECTION_DISTANCE_PX = 56
    DETECTION_RADIUS_PX = 20

    DEFAULT_VEHICLE_COLORS: Sequence[ColorRGB] = (
        (86, 168, 255),
        (255, 88, 88),
        (100, 226, 170),
        (246, 191, 90),
        (180, 120, 255),
        (255, 160, 100),
    )

    LEGEND_ITEMS: Sequence[Tuple[str, ColorRGB]] = (
        ("GO", (0, 255, 127)),
        ("STOP", (255, 60, 60)),
        ("CONFLICT", (255, 136, 0)),
    )

    SPEED_UNIT = "kmh"

    SCREENSHOT_DIR = "screenshots"
