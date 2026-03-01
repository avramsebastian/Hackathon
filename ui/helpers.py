"""
ui/helpers.py
=============
Pure utility functions shared across UI modules:
coordinate transforms, alpha-surface drawing, direction ↔ heading mapping,
interpolation, and awareness-zone logic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any, Optional

import pygame

from ui.constants import AWARENESS_DIVISOR

# ── Direction / heading ───────────────────────────────────────────────────────

_DIR_TO_HEADING: Dict[str, float] = {
    "EAST":  0.0,
    "NORTH": 90.0,
    "WEST":  180.0,
    "SOUTH": 270.0,
}


def direction_to_heading(direction: str) -> float:
    """Convert a cardinal direction string to degrees (0 = East, CCW)."""
    return _DIR_TO_HEADING.get(direction.upper(), 0.0)


# ── Interpolation helpers ─────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def angle_lerp(a: float, b: float, t: float) -> float:
    """Shortest-arc angle interpolation (degrees)."""
    diff = (b - a) % 360
    if diff > 180:
        diff -= 360
    return (a + diff * t) % 360


def interpolate_vehicles(
    prev: List[Dict[str, Any]],
    curr: List[Dict[str, Any]],
    t: float,
) -> List[Dict[str, Any]]:
    """Return a new vehicle list with positions lerped from *prev* to *curr*."""
    if not prev:
        return curr
    prev_map = {v["id"]: v for v in prev}
    result = []
    for v in curr:
        p = prev_map.get(v["id"])
        if p is None:
            result.append(v)
            continue
        out = dict(v)
        out["x"] = lerp(p["x"], v["x"], t)
        out["y"] = lerp(p["y"], v["y"], t)
        out["speed"] = lerp(p["speed"], v["speed"], t)
        # Interpolate velocity direction for smooth visual rotation.
        if "vx" in v and "vx" in p:
            out["vx"] = lerp(p["vx"], v["vx"], t)
            out["vy"] = lerp(p["vy"], v["vy"], t)
        result.append(out)
    return result


# ── Awareness / slow-down ────────────────────────────────────────────────────

def is_approaching(vehicle: Dict[str, Any]) -> bool:
    """True if the vehicle hasn't yet reached its current intersection centre."""
    cx = vehicle.get("int_cx", 0.0)
    cy = vehicle.get("int_cy", 0.0)
    x = vehicle["x"] - cx
    y = vehicle["y"] - cy
    d = vehicle.get("direction", "").upper()
    if d == "EAST":  return x < 0
    if d == "WEST":  return x > 0
    if d == "NORTH": return y < 0
    if d == "SOUTH": return y > 0
    return False


def should_slow_down(vehicle: Dict[str, Any], ml_decision: str) -> bool:
    """
    Combine awareness-distance rule with ML decision.

    awareness_distance_m = speed_kmh / 5
    should_slow_down = STOP decision OR (approaching AND within awareness zone)
    """
    cx = vehicle.get("int_cx", 0.0)
    cy = vehicle.get("int_cy", 0.0)
    dist = math.hypot(vehicle["x"] - cx, vehicle["y"] - cy)
    awareness = vehicle.get("speed", 0) / AWARENESS_DIVISOR
    geo_warning = is_approaching(vehicle) and dist <= awareness
    return ml_decision == "STOP" or geo_warning


# ── Alpha drawing helpers ────────────────────────────────────────────────────

def draw_alpha_rect(
    target: pygame.Surface,
    color: Tuple[int, ...],
    rect: pygame.Rect,
    border_radius: int = 0,
) -> None:
    """Draw a semi-transparent rectangle (colour tuple with 4 channels)."""
    tmp = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    pygame.draw.rect(tmp, color, (0, 0, rect.w, rect.h), border_radius=border_radius)
    target.blit(tmp, rect.topleft)


def draw_alpha_circle(
    target: pygame.Surface,
    color: Tuple[int, ...],
    centre: Tuple[int, int],
    radius: int,
) -> None:
    """Draw a semi-transparent circle."""
    if radius < 1:
        return
    size = radius * 2
    tmp = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(tmp, color, (radius, radius), radius)
    target.blit(tmp, (centre[0] - radius, centre[1] - radius))


# ── Text helper ──────────────────────────────────────────────────────────────

def render_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, ...] = (230, 230, 235),
    anchor: str = "topleft",
) -> pygame.Rect:
    """Render text with flexible *anchor* ('topleft', 'center', 'midright' …)."""
    img = font.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect
