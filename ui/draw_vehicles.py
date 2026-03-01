"""
ui/draw_vehicles.py
===================
Renders top-down car sprites, headlights, route lines, and
awareness-zone indicators.  Everything is heading-aware and colour-coded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pygame

from ui.constants import (
    CAR_LENGTH, CAR_WIDTH, HEADLIGHT_R, HEADLIGHT_INSET,
    AWARENESS_DIVISOR, COLOR_LANE_WHITE,
)
from ui.types import Camera
from ui.helpers import (
    direction_to_heading,
    should_slow_down,
    draw_alpha_circle,
)

_PRETURN_BLINK_DISTANCE_M = 12.0


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC  draw_all_vehicles()
# ══════════════════════════════════════════════════════════════════════════════

def draw_all_vehicles(
    screen: pygame.Surface,
    camera: Camera,
    vehicles: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    frame: int = 0,
) -> None:
    """Render every vehicle: route → awareness zone → car body + headlights."""
    # 1. Route lines (behind everything)
    for v in vehicles:
        _draw_route_line(screen, camera, v)

    # 2. Awareness zones
    for v in vehicles:
        dec = decisions.get(v["id"], {})
        ml = dec.get("decision", "none")
        if should_slow_down(v, ml):
            _draw_awareness_ring(screen, camera, v, frame)

    # 3. Car body + headlights (on top)
    for v in vehicles:
        _draw_vehicle(screen, camera, v, frame)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE
# ══════════════════════════════════════════════════════════════════════════════

# ── Route line ────────────────────────────────────────────────────────────────

def _draw_route_line(
    screen: pygame.Surface, cam: Camera, vehicle: Dict[str, Any]
) -> None:
    """Draw the vehicle's road_line as a faded polyline in its colour."""
    road_line = vehicle.get("road_line")
    if not road_line or len(road_line) < 2:
        return
    color = _get_color(vehicle)
    faded = (*color, 70)  # RGBA

    pts = [cam.world_to_screen(p[0], p[1]) for p in road_line]
    ipts = [(int(p[0]), int(p[1])) for p in pts]

    # Draw on alpha surface
    if len(ipts) < 2:
        return
    # Compute bounding box
    xs = [p[0] for p in ipts]
    ys = [p[1] for p in ipts]
    min_x, max_x = min(xs) - 4, max(xs) + 4
    min_y, max_y = min(ys) - 4, max(ys) + 4
    w = max(1, max_x - min_x)
    h = max(1, max_y - min_y)

    tmp = pygame.Surface((w, h), pygame.SRCALPHA)
    local_pts = [(p[0] - min_x, p[1] - min_y) for p in ipts]
    thickness = max(2, int(cam.zoom * 0.8))
    pygame.draw.lines(tmp, faded, False, local_pts, thickness)
    screen.blit(tmp, (min_x, min_y))


# ── Awareness ring ────────────────────────────────────────────────────────────

def _draw_awareness_ring(
    screen: pygame.Surface, cam: Camera,
    vehicle: Dict[str, Any], frame: int,
) -> None:
    """Pulsing coloured ring around a vehicle that should slow down."""
    sx, sy = cam.world_to_screen(vehicle["x"], vehicle["y"])
    pulse = 0.7 + 0.3 * math.sin(frame * 0.15)
    base_r = int(max(6, CAR_LENGTH * cam.zoom * 0.8))
    r = int(base_r * pulse)
    alpha = int(120 * pulse)
    color = _get_color(vehicle)
    draw_alpha_circle(screen, (*color, alpha), (int(sx), int(sy)), r)


# ── Vehicle body ──────────────────────────────────────────────────────────────

def _draw_vehicle(
    screen: pygame.Surface, cam: Camera, vehicle: Dict[str, Any], frame: int = 0
) -> None:
    """
    Draw a top-down car polygon rotated by heading, with headlights.
    The base shape points EAST (heading 0°).
    """
    # Compute heading from actual velocity vector for smooth rotation.
    vx = vehicle.get("vx", 0.0)
    vy = vehicle.get("vy", 0.0)
    if abs(vx) > 1e-6 or abs(vy) > 1e-6:
        heading = math.degrees(math.atan2(vy, vx))   # 0=East, CCW positive
    else:
        heading = direction_to_heading(vehicle.get("direction", "EAST"))
    color = _get_color(vehicle)
    sx, sy = cam.world_to_screen(vehicle["x"], vehicle["y"])

    # Pixel sizes
    cl = max(6, CAR_LENGTH * cam.zoom)
    cw = max(4, CAR_WIDTH * cam.zoom)
    hl_r = max(1, int(HEADLIGHT_R * cam.zoom))
    hl_off = max(1, HEADLIGHT_INSET * cam.zoom)

    # Build car surface (pointing right = EAST)
    surf_w = int(cl + 6)
    surf_h = int(cw + 6)
    car_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
    cx, cy = surf_w // 2, surf_h // 2

    # --- Drop shadow ---
    shadow_pts = _car_polygon(cx + 2, cy + 2, cl, cw)
    pygame.draw.polygon(car_surf, (0, 0, 0, 40), shadow_pts)

    # --- Body ---
    body_pts = _car_polygon(cx, cy, cl, cw)
    pygame.draw.polygon(car_surf, color, body_pts)
    # Outline
    pygame.draw.polygon(car_surf, _darken(color, 40), body_pts, max(1, int(cam.zoom * 0.3)))

    # --- Windshield (slightly darker strip near front) ---
    ws_x = cx + int(cl * 0.2)
    ws_w = max(2, int(cl * 0.12))
    ws_h = max(2, int(cw * 0.6))
    pygame.draw.rect(car_surf, _darken(color, 60), (ws_x, cy - ws_h // 2, ws_w, ws_h))

    # --- Headlights ---
    hl_y_top = cy - int(cw * 0.35)
    hl_y_bot = cy + int(cw * 0.35)
    hl_x = cx + int(cl * 0.42)
    pygame.draw.circle(car_surf, (255, 255, 200), (hl_x, hl_y_top), hl_r)
    pygame.draw.circle(car_surf, (255, 255, 200), (hl_x, hl_y_bot), hl_r)

    # --- Tail-lights ---
    tl_x = cx - int(cl * 0.42)
    pygame.draw.circle(car_surf, (200, 30, 30), (tl_x, hl_y_top), max(1, hl_r - 1))
    pygame.draw.circle(car_surf, (200, 30, 30), (tl_x, hl_y_bot), max(1, hl_r - 1))
    _draw_turn_indicators(car_surf, vehicle, cx, cy, cl, cw, frame, cam.zoom)

    # Rotate & blit
    rotated = pygame.transform.rotate(car_surf, heading)
    rect = rotated.get_rect(center=(int(sx), int(sy)))
    screen.blit(rotated, rect)

    # --- Blue gyrophars for priority vehicles ---
    if vehicle.get("priority", False):
        _draw_gyrophars(screen, int(sx), int(sy), cl, cw, heading, frame)

    # --- ID label above car ---
    if cam.zoom >= 2.0:
        font = pygame.font.SysFont("arial,helvetica", max(9, int(cam.zoom * 3.5)))
        lbl = font.render(vehicle["id"], True, (255, 255, 255))
        lbl_bg = pygame.Surface((lbl.get_width() + 4, lbl.get_height() + 2), pygame.SRCALPHA)
        lbl_bg.fill((0, 0, 0, 100))
        screen.blit(lbl_bg, (int(sx) - lbl_bg.get_width() // 2, int(sy) - int(cw) - 14))
        screen.blit(lbl, (int(sx) - lbl.get_width() // 2, int(sy) - int(cw) - 13))


def _car_polygon(cx: int, cy: int, length: float, width: float) -> List[Tuple[int, int]]:
    """
    Six-point polygon for a simplified top-down car shape pointing right.
    Slightly tapered at the front, squared at the back.
    """
    hl = length / 2
    hw = width / 2
    taper = width * 0.15
    return [
        (int(cx - hl), int(cy - hw)),          # rear-left
        (int(cx + hl * 0.6), int(cy - hw)),    # front-left shoulder
        (int(cx + hl), int(cy - hw + taper)),   # front-left taper
        (int(cx + hl), int(cy + hw - taper)),   # front-right taper
        (int(cx + hl * 0.6), int(cy + hw)),    # front-right shoulder
        (int(cx - hl), int(cy + hw)),          # rear-right
    ]


# ── Colour helpers ────────────────────────────────────────────────────────────

def _get_color(vehicle: Dict[str, Any]) -> Tuple[int, int, int]:
    # Priority vehicles are white
    if vehicle.get("priority", False):
        return (255, 255, 255)
    c = vehicle.get("color", (180, 180, 180))
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return (c[0], c[1], c[2])
    return (180, 180, 180)


def _draw_gyrophars(
    screen: pygame.Surface,
    sx: int, sy: int,
    car_length: float, car_width: float,
    heading: float, frame: int
) -> None:
    """Draw flashing blue gyrophars on roof of priority vehicle."""
    # Two lights on the roof, offset left/right from center
    offset = car_width * 0.25
    rad = math.radians(heading)
    # Perpendicular direction for left/right offset
    perp_x = -math.sin(rad) * offset
    perp_y = math.cos(rad) * offset
    
    # Alternating flash pattern
    flash = (frame // 4) % 2  # switches every 4 frames
    light_r = max(2, int(car_width * 0.2))
    
    # Left light
    lx, ly = int(sx + perp_x), int(sy + perp_y)
    # Right light  
    rx, ry = int(sx - perp_x), int(sy - perp_y)
    
    # Flash blue alternately
    blue_bright = (0, 100, 255)
    blue_dim = (0, 50, 150)
    
    if flash == 0:
        pygame.draw.circle(screen, blue_bright, (lx, ly), light_r)
        pygame.draw.circle(screen, blue_dim, (rx, ry), light_r)
    else:
        pygame.draw.circle(screen, blue_dim, (lx, ly), light_r)
        pygame.draw.circle(screen, blue_bright, (rx, ry), light_r)


def _draw_turn_indicators(
    surface: pygame.Surface,
    vehicle: Dict[str, Any],
    cx: int,
    cy: int,
    car_length: float,
    car_width: float,
    frame: int,
    zoom: float,
) -> None:
    """Blink indicator during the turn and shortly before the stop line."""
    intent = str(vehicle.get("turn_intent", "FORWARD")).upper()
    if intent not in ("LEFT", "RIGHT"):
        return

    dist_to_line_raw = vehicle.get("dist_to_stop_line")
    try:
        dist_to_line = float(dist_to_line_raw)
    except (TypeError, ValueError):
        dist_to_line = float("inf")

    active_turn = bool(vehicle.get("is_turning", False))
    preturn_window = 0.0 < dist_to_line <= _PRETURN_BLINK_DISTANCE_M
    if not (active_turn or preturn_window):
        return

    # Toggle roughly 5 times per second at 60 FPS.
    if (frame // 6) % 2 == 1:
        return

    front_x = cx + int(car_length * 0.42)
    rear_x = cx - int(car_length * 0.42)
    side_y = cy - int(car_width * 0.45) if intent == "LEFT" else cy + int(car_width * 0.45)
    rad = max(1, int(zoom * 0.9))
    amber = (255, 170, 20)
    pygame.draw.circle(surface, amber, (front_x, side_y), rad)
    pygame.draw.circle(surface, amber, (rear_x, side_y), rad)


def _darken(color: Tuple[int, int, int], amount: int = 40) -> Tuple[int, int, int]:
    return tuple(max(0, c - amount) for c in color)  # type: ignore[return-value]
