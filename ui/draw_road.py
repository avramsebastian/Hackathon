"""
ui/draw_road.py
===============
Renders the Google-Maps-style 2D map:
  grass background, road surfaces, sidewalks, lane markings,
  intersection box, traffic signs, and decorative neighbourhood elements.

All functions are *pure renderers* — they read data and draw to a surface.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pygame

from ui.constants import (
    COLOR_GRASS, COLOR_GRASS_DARK, COLOR_ROAD, COLOR_ROAD_EDGE,
    COLOR_SIDEWALK, COLOR_LANE_WHITE, COLOR_INTERSECTION,
    COLOR_STOP_RED, COLOR_STOP_WHITE,
    COLOR_YIELD_RED, COLOR_YIELD_WHITE,
    COLOR_PRIORITY_YELLOW, COLOR_PRIORITY_WHITE,
    COLOR_SIGN_POLE,
    COLOR_TREE_TRUNK, COLOR_TREE_CANOPY_A, COLOR_TREE_CANOPY_B,
    COLOR_HOUSE_WALL, COLOR_HOUSE_ROOF_A, COLOR_HOUSE_ROOF_B,
    COLOR_HOUSE_DOOR, COLOR_HOUSE_WINDOW,
    COLOR_LIGHT_RED, COLOR_LIGHT_YELLOW, COLOR_LIGHT_GREEN,
    COLOR_LIGHT_OFF, COLOR_LIGHT_HOUSING,
    ROAD_HALF_W, SIDEWALK_W, ROAD_LENGTH, LANE_WIDTH_M,
    DASH_LEN, DASH_GAP,
)
from ui.types import Camera
from ui.helpers import draw_alpha_rect, draw_alpha_circle

# ── Pre-computed decoration positions ─────────────────────────────────────────
# (kind, world_x, world_y, extra…)
_DECORATIONS: List[Tuple[Any, ...]] = [
    # NE quadrant
    ("tree",  20,  22,  2.8),
    ("tree",  38,  18,  3.2),
    ("tree",  55,  32,  2.4),
    ("tree",  72,  24,  3.0),
    ("tree",  48,  58,  2.6),
    ("house", 30,  42,  12, 9, COLOR_HOUSE_ROOF_A),
    ("house", 58,  50,  10, 8, COLOR_HOUSE_ROOF_B),
    # NW quadrant
    ("tree", -24,  20,  2.6),
    ("tree", -42,  35,  3.0),
    ("tree", -58,  22,  2.2),
    ("tree", -70,  42,  3.4),
    ("house", -32, 44, 11, 9, COLOR_HOUSE_ROOF_B),
    ("house", -55, 56, 10, 8, COLOR_HOUSE_ROOF_A),
    # SE quadrant
    ("tree",  24, -20,  2.8),
    ("tree",  40, -36,  3.0),
    ("tree",  62, -48,  2.4),
    ("tree",  32, -58,  2.6),
    ("house", 48, -30, 12, 9, COLOR_HOUSE_ROOF_A),
    # SW quadrant
    ("tree", -22, -24,  3.0),
    ("tree", -50, -32,  2.6),
    ("tree", -68, -52,  3.2),
    ("tree", -30, -58,  2.4),
    ("house", -38, -42, 11, 10, COLOR_HOUSE_ROOF_B),
    ("house", -60, -62, 10, 8, COLOR_HOUSE_ROOF_A),
]


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC  draw_map()  — single entry point
# ══════════════════════════════════════════════════════════════════════════════

def draw_map(
    screen: pygame.Surface,
    camera: Camera,
    intersection: Dict[str, Any],
) -> None:
    """Draw the complete map background in correct z-order."""
    _draw_grass(screen, camera)
    _draw_road_shadows(screen, camera)
    _draw_road_surfaces(screen, camera)
    _draw_sidewalks(screen, camera)
    _draw_intersection_box(screen, camera)
    _draw_lane_markings(screen, camera)
    _draw_edge_lines(screen, camera)
    _draw_decorations(screen, camera)
    sem = intersection.get("semaphore", {})
    if sem.get("enabled", False):
        _draw_semaphores(screen, camera, sem)
    else:
        _draw_signs(screen, camera, intersection)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE helpers
# ══════════════════════════════════════════════════════════════════════════════

def _world_rect(camera: Camera, x1: float, y1: float, x2: float, y2: float) -> pygame.Rect:
    """Convert two world-space corners to a screen-space Rect (y-flipped)."""
    sx1, sy1 = camera.world_to_screen(min(x1, x2), max(y1, y2))
    sx2, sy2 = camera.world_to_screen(max(x1, x2), min(y1, y2))
    return pygame.Rect(int(sx1), int(sy1), max(1, int(sx2 - sx1)), max(1, int(sy2 - sy1)))


# ── Grass ─────────────────────────────────────────────────────────────────────

def _draw_grass(screen: pygame.Surface, cam: Camera) -> None:
    screen.fill(COLOR_GRASS)


# ── Road shadows (subtle 3-D depth) ──────────────────────────────────────────

def _draw_road_shadows(screen: pygame.Surface, cam: Camera) -> None:
    off = max(2, int(1.5 * cam.zoom))  # shadow offset in pixels
    hw = ROAD_HALF_W
    rl = ROAD_LENGTH
    dark = COLOR_GRASS_DARK
    # Horizontal shadow
    r = _world_rect(cam, -rl, -hw, rl, hw)
    r.move_ip(off, off)
    pygame.draw.rect(screen, dark, r)
    # Vertical shadow
    r = _world_rect(cam, -hw, -rl, hw, rl)
    r.move_ip(off, off)
    pygame.draw.rect(screen, dark, r)


# ── Road surfaces ────────────────────────────────────────────────────────────

def _draw_road_surfaces(screen: pygame.Surface, cam: Camera) -> None:
    hw = ROAD_HALF_W
    rl = ROAD_LENGTH
    # Horizontal
    pygame.draw.rect(screen, COLOR_ROAD, _world_rect(cam, -rl, -hw, rl, hw))
    # Vertical
    pygame.draw.rect(screen, COLOR_ROAD, _world_rect(cam, -hw, -rl, hw, rl))


# ── Sidewalks ────────────────────────────────────────────────────────────────

def _draw_sidewalks(screen: pygame.Surface, cam: Camera) -> None:
    hw = ROAD_HALF_W
    sw = SIDEWALK_W
    rl = ROAD_LENGTH
    # Horizontal road sidewalks (top & bottom)
    pygame.draw.rect(screen, COLOR_SIDEWALK, _world_rect(cam, -rl, hw, rl, hw + sw))
    pygame.draw.rect(screen, COLOR_SIDEWALK, _world_rect(cam, -rl, -hw - sw, rl, -hw))
    # Vertical road sidewalks (left & right)
    pygame.draw.rect(screen, COLOR_SIDEWALK, _world_rect(cam, -hw - sw, -rl, -hw, rl))
    pygame.draw.rect(screen, COLOR_SIDEWALK, _world_rect(cam, hw, -rl, hw + sw, rl))


# ── Intersection box ─────────────────────────────────────────────────────────

def _draw_intersection_box(screen: pygame.Surface, cam: Camera) -> None:
    hw = ROAD_HALF_W
    rect = _world_rect(cam, -hw, -hw, hw, hw)
    # Drop shadow
    shadow = rect.copy()
    shadow.move_ip(3, 3)
    draw_alpha_rect(screen, (0, 0, 0, 35), shadow)
    pygame.draw.rect(screen, COLOR_INTERSECTION, rect)


# ── Lane markings (dashed white centre lines) ────────────────────────────────

def _draw_lane_markings(screen: pygame.Surface, cam: Camera) -> None:
    hw = ROAD_HALF_W
    rl = ROAD_LENGTH
    dl = DASH_LEN
    dg = DASH_GAP
    period = dl + dg

    # Horizontal centre line — west arm
    x = -rl
    while x < -hw:
        x1s, y1s = cam.world_to_screen(x, 0)
        x2s, _   = cam.world_to_screen(min(x + dl, -hw), 0)
        w = max(1, int(x2s - x1s))
        h = max(1, int(2 * cam.zoom / 3))
        pygame.draw.rect(screen, COLOR_LANE_WHITE, (int(x1s), int(y1s) - h // 2, w, h))
        x += period

    # Horizontal centre line — east arm
    x = hw
    while x < rl:
        x1s, y1s = cam.world_to_screen(x, 0)
        x2s, _   = cam.world_to_screen(min(x + dl, rl), 0)
        w = max(1, int(x2s - x1s))
        h = max(1, int(2 * cam.zoom / 3))
        pygame.draw.rect(screen, COLOR_LANE_WHITE, (int(x1s), int(y1s) - h // 2, w, h))
        x += period

    # Vertical centre line — south arm
    y = -rl
    while y < -hw:
        x1s, y1s = cam.world_to_screen(0, y + dl)
        _,   y2s = cam.world_to_screen(0, y)
        w = max(1, int(2 * cam.zoom / 3))
        hh = max(1, int(y2s - y1s))
        pygame.draw.rect(screen, COLOR_LANE_WHITE, (int(x1s) - w // 2, int(y1s), w, hh))
        y += period

    # Vertical centre line — north arm
    y = hw
    while y < rl:
        x1s, y1s = cam.world_to_screen(0, y + dl)
        _,   y2s = cam.world_to_screen(0, y)
        w = max(1, int(2 * cam.zoom / 3))
        hh = max(1, int(y2s - y1s))
        pygame.draw.rect(screen, COLOR_LANE_WHITE, (int(x1s) - w // 2, int(y1s), w, hh))
        y += period


# ── Road edge lines (solid white) ────────────────────────────────────────────

def _draw_edge_lines(screen: pygame.Surface, cam: Camera) -> None:
    hw = ROAD_HALF_W
    rl = ROAD_LENGTH
    thickness = max(1, int(cam.zoom * 0.4))

    edges = [
        # Horizontal road
        (-rl, hw, rl, hw),
        (-rl, -hw, rl, -hw),
        # Vertical road
        (hw, -rl, hw, rl),
        (-hw, -rl, -hw, rl),
    ]
    for wx1, wy1, wx2, wy2 in edges:
        p1 = cam.world_to_screen(wx1, wy1)
        p2 = cam.world_to_screen(wx2, wy2)
        pygame.draw.line(screen, COLOR_ROAD_EDGE, _i2(p1), _i2(p2), thickness)


# ── Traffic signs ─────────────────────────────────────────────────────────────

# Where to draw the sign for each approach (right side of road before intersection).
# Keys = cardinal direction the sign faces; values = (world_x_offset, world_y_offset).
_SIGN_OFFSETS: Dict[str, Tuple[float, float]] = {
    "W": (-ROAD_HALF_W - 0.5, -ROAD_HALF_W - 0.5),   # SW corner (right of east-bound)
    "E": ( ROAD_HALF_W + 0.5,  ROAD_HALF_W + 0.5),   # NE corner (right of west-bound)
    "N": (-ROAD_HALF_W - 0.5,  ROAD_HALF_W + 0.5),   # NW corner (right of south-bound)
    "S": ( ROAD_HALF_W + 0.5, -ROAD_HALF_W - 0.5),   # SE corner (right of north-bound)
}


def _draw_signs(screen: pygame.Surface, cam: Camera, intersection: Dict[str, Any]) -> None:
    signs = intersection.get("signs", {})
    for direction, sign_name in signs.items():
        offset = _SIGN_OFFSETS.get(direction.upper())
        if not offset:
            continue
        sx, sy = cam.world_to_screen(offset[0], offset[1])
        _draw_single_sign(screen, cam, int(sx), int(sy), sign_name.upper())


def _draw_single_sign(
    screen: pygame.Surface, cam: Camera, sx: int, sy: int, name: str
) -> None:
    """Draw one traffic sign at screen position (sx, sy)."""
    s = max(5, int(3.0 * cam.zoom))  # sign half-size

    # Pole
    pole_h = int(s * 1.6)
    pw = max(1, int(cam.zoom * 0.4))
    pygame.draw.line(screen, COLOR_SIGN_POLE, (sx, sy), (sx, sy + pole_h), pw)

    if name == "STOP":
        _draw_hexagon(screen, sx, sy, s, COLOR_STOP_RED)
        # tiny white "STOP" text
        if cam.zoom >= 2.5:
            font = pygame.font.SysFont("arial,helvetica", max(8, int(s * 0.6)))
            txt = font.render("STOP", True, COLOR_STOP_WHITE)
            screen.blit(txt, (sx - txt.get_width() // 2, sy - txt.get_height() // 2))

    elif name == "YIELD":
        _draw_yield_triangle(screen, sx, sy, s)

    elif name == "PRIORITY":
        _draw_diamond(screen, sx, sy, s, COLOR_PRIORITY_YELLOW, COLOR_PRIORITY_WHITE)

    else:  # uncontrolled
        pass  # no sign rendered


def _draw_hexagon(
    surface: pygame.Surface, cx: int, cy: int, r: int, color: Tuple[int, ...]
) -> None:
    pts = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        pts.append((cx + int(r * math.cos(angle)), cy - int(r * math.sin(angle))))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (255, 255, 255), pts, max(1, r // 5))


def _draw_yield_triangle(
    surface: pygame.Surface, cx: int, cy: int, r: int
) -> None:
    # Inverted triangle (point down)
    pts = [
        (cx - r, cy - int(r * 0.7)),
        (cx + r, cy - int(r * 0.7)),
        (cx, cy + int(r * 0.9)),
    ]
    pygame.draw.polygon(surface, COLOR_YIELD_WHITE, pts)
    pygame.draw.polygon(surface, COLOR_YIELD_RED, pts, max(1, r // 4))


def _draw_diamond(
    surface: pygame.Surface, cx: int, cy: int, r: int,
    fill: Tuple[int, ...], border: Tuple[int, ...]
) -> None:
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surface, fill, pts)
    pygame.draw.polygon(surface, border, pts, max(1, r // 5))


# ── Decorative neighbourhood elements ────────────────────────────────────────

# ── Traffic-light semaphores ──────────────────────────────────────────────────

# Position each traffic light on the right side of the approaching lane,
# just outside the intersection box at the stop line (~10.5 m from centre).
_LIGHT_OFFSETS: Dict[str, Tuple[float, float]] = {
    "W": (-ROAD_HALF_W - 0.5, -ROAD_HALF_W - 0.5),   # SW corner
    "E": ( ROAD_HALF_W + 0.5,  ROAD_HALF_W + 0.5),   # NE corner
    "N": (-ROAD_HALF_W - 0.5,  ROAD_HALF_W + 0.5),   # NW corner
    "S": ( ROAD_HALF_W + 0.5, -ROAD_HALF_W - 0.5),   # SE corner
}

_COLOR_FOR_PHASE = {
    "GREEN":  COLOR_LIGHT_GREEN,
    "YELLOW": COLOR_LIGHT_YELLOW,
    "RED":    COLOR_LIGHT_RED,
}

def _draw_semaphores(
    screen: pygame.Surface, cam: Camera, sem: Dict[str, Any]
) -> None:
    """Render a 3-bulb traffic light for every approach."""
    colors = sem.get("colors", {})
    for approach, (wx, wy) in _LIGHT_OFFSETS.items():
        phase = colors.get(approach, "RED")
        sx, sy = cam.world_to_screen(wx, wy)
        sx, sy = int(sx), int(sy)
        _draw_single_light(screen, cam, sx, sy, phase)


def _draw_single_light(
    screen: pygame.Surface, cam: Camera, sx: int, sy: int, phase: str
) -> None:
    """Draw one 3-bulb vertical traffic light at screen pos (sx, sy)."""
    bulb_r = max(3, int(1.8 * cam.zoom))
    spacing = int(bulb_r * 2.4)
    housing_w = bulb_r * 2 + max(2, int(cam.zoom * 0.6))
    housing_h = spacing * 2 + bulb_r * 2 + max(2, int(cam.zoom * 0.6))

    # Pole
    pole_h = int(bulb_r * 3)
    pw = max(1, int(cam.zoom * 0.45))
    pygame.draw.line(screen, COLOR_SIGN_POLE, (sx, sy + housing_h // 2),
                     (sx, sy + housing_h // 2 + pole_h), pw)

    # Housing rectangle
    hr = pygame.Rect(sx - housing_w // 2, sy - housing_h // 2, housing_w, housing_h)
    pygame.draw.rect(screen, COLOR_LIGHT_HOUSING, hr, border_radius=max(1, bulb_r // 2))

    # Bulbs: top=RED, mid=YELLOW, bot=GREEN
    bulb_defs = [
        ("RED",    sy - spacing),
        ("YELLOW", sy),
        ("GREEN",  sy + spacing),
    ]
    for bulb_phase, by in bulb_defs:
        if bulb_phase == phase:
            c = _COLOR_FOR_PHASE[bulb_phase]
        else:
            c = COLOR_LIGHT_OFF
        pygame.draw.circle(screen, c, (sx, by), bulb_r)


# ── Decorative neighbourhood elements (cont.) ────────────────────────────────

def _draw_decorations(screen: pygame.Surface, cam: Camera) -> None:
    for dec in _DECORATIONS:
        kind = dec[0]
        if kind == "tree":
            _draw_tree(screen, cam, dec[1], dec[2], dec[3])
        elif kind == "house":
            _draw_house(screen, cam, dec[1], dec[2], dec[3], dec[4], dec[5])



def _draw_tree(
    screen: pygame.Surface, cam: Camera, wx: float, wy: float, size: float
) -> None:
    sx, sy = cam.world_to_screen(wx, wy)
    sx, sy = int(sx), int(sy)
    cr = max(3, int(size * cam.zoom))
    tr_w = max(1, int(size * 0.25 * cam.zoom))
    tr_h = max(2, int(size * 0.5 * cam.zoom))
    # Shadow
    draw_alpha_circle(screen, (0, 0, 0, 25), (sx + 2, sy + 2), cr)
    # Trunk
    pygame.draw.rect(screen, COLOR_TREE_TRUNK, (sx - tr_w // 2, sy, tr_w, tr_h))
    # Canopy
    pygame.draw.circle(screen, COLOR_TREE_CANOPY_A, (sx, sy - int(size * 0.2 * cam.zoom)), cr)
    # Highlight
    pygame.draw.circle(screen, COLOR_TREE_CANOPY_B, (sx - cr // 4, sy - int(size * 0.35 * cam.zoom)), cr // 2)


def _draw_house(
    screen: pygame.Surface, cam: Camera,
    wx: float, wy: float, w: float, h: float,
    roof_color: Tuple[int, ...],
) -> None:
    rect = _world_rect(cam, wx - w / 2, wy - h / 2, wx + w / 2, wy + h / 2)
    # Shadow
    shadow = rect.copy()
    shadow.move_ip(3, 3)
    draw_alpha_rect(screen, (0, 0, 0, 30), shadow)
    # Wall
    pygame.draw.rect(screen, COLOR_HOUSE_WALL, rect)
    # Roof (top third)
    roof_h = max(2, rect.h // 3)
    roof_rect = pygame.Rect(rect.x, rect.y, rect.w, roof_h)
    pygame.draw.rect(screen, roof_color, roof_rect)
    # Door
    dw = max(2, rect.w // 5)
    dh = max(3, rect.h // 3)
    pygame.draw.rect(screen, COLOR_HOUSE_DOOR, (rect.centerx - dw // 2, rect.bottom - dh, dw, dh))
    # Window(s)
    win_sz = max(2, rect.w // 6)
    if rect.w > 20:
        pygame.draw.rect(screen, COLOR_HOUSE_WINDOW, (rect.x + rect.w // 5, rect.y + roof_h + 2, win_sz, win_sz))
        pygame.draw.rect(screen, COLOR_HOUSE_WINDOW, (rect.right - rect.w // 5 - win_sz, rect.y + roof_h + 2, win_sz, win_sz))




# ── tiny helpers ──────────────────────────────────────────────────────────────

def _i2(pair: Tuple[float, float]) -> Tuple[int, int]:
    return int(pair[0]), int(pair[1])
