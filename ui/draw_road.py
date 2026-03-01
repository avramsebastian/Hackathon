"""
ui/draw_road.py
===============
Renders the Google-Maps-style 2D map for a multi-intersection road network:
  grass background, road surfaces, sidewalks, lane markings,
  intersection boxes, traffic signs/semaphores, and decorative elements.

All functions are *pure renderers* — they read data and draw to a surface.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Set, Tuple

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
    ROAD_HALF_W, SIDEWALK_W, LANE_WIDTH_M,
    DASH_LEN, DASH_GAP,
)
from ui.types import Camera
from ui.helpers import draw_alpha_rect, draw_alpha_circle

# ── Constants ─────────────────────────────────────────────────────────────────
_DEFAULT_TERMINAL_ARM_LEN = 80.0  # default for terminal road arms (m)
_APPROACHES = ("N", "S", "E", "W")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL: compute arm lengths from network data
# ══════════════════════════════════════════════════════════════════════════════

def _compute_terminal_lengths(
    intersections: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute terminal arm lengths per direction based on network extent.
    
    For narrow networks, extend arms in that direction to fill space.
    """
    if not intersections:
        return {a: _DEFAULT_TERMINAL_ARM_LEN for a in _APPROACHES}
    
    xs = [i["center"][0] for i in intersections]
    ys = [i["center"][1] for i in intersections]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    net_width = max_x - min_x if max_x > min_x else 0.0
    net_height = max_y - min_y if max_y > min_y else 0.0
    
    # Target aspect ratio ~16:9
    target_aspect = 16.0 / 9.0
    
    # Compute how much to extend in each direction
    if net_height > 0:
        target_width = net_height * target_aspect
        if net_width < target_width:
            extra_x = (target_width - net_width) / 2.0
        else:
            extra_x = 0.0
    else:
        extra_x = 150.0  # single row - extend horizontally
    
    if net_width > 0:
        target_height = net_width / target_aspect
        if net_height < target_height:
            extra_y = (target_height - net_height) / 2.0
        else:
            extra_y = 0.0
    else:
        extra_y = 150.0  # single column - extend vertically
    
    return {
        "N": _DEFAULT_TERMINAL_ARM_LEN + extra_y,
        "S": _DEFAULT_TERMINAL_ARM_LEN + extra_y,
        "E": _DEFAULT_TERMINAL_ARM_LEN + extra_x,
        "W": _DEFAULT_TERMINAL_ARM_LEN + extra_x,
    }


def _arm_lengths(
    intersections: List[Dict[str, Any]],
    roads_data: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Per-intersection arm lengths.  Connected arms extend half the
    distance to the neighbour; terminal arms extend dynamically."""
    centers = {i["id"]: i["center"] for i in intersections}
    terminal_lens = _compute_terminal_lengths(intersections)
    result: Dict[str, Dict[str, float]] = {}
    for i in intersections:
        result[i["id"]] = {a: terminal_lens[a] for a in _APPROACHES}
    for road in roads_data:
        c1 = centers[road["from_id"]]
        c2 = centers[road["to_id"]]
        dist = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
        half = dist / 2.0
        result[road["from_id"]][road["from_arm"]] = half
        result[road["to_id"]][road["to_arm"]] = half
    return result


def _connected_set(
    roads_data: List[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """Map *int_id* -> set of connected arm directions."""
    out: Dict[str, Set[str]] = {}
    for road in roads_data:
        out.setdefault(road["from_id"], set()).add(road["from_arm"])
        out.setdefault(road["to_id"], set()).add(road["to_arm"])
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC  draw_map()  — single entry point
# ══════════════════════════════════════════════════════════════════════════════

def draw_map(
    screen: pygame.Surface,
    camera: Camera,
    intersection: Dict[str, Any],
) -> None:
    """Draw the complete multi-intersection map background in z-order."""
    intersections = intersection.get("intersections", [])
    roads_data = intersection.get("roads", [])
    arms = _arm_lengths(intersections, roads_data)

    _draw_grass(screen, camera)

    # Road shadows
    for info in intersections:
        cx, cy = info["center"]
        _draw_road_shadows_at(screen, camera, cx, cy, arms[info["id"]])

    # Road surfaces
    for info in intersections:
        cx, cy = info["center"]
        _draw_road_surfaces_at(screen, camera, cx, cy, arms[info["id"]])

    # Sidewalks
    for info in intersections:
        cx, cy = info["center"]
        _draw_sidewalks_at(screen, camera, cx, cy, arms[info["id"]])

    # Intersection boxes
    for info in intersections:
        cx, cy = info["center"]
        _draw_intersection_box_at(screen, camera, cx, cy)

    # Lane markings
    for info in intersections:
        cx, cy = info["center"]
        _draw_lane_markings_at(screen, camera, cx, cy, arms[info["id"]])

    # Edge lines
    for info in intersections:
        cx, cy = info["center"]
        _draw_edge_lines_at(screen, camera, cx, cy, arms[info["id"]])

    # Decorations (generated per-intersection)
    _draw_decorations(screen, camera, intersections, roads_data)

    # Signs / semaphores per intersection
    for info in intersections:
        cx, cy = info["center"]
        sem = info.get("semaphore", {})
        if sem.get("enabled", False):
            _draw_semaphores(screen, camera, sem, cx, cy)
        else:
            _draw_signs(screen, camera, info, cx, cy)


# ══════════════════════════════════════════════════════════════════════════════
#  PRIVATE helpers
# ══════════════════════════════════════════════════════════════════════════════

def _world_rect(cam: Camera, x1: float, y1: float, x2: float, y2: float) -> pygame.Rect:
    """Convert two world-space corners to a screen-space Rect (y-flipped)."""
    sx1, sy1 = cam.world_to_screen(min(x1, x2), max(y1, y2))
    sx2, sy2 = cam.world_to_screen(max(x1, x2), min(y1, y2))
    return pygame.Rect(int(sx1), int(sy1), max(1, int(sx2 - sx1)), max(1, int(sy2 - sy1)))


# ── Grass ─────────────────────────────────────────────────────────────────────

def _draw_grass(screen: pygame.Surface, cam: Camera) -> None:
    screen.fill(COLOR_GRASS)


# ── Road shadows ──────────────────────────────────────────────────────────────

def _draw_road_shadows_at(
    screen: pygame.Surface, cam: Camera,
    cx: float, cy: float, arm_len: Dict[str, float],
) -> None:
    off = max(2, int(1.5 * cam.zoom))
    hw = ROAD_HALF_W
    dark = COLOR_GRASS_DARK
    # Horizontal shadow
    r = _world_rect(cam, cx - arm_len["W"], cy - hw, cx + arm_len["E"], cy + hw)
    r.move_ip(off, off)
    pygame.draw.rect(screen, dark, r)
    # Vertical shadow
    r = _world_rect(cam, cx - hw, cy - arm_len["S"], cx + hw, cy + arm_len["N"])
    r.move_ip(off, off)
    pygame.draw.rect(screen, dark, r)


# ── Road surfaces ────────────────────────────────────────────────────────────

def _draw_road_surfaces_at(
    screen: pygame.Surface, cam: Camera,
    cx: float, cy: float, arm_len: Dict[str, float],
) -> None:
    hw = ROAD_HALF_W
    # Horizontal
    pygame.draw.rect(screen, COLOR_ROAD,
                     _world_rect(cam, cx - arm_len["W"], cy - hw,
                                 cx + arm_len["E"], cy + hw))
    # Vertical
    pygame.draw.rect(screen, COLOR_ROAD,
                     _world_rect(cam, cx - hw, cy - arm_len["S"],
                                 cx + hw, cy + arm_len["N"]))


# ── Sidewalks ────────────────────────────────────────────────────────────────

def _draw_sidewalks_at(
    screen: pygame.Surface, cam: Camera,
    cx: float, cy: float, arm_len: Dict[str, float],
) -> None:
    hw = ROAD_HALF_W
    sw = SIDEWALK_W
    # Horizontal top & bottom
    ew = arm_len["E"]
    ww = arm_len["W"]
    pygame.draw.rect(screen, COLOR_SIDEWALK,
                     _world_rect(cam, cx - ww, cy + hw, cx + ew, cy + hw + sw))
    pygame.draw.rect(screen, COLOR_SIDEWALK,
                     _world_rect(cam, cx - ww, cy - hw - sw, cx + ew, cy - hw))
    # Vertical left & right
    nn = arm_len["N"]
    ss = arm_len["S"]
    pygame.draw.rect(screen, COLOR_SIDEWALK,
                     _world_rect(cam, cx - hw - sw, cy - ss, cx - hw, cy + nn))
    pygame.draw.rect(screen, COLOR_SIDEWALK,
                     _world_rect(cam, cx + hw, cy - ss, cx + hw + sw, cy + nn))


# ── Intersection box ─────────────────────────────────────────────────────────

def _draw_intersection_box_at(
    screen: pygame.Surface, cam: Camera, cx: float, cy: float,
) -> None:
    hw = ROAD_HALF_W
    rect = _world_rect(cam, cx - hw, cy - hw, cx + hw, cy + hw)
    shadow = rect.copy()
    shadow.move_ip(3, 3)
    draw_alpha_rect(screen, (0, 0, 0, 35), shadow)
    pygame.draw.rect(screen, COLOR_INTERSECTION, rect)


# ── Lane markings ────────────────────────────────────────────────────────────

def _draw_lane_markings_at(
    screen: pygame.Surface, cam: Camera,
    cx: float, cy: float, arm_len: Dict[str, float],
) -> None:
    hw = ROAD_HALF_W
    dl = DASH_LEN
    dg = DASH_GAP
    period = dl + dg

    # West arm (horizontal dashes)
    x = cx - arm_len["W"]
    while x < cx - hw:
        x1s, y1s = cam.world_to_screen(x, cy)
        x2s, _ = cam.world_to_screen(min(x + dl, cx - hw), cy)
        w = max(1, int(x2s - x1s))
        h = max(1, int(2 * cam.zoom / 3))
        pygame.draw.rect(screen, COLOR_LANE_WHITE,
                         (int(x1s), int(y1s) - h // 2, w, h))
        x += period

    # East arm
    x = cx + hw
    while x < cx + arm_len["E"]:
        x1s, y1s = cam.world_to_screen(x, cy)
        x2s, _ = cam.world_to_screen(min(x + dl, cx + arm_len["E"]), cy)
        w = max(1, int(x2s - x1s))
        h = max(1, int(2 * cam.zoom / 3))
        pygame.draw.rect(screen, COLOR_LANE_WHITE,
                         (int(x1s), int(y1s) - h // 2, w, h))
        x += period

    # South arm (vertical dashes)
    y = cy - arm_len["S"]
    while y < cy - hw:
        x1s, y1s = cam.world_to_screen(cx, y + dl)
        _, y2s = cam.world_to_screen(cx, y)
        w = max(1, int(2 * cam.zoom / 3))
        hh = max(1, int(y2s - y1s))
        pygame.draw.rect(screen, COLOR_LANE_WHITE,
                         (int(x1s) - w // 2, int(y1s), w, hh))
        y += period

    # North arm
    y = cy + hw
    while y < cy + arm_len["N"]:
        x1s, y1s = cam.world_to_screen(cx, y + dl)
        _, y2s = cam.world_to_screen(cx, y)
        w = max(1, int(2 * cam.zoom / 3))
        hh = max(1, int(y2s - y1s))
        pygame.draw.rect(screen, COLOR_LANE_WHITE,
                         (int(x1s) - w // 2, int(y1s), w, hh))
        y += period


# ── Road edge lines ──────────────────────────────────────────────────────────

def _draw_edge_lines_at(
    screen: pygame.Surface, cam: Camera,
    cx: float, cy: float, arm_len: Dict[str, float],
) -> None:
    hw = ROAD_HALF_W
    thickness = max(1, int(cam.zoom * 0.4))

    edges = [
        # Horizontal road top & bottom edges
        (cx - arm_len["W"], cy + hw, cx + arm_len["E"], cy + hw),
        (cx - arm_len["W"], cy - hw, cx + arm_len["E"], cy - hw),
        # Vertical road left & right edges
        (cx + hw, cy - arm_len["S"], cx + hw, cy + arm_len["N"]),
        (cx - hw, cy - arm_len["S"], cx - hw, cy + arm_len["N"]),
    ]
    for wx1, wy1, wx2, wy2 in edges:
        p1 = cam.world_to_screen(wx1, wy1)
        p2 = cam.world_to_screen(wx2, wy2)
        pygame.draw.line(screen, COLOR_ROAD_EDGE, _i2(p1), _i2(p2), thickness)


# ── Traffic signs ─────────────────────────────────────────────────────────────

_SIGN_OFFSETS: Dict[str, Tuple[float, float]] = {
    "W": (-ROAD_HALF_W - 0.5, -ROAD_HALF_W - 0.5),
    "E": ( ROAD_HALF_W + 0.5,  ROAD_HALF_W + 0.5),
    "N": (-ROAD_HALF_W - 0.5,  ROAD_HALF_W + 0.5),
    "S": ( ROAD_HALF_W + 0.5, -ROAD_HALF_W - 0.5),
}


def _draw_signs(
    screen: pygame.Surface, cam: Camera,
    int_info: Dict[str, Any], cx: float, cy: float,
) -> None:
    signs = int_info.get("signs", {})
    for direction, sign_name in signs.items():
        offset = _SIGN_OFFSETS.get(direction.upper())
        if not offset:
            continue
        sx, sy = cam.world_to_screen(cx + offset[0], cy + offset[1])
        _draw_single_sign(screen, cam, int(sx), int(sy), sign_name.upper())


def _draw_single_sign(
    screen: pygame.Surface, cam: Camera, sx: int, sy: int, name: str,
) -> None:
    s = max(5, int(3.0 * cam.zoom))
    pole_h = int(s * 1.6)
    pw = max(1, int(cam.zoom * 0.4))
    pygame.draw.line(screen, COLOR_SIGN_POLE, (sx, sy), (sx, sy + pole_h), pw)

    if name == "STOP":
        _draw_hexagon(screen, sx, sy, s, COLOR_STOP_RED)
        if cam.zoom >= 2.5:
            font = pygame.font.SysFont("arial,helvetica", max(8, int(s * 0.6)))
            txt = font.render("STOP", True, COLOR_STOP_WHITE)
            screen.blit(txt, (sx - txt.get_width() // 2,
                              sy - txt.get_height() // 2))
    elif name == "YIELD":
        _draw_yield_triangle(screen, sx, sy, s)
    elif name == "PRIORITY":
        _draw_diamond(screen, sx, sy, s,
                      COLOR_PRIORITY_YELLOW, COLOR_PRIORITY_WHITE)


def _draw_hexagon(
    surface: pygame.Surface, cx: int, cy: int, r: int, color: Tuple[int, ...],
) -> None:
    pts = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        pts.append((cx + int(r * math.cos(angle)),
                     cy - int(r * math.sin(angle))))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (255, 255, 255), pts, max(1, r // 5))


def _draw_yield_triangle(
    surface: pygame.Surface, cx: int, cy: int, r: int,
) -> None:
    pts = [
        (cx - r, cy - int(r * 0.7)),
        (cx + r, cy - int(r * 0.7)),
        (cx, cy + int(r * 0.9)),
    ]
    pygame.draw.polygon(surface, COLOR_YIELD_WHITE, pts)
    pygame.draw.polygon(surface, COLOR_YIELD_RED, pts, max(1, r // 4))


def _draw_diamond(
    surface: pygame.Surface, cx: int, cy: int, r: int,
    fill: Tuple[int, ...], border: Tuple[int, ...],
) -> None:
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surface, fill, pts)
    pygame.draw.polygon(surface, border, pts, max(1, r // 5))


# ── Traffic-light semaphores ──────────────────────────────────────────────────

_LIGHT_OFFSETS: Dict[str, Tuple[float, float]] = {
    "W": (-ROAD_HALF_W - 0.5, -ROAD_HALF_W - 0.5),
    "E": ( ROAD_HALF_W + 0.5,  ROAD_HALF_W + 0.5),
    "N": (-ROAD_HALF_W - 0.5,  ROAD_HALF_W + 0.5),
    "S": ( ROAD_HALF_W + 0.5, -ROAD_HALF_W - 0.5),
}

_COLOR_FOR_PHASE = {
    "GREEN":  COLOR_LIGHT_GREEN,
    "YELLOW": COLOR_LIGHT_YELLOW,
    "RED":    COLOR_LIGHT_RED,
}


def _draw_semaphores(
    screen: pygame.Surface, cam: Camera,
    sem: Dict[str, Any], cx: float, cy: float,
) -> None:
    colors = sem.get("colors", {})
    for approach, (wx, wy) in _LIGHT_OFFSETS.items():
        phase = colors.get(approach, "RED")
        sx, sy = cam.world_to_screen(cx + wx, cy + wy)
        _draw_single_light(screen, cam, int(sx), int(sy), phase)


def _draw_single_light(
    screen: pygame.Surface, cam: Camera, sx: int, sy: int, phase: str,
) -> None:
    bulb_r = max(2, int(1.0 * cam.zoom))
    spacing = int(bulb_r * 2.3)
    housing_w = bulb_r * 2 + max(1, int(cam.zoom * 0.4))
    housing_h = spacing * 2 + bulb_r * 2 + max(1, int(cam.zoom * 0.4))

    pole_h = int(bulb_r * 2.5)
    pw = max(1, int(cam.zoom * 0.35))
    pygame.draw.line(screen, COLOR_SIGN_POLE,
                     (sx, sy + housing_h // 2),
                     (sx, sy + housing_h // 2 + pole_h), pw)

    hr = pygame.Rect(sx - housing_w // 2, sy - housing_h // 2,
                     housing_w, housing_h)
    pygame.draw.rect(screen, COLOR_LIGHT_HOUSING, hr,
                     border_radius=max(1, bulb_r // 2))

    bulb_defs = [
        ("RED",    sy - spacing),
        ("YELLOW", sy),
        ("GREEN",  sy + spacing),
    ]
    for bulb_phase, by in bulb_defs:
        c = _COLOR_FOR_PHASE[bulb_phase] if bulb_phase == phase else COLOR_LIGHT_OFF
        pygame.draw.circle(screen, c, (sx, by), bulb_r)


# ── Decorative elements ──────────────────────────────────────────────────────

def _generate_decorations(
    intersections: List[Dict[str, Any]],
    roads_data: List[Dict[str, Any]],
) -> List[Tuple[Any, ...]]:
    """Procedurally generate trees/houses around each intersection,
    avoiding road corridors and other intersections."""
    hw = ROAD_HALF_W + SIDEWALK_W + 4.0  # keep away from road
    decs: List[Tuple[Any, ...]] = []
    rng = __import__("random").Random(42)  # deterministic seed

    # Build set of road corridor rectangles to avoid
    occupied: List[Tuple[float, float, float, float]] = []
    for info in intersections:
        ccx, ccy = info["center"]
        occupied.append((ccx - 20, ccy - 20, ccx + 20, ccy + 20))

    def _in_road(px: float, py: float) -> bool:
        for info in intersections:
            ccx, ccy = info["center"]
            if abs(px - ccx) < hw or abs(py - ccy) < hw:
                if abs(px - ccx) < 90 or abs(py - ccy) < 90:
                    return True
        return False

    roof_colors = [COLOR_HOUSE_ROOF_A, COLOR_HOUSE_ROOF_B]
    for info in intersections:
        ccx, ccy = info["center"]
        # Scatter trees around each intersection
        for _ in range(8):
            dx = rng.uniform(hw + 2, hw + 50) * rng.choice([-1, 1])
            dy = rng.uniform(hw + 2, hw + 50) * rng.choice([-1, 1])
            px, py = ccx + dx, ccy + dy
            if not _in_road(px, py):
                decs.append(("tree", px, py, rng.uniform(2.2, 3.4)))
        # Scatter houses
        for _ in range(3):
            dx = rng.uniform(hw + 8, hw + 45) * rng.choice([-1, 1])
            dy = rng.uniform(hw + 8, hw + 45) * rng.choice([-1, 1])
            px, py = ccx + dx, ccy + dy
            if not _in_road(px, py):
                decs.append(("house", px, py,
                             rng.uniform(9, 13), rng.uniform(7, 10),
                             rng.choice(roof_colors)))
    return decs


# Module-level cache so decorations are only generated once per session.
_cached_decorations: List[Tuple[Any, ...]] = []
_cached_dec_key: str = ""


def _draw_decorations(
    screen: pygame.Surface, cam: Camera,
    intersections: List[Dict[str, Any]],
    roads_data: List[Dict[str, Any]],
) -> None:
    global _cached_decorations, _cached_dec_key
    key = str([(i["id"], i["center"]) for i in intersections])
    if key != _cached_dec_key:
        _cached_decorations = _generate_decorations(intersections, roads_data)
        _cached_dec_key = key
    for dec in _cached_decorations:
        kind = dec[0]
        if kind == "tree":
            _draw_tree(screen, cam, dec[1], dec[2], dec[3])
        elif kind == "house":
            _draw_house(screen, cam, dec[1], dec[2], dec[3], dec[4], dec[5])


def _draw_tree(
    screen: pygame.Surface, cam: Camera, wx: float, wy: float, size: float,
) -> None:
    sx, sy = cam.world_to_screen(wx, wy)
    sx, sy = int(sx), int(sy)
    cr = max(3, int(size * cam.zoom))
    tr_w = max(1, int(size * 0.25 * cam.zoom))
    tr_h = max(2, int(size * 0.5 * cam.zoom))
    draw_alpha_circle(screen, (0, 0, 0, 25), (sx + 2, sy + 2), cr)
    pygame.draw.rect(screen, COLOR_TREE_TRUNK,
                     (sx - tr_w // 2, sy, tr_w, tr_h))
    pygame.draw.circle(screen, COLOR_TREE_CANOPY_A,
                       (sx, sy - int(size * 0.2 * cam.zoom)), cr)
    pygame.draw.circle(screen, COLOR_TREE_CANOPY_B,
                       (sx - cr // 4, sy - int(size * 0.35 * cam.zoom)),
                       cr // 2)


def _draw_house(
    screen: pygame.Surface, cam: Camera,
    wx: float, wy: float, w: float, h: float,
    roof_color: Tuple[int, ...],
) -> None:
    rect = _world_rect(cam, wx - w / 2, wy - h / 2, wx + w / 2, wy + h / 2)
    shadow = rect.copy()
    shadow.move_ip(3, 3)
    draw_alpha_rect(screen, (0, 0, 0, 30), shadow)
    pygame.draw.rect(screen, COLOR_HOUSE_WALL, rect)
    roof_h = max(2, rect.h // 3)
    roof_rect = pygame.Rect(rect.x, rect.y, rect.w, roof_h)
    pygame.draw.rect(screen, roof_color, roof_rect)
    dw = max(2, rect.w // 5)
    dh = max(3, rect.h // 3)
    pygame.draw.rect(screen, COLOR_HOUSE_DOOR,
                     (rect.centerx - dw // 2, rect.bottom - dh, dw, dh))
    win_sz = max(2, rect.w // 6)
    if rect.w > 20:
        pygame.draw.rect(screen, COLOR_HOUSE_WINDOW,
                         (rect.x + rect.w // 5, rect.y + roof_h + 2,
                          win_sz, win_sz))
        pygame.draw.rect(screen, COLOR_HOUSE_WINDOW,
                         (rect.right - rect.w // 5 - win_sz,
                          rect.y + roof_h + 2, win_sz, win_sz))


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _i2(pair: Tuple[float, float]) -> Tuple[int, int]:
    return int(pair[0]), int(pair[1])
