"""
ui/pygame_view.py
=================
Main Pygame loop that ties everything together:

 1. **Launch screen** — stylised title card with START SIMULATION button.
 2. **Simulation screen** — map + vehicles + HUD + control bar.

The only public symbol is :func:`run_pygame_view`; it blocks until the
user closes the window.

Bridge dependency
-----------------
The *bridge* object must expose:

    get_vehicles()           → List[dict]
    get_ml_decision(id: str) → dict   {"decision", …}
    get_intersection()       → dict   {"signs", "lane_count", "box_size"}
    is_finished()            → bool
    reset()                  → None
    set_paused(bool)         → None
"""

from __future__ import annotations

import time
import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import pygame

from ui.constants import (
    DEFAULT_ZOOM, MIN_ZOOM, MAX_ZOOM, ZOOM_STEP,
    CONTROL_BAR_H,
    COLOR_LAUNCH_BG, COLOR_LAUNCH_GRID, COLOR_LAUNCH_ROAD,
    COLOR_LAUNCH_DASH, COLOR_LAUNCH_TITLE, COLOR_LAUNCH_SUB,
    COLOR_LAUNCH_BTN, COLOR_LAUNCH_BTN_H, COLOR_HUD_TEXT,
    COLOR_HUD_DIM, COLOR_WARNING,
    FONT_SM, FONT_MD, FONT_LG, FONT_XL, FONT_TTL,
)
from ui.types import Camera, ButtonRect
from ui.draw_road import draw_map
from ui.draw_vehicles import draw_all_vehicles
from ui.hud import draw_top_bar, draw_vehicle_panel, draw_control_bar

log = logging.getLogger("pygame_view")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_pygame_view(
    bridge: Any,
    width: int = 1000,
    height: int = 700,
    fps: int = 60,
) -> None:
    """
    Open the Pygame window, show the launch screen, then the simulation.
    Blocks until the window is closed.
    """
    pygame.init()
    pygame.display.set_caption("V2X Intersection Safety Simulator")
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    fonts = _load_fonts()

    # ── Get network info from bridge ──
    num_intersections, grid_cols, grid_rows = bridge.get_grid_info()
    bounds_x, bounds_y = bridge.get_network_bounds(width, height)
    network_width = bounds_x[1] - bounds_x[0]
    network_height = bounds_y[1] - bounds_y[0]
    
    # Compute min_zoom based on intersection count
    # More intersections = allow more zoom out
    # 2 intersections: min_zoom ~3.0, 9 intersections: min_zoom ~1.5
    base_min_zoom = 3.5 - (num_intersections - 2) * 0.3
    computed_min_zoom = max(1.2, min(3.5, base_min_zoom))
    
    # Calculate zoom to fit the network (fill ~75% of screen)
    zoom_to_fit_x = 0.75 * width / network_width if network_width > 0 else DEFAULT_ZOOM
    zoom_to_fit_y = 0.75 * height / network_height if network_height > 0 else DEFAULT_ZOOM
    initial_zoom = min(zoom_to_fit_x, zoom_to_fit_y)
    
    # Clamp zoom to computed min
    initial_zoom = max(computed_min_zoom, min(initial_zoom, MAX_ZOOM))

    # ── state ──
    scene: str = "launch"  # "launch" | "sim"
    # Center camera on the middle of the intersection network
    camera = Camera(width, height, zoom=initial_zoom)
    camera.world_x = (bounds_x[0] + bounds_x[1]) / 2  # center X
    camera.world_y = (bounds_y[0] + bounds_y[1]) / 2  # center Y
    
    # Store dynamic bounds for clamping
    world_bounds_x = bounds_x
    world_bounds_y = bounds_y
    dynamic_min_zoom = computed_min_zoom
    
    paused = False
    sim_time = 0.0
    frame = 0

    # Vehicle interpolation
    prev_vehicles: List[Dict[str, Any]] = []
    curr_vehicles: List[Dict[str, Any]] = []
    last_poll = time.time()
    poll_interval = 0.1  # 10 Hz, matching bridge default

    # Panning state
    panning = False
    pan_origin: Optional[Tuple[int, int]] = None
    cam_origin: Optional[Tuple[float, float]] = None

    running = True
    while running:
        dt = clock.tick(fps) / 1000.0
        mouse_pos = pygame.mouse.get_pos()
        frame += 1

        # ── Events ────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if scene == "launch":
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        scene = "sim"
                elif scene == "sim":
                    _handle_sim_key(event.key, bridge, camera, world_bounds_x, world_bounds_y)
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        bridge.set_paused(paused)
                    elif event.key == pygame.K_r:
                        bridge.reset()
                        sim_time = 0.0
                        prev_vehicles = []
                        curr_vehicles = []
                    elif event.key == pygame.K_n:
                        bridge.new_scenario()
                        sim_time = 0.0
                        prev_vehicles = []
                        curr_vehicles = []
                        # Recompute bounds and zoom for new network
                        num_intersections, grid_cols, grid_rows = bridge.get_grid_info()
                        bounds_x, bounds_y = bridge.get_network_bounds(width, height)
                        network_width = bounds_x[1] - bounds_x[0]
                        network_height = bounds_y[1] - bounds_y[0]
                        base_min_zoom = 3.5 - (num_intersections - 2) * 0.3
                        computed_min_zoom = max(1.2, min(3.5, base_min_zoom))
                        zoom_to_fit_x = 0.75 * width / network_width if network_width > 0 else DEFAULT_ZOOM
                        zoom_to_fit_y = 0.75 * height / network_height if network_height > 0 else DEFAULT_ZOOM
                        initial_zoom = min(zoom_to_fit_x, zoom_to_fit_y)
                        initial_zoom = max(computed_min_zoom, min(initial_zoom, MAX_ZOOM))
                        camera.zoom = initial_zoom
                        camera.world_x = (bounds_x[0] + bounds_x[1]) / 2
                        camera.world_y = (bounds_y[0] + bounds_y[1]) / 2
                        world_bounds_x = bounds_x
                        world_bounds_y = bounds_y
                        dynamic_min_zoom = computed_min_zoom
                    elif event.key == pygame.K_F12:
                        _screenshot(screen)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if scene == "launch":
                    # START button hit-test is handled below via btn_rect
                    pass
                elif scene == "sim":
                    if event.button == 3 or event.button == 2:  # right / middle
                        panning = True
                        pan_origin = event.pos
                        cam_origin = (camera.world_x, camera.world_y)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (2, 3):
                    panning = False

            elif event.type == pygame.MOUSEMOTION and panning:
                if pan_origin and cam_origin:
                    dx = event.pos[0] - pan_origin[0]
                    dy = event.pos[1] - pan_origin[1]
                    camera.world_x = cam_origin[0] - dx / camera.zoom
                    camera.world_y = cam_origin[1] + dy / camera.zoom
                    _clamp_camera(camera, world_bounds_x, world_bounds_y)

            elif event.type == pygame.MOUSEWHEEL and scene == "sim":
                # Zoom toward cursor
                _zoom_toward(camera, mouse_pos, event.y, world_bounds_x, world_bounds_y)

        # ── Render ────────────────────────────────────────────────────────
        if scene == "launch":
            btn_rect = _draw_launch_screen(screen, fonts, mouse_pos, frame)
            # Check START click
            if pygame.mouse.get_pressed()[0] and btn_rect.collidepoint(mouse_pos):
                scene = "sim"

        elif scene == "sim":
            if not paused and not bridge.is_finished():
                sim_time += dt

            # Poll bridge
            now = time.time()
            if now - last_poll >= poll_interval:
                prev_vehicles = curr_vehicles
                curr_vehicles = bridge.get_vehicles()
                last_poll = now

            # Interpolation factor
            t = min(1.0, (now - last_poll) / poll_interval) if poll_interval > 0 else 1.0
            from ui.helpers import interpolate_vehicles
            interp = interpolate_vehicles(prev_vehicles, curr_vehicles, t)

            # Decisions
            if hasattr(bridge, "get_all_ml_decisions"):
                decisions = bridge.get_all_ml_decisions()
            else:
                decisions: Dict[str, Dict[str, Any]] = {}
                for v in curr_vehicles:
                    decisions[v["id"]] = bridge.get_ml_decision(v["id"])

            intersection = bridge.get_intersection()

            # Draw layers
            draw_map(screen, camera, intersection)
            draw_all_vehicles(screen, camera, interp, decisions, frame)
            draw_top_bar(screen, interp, sim_time, paused)
            draw_vehicle_panel(screen, interp, decisions, frame)
            ctrl_btns = draw_control_bar(screen, paused, mouse_pos)

            # Control-bar click handling
            if pygame.mouse.get_pressed()[0]:
                for btn in ctrl_btns:
                    if btn.contains(*mouse_pos):
                        if btn.label == "start":
                            paused = False
                            bridge.set_paused(False)
                        elif btn.label == "pause":
                            paused = True
                            bridge.set_paused(True)
                        elif btn.label == "reset":
                            bridge.reset()
                            sim_time = 0.0
                            prev_vehicles = []
                            curr_vehicles = []
                        elif btn.label == "new":
                            bridge.new_scenario()
                            sim_time = 0.0
                            prev_vehicles = []
                            curr_vehicles = []
                            # Recompute bounds and zoom for new network
                            num_intersections, grid_cols, grid_rows = bridge.get_grid_info()
                            bounds_x, bounds_y = bridge.get_network_bounds(width, height)
                            network_width = bounds_x[1] - bounds_x[0]
                            network_height = bounds_y[1] - bounds_y[0]
                            base_min_zoom = 3.5 - (num_intersections - 2) * 0.3
                            computed_min_zoom = max(1.2, min(3.5, base_min_zoom))
                            zoom_to_fit_x = 0.75 * width / network_width if network_width > 0 else DEFAULT_ZOOM
                            zoom_to_fit_y = 0.75 * height / network_height if network_height > 0 else DEFAULT_ZOOM
                            initial_zoom = min(zoom_to_fit_x, zoom_to_fit_y)
                            initial_zoom = max(computed_min_zoom, min(initial_zoom, MAX_ZOOM))
                            camera.zoom = initial_zoom
                            camera.world_x = (bounds_x[0] + bounds_x[1]) / 2
                            camera.world_y = (bounds_y[0] + bounds_y[1]) / 2
                            world_bounds_x = bounds_x
                            world_bounds_y = bounds_y
                            dynamic_min_zoom = computed_min_zoom

            # Finished overlay
            if bridge.is_finished():
                _draw_finished_overlay(screen, fonts)

        pygame.display.flip()

    pygame.quit()


# ══════════════════════════════════════════════════════════════════════════════
#  LAUNCH SCREEN
# ══════════════════════════════════════════════════════════════════════════════

def _draw_launch_screen(
    screen: pygame.Surface,
    fonts: Dict[str, pygame.font.Font],
    mouse_pos: Tuple[int, int],
    frame: int,
) -> pygame.Rect:
    """Draw the title card and return the START button Rect."""
    w, h = screen.get_size()
    screen.fill(COLOR_LAUNCH_BG)

    # Subtle grid
    for x in range(0, w, 40):
        pygame.draw.line(screen, COLOR_LAUNCH_GRID, (x, 0), (x, h))
    for y in range(0, h, 40):
        pygame.draw.line(screen, COLOR_LAUNCH_GRID, (0, y), (w, y))

    # Decorative road strip
    road_y = h // 2 + 10
    pygame.draw.rect(screen, COLOR_LAUNCH_ROAD, (0, road_y - 25, w, 50))
    for dx in range(0, w, 40):
        pygame.draw.rect(screen, COLOR_LAUNCH_DASH, (dx, road_y - 1, 20, 2))

    # Animated tiny car on road
    car_x = int((frame * 1.8) % (w + 100)) - 50
    pygame.draw.rect(screen, (86, 168, 255), (car_x, road_y - 8, 30, 14), border_radius=4)
    pygame.draw.circle(screen, (255, 255, 200), (car_x + 28, road_y - 4), 3)
    pygame.draw.circle(screen, (255, 255, 200), (car_x + 28, road_y + 4), 3)

    # Title
    t1 = fonts["title"].render("INTERSECTION", True, COLOR_LAUNCH_TITLE)
    t2 = fonts["xlarge"].render("VISIBILITY  ASSISTANT", True, COLOR_LAUNCH_SUB)
    screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 4 - 30))
    screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 4 + 40))

    # Subtitle
    sub = fonts["small"].render("Pygame 2-D Simulation  •  V2X Safety", True, COLOR_HUD_DIM)
    screen.blit(sub, (w // 2 - sub.get_width() // 2, h // 4 + 80))

    # START button
    btn_w, btn_h = 280, 54
    btn_x = w // 2 - btn_w // 2
    btn_y = h * 3 // 4 - btn_h // 2
    btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
    hovered = btn_rect.collidepoint(mouse_pos)
    btn_color = COLOR_LAUNCH_BTN_H if hovered else COLOR_LAUNCH_BTN

    # Glow
    if hovered:
        glow = pygame.Surface((btn_w + 24, btn_h + 24), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*btn_color, 35), (0, 0, btn_w + 24, btn_h + 24), border_radius=18)
        screen.blit(glow, (btn_x - 12, btn_y - 12))

    pygame.draw.rect(screen, btn_color, btn_rect, border_radius=12)
    txt = fonts["medium"].render("START  SIMULATION", True, (255, 255, 255))
    screen.blit(txt, (btn_x + btn_w // 2 - txt.get_width() // 2, btn_y + btn_h // 2 - txt.get_height() // 2))

    # Controls hint
    hint = fonts["small"].render("Press ENTER or click the button", True, COLOR_HUD_DIM)
    screen.blit(hint, (w // 2 - hint.get_width() // 2, btn_y + btn_h + 18))

    return btn_rect


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_finished_overlay(
    screen: pygame.Surface, fonts: Dict[str, pygame.font.Font]
) -> None:
    w, h = screen.get_size()
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))
    txt = fonts["xlarge"].render("SIMULATION  COMPLETE", True, COLOR_HUD_TEXT)
    screen.blit(txt, (w // 2 - txt.get_width() // 2, h // 2 - txt.get_height() // 2))
    sub = fonts["medium"].render("Press R to reset", True, COLOR_HUD_DIM)
    screen.blit(sub, (w // 2 - sub.get_width() // 2, h // 2 + 30))


def _handle_sim_key(key: int, bridge: Any, camera: Camera, bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> None:
    """Handle zoom and pan keys (non-toggle)."""
    if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        camera.zoom = min(MAX_ZOOM, camera.zoom + ZOOM_STEP)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        camera.zoom = max(MIN_ZOOM, camera.zoom - ZOOM_STEP)
    elif key == pygame.K_UP:
        camera.world_y += 10.0 / camera.zoom
    elif key == pygame.K_DOWN:
        camera.world_y -= 10.0 / camera.zoom
    elif key == pygame.K_LEFT:
        camera.world_x -= 10.0 / camera.zoom
    elif key == pygame.K_RIGHT:
        camera.world_x += 10.0 / camera.zoom
    _clamp_camera(camera, bounds_x, bounds_y)


def _clamp_camera(camera: Camera, bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> None:
    """Clamp camera position so the visible area stays within road bounds.
    
    The camera position is the world point at screen center, so we must
    account for the visible half-width/height based on zoom level.
    """
    # Calculate visible area half-dimensions in world space
    visible_half_w = camera.screen_w / (2.0 * camera.zoom)
    visible_half_h = camera.screen_h / (2.0 * camera.zoom)
    
    # Calculate allowed camera position range
    # Camera must be far enough from edges that visible area stays inside bounds
    min_cam_x = bounds_x[0] + visible_half_w
    max_cam_x = bounds_x[1] - visible_half_w
    min_cam_y = bounds_y[0] + visible_half_h
    max_cam_y = bounds_y[1] - visible_half_h
    
    # If the visible area is larger than the bounds, center the camera
    if min_cam_x > max_cam_x:
        camera.world_x = (bounds_x[0] + bounds_x[1]) / 2
    else:
        camera.world_x = max(min_cam_x, min(max_cam_x, camera.world_x))
    
    if min_cam_y > max_cam_y:
        camera.world_y = (bounds_y[0] + bounds_y[1]) / 2
    else:
        camera.world_y = max(min_cam_y, min(max_cam_y, camera.world_y))


def _zoom_toward(camera: Camera, mouse_pos: Tuple[int, int], direction: int, bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> None:
    """Zoom in/out centred on mouse cursor."""
    old_zoom = camera.zoom
    if direction > 0:
        camera.zoom = min(MAX_ZOOM, camera.zoom + ZOOM_STEP)
    else:
        camera.zoom = max(MIN_ZOOM, camera.zoom - ZOOM_STEP)
    # Adjust world offset so the point under the cursor stays fixed
    mx, my = mouse_pos
    cx, cy = camera.screen_w / 2, camera.screen_h / 2
    factor = 1.0 / camera.zoom - 1.0 / old_zoom
    camera.world_x += (mx - cx) * factor
    camera.world_y -= (my - cy) * factor
    _clamp_camera(camera, bounds_x, bounds_y)


def _screenshot(screen: pygame.Surface) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"screenshots/sim_{ts}.png"
    try:
        import os
        os.makedirs("screenshots", exist_ok=True)
        pygame.image.save(screen, path)
        log.info("Screenshot saved to %s", path)
    except Exception as e:
        log.warning("Could not save screenshot: %s", e)


def _load_fonts() -> Dict[str, pygame.font.Font]:
    family = "arial,helvetica"
    return {
        "small":  pygame.font.SysFont(family, FONT_SM),
        "medium": pygame.font.SysFont(family, FONT_MD),
        "large":  pygame.font.SysFont(family, FONT_LG, bold=True),
        "xlarge": pygame.font.SysFont(family, FONT_XL, bold=True),
        "title":  pygame.font.SysFont(family, FONT_TTL, bold=True),
    }
