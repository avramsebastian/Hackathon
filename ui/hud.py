"""
ui/hud.py
=========
Heads-up display panels drawn on top of the simulation:

 • Top bar      — sim clock, active vehicle count, paused badge
 • Vehicle panel — bottom-left list of vehicles with speed + SLOW DOWN
 • Legend        — bottom-right icon/colour key
 • Control bar   — bottom strip with Start / Pause / Reset buttons
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pygame

from ui.constants import (
    COLOR_HUD_BG, COLOR_HUD_BORDER, COLOR_HUD_TEXT, COLOR_HUD_DIM,
    COLOR_HUD_ACCENT, COLOR_WARNING,
    COLOR_BTN_BG, COLOR_BTN_HOVER, COLOR_BTN_TEXT,
    COLOR_STOP_RED, COLOR_YIELD_RED, COLOR_PRIORITY_YELLOW,
    HUD_PANEL_W, HUD_PAD, HUD_ROW_H, CONTROL_BAR_H,
    BTN_W, BTN_H, LEGEND_W, TOP_BAR_H,
    FONT_SM, FONT_MD, FONT_LG,
)
from ui.helpers import draw_alpha_rect, render_text, should_slow_down
from ui.types import ButtonRect


# ══════════════════════════════════════════════════════════════════════════════
#  FONTS (lazy-loaded singleton)
# ══════════════════════════════════════════════════════════════════════════════

_fonts: Dict[str, pygame.font.Font] = {}


def _f(key: str) -> pygame.font.Font:
    if not _fonts:
        _fonts["sm"] = pygame.font.SysFont("arial,helvetica", FONT_SM)
        _fonts["md"] = pygame.font.SysFont("arial,helvetica", FONT_MD)
        _fonts["lg"] = pygame.font.SysFont("arial,helvetica", FONT_LG, bold=True)
    return _fonts[key]


# ══════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ══════════════════════════════════════════════════════════════════════════════

def draw_top_bar(
    screen: pygame.Surface,
    vehicles: List[Dict[str, Any]],
    sim_time: float,
    paused: bool,
) -> None:
    w = screen.get_width()
    draw_alpha_rect(screen, COLOR_HUD_BG, pygame.Rect(0, 0, w, TOP_BAR_H), border_radius=0)

    # Clock
    mins = int(sim_time) // 60
    secs = int(sim_time) % 60
    ms = int((sim_time % 1) * 10)
    clock_str = f"{mins:02d}:{secs:02d}.{ms}"
    render_text(screen, _f("md"), clock_str, (HUD_PAD, 7), COLOR_HUD_TEXT)

    # Vehicle count
    n = len(vehicles)
    render_text(screen, _f("sm"), f"Vehicles: {n}", (w // 2, 9), COLOR_HUD_DIM, anchor="midtop")

    # Paused badge
    if paused:
        render_text(screen, _f("md"), "⏸  PAUSED", (w - HUD_PAD, 7), COLOR_WARNING, anchor="topright")


# ══════════════════════════════════════════════════════════════════════════════
#  VEHICLE PANEL  (bottom-left, expands upward)
# ══════════════════════════════════════════════════════════════════════════════

def draw_vehicle_panel(
    screen: pygame.Surface,
    vehicles: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    frame: int,
) -> None:
    sh = screen.get_height()
    max_panel_h = max(120, sh - CONTROL_BAR_H - TOP_BAR_H - 16)
    base_h = HUD_PAD * 2 + 24
    more_h = 18
    max_rows = max(1, (max_panel_h - base_h - more_h) // HUD_ROW_H)

    vehicles_sorted = sorted(vehicles, key=lambda v: str(v.get("id", "")))
    visible = vehicles_sorted[:max_rows]
    hidden = max(0, len(vehicles_sorted) - len(visible))

    panel_h = base_h + HUD_ROW_H * max(1, len(visible))
    if hidden > 0:
        panel_h += more_h

    panel_y = sh - CONTROL_BAR_H - panel_h - 6
    panel_rect = pygame.Rect(6, panel_y, HUD_PANEL_W, panel_h)

    draw_alpha_rect(screen, COLOR_HUD_BG, panel_rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_HUD_BORDER, panel_rect, 1, border_radius=8)

    # Header
    render_text(screen, _f("md"), "Active Vehicles", (panel_rect.x + HUD_PAD, panel_rect.y + HUD_PAD), COLOR_HUD_ACCENT)

    y = panel_rect.y + HUD_PAD + 24
    for v in visible:
        vid = v["id"]
        speed = v.get("speed", 0.0)
        color = _tuple_color(v.get("color", (180, 180, 180)))
        dec = decisions.get(vid, {})
        ml = dec.get("decision", "none")
        slow = should_slow_down(v, ml)

        # Colour swatch
        pygame.draw.rect(screen, color, (panel_rect.x + HUD_PAD, y + 5, 14, 14), border_radius=3)

        # ID
        render_text(screen, _f("sm"), vid, (panel_rect.x + HUD_PAD + 20, y + 4), COLOR_HUD_TEXT)

        # Speed
        render_text(
            screen, _f("sm"),
            f"{speed:.0f} km/h",
            (panel_rect.x + HUD_PAD + 120, y + 4),
            COLOR_HUD_DIM,
        )

        # Blinking SLOW DOWN
        if slow:
            blink = (frame // 18) % 2 == 0  # ~0.6 s cycle at 60 fps
            if blink:
                render_text(
                    screen, _f("sm"),
                    "● SLOW DOWN",
                    (panel_rect.x + HUD_PAD + 190, y + 4),
                    COLOR_WARNING,
                )

        y += HUD_ROW_H

    if hidden > 0:
        render_text(
            screen,
            _f("sm"),
            f"+{hidden} more",
            (panel_rect.x + HUD_PAD, panel_rect.bottom - HUD_PAD - 12),
            COLOR_HUD_DIM,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LEGEND  (bottom-right)
# ══════════════════════════════════════════════════════════════════════════════

def draw_legend(screen: pygame.Surface) -> None:
    sw, sh = screen.get_size()
    items = [
        ("Route line",        (86, 168, 255)),
        ("Stop sign",         COLOR_STOP_RED),
        ("Yield sign",        COLOR_YIELD_RED),
        ("Priority sign",     COLOR_PRIORITY_YELLOW),
        ("Slow-down zone",    COLOR_WARNING),
    ]
    panel_h = HUD_PAD * 2 + len(items) * 22 + 24
    panel_x = sw - LEGEND_W - 6
    panel_y = sh - CONTROL_BAR_H - panel_h - 6
    panel_rect = pygame.Rect(panel_x, panel_y, LEGEND_W, panel_h)

    draw_alpha_rect(screen, COLOR_HUD_BG, panel_rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_HUD_BORDER, panel_rect, 1, border_radius=8)

    render_text(screen, _f("md"), "Legend", (panel_x + HUD_PAD, panel_y + HUD_PAD), COLOR_HUD_ACCENT)

    y = panel_y + HUD_PAD + 24
    for label, color in items:
        pygame.draw.rect(screen, color, (panel_x + HUD_PAD, y + 2, 12, 12), border_radius=2)
        render_text(screen, _f("sm"), label, (panel_x + HUD_PAD + 18, y), COLOR_HUD_TEXT)
        y += 22


# ══════════════════════════════════════════════════════════════════════════════
#  CONTROL BAR  (bottom strip)
# ══════════════════════════════════════════════════════════════════════════════

def draw_control_bar(
    screen: pygame.Surface,
    paused: bool,
    mouse_pos: Tuple[int, int],
) -> List[ButtonRect]:
    """
    Draw the control bar and return a list of :class:`ButtonRect` objects
    so the caller can detect clicks.
    """
    sw, sh = screen.get_size()
    bar_rect = pygame.Rect(0, sh - CONTROL_BAR_H, sw, CONTROL_BAR_H)
    draw_alpha_rect(screen, COLOR_HUD_BG, bar_rect)
    pygame.draw.line(screen, COLOR_HUD_BORDER, (0, bar_rect.y), (sw, bar_rect.y))

    buttons: List[ButtonRect] = []
    labels = [("▶  Start", "start"), ("⏸  Pause", "pause"), ("↺  Reset", "reset"), ("🔄  New", "new")]
    total_w = len(labels) * (BTN_W + 10) - 10
    bx = sw // 2 - total_w // 2
    by = bar_rect.y + (CONTROL_BAR_H - BTN_H) // 2

    for text, name in labels:
        rect = pygame.Rect(bx, by, BTN_W, BTN_H)
        hovered = rect.collidepoint(mouse_pos)

        # Highlight the active state button
        if (name == "pause" and paused) or (name == "start" and not paused):
            bg = COLOR_HUD_ACCENT
        else:
            bg = COLOR_BTN_HOVER if hovered else COLOR_BTN_BG

        pygame.draw.rect(screen, bg, rect, border_radius=6)
        render_text(screen, _f("sm"), text, rect.center, COLOR_BTN_TEXT, anchor="center")
        buttons.append(ButtonRect(name, bx, by, BTN_W, BTN_H))
        bx += BTN_W + 10

    return buttons


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tuple_color(c: Any) -> Tuple[int, int, int]:
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    return (180, 180, 180)
