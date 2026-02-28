#!/usr/bin/env python3
"""HUD panel, legend, debug overlay, splash screen, and pause banner (mixin)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pygame


class HudRenderer:
    """Mixin that draws every overlay / HUD element."""

    # ------------------------------------------------------------------ #
    #  Main HUD panel                                                      #
    # ------------------------------------------------------------------ #

    def draw_hud(
        self,
        surface: pygame.Surface,
        vehicles: Sequence[Any],
        ml_decisions: Mapping[str, Any],
        tick: float,
    ) -> None:
        if self.font_small is None or self.font_tiny is None:
            return

        max_visible = 4
        total = len(vehicles)
        scroll = min(self._hud_scroll_offset, max(0, total - max_visible))
        self._hud_scroll_offset = scroll
        visible_vehicles = list(vehicles[scroll : scroll + max_visible])

        row_height = 52
        header_h = 30
        panel_height = header_h + max(1, len(visible_vehicles)) * row_height + 8
        panel_width = 270
        panel_rect = pygame.Rect(16, self.height - panel_height - 16, panel_width, panel_height)

        pygame.draw.rect(surface, self.HUD_BG_COLOR, panel_rect, border_radius=6)
        pygame.draw.rect(surface, self.HUD_BORDER_COLOR, panel_rect, width=1, border_radius=6)

        # Header
        hdr_y = panel_rect.y + 6
        hdr_text = self.font_tiny.render(
            f"VEHICLES {total}   CONFLICTS {self._conflict_total}",
            True,
            (180, 180, 180),
        )
        surface.blit(hdr_text, (panel_rect.x + 10, hdr_y))
        if total > max_visible:
            scroll_hint = self.font_tiny.render(
                f"[{scroll + 1}-{scroll + len(visible_vehicles)}/{total}]",
                True,
                (120, 120, 120),
            )
            surface.blit(scroll_hint, (panel_rect.x + panel_width - 80, hdr_y))

        y = panel_rect.y + header_h
        blink_on = int((tick * 1000) // self.HUD_BLINK_MS) % 2 == 0
        bar_w = 80
        bar_h = 6

        for vehicle in visible_vehicles:
            vehicle_id = self._vehicle_id(vehicle)
            state = self.vehicle_states.get(vehicle_id)
            color = state.color if state else (255, 255, 255)
            effective_speed_mps = self._speed_mps(vehicle) * (
                state.speed_scale_current if state else 1.0
            )
            speed_kmh = self._speed_to_kmh(effective_speed_mps)

            raw_decision = ml_decisions.get(vehicle_id, "none")
            decision = self._normalize_decision(raw_decision)

            # Row: ID + speed
            id_text = self.font_small.render(f"ID {vehicle_id}".upper(), True, color)
            speed_text = self.font_tiny.render(
                f"SPEED {speed_kmh:>4.1f} KM/H", True, (240, 240, 240)
            )
            surface.blit(id_text, (panel_rect.x + 10, y))
            surface.blit(speed_text, (panel_rect.x + 10, y + 16))

            if decision == "stop" and blink_on:
                warn = self.font_tiny.render("SLOW DOWN", True, self.WARNING_COLOR)
                surface.blit(warn, (panel_rect.x + 165, y + 16))

            # Confidence bar
            conf_go = self._extract_confidence(raw_decision, "go")
            conf_stop = self._extract_confidence(raw_decision, "stop")
            if conf_go is not None or conf_stop is not None:
                bx = panel_rect.x + 10
                by = y + 32
                pygame.draw.rect(surface, (40, 40, 40), (bx, by, bar_w, bar_h), border_radius=2)
                go_val = conf_go if conf_go is not None else (1.0 - (conf_stop or 0.0))
                go_px = max(0, min(bar_w, int(go_val * bar_w)))
                if go_px > 0:
                    pygame.draw.rect(
                        surface, self.GO_COLOR, (bx, by, go_px, bar_h), border_radius=2
                    )
                stop_val = conf_stop if conf_stop is not None else (1.0 - (conf_go or 0.0))
                stop_px = max(0, min(bar_w - go_px, int(stop_val * bar_w)))
                if stop_px > 0:
                    pygame.draw.rect(
                        surface,
                        self.STOP_COLOR,
                        (bx + bar_w - stop_px, by, stop_px, bar_h),
                        border_radius=2,
                    )
                lbl = self.font_tiny.render(
                    f"GO {go_val * 100:.0f}%  STOP {stop_val * 100:.0f}%",
                    True,
                    (160, 160, 160),
                )
                surface.blit(lbl, (bx + bar_w + 6, by - 2))

            y += row_height

    # ------------------------------------------------------------------ #
    #  Splash screen                                                       #
    # ------------------------------------------------------------------ #

    def _draw_splash(self, surface: pygame.Surface, tick: float) -> None:
        if self.font_title is None or self.font_small is None:
            return
        title = self.font_title.render("INTERSECTION SAFETY SIM", True, (240, 240, 240))
        surface.blit(
            title,
            title.get_rect(center=(self.width // 2, self.height // 2 - 30)),
        )
        if int(tick * 2) % 2 == 0:
            prompt = self.font_small.render("Press any key to start", True, (160, 160, 160))
            surface.blit(
                prompt,
                prompt.get_rect(center=(self.width // 2, self.height // 2 + 20)),
            )
        lines = [
            "SPACE  Pause/Resume",
            "+ / -  Zoom in/out",
            "R      Reset view",
            "L      Toggle legend",
            "F3     Debug overlay",
            "F12    Screenshot",
            "UP/DN  Scroll HUD",
        ]
        y = self.height // 2 + 60
        for line in lines:
            t = self.font_tiny.render(line, True, (100, 100, 100)) if self.font_tiny else None
            if t:
                surface.blit(t, t.get_rect(center=(self.width // 2, y)))
                y += 16

    # ------------------------------------------------------------------ #
    #  Legend                                                               #
    # ------------------------------------------------------------------ #

    def _draw_legend(self, surface: pygame.Surface) -> None:
        if self.font_tiny is None:
            return
        x = self.width - 120
        y = self.height - 16 - len(self.LEGEND_ITEMS) * 18 - 8
        box_w, box_h = 112, len(self.LEGEND_ITEMS) * 18 + 10
        pygame.draw.rect(
            surface, self.HUD_BG_COLOR, (x - 6, y - 4, box_w, box_h), border_radius=4
        )
        pygame.draw.rect(
            surface, self.HUD_BORDER_COLOR, (x - 6, y - 4, box_w, box_h), width=1, border_radius=4
        )
        for label, color in self.LEGEND_ITEMS:
            pygame.draw.circle(surface, color, (x + 4, y + 6), 4)
            text = self.font_tiny.render(label, True, (200, 200, 200))
            surface.blit(text, (x + 14, y))
            y += 18

    # ------------------------------------------------------------------ #
    #  Debug / FPS overlay                                                 #
    # ------------------------------------------------------------------ #

    def _draw_debug_overlay(
        self, surface: pygame.Surface, vehicles: Sequence[Any], dt: float
    ) -> None:
        if self.font_tiny is None:
            return
        fps = self.clock.get_fps() if self.clock else 0.0
        lines = [
            f"FPS  {fps:.1f}",
            f"DT   {dt * 1000:.1f} ms",
            f"VEH  {len(vehicles)}",
            f"ZOOM {self.zoom:.1f}x",
            f"RES  {self.width}x{self.height}",
            f"TIME {self.time_seconds:.1f}s",
            f"CONF {self._conflict_total}",
        ]
        x, y = 16, 16
        for line in lines:
            text = self.font_tiny.render(line, True, (0, 255, 127))
            surface.blit(text, (x, y))
            y += 14

    # ------------------------------------------------------------------ #
    #  Pause banner                                                        #
    # ------------------------------------------------------------------ #

    def _draw_pause_banner(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (0, 0))
        if self.font_title:
            text = self.font_title.render("PAUSED", True, (220, 220, 220))
            surface.blit(text, text.get_rect(center=(self.width // 2, self.height // 2)))
