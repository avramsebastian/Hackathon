#!/usr/bin/env python3
"""Road, lane-marking, and sign rendering (mixin)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import pygame

from .types import ColorRGB, ColorRGBA


class RoadRenderer:
    """Mixin that draws the intersection roads, lanes, and traffic signs."""

    # --- public draw methods ------------------------------------------ #

    def draw_road(self, surface: pygame.Surface, intersection: Any) -> None:
        lanes_h, lanes_v = self._lane_counts(intersection)
        horizontal_road_width = max(2, lanes_h) * self.LANE_WIDTH
        vertical_road_width = max(2, lanes_v) * self.LANE_WIDTH

        cx = int(self.center.x)
        cy = int(self.center.y)

        horizontal_rect = pygame.Rect(
            0, cy - horizontal_road_width // 2, self.width, horizontal_road_width
        )
        vertical_rect = pygame.Rect(
            cx - vertical_road_width // 2, 0, vertical_road_width, self.height
        )
        pygame.draw.rect(surface, self.ROAD_COLOR, horizontal_rect)
        pygame.draw.rect(surface, self.ROAD_COLOR, vertical_rect)

        intersection_rect = self._intersection_rect(intersection)
        pygame.draw.rect(surface, self.ROAD_COLOR, intersection_rect)

    def draw_lane_markings(self, surface: pygame.Surface, intersection: Any) -> None:
        lanes_h, lanes_v = self._lane_counts(intersection)
        horizontal_road_width = max(2, lanes_h) * self.LANE_WIDTH
        vertical_road_width = max(2, lanes_v) * self.LANE_WIDTH

        cx = int(self.center.x)
        cy = int(self.center.y)
        half_box = self._intersection_rect(intersection).width // 2

        # Horizontal road edges and center lines.
        y_top = cy - horizontal_road_width // 2
        y_bottom = cy + horizontal_road_width // 2
        pygame.draw.line(surface, self.LANE_EDGE_COLOR, (0, y_top), (self.width, y_top), 2)
        pygame.draw.line(surface, self.LANE_EDGE_COLOR, (0, y_bottom), (self.width, y_bottom), 2)
        self._draw_dashed_line(surface, self.LANE_DASH_COLOR, (0, cy), (cx - half_box, cy), 2)
        self._draw_dashed_line(
            surface, self.LANE_DASH_COLOR, (cx + half_box, cy), (self.width, cy), 2
        )
        for lane_i in range(1, max(2, lanes_h)):
            y = y_top + lane_i * self.LANE_WIDTH
            self._draw_dashed_line(surface, self.LANE_DASH_COLOR, (0, y), (cx - half_box, y), 1)
            self._draw_dashed_line(
                surface, self.LANE_DASH_COLOR, (cx + half_box, y), (self.width, y), 1
            )

        # Vertical road edges and center lines.
        x_left = cx - vertical_road_width // 2
        x_right = cx + vertical_road_width // 2
        pygame.draw.line(surface, self.LANE_EDGE_COLOR, (x_left, 0), (x_left, self.height), 2)
        pygame.draw.line(surface, self.LANE_EDGE_COLOR, (x_right, 0), (x_right, self.height), 2)
        self._draw_dashed_line(surface, self.LANE_DASH_COLOR, (cx, 0), (cx, cy - half_box), 2)
        self._draw_dashed_line(
            surface, self.LANE_DASH_COLOR, (cx, cy + half_box), (cx, self.height), 2
        )
        for lane_i in range(1, max(2, lanes_v)):
            x = x_left + lane_i * self.LANE_WIDTH
            self._draw_dashed_line(surface, self.LANE_DASH_COLOR, (x, 0), (x, cy - half_box), 1)
            self._draw_dashed_line(
                surface, self.LANE_DASH_COLOR, (x, cy + half_box), (x, self.height), 1
            )



    def draw_signs(self, surface: pygame.Surface, intersection: Any) -> None:
        signs = self._extract_signs(intersection)
        if not signs:
            signs = {"N": "STOP", "S": "STOP", "E": "YIELD", "W": "YIELD"}

        sign_positions = self._right_side_sign_positions(intersection)

        for key, sign_type in signs.items():
            approach = key.upper()[:1]
            pos = sign_positions.get(approach)
            if pos is None:
                continue
            normalized = str(sign_type).strip().upper()
            if normalized == "STOP":
                self._draw_stop_sign(surface, pos)
            elif normalized == "YIELD":
                self._draw_yield_sign(surface, pos)
            elif normalized in {"PRIORITY", "PRIORITY_ROAD"}:
                self._draw_priority_sign(surface, pos)

    # --- intersection geometry helpers -------------------------------- #

    def _intersection_rect(self, intersection: Any) -> pygame.Rect:
        size = self._get(
            intersection, "box_size", "intersection_size", default=self.INTERSECTION_BOX_SIZE
        )
        try:
            size_px = int(size)
        except (TypeError, ValueError):
            size_px = self.INTERSECTION_BOX_SIZE
        size_px = max(60, size_px)
        return pygame.Rect(
            int(self.center.x - size_px / 2),
            int(self.center.y - size_px / 2),
            size_px,
            size_px,
        )

    def _lane_counts(self, intersection: Any) -> Tuple[int, int]:
        lane_count = self._get(intersection, "lane_count", "lanes", default=2)
        if isinstance(lane_count, Mapping):
            horizontal = self._as_int(lane_count.get("horizontal"), default=2)
            vertical = self._as_int(lane_count.get("vertical"), default=2)
            return max(1, horizontal), max(1, vertical)
        if (
            isinstance(lane_count, Sequence)
            and not isinstance(lane_count, (str, bytes))
            and len(lane_count) >= 2
        ):
            return max(1, self._as_int(lane_count[0], 2)), max(1, self._as_int(lane_count[1], 2))
        count = max(1, self._as_int(lane_count, default=2))
        return count, count

    def _extract_signs(self, intersection: Any) -> Dict[str, str]:
        signs = self._get(intersection, "signs", "traffic_signs", default={})
        if isinstance(signs, Mapping):
            result: Dict[str, str] = {}
            for key, value in signs.items():
                result[str(key).upper()[:1]] = str(value)
            return result
        return {}

    def _right_side_sign_positions(self, intersection: Any) -> Dict[str, pygame.Vector2]:
        rect = self._intersection_rect(intersection)
        lanes_h, lanes_v = self._lane_counts(intersection)
        horizontal_road_width = max(2, lanes_h) * self.LANE_WIDTH
        vertical_road_width = max(2, lanes_v) * self.LANE_WIDTH
        side_offset = 18
        front_offset = 26

        return {
            "N": pygame.Vector2(
                self.center.x - vertical_road_width * 0.5 - side_offset,
                rect.top - front_offset,
            ),
            "S": pygame.Vector2(
                self.center.x + vertical_road_width * 0.5 + side_offset,
                rect.bottom + front_offset,
            ),
            "E": pygame.Vector2(
                rect.right + front_offset,
                self.center.y - horizontal_road_width * 0.5 - side_offset,
            ),
            "W": pygame.Vector2(
                rect.left - front_offset,
                self.center.y + horizontal_road_width * 0.5 + side_offset,
            ),
        }

    # --- primitive drawing helpers ------------------------------------ #

    def _draw_direction_arrow(
        self,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        direction: pygame.Vector2,
        color: ColorRGB,
    ) -> None:
        if direction.length_squared() == 0:
            return
        vec = direction.normalize()
        length = 20
        tip = pos + vec * length
        base = pos - vec * (length * 0.45)
        perp = pygame.Vector2(-vec.y, vec.x)
        wing1 = tip - vec * 8 + perp * 5
        wing2 = tip - vec * 8 - perp * 5

        pygame.draw.line(
            surface, color, (int(base.x), int(base.y)), (int(tip.x), int(tip.y)), 2
        )
        pygame.draw.polygon(
            surface,
            color,
            [
                (int(tip.x), int(tip.y)),
                (int(wing1.x), int(wing1.y)),
                (int(wing2.x), int(wing2.y)),
            ],
        )

    @staticmethod
    def _draw_dashed_line(
        surface: pygame.Surface,
        color: ColorRGB,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width: int = 1,
        dash: int = 10,
        gap: int = 7,
    ) -> None:
        start_v = pygame.Vector2(start)
        end_v = pygame.Vector2(end)
        line_vec = end_v - start_v
        length = line_vec.length()
        if length == 0:
            return
        direction = line_vec / length
        step = dash + gap
        traveled = 0.0
        while traveled < length:
            dash_start = start_v + direction * traveled
            dash_end = start_v + direction * min(length, traveled + dash)
            pygame.draw.line(
                surface,
                color,
                (int(dash_start.x), int(dash_start.y)),
                (int(dash_end.x), int(dash_end.y)),
                width,
            )
            traveled += step

    @staticmethod
    def _draw_dashed_circle(
        surface: pygame.Surface,
        center: Tuple[int, int],
        radius: int,
        color: ColorRGBA,
        width: int = 1,
        dash_angle_deg: float = 11.0,
        gap_angle_deg: float = 8.0,
    ) -> None:
        if radius <= 0:
            return
        rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        angle = 0.0
        while angle < 360.0:
            start_rad = math.radians(angle)
            end_rad = math.radians(min(360.0, angle + dash_angle_deg))
            pygame.draw.arc(surface, color, rect, start_rad, end_rad, width)
            angle += dash_angle_deg + gap_angle_deg

    def _draw_stop_sign(self, surface: pygame.Surface, pos: pygame.Vector2) -> None:
        radius = 9
        points = []
        for i in range(8):
            angle = math.radians(22.5 + i * 45.0)
            points.append((pos.x + radius * math.cos(angle), pos.y + radius * math.sin(angle)))
        pygame.draw.polygon(surface, (185, 45, 45), points)
        pygame.draw.polygon(surface, (225, 225, 225), points, width=1)
        if self.font_tiny is not None:
            text = self.font_tiny.render("STOP", True, (245, 245, 245))
            surface.blit(text, text.get_rect(center=(int(pos.x), int(pos.y))))

    @staticmethod
    def _draw_yield_sign(surface: pygame.Surface, pos: pygame.Vector2) -> None:
        radius = 11
        points = [
            (pos.x, pos.y - radius),
            (pos.x - radius, pos.y + radius * 0.8),
            (pos.x + radius, pos.y + radius * 0.8),
        ]
        pygame.draw.polygon(surface, (238, 238, 238), points, width=1)

    @staticmethod
    def _draw_priority_sign(surface: pygame.Surface, pos: pygame.Vector2) -> None:
        radius = 9
        points = [
            (pos.x, pos.y - radius),
            (pos.x + radius, pos.y),
            (pos.x, pos.y + radius),
            (pos.x - radius, pos.y),
        ]
        pygame.draw.polygon(surface, (240, 185, 72), points)
        pygame.draw.polygon(surface, (60, 60, 60), points, width=1)
