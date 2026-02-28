#!/usr/bin/env python3
"""Vehicle sprite rendering, detection zones, animation, and state sync (mixin)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pygame

from .types import ColorRGB, ColorRGBA, VehicleRenderState


class VehicleRenderer:
    """Mixin that draws vehicles, detection zones, conflict highlights, and handles animation."""

    # ------------------------------------------------------------------ #
    #  Public draw methods                                                 #
    # ------------------------------------------------------------------ #

    def draw_vehicle(self, surface: pygame.Surface, vehicle: Any) -> None:
        vehicle_id = self._vehicle_id(vehicle)
        state = self.vehicle_states.get(vehicle_id)
        if state is None:
            return

        w, h = 26, 13
        sprite = pygame.Surface((w, h), pygame.SRCALPHA)

        # Body
        body = pygame.Rect(0, 0, w, h)
        pygame.draw.rect(sprite, state.color, body, border_radius=3)

        # Windshield
        ws_rect = pygame.Rect(w - 10, 2, 7, h - 4)
        r, g, b = state.color
        glass = (max(0, r - 60), max(0, g - 60), max(0, b - 60), 180)
        pygame.draw.rect(sprite, glass, ws_rect, border_radius=2)

        # Headlights
        hl_color = (255, 248, 200)
        pygame.draw.circle(sprite, hl_color, (w - 2, 3), 2)
        pygame.draw.circle(sprite, hl_color, (w - 2, h - 3), 2)

        # Taillights
        tl_color = (200, 40, 40)
        pygame.draw.circle(sprite, tl_color, (2, 3), 2)
        pygame.draw.circle(sprite, tl_color, (2, h - 3), 2)

        # Border
        pygame.draw.rect(sprite, (235, 235, 235), body, width=1, border_radius=3)

        direction = state.travel_vec if state.travel_vec.length_squared() > 0 else pygame.Vector2(1, 0)
        angle = -math.degrees(math.atan2(direction.y, direction.x))
        rotated = pygame.transform.rotate(sprite, angle)
        dest = rotated.get_rect(center=(int(state.render_pos.x), int(state.render_pos.y)))
        surface.blit(rotated, dest)

    def draw_road_line(self, surface: pygame.Surface, vehicle: Any) -> None:
        vehicle_id = self._vehicle_id(vehicle)
        state = self.vehicle_states.get(vehicle_id)
        if state is None:
            return
        if len(state.path_points) < 2:
            return

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        color = (*state.color, self.ROAD_LINE_ALPHA)
        points = [(int(p.x), int(p.y)) for p in state.path_points]
        pygame.draw.lines(overlay, color, False, points, 4)
        surface.blit(overlay, (0, 0))

    def draw_detection_zone(self, surface: pygame.Surface, vehicle: Any, ml_decision: Any) -> None:
        vehicle_id = self._vehicle_id(vehicle)
        state = self.vehicle_states.get(vehicle_id)
        if state is None:
            return

        approach_vec = self._infer_approach_vector(vehicle, state.render_pos)
        if state.approach_vec.length_squared() > 0:
            approach_vec = state.approach_vec
        circle_center = self.center + approach_vec * self.DETECTION_DISTANCE_PX
        distance = state.render_pos.distance_to(circle_center)
        state.in_detection = distance <= self.DETECTION_RADIUS_PX

        decision = self._normalize_decision(ml_decision)
        if decision == "stop":
            target = (*self.STOP_COLOR, self.ZONE_STOP_ALPHA)
        elif state.in_detection and decision == "go":
            target = (*self.GO_COLOR, self.ZONE_GO_ALPHA)
        else:
            target = (0, 0, 0, 0)
        self._update_zone_transition(state, target)

        if state.zone_current[3] > 0:
            lane_end = circle_center
            if decision == "stop":
                # STOP should paint lane segment from current car position to the intersection.
                to_vehicle = state.render_pos - self.center
                if to_vehicle.length_squared() > 0 and to_vehicle.dot(approach_vec) > 0:
                    lane_end = state.render_pos
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.line(
                overlay,
                state.zone_current,
                (int(self.center.x), int(self.center.y)),
                (int(lane_end.x), int(lane_end.y)),
                self.LANE_WIDTH + 8,
            )
            surface.blit(overlay, (0, 0))

    def draw_conflict_highlight(self, surface: pygame.Surface, intersection: Any, tick: float) -> None:
        alpha = self.CONFLICT_ALPHA_MIN + (
            (math.sin(2 * math.pi * tick) + 1.0) * 0.5
            * (self.CONFLICT_ALPHA_MAX - self.CONFLICT_ALPHA_MIN)
        )

        glow_rect = self._intersection_rect(intersection).inflate(28, 28)
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(
            overlay,
            (*self.CONFLICT_COLOR, int(alpha)),
            glow_rect,
            border_radius=10,
        )
        surface.blit(overlay, (0, 0))

    # ------------------------------------------------------------------ #
    #  Animation                                                           #
    # ------------------------------------------------------------------ #

    def animate_vehicle(self, vehicle: Any, ml_decision: Any, delta_time: float) -> None:
        vehicle_id = self._vehicle_id(vehicle)
        state = self.vehicle_states.get(vehicle_id)
        if state is None:
            return

        decision = self._normalize_decision(ml_decision)
        target_scale = 0.35 if decision == "stop" else 1.0
        blend = min(1.0, delta_time * 6.0)
        state.speed_scale_current += (target_scale - state.speed_scale_current) * blend

        speed_px = self._speed_mps(vehicle) * self.PIXELS_PER_METER * state.speed_scale_current
        if len(state.path_points) >= 2 and state.path_index < len(state.path_points):
            target = state.path_points[state.path_index]
            to_target = target - state.render_pos
            distance = to_target.length()
            step = max(0.0, speed_px * delta_time)
            if distance > 0 and step > 0:
                if distance <= step:
                    state.render_pos = target
                    if state.path_index < len(state.path_points) - 1:
                        state.path_index += 1
                        next_vec = state.path_points[state.path_index] - state.render_pos
                        if next_vec.length_squared() > 0:
                            state.travel_vec = next_vec.normalize()
                else:
                    movement = to_target.normalize() * step
                    state.render_pos += movement
                    state.travel_vec = movement.normalize()
        else:
            direction = self._infer_travel_vector(vehicle, state.render_pos)
            state.render_pos += direction * speed_px * delta_time
            state.travel_vec = direction

        reported_position = self._extract_vehicle_position(vehicle)
        if reported_position is not None:
            state.render_pos = state.render_pos.lerp(reported_position, 0.20)

    # ------------------------------------------------------------------ #
    #  Vehicle state management                                            #
    # ------------------------------------------------------------------ #

    def _sync_vehicle_states(self, vehicles: Sequence[Any]) -> None:
        active_ids = set()
        for index, vehicle in enumerate(vehicles):
            vehicle_id = self._vehicle_id(vehicle)
            active_ids.add(vehicle_id)

            color = self._vehicle_color(vehicle, index)
            start_pos = self._extract_vehicle_position(vehicle)
            if start_pos is None:
                start_pos = self._fallback_start_position(vehicle, index)

            if vehicle_id not in self.vehicle_states:
                state = VehicleRenderState(
                    color=color,
                    render_pos=start_pos,
                    approach_vec=self._infer_approach_vector(vehicle, start_pos),
                )
                self.vehicle_states[vehicle_id] = state
            else:
                state = self.vehicle_states[vehicle_id]
                state.color = color
                if state.render_pos.length_squared() == 0:
                    state.render_pos = start_pos
                if state.approach_vec.length_squared() == 0:
                    state.approach_vec = self._infer_approach_vector(vehicle, state.render_pos)

            state.path_points = self._extract_path_points(vehicle, state.approach_vec)
            if not state.path_points:
                state.path_points = [state.render_pos, self.center.copy()]
            else:
                entry_vec = state.path_points[0] - self.center
                if entry_vec.length_squared() > 0:
                    state.approach_vec = entry_vec.normalize()
            state.path_index = min(state.path_index, len(state.path_points) - 1)

        stale_ids = [vid for vid in self.vehicle_states if vid not in active_ids]
        for vid in stale_ids:
            del self.vehicle_states[vid]

    # ------------------------------------------------------------------ #
    #  Path / position helpers                                             #
    # ------------------------------------------------------------------ #

    def _extract_path_points(
        self, vehicle: Any, approach_vec: pygame.Vector2
    ) -> List[pygame.Vector2]:
        raw_path = self._get(vehicle, "road_line", "path", "desired_path")
        path_points: List[pygame.Vector2] = []
        if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
            for point in raw_path:
                parsed = self._point_to_screen(point)
                if parsed is not None:
                    path_points.append(parsed)
        if len(path_points) >= 2:
            return path_points

        approach = (
            approach_vec
            if approach_vec.length_squared() > 0
            else self._infer_approach_from_field(vehicle)
        )
        if approach.length_squared() == 0:
            approach = pygame.Vector2(0, -1)
        long_span = max(self.width, self.height) * 0.65
        start = self.center + approach * long_span

        # Generate turn paths for LEFT / RIGHT directions
        direction = self._get(vehicle, "direction")
        if isinstance(direction, str):
            rel_dir = direction.strip().upper()
            heading = -approach.normalize() if approach.length_squared() > 0 else pygame.Vector2(1, 0)
            if rel_dir == "LEFT":
                exit_vec = pygame.Vector2(heading.y, -heading.x)
                end = self.center + exit_vec * long_span
                return [start, self.center.copy(), end]
            elif rel_dir == "RIGHT":
                exit_vec = pygame.Vector2(-heading.y, heading.x)
                end = self.center + exit_vec * long_span
                return [start, self.center.copy(), end]

        # FORWARD or default: straight through
        end = self.center - approach * long_span
        return [start, self.center.copy(), end]

    def _fallback_start_position(self, vehicle: Any, index: int) -> pygame.Vector2:
        approach = self._infer_approach_from_field(vehicle)
        if approach.length_squared() == 0:
            cycle = [
                pygame.Vector2(0, -1),
                pygame.Vector2(1, 0),
                pygame.Vector2(0, 1),
                pygame.Vector2(-1, 0),
            ]
            approach = cycle[index % len(cycle)]
        return self.center + approach * (self.DETECTION_DISTANCE_PX + 90)

    def _extract_vehicle_position(self, vehicle: Any) -> Optional[pygame.Vector2]:
        position = self._get(vehicle, "position", "pos")
        if position is not None:
            point = self._point_to_screen(position)
            if point is not None:
                return point

        x = self._get(vehicle, "x")
        y = self._get(vehicle, "y")
        if x is None or y is None:
            return None
        try:
            return self._to_screen_point(float(x), float(y))
        except (TypeError, ValueError):
            return None

    def _point_to_screen(self, point: Any) -> Optional[pygame.Vector2]:
        if isinstance(point, Mapping):
            x = point.get("x")
            y = point.get("y")
            if x is None or y is None:
                return None
            try:
                return self._to_screen_point(float(x), float(y))
            except (TypeError, ValueError):
                return None
        if (
            isinstance(point, Sequence)
            and not isinstance(point, (str, bytes))
            and len(point) >= 2
        ):
            try:
                return self._to_screen_point(float(point[0]), float(point[1]))
            except (TypeError, ValueError):
                return None
        return None

    def _to_screen_point(self, x: float, y: float) -> pygame.Vector2:
        if max(abs(x), abs(y)) <= 200.0:
            return pygame.Vector2(
                self.center.x + x * self.PIXELS_PER_METER,
                self.center.y - y * self.PIXELS_PER_METER,
            )
        if -1.0 <= x <= 1.0 and -1.0 <= y <= 1.0:
            return pygame.Vector2(
                self.center.x + x * self.width * 0.45,
                self.center.y - y * self.height * 0.45,
            )
        return pygame.Vector2(x, y)

    # ------------------------------------------------------------------ #
    #  Speed helpers                                                       #
    # ------------------------------------------------------------------ #

    def _speed_mps(self, vehicle: Any) -> float:
        raw = self._as_float(self._get(vehicle, "speed", "velocity", default=0.0))
        speed = abs(raw if raw is not None else 0.0)
        unit = str(self._get(vehicle, "speed_unit", "unit", default="kmh")).strip().lower()
        if unit in {"kmh", "kph", "km/h"}:
            return speed / 3.6
        if unit in {"mph"}:
            return speed * 0.44704
        if speed > 70:
            return speed / 3.6
        return speed

    @staticmethod
    def _speed_to_kmh(speed_mps: float) -> float:
        return speed_mps * 3.6

    # ------------------------------------------------------------------ #
    #  Direction inference                                                 #
    # ------------------------------------------------------------------ #

    def _infer_travel_vector(self, vehicle: Any, render_pos: pygame.Vector2) -> pygame.Vector2:
        heading = self._get(vehicle, "heading", "angle")
        if isinstance(heading, (int, float)):
            radians = math.radians(float(heading))
            vec = pygame.Vector2(math.cos(radians), -math.sin(radians))
            if vec.length_squared() > 0:
                return vec.normalize()

        direction = self._get(vehicle, "direction")
        if isinstance(direction, str):
            key = direction.strip().upper()

            # Relative directions: FORWARD / LEFT / RIGHT
            if key in {"FORWARD", "LEFT", "RIGHT"}:
                approach = self._infer_approach_vector(vehicle, render_pos)
                heading_vec = -approach
                if heading_vec.length_squared() == 0:
                    heading_vec = pygame.Vector2(1, 0)
                heading_vec = heading_vec.normalize()
                if key == "FORWARD":
                    return heading_vec
                elif key == "LEFT":
                    # 90° left turn in screen coords (Y-down)
                    return pygame.Vector2(heading_vec.y, -heading_vec.x)
                else:  # RIGHT
                    return pygame.Vector2(-heading_vec.y, heading_vec.x)

            # Cardinal directions
            if key in {"N", "NORTH", "SOUTHBOUND"}:
                return pygame.Vector2(0, 1)
            if key in {"S", "SOUTH", "NORTHBOUND"}:
                return pygame.Vector2(0, -1)
            if key in {"E", "EAST", "WESTBOUND"}:
                return pygame.Vector2(-1, 0)
            if key in {"W", "WEST", "EASTBOUND"}:
                return pygame.Vector2(1, 0)

        approach = self._infer_approach_vector(vehicle, render_pos)
        return -approach

    def _infer_approach_vector(self, vehicle: Any, render_pos: pygame.Vector2) -> pygame.Vector2:
        field_vec = self._infer_approach_from_field(vehicle)
        if field_vec.length_squared() > 0:
            return field_vec

        delta = render_pos - self.center
        if abs(delta.x) >= abs(delta.y):
            return pygame.Vector2(1, 0) if delta.x >= 0 else pygame.Vector2(-1, 0)
        return pygame.Vector2(0, 1) if delta.y >= 0 else pygame.Vector2(0, -1)

    def _infer_approach_from_field(self, vehicle: Any) -> pygame.Vector2:
        approach = self._get(vehicle, "approach", "entry", "from_direction")
        if not isinstance(approach, str):
            return pygame.Vector2()
        key = approach.strip().upper()
        if key in {"N", "NORTH"}:
            return pygame.Vector2(0, -1)
        if key in {"S", "SOUTH"}:
            return pygame.Vector2(0, 1)
        if key in {"E", "EAST"}:
            return pygame.Vector2(1, 0)
        if key in {"W", "WEST"}:
            return pygame.Vector2(-1, 0)
        return pygame.Vector2()

    # ------------------------------------------------------------------ #
    #  Zone transition                                                     #
    # ------------------------------------------------------------------ #

    def _update_zone_transition(self, state: VehicleRenderState, target: ColorRGBA) -> None:
        if target != state.zone_target:
            state.zone_from = state.zone_current
            state.zone_target = target
            state.zone_frame = 0

        if state.zone_frame >= self.ZONE_TRANSITION_FRAMES:
            state.zone_current = state.zone_target
            return

        t = min(1.0, (state.zone_frame + 1) / float(self.ZONE_TRANSITION_FRAMES))
        state.zone_current = self._lerp_color(state.zone_from, state.zone_target, t)
        state.zone_frame += 1

    @staticmethod
    def _lerp_color(a: ColorRGBA, b: ColorRGBA, t: float) -> ColorRGBA:
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
            int(a[3] + (b[3] - a[3]) * t),
        )

    # ------------------------------------------------------------------ #
    #  Vehicle identity / color                                            #
    # ------------------------------------------------------------------ #

    def _vehicle_id(self, vehicle: Any) -> str:
        identifier = self._get(vehicle, "id", "vehicle_id", "name")
        if identifier is None:
            return f"VEH-{id(vehicle)}"
        return str(identifier).upper()

    def _vehicle_color(self, vehicle: Any, index: int) -> ColorRGB:
        raw = self._get(vehicle, "color", "colour")
        parsed = self._parse_color(raw)
        if parsed is not None:
            return parsed
        return self.DEFAULT_VEHICLE_COLORS[index % len(self.DEFAULT_VEHICLE_COLORS)]
