#!/usr/bin/env python3
"""
Traffic policy primitives for safety distance and priority scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafetyPolicy:
    lane_offset_m: float = 7.0
    pass_threshold_m: float = 10.0
    coast_decel_kmh_s: float = 60.0
    max_accel_kmh_s: float = 18.0
    max_brake_kmh_s: float = 90.0

    # Safety envelope.
    base_collision_radius_m: float = 2.2
    reaction_time_s: float = 0.55
    horizon_s: float = 0.65
    min_pair_distance_m: float = 4.8
    max_pair_distance_m: float = 11.0

    # ML decision gating.
    ml_stop_soft_radius_m: float = 30.0
    ml_stop_hard_radius_m: float = 14.0
    ml_stop_soft_speed_kmh: float = 24.0

    # Priority model.
    speed_limit_kmh: float = 42.0

    # Virtual signal scheduler (extensible to real V2I semaphores).
    signal_window_s: float = 2.6
    signal_switch_margin: float = 0.45
    signal_control_radius_m: float = 36.0
    red_soft_speed_kmh: float = 8.0
    red_hard_radius_m: float = 18.0

    # Spawn envelope.
    spawn_min_radius_m: float = 70.0
    spawn_max_radius_m: float = 140.0
    spawn_min_gap_m: float = 18.0
    spawn_max_attempts: int = 300


_ROLE_WEIGHT = {
    "civilian": 0.0,
    "bus": 0.4,
    "taxi": 0.2,
    "police": 2.0,
    "ambulance": 2.5,
    "fire": 2.3,
}


def to_mps(speed_kmh: float) -> float:
    return max(0.0, float(speed_kmh)) / 3.6


def braking_distance_m(speed_kmh: float, max_brake_kmh_s: float) -> float:
    """
    Stopping distance with constant deceleration.
    """
    v = to_mps(speed_kmh)
    decel_mps2 = max(0.1, max_brake_kmh_s / 3.6)
    return (v * v) / (2.0 * decel_mps2)


def pair_safe_distance_m(car_a: Any, car_b: Any, policy: SafetyPolicy) -> float:
    """
    Dynamic pair distance based on speed and braking ability.
    """
    va = to_mps(getattr(car_a, "speed", 0.0))
    vb = to_mps(getattr(car_b, "speed", 0.0))
    reaction = (va + vb) * policy.reaction_time_s * 0.5
    braking = (
        braking_distance_m(getattr(car_a, "speed", 0.0), policy.max_brake_kmh_s)
        + braking_distance_m(getattr(car_b, "speed", 0.0), policy.max_brake_kmh_s)
    ) * 0.5
    base = max(policy.min_pair_distance_m, policy.base_collision_radius_m * 2.0)
    safe = base + reaction + 0.35 * braking
    return max(policy.min_pair_distance_m, min(policy.max_pair_distance_m, safe))


def danger_score(car: Any, policy: SafetyPolicy) -> float:
    """
    Higher score => should receive higher scheduling priority.
    """
    speed = max(0.0, float(getattr(car, "speed", 0.0)))
    speed_limit = max(1.0, float(getattr(car, "speed_limit_kmh", policy.speed_limit_kmh)))
    speed_factor = min(2.0, speed / speed_limit)
    overspeed = max(0.0, speed - speed_limit) / speed_limit
    stop_dist = braking_distance_m(speed, policy.max_brake_kmh_s)
    wait_s = max(0.0, float(getattr(car, "wait_s", 0.0)))
    role = str(getattr(car, "role", "civilian")).lower()
    role_bonus = _ROLE_WEIGHT.get(role, 0.0)
    return (
        role_bonus
        + speed_factor
        + (1.2 * overspeed)
        + min(2.0, stop_dist / 35.0)
        + min(2.0, wait_s / 6.0)
    )
