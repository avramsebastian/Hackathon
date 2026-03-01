#!/usr/bin/env python3
"""
sim/traffic_policy.py
=====================
Tunable safety, physics and scheduling parameters for the intersection
simulation.  Every constant lives in the frozen :class:`SafetyPolicy`
dataclass so that experiments can swap policies without touching code.

Also provides three stateless scoring / distance helpers:

* :func:`danger_score` — scheduling priority for a vehicle.
* :func:`pair_safe_distance_m` — dynamic minimum pair distance.
* :func:`braking_distance_m` — constant-deceleration stopping distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafetyPolicy:
    """Immutable bag of every tunable simulation parameter.

    Groups: geometry, longitudinal control, safety envelope,
    stop-sign enforcement, virtual signal scheduler, collision
    guard / resolver, spawn envelope.
    """

    # ── Geometry ──────────────────────────────────────────────────────────
    lane_offset_m: float = 7.0
    """Perpendicular offset of the lane centre from the road axis."""

    pass_threshold_m: float = 0.0
    """Distance past the origin at which a car is considered *through*."""

    # ── Longitudinal control ──────────────────────────────────────────────
    coast_decel_kmh_s: float = 120.0
    """Deceleration (km/h per s) applied after all cars have passed."""

    max_accel_kmh_s: float = 60.0
    """Maximum acceleration rate (km/h per s)."""

    max_brake_kmh_s: float = 180.0
    """Maximum braking rate (km/h per s)."""

    green_launch_accel_kmh_s: float = 120.0
    """Aggressive acceleration when launching from a green light with clear road."""

    creep_filter_kmh: float = 3.0
    """Speeds below this threshold are snapped to zero (anti-creep filter)."""

    # ── ML-first longitudinal control ─────────────────────────────────────
    ml_stop_target_speed_kmh: float = 0.0
    """Target speed when the ML model says STOP (no explicit target)."""

    ml_max_target_speed_kmh: float = 120.0
    """Hard clamp on any explicit ``target_speed_kmh`` from ML."""

    # ── Priority / speed limit ────────────────────────────────────────────
    speed_limit_kmh: float = 84.0
    """Posted speed limit for scoring and spawn range."""

    # ── Safety envelope ───────────────────────────────────────────────────
    base_collision_radius_m: float = 2.2
    """Minimum physical clearance around each vehicle."""

    reaction_time_s: float = 0.55
    """Assumed reaction time for dynamic safe-distance calculation."""

    horizon_s: float = 1.2
    """Look-ahead horizon for predicted collision guard."""

    min_pair_distance_m: float = 4.8
    """Lower clamp on pair safe distance."""

    max_pair_distance_m: float = 11.0
    """Upper clamp on pair safe distance."""

    # ── Stop-sign enforcement ─────────────────────────────────────────────
    stop_sign_wait_s: float = 0.2
    """Mandatory stop duration at a STOP sign (seconds)."""

    stop_line_offset_m: float = 10.5
    """Distance from the intersection centre to the stop line.

    Must be close to ``ROAD_HALF_W`` so cars stop near the
    traffic-light / sign rendering position.
    """

    stop_brake_zone_m: float = 25.0
    """Start decelerating this far before the stop line."""

    # ── Virtual signal scheduler (optional) ───────────────────────────────
    world_signal_scheduler_enabled: bool = False
    """Enable the world-level green-phase scheduler."""

    signal_window_s: float = 2.6
    """Minimum green phase duration (seconds)."""

    signal_switch_margin: float = 0.45
    """Score margin required to switch the green approach."""

    signal_control_radius_m: float = 36.0
    """Radius within which cars are affected by the virtual signal."""

    red_soft_speed_kmh: float = 8.0
    """Speed cap for 'red' cars outside the hard-stop zone."""

    red_hard_radius_m: float = 18.0
    """Radius within which 'red' cars must fully stop."""

    # ── Collision guard / overlap resolver ─────────────────────────────────
    world_collision_guard_enabled: bool = True
    """Enable pair-wise collision guard."""

    world_overlap_resolver_enabled: bool = True
    """Last-resort push-apart when two cars physically overlap."""

    collision_guard_soft_speed_kmh: float = 10.0
    """Speed cap applied by the collision guard for far conflicts."""

    # ── Spawn envelope ────────────────────────────────────────────────────
    spawn_min_radius_m: float = 70.0
    """Closest spawn distance from the intersection centre."""

    spawn_max_radius_m: float = 140.0
    """Farthest spawn distance from the intersection centre."""

    spawn_min_gap_m: float = 18.0
    """Minimum clearance between any two spawned cars."""

    spawn_max_attempts: int = 300
    """Random-placement attempts before deterministic fallback."""

    # ── Semaphore (traffic light) ─────────────────────────────────────────
    semaphore_enabled: bool = True
    """Enable traffic-light control; overrides sign-based enforcement."""

    semaphore_green_s: float = 8.0
    """Duration of the green phase per axis (seconds)."""

    semaphore_yellow_s: float = 2.0
    """Duration of the yellow phase (seconds)."""

    semaphore_red_clearance_s: float = 1.0
    """All-red clearance interval between phases (seconds)."""

    semaphore_brake_zone_m: float = 30.0
    """Distance before stop line at which a red/yellow light triggers braking."""

    semaphore_stop_line_m: float = 10.5
    """Distance of the semaphore stop line from intersection centre."""


_ROLE_WEIGHT = {
    "civilian": 0.0,
    "bus": 0.4,
    "taxi": 0.2,
    "police": 2.0,
    "ambulance": 2.5,
    "fire": 2.3,
}


def to_mps(speed_kmh: float) -> float:
    """Convert km/h to m/s, clamping negatives to zero."""
    return max(0.0, float(speed_kmh)) / 3.6


def braking_distance_m(speed_kmh: float, max_brake_kmh_s: float) -> float:
    """Stopping distance (m) with constant deceleration.

    Parameters
    ----------
    speed_kmh : float
        Current speed in km/h.
    max_brake_kmh_s : float
        Braking deceleration in km/h per second.
    """
    v = to_mps(speed_kmh)
    decel_mps2 = max(0.1, max_brake_kmh_s / 3.6)
    return (v * v) / (2.0 * decel_mps2)


def pair_safe_distance_m(car_a: Any, car_b: Any, policy: SafetyPolicy) -> float:
    """Dynamic minimum pair distance based on speed and braking ability.

    Combines a fixed base radius, reaction-time buffer and
    braking-distance estimate for both vehicles.
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
    """Scheduling-priority score for *car*.

    Higher score ⇒ higher priority (should get right of way).
    Factors: role weight, normalised speed, overspeed penalty,
    stopping distance, accumulated wait time.
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
