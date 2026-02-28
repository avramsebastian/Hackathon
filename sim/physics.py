#!/usr/bin/env python3
"""
sim/physics.py
==============
Low-level physics helpers used by :mod:`sim.world` and :mod:`sim.traffic_policy`.

Keeping these in a separate module avoids circular imports and makes unit
testing straightforward.
"""

from __future__ import annotations

import math


def kmh_to_mps(speed_kmh: float) -> float:
    """Convert km/h to m/s, clamping negatives to zero."""
    return max(0.0, float(speed_kmh)) / 3.6


def mps_to_kmh(speed_mps: float) -> float:
    """Convert m/s to km/h."""
    return max(0.0, float(speed_mps)) * 3.6


def braking_distance(speed_kmh: float, decel_kmh_s: float) -> float:
    """Stopping distance assuming constant deceleration.

    Parameters
    ----------
    speed_kmh : float
        Current speed in km/h.
    decel_kmh_s : float
        Deceleration rate in km/h per second.

    Returns
    -------
    float
        Distance in metres needed to reach zero speed.
    """
    v = kmh_to_mps(speed_kmh)
    a = max(0.1, decel_kmh_s / 3.6)  # m/s²
    return (v * v) / (2.0 * a)


def axial_distance(x: float, y: float, vx: float, vy: float, offset: float) -> float:
    """Signed distance from *(x, y)* to a line perpendicular to travel at *offset* from the origin.

    Positive → the point has not yet reached the line.
    Zero / negative → the point has passed it.

    Parameters
    ----------
    x, y : float
        Position in world coordinates.
    vx, vy : float
        Unit velocity components (+1/−1/0).
    offset : float
        Distance of the line from the origin along the travel axis.
    """
    if vx > 0:       # heading east  → line at x = −offset
        return -x - offset
    elif vx < 0:     # heading west  → line at x = +offset
        return x - offset
    elif vy < 0:     # heading south → line at y = +offset
        return y - offset
    elif vy > 0:     # heading north → line at y = −offset
        return -y - offset
    return math.hypot(x, y)
