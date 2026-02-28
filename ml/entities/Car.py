"""
ml/entities/Car.py
==================
Lightweight car representation used by the ML feature-extraction pipeline.

Not to be confused with :class:`sim.world.Car` which carries full simulation
state.  This class only stores the fields needed for feature computation.
"""

import math


class Car:
    """A vehicle with position, direction and speed.

    Parameters
    ----------
    x, y : float
        World-space coordinates (metres from intersection centre).
    direction : Directions enum or float
        Travel direction enum value.
    speed : float
        Current speed in km/h.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0,
                 direction=0.0, speed: float = 0.0):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed

    def distance_to_center(self) -> float:
        """Euclidean distance from the intersection centre."""
        return math.hypot(self.x, self.y)
