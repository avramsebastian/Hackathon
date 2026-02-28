"""
ml/entities/Directions.py
=========================
Enum for vehicle intended manoeuvre.
"""

from enum import Enum


class Directions(Enum):
    """Intended manoeuvre at the intersection."""
    LEFT = 0
    RIGHT = 1
    FORWARD = 2