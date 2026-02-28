"""
ml/entities/Sign.py
===================
Enum for traffic sign types.
"""

from enum import Enum


class Sign(Enum):
    """Traffic sign visible to the ego vehicle."""
    STOP = 0
    YIELD = 1
    PRIORITY = 2
    NO_SIGN = 3