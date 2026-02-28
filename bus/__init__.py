"""
V2XBus package: Provides in-memory V2X messaging infrastructure.

Modules:
    - message: V2XMessage dataclass
    - v2x_bus: V2XBus transport class
    - metrics: BusMetrics for tracking statistics
    - utils: helper functions (ID generation, latency, packet faults)
"""

from .message import V2XMessage
from .v2x_bus import V2XBus
from .metrics import BusMetrics
from .utils   import new_msg_id, simulate_latency, maybe_drop, maybe_corrupt

__all__ = [
    "V2XMessage",
    "V2XBus",
    "BusMetrics",
    "new_msg_id",
    "simulate_latency",
    "maybe_drop",
    "maybe_corrupt",
]
