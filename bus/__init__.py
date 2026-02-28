"""
bus \u2014 In-memory V2X messaging infrastructure
=============================================

Provides a lightweight pub/sub transport layer with optional packet-loss
and latency simulation, suitable for prototyping V2V and V2I workflows
without a real network stack.

Modules
-------
message
    :class:`V2XMessage` dataclass.
v2x_bus
    :class:`V2XBus` publish / poll / ack transport.
metrics
    :class:`BusMetrics` counter snapshot.
utils
    ID generation, latency sleep, fault injection.
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
