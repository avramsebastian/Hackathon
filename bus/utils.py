"""
Utility functions for V2XBus:
    - ID generation
    - latency simulation
    - fault injection (packet drop, payload corruption)
"""

import uuid
import time
import random
import logging

log = logging.getLogger(__name__)

# ---------- ID Helpers ----------
def new_msg_id() -> str:
    """
    Generate a globally unique message ID.

    Returns:
        str: UUID string for a new message.
    """
    return str(uuid.uuid4())

# ---------- Latency / Timing ----------
def simulate_latency(ms: int):
    """
    Simulate network latency by sleeping for a given number of milliseconds.

    Args:
        ms (int): Number of milliseconds to sleep.
    """
    if ms > 0:
        time.sleep(ms / 1000.0)

# ---------- Fault / Packet Helpers ----------
def maybe_drop(drop_rate: float) -> bool:
    """
    Decide whether to randomly drop a packet based on the drop rate.

    Args:
        drop_rate (float): Probability (0.0–1.0) that the packet will be dropped.

    Returns:
        bool: True if the packet should be dropped, False otherwise.
    """
    if drop_rate <= 0.0:
        return False
    result = random.random() < drop_rate
    if result:
        log.debug("Packet dropped by utils.maybe_drop")
    return result

def maybe_corrupt(payload: dict, corruption_rate: float = 0.0) -> dict:
    """
    Randomly corrupt a payload for demonstration purposes.

    Args:
        payload (dict): Original message payload.
        corruption_rate (float): Probability (0.0–1.0) to corrupt the payload.

    Returns:
        dict: Original or modified payload with '_corrupted' flag if corrupted.
    """
    if random.random() < corruption_rate:
        payload = payload.copy()
        payload["_corrupted"] = True
        log.debug("Payload corrupted by utils.maybe_corrupt")
    return payload
