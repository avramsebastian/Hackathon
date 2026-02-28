#!/usr/bin/env python3

import uuid
import time
import random
import logging

log = logging.getLogger(__name__)

# ---------- ID Helpers ----------
def new_msg_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())

# ---------- Latency / Timing ----------
def simulate_latency(ms: int):
    """Simulate network latency in milliseconds."""
    if ms > 0:
        time.sleep(ms / 1000.0)

# ---------- Fault / Packet Helpers ----------
def maybe_drop(drop_rate: float) -> bool:
    """Randomly decide whether to drop a packet."""
    if drop_rate <= 0.0:
        return False
    result = random.random() < drop_rate
    if result:
        log.debug("Packet dropped by utils.maybe_drop")
    return result

def maybe_corrupt(payload: dict, corruption_rate: float = 0.0) -> dict:
    """
    Randomly corrupt a payload for demonstration purposes.
    Currently adds a flag; can be expanded for real corruption.
    """
    if random.random() < corruption_rate:
        payload = payload.copy()
        payload["_corrupted"] = True
        log.debug("Payload corrupted by utils.maybe_corrupt")
    return payload
