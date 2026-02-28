#!/usr/bin/env python3

import time
import uuid
import random
import logging
from typing import Dict, List, Optional
from .message import V2XMessage

log = logging.getLogger(__name__)

class V2XBus:
    def __init__(self, drop_rate: float = 0.0, latency_ms: int = 0):
        self._topics: Dict[str, List[V2XMessage]] = {}
        self._pending_ack: Dict[str, float] = {}
        self.drop_rate = drop_rate
        self.latency_ms = latency_ms

    def publish(self, topic: str, sender: str, payload: dict, require_ack: bool = False) -> Optional[str]:
        if self._maybe_drop():
            log.warning("packet_dropped topic=%s sender=%s", topic, sender)
            return None

        msg_id = str(uuid.uuid4())
        msg = V2XMessage(id=msg_id, topic=topic, sender=sender, payload=payload, ts=time.time(), require_ack=require_ack)
        self._topics.setdefault(topic, []).append(msg)

        if require_ack:
            self._pending_ack[msg_id] = msg.ts

        log.info("publish topic=%s sender=%s id=%s", topic, sender, msg_id)
        return msg_id

    def poll(self, topic: str) -> List[V2XMessage]:
        msgs = self._topics.get(topic, [])
        self._topics[topic] = []
        return msgs

    def ack(self, msg_id: str):
        if msg_id in self._pending_ack:
            self._pending_ack.pop(msg_id, None)
            log.info("ack_received id=%s", msg_id)

    def pending_acks(self, timeout_s: float = 1.0) -> List[str]:
        now = time.time()
        expired = [msg_id for msg_id, ts in self._pending_ack.items() if now - ts > timeout_s]
        for msg_id in expired:
            log.warning("ack_timeout id=%s", msg_id)
        return expired

    def _maybe_drop(self) -> bool:
        return random.random() < self.drop_rate
