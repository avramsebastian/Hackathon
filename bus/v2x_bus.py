"""
V2XBus: In-memory pub/sub system for V2V and V2I communication.

Supports:
    - Topic-based messaging
    - Optional ACKs with timeout tracking
    - Packet drop and latency simulation
    - Logging of events

Intended usage:
    - Cars publish state to 'v2v.state' or 'v2i.state'
    - Infrastructure polls 'v2i.state' and responds via 'i2v.command'
"""

import time
import uuid
import random
import logging
from typing import Dict, List, Optional
from .message import V2XMessage

log = logging.getLogger(__name__)


class V2XBus:
    """
    Transport layer for vehicle-to-everything (V2X) messages.

    Attributes:
        drop_rate (float): Probability of randomly dropping a packet.
        latency_ms (int): Simulated latency in milliseconds.
    """

    def __init__(self, drop_rate: float = 0.0, latency_ms: int = 0):
        """
        Initialize a V2XBus instance.

        Args:
            drop_rate (float): Chance of randomly dropping a message (0.0 to 1.0).
            latency_ms (int): Optional simulated latency in milliseconds for published messages.
        """
        self._topics: Dict[str, List[V2XMessage]] = {}
        self._pending_ack: Dict[str, float] = {}
        self.drop_rate = drop_rate
        self.latency_ms = latency_ms

    def publish(
        self,
        topic: str,
        sender: str,
        payload: dict,
        require_ack: bool = False,
    ) -> Optional[str]:
        """
        Publish a message to a specific topic.

        Args:
            topic (str): The topic name (e.g., 'v2v.state', 'v2i.state', 'i2v.command').
            sender (str): ID of the sender (e.g., 'car_1', 'tl_1').
            payload (dict): Arbitrary data dictionary representing the message contents.
            require_ack (bool): If True, the message expects an ACK from the receiver.

        Returns:
            Optional[str]: The unique message ID if successfully published, or None if dropped.
        """
        if self._maybe_drop():
            log.warning("packet_dropped topic=%s sender=%s", topic, sender)
            return None

        msg_id = str(uuid.uuid4())
        msg = V2XMessage(
            id=msg_id,
            topic=topic,
            sender=sender,
            payload=payload,
            ts=time.time(),
            require_ack=require_ack,
        )
        self._topics.setdefault(topic, []).append(msg)

        if require_ack:
            self._pending_ack[msg_id] = msg.ts

        log.info("publish topic=%s sender=%s id=%s", topic, sender, msg_id)
        return msg_id

    def poll(self, topic: str) -> List[V2XMessage]:
        """
        Retrieve and clear all messages from a given topic.

        Args:
            topic (str): The topic name to poll messages from.

        Returns:
            List[V2XMessage]: List of messages published to the topic since the last poll.
        """
        msgs = self._topics.get(topic, [])
        self._topics[topic] = []
        return msgs

    def ack(self, msg_id: str):
        """
        Mark a previously published message as acknowledged.

        Args:
            msg_id (str): The ID of the message being acknowledged.

        Side Effects:
            Removes the message from the pending ACK tracking dictionary.
        """
        if msg_id in self._pending_ack:
            self._pending_ack.pop(msg_id, None)
            log.info("ack_received id=%s", msg_id)

    def pending_acks(self, timeout_s: float = 1.0) -> List[str]:
        """
        Retrieve and remove any messages whose ACKs have timed out.

        Args:
            timeout_s (float): Time in seconds before a pending ACK is considered expired.

        Returns:
            List[str]: List of message IDs that timed out waiting for an ACK.

        Side Effects:
            Removes expired message IDs from pending ACK tracking.
        """
        now = time.time()
        expired = [
            msg_id
            for msg_id, ts in self._pending_ack.items()
            if now - ts > timeout_s
        ]
        for msg_id in expired:
            log.warning("ack_timeout id=%s", msg_id)
            self._pending_ack.pop(msg_id, None)
        return expired

    def _maybe_drop(self) -> bool:
        """
        Decide whether to randomly drop a message based on drop_rate.

        Returns:
            bool: True if the message should be dropped, False otherwise.

        Note:
            This is an internal helper method used by publish().
        """
        return random.random() < self.drop_rate
