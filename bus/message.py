"""
V2XMessage: Data structure representing a message transmitted over the V2XBus.
"""

from dataclasses import dataclass

@dataclass
class V2XMessage:
    """
    Represents a single message sent via the V2XBus.

    Attributes:
        id (str): Unique identifier for the message.
        topic (str): The topic of the message (e.g., 'v2v.state', 'v2i.state', 'i2v.command').
        sender (str): ID of the sender (e.g., 'car_1', 'tl_1').
        payload (dict): Arbitrary dictionary containing message contents.
        ts (float): Timestamp (in seconds) when the message was created.
        require_ack (bool): If True, the message expects an acknowledgment from the receiver.
    """
    id: str
    topic: str
    sender: str
    payload: dict
    ts: float
    require_ack: bool = False
