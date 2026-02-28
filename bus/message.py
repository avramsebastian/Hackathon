#!/usr/bin/env python3

from dataclasses import dataclass

@dataclass
class V2XMessage:
    id: str
    topic: str          # "v2v.state", "v2i.state", "i2v.command"
    sender: str         # "car_1", "tl_1"
    payload: dict
    ts: float
    require_ack: bool = False
