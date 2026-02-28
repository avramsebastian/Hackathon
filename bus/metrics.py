#!/usr/bin/env python3

class BusMetrics:
    def __init__(self):
        self.published = 0
        self.dropped = 0
        self.acked = 0
        self.ack_timeouts = 0

    def report(self):
        return {
            "published": self.published,
            "dropped": self.dropped,
            "acked": self.acked,
            "ack_timeouts": self.ack_timeouts
        }
