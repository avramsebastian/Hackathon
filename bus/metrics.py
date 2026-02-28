"""
BusMetrics: Tracks simple statistics for V2XBus message flow.
"""

class BusMetrics:
    """
    Tracks metrics for published messages, drops, ACKs, and timeouts.

    Attributes:
        published (int): Total number of messages successfully published.
        dropped (int): Number of messages dropped due to simulated faults.
        acked (int): Number of messages acknowledged.
        ack_timeouts (int): Number of messages that timed out waiting for ACK.
    """

    def __init__(self):
        """Initialize all counters to zero."""
        self.published = 0
        self.dropped = 0
        self.acked = 0
        self.ack_timeouts = 0

    def report(self) -> dict:
        """
        Return a snapshot of current metrics.

        Returns:
            dict: Dictionary containing 'published', 'dropped', 'acked', and 'ack_timeouts' counters.
        """
        return {
            "published": self.published,
            "dropped": self.dropped,
            "acked": self.acked,
            "ack_timeouts": self.ack_timeouts
        }
