"""
sim/network.py
==============
Road-network topology for multi-intersection simulations.

Defines :class:`IntersectionNode`, :class:`RoadSegment`, and
:class:`RoadNetwork` — a lightweight graph that locates intersections
in world space and records which arms connect to each other.

The default layout is an **L-shape** of three intersections:

.. code-block:: text

   INT_A (−150, 0)  ──── road ────  INT_B (0, 0)
       semaphore                      signs │
                                            │ road
                                            │
                                      INT_C (0, −150)
                                        semaphore

:func:`default_network` builds this layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Intersection node ─────────────────────────────────────────────────────────

@dataclass
class IntersectionNode:
    """A single crossroads in the network.

    Parameters
    ----------
    id : str
        Unique identifier (e.g. ``"INT_A"``).
    cx, cy : float
        Intersection centre in world-space metres.
    has_semaphore : bool
        When *True* the intersection uses a traffic-light cycle;
        otherwise it uses signs (STOP / YIELD / PRIORITY).
    priority_axis : str
        ``"EW"`` or ``"NS"`` — determines sign layout and initial
        semaphore green axis.
    """

    id: str
    cx: float
    cy: float
    has_semaphore: bool = True
    priority_axis: str = "EW"

    # Runtime semaphore state (mutated by World)
    sem_green_axis: str = ""
    sem_phase: str = "GREEN"
    sem_timer: float = 0.0

    def __post_init__(self) -> None:
        if not self.sem_green_axis:
            self.sem_green_axis = self.priority_axis


# ── Road segment ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoadSegment:
    """An undirected road connecting two intersection arms.

    ``from_id``/``from_arm`` is one end;
    ``to_id``/``to_arm`` is the other.
    Cars exiting ``from_id`` via ``from_arm`` arrive at ``to_id``
    on ``to_arm``.
    """

    from_id: str
    from_arm: str          # "N" / "S" / "E" / "W"
    to_id: str
    to_arm: str


# ── Road network ──────────────────────────────────────────────────────────────

class RoadNetwork:
    """Graph of intersections connected by road segments.

    Provides helpers used by :class:`~sim.world.World` and the UI:

    * **terminal_arms** — arms that are *not* connected to another
      intersection (cars spawn and despawn here).
    * **connected_arm** — follow a road from one intersection to
      another.
    * **arm_world_endpoint** — world-coordinate at the far end of a
      terminal arm (for spawn positioning and road rendering).
    """

    def __init__(
        self,
        intersections: List[IntersectionNode],
        roads: List[RoadSegment],
        *,
        road_half_w: float = 10.0,
        terminal_arm_length: float = 80.0,
    ) -> None:
        self.intersections: Dict[str, IntersectionNode] = {
            n.id: n for n in intersections
        }
        self.roads = list(roads)
        self.road_half_w = road_half_w
        self.terminal_arm_length = terminal_arm_length

        # Pre-compute connection lookup:  (int_id, arm) → (other_id, other_arm)
        self._connections: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for road in self.roads:
            self._connections[(road.from_id, road.from_arm)] = (road.to_id, road.to_arm)
            self._connections[(road.to_id, road.to_arm)] = (road.from_id, road.from_arm)

    # ── queries ───────────────────────────────────────────────────────────

    def connected_arm(
        self, int_id: str, arm: str,
    ) -> Optional[Tuple[str, str]]:
        """If *arm* of *int_id* connects to another intersection,
        return ``(other_id, other_arm)``; else ``None``."""
        return self._connections.get((int_id, arm))

    def is_terminal(self, int_id: str, arm: str) -> bool:
        return (int_id, arm) not in self._connections

    def terminal_arms(self) -> List[Tuple[str, str]]:
        """Return ``[(int_id, arm), ...]`` for every arm that is a dead end."""
        result: List[Tuple[str, str]] = []
        for node in self.intersections.values():
            for arm in ("N", "S", "E", "W"):
                if self.is_terminal(node.id, arm):
                    result.append((node.id, arm))
        return result

    def arm_direction_vector(self, arm: str) -> Tuple[float, float]:
        """Unit velocity vector for a car *entering* along this arm
        (i.e. heading toward the intersection centre)."""
        return {
            "W": (1.0, 0.0),
            "E": (-1.0, 0.0),
            "N": (0.0, -1.0),
            "S": (0.0, 1.0),
        }[arm]

    def arm_spawn_position(
        self,
        int_id: str,
        arm: str,
        distance: float,
        lane_offset: float = 7.0,
    ) -> Tuple[float, float, float, float]:
        """World (x, y, vx, vy) for a car spawning on a terminal arm.

        *distance* is how far from the intersection centre the car starts.
        """
        node = self.intersections[int_id]
        vx, vy = self.arm_direction_vector(arm)
        # Position along the arm axis, offset to the correct lane
        if arm == "W":
            return (node.cx - distance, node.cy - lane_offset, vx, vy)
        elif arm == "E":
            return (node.cx + distance, node.cy + lane_offset, vx, vy)
        elif arm == "N":
            return (node.cx - lane_offset, node.cy + distance, vx, vy)
        else:  # S
            return (node.cx + lane_offset, node.cy - distance, vx, vy)

    def arm_exit_position(
        self,
        int_id: str,
        arm: str,
        lane_offset: float = 7.0,
    ) -> Tuple[float, float]:
        """World (x, y) far endpoint when *exiting* ``int_id`` via ``arm``.

        For terminal arms this is the despawn line.
        For connected arms it is the stop line of the receiving intersection.
        """
        conn = self.connected_arm(int_id, arm)
        if conn:
            other_id, _other_arm = conn
            other = self.intersections[other_id]
            return (other.cx, other.cy)
        # Terminal → just go far out
        node = self.intersections[int_id]
        length = self.terminal_arm_length
        if arm == "E":
            return (node.cx + length, node.cy)
        elif arm == "W":
            return (node.cx - length, node.cy)
        elif arm == "N":
            return (node.cx, node.cy + length)
        else:
            return (node.cx, node.cy - length)


# ── Default layout ────────────────────────────────────────────────────────────

_SEP = 150.0  # centre-to-centre distance between intersections


def default_network() -> RoadNetwork:
    """Build the standard L-shaped 3-intersection layout.

    .. code-block:: text

       INT_A (−150, 0) ── road ── INT_B (0, 0)
           semaphore                 signs │
                                          │ road
                                          │
                                    INT_C (0, −150)
                                      semaphore
    """
    nodes = [
        IntersectionNode(
            id="INT_A",
            cx=-_SEP,
            cy=0.0,
            has_semaphore=True,
            priority_axis="EW",
        ),
        IntersectionNode(
            id="INT_B",
            cx=0.0,
            cy=0.0,
            has_semaphore=False,       # signs only
            priority_axis="EW",
        ),
        IntersectionNode(
            id="INT_C",
            cx=0.0,
            cy=-_SEP,
            has_semaphore=True,
            priority_axis="NS",
        ),
    ]

    roads = [
        RoadSegment(from_id="INT_A", from_arm="E",
                    to_id="INT_B",   to_arm="W"),
        RoadSegment(from_id="INT_B", from_arm="S",
                    to_id="INT_C",   to_arm="N"),
    ]

    return RoadNetwork(nodes, roads)
