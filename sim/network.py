"""
sim/network.py
==============
Road-network topology for multi-intersection simulations.

Defines :class:`IntersectionNode`, :class:`RoadSegment`, and
:class:`RoadNetwork` — a lightweight graph that locates intersections
in world space and records which arms connect to each other.

:func:`default_network` builds a random grid of intersections.
"""

from __future__ import annotations

import math
import random
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

    def get_bounds(
        self,
        screen_width: float = 1280.0,
        screen_height: float = 720.0,
        base_margin: float = 50.0,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((min_x, max_x), (min_y, max_y)) world bounds of the network.
        
        Dynamically extends bounds in narrow dimensions to match screen aspect ratio.
        Base margin (50m) is less than terminal arm length (80m) so roads always
        extend beyond the visible bounds, hiding road endings.
        """
        if not self.intersections:
            return ((-100.0, 100.0), (-100.0, 100.0))
        
        min_x = min(n.cx for n in self.intersections.values())
        max_x = max(n.cx for n in self.intersections.values())
        min_y = min(n.cy for n in self.intersections.values())
        max_y = max(n.cy for n in self.intersections.values())
        
        # Current network dimensions (just intersections)
        net_width = max_x - min_x if max_x > min_x else 1.0
        net_height = max_y - min_y if max_y > min_y else 1.0
        
        # Screen aspect ratio
        screen_aspect = screen_width / screen_height
        net_aspect = net_width / net_height
        
        # Extend margins to match screen aspect ratio
        if net_aspect < screen_aspect:
            # Network is too narrow horizontally - extend X margins
            target_width = net_height * screen_aspect
            extra_margin_x = (target_width - net_width) / 2.0
            margin_x = base_margin + extra_margin_x
            margin_y = base_margin
        else:
            # Network is too short vertically - extend Y margins
            target_height = net_width / screen_aspect
            extra_margin_y = (target_height - net_height) / 2.0
            margin_x = base_margin
            margin_y = base_margin + extra_margin_y
        
        return (
            (min_x - margin_x, max_x + margin_x),
            (min_y - margin_y, max_y + margin_y),
        )
    
    def get_grid_info(self) -> Tuple[int, int, int]:
        """Return (num_intersections, grid_cols, grid_rows)."""
        if not self.intersections:
            return (0, 0, 0)
        
        xs = sorted(set(n.cx for n in self.intersections.values()))
        ys = sorted(set(n.cy for n in self.intersections.values()))
        return (len(self.intersections), len(xs), len(ys))


# ── Default layout ────────────────────────────────────────────────────────────

_SEP = 150.0  # centre-to-centre distance between intersections


def default_network(seed: Optional[int] = None) -> RoadNetwork:
    """Build a random grid of intersections.

    Generates a 2x2 to 3x3 grid with some intersections randomly removed,
    ensuring all remaining intersections are connected. If the layout would
    leave dangling roads (like an L-shape), the missing corner is filled in.
    
    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """
    rng = random.Random(seed)
    
    # Random grid size: 2x2, 2x3, 3x2, or 3x3
    rows = rng.randint(2, 3)
    cols = rng.randint(2, 3)
    
    # Create all possible grid positions
    # Grid is centered around (0, 0)
    offset_x = -(cols - 1) * _SEP / 2
    offset_y = -(rows - 1) * _SEP / 2
    
    all_positions = []
    for r in range(rows):
        for c in range(cols):
            cx = offset_x + c * _SEP
            cy = offset_y + r * _SEP
            all_positions.append((r, c, cx, cy))
    
    # Decide how many intersections to keep (at least 2, at most all)
    min_count = max(2, rows * cols - 2)  # Remove at most 2
    max_count = rows * cols
    count = rng.randint(min_count, max_count)
    
    # If we're removing some, do it randomly but ensure connectivity
    if count < len(all_positions):
        # Start with all positions in a set
        grid = {(r, c): (cx, cy) for r, c, cx, cy in all_positions}
        
        # Try to remove some while maintaining connectivity
        positions_to_try_remove = list(grid.keys())
        rng.shuffle(positions_to_try_remove)
        
        for pos in positions_to_try_remove:
            if len(grid) <= count:
                break
            # Check if removing this would break connectivity
            test_grid = dict(grid)
            del test_grid[pos]
            if _is_connected(test_grid):
                del grid[pos]
        
        selected = [(r, c, cx, cy) for (r, c), (cx, cy) in grid.items()]
    else:
        selected = all_positions
    
    # Check for L-shape or incomplete corners and fill them in
    selected = _fill_missing_corners(selected, rows, cols, offset_x, offset_y)
    
    # Ensure even number of intersections
    if len(selected) % 2 != 0:
        # Try to add another intersection - find an empty adjacent slot
        positions = {(r, c) for r, c, _, _ in selected}
        added = False
        for r, c, _, _ in list(selected):
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in positions:
                    nx = offset_x + nc * _SEP
                    ny = offset_y + nr * _SEP
                    selected.append((nr, nc, nx, ny))
                    positions.add((nr, nc))
                    added = True
                    break
            if added:
                break
        
        # If couldn't add (full grid), remove a corner to make even
        if not added and len(selected) > 2:
            # Remove a corner intersection that maintains connectivity
            corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
            for corner in corners:
                if corner in positions:
                    # Check if removing maintains connectivity
                    test_positions = positions - {corner}
                    test_grid = {p: (0, 0) for p in test_positions}
                    if _is_connected(test_grid):
                        selected = [(r, c, cx, cy) for r, c, cx, cy in selected if (r, c) != corner]
                        break
    
    # Create nodes
    nodes = []
    node_map = {}  # (row, col) -> node_id
    for i, (r, c, cx, cy) in enumerate(selected):
        node_id = f"INT_{chr(65 + i)}"  # INT_A, INT_B, etc.
        # Alternate between semaphore and signs, with random priority axis
        has_sem = (i % 2 == 0) if len(selected) > 2 else True
        priority = rng.choice(["EW", "NS"])
        nodes.append(IntersectionNode(
            id=node_id,
            cx=cx,
            cy=cy,
            has_semaphore=has_sem,
            priority_axis=priority,
        ))
        node_map[(r, c)] = node_id
    
    # Create roads between adjacent intersections
    roads = []
    for (r, c), node_id in node_map.items():
        # Connect to the east neighbor
        if (r, c + 1) in node_map:
            roads.append(RoadSegment(
                from_id=node_id,
                from_arm="E",
                to_id=node_map[(r, c + 1)],
                to_arm="W",
            ))
        # Connect to the north neighbor
        if (r + 1, c) in node_map:
            roads.append(RoadSegment(
                from_id=node_id,
                from_arm="N",
                to_id=node_map[(r + 1, c)],
                to_arm="S",
            ))
    
    return RoadNetwork(nodes, roads)


def _is_connected(grid: Dict[Tuple[int, int], Tuple[float, float]]) -> bool:
    """Check if all positions in the grid are connected via adjacency."""
    if len(grid) <= 1:
        return True
    
    positions = set(grid.keys())
    start = next(iter(positions))
    visited = set()
    queue = [start]
    
    while queue:
        pos = queue.pop(0)
        if pos in visited:
            continue
        visited.add(pos)
        r, c = pos
        # Check all 4 neighbors
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if (nr, nc) in positions and (nr, nc) not in visited:
                queue.append((nr, nc))
    
    return len(visited) == len(positions)


def _fill_missing_corners(
    selected: List[Tuple[int, int, float, float]],
    rows: int,
    cols: int,
    offset_x: float,
    offset_y: float,
) -> List[Tuple[int, int, float, float]]:
    """Fill in missing corners to avoid L-shapes with dangling roads.
    
    If we have an L-shape (3 intersections forming an L), add the 4th corner.
    """
    positions = {(r, c) for r, c, _, _ in selected}
    
    # For each pair of adjacent intersections that share a potential corner,
    # check if adding that corner would complete a square
    new_positions = list(selected)
    
    for r, c, cx, cy in selected:
        # Check if we have neighbors that would benefit from a corner fill
        has_east = (r, c + 1) in positions
        has_north = (r + 1, c) in positions
        has_west = (r, c - 1) in positions
        has_south = (r - 1, c) in positions
        
        # Check diagonal corners
        corners_to_check = []
        if has_east and has_north and (r + 1, c + 1) not in positions:
            corners_to_check.append((r + 1, c + 1))
        if has_west and has_north and (r + 1, c - 1) not in positions:
            corners_to_check.append((r + 1, c - 1))
        if has_east and has_south and (r - 1, c + 1) not in positions:
            corners_to_check.append((r - 1, c + 1))
        if has_west and has_south and (r - 1, c - 1) not in positions:
            corners_to_check.append((r - 1, c - 1))
        
        for corner_r, corner_c in corners_to_check:
            if 0 <= corner_r < rows and 0 <= corner_c < cols:
                if (corner_r, corner_c) not in positions:
                    corner_cx = offset_x + corner_c * _SEP
                    corner_cy = offset_y + corner_r * _SEP
                    new_positions.append((corner_r, corner_c, corner_cx, corner_cy))
                    positions.add((corner_r, corner_c))
    
    return new_positions
