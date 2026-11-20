"""Dinic max-flow with explicit residual edges and iterative blocking-flow DFS.

No graph library or built-in minimum-cut routine is used. The iterative DFS
avoids Python's recursion limit on wide images and long residual paths.
"""

from collections import deque
from dataclasses import dataclass
import math

from .control import RunControl


@dataclass(slots=True)
class _Edge:
    to: int
    reverse: int  # index of the paired residual edge in graph[to]
    capacity: float


class Dinic:
    def __init__(self, nodes: int, control: RunControl | None = None):
        self.control = control or RunControl()
        self.graph: list[list[_Edge]] = []
        for node in range(nodes):
            if node % 1024 == 0:
                self.control.checkpoint("Preparing graph...")
            self.graph.append([])

    def add_edge(self, start: int, end: int, capacity: float) -> None:
        if start == end:
            raise ValueError("Self loops are unnecessary for this network.")
        if not math.isfinite(capacity) or capacity < 0:
            raise ValueError("Capacities must be finite and non-negative.")
        forward = _Edge(end, len(self.graph[end]), float(capacity))
        reverse = _Edge(start, len(self.graph[start]), 0.0)
        self.graph[start].append(forward)
        self.graph[end].append(reverse)

    def _levels(self, source: int) -> list[int]:
        """BFS distance from source in the residual graph (-1 = unreachable)."""
        levels = [-1] * len(self.graph)
        levels[source] = 0
        queue = deque([source])
        visited = 0
        while queue:
            if visited % 1024 == 0:
                self.control.checkpoint("Searching graph paths...")
            visited += 1
            node = queue.popleft()
            for edge in self.graph[node]:
                if edge.capacity > 0 and levels[edge.to] < 0:
                    levels[edge.to] = levels[node] + 1
                    queue.append(edge.to)
        return levels

    def _augment(self, source: int, sink: int, levels: list[int], current: list[int]) -> float:
        """Push flow along one source->sink path of the level graph; 0 if none is left.

        current[node] is the next edge to try, so exhausted edges are never rescanned.
        """
        path: list[tuple[int, int]] = []  # (node, edge index) from the source
        node = source
        work = 0
        while True:
            if work % 1024 == 0:
                self.control.checkpoint("Augmenting graph paths...")
            work += 1
            if node == sink:
                edges = [self.graph[start][index] for start, index in path]
                amount = min(edge.capacity for edge in edges)
                for edge in edges:
                    edge.capacity -= amount
                    self.graph[edge.to][edge.reverse].capacity += amount
                return amount
            edges = self.graph[node]
            while current[node] < len(edges):
                edge = edges[current[node]]
                if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                    break
                current[node] += 1
            if current[node] < len(edges):  # advance one level
                path.append((node, current[node]))
                node = edges[current[node]].to
            elif path:  # dead end: step back and skip the edge that led here
                node, _ = path.pop()
                current[node] += 1
            else:
                return 0.0

    def max_flow(self, source: int, sink: int) -> float:
        if source == sink:
            raise ValueError("Source and sink must differ.")
        total = 0.0
        phase = 0
        while True:
            phase += 1
            self.control.checkpoint(f"Solving minimum cut: phase {phase}...")
            levels = self._levels(source)
            if levels[sink] < 0:
                return total
            current = [0] * len(self.graph)
            while amount := self._augment(source, sink, levels, current):
                total += amount

    def source_side(self, source: int) -> set[int]:
        """Reachability in the final residual graph defines the minimum cut."""
        return {node for node, level in enumerate(self._levels(source)) if level >= 0}
