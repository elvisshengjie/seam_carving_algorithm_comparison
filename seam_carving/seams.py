"""Minimum seam selection: rolling-row DP, local greedy, and s-t graph cut."""

from dataclasses import dataclass
import numpy as np

from .flow import Dinic
from .control import RunControl

ALGORITHMS = ("dp", "greedy", "graph")
DIRECTIONS = ("vertical", "horizontal")


@dataclass(frozen=True)
class Seam:
    # Vertical: column per row. Horizontal: row per column.
    indices: np.ndarray
    cost: float
    algorithm: str
    direction: str


def _validate_energy(energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 2 or min(energy.shape) < 1:
        raise ValueError("Energy must be a non-empty two-dimensional matrix.")
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Energy values must be finite and non-negative.")
    # A representable total is needed by the DP and capacity calculations.
    if not np.isfinite(energy.sum()):
        raise ValueError("Total energy exceeds float64 range.")
    return energy


def _dp(energy: np.ndarray, control: RunControl) -> np.ndarray:
    height, width = energy.shape
    previous = energy[0].copy()
    # Only the previous cost row is needed, but reconstruction needs parents.
    parents = np.zeros((height, width), dtype=np.int8)
    columns = np.arange(width)
    for row in range(1, height):
        control.checkpoint()
        padded = np.pad(previous, (1, 1), constant_values=np.inf)
        candidates = np.stack((padded[:-2], padded[1:-1], padded[2:]))
        choices = np.argmin(candidates, axis=0)
        parents[row] = choices - 1
        previous = energy[row] + candidates[choices, columns]
    seam = np.empty(height, dtype=np.int64)
    seam[-1] = int(np.argmin(previous))
    for row in range(height - 1, 0, -1):
        seam[row - 1] = seam[row] + int(parents[row, seam[row]])
    return seam


def _greedy(energy: np.ndarray, control: RunControl) -> np.ndarray:
    height, width = energy.shape
    seam = np.empty(height, dtype=np.int64)
    seam[0] = int(np.argmin(energy[0]))
    for row in range(1, height):
        control.checkpoint()
        left = max(0, int(seam[row - 1]) - 1)
        right = min(width, int(seam[row - 1]) + 2)
        seam[row] = left + int(np.argmin(energy[row, left:right]))
    return seam


def _graph_cut(energy: np.ndarray, control: RunControl) -> np.ndarray:
    """Encode each possible seam pixel as a finite-capacity cut edge.

    Row r has boundary nodes b=0..width. Edge (r,b)->(r,b+1) costs
    energy[r,b]. Force b=0 onto the source side and b=width onto the sink
    side. Infinite backward edges make each source-side row a prefix.
    Infinite diagonals (r,b)->(r+/-1,b-1) enforce adjacent seam columns.

    The extra boundary after the last pixel makes that pixel removable too;
    directly forcing the last *pixel* to the sink would exclude it.
    """
    height, width = energy.shape
    stride = width + 1
    source = height * stride
    sink = source + 1
    network = Dinic(sink + 1, control)
    # Rescale to control magnitudes; positive scaling preserves optimal cuts.
    largest = float(energy.max())
    costs = energy / largest if largest > 0 else energy
    # Every all-finite cut costs <= height*width; this dominates that bound.
    infinite = float(height * width + 1)
    for row in range(height):
        control.checkpoint(f"Building graph: row {row + 1}/{height}...")
        base = row * stride
        network.add_edge(source, base, infinite)
        network.add_edge(base + width, sink, infinite)
        for col in range(width):
            network.add_edge(base + col, base + col + 1, float(costs[row, col]))
            network.add_edge(base + col + 1, base + col, infinite)
        if row + 1 < height:
            below = base + stride
            for boundary in range(1, width + 1):
                network.add_edge(base + boundary, below + boundary - 1, infinite)
                network.add_edge(below + boundary, base + boundary - 1, infinite)
    network.max_flow(source, sink)
    reachable = network.source_side(source)
    seam = np.empty(height, dtype=np.int64)
    control.checkpoint("Reading minimum cut...")
    for row in range(height):
        seam[row] = max(col for col in range(width) if row * stride + col in reachable)
    return seam


def find_seam(energy: np.ndarray, algorithm: str = "dp",
              direction: str = "vertical", *, control: RunControl | None = None) -> Seam:
    """Find one seam without modifying energy. DP and graph are globally optimal
    for this fixed energy map; greedy is not. Equal-cost paths may differ.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm must be one of {ALGORITHMS}.")
    if direction not in DIRECTIONS:
        raise ValueError(f"Direction must be one of {DIRECTIONS}.")
    control = control or RunControl()
    control.checkpoint()
    energy = _validate_energy(energy)
    oriented = energy if direction == "vertical" else energy.T
    solver = {"dp": _dp, "greedy": _greedy, "graph": _graph_cut}[algorithm]
    indices = solver(oriented, control)
    cost = float(oriented[np.arange(oriented.shape[0]), indices].sum())
    return Seam(indices, cost, algorithm, direction)
