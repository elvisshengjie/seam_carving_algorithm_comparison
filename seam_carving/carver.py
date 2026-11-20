"""Shared resizing pipeline: recompute energy and remove one seam per step."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable
import numpy as np

from .energy import compute_energy, validate_image
from .seams import ALGORITHMS, DIRECTIONS, Seam, find_seam
from .control import RunControl


@dataclass(frozen=True)
class Step:
    number: int
    direction: str
    width_before: int
    height_before: int
    seam_cost: float
    energy_seconds: float
    find_seconds: float
    remove_seconds: float


@dataclass
class ResizeResult:
    image: np.ndarray
    steps: list[Step]
    elapsed_seconds: float


def validate_seam(shape: tuple, seam: Seam) -> None:
    if seam.direction not in DIRECTIONS:
        raise ValueError("Unknown seam direction.")
    height, width = shape[:2]
    length, limit = (height, width) if seam.direction == "vertical" else (width, height)
    indices = np.asarray(seam.indices)
    if indices.shape != (length,) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("A seam must contain one integer index per traversed row/column.")
    if (indices < 0).any() or (indices >= limit).any():
        raise ValueError("Seam index lies outside the image.")
    if (np.abs(np.diff(indices)) > 1).any():
        raise ValueError("Adjacent seam indices must differ by at most one.")


def remove_seam(image: np.ndarray, seam: Seam) -> np.ndarray:
    image = validate_image(image)
    validate_seam(image.shape, seam)
    oriented = image if seam.direction == "vertical" else image.swapaxes(0, 1)
    height, width = oriented.shape[:2]
    if width <= 1:
        raise ValueError("Cannot remove the last row or column.")
    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam.indices] = False
    removed = oriented[keep].reshape(height, width - 1, 3)
    return removed if seam.direction == "vertical" else removed.swapaxes(0, 1).copy()


def validate_target(image: np.ndarray, width: int, height: int) -> tuple[int, int]:
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("Target dimensions must be integers.")
    width, height = int(width), int(height)
    if not 1 <= width <= image.shape[1] or not 1 <= height <= image.shape[0]:
        raise ValueError("Targets must be positive and no larger than the input image.")
    return width, height


def resize(image: np.ndarray, width: int, height: int, algorithm: str = "dp",
           order: str = "width-first",
           callback: Callable[[np.ndarray, Seam, Step], None] | None = None,
           *, control: RunControl | None = None) -> ResizeResult:
    """Shrink to width x height. Callback sees each seam BEFORE removal.

    Callback arrays are borrowed, read-only-by-contract; copy if retaining them.
    The reported elapsed time includes callback work. Benchmarks omit callbacks.
    This project removes seams; enlargement by insertion is not implemented.
    Optional control reports computation stages and cancels inside seam solving.
    """
    image = validate_image(image)
    width, height = validate_target(image, width, height)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithm must be one of {ALGORITHMS}.")
    if order not in ("width-first", "height-first"):
        raise ValueError("Order must be width-first or height-first.")
    control = control or RunControl()
    # Seams to remove in each direction: vertical seams reduce width, horizontal reduce height.
    plan = [("vertical", image.shape[1] - width), ("horizontal", image.shape[0] - height)]
    if order == "height-first":
        plan.reverse()
    total_steps = sum(count for _, count in plan)
    current = image.copy()
    steps: list[Step] = []
    started = perf_counter()
    control.checkpoint()
    for direction, count in plan:
        for _ in range(count):
            control.checkpoint(f"Seam {len(steps) + 1}/{total_steps}: calculating energy...")
            t0 = perf_counter()
            energy = compute_energy(current)  # recomputed on the current image every step
            t1 = perf_counter()
            seam = find_seam(energy, algorithm, direction, control=control)
            t2 = perf_counter()
            control.checkpoint()
            next_image = remove_seam(current, seam)
            t3 = perf_counter()
            step = Step(len(steps) + 1, direction, current.shape[1], current.shape[0],
                        seam.cost, t1 - t0, t2 - t1, t3 - t2)
            steps.append(step)
            if callback is not None:
                callback(current, seam, step)
            current = next_image
    return ResizeResult(current, steps, perf_counter() - started)
