"""Sobel gradient energy, implemented with NumPy rather than a resizing library."""

import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """Accept uint8 RGB arrays. File loading converts other formats to RGB."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected a non-empty H x W x 3 RGB image.")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB values in [0, 255].")
    return image


def compute_energy(image: np.ndarray) -> np.ndarray:
    """Return float64 Sobel magnitude of grayscale luminance, shape H x W.

    Edge padding replicates boundary pixels. Converting before subtraction
    prevents uint8 wraparound. A constant image has zero energy, including edges.
    """
    image = validate_image(image)
    gray = image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    p = np.pad(gray, 1, mode="edge")
    gx = ((p[:-2, 2:] - p[:-2, :-2]) + 2 * (p[1:-1, 2:] - p[1:-1, :-2])
          + (p[2:, 2:] - p[2:, :-2]))
    gy = ((p[2:, :-2] - p[:-2, :-2]) + 2 * (p[2:, 1:-1] - p[:-2, 1:-1])
          + (p[2:, 2:] - p[:-2, 2:]))
    return np.hypot(gx, gy)
