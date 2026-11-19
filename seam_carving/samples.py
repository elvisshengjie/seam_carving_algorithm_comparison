"""Reproducible synthetic inputs; no downloads or external image licences."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def create_samples(directory: str | Path, width: int = 96, height: int = 64) -> list[Path]:
    if width < 32 or height < 24:
        raise ValueError("Demo samples must be at least 32 x 24.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    # Low-detail background with two foreground objects.
    y = np.linspace(0, 1, height)[:, None, None]
    sky = np.array([65, 155, 225])[None, None, :] * (1 - y)
    sky += np.array([190, 220, 238])[None, None, :] * y
    background = np.repeat(sky, width, axis=1).astype(np.uint8)
    landscape = Image.fromarray(background)
    draw = ImageDraw.Draw(landscape)
    draw.rectangle((0, height * 3 // 4, width, height), fill=(80, 135, 55))
    draw.rectangle((width * 5 // 8, height // 4, width * 3 // 4, height * 4 // 5), fill=(175, 110, 60))
    draw.rectangle((width * 5 // 8 + 3, height // 3, width * 3 // 4 - 3, height * 3 // 4), fill=(45, 48, 55))
    draw.ellipse((width // 5, height // 2, width // 5 + 6, height // 2 + 6), fill=(35, 35, 45))
    draw.line((width // 5 + 3, height // 2 + 6, width // 5 + 3, height * 4 // 5), fill=(35, 35, 45), width=3)
    path = directory / "landscape.png"
    landscape.save(path)
    paths.append(path)
    # Structure across most columns: a more difficult image for seam carving.
    grid = Image.new("RGB", (width, height), (235, 228, 205))
    draw = ImageDraw.Draw(grid)
    for x in range(0, width, 8):
        draw.line((x, 0, x, height), fill=(35, 65, 95), width=2)
    for yy in range(0, height, 8):
        draw.line((0, yy, width, yy), fill=(35, 65, 95), width=2)
    draw.ellipse((width // 3, height // 4, width * 2 // 3, height * 3 // 4), fill=(215, 75, 65))
    path = directory / "structures.png"
    grid.save(path)
    paths.append(path)
    # Highly textured input where no clearly unimportant region exists.
    noise = np.random.default_rng(2026).integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = directory / "texture.png"
    Image.fromarray(noise).save(path)
    paths.append(path)
    return paths
