"""Image I/O, seam overlays, fixed-size animation frames, and result panels."""

from dataclasses import asdict, fields
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .carver import ResizeResult, Step, validate_seam
from .seams import Seam


def load_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def draw_seam(image: np.ndarray, seam: Seam) -> np.ndarray:
    validate_seam(image.shape, seam)
    result = image.copy()
    if seam.direction == "vertical":
        result[np.arange(image.shape[0]), seam.indices] = [255, 30, 30]
    else:
        result[seam.indices, np.arange(image.shape[1])] = [255, 30, 30]
    return result


def energy_image(energy: np.ndarray) -> np.ndarray:
    maximum = float(energy.max())
    gray = np.zeros(energy.shape, dtype=np.uint8) if maximum == 0 else (
        energy / maximum * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def make_panel(items: list[tuple[str, np.ndarray]], columns: int = 3,
               cell_size: tuple[int, int] = (340, 260)) -> Image.Image:
    """Fit images without distortion; labels include actual pixel dimensions."""
    cell_width, cell_height = cell_size
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#17212b")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(items):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        image = Image.fromarray(array)
        draw.text((x + 12, y + 10), label, fill="white")
        draw.text((x + 12, y + 28), f"{image.width} x {image.height} pixels", fill="#a8c1d1")
        preview = ImageOps.contain(image, (cell_width - 24, cell_height - 64), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x + (cell_width - preview.width) // 2,
                               y + 52 + (cell_height - 64 - preview.height) // 2))
    return canvas


class FrameRecorder:
    """Write EVERY seam to PNG immediately; assemble a GIF only on request.

    PNG frames keep original canvas size, so varying image dimensions do not
    produce shifting GIF frames. Disk recording does not retain full RGB frames
    in memory. GIF export loads paletted frames and may use substantial memory.
    """
    def __init__(self, directory: str | Path, shape: tuple):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.size = (shape[1], shape[0] + 32)
        self.paths: list[Path] = []

    def _frame(self, image: np.ndarray, title: str) -> Image.Image:
        frame = Image.new("RGB", self.size, "#17212b")
        frame.paste(Image.fromarray(image), (0, 32))
        ImageDraw.Draw(frame).text((5, 10), title, fill="white")
        return frame

    def __call__(self, image: np.ndarray, seam: Seam, step: Step) -> None:
        frame = self._frame(draw_seam(image, seam), f"{step.number}: {step.direction} seam")
        path = self.directory / f"frame_{len(self.paths):05d}.png"
        frame.save(path)
        self.paths.append(path)

    def finish(self, image: np.ndarray) -> None:
        path = self.directory / f"frame_{len(self.paths):05d}.png"
        self._frame(image, "Finished").save(path)
        self.paths.append(path)

    def save_gif(self, path: str | Path, duration_ms: int = 100) -> None:
        if not self.paths:
            raise ValueError("No recorded frames to export.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Paletted frames use less memory than retaining full RGB animation frames.
        frames = []
        for frame_path in self.paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").quantize(colors=256, method=Image.Quantize.FASTOCTREE))
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0, disposal=2, optimize=False)


def save_run(result: ResizeResult, directory: str | Path, metadata: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_image(result.image, directory / "resized.png")
    with (directory / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(Step)])
        writer.writeheader()
        writer.writerows(asdict(step) for step in result.steps)
    report = dict(metadata, elapsed_seconds=result.elapsed_seconds,
                  seams_removed=len(result.steps),
                  output_width=result.image.shape[1], output_height=result.image.shape[0])
    (directory / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
