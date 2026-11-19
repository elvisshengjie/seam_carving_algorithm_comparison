"""Fair per-image comparisons, raw timings, seam costs, and visual evidence."""

import csv
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import PIL
from PIL import Image, ImageDraw

from .carver import resize, validate_target
from .energy import compute_energy
from .seams import ALGORITHMS, DIRECTIONS, find_seam
from .visualization import draw_seam, energy_image, load_image, make_panel, save_image, save_run


def compare(input_path: str | Path, width: int, height: int, directory: str | Path,
            repeats: int = 3, order: str = "width-first") -> dict:
    if repeats < 1:
        raise ValueError("Repeats must be positive.")
    original = load_image(input_path)
    validate_target(original, width, height)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    energy = compute_energy(original)
    save_image(energy_image(energy), directory / "energy.png")
    seam_rows = []
    panels = []
    # This evaluates all methods on EXACTLY the same fixed energy map.
    for direction in DIRECTIONS:
        for algorithm in ALGORITHMS:
            start = perf_counter()
            seam = find_seam(energy, algorithm, direction)
            seconds = perf_counter() - start
            seam_rows.append(dict(algorithm=algorithm, direction=direction,
                                  cost=seam.cost, find_seconds=seconds,
                                  indices=seam.indices.tolist()))
            overlay = draw_seam(original, seam)
            panels.append((f"{algorithm} {direction}: cost {seam.cost:.2f}", overlay))
            save_image(overlay, directory / f"{algorithm}_{direction}_seam.png")
    make_panel(panels).save(directory / "first_seams.png")

    samples = {algorithm: [] for algorithm in ALGORITHMS}
    raw_rows = []
    outputs = {}
    for repeat in range(repeats):
        # Rotate order to reduce systematic first/last-run bias.
        algorithms = ALGORITHMS[repeat % 3:] + ALGORITHMS[:repeat % 3]
        for algorithm in algorithms:
            result = resize(original, width, height, algorithm, order)
            samples[algorithm].append(result.elapsed_seconds)
            raw_rows.append(dict(algorithm=algorithm, repeat=repeat + 1,
                                 total_seconds=result.elapsed_seconds,
                                 energy_seconds=sum(s.energy_seconds for s in result.steps),
                                 find_seconds=sum(s.find_seconds for s in result.steps),
                                 remove_seconds=sum(s.remove_seconds for s in result.steps)))
            if repeat == 0:
                outputs[algorithm] = result.image
                save_run(result, directory / algorithm,
                         dict(algorithm=algorithm, order=order, input=str(Path(input_path).resolve())))
    with (directory / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    summary = {}
    for algorithm, values in samples.items():
        summary[algorithm] = dict(mean_seconds=float(np.mean(values)),
                                  median_seconds=float(np.median(values)),
                                  std_seconds=float(np.std(values)),
                                  runs_seconds=values)
    image = Image.fromarray(original)
    x, y = (image.width - width) // 2, (image.height - height) // 2
    cropped = np.array(image.crop((x, y, x + width, y + height)))
    scaled = np.array(image.resize((width, height), Image.Resampling.LANCZOS))
    save_image(cropped, directory / "crop.png")
    save_image(scaled, directory / "scale.png")
    panels = [("Original", original), ("Center crop (baseline)", cropped),
              ("Uniform scaling (baseline)", scaled)]
    panels += [(f"{a}: median {summary[a]['median_seconds']:.4f}s", outputs[a]) for a in ALGORITHMS]
    make_panel(panels).save(directory / "comparison.png")
    _timing_chart(summary).save(directory / "runtime.png")

    checks = {}
    for direction in DIRECTIONS:
        costs = {row["algorithm"]: row["cost"] for row in seam_rows if row["direction"] == direction}
        checks[direction] = dict(dp_graph_equal_cost=bool(np.isclose(costs["dp"], costs["graph"], rtol=1e-9, atol=1e-8)),
                                 greedy_excess_energy=costs["greedy"] - costs["dp"])
    report = dict(input=str(Path(input_path).resolve()), input_size=[image.width, image.height],
                  target_size=[width, height], repeats=repeats, order=order,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   pillow=PIL.__version__, platform=platform.platform()),
                  timings=summary, initial_seams=seam_rows, initial_seam_checks=checks,
                  notes=["Resize timings exclude file I/O, overlays, GIFs, and energy/seam previews.",
                         "Initial seam timings include selection only; energy is precomputed.",
                         "DP/graph optimality is per fixed energy map, not a global guarantee for multiple removals.",
                         "Tied optimal seams can differ and produce different later images.",
                         "Removed energy is not an objective measure of perceptual image quality."])
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Measured comparison", "", f"Input: `{Path(input_path).name}`",
             f"Size: {image.width} x {image.height} -> {width} x {height}",
             f"Repeats: {repeats}; removal order: {order}", "",
             "| Algorithm | Median seconds | Mean seconds | Std seconds |",
             "| --- | ---: | ---: | ---: |"]
    for a in ALGORITHMS:
        s = summary[a]
        lines.append(f"| {a} | {s['median_seconds']:.6f} | {s['mean_seconds']:.6f} | {s['std_seconds']:.6f} |")
    lines += ["", "![Resized images](comparison.png)", "", "![Initial seams](first_seams.png)", "",
              "Inspect foreground shape, straight lines, and texture in the output images. "
              "These synthetic results do not establish quality on natural photographs.", "",
              "See summary.json for initial seam costs, environment, and interpretation limits."]
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _timing_chart(summary: dict) -> Image.Image:
    image = Image.new("RGB", (850, 245), "#17212b")
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), "Median full resize time (seconds, linear scale)", fill="white")
    maximum = max(s["median_seconds"] for s in summary.values()) or 1
    for index, algorithm in enumerate(ALGORITHMS):
        seconds = summary[algorithm]["median_seconds"]
        y = 60 + index * 55
        draw.text((20, y + 9), algorithm, fill="white")
        length = max(1, int(seconds / maximum * 550))
        draw.rectangle((100, y, 100 + length, y + 30), fill=("#49c6b0", "#f6bd60", "#6ca7ee")[index])
        draw.text((665, y + 9), f"{seconds:.6f}s", fill="white")
    return image
