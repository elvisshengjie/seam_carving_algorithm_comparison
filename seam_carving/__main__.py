"""Run with: python -m seam_carving --help"""

import argparse
import json
from pathlib import Path

from .carver import resize
from .experiments import compare
from .samples import create_samples
from .seams import ALGORITHMS
from .visualization import FrameRecorder, load_image, save_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Content-aware shrinking using DP, greedy, or min-cut/max-flow.")
    commands = parser.add_subparsers(dest="command", required=True)
    carve = commands.add_parser("resize", help="Shrink an image and optionally record every seam.")
    carve.add_argument("input", type=Path)
    carve.add_argument("--width", type=int, help="Target width; defaults to the original width.")
    carve.add_argument("--height", type=int, help="Target height; defaults to the original height.")
    carve.add_argument("--algorithm", choices=ALGORITHMS, default="dp")
    carve.add_argument("--order", choices=("width-first", "height-first"), default="width-first")
    carve.add_argument("--output", type=Path, default=Path("output/resize"), help="Output directory.")
    carve.add_argument("--frames", action="store_true", help="Save one PNG per seam plus the final frame.")
    carve.add_argument("--gif", action="store_true", help="Also assemble an animated GIF; implies --frames.")

    experiment = commands.add_parser("compare", help="Measure all three algorithms on the same input.")
    experiment.add_argument("input", type=Path)
    experiment.add_argument("--width", type=int, required=True)
    experiment.add_argument("--height", type=int, required=True)
    experiment.add_argument("--repeats", type=int, default=3)
    experiment.add_argument("--order", choices=("width-first", "height-first"), default="width-first")
    experiment.add_argument("--output", type=Path, default=Path("output/comparison"))

    demo = commands.add_parser("demo", help="Generate three small sample images and compare all methods.")
    demo.add_argument("--output", type=Path, default=Path("output/demo"))
    demo.add_argument("--repeats", type=int, default=1)
    commands.add_parser("gui", help="Open the interactive desktop demo.")
    args = parser.parse_args(argv)
    try:
        if args.command == "resize":
            image = load_image(args.input)
            width = image.shape[1] if args.width is None else args.width
            height = image.shape[0] if args.height is None else args.height
            recorder = FrameRecorder(args.output / "frames", image.shape) if args.frames or args.gif else None
            result = resize(image, width, height, args.algorithm, args.order, recorder)
            save_run(result, args.output, dict(input=str(args.input.resolve()),
                                              algorithm=args.algorithm, order=args.order,
                                              timing_includes_frame_recording=recorder is not None))
            if recorder:
                recorder.finish(result.image)
                if args.gif:
                    recorder.save_gif(args.output / "seams.gif")
            print(f"Saved {width} x {height} image to {args.output / 'resized.png'}")
            print(f"Removed {len(result.steps)} seams in {result.elapsed_seconds:.4f}s")
        elif args.command == "compare":
            report = compare(args.input, args.width, args.height, args.output, args.repeats, args.order)
            print(json.dumps(report["timings"], indent=2))
            print(f"Comparison saved to {args.output.resolve()}")
        elif args.command == "demo":
            if args.repeats < 1:
                raise ValueError("Repeats must be positive.")
            paths = create_samples(args.output / "inputs")
            for path in paths:
                print(f"Comparing {path.name} ...", flush=True)
                compare(path, 80, 56, args.output / path.stem, args.repeats)
            image = load_image(paths[0])
            recorder = FrameRecorder(args.output / "animation" / "frames", image.shape)
            result = resize(image, 80, 56, callback=recorder)
            recorder.finish(result.image)
            recorder.save_gif(args.output / "animation" / "seams.gif")
            print(f"Demo completed: {args.output.resolve()}")
        else:
            from .gui import launch
            launch()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
