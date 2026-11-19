# Seam Carving - Algorithm Analysis Assignment 2

CSD3130/CSD3131 Assignment 2, Group 6.

A runnable Python project implementing content-aware image **shrinking** with
dynamic programming, a local greedy algorithm, and graph cut via a custom Dinic
min-cut/max-flow solver. All methods support vertical and horizontal seams and
recompute the energy map after every removal. No content-aware resizing library
or graph solver is used.

## Team repository

[GitHub repository](https://github.com/elvisshengjie/seam_carving_algorithm_comparison)

```powershell
git clone https://github.com/elvisshengjie/seam_carving_algorithm_comparison.git
cd seam_carving_algorithm_comparison
```

## Start here

Requires Python 3.11 or newer. In a terminal opened in this project directory:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m seam_carving demo
python -m seam_carving gui
```

The demo generates three reproducible 96 x 64 synthetic images, resizes them to
80 x 56 with all three algorithms, and writes results under `output/demo/`.
The texture case is intentionally harder and can take minutes with graph cut.
The desktop interface lets you open an image, edit target dimensions, select an
algorithm and removal order, preview seams, save the result, and export a GIF.
For graph testing with a large photo, click **Small test copy** to explicitly
prepare an input of up to 120 x 80 pixels. The loaded working input and target
dimensions update together; use this same input for fair comparisons with DP
and greedy. The source file is preserved; open it again to restore full size.
Both previews use the original image's display scale, with gray space showing
removed width/height. Labels show actual pixel dimensions and percentage reductions.
Tkinter is included in standard Windows Python installations; the CLI works
without Tkinter. Start with `output/demo/inputs/landscape.png` for a quick GUI demo.

For an isolated environment without activating PowerShell scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m seam_carving gui
```

Use the same `.venv` Python executable for the other commands if using this option.

## Resize your own image

```powershell
python -m seam_carving resize "photo.jpg" --width 400 --height 250 --algorithm dp --output output/my-photo --gif
```

Target width and height are **pixels**, not percentages. Omit either dimension
to preserve it. Both dimensions must lie between 1 and the original dimension.
Enlargement by seam insertion is outside this implementation. File paths with
spaces need quotes. The source file is never overwritten by these commands.

Algorithms: `dp`, `greedy`, `graph`. Default removal order: `width-first`.
Use `--order height-first` to remove horizontal seams first. Order can affect
the final image. `--frames` records every seam as PNG; `--gif` also assembles
an animation. These options increase runtime and storage use. The GUI reports
the current computation stage and elapsed time even before the first seam
finishes. Cancel is checked during graph construction and inside max-flow
search/augmentation, as well as between seams.

Each resize writes:

| File | Purpose |
| --- | --- |
| `resized.png` | Final RGB output |
| `steps.csv` | Direction, seam energy, dimensions, and stage timings per removal |
| `run.json` | Parameters and total elapsed time |
| `frames/frame_*.png` | Every seam in red, then a final frame, if recording |
| `seams.gif` | Animated seam-removal sequence, if requested |

Animation frames keep the original canvas size. The shrinking image occupies
the top-left corner below a caption, rather than being stretched to fit.
Use a new output directory for each experiment; existing same-name output files
are replaced. A recorder's GIF uses only frames from the current run.

## Compare all three methods

```powershell
python -m seam_carving compare "output/demo/inputs/landscape.png" --width 80 --height 56 --repeats 3 --output output/my-comparison
```

Compare uses the exact same RGB input, Sobel energy definition, target size, and
removal order for every algorithm. It rotates algorithm execution order between
repeats and excludes image file I/O and visualizations from resize timings.
Run without other heavy applications for cleaner measurements. There is no
separate warm-up run; all measured runs are retained.

| File | Evidence provided |
| --- | --- |
| `comparison.png` | Original, crop/scale baselines, and all three resized outputs |
| `first_seams.png` | Each method's first vertical/horizontal seam on the original image |
| `energy.png` | Original Sobel energy map |
| `runtime.png` | Linear-scale chart of median full-resize times |
| `timings.csv` | Every measured repeat, including energy/find/remove stage totals |
| `summary.json` | Costs, seam coordinates, mean/median/std, versions, and checks |
| `results.md` | Measured results with linked figures |
| `dp/`, `greedy/`, `graph/` | Final outputs and step records from each method's first run |

DP and graph cut should have equal minimum cost on the **same fixed energy map**,
within floating-point tolerance. They may return different paths when optimal
seams tie. Their later images can then differ. The minimum energy seam at each
step does not guarantee a globally optimal sequence of removals or the best
perceptual quality. Initial seam timings exclude energy computation; the full
resize timings include it. `run.json` for GUI/recorded runs includes recording
overhead and should not be used as a fair algorithm benchmark.

The graph solver is written in Python for clarity and independent implementation.
Begin graph experiments around 100 x 70 pixels and a modest number of seams;
high texture and larger images can be slow and use substantial memory. No input
is silently downscaled. To prevent impractical allocations in the interactive
app, the GUI asks for a small test input when graph cut would process more than
40,000 input pixels. This conservative GUI limit is not an algorithmic bound;
the CLI/API still allow larger graphs and can be slow or exhaust memory.
For natural-image experiments, explicitly prepare the
same manageable input resolution for all methods and document that preparation.

## Code map

| Module | Responsibility |
| --- | --- |
| `seam_carving/energy.py` | Float64 luminance and 3 x 3 Sobel magnitude |
| `seam_carving/seams.py` | Rolling-row DP, greedy selection, seam graph construction |
| `seam_carving/flow.py` | Custom Dinic residual network, max-flow, residual BFS |
| `seam_carving/control.py` | Cooperative cancellation and throttled stage reporting |
| `seam_carving/carver.py` | Seam validation/removal and repeated two-axis shrinking |
| `seam_carving/visualization.py` | Image I/O, overlays, panels, frames, GIF, CSV/JSON |
| `seam_carving/experiments.py` | Fair comparison and measured result exports |
| `seam_carving/samples.py` | Seeded synthetic test images |
| `seam_carving/gui.py` | Tkinter interface; background resize worker |
| `seam_carving/__main__.py` | Command-line entry points (`resize`, `compare`, `demo`, `gui`) |
| `scripts/package_project.py` | Builds the source ZIP that CI uploads |
| `tests/` | Exhaustive oracles, edge cases, I/O, CLI, GUI integration |
| `CSD3130_CSD3131_Presentation.pptx` | Presentation slides |

## Tests and CI

```powershell
python -m unittest discover -s tests -v
python scripts/package_project.py
```

Tests include an independent exhaustive seam oracle, exhaustive cuts for tiny
flow networks, a strict greedy counterexample, cheapest last-column seams,
single rows/columns, ties, floating-point energies, pixel-preserving removal,
two-axis resizing, energy recomputation, animation decoding, CLI errors, and a
hidden-window GUI load/resize/save check. The GUI check skips when Tk/display
is unavailable. Test temporary files are kept within `tmp/tests/`.
Additional tests verify cancellation during graph construction/search/augmentation,
unchanged optimal seam costs with progress enabled, GUI feedback before the first
seam finishes, and the large-input check followed by a successful small-copy run.

`.github/workflows/ci.yml` runs on every push and pull request. It tests on
Windows and Linux with Python 3.11/3.12, runs the demo on one job, packages
source, and uploads artifacts. This provides continuous integration and
artifact delivery. It does not deploy an external service. On headless Linux
the GUI test skips.

## Assignment coverage

| Requirement | Implementation/evidence |
| --- | --- |
| DP, vertical and horizontal | `seams.py`, shared transpose-based orientation |
| Recalculate energy after every seam | `carver.resize` |
| Greedy comparison | `seams.py`, `compare`, greedy counterexample test |
| Graph cut using min-cut/max-flow | `seams.py` + custom `flow.Dinic` |
| Seam, quality, runtime comparison | `compare` outputs and `results.md`; report Sections 10 to 16 |
| Item 2 and complexity discussion | Report Sections 3, 7 and 8 |
| Full seam visualization | `--frames`, `--gif`, desktop recording |
| Interactive resizing | `python -m seam_carving gui` |
| CI/artifact delivery | `.github/workflows/ci.yml` |
| Report and presentation | Report submitted on xSITE; `CSD3130_CSD3131_Presentation.pptx` |
| AI usage declaration | Report Section 17 |

Still to do before submission: record the presentation (at most 2 minutes per
person, 12 minutes in total), add its video link to the report, complete peer
evaluation, and submit the report and slides on xSITE by Friday 20 November,
23:59. Everyone should be able to explain their submitted code for the
closed-book project quiz on Tuesday 24 November.

## References

- Assignment brief and rubric supplied with this workspace.
- [NumPy padding API](https://numpy.org/doc/stable/reference/generated/numpy.pad.html), used for replicated boundary handling.
- [Pillow image/GIF documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif), used for I/O and animation export.
- Official GitHub Actions documentation: [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python), [upload-artifact](https://github.com/actions/upload-artifact).
