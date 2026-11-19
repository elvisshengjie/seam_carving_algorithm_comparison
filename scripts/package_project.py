"""Create a source-only ZIP, excluding PDFs, recordings, caches, and local data."""

import argparse
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/seam-carving-project.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [root / name for name in ("README.md", "CONTRIBUTING.md", "requirements.txt",
                                       ".gitignore", ".gitattributes")]
    for name in ("seam_carving", "tests", "docs", "scripts", ".github"):
        sources.extend(path for path in (root / name).rglob("*")
                       if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources):
            archive.write(source, source.relative_to(root))
    print(f"Packaged {len(sources)} source files: {output}")


if __name__ == "__main__":
    main()
