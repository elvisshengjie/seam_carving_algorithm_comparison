"""Keep test files inside the project, including in restricted environments."""

from pathlib import Path
import tempfile

TEST_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "tests"


def temporary_directory():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_ROOT)
