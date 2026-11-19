import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from PIL import Image
from _support import temporary_directory


PROJECT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "seam_carving", *map(str, args)],
                              cwd=PROJECT, text=True, capture_output=True, timeout=60)

    def test_resize_cli_with_animation(self):
        with temporary_directory() as directory:
            directory = Path(directory)
            path = directory / "input.png"
            Image.fromarray(np.full((6, 8, 3), 100, dtype=np.uint8)).save(path)
            result = self.run_cli("resize", path, "--width", 6, "--height", 5,
                                  "--algorithm", "graph", "--gif", "--output", directory / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((directory / "out" / "run.json").read_text())
            self.assertEqual(report["seams_removed"], 3)
            self.assertTrue((directory / "out" / "seams.gif").is_file())

    def test_bad_targets_and_missing_input_exit_cleanly(self):
        missing = self.run_cli("resize", "does-not-exist.png")
        self.assertEqual(missing.returncode, 2)
        self.assertNotIn("Traceback", missing.stderr)
        with temporary_directory() as directory:
            path = Path(directory) / "input.png"
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
            bad = self.run_cli("resize", path, "--width", 3, "--output", Path(directory) / "out")
            self.assertEqual(bad.returncode, 2)
            self.assertNotIn("Traceback", bad.stderr)


if __name__ == "__main__":
    unittest.main()
