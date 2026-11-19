from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from _support import temporary_directory

from seam_carving import Seam, compute_energy, find_seam, resize
from seam_carving.carver import remove_seam, validate_seam
from seam_carving.experiments import compare
from seam_carving.samples import create_samples
from seam_carving.visualization import FrameRecorder, draw_seam, load_image, save_run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(51).integers(0, 256, (8, 10, 3), dtype=np.uint8)

    def test_exact_dimensions_and_no_input_mutation_all_algorithms(self):
        original = self.image.copy()
        for algorithm in ("dp", "greedy", "graph"):
            for order in ("width-first", "height-first"):
                with self.subTest(algorithm=algorithm, order=order):
                    result = resize(self.image, 7, 6, algorithm, order)
                    self.assertEqual(result.image.shape, (6, 7, 3))
                    self.assertEqual(len(result.steps), 5)
                    self.assertEqual([s.number for s in result.steps], list(range(1, 6)))
                    self.assertTrue(all(s.seam_cost >= 0 for s in result.steps))
        np.testing.assert_array_equal(self.image, original)

    def test_transposition_equivalence(self):
        for algorithm in ("dp", "greedy", "graph"):
            a = resize(self.image, 7, 6, algorithm, "width-first").image
            b = resize(self.image.swapaxes(0, 1), 6, 7, algorithm, "height-first").image.swapaxes(0, 1)
            np.testing.assert_array_equal(a, b)

    def test_removal_preserves_every_other_pixel(self):
        indices = np.array([1, 2, 3, 3, 4, 5, 5, 6])
        seam = Seam(indices, 0, "dp", "vertical")
        result = remove_seam(self.image, seam)
        for row, col in enumerate(indices):
            np.testing.assert_array_equal(result[row], np.delete(self.image[row], col, axis=0))
        horizontal = Seam(np.array([1, 1, 2, 3, 4, 4, 5, 5, 4, 3]), 0, "dp", "horizontal")
        result = remove_seam(self.image, horizontal)
        for col, row in enumerate(horizontal.indices):
            np.testing.assert_array_equal(result[:, col], np.delete(self.image[:, col], row, axis=0))

    def test_energy_recomputed_on_current_image_at_every_step(self):
        visited = []
        expected_image = self.image.copy()

        def callback(image, seam, step):
            nonlocal expected_image
            np.testing.assert_array_equal(image, expected_image)
            expected = find_seam(compute_energy(image), seam.algorithm, seam.direction)
            self.assertEqual(seam.cost, expected.cost)
            np.testing.assert_array_equal(seam.indices, expected.indices)
            expected_image = remove_seam(image, seam)
            visited.append(step.number)

        result = resize(self.image, 6, 5, callback=callback)
        self.assertEqual(len(visited), 7)
        np.testing.assert_array_equal(result.image, expected_image)

    def test_noop_and_shrink_to_single_pixel(self):
        result = resize(self.image, 10, 8)
        np.testing.assert_array_equal(result.image, self.image)
        self.assertIsNot(result.image, self.image)
        self.assertEqual(result.steps, [])
        for algorithm in ("dp", "greedy", "graph"):
            self.assertEqual(resize(self.image, 1, 1, algorithm).image.shape, (1, 1, 3))

    def test_invalid_images_targets_and_seams(self):
        for width, height in ((0, 4), (11, 4), (5, 9), (-1, 4), (4.5, 4), (True, 4)):
            with self.subTest(target=(width, height)), self.assertRaises(ValueError):
                resize(self.image, width, height)
        for image in (np.zeros((2, 2)), np.zeros((0, 2, 3), dtype=np.uint8), self.image.astype(float)):
            with self.assertRaises(ValueError):
                resize(image, 1, 1)
        with self.assertRaises(ValueError):
            resize(self.image, 5, 5, order="random")
        for indices in ([0] * 7, [0.0] * 8, [-1] * 8, [10] * 8, [0, 3, 0, 0, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                remove_seam(self.image, Seam(np.asarray(indices), 0, "dp", "vertical"))
        with self.assertRaises(ValueError):
            remove_seam(np.zeros((2, 1, 3), dtype=np.uint8), Seam(np.array([0, 0]), 0, "dp", "vertical"))

    def test_frames_gif_and_run_files(self):
        with temporary_directory() as directory:
            directory = Path(directory)
            recorder = FrameRecorder(directory / "frames", self.image.shape)
            result = resize(self.image, 8, 7, callback=recorder)
            recorder.finish(result.image)
            recorder.save_gif(directory / "seams.gif")
            save_run(result, directory, {"algorithm": "dp"})
            self.assertEqual(len(recorder.paths), 4)
            for path in recorder.paths:
                with Image.open(path) as frame:
                    self.assertEqual(frame.size, (10, 40))
            with Image.open(directory / "seams.gif") as gif:
                self.assertEqual(gif.n_frames, 4)
                for i in range(gif.n_frames):
                    gif.seek(i)
                    gif.load()
            np.testing.assert_array_equal(load_image(directory / "resized.png"), result.image)
            self.assertTrue((directory / "steps.csv").read_text().startswith("number,direction"))

    def test_rgb_loading_from_grayscale_and_rgba(self):
        with temporary_directory() as directory:
            path = Path(directory) / "image.png"
            for mode, shape in (("L", (4, 5)), ("RGBA", (4, 5, 4))):
                Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)
                self.assertEqual(load_image(path).shape, (4, 5, 3))

    def test_comparison_exports_measured_results_and_cost_checks(self):
        with temporary_directory() as directory:
            directory = Path(directory)
            path = directory / "input.png"
            Image.fromarray(self.image).save(path)
            report = compare(path, 8, 7, directory / "compare", repeats=2)
            for algorithm in ("dp", "greedy", "graph"):
                self.assertEqual(len(report["timings"][algorithm]["runs_seconds"]), 2)
                with Image.open(directory / "compare" / algorithm / "resized.png") as image:
                    self.assertEqual(image.size, (8, 7))
            for checks in report["initial_seam_checks"].values():
                self.assertTrue(checks["dp_graph_equal_cost"])
                self.assertGreaterEqual(checks["greedy_excess_energy"], -1e-8)
            self.assertEqual(len((directory / "compare" / "timings.csv").read_text().splitlines()), 7)
            for name in ("energy.png", "comparison.png", "first_seams.png", "runtime.png", "summary.json", "results.md"):
                self.assertTrue((directory / "compare" / name).is_file())

    def test_samples_are_reproducible(self):
        with temporary_directory() as directory:
            paths = create_samples(Path(directory) / "a")
            others = create_samples(Path(directory) / "b")
            self.assertEqual(len(paths), 3)
            for a, b in zip(paths, others):
                self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
