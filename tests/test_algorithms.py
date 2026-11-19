"""Independent oracles: enumerate legal seams and all cuts on tiny inputs."""

from itertools import product
import unittest
import numpy as np

from seam_carving import compute_energy, find_seam
from seam_carving.carver import validate_seam
from seam_carving.flow import Dinic


def exhaustive_minimum(energy):
    height, width = energy.shape
    best = float("inf")
    for path in product(range(width), repeat=height):
        if all(abs(path[row] - path[row - 1]) <= 1 for row in range(1, height)):
            best = min(best, sum(float(energy[row, col]) for row, col in enumerate(path)))
    return best


class SeamTests(unittest.TestCase):
    def test_dp_and_graph_against_exhaustive_oracle(self):
        rng = np.random.default_rng(1207)
        cases = 0
        for height in range(1, 6):
            for width in range(1, 5):
                for _ in range(8):
                    energy = rng.integers(0, 20, (height, width)).astype(float)
                    expected = exhaustive_minimum(energy)
                    for algorithm in ("dp", "graph"):
                        with self.subTest(shape=energy.shape, algorithm=algorithm, case=cases):
                            seam = find_seam(energy, algorithm)
                            validate_seam(energy.shape, seam)
                            self.assertEqual(seam.cost, expected)
                    cases += 1

    def test_float_energies_and_transposed_oracle(self):
        rng = np.random.default_rng(58)
        for _ in range(40):
            energy = rng.random((4, 5)) * 100
            expected = exhaustive_minimum(energy.T)
            for algorithm in ("dp", "graph"):
                seam = find_seam(energy, algorithm, "horizontal")
                validate_seam(energy.shape, seam)
                self.assertAlmostEqual(seam.cost, expected, places=9)

    def test_last_column_and_last_row_are_allowed(self):
        energy = np.array([[8., 8., 0.]] * 4)
        for algorithm in ("dp", "greedy", "graph"):
            np.testing.assert_array_equal(find_seam(energy, algorithm).indices, [2] * 4)
            np.testing.assert_array_equal(find_seam(energy.T, algorithm, "horizontal").indices, [2] * 4)

    def test_ties_zero_energy_and_single_pixel(self):
        for shape in ((1, 1), (1, 8), (8, 1), (4, 5)):
            for algorithm in ("dp", "greedy", "graph"):
                energy = np.zeros(shape)
                for direction in ("vertical", "horizontal"):
                    seam = find_seam(energy, algorithm, direction)
                    validate_seam(shape, seam)
                    self.assertEqual(seam.cost, 0)
                    np.testing.assert_array_equal(seam.indices, find_seam(energy, algorithm, direction).indices)

    def test_greedy_can_be_strictly_worse(self):
        energy = np.array([[1, 2, 9, 9], [1, 9, 1, 9], [100, 100, 100, 1]], dtype=float)
        self.assertEqual(find_seam(energy, "greedy").cost, 102)
        self.assertEqual(find_seam(energy, "dp").cost, 4)
        self.assertEqual(find_seam(energy, "graph").cost, 4)

    def test_bad_energy_and_options_rejected(self):
        for energy in (np.empty((0, 2)), np.array([1, 2]), [[-1]], [[np.nan]], [[np.inf]]):
            with self.subTest(energy=energy), self.assertRaises(ValueError):
                find_seam(energy)
        with self.assertRaises(ValueError):
            find_seam([[1]], "unknown")
        with self.assertRaises(ValueError):
            find_seam([[1]], direction="diagonal")


class EnergyTests(unittest.TestCase):
    def test_constant_images_have_zero_energy_including_boundaries(self):
        for shape in ((1, 1, 3), (1, 8, 3), (8, 1, 3), (5, 9, 3)):
            np.testing.assert_allclose(compute_energy(np.full(shape, 173, dtype=np.uint8)), 0)

    def test_matches_direct_sobel_convolution(self):
        image = np.random.default_rng(19).integers(0, 256, (6, 7, 3), dtype=np.uint8)
        gray = image.astype(float) @ np.array([0.2126, 0.7152, 0.0722])
        padded = np.pad(gray, 1, mode="edge")
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = kx.T
        reference = np.empty(gray.shape)
        for row in range(gray.shape[0]):
            for col in range(gray.shape[1]):
                patch = padded[row:row + 3, col:col + 3]
                reference[row, col] = np.hypot(np.sum(patch * kx), np.sum(patch * ky))
        np.testing.assert_allclose(compute_energy(image), reference, atol=1e-10)

    def test_uint8_subtraction_does_not_wrap(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[:, 2:] = 255
        energy = compute_energy(image)
        np.testing.assert_allclose(energy[:, 1:3], 1020)
        np.testing.assert_allclose(energy[:, [0, 3]], 0)


class FlowTests(unittest.TestCase):
    def test_known_network(self):
        graph = Dinic(6)
        for edge in [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4),
                     (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7),
                     (3, 5, 20), (4, 5, 4)]:
            graph.add_edge(*edge)
        self.assertEqual(graph.max_flow(0, 5), 23)
        self.assertNotIn(5, graph.source_side(0))

    def test_random_networks_against_all_cuts(self):
        rng = np.random.default_rng(30)
        for _ in range(40):
            nodes = 6
            edges = [(u, v, float(rng.integers(1, 10)) / 4)
                     for u in range(nodes) for v in range(nodes)
                     if u != v and rng.random() < 0.3]
            graph = Dinic(nodes)
            for edge in edges:
                graph.add_edge(*edge)
            expected = float("inf")
            for bits in product((False, True), repeat=nodes - 2):
                source_side = {0} | {i + 1 for i, bit in enumerate(bits) if bit}
                capacity = sum(c for u, v, c in edges if u in source_side and v not in source_side)
                expected = min(expected, capacity)
            self.assertEqual(graph.max_flow(0, nodes - 1), expected)

    def test_long_path_uses_no_recursive_dfs(self):
        graph = Dinic(1500)
        for node in range(1499):
            graph.add_edge(node, node + 1, 1)
        self.assertEqual(graph.max_flow(0, 1499), 1)


if __name__ == "__main__":
    unittest.main()
