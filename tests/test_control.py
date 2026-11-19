"""Cancellation must interrupt one seam, not wait for it to finish."""

import threading
import unittest
import numpy as np

from seam_carving import ComputationCancelled, RunControl, find_seam, resize
from seam_carving.flow import Dinic


class ControlTests(unittest.TestCase):
    def test_pre_cancelled_resize_returns_no_result_or_step(self):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        steps = []
        for algorithm in ("dp", "greedy", "graph"):
            with self.subTest(algorithm=algorithm), self.assertRaises(ComputationCancelled):
                resize(image, 9, 8, algorithm, callback=lambda *args: steps.append(args),
                       control=RunControl(lambda: True))
        self.assertEqual(steps, [])

    def test_cancel_while_constructing_graph(self):
        event = threading.Event()
        messages = []

        def status(message):
            messages.append(message)
            if "Building graph: row 2/" in message:
                event.set()

        with self.assertRaises(ComputationCancelled):
            find_seam(np.ones((6, 7)), "graph",
                      control=RunControl(event.is_set, status, min_interval=0))
        self.assertTrue(any("Building graph: row 2/" in m for m in messages))

    def test_cancel_inside_flow_search(self):
        event = threading.Event()
        control = RunControl(event.is_set, lambda message: event.set()
                             if message.startswith("Searching graph paths") else None,
                             min_interval=0)
        graph = Dinic(3, control)
        graph.add_edge(0, 1, 2)
        graph.add_edge(1, 2, 2)
        with self.assertRaises(ComputationCancelled):
            graph.max_flow(0, 2)

    def test_cancel_inside_flow_augmentation(self):
        event = threading.Event()
        control = RunControl(event.is_set, lambda message: event.set()
                             if message.startswith("Augmenting graph paths") else None,
                             min_interval=0)
        graph = Dinic(3, control)
        graph.add_edge(0, 1, 2)
        graph.add_edge(1, 2, 2)
        with self.assertRaises(ComputationCancelled):
            graph.max_flow(0, 2)

    def test_control_preserves_seam_cost_and_reports_work_before_completion(self):
        energy = np.random.default_rng(613).random((8, 10)) * 100
        messages = []
        expected = find_seam(energy, "dp")
        seam = find_seam(energy, "graph", control=RunControl(on_status=messages.append, min_interval=0))
        self.assertAlmostEqual(seam.cost, expected.cost, places=9)
        self.assertTrue(any(m.startswith("Building graph") for m in messages))
        self.assertTrue(any(m.startswith("Solving minimum cut") for m in messages))

    def test_cancel_at_each_early_checkpoint_does_not_mutate_input(self):
        energy = np.random.default_rng(719).random((8, 10))
        original = energy.copy()
        for stop in range(1, 31):
            calls = 0

            def should_cancel():
                nonlocal calls
                calls += 1
                return calls >= stop

            with self.subTest(checkpoint=stop), self.assertRaises(ComputationCancelled):
                find_seam(energy, "graph", control=RunControl(should_cancel))
            np.testing.assert_array_equal(energy, original)


if __name__ == "__main__":
    unittest.main()
