"""Hidden-window GUI integration check; skip when Tk/display is unavailable."""

from pathlib import Path
import gc
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from _support import TEST_ROOT, temporary_directory


class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
            from seam_carving.gui import SeamCarvingApp
        except ImportError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except tk.TclError as exc:
            self.skipTest(f"Tk/display unavailable: {exc}")
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.app = SeamCarvingApp(self.root, recording_parent=TEST_ROOT)

    def tearDown(self):
        self.app.cancel.set()
        self.wait_until_done()
        if self.app.worker is not None:
            self.app.worker.join(timeout=2)
            self.assertFalse(self.app.worker.is_alive())
        self.app.close()
        self.app = None
        self.root = None
        # Destroy unreachable Tk object cycles here, on the owning thread,
        # before allocations in a subsequent worker can trigger collection.
        gc.collect()

    def wait_until_done(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.busy)

    def test_worker_preview_and_save(self):
        app = self.app
        try:
            with temporary_directory() as directory:
                directory = Path(directory)
                source = directory / "source.png"
                Image.fromarray(np.random.default_rng(61).integers(0, 256, (6, 8, 3), dtype=np.uint8)).save(source)
                with patch("seam_carving.gui.filedialog.askopenfilename", return_value=str(source)):
                    app.open_image()
                app.width.set(6)
                app.height.set(5)
                app.algorithm.set("graph")
                app.start()
                self.wait_until_done()
                self.assertIsNotNone(app.result)
                self.assertEqual(app.result.image.shape, (5, 6, 3))
                self.assertEqual(len(app.recorder.paths), 4)
                self.assertEqual(app.original_title.get(), "Original: 8 x 6 px")
                self.assertEqual(app.result_title.get(), "Result: 6 x 5 px")
                self.assertIn("8 -> 6 (-2 px, 25.0%)", app.size_change.get())
                self.assertIn("6 -> 5 (-1 px, 16.7%)", app.size_change.get())
                # Both images occupy an original-sized canvas; a smaller result
                # must not be independently enlarged to fill its preview panel.
                self.assertEqual(app.photos[app.left].width(), app.photos[app.right].width())
                self.assertEqual(app.photos[app.left].height(), app.photos[app.right].height())
                output = directory / "saved.png"
                with patch("seam_carving.gui.filedialog.asksaveasfilename", return_value=str(output)):
                    app.save()
                self.assertTrue(output.is_file())
                with Image.open(output) as saved:
                    self.assertEqual(saved.size, (6, 5))
        finally:
            app.cancel.set()

    def test_large_graph_is_checked_before_worker_then_small_copy_runs(self):
        app = self.app
        original = np.full((201, 201, 3), 150, dtype=np.uint8)
        app.set_input(original, "large.png")
        app.width.set(200)
        app.height.set(201)
        app.algorithm.set("graph")
        with patch("seam_carving.gui.messagebox.showwarning") as warning:
            app.start()
            warning.assert_called_once()
        self.assertFalse(app.busy)
        self.assertIsNone(app.result)
        np.testing.assert_array_equal(app.original, original)
        app.small_test_copy()
        self.assertLessEqual(app.original.shape[1], 120)
        self.assertLessEqual(app.original.shape[0], 80)
        self.assertEqual(original.shape, (201, 201, 3))
        app.width.set(app.original.shape[1] - 1)
        app.height.set(app.original.shape[0])
        app.record.set(False)
        app.start()
        self.wait_until_done()
        self.assertIsNotNone(app.result)

    def test_cancel_during_first_seam_before_any_preview(self):
        app = self.app
        app.set_input(np.zeros((8, 10, 3), dtype=np.uint8), "image.png")
        app.algorithm.set("graph")
        app.record.set(False)
        entered = threading.Event()

        def waiting_resize(*args, control, **kwargs):
            control.checkpoint("Solving minimum cut...")
            entered.set()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                control.checkpoint()
                time.sleep(0.005)
            raise AssertionError("GUI failed to cancel during the first seam")

        with patch("seam_carving.gui.resize", side_effect=waiting_resize):
            app.start()
            self.assertTrue(entered.wait(2))
            self.root.update()
            # Poll through the normal event loop so no duplicate timers are created.
            deadline = time.monotonic() + 2
            while "elapsed" not in app.status.get() and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertIn("0/", app.status.get())
            self.assertIn("elapsed", app.status.get())
            app.request_cancel()
            self.wait_until_done()
        self.assertIsNone(app.result)
        self.assertTrue(app.status.get().startswith("Cancelled."))


if __name__ == "__main__":
    unittest.main()
