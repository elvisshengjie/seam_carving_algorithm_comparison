"""Tkinter desktop demo. Only the main thread accesses Tk widgets."""

from pathlib import Path
import queue
import tempfile
import threading
from time import perf_counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .carver import resize, validate_target
from .seams import ALGORITHMS
from .visualization import FrameRecorder, draw_seam, load_image, save_image
from .control import ComputationCancelled, RunControl


# Keep interactive Python graph networks within a practical allocation size.
# This is a GUI resource limit, not a restriction on the seam algorithm/API.
GUI_GRAPH_MAX_PIXELS = 40_000


class SeamCarvingApp:
    def __init__(self, root: tk.Tk, recording_parent: Path | None = None):
        self.root = root
        root.title("Seam Carving - DP / Greedy / Graph Cut")
        root.geometry("1120x710")
        root.minsize(850, 570)
        self.original = None
        self.result = None
        self.recorder = None
        self.busy = False
        self.closing = False
        self.cancel = threading.Event()
        self.messages = queue.Queue()
        self.temp = tempfile.TemporaryDirectory(prefix="seam-carving-", dir=recording_parent)
        self.run_number = 0
        self.worker = None
        self.photos = {}
        self.phase = ""
        self.run_started = 0.0
        self.completed_seams = 0
        self.total_seams = 0
        self.input_name = "Image"
        self.width = tk.IntVar(value=80)
        self.height = tk.IntVar(value=56)
        self.algorithm = tk.StringVar(value="dp")
        self.order = tk.StringVar(value="width-first")
        self.record = tk.BooleanVar(value=True)
        self.original_title = tk.StringVar(value="Original")
        self.result_title = tk.StringVar(value="Seam preview / resized result")
        self.size_change = tk.StringVar(value="")
        self.status = tk.StringVar(value="Load an image to begin. Small images make graph-cut demos faster.")
        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")
        self.load_button = ttk.Button(controls, text="Open image", command=self.open_image)
        self.load_button.grid(row=0, column=0, padx=4, pady=5)
        ttk.Label(controls, text="Width").grid(row=0, column=1)
        self.width_box = ttk.Spinbox(controls, from_=1, to=10000, textvariable=self.width, width=7)
        self.width_box.grid(row=0, column=2, padx=5)
        ttk.Label(controls, text="Height").grid(row=0, column=3)
        self.height_box = ttk.Spinbox(controls, from_=1, to=10000, textvariable=self.height, width=7)
        self.height_box.grid(row=0, column=4, padx=5)
        ttk.Combobox(controls, textvariable=self.algorithm, values=ALGORITHMS,
                     state="readonly", width=9).grid(row=0, column=5, padx=5)
        ttk.Combobox(controls, textvariable=self.order, values=("width-first", "height-first"),
                     state="readonly", width=13).grid(row=0, column=6, padx=5)
        self.run_button = ttk.Button(controls, text="Resize", command=self.start, state="disabled")
        self.run_button.grid(row=0, column=7, padx=5)
        self.cancel_button = ttk.Button(controls, text="Cancel", command=self.request_cancel, state="disabled")
        self.cancel_button.grid(row=0, column=8, padx=5)
        ttk.Checkbutton(controls, text="Record every seam for GIF export", variable=self.record).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=6)
        self.small_button = ttk.Button(controls, text="Small test copy", command=self.small_test_copy,
                                       state="disabled")
        self.small_button.grid(row=1, column=4, padx=5)
        self.save_button = ttk.Button(controls, text="Save resized image", command=self.save, state="disabled")
        self.save_button.grid(row=1, column=5, columnspan=2, padx=5)
        self.gif_button = ttk.Button(controls, text="Export seam GIF", command=self.export_gif, state="disabled")
        self.gif_button.grid(row=1, column=7, columnspan=2, padx=5)
        previews = ttk.Frame(root, padding=12)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure((0, 1), weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, textvariable=self.original_title).grid(row=0, column=0, pady=8)
        ttk.Label(previews, textvariable=self.result_title).grid(row=0, column=1, pady=8)
        self.left = ttk.Label(previews, anchor="center")
        self.left.grid(row=1, column=0, sticky="nsew", padx=5)
        self.right = ttk.Label(previews, anchor="center")
        self.right.grid(row=1, column=1, sticky="nsew", padx=5)
        ttk.Label(previews, text="Both previews use the same scale. Gray space shows the removed area.").grid(
            row=2, column=0, columnspan=2, pady=(8, 3))
        ttk.Label(previews, textvariable=self.size_change, font=("Segoe UI", 10, "bold")).grid(
            row=3, column=0, columnspan=2, pady=(3, 5))
        self.progress = ttk.Progressbar(root, mode="determinate")
        self.progress.pack(fill="x", padx=16, pady=6)
        ttk.Label(root, textvariable=self.status, padding=12, wraplength=1040).pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = root.after(80, self.poll)

    def set_running(self, running: bool):
        """While a worker runs, only Cancel is usable; results must be re-enabled explicitly."""
        idle = "disabled" if running else "normal"
        for button in (self.load_button, self.small_button, self.run_button):
            button.configure(state=idle)
        self.cancel_button.configure(state="normal" if running else "disabled")
        if running:
            self.save_button.configure(state="disabled")
            self.gif_button.configure(state="disabled")

    def show(self, widget, array):
        # Keep one scale and one canvas for both panels. Fitting each image
        # separately would enlarge the result and conceal the removed pixels.
        reference = self.original if self.original is not None else array
        height, width = reference.shape[:2]
        scale = min(500 / width, 360 / height)
        canvas_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image_size = (max(1, round(array.shape[1] * scale)),
                      max(1, round(array.shape[0] * scale)))
        image = Image.fromarray(array).resize(image_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", canvas_size, "#dce2e8")
        canvas.paste(image, (0, 0))
        photo = ImageTk.PhotoImage(canvas)
        self.photos[widget] = photo
        widget.configure(image=photo)

    def show_dimensions(self, array, label):
        height, width = array.shape[:2]
        self.result_title.set(f"{label}: {width} x {height} px")
        original_height, original_width = self.original.shape[:2]
        removed_width, removed_height = original_width - width, original_height - height
        self.size_change.set(
            f"Width: {original_width} -> {width} (-{removed_width} px, {removed_width / original_width:.1%})"
            f"    |    Height: {original_height} -> {height} (-{removed_height} px, {removed_height / original_height:.1%})")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            self.original = load_image(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return
        self.set_input(self.original, Path(path).name)

    def set_input(self, image, name):
        self.original = image
        self.input_name = name
        height, width = image.shape[:2]
        self.width_box.configure(to=width)
        self.height_box.configure(to=height)
        self.width.set(max(1, width * 4 // 5))
        self.height.set(height)
        self.result = self.recorder = None
        self.original_title.set(f"Original: {width} x {height} px")
        self.show(self.left, self.original)
        self.show(self.right, self.original)
        self.show_dimensions(self.original, "Before resizing")
        self.progress.configure(value=0)
        self.set_running(False)
        self.save_button.configure(state="disabled")
        self.gif_button.configure(state="disabled")
        self.status.set(f"{name}: {width} x {height}. Choose smaller target dimensions, then Resize.")

    def small_test_copy(self):
        if self.busy or self.original is None:
            return
        image = Image.fromarray(self.original)
        image.thumbnail((120, 80), Image.Resampling.LANCZOS)
        # This explicitly changes the working input for ALL algorithms; the
        # source file is preserved. Comparisons must use this same test input.
        self.set_input(np.array(image), f"{self.input_name} (small test copy)")
        self.status.set(f"Small test copy: {image.width} x {image.height}. "
                        "Use the same copy to compare all algorithms. Open the source image to restore full size.")

    def request_cancel(self):
        self.cancel.set()
        self.status.set("Cancelling computation...")

    def start(self):
        if self.busy or self.original is None:
            return
        try:
            width, height = validate_target(self.original, self.width.get(), self.height.get())
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Invalid dimensions", str(exc))
            return
        image = self.original.copy()
        algorithm, order, record = self.algorithm.get(), self.order.get(), self.record.get()
        total_seams = image.shape[1] - width + image.shape[0] - height
        if algorithm == "graph" and total_seams and image.shape[0] * image.shape[1] > GUI_GRAPH_MAX_PIXELS:
            messagebox.showwarning(
                "Image too large for interactive graph cut",
                f"This image is {image.shape[1]} x {image.shape[0]} pixels. "
                "Graph cut at this size can take a very long time and use substantial memory.\n\n"
                "Click Small test copy to prepare a manageable input (up to 120 x 80 pixels), "
                "or choose DP for the full image. The source file will stay unchanged.")
            return
        self.busy = True
        self.run_started = perf_counter()
        self.phase = f"Starting {algorithm}..."
        self.completed_seams = 0
        self.total_seams = total_seams
        self.cancel.clear()
        self.run_number += 1
        self.result = self.recorder = None
        self.show(self.right, image)
        self.show_dimensions(image, "Starting")
        self.progress.configure(maximum=max(1, total_seams), value=0)
        self.set_running(True)
        self.status.set(f"Running {algorithm}...")
        directory = Path(self.temp.name) / f"run-{self.run_number}"
        # The worker owns only plain data, a queue, and a cancellation event.
        # Do not retain the app/Tk interpreter through a worker closure.
        messages, cancel_event = self.messages, self.cancel

        def worker():
            try:
                recorder = FrameRecorder(directory, image.shape) if record else None
                last_preview = 0.0
                control = RunControl(cancel_event.is_set,
                                     lambda message: messages.put(("phase", message)))

                def on_step(current, seam, step):
                    nonlocal last_preview
                    control.checkpoint()
                    if recorder:
                        recorder(current, seam, step)
                    now = perf_counter()
                    if step.number == 1 or now - last_preview >= 0.12:
                        messages.put(("progress", (draw_seam(current, seam), step.number)))
                        last_preview = now

                result = resize(image, width, height, algorithm, order, on_step, control=control)
                control.checkpoint()
                if recorder:
                    recorder.finish(result.image)
                messages.put(("done", (result, recorder)))
            except ComputationCancelled:
                messages.put(("cancelled", None))
            except MemoryError:
                messages.put(("error", "Not enough memory. Use Small test copy or select DP."))
            except Exception as exc:
                messages.put(("error", str(exc)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "phase":
                    self.phase = payload
                    continue
                if kind == "progress":
                    image, number = payload
                    self.show(self.right, image)
                    self.show_dimensions(image, "Seam preview")
                    self.progress.configure(value=number)
                    self.completed_seams = number
                    self.phase = "Removing selected seam; red pixels show its path."
                    continue
                self.busy = False
                self.set_running(False)
                if self.closing:
                    self.close()
                    return
                if kind == "done":
                    self.result, self.recorder = payload
                    self.show(self.right, self.result.image)
                    self.show_dimensions(self.result.image, "Result")
                    self.progress.configure(value=self.progress["maximum"])
                    self.save_button.configure(state="normal")
                    self.gif_button.configure(state="normal" if self.recorder else "disabled")
                    h, w = self.result.image.shape[:2]
                    self.status.set(f"Finished: {w} x {h}; {len(self.result.steps)} seams; "
                                    f"{self.result.elapsed_seconds:.3f}s including preview/recording.")
                elif kind == "error":
                    self.status.set("Resize failed.")
                    messagebox.showerror("Resize failed", payload)
                else:
                    self.status.set("Cancelled. Change settings and try again.")
        except queue.Empty:
            pass
        if self.busy:
            elapsed = perf_counter() - self.run_started
            phase = "Cancelling computation..." if self.cancel.is_set() else self.phase
            self.status.set(f"{phase}  |  {self.completed_seams}/{self.total_seams} seams  |  {elapsed:.1f}s elapsed")
        self.poll_id = self.root.after(80, self.poll)

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                save_image(self.result.image, path)
                self.status.set(f"Saved {path}")
            except OSError as exc:
                messagebox.showerror("Cannot save", str(exc))

    def export_gif(self):
        if self.recorder is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("Animated GIF", "*.gif")])
        if path:
            try:
                self.recorder.save_gif(path)
                self.status.set(f"Saved {path}")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Cannot export GIF", str(exc))

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel.set()
            self.status.set("Cancelling computation and closing...")
            return
        self.temp.cleanup()
        self.root.after_cancel(self.poll_id)
        # Release PhotoImages on the GUI thread while the interpreter exists.
        self.photos.clear()
        self.root.destroy()


def launch():
    root = tk.Tk()
    SeamCarvingApp(root)
    root.mainloop()
