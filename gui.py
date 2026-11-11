# GENAI: Gemini and Claude for UI full UI implementation
# Gemini - base code for GUI layout, widgets, and event handling (from previous tasks)
# Claude - adaptation for this task (connecting the thresholding methods, matplotlib plots:
#          histogram, sigma_B^2(t), T(x,y) map, Manual vs Otsu comparison view, Gaussian pre-filter)

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from segmentation_methods import (
    optimized_adaptive_threshold,
    optimized_calculate_image_histogram,
    optimized_otsu_threshold,
    apply_threshold,
)
from filters import gaussian_filter

COMPARE_METHOD = "Compare: Manual vs Otsu"

PREVIEW_SIZE = (360, 260)  # max thumbnail size; fits a pane also in the 3-column layout


class ImageSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation")
        self.root.geometry("1200x800")

        # --- Application state ---
        self.original_image = None   # PIL image in grayscale ("L")
        self.image_array = None      # the same image as np.ndarray (uint8, H x W)
        self.working_array = None    # image actually thresholded: image_array or its filtered version
        self.filter_key = "None"     # name of the current pre-filter, e.g. "Gaussian σ=1.0"
        self.histogram_bins = None   # h(i) of working_array
        self.otsu_cache = {}         # filter_key -> (t*, sigma_B^2), so Otsu runs once per image version
        self.otsu_t = None           # Otsu t* of working_array
        self.sigma_b_squared = None  # sigma_B^2(t) of working_array, t = 0..255
        self.current_mask = None     # last computed mask (uint8, values 0/255)
        self.hist_bars = None
        self.preview_canvases = {}

        # --- 1. Top Control Panel ---
        control_frame = tk.LabelFrame(root, text="Control Panel", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Upload Image Button
        self.upload_btn = tk.Button(control_frame, text="Load Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=5)

        # Thresholding Method Selection
        tk.Label(control_frame, text="Method:").pack(side=tk.LEFT, padx=(15, 2))
        self.method_var = tk.StringVar(value="Manual")
        self.method_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.method_var,
            values=["Manual", "Otsu's Algorithm", "Adaptive", COMPARE_METHOD],
            state="readonly",
            width=24
        )
        self.method_dropdown.pack(side=tk.LEFT, padx=5)
        self.method_dropdown.bind("<<ComboboxSelected>>", self.on_method_changed)

        # Preprocessing Filter Selection
        tk.Label(control_frame, text="Pre-filter:").pack(side=tk.LEFT, padx=(15, 2))
        self.filter_var = tk.StringVar(value="None")
        self.filter_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.filter_var,
            values=["None", "Gaussian"],
            state="readonly",
            width=12
        )
        self.filter_dropdown.pack(side=tk.LEFT, padx=5)
        self.filter_dropdown.bind("<<ComboboxSelected>>", self.on_filter_changed)

        # Gaussian sigma
        tk.Label(control_frame, text="σ:").pack(side=tk.LEFT, padx=(10, 2))
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.sigma_spin = tk.Spinbox(
            control_frame, from_=0.5, to=5.0, increment=0.5,
            textvariable=self.sigma_var, width=5, command=self.on_filter_changed
        )
        self.sigma_spin.pack(side=tk.LEFT, padx=5)

        # Process / Apply Button
        self.process_btn = tk.Button(control_frame, text="Process Image", command=self.process_image, bg="#e0e0e0")
        self.process_btn.pack(side=tk.LEFT, padx=15)

        # Save Mask Button
        self.save_btn = tk.Button(control_frame, text="Save Mask", command=self.save_mask)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        # --- 2. Dynamic Parameters Panel (Sliders, Spinboxes) ---
        self.params_frame = tk.LabelFrame(root, text="Method Parameters", padx=10, pady=5)
        self.params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Histogram log-scale toggle (always visible)
        self.log_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.params_frame, text="Histogram log scale",
            variable=self.log_var, command=self.on_log_toggled
        ).pack(side=tk.RIGHT, padx=5)

        # Sub-panel for Manual Thresholding (threshold slider 't')
        self.manual_frame = tk.Frame(self.params_frame)
        tk.Label(self.manual_frame, text="Threshold t:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.IntVar(value=128)
        self.threshold_slider = tk.Scale(
            self.manual_frame, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.threshold_var, length=250, command=self.on_param_changed
        )
        self.threshold_slider.pack(side=tk.LEFT, padx=5)

        # Sub-panel for Adaptive Thresholding (Window W and Constant C)
        self.adaptive_frame = tk.Frame(self.params_frame)
        tk.Label(self.adaptive_frame, text="Window W (odd):").pack(side=tk.LEFT, padx=5)
        self.window_var = tk.IntVar(value=15)
        self.window_spin = tk.Spinbox(self.adaptive_frame, from_=3, to=99, increment=2, textvariable=self.window_var, width=5)
        self.window_spin.pack(side=tk.LEFT, padx=5)

        tk.Label(self.adaptive_frame, text="Constant C:").pack(side=tk.LEFT, padx=10)
        self.c_var = tk.IntVar(value=5)
        self.c_spin = tk.Spinbox(self.adaptive_frame, from_=-50, to=50, textvariable=self.c_var, width=5)
        self.c_spin.pack(side=tk.LEFT, padx=5)

        # --- 3. Dynamic 2x2 Preview Container ---
        self.preview_container = tk.Frame(root)
        self.preview_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid weight for 2x2 layout
        # uniform=... -> columns/rows get equal size regardless of their content
        self.preview_container.grid_columnconfigure(0, weight=1, uniform="col")
        self.preview_container.grid_columnconfigure(1, weight=1, uniform="col")
        self.preview_container.grid_rowconfigure(0, weight=1, uniform="row")
        self.preview_container.grid_rowconfigure(1, weight=1, uniform="row")

        # Define titles for the 4 preview panes
        previews_list = ["Original Image", "Pre-filtered Image", "Binary Mask", "Histogram"]

        for idx, title in enumerate(previews_list):
            row = idx // 2
            col = idx % 2

            frame = tk.LabelFrame(self.preview_container, text=title)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            if title == "Histogram":
                self.create_histogram_plot(frame)
            else:
                canvas = tk.Label(frame, text=f"No {title.lower()} yet", bg="#f0f0f0")
                canvas.pack(fill=tk.BOTH, expand=True)
                self.preview_canvases[title] = canvas

        # Extra pane on the right (column 2, full height): method-specific plot.
        # Hidden by default, shown only for Otsu (sigma_B^2) and Adaptive (T map).
        self.method_plot_frame = tk.LabelFrame(self.preview_container, text="Method Plot")
        self.method_plot_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.create_method_plot(self.method_plot_frame)
        self.set_method_plot_visible(False)

        # --- 4. Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, padx=5, pady=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Initial visibility update
        self.update_params_visibility()

    # ------------------------------------------------------------------
    # Histogram plot (matplotlib embedded in tkinter)
    # ------------------------------------------------------------------
    def create_histogram_plot(self, parent):
        # Small requested size (the pane stretches it); "constrained" recomputes margins on every resize
        self.figure = Figure(figsize=(3, 2), dpi=100, layout="constrained")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel("Intensity i")
        self.ax.set_ylabel("h(i)")

        # Vertical lines marking the thresholds; we only move them later
        self.manual_line = self.ax.axvline(self.threshold_var.get(), color="red", label="manual t")
        self.otsu_line = self.ax.axvline(0, color="green", linestyle="--", label="Otsu t*")
        # Legend first: it copies line properties (incl. visibility) at creation time
        self.ax.legend(loc="upper right")
        self.otsu_line.set_visible(False)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_histogram(self):
        if self.hist_bars is not None:
            self.hist_bars.remove()
        self.hist_bars = self.ax.bar(np.arange(256), self.histogram_bins, width=1.0, color="gray")
        self.ax.set_yscale("log" if self.log_var.get() else "linear")
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)
        self.plot_canvas.draw_idle()

    def move_line(self, line, t):
        line.set_xdata([t, t])
        self.plot_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Method plot (separate pane on the right: Otsu / Adaptive)
    # ------------------------------------------------------------------
    def create_method_plot(self, parent):
        self.method_figure = Figure(figsize=(3, 3), dpi=100, layout="constrained")
        self.method_canvas = FigureCanvasTkAgg(self.method_figure, master=parent)
        self.method_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def set_method_plot_visible(self, visible, title="Method Plot"):
        self.method_plot_visible = visible
        if visible:
            self.method_plot_frame.config(text=title)
            self.method_plot_frame.grid()
            self.preview_container.grid_columnconfigure(2, weight=1, uniform="col")
            self.method_canvas.draw_idle()
        else:
            self.method_plot_frame.grid_remove()  # hides, but remembers grid position
            self.preview_container.grid_columnconfigure(2, weight=0, uniform="")

    def new_method_axes(self):
        # Each method plot starts from an empty figure (comparison uses a grid of axes)
        self.method_figure.clear()
        return self.method_figure.add_subplot(111)

    def plot_sigma_b_squared(self):
        ax = self.new_method_axes()
        ax.plot(np.arange(256), self.sigma_b_squared, color="green")
        ax.axvline(self.otsu_t, color="green", linestyle="--", label=f"t* = {self.otsu_t}")
        ax.set_xlim(0, 255)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\sigma_B^2(t)$")
        ax.legend(loc="upper right")
        ax.set_title(self.filter_label(), fontsize=9)
        self.set_method_plot_visible(True, "Otsu: between-class variance")

    def plot_threshold_map(self, T, W, C):
        ax = self.new_method_axes()
        # Same gray scale as the image (0..255), so T can be compared with I directly
        ax.imshow(T, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"W = {W}, C = {C}, {self.filter_label()}", fontsize=9)
        ax.axis("off")
        self.set_method_plot_visible(True, "Adaptive: threshold map T(x,y)")

    def plot_comparison(self, t):
        # Grid of masks: columns = manual t / Otsu t*, rows = without filter / with filter (if any)
        rows = [("No filter", "None", self.image_array)]
        if self.filter_key != "None":
            rows.append((self.filter_key, self.filter_key, self.working_array))

        self.method_figure.clear()
        axes = self.method_figure.subplots(len(rows), 2, squeeze=False)
        for r, (row_name, key, image) in enumerate(rows):
            t_star, _ = self.get_otsu(key, image)
            for c, (name, threshold) in enumerate([("Manual t", t), ("Otsu t*", t_star)]):
                ax = axes[r][c]
                ax.imshow(apply_threshold(image, threshold), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{row_name}\n{name} = {threshold}", fontsize=8)
                ax.axis("off")
        self.set_method_plot_visible(True, "Comparison: manual vs Otsu")

    def on_log_toggled(self):
        if self.histogram_bins is not None:
            self.draw_histogram()

    # ------------------------------------------------------------------
    # Parameters / method selection
    # ------------------------------------------------------------------
    def update_params_visibility(self):
        method = self.method_var.get()
        # Hide all parameter sub-panels first
        self.manual_frame.pack_forget()
        self.adaptive_frame.pack_forget()

        # Show relevant parameters based on selected method
        if method in ("Manual", COMPARE_METHOD):
            self.manual_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        elif method == "Adaptive":
            self.adaptive_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def on_method_changed(self, event=None):
        self.update_params_visibility()
        self.status_var.set(f"Selected method: {self.method_var.get()} - click 'Process Image' to compute the mask")

    def on_param_changed(self, value=None):
        # Called by the slider on every move -> live preview
        self.move_line(self.manual_line, self.threshold_var.get())
        if self.image_array is not None and self.method_var.get() in ("Manual", COMPARE_METHOD):
            self.process_image()

    def on_filter_changed(self, event=None):
        if self.image_array is not None:
            self.update_working_image()
            self.status_var.set(f"Pre-filter: {self.filter_label()} - click 'Process Image' to compute the mask")

    # ------------------------------------------------------------------
    # Pre-filtering (working image) and cached Otsu
    # ------------------------------------------------------------------
    def filter_label(self):
        return "no filter" if self.filter_key == "None" else self.filter_key

    def update_working_image(self):
        # Decide which image is thresholded: original or Gaussian-filtered
        if self.filter_var.get() == "Gaussian":
            try:
                sigma = float(self.sigma_var.get())
            except tk.TclError:
                self.status_var.set("Sigma must be a number.")
                return
            self.filter_key = f"Gaussian σ={sigma}"
            self.working_array = gaussian_filter(self.image_array, sigma)
            self.display_image(Image.fromarray(self.working_array), self.preview_canvases["Pre-filtered Image"])
        else:
            self.filter_key = "None"
            self.working_array = self.image_array
            canvas = self.preview_canvases["Pre-filtered Image"]
            canvas.config(image="", text="No filter selected")
            canvas.image = None

        # Old results belong to the previous working image
        self.otsu_t = None
        self.sigma_b_squared = None
        self.otsu_line.set_visible(False)
        self.clear_mask()
        self.set_method_plot_visible(False)

        # Histogram of the image that is actually thresholded
        self.status_var.set("Computing histogram...")
        self.root.update_idletasks()
        _, self.histogram_bins = optimized_calculate_image_histogram(self.working_array)
        self.draw_histogram()

    def get_otsu(self, key, image):
        if key not in self.otsu_cache:
            self.status_var.set("Computing Otsu threshold...")
            self.root.update_idletasks()
            self.otsu_cache[key] = optimized_otsu_threshold(image)
        return self.otsu_cache[key]

    def show_otsu_on_histogram(self):
        self.otsu_t, self.sigma_b_squared = self.get_otsu(self.filter_key, self.working_array)
        self.otsu_line.set_visible(True)
        self.move_line(self.otsu_line, self.otsu_t)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_image(self):
        if self.image_array is None:
            self.status_var.set("Load an image first.")
            return

        method = self.method_var.get()

        if method == "Manual":
            t = self.threshold_var.get()
            self.move_line(self.manual_line, t)
            self.show_mask(apply_threshold(self.working_array, t))
            if self.method_plot_visible:
                self.set_method_plot_visible(False)
            self.status_var.set(f"Manual threshold: t = {t} ({self.filter_label()})")

        elif method == "Otsu's Algorithm":
            self.show_otsu_on_histogram()
            self.show_mask(apply_threshold(self.working_array, self.otsu_t))
            self.plot_sigma_b_squared()
            self.status_var.set(f"Otsu threshold: t* = {self.otsu_t} ({self.filter_label()})")

        elif method == COMPARE_METHOD:
            # Both thresholds on the same histogram + masks side by side in the right pane
            t = self.threshold_var.get()
            self.move_line(self.manual_line, t)
            self.show_otsu_on_histogram()
            self.show_mask(apply_threshold(self.working_array, t))  # 'Save Mask' saves the manual one
            self.plot_comparison(t)
            self.status_var.set(f"Comparison: manual t = {t}, Otsu t* = {self.otsu_t} ({self.filter_label()})")

        elif method == "Adaptive":
            W = self.window_var.get()
            C = self.c_var.get()
            if W % 2 == 0 or W < 3:
                self.status_var.set("Window W must be an odd integer >= 3.")
                return
            self.status_var.set(f"Applying adaptive thresholding (W={W}, C={C})...")
            self.root.update_idletasks()
            mask, T = optimized_adaptive_threshold(self.working_array, W, C)
            self.show_mask(mask)
            self.plot_threshold_map(T, W, C)
            self.status_var.set(f"Adaptive thresholding applied (W={W}, C={C}).")
        else:
            self.status_var.set(f"{method}: not implemented yet")

    def clear_mask(self):
        # Forget the old mask so it can't be saved for a different image
        self.current_mask = None
        canvas = self.preview_canvases["Binary Mask"]
        canvas.config(image="", text="No binary mask yet")
        canvas.image = None

    def show_mask(self, mask):
        self.current_mask = mask
        self.display_image(Image.fromarray(mask), self.preview_canvases["Binary Mask"])

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.original_image = Image.open(file_path).convert("L")
        self.image_array = np.array(self.original_image)

        # New image -> forget results computed for the previous one
        self.otsu_cache = {}

        self.display_image(self.original_image, self.preview_canvases["Original Image"])
        self.update_working_image()  # applies the selected pre-filter + histogram

        self.status_var.set(f"Loaded: {file_path} - click 'Process Image' to compute the mask")

    def save_mask(self):
        if self.current_mask is None:
            self.status_var.set("Nothing to save - compute a mask first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("TIFF files", "*.tiff")])
        if file_path:
            Image.fromarray(self.current_mask).save(file_path)
            self.status_var.set(f"Saved mask to: {file_path}")

    def display_image(self, img, canvas):
        img_copy = img.copy()
        img_copy.thumbnail(PREVIEW_SIZE)
        img_tk = ImageTk.PhotoImage(img_copy)
        canvas.config(image=img_tk, text="")
        canvas.image = img_tk  # keep a reference, otherwise tkinter drops the image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationGUI(root)
    root.mainloop()
