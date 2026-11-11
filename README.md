# Threshold Image Segmentation

A small Python desktop app (Tkinter) that segments grayscale images into object and background by thresholding. It shows the intensity histogram, compares global and local thresholding methods, and saves the binary mask.

## Features

- Load an image (converted to grayscale) and view its intensity histogram (linear or log scale)
- **Manual thresholding**: threshold slider with a live mask preview and a marker line on the histogram
- **Otsu's algorithm** (own implementation): automatic threshold `t*` marked on the histogram, plus a plot of the between-class variance σ²_B(t)
- **Adaptive thresholding**: per-pixel threshold with adjustable window `W` and correction `C`, plus a preview of the threshold map T(x, y)
- **Compare: Manual vs Otsu**: both masks side by side and both thresholds on the same histogram
- **Gaussian pre-filter** (adjustable σ): every method can run on the smoothed image; the comparison view shows results with and without filtering
- Save the mask as PNG or TIFF (8-bit, values 0 and 255 only)

## Methods

| Method | Threshold | Mask |
|---|---|---|
| Manual | `t` chosen by the user | `M = 255 if I ≥ t else 0` |
| Otsu | `t* = argmax σ²_B(t)`, where `σ²_B = ω₀·ω₁·(μ₀ − μ₁)²` | `M = 255 if I ≥ t* else 0` |
| Adaptive | `T(x,y) = μ_W(x,y) − C` (mean of the W×W window, edge padding) | `M = 255 if I(x,y) ≥ T(x,y) else 0` |

All functions in `segmentation_methods.py` take a grayscale `uint8` NumPy array and return a `uint8` mask with values 0/255. The same `apply_threshold` function is used for a single global threshold `t` and for the adaptive threshold map `T`.

## Project structure

- `gui.py`: Tkinter GUI, histogram and method plots (matplotlib)
- `segmentation_methods.py`: histogram, `apply_threshold`, Otsu, adaptive thresholding
- `filters.py`: separable Gaussian filter (reused from the previous image-filtering tasks)

## Requirements and running

Python 3 with NumPy, Pillow and matplotlib (Tkinter is included with the standard Windows installer):

```powershell
py -m pip install numpy Pillow matplotlib
py gui.py
```

## Notes

- AI usage:
    - GUI implementation (`gui.py` file),
    - optimized versions for thresholding functions. Use of numpy methods like cumsum() and vectorization made the methods work much faster (~100-180 times). Details provided in the related file,
    - `filters.py` previous task's own code adaptation for greyscale images.


- Observations:
    - **Manual vs Otsu**: Otsu's `t*` lands in the valley between the two histogram peaks. A manual threshold set below `t*` lets background pixels into the mask (objects "grow", more noise), and one set above `t*` erodes the objects. With the slider at exactly `t*`, both masks are identical, since both use the same `apply_threshold`.
    <img width="1562" height="1068" alt="image" src="https://github.com/user-attachments/assets/f74f66f0-9805-40c6-99d1-cb6737249c7d" />

    - **Gaussian pre-filtering**: noise widens the histogram peaks until they overlap, so no single threshold separates the classes cleanly, and the mask has isolated "salt-and-pepper" pixels. Smoothing before thresholding narrows the peaks and deepens the valley between them. The masks get cleaner and more solid, and Otsu's `t*` becomes more reliable because the histogram is closer to bimodal.
    <img width="1554" height="1062" alt="image" src="https://github.com/user-attachments/assets/46cf3a0b-1651-4630-8964-234585aca375" />
    <img width="1555" height="1061" alt="image" src="https://github.com/user-attachments/assets/c85c5098-ebd7-4ea6-b52d-7d55291d6334" />

    - **Cost of filtering**: smoothing slightly rounds sharp corners and blurs thin details, so a larger σ removes more noise but loses more shape detail.


