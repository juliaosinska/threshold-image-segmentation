import numpy as np

# WŁASNE: from previous tasks - image filtering
# GENAI: Claude – adaptation of my task-3 Gaussian code to 2D grayscale arrays (no uint8 rounding between passes)

def get_separable_gaussian_kernel(sigma=1.0):
    radius = int(np.ceil(3 * sigma))

    x = np.arange(-radius, radius + 1)
    # GENAI: Gemini used to help with using correct Gaussian kernel formula
    kernel = np.exp(-(x**2) / (2 * (sigma**2)))

    kernel /= np.sum(kernel)

    return kernel

def optimized_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # image: 2D float array, kernel: 2D array; returns 2D float array (same size as image)
    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')

    output = np.zeros(image.shape, dtype=np.float64)

    # GENAI: Gemini used to help with optimizing the convolution operation by iterating over the kernel and applying it to the entire image at once
    for n in range(kernel_height):
        for m in range(kernel_width):
            weight = kernel[n, m]

            if weight == 0:
                continue

            piece_of_image = padded_image[n:n + image_height, m:m + image_width]
            output += piece_of_image * weight

    return output

def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    kernel = get_separable_gaussian_kernel(sigma=sigma)
    size = len(kernel)

    kernel_h = kernel.reshape((1, size))
    kernel_v = kernel.reshape((size, 1))

    h_convolved = optimized_convolution(image.astype(np.float64), kernel_h)
    final = optimized_convolution(h_convolved, kernel_v)

    return np.clip(np.round(final), 0, 255).astype(np.uint8)
