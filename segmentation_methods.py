import numpy as np

# MY OWN CODE: calculating image histogram
# def calculate_image_histogram(image: np.ndarray):
#     height, width = image.shape
#     N = height * width
#
#     # 1. Calculate image histogram
#     histogram_bins = np.zeros(256)
#     for i in range (0, height):
#         for j in range (0, width):
#             intensity = image[i,j]
#             histogram_bins[intensity] += 1
#
#     # 2. Normalize to the probability distribution
#     probability = np.zeros(256)
#     for k in range(256):
#         probability[k] = histogram_bins[k] / N
#
#     return probability, histogram_bins

# GenAI: Claude - optimized version of calculate_image_histogram (loops replaced with NumPy)
# ~ 130x faster (400x600 image)
def optimized_calculate_image_histogram(image: np.ndarray):
    N = image.size

    # 1. Calculate image histogram: np.bincount counts how many times each value 0..255 occurs
    histogram_bins = np.bincount(image.ravel(), minlength=256).astype(np.float64)

    # 2. Normalize to the probability distribution (whole array at once)
    probability = histogram_bins / N

    return probability, histogram_bins

# MY OWN CODE: applying the threshold to the image (creating the binary mask)
def apply_threshold(image: np.ndarray, t: int):
    # M(x,y) = 1 if I(x,y) >= t else 0
    mask = image >= t
    return mask.astype(np.uint8) * 255

# MY OWN CODE: otsu thresholding algorithm based on the instructions provided in the task
# def otsu_threshold(image: np.ndarray):
#     probability, _ = calculate_image_histogram(image)
#
#     sigma_b_squared = np.zeros(256)
#
#     # for each possible threshold t calculate the between-class variance
#     for t in range(256):
#         omega_0 = 0.0
#         mi_0 = 0.0
#         mi_1 = 0.0
#
#         for i in range(256):
#             if i <= t:
#                 omega_0 += probability[i]
#                 mi_0 += i * probability[i]
#             else:
#                 mi_1 += i * probability[i]
#
#         omega_1 = 1 - omega_0
#
#         # skip if either class is empty (to avoid division by zero)
#         if omega_0 > 0 and omega_1 > 0:
#             mi_0 = mi_0 / omega_0
#             mi_1 = mi_1 / omega_1
#
#             sigma_b_squared[t] = omega_0 * omega_1 * (mi_0 - mi_1) ** 2
#
#     best_threshold = int(np.argmax(sigma_b_squared))
#
#     return best_threshold, sigma_b_squared

# GenAI: Claude - optimized version of otsu_threshold
# The inner loop is replaced with cumulative sums which give those sums for every t at once
# ~ 100x faster (400x600 image)
def optimized_otsu_threshold(image: np.ndarray):
    probability, histogram_bins = optimized_calculate_image_histogram(image)

    sigma_b_squared = np.zeros(256)

    i = np.arange(256)
    omega_0 = np.cumsum(probability)              # omega_0[t] = sum_{i<=t} p(i)
    omega_1 = 1 - omega_0
    mi_0_sum = np.cumsum(i * probability)         # sum_{i<=t} i*p(i)
    mi_1_sum = mi_0_sum[-1] - mi_0_sum            # sum_{i>t} i*p(i) = total - sum_{i<=t}

    # skip t where either class is empty (to avoid division by zero)
    pixels_0 = np.cumsum(histogram_bins)
    valid = (pixels_0 > 0) & (pixels_0 < image.size)

    mi_0 = mi_0_sum[valid] / omega_0[valid]
    mi_1 = mi_1_sum[valid] / omega_1[valid]

    sigma_b_squared[valid] = omega_0[valid] * omega_1[valid] * (mi_0 - mi_1) ** 2

    best_threshold = int(np.argmax(sigma_b_squared))

    return best_threshold, sigma_b_squared

# MY OWN CODE: adaptive thresholding based on the instructions provided in the task
# def adaptive_threshold(image: np.ndarray, W: int, C: float):
#     window_radius = W // 2
#     padded_image = np.pad(image, window_radius, mode='edge')
#     T = np.zeros(image.shape)
#     height, width = image.shape
#
#     for x in range(height):
#         for y in range(width):
#             window = padded_image[x:x + W, y:y + W]
#             local_threshold = np.mean(window) - C
#             T[x, y] = local_threshold
#
#     return apply_threshold(image, T), T

# GenAI: Claude - optimized version of adaptive_threshold using an integral image
# S[x, y] = sum of all padded pixels above and to the left of (x, y), so the sum of any
# W x W window needs only 4 lookups (independent of W) and is done for all pixels at once
# ~ 180x faster (400x600 image)
def optimized_adaptive_threshold(image: np.ndarray, W: int, C: float):
    window_radius = W // 2
    padded_image = np.pad(image, window_radius, mode='edge').astype(np.int64)
    height, width = image.shape

    # integral image with an extra row and column of zeros at the top/left
    S = np.zeros((padded_image.shape[0] + 1, padded_image.shape[1] + 1), dtype=np.int64)
    S[1:, 1:] = padded_image.cumsum(axis=0).cumsum(axis=1)

    # window for pixel (x, y) is padded_image[x:x + W, y:y + W]  (same as in the loop version)
    window_sum = (S[W:W + height, W:W + width]
                  - S[0:height, W:W + width]
                  - S[W:W + height, 0:width]
                  + S[0:height, 0:width])

    T = window_sum / (W * W) - C

    return apply_threshold(image, T), T
