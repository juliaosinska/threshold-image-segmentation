import numpy as np

# MY OWN CODE: calculating image histogram
def calculate_image_histogram(image: np.ndarray):
    height, width = image.shape
    N = height * width

    # 1. Calculate image histogram
    histogram_bins = np.zeros(256)
    for i in range (0, height):
        for j in range (0, width):
            intensity = image[i,j]
            histogram_bins[intensity] += 1

    # 2. Normalize to the probability distribution
    probability = np.zeros(256)
    for k in range(256):
        probability[k] = histogram_bins[k] / N

    return probability, histogram_bins

# MY OWN CODE: applying the threshold to the image (creating the binary mask)
def apply_threshold(image: np.ndarray, t: int):
    # M(x,y) = 1 if I(x,y) >= t, else 0
    mask = image >= t
    return mask.astype(np.uint8) * 255

# MY OWN CODE: otsu thresholding algorithm based on the instructions provided in the task
def otsu_threshold(image: np.ndarray):
    probability, _ = calculate_image_histogram(image)

    sigma_b_squared = np.zeros(256)

    for t in range(256):
        omega_0 = 0.0
        mi_0 = 0.0
        mi_1 = 0.0

        for i in range(256):
            if i <= t:
                omega_0 += probability[i]
                mi_0 += i * probability[i]
            else:
                mi_1 += i * probability[i]

        omega_1 = 1 - omega_0

        if omega_0 > 0 and omega_1 > 0:
            mi_0 = mi_0 / omega_0
            mi_1 = mi_1 / omega_1

            sigma_b_squared[t] = omega_0 * omega_1 * (mi_0 - mi_1) ** 2

    best_threshold = int(np.argmax(sigma_b_squared))

    return best_threshold, sigma_b_squared
