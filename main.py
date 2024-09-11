
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_histogram(image):
    """Calculate the histogram of a grayscale image."""
    hist, _ = np.histogram(image, bins=256, range=(0, 255))
    return hist / hist.sum()  # Normalize the histogram


def cumulative_means(hist):
    """Compute cumulative means (μ_k) for each threshold."""
    return np.cumsum(hist * np.arange(256))




def total_mean(hist):
    """Compute the global mean of the image."""
    return np.sum(hist * np.arange(256))


def between_class_variance(cumulative_sum, cumulative_mean, global_mean):
    """
    Compute between-class variance for each threshold.
    """
    numerator = (global_mean * cumulative_sum - cumulative_mean) ** 2
    denominator = cumulative_sum * (1 - cumulative_sum)
    denominator[denominator == 0] = 1e-9  # Prevent division by zero, a common trick in numerical computing
    return numerator / denominator



# New valley-related methods and the modified Two-Stage Otsu Method from https://github.com/ps-george/multithreshold/blob/master/otsu.py"""

def normalised_histogram_binning(hist, M=32, L=256):
    """Normalised histogram binning"""
    norm_hist = np.zeros((M, 1), dtype=np.float32)
    N = L // M
    counters = [range(x, x+N) for x in range(0, L, N)]
    for i, C in enumerate(counters):
        norm_hist[i] = 0
        for j in C:
            norm_hist[i] += hist[j]
    norm_hist = (norm_hist / norm_hist.max()) * 100
    return norm_hist


def find_valleys(H):
    """Valley estimation on *H*, H should be normalised-binned-grouped histogram. from https://github.com/ps-george/multithreshold/blob/master/otsu.py"""
    hsize = H.shape[0]
    probs = np.zeros((hsize, 1), dtype=int)
    costs = np.zeros((hsize, 1), dtype=float)
    for i in range(1, hsize-1):
        if H[i] > H[i-1] or H[i] > H[i+1]:
            probs[i] = 0
        elif H[i] < H[i-1] and H[i] == H[i+1]:
            probs[i] = 1
            costs[i] = H[i-1] - H[i]
        elif H[i] == H[i-1] and H[i] < H[i+1]:
            probs[i] = 3
            costs[i] = H[i+1] - H[i]
        elif H[i] < H[i-1] and H[i] < H[i+1]:
            probs[i] = 4
            costs[i] = (H[i-1] + H[i+1]) - 2*H[i]
        elif H[i] == H[i-1] and H[i] == H[i+1]:
            probs[i] = probs[i-1]
            costs[i] = probs[i-1]
    for i in range(1, hsize-1):
        if probs[i] != 0:
            probs[i] = (probs[i-1] + probs[i] + probs[i+1]) // 4
    valleys = [i for i, x in enumerate(probs) if x > 0]
    return valleys


def valley_estimation(hist, M=32, L=256):
    """Valley estimation for histogram. L should be divisible by M.from https://github.com/ps-george/multithreshold/blob/master/otsu.py"""
    norm_hist = normalised_histogram_binning(hist, M, L)
    valleys = find_valleys(norm_hist)
    return valleys


def threshold_valley_regions(hist, valleys, N):
    """Perform Otsu's method over estimated valley regions. from https://github.com/ps-george/multithreshold/blob/master/otsu.py"""
    thresholds = []
    for valley in valleys:
        start_pos = (valley * N) - N
        end_pos = (valley + 2) * N
        h = hist[start_pos:end_pos]
        sub_threshold, val = otsu_threshold(h)
        thresholds.append((start_pos + sub_threshold, val))
    thresholds.sort(key=lambda x: x[1], reverse=True)
    thresholds, values = [list(t) for t in zip(*thresholds)]
    return thresholds

def display_all_methods(image_path):
    """Display and save the results of Otsu's method, multi-threshold Otsu, and modified TSMO side by side."""
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Apply Otsu's thresholding
    optimal_threshold, _ = otsu_threshold(image)
    _, otsu_thresh_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)

    # Apply multi-threshold Otsu's method
    multi_thresholds = otsu_multi_threshold(image, n_classes=3)
    multi_thresh_image = np.where(image > multi_thresholds[0], 255, 0).astype(np.uint8)
    for t in multi_thresholds[1:]:
        multi_thresh_image += np.where(image > t, 255, 0).astype(np.uint8)

    # Apply modified TSMO method
    hist = calculate_histogram(image)
    modified_thresholds = modified_TSMO(hist)
    modified_image = np.where(image > modified_thresholds[0], 255, 0).astype(np.uint8)
    for t in modified_thresholds[1:]:
        modified_image += np.where(image > t, 255, 0).astype(np.uint8)

    # Save the images to disk
    cv2.imwrite("otsu_threshold_image.png", otsu_thresh_image)
    cv2.imwrite("multi_threshold_image.png", multi_thresh_image)
    cv2.imwrite("modified_tsmo_image.png", modified_image)
   
    plot_thresholding_result(
    hist=calculate_histogram(image), 
    otsu_threshold=optimal_threshold, 
    multi_thresholds=multi_thresholds,  
    modified_thresholds=modified_thresholds  
)    



def plot_thresholding_result(hist, otsu_threshold, multi_thresholds, modified_thresholds, save_path='thresholding_result.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(256), hist, label='Histogram', color='blue')
    
    ax.axvline(x=otsu_threshold, color='red', linestyle='--', label='Otsu Threshold')
    
    for i, mt in enumerate(multi_thresholds):
        ax.axvline(x=mt, color='green', linestyle='dotted', label=f'Multi Otsu Threshold {i+1}' if i == 0 else '')
    
    for i, tsm in enumerate(modified_thresholds):
        ax.axvline(x=tsm, color='purple', linestyle='--', label=f'Modified TSMO Threshold {i+1}' if i == 0 else '')

    ax.set_title("Histogram of Brightness with Thresholds")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved as {save_path}")


def otsu_threshold(image):
    """ Otsu Threshold"""

    # Calculate the histogram
    hist = calculate_histogram(image)

    # Compute cumulative sums and cumulative means
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = cumulative_means(hist)
    global_mean = total_mean(hist)

    # Compute between-class variance for each threshold
    between_var = between_class_variance(cumulative_sum, cumulative_mean, global_mean)

    # Select the threshold that maximizes between-class variance
    optimal_threshold = np.argmax(between_var)

    return optimal_threshold, between_var[optimal_threshold]

def otsu_multi_threshold(image, n_classes):
    """Extend Otsu's method to handle multiple thresholds"""
    thresholds = []
    temp_image = image.copy()

    for i in range(n_classes - 1):
        threshold, _ = otsu_threshold(temp_image)  

        thresholds.append(threshold)

        # Update the image after thresholding
        temp_image = np.where(temp_image > threshold, 0, temp_image)

    return thresholds

def modified_TSMO(hist, M=32, L=256):
    """Modified Two-Stage Multithreshold Otsu Method. from https://github.com/ps-george/multithreshold/blob/master/otsu.py"""
    N = L // M
    valleys = valley_estimation(hist, M, L)
    thresholds = threshold_valley_regions(hist, valleys, N)
    return thresholds

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process image with different thresholding methods")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to process the image
    display_all_methods(args.image_path)

if __name__ == "__main__":
    main()