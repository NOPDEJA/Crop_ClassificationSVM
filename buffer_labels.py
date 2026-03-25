import rasterio
import numpy as np
from scipy.ndimage import binary_erosion
import os

def buffer_labels(input_label, output_label, buffer_pixels=3):
    """
    Remove edge pixels from label raster by applying erosion (buffer).
    
    Args:
        input_label (str): Path to input label raster (.tif).
        output_label (str): Path to save the buffered raster.
        buffer_pixels (int): Number of pixels to erode (default 3).
    """
    with rasterio.open(input_label) as src:
        labels = src.read(1)  # read single-band label raster
        profile = src.profile

    # Copy array for output
    buffered = np.zeros_like(labels)

    # Apply buffer (erosion) class by class
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # skip background

    for cls in unique_labels:
        mask = labels == cls
        eroded = binary_erosion(mask, iterations=buffer_pixels)
        buffered[eroded] = cls

    # Save new raster
    profile.update(dtype=rasterio.int16)
    with rasterio.open(output_label, 'w', **profile) as dst:
        dst.write(buffered.astype(rasterio.int16), 1)

    print(f"Buffered label raster saved to: {output_label}")

if __name__ == "__main__":
    input_label = ""
    output_label = ""
    buffer_labels(input_label, output_label, buffer_pixels=3)
