import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from glob import glob

LABEL_FILE = "./label/label_47PQQ_buffered.tif"
INDICES_FOLDER = "./indices"
OUTPUT_FOLDER = "./aligned_features"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def align_index_to_label(label_path, index_path):
    with rasterio.open(label_path) as label_ds, rasterio.open(index_path) as index_ds:
        # Calculate overlapping window
        left = max(label_ds.bounds.left, index_ds.bounds.left)
        bottom = max(label_ds.bounds.bottom, index_ds.bounds.bottom)
        right = min(label_ds.bounds.right, index_ds.bounds.right)
        top = min(label_ds.bounds.top, index_ds.bounds.top)

        if left >= right or bottom >= top:
            raise ValueError(f"No overlap between {label_path} and {index_path}")

        label_window = from_bounds(left, bottom, right, top, transform=label_ds.transform)
        index_window = from_bounds(left, bottom, right, top, transform=index_ds.transform)

        # Read overlapping arrays
        label_array = label_ds.read(1, window=label_window)
        index_array = index_ds.read(1, window=index_window)

        # If necessary, resize index array to match label
        if index_array.shape != label_array.shape:
            index_array = np.resize(index_array, label_array.shape)

        return index_array, label_array

if __name__ == "__main__":
    index_files = glob(os.path.join(INDICES_FOLDER, "*.tif"))
    feature_stack = []
    for i, index_file in enumerate(index_files):
        print(f"Aligning {os.path.basename(index_file)}...")
        index_array, label_array = align_index_to_label(LABEL_FILE, index_file)
        feature_stack.append(index_array)

    # Stack features into (height, width, n_features)
    features = np.stack(feature_stack, axis=-1)
    labels = label_array

    # Flatten for SVM training
    X = features.reshape(-1, features.shape[-1])
    y = labels.flatten()

    # Optionally filter out label=0 (no data)
    mask = y != 0
    X = X[mask]
    y = y[mask]

    # Save for training
    np.savez(os.path.join(OUTPUT_FOLDER, "svm_add_data_features_labels.npz"), X=X, y=y)
    print(f"Saved aligned features and labels to {OUTPUT_FOLDER}/svm_add_data_features_labels.npz")