import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from glob import glob

LABEL_FILE = "./label/label_47PQQ_buffered.tif"
INDICES_FOLDER = "./indices"
OUTPUT_FOLDER = "./aligned_features"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def resample_to_reference(index_path, ref_height, ref_width, ref_transform, ref_crs):
    """Reproject and resample index TIF to exactly match the reference raster grid."""
    dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
    with rasterio.open(index_path) as src:
        reproject(
            source=src.read(1).astype(np.float32),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


if __name__ == "__main__":
    # Read label raster once — this is the alignment reference
    with rasterio.open(LABEL_FILE) as label_ds:
        label_array = label_ds.read(1)
        ref_height   = label_ds.height
        ref_width    = label_ds.width
        ref_transform = label_ds.transform
        ref_crs      = label_ds.crs

    index_files = sorted(glob(os.path.join(INDICES_FOLDER, "*.tif")))
    print(f"Found {len(index_files)} index TIFs to align.")

    feature_stack = []
    for index_file in index_files:
        print(f"Aligning {os.path.basename(index_file)}...")
        arr = resample_to_reference(index_file, ref_height, ref_width, ref_transform, ref_crs)
        feature_stack.append(arr)

    # Stack into (H, W, n_features) then flatten
    features = np.stack(feature_stack, axis=-1)
    X = features.reshape(-1, features.shape[-1])
    y = label_array.flatten()

    # Remove nodata pixels (background=0, sentinel=32767)
    mask = (y != 0) & (y != 32767)
    X = X[mask]
    y = y[mask]

    out_path = os.path.join(OUTPUT_FOLDER, "svm_add_data_features_labels.npz")
    np.savez(out_path, X=X, y=y)
    print(f"Saved to {out_path}")
    print(f"X shape: {X.shape}  (pixels x features={X.shape[1]})")
    print(f"y unique LU codes: {np.unique(y)}")
