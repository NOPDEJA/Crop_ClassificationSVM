import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from glob import glob

LABEL_FILE = "./label/label_47PQQ_buffered.tif"
INDICES_FOLDER = "./indices"
OUTPUT_FOLDER = "./aligned_features"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


if __name__ == "__main__":
    # ── Step 1: read label once, build valid-pixel mask ───────────────────────
    with rasterio.open(LABEL_FILE) as label_ds:
        label_array  = label_ds.read(1)
        ref_height   = label_ds.height
        ref_width    = label_ds.width
        ref_transform = label_ds.transform
        ref_crs      = label_ds.crs

    y_flat = label_array.flatten()
    mask   = (y_flat != 0) & (y_flat != 32767)
    n_valid = int(mask.sum())
    print(f"Label raster: {ref_height}x{ref_width}  valid labeled pixels: {n_valid:,}")

    index_files = sorted(glob(os.path.join(INDICES_FOLDER, "*.tif")))
    n_features  = len(index_files)
    print(f"Found {n_features} index TIFs to align.")

    # Pre-allocate output matrix — only labeled pixels, all features
    # Peak memory per TIF: ~480 MB (one full float32 raster) + this matrix
    X = np.zeros((n_valid, n_features), dtype=np.float32)

    # ── Step 2: reproject each TIF, extract labeled pixels, free full raster ──
    for fi, index_file in enumerate(index_files):
        print(f"[{fi+1}/{n_features}] {os.path.basename(index_file)}")
        dst = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
        with rasterio.open(index_file) as src:
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
        X[:, fi] = dst.flatten()[mask]
        del dst  # free the full raster immediately

    y = y_flat[mask]

    out_path = os.path.join(OUTPUT_FOLDER, "svm_add_data_features_labels.npz")
    np.savez(out_path, X=X, y=y)
    print(f"\nSaved to {out_path}")
    print(f"X shape: {X.shape}  (pixels x features)")
    print(f"y unique LU codes ({len(np.unique(y))}): {np.unique(y)}")
