"""export_pred_to_tif.py

Convert a prediction .npy (row-aligned with the labeled pixels of the
buffered label raster, i.e. the same rows as y in the training NPZ) back
into a GeoTIFF map.

Works for any stage's full-dataset predictions:
  python export_pred_to_tif.py runs/s1_dem/stage1_s1_dem_pred.npy runs/s1_dem/stage1_s1_dem_pred.tif
  python export_pred_to_tif.py runs/s1_dem/end_to_end_lu_pred.npy runs/s1_dem/end_to_end_lu_pred.tif

Pixels outside the labeled area are written as 0 (nodata).
"""
import sys
import numpy as np
import rasterio

LABEL_FILE = "./label/label_47PQQ_buffered.tif"

def export(pred_npy, out_tif, label_file=LABEL_FILE):
    with rasterio.open(label_file) as src:
        label = src.read(1)
        profile = src.profile.copy()

    mask = (label != 0) & (label != 32767)
    preds = np.load(pred_npy)
    if preds.shape[0] != int(mask.sum()):
        raise SystemExit(f"Length mismatch: pred has {preds.shape[0]} rows but label raster "
                         f"has {int(mask.sum())} labeled pixels. Wrong label file or stale pred?")

    full = np.zeros(label.shape, dtype=np.int16)
    full[mask] = preds.astype(np.int16)

    profile.update(dtype=rasterio.int16, count=1, compress="lzw", nodata=0)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(full, 1)
    print(f"Saved {out_tif}  (classes: {np.unique(preds)})")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python export_pred_to_tif.py <pred.npy> <out.tif>")
    export(sys.argv[1], sys.argv[2])
