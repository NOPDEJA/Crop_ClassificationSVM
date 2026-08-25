"""m3_build_features.py

M3 of docs/PLAN_S2_3DATE_COMPETITIVE.md: feature parity with the two competing
studies. Adds MTCI (red-edge chlorophyll) and raw B11 (long SWIR) per date --
the one index family both other arms carry and this one lacks, plus the raw band
their importance analyses rank highly. 24 columns -> 30.

    MTCI = (B06 - B05) / (B05 - B04)

WHY A SEPARATE RASTER DIRECTORY. `align_indices_labels.py` builds its column
order from `sorted(glob("./indices/*.tif"))`. Dropping six files in there would
silently reorder the columns of every existing arm, so any model rebuilt
afterwards would read its features in the wrong order against a saved model.
These go to ./indices_m3 instead and are appended explicitly.

WHY APPEND RATHER THAN REBUILD. The plan requires the new matrix be row-identical
to the current one, so the parcel split, y and every saved index array stay valid.
Rebuilding from scratch would reproduce the same rows -- the mask is just
`(label != 0) & (label != 32767)` -- but appending makes that guarantee structural
instead of something to re-verify, and it is far cheaper. y is asserted identical
regardless.

Temporal deltas are deliberately NOT built. Dec-Oct differences of existing
columns are linear combinations of columns the model already has; a linear SVM on
a Nystroem map gains nothing from them, and they would inflate the column count
while adding no information.

Usage:  python m3_build_features.py            # rasters + NPZ
        python m3_build_features.py --rasters  # rasters only
"""
import os
import sys
from glob import glob

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

S2_FOLDER = "./S2_data"
OUT_RASTERS = "./indices_m3"
LABEL_FILE = "./label/label_47PQQ_buffered.tif"
SRC_NPZ = "./aligned_features/svm_s2_3date_features_labels.npz"
OUT_NPZ = "./aligned_features/svm_s2_3date_m3_features_labels.npz"
DATES = ("2018-10-31", "2018-11-30", "2018-12-31")

BAND = {"B04": 3, "B05": 4, "B06": 5, "B11": 8}   # 1-indexed, see CLAUDE.md
REFLECTANCE_SCALE = 10000.0
MTCI_CLIP = 10.0          # vegetation sits well under this; guards the divide


def log(*a):
    print(*a, flush=True)


def build_rasters():
    os.makedirs(OUT_RASTERS, exist_ok=True)
    for date in DATES:
        src_path = os.path.join(S2_FOLDER, f"47PQQ_{date}.tif")
        if not os.path.exists(src_path):
            raise SystemExit(f"missing composite {src_path}")
        with rasterio.open(src_path) as src:
            red = src.read(BAND["B04"]).astype("float32")
            re1 = src.read(BAND["B05"]).astype("float32")
            re2 = src.read(BAND["B06"]).astype("float32")
            swir1 = src.read(BAND["B11"]).astype("float32")

            # same nodata convention as compute_indices.py: gaps are 0 in every band
            invalid = (red == 0) & (re1 == 0) & (re2 == 0)

            red /= REFLECTANCE_SCALE
            re1 /= REFLECTANCE_SCALE
            re2 /= REFLECTANCE_SCALE
            swir1 /= REFLECTANCE_SCALE

            with np.errstate(divide="ignore", invalid="ignore"):
                mtci = (re2 - re1) / (re1 - red)
            # the denominator crosses zero wherever the red edge is flat, which
            # is most non-vegetated ground -- clip rather than let it blow up
            mtci = np.clip(mtci, -MTCI_CLIP, MTCI_CLIP)

            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1, compress="lzw",
                           nodata=np.nan)
            for name, arr in (("MTCI", mtci), ("B11", swir1)):
                arr = arr.copy()
                arr[invalid | ~np.isfinite(arr)] = np.nan
                out = os.path.join(OUT_RASTERS, f"{name}_47PQQ_{date}.tif")
                with rasterio.open(out, "w", **profile) as dst:
                    dst.write(arr, 1)
                finite = np.isfinite(arr)
                log(f"  wrote {out}  finite {finite.mean():6.2%}  "
                    f"p05 {np.nanpercentile(arr, 5):8.3f}  "
                    f"median {np.nanmedian(arr):8.3f}  "
                    f"p95 {np.nanpercentile(arr, 95):8.3f}")


def build_npz():
    with rasterio.open(LABEL_FILE) as ds:
        lab = ds.read(1)
        h, w, tr, crs = ds.height, ds.width, ds.transform, ds.crs
    flat = lab.flatten()
    mask = (flat != 0) & (flat != 32767)
    n = int(mask.sum())
    log(f"label grid {h}x{w}, {n:,} labelled pixels")

    d = np.load(SRC_NPZ, allow_pickle=True)
    names = list(d["feature_names"].astype(str))
    y_src = d["y"]
    assert y_src.size == n, f"row count changed: {y_src.size:,} vs {n:,}"
    assert np.array_equal(y_src, flat[mask]), "y does not reproduce from the label raster"
    log(f"source matrix {d['X'].shape}, y reproduces from the label raster exactly")

    new_files = []
    for date in DATES:
        for name in ("MTCI", "B11"):
            new_files.append(os.path.join(OUT_RASTERS, f"{name}_47PQQ_{date}.tif"))
    for f in new_files:
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}; run with --rasters first")

    Xsrc = d["X"]
    X = np.empty((n, Xsrc.shape[1] + len(new_files)), dtype=np.float32)
    X[:, :Xsrc.shape[1]] = Xsrc
    del Xsrc, d

    for j, path in enumerate(new_files):
        dst = np.full((h, w), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            reproject(source=src.read(1).astype(np.float32), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tr, dst_crs=crs,
                      resampling=Resampling.bilinear,
                      src_nodata=src.nodata, dst_nodata=np.nan)
        col = dst.flatten()[mask]
        X[:, len(names) + j] = col
        log(f"  col {len(names) + j:2d} {os.path.basename(path):<28} "
            f"finite {np.isfinite(col).mean():6.2%}")
        del dst

    names += [os.path.basename(f) for f in new_files]
    np.savez(OUT_NPZ, X=X, y=y_src, feature_names=np.array(names))
    log(f"\nsaved {OUT_NPZ}  X {X.shape}")
    log("columns: " + ", ".join(names[-6:]) + "  (appended)")


if __name__ == "__main__":
    log("M3 feature build: MTCI + raw B11 per date")
    build_rasters()
    if "--rasters" not in sys.argv:
        build_npz()
