"""prepare_epoch.py

Build the feature matrix and labels for a second epoch (2020 or 2024), so the
2018-trained cascade can be tested on it without retraining anything.

    python prepare_epoch.py 2024

Everything the 2018 arm did is reused rather than reimplemented: the same index
functions, the same reprojection loop in align_indices_labels.py (now
parameterised), the same 3-pixel erosion. Three things genuinely differ and each
is a trap worth naming:

1. **The composites stay zipped.** They are read through GDAL's /vsizip/ handler,
   so 14 GB of extraction is avoided. Verified beforehand: 2018/2020/2024 share
   the same 10-band uint16 10980x10980 structure and consistent reflectance
   scaling, so the BOA_ADD_OFFSET=-1000 that processing baseline 04.00 applies to
   2024 (and not to 2018 or 2020) was already handled when they were built.

2. **Labels must be burned onto the 2018 grid.** rasterize_parcel.py derives its
   raster extent from each shapefile's own bounds. The 2018 survey happened to
   land exactly on the tile grid; a different survey will not, and a raster of a
   different shape would break both the row correspondence with 2018 and the
   churn comparison. Here the reference grid is taken from the 2018 label raster.

3. **Column order must match training, not merely look plausible.** The 2018 arm's
   column order came from sorted(glob("indices/*.tif")). A separate per-year
   indices folder sorts the same way because only the year digits change, but
   "sorts the same way" is an assumption, so it is asserted against the training
   NPZ's feature_names with the year substituted. A silent mismatch here would
   produce numbers that look sane and mean nothing.

Outputs:
  indices_<year>/                                       24 index TIFs
  label/label_47PQQ_<year>.tif, _<year>_buffered.tif    on the 2018 grid
  aligned_features/svm_s2_3date_<year>_features_labels.npz
"""
import os
import subprocess
import sys
import time
from glob import glob

import numpy as np
import rasterio

from compute_indices import compute_indices
from compute_extra_indices import compute_extra_indices

ZIP_TPL = "../{year}.zip"
DATES = ("10-31", "11-30", "12-31")
REF_LABEL = "./label/label_47PQQ.tif"          # 2018 grid, the reference for everything
TRAIN_NPZ = "./aligned_features/svm_s2_3date_features_labels.npz"
SURVEY = {2020: "SHAPEFILE_2563", 2024: "SHAPEFILE_2567"}
BUFFER_PIXELS = 3


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def rasterize_on_reference(shapefile, value_column, reference, out_tif):
    """Burn polygons onto an existing raster's exact grid.

    Deliberately not rasterize_parcel.py: that derives the grid from the
    shapefile's own bounds, which is right for defining a new study area and
    wrong for a second epoch that has to line up with the first.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    with rasterio.open(reference) as ref:
        transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs

    gdf = gpd.read_file(shapefile)
    if gdf.crs is None:
        raise SystemExit(f"{shapefile} has no CRS")
    if gdf.crs != crs:
        log(f"  reprojecting survey {gdf.crs} -> {crs}")
        gdf = gdf.to_crs(crs)

    # LU_ID_L3 is float64 in all three surveys; a null would rasterise to garbage
    # rather than fail, so drop and report instead of trusting it.
    bad = gdf[value_column].isna() | gdf.geometry.isna()
    if bad.any():
        log(f"  dropping {int(bad.sum())} polygons with null {value_column} or geometry")
        gdf = gdf[~bad]

    # Mixed-crop parcels carry a 9-digit compound code, <code4>1<code4> -- 220412302 is
    # cassava sharing a parcel with rubber. All three surveys use them (73 distinct codes in
    # 2018, and more in the later surveys) and they do not fit int16. The 2018 label raster
    # maps every one of them to 32767, the nodata sentinel align_indices_labels.py drops,
    # keeping only the 109 pure four-digit codes; that convention is reproduced here rather
    # than invented, so the epochs stay comparable. Without it rasterio raises on the cast.
    val = gdf[value_column].to_numpy()
    mixed = val > 32767
    if mixed.any():
        log(f"  {int(mixed.sum()):,} mixed-crop parcels -> 32767 (nodata), the 2018 convention")
        val = np.where(mixed, 32767, val)

    arr = rasterize([(g, v) for g, v in zip(gdf.geometry, val)],
                    out_shape=(height, width), transform=transform,
                    fill=0, dtype="int16")
    prof = {"driver": "GTiff", "height": height, "width": width, "count": 1,
            "dtype": "int16", "crs": crs, "transform": transform, "compress": "lzw"}
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(arr, 1)
    log(f"  {out_tif}: {int((arr != 0).sum()):,} labelled px, {len(np.unique(arr)) - 1} classes")
    return arr


def erode_labels(in_tif, out_tif, iterations=BUFFER_PIXELS):
    """Same 3-pixel per-class erosion as buffer_labels.py."""
    from scipy.ndimage import binary_erosion
    with rasterio.open(in_tif) as src:
        labels, prof = src.read(1), src.profile
    out = np.zeros_like(labels)
    for cls in np.unique(labels):
        if cls == 0:
            continue
        out[binary_erosion(labels == cls, iterations=iterations)] = cls
    prof.update(dtype=rasterio.int16, compress="lzw")
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(out.astype(rasterio.int16), 1)
    log(f"  {out_tif}: {int((out != 0).sum()):,} px survive erosion")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("2020", "2024"):
        raise SystemExit("Usage: python prepare_epoch.py <2020|2024>")
    year = int(sys.argv[1])
    idx_dir = f"./indices_{year}"
    lab = f"./label/label_47PQQ_{year}.tif"
    lab_buf = f"./label/label_47PQQ_{year}_buffered.tif"
    out_npz = f"svm_s2_3date_{year}_features_labels.npz"

    # ---- 1. indices, straight out of the zip -------------------------------
    os.makedirs(idx_dir, exist_ok=True)
    zp = ZIP_TPL.format(year=year)
    if not os.path.exists(zp):
        raise SystemExit(f"missing {zp}")
    for d in DATES:
        vsi = f"/vsizip/{zp}/47PQQ_{year}-{d}.tif"
        if len(glob(f"{idx_dir}/*_47PQQ_{year}-{d}.tif")) == 8:
            log(f"indices for {year}-{d} already present, skipping")
            continue
        log(f"indices for {year}-{d}")
        compute_indices(vsi, idx_dir)
        compute_extra_indices(vsi, idx_dir)

    made = sorted(os.path.basename(f) for f in glob(f"{idx_dir}/*.tif"))
    if len(made) != 24:
        raise SystemExit(f"expected 24 index TIFs in {idx_dir}, found {len(made)}")

    # ---- 2. column order must match training exactly ------------------------
    train_names = list(np.load(TRAIN_NPZ, allow_pickle=True)["feature_names"].astype(str))
    expected = [n.replace("2018-", f"{year}-") for n in train_names]
    if made != expected:
        for a, b in zip(expected, made):
            if a != b:
                log(f"  MISMATCH expected {a!r} got {b!r}")
        raise SystemExit("index column order does not match the training feature order")
    log(f"column order matches training feature_names ({len(made)} columns)")

    # ---- 3. labels, on the 2018 grid ---------------------------------------
    import config
    shp = getattr(config, SURVEY[year])
    if not os.path.exists(shp):
        raise SystemExit(f"missing survey shapefile for {year}")
    log(f"rasterising the {year} survey onto the 2018 grid")
    rasterize_on_reference(shp, "LU_ID_L3", REF_LABEL, lab)
    erode_labels(lab, lab_buf)

    # ---- 4. align, through the same loop the 2018 arm used ------------------
    log("aligning features to labels")
    env = dict(os.environ, ALIGN_LABEL=lab_buf, ALIGN_INDICES=idx_dir, ALIGN_OUT=out_npz)
    r = subprocess.run([sys.executable, "-u", "align_indices_labels.py"], env=env)
    if r.returncode != 0:
        raise SystemExit(f"align failed (exit {r.returncode})")

    d = np.load(f"./aligned_features/{out_npz}", allow_pickle=True)
    log(f"DONE  X={d['X'].shape}  y={d['y'].shape}  "
        f"labelled classes={len(np.unique(d['y']))}")
