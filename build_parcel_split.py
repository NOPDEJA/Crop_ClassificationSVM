"""build_parcel_split.py

Parcel-disjoint splits, and a contamination audit of the existing pixel splits.

Every split in this project so far is pixel-level (stage1_weight_scale.py:245,
stage2_weighted.py:243, stage3_new_weight.py:342). Neighbouring pixels of one
parcel are near-duplicates, so train and test rows are not independent and every
conditional metric is inflated. Worse, Stage 3 only samples pixels that Stage 2
routed CORRECTLY (stage3_new_weight.py:317), so the reconstructed "held_out"
population keeps 100% of upstream routing failures against only 15% of the
successes -- it is failure-enriched, not unbiased.

This script produces two things:

  1. PARCEL ID PER NPZ ROW. The parcel survey is rasterised onto the exact label
     grid and reduced with the same mask align_indices_labels.py:27 used, so
     element i is the parcel of NPZ row i.

  2. TWO POPULATIONS:
     (a) `clean`  -- rows in parcels from which NOT ONE pixel was ever fitted.
         This is the only genuinely parcel-disjoint evaluation available for the
         ALREADY-TRAINED models, and needs no retraining.
     (b) a reusable parcel-grouped train/val/test split for the rebuild, so
         every future arm shares one split and stays comparable.

Outputs (in ./splits/):
  parcel_id_row.npy        int32 [n_rows]  parcel of each NPZ row, 0 = none
  parcel_raster.tif        int32 raster, for inspection
  clean_rows_mask.npy      bool  [n_rows]  population (a)
  split_assign.npy         int8  [n_rows]  0=train 1=val 2=test, population (b)
  parcel_split_report.csv  counts and per-class support for both
"""
import os
import time

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

from config import NPZ, LABEL_BUFFERED, SHAPEFILE, RANDOM_STATE, OUT_DIR

# -----------------------
# Config
# -----------------------
OUT_SPLIT_DIR = "./splits"
PARCEL_ID_ROW = f"{OUT_SPLIT_DIR}/parcel_id_row.npy"
PARCEL_RASTER = f"{OUT_SPLIT_DIR}/parcel_raster.tif"
CLEAN_MASK = f"{OUT_SPLIT_DIR}/clean_rows_mask.npy"
SPLIT_ASSIGN = f"{OUT_SPLIT_DIR}/split_assign.npy"
REPORT = f"{OUT_SPLIT_DIR}/parcel_split_report.csv"

# The already-trained arm whose fitted rows define contamination.
TRAINVAL_MASK = f"{OUT_DIR}/trainval_rows_mask.npy"

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2   # test gets the remainder

os.makedirs(OUT_SPLIT_DIR, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if __name__ == "__main__":
    # ── 1. the row mask, replayed exactly as align_indices_labels.py built it ──
    log("reading label raster")
    with rasterio.open(LABEL_BUFFERED) as ds:
        label = ds.read(1)
        height, width = ds.height, ds.width
        transform, crs = ds.transform, ds.crs

    y_flat = label.flatten()
    row_mask = (y_flat != 0) & (y_flat != 32767)
    n_rows = int(row_mask.sum())
    log(f"label grid {height}x{width}, valid rows {n_rows:,}")

    y = np.load(NPZ, allow_pickle=True)["y"]
    assert y.size == n_rows, f"row count mismatch: npz {y.size:,} vs label {n_rows:,}"
    assert np.array_equal(y, y_flat[row_mask]), "row ORDER differs from the npz"
    log("row mask reproduces the npz exactly")

    # ── 2. parcel id raster on that identical grid ────────────────────────────
    log("reading parcel survey")
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    log(f"{len(gdf):,} parcels")

    # 1-based so that 0 stays "no parcel"; int32 because 41k+ overflows int16.
    shapes = ((geom, i) for i, geom in enumerate(gdf.geometry, start=1))
    log("rasterising parcel ids")
    parcel = rasterize(shapes, out_shape=(height, width), transform=transform,
                       fill=0, dtype="int32")

    with rasterio.open(PARCEL_RASTER, "w", driver="GTiff", height=height,
                       width=width, count=1, dtype="int32", crs=crs,
                       transform=transform, compress="lzw") as dst:
        dst.write(parcel, 1)

    parcel_row = parcel.flatten()[row_mask]
    del parcel, y_flat
    np.save(PARCEL_ID_ROW, parcel_row)
    n_orphan = int((parcel_row == 0).sum())
    log(f"rows with no parcel: {n_orphan:,} ({n_orphan / n_rows:.2%})")

    parcels = np.unique(parcel_row)
    parcels = parcels[parcels != 0]
    log(f"parcels carrying at least one row: {len(parcels):,}")

    # ── 3. contamination audit against the existing pixel split ───────────────
    if os.path.exists(TRAINVAL_MASK):
        trainval = np.load(TRAINVAL_MASK)
        assert trainval.size == n_rows
        touched = np.unique(parcel_row[trainval])
        touched = touched[touched != 0]
        log(f"parcels contributing >=1 fitted pixel: {len(touched):,} "
            f"of {len(parcels):,} ({len(touched) / len(parcels):.1%})")

        dirty = np.zeros(int(parcel_row.max()) + 1, dtype=bool)
        dirty[touched] = True
        clean_mask = (parcel_row != 0) & ~dirty[parcel_row]
        np.save(CLEAN_MASK, clean_mask)
        log(f"CLEAN rows (parcel never fitted): {int(clean_mask.sum()):,} "
            f"({clean_mask.mean():.2%} of all rows)")
        log(f"  for comparison, pixel-level ~trainval kept "
            f"{int((~trainval).sum()):,} rows ({(~trainval).mean():.2%})")
    else:
        log(f"WARNING: {TRAINVAL_MASK} missing — skipping contamination audit")
        clean_mask = None

    # ── 4. reusable parcel-grouped split for the rebuild ──────────────────────
    # Stratify by each parcel's dominant LU code so rare crops are represented in
    # all three folds; a parcel is atomic and lands wholly in one fold.
    log("building parcel-grouped train/val/test split")
    order = np.argsort(parcel_row, kind="stable")
    sorted_pid = parcel_row[order]
    starts = np.flatnonzero(np.r_[True, sorted_pid[1:] != sorted_pid[:-1]])
    groups = np.split(order, starts[1:])
    if sorted_pid[starts[0]] == 0:      # drop the no-parcel group
        groups = groups[1:]

    dominant = np.array([np.bincount(y[g]).argmax() for g in groups])
    rng = np.random.default_rng(RANDOM_STATE)
    assign = np.full(n_rows, -1, dtype=np.int8)

    for code in np.unique(dominant):
        idx = np.flatnonzero(dominant == code)
        rng.shuffle(idx)
        n_tr = int(round(TRAIN_FRAC * idx.size))
        n_va = int(round(VAL_FRAC * idx.size))
        for fold, chunk in enumerate((idx[:n_tr], idx[n_tr:n_tr + n_va],
                                      idx[n_tr + n_va:])):
            for gi in chunk:
                assign[groups[gi]] = fold

    np.save(SPLIT_ASSIGN, assign)
    for fold, name in enumerate(("train", "val", "test")):
        n = int((assign == fold).sum())
        log(f"  {name:<5} {n:>12,} rows ({n / n_rows:.2%})")
    log(f"  unassigned (no parcel) {int((assign == -1).sum()):,}")

    # ── 5. report ─────────────────────────────────────────────────────────────
    codes = np.unique(y)
    with open(REPORT, "w", encoding="utf-8-sig") as f:
        f.write("lu_code,all_rows,clean_rows,train_rows,val_rows,test_rows\n")
        for c in codes:
            m = y == c
            clean_n = int((m & clean_mask).sum()) if clean_mask is not None else -1
            f.write(f"{int(c)},{int(m.sum())},{clean_n},"
                    f"{int((m & (assign == 0)).sum())},"
                    f"{int((m & (assign == 1)).sum())},"
                    f"{int((m & (assign == 2)).sum())}\n")
    log(f"wrote {REPORT}")
    log("done")
