"""
s1_compute_features.py

Computes SAR-derived classification features from preprocessed
Sentinel-1 GRD GeoTIFFs (output of s1_preprocess_snap.py).

Processes in blocks to avoid memory errors on large tiles.

Features computed per file:
  VV, VH, VV_VH_RATIO, RVI, RFDI, DPR
(VVVH_DIFF was removed — it was an exact duplicate of VV_VH_RATIO)
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from glob import glob
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "./S1_processed"
OUTPUT_DIR  = "./indices"
VV_BAND     = 1
VH_BAND     = 2
NODATA_DB   = -9999.0
DB_FLOOR    = -30.0
DB_CEIL     = 5.0
BLOCK_ROWS  = 512        # number of rows per processing block — lower = less RAM
                         # 512 rows × 28000 cols × 6 features ≈ 350 MB peak
                         # reduce to 256 if you still get memory errors
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = ["VV", "VH", "VV_VH_RATIO", "RVI", "RFDI", "DPR"]


def db_to_linear(arr):
    return np.power(10.0, arr / 10.0)


def safe_div(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-9, a / b, fill)


def compute_block(vv_db, vh_db):
    """
    Compute all 7 SAR features for one block.
    Inputs are float32 arrays of shape (rows, cols).
    Returns dict feature_name → float32 array.
    """
    vv_db = np.clip(vv_db, DB_FLOOR, DB_CEIL)
    vh_db = np.clip(vh_db, DB_FLOOR, DB_CEIL)

    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    features = {}
    features["VV"]          = vv_db.astype(np.float32)
    features["VH"]          = vh_db.astype(np.float32)
    features["VV_VH_RATIO"] = (vv_db - vh_db).astype(np.float32)

    # RVI = 4*VH / (VV+VH)  — linear domain
    features["RVI"]  = np.clip(
        safe_div(4.0 * vh_lin, vv_lin + vh_lin, fill=0.0), 0.0, 1.0
    ).astype(np.float32)

    # RFDI = (VV-VH) / (VV+VH) — linear domain
    features["RFDI"] = np.clip(
        safe_div(vv_lin - vh_lin, vv_lin + vh_lin, fill=0.0), -1.0, 1.0
    ).astype(np.float32)

    # DPR = VH/VV — linear domain
    features["DPR"]      = safe_div(vh_lin, vv_lin, fill=0.0).astype(np.float32)

    # Free intermediate arrays immediately
    del vv_lin, vh_lin
    return features


def already_done(stem):
    """Return True if all 7 feature GeoTIFFs for this stem already exist."""
    return all(
        os.path.exists(os.path.join(OUTPUT_DIR, f"{name}_{stem}.tif"))
        for name in FEATURE_NAMES
    )


def process_file(tif_path):
    stem = Path(tif_path).stem

    # ── Resume logic: skip if all outputs already written ─────────────────────
    if already_done(stem):
        print(f"\n  [SKIP] All features already exist for: {Path(tif_path).name}")
        return [os.path.join(OUTPUT_DIR, f"{name}_{stem}.tif")
                for name in FEATURE_NAMES]

    print(f"\nProcessing: {Path(tif_path).name}")

    with rasterio.open(tif_path) as src:
        if src.count < 2:
            print(f"  WARNING: expected 2 bands, found {src.count} — skipping.")
            return []

        profile  = src.profile.copy()
        nodata   = src.nodata
        width    = src.width
        height   = src.height
        print(f"  Raster size: {width} cols × {height} rows  "
              f"({width*height/1e6:.1f} M pixels)")

        # ── Open one output file per feature ──────────────────────────────────
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="float32", compress="lzw",
                           nodata=np.nan, bigtiff="YES")

        out_paths   = {}
        out_handles = {}
        for name in FEATURE_NAMES:
            fname = f"{name}_{stem}.tif"
            fpath = os.path.join(OUTPUT_DIR, fname)
            out_paths[name]   = fpath
            out_handles[name] = rasterio.open(fpath, "w", **out_profile)

        # ── Process row-by-row blocks ─────────────────────────────────────────
        n_blocks = (height + BLOCK_ROWS - 1) // BLOCK_ROWS
        for bi, row_off in enumerate(range(0, height, BLOCK_ROWS)):
            rows = min(BLOCK_ROWS, height - row_off)
            win  = Window(0, row_off, width, rows)

            vv_raw = src.read(VV_BAND, window=win).astype(np.float32)
            vh_raw = src.read(VH_BAND, window=win).astype(np.float32)

            # Replace nodata with NaN
            if nodata is not None:
                vv_raw[vv_raw == nodata] = np.nan
                vh_raw[vh_raw == nodata] = np.nan

            valid = (~np.isnan(vv_raw)) & (~np.isnan(vh_raw))

            # Compute features for this block
            feats = compute_block(vv_raw, vh_raw)
            del vv_raw, vh_raw

            # Apply valid mask
            for name in FEATURE_NAMES:
                feats[name][~valid] = np.nan
                out_handles[name].write(feats[name], 1, window=win)
            del feats, valid

            if (bi + 1) % 10 == 0 or (bi + 1) == n_blocks:
                print(f"  Block {bi+1}/{n_blocks} done", end="\r")

        print()   # newline after block progress

        # Close all outputs
        for name in FEATURE_NAMES:
            out_handles[name].close()
            print(f"  Saved {name} → {Path(out_paths[name]).name}")

    return list(out_paths.values())


if __name__ == "__main__":
    tif_files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))

    if not tif_files:
        print(f"No GeoTIFFs found in '{INPUT_DIR}/'.\n"
              "Run s1_preprocess_snap.py first.")
        import sys; sys.exit(1)

    print(f"Found {len(tif_files)} preprocessed S1 file(s).")
    all_outputs = []

    for tif_path in tif_files:
        try:
            outputs = process_file(tif_path)
            all_outputs.extend(outputs)
        except Exception as e:
            print(f"  ERROR processing {tif_path}: {e}")
            continue

    print(f"\n{'='*55}")
    print(f"Total feature files written : {len(all_outputs)}")
    print(f"Output directory            : {OUTPUT_DIR}/")
    print("\nNext step: python align_indices_labels.py")