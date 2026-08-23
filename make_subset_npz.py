"""make_subset_npz.py

Slice the combined DEM+S1+S2 feature NPZ into ablation NPZs using the
feature_names manifest. Same rows, same y, same column order within the
subset — so all runs trained on these share identical train/val/test
pixels (sampling is driven by y with fixed seeds).

Subsets:
  dem_s1    columns ending _dem.tif or containing S1A_IW_GRDH
  s2_only   everything else (the 40 S2 spectral index columns)
  s2_3date  the S2 columns for Oct/Nov/Dec only (24 columns) - the window
            used by the collaborator's XGBoost cascade and the prior RF paper

Usage:
  python make_subset_npz.py dem_s1
  python make_subset_npz.py s2_only
  python make_subset_npz.py s2_3date

X is streamed via a memory-mapped copy of X.npy so the 14.9 GB source never
has to fit in RAM at once.
"""
import os
import sys
import zipfile
import numpy as np

SRC = "./aligned_features/svm_dem_s1_s2_features_labels.npz"
OUT_TPL = "./aligned_features/svm_{subset}_features_labels.npz"
WORKDIR = "./aligned_features/_unpacked"

OCT_DEC = ("2018-10-31", "2018-11-30", "2018-12-31")

def subset_mask(names, subset):
    is_dem = np.char.endswith(names, "_dem.tif")
    is_s1 = np.char.find(names, "S1A_IW_GRDH") >= 0
    is_s2 = ~(is_dem | is_s1)
    if subset == "dem_s1":
        return is_dem | is_s1
    if subset == "s2_only":
        return is_s2
    if subset == "s2_3date":
        in_window = np.zeros(names.size, dtype=bool)
        for d in OCT_DEC:
            in_window |= np.char.find(names, d) >= 0
        return is_s2 & in_window
    raise SystemExit(f"Unknown subset '{subset}' (use dem_s1, s2_only or s2_3date)")


def extract_X(src, workdir):
    """Unpack X.npy from the npz so it can be memory-mapped instead of loaded."""
    path = os.path.join(workdir, "X_src.npy")
    if not os.path.exists(path):
        os.makedirs(workdir, exist_ok=True)
        with zipfile.ZipFile(src) as z, open(path, "wb") as out:
            with z.open("X.npy") as f:
                while True:
                    buf = f.read(64 << 20)
                    if not buf:
                        break
                    out.write(buf)
    return path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_subset_npz.py <dem_s1|s2_only>")
    subset = sys.argv[1]

    d = np.load(SRC, allow_pickle=True)
    names = d["feature_names"].astype(str)
    y = d["y"]
    keep = subset_mask(names, subset)
    print(f"{subset}: keeping {keep.sum()} of {names.size} features")
    for nm in names[keep]:
        print("  ", nm)

    src_X = extract_X(SRC, WORKDIR)
    Xsrc = np.load(src_X, mmap_mode="r")
    print("source X:", Xsrc.shape, Xsrc.dtype)

    cols = np.flatnonzero(keep)
    X = np.empty((Xsrc.shape[0], cols.size), dtype=Xsrc.dtype)
    step = 2_000_000
    for i in range(0, Xsrc.shape[0], step):
        X[i:i + step] = Xsrc[i:i + step][:, cols]
        print(f"  rows {i:,} / {Xsrc.shape[0]:,}", flush=True)

    out = OUT_TPL.format(subset=subset)
    np.savez(out, X=X, y=y, feature_names=names[keep])
    print(f"Saved {out}  X shape: {X.shape}")
