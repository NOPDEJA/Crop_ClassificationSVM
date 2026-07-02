"""make_subset_npz.py

Slice the combined DEM+S1+S2 feature NPZ into ablation NPZs using the
feature_names manifest. Same rows, same y, same column order within the
subset — so all runs trained on these share identical train/val/test
pixels (sampling is driven by y with fixed seeds).

Subsets:
  dem_s1   columns ending _dem.tif or containing S1A_IW_GRDH
  s2_only  everything else (the 40 S2 spectral index columns)

Usage:
  python make_subset_npz.py dem_s1
  python make_subset_npz.py s2_only
"""
import sys
import numpy as np

SRC = "./aligned_features/svm_dem_s1_s2_features_labels.npz"
OUT_TPL = "./aligned_features/svm_{subset}_features_labels.npz"

def subset_mask(names, subset):
    is_dem = np.char.endswith(names, "_dem.tif")
    is_s1 = np.char.find(names, "S1A_IW_GRDH") >= 0
    if subset == "dem_s1":
        return is_dem | is_s1
    if subset == "s2_only":
        return ~(is_dem | is_s1)
    raise SystemExit(f"Unknown subset '{subset}' (use dem_s1 or s2_only)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_subset_npz.py <dem_s1|s2_only>")
    subset = sys.argv[1]

    d = np.load(SRC, allow_pickle=True)
    names = d["feature_names"].astype(str)
    keep = subset_mask(names, subset)
    print(f"{subset}: keeping {keep.sum()} of {names.size} features")
    for nm in names[keep]:
        print("  ", nm)

    X = d["X"][:, keep]
    out = OUT_TPL.format(subset=subset)
    np.savez(out, X=X, y=d["y"], feature_names=names[keep])
    print(f"Saved {out}  X shape: {X.shape}")
