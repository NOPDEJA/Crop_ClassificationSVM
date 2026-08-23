"""compare_epochs.py

The 2018, 2020 and 2024 epochs on one identical population.

    python compare_epochs.py

Each epoch is otherwise scored on its own survey's pixels -- 24.3 M for 2018, 23.3 M for
2020, 21.0 M for 2024, with different class mixes (rubber is 49 % of the 2020 population
against 45.5 % of the 2024 one). Accuracy across different class mixes is not a comparison,
the same trap as the cross-arm splits in S2_SVM_ANALYSIS.md 6.9. Here every epoch is scored
on the pixels labelled in all three surveys that carry the SAME class in all three, so the
only thing varying is which year's reflectance the 2018-trained cascade saw.

Two populations, because the 2018 arm fitted on some of these rows and the others did not:

  stable         every pixel stable across the three surveys
  stable, unseen the subset the v2 arm never fitted on -- the only row on which the 2018
                 column is honest, and the one to quote
"""
import numpy as np
import pandas as pd
import rasterio

from evaluate_flat_15class import to_flat_true, to_flat_pred, score

ARM = "s2_2018_3date_v2"
TAG = "s2_3date_v2"

f = lambda t: rasterio.open(t).read(1).flatten()
valid = lambda a: (a != 0) & (a != 32767)
lab = {2018: f("./label/label_47PQQ_buffered.tif"),
       2020: f("./label/label_47PQQ_2020_buffered.tif"),
       2024: f("./label/label_47PQQ_2024_buffered.tif")}
stable = (valid(lab[2018]) & valid(lab[2020]) & valid(lab[2024])
          & (lab[2018] == lab[2020]) & (lab[2018] == lab[2024]))

r18 = np.flatnonzero(valid(lab[2018]))
unseen = np.zeros(stable.size, dtype=bool)
unseen[r18[~np.load(f"./runs/{ARM}/trainval_rows_mask.npy")]] = True
print(f"stable across all three surveys: {int(stable.sum()):,} px"
      f"   of which never fitted on: {int((stable & unseen).sum()):,}")

ov, per = [], {}
for yr in (2018, 2020, 2024):
    rows_y = np.flatnonzero(valid(lab[yr]))
    if yr == 2018:
        y = np.load("./aligned_features/svm_s2_3date_features_labels.npz", allow_pickle=True)["y"]
        s1 = np.load(f"./runs/{ARM}/stage1_{TAG}_pred.npy")
        fin = np.load(f"./runs/{ARM}/end_to_end_lu_pred.npy")
    else:
        d = f"./runs/xyear_{yr}_from_{ARM}"
        y = np.load(f"./aligned_features/svm_s2_3date_{yr}_features_labels.npz", allow_pickle=True)["y"]
        s1, fin = np.load(f"{d}/stage1_pred.npy"), np.load(f"{d}/end_to_end_lu_pred.npy")
    y = y.astype(np.int32)
    assert rows_y.size == y.size, f"{yr}: {rows_y.size} vs {y.size}"
    yf, pf = to_flat_true(y), to_flat_pred(s1, fin)
    for name, mask in (("stable", stable), ("stable, unseen", stable & unseen)):
        i = np.flatnonzero(mask[rows_y])
        rows, o = score(yf[i], pf[i], fold_forest=True, tag=f"{yr} {name}")
        o["epoch"], o["scope"] = yr, name
        ov.append(o)
        if name == "stable, unseen":
            per[yr] = {r["class"]: round(r["f1"], 4) for r in rows}

pd.set_option("display.width", 200)
o = pd.DataFrame(ov)
print("\n" + o[["epoch", "scope", "n_pixels", "accuracy", "kappa", "f1_weighted", "f1_macro"]].to_string(index=False))
print("\nF1 by class, stable+unseen:\n" + pd.DataFrame(per).to_string())
o.to_csv("./runs/xyear/epoch_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(per).to_csv("./runs/xyear/epoch_comparison_per_class.csv", encoding="utf-8-sig")
