"""score_by_churn.py

Split a cross-year score by whether the ground actually moved.

    python score_by_churn.py 2024 s2_2018_3date_v2

`predict_new_epoch.py` reports the transfer score and, separately, how many pixels changed
class between the 2018 survey and the target epoch's. Neither number alone answers the
question the cross-year test is for: a degraded score can mean the model failed to transfer,
or that the land use genuinely changed and the model is being marked against ground truth it
was never wrong about. Scoring the two populations apart separates them.

  unchanged  the 2024 survey agrees with the 2018 survey. Any error here is transfer error:
             the label is what the model was trained to produce, only the reflectance moved.
  changed    the parcel is recorded as a different class in 2024. Errors here are a mix of
             transfer error and the model correctly reporting the 2018 class it learnt.

Both are restricted to pixels labelled in *both* surveys, so the comparison is like for like.
"""
import sys

import numpy as np
import pandas as pd
import rasterio

from config import ARMS
from evaluate_flat_15class import to_flat_true, to_flat_pred, score

if len(sys.argv) != 3:
    raise SystemExit("Usage: python score_by_churn.py <2020|2024> <model_arm>")
year, arm = sys.argv[1], sys.argv[2]
out_dir = f"./runs/xyear_{year}_from_{arm}"

d = np.load(f"./aligned_features/svm_s2_3date_{year}_features_labels.npz", allow_pickle=True)
y = d["y"].astype(np.int32)
s1 = np.load(f"{out_dir}/stage1_pred.npy")
final = np.load(f"{out_dir}/end_to_end_lu_pred.npy")


def flat(tif):
    with rasterio.open(tif) as s:
        return s.read(1).flatten()


l18, lyr = flat("./label/label_47PQQ_buffered.tif"), flat(f"./label/label_47PQQ_{year}_buffered.tif")
valid = lambda f: (f != 0) & (f != 32767)
# rows of this epoch's NPZ, in the same order align_indices_labels.py wrote them
rows_t = np.flatnonzero(valid(lyr))
assert rows_t.size == y.size, f"{rows_t.size} vs {y.size}"
both = valid(l18)[rows_t]                       # also labelled in 2018
same = np.zeros(y.size, dtype=bool)
same[both] = l18[rows_t[both]] == lyr[rows_t[both]]

y_flat, p_flat = to_flat_true(y), to_flat_pred(s1, final)
pops = {"labelled in both": np.flatnonzero(both),
        "unchanged since 2018": np.flatnonzero(both & same),
        "changed since 2018": np.flatnonzero(both & ~same)}
per, ov = [], []
for name, idx in pops.items():
    rows, o = score(y_flat[idx], p_flat[idx], fold_forest=True, tag=name)
    per += rows
    ov.append(o)
    print(f"{name}: {o}", flush=True)

pd.DataFrame(per).to_csv(f"{out_dir}/churn_split_per_class.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(ov).to_csv(f"{out_dir}/churn_split_overall.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200)
w = pd.DataFrame(per).pivot(index="class", columns="population", values="f1").round(4)
print("\nF1 by class:\n" + w.to_string())
