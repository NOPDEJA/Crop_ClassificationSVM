"""Evaluate Stage-1 predictions against the buffered labels.

stage1_s1_dem_pred.npy is row-aligned with the y array in the training NPZ
(both come from the labeled pixels of label_47PQQ_buffered.tif), so the
comparison is done directly on those arrays — no raster needed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# files
from config import NPZ, OUT_DIR, STAGE1_PRED

pred_file = STAGE1_PRED
npz_file = NPZ
out_csv = f"{OUT_DIR}/stage1_evaluation.csv"

# mapping (same as training: econ=1, water=2, others=3, forest=4)
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}
forest_code = {3100,3101,3200,3201,3300,3301,3401,3501}

def map_to_super(y_array):
    out = np.zeros_like(y_array, dtype=np.uint8)
    econ_mask = np.isin(y_array, list(economic_crops))
    water_mask = np.isin(y_array, list(water_code))
    forest_mask = np.isin(y_array, list(forest_code))
    out[econ_mask] = 1
    out[water_mask] = 2
    out[forest_mask] = 4
    # others (exclude nodata 0, 32767)
    others_mask = (~econ_mask) & (~water_mask) & (~forest_mask) & (y_array != 0) & (y_array != 32767)
    out[others_mask] = 3
    return out

y_lu = np.load(npz_file)["y"].astype(np.int32)
y_pred = np.load(pred_file)

if y_pred.shape[0] != y_lu.shape[0]:
    raise SystemExit(f"Length mismatch: pred={y_pred.shape[0]} labels={y_lu.shape[0]}. "
                     "Regenerate stage1 predictions from the current NPZ.")

y_true = map_to_super(y_lu)
mask = y_true != 0
y_true, y_pred = y_true[mask], y_pred[mask]

labels = [1, 2, 3, 4]
target_names = ["econ", "water", "others", "forest"]

report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(out_csv, encoding="utf-8-sig")
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=labels)

print("Overall accuracy:", acc)
print("Confusion matrix (rows=true, cols=pred; order econ,water,others,forest):\n", cm)
print(f"Saved detailed report to {out_csv}")
