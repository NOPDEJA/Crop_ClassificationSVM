import rasterio
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# files
pred_file = "stage1_predicted_from_indices.tif"
label_file = "./label/label_47PQQ_buffered.tif"
out_csv = "stage1_evaluation.csv"

# mapping (same as training)
economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
water_code = {4101,4102,4103,4201,4202,4203}

def map_to_super(y_array):
    out = np.zeros_like(y_array, dtype=np.uint8)
    flat = y_array.flatten()
    out_flat = np.zeros_like(flat, dtype=np.uint8)
    econ_mask = np.isin(flat, list(economic_crops))
    water_mask = np.isin(flat, list(water_code))
    out_flat[econ_mask] = 1
    out_flat[water_mask] = 2
    # others (exclude nodata 0,32767)
    others_mask = (~econ_mask) & (~water_mask) & (flat != 0) & (flat != 32767)
    out_flat[others_mask] = 3
    return out_flat.reshape(y_array.shape)

# read rasters (assumes same shape/transform)
with rasterio.open(pred_file) as p, rasterio.open(label_file) as l:
    pred = p.read(1)
    label = l.read(1)

# Ensure shapes match
if pred.shape != label.shape:
    raise SystemExit("Shape mismatch between pred and label rasters. Align them first.")

# map label LU_CODE -> superclasses
label_super = map_to_super(label)

# mask to only pixels where label_super != 0
mask = label_super != 0
y_true = label_super[mask].flatten()
y_pred = pred[mask].flatten()

# metrics
report = classification_report(y_true, y_pred, target_names=["econ","water","others"], output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(out_csv, encoding="utf-8-sig")
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("Overall accuracy:", acc)
print("Confusion matrix:\n", cm)
print(f"Saved detailed report to {out_csv}")
