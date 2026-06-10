import rasterio
import numpy as np
from collections import Counter

pred_tif = "stage1_predicted_from_indices.tif"

with rasterio.open(pred_tif) as src:
    pred = src.read(1)
unique, counts = np.unique(pred, return_counts=True)
print("Predicted class counts:")
for k, c in zip(unique, counts):
    print(f"  {int(k)} : {int(c)}")
