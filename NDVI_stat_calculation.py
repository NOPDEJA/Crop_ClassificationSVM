import rasterio 
import numpy as np
import os
import csv
from scipy.stats import mode as stats_mode

# Input files
ndvi_file = "C:/Users/Nop/OneDrive/เดสก์ท็อป/Works/MINEWORK/ASSIGNMENT/college/Project and Research with Prof/LDD/LDD_Scripts/S2_data/47PQQ_2018-11-30_ndvi.tif"
label_file = "C:/Users/Nop/OneDrive/เดสก์ท็อป/Works/MINEWORK/ASSIGNMENT/college/Project and Research with Prof/LDD/LDD_Scripts/label/label_47PQQ.tif"
output_csv = "average_ndvi_by_crop.csv"

# Read NDVI
with rasterio.open(ndvi_file) as ndvi_src:
    ndvi = ndvi_src.read(1).astype("float32")
    ndvi_nodata = ndvi_src.nodata

# Read Labels
with rasterio.open(label_file) as label_src:
    label = label_src.read(1)
    label_nodata = label_src.nodata

# Mask nodata
if ndvi_nodata is not None:
    ndvi[ndvi == ndvi_nodata] = np.nan

if label_nodata is not None:
    label[label == label_nodata] = 0  # Treat label nodata as background

# Get unique LU codes (excluding background = 0)
unique_labels = np.unique(label)
unique_labels = unique_labels[unique_labels != 0]

# Prepare results list
results = []

for lu_code in unique_labels:
    mask = (label == lu_code)
    ndvi_values = ndvi[mask]
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]  # Remove NaN

    if ndvi_values.size > 3000:
        ndvi_values = np.random.choice(ndvi_values, size=3000, replace=False)


    if ndvi_values.size > 0:
        mode_result = stats_mode(ndvi_values, nan_policy='omit')
        if mode_result.count > 0:
            mode_val = float(mode_result.mode)
        else:
            mode_val = None
        stats = {
            'LU_code': int(lu_code),
            'Average_NDVI': round(np.mean(ndvi_values), 4),
            'Median_NDVI': round(np.median(ndvi_values), 4),
            'Mode_NDVI': round(mode_val, 4) if mode_val is not None else None,
            'STD_NDVI': round(np.std(ndvi_values), 4),
            'Min_NDVI': round(np.min(ndvi_values), 4),
            'Max_NDVI': round(np.max(ndvi_values), 4)
        }

        results.append(stats)
    else:
        print(f"⚠️ LU_code {lu_code} has no valid NDVI pixels.")

# Ensure output directory exists
output_dir = os.path.dirname(output_csv)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Write results to CSV
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        'LU_code', 'Average_NDVI', 'Median_NDVI', 'Mode_NDVI',
        'STD_NDVI', 'Min_NDVI', 'Max_NDVI'
    ])
    writer.writeheader()
    writer.writerows(results)

print("✅ NDVI statistics saved to:", output_csv)
