# predict_from_index_tifs.py
import rasterio, os
import numpy as np
from joblib import load
import rasterio.windows

MODEL_PATH = "stage1_model.joblib"
SCALER_PATH = "stage1_scaler.joblib"
OUTPUT_TIF = "stage1_predicted_from_indices.tif"
BLOCK = 1024  # tune if needed

# Put your 15 index files here in the SAME ORDER used for training:
index_files = [
    "./indices/NDVI_47PQQ_2018-10-31.tif",
    "./indices/EVI_47PQQ_2018-10-31.tif",
    "./indices/NDWI_47PQQ_2018-10-31.tif",
    "./indices/BSI_47PQQ_2018-10-31.tif",
    "./indices/NDBI_47PQQ_2018-10-31.tif",
    "./indices/NDVI_47PQQ_2018-11-30.tif",
    "./indices/EVI_47PQQ_2018-11-30.tif",
    "./indices/NDWI_47PQQ_2018-11-30.tif",
    "./indices/BSI_47PQQ_2018-11-30.tif",
    "./indices/NDBI_47PQQ_2018-11-30.tif",
    "./indices/NDVI_47PQQ_2018-12-31.tif",
    "./indices/EVI_47PQQ_2018-12-31.tif",
    "./indices/NDWI_47PQQ_2018-12-31.tif",
    "./indices/BSI_47PQQ_2018-12-31.tif",
    "./indices/NDBI_47PQQ_2018-12-31.tif",
]

# load model + scaler
clf = load(MODEL_PATH)
scaler = load(SCALER_PATH)

# open all index rasters
srcs = [rasterio.open(f) for f in index_files]
# check shapes/proj
first = srcs[0]
H, W = first.height, first.width
for s in srcs:
    if (s.height, s.width) != (H, W):
        raise RuntimeError("Index rasters have different shapes — align them first.")
    if s.crs != first.crs:
        raise RuntimeError("Index rasters have different CRS — reproject/align them.")

profile = first.profile.copy()
profile.update(count=1, dtype=rasterio.uint8, compress='lzw')  # predictions small int

with rasterio.open(OUTPUT_TIF, 'w', **profile) as dst:
    for y_off in range(0, H, BLOCK):
        h = min(BLOCK, H - y_off)
        for x_off in range(0, W, BLOCK):
            w = min(BLOCK, W - x_off)
            window = rasterio.windows.Window(x_off, y_off, w, h)
            # read each index band for the window and stack
            bands = []
            for s in srcs:
                arr = s.read(1, window=window).astype(np.float32)
                bands.append(arr)
            # shape (n_bands, h, w)
            stacked = np.stack(bands, axis=0)
            n_bands = stacked.shape[0]
            arr2 = np.moveaxis(stacked, 0, -1).reshape(-1, n_bands)  # (h*w, n_bands)
            # transform and predict
            arr_s = scaler.transform(arr2)
            preds = clf.predict(arr_s).astype(np.uint8)
            out = preds.reshape(h, w)
            dst.write(out, 1, window=window)

for s in srcs:
    s.close()
print("Saved:", OUTPUT_TIF)
