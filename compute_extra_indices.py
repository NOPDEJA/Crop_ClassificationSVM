# compute_extra_indices.py
import os
from glob import glob
import numpy as np
import rasterio

# Update this mapping to match your raster band order (1-based indices)
BAND_MAPPING = {
    'B02': 1,  # Blue
    'B03': 2,  # Green
    'B04': 3,  # Red
    'B05': 4,  # Red Edge (705nm)
    'B06': 5,
    'B07': 6,
    'B08': 7,  # NIR
    'B8A': 8,
    'B11': 9,  # SWIR1
    'B12': 10  # SWIR2
}

def safe_div(a, b):
    return a / (b + 1e-6)

def compute_extra_indices(raster_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        # read bands as float32
        red = src.read(BAND_MAPPING['B04']).astype('float32')
        green = src.read(BAND_MAPPING['B03']).astype('float32')
        blue = src.read(BAND_MAPPING['B02']).astype('float32')
        nir = src.read(BAND_MAPPING['B08']).astype('float32')
        re = src.read(BAND_MAPPING['B05']).astype('float32')   # red edge candidate
        swir1 = src.read(BAND_MAPPING['B11']).astype('float32')
        swir2 = src.read(BAND_MAPPING['B12']).astype('float32')

        # MSAVI (MSAVI2 form)
        a = (2.0 * nir + 1.0)
        msavi = (a - np.sqrt(np.maximum(a * a - 8.0 * (nir - red), 0.0))) / 2.0

        # MTCI (use red-edge B05)
        # mtci = safe_div((nir - re), (re - red))

        # SWIR/NIR ratio
        swir_nir = safe_div(swir1, nir)

        # SWIR ratio (difference normalized)
        swir_ratio = safe_div((swir1 - swir2), (swir1 + swir2))

        # Optionally add SWIR - NDVI style combos if useful

        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        indices = {
            'MSAVI': msavi,
            # 'MTCI': mtci,
            'SWIR_NIR': swir_nir,
            'SWIR_RATIO': swir_ratio
        }

        base = os.path.splitext(os.path.basename(raster_path))[0]
        for name, arr in indices.items():
            out_path = os.path.join(output_dir, f"{name}_{base}.tif")
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(arr, 1)
            print(f"Saved {name} to {out_path}")

if __name__ == "__main__":
    s2_folder = "./S2_data"
    out_folder = "./indices_extra"
    os.makedirs(out_folder, exist_ok=True)
    s2_files = glob(os.path.join(s2_folder, "*.tif"))
    for f in s2_files:
        compute_extra_indices(f, out_folder)
