# compute_extra_indices.py
import os
from glob import glob
import numpy as np
import rasterio

# Band order of the S2 composites written by tile_download.py.
# That script stacks bands in sorted() string order, which places
# B11/B12 BEFORE B8A ("B11" < "B8A" as strings). Do not "fix" this
# to the natural order — it must match the files on disk.
BAND_MAPPING = {
    'B02': 1,   # Blue
    'B03': 2,   # Green
    'B04': 3,   # Red
    'B05': 4,   # Red Edge (705nm)
    'B06': 5,
    'B07': 6,
    'B08': 7,   # NIR
    'B11': 8,   # SWIR1
    'B12': 9,   # SWIR2
    'B8A': 10,  # Narrow NIR
}

REFLECTANCE_SCALE = 10000.0  # composites are DN; divide to get BOA reflectance

def safe_div(a, b):
    return a / (b + 1e-6)

def compute_extra_indices(raster_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        # read bands as float32
        red = src.read(BAND_MAPPING['B04']).astype('float32')
        nir = src.read(BAND_MAPPING['B08']).astype('float32')
        swir1 = src.read(BAND_MAPPING['B11']).astype('float32')
        swir2 = src.read(BAND_MAPPING['B12']).astype('float32')

        # nodata pixels (cloud gaps / outside footprint) are 0 in every band
        invalid = (red == 0) & (nir == 0) & (swir1 == 0)

        # scale DN -> reflectance so the MSAVI2 "+1" term is valid
        red /= REFLECTANCE_SCALE
        nir /= REFLECTANCE_SCALE
        swir1 /= REFLECTANCE_SCALE
        swir2 /= REFLECTANCE_SCALE

        # MSAVI (MSAVI2 form)
        a = (2.0 * nir + 1.0)
        msavi = (a - np.sqrt(np.maximum(a * a - 8.0 * (nir - red), 0.0))) / 2.0

        # SWIR/NIR ratio
        swir_nir = safe_div(swir1, nir)

        # SWIR ratio (difference normalized)
        swir_ratio = safe_div((swir1 - swir2), (swir1 + swir2))

        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        indices = {
            'MSAVI': msavi,
            'SWIR_NIR': swir_nir,
            'SWIR_RATIO': swir_ratio
        }

        base = os.path.splitext(os.path.basename(raster_path))[0]
        for name, arr in indices.items():
            arr[invalid | ~np.isfinite(arr)] = np.nan
            out_path = os.path.join(output_dir, f"{name}_{base}.tif")
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(arr, 1)
            print(f"Saved {name} to {out_path}")

if __name__ == "__main__":
    s2_folder = "./S2_data"
    # write to ./indices so align_indices_labels.py picks these up automatically
    out_folder = "./indices"
    s2_files = glob(os.path.join(s2_folder, "*.tif"))
    for f in s2_files:
        compute_extra_indices(f, out_folder)
