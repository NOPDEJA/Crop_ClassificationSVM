import rasterio
import numpy as np
import os
from glob import glob

# Band order of the S2 composites written by tile_download.py.
# That script stacks bands in sorted() string order, which places
# B11/B12 BEFORE B8A ("B11" < "B8A" as strings). Do not "fix" this
# to the natural order — it must match the files on disk.
BAND_MAPPING = {
    'B02': 1,   # Blue
    'B03': 2,   # Green
    'B04': 3,   # Red
    'B05': 4,
    'B06': 5,
    'B07': 6,
    'B08': 7,   # NIR
    'B11': 8,   # SWIR1
    'B12': 9,   # SWIR2
    'B8A': 10,  # Narrow NIR
}

REFLECTANCE_SCALE = 10000.0  # composites are DN; divide to get BOA reflectance

def compute_indices(raster_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        red = src.read(BAND_MAPPING['B04']).astype('float32')
        green = src.read(BAND_MAPPING['B03']).astype('float32')
        blue = src.read(BAND_MAPPING['B02']).astype('float32')
        nir = src.read(BAND_MAPPING['B08']).astype('float32')
        swir1 = src.read(BAND_MAPPING['B11']).astype('float32')
        swir2 = src.read(BAND_MAPPING['B12']).astype('float32')

        # nodata pixels (cloud gaps / outside footprint) are 0 in every band
        invalid = (red == 0) & (green == 0) & (blue == 0) & (nir == 0)

        # scale DN -> reflectance so EVI coefficients are valid
        red /= REFLECTANCE_SCALE
        green /= REFLECTANCE_SCALE
        blue /= REFLECTANCE_SCALE
        nir /= REFLECTANCE_SCALE
        swir1 /= REFLECTANCE_SCALE
        swir2 /= REFLECTANCE_SCALE

        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red + 1e-6)
            evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1.0)
            ndwi = (green - nir) / (green + nir + 1e-6)
            bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-6)
            ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
        # EVI denominator can cross zero at aberrant pixels -> clip to the
        # conventional range; non-finite values become NaN below
        evi = np.clip(evi, -3.0, 3.0)

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
        indices = {'NDVI': ndvi, 'EVI': evi, 'NDWI': ndwi, 'BSI': bsi, 'NDBI': ndbi}

        for name, array in indices.items():
            array[invalid | ~np.isfinite(array)] = np.nan
            out_path = os.path.join(output_dir, f"{name}_{os.path.basename(raster_path)}")
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(array, 1)
            print(f"Saved {name} to {out_path}")


if __name__ == "__main__":
    s2_folder = "./S2_data"
    output_folder = "./indices"
    s2_files = glob(os.path.join(s2_folder, "*.tif"))

    for raster_file in s2_files:
        compute_indices(raster_file, output_folder)
