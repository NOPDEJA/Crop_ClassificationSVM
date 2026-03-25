import rasterio
import numpy as np
import os
from glob import glob

# Band mapping (adjust according to your raster)
BAND_MAPPING = {
    'B02': 1,  # Blue
    'B03': 2,  # Green
    'B04': 3,  # Red
    'B05': 4,
    'B06': 5,
    'B07': 6,
    'B08': 7,  # NIR
    'B8A': 8,
    'B11': 9,  # SWIR1
    'B12': 10  # SWIR2
}

def compute_indices(raster_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(raster_path) as src:
        red = src.read(BAND_MAPPING['B04']).astype('float32')
        green = src.read(BAND_MAPPING['B03']).astype('float32')
        blue = src.read(BAND_MAPPING['B02']).astype('float32')
        nir = src.read(BAND_MAPPING['B08']).astype('float32')
        swir1 = src.read(BAND_MAPPING['B11']).astype('float32')
        swir2    = src.read(BAND_MAPPING['B12']).astype('float32')

        ndvi = (nir - red) / (nir + red + 1e-6)
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1e-6)
        ndwi = (green - nir) / (green + nir + 1e-6)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-6)
        ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)

        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        indices = {'NDVI': ndvi, 'EVI': evi, 'NDWI': ndwi, 'BSI': bsi, 'NDBI': ndbi}

        for name, array in indices.items():
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
