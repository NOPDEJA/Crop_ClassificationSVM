import rasterio
import numpy as np

input_file = "C:/Users/Nop/OneDrive/เดสก์ท็อป/Works/MINEWORK/ASSIGNMENT/college/Project and Research with Prof/LDD/LDD_Scripts/S2_data/47PQQ_2018-11-30.tif"
output_path = "C:/Users/Nop/OneDrive/เดสก์ท็อป/Works/MINEWORK/ASSIGNMENT/college/Project and Research with Prof/LDD/LDD_Scripts/S2_data/47PQQ_2018-11-30_ndvi.tif"

# Open multiband image
with rasterio.open(input_file) as src:
    red = src.read(3).astype("float32")   # B04
    nir = src.read(7).astype("float32")   # B08
    meta = src.meta.copy()

# Avoid divide-by-zero
ndvi_numerator = nir - red
ndvi_denominator = nir + red
ndvi = ndvi_numerator / ndvi_denominator
ndvi = np.clip(ndvi, -1, 1)  # Ensure within NDVI range

# Update metadata
meta.update({
    "count": 1,
    "dtype": "float32"
})

# Save NDVI to output file
with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(ndvi, 1)
print(f"NDVI saved to {output_path}")