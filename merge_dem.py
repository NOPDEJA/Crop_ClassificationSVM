
import rasterio
from rasterio.merge import merge
import os

files = ['dem/Copernicus_DSM_COG_10_N12_00_E101_00_DEM.tif',
         'dem/Copernicus_DSM_COG_10_N13_00_E101_00_DEM.tif']

srcs = [rasterio.open(f) for f in files]
mosaic, transform = merge(srcs)
profile = srcs[0].profile.copy()
profile.update(transform=transform, width=mosaic.shape[2], height=mosaic.shape[1])

with rasterio.open('dem/dem_47PQQ.tif', 'w', **profile) as dst:
    dst.write(mosaic)

for s in srcs: s.close()
print('Merged DEM saved to dem/dem_47PQQ.tif')
