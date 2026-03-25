import math

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds


def rasterize_polygons_to_raster(gdf, value_column, output_raster, resolution=10):
    """
    Rasterize polygons in a GeoDataFrame into a raster using values from a specific column.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing polygons.
        value_column (str): Column name to use for raster values.
        output_raster (str): Path to save the output raster file.
        resolution (float): Pixel size of the output raster.
    """
    # Ensure GeoDataFrame is in a projected CRS
    if gdf.crs.is_geographic:
        raise ValueError("GeoDataFrame must be in a projected CRS. Reproject to UTM or another projection.")

    # Get the bounding box of the GeoDataFrame
    minx, miny, maxx, maxy = gdf.total_bounds

    # Adjust bounds to align with resolution
    adjusted_minx = math.floor(minx / resolution) * resolution
    adjusted_miny = math.floor(miny / resolution) * resolution
    adjusted_maxx = math.ceil(maxx / resolution) * resolution
    adjusted_maxy = math.ceil(maxy / resolution) * resolution

    # Define raster dimensions and transformation
    width = int((adjusted_maxx - adjusted_minx) / resolution)
    height = int((adjusted_maxy - adjusted_miny) / resolution)
    transform = from_bounds(adjusted_minx, adjusted_miny, adjusted_maxx, adjusted_maxy, width, height)

    # Prepare geometries and values for rasterization
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]

    # Rasterize the polygons
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,  # Default value for areas with no polygons
        dtype='int16'  # Data type of the raster
    )

    # Save raster to file
    with rasterio.open(
            output_raster,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=raster.dtype,
            crs=gdf.crs,
            transform=transform
    ) as dst:
        dst.write(raster, 1)

if __name__ == "__main__":
    # Example usage
    gdf = gpd.read_file(./LU_RYG_2561.shp")
    rasterize_polygons_to_raster(gdf, value_column='LU_ID_L3', output_raster='lu_raster.tif', resolution=10)
