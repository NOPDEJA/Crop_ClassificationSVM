import geopandas as gpd
from shapely.geometry import box, Point
from mgrs import MGRS
import numpy as np


def extract_mgrs_tiles(shape, mgrs_precision=0):


    # Initialize MGRS converter
    mgrs_converter = MGRS()

    # Collect MGRS tiles
    mgrs_tiles = set()

    geometry = shape.iloc[0].geometry
    # Get bounds of the geometry
    minx, miny, maxx, maxy = geometry.bounds
    # Define a step (adjust for precision; smaller steps give better coverage)
    step = 0.01  # Degrees of latitude/longitude

    # Create a grid of points within bounds
    latitudes = np.arange(miny, maxy, step)
    longitudes = np.arange(minx, maxx, step)

    for lat in latitudes:
        for lon in longitudes:
            point = Point(lon, lat)

            # Check if point intersects the geometry
            if geometry.contains(point):
                # Convert point to MGRS
                mgrs_tile = mgrs_converter.toMGRS(lat, lon, MGRSPrecision=mgrs_precision)
                mgrs_tiles.add(mgrs_tile)

    return mgrs_tiles

if __name__ == "__main__":
    # Example usage
    shapefile_path = "C:/Users/Nop/OneDrive/เดสก์ท็อป/Works/MINEWORK/ASSIGNMENT/college/Project and Research with Prof/LDD/LDD_Scripts/Landuse_Rayoung67/LU_RYG_2567.shp"
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure the shapefile is in WGS84 (EPSG:4326)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    province_shape = gdf[gdf["PROV_NAME"] == "RAYONG"]

    mgrs_tiles = extract_mgrs_tiles(province_shape)

    print("Overlapping MGRS Tiles:")
    print(mgrs_tiles)