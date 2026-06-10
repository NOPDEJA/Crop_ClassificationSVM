"""
compute_dem_features.py

Computes terrain features from a DEM and reprojects/resamples them
to match your Sentinel-2 label raster resolution (10m).

Features computed:
  - Elevation      : raw DEM values
  - Slope          : steepness in degrees
  - Aspect         : compass direction of slope (0–360)
  - TPI            : Topographic Position Index (local relief)
  - Roughness      : local elevation variability

Output:
  One GeoTIFF per feature saved to ./indices/ (same folder as spectral indices)
  so align_indices_labels.py picks them up automatically.

Requirements:
  pip install rasterio numpy scipy richdem
  (richdem is optional — used for slope/aspect; falls back to numpy if absent)

Usage:
  1. Set DEM_PATH to your downloaded DEM .tif
  2. Set LABEL_PATH to your label raster (used as the alignment reference)
  3. Run: python compute_dem_features.py
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile
from scipy.ndimage import uniform_filter, generic_filter

# ── Config ───────────────────────────────────────────────────────────────────
DEM_PATH    = "./dem/dem_47PQQ.tif"          # your input DEM file
LABEL_PATH  = "./label/label_47PQQ_buffered.tif"  # alignment reference
OUTPUT_DIR  = "./indices"                    # same folder as spectral indices
TPI_RADIUS  = 15                             # pixels; controls TPI neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: reproject DEM to match label raster ───────────────────────────────
def reproject_dem_to_reference(dem_path: str, ref_path: str) -> np.ndarray:
    """
    Reproject and resample DEM to exactly match the reference raster
    (same CRS, transform, width, height).
    Returns a float32 numpy array (H, W) and the reference profile.
    """
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(dem_path) as dem_src:
        dem_data = np.empty((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=dem_src.read(1).astype(np.float32),
            destination=dem_data,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=dem_src.nodata,
            dst_nodata=np.nan,
        )

    print(f"DEM reprojected → shape {dem_data.shape}, "
          f"range [{np.nanmin(dem_data):.1f}, {np.nanmax(dem_data):.1f}] m")
    return dem_data, ref_profile


# ── Feature computations ──────────────────────────────────────────────────────
def compute_slope_aspect(dem: np.ndarray, cell_size: float = 10.0):
    """
    Compute slope (degrees) and aspect (degrees 0–360) from DEM.
    Uses a central-difference gradient (same as GIS software).
    cell_size: pixel size in metres.
    """
    # Pad edges to avoid border artefacts
    padded = np.pad(dem, 1, mode='edge')

    dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * cell_size)
    dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    # Convert to compass bearing (0 = North, clockwise)
    aspect_compass = 90.0 - aspect_deg
    aspect_compass = np.where(aspect_compass < 0, aspect_compass + 360.0, aspect_compass)
    # Flat areas → assign 0
    flat = slope_deg < 0.01
    aspect_compass[flat] = 0.0

    return slope_deg.astype(np.float32), aspect_compass.astype(np.float32)


def compute_tpi(dem: np.ndarray, radius: int = 15) -> np.ndarray:
    """
    Topographic Position Index = elevation − mean elevation in neighbourhood.
    Positive  → ridges/hilltops; Negative → valleys/depressions.
    radius: half-window size in pixels.
    """
    size = 2 * radius + 1
    mean_local = uniform_filter(dem, size=size, mode='nearest')
    tpi = (dem - mean_local).astype(np.float32)
    return tpi


def compute_roughness(dem: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Roughness = max − min elevation in a local window.
    Captures fine-scale terrain variability.
    """
    size = 2 * radius + 1
    def range_func(arr):
        return arr.max() - arr.min()
    roughness = generic_filter(dem, function=range_func,
                               size=size, mode='nearest').astype(np.float32)
    return roughness


# ── Save feature to GeoTIFF ───────────────────────────────────────────────────
def save_feature(array: np.ndarray, profile: dict, name: str, output_dir: str):
    """Write a single-band float32 GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype='float32',
        compress='lzw',
        nodata=np.nan,
    )
    # Strip multi-band count if label has count>1
    out_path = os.path.join(output_dir, f"{name}_dem.tif")
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(array, 1)
    print(f"Saved {name} → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Reproject DEM to match label raster
    dem, ref_profile = reproject_dem_to_reference(DEM_PATH, LABEL_PATH)

    # Detect pixel size from reference profile (assume square pixels)
    cell_size = abs(ref_profile['transform'].a)   # metres (e.g. 10.0 for S2)
    print(f"Cell size: {cell_size} m")

    # 2. Compute features
    print("Computing slope and aspect...")
    slope, aspect = compute_slope_aspect(dem, cell_size=cell_size)

    print(f"Computing TPI (radius={TPI_RADIUS} px = {TPI_RADIUS*cell_size:.0f} m)...")
    tpi = compute_tpi(dem, radius=TPI_RADIUS)

    print("Computing roughness...")
    roughness = compute_roughness(dem, radius=3)

    # 3. Save — same naming convention as compute_indices.py
    save_feature(dem,       ref_profile, "ELEVATION", OUTPUT_DIR)
    save_feature(slope,     ref_profile, "SLOPE",     OUTPUT_DIR)
    save_feature(aspect,    ref_profile, "ASPECT",    OUTPUT_DIR)
    save_feature(tpi,       ref_profile, "TPI",       OUTPUT_DIR)
    save_feature(roughness, ref_profile, "ROUGHNESS", OUTPUT_DIR)

    print("\nAll DEM features saved. They will be picked up automatically")
    print(f"by align_indices_labels.py from '{OUTPUT_DIR}/'.")