import rasterio as rio
import numpy as np
from datetime import datetime
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import math
from affine import Affine
from rasterio.io import DatasetReader
from rasterio.warp import (
    calculate_default_transform,
    Resampling,
    reproject,
    transform_bounds,
)
from rasterio.windows import Window


def match_to_reference(
        source: DatasetReader,
        reference: DatasetReader,
        resampling_method: Resampling,
) -> MemoryFile:
    """
    Match a raster to a given reference raster, i.e. reproject to the reference CRS, resample to
    the reference resolution, match the reference transform and clip to the reference bounds.

    Args:
        source: Raster to be matched.
        reference: Raster used as reference for the matching process.
        resampling_method: Resampling method to be used during matching.

    Returns:
        matched_file. Matched source raster, clipped to the reference bounds.
    """

    # Defensive check for CRS presence
    if source.crs is None or reference.crs is None:
        raise ValueError(f"Invalid CRS detected in match_to_reference(): source.crs={source.crs}, reference.crs={reference.crs}")

    # outermost bounds (not necessarily grid aligned) of the source raster in the reference crs
    left, bottom, right, top = transform_bounds(
        source.crs, reference.crs, *source.bounds
    )

    # align to reference grid
    ref_left = reference.bounds.left
    ref_bottom = reference.bounds.bottom

    left = ref_left - math.ceil((ref_left - left) / reference.res[0]) * reference.res[0]
    right = (
            ref_left + math.ceil((right - ref_left) / reference.res[0]) * reference.res[0]
    )
    bottom = (
            ref_bottom
            - math.ceil((ref_bottom - bottom) / reference.res[1]) * reference.res[1]
    )
    top = (
            ref_bottom + math.ceil((top - ref_bottom) / reference.res[1]) * reference.res[1]
    )

    # and intersect with reference raster bounds
    left = max(left, reference.bounds.left)
    bottom = max(bottom, reference.bounds.bottom)
    right = min(right, reference.bounds.right)
    top = min(top, reference.bounds.top)

    if left >= right or bottom >= top:
        msg = "Source and reference rasters do not intersect."
        raise ValueError(msg)

    # reproject
    a, b, c, d, e, f, *_ = reference.transform
    matched_transform = Affine(a, b, left, d, e, top)

    nodata = source.nodata
    if nodata is None and source.profile["dtype"] == "float32":
        nodata = np.nan

    matched_profile = source.profile | {
        "crs": reference.crs,
        "transform": matched_transform,
        "width": round((right - left) / reference.res[0]),
        "height": round((top - bottom) / reference.res[1]),
        "resolution": reference.res,
        "nodata": nodata,
    }

    matched_memfile = MemoryFile()
    with matched_memfile.open(**matched_profile) as matched_ds:
        reproject(
            source=rio.band(source, range(1, source.count + 1)),
            destination=rio.band(matched_ds, range(1, source.count + 1)),
            resampling=resampling_method,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

    return matched_memfile


def reproject_raster(raster, profile, resampling_method):
    reproject_memfile = MemoryFile()
    with reproject_memfile.open(**profile) as reproject_ds:
        reproject(
            source=rio.band(raster, range(1, raster.count + 1)),
            destination=rio.band(reproject_ds, range(1, raster.count + 1)),
            resampling=resampling_method,
            src_nodata=raster.nodata,
            dst_nodata=raster.nodata,
        )
    return reproject_memfile


def reference_overlap(matched: DatasetReader, reference: DatasetReader) -> Window:
    """
    Finds the overlapping extent of a reference raster with a matched raster. Used as a
    follow-up to match_to_reference() in order to also retrieve the overlap of the reference.

    Args:
        matched: Raster dataset that was matched to the reference in
         match_to_reference().
        reference: Raster dataset acting as the reference during matching.

    Returns:
        overlap_window. Window indicating the region of reference overlapping with matched.
    """

    # Extract source raster bounds
    left, bottom, right, top = matched.bounds

    # Calculate top-left corner coordinates of the overlap in reference pixel coordinates
    row_offset, column_offset = (
        round(x) for x in rio.transform.rowcol(reference.transform, left, top)
    )

    # Calculate width and height of the overlap in reference pixel coordinates
    width = min(matched.width, reference.width - column_offset)
    height = min(matched.height, reference.height - row_offset)

    # check for overlap
    if (width < 0) or (height < 0) or (row_offset < 0) or (column_offset < 0):
        msg = "Input rasters to reference_overlap() do not overlap, no output produced!"
        raise ValueError(msg)

    overlap_window = Window(column_offset, row_offset, width, height)
    return overlap_window


class Sentinel2Raster():
    def __init__(self, src_zip, file_name, band_files, boa_effect, scaling_factor):
        self.src_zip = src_zip
        self.band_files = band_files
        self.file_name = file_name
        self.boa_effect = boa_effect
        self.scaling_factor = scaling_factor
        self.__band = self.__reproject_bands()
        self.__band["VAL"] = self.__create_valid_mask()


    def __reproject_bands(self):
        extracted_band = {}
        min_res = 10.0
        ref_raster = None

        for band, file in self.band_files.items():
            band_raster = rio.open(self.src_zip.open_raster(file))

            # Assign CRS if missing (assume EPSG:32647 for your Sentinel-2 data)
            if band_raster.crs is None:
                expected_crs = CRS.from_epsg(32647)
                profile = band_raster.profile.copy()
                profile.update(crs=expected_crs)

                memfile = MemoryFile()
                with memfile.open(**profile) as mem_ds:
                    mem_ds.write(band_raster.read())
                band_raster.close()
                band_raster = memfile.open()

            extracted_band[band] = band_raster

            if band_raster.res[0] == min_res and ref_raster is None:
                ref_raster = band_raster

        for band, raster in extracted_band.items():
            if raster.res != ref_raster.res:
                tmp = extracted_band[band]
                extracted_band[band] = match_to_reference(
                    raster, ref_raster, Resampling.nearest
                ).open()
                tmp.close()

        return extracted_band



    def get_all_band_names(self):
        return sorted(self.__band.keys())

    def get_crs(self):
        return self.__band["B02"].crs

    def get_transform(self):
        return self.__band["B02"].transform

    def __create_reference_transform(self, raster, crs):
        transform, width, height = calculate_default_transform(
            raster.crs,
            crs,
            raster.width,
            raster.height,
            *raster.bounds,
        )
        return transform, width, height

    def __create_valid_mask(self):
        scl_arr = self.__band["SCL"].read(1)
        scl_mask = np.where(
            (
                    scl_arr == 0
            )  # Image NO DATA values. Required, since SNAP produces faulty values for NA pixels otherwise
            | (scl_arr == 1)
            | (scl_arr == 2)
            | (scl_arr == 3)
            | (scl_arr == 8)
            | (scl_arr == 9)
            | (scl_arr == 10),
            0,
            1,
        )
        return scl_mask.astype(np.uint16)

    def get_band_raster(self, band):
        return self.__band[band]

    def get_band_array(self, band):
        if band == "SCL":
            return self.__band[band].read(indexes=1)
        elif band == "VAL":
            return self.__band[band]
        else:
            array = self.__band[band].read(indexes=1) + self.boa_effect
            array[array < 0] = 0
            return array.astype(np.uint16)

    def get_band_meta(self, band):
        return self.__band[band].meta

    def get_doy(self):
        date = self.file_name.split("_")[2][:8]
        year = date[:4]
        month = date[4:6]
        day = date[6:]
        date_object = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
        return date_object.timetuple().tm_yday

    def close(self):
        for band, raster in self.__band.items():
            if isinstance(raster, DatasetReader):
                raster.close()
