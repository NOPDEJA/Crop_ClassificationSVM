import os
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from rasterio import MemoryFile
from rio_cogeo import cog_profiles, cog_translate
import rasterio
from nested_dict import nested_dict
from credential import *
import parsing
import zip_manager as zm
from raster import Sentinel2Raster
from zipfile import ZipFile
from multiprocessing import Pool
from itertools import product
from credential import username, password
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BANDS = {
    "B02": "B02_10m.jp2",
    "B03": "B03_10m.jp2",
    "B04": "B04_10m.jp2",
    "B08": "B08_10m.jp2",
    "B05": "B05_20m.jp2",
    "B06": "B06_20m.jp2",
    "B07": "B07_20m.jp2",
    "B11": "B11_20m.jp2",
    "B12": "B12_20m.jp2",
    "B8A": "B8A_20m.jp2",
    "SCL": "SCL_20m.jp2",
}
MASK_BAND = "VAL"
CPU_NUM = 3


def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                          data=data,
                          )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Keycloak token creation failed. Response from the server was: {r.json()}"
        )
    return r.json()["access_token"]


def is_zip_file_corrupted(file):
    try:
        with ZipFile(file, 'r') as z:
            return z.testzip() is not None
    except Exception as e:
        return True


def download_file(keycloak_token, file_id, file_name):
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({file_id})/$value"

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {keycloak_token}"})

    # Robust retry policy
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    temp_bytes = BytesIO()

    try:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()

            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    temp_bytes.write(chunk)

    except Exception as e:
        raise RuntimeError(f"Download failed for {file_name}: {str(e)}")

    # rewind before ZIP test
    temp_bytes.seek(0)

    # Validate the ZIP
    if is_zip_file_corrupted(temp_bytes):
        raise RuntimeError(f"Downloaded ZIP is corrupted: {file_name}")

    return temp_bytes


def write_cogtif(output_path: Path, raster_array: np.array, src_profile: dict):
    profile = src_profile
    cogtif_file = MemoryFile()
    with MemoryFile() as raster_file:
        with raster_file.open(**profile) as mem_src:
            mem_src.write(raster_array)
            dst_profile = cog_profiles.get("lzw") | profile
            cog_translate(
                mem_src,
                cogtif_file.name,
                dst_profile,
                in_memory=True,
                quiet=True,
            )
    with cogtif_file.open() as translated_file:
        with rasterio.open(output_path, "w", **translated_file.profile) as dst:
            dst.write(translated_file.read())


def aggregation_per_tile(tile_period):
    tile, start_date, end_date = tile_period[0], tile_period[1][0], tile_period[1][1]
    data_collection = "SENTINEL-2"
    product_type = "S2MSI2A"

    output_directory_path = Path("./") / "S2_data"
    # Check if the directory exists
    if not output_directory_path.exists():
        # If the directory doesn't exist, create it
        output_directory_path.mkdir(parents=True)
    file_name = f"{tile}_{end_date}"
    output_path = output_directory_path / f"{file_name}.tif"

    if not os.path.exists(output_path):
        print(tile, start_date, end_date)
        # Query the tiles overlapped with the supplied polygon
        json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/"
                            f"v1/Products?$filter=Collection/Name eq '{data_collection}' "
                            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
                            f"and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
                            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId' "
                            f"and att/OData.CSC.StringAttribute/Value eq '{tile}') "
                            f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
                            f"and ContentDate/Start lt {end_date}T00:00:00.000Z"
                            "&$orderby=ContentDate/Start&$top=100").json()
        # Turn the response to DataFrame
        files_df = pd.DataFrame.from_dict(json['value'])

        satellite_data = nested_dict()
        raster_meta = None
        # Download each file
        for _, file in files_df.iterrows():
            file_name = file.Name.split(".")[0]
            keycloak_token = get_keycloak(username, password)
            try:
                downloaded_zip = download_file(keycloak_token, file.Id, file_name)
            except Exception as e:
                print(e)
                return
            sentinel_zip = zm.ZipFileManager(downloaded_zip)
            file_list = sentinel_zip.list_contents()
            # Read metadata file
            meta_file = list(filter(lambda x: "MTD_MSIL2A.xml" in x, file_list))[0]
            metadata = parsing.MetaParser(sentinel_zip.open_file(meta_file))
            # Extract BOA effect and scaling factor
            boa_offset = metadata.get_boa_offset()
            scaling_factor = 1 / metadata.get_boa_quantification_value()
            band_files = {
                band: list(filter(lambda filename: BANDS[band] in filename, file_list)).pop(0)
                for band in BANDS.keys()
            }

            s2_raster = Sentinel2Raster(
                sentinel_zip, file_name, band_files, boa_offset, scaling_factor
            )
            raster_band_names = [band for band in s2_raster.get_all_band_names() if band not in ["SCL", "VAL"]]
            mask_bool = np.invert(s2_raster.get_band_array(MASK_BAND).astype(bool))
            raster_meta = s2_raster.get_band_meta("B02")
            for band_name in raster_band_names:
                if band_name not in satellite_data:
                    satellite_data[band_name] = []
                band = s2_raster.get_band_array(band_name)
                satellite_data[band_name].append(np.ma.masked_array(band, mask=mask_bool))
            s2_raster.close()

        # Start Aggregation
        aggregated_bands = []
        band_names = sorted(list(satellite_data.keys()))
        for band_name in band_names:
            stacked = np.ma.stack(satellite_data[band_name], axis=0)
            band_median = np.ma.median(stacked, axis=0)
            aggregated_bands.append(band_median)
        stacked_aggregated_bands = np.stack(aggregated_bands, axis=0)
        profile = {
            "driver": "GTiff",
            "count": len(satellite_data.keys()),
            "dtype": np.uint16,
            "width": raster_meta['width'],
            "height": raster_meta['height'],
            'crs': raster_meta['crs'],
            'transform': raster_meta['transform']
        }
        write_cogtif(output_path, stacked_aggregated_bands, profile)
    return


if __name__ == "__main__":

    # Intialize the parameters
    tiles = ['47PQQ']
    dates = [('2018-04-01', '2018-04-30')]

    tile_dates = list(product(tiles, dates))

    with Pool(CPU_NUM) as p:
        p.map(aggregation_per_tile, tile_dates)