import os
from io import BytesIO
from pathlib import Path
import pandas as pd
import requests
from zipfile import ZipFile
from multiprocessing import Pool
from itertools import product
from credential import username, password
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── AOI bounding box for Rayong / tile 47PQQ (WGS84) ─────────────────────────
# Adjust if your study area is different
TILE_BBOX_WKT = "POLYGON((101.0 12.5, 102.0 12.5, 102.0 13.5, 101.0 13.5, 101.0 12.5))"

CPU_NUM = 2   # Keep low — S1 files are ~1 GB each


# ── Identical to tile_download.py ─────────────────────────────────────────────
def get_keycloak(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
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

    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    temp_bytes = BytesIO()

    try:
        with session.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            downloaded = 0
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                if chunk:
                    temp_bytes.write(chunk)
                    downloaded += len(chunk)
            print(f"  Downloaded {downloaded / 1e6:.1f} MB  ({file_name})")

    except Exception as e:
        raise RuntimeError(f"Download failed for {file_name}: {str(e)}")

    # rewind before ZIP test
    temp_bytes.seek(0)

    # Validate the ZIP
    if is_zip_file_corrupted(temp_bytes):
        raise RuntimeError(f"Downloaded ZIP is corrupted: {file_name}")

    # ── FIX: seek back to 0 after corruption check ────────────────────────────
    # is_zip_file_corrupted() reads through the BytesIO leaving the pointer
    # at an unknown position. Without this seek the caller gets an empty read.
    temp_bytes.seek(0)

    return temp_bytes


# ── S1-specific download logic (mirrors aggregation_per_tile structure) ────────
def download_per_tile(tile_period):
    tile, start_date, end_date = tile_period[0], tile_period[1][0], tile_period[1][1]
    data_collection = "SENTINEL-1"
    product_type    = "IW_GRDH_1S"

    output_directory_path = Path("./") / "S1_data"
    if not output_directory_path.exists():
        output_directory_path.mkdir(parents=True)

    print(tile, start_date, end_date)

    # Query Copernicus OData — S1 uses footprint intersection, not tileId
    json_resp = requests.get(
        f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        f"?$filter=Collection/Name eq '{data_collection}'"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f"   and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{TILE_BBOX_WKT}')"
        f" and ContentDate/Start gt {start_date}T00:00:00.000Z"
        f" and ContentDate/Start lt {end_date}T00:00:00.000Z"
        f"&$orderby=ContentDate/Start&$top=50"
    ).json()

    files_df = pd.DataFrame.from_dict(json_resp['value'])

    if files_df.empty:
        print(f"  No S1 products found for {tile} {start_date} → {end_date}")
        return

    print(f"  Found {len(files_df)} product(s)")

    for _, file in files_df.iterrows():
        file_name   = file.Name.replace(".SAFE", "")
        output_path = output_directory_path / f"{file_name}.zip"

        if os.path.exists(output_path):
            print(f"  Already exists: {output_path.name}")
            continue

        keycloak_token = get_keycloak(username, password)
        try:
            downloaded_zip = download_file(keycloak_token, file.Id, file_name)
        except Exception as e:
            print(e)
            continue

        # Save raw .zip to disk (SNAP preprocessing happens separately)
        with open(output_path, "wb") as f:
            f.write(downloaded_zip.read())
        print(f"  Saved → {output_path}")

    return


if __name__ == "__main__":

    # Initialise the parameters — same structure as tile_download.py
    tiles = ['47PQQ']
    dates = [('2018-04-01', '2018-04-30')]

    tile_dates = list(product(tiles, dates))

    with Pool(CPU_NUM) as p:
        p.map(download_per_tile, tile_dates)