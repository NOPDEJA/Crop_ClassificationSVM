"""
s1_preprocess_snap.py

Preprocesses Sentinel-1 GRD .zip files using ESA SNAP via subprocess.
Produces terrain-corrected, calibrated backscatter GeoTIFFs ready for
feature computation in s1_compute_features.py.

Processing graph (standard GRD → σ⁰ workflow):
  1. Apply Orbit File          — precise orbit vectors
  2. Remove Thermal Noise      — reduces noise floor artefacts
  3. Radiometric Calibration   — DN  → sigma0 (linear power)
  4. Speckle Filter            — Lee 7×7  (suppresses salt-pepper noise)
  5. Range-Doppler TC          — terrain correction to WGS84/UTM using SRTM
  6. Linear → dB conversion    — 10 * log10(sigma0)
  7. Band subset               — keep Sigma0_VV_db, Sigma0_VH_db only
  8. Export                    — GeoTIFF to ./S1_processed/

Requirements:
  - SNAP 9.x installed (https://step.esa.int/main/download/snap-download/)
  - gpt (Graph Processing Tool) on your PATH
      Windows default: C:/Program Files/snap/bin/gpt.exe
      Linux default  : ~/snap/bin/gpt

Usage:
  python s1_preprocess_snap.py
  
  Or process a single file:
  python s1_preprocess_snap.py ./S1_data/S1A_IW_GRDH_....zip
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from glob import glob

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR    = "./S1_data"        # folder containing raw .zip SAFE files
OUTPUT_DIR   = "./S1_processed"   # where to write preprocessed GeoTIFFs
LOG_DIR      = "./S1_logs"        # optional; saves per-file SNAP log

# Path to SNAP gpt executable — edit if not on PATH
# Windows example: GPT_PATH = "C:/Program Files/snap/bin/gpt.exe"
# Linux  example : GPT_PATH = os.path.expanduser("~/snap/bin/gpt")
GPT_PATH = "D:/esa-snap/bin/gpt.exe"

# Speckle filter settings
SPECKLE_FILTER       = "Lee"          # Lee / Refined Lee / IDAN / Boxcar
SPECKLE_FILTER_SIZE  = "7x7"

# Terrain correction
DEM_NAME             = "SRTM 1Sec HGT"   # auto-downloaded by SNAP
                                           # or "Copernicus 30m Global DEM"
TARGET_CRS           = "EPSG:32647"        # UTM zone 47N  (adjust for your AOI)
PIXEL_SPACING_M      = 10.0               # output resolution in metres

# Parallel SNAP workers (set to number of CPU cores)
SNAP_PARALLELISM     = 4
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def build_gpt_graph(input_path: str, output_path: str) -> str:
    """
    Build a SNAP GPT XML graph string for the standard
    GRD → terrain-corrected σ⁰ dB workflow.
    Returns the XML as a string.
    """
    graph_xml = f"""<graph id="S1_GRD_Preprocessing">
  <version>1.0</version>

  <!-- 1. Read -->
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>{input_path}</file>
    </parameters>
  </node>

  <!-- 2. Apply Orbit File -->
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <!-- 3. Thermal Noise Removal -->
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>

  <!-- 4. Radiometric Calibration  (output: sigma0 linear) -->
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
    </parameters>
  </node>

  <!-- 5. Speckle Filter -->
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <sourceBands/>
      <filter>{SPECKLE_FILTER}</filter>
      <filterSizeX>{SPECKLE_FILTER_SIZE.split("x")[0]}</filterSizeX>
      <filterSizeY>{SPECKLE_FILTER_SIZE.split("x")[1]}</filterSizeY>
      <dampingFactor>2</dampingFactor>
      <estimateENL>true</estimateENL>
    </parameters>
  </node>

  <!-- 6. Range-Doppler Terrain Correction -->
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters>
      <sourceBands/>
      <demName>{DEM_NAME}</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <mapProjection>{TARGET_CRS}</mapProjection>
      <pixelSpacingInMeter>{PIXEL_SPACING_M}</pixelSpacingInMeter>
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <outputComplex>false</outputComplex>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
    </parameters>
  </node>

  <!-- 7. Linear to dB -->
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters>
      <sourceBands/>
    </parameters>
  </node>

  <!-- 8. Write -->
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="LinearToFromdB"/></sources>
    <parameters>
      <file>{output_path}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>

</graph>
"""
    return graph_xml


def preprocess_file(zip_path: str) -> str | None:
    """
    Preprocess a single S1 GRD zip file.
    Returns the output path on success, None on failure.
    """
    zip_path = Path(zip_path)
    stem = zip_path.stem  # e.g. S1A_IW_GRDH_1SDV_20180415T...
    output_path = Path(OUTPUT_DIR) / f"{stem}_preprocessed.tif"
    log_path    = Path(LOG_DIR)   / f"{stem}.log"

    if output_path.exists():
        print(f"  [SKIP] Already processed: {output_path.name}")
        return str(output_path)

    print(f"  Processing: {zip_path.name}")

    # Write the GPT graph to a temp file
    graph_xml = build_gpt_graph(str(zip_path.resolve()), str(output_path.resolve()))
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, prefix="s1_graph_"
    ) as tmp:
        tmp.write(graph_xml)
        graph_path = tmp.name

    cmd = [
        GPT_PATH,
        graph_path,
        f"-q {SNAP_PARALLELISM}",   # parallel workers
        "-e",                         # show progress
    ]

    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                " ".join(cmd),
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,   # 1-hour timeout per file
            )

        if result.returncode != 0:
            print(f"  ERROR (code {result.returncode}): see {log_path}")
            return None

        print(f"  → Saved: {output_path.name}")
        return str(output_path)

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT processing {zip_path.name}")
        return None
    finally:
        os.unlink(graph_path)


def verify_snap_available():
    """Check that gpt is on the PATH and print version."""
    try:
        result = subprocess.run(
            [GPT_PATH, "--version"],
            capture_output=True, text=True, timeout=30
        )
        print(f"SNAP GPT found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print(
            f"ERROR: '{GPT_PATH}' not found.\n"
            "Install SNAP from https://step.esa.int/main/download/snap-download/\n"
            "Then either:\n"
            "  (a) Add SNAP's bin/ folder to your PATH, or\n"
            "  (b) Set GPT_PATH at the top of this script."
        )
        return False


if __name__ == "__main__":
    if not verify_snap_available():
        sys.exit(1)

    # Allow single-file argument from CLI
    if len(sys.argv) > 1:
        zip_files = sys.argv[1:]
    else:
        zip_files = sorted(glob(os.path.join(INPUT_DIR, "*.zip")))

    if not zip_files:
        print(f"No .zip files found in {INPUT_DIR}/")
        sys.exit(0)

    print(f"\nFound {len(zip_files)} file(s) to preprocess.\n")
    success, failed = [], []

    for zf in zip_files:
        result = preprocess_file(zf)
        (success if result else failed).append(zf)

    print(f"\n{'='*50}")
    print(f"Completed: {len(success)} / {len(zip_files)}")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")
    print(f"Preprocessed GeoTIFFs saved to: {OUTPUT_DIR}/")
    print("Next step: run  python s1_compute_features.py")