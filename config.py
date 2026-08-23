"""config.py

Single place to switch which experiment arm the pipeline runs.

Change RUN + TAG + NPZ here and every stage writes/reads a consistent set of
paths. Nothing else in the pipeline should hardcode a run directory or an
artifact filename.

Arms used so far:
    RUN                 TAG            NPZ
    s1_dem_v2           s1_dem         svm_add_data_features_labels.npz   (DEM+S1, 94 cols)
    dem_s1_s2_v2        dem_s1_s2      svm_dem_s1_s2_features_labels.npz  (all, 134 cols)
    s2_2018_3date       s2_3date       svm_s2_3date_features_labels.npz   (S2 Oct-Dec, 24 cols)
    s2_2018_5date       s2_5date       svm_s2_only_features_labels.npz    (S2 all dates, 40 cols)
"""
import os

# -----------------------
# The arm being run
# -----------------------
# Overridable per invocation so a second arm can be trained without editing this
# file (and without a half-edited config pointing analysis scripts at the wrong run):
#   ARM=s2_2018_5date python stage1_weight_scale.py
ARMS = {
    "s2_2018_3date": ("s2_3date", "./aligned_features/svm_s2_3date_features_labels.npz"),
    "s2_2018_5date": ("s2_5date", "./aligned_features/svm_s2_only_features_labels.npz"),
    # Same three dates as the published arm -- so it stays inside the §7 controlled
    # comparison window -- but without the two configuration defects of §6.9. The date
    # window is what the joint paper controls; PCA and kernel capacity are internal.
    "s2_2018_3date_v2": ("s2_3date_v2", "./aligned_features/svm_s2_3date_features_labels.npz"),
}
RUN = os.environ.get("ARM", "s2_2018_3date")
if RUN not in ARMS:
    raise SystemExit(f"Unknown ARM '{RUN}'; known: {', '.join(ARMS)}")
TAG, NPZ = ARMS[RUN]

RANDOM_STATE = 42
PRED_CHUNK = 2_000_000

# -----------------------
# Sampling constants
# -----------------------
# These decide WHICH rows each stage fits on, so reconstruct_sampled_rows.py must
# replay them exactly to recover the held-out population. They used to be declared
# separately in each stage script and again in reconstruct_sampled_rows.py; a change
# to one copy silently scored every published metric on the wrong population. Define
# them here only.

# Stage 1
SAMPLES_PER_LU = 400_000   # cap per LU code, before mapping to superclasses
MIN_CLASS_PIXELS = 200     # drop an LU code with fewer pixels than this
CAP_ECON = 1_000_000       # unique samples per superclass, before the split
CAP_WATER = 500_000
CAP_FOREST = 600_000
CAP_OTHERS = 800_000

# Stage 2
MIN_PIXELS_PER_GROUP = 100
PER_GROUP_CAP = 200_000    # cap per subclass group

# Stage 3
MIN_PIXELS_PER_LU = 100
PER_LU_CAP = 70_000        # cap per LU code, before the split
TARGET_PER_LU = 20_000     # train-only upsampling target for minority codes

# -----------------------
# Run directory
# -----------------------
OUT_DIR = f"./runs/{RUN}"
os.makedirs(OUT_DIR, exist_ok=True)

# Stage 1 artifacts
STAGE1_MODEL = f"{OUT_DIR}/stage1_{TAG}.joblib"
STAGE1_THRESH = f"{OUT_DIR}/stage1_{TAG}_thresholds.json"
STAGE1_REPORT = f"{OUT_DIR}/stage1_{TAG}_report.csv"
STAGE1_PRED = f"{OUT_DIR}/stage1_{TAG}_pred.npy"
STAGE1_PROB = f"{OUT_DIR}/stage1_{TAG}_prob.npy"
VALID_COLS_NPY = f"{OUT_DIR}/stage1_{TAG}_valid_cols.npy"

# Stage 2 artifacts
STAGE2_MODEL = f"{OUT_DIR}/stage2_{TAG}_model.joblib"
STAGE2_MODEL_FULL = f"{OUT_DIR}/stage2_{TAG}_model_fulldata.joblib"
STAGE2_REPORT = f"{OUT_DIR}/stage2_{TAG}_report.csv"
STAGE2_STATS_LU = f"{OUT_DIR}/stage2_{TAG}_stats_per_lu.csv"
STAGE2_STATS_GROUP = f"{OUT_DIR}/stage2_{TAG}_stats_per_group.csv"
STAGE2_TEST_PROB = f"{OUT_DIR}/stage2_{TAG}_test_prob.npy"
STAGE2_TEST_PRED = f"{OUT_DIR}/stage2_{TAG}_test_pred.npy"
STAGE2_CONF_CSV = f"{OUT_DIR}/stage2_{TAG}_confusion_matrix.csv"
STAGE2_META_JSON = f"{OUT_DIR}/stage2_{TAG}_meta.json"
STAGE2_PRED = f"{OUT_DIR}/stage2_{TAG}_pred.npy"

# Stage 3 artifacts ({grp} = orchards / plantation / field)
STAGE3_MODEL_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_model.joblib"
STAGE3_REPORT_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_report.csv"
STAGE3_TEST_PROB_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_test_prob.npy"
STAGE3_TEST_PRED_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_test_pred.npy"
STAGE3_META_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_meta.json"
STAGE3_CONF_TPL = f"{OUT_DIR}/stage3_{TAG}_{{grp}}_confusion_matrix.csv"
STAGE3_TOP_META = f"{OUT_DIR}/stage3_{TAG}_meta.json"

# End-to-end evaluation
E2E_PRED = f"{OUT_DIR}/end_to_end_lu_pred.npy"
E2E_REPORT = f"{OUT_DIR}/end_to_end_report.csv"
E2E_SUMMARY = f"{OUT_DIR}/end_to_end_summary.csv"

# Feature importance
PI_JSON = f"{OUT_DIR}/stage1_{TAG}_features_importance.json"

# -----------------------
# Data paths (shared by all arms)
# -----------------------
INDICES_DIR = "./indices"
DEM_TIF = "./dem/dem_47PQQ.tif"
LABEL_TIF = "./label/label_47PQQ.tif"
LABEL_BUFFERED = "./label/label_47PQQ_buffered.tif"

# LDD parcel surveys (Buddhist-era year in the filename)
LDD_ROOT = "../"
SHAPEFILE_2561 = f"{LDD_ROOT}LDD_Scripts/Rayong61/LU_RYG_2561.shp"          # 2018
SHAPEFILE_2563 = f"{LDD_ROOT}Rayong_61_63 (extract.me)/ระยอง63/LU_RYG_2563.shp"  # 2020
SHAPEFILE_2567 = f"{LDD_ROOT}Landuse_Rayoung67/LU_RYG_2567.shp"             # 2024
SHAPEFILE = SHAPEFILE_2561  # survey used for the current training labels
