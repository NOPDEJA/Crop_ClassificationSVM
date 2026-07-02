"""evaluate_end_to_end.py

Composes the full Stage-1 -> Stage-2 -> Stage-3 cascade over ALL labeled
pixels and scores the final LU_CODE prediction against ground truth.

Per-stage reports answer "how good is this stage on the pixels routed to
it?" — this script answers the question LDD cares about: "if I ask the
pipeline what crop grows at a pixel, how often is it right?"  Stage-1/2
routing errors count as end-to-end misses here, unlike the per-stage CSVs.

Pipeline composition:
  stage1_pred != 1 (not econ)          -> final = 0 (no crop code)
  stage1_pred == 1, stage2_pred == g   -> final = stage3 model g's LU_CODE
  stage2_pred == 4 (other_econ)        -> final = 0 (no Stage-3 model exists)

Outputs (in runs/s1_dem/):
  end_to_end_lu_pred.npy   final LU_CODE per NPZ row (int16, 0 = none)
  end_to_end_report.csv    per-LU precision/recall/F1, end-to-end
  end_to_end_summary.csv   subclass-level rollup + overall stats
"""
import os
import numpy as np
import joblib
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report

# -----------------------
# Config
# -----------------------
NPZ = "./aligned_features/svm_add_data_features_labels.npz"
OUT_DIR = "runs/s1_dem/"

STAGE1_PRED = f"{OUT_DIR}stage1_s1_dem_pred.npy"
STAGE2_PRED = f"{OUT_DIR}stage2_s1_dem_pred.npy"
VALID_COLS_NPY = f"{OUT_DIR}stage1_s1_dem_valid_cols.npy"
STAGE3_MODEL_TPL = f"{OUT_DIR}stage3_s1_dem_{{grp}}_model.joblib"

OUT_PRED = f"{OUT_DIR}end_to_end_lu_pred.npy"
OUT_REPORT = f"{OUT_DIR}end_to_end_report.csv"
OUT_SUMMARY = f"{OUT_DIR}end_to_end_summary.csv"

PRED_CHUNK = 2_000_000

economic_crops = {2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420}
orchards_codes = {2403, 2404, 2407, 2413, 2416, 2419, 2420}
plantation_codes = {2302, 2303, 2405}
field_codes = {2101, 2204, 2205}
SUBCLASS_GROUPS = {1: "orchards", 2: "plantation", 3: "field"}  # stage2 label -> model name

LU_NAMES = {
    2101: "Rice", 2204: "Cassava", 2205: "Pineapple",
    2302: "Rubber", 2303: "Oil palm", 2405: "Coconut",
    2403: "Durian", 2404: "Rambutan", 2407: "Mango", 2413: "Longan",
    2416: "Jackfruit", 2419: "Mangosteen", 2420: "Langsat",
}

def true_subclass(codes):
    sub = np.zeros_like(codes, dtype=np.int8)
    sub[np.isin(codes, list(orchards_codes))] = 1
    sub[np.isin(codes, list(plantation_codes))] = 2
    sub[np.isin(codes, list(field_codes))] = 3
    return sub

if __name__ == "__main__":
    # ── Load inputs ────────────────────────────────────────────────────────
    d = np.load(NPZ, allow_pickle=True)
    y_true = d["y"].astype(np.int32)
    X_all = d["X"].astype(np.float32)
    valid_cols = np.load(VALID_COLS_NPY) if os.path.exists(VALID_COLS_NPY) else None
    if valid_cols is not None:
        X_all = X_all[:, valid_cols]
    stage1_pred = np.load(STAGE1_PRED)
    stage2_pred = np.load(STAGE2_PRED)
    n = y_true.shape[0]
    assert stage1_pred.shape[0] == n and stage2_pred.shape[0] == n, "pred arrays not aligned with NPZ"

    # ── Compose: run each Stage-3 model on its routed pixels ───────────────
    if os.path.exists(OUT_PRED):
        print("Found existing composed predictions:", OUT_PRED)
        final = np.load(OUT_PRED)
    else:
        final = np.zeros(n, dtype=np.int16)
        for grp_label, grp_name in SUBCLASS_GROUPS.items():
            model_path = STAGE3_MODEL_TPL.format(grp=grp_name)
            if not os.path.exists(model_path):
                print(f"[{grp_name}] model missing ({model_path}) — pixels stay 0")
                continue
            model = joblib.load(model_path)
            idxs = np.flatnonzero((stage1_pred == 1) & (stage2_pred == grp_label))
            print(f"[{grp_name}] predicting {idxs.size:,} routed pixels...")
            for s in range(0, idxs.size, PRED_CHUNK):
                e = min(idxs.size, s + PRED_CHUNK)
                sel = idxs[s:e]
                final[sel] = model.predict(X_all[sel]).astype(np.int16)
                print(f"  chunk {s}:{e}")
        np.save(OUT_PRED, final)
        print("Saved composed LU predictions:", OUT_PRED)

    # ── Score end-to-end on economic-crop LU codes ─────────────────────────
    # y for scoring: true LU code if econ, else 0. pred: composed code (0 = none).
    econ_codes = sorted(economic_crops)
    y_eval = np.where(np.isin(y_true, econ_codes), y_true, 0).astype(np.int32)

    report = classification_report(y_eval, final, labels=econ_codes, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.insert(0, "crop", [LU_NAMES.get(int(float(i)), "") if str(i).replace('.','',1).isdigit() else "" for i in df.index])
    df.to_csv(OUT_REPORT, encoding="utf-8-sig")
    print("\nEnd-to-end per-LU report (precision/recall/F1 include routing errors):")
    print(df.to_string())
    print("Saved:", OUT_REPORT)

    # ── Rollups ────────────────────────────────────────────────────────────
    rows = []
    true_econ = np.isin(y_true, econ_codes)
    n_true_econ = int(true_econ.sum())
    correct = (final == y_true) & true_econ

    rows.append({"level": "overall", "group": "all_econ",
                 "true_pixels": n_true_econ,
                 "end_to_end_recall": correct.sum() / max(n_true_econ, 1)})

    sub_true = true_subclass(y_true)
    for g, gname in SUBCLASS_GROUPS.items():
        m = sub_true == g
        rows.append({"level": "subclass", "group": gname,
                     "true_pixels": int(m.sum()),
                     "end_to_end_recall": float(correct[m].sum() / max(m.sum(), 1))})

    # routing losses: where do true-econ pixels fall out of the cascade?
    lost_s1 = true_econ & (stage1_pred != 1)
    lost_s2 = true_econ & (stage1_pred == 1) & (stage2_pred != true_subclass(y_true))
    rows.append({"level": "loss", "group": "dropped_at_stage1 (not pred econ)",
                 "true_pixels": int(lost_s1.sum()),
                 "end_to_end_recall": ""})
    rows.append({"level": "loss", "group": "misrouted_at_stage2 (wrong subclass)",
                 "true_pixels": int(lost_s2.sum()),
                 "end_to_end_recall": ""})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    print("\nEnd-to-end summary:")
    print(summary.to_string(index=False))
    print("Saved:", OUT_SUMMARY)
