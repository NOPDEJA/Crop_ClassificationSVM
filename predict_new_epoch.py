"""predict_new_epoch.py

Apply an already-trained 2018 cascade to a different epoch. Inference only --
no stage is refitted, no hyperparameter is re-searched.

    python predict_new_epoch.py 2024 s2_2018_3date_v2

Only the three-date arms can be tested this way: no dry-season imagery exists for
2020 or 2024, so the five-date arm has no matching feature set.

**Two populations are scored, and the difference between them matters.**

  all          every labelled pixel of the target epoch. The model never saw these
               *reflectances*, so this is honest temporal validation -- but the
               parcels are the same ground the 2018 model trained on, so it is
               temporal transfer only, not spatial.
  unseen_2018  the subset that was also held out spatially in 2018. Never fitted
               on in either sense. This is the strict number.

**Disagreement here is not all model error.** Scored against the epoch's own
survey, three things are mixed: model transfer, genuine land-use change, and
differences between LDD survey rounds. The label-churn table printed below
measures the second directly -- how many pixels changed class between the 2018
survey and this one -- so the model's errors can be split into those that fall on
ground that moved and those that do not. Without it a poor score is uninterpretable.

Outputs (in runs/xyear_<year>_from_<arm>/):
  stage1_pred.npy, stage2_pred.npy, end_to_end_lu_pred.npy
  flat15_per_class.csv, flat15_overall.csv, label_churn.csv
  confusion/ ... the standard matrix set
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import rasterio

from config import ARMS, PRED_CHUNK
from evaluate_flat_15class import (to_flat_true, to_flat_pred, score, CROPS, NAMES)
from confusion_report import confusion_set, write_confusion_set

SUBCLASS = {1: "orchards", 2: "plantation", 3: "field"}
REF_2018_BUF = "./label/label_47PQQ_buffered.tif"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def pixel_index(label_tif):
    """Flat pixel positions of the rows an aligned NPZ contains.

    align_indices_labels.py keeps pixels where label != 0 and != 32767 and saves no
    index, so it is recomputed here by the same rule. Both epochs sit on the 2018
    grid, so these indices are directly comparable between them.
    """
    with rasterio.open(label_tif) as s:
        f = s.read(1).flatten()
    return np.flatnonzero((f != 0) & (f != 32767)), f


def chunked_predict(model, X, rows, out, dtype):
    for s in range(0, rows.size, PRED_CHUNK):
        sel = rows[s:s + PRED_CHUNK]
        out[sel] = model.predict(X[sel]).astype(dtype)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python predict_new_epoch.py <2020|2024> <model_arm>")
    year, arm = sys.argv[1], sys.argv[2]
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}; known: {', '.join(ARMS)}")
    tag = ARMS[arm][0]
    src = f"./runs/{arm}"
    out_dir = f"./runs/xyear_{year}_from_{arm}"
    os.makedirs(out_dir, exist_ok=True)
    log(f"epoch {year}  models from {arm} (tag {tag})  -> {out_dir}")

    npz = f"./aligned_features/svm_s2_3date_{year}_features_labels.npz"
    d = np.load(npz, allow_pickle=True)
    X, y = d["X"], d["y"].astype(np.int32)
    log(f"X={X.shape} y={y.shape}")

    vc_path = f"{src}/stage1_{tag}_valid_cols.npy"
    if os.path.exists(vc_path):
        vc = np.load(vc_path)
        X = X[:, vc]
        log(f"applied valid_cols -> {X.shape[1]} features")

    # ---- cascade, inference only -------------------------------------------
    log("Stage 1")
    m1 = joblib.load(f"{src}/stage1_{tag}.joblib")
    s1 = np.zeros(y.size, dtype=np.uint8)
    chunked_predict(m1, X, np.arange(y.size), s1, np.uint8)
    del m1
    econ = np.flatnonzero(s1 == 1)
    log(f"  routed econ: {econ.size:,} of {y.size:,}")

    log("Stage 2")
    m2 = joblib.load(f"{src}/stage2_{tag}_model.joblib")
    s2 = np.zeros(y.size, dtype=np.int32)
    chunked_predict(m2, X, econ, s2, np.int32)
    del m2

    log("Stage 3")
    final = np.zeros(y.size, dtype=np.int16)
    for lab, grp in SUBCLASS.items():
        rows = np.flatnonzero((s1 == 1) & (s2 == lab))
        mp = f"{src}/stage3_{tag}_{grp}_model.joblib"
        if rows.size == 0 or not os.path.exists(mp):
            log(f"  {grp}: {rows.size:,} px, model present={os.path.exists(mp)} -- skipped")
            continue
        m3 = joblib.load(mp)
        chunked_predict(m3, X, rows, final, np.int16)
        del m3
        log(f"  {grp}: {rows.size:,} px")
    # stage2 == 4 (other_econ) has no Stage-3 model; those stay 0, as in
    # evaluate_end_to_end.py
    del X

    np.save(f"{out_dir}/stage1_pred.npy", s1)
    np.save(f"{out_dir}/stage2_pred.npy", s2)
    np.save(f"{out_dir}/end_to_end_lu_pred.npy", final)

    # ---- populations --------------------------------------------------------
    pix_t, _ = pixel_index(f"./label/label_47PQQ_{year}_buffered.tif")
    pix_18, _ = pixel_index(REF_2018_BUF)
    if pix_t.size != y.size:
        raise SystemExit(f"row/pixel mismatch: {pix_t.size} vs {y.size}")
    fitted18 = np.load(f"{src}/trainval_rows_mask.npy")
    unseen_pix = pix_18[~fitted18]
    strict = np.flatnonzero(np.isin(pix_t, unseen_pix))
    pops = {"all": np.arange(y.size), "unseen_2018": strict}
    log(f"populations: all={y.size:,}  unseen_2018={strict.size:,}")

    y_flat, p_flat = to_flat_true(y), to_flat_pred(s1, final)
    per_class, overall = [], []
    for name, idx in pops.items():
        rows, ov = score(y_flat[idx], p_flat[idx], fold_forest=True, tag=name)
        per_class += rows
        overall.append(ov)
        log(f"{name}: {ov}")

    pd.DataFrame(per_class).to_csv(f"{out_dir}/flat15_per_class.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(overall).to_csv(f"{out_dir}/flat15_overall.csv",
                                 index=False, encoding="utf-8-sig")
    write_confusion_set(confusion_set(y, s1, s2, final, pops["all"]), out_dir, tag=f"xyear{year}")

    # ---- how much of the ground actually moved ------------------------------
    with rasterio.open(REF_2018_BUF) as s:
        l18 = s.read(1).flatten()
    with rasterio.open(f"./label/label_47PQQ_{year}_buffered.tif") as s:
        lyr = s.read(1).flatten()
    both = (l18 != 0) & (l18 != 32767) & (lyr != 0) & (lyr != 32767)
    a, b = l18[both], lyr[both]
    churn = [{"scope": "all labelled in both surveys", "pixels": int(both.sum()),
              "unchanged": int((a == b).sum()),
              "changed_share": round(float((a != b).mean()), 4)}]
    for code, nm in CROPS.items():
        m = a == code
        if m.sum():
            churn.append({"scope": f"true {nm} in 2018", "pixels": int(m.sum()),
                          "unchanged": int((b[m] == code).sum()),
                          "changed_share": round(float((b[m] != code).mean()), 4)})
    cdf = pd.DataFrame(churn)
    cdf.to_csv(f"{out_dir}/label_churn.csv", index=False, encoding="utf-8-sig")
    log(f"\nland-use churn, 2018 survey vs {year} survey:\n" + cdf.to_string(index=False))
    log(f"saved to {out_dir}")
