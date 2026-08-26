"""probe_fused_grouped.py

E2 of docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md: the fused-stack (DEM + S1 +
S2) parcel-grouped probe -- the cheapest direct falsifier of whether the
rare-crop ceiling is optical (S2-only) rather than data (parcels).

Split, sampling, and pipeline are copied from probe_dry_season_grouped.py
verbatim: same crop-wise parcel halving (seed 42), same N_PER_SIDE=4,000 draw,
same median imputation + StandardScaler + Nystroem(800) + OneVsRestClassifier
(LinearSVC) pipeline. The only change is the columns each arm sees:

  Arm A (control):   the 40 S2-index columns (NDVI/EVI/NDWI/BSI/NDBI/MSAVI/SWIR*)
  Arm B (treatment): all 153 columns (adds 5 DEM + 108 S1 columns)

Gamma follows each arm's OWN feature count (1.0 / n_features), never Arm A's
gamma reused for Arm B -- the gamma-scale lesson from combined-run-kernel-dilution.

Falsifier (predeclared in runs/probe_fused_grouped/PREDECLARATION.md): any of
coconut/mangosteen/rambutan/longan gains >= +0.10 F1 in B over A.

Data facts asserted below (verified 2026-08-26, re-verified here, not assumed):
  aligned_features/svm_dem_s1_s2_features_labels.npz -- keys X, y, feature_names;
  153 named features; zero VVVH_DIFF columns; 24,323,769 rows.
  aligned_features/_unpacked/X_src.npy is the same matrix pre-extracted.
Do NOT use svm_add_data_features_labels.npz (no feature_names; superseded).

Writes runs/probe_fused_grouped/per_class.csv and summary.csv.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SRC_NPZ = "./aligned_features/svm_dem_s1_s2_features_labels.npz"
SRC_X = "./aligned_features/_unpacked/X_src.npy"
S2_ONLY_NPZ = "./aligned_features/svm_s2_3date_features_labels.npz"
OUT = "./runs/probe_fused_grouped"
SEED = 42
N_PER_SIDE = 4_000
N_COMPONENTS = int(os.environ.get("PROBE_NCOMP", 800))

CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(SRC_NPZ, allow_pickle=True)
    names = d["feature_names"].astype(str)
    assert names.size == 153, f"expected 153 named features, got {names.size}"
    vvvh = np.flatnonzero(np.char.find(names, "VVVH_DIFF") >= 0)
    assert vvvh.size == 0, f"found {vvvh.size} VVVH_DIFF columns -- npz is stale"

    y_fused = d["y"].astype(np.int32)
    y_s2 = np.load(S2_ONLY_NPZ, allow_pickle=True)["y"].astype(np.int32)
    assert np.array_equal(y_fused, y_s2), "fused npz y != S2-only npz y"
    y = y_fused
    del y_fused, y_s2

    parcels = np.load("./splits/parcel_id_row.npy")
    assert parcels.size == y.size == 24_323_769, \
        f"row count mismatch: parcels {parcels.size} y {y.size}"

    s2_cols = np.flatnonzero(np.char.find(names, "47PQQ") >= 0)
    assert s2_cols.size == 40, f"expected 40 S2 columns, got {s2_cols.size}"
    all_cols = np.arange(names.size)
    log(f"{s2_cols.size} S2 columns of {all_cols.size} total; y = {y.size:,} rows")

    # --- parcel-atomic halving inside each crop, then pixels (identical to
    # probe_dry_season_grouped.py) --------------------------------------------
    tr_take, te_take, counts = [], [], []
    for code, name in CROPS.items():
        idx = np.flatnonzero(y == code)
        pid = parcels[idx]
        uniq = np.unique(pid)
        order = rng.permutation(uniq.size)
        a_par = uniq[order[: uniq.size // 2]]
        b_par = uniq[order[uniq.size // 2:]]
        a = idx[np.isin(pid, a_par)]
        b = idx[np.isin(pid, b_par)]
        assert np.intersect1d(a_par, b_par).size == 0
        ka, kb = min(N_PER_SIDE, a.size), min(N_PER_SIDE, b.size)
        tr_take.append(rng.choice(a, ka, replace=False))
        te_take.append(rng.choice(b, kb, replace=False))
        counts.append({"crop": name, "lu_code": code,
                       "parcels_total": int(uniq.size),
                       "train_parcels": int(a_par.size), "test_parcels": int(b_par.size),
                       "train_pixels": int(ka), "test_pixels": int(kb)})
        log(f"  {name:<12}{uniq.size:>6} parcels -> {a_par.size:>5}/{b_par.size:<5} "
            f"train/test, pixels {ka:,}/{kb:,}")

    tr_rows = np.sort(np.concatenate(tr_take))
    te_rows = np.sort(np.concatenate(te_take))
    assert np.intersect1d(parcels[tr_rows], parcels[te_rows]).size == 0, \
        "a parcel is on both sides"
    log(f"parcel-disjoint: train {tr_rows.size:,}  test {te_rows.size:,}")

    rows = np.concatenate([tr_rows, te_rows])
    order = np.argsort(rows, kind="stable")
    Xsrc = np.load(SRC_X, mmap_mode="r")
    srt = rows[order]
    step = 50_000
    buf = np.empty((rows.size, all_cols.size), dtype=np.float32)
    for i in range(0, srt.size, step):
        buf[i:i + step] = Xsrc[srt[i:i + step]]
    del Xsrc
    X = np.empty_like(buf)
    X[order] = buf
    del buf
    ys = y[rows]
    tr = np.arange(tr_rows.size)
    te = np.arange(tr_rows.size, rows.size)
    log(f"X = {X.shape}")

    arms = {"A_s2only": s2_cols, "B_fused": all_cols}
    labels = list(CROPS)

    per_class, summary = [], []
    for arm, cols in arms.items():
        log(f"--- {arm}: {cols.size} features")
        t0 = time.time()
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Nystroem(gamma=1.0 / cols.size, n_components=N_COMPONENTS,
                     random_state=SEED),
            OneVsRestClassifier(LinearSVC(C=1.0, dual=False, max_iter=5000,
                                          random_state=SEED), n_jobs=1))
        pipe.fit(X[tr][:, cols], ys[tr])
        pred = pipe.predict(X[te][:, cols])
        rep = classification_report(ys[te], pred, labels=labels,
                                    output_dict=True, zero_division=0)
        by_crop = {c["lu_code"]: c for c in counts}
        for lab in labels:
            r = rep[str(lab)]
            per_class.append({"arm": arm, "class": CROPS[lab], "lu_code": lab,
                              "precision": round(r["precision"], 4),
                              "recall": round(r["recall"], 4),
                              "f1": round(r["f1-score"], 4),
                              "test_parcels": by_crop[lab]["test_parcels"],
                              "train_parcels": by_crop[lab]["train_parcels"],
                              "test_pixels": by_crop[lab]["test_pixels"]})
        s = {"arm": arm, "n_features": int(cols.size), "split": "parcel-grouped",
             "macro_f1": round(f1_score(ys[te], pred, average="macro", zero_division=0), 4),
             "accuracy": round(accuracy_score(ys[te], pred), 4),
             "fit_seconds": round(time.time() - t0, 1)}
        summary.append(s)
        log(f"    {s}")

    pc = pd.DataFrame(per_class)
    pc.to_csv(f"{OUT}/per_class.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig")
    log("\n" + pc.pivot(index="class", columns="arm", values="f1").to_string())
    log("\n" + pd.DataFrame(summary).to_string(index=False))

    piv = pc.pivot(index="class", columns="arm", values="f1")
    piv["delta_B_minus_A"] = (piv["B_fused"] - piv["A_s2only"]).round(4)
    piv.to_csv(f"{OUT}/delta.csv", encoding="utf-8-sig")
    log("\ndelta (B - A):\n" + piv.to_string())

    falsifier_crops = ["Coconut", "Mangosteen", "Rambutan", "Longan"]
    hits = piv.loc[piv.index.isin(falsifier_crops), "delta_B_minus_A"]
    triggered = hits[hits >= 0.10]
    if len(triggered):
        log(f"FALSIFIER TRIGGERED: {dict(triggered)} -- optical-ceiling branch")
    else:
        log(f"falsifier not triggered. deltas on watch crops: {dict(hits)}")
    log("DONE")
