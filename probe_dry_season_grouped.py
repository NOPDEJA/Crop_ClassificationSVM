"""probe_dry_season_grouped.py

Section 4 of docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md: `probe_dry_season.py`
rebuilt so that no parcel appears on both sides.

The original splits its sample with a random pixel permutation, and
`probe_replay_overlap.py` shows that 87.5 % to 95.7 % of its rare-crop test
parcels are also in its training half. That is what forced the withdrawal of our
"the rare crops are not spectrally limited" claim. This rebuild changes the split
and nothing else: same three arms, same uniform prior, same 800 Nystroem
components, same seed, same architecture.

ONE DEVIATION from the plan's wording, stated rather than hidden. The plan says
GroupShuffleSplit on parcel ID. A plain group split ignores the label, and with
10 Langsat parcels it can put every one of them on one side, which produces an
undefined F1 rather than a result. So the parcels are halved WITHIN each crop --
the same construction `halve_by_parcel` uses in train_parcel_cascade.py -- which
is still parcel-atomic and additionally guarantees every crop is present on both
sides. Disjointness is asserted, not assumed.

Pixels are drawn AFTER the parcel split, up to N_PER_SIDE per crop per side, so
the prior stays uniform in both halves instead of following whichever half
happened to get the big parcels.

Writes runs/probe_dry_season/per_class_parcel_grouped.csv and
       runs/probe_dry_season/summary_parcel_grouped.csv
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
OUT = "./runs/probe_dry_season"
SEED = int(os.environ.get("PROBE_SEED", 42))
N_PER_SIDE = 4_000          # the original drew 8,000 per crop and halved it
N_COMPONENTS = int(os.environ.get("PROBE_NCOMP", 800))

CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}
WET = ("2018-10-31", "2018-11-30", "2018-12-31")
DRY = ("2018-03-31", "2018-04-30")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    names = np.load(SRC_NPZ, allow_pickle=True)["feature_names"].astype(str)
    y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
                allow_pickle=True)["y"].astype(np.int32)
    parcels = np.load("./splits/parcel_id_row.npy")
    s2_cols = np.flatnonzero(np.char.find(names, "47PQQ") >= 0)
    s2_names = names[s2_cols]
    log(f"{s2_cols.size} S2 columns; y = {y.size:,} rows")

    # --- parcel-atomic halving inside each crop, then pixels ------------------
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
    X = np.empty((rows.size, s2_cols.size), dtype=np.float32)
    srt = rows[order]
    step = 50_000
    buf = np.empty((rows.size, s2_cols.size), dtype=np.float32)
    for i in range(0, srt.size, step):
        buf[i:i + step] = Xsrc[srt[i:i + step]][:, s2_cols]
    del Xsrc
    X[order] = buf
    del buf
    ys = y[rows]
    tr = np.arange(tr_rows.size)
    te = np.arange(tr_rows.size, rows.size)
    log(f"X = {X.shape}")

    arms = {
        "wet3": np.flatnonzero([any(d in n for d in WET) for n in s2_names]),
        "dry2": np.flatnonzero([any(d in n for d in DRY) for n in s2_names]),
        "all5": np.arange(s2_names.size),
    }
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
    pc.to_csv(f"{OUT}/per_class_parcel_grouped.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(f"{OUT}/summary_parcel_grouped.csv", index=False,
                                 encoding="utf-8-sig")
    log("\n" + pc.pivot(index="class", columns="arm", values="f1").to_string())
    log("\n" + pd.DataFrame(summary).to_string(index=False))

    # the comparison the report actually needs: same crops, pixel split vs parcel split
    old = f"{OUT}/per_class_s{SEED}_n{N_COMPONENTS}.csv"
    if os.path.exists(old):
        o = pd.read_csv(old)
        m = (o[o.arm == "all5"][["class", "f1"]].rename(columns={"f1": "pixel_split"})
             .merge(pc[pc.arm == "all5"][["class", "f1", "test_parcels"]]
                    .rename(columns={"f1": "parcel_split"}), on="class"))
        m["delta"] = (m.parcel_split - m.pixel_split).round(4)
        log("\nall5 arm, pixel split vs parcel split:\n" + m.to_string(index=False))
