"""probe_dry_season.py

Does the dry-season S2 window carry crop information the current arm throws away?

`s2_2018_3date` uses Oct-31 / Nov-30 / Dec-31 only. Five dates exist on disk; the
two unused ones are 2018-03-31 and 2018-04-30. That matters here specifically:
Para rubber is deciduous and winters (sheds leaves) Feb-Mar, refoliating in April,
while durian, mangosteen, rambutan and longan are evergreen. Oct-Dec is the tail of
the monsoon, when every one of them is at full canopy -- which is the window the
model currently sees. §6.5 measured the consequence: the cascade redistributes
rubber into all twelve other crops.

This is the cheap test before a ~30 h retrain. Same architecture family as the real
stages (Nystroem RBF -> LinearSVC -> OvR), equal samples per crop so the prior is
uniform in every arm and only the FEATURES differ, on a stratified subsample.

Arms:
  wet3   the 24 columns the current arm uses      (Oct, Nov, Dec)
  dry2   the 16 unused columns                    (Mar, Apr)
  all5   all 40 S2 columns                        (Mar, Apr, Oct, Nov, Dec)

Writes runs/probe_dry_season/{per_class.csv,summary.csv,ndvi_phenology.csv}
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, classification_report

SRC_NPZ = "./aligned_features/svm_dem_s1_s2_features_labels.npz"
SRC_X = "./aligned_features/_unpacked/X_src.npy"
OUT = "./runs/probe_dry_season"
import os
SEED = int(os.environ.get("PROBE_SEED", 42))
N_PER_CROP = 8_000
N_COMPONENTS = int(os.environ.get("PROBE_NCOMP", 300))

CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}
WET = ("2018-10-31", "2018-11-30", "2018-12-31")
DRY = ("2018-03-31", "2018-04-30")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    names = np.load(SRC_NPZ, allow_pickle=True)["feature_names"].astype(str)
    y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
                allow_pickle=True)["y"].astype(np.int32)
    s2_cols = np.flatnonzero(np.char.find(names, "47PQQ") >= 0)
    s2_names = names[s2_cols]
    log(f"{s2_cols.size} S2 columns; y = {y.size:,} rows")

    # stratified subsample, equal per crop -> uniform prior in every arm
    take = []
    for code in CROPS:
        idx = np.flatnonzero(y == code)
        k = min(N_PER_CROP, idx.size)
        take.append(rng.choice(idx, size=k, replace=False))
        log(f"  {CROPS[code]:12s} {idx.size:>10,} available -> {k:,}")
    rows = np.sort(np.concatenate(take))
    log(f"sampling {rows.size:,} rows from the memmap")

    Xsrc = np.load(SRC_X, mmap_mode="r")
    X = np.empty((rows.size, s2_cols.size), dtype=np.float32)
    step = 50_000
    for i in range(0, rows.size, step):
        X[i:i + step] = Xsrc[rows[i:i + step]][:, s2_cols]
    del Xsrc
    ys = y[rows]
    log(f"X = {X.shape}")

    # phenology evidence: mean NDVI per crop per date, the mechanism itself
    ndvi = {n.split("_")[-1].replace(".tif", ""): j
            for j, n in enumerate(s2_names) if n.startswith("NDVI_")}
    phen = pd.DataFrame(
        {d: [np.nanmean(X[ys == c, j]) for c in CROPS] for d, j in sorted(ndvi.items())},
        index=[CROPS[c] for c in CROPS]).round(4)
    phen.to_csv(f"{OUT}/ndvi_phenology.csv", encoding="utf-8-sig")
    log("mean NDVI by crop x date:\n" + phen.to_string())

    arms = {
        "wet3": np.flatnonzero([any(d in n for d in WET) for n in s2_names]),
        "dry2": np.flatnonzero([any(d in n for d in DRY) for n in s2_names]),
        "all5": np.arange(s2_names.size),
    }

    perm = rng.permutation(rows.size)
    cut = rows.size // 2
    tr, te = perm[:cut], perm[cut:]
    labels = list(CROPS)

    per_class, summary = [], []
    for arm, cols in arms.items():
        log(f"--- {arm}: {cols.size} features")
        t0 = time.time()
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            # Nystroem has no gamma="scale". After StandardScaler the variance is 1,
            # so sklearn's 'scale' rule (1/(n_features*X.var())) reduces to 1/n_features
            # -- and it must be recomputed per arm, since the arms differ in width.
            Nystroem(gamma=1.0 / cols.size, n_components=N_COMPONENTS,
                     random_state=SEED),
            OneVsRestClassifier(LinearSVC(C=1.0, dual=False, max_iter=5000,
                                          random_state=SEED), n_jobs=-1))
        pipe.fit(X[tr][:, cols], ys[tr])
        pred = pipe.predict(X[te][:, cols])
        rep = classification_report(ys[te], pred, labels=labels,
                                    output_dict=True, zero_division=0)
        for lab in labels:
            r = rep[str(lab)]
            per_class.append({"arm": arm, "class": CROPS[lab],
                              "precision": round(r["precision"], 4),
                              "recall": round(r["recall"], 4),
                              "f1": round(r["f1-score"], 4)})
        s = {"arm": arm, "n_features": int(cols.size),
             "macro_f1": round(f1_score(ys[te], pred, average="macro", zero_division=0), 4),
             "accuracy": round(accuracy_score(ys[te], pred), 4),
             "fit_seconds": round(time.time() - t0, 1)}
        summary.append(s)
        log(f"    {s}")

    pd.DataFrame(per_class).to_csv(f"{OUT}/per_class_s{SEED}_n{N_COMPONENTS}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(f"{OUT}/summary_s{SEED}_n{N_COMPONENTS}.csv", index=False, encoding="utf-8-sig")
    log("\n" + pd.DataFrame(per_class).pivot(index="class", columns="arm",
                                             values="f1").to_string())
    log("\n" + pd.DataFrame(summary).to_string(index=False))
