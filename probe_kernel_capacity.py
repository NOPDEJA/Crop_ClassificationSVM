"""probe_kernel_capacity.py

Stages 2 and 3 search Nystroem n_components over [50, 100, 150]. That ceiling was
never justified against a measurement, and probe_dry_season.py found the 5-date
feature set is capacity-limited: its advantage over the 3-date set grows from +0.033
to +0.063 when components go 300 -> 800. If the curve is still climbing at 150, the
species stages are throttled and the 5-date arm will under-deliver for a reason that
has nothing to do with the features.

Same balanced 13-crop subsample as the other probes, which is the closest cheap
analogue of Stage 3's fitting conditions.
"""
import time
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score

SRC_NPZ = "./aligned_features/svm_dem_s1_s2_features_labels.npz"
SRC_X = "./aligned_features/_unpacked/X_src.npy"
OUT = "./runs/probe_dry_season"
SEED, N_PER_CROP = 42, 8_000
CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}
WET = ("2018-10-31", "2018-11-30", "2018-12-31")
GRID = [50, 100, 150, 300, 600, 1000]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    names = np.load(SRC_NPZ, allow_pickle=True)["feature_names"].astype(str)
    y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
                allow_pickle=True)["y"].astype(np.int32)
    s2_cols = np.flatnonzero(np.char.find(names, "47PQQ") >= 0)
    s2_names = names[s2_cols]
    take = [rng.choice(np.flatnonzero(y == c),
                       size=min(N_PER_CROP, (y == c).sum()), replace=False)
            for c in CROPS]
    rows = np.sort(np.concatenate(take))
    Xsrc = np.load(SRC_X, mmap_mode="r")
    X = np.empty((rows.size, s2_cols.size), dtype=np.float32)
    for i in range(0, rows.size, 50_000):
        X[i:i + 50_000] = Xsrc[rows[i:i + 50_000]][:, s2_cols]
    del Xsrc
    ys = y[rows]
    perm = rng.permutation(rows.size)
    tr, te = perm[:rows.size // 2], perm[rows.size // 2:]

    arms = {"wet3": np.flatnonzero([any(d in n for d in WET) for n in s2_names]),
            "all5": np.arange(s2_names.size)}
    out = []
    for arm, cols in arms.items():
        for nc in GRID:
            t0 = time.time()
            pipe = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                Nystroem(gamma=1.0 / cols.size, n_components=nc, random_state=SEED),
                OneVsRestClassifier(LinearSVC(C=1.0, dual=False, max_iter=5000,
                                              random_state=SEED), n_jobs=-1))
            pipe.fit(X[tr][:, cols], ys[tr])
            pred = pipe.predict(X[te][:, cols])
            r = {"arm": arm, "n_components": nc,
                 "macro_f1": round(f1_score(ys[te], pred, average="macro", zero_division=0), 4),
                 "accuracy": round(accuracy_score(ys[te], pred), 4),
                 "fit_seconds": round(time.time() - t0, 1)}
            out.append(r)
            log(f"  {r}")

    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/kernel_capacity.csv", index=False, encoding="utf-8-sig")
    log("\n" + df.pivot(index="n_components", columns="arm",
                        values="macro_f1").to_string())
