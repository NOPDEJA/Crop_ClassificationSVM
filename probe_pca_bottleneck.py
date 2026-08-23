"""probe_pca_bottleneck.py

Stage 1 runs PCA(n_components=10) before the Nystroem kernel. The 3-date arm feeds
it 24 columns; the 5-date arm would feed it 40. If 10 components is already a hard
bottleneck, the two extra acquisition dates get projected away before the classifier
ever sees them and a 5-date retrain buys nothing -- so measure that before spending
the training time, not after.

Same subsample and architecture as probe_dry_season.py, with PCA inserted where the
real stage has it. Arms: wet3 (24 cols) vs all5 (40 cols), each at several PCA widths,
plus a no-PCA reference.
"""
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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
SEED, N_PER_CROP, N_COMPONENTS = 42, 8_000, 300
CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}
WET = ("2018-10-31", "2018-11-30", "2018-12-31")


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
    log(f"X = {X.shape}")

    arms = {"wet3": np.flatnonzero([any(d in n for d in WET) for n in s2_names]),
            "all5": np.arange(s2_names.size)}
    perm = rng.permutation(rows.size)
    tr, te = perm[:rows.size // 2], perm[rows.size // 2:]

    out = []
    for arm, cols in arms.items():
        # how much of the arm's variance does PCA(10) actually keep?
        z = StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(X[tr][:, cols]))
        ev = PCA(n_components=min(20, cols.size), random_state=SEED).fit(z).explained_variance_ratio_
        log(f"{arm}: PCA cumulative variance @5/10/15 = "
            f"{ev[:5].sum():.4f} / {ev[:10].sum():.4f} / {ev[:15].sum():.4f}")

        for npca in [10, 14, 20, None]:
            if npca is not None and npca > cols.size:
                continue
            steps = [SimpleImputer(strategy="median"), StandardScaler()]
            if npca:
                steps.append(PCA(n_components=npca, random_state=SEED))
            width = npca or cols.size
            steps += [Nystroem(gamma=1.0 / width, n_components=N_COMPONENTS,
                               random_state=SEED),
                      OneVsRestClassifier(LinearSVC(C=1.0, dual=False, max_iter=5000,
                                                    random_state=SEED), n_jobs=-1)]
            pipe = make_pipeline(*steps)
            pipe.fit(X[tr][:, cols], ys[tr])
            pred = pipe.predict(X[te][:, cols])
            r = {"arm": arm, "n_features": int(cols.size), "pca": npca or "none",
                 "macro_f1": round(f1_score(ys[te], pred, average="macro", zero_division=0), 4),
                 "accuracy": round(accuracy_score(ys[te], pred), 4),
                 "pca_var_kept": round(float(ev[:npca].sum()), 4) if npca else 1.0}
            out.append(r)
            log(f"    {r}")

    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/pca_bottleneck.csv", index=False, encoding="utf-8-sig")
    log("\n" + df.pivot(index="pca", columns="arm", values="macro_f1").to_string())
