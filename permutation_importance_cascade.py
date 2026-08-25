"""permutation_importance_cascade.py

Permutation feature importance for the reported cascade, on Stage 1 and on the
orchards expert (D9 of docs/PLAN_2026-08-26_WEIGHTED_RUN.md).

Those two are picked deliberately. Stage 1 decides candidacy, which is the gate
everything downstream sits behind, and the orchards expert is where the seven rare
crops are actually separated -- so if MTCI and raw B11 (the two columns M3 added)
earn their place anywhere, it is there. Stages 2 and 3's other experts are left for
later; the point here is to answer one question, not to produce thirty numbers.

METHOD. A column is shuffled across rows and the model re-scored; the drop in macro
F1 is that column's importance. Shuffling rather than dropping keeps the fitted
model valid -- refitting without a column would measure a different model. Because
the three dates of each index are highly correlated, a single-column permutation
UNDERSTATES a family's importance: shuffling October NDVI leaves November and
December NDVI to carry the signal. The per-family totals printed at the end are the
honest way to read this, and even they are a lower bound.

The population is the VALIDATION fold, never fold 2. Stage 1 is scored over a
random sample of it; the orchards expert is scored over the validation rows whose
TRUE label is one of the seven orchard crops, which is the population it was fitted
against.

Env:
  RUN_DIR=<dir>    run whose models to probe
  N_ROWS=150000    rows sampled per model
  N_REPEATS=3      shuffles per column
"""
import csv
import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import f1_score

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m5")
N_ROWS = int(os.environ.get("N_ROWS", 150_000))
N_REPEATS = int(os.environ.get("N_REPEATS", 3))
SPLIT_ASSIGN = "./splits/split_assign.npy"
SEED = 42
ECON = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
WATER = {4101, 4102, 4103, 4201, 4202, 4203}
FOREST = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
ORCHARDS = [2403, 2404, 2407, 2413, 2416, 2419, 2420]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_model(path):
    """Unpickle a model saved by train_parcel_cascade.py run as a script.

    joblib records PlattCalibrated's module as `__main__`, because that is where
    it lived when the run pickled it. Any other process therefore has to put the
    class back under that name before unpickling, or find_class raises.
    """
    import sys
    import train_parcel_cascade as tpc
    sys.modules["__main__"].PlattCalibrated = tpc.PlattCalibrated
    return joblib.load(path)


def family(name):
    """'NDVI_47PQQ_2018-10-31.tif' -> ('NDVI', 'Oct')."""
    stem = name.rsplit(".tif", 1)[0]
    idx = stem.split("_47PQQ_")[0]
    month = {"10": "Oct", "11": "Nov", "12": "Dec"}[stem[-5:-3]]
    return idx, month


def macro_f1(model, X, y, labels):
    return f1_score(y, model.predict(X), labels=labels, average="macro",
                    zero_division=0)


def probe(model, X, y, labels, names, tag, rng):
    base = macro_f1(model, X, y, labels)
    log(f"{tag}: baseline macro F1 {base:.4f} over {y.size:,} rows, "
        f"{len(labels)} classes")
    rows = []
    for j, name in enumerate(names):
        keep = X[:, j].copy()
        drops = []
        for _ in range(N_REPEATS):
            X[:, j] = keep[rng.permutation(X.shape[0])]
            drops.append(base - macro_f1(model, X, y, labels))
        X[:, j] = keep
        idx, month = family(name)
        rows.append({"model": tag, "feature": name, "index": idx, "month": month,
                     "baseline_macro_f1": round(float(base), 4),
                     "importance": round(float(np.mean(drops)), 5),
                     "std": round(float(np.std(drops)), 5)})
        log(f"  {idx:<11}{month}   {np.mean(drops):+.5f}")
    return rows


def main():
    manifest = json.load(open(f"{RUN}/manifest.json"))
    if manifest.get("class_weight") == "sqrt":
        import sklearn
        sklearn.set_config(enable_metadata_routing=True)
    d = np.load(manifest["npz"], allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    valid_cols = np.load(f"{RUN}/valid_cols.npy")
    X = d["X"].astype(np.float32)
    if valid_cols.size != X.shape[1]:
        X = X[:, valid_cols]
        names = [names[c] for c in valid_cols]
    y = d["y"].astype(np.int32)
    va = np.flatnonzero(np.load(SPLIT_ASSIGN) == 1)
    rng = np.random.default_rng(SEED)
    log(f"{RUN}: {len(names)} features, validation fold {va.size:,} rows")

    out = []

    # ---- Stage 1, over a random sample of the whole validation fold ----------
    sup = np.where(np.isin(y, list(ECON)), 1,
          np.where(np.isin(y, list(WATER)), 2,
          np.where(np.isin(y, list(FOREST)), 4, 3))).astype(np.int32)
    idx = np.sort(rng.choice(va, min(N_ROWS, va.size), replace=False))
    out += probe(load_model(f"{RUN}/stage1_model.joblib"),
                 X[idx].copy(), sup[idx], [1, 2, 3, 4], names, "stage1", rng)

    # ---- orchards expert, over its own true-membership population ------------
    own = va[np.isin(y[va], ORCHARDS)]
    idx = np.sort(rng.choice(own, min(N_ROWS, own.size), replace=False))
    out += probe(load_model(f"{RUN}/stage3_orchards_model.joblib"),
                 X[idx].copy(), y[idx], ORCHARDS, names, "stage3_orchards", rng)

    with open(f"{RUN}/permutation_importance.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    # ---- per-family totals, which is how correlated dates should be read -----
    fam = []
    for tag in ("stage1", "stage3_orchards"):
        sub = [r for r in out if r["model"] == tag]
        tot = {}
        for r in sub:
            tot[r["index"]] = tot.get(r["index"], 0.0) + r["importance"]
        log(f"\n{tag}: importance summed over the three dates")
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
            log(f"  {k:<12}{v:+.5f}")
            fam.append({"model": tag, "index": k, "importance_3date_sum": round(v, 5)})
    with open(f"{RUN}/permutation_importance_by_index.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fam[0]))
        w.writeheader(); w.writerows(fam)
    log(f"\nwrote {RUN}/permutation_importance{{,_by_index}}.csv")


if __name__ == "__main__":
    main()
