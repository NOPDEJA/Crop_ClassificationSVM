"""score_fold2.py

The single, gated read of fold 2 for a run that was trained with SKIP_TEST=1.

`train_parcel_cascade.py` with SKIP_TEST=1 never touches fold 2, which is what
makes the one read afterwards a real held-out evaluation instead of the third or
fourth look at the same rows. This script performs that read: it loads the saved
models, predicts fold 2, composes the cascade at a PREDECLARED operating point and
scores it. It refits nothing and selects nothing.

Run it only when the run's predeclaration says the gate opened, and only once.
The alphas must come from `opsweep_selection.json` -- that is, they must have been
selected on fold 1's calibration half before this script was ever invoked. Passing
alphas chosen any other way turns the number into a tuned one, quietly.

Composition and scoring are copied from train_parcel_cascade.py's own compose block
so the number means exactly what M5's 0.2344 means: hard routing, the whole test
fold, non-crop truth mapped to 0, 13-crop macro F1.

Env:
  RUN_DIR=<dir>       run to score (must contain the joblib models)
  ALPHA2/ALPHA3       operating point; default reads opsweep_selection.json
  CONFIRM=yes         required, so this cannot happen by reflex
"""
import csv
import json
import os
import time

import joblib
import numpy as np
from sklearn.metrics import classification_report

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_weighted")
SPLIT_ASSIGN = "./splits/split_assign.npy"
CHUNK = int(os.environ.get("PRED_CHUNK_OVERRIDE", 400_000))
SINK = 4
STD = {1: ("orchards", [2403, 2404, 2407, 2413, 2416, 2419, 2420]),
       2: ("plantation", [2302, 2303, 2405]),
       3: ("field", [2101, 2204, 2205])}
MERGED = {1: ("tree", [2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420]),
          3: ("field", [2101, 2204, 2205])}
CROPS = sorted(c for _, cs in STD.values() for c in cs)


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


def reweight(P, ratio, alpha):
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


def chunked_proba(model, X, idx):
    out = np.zeros((idx.size, len(model.classes_)), dtype=np.float32)
    for s in range(0, idx.size, CHUNK):
        e = min(idx.size, s + CHUNK)
        out[s:e] = model.predict_proba(X[idx[s:e]]).astype(np.float32)
        log(f"    {e:,}/{idx.size:,}")
    return out


def main():
    if os.environ.get("CONFIRM") != "yes":
        raise SystemExit("refusing to read fold 2 without CONFIRM=yes")
    manifest = json.load(open(f"{RUN}/manifest.json"))
    if manifest.get("class_weight") == "sqrt":
        import sklearn                       # the models carry routing requests
        sklearn.set_config(enable_metadata_routing=True)
    groups = MERGED if os.path.exists(f"{RUN}/stage3_tree_model.joblib") else STD
    gname = {g: n for g, (n, _) in groups.items()}
    codes = {g: c for g, (_, c) in groups.items()}

    sel = json.load(open(f"{RUN}/opsweep_selection.json"))
    a2 = float(os.environ.get("ALPHA2", sel["alpha2"]))
    a3 = float(os.environ.get("ALPHA3", sel["alpha3"]))
    log(f"FOLD 2 READ -- {RUN}   operating point ({a2}, {a3})")
    log(f"  selected on the calibration half; its tune-half score was "
        f"{sel['tune_macro_f1']:.4f}")

    d = np.load(manifest["npz"], allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int32)
    valid_cols = np.load(f"{RUN}/valid_cols.npy")
    if valid_cols.size != X.shape[1]:
        X = X[:, valid_cols]
    assert X.shape[1] == manifest["n_features"], (X.shape, manifest["n_features"])
    asg = np.load(SPLIT_ASSIGN)
    te = np.flatnonzero(asg == 2)
    log(f"  test fold {te.size:,} rows, {X.shape[1]} features")

    # ---- Stage 1 on fold 2 -> candidacy, exactly the deployed argmax rule ----
    m1 = load_model(f"{RUN}/stage1_model.joblib")
    log("  stage 1 on fold 2")
    p1 = chunked_proba(m1, X, te)
    np.save(f"{RUN}/stage1_prob_test.npy", p1)
    econ_col = list(m1.classes_).index(1)
    c_te = te[m1.classes_[p1.argmax(1)] == 1]
    pe = p1[m1.classes_[p1.argmax(1)] == 1, econ_col]
    del p1
    log(f"  stage-2 candidates on fold 2: {c_te.size:,} ({c_te.size / te.size:.1%})")

    # ---- reweighting denominators: the CALIBRATION half, as at fit time ----
    g_of = np.zeros(y.size, dtype=np.int32)
    for g in groups:
        g_of[np.isin(y, codes[g])] = g
    g_of[g_of == 0] = SINK
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    c_va_cal = np.intersect1d(c_va, np.load(f"{RUN}/val_cal_idx.npy"))

    m2 = load_model(f"{RUN}/stage2_model.joblib")
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in m2.classes_])
    ratio2 = ((1.0 / m2.classes_.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]
    log("  stage 2 on fold-2 candidates")
    p2 = chunked_proba(m2, X, c_te)
    np.save(f"{RUN}/stage2_prob_test.npy", p2)
    np.save(f"{RUN}/stage2_test_idx.npy", c_te)

    experts, e_classes, ratio3, p3 = {}, {}, {}, {}
    for g in groups:
        experts[g] = load_model(f"{RUN}/stage3_{gname[g]}_model.joblib")
        e_classes[g] = experts[g].classes_
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, codes[g])))
        pi = np.array([(y[own] == c).mean() for c in e_classes[g]])
        ratio3[g] = ((1.0 / len(codes[g])) / np.where(pi > 0, pi, 1e-9))[None, :]
        log(f"  stage 3 {gname[g]} on fold-2 candidates")
        p3[g] = chunked_proba(experts[g], X, c_te)
        np.save(f"{RUN}/stage3_{gname[g]}_prob_test.npy", p3[g])

    # ---- compose (hard routing, and the joint rule for reference) ----
    hard = np.zeros(y.size, dtype=np.int32)
    g_hat = m2.classes_[reweight(p2, ratio2, a2).argmax(1)]
    for g in groups:
        m = g_hat == g
        if m.any():
            hard[c_te[m]] = e_classes[g][reweight(p3[g][m], ratio3[g], a3).argmax(1)]

    joint = np.zeros(y.size, dtype=np.int32)
    g_col = {g: list(m2.classes_).index(g) for g in m2.classes_}
    scores = np.zeros((c_te.size, len(CROPS)), dtype=np.float32)
    for j, code in enumerate(CROPS):
        g = next(g for g, cs in codes.items() if code in cs)
        k = list(e_classes[g]).index(code)
        scores[:, j] = pe * p2[:, g_col[g]] * p3[g][:, k]
    not_crop = (1.0 - pe) + pe * p2[:, g_col[SINK]]
    best = scores.argmax(1)
    take = scores[np.arange(c_te.size), best] > not_crop
    joint[c_te[take]] = np.array(CROPS)[best[take]]

    np.save(f"{RUN}/pred_hard.npy", hard)
    np.save(f"{RUN}/pred_joint.npy", joint)

    # ---- score, strict convention ----
    y_eval = np.where(np.isin(y, CROPS), y, 0).astype(np.int32)
    results = {}
    for nm, pred in (("hard", hard), ("joint", joint)):
        rep = classification_report(y_eval[te], pred[te], labels=CROPS,
                                    output_dict=True, zero_division=0)
        mf, wf = rep["macro avg"]["f1-score"], rep["weighted avg"]["f1-score"]
        log(f"  {nm}: rows {te.size:,}  macro F1 {mf:.4f}  weighted F1 {wf:.4f}")
        results[nm] = {"rows": int(te.size), "macro_f1": round(mf, 4),
                       "weighted_f1": round(wf, 4)}
        with open(f"{RUN}/report_{nm}.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["lu_code", "precision", "recall", "f1", "support"])
            for c in CROPS:
                r = rep[str(c)]
                w.writerow([c, round(r["precision"], 4), round(r["recall"], 4),
                            round(r["f1-score"], 4), int(r["support"])])

    assert np.all(hard[asg != 2] == 0), "predictions leaked outside fold 2"
    manifest["alpha2"], manifest["alpha3"] = a2, a3
    manifest["results"] = results
    manifest["fold2_read"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{RUN}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("DONE -- fold 2 has now been read for this run. Do not run this again.")


if __name__ == "__main__":
    main()
