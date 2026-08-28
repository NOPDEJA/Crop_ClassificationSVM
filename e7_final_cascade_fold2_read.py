"""e7_final_cascade_fold2_read.py

E7 of docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md: THE single predeclared
fold-2 read for this entire plan. No other step in this plan's execution has
loaded, scored, or printed anything derived from fold-2 predictions.

FINAL CONFIG (M5 plus every component that passed its gate):
  Stage 1        M5's frozen model (untouched by every experiment)
  Stage 2        E4's retuned model: n_components=1200, gamma=0.5x rule,
                 C=30 (Gate G4 PASS), fit on s2mass's exact pool/calibration
                 rows with the subtype-mass sample weights (Gate G3 PASS,
                 already baked into this fit -- E4 reused s2mass's saved
                 pool/cal/weights verbatim, only the hyperparameters changed)
  Orchards       E4's retuned model: same hyperparameters as Stage 2,
                 UNWEIGHTED (Gate G4 PASS; Gate G5's weighted version FAILED,
                 so the E5 orchards-treatment model is NOT used)
  Plantation     M5's frozen, original model (Gate G5 FAILED, no change)
  Field          M5's frozen, original model (untouched by every experiment)

READ RULE:
  1. Build a "final" virtual FOLD-1 validation directory from already-fit
     prob_val arrays (E4's Stage-2 + orchards; M5's Stage-1/plantation/field),
     and select the operating-point cell (alpha2, alpha3) that maximises
     macro F1 on fold 1's CALIBRATION half via sweep_operating_point.py --
     mechanical, no fold-2 data touched.
  2. Predict E4's Stage-2 and orchards models on fold 2's candidates (M5's
     frozen stage2_test_idx.npy -- Stage 1 is untouched, so the candidate set
     is exactly M5's). Reuse M5's frozen plantation/field test predictions.
  3. Compose with the SAME hard-rule construction train_parcel_cascade.py
     uses (ratio2/ratio3 denominators from the calibration population's
     prior), at the cal-selected (alpha2, alpha3) from step 1.
  4. Score fold 2 ONCE with the strict convention (full population, non-crop
     truth mapped to 0). No second read under any outcome.

PLANNING SCENARIO (not a forecast): base 0.24-0.26, upside 0.28. Report
whatever lands, including a loss against M5's 0.2344.

Env:
  M5=<dir>   frozen baseline (default ./runs/s2_2018_3date_parcel_m5)
  E4=<dir>   gated E4 run (default ./runs/retune_stage2_orchards)
  OUT=<dir>  output (default ./runs/s2_2018_3date_parcel_e7_final)
"""
import csv
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

M5 = os.environ.get("M5", "./runs/s2_2018_3date_parcel_m5")
E4 = os.environ.get("E4", "./runs/retune_stage2_orchards")
OUT = os.environ.get("OUT", "./runs/s2_2018_3date_parcel_e7_final")
CHUNK = 400_000

GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},
          2: {2302, 2303, 2405},
          3: {2101, 2204, 2205}}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
SINK = 4
CROPS = sorted(c for codes in GROUPS.values() for c in codes)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class PlattCalibrated:
    """Must be defined here (not just imported) so joblib.load() can resolve
    the __main__.PlattCalibrated class the E4/M5 models were pickled under."""

    def __init__(self, base):
        self.base = base
        self.classes_ = base.classes_

    def _scores(self, X):
        s = self.base.decision_function(X)
        return s.reshape(-1, 1) if s.ndim == 1 else s

    def fit(self, X, y):
        raise NotImplementedError("E7 only loads and predicts with already-fit models")

    def predict_proba(self, X):
        S = self._scores(X)
        import numpy as _np
        P = _np.empty_like(S, dtype=_np.float64)
        for k, sig in enumerate(self.cal_):
            P[:, k] = 1.0 / (1.0 + _np.exp(-S[:, k])) if sig is None else sig.predict(S[:, k])
        tot = P.sum(1, keepdims=True)
        tot[tot == 0] = 1.0
        return P / tot


def chunked_proba(model, X, idx):
    out = np.zeros((idx.size, len(model.classes_)), dtype=np.float32)
    for s in range(0, idx.size, CHUNK):
        e = min(idx.size, s + CHUNK)
        out[s:e] = model.predict_proba(X[idx[s:e]]).astype(np.float32)
        log(f"    {e:,}/{idx.size:,}")
    return out


def reweight(P, ratio, alpha):
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


def build_final_val_dir():
    d = f"{OUT}/final_val"
    os.makedirs(d, exist_ok=True)
    shared = ["val_cal_idx.npy", "val_tune_idx.npy", "stage2_val_idx.npy",
              "stage1_pred.npy", "stage1_train_idx.npy", "stage1_route_oof_train.npy",
              "stage3_plantation_prob_val.npy", "stage3_field_prob_val.npy", "valid_cols.npy"]
    for f in shared:
        dst = f"{d}/{f}"
        if not os.path.exists(dst):
            os.link(f"{M5}/{f}", dst)
    dst = f"{d}/stage2_prob_val.npy"
    if not os.path.exists(dst):
        os.link(f"{E4}/stage2_retuned_prob_val.npy", dst)
    dst = f"{d}/stage3_orchards_prob_val.npy"
    if not os.path.exists(dst):
        os.link(f"{E4}/stage3_orchards_retuned_prob_val.npy", dst)
    m5 = json.load(open(f"{M5}/manifest.json"))
    json.dump({"npz": m5["npz"], "arm": "e7_final"}, open(f"{d}/manifest.json", "w"), indent=2)
    return d


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    log("=== E7: final consolidated cascade -- STEP 1: select operating point on fold 1 only ===")

    final_val_dir = build_final_val_dir()
    env = dict(os.environ, RUN_DIR=final_val_dir)
    r = subprocess.run([sys.executable, "-u", "sweep_operating_point.py"], env=env)
    if r.returncode:
        raise SystemExit("sweep_operating_point.py failed on the final config -- abort, fold 2 untouched")
    sel = json.load(open(f"{final_val_dir}/opsweep_selection.json"))
    alpha2, alpha3 = sel["alpha2"], sel["alpha3"]
    log(f"  SELECTED on fold 1's calibration half: alpha2={alpha2} alpha3={alpha3}"
        f"  (cal {sel['cal_macro_f1']:.4f}, tune {sel['tune_macro_f1']:.4f} -- informational, not the read)")

    with open(f"{OUT}/selected_operating_point.json", "w") as f:
        json.dump(sel, f, indent=2)

    log("\n=== STEP 2: predicting the final config on fold 2 (irreversible from here) ===")

    meta5 = json.load(open(f"{M5}/manifest.json"))
    npz = meta5["npz"]
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(np.int32)
    X = d["X"].astype(np.float32)
    del d
    valid_cols = np.load(f"{M5}/valid_cols.npy")
    if valid_cols.size != X.shape[1]:
        X = X[:, valid_cols]
    log(f"X {X.shape}")

    c_te = np.load(f"{M5}/stage2_test_idx.npy")     # Stage 1 untouched -> M5's candidates are exact
    va_cal = np.load(f"{M5}/val_cal_idx.npy")

    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK
    c_va_cal = np.intersect1d(np.load(f"{M5}/stage2_val_idx.npy"), va_cal)

    log("  predicting E4's retuned Stage-2 model on fold-2 candidates")
    m2 = joblib.load(f"{E4}/stage2_retuned_model.joblib")
    p2_test = chunked_proba(m2, X, c_te)
    np.save(f"{OUT}/stage2_prob_test.npy", p2_test)
    del m2

    log("  predicting E4's retuned orchards model on fold-2 candidates")
    m3_orch = joblib.load(f"{E4}/stage3_orchards_retuned_model.joblib")
    p3_orch_test = chunked_proba(m3_orch, X, c_te)
    np.save(f"{OUT}/stage3_orchards_prob_test.npy", p3_orch_test)
    del m3_orch

    log("  reusing M5's frozen plantation and field test predictions (untouched by every gate)")
    p3_plant_test = np.load(f"{M5}/stage3_plantation_prob_test.npy")
    p3_field_test = np.load(f"{M5}/stage3_field_prob_test.npy")

    e_classes = {
        1: np.array(sorted(GROUPS[1])),   # orchards -- E4's model classes_, same sorted crop codes
        2: np.array(sorted(GROUPS[2])),   # plantation -- M5's frozen model
        3: np.array(sorted(GROUPS[3])),   # field -- M5's frozen model
    }
    p3_test = {1: p3_orch_test, 2: p3_plant_test, 3: p3_field_test}
    classes2 = np.array([1, 2, 3, 4])

    # ratio2/ratio3 denominators from the CALIBRATION population's prior --
    # identical construction to train_parcel_cascade.py's own compose block
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in classes2])
    ratio2 = ((1.0 / classes2.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]
    ratio3 = {}
    for g, codes in GROUPS.items():
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, list(codes))))
        pi = np.array([(y[own] == c).mean() for c in e_classes[g]])
        ratio3[g] = ((1.0 / len(codes)) / np.where(pi > 0, pi, 1e-9))[None, :]

    log(f"  composing at the predeclared cell alpha2={alpha2} alpha3={alpha3}")
    hard = np.zeros(y.size, dtype=np.int32)
    g_hat = classes2[reweight(p2_test, ratio2, alpha2).argmax(1)]
    for g in GROUPS:
        sel_mask = g_hat == g
        if sel_mask.any():
            hard[c_te[sel_mask]] = e_classes[g][reweight(p3_test[g][sel_mask], ratio3[g], alpha3).argmax(1)]
    np.save(f"{OUT}/pred_hard.npy", hard)

    te = np.flatnonzero(np.load("./splits/split_assign.npy") == 2)
    assert np.array_equal(np.sort(c_te), np.intersect1d(c_te, te)), "c_te must be a subset of fold 2"

    log("\n=== STEP 3: scoring fold 2 ONCE, strict convention ===")
    y_eval = np.where(np.isin(y, CROPS), y, 0).astype(np.int32)
    rep = classification_report(y_eval[te], hard[te], labels=CROPS, output_dict=True, zero_division=0)
    macro_f1 = rep["macro avg"]["f1-score"]
    weighted_f1 = rep["weighted avg"]["f1-score"]
    log(f"  FINAL strict test macro F1 = {macro_f1:.4f}  weighted F1 = {weighted_f1:.4f}"
        f"  ({te.size:,} rows)")

    m5_macro = meta5.get("results", {}).get("hard", {}).get("macro_f1")
    log(f"  M5 baseline strict test macro F1 = {m5_macro}")
    log(f"  delta vs M5: {macro_f1 - m5_macro:+.4f}" if m5_macro is not None else "  (M5 baseline unavailable)")
    log("  planning scenario was base 0.24-0.26, upside 0.28 (not a forecast)")

    with open(f"{OUT}/report_hard.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["lu_code", "precision", "recall", "f1", "support"])
        for c in CROPS:
            r = rep[str(c)]
            w.writerow([c, round(r["precision"], 4), round(r["recall"], 4),
                       round(r["f1-score"], 4), int(r["support"])])

    manifest = {"m5": M5, "e4": E4, "npz": npz,
               "selected_operating_point": {"alpha2": alpha2, "alpha3": alpha3},
               "results": {"hard": {"rows": int(te.size), "macro_f1": round(float(macro_f1), 4),
                                    "weighted_f1": round(float(weighted_f1), 4)}},
               "m5_baseline_macro_f1": m5_macro,
               "delta_vs_m5": round(float(macro_f1 - m5_macro), 4) if m5_macro is not None else None,
               "gates_carried": {"G3_subtype_mass_stage2_weights": "PASS (baked into E4's Stage-2 fit)",
                                 "G4_stage2_orchards_retune": "PASS",
                                 "G5_stage3_subtype_mass_plantation_orchards": "FAIL (not used)"},
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"\nwrote {OUT}/manifest.json and {OUT}/report_hard.csv")
    log("DONE -- fold 2 has now been read exactly once for this plan")
