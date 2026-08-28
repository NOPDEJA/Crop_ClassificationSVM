"""e4_retune_stage2_orchards.py

E4 of docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md: an honest hyperparameter
retune of Stage 2 and the orchards expert. Current values (P23: n_components=600,
gamma=None i.e. 1/n_features, C=10) were selected by the v2 randomized searches
on a pixel-leaky inner CV scored on accuracy -- the largest known validity
defect still standing (report Sec.8.3). Every prior search hit its capacity
ceiling, so capacity may be undersized too.

DESIGN, mapped onto this repo's frozen artifacts (no rng-sequence replay
needed -- unlike s2mass_stage2.py, nothing here has to be byte-identical to
M5's pool, because this is a NEW configuration, not a reproducibility check):

  Stage-2 search population: cap(cand_tr, g_of, 50_000) -- 50k/group, 200k
    total, drawn from the OUT-OF-FOLD econ candidates in fold 0 (M5's saved
    stage1_train_idx.npy / stage1_route_oof_train.npy). A subsample of the
    800k production pool, sized to keep the search overnight-ish.
  Orchards search population: cap(orchard rows in fold 0, y, PER_LU_CAP) --
    the SAME "natural capped fit set" production already uses; no further
    subsampling, so the search and the final refit use one population.

  GroupKFold(3) on parcel_id_row, scoring='f1_macro', grid C x gamma x
  n_components (<=24 cells), N_JOBS=1 throughout (BLAS-crash lesson).

  Gate G4 refit, paired against the CURRENT BEST KNOWN config (G3 passed, so
  that is M5 + s2mass's subtype-mass Stage-2, per the plan's "M5+G3 weights,
  run them jointly"):
    control:   s2mass's already-fit, already-gated Stage-2 TREATMENT arm
               (production hyperparams P23, subtype-mass weights) + M5's
               original (unweighted, untuned) orchards expert.
    treatment: Stage-2 refit on the IDENTICAL s2mass pool/cal/weights (only
               the hyperparameters change) + the orchards expert refit on its
               natural capped set at the new hyperparameters (unweighted,
               matching M5's style -- E5 is where Stage-3 weighting is tested,
               not here).
    Plantation and field experts are untouched in both arms (frozen from M5).

  Carry iff treatment tune macro F1 >= control + 0.002, alive-crops guard
  (crop count with F1 >= 0.01 must not fall).

Env:
  M5=<dir>        frozen run (default ./runs/s2_2018_3date_parcel_m5)
  S2MASS=<dir>    gated s2mass run, source of Stage-2's pool/cal/weights
                  (default ./runs/s2_2018_3date_parcel_s2mass)
  OUT=<dir>       output (default ./runs/retune_stage2_orchards)
"""
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import _SigmoidCalibration
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import RANDOM_STATE, PER_LU_CAP

M5 = os.environ.get("M5", "./runs/s2_2018_3date_parcel_m5")
S2MASS = os.environ.get("S2MASS", "./runs/s2_2018_3date_parcel_s2mass")
OUT = os.environ.get("OUT", "./runs/retune_stage2_orchards")
PARCEL_ID = "./splits/parcel_id_row.npy"

SEARCH_SEED = 777                       # independent of M5's replay rng -- new populations
STAGE2_SEARCH_PER_GROUP = 50_000
GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},
          2: {2302, 2303, 2405},
          3: {2101, 2204, 2205}}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
ORCHARD_CODES = sorted(GROUPS[1])
SINK = 4
CHUNK = 400_000
CALIB_MAX = 300_000

C_GRID = [1, 10, 30]
GAMMA_MULT = [0.25, 0.5, 1, 2]
COMP_STAGE2 = [600, 1200]
COMP_ORCH = [800, 1200]

os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cap(idx, labels, per_class, gen):
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else gen.choice(w, per_class, replace=False))
    return np.sort(np.concatenate(parts))


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("nyst", Nystroem(kernel="rbf", random_state=RANDOM_STATE)),
        ("clf", OneVsRestClassifier(
            LinearSVC(class_weight=None, max_iter=5000, random_state=RANDOM_STATE), n_jobs=1)),
    ])


def run_search(X_pop, y_pop, groups_pop, comp_grid, gamma_grid, tag):
    log(f"--- {tag} search: {X_pop.shape[0]:,} rows, {len(comp_grid) * len(gamma_grid) * len(C_GRID)} candidates")
    t0 = time.time()
    splits = list(GroupKFold(n_splits=3).split(X_pop, y_pop, groups=groups_pop))
    for i, (a, b) in enumerate(splits):
        log(f"    fold {i}: train {a.size:,} test {b.size:,}")
    grid = {"nyst__n_components": comp_grid, "nyst__gamma": gamma_grid,
            "clf__estimator__C": C_GRID}
    gs = GridSearchCV(make_pipeline(), grid, scoring="f1_macro", cv=splits,
                      n_jobs=1, refit=False, verbose=1)
    gs.fit(X_pop, y_pop)
    df = pd.DataFrame(gs.cv_results_)
    df.to_csv(f"{OUT}/search_{tag}.csv", index=False, encoding="utf-8-sig")
    best = df.loc[df["mean_test_score"].idxmax()]
    winner = {"n_components": int(best["param_nyst__n_components"]),
              "gamma": float(best["param_nyst__gamma"]),
              "C": float(best["param_clf__estimator__C"]),
              "mean_test_score": round(float(best["mean_test_score"]), 4),
              "std_test_score": round(float(best["std_test_score"]), 4)}
    log(f"    {tag} winner: {winner}  ({(time.time() - t0) / 60:.1f} min)")
    return winner


class PlattCalibrated:
    def __init__(self, base):
        self.base = base
        self.classes_ = base.classes_

    def _scores(self, X):
        s = self.base.decision_function(X)
        return s.reshape(-1, 1) if s.ndim == 1 else s

    def fit(self, X, y):
        S = self._scores(X)
        self.cal_ = []
        for k, c in enumerate(self.classes_):
            yk = (y == c).astype(int)
            if yk.sum() < 2 or yk.sum() == yk.size:
                self.cal_.append(None)
                continue
            sig = _SigmoidCalibration()
            sig.fit(S[:, k], yk)
            self.cal_.append(sig)
        return self

    def predict_proba(self, X):
        S = self._scores(X)
        P = np.empty_like(S, dtype=np.float64)
        for k, sig in enumerate(self.cal_):
            P[:, k] = 1.0 / (1.0 + np.exp(-S[:, k])) if sig is None else sig.predict(S[:, k])
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


def fit_final(X, y_target, fit_idx, cal_idx, p, tag, weighted=False, sw=None):
    log(f"  {tag}: final fit on {fit_idx.size:,} rows  params={p}")
    if weighted:
        # set_fit_request() itself requires metadata routing to already be
        # enabled at the point it's called, so the pipeline must be built
        # INSIDE the config context, not just fitted inside it.
        with sklearn.config_context(enable_metadata_routing=True):
            base_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler().set_fit_request(sample_weight=False)),
                ("nyst", Nystroem(kernel="rbf", n_components=p["n_components"], gamma=p["gamma"],
                                  random_state=RANDOM_STATE)),
                ("svc", LinearSVC(C=p["C"], class_weight=None, max_iter=5000,
                                  random_state=RANDOM_STATE).set_fit_request(sample_weight=True)),
            ])
            base = OneVsRestClassifier(base_pipe)
            base.fit(X[fit_idx], y_target[fit_idx], sample_weight=sw)
    else:
        base_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("nyst", Nystroem(kernel="rbf", n_components=p["n_components"], gamma=p["gamma"],
                              random_state=RANDOM_STATE)),
            ("svc", LinearSVC(C=p["C"], class_weight=None, max_iter=5000,
                              random_state=RANDOM_STATE)),
        ])
        base = OneVsRestClassifier(base_pipe)
        base.fit(X[fit_idx], y_target[fit_idx])
    log(f"  {tag}: sigmoid calibration on {cal_idx.size:,} rows")
    return PlattCalibrated(base).fit(X[cal_idx], y_target[cal_idx])


def build_arm_dir(out_root, arm, stage2_prob_val, stage3_orchards_prob_val):
    d = f"{out_root}/{arm}"
    os.makedirs(d, exist_ok=True)
    shared = ["val_cal_idx.npy", "val_tune_idx.npy", "stage2_val_idx.npy",
              "stage1_pred.npy", "stage1_train_idx.npy", "stage1_route_oof_train.npy",
              "stage3_plantation_prob_val.npy", "stage3_field_prob_val.npy", "valid_cols.npy"]
    for f in shared:
        dst = f"{d}/{f}"
        if not os.path.exists(dst):
            os.link(f"{M5}/{f}", dst)
    np.save(f"{d}/stage2_prob_val.npy", stage2_prob_val)
    np.save(f"{d}/stage3_orchards_prob_val.npy", stage3_orchards_prob_val)
    m5 = json.load(open(f"{M5}/manifest.json"))
    json.dump({"npz": m5["npz"], "arm": arm}, open(f"{d}/manifest.json", "w"), indent=2)
    return d


def run_sweep(run_dir):
    env = dict(os.environ, RUN_DIR=run_dir)
    r = subprocess.run([sys.executable, "-u", "sweep_operating_point.py"], env=env)
    if r.returncode:
        raise SystemExit(f"sweep_operating_point.py failed on {run_dir}")
    return json.load(open(f"{run_dir}/opsweep_selection.json"))


if __name__ == "__main__":
    log("=== E4 honest retune: Stage 2 + orchards expert ===")
    meta5 = json.load(open(f"{M5}/manifest.json"))
    npz = meta5["npz"]
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(np.int32)
    X = d["X"].astype(np.float32)
    del d
    parcels = np.load(PARCEL_ID)
    valid_cols = np.load(f"{M5}/valid_cols.npy")
    if valid_cols.size != X.shape[1]:
        X = X[:, valid_cols]
    log(f"X {X.shape}")

    tr = np.load(f"{M5}/stage1_train_idx.npy")
    route_oof = np.load(f"{M5}/stage1_route_oof_train.npy")
    assert route_oof.size == tr.size
    cand_tr = tr[route_oof == 1]
    log(f"train fold {tr.size:,}; OOF econ candidates {cand_tr.size:,}")

    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK

    va_cal = np.load(f"{M5}/val_cal_idx.npy")
    c_va = np.load(f"{M5}/stage2_val_idx.npy")

    gen = np.random.default_rng(SEARCH_SEED)

    # ---------------- search populations -------------------------------------
    s2_idx = cap(cand_tr, g_of[cand_tr], STAGE2_SEARCH_PER_GROUP, gen)
    log(f"stage-2 search population {s2_idx.size:,} "
        f"{dict(zip(*np.unique(g_of[s2_idx], return_counts=True)))}")

    orch_tr = np.intersect1d(tr, np.flatnonzero(np.isin(y, ORCHARD_CODES)))
    orch_idx = cap(orch_tr, y[orch_tr], PER_LU_CAP, gen)
    log(f"orchards natural capped fit set {orch_idx.size:,} "
        f"{dict(zip(*np.unique(y[orch_idx], return_counts=True)))}")

    n_features = X.shape[1]
    gamma_grid = [round(m / n_features, 8) for m in GAMMA_MULT]
    log(f"n_features={n_features}  gamma grid (current rule x{GAMMA_MULT}) = {gamma_grid}")

    # ---------------- honest GroupKFold(3) searches --------------------------
    if os.environ.get("RESUME_FROM_WINNERS", "0") == "1" and os.path.exists(f"{OUT}/winners.json"):
        prior = json.load(open(f"{OUT}/winners.json"))
        winner_stage2, winner_orchards = prior["stage2"], prior["orchards"]
        log(f"RESUME_FROM_WINNERS: reusing saved winners.json (search NOT rerun)")
        log(f"  stage2 winner: {winner_stage2}")
        log(f"  orchards winner: {winner_orchards}")
    else:
        winner_stage2 = run_search(X[s2_idx], g_of[s2_idx], parcels[s2_idx],
                                   COMP_STAGE2, gamma_grid, "stage2")
        winner_orchards = run_search(X[orch_idx], y[orch_idx], parcels[orch_idx],
                                     COMP_ORCH, gamma_grid, "orchards")
        json.dump({"stage2": winner_stage2, "orchards": winner_orchards},
                  open(f"{OUT}/winners.json", "w"), indent=2)

    # ---------------- Gate G4: refit winners at full size ---------------------
    log("\n=== Gate G4: full-size refit and paired comparison ===")

    # Stage 2: reuse s2mass's EXACT pool/cal/weights -- only hyperparams change
    fit2 = np.load(f"{S2MASS}/stage2_fit_idx.npy")
    cal2 = np.load(f"{S2MASS}/stage2_cal_idx.npy")
    w2 = np.load(f"{S2MASS}/stage2_sample_weight.npy")
    p2_new = dict(n_components=winner_stage2["n_components"], gamma=winner_stage2["gamma"],
                  C=winner_stage2["C"])
    m2_new = fit_final(X, g_of, fit2, cal2, p2_new, "stage2-retuned", weighted=True, sw=w2)
    joblib.dump(m2_new, f"{OUT}/stage2_retuned_model.joblib")
    log("  stage2-retuned: predicting on validation candidates")
    stage2_prob_val_new = chunked_proba(m2_new, X, c_va)
    np.save(f"{OUT}/stage2_retuned_prob_val.npy", stage2_prob_val_new)

    # Orchards: refit on the SAME population the search used, new hyperparams,
    # unweighted -- matching M5's style (E5 tests Stage-3 weighting, not E4)
    orch_va_cal = np.intersect1d(va_cal, np.flatnonzero(np.isin(y, ORCHARD_CODES)))
    p3_new = dict(n_components=winner_orchards["n_components"], gamma=winner_orchards["gamma"],
                  C=winner_orchards["C"])
    m3_new = fit_final(X, y, orch_idx, orch_va_cal, p3_new, "orchards-retuned", weighted=False)
    joblib.dump(m3_new, f"{OUT}/stage3_orchards_retuned_model.joblib")
    log("  orchards-retuned: predicting on validation candidates")
    stage3_orch_prob_val_new = chunked_proba(m3_new, X, c_va)
    np.save(f"{OUT}/stage3_orchards_retuned_prob_val.npy", stage3_orch_prob_val_new)

    # control: current best-known config (G3 passed -> s2mass treatment Stage 2
    # + M5's original, untuned orchards expert)
    control_stage2 = np.load(f"{S2MASS}/stage2_treatment_prob_val.npy")
    control_orchards = np.load(f"{M5}/stage3_orchards_prob_val.npy")

    control_dir = build_arm_dir(OUT, "control", control_stage2, control_orchards)
    treatment_dir = build_arm_dir(OUT, "treatment", stage2_prob_val_new, stage3_orch_prob_val_new)

    sel_control = run_sweep(control_dir)
    sel_treatment = run_sweep(treatment_dir)

    delta = sel_treatment["tune_macro_f1"] - sel_control["tune_macro_f1"]
    alive_guard = sel_treatment["tune_alive"] >= sel_control["tune_alive"]
    gate_g4 = (delta >= 0.002) and alive_guard

    log("\n================ GATE G4 ================")
    log(f"  control    tune {sel_control['tune_macro_f1']:.4f}  ({sel_control['tune_alive']} alive)"
        f"  cell ({sel_control['alpha2']}, {sel_control['alpha3']})")
    log(f"  treatment  tune {sel_treatment['tune_macro_f1']:.4f}  ({sel_treatment['tune_alive']} alive)"
        f"  cell ({sel_treatment['alpha2']}, {sel_treatment['alpha3']})")
    log(f"  delta (treatment - control): {delta:+.4f}")
    log(f"  alive-crops guard (treatment >= control): {alive_guard}")
    log(f"  GATE G4 -> {'PASS: winners enter E7' if gate_g4 else 'FAIL: E4 winners stay out'}")

    with open(f"{OUT}/gate_g4.json", "w") as f:
        json.dump({"winner_stage2": winner_stage2, "winner_orchards": winner_orchards,
                   "control": sel_control, "treatment": sel_treatment,
                   "delta": round(delta, 4), "alive_guard": alive_guard,
                   "gate_g4_pass": gate_g4}, f, indent=2)
    log("DONE")
