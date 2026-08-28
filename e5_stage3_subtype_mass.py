"""e5_stage3_subtype_mass.py

E5 of docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md: the s2mass-style paired
subtype-mass design, applied INSIDE the plantation and orchards Stage-3
experts instead of Stage 2's group labels. Motivation: coconut's Stage-2
routing doubled (13.7% -> 26.6%) under the s2mass treatment while its F1
stayed at 0.001 -- the binding constraint moved into the plantation expert
(96.8% rubber / 0.08% coconut in its raw fit population). This is the direct
test of whether reweighting the CROP MIX inside each expert helps.

Stage 1 and Stage 2 are FROZEN AT THE E4 WINNER (Gate G4 passed 2026-08-28:
n_components=1200, gamma=0.5x rule, C=30 for both Stage 2 and orchards) --
reused directly from runs/retune_stage2_orchards/treatment/, per the plan's
"use the E4 winner if G4 passed". Field is untouched by E5 (frozen from M5).

Orchards population is reproduced BIT-IDENTICAL to E4's by replaying the same
SEARCH_SEED=777 rng draw sequence (stage-2 search draw, then orchards draw,
then this script's new plantation draw) -- so E5's orchards CONTROL is
literally E4's already-fit orchards-retuned model (reused, not refit): same
rows, same hyperparameters, unweighted. Only orchards TREATMENT (weighted,
same rows/hyperparams) and both plantation arms (new population, M5's
original Stage-3 hyperparams since E4 never retuned plantation) are new fits.

Per-row weight (inside each expert): w_c = sqrt(m_max/m_c) over POST-CAP
within-expert crop counts, renormalised so the expert's total fit mass is
unchanged -- identical formula to s2mass_stage2.py's subtype_weights(),
applied to Stage-3's crop labels instead of Stage-2's group labels.

Gate G5: same numeric form as G3 (treatment tune macro F1 >= control + 0.002,
alive-crops guard). E5's own budget (~1-2h) is a single paired comparison,
not G3's 3-draw sensitivity check -- G3's replication answered a different
question (pool-draw sensitivity of an already-passed result); this is a new
experiment being gated for the first time.

Env:
  M5=<dir>      frozen run (default ./runs/s2_2018_3date_parcel_m5)
  E4=<dir>      gated E4 run (default ./runs/retune_stage2_orchards)
  OUT=<dir>     output (default ./runs/s3mass_experts)
"""
import json
import os
import subprocess
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.calibration import _SigmoidCalibration
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import RANDOM_STATE, PER_LU_CAP

M5 = os.environ.get("M5", "./runs/s2_2018_3date_parcel_m5")
E4 = os.environ.get("E4", "./runs/retune_stage2_orchards")
OUT = os.environ.get("OUT", "./runs/s3mass_experts")
PARCEL_ID = "./splits/parcel_id_row.npy"

SEARCH_SEED = 777           # MUST match e4_retune_stage2_orchards.py's seed and draw order
STAGE2_SEARCH_PER_GROUP = 50_000

GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},
          2: {2302, 2303, 2405},
          3: {2101, 2204, 2205}}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
SINK = 4
CHUNK = 400_000
CALIB_MAX = 300_000

os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def cap(idx, labels, per_class, gen):
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else gen.choice(w, per_class, replace=False))
    return np.sort(np.concatenate(parts))


def subtype_weights(fit_idx, y):
    """w_c = sqrt(m_max/m_c) over this expert's own post-cap crop counts,
    renormalised so total fit mass is unchanged. Identical formula to
    s2mass_stage2.py's subtype_weights(), applied with no group/sink split
    since every row here already belongs to one expert."""
    codes = y[fit_idx]
    uniq, cnt = np.unique(codes, return_counts=True)
    raw = np.sqrt(cnt.max() / cnt)
    w = raw[np.searchsorted(uniq, codes)]
    w *= fit_idx.size / w.sum()
    table = [{"crop": NM[int(c)], "lu_code": int(c), "rows": int(n),
             "weight": round(float(w[codes == c][0]), 4)} for c, n in zip(uniq, cnt)]
    assert abs(w.sum() - fit_idx.size) < 1e-6
    return w, table


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


def fit_final(X, y, fit_idx, cal_idx, p, tag, weighted, sw=None):
    log(f"  {tag}: final fit on {fit_idx.size:,} rows  params={p}  weighted={weighted}")
    if weighted:
        # set_fit_request() needs metadata routing enabled at the point it's
        # called -- the WHOLE pipeline must be built inside the config
        # context, not just fitted inside it (E4's bug, fixed here from the start).
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
            base.fit(X[fit_idx], y[fit_idx], sample_weight=sw)
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
        base.fit(X[fit_idx], y[fit_idx])
    log(f"  {tag}: sigmoid calibration on {cal_idx.size:,} rows")
    return PlattCalibrated(base).fit(X[cal_idx], y[cal_idx])


def build_arm_dir(arm, stage3_orchards_pv, stage3_plantation_pv):
    d = f"{OUT}/{arm}"
    os.makedirs(d, exist_ok=True)
    shared = ["val_cal_idx.npy", "val_tune_idx.npy", "stage2_val_idx.npy",
              "stage1_pred.npy", "stage1_train_idx.npy", "stage1_route_oof_train.npy",
              "stage3_field_prob_val.npy", "valid_cols.npy"]
    for f in shared:
        dst = f"{d}/{f}"
        if not os.path.exists(dst):
            os.link(f"{M5}/{f}", dst)
    # Stage 2 frozen at the E4 winner (Gate G4 passed) -- E4's treatment arm
    dst = f"{d}/stage2_prob_val.npy"
    if not os.path.exists(dst):
        os.link(f"{E4}/treatment/stage2_prob_val.npy", dst)
    np.save(f"{d}/stage3_orchards_prob_val.npy", stage3_orchards_pv)
    np.save(f"{d}/stage3_plantation_prob_val.npy", stage3_plantation_pv)
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
    log("=== E5: Stage-3 subtype mass, factorial against the E4 winner ===")
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
    cand_tr = tr[route_oof == 1]
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK

    va_cal = np.load(f"{M5}/val_cal_idx.npy")
    c_va = np.load(f"{M5}/stage2_val_idx.npy")

    # ---- replay E4's exact draw sequence to reproduce orch_idx bit-identical,
    # then draw a NEW plantation population continuing the same generator ----
    gen = np.random.default_rng(SEARCH_SEED)
    s2_idx = cap(cand_tr, g_of[cand_tr], STAGE2_SEARCH_PER_GROUP, gen)     # E4's draw 1 (unused here)
    orch_tr = np.intersect1d(tr, np.flatnonzero(np.isin(y, list(GROUPS[1]))))
    orch_idx = cap(orch_tr, y[orch_tr], PER_LU_CAP, gen)                   # E4's draw 2 -- must match
    plant_tr = np.intersect1d(tr, np.flatnonzero(np.isin(y, list(GROUPS[2]))))
    plant_idx = cap(plant_tr, y[plant_tr], PER_LU_CAP, gen)                # NEW draw 3

    log(f"orchards fit set {orch_idx.size:,} "
        f"{dict(zip(*np.unique(y[orch_idx], return_counts=True)))}")
    log(f"plantation fit set {plant_idx.size:,} "
        f"{dict(zip(*np.unique(y[plant_idx], return_counts=True)))}")

    orch_va_cal = np.intersect1d(va_cal, np.flatnonzero(np.isin(y, list(GROUPS[1]))))
    plant_va_cal = np.intersect1d(va_cal, np.flatnonzero(np.isin(y, list(GROUPS[2]))))

    e4_winners = json.load(open(f"{E4}/winners.json"))
    p3_orchards = dict(n_components=e4_winners["orchards"]["n_components"],
                       gamma=e4_winners["orchards"]["gamma"], C=e4_winners["orchards"]["C"])
    p3_plantation = dict(meta5["params_stage3"])   # M5's original Stage-3 config; untouched by E4
    log(f"orchards hyperparams (E4 winner): {p3_orchards}")
    log(f"plantation hyperparams (M5 original, untouched by E4): {p3_plantation}")

    w_orch, table_orch = subtype_weights(orch_idx, y)
    w_plant, table_plant = subtype_weights(plant_idx, y)
    for name, table in (("orchards", table_orch), ("plantation", table_plant)):
        log(f"  {name} subtype weights (post-cap counts):")
        for r in table:
            log(f"    {r['crop']:<12}{r['rows']:>8,}  w={r['weight']:.3f}")
    np.save(f"{OUT}/orchards_sample_weight.npy", w_orch)
    np.save(f"{OUT}/plantation_sample_weight.npy", w_plant)

    # ---- orchards CONTROL: reuse E4's already-fit, byte-identical model ----
    log("orchards control: reusing E4's orchards-retuned model (same rows, same hyperparams, unweighted)")
    orch_control_pv = np.load(f"{E4}/stage3_orchards_retuned_prob_val.npy")

    # ---- orchards TREATMENT: new weighted fit, same rows/hyperparams -------
    m_orch_t = fit_final(X, y, orch_idx, orch_va_cal, p3_orchards, "orchards-treatment",
                         weighted=True, sw=w_orch)
    joblib.dump(m_orch_t, f"{OUT}/orchards_treatment_model.joblib")
    orch_treatment_pv = chunked_proba(m_orch_t, X, c_va)
    np.save(f"{OUT}/orchards_treatment_prob_val.npy", orch_treatment_pv)
    del m_orch_t

    # ---- plantation CONTROL and TREATMENT: both new ------------------------
    m_plant_c = fit_final(X, y, plant_idx, plant_va_cal, p3_plantation, "plantation-control",
                          weighted=False)
    joblib.dump(m_plant_c, f"{OUT}/plantation_control_model.joblib")
    plant_control_pv = chunked_proba(m_plant_c, X, c_va)
    np.save(f"{OUT}/plantation_control_prob_val.npy", plant_control_pv)
    del m_plant_c

    m_plant_t = fit_final(X, y, plant_idx, plant_va_cal, p3_plantation, "plantation-treatment",
                          weighted=True, sw=w_plant)
    joblib.dump(m_plant_t, f"{OUT}/plantation_treatment_model.joblib")
    plant_treatment_pv = chunked_proba(m_plant_t, X, c_va)
    np.save(f"{OUT}/plantation_treatment_prob_val.npy", plant_treatment_pv)
    del m_plant_t

    # ---- score both arms -----------------------------------------------------
    control_dir = build_arm_dir("control", orch_control_pv, plant_control_pv)
    treatment_dir = build_arm_dir("treatment", orch_treatment_pv, plant_treatment_pv)
    sel_control = run_sweep(control_dir)
    sel_treatment = run_sweep(treatment_dir)

    delta = sel_treatment["tune_macro_f1"] - sel_control["tune_macro_f1"]
    alive_guard = sel_treatment["tune_alive"] >= sel_control["tune_alive"]
    gate_g5 = (delta >= 0.002) and alive_guard

    control_crop = {r["crop"]: r for r in
                    __import__("csv").DictReader(open(f"{control_dir}/opsweep_selected_tune.csv",
                                                       encoding="utf-8-sig"))}
    treatment_crop = {r["crop"]: r for r in
                      __import__("csv").DictReader(open(f"{treatment_dir}/opsweep_selected_tune.csv",
                                                         encoding="utf-8-sig"))}
    coco_c = control_crop.get("Coconut", {})
    coco_t = treatment_crop.get("Coconut", {})

    log("\n================ GATE G5 ================")
    log(f"  control    tune {sel_control['tune_macro_f1']:.4f}  ({sel_control['tune_alive']} alive)"
        f"  cell ({sel_control['alpha2']}, {sel_control['alpha3']})")
    log(f"  treatment  tune {sel_treatment['tune_macro_f1']:.4f}  ({sel_treatment['tune_alive']} alive)"
        f"  cell ({sel_treatment['alpha2']}, {sel_treatment['alpha3']})")
    log(f"  delta (treatment - control): {delta:+.4f}")
    log(f"  alive-crops guard (treatment >= control): {alive_guard}")
    log(f"  GATE G5 -> {'PASS: E5 experts enter E7' if gate_g5 else 'FAIL: E5 experts stay out'}")
    log(f"\n  coconut watch -- F1 control {coco_c.get('f1', 'n/a')} -> "
        f"treatment {coco_t.get('f1', 'n/a')}  (support {coco_c.get('support', 'n/a')})")

    with open(f"{OUT}/gate_g5.json", "w") as f:
        json.dump({"orchards_hyperparams": p3_orchards, "plantation_hyperparams": p3_plantation,
                   "control": sel_control, "treatment": sel_treatment,
                   "delta": round(delta, 4), "alive_guard": alive_guard,
                   "gate_g5_pass": gate_g5,
                   "coconut_f1_control": coco_c.get("f1"), "coconut_f1_treatment": coco_t.get("f1")},
                  f, indent=2)
    log("DONE")
