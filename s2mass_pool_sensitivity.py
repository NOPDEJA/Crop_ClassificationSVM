"""s2mass_pool_sensitivity.py

E3 of docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md: is the s2mass +0.0081 an
artifact of one particular 200k-per-group draw?

s2mass_stage2.py replayed M5's rng sequence EXACTLY, so its "control" and
"treatment" fit on the identical pool M5 used -- the +0.0081 has never been
measured against a different draw of who ends up in that pool. This script
reuses all of s2mass_stage2.py's replay machinery (verbatim, same checks) to
rebuild c_tr / c_va / c_va_cal / g_of / route exactly as M5 had them, but then
substitutes an INDEPENDENT rng, seeded per draw (1001, 1002, 1003 -- for the
pool draw only), for the one line that used to consume the shared replay rng:
`fit2 = cap(c_tr, g_of[c_tr], PER_GROUP_CAP)`.

The calibration rows are NOT redrawn: cal2 is loaded verbatim from
runs/s2_2018_3date_parcel_s2mass/stage2_cal_idx.npy so every draw calibrates
on the same rows the original +0.0081 result did, isolating the pool draw as
the only thing that varies between this script's three arms and the original.

Stage 1 and Stage 3 stay frozen from M5 (hard-linked, per s2mass_score.py's
pattern) for every draw's scoring.

Read: per-draw paired delta (treatment - control), at the cal-selected cell
(sweep_operating_point.py's own selection) and at the fixed cell (0.3, 0.6).

Gate G3 (predeclared in runs/s2mass_pool_sensitivity/PREDECLARATION.md): the
subtype weights enter E7 iff treatment >= control in ALL three draws and the
mean delta >= +0.002.

Env:
  M5=<dir>        frozen run to inherit Stage 1/3 from (default s2_2018_3date_parcel_m5)
  S2MASS=<dir>    original s2mass run, source of the frozen cal2 (default
                  s2_2018_3date_parcel_s2mass)
  OUT=<dir>       output root (default ./runs/s2mass_pool_sensitivity)
  SEEDS=1001,1002,1003
"""
import csv
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

from config import (RANDOM_STATE, SAMPLES_PER_LU, CAP_ECON, CAP_WATER,
                    CAP_FOREST, CAP_OTHERS, PER_GROUP_CAP)

M5 = os.environ.get("M5", "./runs/s2_2018_3date_parcel_m5")
S2MASS = os.environ.get("S2MASS", "./runs/s2_2018_3date_parcel_s2mass")
OUT = os.environ.get("OUT", "./runs/s2mass_pool_sensitivity")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1001,1002,1003").split(",")]
FIXED_CELL = (0.3, 0.6)
SPLIT_ASSIGN = "./splits/split_assign.npy"
PARCEL_ID = "./splits/parcel_id_row.npy"

P23 = dict(n_components=600, gamma=None, C=10.0)
CHUNK = 400_000
CALIB_MAX = 300_000
CROSSFIT_S1 = 3

ECON = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
WATER = {4101, 4102, 4103, 4201, 4202, 4203}
FOREST = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
GROUPS = {1: {2403, 2404, 2407, 2413, 2416, 2419, 2420},
          2: {2302, 2303, 2405},
          3: {2101, 2204, 2205}}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
SINK = 4
CAPS1 = {1: CAP_ECON, 2: CAP_WATER, 4: CAP_FOREST, 3: CAP_OTHERS}
SHARED = ["val_cal_idx.npy", "val_tune_idx.npy", "stage2_val_idx.npy",
          "stage1_pred.npy", "stage1_train_idx.npy", "stage1_route_oof_train.npy",
          "stage3_orchards_prob_val.npy", "stage3_plantation_prob_val.npy",
          "stage3_field_prob_val.npy", "valid_cols.npy"]

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)   # the SHARED replay rng, M5-identical


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------
# verbatim from s2mass_stage2.py -- consumes `rng`, so any edit changes which
# rows M5 fitted on and breaks the replay
# --------------------------------------------------------------------------
def cap(idx, labels, per_class, limit_total=None, gen=None):
    gen = gen or rng
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else gen.choice(w, per_class, replace=False))
    out = np.concatenate(parts)
    if limit_total and out.size > limit_total:
        out = gen.choice(out, limit_total, replace=False)
    return np.sort(out)


def stage1_cap(idx, y, sup):
    lu = cap(idx, y[idx], SAMPLES_PER_LU)
    parts = []
    for sc, lim in CAPS1.items():
        w = lu[sup[lu] == sc]
        parts.append(w if w.size <= lim else rng.choice(w, lim, replace=False))
    return np.sort(np.concatenate(parts))


def halve_by_parcel(idx, y, parcels):
    pid = parcels[idx]
    uniq, inv = np.unique(pid, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size))
    plab = y[idx][order][starts]
    take = np.zeros(uniq.size, dtype=bool)
    for c in np.unique(plab):
        w = np.flatnonzero(plab == c)
        w = w[rng.permutation(w.size)]
        take[w[: w.size // 2]] = True
    m = take[inv]
    return idx[m], idx[~m]


def draw_calib(cal_idx):
    if cal_idx.size > CALIB_MAX:
        return np.sort(rng.choice(cal_idx, CALIB_MAX, replace=False))
    return cal_idx


# --------------------------------------------------------------------------
def base_pipe(p, weighted):
    scaler = StandardScaler()
    svc = LinearSVC(C=p["C"], class_weight=None, max_iter=5000,
                    random_state=RANDOM_STATE)
    if weighted:
        scaler = scaler.set_fit_request(sample_weight=False)
        svc = svc.set_fit_request(sample_weight=True)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
        ("nyst", Nystroem(kernel="rbf", n_components=p["n_components"],
                          gamma=p["gamma"], random_state=RANDOM_STATE)),
        ("svc", svc),
    ])


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
                log(f"    calibration fallback for class {c}")
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


def subtype_weights(fit_idx, g_of, y):
    w = np.ones(fit_idx.size, dtype=np.float64)
    table = []
    for g in sorted(np.unique(g_of[fit_idx])):
        sel = np.flatnonzero(g_of[fit_idx] == g)
        n_g = sel.size
        if g == SINK:
            table.append({"group": GNAME[g], "crop": "(non-crop)", "lu_code": 0,
                          "rows": n_g, "weight": 1.0})
            continue
        codes = y[fit_idx[sel]]
        uniq, cnt = np.unique(codes, return_counts=True)
        raw = np.sqrt(cnt.max() / cnt)
        wg = raw[np.searchsorted(uniq, codes)]
        wg *= n_g / wg.sum()
        w[sel] = wg
        for c, n in zip(uniq, cnt):
            table.append({"group": GNAME[g], "crop": NM[int(c)], "lu_code": int(c),
                          "rows": int(n), "weight": round(float(wg[codes == c][0]), 4)})
        assert abs(wg.sum() - n_g) < 1e-6
    return w, table


def build_arm_dir(draw_dir, arm):
    """Hard-link M5's frozen Stage-1/3 arrays + this draw's stage2_prob_val.npy."""
    d = f"{draw_dir}/{arm}"
    os.makedirs(d, exist_ok=True)
    for f in SHARED:
        dst = f"{d}/{f}"
        if not os.path.exists(dst):
            os.link(f"{M5}/{f}", dst)
    dst = f"{d}/stage2_prob_val.npy"
    if not os.path.exists(dst):
        os.link(f"{draw_dir}/stage2_{arm}_prob_val.npy", dst)
    m5 = json.load(open(f"{M5}/manifest.json"))
    json.dump({"npz": m5["npz"], "arm": arm, "stage1_from": M5, "stage3_from": M5,
               "params_stage23": m5["params_stage23"], "params_stage3": m5["params_stage3"]},
              open(f"{d}/manifest.json", "w"), indent=2)
    return d


def run_sweep(run_dir):
    env = dict(os.environ, RUN_DIR=run_dir, ALSO=f"{FIXED_CELL[0]},{FIXED_CELL[1]}")
    r = subprocess.run([sys.executable, "-u", "sweep_operating_point.py"], env=env)
    if r.returncode:
        raise SystemExit(f"sweep_operating_point.py failed on {run_dir}")
    return json.load(open(f"{run_dir}/opsweep_selection.json"))


if __name__ == "__main__":
    log("=== s2mass pool-draw sensitivity ===  M5 =", M5, " S2MASS =", S2MASS, " OUT =", OUT)
    meta5 = json.load(open(f"{M5}/manifest.json"))
    npz = meta5["npz"]
    d = np.load(npz, allow_pickle=True)
    y = d["y"].astype(np.int32)
    asg = np.load(SPLIT_ASSIGN)
    parcels = np.load(PARCEL_ID)
    tr = np.flatnonzero(asg == 0)
    va = np.flatnonzero(asg == 1)
    sup = np.where(np.isin(y, list(ECON)), 1,
          np.where(np.isin(y, list(WATER)), 2,
          np.where(np.isin(y, list(FOREST)), 4, 3))).astype(np.int32)

    # ---------------- replay M5's rng sequence up to (not including) the
    # stage-2 pool draw -- identical to s2mass_stage2.py's checks 1-10 -------
    log("replaying M5's sampling sequence up to the stage-2 pool draw")
    checks = []

    va_cal, va_tune = halve_by_parcel(va, y, parcels)
    ok = (np.array_equal(va_cal, np.load(f"{M5}/val_cal_idx.npy")) and
          np.array_equal(va_tune, np.load(f"{M5}/val_tune_idx.npy")))
    checks.append(("fold-1 halving byte-identical to M5", ok))

    fit1 = stage1_cap(tr, y, sup)
    dist = {int(k): int(v) for k, v in zip(*np.unique(sup[fit1], return_counts=True))}
    want = {int(k): int(v) for k, v in meta5["stage1_train"].items()}
    checks.append((f"stage-1 fit distribution {dist}", dist == want))

    draw_calib(va_cal)

    uniq = np.unique(parcels[tr])
    part_of_parcel = rng.permutation(uniq.size) % CROSSFIT_S1
    part = part_of_parcel[np.searchsorted(uniq, parcels[tr])]
    sizes = []
    for kk in range(CROSSFIT_S1):
        hold = part == kk
        fi = stage1_cap(tr[~hold], y, sup)
        sizes.append((int(fi.size), int(hold.sum())))
        draw_calib(va_cal)
    want_sizes = [(2849354, 3468903), (2746302, 6609230), (2867194, 4764161)]
    checks.append((f"crossfit fit/route sizes {sizes}", sizes == want_sizes))

    route = np.load(f"{M5}/stage1_pred.npy").copy()
    assert np.array_equal(np.load(f"{M5}/stage1_train_idx.npy"), tr)
    route[tr] = np.load(f"{M5}/stage1_route_oof_train.npy")
    cand = np.flatnonzero(route == 1)
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, list(codes))] = g
    g_of[g_of == 0] = SINK
    c_tr = np.intersect1d(cand, tr)
    c_va = np.intersect1d(cand, va)
    c_va_cal = np.intersect1d(cand, va_cal)
    n_sink = int((g_of[c_tr] == SINK).sum())
    checks.append((f"stage-2 sink training rows {n_sink:,}",
                   n_sink == meta5["stage2_sink_train"]))
    checks.append(("validation candidates byte-identical to M5",
                   np.array_equal(c_va, np.load(f"{M5}/stage2_val_idx.npy"))))

    for name, ok in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        raise SystemExit("replay diverged from M5 -- draws would not be comparable")

    # cal2 is loaded from the ORIGINAL s2mass run, not redrawn -- "the same
    # calibration rows" for every pool draw, per the plan.
    cal2 = np.load(f"{S2MASS}/stage2_cal_idx.npy")
    log(f"  reusing s2mass's calibration rows: {cal2.size:,}")

    X = d["X"].astype(np.float32)
    del d
    vc = np.load(f"{M5}/valid_cols.npy")
    if vc.size != X.shape[1]:
        X = X[:, vc]
    log(f"X {X.shape}")

    results = []
    for seed in SEEDS:
        draw_dir = f"{OUT}/draw_{seed}"
        os.makedirs(draw_dir, exist_ok=True)
        log(f"\n===== draw seed {seed} =====")
        gen = np.random.default_rng(seed)
        fit2 = cap(c_tr, g_of[c_tr], PER_GROUP_CAP, gen=gen)
        assert fit2.size == 800_000, f"unexpected pool size {fit2.size:,}"
        np.save(f"{draw_dir}/stage2_fit_idx.npy", fit2)
        np.save(f"{draw_dir}/stage2_cal_idx.npy", cal2)
        log(f"  pool groups "
            f"{ {GNAME[int(k)]: int(v) for k, v in zip(*np.unique(g_of[fit2], return_counts=True))} }")

        w, table = subtype_weights(fit2, g_of, y)
        with open(f"{draw_dir}/mass_table.csv", "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)
        np.save(f"{draw_dir}/stage2_sample_weight.npy", w)

        manifest = {"seed": seed, "m5": M5, "s2mass": S2MASS, "npz": npz,
                    "params_stage23": P23, "stage2_fit_rows": int(fit2.size),
                    "stage2_cal_rows": int(cal2.size),
                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}

        for arm in ("control", "treatment"):
            t0 = time.time()
            log(f"--- seed {seed} / {arm}: base fit on {fit2.size:,} rows")
            if arm == "treatment":
                with sklearn.config_context(enable_metadata_routing=True):
                    base = OneVsRestClassifier(base_pipe(P23, True))
                    base.fit(X[fit2], g_of[fit2], sample_weight=w)
            else:
                base = OneVsRestClassifier(base_pipe(P23, False))
                base.fit(X[fit2], g_of[fit2])
            log(f"  {arm}: sigmoid calibration on {cal2.size:,} rows")
            m2 = PlattCalibrated(base).fit(X[cal2], g_of[cal2])
            joblib.dump(m2, f"{draw_dir}/stage2_{arm}_model.joblib")
            log(f"  {arm}: predicting on {c_va.size:,} validation candidates")
            np.save(f"{draw_dir}/stage2_{arm}_prob_val.npy", chunked_proba(m2, X, c_va))
            manifest[f"{arm}_minutes"] = round((time.time() - t0) / 60, 1)
            log(f"  {arm}: done in {manifest[f'{arm}_minutes']} min")
            del base, m2

        manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(f"{draw_dir}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # ---------------- score both arms ------------------------------------
        arm_dirs = {arm: build_arm_dir(draw_dir, arm) for arm in ("control", "treatment")}
        sel = {arm: run_sweep(d) for arm, d in arm_dirs.items()}
        delta_cal_sel = sel["treatment"]["tune_macro_f1"] - sel["control"]["tune_macro_f1"]
        delta_fixed = sel["treatment"]["fixed_cell_tune_macro_f1"] - sel["control"]["fixed_cell_tune_macro_f1"]
        row = {"seed": seed,
               "control_cal_selected_tune_f1": sel["control"]["tune_macro_f1"],
               "treatment_cal_selected_tune_f1": sel["treatment"]["tune_macro_f1"],
               "delta_cal_selected": round(delta_cal_sel, 4),
               "control_fixed_cell_tune_f1": sel["control"]["fixed_cell_tune_macro_f1"],
               "treatment_fixed_cell_tune_f1": sel["treatment"]["fixed_cell_tune_macro_f1"],
               "delta_fixed_cell": round(delta_fixed, 4)}
        results.append(row)
        log(f"  seed {seed}: delta(cal-selected) {delta_cal_sel:+.4f}"
            f"  delta(fixed 0.3,0.6) {delta_fixed:+.4f}")

    with open(f"{OUT}/pool_sensitivity_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    deltas = [r["delta_cal_selected"] for r in results]
    mean_delta = round(sum(deltas) / len(deltas), 4)
    all_favour = all(r["treatment_cal_selected_tune_f1"] >= r["control_cal_selected_tune_f1"]
                      for r in results)
    gate_g3 = all_favour and mean_delta >= 0.002
    log("\n================ GATE G3 ================")
    log(f"  per-draw delta (cal-selected): {deltas}  range [{min(deltas):+.4f}, {max(deltas):+.4f}]")
    log(f"  mean delta: {mean_delta:+.4f}")
    log(f"  treatment >= control in all draws: {all_favour}")
    log(f"  GATE G3 -> {'PASS: weights enter E7' if gate_g3 else 'FAIL: weights stay out'}")
    with open(f"{OUT}/gate_g3.json", "w") as f:
        json.dump({"deltas_cal_selected": deltas, "mean_delta": mean_delta,
                   "range": [min(deltas), max(deltas)], "all_favour_treatment": all_favour,
                   "gate_g3_pass": gate_g3}, f, indent=2)
    log("DONE")
