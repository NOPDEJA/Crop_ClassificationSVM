"""s2mass_stage2.py

Section 3 of docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md: two paired Stage-2 fits
that differ only in how the loss mass is distributed across crop SUBTYPES inside
each of the four group labels.

Why this is not a cascade run. Stage 2's caps balance the four group labels at
200,000 rows each and then sample uniformly inside each group, so the plantation
fit is 96.82% rubber and 0.08% coconut. Replacing the learned route with the true
group -- every model frozen -- is worth +0.1491 tune-half macro F1
(runs/s2_2018_3date_parcel_m5/oracle_routing.csv). That headroom sits entirely in
Stage 2, so Stage 1 and Stage 3 are held frozen from M5 and only Stage 2 is refitted.

  control    identical config to M5's Stage 2, no weights, fresh fit.
             Its distance from M5's 0.2294 IS the retrain noise floor.
  treatment  same, plus per-row weights w = sqrt(m_maxc / m_c) over the post-cap
             subtype counts inside the row's group, renormalised so each group's
             TOTAL mass stays exactly its row count. Group balance is untouched;
             only the distribution inside each group moves. Sink rows keep 1.

The training pool is not re-drawn, it is REPLAYED. train_parcel_cascade.py draws
every sample from one np.random.default_rng(42), so the Stage-2 cap draw depends
on the whole preceding sequence of draws. This script replays that sequence
exactly -- no model is fitted during the replay, because the only rng consumers
are the halving, the caps and the calibration subsamples, and the one
model-dependent input (the out-of-fold Stage-1 routes) was saved by M5. Six
checkpoints against M5's saved arrays and its log confirm the state stayed
aligned; if any fails the control is not a control and the script stops.

Env:
  REPLAY_ONLY=1   stop after the pool replay and the mass table (D4's pre-launch
                  check). Fits nothing, loads no feature matrix.
  M5=<dir>        frozen run to inherit from (default ./runs/s2_2018_3date_parcel_m5)
  OUT=<dir>       output (default ./runs/s2_2018_3date_parcel_s2mass)
"""
import csv
import json
import os
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
OUT = os.environ.get("OUT", "./runs/s2_2018_3date_parcel_s2mass")
REPLAY_ONLY = os.environ.get("REPLAY_ONLY", "0") == "1"
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

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------
# verbatim from train_parcel_cascade.py -- these consume `rng`, so any edit
# changes which rows M5 fitted on and breaks the replay
# --------------------------------------------------------------------------
def cap(idx, labels, per_class, limit_total=None):
    parts = []
    for c in np.unique(labels):
        w = idx[labels == c]
        parts.append(w if w.size <= per_class else rng.choice(w, per_class, replace=False))
    out = np.concatenate(parts)
    if limit_total and out.size > limit_total:
        out = rng.choice(out, limit_total, replace=False)
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
    """The one rng draw inside fit_calibrated, with the fitting removed."""
    if cal_idx.size > CALIB_MAX:
        return np.sort(rng.choice(cal_idx, CALIB_MAX, replace=False))
    return cal_idx


# --------------------------------------------------------------------------
def base_pipe(p, weighted):
    scaler = StandardScaler()
    svc = LinearSVC(C=p["C"], class_weight=None, max_iter=5000,
                    random_state=RANDOM_STATE)
    if weighted:
        # only the SVC consumes the weights; the scaler must decline explicitly
        # or routing refuses to pass anything through the pipeline at all
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
    """Copied from train_parcel_cascade.py so both arms calibrate identically."""

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
    """D3: sqrt-tempered over post-cap SUBTYPE counts, renormalised per group.

    Returns the weight vector and the mass table. Group totals are preserved
    exactly, so this moves mass inside a group and never between groups.
    """
    w = np.ones(fit_idx.size, dtype=np.float64)
    table = []
    for g in sorted(np.unique(g_of[fit_idx])):
        sel = np.flatnonzero(g_of[fit_idx] == g)
        n_g = sel.size
        if g == SINK:                      # one subtype by construction
            table.append({"group": GNAME[g], "crop": "(non-crop)", "lu_code": 0,
                          "rows": n_g, "mass_before": float(n_g),
                          "share_before": 1.0, "mass_after": float(n_g),
                          "share_after": 1.0, "weight": 1.0})
            continue
        codes = y[fit_idx[sel]]
        uniq, cnt = np.unique(codes, return_counts=True)
        raw = np.sqrt(cnt.max() / cnt)
        wg = raw[np.searchsorted(uniq, codes)]
        wg *= n_g / wg.sum()               # group total mass unchanged
        w[sel] = wg
        for c, n in zip(uniq, cnt):
            m_after = float(wg[codes == c].sum())
            table.append({"group": GNAME[g], "crop": NM[int(c)], "lu_code": int(c),
                          "rows": int(n), "mass_before": float(n),
                          "share_before": round(n / n_g, 6),
                          "mass_after": round(m_after, 2),
                          "share_after": round(m_after / n_g, 6),
                          "weight": round(float(wg[codes == c][0]), 4)})
        assert abs(wg.sum() - n_g) < 1e-6
    return w, table


if __name__ == "__main__":
    log("=== stage-2 subtype mass ===  M5 =", M5, " OUT =", OUT)
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

    # ---------------- replay M5's rng sequence up to the Stage-2 draw --------
    log("replaying M5's sampling sequence (no model is fitted here)")
    checks = []

    va_cal, va_tune = halve_by_parcel(va, y, parcels)          # draw 1
    ok = (np.array_equal(va_cal, np.load(f"{M5}/val_cal_idx.npy")) and
          np.array_equal(va_tune, np.load(f"{M5}/val_tune_idx.npy")))
    checks.append(("fold-1 halving byte-identical to M5", ok))

    fit1 = stage1_cap(tr, y, sup)                              # draw 2
    dist = {int(k): int(v) for k, v in zip(*np.unique(sup[fit1], return_counts=True))}
    want = {int(k): int(v) for k, v in meta5["stage1_train"].items()}
    checks.append((f"stage-1 fit distribution {dist}", dist == want))

    draw_calib(va_cal)                                         # draw 3

    uniq = np.unique(parcels[tr])                              # draw 4
    part_of_parcel = rng.permutation(uniq.size) % CROSSFIT_S1
    part = part_of_parcel[np.searchsorted(uniq, parcels[tr])]
    sizes = []
    for kk in range(CROSSFIT_S1):
        hold = part == kk
        fi = stage1_cap(tr[~hold], y, sup)                     # draws 5, 7, 9
        sizes.append((int(fi.size), int(hold.sum())))
        draw_calib(va_cal)                                     # draws 6, 8, 10
    want_sizes = [(2849354, 3468903), (2746302, 6609230), (2867194, 4764161)]
    checks.append((f"crossfit fit/route sizes {sizes}", sizes == want_sizes))

    # the one model-dependent input, taken from M5's saved arrays
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

    fit2 = cap(c_tr, g_of[c_tr], PER_GROUP_CAP)                # draw 11
    cal2 = draw_calib(c_va_cal)                                # draw 12
    checks.append((f"stage-2 pool {fit2.size:,} rows", fit2.size == 800_000))

    for name, ok in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(ok for _, ok in checks):
        raise SystemExit("replay diverged from M5 -- the control would not be a control")
    np.save(f"{OUT}/stage2_fit_idx.npy", fit2)
    np.save(f"{OUT}/stage2_cal_idx.npy", cal2)
    log("  pool groups "
        f"{ {GNAME[int(k)]: int(v) for k, v in zip(*np.unique(g_of[fit2], return_counts=True))} }")
    log(f"  stage-2 calibration rows {cal2.size:,}")

    # ---------------- D4: the mass table, before anything is fitted ----------
    w, table = subtype_weights(fit2, g_of, y)
    with open(f"{OUT}/mass_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)
    log("")
    log("  subtype mass inside each group, before -> after weighting")
    for r in table:
        log(f"    {r['group']:<11}{r['crop']:<12}{r['rows']:>8,}  "
            f"{r['share_before']:>9.4%} -> {r['share_after']:>8.4%}  w={r['weight']:.3f}")
    moved = {r["group"] for r in table
             if abs(r["share_after"] - r["share_before"]) > 1e-6}
    log(f"  groups whose subtype mass moved: {sorted(moved)}")
    if not {"orchards", "plantation", "field"} <= moved:
        raise SystemExit("D4: a named group's masses are unchanged -- vacuous arm, stopping")
    coco = [r for r in table if r["crop"] == "Coconut"][0]
    log(f"  coconut mass share in plantation "
        f"{coco['share_before']:.4%} -> {coco['share_after']:.4%}")
    np.save(f"{OUT}/stage2_sample_weight.npy", w)

    if REPLAY_ONLY:
        log("REPLAY_ONLY: stopping before the fits")
        raise SystemExit(0)

    # ---------------- the two fits ------------------------------------------
    X = d["X"].astype(np.float32)
    del d
    vc = np.load(f"{M5}/valid_cols.npy")
    if vc.size != X.shape[1]:
        X = X[:, vc]
    log(f"X {X.shape}")

    manifest = {"m5": M5, "npz": npz, "params_stage23": P23,
                "stage2_fit_rows": int(fit2.size), "stage2_cal_rows": int(cal2.size),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    for arm in ("control", "treatment"):
        t0 = time.time()
        log(f"--- {arm}: base fit on {fit2.size:,} rows")
        if arm == "treatment":
            # routing is enabled only around this fit, so the control runs the
            # byte-for-byte M5 code path
            with sklearn.config_context(enable_metadata_routing=True):
                base = OneVsRestClassifier(base_pipe(P23, True))
                base.fit(X[fit2], g_of[fit2], sample_weight=w)
        else:
            base = OneVsRestClassifier(base_pipe(P23, False))
            base.fit(X[fit2], g_of[fit2])
        log(f"  {arm}: sigmoid calibration on {cal2.size:,} rows")
        m2 = PlattCalibrated(base).fit(X[cal2], g_of[cal2])
        joblib.dump(m2, f"{OUT}/stage2_{arm}_model.joblib")
        log(f"  {arm}: predicting on {c_va.size:,} validation candidates")
        np.save(f"{OUT}/stage2_{arm}_prob_val.npy", chunked_proba(m2, X, c_va))
        manifest[f"{arm}_minutes"] = round((time.time() - t0) / 60, 1)
        log(f"  {arm}: done in {manifest[f'{arm}_minutes']} min")
        del base, m2

    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log("DONE")
