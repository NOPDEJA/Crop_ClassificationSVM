"""stage2_diagnostics.py

The three §4 figures the report quotes without an artifact behind them, saved as
CSVs so every number in the report is traceable (F4 of the 2026-08-26 plan):

  stage2_pool_composition.csv   what the Stage-2 cap draws FROM, per group per crop
  stage2_routing_accuracy.csv   share of each crop Stage 2 sends to the right group
  oracle_routing.csv            tune-half macro F1 when the learned route is
                                replaced by the true group, every model frozen

Nothing is fitted here. It reads M5's saved validation probabilities and the
saved Stage-1 routes, and it scores through the same strict convention
sweep_operating_point.py uses: the whole tuning half, non-crop truth mapped to 0,
the 13 crop labels fixed.

Env:
  RUN_DIR=<dir>   run to diagnose (default ./runs/s2_2018_3date_parcel_m5)
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import f1_score

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m5")
SINK = 4
GROUPS = {1: ("orchards", [2403, 2404, 2407, 2413, 2416, 2419, 2420]),
          2: ("plantation", [2302, 2303, 2405]),
          3: ("field", [2101, 2204, 2205])}
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
CROPS = sorted(NM)
PER_GROUP_CAP = 200_000


def log(*a):
    print(*a, flush=True)


def reweight(P, ratio, alpha):
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log("wrote", path)


def main():
    npz = json.load(open(f"{RUN}/manifest.json"))["npz"]
    y = np.load(npz, allow_pickle=True)["y"].astype(np.int32)
    asg = np.load("./splits/split_assign.npy")
    sel = json.load(open(f"{RUN}/opsweep_selection.json"))
    a2, a3 = sel["alpha2"], sel["alpha3"]
    log(f"{RUN}: cal-selected cell ({a2}, {a3}), tune macro F1 {sel['tune_macro_f1']}")

    g_of = np.zeros(y.size, dtype=np.int32)
    for g, (_, codes) in GROUPS.items():
        g_of[np.isin(y, codes)] = g
    g_of[g_of == 0] = SINK

    # ---------------- pool composition (the population the cap draws from) ----
    tr = np.load(f"{RUN}/stage1_train_idx.npy")
    route_tr = np.load(f"{RUN}/stage1_route_oof_train.npy")
    c_tr = tr[route_tr == 1]
    log(f"fold-0 stage-2 candidates {c_tr.size:,}")
    rows = []
    for g, (gn, codes) in GROUPS.items():
        own = c_tr[g_of[c_tr] == g]
        for c in codes:
            n = int((y[own] == c).sum())
            rows.append({"group": gn, "crop": NM[c], "lu_code": c,
                         "pool_rows": n,
                         "share_of_group": round(n / own.size, 6),
                         "expected_rows_in_cap_draw":
                             int(round(PER_GROUP_CAP * n / own.size))})
        log(f"  {gn}: pool {own.size:,}")
    n_sink = int((g_of[c_tr] == SINK).sum())
    rows.append({"group": "sink", "crop": "(non-crop)", "lu_code": 0,
                 "pool_rows": n_sink, "share_of_group": 1.0,
                 "expected_rows_in_cap_draw": PER_GROUP_CAP})
    write_csv(f"{RUN}/stage2_pool_composition.csv", rows)

    # ---------------- validation-side objects, as the sweep builds them -------
    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    in_cal = np.isin(c_va, cal, assume_unique=True)
    c_va_cal = np.intersect1d(c_va, cal)
    classes2 = np.array([1, 2, 3, 4])
    P2 = np.load(f"{RUN}/stage2_prob_val.npy")
    assert P2.shape == (c_va.size, 4), P2.shape
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in classes2])
    ratio2 = ((1.0 / classes2.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]

    P3, ratio3, cls3 = {}, {}, {}
    for g, (gn, codes) in GROUPS.items():
        P3[g] = np.load(f"{RUN}/stage3_{gn}_prob_val.npy")
        cls3[g] = np.array(sorted(codes))
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, codes)))
        pi = np.array([(y[own] == c).mean() for c in cls3[g]])
        ratio3[g] = ((1.0 / len(codes)) / np.where(pi > 0, pi, 1e-9))[None, :]

    m_tune = ~in_cal
    pos_tune = np.searchsorted(tune, c_va[m_tune])
    y_tune = np.where(np.isin(y[tune], CROPS), y[tune], 0)

    def score(code):
        pred = np.zeros(tune.size, dtype=np.int32)
        pred[pos_tune] = code[m_tune]
        return f1_score(y_tune, pred, labels=CROPS, average=None, zero_division=0)

    def compose(g_hat, alpha3):
        code = np.zeros(c_va.size, dtype=np.int32)
        for g in GROUPS:
            s = g_hat == g
            if s.any():
                code[s] = cls3[g][reweight(P3[g][s], ratio3[g], alpha3).argmax(1)]
        return code

    # ---------------- routing accuracy per crop ------------------------------
    rows = []
    for name, alpha in (("argmax", 0.0), (f"alpha2={a2}", a2)):
        g_hat = classes2[reweight(P2, ratio2, alpha).argmax(1)]
        for c in CROPS:
            m = m_tune & (y[c_va] == c)
            n = int(m.sum())
            rows.append({"operating_point": name, "crop": NM[c], "lu_code": c,
                         "tune_candidates": n,
                         "routed_to_true_group":
                             round(float((g_hat[m] == g_of[c_va][m]).mean()), 4)
                             if n else 0.0})
    write_csv(f"{RUN}/stage2_routing_accuracy.csv", rows)
    for r in rows:
        if r["operating_point"] != "argmax":
            log(f"  {r['crop']:<12}{r['routed_to_true_group']:>8.1%}  "
                f"({r['tune_candidates']:,} tune candidates)")

    # ---------------- oracle routing -----------------------------------------
    learned = classes2[reweight(P2, ratio2, a2).argmax(1)]
    f_learned = score(compose(learned, a3))
    f_oracle = score(compose(g_of[c_va], a3))
    log(f"\n  learned route  tune macro F1 {f_learned.mean():.4f}")
    log(f"  oracle route   tune macro F1 {f_oracle.mean():.4f}")
    log(f"  headroom       {f_oracle.mean() - f_learned.mean():+.4f}")
    rows = [{"crop": NM[c], "lu_code": c,
             "tune_support": int((y_tune == c).sum()),
             "f1_learned_route": round(float(f_learned[i]), 4),
             "f1_oracle_route": round(float(f_oracle[i]), 4),
             "delta": round(float(f_oracle[i] - f_learned[i]), 4)}
            for i, c in enumerate(CROPS)]
    rows.append({"crop": "MACRO", "lu_code": 0, "tune_support": int(tune.size),
                 "f1_learned_route": round(float(f_learned.mean()), 4),
                 "f1_oracle_route": round(float(f_oracle.mean()), 4),
                 "delta": round(float(f_oracle.mean() - f_learned.mean()), 4)})
    write_csv(f"{RUN}/oracle_routing.csv", rows)


if __name__ == "__main__":
    main()
