"""m1_calibration_audit.py

M1 of docs/PLAN_S2_3DATE_COMPETITIVE.md: is the cascade actually calibrated?

This decides how M2 must be described. Every stage of the parcel run fits its
Platt sigmoids on validation rows drawn at NATURAL priors, so a mathematically
correct correction toward those same priors should be close to a no-op. If the
calibrators are already good, then sweeping the (alpha_2, alpha_3) exponents is
OPERATING-POINT TUNING and must be labelled as such -- not "prior correction",
which implies it is repairing a distortion that the numbers here would show does
not exist.

Nothing is refitted. This reads the saved *_prob_val.npy arrays only.

ONE DEPARTURE FROM THE PLAN. It asks for the audit "on the calibration half".
That half is what the sigmoids were fitted on, so scores there are optimistic --
it measures how well calibration fitted, not how well it generalises. Both halves
are reported. The tuning half is the honest number; the gap between them is how
much the calibrator overfitted, which is worth knowing on its own because several
crops have very few calibration positives.

Env:
  RUN_DIR=<dir>   run to audit (default ./runs/s2_2018_3date_parcel_m0)
"""
import csv
import os

import numpy as np

from config import NPZ

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
SPLIT_ASSIGN = "./splits/split_assign.npy"
N_BINS = 10

ECON = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
WATER = {4101, 4102, 4103, 4201, 4202, 4203}
FOREST = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
GROUPS = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
          2: [2302, 2303, 2405],
          3: [2101, 2204, 2205]}
SINK = 4
SUPNAME = {1: "economic", 2: "water", 3: "others", 4: "forest"}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
CROPNAME = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
            2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
            2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
            2420: "Langsat"}


def ece_and_brier(p, hit):
    """Expected calibration error (equal-width bins) and Brier score.

    `p` is the predicted probability of the class, `hit` the 0/1 outcome. ECE is
    the support-weighted mean gap between predicted confidence and observed
    frequency, so it answers "when this model says 0.7, does it happen 70% of
    the time?" -- which is exactly what prior correction would be repairing.
    """
    brier = float(np.mean((p - hit) ** 2))
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, N_BINS - 1)
    tot = p.size
    ece = 0.0
    rows = []
    for k in range(N_BINS):
        m = b == k
        n = int(m.sum())
        if not n:
            rows.append((k, 0, np.nan, np.nan))
            continue
        conf, freq = float(p[m].mean()), float(hit[m].mean())
        ece += (n / tot) * abs(conf - freq)
        rows.append((k, n, conf, freq))
    return ece, brier, rows


def audit(tag, P, classes, y_true, halves, out_rows, rel_rows, names):
    """Per-class calibration on each half of fold 1."""
    print(f"\n{'=' * 74}\n{tag}\n{'=' * 74}")
    print(f"{'class':<12}{'half':<6}{'n':>10}{'pred prev':>11}{'obs prev':>10}"
          f"{'ratio':>8}{'ECE':>8}{'Brier':>9}")
    for hname, hmask in halves:
        for j, c in enumerate(classes):
            p = P[hmask, j].astype(np.float64)
            hit = (y_true[hmask] == c).astype(np.float64)
            if p.size == 0:
                continue
            ece, brier, rel = ece_and_brier(p, hit)
            pred_prev, obs_prev = float(p.mean()), float(hit.mean())
            ratio = pred_prev / obs_prev if obs_prev > 0 else np.nan
            label = names[c]
            print(f"{label:<12}{hname:<6}{p.size:>10,}{pred_prev:>11.5f}"
                  f"{obs_prev:>10.5f}{ratio:>8.2f}{ece:>8.4f}{brier:>9.5f}")
            out_rows.append({"stage": tag, "class": label, "half": hname,
                             "n": p.size, "pred_prevalence": round(pred_prev, 6),
                             "obs_prevalence": round(obs_prev, 6),
                             "prev_ratio": round(ratio, 4) if obs_prev > 0 else "",
                             "ece": round(ece, 5), "brier": round(brier, 6)})
            if hname == "tune":
                for k, n, conf, freq in rel:
                    rel_rows.append({"stage": tag, "class": label, "bin": k,
                                     "n": n,
                                     "mean_pred": "" if n == 0 else round(conf, 5),
                                     "obs_freq": "" if n == 0 else round(freq, 5)})


if __name__ == "__main__":
    print(f"M1 calibration audit -- {RUN}")
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    asg = np.load(SPLIT_ASSIGN)

    va = np.load(f"{RUN}/stage1_val_idx.npy")
    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    in_cal = np.isin(va, cal, assume_unique=True)
    print(f"fold 1: {va.size:,} rows -> {int(in_cal.sum()):,} calibrate / "
          f"{int((~in_cal).sum()):,} tune")

    out_rows, rel_rows = [], []

    # ---- Stage 1 ---------------------------------------------------------
    sup = np.where(np.isin(y, list(ECON)), 1,
          np.where(np.isin(y, list(WATER)), 2,
          np.where(np.isin(y, list(FOREST)), 4, 3))).astype(np.int32)
    P1 = np.load(f"{RUN}/stage1_prob_val.npy")
    audit("stage1 (superclass)", P1, [1, 2, 3, 4], sup[va],
          [("cal", in_cal), ("tune", ~in_cal)], out_rows, rel_rows, SUPNAME)
    del P1

    # ---- Stage 2 ---------------------------------------------------------
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, codes)] = g
    g_of[g_of == 0] = SINK
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    c_in_cal = np.isin(c_va, cal, assume_unique=True)
    P2 = np.load(f"{RUN}/stage2_prob_val.npy")
    audit("stage2 (subclass + sink)", P2, [1, 2, 3, 4], g_of[c_va],
          [("cal", c_in_cal), ("tune", ~c_in_cal)], out_rows, rel_rows, GNAME)
    del P2

    # ---- Stage 3 ---------------------------------------------------------
    # Each expert scores every candidate, but it was calibrated only on rows
    # whose TRUE crop is in its group, so that is the population to audit.
    for g, codes in GROUPS.items():
        P3 = np.load(f"{RUN}/stage3_{GNAME[g]}_prob_val.npy")
        own = np.isin(y[c_va], codes)
        audit(f"stage3 {GNAME[g]}", P3, sorted(codes), y[c_va],
              [("cal", own & c_in_cal), ("tune", own & ~c_in_cal)],
              out_rows, rel_rows, CROPNAME)
        del P3

    # ---- is there even a stable "true prior" to correct toward? -----------
    # Prior correction assumes pi_true is a property of the world. The folds are
    # disjoint PARCEL samples, so if prevalence swings between them there is no
    # single pi_true, and an exponent fitted on one fold is an operating point
    # for that fold rather than a correction toward truth.
    te = np.flatnonzero(asg == 2)
    pops = [("cal", cal), ("tune", tune), ("test", te)]
    print(f"\n{'=' * 74}\nclass prevalence across disjoint parcel samples\n{'=' * 74}")
    print(f"{'class':<12}{'within':<12}" + "".join(f"{n:>11}" for n, _ in pops) + f"{'max/min':>10}")
    prev_rows = []
    for gl, codes in GROUPS.items():
        for c in sorted(codes):
            vals = []
            for _, idx in pops:
                own = np.isin(y[idx], codes)
                vals.append(float((y[idx][own] == c).mean()) if own.any() else np.nan)
            spread = max(vals) / min(vals) if min(vals) > 0 else np.inf
            print(f"{CROPNAME[c]:<12}{GNAME[gl]:<12}"
                  + "".join(f"{v:>11.5f}" for v in vals) + f"{spread:>10.2f}")
            prev_rows.append({"class": CROPNAME[c], "within": GNAME[gl],
                              **{n: round(v, 6) for (n, _), v in zip(pops, vals)},
                              "max_over_min": round(spread, 3)})
    print(f"\n{'superclass':<12}{'within':<12}" + "".join(f"{n:>11}" for n, _ in pops) + f"{'max/min':>10}")
    for sc in (1, 2, 3, 4):
        vals = [float((sup[idx] == sc).mean()) for _, idx in pops]
        spread = max(vals) / min(vals) if min(vals) > 0 else np.inf
        print(f"{SUPNAME[sc]:<12}{'all':<12}" + "".join(f"{v:>11.5f}" for v in vals)
              + f"{spread:>10.2f}")
        prev_rows.append({"class": SUPNAME[sc], "within": "all",
                          **{n: round(v, 6) for (n, _), v in zip(pops, vals)},
                          "max_over_min": round(spread, 3)})
    with open(f"{RUN}/m1_prior_stability.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(prev_rows[0]))
        w.writeheader()
        w.writerows(prev_rows)

    with open(f"{RUN}/m1_calibration.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with open(f"{RUN}/m1_reliability.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0]))
        w.writeheader()
        w.writerows(rel_rows)
    print(f"\nwrote {RUN}/m1_calibration.csv and m1_reliability.csv")
