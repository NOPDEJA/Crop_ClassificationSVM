"""m2_operating_point.py

M2 of docs/PLAN_S2_3DATE_COMPETITIVE.md: sweep the (alpha_2, alpha_3) exponents on
the M0 parcel run and measure what they buy.

WHY THIS IS NOT "PRIOR CORRECTION", AND WHY THE PLAN'S DENOMINATOR IS WRONG HERE
--------------------------------------------------------------------------------
The plan says to take the denominator from the training population's priors, as
`sweep_prior_alpha.py` does for the v2 arm. That is right for v2, whose models were
fitted with class_weight='balanced' plus upsampling and calibrated by internal CV on
that rebalanced sample -- so their probabilities sat at a rebalanced prior and pulling
them back toward the tile's real prior was a genuine correction.

It is wrong for M0. M0 uses caps only, no weighting, no upsampling, and fits its Platt
sigmoids on validation rows at NATURAL priors. M1 confirms the result: predicted over
observed prevalence is 0.86-1.29 at Stage 1 and 0.91-1.26 at Stage 2 on both halves of
fold 1. The probabilities are already at natural priors. Dividing by the capped training
prior would apply a correction that calibration has already performed -- double-counting
it -- and would look like a large effect while being an error.

M1 also showed there is no stable pi_true to correct toward: across the three disjoint
parcel samples, within-group prevalence swings 25.6x for Langsat, 4.9x for Oil palm and
3.7x for Mango. "The true prior" is a property of which parcels were drawn.

So this sweep is an OPERATING POINT, stated plainly: it reweights each class from the
natural prior toward uniform,

    p'(c|x)  proportional to  p(c|x) * ( (1/K) / pi_cal(c) ) ** alpha

with alpha=0 the argmax M0 already reports and alpha=1 full rebalancing. The question it
answers is not "what is the correct prior" but "how much rare-class recall can be bought,
and at what cost to macro F1". Stage 1 is held fixed, so candidacy never changes.

SELECTION DISCIPLINE. Selection is on the TUNING half of fold 1 only. Fold 2 is not read
here -- that is M5's single predeclared evaluation. The calibration half is scored too,
not to select on, but to test whether the chosen point is stable across two disjoint
parcel samples. M1 predicts it will not be.

Env:
  RUN_DIR=<dir>   run to sweep (default ./runs/s2_2018_3date_parcel_m0)
"""
import csv
import os

import numpy as np
from sklearn.metrics import f1_score

from config import NPZ, PER_LU_CAP

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
SPLIT_ASSIGN = "./splits/split_assign.npy"
GROUPS = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
          2: [2302, 2303, 2405],
          3: [2101, 2204, 2205]}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
SINK = 4
CROPS = sorted(c for cs in GROUPS.values() for c in cs)
AXIS = [round(v, 2) for v in np.arange(0.0, 1.201, 0.1)]
# logged Stage-3 fit totals; Amendment 4 requires these be re-derived, not reused
EXPECTED_FIT3 = {1: 172_371, 2: 148_344, 3: 210_000}


def log(*a):
    print(*a, flush=True)


def corrected(P, ratio, alpha):
    """p * ratio**alpha, renormalised. alpha=0 leaves P untouched."""
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


if __name__ == "__main__":
    log(f"M2 operating-point sweep -- {RUN}")
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    asg = np.load(SPLIT_ASSIGN)
    tr = np.flatnonzero(asg == 0)

    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    in_cal = np.isin(c_va, cal, assume_unique=True)

    # ---- Amendment 4: Stage-3 denominators re-derived from THIS run -------
    # fit3 = cap(group train rows, per LU, PER_LU_CAP), so the fitted count per
    # class is min(N_c in the training fold, PER_LU_CAP) -- deterministic. The
    # totals are checked against the run log; a mismatch means the sampling rule
    # changed and the denominators would be silently wrong.
    fit3 = {}
    for g, codes in GROUPS.items():
        counts = np.array([min(int((y[tr] == c).sum()), PER_LU_CAP) for c in sorted(codes)])
        fit3[g] = counts
        got, want = int(counts.sum()), EXPECTED_FIT3[g]
        status = "OK" if got == want else "MISMATCH"
        log(f"  stage3 {GNAME[g]:<11} fitted {got:>8,} vs logged {want:>8,}  {status}")
        if got != want:
            raise SystemExit(
                f"stage-3 fitted counts for {GNAME[g]} do not reproduce the run log "
                f"({got} vs {want}); the denominators would be wrong. Refusing to sweep.")

    # ---- reweighting targets, from the CALIBRATION half ------------------
    # The probabilities sit at the calibration population's priors (M1), so that
    # is the denominator. Target is uniform: alpha is "how far toward balanced".
    g_of = np.zeros(y.size, dtype=np.int32)
    for g, codes in GROUPS.items():
        g_of[np.isin(y, codes)] = g
    g_of[g_of == 0] = SINK

    pi2 = np.array([(g_of[c_va][in_cal] == g).mean() for g in (1, 2, 3, SINK)])
    ratio2 = (0.25 / pi2)[None, :]
    log("\n  stage2 pi_cal " + " ".join(f"{GNAME[g]}={p:.4f}" for g, p in zip((1, 2, 3, SINK), pi2)))

    ratio3 = {}
    for g, codes in GROUPS.items():
        own = np.isin(y[c_va], codes) & in_cal
        pi = np.array([(y[c_va][own] == c).mean() for c in sorted(codes)])
        pi = np.where(pi > 0, pi, 1e-9)
        ratio3[g] = ((1.0 / len(codes)) / pi)[None, :]

    P2 = np.load(f"{RUN}/stage2_prob_val.npy")
    P3 = {g: np.load(f"{RUN}/stage3_{GNAME[g]}_prob_val.npy") for g in GROUPS}
    classes2 = np.array([1, 2, 3, SINK])
    classes3 = {g: np.array(sorted(codes)) for g, codes in GROUPS.items()}

    halves = {}
    for name, idx in (("cal", cal), ("tune", tune)):
        rows = idx
        y_eval = np.where(np.isin(y[rows], CROPS), y[rows], 0)
        # position of each candidate within this half's row list
        pos = np.searchsorted(rows, c_va[in_cal if name == "cal" else ~in_cal])
        halves[name] = (rows, y_eval, pos, in_cal if name == "cal" else ~in_cal)
        log(f"  {name}: {rows.size:,} rows, {pos.size:,} candidates")

    log(f"\n  sweeping {len(AXIS)}x{len(AXIS)} = {len(AXIS) ** 2} cells")
    out = []
    for a2 in AXIS:
        P2c = corrected(P2, ratio2, a2)
        g_hat = classes2[P2c.argmax(1)]
        for a3 in AXIS:
            code = np.zeros(c_va.size, dtype=np.int32)
            for g in GROUPS:
                sel = g_hat == g
                if sel.any():
                    code[sel] = classes3[g][corrected(P3[g][sel], ratio3[g], a3).argmax(1)]
            row = {"alpha2": a2, "alpha3": a3}
            for name in ("cal", "tune"):
                rows, y_eval, pos, m = halves[name]
                pred = np.zeros(rows.size, dtype=np.int32)
                pred[pos] = code[m]
                f = f1_score(y_eval, pred, labels=CROPS, average=None, zero_division=0)
                row[f"{name}_macro_f1"] = round(float(f.mean()), 4)
                row[f"{name}_alive"] = int((f >= 0.01).sum())
            out.append(row)
        log(f"    alpha2={a2} done")

    with open(f"{RUN}/m2_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    base = next(r for r in out if r["alpha2"] == 0.0 and r["alpha3"] == 0.0)
    sel = max(out, key=lambda r: r["tune_macro_f1"])          # selected on TUNE only
    ora = max(out, key=lambda r: r["cal_macro_f1"])           # what CAL would have picked
    log(f"\n  baseline (0,0)      tune {base['tune_macro_f1']:.4f} ({base['tune_alive']} alive)"
        f"   cal {base['cal_macro_f1']:.4f}")
    log(f"  selected on tune    a2={sel['alpha2']} a3={sel['alpha3']}"
        f"   tune {sel['tune_macro_f1']:.4f} ({sel['tune_alive']} alive)"
        f"   cal {sel['cal_macro_f1']:.4f}")
    log(f"  best on cal         a2={ora['alpha2']} a3={ora['alpha3']}"
        f"   cal {ora['cal_macro_f1']:.4f}   tune {ora['tune_macro_f1']:.4f}")
    log(f"\n  STABILITY: the two disjoint parcel halves "
        f"{'AGREE' if (sel['alpha2'], sel['alpha3']) == (ora['alpha2'], ora['alpha3']) else 'DISAGREE'}"
        f" on the operating point")
    log(f"  transfer cost of selecting on tune, scored on cal: "
        f"{sel['cal_macro_f1'] - ora['cal_macro_f1']:+.4f}")
    log(f"\nwrote {RUN}/m2_sweep.csv")
