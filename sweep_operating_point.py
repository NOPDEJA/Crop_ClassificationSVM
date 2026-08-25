"""sweep_operating_point.py

Select an operating point (alpha2, alpha3) HONESTLY, from saved *_prob_val.npy.

This is the gate procedure of docs/PLAN_2026-08-26_WEIGHTED_RUN.md section 3, and
it is deliberately one script rather than two: the weighted run and M5 must be
compared on numbers produced by the identical grid, the identical selection half,
the identical scorer and the identical rows, or the gate means nothing.

DIFFERENCE FROM m2_operating_point.py. That script selected on the TUNING half and
scored the calibration half as a stability check. Selecting on tune and then also
reporting tune is selection on the reported number, so the score it reports is
optimistic -- measured at roughly +0.047 for the M2 argmax. Here selection happens
on the CALIBRATION half and the selected cell is scored ONCE on the tuning half,
which the selection never saw. The tuning half is therefore a clean held-out
score, at the cost of selecting against rows the Platt sigmoids were fitted on.
That is the honest trade: selection is in-sample to calibration, the reported
number is not in-sample to anything.

Scoring is the strict convention (strict-scoring-convention in project memory):
the whole half, non-crop truth mapped to 0, the 13 crop labels fixed, so false
positives on non-crop pixels count against precision.

Taxonomy is inferred from which stage3_*_prob_val.npy the run wrote, so a
MERGE_TREE run scores through the same code path.

Env:
  RUN_DIR=<dir>   run to sweep
  GRID_MAX=1.2    top of the alpha axis (13 values at 0.1 spacing -> 169 cells)
  ALSO=0.2,0.7    extra cell to report explicitly, for cross-run comparison
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import f1_score

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m5")
GRID_MAX = float(os.environ.get("GRID_MAX", 1.2))
ALSO = tuple(float(v) for v in os.environ.get("ALSO", "0.2,0.7").split(","))
AXIS = [round(v, 2) for v in np.arange(0.0, GRID_MAX + 1e-9, 0.1)]
SINK = 4
STD = {1: ("orchards", [2403, 2404, 2407, 2413, 2416, 2419, 2420]),
       2: ("plantation", [2302, 2303, 2405]),
       3: ("field", [2101, 2204, 2205])}
MERGED = {1: ("tree", [2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420]),
          3: ("field", [2101, 2204, 2205])}
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
CROPS = sorted(NM)


def log(*a):
    print(*a, flush=True)


def reweight(P, ratio, alpha):
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


def main():
    groups = MERGED if os.path.exists(f"{RUN}/stage3_tree_prob_val.npy") else STD
    gname = {g: n for g, (n, _) in groups.items()}
    codes = {g: c for g, (_, c) in groups.items()}
    # y comes from the matrix this run was trained on; every arm's npz carries the
    # same rows in the same order, but reading the run's own is one less assumption
    npz = json.load(open(f"{RUN}/manifest.json"))["npz"]
    log(f"sweep {RUN}")
    log(f"  taxonomy {[(gname[g], len(codes[g])) for g in groups]} + sink   y from {npz}")

    y = np.load(npz, allow_pickle=True)["y"].astype(np.int32)
    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    in_cal = np.isin(c_va, cal, assume_unique=True)
    c_va_cal = np.intersect1d(c_va, cal)
    assert np.intersect1d(cal, tune).size == 0, "cal and tune overlap"

    g_of = np.zeros(y.size, dtype=np.int32)
    for g in groups:
        g_of[np.isin(y, codes[g])] = g
    g_of[g_of == 0] = SINK

    # Denominators from the CALIBRATION half only, exactly as train_parcel_cascade
    # composes them, so a cell selected here means the same thing at compose time.
    classes2 = np.array(sorted(list(groups) + [SINK]))
    P2 = np.load(f"{RUN}/stage2_prob_val.npy")
    assert P2.shape[0] == c_va.size and P2.shape[1] == classes2.size, (P2.shape, classes2)
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in classes2])
    ratio2 = ((1.0 / classes2.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]

    P3, ratio3, cls3 = {}, {}, {}
    for g in groups:
        P3[g] = np.load(f"{RUN}/stage3_{gname[g]}_prob_val.npy")
        cls3[g] = np.array(sorted(codes[g]))
        assert P3[g].shape == (c_va.size, cls3[g].size), (g, P3[g].shape)
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, codes[g])))
        pi = np.array([(y[own] == c).mean() for c in cls3[g]])
        ratio3[g] = ((1.0 / len(codes[g])) / np.where(pi > 0, pi, 1e-9))[None, :]

    halves = {}
    for name, rows, m in (("cal", cal, in_cal), ("tune", tune, ~in_cal)):
        halves[name] = (rows, np.where(np.isin(y[rows], CROPS), y[rows], 0),
                        np.searchsorted(rows, c_va[m]), m)
        log(f"  {name}: {rows.size:,} rows, {int(m.sum()):,} candidates")

    def score(name, code):
        rows, y_eval, pos, m = halves[name]
        pred = np.zeros(rows.size, dtype=np.int32)
        pred[pos] = code[m]
        return f1_score(y_eval, pred, labels=CROPS, average=None, zero_division=0)

    def compose(a3, g_hat):
        code = np.zeros(c_va.size, dtype=np.int32)
        for g in groups:
            sel = g_hat == g
            if sel.any():
                code[sel] = cls3[g][reweight(P3[g][sel], ratio3[g], a3).argmax(1)]
        return code

    log(f"\n  sweeping {len(AXIS)}x{len(AXIS)} = {len(AXIS) ** 2} cells")
    out = []
    for a2 in AXIS:
        g_hat = classes2[reweight(P2, ratio2, a2).argmax(1)]
        for a3 in AXIS:
            code = compose(a3, g_hat)
            row = {"alpha2": a2, "alpha3": a3}
            for name in ("cal", "tune"):
                f = score(name, code)
                row[f"{name}_macro_f1"] = round(float(f.mean()), 4)
                row[f"{name}_alive"] = int((f >= 0.01).sum())
            out.append(row)
        log(f"    alpha2={a2} done")

    with open(f"{RUN}/opsweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    sel = max(out, key=lambda r: r["cal_macro_f1"])           # SELECTED ON CAL
    base = next(r for r in out if r["alpha2"] == 0.0 and r["alpha3"] == 0.0)
    also = next(r for r in out if (r["alpha2"], r["alpha3"]) == ALSO)
    log(f"\n  baseline (0.0, 0.0)   cal {base['cal_macro_f1']:.4f}"
        f"   tune {base['tune_macro_f1']:.4f} ({base['tune_alive']} alive)")
    log(f"  fixed cell {ALSO}   cal {also['cal_macro_f1']:.4f}"
        f"   tune {also['tune_macro_f1']:.4f} ({also['tune_alive']} alive)")
    log(f"  SELECTED on cal      a2={sel['alpha2']} a3={sel['alpha3']}"
        f"   cal {sel['cal_macro_f1']:.4f}")
    log(f"  GATE NUMBER (tune)   {sel['tune_macro_f1']:.4f}"
        f"   ({sel['tune_alive']} of 13 crops >= 0.01)")
    log(f"  optimism of selecting and reporting on the same half: "
        f"{sel['cal_macro_f1'] - sel['tune_macro_f1']:+.4f}")

    # per-crop breakdown of the selected cell on the tuning half
    g_hat = classes2[reweight(P2, ratio2, sel["alpha2"]).argmax(1)]
    f = score("tune", compose(sel["alpha3"], g_hat))
    _, y_eval, _, _ = halves["tune"]
    rows = [{"crop": NM[c], "lu_code": c, "f1": round(float(f[i]), 4),
             "support": int((y_eval == c).sum())} for i, c in enumerate(CROPS)]
    with open(f"{RUN}/opsweep_selected_tune.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log("")
    for r in rows:
        log(f"  {r['crop']:<12}{r['f1']:>9.4f}   {r['support']:,}")

    with open(f"{RUN}/opsweep_selection.json", "w") as fh:
        json.dump({"run": RUN, "grid": [AXIS[0], AXIS[-1], len(AXIS)],
                   "selected_on": "cal", "alpha2": sel["alpha2"],
                   "alpha3": sel["alpha3"], "cal_macro_f1": sel["cal_macro_f1"],
                   "tune_macro_f1": sel["tune_macro_f1"],
                   "tune_alive": sel["tune_alive"],
                   "baseline_tune_macro_f1": base["tune_macro_f1"],
                   "fixed_cell": list(ALSO),
                   "fixed_cell_tune_macro_f1": also["tune_macro_f1"]}, fh, indent=2)
    log(f"\nwrote {RUN}/opsweep.csv, opsweep_selected_tune.csv, opsweep_selection.json")


if __name__ == "__main__":
    main()
