"""validate_alpha_split.py

Split-half validation of the operating point, plus the comparison against the prior
Random Forest paper.

The operating point of §6.7 (alpha_2=0.8, alpha_3=0.3) was chosen by ranking 256 cells
on `held_out` and then reported on `held_out`. That is selection on the evaluation set,
so the reported figures are optimistically biased. The bias should be small -- two
scalar parameters fitted against 21 million pixels, on a response surface measured to be
flat -- but "should be small" is not a measurement.

So measure it. `held_out` is split in half at random:

    A (selection)  choose alpha by the same rule as §6.7: best econ-13 macro F1
                   among cells that keep all 13 crops alive
    B (reporting)  score the chosen alpha, having never been looked at during selection

Three numbers on B make the bias legible:

    baseline        alpha = (0, 0)
    selected        the alpha chosen on A            <- the honest, unbiased figure
    oracle          the best alpha on B itself       <- what selection could have got
                                                        away with; the gap is the bias

Also scores the `matched` population drawn from B -- held-out pixels resampled to the
prior RF paper's per-class test supports, so class priors match and weighted metrics are
comparable with it.

Writes runs/<RUN>/alpha_split_validation_{grid,summary,per_class}.csv
"""
import time
import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE
from sweep_prior_alpha import Context, compose
from evaluate_flat_15class import (to_flat_true, to_flat_pred, score, matched_subset,
                                   CROPS, NAMES, RESERVOIR, OTHERS, FOREST)
from confusion_report import _matrix, per_class_from_flat, confusion_set, write_confusion_set

ALIVE_EPS = 0.001
AXIS = [round(v, 2) for v in np.arange(0.0, 1.001, 0.1)] + [0.3, 0.8]
AXIS = sorted(set(AXIS))
SPLIT_SEED = 20260821

# Prior RF paper, its own test split (S2_SVM_ANALYSIS.md §6.3)
RF_PAPER = {"accuracy": 0.716, "kappa": 0.678, "f1_weighted": 0.714, "n_pixels": 303_947}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def econ_scores(y, final, idx, codes):
    """econ-13 macro F1 and classes-alive on rows `idx`.

    Label 0 ("no code") must be in the label set: a true-econ pixel the cascade dropped
    at Stage 1, or routed to other_econ, predicts 0. Without that column those pixels
    leave the matrix and recall is computed against a shrunken denominator.
    """
    m = np.isin(y[idx], codes)
    cm = _matrix(y[idx][m], final[idx][m], codes + [0],
                 [CROPS[c] for c in codes] + ["(no code)"])
    pc = per_class_from_flat(cm).iloc[:len(codes)]
    return float(pc.f1.mean()), int((pc.f1 >= ALIVE_EPS).sum())


def overall(y_flat, p_flat, idx):
    lab15 = list(CROPS) + [RESERVOIR, OTHERS]
    yf = np.where(y_flat[idx] == FOREST, OTHERS, y_flat[idx])
    pf = np.where(p_flat[idx] == FOREST, OTHERS, p_flat[idx])
    cm = _matrix(yf, pf, lab15, [NAMES[l] for l in lab15])
    v = cm.values.astype(np.float64)
    acc = np.diag(v).sum() / v.sum()
    pe = (v.sum(0) * v.sum(1)).sum() / v.sum() ** 2
    pc = per_class_from_flat(cm)
    w = pc["support"] / pc["support"].sum()
    return {"accuracy": round(acc, 4), "kappa": round((acc - pe) / (1 - pe), 4),
            "f1_weighted": round(float((pc.f1 * w).sum()), 4),
            "f1_macro": round(float(pc.f1.mean()), 4)}


if __name__ == "__main__":
    ctx = Context()
    y, s1, held = ctx.y, ctx.s1, ctx.held
    codes = list(CROPS)

    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(held.size)
    A, B = held[perm[:held.size // 2]], held[perm[held.size // 2:]]
    log(f"held_out {held.size:,} -> selection A {A.size:,} | reporting B {B.size:,}")
    log(f"grid {len(AXIS)}x{len(AXIS)} = {len(AXIS)**2} cells: {AXIS}")

    rows = []
    for a2 in AXIS:
        for a3 in AXIS:
            final, _ = compose(ctx, a2, a3)
            fA, nA = econ_scores(y, final, A, codes)
            fB, nB = econ_scores(y, final, B, codes)
            rows.append({"alpha2": a2, "alpha3": a3,
                         "A_econ13_macro_f1": round(fA, 4), "A_alive": nA,
                         "B_econ13_macro_f1": round(fB, 4), "B_alive": nB})
        log(f"  alpha2={a2} done ({len(rows)} cells)")

    g = pd.DataFrame(rows)
    g.to_csv(f"{OUT_DIR}/alpha_split_validation_grid.csv", index=False, encoding="utf-8-sig")

    # selection rule, applied to A only -- identical to the rule used in §6.7
    fullA = g[g.A_alive == len(codes)]
    sel = (fullA if len(fullA) else g).loc[
        (fullA if len(fullA) else g).A_econ13_macro_f1.idxmax()]
    # what selection could have achieved had it cheated and looked at B
    fullB = g[g.B_alive == len(codes)]
    ora = (fullB if len(fullB) else g).loc[
        (fullB if len(fullB) else g).B_econ13_macro_f1.idxmax()]
    base = g[(g.alpha2 == 0.0) & (g.alpha3 == 0.0)].iloc[0]
    pub = g[(g.alpha2 == 0.8) & (g.alpha3 == 0.3)].iloc[0]

    log(f"selected on A : a2={sel.alpha2} a3={sel.alpha3} "
        f"(A={sel.A_econ13_macro_f1}, alive {sel.A_alive}) -> B={sel.B_econ13_macro_f1}")
    log(f"oracle on B   : a2={ora.alpha2} a3={ora.alpha3} -> B={ora.B_econ13_macro_f1}")
    log(f"published pt  : a2=0.8 a3=0.3 -> B={pub.B_econ13_macro_f1}")
    log(f"SELECTION BIAS: {round(ora.B_econ13_macro_f1 - sel.B_econ13_macro_f1, 4)} "
        f"macro F1 (oracle-on-B minus selected-on-A, both scored on B)")

    y_flat = to_flat_true(y)
    rng2 = np.random.default_rng(RANDOM_STATE)
    matched_B = B[matched_subset(y_flat[B], rng2)]
    log(f"matched drawn from B: {matched_B.size:,} px "
        f"(prior RF paper test split: {RF_PAPER['n_pixels']:,})")

    variants = {"baseline (a=0,0)": (0.0, 0.0),
                f"selected on A (a={sel.alpha2},{sel.alpha3})": (sel.alpha2, sel.alpha3),
                "published (a=0.8,0.3)": (0.8, 0.3)}
    summary, per_class = [], []
    for name, (a2, a3) in variants.items():
        final, s2f = compose(ctx, a2, a3)
        p_flat = to_flat_pred(s1, final)
        f_all, n_all = econ_scores(y, final, B, codes)
        for pop, idx in (("B (reporting half)", B), ("matched from B", matched_B)):
            o = overall(y_flat, p_flat, idx)
            summary.append({"variant": name, "population": pop, "n_pixels": int(idx.size),
                            **o, "econ13_macro_f1": round(f_all, 4) if "B (" in pop else None})
        rws, _ = score(y_flat[B], p_flat[B], fold_forest=True, tag=name)
        per_class += rws
        log(f"{name}: {summary[-2]}")

    summary.append({"variant": "prior RF paper (its own test split)",
                    "population": "matched-equivalent",
                    "n_pixels": RF_PAPER["n_pixels"], "accuracy": RF_PAPER["accuracy"],
                    "kappa": RF_PAPER["kappa"], "f1_weighted": RF_PAPER["f1_weighted"],
                    "f1_macro": None, "econ13_macro_f1": None})

    sdf = pd.DataFrame(summary)
    sdf.to_csv(f"{OUT_DIR}/alpha_split_validation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(f"{OUT_DIR}/alpha_split_validation_per_class.csv",
                                   index=False, encoding="utf-8-sig")
    log("\n" + sdf.to_string(index=False))
    log(f"saved to {OUT_DIR}/alpha_split_validation_*.csv")
