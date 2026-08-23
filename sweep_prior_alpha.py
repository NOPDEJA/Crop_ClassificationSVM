"""sweep_prior_alpha.py

The post-hoc prior sweep of .scratch/class-prior-regime (ticket 04).

Sweeps two axes over the saved full-tile probability arrays -- nothing is refitted,
so a cell costs seconds instead of the hour a retrain costs:

    alpha_2   Stage-2 routing prior   (orchards / plantation / field)
    alpha_3   Stage-3 species prior   (crop within group)

Stage 1 is held fixed. Its rule is not argmax but a sequential overwrite
(stage1_weight_scale.py:315-323), so correcting its probabilities would silently
change the operating point; it gets its own ticket.

**Parameterisation.** The obvious form, pi_target ∝ N_c^alpha, assumes the model's
probabilities carry a uniform prior, so that alpha=0 is a no-op. That holds at Stage 2
(fitted on exactly 140,000 pixels per subclass) but NOT at Stage 3, whose sampled
distribution is capped-and-upsampled, roughly N_c^0.27. Both stages calibrate with
CalibratedClassifierCV(sigmoid) fitted on that sampled data, so the probabilities are
anchored to the *sampled* prior, not to the class_weight-adjusted one. Using N_c^alpha
directly would therefore make alpha=0 a real correction at Stage 3 and the sweep would
not contain the baseline.

The geometric form fixes this and reduces to the intended one wherever pi_train is
uniform:

    pi_target(alpha) ∝ pi_train^(1-alpha) · pi_true^alpha
    equivalently   p'(c|x) ∝ p(c|x) · (pi_true(c)/pi_train(c))^alpha

alpha is then the *strength* of the correction toward the true prior: 0 = the model as
trained, 1 = fully corrected. Landmarks keep their meaning -- alpha=1 is §6.6's oracle,
alpha_2=0.858 is §6.6's EM correction, alpha=0.552 is the predicted
parcel-proportional exponent.

pi_true is counted on the population actually routed to each stage, recomputed per
alpha_2 because alpha_2 changes what Stage 3 receives.

Outputs (in runs/<RUN>/sweep/):
  sweep_results.csv        one row per (alpha_2, alpha_3) cell
  confusion/<tag>_*.csv    full matrix set, landmarks + winners only
"""
import json
import itertools
import time
import numpy as np
import pandas as pd

from config import NPZ, OUT_DIR, TAG, STAGE1_PRED, STAGE3_META_TPL
from evaluate_flat_15class import (to_flat_true, to_flat_pred, CROPS, NAMES,
                                   RESERVOIR, OTHERS, FOREST)
from confusion_report import (confusion_set, write_confusion_set,
                              per_class_from_flat, _matrix)

GROUPS = {1: "orchards", 2: "plantation", 3: "field"}
GROUP_CODES = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
               2: [2302, 2303, 2405],
               3: [2101, 2204, 2205]}
ALIVE_EPS = 0.001          # a class counts as alive at F1 >= this
LANDMARKS = [0.0, 0.552, 0.858, 1.0]

SWEEP_DIR = f"{OUT_DIR}/sweep"


def axis():
    """[0,1] step 0.1, densified to 0.05 across the predicted CI, plus landmarks.

    SWEEP_AXIS=0,1 runs a named subset — used to check the alpha=0 cell reproduces
    the baseline before committing to the full grid.
    """
    import os
    env = os.environ.get("SWEEP_AXIS")
    if env:
        return [float(v) for v in env.split(",")]
    a = set(np.round(np.arange(0.0, 1.001, 0.1), 3))
    a |= set(np.round(np.arange(0.45, 0.701, 0.05), 3))   # CI [0.415, 0.689]
    a |= set(LANDMARKS)
    return sorted(a)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def empirical_prior(y_sub, classes):
    c = np.array([(y_sub == k).sum() for k in classes], dtype=np.float64)
    return c / c.sum() if c.sum() else np.full(len(classes), 1.0 / len(classes))


def corrected(p, ratio, alpha):
    """p'(c|x) ∝ p(c|x) · ratio(c)^alpha, where ratio = pi_true / pi_train."""
    if alpha == 0.0:
        return p
    w = p * (ratio ** alpha)
    return w / w.sum(axis=1, keepdims=True)


class Context:
    """Everything a cell needs, loaded once.

    Module-level so apply_operating_point.py composes the cascade through the exact
    same code the sweep ranked -- a second copy of this arithmetic is precisely how
    the published operating point would drift from the curve that chose it.
    """

    def __init__(self):
        self.y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
        self.s1 = np.load(STAGE1_PRED)
        self.econ_idx = np.load(f"{OUT_DIR}/stage2_{TAG}_full_idx.npy")
        self.p2 = np.load(f"{OUT_DIR}/stage2_{TAG}_full_prob.npy")
        self.s2_classes = np.load(f"{OUT_DIR}/stage2_{TAG}_full_classes.npy")
        self.fitted = np.load(f"{OUT_DIR}/trainval_rows_mask.npy")
        self.held = np.flatnonzero(~self.fitted)
        self.y_econ = self.y[self.econ_idx]

        self.p3, self.c3, self.ratio3_den = {}, {}, {}
        for lab, grp in GROUPS.items():
            self.p3[lab] = np.load(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_prob.npy")
            self.c3[lab] = np.load(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_classes.npy")
            meta = json.load(open(STAGE3_META_TPL.format(grp=grp), encoding="utf-8"))
            tf = meta["counts"]["train_final"]
            cnt = np.array([tf[str(c)] for c in self.c3[lab]], dtype=np.float64)
            self.ratio3_den[lab] = cnt / cnt.sum()      # pi_train, the sampled prior

        # Stage 2 was fitted on exactly 140,000 per subclass -> pi_train uniform.
        pi2_train = np.full(len(self.s2_classes), 1.0 / len(self.s2_classes))
        sub_of = np.zeros(self.y_econ.shape, dtype=np.int32)
        for lab, codes in GROUP_CODES.items():
            sub_of[np.isin(self.y_econ, codes)] = lab
        self.ratio2 = empirical_prior(sub_of[sub_of > 0], self.s2_classes) / pi2_train


def compose(ctx, a2, a3):
    """Run the cascade at one grid point. Returns (final_lu, stage2_group_full).

    Both are full-length arrays aligned to the NPZ, so they drop straight into
    evaluate_flat_15class / confusion_report in place of the saved predictions.
    """
    g = ctx.s2_classes[corrected(ctx.p2, ctx.ratio2, a2).argmax(1)]
    code = np.zeros(ctx.econ_idx.size, dtype=np.int16)
    for lab in GROUPS:
        m = g == lab
        if not m.any():
            continue
        ytr = ctx.y_econ[m]
        r = (empirical_prior(ytr[np.isin(ytr, ctx.c3[lab])], ctx.c3[lab])
             / ctx.ratio3_den[lab])
        code[m] = ctx.c3[lab][corrected(ctx.p3[lab][m], r, a3).argmax(1)]
    final = np.zeros(ctx.y.size, dtype=np.int16)
    final[ctx.econ_idx] = code
    g_full = np.zeros(ctx.y.size, dtype=np.int32)
    g_full[ctx.econ_idx] = g
    return final, g_full


if __name__ == "__main__":
    import os
    os.makedirs(SWEEP_DIR, exist_ok=True)

    log("loading")
    ctx = Context()
    y, s1, econ_idx = ctx.y, ctx.s1, ctx.econ_idx
    p2, s2_classes, ratio2 = ctx.p2, ctx.s2_classes, ctx.ratio2
    p3, c3, ratio3_den = ctx.p3, ctx.c3, ctx.ratio3_den
    held, y_econ = ctx.held, ctx.y_econ

    y_flat_held = to_flat_true(y[held])
    lab15 = list(CROPS) + [RESERVOIR, OTHERS]
    names15 = [NAMES[l] for l in lab15]
    econ_codes = list(CROPS)
    ax = axis()
    log(f"grid {len(ax)} x {len(ax)} = {len(ax)**2} cells; axis = {ax}")

    def cell(a2, a3):
        final_, g_full = compose(ctx, a2, a3)
        return final_, g_full[econ_idx]

    def write_cells(cells):
        """Full matrix set for named grid points."""
        s2_full = np.zeros(y.size, dtype=np.int32)
        for a2, a3 in cells:
            final_, g_ = cell(a2, a3)
            s2_full[:] = 0
            s2_full[econ_idx] = g_
            write_confusion_set(confusion_set(y, s1, s2_full, final_, held),
                                SWEEP_DIR, f"a2-{a2}_a3-{a3}")
            log(f"  matrices for a2={a2} a3={a3}")

    # matrices-only mode: "0.8,0.3;0.8,0.8" — recompute named cells without resweeping
    import os
    only = os.environ.get("SWEEP_MATRIX_CELLS")
    if only:
        write_cells([tuple(float(v) for v in c.split(",")) for c in only.split(";")])
        raise SystemExit(0)

    rows, keep_mats = [], {}
    for a2 in ax:
        g = s2_classes[corrected(p2, ratio2, a2).argmax(1)]
        # pi_true for Stage 3 depends on what alpha_2 routed here, so recompute it
        slices, ratio3 = {}, {}
        for lab in GROUPS:
            m = g == lab
            slices[lab] = (m, p3[lab][m])
            ytr = y_econ[m]
            ratio3[lab] = (empirical_prior(ytr[np.isin(ytr, c3[lab])], c3[lab])
                           / ratio3_den[lab])

        for a3 in ax:
            code = np.zeros(econ_idx.size, dtype=np.int16)
            for lab in GROUPS:
                m, pg = slices[lab]
                if not m.any():
                    continue
                code[m] = c3[lab][corrected(pg, ratio3[lab], a3).argmax(1)]
            final = np.zeros(y.size, dtype=np.int16)
            final[econ_idx] = code

            p_flat = to_flat_pred(s1[held], final[held])
            yf = np.where(y_flat_held == FOREST, OTHERS, y_flat_held)
            pf = np.where(p_flat == FOREST, OTHERS, p_flat)
            cm = _matrix(yf, pf, lab15, names15)
            pc = per_class_from_flat(cm).set_index("class")

            v = cm.values.astype(np.float64)
            acc = np.diag(v).sum() / v.sum()
            pe = (v.sum(0) * v.sum(1)).sum() / v.sum() ** 2
            kappa = (acc - pe) / (1 - pe)

            # Ranking metric: macro over the 13 crops on true-econ pixels only, the
            # definition the map fixed (baseline 0.1678).
            # Label 0 ("no code") must be in the label set: a true-econ pixel the
            # cascade dropped at Stage 1, or routed to other_econ, predicts 0. Without
            # that column those pixels leave the matrix entirely and recall is
            # computed against a shrunken denominator -- which inflated it to 0.3483
            # against the reference 0.2360.
            hm = np.isin(y[held], econ_codes)
            ce = _matrix(y[held][hm], final[held][hm], econ_codes + [0],
                         [CROPS[c] for c in econ_codes] + ["(no code)"])
            pe13 = per_class_from_flat(ce).iloc[:len(econ_codes)]
            crop_f1 = pc.loc[[CROPS[c] for c in econ_codes], "f1"]

            rows.append({
                "alpha2": a2, "alpha3": a3,
                "econ13_macro_f1": round(pe13.f1.mean(), 4),          # ranking
                "classes_alive": int((pe13.f1 >= ALIVE_EPS).sum()),   # co-primary
                "crop_macro_f1_flat": round(crop_f1.mean(), 4),       # stricter view
                "flat15_macro_f1": round(pc.f1.mean(), 4),
                "accuracy": round(acc, 4), "kappa": round(kappa, 4),
                "econ13_macro_precision": round(pe13.precision.mean(), 4),
                "econ13_macro_recall": round(pe13.recall.mean(), 4),
            })

            if a2 == a3 and a2 in LANDMARKS:
                keep_mats[f"a2-{a2}_a3-{a3}"] = (final, g)
        log(f"  alpha2={a2} done ({len(rows)} cells)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{SWEEP_DIR}/sweep_results.csv", index=False, encoding="utf-8-sig")
    log("saved sweep_results.csv")

    best = df.loc[df.econ13_macro_f1.idxmax()]
    full = df[df.classes_alive == len(econ_codes)]
    best_alive = full.loc[full.econ13_macro_f1.idxmax()] if len(full) else None
    log(f"best macro F1      : a2={best.alpha2} a3={best.alpha3} "
        f"f1={best.econ13_macro_f1} alive={best.classes_alive}")
    if best_alive is not None:
        log(f"best all-13-alive  : a2={best_alive.alpha2} a3={best_alive.alpha3} "
            f"f1={best_alive.econ13_macro_f1}")

    # matrices for the landmark diagonal plus both winners (spec: landmarks + winner)
    cells = [(a, a) for a in LANDMARKS]
    cells.append((best.alpha2, best.alpha3))
    if best_alive is not None:
        cells.append((best_alive.alpha2, best_alive.alpha3))
    write_cells(sorted(set(cells)))
    log(f"wrote confusion matrices for {len(set(cells))} cells")
