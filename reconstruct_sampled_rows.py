"""reconstruct_sampled_rows.py

Rebuild the exact set of NPZ rows that each stage drew into its sample.

Why: end-to-end scoring runs over every labeled pixel, including pixels the
models were fitted on, which makes the score optimistic. Every sampling step in
Stages 1-3 is driven by a seeded default_rng(RANDOM_STATE) and depends only on
y (and the saved stage predictions) - never on X - so the selection can be
replayed here without retraining anything.

Two masks are produced over all NPZ rows:

  sampled_rows_mask.npy   True = the row entered some stage's sample at all
  trainval_rows_mask.npy  True = the row was in the TRAIN or VAL split of some
                          stage, i.e. a model was actually fitted on it

Score on ~trainval, not ~sampled. Excluding every sampled row biases the residue
towards failure: a stage's per-LU cap (70k) exceeds the number of pixels a rare
class has surviving upstream routing, so *all* of its correctly-routed pixels are
absorbed into the sample and only mis-routed ones remain. That drives recall for
rare classes to exactly 0. Rows that appear only in a stage's held-out test split
are legitimately unfitted and unbiased with respect to correctness.

Each stage splits 70/15/15 with a fixed seed, so the split is replayed here the
same way the sample is - including the shuffles that set row order, since
train_test_split partitions by position.

Usage:
  python reconstruct_sampled_rows.py
"""
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from config import (NPZ, RANDOM_STATE, OUT_DIR, STAGE1_PRED, STAGE2_PRED,
                    SAMPLES_PER_LU, MIN_CLASS_PIXELS,
                    CAP_ECON, CAP_WATER, CAP_FOREST, CAP_OTHERS,
                    MIN_PIXELS_PER_GROUP, PER_GROUP_CAP,
                    MIN_PIXELS_PER_LU, PER_LU_CAP)

economic_crops = {2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420}
water_code = {4101, 4102, 4103, 4201, 4202, 4203}
forest_code = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}
orchards_codes = {2403, 2404, 2407, 2413, 2416, 2419, 2420}
plantation_codes = {2302, 2303, 2405}
field_codes = {2101, 2204, 2205}
SUBCLASS_TO_CODES = {
    1: sorted(orchards_codes),
    2: sorted(plantation_codes),
    3: sorted(field_codes),
    4: sorted(c for c in economic_crops
              if c not in (orchards_codes | plantation_codes | field_codes)),
}

OUT_MASK = f"{OUT_DIR}/sampled_rows_mask.npy"
OUT_TRAINVAL = f"{OUT_DIR}/trainval_rows_mask.npy"


def trainval_positions(strat):
    """Replay the 70/15/15 split; return positions that were train or val."""
    pos = np.arange(strat.size)
    tr, rest = train_test_split(pos, test_size=0.3, stratify=strat, random_state=RANDOM_STATE)
    val, _te = train_test_split(rest, test_size=0.5, stratify=strat[rest], random_state=RANDOM_STATE)
    return np.concatenate([tr, val])


def stage1_rows(y):
    """Replay load_and_sample_per_lu + rebalance_by_superclass."""
    pos_valid = np.flatnonzero((y != 0) & (y != 32767))
    yv = y[pos_valid]

    uniques, counts = np.unique(yv, return_counts=True)
    keep_codes = uniques[counts >= MIN_CLASS_PIXELS]
    keep_mask = np.isin(yv, keep_codes)
    pos_f, yf = pos_valid[keep_mask], yv[keep_mask]

    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for code in keep_codes:
        idxs = np.flatnonzero(yf == code)
        n_take = min(SAMPLES_PER_LU, idxs.size)
        parts.append(rng.choice(idxs, size=n_take, replace=False))
    sample_idx = np.concatenate(parts)          # positions into yf, i.e. rows of Xs
    ys_codes = yf[sample_idx]
    print(f"  stage1 per-LU sample: {sample_idx.size:,} rows")

    ys_super = np.where(np.isin(ys_codes, list(economic_crops)), 1,
                np.where(np.isin(ys_codes, list(water_code)), 2,
                np.where(np.isin(ys_codes, list(forest_code)), 4, 3)))

    # rebalance_by_superclass uses a fresh rng with the same seed
    rng = np.random.default_rng(RANDOM_STATE)

    def cap(idxs, c):
        return idxs if c is None else rng.choice(idxs, size=min(len(idxs), c), replace=False)

    kept = np.concatenate([
        cap(np.flatnonzero(ys_super == 1), CAP_ECON),
        cap(np.flatnonzero(ys_super == 2), CAP_WATER),
        cap(np.flatnonzero(ys_super == 4), CAP_FOREST),
        cap(np.flatnonzero(ys_super == 3), CAP_OTHERS),
    ])
    rng.shuffle(kept)          # matches rebalance_by_superclass; sets row order
    print(f"  stage1 after rebalance: {kept.size:,} rows",
          dict(Counter(ys_super[kept])))
    rows = pos_f[sample_idx[kept]]
    return rows, rows[trainval_positions(ys_super[kept])]


def stage2_rows(y, stage1_pred):
    """Replay Stage-2 selection: econ-predicted AND truly econ, capped per group."""
    rows = np.flatnonzero((stage1_pred == 1) & np.isin(y, list(economic_crops)))
    y_sub = np.where(np.isin(y[rows], list(orchards_codes)), 1,
             np.where(np.isin(y[rows], list(plantation_codes)), 2,
             np.where(np.isin(y[rows], list(field_codes)), 3, 4)))

    uniques, counts = np.unique(y_sub, return_counts=True)
    keep_groups = uniques[counts >= MIN_PIXELS_PER_GROUP]
    sel = np.isin(y_sub, keep_groups)
    rows, y_sub = rows[sel], y_sub[sel]

    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    for c in np.unique(y_sub):
        idxs = np.flatnonzero(y_sub == c)
        parts.append(rng.choice(idxs, size=PER_GROUP_CAP, replace=False)
                     if len(idxs) > PER_GROUP_CAP else idxs)
    kept = np.concatenate(parts)
    kept = kept[rng.permutation(kept.size)]     # matches cap_per_group
    print(f"  stage2 sample: {kept.size:,} rows")
    out = rows[kept]
    return out, out[trainval_positions(y_sub[kept])]


def stage3_rows(y, stage1_pred, stage2_pred):
    """Replay Stage-3 selection per subclass group."""
    econ_mask = (stage1_pred == 1)
    out, tv = [], []
    for subclass_label, lu_list in SUBCLASS_TO_CODES.items():
        rows = np.flatnonzero(econ_mask & (stage2_pred == subclass_label) & np.isin(y, lu_list))
        if rows.size == 0:
            continue
        y_sub = y[rows]
        uniques, counts = np.unique(y_sub, return_counts=True)
        keep_lu = uniques[counts >= MIN_PIXELS_PER_LU]
        if keep_lu.size == 0:
            continue
        sel = np.isin(y_sub, keep_lu)
        rows, y_sub = rows[sel], y_sub[sel]

        rng = np.random.default_rng(RANDOM_STATE)   # fresh per call
        parts = []
        for c in np.unique(y_sub):
            idxs = np.flatnonzero(y_sub == c)
            keep_n = min(len(idxs), PER_LU_CAP)
            parts.append(rng.choice(idxs, size=keep_n, replace=False)
                         if len(idxs) > keep_n else idxs)
        kept = np.concatenate(parts)
        kept = kept[rng.permutation(kept.size)]  # matches rebalance_lu_distribution
        print(f"  stage3 group {subclass_label}: {kept.size:,} rows")
        sel = rows[kept]
        out.append(sel)
        tv.append(sel[trainval_positions(y_sub[kept])])
    empty = np.array([], dtype=np.int64)
    return (np.concatenate(out) if out else empty,
            np.concatenate(tv) if tv else empty)


if __name__ == "__main__":
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    n = y.size
    seen = np.zeros(n, dtype=bool)
    trainval = np.zeros(n, dtype=bool)

    print("Stage 1:")
    r, t = stage1_rows(y)
    seen[r] = True; trainval[t] = True

    if os.path.exists(STAGE1_PRED):
        s1 = np.load(STAGE1_PRED)
        print("Stage 2:")
        r, t = stage2_rows(y, s1)
        seen[r] = True; trainval[t] = True
        if os.path.exists(STAGE2_PRED):
            s2 = np.load(STAGE2_PRED)
            print("Stage 3:")
            r, t = stage3_rows(y, s1, s2)
            seen[r] = True; trainval[t] = True
        else:
            print("Stage 3: stage2 predictions missing — skipped")
    else:
        print("Stages 2/3: stage1 predictions missing — skipped")

    labeled = (y != 0) & (y != 32767)
    np.save(OUT_MASK, seen)
    np.save(OUT_TRAINVAL, trainval)
    print(f"\nRows total             : {n:,}")
    print(f"Rows labeled           : {labeled.sum():,}")
    print(f"Rows sampled at all    : {seen.sum():,}")
    print(f"Rows fitted (train/val): {trainval.sum():,}")
    print(f"Scoreable (not fitted) : {(labeled & ~trainval).sum():,}")
    print("Saved:", OUT_MASK)
    print("Saved:", OUT_TRAINVAL)
