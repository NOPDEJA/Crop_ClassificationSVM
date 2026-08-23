"""confusion_report.py

The full confusion-matrix set for one cascade prediction, in one call.

§6.5 only became legible when the confusion was read *backwards* — for each
predicted class, what was actually there. Scalar metrics hid that 83 % of predicted
oil palm is truly rubber. A sweep judged on macro F1 alone would repeat that
mistake, so every sweep point emits these.

Six matrices, each in three views:

  stage1_superclass     4x4   econ / water / forest / others
  stage2_subclass       3x3   orchards / plantation / field
  stage3_<group>        crop x crop, within each Stage-2 group
  flat15                crop x crop, 13 crops + Reservoir + Others (forest folded)
  flat16_forest_split   as flat15, with Forest kept as its own class

  _counts      raw
  _recall      row-normalised   — of the true class, where did it go?
  _precision   column-normalised — of this prediction, what was really there?

The precision view is the one that exposes prior collapse; it is produced for every
matrix, not only flat15, because the Stage-3 groups are where the collapse is worst.

Usage:
    mats = confusion_set(y, s1_pred, s2_pred, final_lu, idx)
    write_confusion_set(mats, OUT_DIR, tag="baseline")
"""
import os
import numpy as np
import pandas as pd

from evaluate_flat_15class import (to_flat_true, to_flat_pred, CROPS, NAMES,
                                   RESERVOIR, OTHERS, FOREST,
                                   water_code, forest_code)

SUPERCLASS = {1: "econ", 2: "water", 3: "others", 4: "forest"}
SUBCLASS = {1: "orchards", 2: "plantation", 3: "field"}
GROUP_CODES = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
               2: [2302, 2303, 2405],
               3: [2101, 2204, 2205]}


def _matrix(y_true, y_pred, labels, names):
    """Confusion counts as a labelled DataFrame. Rows = true, columns = predicted."""
    pos = {int(l): i for i, l in enumerate(labels)}
    k = len(labels)
    keep = np.isin(y_true, labels) & np.isin(y_pred, labels)
    if not keep.any():
        return pd.DataFrame(np.zeros((k, k), dtype=np.int64), index=names, columns=names)
    r = np.array([pos[int(v)] for v in y_true[keep]])
    c = np.array([pos[int(v)] for v in y_pred[keep]])
    cm = np.bincount(r * k + c, minlength=k * k).reshape(k, k)
    return pd.DataFrame(cm, index=names, columns=names)


def true_superclass(y):
    sup = np.full(y.shape, 3, dtype=np.int8)
    sup[np.isin(y, list(CROPS))] = 1
    sup[np.isin(y, list(water_code))] = 2
    sup[np.isin(y, list(forest_code))] = 4
    return sup


def true_subclass(y):
    sub = np.zeros(y.shape, dtype=np.int8)
    for lab, codes in GROUP_CODES.items():
        sub[np.isin(y, codes)] = lab
    return sub


def confusion_set(y, s1_pred, s2_pred, final_lu, idx):
    """Every confusion matrix for one cascade prediction, over rows `idx`.

    y, s1_pred, s2_pred, final_lu are full-length arrays aligned to the NPZ;
    idx selects the population (normally held_out).
    """
    y, s1, s2, fin = y[idx], s1_pred[idx], s2_pred[idx], final_lu[idx]
    mats = {}

    sup_lab = [1, 2, 4, 3]
    mats["stage1_superclass"] = _matrix(true_superclass(y), s1, sup_lab,
                                        [SUPERCLASS[l] for l in sup_lab])

    # Stage 2 scores only pixels it was actually given, and only those whose true
    # class has a subclass at all — pixels Stage 1 wrongly routed here have none.
    routed = (s1 == 1)
    sub_lab = [1, 2, 3]
    mats["stage2_subclass"] = _matrix(true_subclass(y)[routed], s2[routed], sub_lab,
                                      [SUBCLASS[l] for l in sub_lab])

    for lab, grp in SUBCLASS.items():
        m = routed & (s2 == lab)
        codes = sorted(GROUP_CODES[lab])
        mats[f"stage3_{grp}"] = _matrix(y[m], fin[m], codes,
                                        [CROPS[c] for c in codes])

    y_flat, p_flat = to_flat_true(y), to_flat_pred(s1, fin)
    lab16 = list(CROPS) + [RESERVOIR, OTHERS, FOREST]
    mats["flat16_forest_split"] = _matrix(y_flat, p_flat, lab16,
                                          [NAMES[l] for l in lab16])

    yf = np.where(y_flat == FOREST, OTHERS, y_flat)
    pf = np.where(p_flat == FOREST, OTHERS, p_flat)
    lab15 = list(CROPS) + [RESERVOIR, OTHERS]
    mats["flat15"] = _matrix(yf, pf, lab15, [NAMES[l] for l in lab15])

    return mats


def write_confusion_set(mats, out_dir, tag):
    """Write counts / recall / precision views. Returns the directory written to."""
    d = os.path.join(out_dir, "confusion")
    os.makedirs(d, exist_ok=True)
    for name, cm in mats.items():
        v = cm.values.astype(np.float64)
        rows = v.sum(axis=1, keepdims=True)
        cols = v.sum(axis=0, keepdims=True)
        views = {
            "counts": cm,
            "recall": pd.DataFrame(np.divide(v, rows, out=np.zeros_like(v), where=rows > 0),
                                   index=cm.index, columns=cm.columns).round(4),
            "precision": pd.DataFrame(np.divide(v, cols, out=np.zeros_like(v), where=cols > 0),
                                      index=cm.index, columns=cm.columns).round(4),
        }
        for view, df in views.items():
            df.to_csv(os.path.join(d, f"{tag}_{name}_{view}.csv"), encoding="utf-8-sig")
    return d


def per_class_from_flat(cm):
    """Precision/recall/F1 per class from a flat confusion matrix.

    Exists so the sweep can derive its scalar metrics from the same matrices the
    human reads, rather than from a separate code path that could disagree.
    """
    v = cm.values.astype(np.float64)
    tp = np.diag(v)
    prec = np.divide(tp, v.sum(0), out=np.zeros_like(tp), where=v.sum(0) > 0)
    rec = np.divide(tp, v.sum(1), out=np.zeros_like(tp), where=v.sum(1) > 0)
    den = prec + rec
    f1 = np.divide(2 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    return pd.DataFrame({"class": cm.index, "precision": prec, "recall": rec,
                         "f1": f1, "support": v.sum(1).astype(np.int64)})
