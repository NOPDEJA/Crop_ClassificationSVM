"""evaluate_flat_15class.py

Re-score the SVM cascade's output as a flat classification problem so it can be
compared against models that are not cascades (the prior Random Forest paper,
and the collaborator's XGBoost).

The cascade's native metrics are per-stage and conditional; they cannot be put
in the same table as a flat model's. Here the cascade's final answer for every
pixel is collapsed into one label set:

    13 economic crop LU codes + reservoir + others        (15 classes)

Forest is folded into "others" to match the comparison taxonomy, but is also
reported on its own row so a genuinely strong result is not hidden.

Three scored populations, because the population changes the number as much as
the model does:

  all       every labeled pixel - comparable to the project's existing
            end_to_end_summary.csv, but optimistic: it includes pixels the
            models were fitted on
  held_out  pixels no stage ever FITTED on (not in any stage's train or val
            split). This is the honest number and the one to publish. Note it is
            deliberately not "never sampled": a stage's per-LU cap absorbs every
            correctly-routed pixel of a rare class, so the never-sampled residue
            contains only mis-routed pixels and scores 0 by construction
  matched   held_out pixels subsampled to the prior RF paper's per-class test
            supports, so class priors match and weighted metrics are comparable

Usage:
  python reconstruct_sampled_rows.py     # first, builds the seen/unseen mask
  python evaluate_flat_15class.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             accuracy_score, f1_score)

from config import NPZ, RANDOM_STATE, OUT_DIR, STAGE1_PRED, STAGE2_PRED, E2E_PRED

TRAINVAL_MASK = f"{OUT_DIR}/trainval_rows_mask.npy"
OUT_PER_CLASS = f"{OUT_DIR}/flat15_per_class.csv"
OUT_OVERALL = f"{OUT_DIR}/flat15_overall.csv"

RESERVOIR, OTHERS, FOREST = 9001, 9002, 9003

CROPS = {
    2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Para rubber",
    2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
    2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
    2420: "Langsat",
}
EXTRA = {RESERVOIR: "Reservoir", OTHERS: "Others", FOREST: "Forest"}
NAMES = {**CROPS, **EXTRA}

water_code = {4101, 4102, 4103, 4201, 4202, 4203}
forest_code = {3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501}

# Prior RF paper's test supports = its Table I pixel counts / 10 (80:10:10 split).
# Taken from Table I rather than Table V: Table V's support column is shifted by
# one row because Rambutan's entry is blank.
RF_SUPPORTS = {
    2101: 15_656, 2204: 38_757, 2205: 38_656, 2302: 38_555, 2303: 38_990,
    2403: 39_312, 2404: 781, 2405: 2_508, 2407: 5_615, 2413: 2_790,
    2416: 4_519, 2419: 2_657, 2420: 196, RESERVOIR: 37_425, OTHERS: 37_531,
}


def to_flat_true(y):
    out = np.full(y.shape, OTHERS, dtype=np.int32)
    out[np.isin(y, list(water_code))] = RESERVOIR
    out[np.isin(y, list(forest_code))] = FOREST
    for code in CROPS:
        out[y == code] = code
    return out


def to_flat_pred(stage1_pred, final_lu):
    """Collapse the cascade's routing + final code into one flat label."""
    out = np.full(stage1_pred.shape, OTHERS, dtype=np.int32)
    out[stage1_pred == 2] = RESERVOIR
    out[stage1_pred == 4] = FOREST
    # econ-routed pixels: use the Stage-3 code; 0 means the cascade produced no
    # code (routed to other_econ, which has no Stage-3 model) -> counts as Others
    econ = (stage1_pred == 1) & (final_lu != 0)
    out[econ] = final_lu[econ]
    return out


def score(y_true, y_pred, fold_forest, tag):
    if fold_forest:
        y_true = np.where(y_true == FOREST, OTHERS, y_true)
        y_pred = np.where(y_pred == FOREST, OTHERS, y_pred)
        labels = list(CROPS) + [RESERVOIR, OTHERS]
    else:
        labels = list(CROPS) + [RESERVOIR, OTHERS, FOREST]

    rep = classification_report(y_true, y_pred, labels=labels,
                                output_dict=True, zero_division=0)
    rows = []
    for lab in labels:
        r = rep[str(lab)]
        rows.append({"population": tag, "class": NAMES[lab],
                     "precision": round(r["precision"], 4),
                     "recall": round(r["recall"], 4),
                     "f1": round(r["f1-score"], 4),
                     "support": int(r["support"])})
    overall = {
        "population": tag,
        "n_pixels": int(y_true.size),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    return rows, overall


def matched_subset(y_flat, rng):
    """Subsample to the RF paper's class priors."""
    take = []
    for lab, n in RF_SUPPORTS.items():
        idxs = np.flatnonzero(y_flat == lab)
        if idxs.size == 0:
            continue
        k = min(n, idxs.size)
        take.append(rng.choice(idxs, size=k, replace=False))
        if idxs.size < n:
            print(f"  matched: {NAMES[lab]} has only {idxs.size:,} of {n:,} needed")
    return np.concatenate(take)


if __name__ == "__main__":
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    s1 = np.load(STAGE1_PRED)
    final = np.load(E2E_PRED)
    assert s1.size == y.size == final.size, "prediction arrays not aligned with NPZ"

    y_flat = to_flat_true(y)
    p_flat = to_flat_pred(s1, final)

    if os.path.exists(TRAINVAL_MASK):
        fitted = np.load(TRAINVAL_MASK)
    else:
        raise SystemExit("Run reconstruct_sampled_rows.py first (train/val mask missing)")

    populations = {
        "all": np.arange(y.size),
        "held_out": np.flatnonzero(~fitted),
    }
    rng = np.random.default_rng(RANDOM_STATE)
    held_idx = populations["held_out"]
    populations["matched"] = held_idx[matched_subset(y_flat[held_idx], rng)]

    per_class, overall = [], []
    for tag, idx in populations.items():
        print(f"\n=== {tag}: {idx.size:,} pixels ===")
        rows, ov = score(y_flat[idx], p_flat[idx], fold_forest=True, tag=tag)
        per_class += rows
        overall.append(ov)
        print(pd.DataFrame(rows).to_string(index=False))
        print(ov)

    # forest as its own row (16-class view), unseen population only
    rows_f, _ = score(y_flat[held_idx], p_flat[held_idx],
                      fold_forest=False, tag="held_out_forest_split")
    per_class += [r for r in rows_f if r["class"] == "Forest"]

    pd.DataFrame(per_class).to_csv(OUT_PER_CLASS, index=False, encoding="utf-8-sig")
    pd.DataFrame(overall).to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    print("\nSaved:", OUT_PER_CLASS)
    print("Saved:", OUT_OVERALL)

    # confusion matrices, held_out only — the population the paper reports
    from confusion_report import confusion_set, write_confusion_set
    s2 = np.load(STAGE2_PRED)
    mats = confusion_set(y, s1, s2, final, held_idx)
    print("Saved confusion matrices to:",
          write_confusion_set(mats, OUT_DIR, tag="baseline"))
