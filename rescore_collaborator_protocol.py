"""rescore_collaborator_protocol.py

W1 of PLAN.md: re-score an already-trained arm over a population built the way the
collaborator's XGBoost study builds its evaluation set, so the two sets of figures
can be read side by side.

WHAT THIS IS NOT
----------------
This is **not** a head-to-head comparison, and must not be labelled one.

  * Assigning a fresh 60/20/20 split to an already-trained prediction array does
    not recreate a train/test split: rows landing in the nominal "CV" partition
    include pixels this SVM was fitted on. No partitioning is therefore applied
    here at all -- the whole sampled population is scored, and the number is
    descriptive.
  * Their sampling uses an unseeded `np.random.choice` (`extract_crops.py`), so
    their exact sample cannot be reproduced. This script seeds for our own
    reproducibility, which is a deliberate divergence.

WHAT IT IS
----------
A prevalence- and metric-normalised rescore. The collaborator caps each class at
200,000 pixels, which leaves rubber at 200,000 against Langsat's ~1,800 (roughly
100:1) instead of the natural tile's ~6,700:1. Their metrics are computed over
that capped population; ours over the natural one. Since rare-class precision is
largely a function of prevalence, this script shows how much of the apparent gap
between the two studies is population rather than learner.

Their taxonomy is the 13 crops plus a single `others` absorbing everything else,
and their labels are 3-px eroded -- which our buffered label raster already is,
so sampling NPZ rows is equivalent to sampling their eroded masks.

Outputs (in runs/<ARM>/):
  collaborator_protocol_rescore.csv          per-class, both populations
  collaborator_protocol_rescore_summary.csv  macro/weighted summary
"""
import csv

import numpy as np
from sklearn.metrics import classification_report

from config import NPZ, OUT_DIR, E2E_PRED

SAMPLES_PER_CLASS = 200_000     # extract_crops.py: samples_per_class
OTHERS = 9999                   # extract_crops.py: the 14th class id
SEED = 42

CROPS = {2101: "rice", 2204: "cassava", 2205: "pineapple", 2302: "rubber",
         2303: "oil_palm", 2403: "durian", 2404: "rambutan", 2405: "coconut",
         2407: "mango", 2413: "longan", 2416: "jackfruit", 2419: "mangosteen",
         2420: "longkong"}
CODES = sorted(CROPS)
LABELS = CODES + [OTHERS]
NAMES = [CROPS[c] for c in CODES] + ["others"]


def flatten(y, pred):
    """Collapse to their 14-class taxonomy: the 13 crops, everything else `others`."""
    yt = np.where(np.isin(y, CODES), y, OTHERS).astype(np.int32)
    yp = np.where(np.isin(pred, CODES), pred, OTHERS).astype(np.int32)
    return yt, yp


def score(yt, yp, label):
    rep = classification_report(yt, yp, labels=LABELS, target_names=NAMES,
                                output_dict=True, zero_division=0)
    print(f"\n=== {label}: {yt.size:,} rows")
    print(f"    macro F1 {rep['macro avg']['f1-score']:.4f}    "
          f"weighted F1 {rep['weighted avg']['f1-score']:.4f}    "
          f"accuracy {rep['accuracy']:.4f}")
    return rep


if __name__ == "__main__":
    y = np.load(NPZ, allow_pickle=True)["y"]
    pred = np.load(E2E_PRED)
    assert y.size == pred.size
    yt_all, yp_all = flatten(y, pred)

    # --- population 1: the natural tile, what this study has been quoting ------
    rep_nat = score(yt_all, yp_all, "natural tile population")

    # --- population 2: capped at 200k per class, their protocol ---------------
    rng = np.random.default_rng(SEED)
    picks = []
    print(f"\ncapped sample (<= {SAMPLES_PER_CLASS:,} per class):")
    for c, name in zip(LABELS, NAMES):
        idx = np.flatnonzero(yt_all == c)
        take = idx if idx.size <= SAMPLES_PER_CLASS else rng.choice(
            idx, SAMPLES_PER_CLASS, replace=False)
        picks.append(take)
        print(f"  {name:<11} available {idx.size:>10,}  ->  sampled {take.size:>8,}")
    sample = np.concatenate(picks)
    rep_cap = score(yt_all[sample], yp_all[sample], "capped 200k/class population")

    imb_nat = max(np.bincount(yt_all)[LABELS]) / min(np.bincount(yt_all)[LABELS])
    counts_cap = np.array([p.size for p in picks])
    print(f"\nmost:least common class -- natural {imb_nat:,.0f}:1, "
          f"capped {counts_cap.max() / counts_cap.min():,.0f}:1")

    # --- write ----------------------------------------------------------------
    rows = []
    for pop, rep in (("natural", rep_nat), ("capped_200k", rep_cap)):
        for name in NAMES:
            d = rep[name]
            rows.append({"population": pop, "crop": name,
                         "support": int(d["support"]),
                         "precision": round(d["precision"], 4),
                         "recall": round(d["recall"], 4),
                         "f1": round(d["f1-score"], 4)})
    with open(f"{OUT_DIR}/collaborator_protocol_rescore.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = [{"population": pop, "rows": int(rep["macro avg"]["support"]),
                "accuracy": round(rep["accuracy"], 4),
                "macro_f1": round(rep["macro avg"]["f1-score"], 4),
                "weighted_f1": round(rep["weighted avg"]["f1-score"], 4)}
               for pop, rep in (("natural", rep_nat), ("capped_200k", rep_cap))]
    with open(f"{OUT_DIR}/collaborator_protocol_rescore_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"\nwrote {OUT_DIR}/collaborator_protocol_rescore{{,_summary}}.csv")
