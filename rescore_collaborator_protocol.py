"""rescore_collaborator_protocol.py

Re-score one of our parcel-cascade runs under the collaborator's CURRENT protocol,
so the two studies' numbers can be read side by side without pretending they are a
head-to-head.

Pinned to github.com/Gunkartan/geospatial at commit a9a7ca0 (2026-08-25), files
`code/inference/inference.py`, `code/extraction/extract_all.py`,
`code/extraction/extract_crops.py`, `code/training/train_crops.py`. The 2026-08-25
version of this script mirrored their OLD protocol, in which there was no composed
cascade at all. There is one now, and it changes the population their crop numbers
are computed on, so the old rescore is superseded.

WHAT THEIR PROTOCOL ACTUALLY IS, AT THAT COMMIT
-----------------------------------------------
1. `extract_all.py` builds the inference population by RESERVOIR SAMPLING at most
   200,000 pixels per class, seeded `default_rng(42)`, over the whole tile. "Class"
   there means each distinct LU code that starts with 4 (water) or 1 (building),
   each of the 13 crop codes, and a single 9999 bucket for every other valid label
   (`get_sampling_classes`). Unlike `extract_crops.py`, this sampling applies NO
   3-pixel erosion.
2. `inference.py` chains three models. Rows with P(water) >= 0.56 are dropped, then
   rows with P(building) >= 0.56 are dropped from what remains.
3. The 14-class crop XGBoost classifies the SURVIVORS, and the crop
   `classification_report` is computed on the survivors only, with truth mapped to
   9999 for anything outside the 13 crops. Their macro is therefore over 14 labels,
   the 14th being 'others' -- a class their crop model was trained on, capped at
   200,000 like every other (`extract_crops.py`).

WHAT THIS SCRIPT REPRODUCES, AND WHAT IT ONLY APPROXIMATES
-----------------------------------------------------------
Reproduced:
  * the per-LU-code 200,000 reservoir cap, including the single 9999 bucket, and
    the same seed (42). Their reservoir over a raster scan and our uniform draw
    over the same rows are both uniform samples of the same population, so this is
    an equivalence rather than a copy.
  * the 14-label macro including 'others', which is the metric change that moves
    the number most.
  * the survivor-only denominator: the crop report is computed after the filters.
  * the water filter at exactly their 0.56, using our Stage-1 calibrated P(water).

Approximated, and this must be said in the report text:
  * THE BUILDING FILTER. We have no buildings model. Buildings live inside our
    Stage-1 'others' superclass together with roads, bare ground and everything
    else non-crop non-water non-forest, so there is no probability of ours that
    means what theirs means. Two variants are therefore reported: `water_only`,
    which skips the building stage entirely, and `water_plus_others`, which uses
    P(others) >= 0.56 as a stand-in. The truth is bracketed by the two -- the
    stand-in drops strictly more non-crop rows than a real building filter would,
    so it flatters us, and `water_only` drops strictly fewer, so it does not.
  * THE POPULATION. Theirs is the whole tile. Ours is fold 2 only, because that is
    the only ground our cascade has a legitimate prediction for. This makes our
    denominator a parcel-disjoint held-out third and theirs a sample that overlaps
    the pixels their models were fitted on (see below).
  * erosion. `extract_all.py` does not erode; our label raster already is 3-px
    eroded, which is closer to their TRAINING sample than to their inference one.

THEIR SAMPLE IS DRAWN FROM GROUND THEIR MODELS TRAINED ON
----------------------------------------------------------
`extract_crops.py` samples up to 200,000 eroded pixels per class for training and
`train_crops.py` splits THAT csv 60/20/20 by row. `extract_all.py` then re-samples
up to 200,000 pixels per class from the same tile for inference. Nothing separates
the two draws spatially. For any class with fewer than 200,000 eroded pixels --
which is every crop except rubber, and by a wide margin for the rare orchards --
the training draw is essentially the whole class, so the inference draw lands on
pixels the model was fitted on. The overlap is concentrated exactly in the rare
classes a macro average rewards most.

Env:
  RUN_DIR=<dir>   parcel-cascade run to rescore (needs pred_hard.npy and
                  stage1_prob_test.npy)
"""
import csv
import json
import os

import numpy as np
from sklearn.metrics import classification_report

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m5")
SPLIT_ASSIGN = "./splits/split_assign.npy"
SAMPLES_PER_CLASS = 200_000     # extract_all.py: sample_size
OTHERS = 9999                   # extract_all.py: the catch-all sampling class
THRESH = 0.56                   # inference.py: both filter thresholds
SEED = 42                       # extract_all.py: default_rng(42)

CROPS = {2101: "rice", 2204: "cassava", 2205: "pineapple", 2302: "rubber",
         2303: "oil_palm", 2403: "durian", 2404: "rambutan", 2405: "coconut",
         2407: "mango", 2413: "longan", 2416: "jackfruit", 2419: "mangosteen",
         2420: "longkong"}
CODES = sorted(CROPS)
LABELS = CODES + [OTHERS]
NAMES = [CROPS[c] for c in CODES] + ["others"]


def log(*a):
    print(*a, flush=True)


def sampling_class(y):
    """extract_all.get_sampling_classes: keep water/building/crop LU codes as
    themselves, bucket every other valid label into 9999, drop label 0."""
    s = np.where(np.isin(y, CODES), y, 0).astype(np.int64)
    txt = y.astype(str)
    keep = np.char.startswith(txt, "4") | np.char.startswith(txt, "1")
    s = np.where((s == 0) & keep, y, s)
    s = np.where((s == 0) & (y != 0), OTHERS, s)
    return s


def report(y_true, y_pred, label):
    yt = np.where(np.isin(y_true, CODES), y_true, OTHERS).astype(np.int32)
    yp = np.where(np.isin(y_pred, CODES), y_pred, OTHERS).astype(np.int32)
    rep = classification_report(yt, yp, labels=LABELS, target_names=NAMES,
                                output_dict=True, zero_division=0)
    log(f"\n=== {label}: {yt.size:,} rows")
    log(f"    macro F1 (14 labels, incl. others) {rep['macro avg']['f1-score']:.4f}"
        f"    weighted F1 {rep['weighted avg']['f1-score']:.4f}"
        f"    accuracy {rep['accuracy']:.4f}")
    crop_only = np.mean([rep[n]["f1-score"] for n in NAMES[:-1]])
    log(f"    macro F1 over the 13 crops only {crop_only:.4f}")
    return rep, crop_only


if __name__ == "__main__":
    manifest = json.load(open(f"{RUN}/manifest.json"))
    y = np.load(manifest["npz"], allow_pickle=True)["y"].astype(np.int32)
    pred = np.load(f"{RUN}/pred_hard.npy")
    asg = np.load(SPLIT_ASSIGN)
    te = np.flatnonzero(asg == 2)
    assert y.size == pred.size == asg.size
    log(f"rescoring {RUN} under the collaborator protocol at commit a9a7ca0")
    log(f"  population: fold 2 only, {te.size:,} rows")

    # ---- their sampling: <= 200k per sampling class, over our test fold -------
    rng = np.random.default_rng(SEED)
    sc = sampling_class(y[te])
    picks = []
    log(f"\ncapped sample (<= {SAMPLES_PER_CLASS:,} per sampling class):")
    for c in np.unique(sc):
        if c == 0:
            continue
        idx = np.flatnonzero(sc == c)
        take = idx if idx.size <= SAMPLES_PER_CLASS else rng.choice(
            idx, SAMPLES_PER_CLASS, replace=False)
        picks.append(take)
        log(f"  {int(c):<7} available {idx.size:>10,}  ->  sampled {take.size:>8,}")
    sample = te[np.sort(np.concatenate(picks))]
    log(f"  sampled population {sample.size:,} rows")

    # ---- their chain: drop P(water) >= 0.56, then drop P(building) >= 0.56 ----
    m1_classes = [1, 2, 3, 4]           # econ, water, others, forest
    p1 = np.load(f"{RUN}/stage1_prob_test.npy")
    assert p1.shape[0] == te.size, (p1.shape, te.size)
    pos = np.searchsorted(te, sample)
    p_water = p1[pos, m1_classes.index(2)]
    p_others = p1[pos, m1_classes.index(3)]

    variants = {
        "water_only": p_water < THRESH,
        "water_plus_others": (p_water < THRESH) & (p_others < THRESH),
    }

    rows, summary = [], []
    for name, keep in variants.items():
        log(f"\n--- variant {name}: {int(keep.sum()):,} survivors of {keep.size:,}"
            f" ({keep.mean():.1%})")
        rep, crop_only = report(y[sample][keep], pred[sample][keep], name)
        for n in NAMES:
            d = rep[n]
            rows.append({"variant": name, "crop": n, "support": int(d["support"]),
                         "precision": round(d["precision"], 4),
                         "recall": round(d["recall"], 4),
                         "f1": round(d["f1-score"], 4)})
        summary.append({"variant": name, "rows": int(keep.sum()),
                        "survivor_rate": round(float(keep.mean()), 4),
                        "accuracy": round(rep["accuracy"], 4),
                        "macro_f1_14": round(rep["macro avg"]["f1-score"], 4),
                        "macro_f1_13_crops": round(float(crop_only), 4),
                        "weighted_f1": round(rep["weighted avg"]["f1-score"], 4)})

    # the same run under OUR convention, for the reader who wants the bridge
    log("\n--- for reference, our own strict convention on the same fold")
    rep, crop_only = report(y[te], pred[te], "strict: full fold 2, 13-crop macro")
    summary.append({"variant": "ours_strict_full_fold2", "rows": int(te.size),
                    "survivor_rate": 1.0, "accuracy": round(rep["accuracy"], 4),
                    "macro_f1_14": round(rep["macro avg"]["f1-score"], 4),
                    "macro_f1_13_crops": round(float(crop_only), 4),
                    "weighted_f1": round(rep["weighted avg"]["f1-score"], 4)})

    with open(f"{RUN}/collaborator_protocol_rescore.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(f"{RUN}/collaborator_protocol_rescore_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    log(f"\nwrote {RUN}/collaborator_protocol_rescore{{,_summary}}.csv")
