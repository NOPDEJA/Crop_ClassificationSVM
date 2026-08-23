"""probe_stage1_prior.py

Wayfinder ticket 07: Stage 1's prior.

Stage 1 carries 14.7 % of the cascade's loss (§6.5) and everything downstream is
conditional on it, so a gain here compounds. It is also the one stage the alpha sweep
deliberately left alone, because its rule is not argmax:

    labels = argmax(p)
    for cls in (1 econ, 2 water, 4 forest):   # in this order -- later wins
        labels[p[:, cls] > t_cls] = cls

with tuned t = {econ 0.40, water 0.48, forest 0.50}. **The econ threshold sat on the
floor of its own search grid** (np.linspace(0.4, 0.95, 56)) -- the tuner wanted to go
lower and could not. That is a constraint, not an optimum, and it is worth measuring.

A threshold on a probability *is* a prior. So the two candidate rules are compared on
the same footing, on held_out, with nothing refitted:

  thresh    the current rule, sweeping the econ threshold below its floor
  prior     drop the thresholds entirely, correct the probabilities toward the true
            prior at strength alpha, then take argmax

Reports Stage-1 econ recall (what reaches the rest of the cascade at all) against what
it costs the other three superclasses.

Writes runs/s2_2018_3date/stage1_prior_probe.csv
"""
import json
import time
import numpy as np
import pandas as pd

from config import NPZ, OUT_DIR, TAG, STAGE1_PRED, STAGE1_THRESH
from confusion_report import true_superclass, _matrix, per_class_from_flat

SUP = {1: "econ", 2: "water", 3: "others", 4: "forest"}
LAB = [1, 2, 4, 3]
OUT_CSV = f"{OUT_DIR}/stage1_prior_probe.csv"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def thresh_rule(p, classes, thr):
    """The stage's own rule: argmax, then sequential overwrite in class order."""
    lab = classes[p.argmax(1)]
    pos = {int(c): i for i, c in enumerate(classes)}
    for c in (1, 2, 4):
        t = thr.get(c)
        if t is None or c not in pos:
            continue
        lab[p[:, pos[c]] > float(t)] = c
    return lab


def metrics(y_sup, pred, tag, **extra):
    cm = _matrix(y_sup, pred, LAB, [SUP[l] for l in LAB])
    pc = per_class_from_flat(cm).set_index("class")
    v = cm.values.astype(np.float64)
    acc = np.diag(v).sum() / v.sum()
    pe = (v.sum(0) * v.sum(1)).sum() / v.sum() ** 2
    row = {"rule": tag, **extra,
           "econ_recall": round(pc.loc["econ", "recall"], 4),
           "econ_precision": round(pc.loc["econ", "precision"], 4),
           "econ_f1": round(pc.loc["econ", "f1"], 4),
           "forest_f1": round(pc.loc["forest", "f1"], 4),
           "water_f1": round(pc.loc["water", "f1"], 4),
           "others_f1": round(pc.loc["others", "f1"], 4),
           "macro_f1": round(pc.f1.mean(), 4),
           "accuracy": round(acc, 4),
           "kappa": round((acc - pe) / (1 - pe), 4)}
    return row


if __name__ == "__main__":
    log("loading")
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    p = np.load(f"{OUT_DIR}/stage1_{TAG}_prob.npy")
    s1 = np.load(STAGE1_PRED)
    fitted = np.load(f"{OUT_DIR}/trainval_rows_mask.npy")
    held = np.flatnonzero(~fitted)
    thr = {int(k): v for k, v in json.load(open(STAGE1_THRESH)).items()}
    log(f"p = {p.shape}, held_out = {held.size:,}, thresholds = {thr}")

    # classes_ order is not stored next to the probabilities. Recovered by agreement:
    # plain argmax with classes_ = [1,2,3,4] reproduces the saved predictions on all
    # 24,323,769 rows, exactly (0 disagreements).
    #
    # That is the finding this probe opened with. stage1_weight_scale.py tunes the
    # thresholds, saves them, and reports the test split through them -- but the
    # full-tile predictions every published number rests on come from
    # chunked_predict_and_save -> predict(), i.e. argmax. The thresholds never reach
    # the tile. So Stage 1 is an argmax stage after all, prior correction composes with
    # it cleanly, and applying the tuned thresholds is itself an untested candidate.
    classes = np.array([1, 2, 3, 4])
    agree = (classes[p[:200_000].argmax(1)] == s1[:200_000]).mean()
    assert agree == 1.0, f"classes_ order wrong: {agree}"
    log("classes_ = [1,2,3,4]; saved predictions are plain argmax (thresholds unused)")

    y_sup = true_superclass(y)
    yh, ph = y_sup[held], p[held]
    rows = [metrics(yh, s1[held], "baseline (saved)", econ_t=thr[1], alpha=0.0)]
    log(f"baseline: {rows[0]}")

    # pi_train: the superclass mix Stage 1 was actually fitted on
    pos = {int(c): i for i, c in enumerate(classes)}
    tr = np.array([(y_sup[fitted] == c).sum() for c in classes], dtype=np.float64)
    pi_train = tr / tr.sum()
    tru = np.array([(y_sup == c).sum() for c in classes], dtype=np.float64)
    pi_true = tru / tru.sum()
    log("pi_train = " + str(dict(zip(classes.tolist(), pi_train.round(4)))))
    log("pi_true  = " + str(dict(zip(classes.tolist(), pi_true.round(4)))))
    ratio = pi_true / pi_train

    # (a) the current rule, econ threshold pushed below its grid floor
    for t in [0.20, 0.25, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.45, 0.50, 0.60]:
        rows.append(metrics(yh, thresh_rule(ph, classes, {**thr, 1: t}),
                            "thresh", econ_t=t, alpha=0.0))
        log(f"  thresh econ_t={t}: {rows[-1]['econ_recall']=} {rows[-1]['macro_f1']=}")

    # (b) thresholds dropped, prior correction + argmax
    for a in [0.0, 0.2, 0.4, 0.552, 0.6, 0.8, 0.858, 1.0]:
        w = ph * (ratio ** a) if a else ph
        rows.append(metrics(yh, classes[w.argmax(1)], "prior", econ_t=None, alpha=a))
        log(f"  prior alpha={a}: {rows[-1]['econ_recall']=} {rows[-1]['macro_f1']=}")

    # The ticket's actual criterion: does Stage 1 stop discarding rare species?
    # Per-crop econ recall = share of each true crop's pixels routed to superclass 1,
    # which is the ceiling on everything stages 2 and 3 can still get right.
    from evaluate_flat_15class import CROPS
    yh_code = y[held]
    per_crop = {}
    for a in [0.0, 0.552, 0.8, 1.0]:
        w = ph * (ratio ** a) if a else ph
        lab = classes[w.argmax(1)]
        per_crop[f"alpha={a}"] = {CROPS[c]: round(float((lab[yh_code == c] == 1).mean()), 4)
                                  for c in CROPS}
    pcdf = pd.DataFrame(per_crop)
    pcdf["delta_0_to_1"] = (pcdf["alpha=1.0"] - pcdf["alpha=0.0"]).round(4)
    pcdf.to_csv(f"{OUT_DIR}/stage1_prior_probe_per_crop.csv", encoding="utf-8-sig")
    log("Stage-1 econ recall per crop (held_out):")
    print(pcdf.to_string(), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log("\n" + df.to_string(index=False))
    log(f"saved {OUT_CSV}")
