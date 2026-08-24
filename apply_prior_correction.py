"""apply_prior_correction.py

Tests option 1 of docs/S2_SVM_ANALYSIS.md 6.4 / 6.5: every stage trains on rebalanced
data and is then applied to a tile with the real class priors, so the cascade
redistributes rubber across the twelve rarer crops.  Post-hoc prior correction
undoes that at inference time, with no retraining:

    p_corrected(c|x)  proportional to  p(c|x) * pi_target(c) / pi_train(c)

pi_train comes from the counts each stage actually fitted on (Stage 2 is exactly
uniform; Stage 3's are in its meta JSON, after upsampling).  pi_target is
estimated two ways:

  em      Saerens-Latinne-Decaestecker EM on the probabilities alone.  Uses no
          labels, so it is what a deployed pipeline could actually do.
  oracle  the true class distribution of the routed pixels.  Not deployable;
          it bounds what prior correction can buy.

Stage 1 is left untouched, so routing into the cascade is identical to the
baseline and the comparison isolates the Stage-2/3 effect.

Requires: save_stage23_probs.py, save_stage3_probs_all_econ.py,
          reconstruct_sampled_rows.py

Outputs (in runs/<RUN>/):
  prior_correction_overall.csv    flat-15 metrics, baseline vs em vs oracle
  prior_correction_per_class.csv  per-class P/R/F1 for each variant
  prior_correction_econ.csv       end-to-end econ P/R/F1 per variant
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from config import NPZ, RANDOM_STATE, OUT_DIR, TAG, STAGE1_PRED, STAGE3_META_TPL
from evaluate_flat_15class import (to_flat_true, to_flat_pred, score,
                                   matched_subset, CROPS)

GROUPS = {1: "orchards", 2: "plantation", 3: "field"}
GROUP_CODES = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
               2: [2302, 2303, 2405],
               3: [2101, 2204, 2205]}
EM_ITERS = 50
EM_TOL = 1e-6


def em_priors(p, pi_train):
    """Saerens-Latinne-Decaestecker: estimate test priors from probabilities."""
    pi = pi_train.copy()
    for _ in range(EM_ITERS):
        w = p * (pi / pi_train)
        w /= w.sum(axis=1, keepdims=True)
        pi_new = w.mean(axis=0)
        done = np.abs(pi_new - pi).max() < EM_TOL
        pi = pi_new
        if done:
            break
    return pi


def correct(p, pi_train, pi_target):
    w = p * (pi_target / pi_train)
    return w / w.sum(axis=1, keepdims=True)


def true_prior(y_sub, classes):
    """Empirical prior of `classes` among pixels whose true label is one of them."""
    cnt = np.array([(y_sub == c).sum() for c in classes], dtype=np.float64)
    if cnt.sum() == 0:
        return np.full(len(classes), 1.0 / len(classes))
    return cnt / cnt.sum()


if __name__ == "__main__":
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    s1 = np.load(STAGE1_PRED)
    econ_idx = np.load(f"{OUT_DIR}/stage2_{TAG}_full_idx.npy")
    p2 = np.load(f"{OUT_DIR}/stage2_{TAG}_full_prob.npy")
    s2_classes = np.load(f"{OUT_DIR}/stage2_{TAG}_full_classes.npy")
    y_econ = y[econ_idx]

    # Stage 2 fitted on exactly 140,000 pixels per subclass -> uniform prior.
    pi2_train = np.full(len(s2_classes), 1.0 / len(s2_classes))

    p3, c3, pi3_train = {}, {}, {}
    for lab, grp in GROUPS.items():
        p3[lab] = np.load(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_prob.npy")
        c3[lab] = np.load(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_classes.npy")
        meta = json.load(open(STAGE3_META_TPL.format(grp=grp), encoding="utf-8"))
        tf = meta["counts"]["train_final"]
        cnt = np.array([tf[str(c)] for c in c3[lab]], dtype=np.float64)
        pi3_train[lab] = cnt / cnt.sum()

    # -- target priors ------------------------------------------------------
    sub_of = np.zeros(y_econ.shape, dtype=np.int32)
    for lab, codes in GROUP_CODES.items():
        sub_of[np.isin(y_econ, codes)] = lab

    targets = {
        "em": {"s2": em_priors(p2, pi2_train),
               **{lab: em_priors(p3[lab], pi3_train[lab]) for lab in GROUPS}},
        "oracle": {"s2": true_prior(sub_of[sub_of > 0], s2_classes),
                   **{lab: true_prior(y_econ[np.isin(y_econ, c3[lab])], c3[lab])
                      for lab in GROUPS}},
    }

    print("Stage-2 priors  train :", np.round(pi2_train, 4))
    for m in targets:
        print(f"                {m:>6}:", np.round(targets[m]['s2'], 4))
    for lab, grp in GROUPS.items():
        print(f"Stage-3 {grp} classes {list(c3[lab])}")
        print("      train :", np.round(pi3_train[lab], 4))
        for m in targets:
            print(f"      {m:>6}:", np.round(targets[m][lab], 4))

    # -- compose each variant -----------------------------------------------
    def compose(mode):
        """mode: baseline | em | oracle | em_s2only | oracle_s2only.

        The *_s2only variants correct Stage 2's uniform prior but leave Stage 3
        at argmax, because EM estimates the Stage-2 prior well and the Stage-3
        ones badly (see the printed priors)."""
        base = mode.replace("_s2only", "")
        if mode == "baseline":
            g = s2_classes[p2.argmax(1)]
        else:
            g = s2_classes[correct(p2, pi2_train, targets[base]["s2"]).argmax(1)]
        code = np.zeros(econ_idx.size, dtype=np.int16)
        for lab in GROUPS:
            m = g == lab
            if not m.any():
                continue
            pg = p3[lab][m]
            if mode in ("em", "oracle"):
                pg = correct(pg, pi3_train[lab], targets[base][lab])
            code[m] = c3[lab][pg.argmax(1)]
        final = np.zeros(y.size, dtype=np.int16)
        final[econ_idx] = code
        return final

    fitted = np.load(f"{OUT_DIR}/trainval_rows_mask.npy")
    held = np.flatnonzero(~fitted)
    y_flat = to_flat_true(y)
    rng = np.random.default_rng(RANDOM_STATE)
    pops = {"all": np.arange(y.size), "held_out": held}
    pops["matched"] = held[matched_subset(y_flat[held], rng)]

    per_class, overall, econ_rows = [], [], []
    econ_codes = list(CROPS)
    for mode in ("baseline", "em_s2only", "oracle_s2only", "em", "oracle"):
        final = compose(mode)
        p_flat = to_flat_pred(s1, final)
        for tag, idx in pops.items():
            rows, ov = score(y_flat[idx], p_flat[idx], fold_forest=True,
                             tag=f"{mode}/{tag}")
            per_class += rows
            overall.append(ov)

        # Whole held-out population, non-crop truth mapped to 0 -- NOT masked to
        # crop-truth rows. Masking first hides every crop prediction made on a
        # non-crop pixel, so those false positives never reach precision and the
        # macro F1 comes out roughly 0.034 too high. Same convention as
        # train_parcel_cascade.py and evaluate_end_to_end.py:97.
        yt = np.where(np.isin(y[held], econ_codes), y[held], 0)
        rep = classification_report(yt, final[held], labels=econ_codes,
                                    output_dict=True, zero_division=0)
        for key, name in [(str(c), CROPS[c]) for c in econ_codes] + [("macro avg", "MACRO")]:
            r = rep[key]
            econ_rows.append({"variant": mode, "crop": name,
                              "precision": round(r["precision"], 4),
                              "recall": round(r["recall"], 4),
                              "f1": round(r["f1-score"], 4),
                              "support": int(r["support"])})
        print(f"\n=== {mode} ===")
        print(pd.DataFrame([o for o in overall
                            if o["population"].startswith(mode + "/")]).to_string(index=False))
        print(pd.DataFrame([r for r in econ_rows
                            if r["variant"] == mode]).to_string(index=False))

    pd.DataFrame(overall).to_csv(f"{OUT_DIR}/prior_correction_overall.csv",
                                 index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(f"{OUT_DIR}/prior_correction_per_class.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(econ_rows).to_csv(f"{OUT_DIR}/prior_correction_econ.csv",
                                   index=False, encoding="utf-8-sig")
    print("\nSaved prior_correction_{overall,per_class,econ}.csv in", OUT_DIR)
