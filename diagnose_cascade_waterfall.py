"""diagnose_cascade_waterfall.py

Where does each crop die?

The Stage-3 experts, measured in isolation on fold 1's tuning half, score Rambutan
0.22, Mango 0.42 and Mangosteen 0.12. End to end on fold 2 the same crops score
0.0000, 0.0356 and 0.0052. The species are separable; something upstream is
removing them. This traces every true crop pixel through the cascade and reports
the recall surviving each stage, so the loss is attributed rather than guessed.

Four columns per crop, all as a fraction of that crop's true test pixels:

  stage1   Stage 1 called it economic (otherwise it leaves the cascade for good)
  stage2   ... AND Stage 2 routed it to the group that actually contains it
  final    ... AND the prediction is the correct crop code
  expert   what the group's expert scores on its own, for comparison

Deliberately loads no feature matrix -- only labels, routes and predictions -- so
it can run beside a training job without competing for memory.

Env:
  RUN_DIR=<dir>   run to diagnose (default ./runs/s2_2018_3date_parcel_m0)
"""
import csv
import os

import numpy as np

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
NPZ = "./aligned_features/svm_s2_3date_features_labels.npz"
SPLIT_ASSIGN = "./splits/split_assign.npy"
GROUPS = {1: [2403, 2404, 2407, 2413, 2416, 2419, 2420],
          2: [2302, 2303, 2405],
          3: [2101, 2204, 2205]}
GNAME = {1: "orchards", 2: "plantation", 3: "field", 4: "sink"}
SINK = 4
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
CROPS = sorted(NM)


def main():
    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    asg = np.load(SPLIT_ASSIGN)
    te = np.flatnonzero(asg == 2)
    del asg

    s1 = np.load(f"{RUN}/stage1_pred.npy")
    pred = np.load(f"{RUN}/pred_hard.npy")
    c_te = np.load(f"{RUN}/stage2_test_idx.npy")
    p2 = np.load(f"{RUN}/stage2_prob_test.npy")
    g_hat_of_cand = np.array([1, 2, 3, SINK])[p2.argmax(1)]
    del p2

    # group Stage 2 assigned, over the whole tile; 0 = never reached Stage 2
    g_hat = np.zeros(y.size, dtype=np.int8)
    g_hat[c_te] = g_hat_of_cand
    del g_hat_of_cand, c_te

    y_te = y[te]
    s1_te = s1[te]
    pred_te = pred[te]
    g_te = g_hat[te]
    del s1, pred, g_hat, y

    group_of = {c: g for g, cs in GROUPS.items() for c in cs}

    rows = []
    print(f"cascade waterfall on fold 2 -- {RUN}")
    print(f"{'crop':<12}{'true px':>10}{'stage1':>9}{'stage2':>9}{'final':>9}"
          f"{'  lost@1':>9}{'lost@2':>9}{'lost@3':>9}")
    for c in CROPS:
        m = y_te == c
        n = int(m.sum())
        if n == 0:
            continue
        surv1 = int((m & (s1_te == 1)).sum())
        surv2 = int((m & (s1_te == 1) & (g_te == group_of[c])).sum())
        final = int((m & (pred_te == c)).sum())
        r1, r2, r3 = surv1 / n, surv2 / n, final / n
        print(f"{NM[c]:<12}{n:>10,}{r1:>9.3f}{r2:>9.3f}{r3:>9.3f}"
              f"{1 - r1:>9.3f}{r1 - r2:>9.3f}{r2 - r3:>9.3f}")
        rows.append({"crop": NM[c], "lu_code": c, "true_px": n,
                     "recall_after_stage1": round(r1, 4),
                     "recall_after_stage2": round(r2, 4),
                     "recall_final": round(r3, 4),
                     "lost_at_stage1": round(1 - r1, 4),
                     "lost_at_stage2": round(r1 - r2, 4),
                     "lost_at_stage3": round(r2 - r3, 4)})

    # where do the misrouted ones go?
    print(f"\nwhere Stage 2 sends each crop (share of pixels that survived Stage 1)")
    print(f"{'crop':<12}" + "".join(f"{GNAME[g]:>12}" for g in (1, 2, 3, SINK)))
    for c in CROPS:
        m = (y_te == c) & (s1_te == 1)
        n = int(m.sum())
        if n == 0:
            continue
        shares = [(m & (g_te == g)).sum() / n for g in (1, 2, 3, SINK)]
        star = "  <-- own group: " + GNAME[group_of[c]]
        print(f"{NM[c]:<12}" + "".join(f"{s:>12.3f}" for s in shares) + star)

    with open(f"{RUN}/cascade_waterfall.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RUN}/cascade_waterfall.csv")


if __name__ == "__main__":
    main()
