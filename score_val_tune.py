"""score_val_tune.py

Compose and score a cascade on fold 1's TUNING HALF, from saved *_prob_val.npy.

Fold 2 gets one read, on a predeclared configuration (M5). Everything explored
afterwards -- and the tree merge is a new hypothesis generated from M0's own
diagnostics -- has to be judged somewhere else. This is that somewhere.

Comparable references on this exact population, from m2_operating_point.py:

    M0, alpha (0.0, 0.0)   macro F1 0.2067   7 crops >= 0.01
    M0, alpha (0.2, 0.7)   macro F1 0.2179  10 crops >= 0.01

so any run scored here can be set beside those two numbers directly.

Handles either taxonomy: it infers the groups from which stage3_*_prob_val.npy
files the run wrote, so a MERGE_TREE run (tree/field) and a standard one
(orchards/plantation/field) both work without a flag.

Scoring uses the strict convention -- the whole tuning half, non-crop truth mapped
to 0, labels fixed -- so false positives on non-crop pixels count against
precision. See strict-scoring-convention in the project memory.

Env:
  RUN_DIR=<dir>   run to score
  ALPHA2/ALPHA3   operating point (default 0,0 = plain argmax)
"""
import csv
import os

import numpy as np
from sklearn.metrics import f1_score

RUN = os.environ.get("RUN_DIR", "./runs/s2_2018_3date_parcel_m0")
ALPHA2 = float(os.environ.get("ALPHA2", 0.0))
ALPHA3 = float(os.environ.get("ALPHA3", 0.0))
NPZ = "./aligned_features/svm_s2_3date_features_labels.npz"   # y only; same rows
SINK = 4
STD = {1: ("orchards", [2403, 2404, 2407, 2413, 2416, 2419, 2420]),
       2: ("plantation", [2302, 2303, 2405]),
       3: ("field", [2101, 2204, 2205])}
MERGED = {1: ("tree", [2302, 2303, 2403, 2404, 2405, 2407, 2413, 2416, 2419, 2420]),
          3: ("field", [2101, 2204, 2205])}
NM = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
      2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
      2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
      2420: "Langsat"}
CROPS = sorted(NM)


def reweight(P, ratio, alpha):
    if alpha == 0.0:
        return P
    out = P * (ratio ** alpha)
    tot = out.sum(1, keepdims=True)
    tot[tot == 0] = 1.0
    return out / tot


def main():
    groups = MERGED if os.path.exists(f"{RUN}/stage3_tree_prob_val.npy") else STD
    gname = {g: n for g, (n, _) in groups.items()}
    codes = {g: c for g, (_, c) in groups.items()}
    print(f"scoring {RUN} on fold 1's tuning half")
    print(f"  taxonomy: {[(gname[g], len(codes[g])) for g in groups]}  + sink")

    y = np.load(NPZ, allow_pickle=True)["y"].astype(np.int32)
    cal = np.load(f"{RUN}/val_cal_idx.npy")
    tune = np.load(f"{RUN}/val_tune_idx.npy")
    c_va = np.load(f"{RUN}/stage2_val_idx.npy")
    in_cal = np.isin(c_va, cal, assume_unique=True)
    c_va_cal = np.intersect1d(c_va, cal)

    g_of = np.zeros(y.size, dtype=np.int32)
    for g in groups:
        g_of[np.isin(y, codes[g])] = g
    g_of[g_of == 0] = SINK

    classes2 = np.array(sorted(list(groups) + [SINK]))
    P2 = np.load(f"{RUN}/stage2_prob_val.npy")
    assert P2.shape[1] == classes2.size, (P2.shape, classes2)
    pi2 = np.array([(g_of[c_va_cal] == g).mean() for g in classes2])
    ratio2 = ((1.0 / classes2.size) / np.where(pi2 > 0, pi2, 1e-9))[None, :]

    P3, ratio3, cls3 = {}, {}, {}
    for g in groups:
        P3[g] = np.load(f"{RUN}/stage3_{gname[g]}_prob_val.npy")
        cls3[g] = np.array(sorted(codes[g]))
        own = np.intersect1d(c_va_cal, np.flatnonzero(np.isin(y, codes[g])))
        pi = np.array([(y[own] == c).mean() for c in cls3[g]])
        ratio3[g] = ((1.0 / len(codes[g])) / np.where(pi > 0, pi, 1e-9))[None, :]

    g_hat = classes2[reweight(P2, ratio2, ALPHA2).argmax(1)]
    code = np.zeros(c_va.size, dtype=np.int32)
    for g in groups:
        sel = g_hat == g
        if sel.any():
            code[sel] = cls3[g][reweight(P3[g][sel], ratio3[g], ALPHA3).argmax(1)]

    m = ~in_cal
    y_eval = np.where(np.isin(y[tune], CROPS), y[tune], 0)
    pred = np.zeros(tune.size, dtype=np.int32)
    pred[np.searchsorted(tune, c_va[m])] = code[m]

    f = f1_score(y_eval, pred, labels=CROPS, average=None, zero_division=0)
    print(f"\n  alpha ({ALPHA2}, {ALPHA3})   rows {tune.size:,}")
    print(f"  {'crop':<12}{'F1':>9}   support")
    rows = []
    for i, c in enumerate(CROPS):
        n = int((y_eval == c).sum())
        print(f"  {NM[c]:<12}{f[i]:>9.4f}   {n:,}")
        rows.append({"crop": NM[c], "lu_code": c, "f1": round(float(f[i]), 4),
                     "support": n})
    print(f"  {'MACRO':<12}{f.mean():>9.4f}")
    print(f"  {'alive>=0.01':<12}{int((f >= 0.01).sum()):>9}")
    print("\n  reference on this same population: M0 (0,0) 0.2067 / 7 alive,"
          " M0 (0.2,0.7) 0.2179 / 10 alive")

    tag = f"a{ALPHA2}_{ALPHA3}".replace(".", "")
    with open(f"{RUN}/val_tune_score_{tag}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {RUN}/val_tune_score_{tag}.csv")


if __name__ == "__main__":
    main()
