"""paired_bootstrap_m0.py

The paired parcel bootstrap behind §2's "the repairs cost nothing", saved as an
artifact with a fixed seed so the number in the report is the number on disk
(F4 of docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md).

M0 against the parcel-disjoint baseline. Both were scored on the SAME fold 2 --
same 5,500,269 rows, same parcels -- so this is a legitimate paired comparison,
unlike 0.2248 against the leaky v2's 0.2245, which are different populations.

Parcels are the resampling unit, not pixels: pixels inside a parcel are near
duplicates, so a pixel bootstrap would report an interval far too tight.

Writes runs/s2_2018_3date_parcel_m0/m0_bootstrap.csv.
"""
import csv

import numpy as np

CROPS = np.array([2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405,
                  2407, 2413, 2416, 2419, 2420])
REPS = 400
SEED = 20260826
A_DIR = "./runs/s2_2018_3date_parcel"       # baseline
B_DIR = "./runs/s2_2018_3date_parcel_m0"    # the protocol repairs, nothing else
OUT = f"{B_DIR}/m0_bootstrap.csv"

rng = np.random.default_rng(SEED)
y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
            allow_pickle=True)["y"].astype(np.int32)
asg = np.load("./splits/split_assign.npy")
par = np.load("./splits/parcel_id_row.npy")
te = np.flatnonzero(asg == 2)

yt = np.where(np.isin(y[te], CROPS), y[te], 0)
pa = np.load(f"{A_DIR}/pred_hard.npy")[te]
pb = np.load(f"{B_DIR}/pred_hard.npy")[te]

uniq, pidx = np.unique(par[te], return_inverse=True)
P = uniq.size
print(f"test fold: {te.size:,} rows in {P:,} parcels", flush=True)


def per_parcel(pred):
    """(3, 13, P) counts of tp/fp/fn per class per parcel."""
    out = np.zeros((3, CROPS.size, P), dtype=np.int64)
    for k, c in enumerate(CROPS):
        tc, pc = yt == c, pred == c
        out[0, k] = np.bincount(pidx, weights=tc & pc, minlength=P)
        out[1, k] = np.bincount(pidx, weights=~tc & pc, minlength=P)
        out[2, k] = np.bincount(pidx, weights=tc & ~pc, minlength=P)
    return out


def macro(counts):
    tp, fp, fn = counts
    p = np.divide(tp, tp + fp, out=np.zeros(tp.shape, float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros(tp.shape, float), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(tp.shape, float), where=(p + r) > 0)
    return f.mean(0) if f.ndim > 1 else f.mean()


A, B = per_parcel(pa), per_parcel(pb)
pt_a, pt_b = macro(A.sum(2)), macro(B.sum(2))
print(f"point estimates: baseline {pt_a:.4f}  M0 {pt_b:.4f}  delta {pt_b - pt_a:+.4f}",
      flush=True)

da, db = np.empty(REPS), np.empty(REPS)
for i in range(REPS):
    s = rng.integers(0, P, P)
    da[i] = macro(A[:, :, s].sum(2))
    db[i] = macro(B[:, :, s].sum(2))
d = db - da

rows = [
    {"quantity": "baseline macro F1", "point": round(float(pt_a), 4),
     "mean": round(float(da.mean()), 4), "ci_lo": round(float(np.percentile(da, 2.5)), 4),
     "ci_hi": round(float(np.percentile(da, 97.5)), 4)},
    {"quantity": "M0 macro F1", "point": round(float(pt_b), 4),
     "mean": round(float(db.mean()), 4), "ci_lo": round(float(np.percentile(db, 2.5)), 4),
     "ci_hi": round(float(np.percentile(db, 97.5)), 4)},
    {"quantity": "paired delta (M0 - baseline)", "point": round(float(pt_b - pt_a), 5),
     "mean": round(float(d.mean()), 5), "ci_lo": round(float(np.percentile(d, 2.5)), 5),
     "ci_hi": round(float(np.percentile(d, 97.5)), 5)},
    {"quantity": f"P(M0 > baseline), {REPS} reps, seed {SEED}",
     "point": round(float((d > 0).mean()), 4), "mean": "", "ci_lo": "", "ci_hi": ""},
    {"quantity": f"resampling unit: parcels ({P:,} of them), rows {te.size:,}",
     "point": "", "mean": "", "ci_lo": "", "ci_hi": ""},
]
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    print(f"  {r['quantity']:<46} {r['point']}  [{r['ci_lo']}, {r['ci_hi']}]", flush=True)
print("verdict:", "delta EXCLUDES zero" if np.percentile(d, 2.5) > 0
      else "delta INCLUDES zero -- no detectable difference", flush=True)
print("wrote", OUT, flush=True)
