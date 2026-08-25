"""Paired parcel bootstrap: M5 (features+capacity+operating point) vs M0.

Both were scored on the SAME fold 2 -- same 5,500,269 rows, same 5,824 parcels --
so this is a legitimate paired comparison, unlike 0.2248 against the leaky v2's
0.2245 (different populations entirely).

Parcels are the resampling unit, not pixels: pixels inside a parcel are
near-duplicates, so a pixel bootstrap would report a interval far too tight.
"""
import numpy as np

CROPS = np.array([2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405,
                  2407, 2413, 2416, 2419, 2420])
REPS = 400
rng = np.random.default_rng(20260825)

y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
            allow_pickle=True)["y"].astype(np.int32)
asg = np.load("splits/split_assign.npy")
par = np.load("splits/parcel_id_row.npy")
te = np.flatnonzero(asg == 2)

yt = np.where(np.isin(y[te], CROPS), y[te], 0)
pa = np.load("runs/s2_2018_3date_parcel_m0/pred_hard.npy")[te]
pb = np.load("runs/s2_2018_3date_parcel_m5/pred_hard.npy")[te]

uniq, pidx = np.unique(par[te], return_inverse=True)
P = uniq.size
print(f"test fold: {te.size:,} rows in {P:,} parcels")


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
    """Macro F1 over the 13 crops from summed tp/fp/fn."""
    tp, fp, fn = counts
    p = np.divide(tp, tp + fp, out=np.zeros(tp.shape, float), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros(tp.shape, float), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros(tp.shape, float), where=(p + r) > 0)
    return f.mean(0) if f.ndim > 1 else f.mean()


A, B = per_parcel(pa), per_parcel(pb)
print(f"point estimates: baseline {macro(A.sum(2)):.4f}   M0 {macro(B.sum(2)):.4f}"
      f"   delta {macro(B.sum(2)) - macro(A.sum(2)):+.4f}")

da = np.empty(REPS)
db = np.empty(REPS)
for i in range(REPS):
    s = rng.integers(0, P, P)
    da[i] = macro(A[:, :, s].sum(2))
    db[i] = macro(B[:, :, s].sum(2))
d = db - da

print(f"\n{REPS} parcel-resampled replicates")
print(f"  baseline  {da.mean():.4f}  95% CI [{np.percentile(da, 2.5):.4f}, {np.percentile(da, 97.5):.4f}]")
print(f"  M0        {db.mean():.4f}  95% CI [{np.percentile(db, 2.5):.4f}, {np.percentile(db, 97.5):.4f}]")
print(f"  PAIRED delta {d.mean():+.4f}  95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
print(f"  P(M0 > baseline) = {(d > 0).mean():.3f}")
print("\n  verdict:", "delta EXCLUDES zero -- M0 is better" if np.percentile(d, 2.5) > 0
      else ("delta EXCLUDES zero -- M0 is worse" if np.percentile(d, 97.5) < 0
            else "delta INCLUDES zero -- no detectable difference"))
