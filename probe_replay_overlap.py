"""probe_replay_overlap.py

The artifact behind §5's withdrawn claim (F4 of the 2026-08-26 plan).

`probe_dry_season.py` splits its sample with a random pixel permutation. This
replays that exact split -- same seed, same per-crop draw, same permutation, no
model fitted -- and attaches parcel IDs to both sides, so the table says how much
of the probe's "test" ground the probe had already trained on.

Writes runs/probe_dry_season/probe_replay_overlap.csv.
"""
import csv

import numpy as np

SEED = 42
N_PER_CROP = 8_000
OUT = "./runs/probe_dry_season/probe_replay_overlap.csv"
CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
         2303: "OilPalm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}

rng = np.random.default_rng(SEED)
y = np.load("./aligned_features/svm_s2_3date_features_labels.npz",
            allow_pickle=True)["y"].astype(np.int32)
parcels = np.load("./splits/parcel_id_row.npy")

# --- verbatim from probe_dry_season.py: the per-crop draw, then the permutation
take = []
for code in CROPS:
    idx = np.flatnonzero(y == code)
    k = min(N_PER_CROP, idx.size)
    take.append(rng.choice(idx, size=k, replace=False))
rows = np.sort(np.concatenate(take))
perm = rng.permutation(rows.size)
cut = rows.size // 2
tr, te = perm[:cut], perm[cut:]
print(f"probe sample {rows.size:,} pixels; train {tr.size:,} test {te.size:,}", flush=True)

ys = y[rows]
ptr, pte = parcels[rows[tr]], parcels[rows[te]]

out = []
for code, name in CROPS.items():
    a = np.unique(ptr[ys[tr] == code])
    b = np.unique(pte[ys[te] == code])
    shared = np.intersect1d(a, b).size
    out.append({"crop": name, "lu_code": code,
                "train_parcels": int(a.size), "test_parcels": int(b.size),
                "shared_parcels": int(shared),
                "share_of_test_also_in_train":
                    round(shared / b.size, 4) if b.size else 0.0,
                "test_pixels": int((ys[te] == code).sum())})

with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

for r in sorted(out, key=lambda r: -r["share_of_test_also_in_train"]):
    print(f"  {r['crop']:<12}{r['train_parcels']:>6} train {r['test_parcels']:>6} test "
          f"{r['shared_parcels']:>6} shared   {r['share_of_test_also_in_train']:.1%}",
          flush=True)
print("wrote", OUT, flush=True)
