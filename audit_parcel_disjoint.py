"""audit_parcel_disjoint.py

W2 of PLAN.md: score an already-trained arm on the rows that live in parcels no
stage ever fitted (`splits/clean_rows_mask.npy`), alongside the two populations
the study has been quoting.

This is a LIMITED AUDIT, not a causal leakage penalty. The clean rows are what is
left over after a pixel-level sampler happened to miss whole parcels, so their
class composition is arbitrary -- Longan has none at all, Rice eleven. It bounds
the problem; it does not price it. A causal estimate needs matched
pixel-versus-parcel runs with identical train fractions and evaluation
composition, which is post-report work.

Confidence intervals are bootstrapped over PARCELS, not pixels: pixels within a
parcel are near-duplicates, so a pixel bootstrap would report intervals far
narrower than the evidence supports.

Outputs (in runs/<ARM>/):
  parcel_disjoint_audit.csv         per-class metrics on each population
  parcel_disjoint_audit_summary.csv macro/weighted summary + bootstrap CI
"""
import numpy as np
from sklearn.metrics import classification_report, f1_score

from config import NPZ, OUT_DIR, E2E_PRED, TAG

CLEAN_MASK = "./splits/clean_rows_mask.npy"
PARCEL_ID = "./splits/parcel_id_row.npy"
TRAINVAL = f"{OUT_DIR}/trainval_rows_mask.npy"

CROPS = {2101: "Rice", 2204: "Cassava", 2205: "Pineapple", 2302: "Rubber",
         2303: "Oil palm", 2403: "Durian", 2404: "Rambutan", 2405: "Coconut",
         2407: "Mango", 2413: "Longan", 2416: "Jackfruit", 2419: "Mangosteen",
         2420: "Langsat"}
CODES = sorted(CROPS)
N_BOOT = 2000
RNG = np.random.default_rng(42)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=CODES, average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, parcels, n_boot=N_BOOT):
    """Resample whole parcels with replacement; report the 2.5/97.5 percentiles."""
    uniq = np.unique(parcels)
    # index rows by parcel once, so each draw is a concatenation not a scan
    order = np.argsort(parcels, kind="stable")
    sp = parcels[order]
    starts = np.flatnonzero(np.r_[True, sp[1:] != sp[:-1]])
    groups = np.split(order, starts[1:])
    pos = {p: i for i, p in enumerate(sp[starts])}

    stats = np.empty(n_boot)
    for b in range(n_boot):
        draw = RNG.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([groups[pos[p]] for p in draw])
        stats[b] = macro_f1(y_true[idx], y_pred[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


if __name__ == "__main__":
    y = np.load(NPZ, allow_pickle=True)["y"]
    pred = np.load(E2E_PRED)
    clean = np.load(CLEAN_MASK)
    parcels = np.load(PARCEL_ID)
    trainval = np.load(TRAINVAL)
    assert y.size == pred.size == clean.size == parcels.size == trainval.size

    # Score exactly as evaluate_end_to_end.py does: keep every row in the
    # population and set non-crop truth to 0, so a crop predicted on a non-crop
    # pixel counts as a false positive. Restricting to true-crop rows would hide
    # precisely the contaminant false positives that dominate the error.
    y_eval = np.where(np.isin(y, CODES), y, 0).astype(np.int32)

    populations = {
        "all": np.ones(y.size, dtype=bool),
        "pixel_held_out": ~trainval,          # what the study has been quoting
        "parcel_disjoint": clean,             # the honest, tiny one
    }

    rows, summary = [], []
    for name, mask in populations.items():
        yt, yp = y_eval[mask], pred[mask]
        n = int(mask.sum())
        rep = classification_report(yt, yp, labels=CODES, output_dict=True,
                                    zero_division=0)
        mf = rep["macro avg"]["f1-score"]
        wf = rep["weighted avg"]["f1-score"]

        if name == "parcel_disjoint":
            lo, hi = bootstrap_ci(yt, yp, parcels[mask])
            n_parcels = int(np.unique(parcels[mask]).size)
        else:
            lo = hi = float("nan")
            n_parcels = int(np.unique(parcels[mask]).size)

        print(f"\n=== {name}: {n:,} rows in {n_parcels:,} parcels")
        print(f"    macro F1 {mf:.4f}   weighted F1 {wf:.4f}"
              + (f"   95% CI over parcels [{lo:.4f}, {hi:.4f}]" if lo == lo else ""))
        summary.append({"population": name, "rows": n, "parcels": n_parcels,
                        "macro_f1": round(mf, 4), "weighted_f1": round(wf, 4),
                        "ci_lo": round(lo, 4) if lo == lo else "",
                        "ci_hi": round(hi, 4) if hi == hi else ""})

        for c in CODES:
            d = rep[str(c)]
            rows.append({"population": name, "lu_code": c, "crop": CROPS[c],
                         "support": int(d["support"]),
                         "precision": round(d["precision"], 4),
                         "recall": round(d["recall"], 4),
                         "f1": round(d["f1-score"], 4)})

    import csv
    with open(f"{OUT_DIR}/parcel_disjoint_audit.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(f"{OUT_DIR}/parcel_disjoint_audit_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print(f"\nwrote {OUT_DIR}/parcel_disjoint_audit{{,_summary}}.csv")
