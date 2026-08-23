"""apply_operating_point.py

Emit the prior-corrected cascade output at one operating point, and score it through
the same machinery as the uncorrected baseline.

The sweep (runs/<RUN>/sweep/sweep_results.csv) ranked 256 points but only ever held
their predictions in memory. This writes the chosen point out as real prediction
arrays plus the standard flat-15 tables and confusion matrices, so the corrected
result can go in the paper next to the uncorrected one.

Default point is (alpha_2 0.8, alpha_3 0.3): the best cell that keeps all 13 crops
alive. The best cell overall, (0.8, 0.8), scores 0.2203 against 0.2129 but kills three
crops outright -- 0.0074 macro F1 is not worth reporting three species as absent from a
province where they are mapped.

Everything is written under a distinct tag; the uncorrected baseline is not touched.

Usage:
  python apply_operating_point.py            # (0.8, 0.3)
  python apply_operating_point.py 0.8 0.8
"""
import sys
import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE
from sweep_prior_alpha import Context, compose, log
from evaluate_flat_15class import (to_flat_true, to_flat_pred, score,
                                   matched_subset, NAMES)
from confusion_report import confusion_set, write_confusion_set

if __name__ == "__main__":
    a2 = float(sys.argv[1]) if len(sys.argv) > 2 else 0.8
    a3 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
    tag = f"corrected_a2-{a2}_a3-{a3}"
    log(f"operating point alpha_2={a2} alpha_3={a3} -> tag {tag}")

    ctx = Context()
    final, s2_full = compose(ctx, a2, a3)
    y, s1, held = ctx.y, ctx.s1, ctx.held

    np.save(f"{OUT_DIR}/end_to_end_lu_pred_{tag}.npy", final)
    np.save(f"{OUT_DIR}/stage2_pred_{tag}.npy", s2_full)
    log("saved corrected prediction arrays")

    y_flat, p_flat = to_flat_true(y), to_flat_pred(s1, final)
    populations = {"all": np.arange(y.size), "held_out": held}
    rng = np.random.default_rng(RANDOM_STATE)
    populations["matched"] = held[matched_subset(y_flat[held], rng)]

    per_class, overall = [], []
    for name, idx in populations.items():
        rows, ov = score(y_flat[idx], p_flat[idx], fold_forest=True, tag=name)
        per_class += rows
        overall.append(ov)
        log(f"{name}: {ov}")

    rows_f, _ = score(y_flat[held], p_flat[held], fold_forest=False,
                      tag="held_out_forest_split")
    per_class += [r for r in rows_f if r["class"] == "Forest"]

    pc = f"{OUT_DIR}/flat15_per_class_{tag}.csv"
    ov = f"{OUT_DIR}/flat15_overall_{tag}.csv"
    pd.DataFrame(per_class).to_csv(pc, index=False, encoding="utf-8-sig")
    pd.DataFrame(overall).to_csv(ov, index=False, encoding="utf-8-sig")
    log(f"saved {pc}\n       {ov}")

    d = write_confusion_set(confusion_set(y, s1, s2_full, final, held), OUT_DIR, tag=tag)
    log(f"saved confusion matrices to {d}")

    base = pd.read_csv(f"{OUT_DIR}/flat15_per_class.csv")
    new = pd.DataFrame(per_class)
    cmp = (base[base.population == "held_out"][["class", "f1"]]
           .merge(new[new.population == "held_out"][["class", "f1"]],
                  on="class", suffixes=("_baseline", "_corrected")))
    cmp["delta"] = (cmp.f1_corrected - cmp.f1_baseline).round(4)
    log("\nheld_out per-class F1, baseline vs corrected:\n" + cmp.to_string(index=False))
