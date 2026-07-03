---
name: regression-check
description: Compare a newly finished stage's report.csv against the documented baseline in a prior run's REPORT.md (or report.csv) to catch silent metric regressions before accepting a new run as the new baseline. Use after any stage1/2/3 training run completes, before retraining downstream stages on top of it, or when the user asks to check for regressions.
---

# SVM stage regression check

Regression check, not experiment design — a baseline already exists (a prior run's
`report.csv` or its `REPORT.md` writeup under `runs/<name>/`). This skill re-reads the
new run's report and diffs it against that baseline; it does not launch training.

## Why this exists

The DEM+S1+S2 combined run (`runs/dem_s1_s2`) silently regressed at every stage — Stage 1
accuracy 0.667→0.648, Stage 2 0.509→0.466, end-to-end econ recall 23.1%→19.0% — and it took
a full session of manual diffing across CSVs to notice and diagnose (RBF gamma ~100x
off-scale + kernel dilution from 94 redundant SAR/DEM columns). A `report.csv` diff at
run-completion time would have caught it immediately instead of requiring a fresh
investigation.

## Steps

1. **Identify the new run and its baseline.** The new run's report lives at
   `runs/<name>/stage{1,2,3}_..._report.csv` (or `..._model.joblib` sibling). Find the
   comparison baseline: prefer a sibling run directory's `REPORT.md` documenting the same
   chain type (e.g. `runs/dem_s1_s2/REPORT.md` for a `dem_s1_s2_v2` rerun), else the
   immediately prior run's `report.csv` for the same stage.

2. **Diff per-class metrics.** From each report.csv, compare `precision`, `recall`,
   `f1-score` per class and the overall `accuracy` row. Flag:
   - Any class F1 or overall accuracy dropping **>0.03 absolute** vs baseline.
   - A pattern where *most* classes drop together (broad regression) rather than one
     class trading off against another — that shape matches the kernel-dilution failure
     mode, not normal hyperparameter noise.
   - Classes that "improved" only because their routed/support pixel count shrank a lot
     vs baseline (check the `support` column) — that's survivor bias, not a real gain.

3. **Check for the known off-scale-gamma symptom** if the model is an RBF/Nystroem SVM:
   read the run's `meta.json` or `run.log` for the winning `nyst__gamma` and
   `nyst__n_components`. If gamma is pinned at the grid's largest value, or
   `n_components` is pinned at the grid's ceiling in every stage, the search range itself
   is likely miscalibrated (see `[[combined-run-kernel-dilution]]` memory) — the same
   failure that caused the original incident above.

4. **Report, don't silently fix.** Summarize pass/fail per class and per stage. If
   something regressed, name the likely suspect (recent config diff, feature-set change,
   grid miscalibration) and let the user decide whether to accept, investigate, or rerun —
   per this project's Surgical Changes rule, don't retrain or edit config while just
   checking for regressions.
