# Peer review of the amended competitiveness plan — results

**Answers:** `docs/HANDOFF_PLAN_AMEND_REVIEW.md` (commit `4ed2614`), section 7.
**Reviewers:** Codex (`codex exec`, adversarial brief, ran the repro itself) with an
independent parallel check of items 3 and 4 by this session. Where both looked, they
agree. **Date:** 2026-08-25.

## 1. The §2 repro — CONFIRMED

Codex ran the handoff's snippet unchanged against the artifacts:

```
strict macro=0.2248 weighted=0.8018
masked macro=0.2405 weighted=0.8309
non-crop rows predicted as crop: 278123   (7.72% of 3,604,150 crop predictions)
```

Exactly the expected numbers. Amendment `a27ea2e`'s numerical account stands.

## 2. Consequence 4 — CUT the causal claim, keep the historical correction

The ruling the handoff asked for: **0.2248 vs 0.2245 is not a comparison.** The two
numbers score different populations (fold 2's 3.80M crop rows vs the whole tile's
15.16M including fitted pixels), and the runs differ in split, balancing, sink,
Stage-3 population and calibration simultaneously. Even the softened "removing
leakage cost no measurable accuracy" is an unsupported causal claim. What survives:
the earlier +0.016 was a scoring artifact, and the project currently has **no
controlled estimate of what leakage cost**. Measuring it would need a matched
pixel-vs-parcel experiment on the same evaluation parcels. Consequence 4 should be
rewritten to say exactly that and no more. (The project memory has been updated to
match.)

## 3. The §3.4 sweep over the ad-hoc scorers — the blast radius is REAL

Two independent sweeps (Codex and this session) reached identical verdicts:

| script | verdict |
|---|---|
| `rescore_collaborator_protocol.py` | safe — maps non-crop truth/pred into a fixed 14-class taxonomy, keeps every row (`:52-66`) |
| `audit_parcel_disjoint.py` | safe — strict convention, non-crop truth → 0 before scoring (`:70-87`) |
| `diagnose_error_budget.py` | not applicable — recall/loss decomposition only, computes no precision |
| `apply_prior_correction.py` | **defect present** — `prior_correction_econ.csv` masks to true-econ rows before precision (`:159-168`); its flat overall report is safe |
| `sweep_prior_alpha.py` | **defect present, and it reached model selection** — `econ13_macro_f1` masks to true crops first (`:228-244`) and is the sweep's ranking metric; at the chosen (0.8, 0.8) cell: masked 0.2428 vs strict-flat 0.2051 |

So §3.4's generalisation is not over-fitted to one script — the same defect lives in
two more, and in one of them it chose the published α optimum. **Consequence:
re-rank `runs/s2_2018_3date_v2/sweep/sweep_results.csv` by `crop_macro_f1_flat`
before ever reusing "the swept optimum α₂=0.8, α₃=0.8"; the optimum may move.**

## 4. The Stage-3 denominator hazard — CONFIRMED; fourth amendment warranted

Verified from primary sources by both reviewers, down to the per-class decomposition:

- Stage 2: base fit exactly 800,000 = 4 × `PER_GROUP_CAP` (`run.log:74`) — the
  uniform-π_train assumption survives; only the comment's number is stale.
- Stage 3 caps did **not** bind uniformly: orchards 172,371
  (70,000+3,770+43,199+14,483+27,008+12,272+1,639), plantation 148,344
  (70,000+70,000+8,344), field 210,000 (3 × 70,000) — `run.log:90,93,96`.
- `sweep_prior_alpha.py:115-125` reads `ratio3_den` from whichever run's Stage-3
  metadata `config.py` selects, and the parcel arm is not registered in `ARMS`
  (`config.py:24-35`), so silently inheriting v2's denominators is a realistic
  failure, wrong for two of three experts.

The amendment M2 needs, sharpened by the review: **do not reuse v2 `train_final`
metadata at all. Derive and save source-prior vectors from the parcel run's actual
calibration subsets (natural-prior fold-1 rows), aligned to each model's
`classes_`, including the Stage-2 sink.** Base-fit counts are not the correct
denominator either, because the parcel run's probabilities were calibrated on
natural priors after the capped fit (`train_parcel_cascade.py:162-174`).

## 5. Remaining overclaims found in the amended plan

1. **`0.3965` is mislabeled in §0's table** — it sits in the "crop-13 macro F1"
   column but `rescore_collaborator_protocol.py` averages 13 crops **plus
   `others`** (flat-14). Re-derived exactly: crop-13 = 0.3880, flat-14 = 0.3965
   (`others` F1 0.5078). The plan's own table breaks its own consequence-3 rule.
   The "roughly 0.40 vs 0.60" sentence compounds it (flat-14 at 200k vs crop-13 at
   ~390k) and should be removed rather than caveated.
2. **"Only cells sharing a cap may be compared" is necessary, not sufficient** —
   epoch, leakage, spatial sample and taxonomy must also match; the sentence
   overpromises what the cap column fixes.
3. **The weighted-F1 "exceeds" claim** mixes crop-13 weighted (0.8018) with flat-15
   weighted (0.714) on top of the rubber-share problem. The review's
   recommendation: native summaries only, never an "exceeds" sentence.
4. **M5's "single fold-2 evaluation"** cannot restore an untouched test set — fold 2
   was read on 2026-08-24. Correct framing: a locked, predeclared exploratory
   re-evaluation.
5. **Temporal deltas** — "no new information" is true but does not by itself predict
   no performance change under RBF/Nystroem distance geometry; the ablation framing
   is right, the categorical dismissal is not.

## Amendment queue (for the plan's owner; none applied here)

- A4: M2 — Stage-3 denominator provenance (item 4 above).
- A5: §0 — fix the 0.3965 column label / add the 0.3880 crop-13 figure; delete the
  "0.40 vs 0.60" sentence; soften "only cells sharing a cap may be compared".
- A6: §0 consequence 4 — cut to the historical correction (item 2 above).
- A7 (small): M5 wording; temporal-delta justification; add the sweep re-ranking
  consequence from item 3.

Raw reviewer output: `.codex_peer_review2.txt` in this worktree (untracked);
regenerate with `codex exec` against the handoff if needed.
