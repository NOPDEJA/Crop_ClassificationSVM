# Plan — cost-sensitive weighted run and the 2026-08-27 report

**Written:** 2026-08-26, after a grilling session with Nop. Every decision below is his,
made explicitly; do not relitigate them, but do flag anything that contradicts the repo.
**Executor:** a fresh Claude Opus 5 session.
**Reviewer:** Codex, adversarial peer review (§7).
**Hard deadline:** the report reaches the professor **Thursday 2026-08-27 before 23:00**.
A full cascade run takes ~6 h, so the run must launch Wednesday night.

---

## 0. State as of writing

- **Current best:** `runs/s2_2018_3date_parcel_m5/` — test macro F1 **0.2344** hard /
  0.2133 joint, weighted F1 0.7974, 12/13 crops non-zero (Langsat 0). 30-column matrix
  (`./aligned_features/svm_s2_3date_m3_features_labels.npz`: 24 cols + MTCI + raw B11 per
  date), repairs from `docs/PLAN_S2_3DATE_COMPETITIVE.md` §3, operating point
  (α₂, α₃) = (0.2, 0.7) predeclared from M2.
- **Tree probe:** `runs/s2_2018_3date_parcel_tree/` (MERGE_TREE=1, orchards+plantation
  collapsed into one expert). Fold-1 tune half only, never read fold 2. Tune macro F1 at
  (0.2, 0.7) ≈ **0.2158** (`val_tune_score_a02_07.csv`). Reported as a negative result
  pending the M5 tune-half comparison in step 1 — do not call it settled before that.
- **Fold 2 has been read 3 times** (baseline, M0, M5). It gets at most **one** more read
  under this plan, gated by §3.
- The root `train_parcel_cascade.py` is the repaired script and is **uncommitted**, as are
  the m5/tree run dirs. Commit before editing anything.
- `m1_calibration_audit.py`, `m2_operating_point.py`, `m3_*.py`, `m4_capacity_probe.py`
  exist only in `.claude/worktrees/implement-competitive-plan/` — copy or adapt what you
  need; do not work in the worktree.
- **Collaborator update (2026-08-25 commits, github.com/Gunkartan/geospatial):** they now
  have a composed cascade. `code/inference/inference.py` chains water → buildings → crops:
  P(water) ≥ 0.56 removed, then P(building) ≥ 0.56 removed, flat 14-class XGBoost on the
  survivors, non-crop truth mapped to 9999/'others'. Their population is a reservoir
  sample capped at 200,000 per class (`extract_all.py`), their crop report is computed on
  filter survivors only, and their macro is over 14 classes including 'others'. Report §6
  of `docs/REPORT_2026-08-25.md` is outdated on the "no composed cascade" claim.
  `rescore_collaborator_protocol.py` mirrors their OLD protocol and must be updated.

Interpreter: `C:\Conda_environment\envs\svm_env\python.exe` (system Python 3.13 has numpy
but no sklearn/pandas). 32 GB machine, serial runs only.

## 1. Settled decisions (do not reopen)

| # | decision |
|---|---|
| D1 | Report = results + problems faced + solution + analysis. Consolidation story; tonight's run is a bonus either way. |
| D2 | Exactly **one** retrain, launched Wednesday night. Thursday is scoring and writing only. Post-hoc work unlimited. |
| D3 | The retrain is **class weights only** — one variable against M5. Δ-date features, texture, cap changes: next-steps section, not this run. |
| D4 | Weights on **Stage 2 (4 groups incl. sink) and all three Stage-3 experts**. Stage 1 untouched. |
| D5 | Scheme: **tempered, w_i = sqrt(n_max / n_i)** from **post-cap** training counts of the model being fitted. No manual dicts (rare-crop validation support is 9–13 px; untunable honestly). |
| D6 | Caps frozen at current `config.py` values (400k/LU; 1M/500k/600k/800k; 200k/group; 70k/LU). Rationale: caps only bite majority classes — raising totals adds zero rare-class rows and worsens the imbalance the weights fight. |
| D7 | **One pre-committed fold-2 read**, under the §3 gate. Otherwise tune-half negative result. |
| D8 | No joint table with the collaborator. The report names the two remaining protocol gaps (200k/class cap; survivor-biased crop denominator) and shows our score under their protocol. |
| D9 | Feature importance: permutation, **Stage 1 + orchards expert**, 30 columns, ~150k val-fold rows, on the reported model (weighted run if it wins the gate, else M5). |
| D10 | Report file `docs/REPORT_2026-08-27.md`, same shape as `REPORT_2026-08-25.md`, prose in **Nop's voice** per the skill at `C:\Users\Nop\Downloads\SKILL.md` (first person, and/so/but chaining, say the unflattering thing, name specifics, close paragraphs on the takeaway, **no semicolons, no em-dashes**). The skill governs the report only, not repo docs like this one. |
| D11 | Tree probe reported as a negative result with its number. |

## 2. Execution steps (Wednesday night)

Each step has a verify gate. Stop and reassess on any failure; do not improvise past it.

1. **Commit current state** on `dev`: repaired `train_parcel_cascade.py`, run dirs
   (m5, tree, m0 CSVs), audit scripts. → verify: `git status` clean of the files you need.
2. **Compute M5's cal-sweep and tune-half score.** Adapt `m2_operating_point.py` to read
   `runs/s2_2018_3date_parcel_m5/*_prob_val.npy` + `val_cal_idx.npy`/`val_tune_idx.npy`:
   169-cell (α₂, α₃) grid, **select on the calibration half** by strict
   `crop_macro_f1_flat`, then score the selected cell on the **tune half**. Save both
   numbers and the CSV into the m5 run dir. This is the gate baseline and the fair
   comparator for the tree's 0.2158. → verify: selected cell and tune score printed;
   sanity: tune ≠ cal score, both plausible (0.15–0.30).
3. **Edit `train_parcel_cascade.py`**: add tempered class weights per D4/D5, behind an
   env flag (e.g. `CLASS_WEIGHT=sqrt`), default off so M5 remains reproducible. Weights
   computed from the post-cap counts of each fitted model (`fit2`, `fit3` populations).
   → verify: `SMOKE=1 CLASS_WEIGHT=sqrt` completes; logged weight dicts show rubber ≈ 1
   and the rarest class the largest weight; `CLASS_WEIGHT` unset reproduces M5 behaviour.
4. **Write the predeclaration** (§3 verbatim) to
   `runs/s2_2018_3date_parcel_weighted/PREDECLARATION.md` **before** launch.
5. **Launch** the full run, detached/unbuffered, `SKIP_TEST=1`, output
   `runs/s2_2018_3date_parcel_weighted/`. (~6 h; the long-training-workflow memory has the
   babysitting pattern.) → verify: log advancing past Stage-1 fit before leaving it.
6. **While it runs:** update `rescore_collaborator_protocol.py` to the new chained
   protocol (0.56 sequential filters → survivor population → 14-class classification
   report incl. 'others' → their capped sampling). Note in code comments which of their
   choices it reproduces vs. approximates. Also check whether their capped sample is drawn
   from ground their models trained on; one factual sentence for the report if so.
   → verify: runs against M5's saved test probabilities without error.
7. **Prep the permutation-importance script** (D9) so Thursday only executes it.

## 3. Predeclaration (copy verbatim into the run dir before launch)

> Run `s2_2018_3date_parcel_weighted`, launched 2026-08-26. Identical to M5 except:
> tempered class weights w_i = sqrt(n_max/n_i) from post-cap training counts, applied to
> Stage 2 and all Stage-3 experts. Stage 1, caps, matrix, folds, calibration unchanged.
> Operating point: (α₂, α₃) selected on the calibration half by strict crop_macro_f1_flat
> over the 169-cell grid, scored once on the tune half. Gate: **iff** the tune-half score
> exceeds M5's tune-half score obtained by the same procedure, fold 2 is read once (hard
> routing, strict full-population convention) and becomes the report headline. Otherwise
> fold 2 is not read and the run is reported as a tune-half negative result. No other
> fold-2 access is permitted. The tune half is never used for selection.

## 4. Thursday schedule

| time | task |
|---|---|
| morning | score the weighted run: cal-half sweep → tune-half score → apply the gate; conditional single fold-2 read |
| morning | collaborator-protocol rescore of the reported model (step 6 script) |
| midday | permutation importance per D9 |
| afternoon | write `docs/REPORT_2026-08-27.md`; Codex review (§7); fix; deliver **before 23:00** with margin |

If the run crashed overnight: no relaunch (D2 — Thursday is not for training). The report
ships on M5 with the negative/incomplete result stated plainly.

## 5. Report outline

1. Summary (three-to-four numbered things that happened).
2. Results: 0.2248 → 0.2283 (M0, repairs cost nothing, CI [+0.0001, +0.0070]) → 0.2344
   (M5, test) → tonight's run either way. Per-crop table old vs new.
3. Problem faced: 6,685:1 imbalance; the prior/operating-point dial dominates
   architecture (the §4 story of the previous report, updated).
4. Solution tried: cost-sensitive SVM (cite Waske 2009, RUESVMs 2020), tempered weights,
   honest α selection (cite Saerens 2002 for the prior-correction family).
5. Analysis: feature importance (does MTCI/B11 carry the orchard signal?); tree-merge
   negative result vs M5 tune-half; M5 vs M0 attribution caveat (features+α changed
   together).
6. Collaborator §6 rewrite: their new chained cascade, convergence on FP-counting, the
   two remaining gaps (D8), our score under their protocol.
7. Limitations (carry forward: fold-2 read count, frozen hyperparameters, fold-support
   variance, Langsat).
8. Next steps: Δ-date features (rubber defoliation Oct→Dec — Cai 2022, Wang 2023 Nature),
   cap-strategy experiment (hand caps vs uniform vs √n-proportional), parcel aggregation
   as a labeled side-analysis (Belgiu & Csillik 2018, Matvienko 2020), red-edge columns
   (Clevers & Gitelson 2013), full-cascade importance. Cite Chen et al. 2024 (10 m crop
   mapping, eastern Thailand, OA 85.6 % — with a dense 2017–23 harmonic series) as the
   3-date ceiling reference; Foody 2020 for metric protocol; Waśniewski 2022 +
   Turkoglu 2021 for hierarchical-vs-flat; texture cited not built (Nomura & Mitchard
   2018). Full citation list in the session transcript of 2026-08-26; all DOIs verified
   by the research agent.

## 6. Peer-review brief (Codex)

Review adversarially after execution, before the report ships. Two retracted findings in
this project's history were caught only on re-derivation; assume something here is wrong
too. Priority order:

1. **Gate fairness.** M5's tune score and the weighted run's tune score must come from
   the *identical* procedure (same grid, same cal-half selection, same strict scorer,
   same rows). Any asymmetry invalidates the gate. Re-derive both numbers from the saved
   `*_prob_val.npy` arrays; do not trust printed logs.
2. **Weight correctness.** Recompute w_i = sqrt(n_max/n_i) from the logged post-cap
   counts and compare to the dicts the run logged. Check the weights were computed from
   the counts of the model being fitted (Stage-2 groups vs Stage-3 per-expert), not from
   global counts, and that the sink class got a weight (not silently excluded).
3. **No fold-2 leakage.** Grep every script run since M5 for reads of fold-2 rows
   (`asg==2`) outside the single gated read. The tree run's SKIP_TEST claim too.
4. **Scoring convention.** Every number destined for the report uses the strict
   convention: full population, non-crop truth mapped to 0, 13-crop macro. Three of six
   scorers had this defect once; check any new or adapted scorer (step 2's, step 6's).
5. **Collaborator rescore.** Verify the updated script's filter order, thresholds (0.56),
   survivor population, 14-class macro incl. 'others', and capped sampling actually match
   `inference.py`/`extract_all.py` at the pinned commit (a9a7ca0), and that approximations
   are declared in the report text.
6. **Report claims vs. artifacts.** Every number in `REPORT_2026-08-27.md` traceable to a
   CSV/manifest in the repo. Flag any tune-vs-cal confusion (the report must quote tune),
   any comparison across differently-defined populations, and any claim the predeclaration
   does not license (e.g. calling the M0→M5 delta attributable — it is not; features and α
   changed together).
7. **Voice check** (light): the report should follow the SKILL.md rules — flag semicolons,
   em-dashes, "furthermore/thus/leveraged", and third-person hedging.

Deliver findings as a ranked list with a re-derivation command or file:line for each.

## 7. Out of scope (explicitly)

Cap/sample-size changes, Δ-date features, texture/GLCM, parcel-majority aggregation as a
headline number, per-rare-crop manual weights or thresholds, any joint table, any second
retrain, `2018/` folder, and the archived per-stage pixel-split pipeline.
