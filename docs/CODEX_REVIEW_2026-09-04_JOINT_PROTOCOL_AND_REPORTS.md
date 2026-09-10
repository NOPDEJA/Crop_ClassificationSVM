# Codex Review for Claude: Joint Protocol and Latest Reports

**Date:** 2026-09-04  
**Scope:** Review and revise the current root-level joint-paper documents. Do not modify
`2018/`, trained models, run artifacts, split arrays, labels, or reported measurements.

## Review outcome

The joint-paper direction is promising, but `docs/JOINT_PROTOCOL_2026-09-03.md` is not yet an
executable 2 x 2 factorial protocol. Its proposed controls invalidate the claim that two cells
already exist, its architecture levels are not applied consistently across algorithms, and it
omits controls for missing-value eligibility and the sampled training population.

Revise the protocol before sending it to the collaborator or professor. Preserve the useful
parts: parcel-disjoint evaluation, full-population scoring, stage-by-stage error accounting,
rare-class parcel support, and disagreement mapping.

## Files reviewed

- `docs/JOINT_PROTOCOL_2026-09-03.md`
- `docs/REPORT_2026-08-28_E7_FINAL_READ.md`
- `docs/PROFESSOR_PROGRESS_REPORT_2026-08-27.md` (updated 2026-09-03)
- `runs/s2_2018_3date_parcel_e7_final/report_hard.csv`
- `runs/s2_2018_3date_parcel_e7_final/manifest.json`
- `runs/s2_2018_3date_parcel_e7_final/PREDECLARATION.md`
- `runs/s2mass_pool_sensitivity/gate_g3.json`
- `runs/retune_stage2_orchards/gate_g4.json`
- `runs/s3mass_experts/gate_g5.json`
- `C:/Users/Nop/Downloads/Report 2 September 2026.pdf`

## Findings, ordered by severity

### P0 - Neither claimed existing factorial cell satisfies the proposed controls

The matrix at joint-protocol lines 50-54 marks SVM/E7 and the collaborator's XGBoost result as
existing cells. They are only existing implementations or preliminary results:

- E7 uses the SVM 30-feature set, not the proposed common 15 columns.
- XGBoost does not use the shared parcel split, population, denominator, or held-out partition.
- The PDF reports the result from the pipeline with the shared-dataset NaN deletion problem,
  before a verified score from the repaired per-model extraction path.

**Required change:** Mark all four aligned cells as `to be run`. Describe E7 and the 2 September
XGBoost result as motivation/baselines that are excluded from the factorial estimates.

### P0 - The architecture factor is not coherent

The matrix calls the XGBoost pipeline a `flat 13-class` architecture. End to end, it is a
sequential rejection cascade: water rejection, building rejection, then a flat crop head. A
direct flat SVM would therefore not be the same architecture.

**Required change:** Choose one of these designs and state it exactly:

1. Preferred for the current paper angle: architecture = **semantic routing cascade** versus
   **sequential rejection cascade**. Implement both architectures with both algorithms.
2. Alternative: architecture = **direct flat 14-label classifier** versus one fully specified
   shared cascade. In that case, the current XGBoost pipeline is not the flat cell.

Every cell must emit the same final label set: 13 target crops plus `others`.

### P0 - A three-cell result is not a complete 2 x 2 factorial

The protocol says cell C may be dropped while retaining the design. With one cell missing, the
algorithm effect, architecture effect, and interaction cannot all be identified as claimed.

**Required change:** Require four aligned cells for factorial language. If only three are
feasible, rename the study as controlled pairwise comparisons and limit the claims accordingly.

### P1 - The label artifact is identified incorrectly

Joint-protocol line 111 names `label/label_47PQQ.tif` as the 3-pixel-eroded artifact. The code
defines:

- raw: `label/label_47PQQ.tif`
- eroded: `label/label_47PQQ_buffered.tif`

See `config.py:114-115`, `buffer_labels.py:39-40`, and `align_indices_labels.py:10`.

**Required change:** Make `label/label_47PQQ_buffered.tif` the artifact of record wherever the
protocol requires 30 m erosion. State separately that the raw raster is its source.

### P1 - The split description is stale

Joint-protocol line 109 says both sides currently split at pixel level. The current E7 SVM is
parcel-disjoint; only earlier SVM experiments and the current XGBoost workflow used pixel-level
splits.

**Required change:** State that XGBoost requires migration to the existing parcel split, while
E7 already demonstrates the SVM parcel-disjoint machinery.

### P1 - Row eligibility and missing-value handling are not frozen

The PDF's central issue is that a NaN in water/building texture features removed a row whose crop
features were usable. Yet the six controls do not define which raster pixels are eligible when
common features contain NaN or infinity.

**Required change:** Add an explicit control containing:

- a shared raster grid and linear pixel identifier;
- one frozen evaluation-row mask over the common features;
- a declared missing-value policy;
- proof that every cell predicts the same ordered test pixel IDs.

Do not let each model silently delete a different set of rows. Save the final test pixel IDs or
Boolean mask as an artifact and hash it.

### P1 - The training population is not held constant

The protocol permits the flat SVM to use a roughly 200,000-per-class cap but does not require
XGBoost to use the same sampled training rows. That confounds algorithm with training volume and
sample composition.

**Required change:** For the algorithm comparison, give corresponding SVM and XGBoost cells the
same ordered training pixel IDs. Use a seeded cap if computationally necessary. An additional
XGBoost-full-data arm may be reported as a resource/data-volume ablation, not as the pure
algorithm cell. Evaluation remains natural and uncapped.

### P1 - E7's test fold is not globally untouched

The E7 predeclaration correctly says fold 2 was read once during the E1-E7 plan. The same test
partition had already produced M5 and earlier checkpoint results, so it is not a globally fresh
paper test set.

**Required change:** Either lock a genuinely new paper test partition before the joint runs, or
describe the existing fold as previously observed and avoid `read once` language that implies it
was untouched across the full project.

### P1 - The PDF does not isolate the cause of the crop-score decline

The NaN-row deletion mechanism is plausible and the separate-dataset fix is appropriate, but the
support chart does not show a general population collapse:

- standalone support totals approximately 304,081 rows;
- pipeline support totals 755,295 rows;
- pipeline support is lower for 9 classes and higher for 5, with large increases for cassava,
  pineapple, rubber, oil palm, and `others`.

Lower support also does not mechanically lower F1. The two results use materially different
class distributions and evaluation populations.

**Required change:** Describe NaN deletion as `one identified mechanism`, not the established
sole cause. Require an old-versus-repaired comparison on the same frozen pixel IDs before making
a causal statement.

### P1 - The SVM report overstates certainty once

The E7 measurements are correctly reproduced by the saved artifacts:

- rows: 5,500,269;
- macro F1: 0.2429;
- weighted F1: 0.7949;
- M5 macro F1: 0.2344;
- observed delta: +0.0085.

However, the updated professor report says `the improvement is real` while also stating that the
parcel-bootstrap uncertainty is about +/-0.014, wider than +0.0085.

**Required change:** Use `observed test-fold gain` or `point-estimate gain`. State that it is not
clearly separated from parcel-level uncertainty. Also surface that weighted F1 decreased from
0.7974 to 0.7949 while macro F1 increased.

### P2 - G3's mean treatment effect is not a noise floor

`docs/REPORT_2026-08-28_E7_FINAL_READ.md:67` calls the G3 results a pool-draw noise floor with
mean +0.0078 and range +0.0072 to +0.0087. The nonzero mean is the measured treatment effect. The
between-draw spread describes sensitivity to the pool draw.

**Required change:** Replace `noise floor` with `repeatability across three pool draws`; do not
use it to classify unrelated per-crop E7 changes as noise.

### P2 - Zero-support macro naming is ambiguous

Using the rounded PDF table:

- 13-crop macro F1 including zero-support Longkong is approximately 0.3692;
- excluding Longkong gives approximately 0.400 over 12 supported crops;
- crops plus `others`, including Longkong, gives the reported approximately 0.38.

**Required change:** Freeze an explicit ordered label list. If support-zero targets are excluded,
name the statistic `macro F1 over supported target crops`, report its denominator, and separately
state that Longkong was unevaluable. Do not call a 12-class average a 13-crop macro without a
qualification.

## Required revision sequence

1. Correct factual errors: label filename, split status, and zero-support naming.
2. Redefine the architecture levels and rebuild the matrix so the same two levels apply to both
   algorithms.
3. Mark every aligned factorial cell as pending.
4. Add controls for evaluation pixel IDs/missing values and identical sampled training rows.
5. Correct causal and uncertainty wording in the E7 and professor-facing reports without
   changing any numeric result.
6. Add a short execution manifest specification for each future cell: data hash, pixel-ID hashes,
   split hash, feature names/order, training sample hash, labels, hyperparameter-search budget,
   selected settings, thresholds, seed, and final prediction hash.

## Completion criteria

The revision is complete only when all of the following are true:

- The two architecture levels mean the same thing in both algorithm columns.
- Four pending aligned runs are required for any factorial-effect or interaction claim.
- No current SVM or XGBoost score is presented as an aligned factorial cell.
- `label_47PQQ_buffered.tif` is consistently identified as the eroded label artifact.
- Every future cell is required to train and score on shared, hashed pixel identities.
- Missing-value policy and training sampling are explicit controls.
- E7 is described as an observed +0.0085 macro-F1 point-estimate gain with weighted-F1 trade-off
  and parcel-level uncertainty.
- The PDF's NaN issue is presented as a plausible, testable contributor rather than an isolated
  causal conclusion.
- Macro-F1 denominators and support-zero exclusions are explicit in every comparison table.

Return a concise change summary and list every modified document. Preserve all source run
artifacts and historical measurements unchanged.
