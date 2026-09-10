# Disposition of the Codex review of 2026-09-10

Reviewed document: `docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md` — a discussion draft, never
sent to the collaborator. Review: `docs/CODEX_REVIEW_2026-09-10_XGB_ARM.md`.

**Every finding is accepted.** One is accepted with a deliberate deviation, recorded in full
below rather than quietly narrowed. No numeric result changed anywhere: E7 is still 0.2429 on
5,500,269 rows, and no model was refitted.

The review's bottom line was "proceed with the crossover, but do not package this document
yet." That is what happened. The draft is superseded by three artifacts that are checkable
where it was only assertive:

| artifact | what it fixes |
|---|---|
| `cascade_algo.py` | the seam exists in code, so "same architecture" is a diff |
| `E7_EFFECTIVE_CONFIG.json` | E7's configuration in one file with hashes, for the first time |
| `docs/HANDOFF_XGB_CASCADE.md` | what he runs, in order, with the asymmetries disclosed |

---

## The P0: the draft described a configuration E7 did not use

**Accepted, and it was the most damaging error in the draft.** The Tier-A table specified
`sqrt(n_max/n_c)` weights for Stage 2 and every Stage-3 expert. E7 used neither.

What E7 actually used, now encoded in `configs/e7_svm.json` and `E7_EFFECTIVE_CONFIG.json`:

| stage | E7 reality | the draft claimed |
|---|---|---|
| Stage 1 | unweighted, M5's frozen model | unweighted (correct) |
| Stage 2 | **subtype-mass** weights, renormalised per group | `sqrt(n_max/n_c)` |
| orchards | **unweighted** (Gate G5 failed at −0.0012) | `sqrt(n_max/n_c)` |
| plantation, field | **unweighted**, M5's originals | `sqrt(n_max/n_c)` |
| operating point | 0.4 / 0.5, selected on the **calibration** half | said tuning half |

Implemented as `CLASS_WEIGHT=subtype` in `train_parcel_cascade.py`, a third scheme alongside
the existing `sqrt`: subtype-mass at Stage 2 only, experts untouched. The existing `sqrt` path
is unchanged so no earlier run's reproducibility is affected.

## The six factual claims

| claim | verdict | disposition |
|---|---|---|
| `base_pipe()` is the only estimator constructor | **wrong** | Accepted. The seam is now `make_base()` plus `cascade_algo.py`, and the module docstring names the one-vs-rest wrapper and the per-class sigmoids as part of the learning rule rather than pretending the seam is one function. |
| `PlattCalibrated` needs `decision_function` | confirmed | Resolved without touching `PlattCalibrated` at all: `XGBMarginClassifier.decision_function` returns XGBoost's pre-softmax margins, which is the interface the calibrator already expects. |
| one-vs-rest exists to route `sample_weight` | **wrong** | Accepted. It defines the fitted losses and the margins being calibrated. Dropping it for XGBoost is disclosed as a change to the algorithm, not as removing overhead. |
| cross-fitting makes Stage-2 rows algorithm-dependent | confirmed | Accepted, and it became the framework rule in the protocol's section 3. |
| P1/P23 leaky, E4 group-clean | confirmed, qualified | Accepted including the qualification: E4 searched Stage 2 **unweighted** and refitted it **weighted**. `xgb_search.py` copies that quirk deliberately, with a comment saying why. |
| both E4 winners at the maximum of every axis | **wrong** | Accepted. `n_components` and `C` were maximal; `gamma` landed at 0.5x the rule, interior to its grid. The "categorically under-tuned" claim is retracted and the corrected version is in `E7_EFFECTIVE_CONFIG.json` under `grid_ceiling`. |

## Answers Q1 to Q5

**Q1, missing values.** Accepted as recommended: median imputation for the run of record,
native NaN handling as `XGB_NATIVE_NAN=1`, a separately labelled ablation. The review's extra
point is also accepted and recorded in the config: because Stage-2 fit populations are
algorithm-dependent, the two arms fit their medians on different rows, so the policy is
identical but the imputed values are not guaranteed to be. The draft's "byte-identical
inputs" claim is withdrawn.

**Q2, Platt-scaling XGBoost.** Accepted, and the draft's `log(p)` fallback is dropped
entirely — it was an arbitrary transformation and behaved badly against the existing
fallback. Both arms now feed the same calibrator a raw per-class score: one-vs-rest decision
margins from the SVM, pre-softmax multiclass margins from XGBoost. The phrase "already
calibrated" is removed; exposing `predict_proba` is not evidence of calibration.

**Q3, tuning budget — accepted with a deliberate deviation.** The review's position is that a
full symmetric clean retune is *necessary* before the paired difference can be attributed to
the algorithm. We agree with the reasoning and are not doing it first. The order is: ship the
XGBoost package, run the SVM-in-his-cascade half, then run the clean SVM retune. The
consequence is stated in three places rather than hidden — the handoff, the protocol, and the
config file all say that until the retune lands, the pair measures *algorithm plus tuning
regime*, and that the asymmetry favours the XGBoost arm because it searches all five stages
while ours searched two. This is a sequencing decision, not a disagreement.

**Q4, dropping one-vs-rest.** Accepted, including the retraction. The claim that the
weighting "transfers exactly" is withdrawn. The control is now worded as *the same
observation-level weight vector is passed to each algorithm's native loss*, never *the same
cost-sensitive learning rule*. An unweighted XGBoost run is offered as a sensitivity arm.

**Q5, mis-filed Tier-A items.** All accepted. `CROSSFIT_S1` stays Tier A as a leakage-control
rule. `CALIB_MAX` stays Tier A because both arms do use the common calibrator. The
operating-point *procedure* is Tier A and the *selected values* are per-arm. `CHUNK` moved out
of Tier B entirely into "neither scientific tier". `SKIP_TEST` is labelled experimental
governance rather than architecture.

## The attacks on the tier assignments

**Row identities (P0).** Accepted, and it forced the protocol change. The review put two
options: allow downstream rows to be treatment-induced mediators, or force identical Stage-2
rows via an external router. We took the first. The second would mean the XGBoost cell's
Stage 1 was never tested. What is frozen: the fold-0 pool, every cap, the cross-fit parcel
partition, the fold-1 halves, the evaluation rows. What floats is strictly downstream of the
treatment. Every run now saves and hashes its fit and calibration indices — including
**Stage 1's, which was the one set M5 never saved**; `train_parcel_cascade.py` now writes
`stage1_fit_idx.npy`.

**Operating point (P1).** Accepted both parts. Each arm selects its own alphas; only the
grid, the selection partition and the objective are shared. The factual correction is also
accepted: E7 selected on the **calibration** half, not the tuning half as the draft said.

**Cost sensitivity (P1).** Accepted, superseded by the P0 fix above.

**`CHUNK` (P1) and one-vs-rest wording (P1).** Both accepted, as above.

## What the draft missed

| finding | disposition |
|---|---|
| **P0** run-of-record decision rule not frozen | Accepted. **Hard routing is the primary endpoint**, frozen in the handoff, the protocol and the config. Joint is reported but secondary. Without this either arm could pick the flattering composition after seeing both. |
| **P0** wrong scorer cited | Accepted. `evaluate_end_to_end.py` scores every NPZ row and would give a train-inclusive number. It is now listed under `do_not_use` in the config, with the correct fold-2 scorers named. |
| **P0** feature identity and order absent | Accepted. The config carries the NPZ SHA-256, row and column counts, the feature names in order where the NPZ stores them, and the `valid_cols` hash. |
| **P1** E4 tuned a different loss from the deployed one | Accepted and disclosed; the XGBoost search copies the procedure rather than fixing one arm. |
| **P1** stage-specific calibration populations unspecified | Accepted. All three are now written into the config, including the architectural control the draft omitted: **Stage-3 experts are trained and calibrated by TRUE group membership**, regardless of whether the upstream stages routed the pixel correctly. |
| **P1** class ordering and label encoding unspecified | Accepted. `XGBMarginClassifier` encodes to `0..K-1` through a sorted `classes_`, asserts the encoding is reversible, and asserts the returned margin column count matches the declared class list on every call. A silent permutation here would relabel the crops and raise nowhere else. |
| **P1** RNG replay is not a sufficient contract | Accepted. Hashes of actual row identities, not seed-plus-code-path equivalence. Stage 1's indices are now saved, closing the last gap. |
| **P1** early stopping ungoverned | Accepted. **Forbidden.** `n_estimators` is a grid axis instead. Its evaluation rows would otherwise enter model selection, and no partition is free to host them. |
| **P1** test partition not globally untouched | Accepted. Not resolvable by us alone: it is decision 2 in the protocol, flagged time-critical, because after the XGBoost cell runs it is too late to lock a new partition. |
| **P2** consolidated config insufficient | Accepted. Every field the review listed is in `emit_e7_config.py`, including the weight-vector hash, the calibration score definition and fallback, the operating-point grid and selection partition, and the endpoint. |

## Verification performed before accepting

Claims were checked against the artifacts, not taken on the review's word.

| checked | result |
|---|---|
| M5's actual per-stage parameters | `runs/s2_2018_3date_parcel_m5/manifest.json`: `class_weight: null`, Stage-3 at `n_components: 1200`. Confirms Stage 2 and every expert were unweighted in M5. |
| E4's winners | `runs/retune_stage2_orchards/winners.json`: both stages at 1200 / C=30 / gamma 0.01666667. Gamma is 0.5x the rule, **interior** to the grid — the review's correction holds. |
| E7's operating point | `runs/.../e7_final/selected_operating_point.json`: `selected_on: "cal"`, alpha2 0.4, alpha3 0.5. The draft was wrong. |
| Whether the Stage-2 pool is reproducible in one script | `s2mass_stage2.py` **replays** the trainer's seed-42 sequence to draw 11 rather than drawing fresh, and asserts the pool is 800,000 rows. So a single-script run does reproduce Stage 2's pool; the orchards expert's population still differs, drawn from E4's seed 777. Recorded as the reproduction caveat. |
| XGBoost's margin interface | `predict(X, output_margin=True)` returns `(n, K)` pre-softmax scores whose softmax equals `predict_proba`, and the column order follows the `0..K-1` encoding. Verified on a synthetic multiclass fit before the wrapper was written. |
| The SVM path is unchanged | Two `SMOKE=1` runs of the trainer, before and after the edits, on the same NPZ and seed, compared by prediction hash. See `docs/HANDOFF_XGB_CASCADE.md` and the seam commit message. |

## Not done, and why

**The clean SVM retune.** See Q3. Sequenced after the second half of the crossover, disclosed
in the meantime.

**A new test partition.** Requires agreement from the professor and the collaborator, because
it costs both sides a re-read. Raised as protocol decision 2.

**Nothing under `2018/`, and no trained model, run artifact, split array, label raster or
reported measurement was modified.**
