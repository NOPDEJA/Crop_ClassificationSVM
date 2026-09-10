# Codex Review of 2026-09-10 — XGB Arm Config Surface

Reviewed document: `docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md`.
Produced by `codex exec` (codex-cli 0.149.0, read-only sandbox) against the working tree.
Verbatim below; disposition to follow.

---

# P0 — The document does not describe the SVM configuration it proposes to match

This is the most serious problem. Tier A says the reference cascade uses `sqrt(n_max/n_c)` weights for Stage 2 and every Stage-3 expert (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:67,73-75`). E7 does not:

- Stage 2 uses subtype-mass weights, calculated within each crop group and renormalized to preserve group mass (`s2mass_stage2.py:197-229`; `e7_final_cascade_fold2_read.py:7-13`).
- The E4 orchards model is unweighted (`e4_retune_stage2_orchards.py:309-315`; `e7_final_cascade_fold2_read.py:14-16`).
- Plantation and field are the original unweighted M5 experts (`e7_final_cascade_fold2_read.py:17-18`).
- E7 selected `alpha2=0.4, alpha3=0.5` on the calibration half, not the tuning half (`sweep_operating_point.py:148-160`; `e7_final_cascade_fold2_read.py:140-148`).

If the collaborator implements the Tier-A table literally, the XGBoost arm will not share E7’s training loss or operating-point selection. That breaks the comparison.

The crossover idea itself is not unsound. Controlled comparisons within each architecture/dataset pair are defensible. But this document is not yet a valid execution contract.

## Verification of the six factual claims

| Claim | Verdict | Evidence |
|---|---|---|
| `base_pipe()` is the only estimator constructor | **WRONG** | It constructs the base pipeline at `train_parcel_cascade.py:213-229`, but the file separately constructs `OneVsRestClassifier` at `:303` and one `_SigmoidCalibration` per class at `:276-278`. The seam is therefore not confined to `base_pipe()`. |
| `PlattCalibrated` requires `decision_function` and cannot accept XGBoost unmodified | **CONFIRMED** | `_scores()` unconditionally calls `self.base.decision_function(X)` at `train_parcel_cascade.py:260-262`. A normal `XGBClassifier` does not expose that interface, so an adapter or calibration rewrite is required. |
| `OneVsRestClassifier` exists to route `sample_weight` past `LabelBinarizer` | **WRONG** | The wrapper is always constructed, including unweighted fits (`train_parcel_cascade.py:303-311`). The comment says the presence of OvR/`LabelBinarizer` is why a multiclass `class_weight` dictionary cannot work and why row weights are used (`:135-143`). Metadata routing delivers the weights; OvR does not exist to route them. It also defines the independent-binary learning rule and the margins being calibrated. |
| `crossfit_stage1_econ` makes Stage-2 rows depend on Stage-1 predictions and therefore the algorithm | **CONFIRMED** | Each cross-fit model predicts its held-out part at `train_parcel_cascade.py:413-417`; those predictions replace the fold-0 routes at `:505-507`; Stage-2 candidates are then `route == 1` at `:523`, with the training subset formed at `:546`. |
| P1/P23 came from leaky inner CV, while E4 used parcel-grouped `GroupKFold` | **CONFIRMED, with qualification** | The frozen provenance is explicit at `train_parcel_cascade.py:107-113`. E4 uses `GroupKFold(3)` and macro F1 at `e4_retune_stage2_orchards.py:114-124`, using `parcel_id_row` at `:287-290`. However, E4’s Stage-2 search was unweighted, while its final refit used saved subtype-mass weights (`:120-124` versus `:297-304`). It was group-clean, but it did not tune the exact final loss. |
| Both E4 winners were at the maximum of every grid axis | **WRONG** | Both used maximum `n_components=1200` and `C=30`, but `gamma=0.01666667`, corresponding to multiplier `0.5`; the gamma grid was `{0.25, 0.5, 1, 2}/n_features` (`e4_retune_stage2_orchards.py:84-87`; `runs/retune_stage2_orchards/winners.json:2-14`). Gamma was not at its maximum. Therefore the claim that the search proves both stages are categorically “under-tuned” is too strong. |

# Answers to Q1–Q5

## P1 — Q1: Missing-value policy

Control 3 binds as currently written. It requires one missing-value policy “declared once and applied identically” (`docs/JOINT_PROTOCOL_2026-09-03.md:130-132`). Keeping the same rows but changing feature values still changes the experimental input. Equal eligibility alone does not make median-imputed and native-NaN inputs identical.

There is a legitimate alternative interpretation: freeze only the eligibility mask and treat missing-value representation as algorithm-specific preprocessing, consistent with the protocol’s statement that preprocessing belongs to the algorithm (`docs/JOINT_PROTOCOL_2026-09-03.md:162-164`). But that requires an explicit protocol amendment and changes the estimand from “estimator swap on identical values” to “algorithm with its native preprocessing.”

Recommendation: median imputation for the run of record; native XGBoost NaN handling as a labelled ablation.

The document’s “byte-identical inputs” claim at `docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:122-123` also needs qualification. Stage-2 fit populations are algorithm-dependent, so separately fitted median imputers can learn different medians even under the same nominal policy.

## P1 — Q2: Platt-scaling XGBoost

Yes, post-hoc calibration of XGBoost is defensible. XGBoost probabilities are not guaranteed to be calibrated merely because the model exposes `predict_proba`.

What is not defensible without justification is the proposed `log(p)` fallback (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:108-110`). That is an arbitrary score transformation, and it behaves especially badly with the fallback sigmoid at `train_parcel_cascade.py:269-285`.

The cleaner common rule is:

- SVM supplies its decision margins.
- XGBoost supplies its native raw multiclass margins.
- The same per-class sigmoid-fitting procedure, calibration rows, class order, fallback rule and renormalization are applied to both.

Calibration should then be assessed on data not used to fit the sigmoid. Do not describe XGBoost as “already calibrated.”

## P0 — Q3: Tuning budget

A full symmetric clean retune is necessary if the paper wants to attribute the paired difference to the algorithm.

The leaky origin of P1 does not, by itself, leak into a genuinely untouched test score: it mainly makes parameter selection unreliable. The fatal problem is asymmetric tuning opportunity. Giving XGBoost a clean per-stage search while SVM Stage 1 and most experts retain parameters selected under another procedure makes the comparison “algorithm plus tuning regime.”

Retune both arms under the same:

- parcel-grouped search protocol;
- stage-specific training population rule;
- final weighting rule;
- metric and fixed label denominator;
- number of evaluated configurations or other declared budget;
- validation/test discipline.

Matching “24 candidates × 3 folds” is not sufficient if SVM tunes only Stage 2 and orchards while XGBoost tunes every stage. Stage 1 deserves priority because it controls every downstream candidate population, but all deployed experts must receive symmetric treatment.

## P1 — Q4: Dropping `OneVsRestClassifier`

Dropping OvR is right if “XGBoost” means its native multiclass algorithm. Keeping OvR would compare SVM against an ensemble of independently trained binary XGBoost models, not ordinary multiclass XGBoost.

But the document must retract the claim that the weighting transfers “exactly” in substance (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:73-75`). The numeric row vector can be identical; its effect under a native softmax loss is not identical to its effect across independent OvR losses.

Define the control as “the same observation-level weight vector is passed to each algorithm’s native loss,” not “the same cost-sensitive learning rule.” An unweighted sensitivity arm would help determine whether the result is dominated by this loss interaction.

Also, the actual E7 Stage-2 vector is the subtype-mass vector, not the Tier-A class-weight vector.

## P1 — Q5: Algorithm-specific items in Tier A

`CROSSFIT_S1=3` belongs in Tier A. It is a leakage-control/training-topology rule. Letting arms choose different values changes how much fold-0 data each cross-fit Stage-1 model receives and changes Stage-2’s population.

`CALIB_MAX=300_000` belongs in Tier A only if both arms use the common post-hoc calibrator. If XGBoost uses native probabilities, the knob is inapplicable to that arm and the calibration design has become algorithm-specific.

Other mixed or mis-filed Tier-A entries:

- The calibration protocol can be Tier A; the estimator-specific score extraction cannot.
- The operating-point grid and selection data can be Tier A; the selected `alpha2/alpha3` should be arm-specific fitted parameters, hence Tier B.
- Missing-row eligibility is Tier A. Missing-value transformation is Tier A only under the strict estimator-swap estimand.
- The exact weight vector may be Tier A. Its effect under a particular loss is algorithm-specific and must not be claimed equivalent.
- `SKIP_TEST` is experimental governance, not cascade architecture, although it should remain a mandatory shared control.

# Attack on the tier assignments

## P0 — Tier A freezes rules but not the required row identities

The protocol requires the same ordered training pixel IDs (`docs/JOINT_PROTOCOL_2026-09-03.md:133`). The document freezes caps and seeds, not the resulting IDs.

That is inadequate because:

- Stage-2 IDs depend on algorithm-specific Stage-1 predictions (`train_parcel_cascade.py:505-557`).
- Calibration IDs are subsampled through a shared stateful RNG (`:191,313-314`).
- Cross-fit partitions and cross-fit fit samples are also generated from that RNG (`:407-415`).

There is a direct contradiction between control 5 and algorithm-dependent Stage-2 routing. The paper must choose one definition:

1. Freeze the eligible fold-0 pool and the conditional sampling procedure, while explicitly allowing downstream row IDs to be treatment-induced mediators; or
2. Require identical Stage-2 IDs, which would mean routing both arms with a frozen external/oracle router and would no longer test each algorithm’s complete cascade.

Do not silently use SVM-generated routes for XGBoost.

## P1 — Tier A conflates operating-point procedure with selected values

The selection grid, selection partition and objective should be fixed. The winning alpha values should be selected independently per arm. Forcing E7’s `0.4/0.5` onto XGBoost would be equivalent to forcing SVM-specific tuned thresholds onto a different score distribution.

The document also factually says the sweep occurs on the tuning half (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:68,146-151`), while the E7 procedure selects on the calibration half (`sweep_operating_point.py:148-160`).

## P1 — Tier A’s cost-sensitivity entry is the wrong treatment

Before deciding its tier, replace it with the actual effective E7 policy. At present it asks XGBoost to match a weighting configuration the SVM comparator did not use.

## P1 — Tier B puts `CHUNK` among algorithm choices

`CHUNK` is an implementation/resource setting, not an algorithm-arm hyperparameter. It belongs in neither scientific tier. It may vary only after verifying that probabilities and row order are unchanged (`train_parcel_cascade.py:322-336`).

## P1 — Tier B’s OvR description understates the change

Removing OvR is defensible, but not “pure overhead.” OvR determines the fitted losses, score matrix and inputs to calibration (`train_parcel_cascade.py:241-288,303`). Changing it is part of the algorithm definition and must be disclosed as such.

## P2 — Scaling and Nyström are correctly in Tier B

Dropping `StandardScaler` and `Nystroem` for native XGBoost is appropriate. They are SVM accommodations, not shared architecture (`train_parcel_cascade.py:213-229`).

# What the document missed

## P0 — It does not freeze the run-of-record decision rule

The code produces both:

- hard routing with reweighted Stage-2/3 argmax (`train_parcel_cascade.py:648-655`);
- joint probability composition with an explicit not-crop mass (`:657-670`).

E7 reports the hard rule only (`e7_final_cascade_fold2_read.py:209-225`). The handoff must explicitly freeze “hard cascade is the primary endpoint.” Otherwise either arm can choose the more favorable composition.

## P0 — The cited scoring script does not implement the parcel-held-out score

The Tier-A table cites `evaluate_end_to_end.py:97`, but that legacy script scores every NPZ row; it does not restrict evaluation to fold 2 (`evaluate_end_to_end.py:61-70,94-102`). The current cascade’s strict fold-2 scorer is at `train_parcel_cascade.py:675-685`, and E7’s is at `e7_final_cascade_fold2_read.py:218-225`.

Shipping the cited script could produce a train-inclusive result.

## P0 — Feature identity and order are absent from the Tier-A contract

The joint protocol explicitly warns that differing features convert the study into a comparison of systems rather than algorithms (`docs/JOINT_PROTOCOL_2026-09-03.md:138-164`). The handoff needs the exact NPZ hash, feature names in order, `valid_cols` hash and row ordering—not merely an NPZ filename mentioned in a future consolidated manifest.

## P1 — E4 tuned a different Stage-2 loss from the deployed one

E4’s grid search calls `gs.fit(X_pop, y_pop)` without sample weights (`e4_retune_stage2_orchards.py:114-124`). Its final Stage-2 refit uses subtype-mass weights (`:297-304`). An XGBoost search using its final weights would therefore have a different tuning protocol from the SVM reference.

## P1 — Stage-specific calibration populations are not specified

They are materially different:

- Stage 1: all `va_cal`;
- Stage 2: only rows Stage 1 predicted economic (`train_parcel_cascade.py:546-560`);
- Stage 3: true-group calibration rows, regardless of upstream success (`:575-589`).

That last rule—experts trained and calibrated by true membership—is an important architectural control missing from Tier A.

## P1 — Probability class ordering and XGBoost label encoding are unspecified

The composition assumes stable `classes_` ordering (`train_parcel_cascade.py:631-666`). Stage-3 labels are raw LU codes such as 2101 and 2420. The handoff needs an explicit reversible `0..K-1` encoding for XGBoost and assertions that returned probability columns match the declared class list. A silent column-order error would invalidate the entire cascade.

## P1 — Stateful RNG replay is not a sufficient cross-arm contract

One global generator controls fold halving, caps, cross-fit parcel allocation and calibration subsampling (`train_parcel_cascade.py:191,313-314,342-360,382-386,407-415`). Any changed call sequence changes later samples. Save and hash every actual fit/calibration index and cross-fit parcel assignment; do not rely on seed plus code-path equivalence.

## P1 — Early stopping is not governed

If XGBoost uses early stopping, its evaluation rows become part of model selection. Those rows must come from a parcel-disjoint inner split within the permitted search population—not fold 1 calibration/tuning or fold 2. The document lists `n_estimators` but does not regulate early stopping.

## P1 — The test partition is not globally untouched

The joint protocol already records that fold 2 has been observed in M5 and earlier checkpoints (`docs/JOINT_PROTOCOL_2026-09-03.md:145-154`). The new paired comparison should preferably lock a new paper test partition. Otherwise it must be described as evaluation on a previously observed partition; `SKIP_TEST=1` cannot restore global blindness.

## P2 — The proposed consolidated config is necessary but insufficient

`E7_EFFECTIVE_CONFIG.json` should additionally contain:

- the actual subtype-mass weight vector/hash;
- all fit, calibration and evaluation row hashes;
- cross-fit parcel assignments;
- feature names and order;
- class-to-column mappings;
- hard-versus-joint endpoint;
- calibration score definition and fallback;
- operating-point grid, selection partition and tie-breaking rule;
- XGBoost objective, early-stopping policy, software version and thread count.

Bottom line: proceed with the crossover, but do not package this document yet. First reconcile it with the actual E7 model, resolve the Stage-2-row/control-5 contradiction, and specify one algorithm-neutral calibration and tuning protocol.
tokens used
81,177
# P0 — The document does not describe the SVM configuration it proposes to match

This is the most serious problem. Tier A says the reference cascade uses `sqrt(n_max/n_c)` weights for Stage 2 and every Stage-3 expert (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:67,73-75`). E7 does not:

- Stage 2 uses subtype-mass weights, calculated within each crop group and renormalized to preserve group mass (`s2mass_stage2.py:197-229`; `e7_final_cascade_fold2_read.py:7-13`).
- The E4 orchards model is unweighted (`e4_retune_stage2_orchards.py:309-315`; `e7_final_cascade_fold2_read.py:14-16`).
- Plantation and field are the original unweighted M5 experts (`e7_final_cascade_fold2_read.py:17-18`).
- E7 selected `alpha2=0.4, alpha3=0.5` on the calibration half, not the tuning half (`sweep_operating_point.py:148-160`; `e7_final_cascade_fold2_read.py:140-148`).

If the collaborator implements the Tier-A table literally, the XGBoost arm will not share E7’s training loss or operating-point selection. That breaks the comparison.

The crossover idea itself is not unsound. Controlled comparisons within each architecture/dataset pair are defensible. But this document is not yet a valid execution contract.

## Verification of the six factual claims

| Claim | Verdict | Evidence |
|---|---|---|
| `base_pipe()` is the only estimator constructor | **WRONG** | It constructs the base pipeline at `train_parcel_cascade.py:213-229`, but the file separately constructs `OneVsRestClassifier` at `:303` and one `_SigmoidCalibration` per class at `:276-278`. The seam is therefore not confined to `base_pipe()`. |
| `PlattCalibrated` requires `decision_function` and cannot accept XGBoost unmodified | **CONFIRMED** | `_scores()` unconditionally calls `self.base.decision_function(X)` at `train_parcel_cascade.py:260-262`. A normal `XGBClassifier` does not expose that interface, so an adapter or calibration rewrite is required. |
| `OneVsRestClassifier` exists to route `sample_weight` past `LabelBinarizer` | **WRONG** | The wrapper is always constructed, including unweighted fits (`train_parcel_cascade.py:303-311`). The comment says the presence of OvR/`LabelBinarizer` is why a multiclass `class_weight` dictionary cannot work and why row weights are used (`:135-143`). Metadata routing delivers the weights; OvR does not exist to route them. It also defines the independent-binary learning rule and the margins being calibrated. |
| `crossfit_stage1_econ` makes Stage-2 rows depend on Stage-1 predictions and therefore the algorithm | **CONFIRMED** | Each cross-fit model predicts its held-out part at `train_parcel_cascade.py:413-417`; those predictions replace the fold-0 routes at `:505-507`; Stage-2 candidates are then `route == 1` at `:523`, with the training subset formed at `:546`. |
| P1/P23 came from leaky inner CV, while E4 used parcel-grouped `GroupKFold` | **CONFIRMED, with qualification** | The frozen provenance is explicit at `train_parcel_cascade.py:107-113`. E4 uses `GroupKFold(3)` and macro F1 at `e4_retune_stage2_orchards.py:114-124`, using `parcel_id_row` at `:287-290`. However, E4’s Stage-2 search was unweighted, while its final refit used saved subtype-mass weights (`:120-124` versus `:297-304`). It was group-clean, but it did not tune the exact final loss. |
| Both E4 winners were at the maximum of every grid axis | **WRONG** | Both used maximum `n_components=1200` and `C=30`, but `gamma=0.01666667`, corresponding to multiplier `0.5`; the gamma grid was `{0.25, 0.5, 1, 2}/n_features` (`e4_retune_stage2_orchards.py:84-87`; `runs/retune_stage2_orchards/winners.json:2-14`). Gamma was not at its maximum. Therefore the claim that the search proves both stages are categorically “under-tuned” is too strong. |

# Answers to Q1–Q5

## P1 — Q1: Missing-value policy

Control 3 binds as currently written. It requires one missing-value policy “declared once and applied identically” (`docs/JOINT_PROTOCOL_2026-09-03.md:130-132`). Keeping the same rows but changing feature values still changes the experimental input. Equal eligibility alone does not make median-imputed and native-NaN inputs identical.

There is a legitimate alternative interpretation: freeze only the eligibility mask and treat missing-value representation as algorithm-specific preprocessing, consistent with the protocol’s statement that preprocessing belongs to the algorithm (`docs/JOINT_PROTOCOL_2026-09-03.md:162-164`). But that requires an explicit protocol amendment and changes the estimand from “estimator swap on identical values” to “algorithm with its native preprocessing.”

Recommendation: median imputation for the run of record; native XGBoost NaN handling as a labelled ablation.

The document’s “byte-identical inputs” claim at `docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:122-123` also needs qualification. Stage-2 fit populations are algorithm-dependent, so separately fitted median imputers can learn different medians even under the same nominal policy.

## P1 — Q2: Platt-scaling XGBoost

Yes, post-hoc calibration of XGBoost is defensible. XGBoost probabilities are not guaranteed to be calibrated merely because the model exposes `predict_proba`.

What is not defensible without justification is the proposed `log(p)` fallback (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:108-110`). That is an arbitrary score transformation, and it behaves especially badly with the fallback sigmoid at `train_parcel_cascade.py:269-285`.

The cleaner common rule is:

- SVM supplies its decision margins.
- XGBoost supplies its native raw multiclass margins.
- The same per-class sigmoid-fitting procedure, calibration rows, class order, fallback rule and renormalization are applied to both.

Calibration should then be assessed on data not used to fit the sigmoid. Do not describe XGBoost as “already calibrated.”

## P0 — Q3: Tuning budget

A full symmetric clean retune is necessary if the paper wants to attribute the paired difference to the algorithm.

The leaky origin of P1 does not, by itself, leak into a genuinely untouched test score: it mainly makes parameter selection unreliable. The fatal problem is asymmetric tuning opportunity. Giving XGBoost a clean per-stage search while SVM Stage 1 and most experts retain parameters selected under another procedure makes the comparison “algorithm plus tuning regime.”

Retune both arms under the same:

- parcel-grouped search protocol;
- stage-specific training population rule;
- final weighting rule;
- metric and fixed label denominator;
- number of evaluated configurations or other declared budget;
- validation/test discipline.

Matching “24 candidates × 3 folds” is not sufficient if SVM tunes only Stage 2 and orchards while XGBoost tunes every stage. Stage 1 deserves priority because it controls every downstream candidate population, but all deployed experts must receive symmetric treatment.

## P1 — Q4: Dropping `OneVsRestClassifier`

Dropping OvR is right if “XGBoost” means its native multiclass algorithm. Keeping OvR would compare SVM against an ensemble of independently trained binary XGBoost models, not ordinary multiclass XGBoost.

But the document must retract the claim that the weighting transfers “exactly” in substance (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:73-75`). The numeric row vector can be identical; its effect under a native softmax loss is not identical to its effect across independent OvR losses.

Define the control as “the same observation-level weight vector is passed to each algorithm’s native loss,” not “the same cost-sensitive learning rule.” An unweighted sensitivity arm would help determine whether the result is dominated by this loss interaction.

Also, the actual E7 Stage-2 vector is the subtype-mass vector, not the Tier-A class-weight vector.

## P1 — Q5: Algorithm-specific items in Tier A

`CROSSFIT_S1=3` belongs in Tier A. It is a leakage-control/training-topology rule. Letting arms choose different values changes how much fold-0 data each cross-fit Stage-1 model receives and changes Stage-2’s population.

`CALIB_MAX=300_000` belongs in Tier A only if both arms use the common post-hoc calibrator. If XGBoost uses native probabilities, the knob is inapplicable to that arm and the calibration design has become algorithm-specific.

Other mixed or mis-filed Tier-A entries:

- The calibration protocol can be Tier A; the estimator-specific score extraction cannot.
- The operating-point grid and selection data can be Tier A; the selected `alpha2/alpha3` should be arm-specific fitted parameters, hence Tier B.
- Missing-row eligibility is Tier A. Missing-value transformation is Tier A only under the strict estimator-swap estimand.
- The exact weight vector may be Tier A. Its effect under a particular loss is algorithm-specific and must not be claimed equivalent.
- `SKIP_TEST` is experimental governance, not cascade architecture, although it should remain a mandatory shared control.

# Attack on the tier assignments

## P0 — Tier A freezes rules but not the required row identities

The protocol requires the same ordered training pixel IDs (`docs/JOINT_PROTOCOL_2026-09-03.md:133`). The document freezes caps and seeds, not the resulting IDs.

That is inadequate because:

- Stage-2 IDs depend on algorithm-specific Stage-1 predictions (`train_parcel_cascade.py:505-557`).
- Calibration IDs are subsampled through a shared stateful RNG (`:191,313-314`).
- Cross-fit partitions and cross-fit fit samples are also generated from that RNG (`:407-415`).

There is a direct contradiction between control 5 and algorithm-dependent Stage-2 routing. The paper must choose one definition:

1. Freeze the eligible fold-0 pool and the conditional sampling procedure, while explicitly allowing downstream row IDs to be treatment-induced mediators; or
2. Require identical Stage-2 IDs, which would mean routing both arms with a frozen external/oracle router and would no longer test each algorithm’s complete cascade.

Do not silently use SVM-generated routes for XGBoost.

## P1 — Tier A conflates operating-point procedure with selected values

The selection grid, selection partition and objective should be fixed. The winning alpha values should be selected independently per arm. Forcing E7’s `0.4/0.5` onto XGBoost would be equivalent to forcing SVM-specific tuned thresholds onto a different score distribution.

The document also factually says the sweep occurs on the tuning half (`docs/XGB_ARM_IN_OUR_CASCADE_2026-09-10.md:68,146-151`), while the E7 procedure selects on the calibration half (`sweep_operating_point.py:148-160`).

## P1 — Tier A’s cost-sensitivity entry is the wrong treatment

Before deciding its tier, replace it with the actual effective E7 policy. At present it asks XGBoost to match a weighting configuration the SVM comparator did not use.

## P1 — Tier B puts `CHUNK` among algorithm choices

`CHUNK` is an implementation/resource setting, not an algorithm-arm hyperparameter. It belongs in neither scientific tier. It may vary only after verifying that probabilities and row order are unchanged (`train_parcel_cascade.py:322-336`).

## P1 — Tier B’s OvR description understates the change

Removing OvR is defensible, but not “pure overhead.” OvR determines the fitted losses, score matrix and inputs to calibration (`train_parcel_cascade.py:241-288,303`). Changing it is part of the algorithm definition and must be disclosed as such.

## P2 — Scaling and Nyström are correctly in Tier B

Dropping `StandardScaler` and `Nystroem` for native XGBoost is appropriate. They are SVM accommodations, not shared architecture (`train_parcel_cascade.py:213-229`).

# What the document missed

## P0 — It does not freeze the run-of-record decision rule

The code produces both:

- hard routing with reweighted Stage-2/3 argmax (`train_parcel_cascade.py:648-655`);
- joint probability composition with an explicit not-crop mass (`:657-670`).

E7 reports the hard rule only (`e7_final_cascade_fold2_read.py:209-225`). The handoff must explicitly freeze “hard cascade is the primary endpoint.” Otherwise either arm can choose the more favorable composition.

## P0 — The cited scoring script does not implement the parcel-held-out score

The Tier-A table cites `evaluate_end_to_end.py:97`, but that legacy script scores every NPZ row; it does not restrict evaluation to fold 2 (`evaluate_end_to_end.py:61-70,94-102`). The current cascade’s strict fold-2 scorer is at `train_parcel_cascade.py:675-685`, and E7’s is at `e7_final_cascade_fold2_read.py:218-225`.

Shipping the cited script could produce a train-inclusive result.

## P0 — Feature identity and order are absent from the Tier-A contract

The joint protocol explicitly warns that differing features convert the study into a comparison of systems rather than algorithms (`docs/JOINT_PROTOCOL_2026-09-03.md:138-164`). The handoff needs the exact NPZ hash, feature names in order, `valid_cols` hash and row ordering—not merely an NPZ filename mentioned in a future consolidated manifest.

## P1 — E4 tuned a different Stage-2 loss from the deployed one

E4’s grid search calls `gs.fit(X_pop, y_pop)` without sample weights (`e4_retune_stage2_orchards.py:114-124`). Its final Stage-2 refit uses subtype-mass weights (`:297-304`). An XGBoost search using its final weights would therefore have a different tuning protocol from the SVM reference.

## P1 — Stage-specific calibration populations are not specified

They are materially different:

- Stage 1: all `va_cal`;
- Stage 2: only rows Stage 1 predicted economic (`train_parcel_cascade.py:546-560`);
- Stage 3: true-group calibration rows, regardless of upstream success (`:575-589`).

That last rule—experts trained and calibrated by true membership—is an important architectural control missing from Tier A.

## P1 — Probability class ordering and XGBoost label encoding are unspecified

The composition assumes stable `classes_` ordering (`train_parcel_cascade.py:631-666`). Stage-3 labels are raw LU codes such as 2101 and 2420. The handoff needs an explicit reversible `0..K-1` encoding for XGBoost and assertions that returned probability columns match the declared class list. A silent column-order error would invalidate the entire cascade.

## P1 — Stateful RNG replay is not a sufficient cross-arm contract

One global generator controls fold halving, caps, cross-fit parcel allocation and calibration subsampling (`train_parcel_cascade.py:191,313-314,342-360,382-386,407-415`). Any changed call sequence changes later samples. Save and hash every actual fit/calibration index and cross-fit parcel assignment; do not rely on seed plus code-path equivalence.

## P1 — Early stopping is not governed

If XGBoost uses early stopping, its evaluation rows become part of model selection. Those rows must come from a parcel-disjoint inner split within the permitted search population—not fold 1 calibration/tuning or fold 2. The document lists `n_estimators` but does not regulate early stopping.

## P1 — The test partition is not globally untouched

The joint protocol already records that fold 2 has been observed in M5 and earlier checkpoints (`docs/JOINT_PROTOCOL_2026-09-03.md:145-154`). The new paired comparison should preferably lock a new paper test partition. Otherwise it must be described as evaluation on a previously observed partition; `SKIP_TEST=1` cannot restore global blindness.

## P2 — The proposed consolidated config is necessary but insufficient

`E7_EFFECTIVE_CONFIG.json` should additionally contain:

- the actual subtype-mass weight vector/hash;
- all fit, calibration and evaluation row hashes;
- cross-fit parcel assignments;
- feature names and order;
- class-to-column mappings;
- hard-versus-joint endpoint;
- calibration score definition and fallback;
- operating-point grid, selection partition and tie-breaking rule;
- XGBoost objective, early-stopping policy, software version and thread count.

Bottom line: proceed with the crossover, but do not package this document yet. First reconcile it with the actual E7 model, resolve the Stage-2-row/control-5 contradiction, and specify one algorithm-neutral calibration and tuning protocol.
