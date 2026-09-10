# Disposition of the Codex Review of 2026-09-04

Reviewed document: `docs/CODEX_REVIEW_2026-09-04_JOINT_PROTOCOL_AND_REPORTS.md`.
All eleven findings were checked against the code and the run artifacts before being acted on.
**All eleven are accepted.** None is disputed. No numeric result was changed anywhere; every edit
is to wording, structure, or protocol design.

## Verification performed before accepting

| Claim | How it was checked | Verdict |
|---|---|---|
| `label_47PQQ_buffered.tif` is the eroded artifact, not `label_47PQQ.tif` | `config.py:114-115`, `buffer_labels.py:39-40`, `align_indices_labels.py:10` — the aligner's default `ALIGN_LABEL` is the buffered raster | **Confirmed. My error.** |
| The split claim was stale | E7 runs on `splits/parcel_id_row.npy` / `split_assign.npy`; only earlier SVM work and the current XGBoost workflow split at pixel level | **Confirmed. My error.** |
| E7 does not use the proposed common features | `runs/s2_2018_3date_parcel_e7_final/manifest.json` names `svm_s2_3date_m3_features_labels.npz`, the 30-feature set | **Confirmed** |
| The architecture factor was incoherent | My own section 2 described the collaborator's system as a three-stage rejection cascade while my section 3 matrix called it "flat 13-class" | **Confirmed, and internally self-contradictory** |
| E7 measurements reproduce | `report_hard.csv` and `run2.log:75` give 0.2429 / 0.7949 / 5,500,269; M5 gives 0.2344 | **Confirmed, numbers unchanged** |
| The collaborator's standalone numbers are a CV partition | `train_crops.py:34-37,55,72` — reports on `y_cv`, and `x_test`/`y_test` are never evaluated | **Confirmed, and it strengthens the finding — see below** |
| Zero-support macro arithmetic | 4.80/13 = 0.3692; 4.80/12 = 0.400; (4.80+0.50)/14 = 0.379 | **Confirmed** |

## A further correction, raised by the user after this review

The user pointed out that **the 2 September report's numbers come from the old pipeline, not the
repaired one.** This is confirmed by the commit history: the per-model extraction split landed in
`095da3e` and `16086c1` on 2026-09-01, with the crop half in `992ec12` and `32c8813` on 2026-09-02,
and `full_pipeline.py` in `f5f1a6a` the same day. The report analyses the pipeline those commits
replaced.

Two consequences, both now applied:

1. **0.38 must never be quoted as "the XGBoost result".** It is the score of a configuration its
   own author has already diagnosed and replaced, and the repaired pipeline has not been scored
   yet. The current XGBoost system's performance is simply unknown.
2. **The population question is answered, and I had been reading the wrong code.** I based the
   "23 times smaller, unexplained" question on the *new* extractors, which apply no cap. The
   number came from the *old* `extract_all.py`, which drew a reservoir sample capped at 200,000
   per class with a seeded `default_rng(42)`, built the single shared water-plus-crop table, and
   then deleted rows with missing values. Rubber at 197,590 is that cap, not a mystery. The open
   question is now the narrower and answerable one of what population the *repaired* pipeline
   evaluates.

This also refines finding P1-9: both of his reported results are drawn from capped samples, and
the standalone-versus-pipeline difference is a 20 percent CV partition against a full inference
sample, with different missing-value rules applied to each.

## One addition to finding P1 (PDF causal isolation)

Codex reported that the collaborator's standalone support totals about 304,081 rows against the
pipeline's 755,295, so the pipeline population is the **larger** of the two. The code explains
why, and makes the point sharper than "the chart does not show a general collapse":

`train_crops.py` splits 60/20/20 and reports `classification_report` on the **20 percent
cross-validation partition**. The pipeline report, by contrast, scores a full inference
population. So the two support charts compare a one-fifth slice against a whole population. That
is not a like-for-like support comparison in either direction, and the nine classes that fall did
so despite the denominator being roughly 2.5 times larger.

This is recorded as question 2 in the protocol's section 8.

## Disposition table

| # | Finding | Disposition | Where |
|---|---|---|---|
| P0-1 | Neither claimed cell satisfies the controls | **Accepted.** All four cells marked *to be run*. E7 and the 2 Sept result demoted to motivation/baselines, explicitly barred from contributing to any estimated effect | protocol §3; prof report §12; companion §22 |
| P0-2 | Architecture factor incoherent | **Accepted, option 1 taken.** The two levels are now *semantic routing cascade* vs *sequential rejection cascade*, both implemented with both algorithms. Every cell emits the same 13 crops + `others` | protocol §3; prof report §12; companion §22 |
| P0-3 | Three cells is not a factorial | **Accepted.** Four aligned cells are now required for the words *main effect* and *interaction*; the three-cell fallback is renamed *controlled pairwise comparisons* with reduced claims. The earlier "drop cell C" advice is retracted in text | protocol §3; prof report §12; companion §22 |
| P1-4 | Label artifact misidentified | **Accepted.** `label_47PQQ_buffered.tif` is now the artifact of record wherever erosion is required, with the raw raster named as its source | protocol §5 control 6, §10; companion §22 item 5 |
| P1-5 | Split description stale | **Accepted.** Now states the SVM side is already parcel-disjoint and the XGBoost side needs migrating, while noting earlier SVM work had the same defect | protocol §5 control 4; companion §22 item 4 |
| P1-6 | Row eligibility / missing values not frozen | **Accepted.** New controls 2 and 3: one shared grid and linear pixel ID, one evaluation-row mask over the **common** features only, saved and hashed; a declared missing-value policy; and proof that every cell predicts the same ordered test pixel IDs | protocol §5 controls 2–3, §7; companion §22 item 2 |
| P1-7 | Training population not held constant | **Accepted.** New control 5: corresponding cells get the same ordered training pixel IDs including any cap; an uncapped XGBoost arm is permitted only as a labelled data-volume ablation; evaluation stays uncapped | protocol §5 control 5; companion §22 item 3 |
| P1-8 | E7's fold is not globally untouched | **Accepted.** "Read once" is now everywhere qualified as *within the E1–E7 plan*, with fold 2 described as a previously observed partition. The protocol requires choosing between locking a new paper partition or reporting the exposure honestly; this is also now a question for the professor | protocol §5 test-partition caveat; prof report §14 table + §15 q7; companion §16, §22 item 8 |
| P1-9 | PDF does not isolate the cause | **Accepted and extended.** NaN deletion is now "one identified mechanism", causation requires an old-vs-repaired comparison on frozen pixel IDs, and the CV-vs-full-population confound is documented. Further refined after the user's old-pipeline correction above | protocol §8 questions 2–4; companion §21 |
| P1-10 | SVM report overstates certainty once | **Accepted.** "The improvement is real … on a fold I had not read" is replaced by an *observed test-fold gain* of 0.0085 that is explicitly not separated from the ±0.014 parcel-bootstrap interval, with the weighted-F1 decrease from 0.7974 to 0.7949 surfaced in the same paragraph | prof report §4; companion §16 |
| P2-11 | G3 mean is not a noise floor | **Accepted.** The bullet now states that **no noise floor for per-crop E7 deltas has been estimated**, and explains that G3's nonzero mean is its treatment effect while its spread is pool-draw sensitivity for that experiment only | E7 report §"per-crop" bullet |
| P2-12 | Zero-support macro naming ambiguous | **Accepted.** New §6 tabulates all three candidate statistics with denominators and fixes the naming rule; the appendix now labels 0.2429 as "13 evaluable" and 0.3692 as "13, one of them zero-support" | protocol §6, appendix |

## Modified documents

| File | Change |
|---|---|
| `docs/JOINT_PROTOCOL_2026-09-03.md` | Substantially rewritten. §3 architecture factor redefined and all cells marked pending; §5 grew from six controls to eight; new §6 on macro naming; new §7 execution manifest; §8 expanded to seven open questions; §10 and appendix corrected |
| `docs/PROFESSOR_PROGRESS_REPORT_2026-08-27.md` | §4 certainty wording and weighted-F1 trade-off; §12 matrix and architecture levels rewritten with both earlier mistakes explicitly retracted in the text; §14 partition caveat and next-steps item 2; §15 new question 7; title updated |
| `docs/PROFESSOR_PROGRESS_REPORT_2026-08-27_TECHNICAL_COMPANION.md` | §16 observed-gain wording, weighted-F1 note, and a "what read once means" paragraph; §22 architecture levels, pending cells, three-cell rule, and two new controls with the list renumbered to nine items |
| `docs/REPORT_2026-08-28_E7_FINAL_READ.md` | The per-crop bullet's "noise floor" claim retracted and corrected in place |

## Not done, and why

**The execution manifest is specified but not implemented.** Protocol §7 lists the required
fields, and no cell can be quoted without one. Building the tooling that emits and checks those
hashes is work for when the protocol is agreed, not before — the field list may change in
negotiation, and implementing against an unagreed spec would be wasted.

Nothing under `2018/`, and no trained model, run artifact, split array, label raster, or reported
measurement, was modified.
