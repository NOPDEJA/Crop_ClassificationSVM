# Handoff for Codex: the ceiling verdict on RF-paper competence

> **Reviewed by Codex 2026-08-26 — all six findings applied in place; §8 records the
> dispositions.** Net outcome: the data-limited warning is **sustained**, the categorical
> ceiling is **overturned** to "no credible reason to expect", the 0.25–0.28 forecast is
> **withdrawn** as a predicted interval, and the RF comparison is reframed as an unaudited
> protocol-sensitivity discussion. §4's list now carries Codex's execution order; the runnable
> version is `docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md`.

**Date:** 2026-08-26
**Scope:** the root-level three-date S2-only parcel-disjoint cascade (`runs/s2_2018_3date_parcel*`),
validation artifacts only. No fold-2 read is proposed anywhere in this document except P6, which
is a single predeclared read at the end of the plan.
**What is being reviewed:** a verdict, not a run. The professor's benchmark is the prior RF paper
(flat 15-class Random Forest, Sentinel-2, 2024+2020 epoch), whose orchard / rare-class F1 is about
0.33 (per-class table quoted from the paper, not present in this repo — verify against the PDF if
you have it). The question decided here is whether any plan on the current dataset gets our rare
classes to that level. My original answer — "no under an honest protocol, yes-but-meaninglessly
under theirs" — was **overstated in both halves**; the post-review answer is: *current evidence
gives no credible reason to expect it under the planned S2-only interventions, and the RF paper's
own number is unaudited rather than meaningless.*

Everything below uses the strict scoring convention (full population, non-crop truth mapped to 0)
unless a row says otherwise. All artifact paths are relative to the repo root.

---

## 1. State as of this handoff, verified

Tree clean at `f2a3cb9`. Everything in `docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md` executed; all
six delta-review findings applied (`docs/CODEX_DELTA_REVIEW_2026-08-27.md`). The shipping report is
`docs/REPORT_2026-08-27.md`.

| quantity | value | artifact |
|---|---|---|
| M5 strict test macro F1 (5,500,269 px) | **0.2344** | `runs/s2_2018_3date_parcel_m5/report_hard.csv` |
| whole year of gains, baseline → M5, test fold | +0.0096 | report §2 |
| M5 tune-half macro F1 | 0.2294 | `opsweep_selected_tune.csv` |
| subtype-mass treatment, tune half | 0.2375 (**+0.0081**; +0.0069 at fixed cell, all 169 cells favour it) | `runs/s2_2018_3date_parcel_s2mass/s2mass_summary.csv` |
| oracle Stage-2 routing, tune half | 0.3785 (**+0.1491**; +0.1452 remains after treatment) | `runs/s2_2018_3date_parcel_m5/oracle_routing.csv` |
| M5 under collaborator protocol (14 labels) | 0.3078 | `collaborator_protocol_rescore_summary.csv` |
| isolated probe, pixel split → parcel split | 0.5852 → **0.3945** | `runs/probe_dry_season/per_class_parcel_grouped.csv` |
| parcel-half level gap at identical settings | ~0.048 | report §5 |
| tempered class weights (gate) | lost, 0.2272 vs 0.2294 | report §4 |
| tree-merge probe | lost, 0.2158 vs 0.2283 | report §4 |

---

## 2. The verdict, stated as attackable claims

### V1 — corrected: severe protocol sensitivity is demonstrated on our data; the RF paper itself is unaudited

The RF paper's evaluation is a pixel-level split (parcels shared between train and test) on a
prior-matched ~303,947-pixel population. Our own data now measures what that measuring stick is
worth: replaying our isolated probe's pixel split with parcel IDs shows 87.5–95.7 % of rare-crop
test parcels also sit in training (`probe_replay_overlap.csv`), and moving the same probe from
pixel split to parcel split costs the rare crops 0.18 to 0.68 of F1 each:

| crop | F1, pixel split (their kind of protocol) | F1, parcel split (honest) |
|---|---|---|
| Mango | 0.6150 | 0.4350 |
| Coconut | 0.6329 | 0.3782 |
| Rambutan | 0.6092 | 0.3407 |
| Mangosteen | 0.5385 | 0.2753 |
| Jackfruit | 0.4540 | 0.2510 |
| Longan | 0.5543 | 0.2357 |
| Langsat | 0.6774 | **0.0000** |

~~"We already exceed 0.33 under their protocol"~~ — **struck by review, and the review is right.**
The pixel-split probe is a uniform-prior, crop-only, five-date experiment with no non-crop false
positives; the RF study is a flat 15-class evaluation on a different epoch and population. Those
are not equivalent protocols, so the 0.45–0.68 column cannot be read as beating their number. What
stands is narrower: **severe protocol sensitivity is demonstrated on our dataset**, and the RF
paper's generalization performance is **unaudited**, because its split cannot be reconstructed
here. Public paper metadata confirms overall F1 ≈ 0.71 and oil-palm F1 ≈ 0.81; the quoted
rare-class ~0.33 and the paper's splitting method could not be independently verified.

Paper wording to carry (Codex's formulation, adopted verbatim): *"Our experiment demonstrates
severe protocol sensitivity on our dataset. The RF paper's generalization performance remains
unaudited because its split cannot be reconstructed here."* The leaky/matched score belongs in an
appendix or protocol-sensitivity section — never beside the honest score as equivalent evidence.

### V2 — corrected: no credible reason to expect the five dead crops to reach strict F1 0.33 under the planned S2-only interventions

The five are Coconut, Longan, Langsat, Rambutan, Mangosteen (Jackfruit borderline-below, Mango is
the one plausible exception — see V3). Three independent bounds, each conditional in a different
way, which is exactly why I want them attacked jointly:

**(a) The frozen-cascade oracle bound.** Freeze every model and grant Stage 2 perfect routing
(`oracle_routing.csv`, tune half):

| crop | learned route F1 | **oracle route F1** |
|---|---|---|
| Rice | 0.4471 | 0.7732 |
| Durian | 0.4643 | 0.8187 |
| Cassava | 0.3365 | 0.6446 |
| Pineapple | 0.4071 | 0.6615 |
| Mango | 0.0348 | 0.3004 |
| Rambutan | 0.0508 | 0.1752 |
| Jackfruit | 0.0313 | 0.0595 |
| Mangosteen | 0.0312 | 0.0504 |
| Coconut | 0.0013 | **0.0289** |
| Longan | 0.0029 | **0.0030** |
| Langsat | 0.0000 | 0.0000 |

Even with the single largest known defect solved *perfectly*, the five dead crops stay below 0.18
and mostly below 0.06. This bound is conditional on the frozen Stage-3 experts, which carry their
own subtype imbalance, so it is not a true ceiling — a joint routing+Stage-3 fix could exceed it.
That is why bound (b) exists.

**(b) The isolated-probe generalization ceiling.** The parcel-grouped probe is close to a best-case
setting: uniform prior, balanced evaluation population, five dates instead of three, 800 Nyström
components, no cascade to compound errors, no non-crop false positives charged. Its rare-crop
ceiling on unseen parcels is the parcel-split column in V1: 0.24–0.44, with only Mango, Coconut and
Rambutan clearing 0.3. For a rare class, moving from that balanced population to the natural tile
at a fixed classifier can only lower precision (the negatives multiply ~50–1,000×; recall is
unchanged), so balanced-probe F1 is an optimistic bound on natural-population F1 — and the strict
convention then adds non-crop false positives on top. This is an argument, not a theorem: the
cascade is a different classifier and the report itself forbids probe-vs-cascade *comparisons*. I
am using the probe as a *bound under a strictly more favourable protocol*, not as a comparison.
Attack that distinction if it does not hold.

**(c) The prevalence arithmetic.** Strict F1 0.33 with recall 0.5 needs precision ≥ 0.246. Worked
on the test fold (5,500,269 px):

- **Coconut** (2,947 true px): predicted-positive budget ≈ 5,990 px, so ≤ ~4,500 false positives
  across ~5.50 M non-coconut pixels — a false-positive rate of ~8×10⁻⁴, i.e. the plantation expert
  may emit coconut on at most ~1 in 700 of the 3.24 M rubber pixels. Today 83 % of predicted oil
  palm is truly rubber; the same redistribution pressure applies to coconut.
- **Langsat** (191 true px): ≤ ~290 false positives in 5.5 M — one stray Langsat per ~19,000
  pixels. The old cascade over-predicted Langsat **168×**. And with 8 test parcels, any Langsat
  number is noise before it is evidence (report limitation 4).

No plausible fit-side intervention changes these rates by the required orders of magnitude while
the training support is 10–150 parcels per crop with one parcel holding 80 % of Langsat's pixels.

**Review outcome: sustained as a warning, overturned as a ceiling.** None of the three bounds is
valid as a ceiling. (a) is conditional on the frozen Stage-3 experts. (b) is favourable evidence
but not an upper bound on a *different* classifier under class-conditional distribution shift.
(c) fixes recall at 0.5, whereas p = 0.33r/(2r − 0.33) admits, for example, r ≈ 0.20 at
p ≈ 0.94 — a precision-oriented operating point with a very different false-positive budget that
nothing here excludes — and no strict-population precision–recall curve for a jointly retrained
routing/expert system exists. The sentence this project now carries is: **current evidence gives
no credible reason to expect the five rare crops to reach strict F1 0.33 under the planned
S2-only interventions.** It does not, and cannot honestly, say that no configuration can.

### V3 — corrected: the reachable headroom is mid-frequency; the numeric range is a planning scenario, not a forecast

The oracle deltas concentrate in crops with parcels to learn from: Durian +0.354, Rice +0.326,
Cassava +0.308, Mango +0.266, Pineapple +0.255. The subtype-mass result already behaves this way:
two-thirds of its +0.0081 came from oil palm and rice. Any gains from the plan in §4 will be
driven by rice/cassava/durian/oil-palm/mango routing, not by rare-class recovery; Durian and
Mango may individually clear 0.33 strict.

**Review withdrew my original "0.25–0.28" as a predicted interval, and the arithmetic is why.**
Reaching 0.25 from 0.2344 needs +0.203 of summed per-crop F1 across 13 crops; reaching 0.28 needs
+0.593. The only measured winner is +0.0081, on a different validation population, from an
unreplicated pool draw, and the learnable fraction of the +0.1452 oracle headroom is entirely
unmeasured. The numbers this project now carries are **base scenario 0.24–0.26, with 0.28 as
explicit upside** — planning figures to be tested by the single fold-2 read at the end of the
execution plan, not an expectation. For calibration: the year bought +0.0096 and the best single
intervention bought +0.0081 on a tune half.

### V4 — For Langsat and Longan, measurement binds before the model

Langsat: 1,639 training px from 10 parcels (one parcel 1,310 px), 7 tune px from 2 parcels, 191
test px, 8 probe test parcels. Longan — **corrected by review; my draft misattributed Langsat's
13 fold-1 validation pixels to it** — has 3,500 fold-1 px from 18 parcels (calibration half
2,525 px / 9 parcels, tune half 975 px / 9 parcels) and 45 probe test parcels: poorly supported
at parcel level, but not measurement-limited to Langsat's degree, so this claim now binds hard
only for Langsat and by degree for Longan. Two disjoint parcel halves of one fold differ by
~0.048 macro at identical settings — one observed difference, which demonstrates instability but
is not itself a variance estimate — so even a true +0.05 gain on one of these crops is not
reliably observable on this tile. Any plan that claims it will "fix"
these classes is unfalsifiable with the data we hold; the report already says get more parcels or
stop reporting them per class (§8.2), and I am asking you to confirm that stance is right.

### V5 — corrected: data limitation is the leading diagnosis, not yet an exoneration of the architecture

The year's interventions sort cleanly: everything that redistributes attention among existing
parcels (class weights −0.0022, tree merge −0.0125, soft routing −0.0015, SMOTE ruled out at 97.9 %
same-parcel neighbours) fails or barely moves; the one success (+0.0081) worked by moving fitting
mass toward crops that *had parcels to learn from*. Meanwhile every protocol dial (decision rule
+0.015, evaluation population +0.03–0.04, pixel-vs-parcel split +0.19 on the probe) dwarfs every
model dial. That pattern is what "data-limited" looks like from inside.

**Review outcome: softened to "leading diagnosis".** The architecture is not exonerated: the
honest retune is unrun, the fused-stack probe is unrun, the class-weight loss carries unmeasured
full-cascade run variation, and only a narrow family of SVM interventions has been tested. E2 and
E4 in the execution plan are exactly the tests that could shift this diagnosis.

### V6 — What would genuinely move the rare classes (all scope changes, none in the current plan)

1. **More parcels** — other LDD survey years or adjacent tiles. The cross-year transfer study says
   the model reads imagery rather than memorizing locations (−0.067 acc at 2 yr), so cross-year
   parcels are plausible training material, with label-epoch mismatch as the risk to design around.
   This is a data request to the professor / LDD, not an experiment.
2. **Sensor fusion** — the DEM+S1+S2 stack exists in this repo but has never been probed under a
   parcel-grouped split. Falsifier F1 in §5 tests it cheaply. If SAR/terrain lifts the probe
   ceiling for coconut/mangosteen, the "spectral ceiling" is really an "optical ceiling" and the
   fused pipeline becomes the LDD delivery story.
3. **Parcel-majority aggregation** — changes the unit of analysis to what LDD actually consumes;
   likely lifts rare-crop scores substantially but must never be compared to pixel-level figures
   (report §8.6).

---

## 3. Consequence of the verdict for the paper, if it survives

1. Report the RF comparison as a **protocol study**: our matched-protocol number (0.3078 under the
   collaborator's; the RF-matched analogue if we agree one) beside the honest 0.2344, with the
   probe-collapse table as the demonstration of why the sticks disagree.
2. The improvement narrative targets mid-frequency routing (V3), stated as such.
3. Rare five: reported with support caveats, framed as data-limited with V2's three bounds, and the
   data request of V6.1 raised at the meeting.

---

## 4. The plan I would run regardless of the verdict

This is the delegation candidate (a Sonnet-5 session would execute it) once your feedback lands.
Order is by information per hour. Gates follow house rules: predeclare before running, select on
the calibration half, score once on the tune half, fold 2 untouched except P6.

**Review reordered this list, and the reordering is adopted:** P7 → F1 (fused probe, the cheapest
direct falsifier of the central verdict, promoted to second) → P4 → P1 → P2 (factorial against
the P1 winner) → P3 (demoted: date differences are deterministic linear combinations of existing
columns, so they add no information — they alter feature scaling and the kernel metric, making P3
a feature-weighting experiment with lower expected value than first stated) → P6. The executable
version with full specifications is `docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md`; the table
below keeps its original numbering for traceability.

| # | experiment | cost | gate / read | expected size (attack these) |
|---|---|---|---|---|
| P1 | **Honest retune**, GroupKFold on parcel ID, `scoring='f1_macro'`, Stage 2 and orchards expert only; C, gamma, components jointly | overnight | tune-half paired vs M5 Stage-2/expert swap-in | unknown; the current values came from a leaky accuracy-scored search, and every prior search hit its capacity ceiling, so +0.00 to +0.03 |
| P2 | **Stage-3 subtype mass**, same machinery as `s2mass_stage2.py` but inside the plantation and orchards experts (coconut's constraint moved there: routing 13.7→26.6 %, F1 flat) | ~1 h | same paired design | +0.00 to +0.01 |
| P3 | **Date-difference features** (Oct−Nov, Nov−Dec deltas of MSAVI/BSI/EVI/NDVI), probed first in a parcel-grouped expert probe, promoted only if the probe pays | hours | probe gate ≥ +0.01 macro on probe | +0.005 to +0.02 at probe level; rubber-defoliation signal is the mechanism |
| P4 | **Pool-draw sensitivity** of the +0.0081: repeat the paired s2mass fits over ≥3 fresh 200k draws | ~2 h | report spread, no gate | decides whether P6 carries the weights |
| P5 | falsifier F1 below | ~1 h | see §5 | — |
| P6 | **One consolidated cascade** with every surviving winner, single predeclared fold-2 read — the paper's headline | ~3.5 h + read | predeclared before launch | lands V3's 0.25–0.28 or refutes it |
| P7 | hygiene: `train_parcel_cascade.py` persists Stage-2/3 fit and calibration indices (already accepted in report §8.1) | minutes | n/a | prevents another pool-identity argument |

Deliberately excluded, with reasons already on record: pixel SMOTE (97.9 % same-parcel neighbours),
equalize-down-to-Langsat, per-crop manual thresholds (untunable at 7 tune px), soft routing
(lost), tree merge (lost), any second fold-2 read.

---

## 5. Falsifiers for the verdict, ranked

**F1 — Fused-stack parcel-grouped probe.** Rerun the grouped-probe design on the DEM+S1+S2
feature matrix, same crop-wise parcel halving, same uniform prior. If
coconut/mangosteen/rambutan/longan move materially above their S2-only ceiling (say +0.10 F1),
V2(b) weakens and the fused pipeline becomes the rare-class story.

**Corrected artifact instructions (review finding, verified locally 2026-08-26):** use
`aligned_features/svm_dem_s1_s2_features_labels.npz` — 153 named features, zero VVVH_DIFF
columns, 24,323,769 rows matching `splits/parcel_id_row.npy`, labels equal to the S2-only npz —
with `aligned_features/_unpacked/X_src.npy` already extracted as (24,323,769 × 153) float32 for
memmap use. The older `svm_add_data_features_labels.npz` named in my draft has no
`feature_names` key, so duplicate columns cannot be dropped by name; do not use it. Keep explicit
label-array and parcel-length equality assertions in the probe regardless.

**F2 — P1 overshoots.** If the honest retune alone moves the orchards expert by more than ~+0.03
tune macro, the "capacity was never the binding constraint" premise weakens and V3's landing zone
is too low.

**F3 — LDD data.** If the professor can supply the 2020 or 2024 parcel survey for 47PQQ (or a
neighbouring tile), V2 becomes testable with real added support instead of bounds. This is the only
falsifier that could actually rescue the five dead crops.

---

## 6. What I want attacked, in order

1. **V2's joint logic.** Each bound is conditional: (a) freezes Stage 3, (b) transfers a balanced
   bound to a natural population across a different classifier, (c) assumes fit-side interventions
   cannot buy 10³ of false-positive rate. Is there a configuration consistent with all three that
   still reaches 0.33 strict on coconut, longan or langsat? If you find one, the verdict falls.
2. **The 0.25–0.28 landing zone** in V3. It is extrapolated from one +0.0081 tune-half result, an
   unmeasured fraction of +0.1452 oracle headroom, and an unrun retune. Tell me if it should be
   narrower, lower, or refused as unquantifiable.
3. **V1's transfer inference** — is "their 0.33 would collapse like ours did" defensible enough to
   print, or must it stay as "our protocol demonstration, their number unaudited"?
4. **The protocol-parity reporting proposal** (§3.1). Reporting our model under a leaky protocol,
   even flagged, arguably launders the protocol. Is the two-sticks table honest or should the
   matched number be relegated to an appendix?
5. **P1–P7 ordering and gates**, especially whether P4 belongs before P6, and whether F1's row
   alignment risk is handled adequately.
6. Anything in §2 that repeats the class of error already caught twice (pixel-leak probes read as
   generalization, level differences read as bias). I have tried to pre-empt both; verify.

Do not re-litigate settled findings (the withdrawal, the optimism correction, the subtype-mass
result). Never read fold 2. Ranked findings with re-derivation evidence, most severe first, and
state explicitly which claims you verified clean.

---

## 7. Re-derivation recipes

- **Oracle bound per crop:** `runs/s2_2018_3date_parcel_m5/oracle_routing.csv` (columns:
  `f1_learned_route`, `f1_oracle_route`; MACRO row 0.2294 → 0.3785).
- **Probe ceiling and leakage:** `runs/probe_dry_season/per_class_parcel_grouped.csv`,
  `probe_replay_overlap.csv`; scripts `probe_dry_season_grouped.py`, `probe_replay_overlap.py`.
- **Prevalence arithmetic:** F1 = 2pr/(p+r) with r = 0.5 and F1 = 0.33 gives p = 0.246; predicted
  positives = 0.5·support/0.246; false positives = that minus 0.5·support; rate = FP / (5,500,269 −
  support). Supports from `report_hard.csv` per-class rows.
- **Parcel supports:** the crop-pixels-and-parcels-by-split recipe in
  `docs/RESEARCH_2026-08-26_RARE_CROP_NEXT_STEPS.md` §"Re-derivation recipes".
- **Protocol dials:** decision rule and population rows in `docs/REPORT_2026-08-27.md` §3;
  collaborator rescore in `collaborator_protocol_rescore_summary.csv`.

---

## 8. Codex review outcome, 2026-08-26 — dispositions

| # | finding | severity | disposition |
|---|---|---|---|
| 1 | V1 compared non-equivalent protocols; "we already exceed 0.33 under their protocol" unsupported; RF paper's rare-class figure and split method unverifiable | Critical | **applied** — V1 rewritten to "protocol sensitivity demonstrated locally, RF paper unaudited"; Codex's paper wording adopted verbatim; matched scores relegated to appendix status |
| 2 | V2 establishes "unlikely", not "impossible": no bound is a valid ceiling, and a precision-oriented point (r ≈ 0.20, p ≈ 0.94) is not excluded | Critical | **applied** — V2 retitled and closed with the "no credible reason to expect" formulation; categorical ceiling withdrawn |
| 3 | V4 misattributed Langsat's 13 fold-1 validation px to Longan (Longan: 3,500 px / 18 parcels; cal 2,525/9; tune 975/9); 0.048 is instability, not a variance estimate | High | **applied** — V4 corrected on both points |
| 4 | 0.25–0.28 refused as a quantitative forecast (+0.203 to +0.593 summed per-crop F1 required; only +0.0081 measured) | High | **applied** — V3 now carries base 0.24–0.26 / explicit upside 0.28 as a planning scenario only |
| 5 | V5 does not isolate dataset from architecture (retune unrun, fused probe unrun, run-variation unmeasured, narrow intervention family) | Medium | **applied** — softened to "leading diagnosis" |
| 6 | Plan reorder (P7 → F1 → P4 → P1 → P2 factorial → P3 demoted → P6); stale fused-npz instructions; P3 is a feature-weighting experiment | Medium | **applied** — §4 and §5 updated; npz facts re-verified locally (153 named features, rows match `parcel_id_row.npy`, `X_src.npy` is the extracted matrix) |

Verified clean by Codex: the oracle-routing numbers, the probe collapse, the leakage shares, the
exact paired +0.0081317 winning all 169 cells, the coconut arithmetic at r = 0.5, Langsat's
support facts, P4-before-P6, and that no new run or fold-2 read occurred.

**Overall: the data-limited warning is sustained, the categorical ceiling is overturned, the
forecast is withdrawn, and the RF comparison becomes an unaudited protocol-sensitivity
discussion.** The plan survives with the reordering and executes as
`docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md`.

*Prepared by the Claude session of 2026-08-26 after the subtype-mass run and probe rebuild;
reviewed by Codex the same day; corrections applied in place with the original wrong wordings
noted where instructive.*
