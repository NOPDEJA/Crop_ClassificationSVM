# Predeclaration — Stage-2 subtype mass

Written 2026-08-26, **before either fit was launched**, per §3 of
`docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md`. Nothing below is edited after the fact.

## What is being run

Two Stage-2 fits, **control** and **treatment**, differing only in the D3 sample weights.
Stage 1 and Stage 3 are frozen from `runs/s2_2018_3date_parcel_m5`. No stage is refitted
except Stage 2, and the training pool is M5's own pool, replayed rather than re-drawn.

The weight is `w(row) = sqrt(m_maxc / m_c)`, where `m_c` is the post-cap row count of the
row's crop subtype **within its group** and `m_maxc` the largest subtype count in that
group, then renormalised within each group so the group's total mass stays exactly its row
count. Sink rows keep weight 1. Group balance is therefore untouched and only the
distribution inside each group moves.

## Pre-launch checks, all passed before launching

Pool identity, against M5's saved arrays and its `run.log`:

| check | result |
|---|---|
| fold-1 halving byte-identical to M5 | PASS |
| Stage-1 fit distribution `{1: 1000000, 2: 482151, 3: 800000, 4: 600000}` | PASS |
| crossfit fit/route sizes `[(2849354, 3468903), (2746302, 6609230), (2867194, 4764161)]` | PASS |
| Stage-2 sink training rows 1,677,101 | PASS |
| validation candidates byte-identical to M5 | PASS |
| Stage-2 pool 800,000 rows, 200,000 per group | PASS |

D4's mass check (`mass_table.csv`), the lesson of the weighted run. All three named groups
move, so this is not a vacuous arm:

| group | crop | rows | share before | share after | weight |
|---|---|---|---|---|---|
| orchards | Durian | 133,481 | 66.74 % | 38.99 % | 0.584 |
| orchards | Langsat | 1,179 | 0.59 % | 3.66 % | 6.216 |
| plantation | Rubber | 193,381 | 96.69 % | 82.56 % | 0.854 |
| plantation | Oil palm | 6,463 | 3.23 % | 15.09 % | 4.671 |
| plantation | **Coconut** | **156** | **0.078 %** | **2.34 %** | **30.064** |
| field | Rice | 28,774 | 14.39 % | 22.48 % | 1.562 |
| sink | (non-crop) | 200,000 | 100 % | 100 % | 1.000 |

## Metrics

Tune-half strict `crop_macro_f1` at the cal-selected cell, per-crop F1, and Stage-2 routing
accuracy per crop. Strict convention: the whole tuning half, non-crop truth mapped to 0, the
13 crop labels fixed. Selection of (α₂, α₃) happens on the calibration half and the selected
cell is scored **once** on the tuning half, through `sweep_operating_point.py`, the same code
path that produced M5's 0.2294.

## Reading rules, fixed in advance

1. The **noise floor** is `|control − M5| = |control − 0.2294|`.
2. The **effect** is `treatment − control`.
3. The effect is **claimed only if it exceeds the noise floor**.
4. **Fold 2 is not read regardless of the outcome.** Not for the control, not for the
   treatment, not if the treatment wins. The test read is worth more after the honest
   hyperparameter retune, and the headline must not churn hours before the meeting. The
   professor can overrule this at the meeting.
5. If the control lands more than ~0.005 from 0.2294, the pool replay is wrong and nothing
   is reported from either arm until that is diagnosed.

## Expected secondary movement

If the mechanism is real, the Stage-2 routing accuracy of coconut (13.7 % at M5), mango
(11.0 %) and Langsat (28.6 %) should rise, and rubber's (85.4 %) may fall. **Both directions
get reported**, whichever way the headline goes.

## What would make this a negative result

Treatment inside the noise floor of the control, or worse than it. In that case the +0.1491
oracle-routing headroom is real but is not reachable by reweighting Stage-2's loss, and the
next suspect is the calibration of Stage-2's probabilities, since oracle routing bypasses
them entirely.
