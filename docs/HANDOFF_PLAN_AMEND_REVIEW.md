# Handoff — review the amended competitiveness plan

**For:** a fresh Claude Fable 5 session, reviewing adversarially.
**Written:** 2026-08-25.
**Subject:** commit `a27ea2e` on `worktree-plan-amend-s2-3date`, which amends
`docs/PLAN_S2_3DATE_COMPETITIVE.md` (previously `9477bbd` on
`worktree-plan-competitive-s2-3date`).

Your job is not to admire this. One numeric claim in the project's memory was
wrong for a whole day before anyone re-derived it, so assume the same class of
error is still present somewhere below and go looking for it.

---

## 1. What happened, in order

The plan at `9477bbd` was drafted, Codex-reviewed, revised and committed. It was
then checked by re-deriving every number in its §0 from the run artifacts rather
than trusting the write-ups. That check found:

- the plan's own figures were **correct** (0.2248, 0.8018, 0.3965/0.5520, the RF
  macro recomputation, all four `train_parcel_cascade.py` code gaps, the Langsat
  13-row figure, MTCI feasibility);
- but the project **memory** `parcel-split-costs-nothing` was **wrong**, claiming
  the parcel-disjoint run beat the leaky one by ~0.016 when like-for-like it is a
  tie;
- and the plan had three presentational defects that would have leaked
  non-comparable numbers into an email or a paper.

The memory has been rewritten (outside the repo, at
`C:\Users\Nop\.claude\projects\D--MINEWORK-...\memory\parcel-split-costs-nothing.md`)
and three amendments were applied to the plan in `a27ea2e`.

## 2. The central claim to attack first

**Claim:** `parcel_agg.py` computes precision on a masked population, inflating
macro F1 from 0.2248 to 0.2405, and the resulting "+0.016 gain from parcel
disjointness" is an artifact.

**Mechanism:** the script does `m = (asg == 2) & np.isin(y, CODES)` and subsets
*before* its `prf()` counts false positives as `((yp==c) & (yt!=c))`. Non-crop
test rows predicted as a crop are therefore invisible to it. Both
`train_parcel_cascade.py` (`labels=CROPS` on `y_eval[te]`, non-crop truth mapped
to 0) and `evaluate_end_to_end.py:97` use the stricter convention.

**Re-derive it yourself** (needs only numpy; run from the repo root):

```python
import numpy as np
CODES = np.array([2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420])
y    = np.load('./aligned_features/svm_s2_3date_features_labels.npz', allow_pickle=True)['y'].astype(np.int32)
pred = np.load('runs/s2_2018_3date_parcel/pred_hard.npy')
asg  = np.load('splits/split_assign.npy')
te   = asg == 2

def prf(yt, yp):
    out = []
    for c in CODES:
        tp = ((yp==c)&(yt==c)).sum(); fp = ((yp==c)&(yt!=c)).sum(); fn = ((yp!=c)&(yt==c)).sum()
        p = tp/(tp+fp) if tp+fp else 0.; r = tp/(tp+fn) if tp+fn else 0.
        out.append((2*p*r/(p+r) if p+r else 0., (yt==c).sum()))
    return out

A = prf(np.where(np.isin(y,CODES), y, 0)[te], pred[te])     # strict
m = te & np.isin(y, CODES); B = prf(y[m], pred[m])          # parcel_agg.py's way
for nm, r in (('strict', A), ('masked', B)):
    n = sum(x[1] for x in r)
    print(nm, 'macro=%.4f weighted=%.4f' % (sum(x[0] for x in r)/13,
                                            sum(x[0]*x[1] for x in r)/n))
print('non-crop rows predicted as crop:', int((te & ~np.isin(y,CODES) & np.isin(pred,CODES)).sum()))
```

Expected: `strict macro=0.2248 weighted=0.8018`, `masked macro=0.2405
weighted=0.8309`, and `278123` dropped false positives (7.7% of crop
predictions). If you get anything else, the amendment is wrong — say so loudly.

## 3. Where this reasoning is weakest — please push here

1. **Is 0.2248 vs 0.2245 a comparison at all?** They come from different test
   populations: fold 2's 3,800,567 crop rows for the parcel run, versus the whole
   tile's 15,157,223 for the leaky v2 (`end_to_end_report.csv`). The amended plan
   calls the tie "loose" and still leans on it as consequence 4. A defensible
   alternative reading is that the two numbers are simply **not comparable**, that
   the honest statement is "we have no measurement of what leakage cost", and that
   consequence 4 should be softened further or cut. Decide which, and say why.
   This is the single most likely place the amendment is still overclaiming.
2. **RF Table V rounding.** The macro figures (0.602 over 13 crops, 0.625 over 15)
   are recomputed from per-class F1s printed to two decimals, so each carries
   roughly ±0.005. The plan states them as "≈". Check the arithmetic — the 13 crop
   F1s should sum to 7.83 and all 15 to 9.38 — and check the underlying
   assumption that the paper's Table V lists *test* performance while Table IV's
   0.7139 is also test (it does say so, but confirm).
3. **Does the weighted F1 belong in the plan at all?** Rubber is 85.1% of the
   crop-13 test support, so 0.8018 mostly restates rubber's 0.8767. The amendment
   keeps the claim but attaches the share and demotes it to "the weakest claim in
   the plan". A reasonable reviewer might say a number that misleading should not
   be quoted even with a caveat, because caveats do not survive being repeated.
4. **Is §3.4's rewrite over-fitted to one script?** It generalises from a single
   bug in `parcel_agg.py` to a rule for all ad-hoc scorers. Check whether the
   other analysis scripts (`rescore_collaborator_protocol.py`,
   `audit_parcel_disjoint.py`, `diagnose_error_budget.py`,
   `apply_prior_correction.py`, `sweep_prior_alpha.py`) actually share the defect.
   If none of them do, the rule is still right but the framing overstates the
   blast radius. **This has not been checked** — it is the biggest untouched gap
   in the review, and it is the natural first thing for you to do.

## 4. A fourth issue, found but deliberately NOT amended

The user asked for three amendments, so this was left out of `a27ea2e`. It should
probably become a fourth.

M2 says the exponent sweep must not use "the old balanced-regime constants in
`sweep_prior_alpha.py`". That is right, but the stated reason is imprecise, and
the precise version matters:

- **Stage 2 is fine.** `sweep_prior_alpha.py:124` asserts `pi_train` is uniform
  because Stage 2 saw "exactly 140,000 per subclass". The parcel run used
  `PER_GROUP_CAP = 200_000` and its log line 74 shows a Stage-2 base fit on
  exactly 800,000 rows = 4 × 200,000. The cap bound uniformly, so `pi_train` is
  uniform in both runs and the *ratio* is unaffected. Only the comment's number
  is stale.
- **Stage 3 is the real hazard.** The script reads per-class training counts from
  an artifact (`ratio3_den`, lines 121-122) rather than assuming uniformity —
  which is correct behaviour, but those counts describe the **v2** run. In the
  parcel run the Stage-3 caps did *not* bind uniformly: orchards 172,371 and
  plantation 148,344 rows (log lines 90, 93), against field's 210,000 = 3 ×
  `PER_LU_CAP`. Reusing v2's counts file would silently apply the wrong
  denominator to two of the three experts.

Verify both from `runs/s2_2018_3date_parcel/run.log` and `config.py:50-63`, then
decide whether M2 should name the artifact-reuse hazard explicitly.

## 5. Things already verified — re-check only if cheap

| claim | verified how | result |
|---|---|---|
| parcel run macro/weighted | recomputed from `report_hard.csv` | 0.2248 / 0.8017-8 ✓ |
| leaky v2 macro | `end_to_end_report.csv` macro avg row | 0.2245 ✓ |
| capped rescore | `collaborator_protocol_rescore_summary.csv` | 0.3965 / 0.5520 ✓ |
| RF macro 13 / 15 | summed Table V F1s ÷ 13, ÷ 15 | 0.6023 / 0.6253 ✓ |
| RF 0.714 is weighted | its recall 0.7157 == accuracy exactly | ✓ |
| val probs never saved | only `*_prob_test.npy` in the run dir | ✓ gap real |
| no train–val assertion | `train_parcel_cascade.py:227-228` only tr/te, va/te | ✓ gap real |
| train/val parcels disjoint anyway | `np.intersect1d` → 0 | ✓ so it is hardening |
| fold-1 dual use | `fit_calibrated(..., cal_idx=va)` for all 3 stages | ✓ gap real |
| Stage-2 in-sample routes | `train_parcel_cascade.py:262` comment + `fit1 ⊂ fold 0` | ✓ gap real |
| Langsat val rows | counted from `split_assign.npy` | 13 ✓ |
| MTCI buildable | `B04/B05/B06` in both `BAND_MAPPING`s | ✓ |
| `CONTEXT.md` cascade error | lines 39, 52-53 | 3 lines, not 1 ✓ |

## 6. Ground truth locations

- Amended plan: `docs/PLAN_S2_3DATE_COMPETITIVE.md`, commit `a27ea2e`, branch
  `worktree-plan-amend-s2-3date` (pushed). Parent is `9477bbd`.
- Buggy script: `C:\Users\Nop\.claude\jobs\3f1055b7\tmp\parcel_agg.py` — outside
  the repo, still present, still unfixed. Do not fix it silently; it is the
  evidence.
- Codex's review of the *first* draft:
  `C:\Users\Nop\.claude\jobs\98e0719d\tmp\codex_review.txt`.
- RF paper text and Codex's review of it:
  `C:\Users\Nop\.claude\jobs\3f1055b7\tmp\codex_rf_review.md` (Table V is quoted
  verbatim near the end, so the paper PDF is not needed).
- The corrected memory: `parcel-split-costs-nothing.md` in the project memory dir.

## 7. What a good review returns

Not a verdict on whether the plan reads well. Specifically:

1. Confirmation or refutation of §2's repro, with numbers.
2. A decision on §3.1 — keep, soften, or cut consequence 4.
3. The result of the §3.4 sweep over the other ad-hoc scorers.
4. Whether the §4 Stage-3 denominator hazard warrants a fourth amendment.
5. Anything in the plan that is still stated more confidently than its evidence
   supports. That was the original sin here and it will be the next one.
