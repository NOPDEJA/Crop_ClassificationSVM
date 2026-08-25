# Handoff — review M0 through M3 of the competitiveness plan

**For:** a fresh Claude Fable 5 session, reviewing adversarially.
**Written:** 2026-08-25.
**Subject:** commits `7b5122f..8c1be72` on `worktree-implement-competitive-plan`, which
implement `docs/PLAN_S2_3DATE_COMPETITIVE.md` §2.1, all five §3 repairs, and M0–M3.

You wrote the plan and the handoff that started this. That handoff said to assume an
error of the same class is still present and go looking for it. That instruction stands,
turned around: **two numbers below have already been retracted after being written down
as findings, and the second was caught only because the first made me check.** Assume a
third.

---

## 1. What happened, in order

| step | what | outcome |
|---|---|---|
| verify | reproduced the handoff's §2 repro | exact match: strict 0.2248/0.8018, masked 0.2405/0.8309, 278,123 dropped FPs |
| §3.4 | swept the other scorers, the handoff's "biggest untouched gap" | **3 of 6 had the defect**, not the zero it guessed |
| §2.1, §3 | implemented all five repairs | smoke-tested, then M0 |
| M0 | prerequisite run, 5h38m | 0.2283 vs baseline 0.2248 |
| M1 | calibration audit | no stable `π_true` exists |
| M2 | operating-point sweep | +0.0112 macro F1, 7 → 10 crops alive |
| M3 | MTCI + raw B11, 24 → 30 columns | matrix built; ablation in flight |

**Not done:** M4 (capacity), M5 (retrain + the single fold-2 read), M6 (deferred by
default). Fold 2 has been read exactly once since M0, by M0 itself.

---

## 2. The three claims most worth attacking

### 2.1 "The in-sample routes were hiding 10.3% of Stage 2's contamination"

Same fold-0 rows, same M0 Stage-1 model, only the routing differs:

| routing | Stage-2 candidates | sink share |
|---|---|---|
| in-sample | 9,806,157 | 15.657% |
| out-of-fold | 9,944,240 | 17.030% |

Route agreement 94.27%. **Re-derive** (the out-of-fold routes were not persisted by M0 —
that is fixed for M5 but you must reconstruct the in-sample side):

```python
import numpy as np
CROPS=[2101,2204,2205,2302,2303,2403,2404,2405,2407,2413,2416,2419,2420]
G={1:[2403,2404,2407,2413,2416,2419,2420],2:[2302,2303,2405],3:[2101,2204,2205]}
y=np.load('./aligned_features/svm_s2_3date_features_labels.npz',allow_pickle=True)['y'].astype(np.int32)
asg=np.load('splits/split_assign.npy'); tr=np.flatnonzero(asg==0)
s1=np.load('runs/s2_2018_3date_parcel_m0/stage1_pred.npy')
g=np.zeros(y.size,np.int32)
for k,c in G.items(): g[np.isin(y,c)]=k
g[g==0]=4
ins=tr[s1[tr]==1]
print(ins.size, (g[ins]==4).mean())      # expect 9,806,157 and 0.15657
```

The out-of-fold side (9,944,240 / 17.030%) comes from `run.log`, not from an array.
**That asymmetry is a weakness**: one side is recomputed, the other is read off a log line.
If you distrust it, the only fix is rerunning with the current script, which now saves
`stage1_route_oof_train.npy`.

### 2.2 "The protocol repairs cost no accuracy"

Paired parcel bootstrap, 400 replicates, both runs on the identical fold 2 (5,500,269
rows / 5,824 parcels): baseline 0.2248, M0 0.2283, **delta +0.0032, CI [+0.0001, +0.0070]**.

I deliberately did **not** call this a gain. Attack that both ways:

- Too weak? The CI excludes zero and P(M0 > baseline) = 0.975.
- Too strong? The lower bound is +0.0001. A tail percentile from 400 replicates is itself
  noisy, and I did not bootstrap the bootstrap. 400 was chosen for runtime, not by any
  precision argument.

Also: M0 differs from baseline in **two** model-relevant ways at once — halved calibration
set and cross-fitted Stage-2 candidates — so even a real delta is not attributable.

### 2.3 "M2 buys +0.0112 macro F1 and three crops"

(α₂, α₃) = (0.2, 0.7) on the tuning half: 0.2067 → 0.2179, crops alive 7 → 10.
`runs/s2_2018_3date_parcel_m0/m2_sweep.csv` has all 169 cells.

The weak point is what "alive" means. I set the threshold at F1 ≥ 0.01, which is
arbitrary and very low. Mangosteen goes 0.0014 → 0.0437 and Jackfruit 0.0015 → 0.0290 —
real movement, but from numbers that round to zero to numbers that round to zero. **Is
"three crops off the floor" a result or a presentational choice?** I think it is worth
reporting with the raw numbers attached, which is what the plan does, but I would not
defend the phrase "alive" hard.

---

## 3. Two claims I already retracted — check I retracted them correctly

Both were written into the plan as findings before being caught.

1. **"An exponent tuned on one parcel sample may not transfer."** Written into M1's
   consequence 2. M2 then measured it: the two disjoint halves rank all 169 cells at
   **correlation 0.9936** and pick adjacent cells, transfer cost −0.0014. The surface is a
   plateau, not a peak. Now struck through in place with the correction beside it. *Check
   the correction is not itself overstated — 0.9936 is one pair of halves of one fold.*
2. **"Langsat prevalence swings 25.6× between parcel samples."** Led M1's table. It rests
   on **4 / 9 / 191 pixels** and roughly one parcel per fold — small-count noise presented
   as evidence. The M1 table is now split by support, and the finding rests on Oil palm
   (4.9× on 42,868–109,693 px), Mango (3.7×) and Rice (2.9×). *Check the surviving rows
   really do carry it, and that I have not simply moved the goalposts to whichever rows
   still look good.*

---

## 4. Where the reasoning is weakest — push here

1. **M1's central claim may prove too much.** "There is no stable `π_true`" is doing heavy
   work: it reframes M2 from prior correction to operating-point tuning, and it partly
   retires the `prior-correction-sweep-regime` memory. But the evidence is prevalence
   varying between three parcel samples *of a single tile in a single year*. An alternative
   reading is mundane: crops are spatially clustered, parcels are the sampling unit, and
   three draws of ~5,800 parcels will differ — which says the folds are small, not that
   the prior is undefined. Those have different consequences. Decide which, and whether
   M2's reframing survives the weaker reading. **I think this is the most likely place I
   am still overclaiming.**
2. **M2's denominator change was mine, not the plan's.** The plan said to divide by the
   training prior. I argued that double-counts, because M0 calibrates on natural-prior
   rows and M1 shows ratios ≈1. So I divided by the calibration prior toward a uniform
   target instead. If that reasoning is wrong, M2's entire surface is the wrong object and
   (0.2, 0.7) is predeclared for M5 on a bad basis. Check the argument, not just the code.
3. **M3's ablation confounds gamma with features.** `P23` sets `gamma=None` → `1/n_features`,
   so 24 → 30 columns also moves gamma 0.0417 → 0.0333. I let it, on the grounds that M5
   makes the same substitution so the probe reproduces the real decision. That is a defence
   of the *decision*, not of the *measurement*: if 30 columns win, I will not know whether
   MTCI helped or gamma did. M4 is supposed to separate them and has not run.
4. **The M0-vs-baseline comparison I discarded.** I first compared M0's sink share against
   the baseline run's (14.43% → 16.19%) and then discarded it as confounded by the
   calibration change, replacing it with the within-run version in §2.1. Check that the
   within-run version is actually clean and that I have not repeated the same error in a
   subtler form.
5. **Nothing here has been checked against fold 2.** Every number above is fold-1 or
   fold-0. M2's operating point is predeclared but untested. That is the protocol working
   as designed, but it means the plan currently has *no* validated end-to-end improvement.

---

## 5. Verified — re-check only if cheap

| claim | how | result |
|---|---|---|
| handoff §2 repro | rerun verbatim | exact ✓ |
| scorer defect blast radius | read all six scorers | 3 of 6 ✓ |
| masking inflation | masked vs strict column, 256 cells | +0.0344, argmax moves 1 cell ✓ |
| Amendment 4 | `min(N_c, PER_LU_CAP)` vs run log | 172,371 / 148,344 / 210,000 exact ✓ |
| M0 folds vs baseline | manifest | identical ✓ |
| Stage-1 cap refactor | dist after caps vs baseline | byte-identical ✓ |
| cal/tune halves | disjoint, union = fold 1, parcel-disjoint | ✓ |
| all 13 crops in both halves | counted | ✓ (Langsat 4 cal / 9 tune) |
| M3 row identity | `y` reproduces from label raster | 24,323,769 exact ✓ |
| MTCI physical range | median 2.40–2.64, p95 4.50–4.90 | ✓ |

## 6. Ground truth

- Branch `worktree-implement-competitive-plan`, pushed. Plan with all results inline:
  `docs/PLAN_S2_3DATE_COMPETITIVE.md` (M0/M1/M2 RESULT blocks).
- Runs: `runs/s2_2018_3date_parcel_m0/` (M0 + `m1_*.csv` + `m2_sweep.csv`);
  baseline `runs/s2_2018_3date_parcel/`.
- Scripts: `m1_calibration_audit.py`, `m2_operating_point.py`, `m3_build_features.py`,
  `m3_ablation_orchards.py`, and the repaired `train_parcel_cascade.py`.
- Interpreter is `C:\Conda_environment\envs\svm_env\python.exe`; the system Python 3.13
  has numpy but no sklearn or pandas.

## 7. What a good review returns

1. Whether §2.1's asymmetric derivation (array vs log line) is acceptable or must be rerun.
2. A ruling on §4.1 — does M1 prove "no stable prior", or only "small folds"?
3. Whether §4.2's denominator argument holds, since M5 is predeclared on it.
4. Whether "three crops alive" survives contact with the F1 ≥ 0.01 threshold.
5. The third retraction. Two are listed in §3. Find the one I have not noticed.
