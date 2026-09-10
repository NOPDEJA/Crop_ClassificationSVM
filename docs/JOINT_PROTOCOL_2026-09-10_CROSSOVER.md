# Joint Paper Protocol, revision 2: the crossover

Rayong crop classification, 2018 LDD survey, tile 47PQQ.
Drafted 2026-09-10. **Supersedes `docs/JOINT_PROTOCOL_2026-09-03.md`**, which should be read
only for the history of how the design got here.

---

## 0. What changed from the 3 September protocol, and why

The previous protocol proposed a **2x2 factorial**: {SVM, XGBoost} x {routing cascade,
rejection cascade}, all four cells run on a **common 15-column feature set** built from the
collaborator's indices. That design is withdrawn. Two reasons, in order of weight.

1. **It required four new runs before anything could be said.** No existing result was a
   cell, by the previous protocol's own admission. Four aligned runs across two machines and
   two codebases, with a shared feature extraction neither side had built, is not a plan that
   produces a paper this year.
2. **Rule (A) does not actually need it.** The professor's condition is *different algorithm,
   same architecture and same controls*. A single pair satisfying that condition satisfies
   the rule. The factorial was an attempt to answer a larger question — whether routing beats
   rejection — that nobody asked for and that the data cannot cleanly support anyway, since
   the two architectures were developed on different feature sets.

**The replacement is a crossover.** Each side implements the *other* side's algorithm inside
its *own* architecture on its *own* dataset, and each pair is compared only within itself.

| pair | architecture | dataset | algorithms compared | runs where |
|---|---|---|---|---|
| **P1** | our routing cascade | our 30-column S2 3-date arm | SVM vs XGBoost | XGBoost half on his machine |
| **P2** | his rejection cascade | his 15-column dataset | XGBoost vs SVM | SVM half on our machine |

Within a pair, the architecture, the dataset, the split, the labels, the scoring convention
and the population are all held fixed, and exactly one thing varies: the algorithm and the
preprocessing that algorithm needs. That is rule (A), twice.

**What we give up, stated plainly.** The crossover cannot estimate an architecture main
effect or an interaction, because the architecture is confounded with the dataset. The paper
must therefore not use the words *factorial*, *main effect* or *interaction*. It reports **two
controlled algorithm comparisons**, and any statement about routing versus rejection is
qualitative and clearly labelled as such.

**What we gain.** Each side keeps its own contribution and its own data pipeline, no shared
feature extraction has to be built, and each pair is independently publishable even if the
other never finishes.

## 1. The unit of comparison

A **cell** is one algorithm inside one architecture on one arm. A **pair** is two cells
differing only in algorithm. Only cells within a pair may appear in the same table.

Our E7 run is cell P1-SVM: strict macro F1 **0.2429** over 5,500,269 fold-2 rows, hard
routing. It is the comparator for P1-XGB, and it is used **as already run**. We are not
re-running it to match the XGBoost cell, because there is nothing about the XGBoost cell that
requires it to change.

His repaired-pipeline result is **not** cell P2-XGB and must not be quoted as one. It is
scored on a population that is 100% crop pixels with `others` support 0, and the score
appears to be in-sample. Those two questions are still open and are restated in section 6.

## 2. Controls, restated for the crossover

Within a pair, these are fixed. Between pairs, nothing is claimed.

| # | control | value |
|---|---|---|
| 1 | Evaluation population | the arm's natural held-out partition, uncapped. Capping for training is permitted; capping for scoring is not. |
| 2 | Frozen evaluation rows | both cells of a pair must predict the same ordered row identities, proven by hash, not asserted |
| 3 | Missing-value policy | declared once per pair and applied to both cells. No rows dropped in either. |
| 4 | Split | parcel-grouped in P1. P2 needs the same migration on his side, or the pair is measured on ground the models have partly seen. |
| 5 | Training rows | **the same rule, not the same rows.** See section 3. |
| 6 | Labels and class list | the eroded raster of record; 13 crops plus `others`, ordered and explicit |
| 7 | Scoring | strict: full population, non-crop truth mapped to 0, explicit label list, no masking, denominator stated |
| 8 | Reported partition | one partition, fixed in advance, with its prior exposure stated honestly |
| 9 | Endpoint | fixed in advance per pair. In P1 it is **hard** routing. |
| 10 | Tuning budget | declared, and equal per stage across the two cells of a pair |

Controls 9 and 10 are new. Both exist because a review found they were unfixed and either
side could have chosen the favourable option after seeing results.

## 3. The one control that had to change: training rows

The 3 September protocol required **the same ordered training pixel IDs** in corresponding
cells. In a cascade that is not achievable, and pretending otherwise would have produced a
contract nobody could satisfy.

Our Stage 2 trains only on pixels our Stage 1 routed to it. An XGBoost Stage 1 will route a
slightly different set. So the two cells' Stage-2 training rows **cannot** be identical unless
one cell is fed the other's routes, which would mean its own Stage 1 was never tested.

The control is therefore restated as a **framework rule**:

> Stage 2 trains on whatever *your own* Stage 1 routed to it. The rule is fixed and identical
> in both cells; the rows it produces are a consequence of the algorithm, and are saved and
> hashed rather than forced to match.

The rows that *are* frozen: the fold-0 eligible pool, every cap, the cross-fit parcel
partition, the fold-1 calibration and tuning halves, and the evaluation rows. What floats is
strictly downstream of the treatment, which makes it a mediator, not a confound. Every run
saves `*_fit_idx.npy` and `*_cal_idx.npy` with SHA-256 hashes so the difference is measurable.

## 4. What is architecture and what is algorithm

The line is drawn in code, in one file, so that it is checkable by diff rather than by
argument. `cascade_algo.py` is the entire seam. If something is not in that file, it is
architecture and both cells share it.

**Algorithm (his to choose, in P1):** the estimator; the scaler and Nyström kernel map, which
exist only so a linear SVC can approximate an RBF kernel and are therefore dropped rather than
forced on a tree model; the one-vs-rest wrapper, dropped because keeping it would compare
against an ensemble of independent binary XGBoost models rather than XGBoost; all
hyperparameters; and the cell's own operating point.

**Architecture (shared):** the split, the caps, the cross-fitted routing, the calibration, the
operating-point *procedure*, the endpoint, the scorer.

**Calibration is architecture, not algorithm**, and this is the subtle one. The cascade's
decision rule is an argmax over calibrated probabilities and the operating-point sweep divides
by the calibration prior, so a downstream architectural component consumes the calibrator's
output. Both cells therefore go through the same per-class Platt procedure on the same rows.
What differs is only the raw score fed in: the SVM supplies its one-vs-rest decision margins,
XGBoost supplies its pre-softmax multiclass margins. Neither cell uses `predict_proba`
directly, and no log transformation is applied to probabilities to manufacture a score.

## 5. Asymmetries we are disclosing rather than resolving

1. **Our arm is a tuning hybrid; his will be cleaner.** Only Stage 2 and the orchards expert
   were searched honestly under parcel-grouped cross-validation. Stage 1, plantation and field
   remain frozen at values from a randomized search whose inner cross-validation leaked. The
   XGBoost cell searches all five stages. So the XGBoost cell receives **more** clean tuning
   than ours, not less. Until the matching clean SVM retune runs, P1 measures *algorithm plus
   tuning regime*. That retune is planned and is the next thing on our side after P2.
2. **The weight vector transfers; its effect does not.** Both cells receive the same
   observation-level weight vector. Its effect under XGBoost's native softmax loss is not
   identical to its effect across 13 independent binary problems. The control is worded
   accordingly.
3. **Fold 2 is not globally untouched.** Our M5 and earlier checkpoints read it, and E7 read
   it once more. `SKIP_TEST=1` prevents *new* exposure but cannot restore blindness. Either we
   lock a genuinely new partition before the XGBoost cell runs, or the paper describes fold 2
   as a previously observed partition. This is a decision, and it must be made before section
   7 step 1, not after.
4. **Determinism.** XGBoost with histogram tree method and multiple threads is not
   bit-reproducible. Accepted and declared, rather than forced to single-thread.

## 6. Still open on his side

Unchanged from the previous protocol, and still the gate on anything involving his numbers.

1. **Why is `others` support 0?** The 13 crop supports sum to exactly the reported total, so
   the scored population contains no non-crop ground and crop precision cannot be charged for
   a single false positive off-crop. Control 7 requires it back in the denominator.
2. **Is the reported score in-sample?** `infer_crops.py` reads the whole extracted table,
   predicts on all of it, and reports; the crop model was fitted on a draw from the same tile.
   If so the figures are fit quality, not generalization, and the rare crops are the classes
   most affected.
3. The erosion asymmetry, the survivor denominator, and the pixel-level split all remain as
   previously described.

## 7. Execution

**P1, in progress.** We supply the pipeline, the data and a consolidated configuration
contract; he runs five steps documented in `docs/HANDOFF_XGB_CASCADE.md`. The budget is 24
candidates x 3 parcel-grouped folds per stage.

**P2, ours, after P1 ships.** We implement the rejection cascade with an SVM on his dataset,
under his controls, and report it beside his XGBoost number.

**Then:** the clean SVM retune that closes asymmetry 1.

## 8. Decisions requested

1. Do we accept the crossover in place of the 2x2, and accept that the paper therefore
   reports two controlled algorithm comparisons rather than a factorial?
2. Do we lock a new test partition before the XGBoost cell runs, or describe fold 2 as
   previously observed? **This one is time-critical.**
3. Is 24 candidates x 3 folds per stage accepted as the declared budget for both cells?
4. Does he accept the framework rule in section 3 in place of identical training rows?
