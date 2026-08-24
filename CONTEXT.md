# Context

Shared vocabulary for the LDD crop-classification work. This is a glossary, not a
spec — it says what words mean, never how anything is implemented.

## Study

**Tile** — the single Sentinel-2 MGRS tile 47PQQ, covering Rayong province, Thailand.
All work is scoped to this tile.

**Epoch** — the year a set of observations and labels belongs to. Three epochs exist:
2018, 2020, 2024. Imagery and parcel survey must always come from the same epoch;
mixing them silently measures land-use change instead of model quality.

**Parcel survey** — LDD's field-surveyed land-use polygons for one epoch, named by the
Buddhist-era year: *2561* = 2018, *2563* = 2020, *2567* = 2024. The survey is the only
source of ground truth.

**Label** — the crop or land-cover code attached to a pixel, derived from the parcel
survey. A **buffered label** is one whose parcel boundary was contracted inward before
rasterization, so pixels near a parcel edge are excluded; this removes mixed pixels
where a parcel boundary and a real crop boundary disagree.

## Features

**Index** — a spectral quantity computed from Sentinel-2 bands for one date
(NDVI, EVI, NDWI, BSI, NDBI, MSAVI, SWIR_NIR, SWIR_RATIO).

**Feature** — one column of the training matrix: an index at a specific date, or a
terrain or radar quantity. A feature is identified by name, and the *order* of features
is part of the model — a model can only be applied to features in the order it saw.

**Window** — the set of dates an arm draws on. The **3-date window** is Oct/Nov/Dec;
the **5-date window** adds Mar and Apr. The window is a property of the arm, and every
epoch an arm is applied to must supply the same window.

**Texture feature** — a feature derived from a pixel's neighbourhood rather than the
pixel alone (e.g. the mean or variance of a 3×3 window). Texture features carry spatial
information. **Neither study uses any** — this entry previously said the collaborator's
model did, which was checked against `github.com/Gunkartan/geospatial` and is wrong.
Their columns are NDVI, EVI, NDWI (MNDWI form), MTCI and raw long-SWIR, all per-pixel.
Texture is therefore an untried direction for both arms, not a difference between them.

## Models

**Arm** — one experimental configuration: a feature set, a window, an epoch, and a
**split scheme**, trained end to end. Arms are named `<features>_<epoch>_<window>`, e.g.
`s2_2018_3date`. Two arms differing in exactly one property are *comparable*; arms
differing in more than one are not, and their numbers must not be placed in the same
table as if they were. The split scheme is part of an arm's identity because a score is
a property of a model *and* the population it was measured on: two arms split differently
are not comparable even when every other property matches.

**Cascade** — a sequence of models where each one only sees the pixels a previous model
routed to it. **Only this project's SVM is a cascade.** The collaborator's crop model is
a flat 14-class `XGBClassifier` with no routing at all, so there are no stages on their
side to correspond to ours; comparisons must be made end to end, never stage by stage.
An earlier version of this entry called their model a cascade too, and that error made
a stage-by-stage comparison look meaningful when it never was.

**Stage** — one model within the cascade. Stage 1 assigns a superclass; Stage 2 assigns
a subclass within economic crops; Stage 3 assigns the final crop code within a subclass.

**Superclass** — the four coarse classes Stage 1 separates: economic crops, water,
forest, others.

**Subclass** — the grouping Stage 2 assigns within economic crops: orchards, plantation,
field. Its purpose is *routing*, not reporting.

**Routing** — sending a pixel to the next stage's model based on the current stage's
prediction. A pixel routed wrongly cannot be recovered later, because the model it
reaches was never trained on its true class. This is **error propagation**, and it is
the cost a cascade pays for specialization.

**Hard routing** — routing by taking the single highest-scoring branch and committing to
it. **Joint routing** — scoring every branch and combining the probabilities along each
path, so no branch is discarded early. The two are different decision rules over the same
trained models, which makes a comparison between them a clean one: nothing else varies.

**Sink** — a class whose purpose is to absorb what does not belong to any other class,
rather than to name something. Stage 2's fourth subclass is a sink: it receives whatever
Stage 1 called economic that is not one of the 13 reported crops. Without one, every such
pixel is forced to become a crop it is not, which is a manufacturer of false positives.

**Calibration** — fitting a transformation that turns a model's raw scores into
probabilities. Calibration is only meaningful relative to a population: probabilities
fitted on a rebalanced sample describe that sample's class frequencies, not the world's.

**Run** — one execution of a stage producing artifacts. All artifacts of all stages for
one arm live in a single run directory, named after the arm.

## Evaluation

**Population** — the set of pixels a score is computed over. A metric is meaningless
without one, and most of this project's measurement errors have been comparisons between
two differently-defined populations rather than between two models.

**Fitted population** — the pixels any stage used to fit or calibrate a model. A score
computed over pixels drawn from the fitted population is optimistic and is not an
estimate of generalization.

**Split scheme** — the rule deciding which pixels may be fitted on. A **pixel split**
assigns individual pixels independently; a **parcel split** assigns whole parcels, so
every pixel of a parcel lands on the same side.

**Parcel-disjoint** — of a population: containing no pixel from any parcel that
contributed a fitted pixel. Under a pixel split a population can be almost entirely
composed of pixels never individually fitted and still not be parcel-disjoint, because
neighbouring pixels of one parcel are near-duplicates. Only a parcel-disjoint population
estimates performance on ground the model has not seen.

**Conditional metric** — a score computed only over the pixels a previous stage passed
through. Conditional metrics are survivor-biased: they describe the model's behaviour on
an easier population than it faces in use.

**End-to-end metric** — a score computed over all pixels of a true class, counting a
pixel correct only if every stage routed it correctly and the final code matches. This is
the honest number, and it is always lower than any stage's conditional metric.

**Flat scoring** — re-expressing a cascade's final output as a single flat set of classes
so it can be compared against a non-cascaded model. The agreed flat taxonomy is the 13
economic crops, plus *reservoir* and *others*; forest folds into others but is also
reported as its own row so it is not lost.

**Cross-year test** — applying a model trained on one epoch to a later epoch. Its errors
mix two causes: the model failing, and the land genuinely having changed. A
**change-filtered** test restricts scoring to parcels whose code is identical across both
surveys, isolating model failure; the gap between filtered and unfiltered scores measures
the land-use change itself.

**Class prior** — how frequently each class occurs. A *training* prior is whatever the
sampling produced; a *true* prior is what the ground actually holds. When they differ, a
model's scores are systematically shifted, and rare classes are over-predicted in
proportion to the mismatch.

**Prior correction** — reweighting a model's probabilities after training to move them
from the training prior toward the true one. It changes the decision rule, never the
model, so it can be swept cheaply over many settings.

**Operating point** — the particular trade between precision and recall that a chosen
decision rule lands on. There is no single best one: a rule that stops over-predicting
rare crops also stops predicting them at all. Quoting a score without saying which
operating point produced it hides that choice rather than settling it.

## Notes

`NDVI_making.py` and `NDVI_stat_calculation.py` are one-off exploratory scripts from
early in the project. They are not part of any pipeline and are kept only as history.
