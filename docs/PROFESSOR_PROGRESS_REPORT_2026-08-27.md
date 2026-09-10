# Progress Report on Crop Classification in Rayong Province

**Date:** 27 August 2026, updated 3 September 2026  
**Study area:** Sentinel-2 tile 47PQQ, Rayong Province  
**Current model:** Three-stage hierarchical Support Vector Machine (SVM), final consolidated cascade

> **Update of 3 September 2026.** The planned work in section 14 is now finished and the final
> test result is in. The final consolidated cascade reached macro F1 **0.2429**. Sections 4, 9,
> and 11 to 15 have been updated. I have also read the collaborator's 2 September 2026 report and
> re-read their repository at its current version, and section 11 corrects something I reported
> earlier. Section 12 now proposes a different design for the joint paper.

## 1. Purpose of this report

In this report, I explain what I changed in the crop classification model, what worked, what failed, and what I now think is limiting the rare crop classes. I also explain the collaborator's XGBoost workflow in simple terms and propose an angle for combining both studies in a joint conference paper. Their first complete pipeline result arrived on 2 September 2026 and I summarise it in section 11, but the comparison between our two models is still open, because neither model has yet been scored under one agreed evaluation method.

The main score I use is macro F1. F1 combines precision, which asks whether a predicted crop is usually correct, and recall, which asks how much of the true crop the model can find. Macro F1 gives the same importance to every crop, so a rare crop such as Langsat counts as much as rubber even though rubber has millions of pixels and Langsat has only a small number of parcels. Weighted F1 gives more influence to the common classes, which is why it is normally much higher in this dataset. Macro F1 is useful for checking whether the model ignores rare crops, but it also becomes unstable when a class has almost no validation data.

## 2. How the SVM model works

The SVM model I developed uses three Sentinel-2 monthly composites from October, November, and December 2018. The reference crop labels come from the 2018 parcel survey by Thailand's Land Development Department (LDD), and I erode the parcel boundaries by 30 metres to reduce mixed pixels near the edges. From the satellite images, I calculate vegetation, water, bare-soil, built-up, and shortwave infrared indices, and the model classifies each 10-metre pixel through three stages.

1. Stage 1 separates economic crops, water, forest, and other land.
2. Stage 2 takes the pixels identified as economic crops and routes them into field crops, plantations, orchards, or an extra sink group for economic codes outside the 13 target crops.
3. Stage 3 uses a specialist model inside each group to predict the final crop, such as rice, rubber, durian, coconut, or Langsat.

I use this hierarchy because rubber should not compete directly with every land-use type in one large decision. However, the hierarchy also means that an early mistake cannot always be repaired later. For example, if Stage 2 sends a coconut pixel to the orchard group instead of the plantation group, the coconut specialist never receives that pixel.

I divide the data into training, validation, and test groups by complete parcel, using about 61 percent of the pixels for training, 16 percent for validation, and 23 percent for testing. The training group fits the models, the validation group guides choices without touching the final test, and the test group gives the final score. Every agricultural parcel belongs to only one group, so the model cannot learn some pixels from one parcel and then receive other pixels from the same parcel as supposedly unseen test data. This is stricter than a random pixel split, but it is closer to the real task of predicting a new parcel.

For the main score, I keep the complete test population and map non-crop truth to a background label. If the model predicts a crop on a road, forest, water body, or other non-crop area, that mistake reduces crop precision. I use this strict score because the final land-use map will cover the full tile, not only pixels that are already known to be crops.

## 3. How the model turns SVM scores into class probabilities

An SVM's native output is a signed distance from a decision boundary, not a probability. Every place in this report that mentions a "probability," a "probability rule," or a "calibration" is going through the three-step procedure below.

**Step 1: per-class sigmoid calibration.** For each stage, the underlying SVM is trained in a one-versus-rest form, so every class produces its own signed margin. I fit that base SVM once, using only the training parcels. Then, on a separate held-out set of validation parcels, I fit one sigmoid curve per class that maps a margin value to a number between 0 and 1: probability equals one divided by one plus the exponential of (A times the margin, plus B). The two numbers A and B are chosen per class so the curve best matches which margins in the validation parcels truly belonged to that class. This is the method from Platt (1999), *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*, still the standard way to turn an SVM margin into a probability. I use scikit-learn's internal implementation of Platt's fitting procedure, but I call it once per class by hand rather than through scikit-learn's ready-made `CalibratedClassifierCV` wrapper, because that wrapper's internal cross-validation step crashes when a rare class, such as Langsat, has too few positive rows in one of its internal folds, which happens in this dataset. Fitting the base model once on training parcels and the sigmoids only on a separate calibration half of validation parcels keeps this honest: no sigmoid ever sees a row its own base model was fitted on.

**Step 2: combining the per-class sigmoids into one probability vector.** Because each class's sigmoid is fitted independently, the outputs do not automatically add up to 1 across classes for a given pixel. I normalise them by dividing each class's value by the sum across all classes. This is a simplification. The more complete method in the literature, Wu, Lin, and Weng (2004), *Probability Estimates for Multi-class Classification by Pairwise Coupling*, solves for a jointly consistent set of probabilities using every pair of classes at once, rather than each class against the rest independently. I use the simpler sum-to-one normalisation instead, partly because several rare crops have too few calibration pixels to support the pairwise method reliably. I have not measured how much accuracy this simplification costs, and I am flagging it here as an open item rather than a settled choice.

**Step 3: the operating-point rule, which replaces stage-to-stage probability multiplication.** The cascade does not chain probabilities across the three stages by multiplying them into one joint number. Each stage is routed on its own calibrated probability. Before taking the final decision at Stage 2 and Stage 3, though, I apply a prior-correction step: I divide each class's calibrated probability by how common that class actually is in the calibration data, raise that ratio to a tunable power I call alpha, multiply it back into the probability, and renormalise. Alpha equal to 0 leaves the raw calibrated probability untouched. Alpha equal to 1 more fully corrects for the fact that, for example, rubber is far more common in the training pool than coconut, so the raw calibrated probabilities already favour the common classes before any decision is made. I search a grid of alpha values for Stage 2 and Stage 3 together, 169 combinations, and choose the pair that scores best on the calibration half of the validation parcels. I then apply that one fixed choice to the tuning half, which the search never saw, to get the honest number reported elsewhere in this document as the current probability rule. Stage 1 does not receive this correction; it is still a plain argmax on the raw calibrated probability, a known limitation already on record and not yet addressed.

## 4. What each completed run means

The internal development notes used short labels such as M0 and M5, but those labels were only checkpoints during development. M0 referred to the 24-feature protocol-repaired run, and M5 referred to the 30-feature run with larger Stage-3 specialists and the adjusted probability rule. They are not different model families, so I use descriptive names instead. The four completed full runs are the first parcel-disjoint baseline, the protocol-repaired run, the three-date Sentinel-2 run, and the final consolidated cascade.

### First parcel-disjoint baseline

At first, I planned to answer one basic question, which was how the three-stage SVM performs when the test parcels are genuinely unseen. This run used 24 Sentinel-2 features, made from eight indices for each of the three dates. The indices covered vegetation greenness, vegetation adjusted for soil, water, bare soil, built-up land, and two shortwave infrared ratios. I kept the SVM settings that had been selected during the older pixel-level experiments, removed upsampling and automatic class weighting, and limited the number of training pixels from very large classes so rubber could not dominate every fit. I also gave Stage 2 a real fourth sink class for non-target economic land and trained each Stage-3 specialist from the true crop group instead of only the pixels routed correctly by an earlier stage.

This first parcel-disjoint baseline reached macro F1 **0.2248** and weighted F1 **0.8018** on 5,500,269 test pixels. Its purpose was not to maximise the score yet. Its purpose was to replace the older pixel-based evaluation with one number that represented new parcels.

### Protocol-repaired run

Then I kept the same 24 features and the same SVM settings, but I repaired how later stages received training examples. Stage 2 had previously received routes made by a Stage-1 model that had already seen those training pixels, so I divided the training parcels into three parts and produced each part's route with a model fitted on the other two parts. I also divided the validation parcels into a calibration half and a tuning half, so the probability calibration and the later probability-rule choice did not use the same parcels.

This repaired run reached macro F1 **0.2283** and weighted F1 **0.8050**. The gain over 0.2248 was 0.0035, but the parcel-level uncertainty interval still touched zero, so I cannot honestly claim that the repairs improved accuracy. They made the procedure more trustworthy without reducing the result.

### Three-date Sentinel-2 run

After that, I planned to test whether a small feature addition and a larger final specialist could recover more crop classes. I added six columns, which were MTCI and the raw Sentinel-2 B11 band for each of the three dates, so the feature count increased from 24 to 30. MTCI is related to leaf chlorophyll, while B11 is a shortwave infrared band related to vegetation and moisture. I also increased the nonlinear feature-space size of each Stage-3 specialist from 600 to 1,200 components and used a validation-selected probability adjustment for Stage 2 and Stage 3.

This three-date Sentinel-2 model reached macro F1 **0.2344** and weighted F1 **0.7974** on the same 5,500,269 test pixels. It was the best model I had at the time of the first version of this report, and the internal notes call it M5.

### Final consolidated cascade

Finally, I combined only the changes that had passed their written validation rules, wrote the configuration down before training, and used the test fold once. Two changes went in. The first was the Stage-2 subtype weighting from section 6, which redistributes influence to underrepresented crop types inside each Stage-2 group. The second was the retuned settings from the parcel-grouped hyperparameter search, which applied to Stage 2 and the orchard specialist and increased both the penalty term and the nonlinear feature-space size. The rebalancing test inside the plantation and orchard specialists failed its own rule, so it was left out, and the plantation specialist was not changed.

This final consolidated cascade reached macro F1 **0.2429** and weighted F1 **0.7949** on the same 5,500,269 test pixels, which is a gain of **0.0085** over the three-date model.

| Completed full run | Features and main change | Test macro F1 | Test weighted F1 |
|---|---|---:|---:|
| First parcel-disjoint baseline | 24 Sentinel-2 features, parcel-separated fitting and scoring | 0.2248 | 0.8018 |
| Protocol-repaired run | Same 24 features, honest Stage-1 routes for Stage-2 training, separate calibration and tuning parcels | 0.2283 | 0.8050 |
| Three-date Sentinel-2 run | 30 features, larger Stage-3 specialists, adjusted probability rule | 0.2344 | 0.7974 |
| Final consolidated cascade | Stage-2 subtype weighting plus parcel-grouped retuning of Stage 2 and the orchard specialist | **0.2429** | 0.7949 |

Before this final run I wrote down a planning range of 0.24 to 0.26 macro F1, with 0.28 as an upside case. The result of 0.2429 sits at the very bottom of that range, and well below the upside. I want to state the size of this honestly, in three parts.

The gain is an **observed test-fold gain of 0.0085**, which is a point estimate. It is **not clearly separated from parcel-level uncertainty**: the parcel bootstrap interval on macro F1 in this project is about plus or minus 0.014, which is wider than the gain itself. What the written gates do establish is that both retained changes won on validation data before the test fold was touched, which is a different and weaker claim than the gain being statistically distinguishable from zero.

**Weighted F1 moved the other way**, from 0.7974 down to 0.7949. Macro F1 and weighted F1 answer different questions, so this is not a contradiction, but it does mean the final model is very slightly worse at the common-crop-dominated view of the map while being better at the equal-weight-per-crop view. Both numbers belong in any table that reports this run.

For scale, the whole final round of work produced 0.0085, while everything before it, from the first parcel-disjoint baseline onward, produced 0.0096.

Where the gain came from is more informative than the total. Almost all of it is in two crops. Oil palm improved from 0.4158 to **0.5587**, a gain of 0.1429, and rice improved from 0.4181 to 0.4457. Both are mid-frequency crops with enough parcels for a specialist model to learn from, which is exactly what the Stage-2 diagnosis in sections 5 and 6 predicted would happen.

| Crop | Three-date run F1 | Final cascade F1 | Change |
|---|---:|---:|---:|
| Rubber | 0.8721 | 0.8658 | -0.0063 |
| Oil palm | 0.4158 | **0.5587** | **+0.1429** |
| Rice | 0.4181 | 0.4457 | +0.0276 |
| Pineapple | 0.4208 | 0.4205 | -0.0003 |
| Durian | 0.3827 | 0.3642 | -0.0185 |
| Cassava | 0.3451 | 0.3379 | -0.0072 |
| Mango | 0.0659 | 0.0812 | +0.0153 |
| Rambutan | 0.0184 | 0.0406 | +0.0222 |
| Jackfruit | 0.0810 | **0.0198** | **-0.0612** |
| Mangosteen | 0.0132 | 0.0112 | -0.0020 |
| Coconut | 0.0085 | 0.0095 | +0.0010 |
| Longan | 0.0054 | 0.0024 | -0.0030 |
| Langsat | 0.0000 | 0.0000 | 0.0000 |

Two things in that table need to be said plainly rather than summarised away.

First, **the rare crops did not move.** Coconut gained 0.0010 and is still at 0.0095. Longan and Langsat are effectively at zero. This is what I predicted before the run, and it is the clearest evidence I have that no Sentinel-2-only change I tested can rescue these classes. That prediction is now confirmed by a completed test read rather than by validation-half projections.

Second, **jackfruit became worse, by 0.0612, and I did not predict it.** Nothing in the validation-half measurements warned me. The most likely explanation is that the retuned, higher-capacity orchard specialist moved the jackfruit decision boundary in the wrong direction, even though that retuning won its own validation rule overall. I am reporting this as an open question rather than explaining it away, because the test fold has now been read and I do not want to keep tuning against it.

So the final model is useful for the common crops, has improved clearly for one mid-frequency crop, and still does not solve the rare crops.

## 5. What failed and what the failure taught me

Then I tested a cost-sensitive SVM because I expected larger class weights to help the rare crops. I wrote the success rule before running it, and the weighted model had to beat the unchanged model on the same validation rows before it could receive another test-fold score.

The weighted model failed that rule. Its validation macro F1 was 0.2272 compared with 0.2294 for the unchanged current model, so I did not calculate a new full test score. Longan improved slightly, but several other crops lost enough to make the overall result worse.

After that, I checked why the weights did so little, and I found that most of them were exactly 1.00. The sampling caps had already balanced the four Stage-2 groups, so the new weighting formula saw four equal groups and made almost no change. At first, this looked like an experiment that simply failed, but it exposed a more important imbalance inside each group.

Stage 2 balances field crops, plantations, orchards, and the sink group, but it does not balance the crop types inside those groups. In the plantation candidate pool, 96.82 percent of the rows are rubber, 3.10 percent are oil palm, and only 0.08 percent are coconut. When Stage 2 draws 200,000 plantation rows, it receives about 193,645 rubber pixels and only 159 coconut pixels. Coconut and rubber share the same plantation label at this stage, so balancing the four group totals never fixes this internal imbalance.

I checked the size of the routing problem by freezing every trained model and replacing only the learned Stage-2 route with the correct group from the label. This is not a deployable model because the correct route is unknown during real prediction, but it shows how much performance is being lost at that decision. Macro F1 increased from 0.2294 to 0.3785, which is a gain of 0.1491 with every classifier frozen.

That diagnostic does not mean a learned model can recover the full 0.1491. It means Stage-2 routing is a large source of error and is worth improving.

## 6. Rebalancing crop influence inside Stage 2

So I ran a more focused experiment. I kept the total weight of each Stage-2 group unchanged, but I redistributed that weight inside the group so that underrepresented crop types had more influence. Stage 1 and all Stage-3 specialists stayed frozen, and I refitted Stage 2 twice on the same rows. The control used the old weights, and the treatment used the new subtype weights.

The control reproduced the original Stage-2 model exactly, which confirmed that the same training pool was used. The treatment increased validation macro F1 from 0.2294 to **0.2375**, a gain of **0.0081**. It also beat the control at all 169 tested probability settings, and the gain at the original fixed setting was 0.0069.

| Validation result | Control | Subtype-mass treatment |
|---|---:|---:|
| Macro F1 | 0.2294 | **0.2375** |
| Crops with F1 at least 0.01 | 10 of 13 | 10 of 13 |

The result worked, but not in the way I predicted. Coconut routing nearly doubled from 13.7 percent to 26.6 percent, but coconut F1 did not improve. Most of the useful gain came from rice and oil palm, which have enough parcels for the later specialist models to learn from. This told me that giving a rare pixel the correct route is not enough when the final specialist still cannot separate that crop.

The 0.0081 result is measured on one half of the validation fold, and it is not directly comparable to the 0.2344 test score. I also held the 200,000-row training draw fixed when I first measured it, so I repeated the control and treatment fit over three fresh training-pool draws before trusting it further. All three draws favoured the new weights again, with gains of 0.0072, 0.0087, and 0.0074, for a mean of 0.0078. The written rule for carrying the weights into the final cascade was that they had to win in all three draws with an average gain of at least 0.002, and they cleared that rule, so this result is no longer a single-draw measurement.

## 7. The evaluation mistake I had to correct

Earlier in the project, I used an isolated crop probe that randomly split pixels into training and test groups. The probe appeared to show that the rare crops were spectrally separable, and the five-date version reached macro F1 0.5852. I later replayed that split with parcel identities and found that 87.5 to 95.7 percent of the rare-crop test parcels also appeared in training.

I rebuilt the same probe by dividing complete parcels instead of pixels. Macro F1 fell from **0.5852 to 0.3945**, and the damage was concentrated in the rare crops.

| Crop | Pixel-split F1 | Parcel-disjoint F1 |
|---|---:|---:|
| Mango | 0.6150 | 0.4350 |
| Coconut | 0.6329 | 0.3782 |
| Rambutan | 0.6092 | 0.3407 |
| Mangosteen | 0.5385 | 0.2753 |
| Jackfruit | 0.4540 | 0.2510 |
| Longan | 0.5543 | 0.2357 |
| Langsat | 0.6774 | 0.0000 |

I am withdrawing the earlier claim that the rare crops are clearly not limited by spectral information. The pixel split mainly showed that the model could recognise more pixels from parcels it had already seen. It did not show reliable generalisation to new parcels.

However, I am also not claiming that no possible model can reach F1 0.33 for these crops. The correct statement is narrower. With the current three-date Sentinel-2 data and the interventions I have planned, I do not have credible evidence that coconut, longan, Langsat, rambutan, or mangosteen will reach strict F1 0.33. Mango is more promising, and Langsat has too little support for either a positive or negative conclusion.

## 8. New sensor-fusion probe

After an independent technical review challenged my claim about a fixed spectral ceiling, I ran one more parcel-disjoint probe to check whether the limitation was caused specifically by Sentinel-2. I compared the same rows and the same parcel split under two feature sets. The first arm used 40 Sentinel-2 index features from five dates, and the second arm used all 153 available features, adding terrain information from the digital elevation model and Sentinel-1 radar measurements.

The fused feature set increased probe macro F1 from **0.3945 to 0.4401**. Rice gained 0.1084, mangosteen gained 0.0793, mango gained 0.0754, rambutan gained 0.0616, durian gained 0.0590, and oil palm gained 0.0547. Coconut gained only 0.0085, longan gained 0.0285, and Langsat did not move.

| Feature set in the controlled probe | Macro F1 |
|---|---:|
| Sentinel-2 only | 0.3945 |
| Sentinel-1 + terrain + Sentinel-2 | **0.4401** |

Before running it, I wrote that the optical-ceiling explanation would weaken if coconut, mangosteen, rambutan, or longan gained at least 0.10 F1. None reached that threshold, so the written decision rule did not change my conclusion. Still, the 0.0456 macro gain is useful evidence that radar and terrain add information for several mid-frequency crops, and the fused pipeline deserves further study.

These probe scores cannot be placed beside the full cascade score of 0.2429 as if they are competing models. The probe has a balanced crop-only population, five dates, and no non-crop false positives, while the full cascade has three dates and is scored over the natural tile population. The valid comparison is only 0.3945 against 0.4401 inside the same probe.

## 9. What I now think is limiting the model

At this point, I think there are three connected limits.

First, Stage-2 routing is still losing a large amount of recoverable information for rice, cassava, pineapple, durian, oil palm, and mango. These crops have enough parcels for a better route to matter, and the subtype-mass result already moved rice and oil palm in the right direction.

Second, the rare crops have too few independent parcels. Langsat has 1,639 training pixels from only 10 training parcels, and one parcel contains 1,310 of those pixels. Its complete validation fold has only 13 pixels, and its test fold has 191 pixels. A per-class Langsat score is not reliable enough to guide model selection.

Third, three dates of Sentinel-2 may not contain enough generalisable information to separate visually similar orchard species on unseen parcels. The fused probe improved several crops, so I do not think Sentinel-2 is the only useful sensor, but the added information still did not rescue coconut, longan, or Langsat.

The final consolidated cascade has now tested this. Better fitting did improve a mid-frequency crop, and it improved oil palm by a large amount, while the rare crops stayed where they were. So this is no longer a working conclusion that is waiting for evidence. It is the closing position of this stage of the project: the mid-frequency crops respond to routing and capacity work, and the rare crops are limited by how few independent parcels exist. New parcels are the intervention most likely to change the rare-crop result, and no Sentinel-2-only change I tested substituted for them.

## 10. How the current result relates to the earlier Random Forest paper

The Random Forest benchmark I have referred to is the earlier Rayong crop-classification study using Sentinel-2 imagery. It used a flat Random Forest, which means one model predicted the final land-use class directly instead of passing pixels through several stages. The study used a different image period, mainly 2024 with additional 2020 data, and it evaluated 15 land-use classes on about 303,947 test pixels.

That paper reported about 0.716 overall accuracy, 0.678 Cohen's kappa, and 0.714 weighted F1. Its reported overall F1 was about 0.71, oil palm reached about 0.81, and the per-class table has been used as an approximate F1 0.33 reference for the orchard and rare crops.

My current result is not a direct repeat of that experiment. I use 2018 imagery and the 2018 LDD survey, a three-stage SVM, a natural full-tile test population, and macro F1 over the 13 economic crops. I also cannot reconstruct from the available material whether the Random Forest test parcels were separated completely from its training parcels. My parcel experiment shows that this choice strongly changes my own rare-crop scores, but it does not prove what would happen to the Random Forest result.

So I will use the Random Forest paper as related work and a motivation for the crop classes, not as a controlled head-to-head result. A fair classifier comparison requires the same parcels, dates, features, labels, and scoring rule.

## 11. The collaborator's XGBoost workflow

When I first inspected the collaborator's work, I only found separate training components and believed that the complete prediction chain had not been connected. **That statement was wrong and I am correcting it here.** Their prediction chain has been connected since late August, and I re-read their repository at its current version on 3 September 2026 to confirm the details below. Their pipeline now contains three connected models.

1. A water model predicts the probability that a pixel is water. Pixels with water probability at or above 0.56 are removed from the crop candidates.
2. A building model then predicts the probability that each remaining pixel is a building. Pixels with building probability at or above 0.56 are also removed.
3. A flat XGBoost crop model classifies the remaining pixels into 13 economic crops plus an `others` class.

Their design and my SVM design solve the early filtering problem differently. Their cascade removes water and buildings before a flat crop classifier makes the final decision. My SVM cascade first separates four broad land groups, then routes economic crops into field, plantation, and orchard groups, and finally uses a specialist model inside each group.

Their most recent report, dated 2 September 2026, gives their first complete end-to-end pipeline result. Their water model reaches F1 0.865 and their building model reaches F1 0.836, which are both strong. Their crop model reaches accuracy 0.57 and macro F1 0.38 over 14 labels on 755,295 pixels.

**One point about that crop number is essential and I want to state it before anything else. It is a result from the old pipeline, not their current one.** Their report is largely an analysis of why that pipeline was wrong. The pipeline built one shared dataset for all three models and then deleted any row containing a missing value, so a single missing neighbour in the water and building texture features destroyed rows whose crop features were perfectly usable. Their fix, which is to build a separate dataset for each model, is the right fix, and they had already implemented it in code on 1 and 2 September. It would be wrong to quote 0.38 as the XGBoost result, both because the evaluation population differs from mine and because they have themselves replaced the configuration that produced it.

I can also explain the population that produced those old numbers, which I had previously listed as an open question. The old extraction script drew a sample capped at 200,000 pixels per class, built the single shared table, and only then applied the row deletion. That is why their old rubber support was 197,590, just under the cap, rather than the millions of rubber pixels that exist in the tile.

### Their repaired-pipeline result

A result from the repaired pipeline has now arrived, and it changes what I can say. I have read it from a screenshot of the classification report rather than from a file, so the figures below should be confirmed before either of us quotes them in writing.

Their repaired pipeline scores **19,680,774 pixels**, reports a macro average F1 of **0.43** over 14 labels, weighted F1 0.72, and accuracy 0.67. Over the 13 target crops alone the macro F1 is **0.4577**. The sampling cap is gone, which is what I expected, and this settles the population question completely.

Two structural facts follow from their own table, and both are arithmetic rather than interpretation.

**Their evaluation population contains no non-crop pixels at all.** The 13 crop supports add up to 19,680,774 exactly, which is the reported total, leaving no remainder. Their `others` class has support 0. So crop precision in that table cannot be reduced by a single false positive that lands on a road, a forest, a building, or bare ground. This is the same evaluation mistake I had to correct in my own work in section 7, and on my own predictions it was worth about 0.034 macro F1. Their headline 0.43 also averages that empty `others` class into 14 labels, which pushes the headline down by about 0.033 relative to the 13-crop figure. So one choice depresses the headline while another inflates every per-crop value, and I do not want those two to quietly cancel out in a reader's mind.

**Their score also appears to be measured on the data the model was fitted on.** Their inference script reads the whole extracted table, predicts on all of it, and computes the report on that, with no separation between training and evaluation pixels. Their crop model was fitted on a capped sample drawn from the same tile, so for every crop with fewer than 200,000 pixels, which is every crop except rubber, the training sample is close to the entire class and those pixels sit inside the 19.68 million being scored. Their strongest surprises are in exactly those crops: coconut 0.39, longan 0.47, longkong 0.37, against my 0.0095, 0.0024, and 0.0000.

To make the comparison as fair as I currently can, I rescored my own unchanged final predictions on crop-truth rows only, which is the closest match to their population that my data allows.

| Result | Rows | Macro F1 over 13 crops |
|---|---:|---:|
| My final cascade, strict, full population | 5,500,269 | 0.2429 |
| My final cascade, crop-truth rows only | 3,800,567 | **0.2662** |
| Their repaired pipeline | 19,680,774 | **0.4577** |

Changing only the scoring population is worth 0.0233 to my own model. The rest of the gap I cannot yet attribute to the algorithm, because three further differences remain, and I have measured each of them on my own data to be large for precisely these crops. Their score appears to include training pixels. Their evaluation does not erode parcel boundaries while their training does. And their split is by pixel rather than by parcel, which in my own probe moved macro F1 from 0.5852 down to 0.3945, with Langsat falling from 0.6774 to exactly zero.

I want to be careful about tone here, because none of this means their model is weak. It means their number and my number are not answering the same question yet. One useful cross-check points the other way: their rubber precision is 0.98 and mine is 0.980 under the same crop-only convention. On the one crop where support is not the limiting factor, the two systems agree almost exactly.

The two questions I need answered are therefore narrow and cheap for them: why does `others` have support 0, and is the reported score computed on pixels the model was trained on.


Three differences between their standalone result and their pipeline result still need to be separated before that cause is stated in a paper, and I would raise all three as joint methodology rather than as criticism.

1. Their crop report is calculated only on the pixels that survive the water and building filters. A rubber pixel wrongly classified as water is removed from the crop evaluation instead of being charged as a missed rubber pixel, which raises crop recall. The joint comparison needs a full-population score that counts the errors of all three models together.
2. Their training extractor erodes parcel edges, but their pipeline evaluation extractor does not, so the crop model is trained on parcel interiors and scored on all pixels including mixed edges.
3. Their training script still reads the older capped dataset, while their pipeline scores the newer un-eroded filtered population, so the deployed model was not fitted on the distribution it is scored on.

I also want to record two things I checked and found to be **correct** in their code, so they are not raised again as concerns: their Sentinel-2 band indexing and their MTCI formula are both right, and there is no leakage of pixel coordinates into their features.

I can explain this workflow from the code, but I cannot yet state how well their complete model performs. I do not have a verified score from their final cascade under the strict full-population and parcel-disjoint evaluation used for my SVM, and I also do not have my SVM result under their exact evaluation because the SVM has no separate building filter. Their result should stay open until both models are run on the same rows, the same parcel split, the same label definitions, and the same scoring denominator.

I also found a shared protocol risk. Their data preparation splits sampled pixels by row, so pixels from the same parcel can appear in both training and evaluation. I made the same mistake in my earlier probe, and it strongly increased the rare-crop scores. I would raise this as a joint methodological issue to fix together, not as a criticism of their model.

Finally, the evaluation population remains the single largest source of difference between our two sets of numbers, and it is the first thing the joint protocol has to settle. The cap is no longer the issue, because their repaired pipeline removed it. What remains is that their population is crop-only while mine includes all non-crop ground, and that their score appears to be measured in-sample. Both are fixable, and both have to be fixed before any table puts our two numbers side by side.

## 12. Proposed angle for the joint conference paper

I do not think the strongest paper is a simple table asking whether SVM or XGBoost has the larger F1. The evaluation protocol can move the score more than many model changes, and a table made before the protocol is aligned could reward the measuring method instead of the classifier.

### The rule we agreed, and the problem it creates

In our discussion, the condition for a joint paper was that one axis must be held fixed. Either we use different algorithms with the same architecture and the same control variables, or we use the same algorithm with different methods. We chose the first option, and I agree with that choice.

The difficulty is that the two systems do not currently have the same architecture. Both are hierarchical, and both are cascades, but the hierarchies express different ideas. Mine routes pixels semantically, first into four broad land groups and then into field crops, plantations, and orchards, and finally uses a specialist inside each group. Theirs rejects water, then rejects buildings, and then lets one flat 14-label model classify whatever survives. Taken literally, the rule we agreed would require one of us to abandon our architecture.

I do not think either one-sided option is good. If I flatten my model to match theirs, I remove the only structural contribution my study has, and I also ignore a real constraint, which is that a flat SVM over 24.3 million rows is not affordable in the way a flat gradient-boosted tree is. Part of the reason my cascade exists is that limit. If they adopt my cascade instead, the comparison is clean and each of us contributes something distinct, but it quietly assumes my routing design is the better architecture, and that is exactly the question worth asking.

### The resolution I propose: treat architecture as a second factor

Rather than forcing the two architectures to match, I propose we make architecture a second controlled factor and fill a two-by-two table. The two levels of that factor are **two cascades**, not "flat against cascade":

- a **semantic routing cascade**, which is my design, and
- a **sequential rejection cascade**, which is theirs.

My first draft of this proposal described their system as the flat cell, and that was a mistake I want to correct here. End to end, their system is a cascade whose final stage happens to be flat. Comparing it against a genuinely flat SVM would confound the architecture with the presence of the two upstream filters, so both architectures have to be implemented with both algorithms.

| | RBF-SVM | XGBoost |
|---|---|---|
| Semantic routing cascade | to be run | to be run |
| Sequential rejection cascade | to be run | to be run |

Every cell must produce the same final label set, which is the 13 target crops plus an `others` class.

**All four cells are listed as pending, and that is deliberate.** Neither of the two results we already have qualifies as a cell in this table. My final cascade uses my own 30-feature set rather than the agreed common features, and it was produced before these controls existed. Their 2 September result does not use the shared split, population, denominator, or a held-out partition, and it comes from the pipeline before their row-deletion repair. Both remain valuable as motivation and as a record of where each study stood before alignment, but neither can contribute to an estimated effect.

I also want to correct a second thing I wrote in the first draft. I said one cell could be dropped while keeping the design. That is not true: with only three cells, the effect of the algorithm, the effect of the architecture, and the interaction between them cannot all be separated. So the paper may use words like *main effect* and *interaction* only if all four cells are run. If only three are possible, the study should be renamed to controlled pairwise comparisons and the claims reduced to the pairs actually run.

I have written the full version of this, including the eight control variables that must be frozen before either side runs anything and the fairness conditions for tuning each arm, as a separate document, `docs/JOINT_PROTOCOL_2026-09-03.md`, so it can be agreed or amended directly.

### The angle

> **Routing or rejection? A parcel-disjoint comparison of two hierarchical designs for crop mapping under extreme class imbalance in Rayong Province.**

A fair comparison can show where each design loses information, which is more useful than only naming a winner.

I propose four contributions for the paper.

1. **A shared parcel-disjoint evaluation.** Both models should use the same 2018 parcel assignment, the same label erosion, the same 13 crop definitions, and the same full-population scoring rule. If the joint study compares SVM against XGBoost as classifiers, they should also receive the same feature matrix and date window. If the inputs remain different, the paper should call it a comparison of two complete systems instead of attributing the difference to the classifier. A pixel-split result can also be reported as a protocol sensitivity experiment, but it should not be presented as unseen-parcel performance.
2. **A comparison of two hierarchical strategies.** The joint study can compare the SVM's semantic routing against the XGBoost water and building filters, and then trace how much error is created at each stage instead of treating the final F1 as a black box.
3. **A rare-class support analysis.** The paper can show that pixel count is not the same as independent information because thousands of pixels may come from only a few parcels. This is especially important for Langsat, coconut, longan, rambutan, and mangosteen.
4. **A disagreement map between the two models.** When both models agree with high confidence, the output can be treated as a stronger candidate prediction. When they disagree, those parcels can become targets for field checking or for the next LDD survey. This gives a practical use for both models even if one has a higher overall score.

Using both models as a formal ensemble should remain optional for now. To combine their probabilities honestly, the collaborator and I would first need to convert both outputs into the same set of final labels, calibrate both models on the same parcel-disjoint validation data, and train any combining rule without reading the test fold. Until that work is completed, a consensus and disagreement analysis is safer than claiming that an ensemble improves accuracy.

The result of the collaborator's XGBoost model will decide the final paper emphasis. If both models fail on the same rare crops, the paper becomes a strong study of data support and evaluation protocol. If XGBoost recovers classes that the SVM misses, the paper can study whether tree-based filtering handles the decision boundary better. If the SVM recovers classes that XGBoost misses, the agricultural routing structure becomes the main explanation. And if their errors are complementary, the disagreement analysis provides a justified path toward an ensemble.

This keeps the paper open to the result instead of deciding the story before the experiment.

## 13. Current limitations

- The test fold has now been read for the final consolidated cascade, and I am treating it as spent. Any further change to this model must be evaluated on new ground, not on this fold, or the score stops meaning unseen-parcel performance.
- The retuning covered only Stage 2 and the orchard specialist. Stage 1, the field specialist, and the plantation specialist still carry hyperparameters that were originally chosen by a pixel-level search based on accuracy.
- The subtype weighting gain was measured on one fixed training draw before it was repeated over three fresh draws. It passed, but it was still selected and confirmed on validation data from the same fold.
- The fused result is an isolated controlled probe, not a complete Sentinel-1 + terrain + Sentinel-2 cascade result.
- The labels come from the 2018 LDD survey, and mixed-crop compound codes are excluded.
- The earlier Random Forest paper uses a different image period, feature set, model structure, and evaluation population. My experiments show that evaluation protocol strongly changes my SVM scores, but they do not prove how the Random Forest result would change under a parcel split.

## 14. What I planned, and how it ended

After the independent review, I arranged the remaining work so that cheap checks came before another full test score. **All of this plan is now finished.** The table below records what each step did and how it ended.

| Order | Planned work | What changes | Current status |
|---|---|---|---|
| 1 | Save exact training and calibration row identities | Adds reproducibility records only and does not change model behaviour | Completed |
| 2 | Compare Sentinel-2 against Sentinel-1 + terrain + Sentinel-2 on the same unseen parcels | Tests whether radar and terrain add information | Completed, macro F1 increased from 0.3945 to 0.4401 inside the controlled probe |
| 3 | Repeat the Stage-2 crop-rebalancing experiment over three fresh training draws | Tests whether the +0.0081 validation gain survives a different sample | Completed, all three draws favoured the new weights (mean gain 0.0078); the rule for keeping them is satisfied |
| 4 | Retune Stage 2 and the orchard specialist | Replaces the old pixel-based accuracy search with parcel-grouped macro F1 selection | Completed and passed its rule. Both halves of the search chose the same settings independently: double the nonlinear feature-space size, half the previous gamma value, and a higher penalty term. This confirms that every earlier search had been running against its own capacity limit |
| 5 | Rebalance crop influence inside the plantation and orchard specialists | Tests whether coconut and the rare orchards fail inside the final specialist even after correct routing | Completed and **failed** its rule, by 0.0012, and jackfruit fell below the minimum. Coconut itself did move slightly, from 0.0000 to 0.0033 in that probe, but the overall effect was negative, so these weights were left out of the final model |
| 6 | Test date-difference features only if time remains | Changes feature weighting but adds no new image dates | Skipped, as planned |
| 7 | Train one final combined cascade | Includes only changes that passed their written validation rules, then uses the final test once | Completed. Macro F1 **0.2429**, reported in section 4. The first launch failed while loading a model, before any test data was touched, which was one of the abort conditions I had written down; the relaunch read the test fold exactly once within this plan. The same fold had produced earlier checkpoint results, so it is a previously observed partition, not a globally fresh test set |

Two of the four experiments passed their written rules and two did not, and I kept only the two that passed. The final score of 0.2429 landed at the bottom of the 0.24 to 0.26 range I had written down beforehand, and clearly below the 0.28 upside case. I regard the discipline of writing the rules down first as the more valuable outcome of this stage than the 0.0085 itself, because it is what allows me to report the jackfruit regression and the failed specialist rebalancing instead of quietly dropping them.

### What comes next

The test fold is now spent, so the next stage of work is not more tuning of this model.

1. **Settle the joint protocol.** Agree the eight control variables in `docs/JOINT_PROTOCOL_2026-09-03.md` with the collaborator, and get answers to the two narrow questions in section 11: why `others` has support 0 in their repaired result, and whether that result is measured on pixels the model was trained on. Nothing in the joint paper can be measured before this is settled.
2. **Run all four cells of the two-by-two.** Both architectures with both algorithms, on the frozen pixel identities and the shared training rows. None of the four is done; the two results we have today are baselines, not cells.
3. **Try majority-class capping as a training method.** This is the one modelling idea in their study that mine has not tested. Capping rubber at 200,000 rows while Langsat keeps its 1,800 improves the balance the model trains against by roughly 60 times, and it is a different mechanism from the class weights that failed in section 5, because it changes which pixels define the decision boundary rather than only reweighting the loss. I would train on the capped sample and still score strictly on the natural tile.
4. **Explain the jackfruit regression**, using validation data and not the test fold.
5. **Request more parcels.** For the rare crops, the most useful next resource is more independent parcels from another LDD survey year or a neighbouring tile. Model changes can still improve the middle-frequency crops, but I do not want to promise that a new weighting formula can replace missing parcel data. This is now supported by a completed test result rather than by a projection.

## 15. Main points for discussion

1. Is the strict parcel-disjoint, full-population score the correct primary result for the conference paper?
2. Can I obtain additional parcel surveys from 2020, 2024, or a neighbouring tile for the rare crops?
3. Can the collaborator and I agree on one shared split, label mapping, feature set, and scoring denominator before comparing SVM and XGBoost? The proposed wording is in `docs/JOINT_PROTOCOL_2026-09-03.md`.
4. Do you accept the two-by-two design in section 12, with two cascade architectures as the two levels and all four cells run fresh, as the correct way to satisfy the rule that we change the algorithm while holding the control variables fixed? If not, which single architecture should both sides adopt, and who changes?
5. Should the joint paper focus first on the fair comparison and error analysis, with an ensemble kept as optional follow-up work?
6. Is it acceptable to report the final result as 0.2429 with the jackfruit regression left unexplained, or should that be resolved on validation data before the paper is written?
7. For the joint paper, should we lock a genuinely new test partition before any joint run, or report the existing fold 2 as a previously observed partition? It was read once within the recent plan, but earlier checkpoints had already used it, so it is not a globally untouched test set.

The main lesson from this stage is that the two limits I identified behave differently. Routing and capacity work is real and it paid, but it paid in the mid-frequency crops, and oil palm improving by 0.1429 is the clearest example. Rare-crop performance did not move, and it is limited by how few independent parcels exist rather than by anything I can repair by fitting the model differently. The final result of 0.2429 is a modest gain honestly measured, and the more useful outcome is that I can now say which of the two limits is which, with a completed test read behind the statement instead of a projection.
