# Progress Report on Crop Classification in Rayong Province

**Date:** 27 August 2026  
**Study area:** Sentinel-2 tile 47PQQ, Rayong Province  
**Current model:** Three-stage hierarchical Support Vector Machine (SVM)

## 1. Purpose of this report

In this report, I explain what I changed in the crop classification model, what worked, what failed, and what I now think is limiting the rare crop classes. I also explain the collaborator's XGBoost workflow in simple terms and propose an angle for combining both studies in a joint conference paper. Their model result is still open because I have reviewed how their code works, but I do not yet have a verified result produced under the same evaluation method as my SVM model.

The main score I use is macro F1. F1 combines precision, which asks whether a predicted crop is usually correct, and recall, which asks how much of the true crop the model can find. Macro F1 gives the same importance to every crop, so a rare crop such as Langsat counts as much as rubber even though rubber has millions of pixels and Langsat has only a small number of parcels. Weighted F1 gives more influence to the common classes, which is why it is normally much higher in this dataset. Macro F1 is useful for checking whether the model ignores rare crops, but it also becomes unstable when a class has almost no validation data.

## 2. How the SVM model works

The SVM model I developed uses three Sentinel-2 monthly composites from October, November, and December 2018. The reference crop labels come from the 2018 parcel survey by Thailand's Land Development Department (LDD), and I erode the parcel boundaries by 30 metres to reduce mixed pixels near the edges. From the satellite images, I calculate vegetation, water, bare-soil, built-up, and shortwave infrared indices, and the model classifies each 10-metre pixel through three stages.

1. Stage 1 separates economic crops, water, forest, and other land.
2. Stage 2 takes the pixels identified as economic crops and routes them into field crops, plantations, orchards, or an extra sink group for economic codes outside the 13 target crops.
3. Stage 3 uses a specialist model inside each group to predict the final crop, such as rice, rubber, durian, coconut, or Langsat.

I use this hierarchy because rubber should not compete directly with every land-use type in one large decision. However, the hierarchy also means that an early mistake cannot always be repaired later. For example, if Stage 2 sends a coconut pixel to the orchard group instead of the plantation group, the coconut specialist never receives that pixel.

I divide the data into training, validation, and test groups by complete parcel, using about 61 percent of the pixels for training, 16 percent for validation, and 23 percent for testing. The training group fits the models, the validation group guides choices without touching the final test, and the test group gives the final score. Every agricultural parcel belongs to only one group, so the model cannot learn some pixels from one parcel and then receive other pixels from the same parcel as supposedly unseen test data. This is stricter than a random pixel split, but it is closer to the real task of predicting a new parcel.

For the main score, I keep the complete test population and map non-crop truth to a background label. If the model predicts a crop on a road, forest, water body, or other non-crop area, that mistake reduces crop precision. I use this strict score because the final land-use map will cover the full tile, not only pixels that are already known to be crops.

## 3. What each completed run means

The internal development notes used short labels such as M0 and M5, but those labels were only checkpoints during development. M0 referred to the 24-feature protocol-repaired run, and M5 referred to the current 30-feature run with larger Stage-3 specialists and the adjusted probability rule. They are not different model families, so I use descriptive names instead. The three completed full runs are the first parcel-disjoint baseline, the protocol-repaired run, and the current three-date Sentinel-2 run.

### First parcel-disjoint baseline

At first, I planned to answer one basic question, which was how the three-stage SVM performs when the test parcels are genuinely unseen. This run used 24 Sentinel-2 features, made from eight indices for each of the three dates. The indices covered vegetation greenness, vegetation adjusted for soil, water, bare soil, built-up land, and two shortwave infrared ratios. I kept the SVM settings that had been selected during the older pixel-level experiments, removed upsampling and automatic class weighting, and limited the number of training pixels from very large classes so rubber could not dominate every fit. I also gave Stage 2 a real fourth sink class for non-target economic land and trained each Stage-3 specialist from the true crop group instead of only the pixels routed correctly by an earlier stage.

This first parcel-disjoint baseline reached macro F1 **0.2248** and weighted F1 **0.8018** on 5,500,269 test pixels. Its purpose was not to maximise the score yet. Its purpose was to replace the older pixel-based evaluation with one number that represented new parcels.

### Protocol-repaired run

Then I kept the same 24 features and the same SVM settings, but I repaired how later stages received training examples. Stage 2 had previously received routes made by a Stage-1 model that had already seen those training pixels, so I divided the training parcels into three parts and produced each part's route with a model fitted on the other two parts. I also divided the validation parcels into a calibration half and a tuning half, so the probability calibration and the later probability-rule choice did not use the same parcels.

This repaired run reached macro F1 **0.2283** and weighted F1 **0.8050**. The gain over 0.2248 was 0.0035, but the parcel-level uncertainty interval still touched zero, so I cannot honestly claim that the repairs improved accuracy. They made the procedure more trustworthy without reducing the result.

### Current three-date Sentinel-2 run

After that, I planned to test whether a small feature addition and a larger final specialist could recover more crop classes. I added six columns, which were MTCI and the raw Sentinel-2 B11 band for each of the three dates, so the feature count increased from 24 to 30. MTCI is related to leaf chlorophyll, while B11 is a shortwave infrared band related to vegetation and moisture. I also increased the nonlinear feature-space size of each Stage-3 specialist from 600 to 1,200 components and used a validation-selected probability adjustment for Stage 2 and Stage 3.

This is the current three-date Sentinel-2 model. It reached macro F1 **0.2344** and weighted F1 **0.7974** on the same 5,500,269 test pixels.

| Completed full run | Features and main change | Test macro F1 | Test weighted F1 |
|---|---|---:|---:|
| First parcel-disjoint baseline | 24 Sentinel-2 features, parcel-separated fitting and scoring | 0.2248 | 0.8018 |
| Protocol-repaired run | Same 24 features, honest Stage-1 routes for Stage-2 training, separate calibration and tuning parcels | 0.2283 | 0.8050 |
| Current three-date Sentinel-2 run | 30 features, larger Stage-3 specialists, adjusted probability rule | **0.2344** | 0.7974 |

The total macro F1 gain from the first parcel-disjoint baseline to the current run is 0.0096. However, the final step changed the features, Stage-3 size, and probability rule together, so I cannot separate how much each change contributed. The current score is real, but the cause of the whole gain is not isolated.

The strongest classes remain the crops with many parcels. Rubber reaches F1 0.8721, pineapple reaches 0.4208, rice reaches 0.4181, oil palm reaches 0.4158, and durian reaches 0.3827. Some rare crops moved from zero to a small non-zero score, but the values are still too low for operational use. Jackfruit reaches 0.0810, mango reaches 0.0659, rambutan reaches 0.0184, mangosteen reaches 0.0132, coconut reaches 0.0085, longan reaches 0.0054, and Langsat remains at zero.

This is the unflattering part of the result. The current model is useful for the common crops, but it still does not solve the rare crops.

## 4. What failed and what the failure taught me

Then I tested a cost-sensitive SVM because I expected larger class weights to help the rare crops. I wrote the success rule before running it, and the weighted model had to beat the unchanged model on the same validation rows before it could receive another test-fold score.

The weighted model failed that rule. Its validation macro F1 was 0.2272 compared with 0.2294 for the unchanged current model, so I did not calculate a new full test score. Longan improved slightly, but several other crops lost enough to make the overall result worse.

After that, I checked why the weights did so little, and I found that most of them were exactly 1.00. The sampling caps had already balanced the four Stage-2 groups, so the new weighting formula saw four equal groups and made almost no change. At first, this looked like an experiment that simply failed, but it exposed a more important imbalance inside each group.

Stage 2 balances field crops, plantations, orchards, and the sink group, but it does not balance the crop types inside those groups. In the plantation candidate pool, 96.82 percent of the rows are rubber, 3.10 percent are oil palm, and only 0.08 percent are coconut. When Stage 2 draws 200,000 plantation rows, it receives about 193,645 rubber pixels and only 159 coconut pixels. Coconut and rubber share the same plantation label at this stage, so balancing the four group totals never fixes this internal imbalance.

I checked the size of the routing problem by freezing every trained model and replacing only the learned Stage-2 route with the correct group from the label. This is not a deployable model because the correct route is unknown during real prediction, but it shows how much performance is being lost at that decision. Macro F1 increased from 0.2294 to 0.3785, which is a gain of 0.1491 with every classifier frozen.

That diagnostic does not mean a learned model can recover the full 0.1491. It means Stage-2 routing is a large source of error and is worth improving.

## 5. Rebalancing crop influence inside Stage 2

So I ran a more focused experiment. I kept the total weight of each Stage-2 group unchanged, but I redistributed that weight inside the group so that underrepresented crop types had more influence. Stage 1 and all Stage-3 specialists stayed frozen, and I refitted Stage 2 twice on the same rows. The control used the old weights, and the treatment used the new subtype weights.

The control reproduced the original Stage-2 model exactly, which confirmed that the same training pool was used. The treatment increased validation macro F1 from 0.2294 to **0.2375**, a gain of **0.0081**. It also beat the control at all 169 tested probability settings, and the gain at the original fixed setting was 0.0069.

| Validation result | Control | Subtype-mass treatment |
|---|---:|---:|
| Macro F1 | 0.2294 | **0.2375** |
| Crops with F1 at least 0.01 | 10 of 13 | 10 of 13 |

The result worked, but not in the way I predicted. Coconut routing nearly doubled from 13.7 percent to 26.6 percent, but coconut F1 did not improve. Most of the useful gain came from rice and oil palm, which have enough parcels for the later specialist models to learn from. This told me that giving a rare pixel the correct route is not enough when the final specialist still cannot separate that crop.

The 0.0081 result is measured on one half of the validation fold, and it is not directly comparable to the 0.2344 test score. I also held the 200,000-row training draw fixed, so I still need to repeat the experiment with several fresh draws before I carry the new weights into the final cascade. This is the next check, not a detail I should skip.

## 6. The evaluation mistake I had to correct

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

## 7. New sensor-fusion probe

After an independent technical review challenged my claim about a fixed spectral ceiling, I ran one more parcel-disjoint probe to check whether the limitation was caused specifically by Sentinel-2. I compared the same rows and the same parcel split under two feature sets. The first arm used 40 Sentinel-2 index features from five dates, and the second arm used all 153 available features, adding terrain information from the digital elevation model and Sentinel-1 radar measurements.

The fused feature set increased probe macro F1 from **0.3945 to 0.4401**. Rice gained 0.1084, mangosteen gained 0.0793, mango gained 0.0754, rambutan gained 0.0616, durian gained 0.0590, and oil palm gained 0.0547. Coconut gained only 0.0085, longan gained 0.0285, and Langsat did not move.

| Feature set in the controlled probe | Macro F1 |
|---|---:|
| Sentinel-2 only | 0.3945 |
| Sentinel-1 + terrain + Sentinel-2 | **0.4401** |

Before running it, I wrote that the optical-ceiling explanation would weaken if coconut, mangosteen, rambutan, or longan gained at least 0.10 F1. None reached that threshold, so the written decision rule did not change my conclusion. Still, the 0.0456 macro gain is useful evidence that radar and terrain add information for several mid-frequency crops, and the fused pipeline deserves further study.

These probe scores cannot be placed beside the full cascade score of 0.2344 as if they are competing models. The probe has a balanced crop-only population, five dates, and no non-crop false positives, while the full cascade has three dates and is scored over the natural tile population. The valid comparison is only 0.3945 against 0.4401 inside the same probe.

## 8. What I now think is limiting the model

At this point, I think there are three connected limits.

First, Stage-2 routing is still losing a large amount of recoverable information for rice, cassava, pineapple, durian, oil palm, and mango. These crops have enough parcels for a better route to matter, and the subtype-mass result already moved rice and oil palm in the right direction.

Second, the rare crops have too few independent parcels. Langsat has 1,639 training pixels from only 10 training parcels, and one parcel contains 1,310 of those pixels. Its complete validation fold has only 13 pixels, and its test fold has 191 pixels. A per-class Langsat score is not reliable enough to guide model selection.

Third, three dates of Sentinel-2 may not contain enough generalisable information to separate visually similar orchard species on unseen parcels. The fused probe improved several crops, so I do not think Sentinel-2 is the only useful sensor, but the added information still did not rescue coconut, longan, or Langsat.

So the current working conclusion is that the model is limited by both routing and data support. Better fitting can still improve the mid-frequency crops, but new parcels are the intervention most likely to change the rare-crop result.

## 9. How the current result relates to the earlier Random Forest paper

The Random Forest benchmark I have referred to is the earlier Rayong crop-classification study using Sentinel-2 imagery. It used a flat Random Forest, which means one model predicted the final land-use class directly instead of passing pixels through several stages. The study used a different image period, mainly 2024 with additional 2020 data, and it evaluated 15 land-use classes on about 303,947 test pixels.

That paper reported about 0.716 overall accuracy, 0.678 Cohen's kappa, and 0.714 weighted F1. Its reported overall F1 was about 0.71, oil palm reached about 0.81, and the per-class table has been used as an approximate F1 0.33 reference for the orchard and rare crops.

My current result is not a direct repeat of that experiment. I use 2018 imagery and the 2018 LDD survey, a three-stage SVM, a natural full-tile test population, and macro F1 over the 13 economic crops. I also cannot reconstruct from the available material whether the Random Forest test parcels were separated completely from its training parcels. My parcel experiment shows that this choice strongly changes my own rare-crop scores, but it does not prove what would happen to the Random Forest result.

So I will use the Random Forest paper as related work and a motivation for the crop classes, not as a controlled head-to-head result. A fair classifier comparison requires the same parcels, dates, features, labels, and scoring rule.

## 10. The collaborator's XGBoost workflow

When I first inspected the collaborator's work, I only found separate training components and believed that the complete prediction chain had not been connected. I checked the newer version of their inference code, and it now contains three connected models.

1. A water model predicts the probability that a pixel is water. Pixels with water probability at or above 0.56 are removed from the crop candidates.
2. A building model then predicts the probability that each remaining pixel is a building. Pixels with building probability at or above 0.56 are also removed.
3. A flat XGBoost crop model classifies the remaining pixels into 13 economic crops plus an `others` class.

Their design and my SVM design solve the early filtering problem differently. Their cascade removes water and buildings before a flat crop classifier makes the final decision. My SVM cascade first separates four broad land groups, then routes economic crops into field, plantation, and orchard groups, and finally uses a specialist model inside each group.

Their crop report is currently calculated only on the pixels that survive the water and building filters. This means the crop classifier is not charged inside that report for a crop pixel removed by an earlier filter, so the final comparison also needs a full-population score that counts errors from all three models together.

I can explain this workflow from the code, but I cannot yet state how well their complete model performs. I do not have a verified score from their final cascade under the strict full-population and parcel-disjoint evaluation used for my SVM, and I also do not have my SVM result under their exact evaluation because the SVM has no separate building filter. Their result should stay open until both models are run on the same rows, the same parcel split, the same label definitions, and the same scoring denominator.

I also found a shared protocol risk. Their current data preparation appears to split sampled pixels by row, so pixels from the same parcel can appear in both training and evaluation. I made the same mistake in my earlier probe, and it strongly increased the rare-crop scores. I would raise this as a joint methodological issue to fix together, not as a criticism of their model.

## 11. Proposed angle for the joint conference paper

I do not think the strongest paper is a simple table asking whether SVM or XGBoost has the larger F1. The evaluation protocol can move the score more than many model changes, and a table made before the protocol is aligned could reward the measuring method instead of the classifier.

The stronger angle is:

> **Hierarchical crop mapping under extreme class imbalance: a parcel-disjoint comparison of SVM routing and XGBoost filtering in Rayong Province.**

Both models are hierarchical, but their hierarchies represent different ideas. The SVM I developed uses agricultural knowledge to route pixels through economic crops, field crops, plantations, and orchards. The XGBoost system uses exclusion filters for water and buildings, then lets one flat crop model classify the survivors. A fair comparison can show where each design loses information, which is more useful than only naming a winner.

I propose four contributions for the paper.

1. **A shared parcel-disjoint evaluation.** Both models should use the same 2018 parcel assignment, the same label erosion, the same 13 crop definitions, and the same full-population scoring rule. If the joint study compares SVM against XGBoost as classifiers, they should also receive the same feature matrix and date window. If the inputs remain different, the paper should call it a comparison of two complete systems instead of attributing the difference to the classifier. A pixel-split result can also be reported as a protocol sensitivity experiment, but it should not be presented as unseen-parcel performance.
2. **A comparison of two hierarchical strategies.** The joint study can compare the SVM's semantic routing against the XGBoost water and building filters, and then trace how much error is created at each stage instead of treating the final F1 as a black box.
3. **A rare-class support analysis.** The paper can show that pixel count is not the same as independent information because thousands of pixels may come from only a few parcels. This is especially important for Langsat, coconut, longan, rambutan, and mangosteen.
4. **A disagreement map between the two models.** When both models agree with high confidence, the output can be treated as a stronger candidate prediction. When they disagree, those parcels can become targets for field checking or for the next LDD survey. This gives a practical use for both models even if one has a higher overall score.

Using both models as a formal ensemble should remain optional for now. To combine their probabilities honestly, the collaborator and I would first need to convert both outputs into the same set of final labels, calibrate both models on the same parcel-disjoint validation data, and train any combining rule without reading the test fold. Until that work is completed, a consensus and disagreement analysis is safer than claiming that an ensemble improves accuracy.

The result of the collaborator's XGBoost model will decide the final paper emphasis. If both models fail on the same rare crops, the paper becomes a strong study of data support and evaluation protocol. If XGBoost recovers classes that the SVM misses, the paper can study whether tree-based filtering handles the decision boundary better. If the SVM recovers classes that XGBoost misses, the agricultural routing structure becomes the main explanation. And if their errors are complementary, the disagreement analysis provides a justified path toward an ensemble.

This keeps the paper open to the result instead of deciding the story before the experiment.

## 12. Current limitations

- The current three-date Sentinel-2 model's test score has already been examined after several development stages, so I am treating the next full test read as expensive. I will use it only for one final configuration that is written down before training starts.
- The current hyperparameters were originally chosen with a pixel-level search based on accuracy, so I am retuning Stage 2 and the orchard specialist with parcel-grouped cross-validation and macro F1.
- The subtype-mass gain has been measured on one fixed training draw and one validation population. It needs fresh-draw sensitivity checks before promotion.
- The fused result is an isolated controlled probe, not a complete Sentinel-1 + terrain + Sentinel-2 cascade result.
- The labels come from the 2018 LDD survey, and mixed-crop compound codes are excluded.
- The earlier Random Forest paper uses a different image period, feature set, model structure, and evaluation population. My experiments show that evaluation protocol strongly changes my SVM scores, but they do not prove how the Random Forest result would change under a parcel split.

## 13. What I have planned next

After the independent review, I arranged the remaining work so that cheap checks come before another full test score. At the time of this report, the first two steps are complete and the final SVM result is still pending.

| Order | Planned work | What changes | Current status |
|---|---|---|---|
| 1 | Save exact training and calibration row identities | Adds reproducibility records only and does not change model behaviour | Completed |
| 2 | Compare Sentinel-2 against Sentinel-1 + terrain + Sentinel-2 on the same unseen parcels | Tests whether radar and terrain add information | Completed, macro F1 increased from 0.3945 to 0.4401 inside the controlled probe |
| 3 | Repeat the Stage-2 crop-rebalancing experiment over three fresh training draws | Tests whether the +0.0081 validation gain survives a different sample | Next result pending |
| 4 | Retune Stage 2 and the orchard specialist | Replaces the old pixel-based accuracy search with parcel-grouped macro F1 selection | Pending |
| 5 | Rebalance crop influence inside the plantation and orchard specialists | Tests whether coconut and the rare orchards fail inside the final specialist even after correct routing | Pending |
| 6 | Test date-difference features only if time remains | Changes feature weighting but adds no new image dates | Optional |
| 7 | Train one final combined cascade | Includes only changes that passed their written validation rules, then uses the final test once | Pending |

First, I will repeat the Stage-2 subtype-mass control and treatment over three fresh training-pool draws. The new weights will enter the final cascade only if they beat the control in all three draws and the average gain is at least 0.002 macro F1.

Then I will retune Stage 2 and the orchard specialist with three-fold parcel-grouped cross-validation using macro F1, because the old search shared parcels and optimised accuracy. After that, I will test subtype weighting inside the plantation and orchard specialists, especially because better routing did not improve coconut once it reached the plantation model.

Finally, I will combine only the changes that pass their written validation rules and calculate one final strict test result. My current planning range is 0.24 to 0.26 macro F1, with 0.28 treated only as an upside case rather than a prediction. If nothing passes validation, I will keep the current three-date Sentinel-2 model instead of spending another test read on an unchanged configuration.

For the rare crops, the most useful next resource is more independent parcels from another LDD survey year or a neighbouring tile. Model changes can still improve the middle-frequency crops, but I do not want to promise that a new weighting formula can replace missing parcel data.

## 14. Main points for discussion

1. Is the strict parcel-disjoint, full-population score the correct primary result for the conference paper?
2. Can I obtain additional parcel surveys from 2020, 2024, or a neighbouring tile for the rare crops?
3. Can the collaborator and I agree on one shared split, label mapping, feature set, and scoring denominator before comparing SVM and XGBoost?
4. Should the joint paper focus first on the fair comparison and error analysis, with an ensemble kept as optional follow-up work?

The main lesson from this stage is that the model still has real routing headroom, but rare-crop performance is also limited by how few independent parcels exist. I now have a clearer path for improving the classes that have enough support, and I also have a more honest boundary around what the current dataset can answer.
