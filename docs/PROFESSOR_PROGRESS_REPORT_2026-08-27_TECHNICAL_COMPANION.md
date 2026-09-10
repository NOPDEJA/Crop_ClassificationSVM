# Technical Companion to the Progress Report on Crop Classification in Rayong Province

**Date:** 27 August 2026, updated 3 September 2026  
**Study area:** Rayong Province, Thailand  
**Primary system:** Three-stage hierarchical Support Vector Machine  
**Purpose:** A self-contained explanation of every technical term, mathematical step, experimental decision, result, limitation, and planned action in the progress report

## 1. How to read this companion

I wrote this companion because the main progress report is designed to be readable in one sitting, so it cannot stop and define every term. This document does stop. It explains what each part means, how the calculation works, why I used it, what happened in the experiments, and what I can and cannot conclude from the result. The explanation does not require access to my code, saved models, or internal run names.

There are also four status notes from my checks of the progress report, the last two added on 3 September 2026.

First, the report correctly says in the results and planning sections that the Stage-2 subtype-weight experiment was repeated with three fresh training draws and passed its written rule. One sentence in the limitations list still says that those fresh-draw checks are needed. That sentence is stale. The checks are complete, all three favoured the treatment, and the mean improvement was 0.0078 macro F1.

Second, the report describes the probability adjustment in shortened language. The exact implementation does not simply divide a probability by class frequency. It moves each class toward a uniform target frequency by the equation shown in Section 12. This is best understood as a validation-selected operating-point adjustment for macro F1, not as a claim that I recovered the true class frequency of the whole province.

Third, both of the items above are now closed, and so is the whole planned sequence. The retuning experiment finished and passed its gate, the Stage-3 specialist rebalancing finished and **failed** its gate, and one final consolidated cascade was trained and scored once on the test fold at macro F1 **0.2429**. Sections 16, 19, and 21 to 25 have been updated for that result, and Section 22 now proposes a different design for the joint paper.

Fourth, one correction about the collaborator's system. An earlier version of the progress report said their prediction chain was not connected. That was true when I first read their code and it has been false since late August. Their chain is connected, and Section 21 describes it as it now stands after a re-read on 3 September 2026.

## 2. The problem in one complete example

The goal is to turn satellite and terrain measurements into a 10-metre crop map. One pixel covers an area of about 10 by 10 metres, or 100 square metres. For each labelled pixel, I build a list of numerical measurements such as vegetation greenness in October, shortwave-infrared reflectance in November, and radar backscatter. This list is the pixel's **feature vector**. The known land-use code from the Land Development Department survey is the **target label**.

The model does not jump directly from the feature vector to one of 13 crops. It works as a cascade.

1. Stage 1 asks whether the pixel is an economic crop, water, forest, or another land type.
2. If Stage 1 says economic crop, Stage 2 asks whether it belongs to the orchard, plantation, field-crop, or sink group.
3. If Stage 2 selects orchard, plantation, or field, the matching Stage-3 specialist predicts the final crop.

For example, a coconut pixel should follow this route:

```text
economic crop -> plantation -> coconut
```

If Stage 1 calls it forest, the crop is lost immediately. If Stage 2 sends it to orchards, the plantation specialist never sees it, so it cannot later choose coconut. This is called **error propagation** or **routing loss**. The hierarchy reduces the number of unrelated classes competing at once, but an early error can block the correct final answer.

## 3. Study area, spatial data, and labels

### Sentinel-2 tile 47PQQ

Sentinel-2 divides its image products into named map tiles. **47PQQ** is the tile used for this study and it covers the Rayong study area. The code is an identifier, not a model version. The report names it so the geographic image footprint is clear.

### Raster and pixel

A **raster** is a rectangular grid. Every grid cell is a pixel and stores a value. A satellite band can store reflectance, a terrain raster can store elevation, and a label raster can store a land-use code. All layers must place the same real-world location in the same row and column before they can be combined.

A **GeoTIFF** is a TIFF image with map information attached, including its coordinate system, pixel size, and geographic position. This lets different layers be aligned geographically instead of only visually.

### Coordinate reference system

A **coordinate reference system**, or CRS, defines how positions on Earth are represented as map coordinates. The aligned data use EPSG:32647, which is WGS 84 projected into UTM zone 47N. A projected CRS is useful here because distance and pixel size can be expressed in metres.

### Spatial resolution and resampling

The working resolution is 10 metres. Some source measurements do not begin on exactly the same grid, so they are **reprojected** and **resampled** to match the label raster. Reprojection changes the coordinate representation. Resampling estimates values on the destination grid. The important requirement is that a row in the feature matrix and the same row in the label vector refer to the same ground location.

### Parcel and LDD survey label

A **parcel** is one mapped land unit from the 2018 Land Development Department, or LDD, survey. Its land-use category is recorded with an **LU code**, meaning land-use code. I convert the parcel polygons into a raster, which is called **rasterisation**, so every labelled 10-metre pixel receives the code of its parcel.

The model predicts these 13 economic crops:

| Group | LU code | Crop |
|---|---:|---|
| Field | 2101 | Rice |
| Field | 2204 | Cassava |
| Field | 2205 | Pineapple |
| Plantation | 2302 | Rubber |
| Plantation | 2303 | Oil palm |
| Plantation | 2405 | Coconut |
| Orchard | 2403 | Durian |
| Orchard | 2404 | Rambutan |
| Orchard | 2407 | Mango |
| Orchard | 2413 | Longan |
| Orchard | 2416 | Jackfruit |
| Orchard | 2419 | Mangosteen |
| Orchard | 2420 | Langsat |

### Label erosion and mixed pixels

I remove a 30-metre strip from parcel boundaries before training and scoring. At 10-metre resolution, this is a three-pixel erosion. A **mixed pixel** covers more than one real surface, such as half rubber trees and half road, even though the label raster can store only one code. Boundary erosion removes many of these uncertain pixels and reduces disagreement caused by map geometry rather than crop appearance.

This choice also has a cost. Small parcels can lose many pixels, and the remaining sample represents parcel interiors more than boundaries. I accept that trade because the purpose is to learn crop signals from cleaner labels, but it must be used consistently for both models in a fair comparison.

### Background and nodata

**Background** means that the pixel is not one of the 13 target crops in the final crop score. It can be water, forest, road, built land, another agricultural code, or another non-target class. The value 0 is used for this final no-crop state. The raster data also use a separate no-data sentinel in some preparation steps. No-data means that a valid observation is unavailable, not that the land is a true background class.

## 4. Sentinel-2 optical data and the 30 current features

### What Sentinel-2 measures

Sentinel-2 is an optical satellite mission. It measures reflected sunlight in several wavelength bands. Plants, water, soil, and built surfaces reflect different proportions of blue, green, red, near-infrared, red-edge, and shortwave-infrared light. A crop's spectral response also changes with growth stage, water content, canopy structure, and management.

The current full cascade uses one monthly composite from October, November, and December 2018. A **monthly composite** combines usable observations from a month into one representative image and reduces the effect of clouds and individual acquisition noise. Three dates allow the model to observe some seasonal change, although they do not cover a full annual crop cycle.

The source is Sentinel-2 Level-2A. **Bottom-of-atmosphere reflectance** means that atmospheric effects have already been corrected to estimate the proportion of light reflected from the surface. The stored digital numbers are divided by 10,000 before the formulas are calculated, so a stored value such as 3,000 becomes reflectance 0.30.

### Bands used in the formulas

| Band | Plain meaning | Main role here |
|---|---|---|
| B02 | Blue light | EVI and bare-soil contrast |
| B03 | Green light | Water contrast |
| B04 | Red light | Chlorophyll absorption and vegetation contrast |
| B05 | First red-edge band | MTCI denominator |
| B06 | Second red-edge band | MTCI numerator |
| B08 | Near infrared, or NIR | Strong vegetation response |
| B11 | Shortwave infrared 1, or SWIR1 | Moisture, vegetation, and soil response |
| B12 | Shortwave infrared 2, or SWIR2 | Additional moisture and material contrast |

Let `B`, `G`, `R`, `N`, `S1`, and `S2` mean blue, green, red, near infrared, SWIR1, and SWIR2 reflectance. The constant `epsilon = 0.000001` is added to several denominators so division remains numerically stable near zero.

### NDVI

```text
NDVI = (N - R) / (N + R + epsilon)
```

The **Normalized Difference Vegetation Index** becomes larger when near infrared is high and red reflectance is low, a common pattern for healthy green vegetation. I use it because crop canopies and non-vegetated surfaces often differ strongly on this axis. It is not a direct measurement of crop species and it can saturate in dense vegetation.

### EVI

```text
EVI = 2.5 x (N - R) / (N + 6R - 7.5B + 1)
```

The **Enhanced Vegetation Index** adds blue reflectance and correction coefficients to reduce some atmospheric and soil-background effects. In implementation, extreme values are clipped to the range from -3 to 3 because the denominator can become very small at abnormal pixels. I use EVI alongside NDVI because it responds differently in dense vegetation and imperfect background conditions.

### NDWI

```text
NDWI = (G - N) / (G + N + epsilon)
```

The **Normalized Difference Water Index** used here contrasts green and near-infrared reflectance. Open water often has low near-infrared reflectance, so the index helps separate water and wet surfaces. The name NDWI is used for more than one formula in remote-sensing literature, so the equation matters more than the abbreviation.

### BSI

```text
BSI = ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + epsilon)
```

The **Bare Soil Index** contrasts soil-sensitive red and shortwave-infrared response against blue and near infrared. I use it because exposed ground, recently prepared fields, built surfaces, and closed crop canopies do not behave the same way.

### NDBI

```text
NDBI = (S1 - N) / (S1 + N + epsilon)
```

The **Normalized Difference Built-up Index** contrasts SWIR1 and near infrared. Built surfaces can have a different balance from vegetation. It is not a perfect building detector in agricultural landscapes because dry soil can look similar, but it gives the model a useful built and dry-surface signal.

### MSAVI

```text
MSAVI = (2N + 1 - square_root((2N + 1)^2 - 8(N - R))) / 2
```

The **Modified Soil-Adjusted Vegetation Index** reduces the influence of visible soil when vegetation cover is incomplete. I use the MSAVI2 form shown above. It is useful for fields and young crops where soil remains visible between plants.

### SWIR to NIR ratio

```text
SWIR_NIR = S1 / (N + epsilon)
```

This ratio compares shortwave-infrared and near-infrared response. It is sensitive to moisture, canopy condition, soil, and shadow. Very dark pixels can make the ratio explode when `N` is close to zero, so I clip the implemented value to the range from 0 to 20. This protects later scaling from a small number of numerical outliers.

### Normalized SWIR ratio

```text
SWIR_RATIO = (S1 - S2) / (S1 + S2 + epsilon)
```

This index compares the two shortwave-infrared bands. Normalisation reduces the influence of their overall brightness and highlights their relative difference.

### MTCI

Let `RE1` be B05 and `RE2` be B06. Then

```text
MTCI = (RE2 - RE1) / (RE1 - R)
```

The **MERIS Terrestrial Chlorophyll Index** uses the shape of the red edge, which is the rapid reflectance increase between red and near infrared in green vegetation. It is related to canopy chlorophyll. I added it because the collaborator's feature family included it and because the earlier 24-feature SVM had no red-edge-derived column.

The denominator can cross zero on flat or non-vegetated spectra, so implemented values are clipped to -10 through 10. Later permutation tests placed MTCI last among the index families in both Stage 1 and the orchard specialist. That means it added little measured predictive information in the current three-date system. It does not prove that all red-edge bands are useless because one compressed index is not the same as giving the model the original red-edge reflectances.

### Raw B11

The **raw B11 feature** is the surface reflectance in Sentinel-2's SWIR1 band after scaling. It is called raw only because it is not combined into an index. It is still an atmospherically corrected satellite measurement.

I added B11 because an index compresses two or more bands into one number and can discard absolute brightness information. Two pixels can have the same ratio but different original reflectance values. Raw B11 therefore lets the model use information that the derived indices may have removed. Its measured importance was moderate, so it was more useful than MTCI in this run, although the two features were added together and the final test gain cannot be assigned to B11 alone.

### Why there are 24 and then 30 features

The first parcel-disjoint and protocol-repaired systems use eight indices for each of three dates:

```text
8 indices x 3 dates = 24 features
```

The current system appends MTCI and B11 for each date:

```text
24 + (2 new features x 3 dates) = 30 features
```

The internal name **M0** means the protocol-repaired 24-feature checkpoint. **M5** means the current 30-feature checkpoint with larger Stage-3 capacity and a selected operating point. These names are experiment bookkeeping, not mathematical methods and not new model families.

## 5. Sentinel-1 radar features in the sensor-fusion probe

Sentinel-1 is a **synthetic aperture radar**, or SAR, mission. Unlike Sentinel-2, it transmits microwave energy and measures the returned signal, so it does not depend on sunlight and is much less obstructed by cloud. Radar response depends on surface roughness, geometry, water content, and canopy structure. This makes it potentially complementary to optical reflectance.

The processed images provide VV and VH polarisation. **VV** means vertically transmitted and vertically received energy. **VH** means vertically transmitted and horizontally received energy. Cross-polarised VH is often responsive to complex vegetation structure because branches and leaves scatter the signal in different directions.

Radar backscatter is stored in decibels, or dB. The conversion to linear power is

```text
linear power = 10^(dB power / 10)
```

The implemented VV and VH dB values are clipped to -30 through +5 to limit extreme values. For each acquisition, I calculate six features.

1. `VV_dB`, the clipped VV backscatter.
2. `VH_dB`, the clipped VH backscatter.
3. `VV_VH_RATIO = VV_dB - VH_dB`. In dB, subtraction represents a power ratio.
4. `RVI = clip(4 x VH / (VV + VH), 0, 1)`, calculated in the linear domain. RVI is the Radar Vegetation Index and responds to depolarised vegetation scattering.
5. `RFDI = clip((VV - VH) / (VV + VH), -1, 1)`, also in the linear domain. RFDI is the Radar Forest Degradation Index and measures the balance between co-polarised and cross-polarised response.
6. `DPR = VH / VV`, the dual-polarisation ratio in the linear domain.

An older duplicate feature called VVVH_DIFF was removed because it contained exactly the same values as VV_VH_RATIO. Keeping exact duplicates gives the repeated measurement extra influence without adding information, so removal was the correct decision.

The fused matrix contains 18 Sentinel-1 acquisitions and six features per acquisition:

```text
18 acquisitions x 6 features = 108 radar features
```

## 6. Terrain features in the sensor-fusion probe

A **digital elevation model**, or DEM, stores ground elevation. The study uses five terrain features.

### Elevation

Elevation is height in metres. It can be associated with soil, drainage, temperature, accessibility, and which crops are planted, although it is not a crop measurement by itself.

### Slope

For horizontal and vertical elevation gradients `dz/dx` and `dz/dy`, the slope is

```text
slope = arctangent(square_root((dz/dx)^2 + (dz/dy)^2))
```

The result is converted from radians to degrees. A central difference compares elevation on both sides of a pixel. Slope can separate flat agricultural plains from steep land where some crop types are less likely.

### Aspect

**Aspect** is the compass direction that a slope faces. It is calculated from the two elevation gradients and converted so 0 degrees is north and direction increases clockwise. Flat areas are assigned 0 because a direction is not meaningful there. Aspect can affect sunlight and moisture, although its value in a tropical agricultural tile may be smaller than elevation or slope.

### Topographic Position Index

```text
TPI = centre elevation - local mean elevation
```

The **Topographic Position Index** subtracts the mean elevation in a local 31 by 31 pixel neighbourhood from the centre elevation. Positive values indicate a local ridge or high point and negative values indicate a valley or depression. At 10 metres, the radius is 150 metres.

### Roughness

```text
roughness = local maximum elevation - local minimum elevation
```

Roughness is the elevation range in a local 7 by 7 pixel neighbourhood. It describes fine-scale terrain variation.

### Why the fused probe has 153 features

The controlled probe uses five dates of the original eight Sentinel-2 indices, 18 dates of six Sentinel-1 features, and five DEM features:

```text
(5 dates x 8 optical indices) + (18 acquisitions x 6 radar features) + 5 terrain features
= 40 + 108 + 5
= 153 total features
```

This probe does not use the 30-feature current full-cascade matrix. It asks a narrower question under a fixed crop-only experiment: does adding radar and terrain improve unseen-parcel separation compared with the same five-date optical rows? Keeping the rows and split fixed makes the feature family the main changed variable.

## 7. Data preparation before the SVM

### Missing-value imputation

Cloud gaps, invalid ratios, or unavailable observations can produce missing values. A **median imputer** replaces a missing value in each feature with that feature's median calculated from the fitting data. The median is less affected by extreme values than the mean.

The imputer must be fitted only on training data. If a test value helps determine the median, information has moved backwards from evaluation into training.

### Standardisation

Features have different units and ranges, so I standardise each one:

```text
standardised value = (original value - training mean) / training standard deviation
```

After scaling, a typical feature has mean near 0 and standard deviation near 1. This matters because the RBF kernel uses distance. Without scaling, a feature with large numerical units could dominate that distance even if it is not more informative.

### Feature order

The model receives a matrix in which each column has one fixed meaning. Training and inference must use exactly the same column order. A model trained with NDVI in column 1 and EVI in column 2 cannot safely receive the columns reversed later. Any added or removed feature therefore requires a rebuilt matrix and retraining unless the new columns are appended under a separately verified design.

## 8. How the Support Vector Machine works

### Feature vector and decision boundary

For one pixel, let the feature vector be:

```text
x = [x1, x2, ..., xd]
```

The letter `d` is the number of features. A binary linear SVM calculates:

```text
decision score = (w1 x x1) + (w2 x x2) + ... + (wd x xd) + b
```

The weights `w1` through `wd` set the direction of the decision boundary and `b` moves it. The sign of the decision score gives the side of the boundary. Its magnitude is a signed margin score. A positive score supports the target class and a negative score supports the rest, although the raw magnitude is not a probability.

### Margin and training objective

The SVM tries to separate classes with a wide **margin**, meaning a gap around the decision boundary. Perfect separation is often impossible, so the model also pays a loss for rows inside the margin or on the wrong side.

The LinearSVC implementation used after the kernel approximation has a squared-hinge form that can be written as

```text
objective = boundary complexity
          + C x sum of each row's weighted squared-hinge loss

row loss = sample weight x max(0, 1 - true sign x decision score)^2
```

The true sign is +1 for the class and -1 for the rest. `C` controls how strongly fitting errors compete with a simple, regularised boundary. A larger `C` punishes training errors more strongly and can create a more complex fit. A smaller `C` accepts more training error and applies stronger regularisation.

### One versus rest

SVM is naturally binary, so a multi-class stage is built with **one-versus-rest**, or OvR. For `K` classes, I train `K` binary decisions. The orchard model, for example, has one decision for durian against all other orchard crops, another for rambutan against all other orchard crops, and so on. Their calibrated outputs are later compared to choose one class.

### RBF kernel

A straight line in the original features may not separate two crops. The **radial basis function**, or RBF, kernel measures similarity as

```text
RBF similarity = exp(-gamma x squared distance between two feature vectors)
```

Here `exp` is the exponential function. Nearby feature vectors receive similarity near 1 and distant vectors approach 0. The parameter `gamma` controls how quickly similarity falls. A large `gamma` makes influence very local and can overfit. A small `gamma` makes the surface smoother and can underfit.

### Nyström approximation

A full RBF SVM on millions of pixels would require too much memory and time. I therefore use a **Nyström approximation**. It selects a set of landmark rows and builds an explicit nonlinear feature map approximately equal to the RBF kernel:

```text
approximate nonlinear features
= similarities from one pixel to the landmark pixels
  x a correction calculated from similarities among the landmarks
```

The landmark set is a selected subset of training rows. A linear SVM then works on these approximate nonlinear features. This keeps much of the nonlinear behaviour without constructing the full all-pairs kernel matrix.

The number of **Nyström components** is the number of landmark-based dimensions. More components can represent a richer boundary but require more time and memory. The earlier Stage-3 specialists used 600 components and the current full run uses 1,200. The change was intended to give difficult crop specialists more capacity, but it was introduced together with new features and a new operating point, so its individual contribution to the 0.0096 test gain is unknown.

### Hyperparameters

A **hyperparameter** is chosen outside the normal coefficient-fitting process. The main ones here are `C`, `gamma`, and the number of Nyström components. They interact. More components cannot help if `gamma` produces an unsuitable geometry, and a good feature map can still be underfit or overfit under the wrong `C`.

The old values were selected by a pixel-level cross-validation search that optimised accuracy. That is now considered a protocol weakness because parcels could cross folds and accuracy favours common classes. The current retune searches these values with complete parcels kept together and macro F1 as the selection score.

## 9. Why the system is hierarchical

The hierarchy adds agricultural structure. Stage 1 separates broad land cover. Stage 2 groups crops with related management and form. Stage 3 lets a specialist focus on a smaller set.

This can help because a rubber pixel does not need to compete directly against water, rice, Langsat, built land, and every other class in one flat boundary. It can also hurt because routing decisions are hard. The final probability is not a single jointly optimised 13-crop probability. It is the result of separate decisions.

### Stage-1 labels

| Stage-1 value | Meaning |
|---:|---|
| 1 | Economic crops, including the 13 targets |
| 2 | Water |
| 3 | Other valid land uses |
| 4 | Forest |

### Stage-2 labels

| Stage-2 value | Meaning |
|---:|---|
| 1 | Orchards |
| 2 | Plantations |
| 3 | Field crops |
| 4 | Sink |

The **sink class** contains economic land-use codes outside the 13 target crops. It gives those pixels a legitimate destination instead of forcing them into orchard, plantation, or field. No Stage-3 specialist exists for the sink, so its final crop output is background.

### Candidate population

A **candidate** is a pixel allowed to enter a later stage. Stage-2 candidates are pixels routed as economic crops by Stage 1. A crop pixel dropped before that point cannot be recovered by Stage 2 or 3. The composition of the candidate population therefore affects both learning and evaluation.

### Out-of-fold routing

For honest Stage-2 training, the model should see the kind of imperfect Stage-1 routes it will receive at deployment. But if Stage 1 predicts its own training pixels, those routes are too optimistic because the model has already fitted those rows.

I repair this with **out-of-fold**, or OOF, routing. Training parcels are divided into three parts. For each part, Stage 1 is fitted on the other two parts and predicts the held-out part. The three held-out predictions are joined. Every training-row route therefore comes from a Stage-1 model that did not fit that row or its parcel.

This change increased test macro F1 from 0.2248 to 0.2283, but the paired parcel uncertainty interval for the difference included zero. So I treat OOF routing as a validity repair, not as a proven accuracy improvement.

## 10. Data splits, leakage, and generalisation

### Training, validation, and test

The 24,323,769 labelled rows are divided approximately as follows:

| Role | Rows | Approximate share | Purpose |
|---|---:|---:|---|
| Training | 14,842,294 | 61% | Fit SVM boundaries and preprocessing |
| Validation | 3,981,206 | 16% | Calibrate probabilities and choose changes |
| Test | 5,500,269 | 23% | One final estimate on unseen parcels |

The validation fold is divided again into a **calibration half** and a **tuning half**. The calibration half fits the sigmoid curves and selects the operating-point cell. The tuning half reports the chosen cell once without having selected it. This is not a perfectly untouched validation design because operating-point selection uses the same half that fitted the sigmoids, but the quoted tuning score is still out of sample for both tasks.

### Parcel-disjoint split

A **parcel-disjoint** split assigns an entire parcel to only one fold. Pixels within one parcel are spatial neighbours and often share the same planting date, soil, management, and image conditions. Treating them as independent observations exaggerates the amount of new information.

### Pixel leakage

In a random pixel split, some pixels from a parcel can be used for training and nearby pixels from the same parcel can be used for testing. This is **data leakage** because the test is no longer a realistic unseen-parcel task. It does not necessarily copy identical rows, but it lets the model recognise parcel-specific patterns.

The earlier five-date crop probe scored macro F1 0.5852 under a pixel split. After parcel identities were reconstructed, 87.5% to 95.7% of rare-crop test parcels also appeared in training. The parcel-disjoint replay fell to 0.3945, and Langsat fell from 0.6774 to 0.0000. This is why I withdrew the claim that the high pixel-split scores proved generalisable spectral separation.

### Generalisation

**Generalisation** means performing well on data that were not represented during fitting. Here the intended question is not whether the model can label another pixel from a known parcel. It is whether it can label a parcel it has not seen. The parcel-disjoint design is stricter because it matches that question.

### GroupKFold

**GroupKFold** is cross-validation that keeps all rows with the same group identifier together. The current retune uses parcel ID as the group. In three-fold GroupKFold, each candidate hyperparameter setting is fitted three times, each time holding out a different set of complete parcels. The mean macro F1 across those folds selects the candidate.

## 11. From SVM margins to probabilities

### Signed margin

An SVM returns one decision score for each class. A larger margin usually means stronger support, but a margin of 2 is not automatically 80% probability. Different class models can also have different score scales.

### Platt sigmoid calibration

For each class, I fit two numbers called `A` and `B` on held-out calibration parcels:

```text
class sigmoid output = 1 / (1 + exp(A x class margin + B))
```

This is **Platt calibration**. The sigmoid converts an unbounded margin into a value between 0 and 1 by matching observed positives and negatives in the calibration rows. The base SVM is already fitted and is not refitted on these rows.

I fit one sigmoid per OvR class manually because automatic internal cross-validation can create a fold with too few positive examples for a very rare class. If a class does not have enough positives to fit its sigmoid, the implementation has a plain logistic transform as a safeguard. That fallback is a stability measure and is not as well calibrated as a fitted class-specific curve.

### Normalising independent outputs

The independently fitted class outputs do not have to sum to 1. I therefore calculate:

```text
normalised probability for one class
= that class's sigmoid output / sum of all class sigmoid outputs
```

This creates one sum-to-one vector for routing. It is simple and stable, but it is not proof that the numbers are perfectly calibrated multiclass probabilities. A jointly fitted multiclass calibration method could behave differently.

The progress report mentions pairwise coupling as a more complete literature method. Pairwise coupling requires probability estimates for class pairs and solves for a consistent multiclass vector. It is not a direct drop-in transformation of the existing OvR outputs, so adopting it would mean changing and validating the calibration design, not only replacing one line of normalisation.

### Calibration versus discrimination

**Discrimination** asks whether the model ranks the correct class above the wrong class. **Calibration** asks whether predictions assigned probability 0.7 are correct about 70% of the time. A model can rank classes well but have badly scaled probabilities, or have reasonable average calibration and still make poor class decisions. The operating-point adjustment changes decisions after calibration. It does not improve the underlying satellite representation.

## 12. The exact operating-point adjustment

Let `p_c` be the normalised calibrated probability for one class. Let `pi_c` be that class's prevalence among the applicable calibration candidates. If a stage has `K` possible classes, its uniform target is `1 / K`. The adjustment ratio is:

```text
class adjustment ratio = uniform target / calibration prevalence
                       = (1 / K) / pi_c
```

For adjustment strength `alpha`, the adjusted probability is:

```text
unnormalised adjusted value for class c = p_c x ratio_c^alpha

adjusted probability for class c
= unnormalised adjusted value for class c
  / sum of unnormalised adjusted values for all classes
```

Then the predicted class is the **argmax**:

```text
predicted class = class with the largest adjusted probability
```

Argmax means the class with the largest value.

When `alpha = 0`, every powered ratio becomes 1 and the probabilities are unchanged. When `alpha = 1`, the full uniform-target ratio is applied. Values above 1 allow a stronger shift. This favours classes that are rare in the calibration candidates and suppresses classes that are common there.

I search 13 values from 0.0 through 1.2 for Stage 2 and the same 13 values for Stage 3:

```text
13 Stage-2 alpha values x 13 Stage-3 alpha values = 169 combinations
```

The best pair is selected by macro F1 on the calibration half and then scored once on the tuning half. Stage 1 currently uses no such adjustment and takes plain argmax.

I call this an **operating point** because it changes the trade between common and rare classes without refitting the feature boundary. It resembles prior correction mathematically, but the target is uniform because macro F1 weights classes equally. It should not be interpreted as recovering the natural crop prevalence of Rayong.

The stages are routed separately. I do not multiply Stage-1, Stage-2, and Stage-3 probabilities into one joint probability. An earlier joint-product attempt scored worse, so the current hard hierarchy keeps each stage's own decision.

## 13. Evaluation metrics and their equations

For one crop class:

- **True positive**, or TP, means the crop is true and predicted.
- **False positive**, or FP, means the crop is predicted but the truth is another crop or background.
- **False negative**, or FN, means the crop is true but the model predicts something else or drops it.
- **True negative**, or TN, means neither truth nor prediction is that crop.

### Precision

```text
precision = TP / (TP + FP)
```

Precision asks, “Of the pixels predicted as this crop, how many were correct?” A model that predicts coconut everywhere would obtain high recall but extremely low precision.

### Recall

```text
recall = TP / (TP + FN)
```

Recall asks, “Of the true pixels of this crop, how many did the model find?” Routing a true crop to forest or the wrong Stage-2 group creates a false negative in the final score.

### F1

```text
F1 = 2 x precision x recall / (precision + recall)

The same equation using counts is:
F1 = 2TP / (2TP + FP + FN)
```

F1 is the harmonic mean, so it is low when either precision or recall is low. If no correct prediction exists for a crop, its F1 is 0.

### Macro F1

For 13 crops,

```text
macro F1 = sum of the 13 individual crop F1 scores / 13
```

Each crop has equal influence. This is why improving a rare crop can matter as much as improving rubber. It is also why macro F1 is noisy when a crop has only a few parcels or pixels in validation.

### Weighted F1

```text
weighted F1
= sum of (each crop's true-pixel count x that crop's F1)
  / total true-pixel count across the 13 crops
```

Rubber receives much more influence because it has much larger support. Weighted F1 can therefore remain high while several rare classes fail.

### Accuracy

```text
accuracy = number of correct predictions / number of evaluated predictions
```

Accuracy is intuitive but can be dominated by common classes. This is why the new hyperparameter search uses macro F1 instead of accuracy.

### Cohen's kappa

```text
kappa = (observed agreement - expected agreement)
        / (1 - expected agreement)
```

Expected agreement is calculated from the predicted and true class frequencies. Kappa is reported in the earlier Random Forest work, but it does not remove the need for identical test data and label definitions.

### Support

**Support** is the number of true evaluation examples for a class. Pixel support and parcel support are different. Langsat has 1,639 training pixels from only 10 training parcels, and one parcel supplies 1,310 of those pixels. Its validation fold has only 13 pixels and its test fold has 191. Thousands of neighbouring pixels are not equivalent to thousands of independent farms.

### Confusion matrix

A **confusion matrix** places true classes in rows and predicted classes in columns. The diagonal contains correct classifications. Off-diagonal cells show which classes are confused. In a cascade, it is also useful to inspect routing confusion before the final crop confusion.

### Strict full-population score

The main evaluation keeps all 5,500,269 rows in the test fold. Truth is one of the 13 crop codes or background 0. Predictions begin as background and only become a crop if the complete cascade assigns one. The score averages the 13 crop F1 values. Background is not a fourteenth term in macro F1, but crop predictions on background become false positives and reduce crop precision.

This is stricter than scoring only the pixels that survived an earlier filter. It charges the whole system for dropped crops and false crop alarms on non-crop land.

### Balanced crop-only probe

The fused probe first splits each crop's complete parcels into two sides and then draws up to 4,000 pixels per crop on each side. It contains only the 13 crops and gives them roughly equal pixel influence. This makes it useful for testing representation under controlled conditions, but it removes non-crop false positives and changes the class prevalence. Its macro F1 cannot be compared directly with the full-cascade macro F1.

### Bootstrap confidence interval

A **bootstrap** estimates uncertainty by repeatedly resampling the observed units with replacement. Because pixels inside a parcel are correlated, I resample parcels rather than individual pixels. For each bootstrap sample I recalculate both model scores and their paired difference. The middle 95% of these differences forms an approximate 95% confidence interval.

For the protocol repair, the point gain was 0.0035, but the interval included zero. This means the observed data do not clearly distinguish a small real improvement from parcel-to-parcel sampling variation. It does not mean the repaired protocol was unnecessary. It means accuracy improvement was not demonstrated.

## 14. Class imbalance and the attempted solutions

### What imbalance means here

**Class imbalance** means that some classes contribute far more training rows than others. Rubber can supply millions of pixels while a rare crop can supply only a few parcels. An SVM that minimises total loss can perform well by fitting the common class and sacrificing the rare class.

There are two levels of imbalance in Stage 2.

1. Between the four Stage-2 labels, meaning orchard, plantation, field, and sink.
2. Inside a Stage-2 label, meaning crops such as rubber, oil palm, and coconut all sharing the plantation target.

Balancing level 1 does not automatically balance level 2.

### Sampling cap

A **cap** limits the number of rows drawn from a large class. This prevents a huge class from taking almost the entire training pool and keeps computation manageable. The current Stage-2 production pool draws up to 200,000 rows per group, giving 800,000 total rows when all four groups meet the cap.

Caps discard repeated pixels from large classes, which reduces dominance and computational cost. But if the capped plantation pool remains 96.82% rubber and only 0.08% coconut, the internal crop imbalance survives.

### Upsampling

**Upsampling** repeats or resamples minority rows so they appear more often. It was removed from the parcel-disjoint baseline because repeated pixels do not create new independent parcels and can affect every fitted preprocessing step. Repetition can alter the median imputer, the scaling statistics, the Nyström landmarks, and the SVM loss at once. A controlled sample-weight experiment is easier to interpret.

### Cost-sensitive SVM and class weight

A **cost-sensitive** model gives some training mistakes more penalty than others. In the SVM objective, a sample weight multiplies the row's loss. The first weighting attempt used the Stage-2 group counts. But the caps had already made most group counts equal, so most calculated weights were 1.00. The treatment changed little and its validation macro F1 was 0.2272, below the unchanged model's 0.2294. It failed the written rule and did not receive a test score.

This failure was useful because it showed that the relevant imbalance was hidden inside each group.

### Subtype-mass weighting

For one crop subtype inside a Stage-2 group, let `subtype count` be its count after the cap and let `largest subtype count` be the largest crop count in that group. The initial row weight is:

```text
initial subtype weight = square_root(largest subtype count / subtype count)
```

The square root is a **tempering** choice. Full inverse frequency would give very rare classes enormous weight and could destabilise the boundary. Square-root inverse frequency increases their influence more cautiously.

The weights are then renormalised inside each group. For one row, the final weight is:

```text
final row weight
= initial weight for that row's crop
  x number of rows in the group
  / sum of all initial row weights in the group
```

Therefore,

```text
sum of final weights in one group = number of rows in that group
```

The total mass of plantation, orchard, field, and sink remains unchanged. Only the distribution among crop subtypes inside a group changes. This isolates the intended intervention and avoids solving coconut's problem by simply making all plantation rows more important than all field rows.

### Control and treatment

The **control** uses the existing settings and no subtype redistribution. The **treatment** uses the new subtype weights. Both are fitted on the same rows with Stage 1 and Stage 3 frozen. A paired design means the main difference is the treatment itself.

The control reproduced the earlier Stage-2 model byte for byte. A **SHA-256 hash** is a digital fingerprint of a file. Matching hashes are direct evidence that the separately saved model contents are identical. This check was important because the older run had not saved its Stage-2 row indices explicitly, so the training draw had to be reconstructed from its random-number sequence.

### What happened

On the original fixed draw, treatment macro F1 was 0.2375 and control was 0.2294, a gain of 0.0081. Treatment also beat control at every one of the 169 operating-point cells. Coconut's correct Stage-2 route increased from 13.7% to 26.6%, but final coconut F1 did not improve. Rice and oil palm provided much of the final gain.

This result separates two problems. Better routing can help crops whose Stage-3 specialist already has enough independent information. For coconut, a better route still hands the pixel to a specialist that cannot reliably separate coconut from rubber and oil palm.

### Fresh-draw sensitivity

A **training draw** is one random selection of capped pixels from a larger candidate pool. A result might depend on unusually favourable sampled rows, so I repeated control and treatment on three new draws.

| Fresh draw | Treatment minus control macro F1 |
|---:|---:|
| 1001 | +0.0072 |
| 1002 | +0.0087 |
| 1003 | +0.0074 |
| Mean | **+0.0078** |

The predeclared rule required treatment to be at least as good as control in all three draws and the mean gain to be at least 0.002. Both conditions passed. So subtype-mass weighting is approved for the final combined validation experiment. It has not yet produced a new full test score.

## 15. Diagnostics, controls, and decision rules

### Oracle routing

An **oracle** uses information that is available only because the true labels are known. I froze all fitted models and replaced the learned Stage-2 group with the true group. Validation macro F1 rose from 0.2294 to 0.3785, a difference of 0.1491.

This is not deployable because a real map does not know the correct route in advance. It is a diagnostic upper bound for the current downstream models under perfect Stage-2 routing. It shows that routing is important, but it does not promise that a learned router can recover the whole gap.

### Frozen component

A **frozen component** is kept exactly unchanged while another component is tested. Freezing Stage 1 and Stage 3 during the Stage-2 weight experiment reduces confounding. If all three stages changed together, a score difference could not be attributed to Stage 2.

### Predeclaration and gate

A **predeclaration** records the comparison, metric, data split, and acceptance threshold before the result is observed. A **gate** is the written condition a change must pass before it can move forward. This prevents an experiment from being called successful only because one unexpected number looked attractive after the run.

The fresh-draw gate required wins on all three draws and mean gain at least 0.002. The retuning gate requires treatment tuning macro F1 to exceed control by at least 0.002 and requires the number of **alive crops**, crops with F1 at least 0.01, not to decrease.

The alive-crop guard prevents a small average gain from hiding the collapse of one crop to almost zero.

### Falsifier

A **falsifier** is a result that would weaken a proposed explanation. Before the sensor-fusion probe, I wrote that the optical-ceiling explanation would weaken if coconut, mangosteen, rambutan, or longan gained at least 0.10 F1 from adding radar and terrain.

None crossed +0.10, so the prewritten branch did not trigger. But the overall macro gain of 0.0456 and several medium-size crop gains still show that radar and terrain contain useful information. A falsifier threshold is not the same as saying all smaller gains are worthless.

### Confounding

**Confounding** means more than one important factor changed between two results. From the baseline to the current model, the features, Stage-3 capacity, and operating point all changed. The test macro F1 rose by 0.0096, but I cannot assign a fraction of that gain to each cause. The score is valid for the complete current configuration, while the causal explanation remains unresolved.

## 16. What each completed full run means

### First parcel-disjoint baseline

This run used 24 optical features, no upsampling, no automatic class weighting, capped large classes, a real Stage-2 sink, and true-group Stage-3 training. It scored macro F1 0.2248 and weighted F1 0.8018 on 5,500,269 unseen-parcel test rows.

The decision was to establish an honest baseline before trying to maximise the score. Its value is mainly methodological because it replaced the old pixel-split reference with an unseen-parcel result.

### Protocol-repaired run, internally M0

This run kept the same 24 features and SVM settings, but added OOF Stage-1 routes for Stage-2 training and split validation into calibration and tuning parcels. It scored macro F1 0.2283 and weighted F1 0.8050.

The point gain of 0.0035 was not clearly above parcel-level uncertainty. The reason to keep the repairs is that they remove optimism and separate model fitting from model choice, not because the score proved they increase accuracy.

### Three-date optical run, internally M5

This run added MTCI and raw B11 for three dates, increased each Stage-3 Nyström map from 600 to 1,200 components, and used selected Stage-2 and Stage-3 operating strengths. It scored macro F1 0.2344 and weighted F1 0.7974.

| Crop | Current F1 | Interpretation |
|---|---:|---|
| Rubber | 0.8721 | Strong and supported by many parcels |
| Pineapple | 0.4208 | Moderate |
| Rice | 0.4181 | Moderate |
| Oil palm | 0.4158 | Moderate |
| Durian | 0.3827 | Moderate |
| Cassava | 0.3451 | Moderate |
| Jackfruit | 0.0810 | Low |
| Mango | 0.0659 | Low |
| Rambutan | 0.0184 | Very low |
| Mangosteen | 0.0132 | Very low |
| Coconut | 0.0085 | Near zero |
| Longan | 0.0054 | Near zero |
| Langsat | 0.0000 | No successful detection under F1 |

The result supports operational confidence only for the stronger common crops. It does not support the claim that the 13-crop map is reliable for every class.

### Final consolidated cascade

This is the run the whole planned sequence was building toward. It combines only the two interventions that passed their written gates: the Stage-2 subtype-mass weighting described in Section 14, and the retuned Stage-2 and orchard-specialist hyperparameters described in Section 19. The Stage-3 specialist rebalancing failed its gate and is **not** in this configuration, so the plantation specialist is unchanged from M5. The configuration was written down before training, and the test fold was read once within this plan. It scored macro F1 **0.2429** and weighted F1 **0.7949** on the same 5,500,269 unseen-parcel test rows, an observed gain of **0.0085** over M5. Note that weighted F1 moved in the opposite direction, from 0.7974 down to 0.7949.

| Crop | M5 F1 | Final F1 | Change | Interpretation |
|---|---:|---:|---:|---|
| Rubber | 0.8721 | 0.8658 | -0.0063 | Strong, unchanged in practice |
| Oil palm | 0.4158 | **0.5587** | **+0.1429** | The single largest gain in the run |
| Rice | 0.4181 | 0.4457 | +0.0276 | Improved |
| Pineapple | 0.4208 | 0.4205 | -0.0003 | Unchanged |
| Durian | 0.3827 | 0.3642 | -0.0185 | Slightly worse |
| Cassava | 0.3451 | 0.3379 | -0.0072 | Slightly worse |
| Mango | 0.0659 | 0.0812 | +0.0153 | Still low |
| Rambutan | 0.0184 | 0.0406 | +0.0222 | Still very low |
| Jackfruit | 0.0810 | **0.0198** | **-0.0612** | Regressed, unexplained |
| Mangosteen | 0.0132 | 0.0112 | -0.0020 | Still very low |
| Coconut | 0.0085 | 0.0095 | +0.0010 | Still near zero |
| Longan | 0.0054 | 0.0024 | -0.0030 | Still near zero |
| Langsat | 0.0000 | 0.0000 | 0.0000 | Still no detection |

Three technical points about this table.

First, the **distribution of the gain matters more than its size**. Oil palm and rice account for essentially all of it, and both are mid-frequency crops with enough independent parcels for a Stage-3 specialist to fit. This is the outcome the Stage-2 routing diagnosis in Section 14 predicted, so the mechanism and the result agree.

Second, the **rare crops did not move**, which was predicted in writing before the run. Coconut gained 0.0010. This is now a completed test-fold result rather than a validation-half projection, which is why Section 25 no longer lists it as an open question.

Third, **jackfruit regressed by 0.0612 and I cannot explain it**. Nothing in the validation-half measurements from any earlier experiment predicted it. The most plausible mechanism is that the retuned orchard specialist, which has double the Nystrom components and a higher penalty term, moved the jackfruit decision boundary unfavourably even though that retuning won its own paired gate on the overall metric. A gate that is scored on macro F1 can accept a change that helps several crops and harms one, because macro F1 only sees the mean. Diagnosing this properly requires validation data, not another test read, so it stays open.

For completeness, the run also had one abort event. The first launch crashed while loading a saved calibrated model, which was one of the failure modes listed in the predeclaration, and it crashed before any test row was touched. The relaunch read the test fold exactly once, which is what the predeclaration required.

**A limit on what "read once" means.** It means read once *within the E1 to E7 plan*. The same test partition had already produced M5 and several earlier checkpoint results, so fold 2 is a **previously observed partition**, not a globally fresh test set for the whole project. For the joint paper this matters, and Section 22 requires either locking a new partition or describing this one honestly.

## 17. What the sensor-fusion probe does and does not show

The parcel-disjoint crop-only control used 40 Sentinel-2 features and scored macro F1 0.3945. The fused arm used all 153 optical, radar, and terrain features and scored 0.4401. Because rows, labels, parcel split, sampling, model family, and evaluation were held fixed, the difference of 0.0456 is evidence that the added sensors provide useful information in that probe.

Important gains included rice +0.1084, mangosteen +0.0793, mango +0.0754, rambutan +0.0616, durian +0.0590, and oil palm +0.0547. Coconut gained only 0.0085, longan gained 0.0285, and Langsat did not improve.

The probe does not include non-crop land, it uses five optical dates instead of three, and it balances crop sampling. It is therefore a representation test and not a finished mapping system. The valid comparison is 0.3945 against 0.4401 inside the probe. Comparing 0.4401 directly against the strict full-cascade 0.2429 would mix different populations and tasks.

The result weakens any claim that optical data are the only useful source. It does not overturn the conclusion that parcel support and rare-crop generalisation remain severe constraints.

## 18. Why rare crops remain difficult

### Few independent parcels

A model learns variation across land, management, soil, planting age, and image conditions. Ten parcels cannot represent this diversity simply because they contain many neighbouring pixels. For Langsat, one training parcel contributes about 80% of its training pixels. The fitted boundary can easily learn that parcel's local signature and fail elsewhere.

### Similar crop appearance

Perennial tropical crops can have similar green canopies during the same three months. Vegetation indices compress reflectance into a small number of ratios, and similar orchards can overlap in that space. Radar and terrain add different information, but they cannot manufacture seasonal stages or field examples that are absent.

### Routing loss

A true crop must survive broad land classification and receive the right crop group before the specialist can act. The oracle diagnostic shows large routing headroom. The subtype-mass result shows that some of it can be recovered, especially for crops with enough later-stage support.

### Specialist limitation

Correct routing is necessary but not sufficient. Coconut routing almost doubled without improving coconut F1. This means the plantation specialist remains a bottleneck, which is why the next weighting experiment moves inside the plantation and orchard specialists.

### Metric instability

With only 13 validation pixels for Langsat, one or two predictions can move recall sharply. A validation-selected change may therefore be driven by noise. More independent parcels are the most direct solution because they improve both fitting and evaluation reliability.

## 19. The retuning experiment and its result

The current retune replaces an old pixel-level accuracy search with three-fold parcel-grouped macro-F1 selection. It searches:

- `C = 1, 10, or 30`
- `gamma = 0.25, 0.5, 1, or 2 times the current value of 1/30`
- Stage-2 components = `600 or 1,200`
- orchard-specialist components = `800 or 1,200`

This gives 24 settings for Stage 2 and 24 for the orchard specialist. Each setting is evaluated over three parcel folds, so each component requires 72 fits.

The Stage-2 search uses 50,000 rows from each of its four labels, or 200,000 rows. The orchard search uses its naturally capped crop set, including every available rare-crop row below the 70,000-per-crop cap.

The control for the gate is the already-approved subtype-mass Stage 2 combined with the existing orchard specialist. The treatment changes the chosen Stage-2 hyperparameters and orchard-specialist hyperparameters, while field and plantation specialists remain frozen. Treatment is retained only if it gains at least 0.002 tuning macro F1 and does not reduce the number of crops with F1 at least 0.01.

### Result

Both searches finished and the paired gate **passed**, with a gain of 0.0041 tuning macro F1 over the control and no reduction in the number of crops above 0.01.

The informative part is that the two searches, run separately on different label sets, chose the **same** direction independently: 1,200 components rather than 600 or 800, a gamma of half the previous value, and a penalty term of 30 rather than 10. Every one of those is at or toward the more flexible end of the grid.

That agreement is the clearest evidence I have for something I had only suspected. **Every hyperparameter search in this project before this one was selecting against its own capacity limit.** The settings had been inherited from an older pixel-level accuracy search, and when the search was re-run honestly with parcel-grouped folds and macro F1, both components asked for more capacity in the same way. This is also the most likely mechanism behind the jackfruit regression described in Section 16, because more capacity in the orchard specialist changes every orchard boundary and not only the ones that improved.

## 20. The earlier Random Forest result

A **Random Forest** is an ensemble of decision trees. Each tree is fitted with random variation in rows or features, and their predictions are combined. The earlier Rayong study used one flat Random Forest to predict final land-use classes directly. It reported about 0.716 overall accuracy, 0.678 kappa, and 0.714 weighted F1 on about 303,947 test pixels, with an overall reported F1 near 0.71.

It is tempting to place those values beside the current SVM's 0.2429 macro F1, but that would not be a controlled classifier comparison. The studies differ in year, image period, labels, class population, model structure, metric averaging, test size, and possibly parcel separation. A weighted F1 and a macro F1 also answer different questions.

I therefore use the Random Forest paper as related work and as motivation for the crop classes. I do not use it as proof that Random Forest is better or that the current SVM reproduced its task. A fair comparison requires the same rows, parcel split, labels, dates, features, and scoring denominator.

## 21. How XGBoost works and how the collaborator uses it

### Gradient-boosted trees

**XGBoost** is a gradient-boosted decision-tree method. A decision tree repeatedly asks threshold questions such as “Is NDVI below this value?” and sends a row down branches to a final score. One tree is limited, so boosting builds trees sequentially.

At each boosting step, the model adds a new tree to the existing prediction:

```text
new prediction = previous prediction + learning rate x new tree output
```

The **learning rate** controls how much the new tree can change the existing prediction. The new tree is chosen to reduce the current loss, so later trees focus on errors left by earlier trees. XGBoost also regularises tree complexity and uses efficient gradient calculations.

Unlike the SVM's distance-based RBF representation, trees divide feature space with threshold rules. They can represent interactions such as high B11 in November together with low NDWI in October without requiring manual interaction terms. But they can still overfit repeated pixels from the same parcel, so parcel-disjoint evaluation remains necessary.

### Collaborator's three connected models

I re-read their repository on 3 September 2026. Their inference chain works as follows.

1. A water model estimates water probability. A pixel at or above 0.56 is removed from crop candidates.
2. A building model estimates building probability on the remaining pixels. A pixel at or above 0.56 is removed.
3. A flat crop model predicts one of 13 crops or an `others` class for the survivors.

Two implementation details are worth recording because they affect how the comparison must be run. First, the rejection now happens at **extraction** time rather than inside a single inference script: each downstream extractor reads the list of pixel coordinates the upstream model rejected and skips them. The effect is the same, but it means each model has its own dataset file, which was the fix their 2 September report describes. Second, both thresholds are 0.56, and the building threshold appears to have been copied from the water model rather than selected separately for buildings.

Their 2 September 2026 report gives their first complete end-to-end pipeline result: water F1 0.865, building F1 0.836, and a crop model at accuracy 0.57 and macro F1 0.38 over 14 labels on 755,295 pixels. Over the 13 crops alone their macro F1 is 0.3692, because `longkong` enters their 14-label average with support 0.

**That crop result comes from the old pipeline, and this qualification governs everything else in this section.** The report is largely an analysis of why that pipeline was wrong. The pipeline had built one shared dataset for all three models and then deleted any row containing a missing or infinite value. Because the water and building features are 3-by-3 neighbourhood means and variances, one missing neighbour destroys those columns, and the row was deleted even when every crop feature in it was valid. Their fix, one dataset per model, is the right fix and was committed on 1 and 2 September 2026, before the report was circulated.

**The repaired pipeline has not yet produced a reported score.** So the performance of their current system is unknown, and 0.38 must not be quoted as "the XGBoost result" in any comparison. It is the score of a configuration its own author has replaced.

This also resolves what I had listed as an open question about their population. The old extraction script, now deleted from their repository, drew a reservoir sample capped at **200,000 pixels per class** with a seeded generator, assembled the single shared table of water and crop columns, and only then applied the row deletion. That is why their rubber support is 197,590, just under the cap, rather than the millions of rubber pixels present in the tile. Their repaired extractors apply no cap, so their new population should be substantially larger, and its size is now the first thing the joint protocol needs from them.

### Three differences their report does not separate

Their diagnosis is correct, but it is **under-determined** as an explanation of the whole drop, because at least three other things also differ between their standalone result and their pipeline result. Each of these is a joint methodology item, not a criticism.

1. **Erosion asymmetry.** Their training extractor erodes parcel boundaries; their pipeline evaluation extractor does not. So the crop model is fitted on parcel interiors and scored on all pixels, including mixed boundary pixels, which are the hardest ones.
2. **Training and deployment populations differ.** Their training script still reads the older capped, eroded, unfiltered dataset, while inference scores the newer uneroded filtered population. The deployed model was therefore never fitted on the distribution it is scored on.
3. **The two reported results use different partitions.** Their standalone crop numbers come from the 20 percent cross-validation partition of the capped training sample, totalling about 304,081 rows, while their pipeline numbers come from the full capped inference sample of 755,295 rows. The pipeline population is therefore roughly 2.5 times **larger** overall, even though nine classes fall and five rise. A support chart comparing a one-fifth slice against a whole population cannot on its own establish that support collapsed.
4. **The survivor-only denominator**, described immediately below.

Because of these, the row-deletion mechanism should be described as **one identified contributor** rather than the established sole cause. Lower support also does not mechanically lower F1. Establishing causation needs an old-versus-repaired comparison on the same frozen pixel identities, which their repaired code now makes cheap to run.

### The population question, and how it was resolved

I originally recorded this as an unresolved factual matter, on the grounds that their pipeline extractors apply no sampling cap, so every labelled pixel surviving the two filters should appear in the dataset, and yet their rubber support of 197,590 pixels is roughly 50 times smaller than rubber in my tile. That reasoning read the wrong version of their code.

The numbers in the report were not produced by those uncapped extractors. They were produced by the **old** extraction script, since deleted from their repository, which drew a reservoir sample capped at **200,000 pixels per class** using a seeded generator, assembled the single shared table of water and crop columns, and only then deleted rows containing missing values. Their rubber support of 197,590 is that cap, less the rows the deletion removed. There is no mystery.

What remains open is the narrower and directly answerable question of **what population their repaired pipeline evaluates**. Because the repaired extractors apply no cap, it should be considerably larger, and it may be the full labelled tile. That number is the first thing the joint protocol needs from them, because the evaluation population is the single largest source of difference between our two sets of results.

### Two things I checked and found correct

I am recording these so they are not raised again as concerns. Their Sentinel-2 band indexing is correct for their composite layout, and their MTCI formula is the standard red-edge form. Their features also contain **no pixel coordinates**, so there is no positional leakage in their crop model.

The value 0.56 is a **decision threshold**. A probability above the threshold triggers the filter. Raising a threshold normally removes fewer pixels and can increase water or building false negatives. Lowering it removes more pixels and can accidentally discard crops. The final effect must be measured across the whole chain.

The SVM and XGBoost systems therefore use different hierarchy ideas. The SVM performs semantic agricultural routing into field, plantation, and orchard specialists. The XGBoost system performs exclusion filtering for water and buildings and then uses one flat crop classifier.

### Survivor-only denominator

If the crop report is calculated only on pixels that survive the first two filters, it does not charge the crop model for a true crop discarded as water or building. This is a **conditional** score. It can be useful for diagnosing the last model, but it is not the end-to-end map score.

A fair full-population result must begin with the same complete evaluation rows, run all filters, assign a final output to every row, and count crop pixels removed early as false negatives. Non-crop pixels incorrectly retained and labelled as crops must become false positives.

### Why the XGBoost result remains open

I can explain the implemented structure, but the numbers I have describe a pipeline they have already replaced, and the repaired one has not been scored yet. Even setting that aside, I do not have a verified parcel-disjoint, full-population result under the SVM's label definitions and denominator, and I do not have my SVM result under their exact protocol either. Their preparation also splits sampled pixels by row rather than by parcel. Because my own rare-crop score was strongly inflated by that same choice, the shared protocol should be repaired before drawing a model comparison.

This is a methodological alignment problem, not evidence that XGBoost is weak. Its result could be better, worse, or complementary after the same evaluation is applied.

## 22. The joint-paper logic

The strongest paper is not simply “SVM versus XGBoost.” If one model uses an easier split or a narrower denominator, the table would compare measurement protocols as much as algorithms.

The proposed paper question is:

> How do semantic crop routing and water-building filtering behave under extreme class imbalance when both are evaluated on unseen parcels and the complete map population?

### The design rule we are working under, and why it needs care

The condition set for a joint paper is that one axis must be held fixed. Either the two studies use **different algorithms with the same architecture and the same control variables**, or they use the **same algorithm with different methods**. We chose the first.

The difficulty is that both systems are hierarchical but their hierarchies are not the same object. Mine is a **semantic routing cascade**: every pixel is assigned to a branch and a specialist inside that branch makes the final decision. Theirs is a **sequential rejection cascade**: two binary models delete pixels, and one flat head classifies whatever is left. Taken literally, the rule would require one of us to abandon our architecture.

It is important to be precise about what their system is, because an earlier draft of this section was not. Their end-to-end system is **not a flat classifier**. It is a cascade whose final stage is flat. Setting it opposite a genuinely flat SVM would confound the architecture factor with the presence of the two upstream rejection filters, and the resulting table could not be read as an architecture effect at all.

Neither one-sided option is good. If I flatten my system to match theirs, I remove its only structural contribution, and I also ignore a computational asymmetry: a flat RBF SVM over 24.3 million rows is not affordable in the way a flat gradient-boosted tree is, and that constraint is part of why the cascade exists. If they adopt my cascade, the comparison is clean and each side contributes something distinct, but it assumes my routing design is the better architecture, which is exactly the question under test.

### The proposed resolution: a factorial design

Rather than forcing the architectures to match, architecture becomes a **second controlled factor**, and the study fills a two-by-two table. The two levels of the architecture factor are the two cascades, and each is implemented with each algorithm. Every cell emits the same final label set, the 13 target crops plus `others`.

| | RBF-SVM | XGBoost |
|---|---|---|
| Semantic routing cascade | to be run | to be run |
| Sequential rejection cascade | to be run | to be run |

In a **factorial design**, two factors are varied together so that each one's effect can be estimated over the levels of the other. The design yields three quantities that a single head-to-head number cannot give:

- the **main effect of the algorithm**, which is the average difference between the SVM column and the XGBoost column;
- the **main effect of the architecture**, which is the average difference between the flat row and the cascade row;
- the **interaction**, which is whether hierarchy helps a kernel method and a tree ensemble by the same amount. A non-zero interaction is the scientifically interesting outcome, because it would mean the right architecture depends on the algorithm.

**All four cells are pending.** Neither existing result qualifies as a cell. My final cascade uses my own 30-feature Sentinel-2 set rather than the agreed common features and predates these controls; the collaborator's 2 September result uses neither the shared split, nor the shared population, nor the shared denominator, nor a held-out partition, and it comes from the pipeline before their row-deletion repair. Both belong in the paper as motivation and as a record of where each study stood before alignment. Neither may contribute to an estimated effect.

**Three cells are not enough.** With one cell missing, the algorithm effect, the architecture effect, and the interaction cannot all be identified, because each estimate needs both levels of the other factor. The rule is therefore strict: the words *main effect* and *interaction* require four aligned cells. If only three are achievable, the study is renamed to **controlled pairwise comparisons** and the claims shrink to the pairs actually run.

If capacity forces a per-class training cap on the SVM cells, the same capped rows must be given to the XGBoost cells, for the reason given in the controls below.

### Required shared protocol

Both systems should use, in descending order of how much damage each item does if left unaligned:

1. **The same evaluation population**, the natural uncapped tile. Measured on my own unchanged predictions, population alone moves macro F1 by **0.131**, from 0.2654 on the natural tile to 0.3965 on a sample capped at 200,000 per class, and accuracy and weighted F1 move in the opposite direction. This is larger than any modelling difference either study has produced, which is why it is first.
2. **The same frozen evaluation pixel identities.** One shared raster grid, one linear pixel identifier, and one evaluation-row mask computed over the **common** features only, saved and hashed as an artifact. This is the control the collaborator's own 2 September report shows is missing: if each model deletes its own rows for its own reasons, the cells score different populations and nothing is comparable. Every cell must be provable to have predicted the same ordered list of test pixel IDs. The **missing-value policy** is part of this control and is declared once for all cells, with eligibility decided over the common features and never over features that only one arm uses.
3. **The same sampled training rows.** Corresponding cells receive the same ordered training pixel IDs, seeded, including any per-class cap. Otherwise the algorithm comparison is confounded with training volume and sample composition. An uncapped XGBoost arm may be reported as a clearly labelled **data-volume ablation**, never as the algorithm cell. Evaluation stays natural and uncapped regardless.
4. **The same parcel IDs and train, validation, and test assignment.** The SVM side already runs parcel-disjoint and the final cascade demonstrates the machinery; the XGBoost side currently splits at pixel level and needs migrating. Earlier SVM experiments had the same defect, so this is a repair the SVM side has already made rather than a standard invented for the other side.
5. **The same 30-metre label erosion**, applied at both training and evaluation time on both sides. The artifact of record is `label/label_47PQQ_buffered.tif`, which is the eroded raster that the feature-alignment step actually consumes; `label/label_47PQQ.tif` is its raw, uneroded source and is not the comparison artifact.
6. **The same 13 crop definitions and background rule**, with any class at support 0 excluded from the macro average and the exclusion stated.
7. **The same full-population denominator and the same metric equations**, with a fixed explicit label list and no masking applied before the metric.
8. **The same reported partition**, fixed in advance, with its prior exposure stated. Neither side currently has a globally untouched test set: the collaborator's numbers come from a cross-validation partition, and my fold 2 was read once within the recent plan but had been used by earlier checkpoints. The joint paper should either lock a new partition before any joint run or describe the existing one as previously observed.
9. **The same date window and feature matrix** if the paper claims a classifier comparison. The natural common denominator is their 15 columns, which leaves my radar and terrain stack as a clearly separate ablation.

If the feature sets or dates remain different, the paper can still compare two complete mapping systems, but it should not attribute the whole difference to SVM versus XGBoost.

### Two fairness conditions on the factorial

- **Each cell gets its own hyperparameter search at a comparable budget.** My cascade's capacity was retuned specifically for an RBF kernel, as described in Section 19. Dropping an untuned XGBoost into those stages would produce a result that measures tuning effort rather than algorithm, and the same applies in reverse to the flat SVM cell.
- **Preprocessing belongs to the algorithm arm, not to the shared controls.** The SVM requires a Nystrom or PCA map before the kernel; a tree ensemble neither needs nor benefits from one. Forcing identical preprocessing would handicap one arm for no scientific gain, so it should be declared as part of each arm and stated in the methods.

The full signable version of this protocol, including the open questions for the collaborator, is written separately as `docs/JOINT_PROTOCOL_2026-09-03.md`.

### Stage-by-stage error accounting

The paper can measure where each true crop is lost. For SVM, the losses are Stage-1 rejection, Stage-2 wrong-group routing, and Stage-3 crop confusion. For XGBoost, the losses are water rejection, building rejection, and final crop confusion. This makes the comparison interpretable.

### Rare-class support analysis

The paper should report both pixels and independent parcels per crop. It can then show whether a failure is associated with routing, model representation, or insufficient coverage. This is more useful than presenting one F1 value without its data support.

### Disagreement map

A **disagreement map** marks locations where the two systems predict different final labels. High-confidence agreement can identify stronger candidate predictions. Disagreement can identify uncertain parcels for field checking, label review, or future survey collection.

This is useful even if one model has a higher average score because the two model families can make different errors.

### Ensemble

An **ensemble** combines two models into one prediction. A simple vote is not automatically honest because their probabilities may use different labels, scales, calibration data, and candidate populations. A formal combination would require both outputs on the same final labels, shared parcel-disjoint calibration rows, and a combining rule fitted without reading the test fold.

Until that is done, agreement and disagreement analysis is safer than claiming an ensemble gain. If the errors prove complementary, an ensemble becomes a justified follow-up rather than a predetermined story.

### How the open XGBoost result changes the paper angle

- If both systems fail on the same rare crops, the main result concerns parcel support, evaluation leakage, and the limits of the available data.
- If XGBoost recovers crops the SVM misses, the analysis can test whether tree thresholds and flat crop decisions avoid harmful SVM routes.
- If SVM recovers crops XGBoost misses, the agricultural grouping may provide useful structure.
- If their errors are complementary, disagreement-guided validation and an ensemble become the strongest direction.

The paper story should follow the aligned experiment, not be chosen before the result.

## 23. Why each action was chosen, and how it ended

### Rebalance inside Stage-3 specialists

The Stage-2 treatment improved routing but not coconut F1. This points to a bottleneck inside the plantation specialist. Applying subtype weighting separately in plantation and orchard models tests that explanation directly. It is more targeted than globally changing every stage.

**Outcome: this failed its gate and was not kept.** The paired comparison came out at -0.0012, below the required +0.002, and jackfruit fell under the 0.01 alive-crop floor, which the gate also forbids. The mechanism it was testing did show up: coconut moved from 0.0000 to 0.0033 inside that probe, which is consistent with the plantation specialist being a real bottleneck. But the crops that lost outweighed the crop that gained, so the honest reading is that the diagnosis was partly right and the intervention was still not worth keeping. This is the clearest example in the project of a correct hypothesis producing a failed treatment.

### Test date differences only if time remains

A **temporal difference feature** subtracts the same index between two dates, such as December NDVI minus October NDVI. It makes seasonal change explicit. But the current nonlinear model can already use interactions among date-specific columns, and a difference adds no new observation. It may change feature emphasis but is a lower-priority intervention than honest retuning and specialist imbalance.

**Outcome: skipped, as the plan allowed.** It was marked optional in advance precisely so that it could be dropped without changing the plan's conclusions, and the earlier importance measurements gave no reason to promote it.

### Train one final combined cascade

Only interventions that pass their validation gates will be combined. The final test fold has already been examined for several historical checkpoints, so repeatedly testing alternatives would gradually turn it into another validation set. I therefore reserve another strict test read for one written final configuration.

**Outcome: completed, at macro F1 0.2429**, detailed in Section 16. Two of the four candidate interventions passed their gates and were included; two failed and were excluded. The planning range of 0.24 to 0.26 was written before the run, and the result landed at the bottom of it, with the 0.28 upside case not reached. The test fold is now spent, so any further change to this model has to be evaluated on different ground.

### Obtain more parcels

More independent parcels from another survey year or neighbouring tile address the central rare-class limitation directly. They expand variation in geography, management, age, and season, and they make validation scores less unstable. No weighting equation can create independent farms that were never observed.

**Status: this is now the main outstanding request rather than one option among several.** Every optical-only intervention in the plan has been tested, and none moved the rare crops. That is a completed test result rather than a projection, which is what changes this from a preference into the leading recommendation.

### Majority-class capping, the one untested idea

There is one modelling idea in the collaborator's study that mine has not tested. They cap every class at 200,000 training rows. Capping does not raise the rare classes; it **lowers the majority ones**, taking rubber from roughly 3.2 million rows to 200,000 while Langsat keeps its 1,800. The balance the model is fitted against therefore improves by roughly 60 times, even though nothing about the rare crops changed.

This is a different mechanism from the cost-sensitive class weights that failed in Section 14. A class weight rescales the loss contribution of rows that are already in the training set, so the geometry of the fitted boundary changes only through the penalty term. A cap changes **which rows exist**, so it changes which pixels can become support vectors and how dense the kernel neighbourhood around each rare class is. A failure of the first does not predict a failure of the second, which is why it is worth a test rather than an assumption.

The honest form of the experiment is to train on the capped sample and still score strictly on the natural uncapped tile. Training-time capping is a legitimate modelling choice; evaluation-time capping is the confound described in Section 22, and the two must not be conflated.

## 24. What I can conclude now

1. The final strict full-cascade result is macro F1 **0.2429** and weighted F1 0.7949 on 5,500,269 unseen-parcel test rows, an improvement of 0.0085 over the previous 0.2344.
2. That gain is concentrated in two mid-frequency crops. Oil palm improved by 0.1429 and rice by 0.0276, and together they account for essentially all of it.
3. The rare crops did not improve. Coconut moved by 0.0010, longan and Langsat did not move, and this was predicted in writing before the test read. This is now a completed result rather than a projection.
4. The whole final round of work bought 0.0085, while every change from the first parcel-disjoint baseline up to M5 bought 0.0096 in total. Both numbers are small and should be reported as such.
5. The protocol repairs make the experiment more trustworthy, but their 0.0035 score gain was not clearly different from zero under parcel bootstrap uncertainty.
6. Stage-2 routing is a large bottleneck. Perfect diagnostic routing increased validation macro F1 by 0.1491, although that gain is not deployable.
7. Stage-2 subtype-mass weighting produced a repeatable validation gain. It won on the original draw and all three fresh draws, with a fresh-draw mean of +0.0078, and it survived into the final cascade.
8. Every hyperparameter search before the honest retune was selecting against its own capacity limit. Two independent searches on different label sets both chose more components, a smaller gamma, and a larger penalty term.
9. Better coconut routing did not improve final coconut F1, and rebalancing inside the specialists failed its own gate, so the plantation specialist and parcel support both remain limiting.
10. Pixel splitting seriously overstated rare-crop generalisation. The crop-only probe fell from 0.5852 to 0.3945 when complete parcels were separated.
11. Radar and terrain add useful information in a controlled unseen-parcel probe, increasing macro F1 from 0.3945 to 0.4401. They did not rescue every rare crop.
12. MTCI contributed little measured importance in the current three-date system, while raw B11 was more useful. This does not prove that all red-edge information is useless.
13. The two limits I identified behave differently, and the final result separates them. Routing and capacity work pays in crops that have enough independent parcels, and it does not substitute for parcel data in crops that do not.
14. The earlier Random Forest result and the collaborator's XGBoost workflow are not yet controlled comparisons with the SVM.
15. The strongest joint-paper contribution is a shared parcel-disjoint, full-population factorial comparison with stage-by-stage error and parcel-support analysis.

## 25. What I cannot conclude now

1. I cannot claim that the current system solves all 13 crops operationally.
2. I cannot claim that MTCI, larger Stage-3 capacity, or the operating-point rule individually caused a known fraction of the 0.0096 test gain because they changed together. The same applies to the final 0.0085, which combined subtype weighting and retuning in one configuration by design, because the plan spent its single test read on the combination rather than on separating the parts.
3. I cannot claim that three-date Sentinel-2 creates a universal performance ceiling for every possible model.
4. I cannot claim that radar and terrain have improved the complete production cascade because they have only passed a controlled crop-only probe.
5. I cannot claim that perfect learned routing will recover the oracle's full 0.1491 gain.
6. I cannot claim that the collaborator's XGBoost is better or worse until both systems use the same parcels, rows, labels, features where applicable, and scoring denominator.
7. I cannot make a stable positive or negative performance statement for Langsat from its current parcel support.
8. **I cannot explain why jackfruit regressed by 0.0612 in the final cascade.** No validation-half measurement predicted it. The retuned orchard specialist is the most plausible cause, but I have not demonstrated that, and I will not investigate it on the test fold.
9. I cannot claim the final 0.2429 is separated from parcel-level uncertainty. The parcel bootstrap interval on macro F1 in this project is about plus or minus 0.014, which is wider than the 0.0085 gain, so the gain is consistent with the interval rather than clearly outside it. What the gate did establish is that the changes won on validation data before the test fold was touched.
10. I cannot claim that majority-class capping will help, only that it is untested and mechanistically different from the class weighting that failed.

## 26. Compact glossary

| Term | Meaning in this study |
|---|---|
| Alpha | Strength of the uniform-target probability adjustment |
| Argmax | Choosing the class with the largest adjusted probability |
| Calibration | Mapping model scores to probability-like values using held-out data |
| Candidate | Pixel allowed to enter a later cascade stage |
| Cap | Maximum number of sampled rows from a class or group. Applied at training time it is a modelling choice; applied at evaluation time it changes the measured score and becomes a confound |
| Cascade | Models connected so one stage controls what reaches the next |
| Class imbalance | Large difference in sample size among classes |
| Composite | One representative image made from observations in a time window |
| Confounding | Several important changes occur together, so their effects cannot be separated |
| Cross-fitting | Producing training-time predictions with models that did not fit those rows |
| CRS | Rule that maps Earth locations into numerical coordinates |
| dB | Logarithmic unit used for radar power |
| DEM | Raster of elevation |
| Discrimination | Ability to rank or separate correct and incorrect classes |
| Ensemble | A rule that combines predictions from multiple models |
| Epoch or date window | Time period represented by the satellite inputs |
| Factorial design | Varying two factors together, here algorithm and architecture, so each one's effect can be estimated across the levels of the other |
| Falsifier | Prewritten result that would weaken an explanation |
| Feature | Numerical input measurement used by a model |
| Feature vector | All feature values for one pixel |
| Fold | One partition used for training, validation, test, or cross-validation |
| Full-population score | Evaluation that includes all eligible map rows and all early-stage errors |
| Gate | Written acceptance rule for carrying a change forward |
| Generalisation | Performance on genuinely unseen parcels |
| GeoTIFF | Raster image containing geographic reference information |
| GroupKFold | Cross-validation that keeps each parcel inside one fold |
| Hyperparameter | Model setting selected outside coefficient fitting |
| Imputation | Replacement of missing feature values |
| Interaction | In a factorial design, the case where one factor's effect depends on the level of the other, for example hierarchy helping the SVM but not XGBoost |
| Kappa | Chance-adjusted agreement metric |
| Kernel | Similarity function that supports nonlinear decisions |
| Label erosion | Removing boundary pixels from parcel labels |
| Leakage | Evaluation information improperly entering training |
| LU code | Numeric land-use category from the LDD survey |
| Macro F1 | Equal-weight mean of the 13 crop F1 scores. Because it averages, it can accept a change that helps several crops and harms one |
| Main effect | The average effect of one factor across the levels of the other factor in a factorial design |
| Majority-class capping | Reducing the number of training rows taken from abundant classes, which improves training balance without adding any rare-class data |
| Margin | Signed SVM decision score relative to a boundary |
| Mixed pixel | Pixel containing more than one real land surface |
| Nodata | Missing or invalid measurement, not a land class |
| Nyström components | Size of the approximate nonlinear RBF feature map |
| Operating point | Post-calibration rule that changes the class decision tradeoff |
| Oracle | Diagnostic using true information unavailable at deployment |
| OvR | One binary model for each class against all remaining classes |
| Parcel-disjoint | No parcel appears in more than one data partition |
| Prevalence or prior | Fraction of applicable rows belonging to a class |
| Probe | Controlled, limited experiment answering one diagnostic question |
| Raster | Geographic grid of pixel values |
| Regularisation | Constraint that discourages an overly complex fitted boundary |
| Rejection cascade | Cascade in which upstream models delete pixels and one flat model classifies the survivors, as in the collaborator's system |
| Resampling | Estimating values on a new raster grid, or drawing rows for an experiment, depending on context |
| Routing | Sending a pixel from one cascade stage to a later component |
| Routing cascade | Cascade in which every pixel is assigned to a branch and a specialist inside that branch decides, as in the three-stage SVM |
| Sample weight | Multiplier on one training row's loss |
| Scaling | Converting features to comparable numerical ranges |
| Sentinel-1 | Radar satellite data source |
| Sentinel-2 | Optical multispectral satellite data source |
| Sink | Stage-2 destination for non-target economic codes |
| Support | Number of true evaluation examples, preferably also reported as parcels |
| SVM | Margin-based classifier used in the three-stage system |
| Threshold | Cutoff above which a probability triggers a decision |
| Tuning | Choosing settings using non-test data |
| Upsampling | Repeating or resampling minority rows to increase their presence |
| Weighted F1 | F1 average weighted by true crop pixel support |
| XGBoost | Sequential ensemble of regularised decision trees |

## 27. References for the probability methods

Platt, J. C. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. This is the basis for the sigmoid mapping from SVM margins to probability-like outputs.

Wu, T. F., Lin, C. J., and Weng, R. C. (2004). *Probability Estimates for Multi-class Classification by Pairwise Coupling*. This explains a jointly consistent multiclass probability construction from pairwise class estimates.

The main methodological point is that probability conversion, class decision adjustment, and end-to-end accuracy are separate questions. Calibration changes the interpretation of scores, the operating point changes which class wins, and neither step can replace representative parcel data.
