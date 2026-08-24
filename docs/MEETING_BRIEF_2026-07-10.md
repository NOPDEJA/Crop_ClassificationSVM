# Progress Meeting Brief — 2026-07-10

Prep notes for tomorrow's progress meeting. Suggested flow ≈ 15 min + questions.
Files to open are marked 📂. Deeper backup material for Q&A is in
`DECISIONS_AND_ANALYSIS.md`.

---

## 0. The one-sentence opener

> "Since last meeting, end-to-end economic-crop recall went from **23.1 % to 61.7 %**
> — 2.7× — and we proved that Sentinel-2 indices decisively help, after showing that
> the earlier result saying they *hurt* was a hyperparameter-scale artifact."

---

## 1. Presentation flow (what to show, in order)

### Slide/step 1 — Show the map first
📂 `runs/dem_s1_s2_v2/end_to_end_lu_pred_quicklook.png` *(generated for this meeting)*

Talking points:
- This is the full end-to-end product: 13 crop LU codes over all labeled parcels of
  tile 47PQQ, from the best model (`dem_s1_s2_v2`).
- The spatial pattern is agronomically plausible: pineapple belt in the west, cassava
  southwest, rice along the central lowland, rubber dominating the center, and the
  durian/rambutan/mangosteen orchard belt in the east — the model reproduces the known
  geography of Rayong, not just pixel statistics.
- (If asked for the real raster: 📂 `runs/dem_s1_s2_v2/end_to_end_lu_pred.tif` in QGIS,
  compare against 📂 `label/label_47PQQ_buffered.tif`.)

### Slide/step 2 — Where we were vs where we are
📂 `README.md` (headline table) — or quote directly:

| Milestone | End-to-end econ recall |
|---|---|
| DEM+S1 baseline (June, first working cascade) | 23.1 % |
| DEM+S1+S2, first attempt (broken gamma) | 19.0 % ← the mystery |
| DEM+S1 retrained, corrected gamma | 47.3 % |
| **DEM+S1+S2, corrected gamma (current best)** | **61.7 %** |

### Slide/step 3 — The research story of this cycle (the gamma finding)
📂 `runs/dem_s1_s2/REPORT.md` (the "S2 made it worse" report) → then 📂 `runs/dem_s1_s2_v2/REPORT.md`

Tell it as a 4-beat story — this is the intellectually strongest part of the progress:
1. **Anomaly:** adding 40 correctly-computed S2 indices made *every* stage worse
   (Stage 2 accuracy 0.509 → 0.466; Rice F1 0.791 → 0.379).
2. **Rule out the data:** band order verified empirically (band-10 correlates 0.90 with
   B08 at 1.08× → it *is* B8A); indices finite, clipped, cloud gaps known.
3. **Diagnosis:** RBF kernel dilution — 94 redundant SAR/DEM columns dominate the
   distance metric — compounded by a gamma grid {0.5, 1.0} ~100× above the correct
   `1/n_features ≈ 0.007` scale. At that gamma every pixel looks maximally dissimilar
   to every other; adding features makes it worse, which is exactly what we saw.
4. **Confirmation, twice:** a 1-hour Stage-2 rerun with corrected gamma jumped accuracy
   **0.466 → 0.868**; then both full chains (with and without S2) were retrained under
   the *identical* corrected config — the only fair comparison — and S2 won every
   metric at every stage.

Key line for the professor: *"We didn't just fix a bug — we established that
feature-set conclusions are only valid under hyperparameters scaled to that feature
set, and we rebuilt both comparison arms before concluding."*

### Slide/step 4 — Current model quality, stage by stage
📂 `runs/dem_s1_s2_v2/REPORT.md` (scroll to the v2-vs-v2 tables)

| Stage | Metric | no S2 | +S2 |
|---|---|---|---|
| 1 — superclass | accuracy | 0.733 | **0.766** (forest F1 0.880, water F1 0.875) |
| 2 — subclass routing | accuracy | 0.773 | **0.858** |
| 3 — field | Rice F1 | 0.974 | **0.981** |
| 3 — plantation | Rubber / Oil palm F1 | 0.846 / 0.857 | **0.906 / 0.897** |
| 3 — orchards | macro F1 | 0.399 | **0.519** (Mango 0.902, Durian 0.712) |
| **End-to-end** | **econ recall** | **47.3 %** | **61.7 %** |

### Slide/step 5 — Honest accounting: where the remaining 38 % goes
📂 `runs/dem_s1_s2_v2/end_to_end_summary.csv` (exact numbers)

Of 15.16 M true economic-crop pixels:
- **61.7 %** get the correct final LU code
- **18.9 %** dropped at Stage 1 (not predicted econ at all)
- **13.0 %** misrouted at Stage 2 (wrong subclass → wrong Stage-3 model)
- **~6.4 %** wrong code within the right subclass (Stage 3)

Point to make: S2's gain came mostly from *routing* (Stage-2 misrouting fell from
3.09 M to 1.97 M pixels), and the loss table tells us the next gains live in Stage 1
recall and Stage 2 routing — not in polishing Stage 3.

### Slide/step 6 — Problems faced & solved this cycle (one slide, fast)
- **Gamma / kernel dilution** — the headline story above.
- **S2 band-order bug** (Phase-1 legacy): `sorted()` puts B11/B12 *before* B8A, so all
  Phase-1 SWIR indices used swapped bands. Fixed and verified empirically.
- **Longan (2413) missing** from Stage-2 orchard codes → trained as "other_econ". Fixed.
- **Duplicate feature** (VVVH_DIFF ≡ VV_VH_RATIO) removed — pure redundancy feeding
  kernel dilution.
- **Ops hardening:** a machine reboot destroyed 6.5 h of buffered output → all long runs
  now launch detached + unbuffered with health monitoring; ~30 h of training across
  6 stages then completed unattended, surviving one reboot.

---

## 2. Analysis points to be ready to defend

1. **"Orchards accuracy went DOWN (0.829 → 0.635) — is that a regression?"**
   No — the old model predicted Durian for ~everything (82 % of test set was Durian;
   minority recall 2–8 %). Accuracy rewarded that collapse. The new model actually
   learns all 7 species (Mango F1 0.037 → 0.902), nearly doubling macro-F1
   (0.288 → 0.519). Macro-F1 is the fair metric for imbalanced orchards — and in the
   final v2-vs-v2 comparison S2 wins on *both* accuracy and macro-F1 anyway.

2. **"Is 61.7 % good?"**
   Context matters: it is exact 13-class LU-code match, pixel level, after three
   chained models — every stage's error compounds. Per-stage accuracies are 0.77–0.86.
   It is 2.7× the baseline from a month ago, and the loss table shows exactly where the
   next points come from. Do *not* oversell it as final accuracy — see caveat 3.

3. **Known limitations (raise them yourself — it builds credibility):**
   - Pixel-level train/test splits share parcels between train and test → absolute
     numbers are optimistic; parcel-level (group) splits are the next validation step
     before anything is delivered to LDD. Relative comparisons (S2 vs no-S2) remain valid.
   - Stages 2/3 train on pixels the previous stage passed → conditional metrics are
     survivor-biased; the end-to-end number is the honest one (which is why we lead with it).
   - Labels are the 2018 LDD survey (`LU_RYG_2561`); imagery is newer — some label noise
     from land-use change is baked in.

4. **"Why SVM and not deep learning / random forest?"**
   SVM cascade was the agreed project scope; a tree-ensemble (RF/GBM) reference run is
   already on the follow-up list precisely because trees are robust to the redundant-
   feature problem that bit the RBF kernel. It would strengthen the paper as a baseline,
   not replace the pipeline.

---

## 3. Suggestions to propose (and decisions to ask for)

Ordered by priority — ask the professor to confirm/adjust:

1. **Adopt `dem_s1_s2_v2` as the production baseline** (update project docs to point at
   it; retire the pre-fix baselines as historical references).
2. **Parcel-level split validation next** — expect absolute numbers to drop; better we
   report that ourselves before LDD delivery. *(Decision: agree this is required before
   handing results to LDD?)*
3. **Attack the loss table:** Stage-1 econ recall (lower the 0.4 threshold floor, widen
   `n_components` past its 250 ceiling) and Stage-2 routing. *(Decision: is +5–10 pp
   end-to-end recall the right target for next cycle?)*
4. **Durian threshold retuning** — its F1 dipped (0.905 → 0.712) as the model stopped
   over-predicting it; per-class threshold tuning on the orchards model should recover
   most of it without hurting the minority species.
5. **S1 temporal aggregation** (18 dates → monthly/orbit medians, 108 → ~24 columns) —
   removes the redundancy that caused kernel dilution *and* the all-NaN column problem.
6. **RF/GBM reference run** — cheap, strengthens methodology for publication.
7. **Ask LDD:** is a newer parcel survey than 2561 (2018) available for validation? And
   what recall/precision trade-off do they prefer for the delivered map (threshold knob)?

---

## 4. File cheat-sheet (in order of likely use)

| File | Use in meeting |
|---|---|
| 📂 `runs/dem_s1_s2_v2/end_to_end_lu_pred_quicklook.png` | **Opener** — the map, slide-ready |
| 📂 `README.md` | One-page project overview + headline table |
| 📂 `runs/dem_s1_s2_v2/REPORT.md` | **Main results doc** — all v2-vs-v2 comparison tables |
| 📂 `runs/dem_s1_s2_v2/end_to_end_summary.csv` | Exact end-to-end recall + loss accounting |
| 📂 `runs/dem_s1_s2/REPORT.md` | The "S2 hurts" anomaly + kernel-dilution diagnosis (story beat 1–3) |
| 📂 `runs/exp_gamma_scale/stage2_dem_s1_s2_report.csv` | The 1-hour confirmation (acc 0.868) |
| 📂 `DECISIONS_AND_ANALYSIS.md` | Backup for any technical question — glossary, code, deep dives |
| 📂 `runs/dem_s1_s2_v2/end_to_end_lu_pred.tif` (+ `label/label_47PQQ_buffered.tif`) | Only if they want to pan around in QGIS |

Numbers cheat-sheet: end-to-end 23.1 → 19.0 → 47.3 → **61.7 %** · gamma fix alone
0.466 → 0.868 (Stage 2) · Stage 1 acc 0.766 · Stage 2 acc 0.858 · forest F1 0.880 ·
Rice F1 0.981 · Rubber F1 0.906 · orchards macro-F1 0.519 · 24.3 M pixels · 134 features ·
~6.5 h per stage, ~30 h total for both v2 chains.
