# Stage 3 — DEM + Sentinel-1 vs S2-Only Baseline (LU_CODE Classification)

**Run date:** 2026-07-01  
**Input:** Stage 2 subclass predictions → per-subclass fine-grained LU_CODE models  
**Subclasses trained:** orchards, plantation, field  (`other_econ` skipped — no eligible pixels after Stage 2 routing)

> **Important caveat:** Stage 3 trains only on pixels that Stage 2 correctly routed to the right subclass.  
> Since Stage 2 DEM+S1 recall was low (orchards=29.6%, plantation=33.4%, field=93.7%), the Stage 3  
> orchards/plantation models see only a biased ~30% of available labeled data — the pixels most visually  
> distinct in SAR/terrain space. Results for orchards and plantation should be interpreted with this in mind.

> **Additional caveats (found 2026-07-02):**
> 1. Stage 2 was trained with Longan (2413) mislabeled as *other_econ* (missing from its `orchards_codes`), so the Longan pixels reaching this orchards model were only the ones Stage 2 misrouted by its own training — a doubly biased sample. Longan's F1 (0.167) is not trustworthy. Fixed in `stage2_weighted.py` for future runs.
> 2. The S2-only baseline's SWIR-based features (BSI, NDBI, SWIR_NIR, SWIR_RATIO) were computed with swapped bands (see `runs/s1_dem/REPORT.md`), so the baseline F1 columns understate a correct S2 model.

---

## Results at a Glance

### Orchards sub-model (LU_CODEs: 2403, 2404, 2407, 2413, 2416, 2419)

| LU_CODE | Crop | S2-Only F1 | DEM+S1 F1 | Δ F1 |
|---------|------|-----------|-----------|------|
| 2403 | Durian | 0.744 | **0.848** | **+0.104** |
| 2404 | Rambutan | 0.203 | 0.144 | −0.059 |
| 2407 | Mango | 0.722 | 0.086 | −0.636 |
| 2413 | Longan | — | 0.167 | (new) |
| 2416 | Jackfruit | 0.298 | 0.049 | −0.249 |
| 2419 | Mangosteen | 0.284 | 0.066 | −0.218 |
| **Accuracy** | | **0.641** | **0.734** | +0.093 |
| **Macro F1** | | **0.423** | **0.227** | −0.196 |

> S2-only baseline: `2018/stage3_orchards_weightscale_report1.csv` (2026-01-15), 18,896 test pixels  
> DEM+S1: `runs/s1_dem/stage3_s1_dem_orchards_report.csv` (2026-07-01), 14,176 test pixels  
> Note: S2-only included 2420 (Langsat), DEM+S1 replaced it with 2413 (Longan).

---

### Plantation sub-model (LU_CODEs: 2302, 2303, 2405)

| LU_CODE | Crop | S2-Only F1 | DEM+S1 F1 | Δ F1 |
|---------|------|-----------|-----------|------|
| 2302 | Rubber | 0.854 | 0.784 | −0.070 |
| 2303 | Oil palm | 0.842 | 0.698 | −0.144 |
| 2405 | Coconut | 0.250 | 0.311 | **+0.061** |
| **Accuracy** | | **0.841** | **0.747** | −0.094 |
| **Macro F1** | | **0.649** | **0.598** | −0.051 |

> S2-only baseline: `2018/stage3_plantation_weightscale_report1.csv` (2026-01-15), 21,100 test pixels  
> DEM+S1: `runs/s1_dem/stage3_s1_dem_plantation_report.csv` (2026-07-01), 21,081 test pixels

---

### Field sub-model (LU_CODEs: 2101, 2204, 2205)

| LU_CODE | Crop | S2-Only F1 | DEM+S1 F1 | Δ F1 |
|---------|------|-----------|-----------|------|
| 2101 | Rice | 0.785 | **0.791** | **+0.006** |
| 2204 | Cassava | 0.634 | 0.599 | −0.035 |
| 2205 | Pineapple | 0.631 | 0.608 | −0.023 |
| **Accuracy** | | **0.689** | **0.657** | −0.032 |
| **Macro F1** | | **0.683** | **0.666** | −0.017 |

> S2-only baseline: `2018/stage3_field_weightscale_report1.csv` (2026-01-15), 31,500 test pixels  
> DEM+S1: `runs/s1_dem/stage3_s1_dem_field_report.csv` (2026-07-01), 31,500 test pixels

---

## Confusion Matrices

### Orchards DEM+S1 (rows=true, cols=pred)

| | 2403 | 2404 | 2407 | 2413 | 2416 | 2419 |
|---|---|---|---|---|---|---|
| **2403 Durian** | 10,200 | 68 | 95 | 58 | 51 | 28 |
| **2404 Rambutan** | 63 | 21 | 21 | 22 | 18 | 16 |
| **2407 Mango** | 394 | 10 | 23 | 18 | 18 | 8 |
| **2413 Longan** | 770 | 30 | 54 | 106 | 25 | 13 |
| **2416 Jackfruit** | 1,155 | 32 | 43 | 50 | 34 | — |
| **2419 Mangosteen** | 626 | 24 | 43 | 42 | — | 27 |

### Plantation DEM+S1 (rows=true, cols=pred)

| | 2302 | 2303 | 2405 |
|---|---|---|---|
| **2302 Rubber** | 9,671 | 740 | 89 |
| **2303 Oil palm** | 4,446 | 6,039 | 15 |
| **2405 Coconut** | 20 | 31 | 30 |

### Field DEM+S1 (rows=true, cols=pred)

| | 2101 | 2204 | 2205 |
|---|---|---|---|
| **2101 Rice** | 7,218 | 2,459 | 823 |
| **2204 Cassava** | 1,085 | 8,043 | 1,372 |
| **2205 Pineapple** | 1,052 | 3,994 | 5,440 |

---

## Key Observations

### 1. Durian (2403) is the standout DEM+S1 winner (+0.104 F1)

Durian is the only crop where DEM+S1 outperforms S2-only (0.848 vs 0.744). Two factors explain this:
- **Terrain specificity**: Durian orchards in Rayong are planted on gentle slopes distinct from flat paddy/cassava areas; elevation and slope provide strong discriminating signal.
- **SAR canopy structure**: Durian has a dense, rounded crown that produces distinctive double-bounce SAR backscatter patterns.

Durian also dominates the orchards test set (10,500 of 14,176 pixels), which inflates the overall orchards accuracy (0.734) while masking poor performance on all other orchard types.

### 2. Rare orchards collapse to near-zero (Mango, Jackfruit, Mangosteen)

Mango drops from F1=0.722 → 0.086, Jackfruit from 0.298 → 0.049. These crops:
- Have far fewer samples reaching Stage 3 (due to Stage 2's 29.6% orchards recall — only the "obvious" orchards pixels pass through)
- Are structurally similar to Durian in SAR/terrain space, so the Stage 3 model overwhelmingly predicts Durian for any orchard-like pixel

### 3. Plantation stays reasonable — Rubber and Oil palm still separable with SAR

Rubber (2302) F1=0.784, Oil palm (2303) F1=0.698. These are among the most SAR-distinguishable tree crops:
- Rubber has a strong seasonal SAR signature (defoliation during dry season → lower backscatter)
- Oil palm has a distinctive frond geometry with high VH backscatter year-round

The ~0.07–0.14 drop vs S2-only is modest compared to the orchards collapse, confirming SAR partially captures plantation structure.

### 4. Field crops are the most competitive with S2-only (macro F1: 0.666 vs 0.683)

Rice (2101) F1 is essentially equal (0.791 vs 0.785). Rice flooding cycles produce a distinctive SAR temporal signature (low backscatter during flooding). Cassava and pineapple drop slightly but remain usable. Field is the subclass where DEM+S1 is closest to parity with S2-only.

### 5. other_econ produced no training data

No pixels passed the Stage 2 → Stage 3 other_econ filter (Stage 2 recall for other_econ was 0.4%). This subclass is effectively unclassifiable with the current pipeline.

---

## Full Pipeline Summary (DEM+S1)

| Stage | Task | Key Metric | Status |
|-------|------|-----------|--------|
| Stage 1 | Broad 4-class (econ/water/others/forest) | Forest F1=0.816, Econ F1=0.677, Acc=0.667 | ✅ Complete |
| Stage 2 | Econ subclass (orchards/plantation/field) | Acc=0.509, Macro F1=0.365 | ✅ Complete — poor |
| Stage 3 orchards | LU_CODE within orchards | Durian F1=0.848, rest <0.2 | ✅ Complete — Durian only |
| Stage 3 plantation | LU_CODE within plantation | Rubber F1=0.784, Oil palm F1=0.698 | ✅ Complete — usable |
| Stage 3 field | LU_CODE within field | Rice F1=0.791, Macro F1=0.666 | ✅ Complete — competitive |

---

## Suggested Next Steps

1. **Add Sentinel-2 spectral indices (highest priority).** Stage 2 and Stage 3 results confirm that the DEM+S1 feature set is insufficient for crop-type discrimination. A combined DEM+S1+S2 run is expected to recover the large losses in orchards and plantation sub-classification.

2. **Extend the NaN-column drop to each stage's pixel subset.** The 14 additional all-NaN columns within econ pixels (indices 4, 11, 20, 27, 36, 43, 52, 59, 67, 74, 82, 89, 97, 104 after Stage 1 valid_cols) waste model capacity. These should be dropped in a second-pass valid_cols specific to Stage 2/3 training data.

3. **Investigate Durian terrain features.** The large Durian gain (+0.104 F1) suggests DEM slope/TPI features are genuinely useful for certain crops. Feature importance analysis on Stage 3 orchards would confirm which DEM features drive this.

4. **Increase n_components in the next run.** All three stages selected n_components=150 (the search ceiling). Expanding to [150, 200, 300] is likely to improve accuracy at some training time cost.
