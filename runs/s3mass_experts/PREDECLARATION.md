# PREDECLARATION — E5 Stage-3 subtype mass, factorial against the E4 winner

Written before launch, per docs/PLAN_2026-08-26_POSTREVIEW_EXECUTION.md E5.

**Script:** `e5_stage3_subtype_mass.py`.

**Why:** coconut's Stage-2 routing doubled under s2mass's treatment (13.7% ->
26.6%) while its F1 stayed at 0.001 -- the binding constraint moved into the
plantation expert (96.8% rubber / 0.08% coconut in its raw fit population).
This tests reweighting the crop mix INSIDE the plantation and orchards
experts, the direct analogue of the Stage-2 subtype-mass fix.

**Stage 1 and Stage 2: frozen at the E4 winner** (Gate G4 passed
2026-08-28: n_components=1200, gamma=0.5x rule, C=30). Reused directly from
`runs/retune_stage2_orchards/treatment/stage2_prob_val.npy` in both E5 arms.
Field expert: frozen from M5 in both arms (untouched by E5).

**Populations:**
- Orchards: reproduced BIT-IDENTICAL to E4's population by replaying the same
  `SEARCH_SEED=777` draw sequence (E4's stage-2-search draw, then E4's
  orchards draw). Orchards CONTROL is therefore literally E4's already-fit,
  already-gated orchards-retuned model reused (not refit) -- same rows, same
  hyperparameters (E4 winner), unweighted.
- Plantation: a NEW `cap(rows, y, PER_LU_CAP=70,000)` draw, continuing the
  same generator (so it is reproducible but was never drawn by any prior
  script). Hyperparameters: M5's original Stage-3 config
  (n_components=1200, gamma=rule-based, C=10 -- untouched by E4, which only
  retuned orchards among the Stage-3 experts).

**Weights:** `w_c = sqrt(m_max/m_c)` over each expert's OWN post-cap crop
counts, renormalised so that expert's total fit mass is unchanged --
identical formula to `s2mass_stage2.py`, applied inside Stage 3 instead of
Stage 2.

**Arms:**
- control: orchards (E4's reused model, unweighted) + plantation (new,
  unweighted) + Stage 1/2 frozen at E4 winner + field frozen from M5.
- treatment: orchards (new, weighted, same rows/hyperparams as control) +
  plantation (new, weighted, same rows/hyperparams as control) + the same
  frozen Stage 1/2/field.

**Gate G5 (predeclared, same numeric form as G3):** treatment tune macro F1
>= control + 0.002, AND alive-crops guard (crop count with F1 >= 0.01 must
not fall). Single paired comparison (E5's own stated budget is ~1-2h, not a
multi-draw sensitivity check like G3's -- G3 answered a different question,
whether an already-passed result survives a different pool draw; this is a
first-time gate on a new experiment).

**Watch specifically:** coconut's F1, control vs treatment -- report the pair
either way, this experiment exists for coconut.

**Fold 2:** untouched. Every population here derives from
`stage1_train_idx.npy` (fold 0) or `val_cal_idx.npy` (fold 1's calibration
half) only.

**Bug lesson carried over from E4:** any weighted pipeline (`.set_fit_request()`)
is built entirely INSIDE the `sklearn.config_context(enable_metadata_routing=True)`
block, not just fitted inside it.
