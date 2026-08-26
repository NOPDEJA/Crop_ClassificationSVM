# Codex delta re-review brief

Paste the block below to Codex with the repo root as cwd, per §6 of
`docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md`. Everything the plan asked for has been run, so
this is a delta review over new content plus a verification pass on F1–F8.

---

> Delta re-review only. The full review's findings were applied. Verify each of F1–F8 in
> `docs/PLAN_2026-08-26_STAGE2_SUBTYPE_RUN.md` §2 is actually fixed in
> `docs/REPORT_2026-08-27.md`, then review ONLY the new content: the Stage-2 subtype
> experiment section in §4, the probe-rebuild subsection in §5, and the rewritten §8 items 1
> and 2.
>
> For the experiment: re-derive the three tune-half scores (M5 0.2294, control 0.2294,
> treatment 0.2375) from the saved probability arrays with the strict scorer, confirm the
> noise-floor reading rule in `runs/s2_2018_3date_parcel_s2mass/PREDECLARATION.md` was applied
> as written, and recompute the mass table from
> `runs/s2_2018_3date_parcel_s2mass/stage2_sample_weight.npy` and `stage2_fit_idx.npy`.
> Check specifically that the control is a genuine independent refit and not a copy of M5's
> array, and that the reported +0.0081 is not an artefact of the treatment selecting a
> different operating-point cell.
>
> For the probe: confirm the split is parcel-grouped with no parcel on both sides, and that
> the quoted numbers match `runs/probe_dry_season/per_class_parcel_grouped.csv`. Note that
> the split halves parcels within each crop rather than using a plain GroupShuffleSplit, and
> judge whether the report states that deviation honestly enough.
>
> Three things I want attacked hardest, because I am least sure of them:
> 1. The pool replay in `s2mass_stage2.py`. It reconstructs M5's `np.random.default_rng(42)`
>    sequence rather than re-drawing, and the whole control rests on that being exact. Six
>    checkpoints pass, but tell me if a seventh would have caught something.
> 2. The claim that the noise floor is 0.0000015. It holds the pool, the calibration rows and
>    the seed fixed, so it may understate the run-to-run noise the report elsewhere attributes
>    to full-cascade variation. Limitation 6 tries to say exactly that. Check it does not
>    overclaim.
> 3. §2's bootstrap now says the M0 interval does NOT clear zero, reversing the earlier draft.
>    Confirm `runs/s2_2018_3date_parcel_m0/m0_bootstrap.csv` supports the reversal.
>
> Do not re-litigate settled findings. Never read fold 2. Ranked findings with re-derivation
> evidence, most severe first, and state explicitly which items you verified clean.

---

## What changed since the last review, for orientation

| area | change |
|---|---|
| §1 | now five items, item 4 is the new positive result |
| §2 | bootstrap reversed to `[−0.00002, +0.00689]`, F5 wording fixed, weighted row says "not scored" |
| §3 | 6,685:1 named as whole-tile, test fold 16,942:1 given separately, F7 selection-half wording |
| §4 | retitled, and a new subsection reports the subtype-mass experiment |
| §5 | new subsection on the parcel-grouped probe rebuild, plus the dry-season finding |
| §6 | bracketing claim dropped, replaced with two proxy scenarios and "exact value unknown" |
| §7 | limitations 1, 6 and 7 rewritten |
| §8 | items 1 and 2 marked done, replaced with what follows from them |
| footer | figure-to-artifact table replacing the prose list |

## New artifacts to check against

```
runs/s2_2018_3date_parcel_m5/oracle_routing.csv
runs/s2_2018_3date_parcel_m5/stage2_routing_accuracy.csv
runs/s2_2018_3date_parcel_m5/stage2_pool_composition.csv
runs/s2_2018_3date_parcel_m0/m0_bootstrap.csv
runs/probe_dry_season/probe_replay_overlap.csv
runs/probe_dry_season/per_class_parcel_grouped.csv
runs/probe_dry_season/summary_parcel_grouped.csv
runs/s2_2018_3date_parcel_s2mass/PREDECLARATION.md
runs/s2_2018_3date_parcel_s2mass/mass_table.csv
runs/s2_2018_3date_parcel_s2mass/s2mass_summary.csv
runs/s2_2018_3date_parcel_s2mass/s2mass_routing_shift.csv
runs/s2_2018_3date_parcel_s2mass/{control,treatment}/opsweep*.csv
```

Scripts: `stage2_diagnostics.py`, `paired_bootstrap_m0.py`, `probe_replay_overlap.py`,
`probe_dry_season_grouped.py`, `s2mass_stage2.py`, `s2mass_score.py`.

---

## Round 2 outcome, 2026-08-26

All six findings applied. Two changed a conclusion rather than wording.

| finding | severity | disposition |
|---|---|---|
| F3 stale "I report a bracket" at §6 | High | fixed, now "two proxy sensitivity scenarios" |
| noise floor not saved at precision | Medium | `s2mass_score.py` recomputes exact scores and saves 10 decimals, plus both readings of the predeclared floor |
| §1 misnumbers its own result | Low | "fifth item" corrected to "fourth item" |
| probe softened its predeclared collapse branch | High | §5, §7.7 and §8.2 now take the collapse reading, and the probe-versus-cascade routing-loss comparison is deleted as unsupported |
| pool identity not proved by the six checks | Medium | §4 states the six checks are indirect and that M5 never saved the indices, then gives the byte-identical model as the direct proof |
| noise language broader than measured | Medium | narrowed to scoring-path reproducibility, and limitation 6 now says full-cascade variation is unmeasured |

**The seventh check turned out to matter more than a check.** The control's Stage-2 model is
byte-identical to M5's (SHA-256 `1d8662063489cdc5...`, separate files, hardlink count 1), so
the Stage-2 fit is deterministic given the pool. That is the direct pool proof the six
checkpoints could not give. It also means the control never measured retraining variation at
all. Re-predicting with M5's own model in a fresh process reproduces the same 0.0007 spread
that the control showed, while two predictions inside one process are bit-identical, so the
0.0000015 is threaded floating point at predict time. The report now says exactly that, and
limitation 6 returns full-cascade run-to-run variation to the unmeasured column.

Follow-ups accepted into §8: future baseline runs must save Stage-2 fit and calibration
indices or their hashes, and the pool-draw sensitivity of +0.0081 is unmeasured by design.
