"""emit_e7_config.py

Write E7_EFFECTIVE_CONFIG.json: the single file that says what "the same
architecture" means, so that the claim is checkable instead of asserted.

WHY THIS IS NEEDED. E7's configuration has never existed in one place. It is the
union of two run directories -- runs/s2_2018_3date_parcel_m5 and
runs/retune_stage2_orchards -- plus the composition script
e7_final_cascade_fold2_read.py, and E4's directory has no manifest at all, only
winners.json. Asking the collaborator to match a configuration that exists only
as an intersection of three artifacts is asking him to guess.

WHAT IT DELIBERATELY INCLUDES. Hashes of the data, the split, the labels, the
feature order, the fit and calibration row indices, and the Stage-2 weight
vector. Rules alone are not enough: two runs that follow identical rules can
still fit different rows, and the only way to know is to compare the row
identities themselves. Where a hash cannot be computed the field says why rather
than being silently omitted.

Env:
  M5=<dir>   default ./runs/s2_2018_3date_parcel_m5
  E4=<dir>   default ./runs/retune_stage2_orchards
  E7=<dir>   default ./runs/s2_2018_3date_parcel_e7_final
  S2MASS=<dir> default ./runs/s2_2018_3date_parcel_s2mass
  OUT=<file> default ./E7_EFFECTIVE_CONFIG.json
"""
import hashlib
import json
import os

import numpy as np

M5 = os.environ.get("M5", "./runs/s2_2018_3date_parcel_m5")
E4 = os.environ.get("E4", "./runs/retune_stage2_orchards")
E7 = os.environ.get("E7", "./runs/s2_2018_3date_parcel_e7_final")
S2MASS = os.environ.get("S2MASS", "./runs/s2_2018_3date_parcel_s2mass")
OUT = os.environ.get("OUT", "./E7_EFFECTIVE_CONFIG.json")


def sha_file(path, chunk=1 << 24):
    if not os.path.exists(path):
        return {"error": "file not present on this machine"}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(path)}


def sha_arr(path):
    if not os.path.exists(path):
        return {"error": "file not present on this machine"}
    a = np.load(path)
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "shape": list(a.shape), "dtype": str(a.dtype)}


if __name__ == "__main__":
    m5 = json.load(open(f"{M5}/manifest.json"))
    e7 = json.load(open(f"{E7}/manifest.json"))
    winners = json.load(open(f"{E4}/winners.json"))
    npz_path = m5["npz"]

    print("hashing the feature matrix (3 GB, one pass)...")
    npz_hash = sha_file(npz_path)

    # Read X's shape from the .npy header inside the zip rather than decompressing
    # 2.9 GB of features to learn two integers.
    import zipfile
    from numpy.lib import format as npformat
    with zipfile.ZipFile(npz_path) as z:
        with z.open("X.npy") as fh:
            npformat.read_magic(fh)
            shape, _, _ = npformat._read_array_header(fh, (1, 0))
    n_rows, n_cols = shape
    d = np.load(npz_path, allow_pickle=True)
    names = [str(s) for s in d["feature_names"]] if "feature_names" in d.files else None

    cfg = {
      "what_this_is":
        "The effective configuration of run E7, the SVM comparator for the XGBoost "
        "crossover. Tier A is the architecture and is identical in both arms. Tier B "
        "is the algorithm and is the other side's to choose. See "
        "docs/HANDOFF_XGB_CASCADE.md.",
      "comparator_result": {
        "run": E7, "rows": e7["results"]["hard"]["rows"],
        "macro_f1_strict_hard": e7["results"]["hard"]["macro_f1"],
        "weighted_f1": e7["results"]["hard"]["weighted_f1"],
        "baseline_m5_macro_f1": e7.get("m5_baseline_macro_f1"),
        "endpoint": "hard",
        "population": "fold 2, natural (uncapped), non-crop truth mapped to 0",
      },

      "tier_a_architecture": {
        "npz": {"path": npz_path, "rows": int(n_rows), "features": int(n_cols),
                **npz_hash},
        "feature_names_in_order": names or
            "NOT STORED IN THIS NPZ -- order is alphabetical by "
            "glob('./indices/*.tif') in align_indices_labels.py, and valid_cols.npy "
            "records which survived the train-only all-NaN drop",
        "valid_cols": sha_arr(f"{M5}/valid_cols.npy"),
        "split_assign": sha_arr("./splits/split_assign.npy"),
        "parcel_id_row": sha_arr("./splits/parcel_id_row.npy"),
        "label_raster": {"path": "./label/label_47PQQ_buffered.tif",
                         **sha_file("./label/label_47PQQ_buffered.tif"),
                         "note": "3-pixel-eroded; cannot be regenerated from "
                                 "rasterize_parcel.py because compound mixed-crop "
                                 "LU_ID_L3 codes are unhandled, so the raster itself "
                                 "is the artifact of record"},
        "random_state": 42,
        "superclasses": {
          "econ": [2101, 2204, 2205, 2302, 2303, 2403, 2404, 2405, 2407, 2413,
                   2416, 2419, 2420],
          "water": [4101, 4102, 4103, 4201, 4202, 4203],
          "forest": [3100, 3101, 3200, 3201, 3300, 3301, 3401, 3501],
          "others": "every remaining valid code"},
        "stage2_groups": {"orchards": [2403, 2404, 2407, 2413, 2416, 2419, 2420],
                          "plantation": [2302, 2303, 2405],
                          "field": [2101, 2204, 2205],
                          "sink": "everything Stage 1 called economic that is not "
                                  "one of the 13"},
        "merge_tree": False,
        "sampling": {"samples_per_lu": 400000, "cap_econ": 1000000,
                     "cap_water": 500000, "cap_forest": 600000,
                     "cap_others": 800000, "per_group_cap": 200000,
                     "per_lu_cap": 70000},
        "crossfit_s1_parts": m5.get("crossfit_s1_parts"),
        "crossfit_s1_route_agreement": m5.get("crossfit_s1_route_agreement"),
        "fold1_halving": "by parcel, stratified by parcel label",
        "calibration": {
          "scheme": "per-class Platt sigmoid on fold 1's calibration half; the base "
                    "estimator is fitted once on fold 0 and never refitted",
          "calib_max": 300000,
          "sampling": "random, NOT per-class, so natural priors survive",
          "score_fed_to_the_sigmoid":
              "SVM: OneVsRest decision_function margins. XGBoost: predict("
              "output_margin=True), the pre-softmax multiclass scores. Both arms "
              "then use the SAME sigmoid procedure, class order, fallback and "
              "renormalisation. Feeding log(predict_proba) instead would be an "
              "arbitrary transformation and is not permitted.",
          "fallback": "a class with fewer than 2 positives among the calibration "
                      "rows uses the plain logistic of its raw score",
          "stage_populations": {
            "stage1": "all calibration-half rows",
            "stage2": "calibration-half rows the arm's OWN Stage 1 called economic",
            "stage3": "calibration-half rows of the expert's TRUE group, regardless "
                      "of whether Stage 1 or 2 routed them correctly"}},
        "weighting": {
          "scheme": "subtype mass (CLASS_WEIGHT=subtype)",
          "stage1": "unweighted",
          "stage2": "sqrt-tempered over post-cap subtype counts, renormalised so "
                    "each group's TOTAL mass is unchanged -- mass moves only inside "
                    "a group, so Stage 2's four-way routing prior is untouched",
          "stage3": "unweighted, all three experts. Gate G5 tested weighting the "
                    "plantation and orchards experts and FAILED at -0.0012 with the "
                    "alive-crops guard broken, so it is deliberately absent.",
          "weight_vector": sha_arr(f"{S2MASS}/stage2_sample_weight.npy"),
          "control_wording": "the same observation-level weight vector is passed to "
                             "each algorithm's native loss. NOT 'the same "
                             "cost-sensitive learning rule' -- a row weight under a "
                             "native softmax is not equivalent to the same weight "
                             "across 13 independent binary problems."},
        "operating_point": {
          "grid": "alpha2 x alpha3, each 0.0 to 1.2 in 13 steps",
          "selected_on": "fold 1's CALIBRATION half",
          "reported_on": "fold 1's TUNING half",
          "objective": "strict macro F1 over the 13 crops",
          "e7_selected_values": {"alpha2": 0.4, "alpha3": 0.5},
          "per_arm": "each arm selects its OWN alpha pair. E7's values are recorded "
                     "for reference and must NOT be copied into the XGBoost arm."},
        "missing_values": {
          "policy": "SimpleImputer(strategy='median'), fitted on each stage's own "
                    "fit rows",
          "rows_dropped": "none, in either arm, so the population is identical "
                          "regardless",
          "caveat": "Stage-2 fit populations are algorithm-dependent, so the two "
                    "arms' medians are fitted on different rows and can differ. The "
                    "policy is identical; the imputed values are not guaranteed to be.",
          "ablation": "XGB_NATIVE_NAN=1 is a separately labelled run, never the "
                      "headline"},
        "endpoint": {
          "primary": "hard cascade -- commit to the argmax branch at each stage",
          "secondary": "joint composition, reported but not the headline",
          "why_frozen": "the script emits both; without fixing this in advance "
                        "either arm could pick the more favourable one after seeing "
                        "both"},
        "scoring": {
          "convention": "strict -- the whole fold-2 population, non-crop truth "
                        "mapped to 0, an explicit ordered 13-label list, no masking "
                        "before the metric",
          "implemented_by": "train_parcel_cascade.py (the compose-and-score block), "
                            "and e7_final_cascade_fold2_read.py for E7 itself",
          "do_not_use": "evaluate_end_to_end.py -- it scores every NPZ row rather "
                        "than restricting to fold 2, so it would produce a "
                        "train-inclusive number",
          "macro_denominator": 13},
        "test_discipline": {
          "rule": "SKIP_TEST=1 until one predeclared final read",
          "honesty_note": "fold 2 is NOT globally untouched. M5 and earlier "
                          "checkpoints read it, and E7 read it once more. It must be "
                          "described as a previously observed partition."}},

      "tier_b_algorithm_svm_arm": {
        "estimator": "OneVsRestClassifier(Pipeline(SimpleImputer, StandardScaler, "
                     "Nystroem(rbf), LinearSVC))",
        "stage1": m5["params_stage1"],
        "stage2": {"n_components": winners["stage2"]["n_components"],
                   "gamma": winners["stage2"]["gamma"],
                   "C": winners["stage2"]["C"],
                   "source": "E4 GroupKFold(3) search, Gate G4 PASS",
                   "search_mean_f1_macro": winners["stage2"]["mean_test_score"]},
        "stage3": {
          "orchards": {"n_components": winners["orchards"]["n_components"],
                       "gamma": winners["orchards"]["gamma"],
                       "C": winners["orchards"]["C"],
                       "source": "E4 GroupKFold(3) search, Gate G4 PASS",
                       "search_mean_f1_macro": winners["orchards"]["mean_test_score"]},
          "plantation": dict(m5["params_stage3"], source="M5, untouched by E1-E7"),
          "field": dict(m5["params_stage3"], source="M5, untouched by E1-E7")},
        "tuning_provenance": {
          "cleanly_searched": ["stage2", "stage3.orchards"],
          "frozen_from_a_leaky_inner_cv": ["stage1", "stage3.plantation",
                                           "stage3.field"],
          "disclosure": "This arm is a HYBRID. The XGBoost arm searches all five "
                        "stages under GroupKFold, so it receives MORE clean tuning "
                        "than this one, not less. Until the matching clean SVM "
                        "retune is run, the paired difference is 'algorithm plus "
                        "tuning regime', not 'algorithm'.",
          "e4_budget": {"candidates": 24, "folds": 3, "grouping": "parcel_id_row",
                        "scoring": "f1_macro", "population": "fold 0 only",
                        "weighted_search": False,
                        "note": "E4 searched Stage 2 unweighted and refitted it "
                                "weighted. The XGBoost search copies that rather "
                                "than fixing it in one arm only."},
          "grid_ceiling": "Both E4 winners took the MAXIMUM n_components (1200) and "
                          "the MAXIMUM C (30), but gamma landed at 0.5x the rule, "
                          "which is interior to its grid. So the search hit its "
                          "capacity ceiling on two of three axes, not all three -- "
                          "the arm is plausibly under-tuned, but that is not proven "
                          "categorically."}},

      "row_identities": {
        "why": "Freezing rules is not the same as freezing rows. Two runs following "
               "identical rules can still fit different rows, and the Stage-2 "
               "population is algorithm-dependent BY DESIGN.",
        "framework_rule": "Stage 2 trains on whatever your OWN Stage 1 routed to it. "
                          "The rule is fixed and identical in both arms; the "
                          "resulting rows differ and are recorded, not forced to "
                          "match. Routing the XGBoost arm with SVM-generated routes "
                          "would mean its Stage 1 was never tested.",
        "m5_hashes": m5.get("fit_cal_hashes", {}),
        "note": "Stage 1's fit rows were not hashed in M5; the trainer now saves "
                "them as stage1_fit_idx.npy for every future run."},

      "reproduction_caveat":
        "configs/e7_svm.json reproduces E7's RECIPE, not E7's bytes. E7's orchards "
        "expert was fitted on a population drawn from E4's own generator (seed 777), "
        "whereas a single-script run draws it from the main seed-42 sequence: the "
        "same rule, a different draw. Stage 2 does replay byte-identically, because "
        "s2mass_stage2.py replayed the trainer's own RNG to obtain its pool.",
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("wrote", OUT)
    print("npz", npz_path, npz_hash.get("sha256", npz_hash))
    print("feature names stored in npz:", bool(names))
