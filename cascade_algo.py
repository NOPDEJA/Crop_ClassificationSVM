"""cascade_algo.py

The estimator seam of the parcel-disjoint cascade: the one place where the
learning ALGORITHM is chosen. Everything else in train_parcel_cascade.py is
ARCHITECTURE -- the parcel split, the per-stage caps, the cross-fitted Stage-1
routing, the Platt calibration on fold 1's calibration half, the operating-point
sweep, the strict fold-2 scoring -- and is identical whichever algorithm runs.

That division is the whole point. The joint paper's rule (A) is "different
algorithm, same architecture", so the comparison is only valid if the set of
things that differ between the two runs is exactly the contents of this file.

WHAT BELONGS TO THE ALGORITHM (this file, the collaborator's to choose)
  the estimator itself, and the preprocessing that estimator needs:
  StandardScaler and the Nystroem kernel map exist only to let a LINEAR SVC
  approximate an RBF kernel, so they are dropped for XGBoost rather than forced
  on it. OneVsRestClassifier is dropped too -- keeping it would compare the SVM
  against an ensemble of independent binary XGBoost models, which is not what
  "XGBoost" means. This is a real change to the learning rule (it changes the
  fitted losses and the margins being calibrated), not a cosmetic one, and the
  methods section has to say so.

WHAT DOES NOT BELONG TO THE ALGORITHM
  calibration. The cascade's decision rule is an argmax over CALIBRATED
  probabilities and the operating-point sweep divides by the calibration prior,
  so a downstream architectural component consumes the calibrator's output. If
  one arm emitted Platt-scaled probabilities and the other raw predict_proba,
  the two arms would be using different decision rules and their alpha2/alpha3
  would not be comparable. Both arms therefore go through the SAME per-class
  sigmoid procedure, on the same calibration rows, in the same class order.
  What each arm supplies is a RAW PER-CLASS SCORE:

      SVM      OneVsRest decision_function -- per-class margins
      XGBoost  predict(output_margin=True) -- pre-softmax multiclass scores

  Both are "higher means more likely", which is all _SigmoidCalibration needs.
  Note what is NOT done here: feeding the calibrator log(predict_proba) would be
  an arbitrary score transformation and behaves badly against the fallback in
  PlattCalibrated, and describing XGBoost as "already calibrated" would be
  wrong -- exposing predict_proba is not evidence of calibration.

Env (read by train_parcel_cascade.py, documented here because they are this
file's inputs):
  ALGO=svm|xgb      which arm. svm is the default and reproduces M5/E7.
  PARAMS=<file>     per-stage hyperparameters, JSON. See load_params().
  XGB_DEVICE=cpu|cuda
  XGB_NTHREAD=<n>   0 or unset means "all cores"
  XGB_NATIVE_NAN=1  ABLATION ONLY: let XGBoost learn a default split direction
                    for missing values instead of median-imputing. The run of
                    record imputes, so both arms see the same values; this is
                    offered as a separately labelled run, never as the headline.
"""
import json
import os

import numpy as np
from sklearn.impute import SimpleImputer

ALGO = os.environ.get("ALGO", "svm")
if ALGO not in ("svm", "xgb"):
    raise SystemExit(f"ALGO={ALGO!r} is not an arm I know; use 'svm' or 'xgb'")

XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")
XGB_NTHREAD = int(os.environ.get("XGB_NTHREAD", "0"))
XGB_NATIVE_NAN = os.environ.get("XGB_NATIVE_NAN", "0") == "1"

# The collaborator's published crop-model settings (github.com/Gunkartan/geospatial,
# train_crops.py at f5f1a6a). They are the STARTING POINT only, and they are not
# "his tuned values" for this cascade in any meaningful sense: they were selected
# on a pixel split against a flat 14-class head, not on a parcel split inside a
# three-stage router. The contract gives him E4's budget -- 24 candidates x 3
# parcel-grouped folds per stage -- to replace every one of them.
DEFAULT_XGB = dict(n_estimators=800, learning_rate=0.1, max_depth=10,
                   subsample=0.6, colsample_bytree=0.8)


class XGBMarginClassifier:
    """Native multiclass XGBoost behind the interface PlattCalibrated expects.

    Two things this class exists to get right.

    1. LABEL ENCODING. The cascade's labels are LU codes (2101 ... 2420) at
       Stage 3 and superclass codes (1, 2, 3, 4) at Stages 1 and 2 -- never
       0..K-1, which is what XGBoost requires. The mapping is done here and kept
       reversible through classes_, which is sorted so it matches the ordering
       every other estimator in the pipeline uses. The composition step indexes
       probability columns by position (`list(e_classes[g]).index(code)`), so a
       silent column permutation would not raise anywhere -- it would just
       relabel the crops. Hence the assertions.

    2. IMPUTATION. Fitted on the FIT rows only, exactly as the SVM's Pipeline
       does, so the missing-value policy is identical in both arms rather than
       identical in name. It is a separate object here rather than a Pipeline
       step so that sample_weight reaches XGBoost without sklearn metadata
       routing, which the SVM path needs only because LabelBinarizer sits
       between the weights and the estimator.
    """

    def __init__(self, params):
        self.params = dict(params)

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier

        self.classes_ = np.unique(y)
        assert self.classes_.size >= 3, (
            f"XGBMarginClassifier expects a multiclass problem, got "
            f"{self.classes_.size} class(es): {self.classes_}")
        enc = np.searchsorted(self.classes_, y)
        assert np.array_equal(self.classes_[enc], y), "label encoding is not reversible"

        if XGB_NATIVE_NAN:
            self.imputer_ = None
            Xf = X
        else:
            self.imputer_ = SimpleImputer(strategy="median").fit(X)
            Xf = self.imputer_.transform(X)

        p = dict(self.params)
        p.setdefault("tree_method", "hist")
        p.setdefault("random_state", 42)
        p["objective"] = "multi:softprob"
        p["device"] = XGB_DEVICE
        if XGB_NTHREAD:
            p["n_jobs"] = XGB_NTHREAD
        self.clf_ = XGBClassifier(**p)
        self.clf_.fit(Xf, enc, sample_weight=sample_weight)
        return self

    def decision_function(self, X):
        Xf = X if self.imputer_ is None else self.imputer_.transform(X)
        m = self.clf_.predict(Xf, output_margin=True)
        m = m.reshape(-1, 1) if m.ndim == 1 else m
        assert m.shape[1] == self.classes_.size, (
            f"XGBoost returned {m.shape[1]} margin columns for "
            f"{self.classes_.size} declared classes {self.classes_}")
        return m


def make_xgb(params):
    return XGBMarginClassifier(params)


def load_params(path, svm_defaults):
    """Per-stage hyperparameters, for either arm.

    Shape of the file:

        {"stage1": {...},
         "stage2": {...},
         "stage3": {"orchards": {...}, "plantation": {...}, "field": {...}}}

    stage3 may also be a single flat dict, applied to every expert. A missing
    key falls back to the built-in default for that stage, so a file need only
    state what it changes.

    Returns (p1, p23, p3_of_group_name). The SVM defaults are passed in rather
    than duplicated here, because they are frozen constants of the SVM arm and
    belong with it.
    """
    d1, d23, d3 = svm_defaults
    if ALGO == "xgb":
        d1 = d23 = d3 = DEFAULT_XGB
    if not path:
        return dict(d1), dict(d23), {g: dict(d3) for g in ("orchards", "plantation", "field")}

    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    declared = cfg.get("algo")
    if declared and declared != ALGO:
        raise SystemExit(f"{path} declares algo={declared!r} but ALGO={ALGO!r}")

    p1 = dict(d1, **cfg.get("stage1", {}))
    p23 = dict(d23, **cfg.get("stage2", {}))
    s3 = cfg.get("stage3", {})
    if s3 and not any(k in ("orchards", "plantation", "field", "tree") for k in s3):
        s3 = {g: s3 for g in ("orchards", "plantation", "field")}
    p3 = {g: dict(d3, **s3.get(g, {})) for g in ("orchards", "plantation", "field", "tree")}
    return p1, p23, p3


def subtype_weights(fit_idx, g_of, y, gname, sink):
    """E7's actual Stage-2 weighting: sqrt-tempered over post-cap SUBTYPE counts,
    renormalised so each group's TOTAL mass is unchanged.

    Verbatim in substance from s2mass_stage2.py, which is where it was developed
    and gated (G3: replicated across three fresh pool draws, +0.0072/+0.0087/
    +0.0074). It is reproduced here rather than imported because s2mass_stage2.py
    is a one-shot experiment script that replays another run's RNG, and importing
    it would drag that replay in.

    Read the mechanism carefully, because it is NOT the sqrt class weighting the
    CLASS_WEIGHT=sqrt path applies. Mass moves only WITHIN a crop group, never
    between groups: Stage 2's four-way routing prior is left exactly as it was,
    and what changes is the balance of crops inside each group that Stage 2 sees.
    Weighting across groups instead would move the routing prior, which is what
    the alpha2 operating point is for.
    """
    w = np.ones(fit_idx.size, dtype=np.float64)
    table = []
    for g in sorted(np.unique(g_of[fit_idx])):
        sel = np.flatnonzero(g_of[fit_idx] == g)
        n_g = sel.size
        if g == sink:                       # one subtype by construction
            table.append({"group": gname[g], "lu_code": 0, "rows": int(n_g), "weight": 1.0})
            continue
        codes = y[fit_idx[sel]]
        uniq, cnt = np.unique(codes, return_counts=True)
        raw = np.sqrt(cnt.max() / cnt)
        wg = raw[np.searchsorted(uniq, codes)]
        wg *= n_g / wg.sum()                # group total mass unchanged
        w[sel] = wg
        for c, n in zip(uniq, cnt):
            table.append({"group": gname[g], "lu_code": int(c), "rows": int(n),
                          "weight": round(float(wg[codes == c][0]), 4)})
        assert abs(wg.sum() - n_g) < 1e-6, "subtype weights changed a group's total mass"
    return w, table


def manifest_entry():
    """What the run must declare about its algorithm, per protocol section 7."""
    e = {"algo": ALGO}
    if ALGO == "xgb":
        import xgboost
        e.update({"xgboost_version": xgboost.__version__, "device": XGB_DEVICE,
                  "nthread": XGB_NTHREAD or "all", "native_nan": XGB_NATIVE_NAN,
                  "deterministic": XGB_NTHREAD == 1 and XGB_DEVICE == "cpu"})
    return e
