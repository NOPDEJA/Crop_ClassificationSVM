"""save_stage23_probs.py

Stage 2 and Stage 3 saved probabilities only for their held-out test splits, so
post-hoc prior correction (S2_SVM_ANALYSIS.md 6.4, option 1) cannot be applied
over the tile.  This re-runs predict_proba with the already-fitted models --
no retraining -- and stores full-tile probabilities for every routed pixel.

Outputs (in runs/<RUN>/):
  stage2_<TAG>_full_prob.npy    float32 [n_econ, 3]   rows = econ_idx order
  stage2_<TAG>_full_idx.npy     int32   [n_econ]      NPZ row of each
  stage3_<TAG>_<grp>_full_prob.npy / _full_idx.npy / _full_classes.npy
"""
import numpy as np, joblib, time
from config import (NPZ, PRED_CHUNK, OUT_DIR, TAG, STAGE1_PRED, STAGE2_PRED,
                    STAGE2_MODEL, VALID_COLS_NPY, STAGE3_MODEL_TPL)

GROUPS = {1: "orchards", 2: "plantation", 3: "field"}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def probs_for(model, X, idx, chunk=PRED_CHUNK):
    out = np.zeros((idx.size, len(model.classes_)), dtype=np.float32)
    for s in range(0, idx.size, chunk):
        e = min(idx.size, s + chunk)
        out[s:e] = model.predict_proba(X[idx[s:e]]).astype(np.float32)
        log(f"    {e}/{idx.size}")
    return out

if __name__ == "__main__":
    valid_cols = np.load(VALID_COLS_NPY)
    log("loading X")
    X = np.load(NPZ, allow_pickle=True)["X"].astype(np.float32)[:, valid_cols]
    s1 = np.load(STAGE1_PRED)
    s2 = np.load(STAGE2_PRED)

    econ_idx = np.flatnonzero(s1 == 1).astype(np.int32)
    log(f"stage 2: {econ_idx.size} econ pixels")
    m2 = joblib.load(STAGE2_MODEL)
    np.save(f"{OUT_DIR}/stage2_{TAG}_full_prob.npy", probs_for(m2, X, econ_idx))
    np.save(f"{OUT_DIR}/stage2_{TAG}_full_idx.npy", econ_idx)
    np.save(f"{OUT_DIR}/stage2_{TAG}_full_classes.npy", m2.classes_)
    del m2

    for lab, grp in GROUPS.items():
        idx = np.flatnonzero((s1 == 1) & (s2 == lab)).astype(np.int32)
        log(f"stage 3 {grp}: {idx.size} pixels")
        m3 = joblib.load(STAGE3_MODEL_TPL.format(grp=grp))
        np.save(f"{OUT_DIR}/stage3_{TAG}_{grp}_full_prob.npy", probs_for(m3, X, idx))
        np.save(f"{OUT_DIR}/stage3_{TAG}_{grp}_full_idx.npy", idx)
        np.save(f"{OUT_DIR}/stage3_{TAG}_{grp}_full_classes.npy", m3.classes_)
        del m3
    log("done")
