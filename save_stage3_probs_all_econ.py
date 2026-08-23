"""save_stage3_probs_all_econ.py

save_stage23_probs.py stores each Stage-3 model's probabilities only for the
pixels Stage 2 actually routed to it.  Correcting Stage 2's priors moves pixels
between groups, so the corrected cascade needs every Stage-3 model's opinion on
every econ pixel.  This computes that -- again with the already-fitted models,
no retraining.

Rows are aligned to stage2_<TAG>_full_idx.npy (the econ pixel order).

Outputs (in runs/<RUN>/):
  stage3_<TAG>_<grp>_econ_prob.npy     float32 [n_econ, n_classes_of_grp]
  stage3_<TAG>_<grp>_econ_classes.npy  int32   LU codes, column order
"""
import numpy as np, joblib, time
from config import (NPZ, PRED_CHUNK, OUT_DIR, TAG, VALID_COLS_NPY,
                    STAGE3_MODEL_TPL)

GROUPS = ["orchards", "plantation", "field"]

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

if __name__ == "__main__":
    valid_cols = np.load(VALID_COLS_NPY)
    log("loading X")
    X = np.load(NPZ, allow_pickle=True)["X"].astype(np.float32)[:, valid_cols]
    econ_idx = np.load(f"{OUT_DIR}/stage2_{TAG}_full_idx.npy")
    log(f"{econ_idx.size} econ pixels")

    for grp in GROUPS:
        m = joblib.load(STAGE3_MODEL_TPL.format(grp=grp))
        out = np.zeros((econ_idx.size, len(m.classes_)), dtype=np.float32)
        log(f"{grp}: {len(m.classes_)} classes")
        for s in range(0, econ_idx.size, PRED_CHUNK):
            e = min(econ_idx.size, s + PRED_CHUNK)
            out[s:e] = m.predict_proba(X[econ_idx[s:e]]).astype(np.float32)
            log(f"    {grp} {e}/{econ_idx.size}")
        np.save(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_prob.npy", out)
        np.save(f"{OUT_DIR}/stage3_{TAG}_{grp}_econ_classes.npy",
                np.asarray(m.classes_, dtype=np.int32))
        del m, out
    log("done")
