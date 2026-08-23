#!/usr/bin/env bash
# Run the remaining pipeline for the arm configured in config.py, unattended.
# Stage 1 is assumed to be already running; this waits for its predictions to
# land, then runs Stage 2 -> Stage 3 -> end-to-end -> flat 15-class scoring.
# Each step only starts if the previous one produced the artifact it must consume.

set -u
PY=/c/Conda_environment/envs/svm_env/python.exe
# Arm comes from the environment so a second arm can be chained without editing this
# file; config.py resolves the same ARM name to its TAG and NPZ. Exported because
# every python step below has to resolve the same arm.
export ARM=${ARM:-s2_2018_3date}
RUN_DIR=./runs/$ARM
TAG=$($PY -c "import config; print(config.TAG)")
mkdir -p "$RUN_DIR"
log () { echo "[$(date +%H:%M:%S)] $*"; }
log "arm=$ARM tag=$TAG dir=$RUN_DIR use_pca=${USE_PCA:-1}"

S1_PRED=$RUN_DIR/stage1_${TAG}_pred.npy
S2_PRED=$RUN_DIR/stage2_${TAG}_pred.npy
S3_ORCH=$RUN_DIR/stage3_${TAG}_orchards_model.joblib
E2E=$RUN_DIR/end_to_end_lu_pred.npy

# Single-instance lock. Two watchers waiting on the same file will both fire when
# it appears and then race on the same output paths, so refuse to start a second.
LOCK=$RUN_DIR/chain.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    log "another chain is already running (lock $LOCK) — exiting"
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# ---- Stage 1 ---------------------------------------------------------------
# RUN_STAGE1=1 trains it here, so the whole arm is one unattended command. Left off
# by default: the original use was to attach to a Stage 1 already running.
if [ "${RUN_STAGE1:-0}" = "1" ] && [ ! -f "$S1_PRED" ]; then
    log "Stage 1 starting"
    $PY -u stage1_weight_scale.py > $RUN_DIR/stage1_run.log 2> $RUN_DIR/stage1_run.err
    log "Stage 1 exit=$?"
fi

# ---- wait for Stage 1 to finish writing its full-tile predictions ----------
log "waiting for $S1_PRED"
prev=-1
stable=0
while true; do
    if [ -f "$S1_PRED" ]; then
        cur=$(stat -c %s "$S1_PRED" 2>/dev/null || echo 0)
        if [ "$cur" = "$prev" ] && [ "$cur" != "0" ]; then
            stable=$((stable + 1))
        else
            stable=0
        fi
        prev=$cur
        [ "$stable" -ge 3 ] && break
    fi
    sleep 60
done
log "Stage 1 predictions complete ($(stat -c %s "$S1_PRED") bytes)"

# ---- Stage 2 ---------------------------------------------------------------
log "Stage 2 starting"
$PY -u stage2_weighted.py > $RUN_DIR/stage2_run.log 2> $RUN_DIR/stage2_run.err
log "Stage 2 exit=$?"
if [ ! -f "$S2_PRED" ]; then
    log "ABORT: $S2_PRED missing — Stage 2 did not produce full predictions"
    exit 1
fi

# ---- Stage 3 ---------------------------------------------------------------
log "Stage 3 starting"
$PY -u stage3_new_weight.py > $RUN_DIR/stage3_run.log 2> $RUN_DIR/stage3_run.err
log "Stage 3 exit=$?"
if [ ! -f "$S3_ORCH" ]; then
    log "ABORT: $S3_ORCH missing — Stage 3 produced no models"
    exit 1
fi

# ---- End-to-end composition + scoring --------------------------------------
log "end-to-end evaluation starting"
$PY -u evaluate_end_to_end.py > $RUN_DIR/e2e_run.log 2> $RUN_DIR/e2e_run.err
log "end-to-end exit=$?"
if [ ! -f "$E2E" ]; then
    log "ABORT: $E2E missing"
    exit 1
fi

# ---- Honest scoring: which rows did the cascade actually see, then flat 15 --
log "reconstructing sampled rows"
$PY -u reconstruct_sampled_rows.py > $RUN_DIR/sampled_rows.log 2>&1
log "flat 15-class scoring"
$PY -u evaluate_flat_15class.py > $RUN_DIR/flat15.log 2>&1
log "CHAIN COMPLETE"
