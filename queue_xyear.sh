#!/usr/bin/env bash
# Cross-year transfer, queued behind s2_2018_3date_v2. 2024 first, then 2020.
#
# 2024 leads because the prior RF paper used 2024 (+2020) imagery: running the
# cascade on 2024 against LU_RYG_2567 removes the epoch mismatch, which is one of
# the four caveats currently softening that comparison in §6.3.
#
# Inference only -- nothing is refitted. Only the three-date arms can be tested
# this way: no dry-season imagery exists for 2020 or 2024.
set -u
PY=/c/Conda_environment/envs/svm_env/python.exe
V2=./runs/s2_2018_3date_v2/chain.log
OUT=./runs/xyear
mkdir -p "$OUT"

log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "queued; waiting for the s2_2018_3date_v2 chain to end"
until grep -qE "CHAIN COMPLETE|ABORT" "$V2" 2>/dev/null; do sleep 300; done
log "v2 chain ended -> $(grep -E 'CHAIN COMPLETE|ABORT' "$V2" | tail -1)"

# Prefer the defect-free arm; fall back to the published one if v2 did not finish
# its Stage 3, so a failure there does not silently cancel the cross-year work.
ARM=s2_2018_3date_v2
if [ ! -f "./runs/$ARM/stage3_s2_3date_v2_orchards_model.joblib" ]; then
    log "WARNING: $ARM has no Stage-3 orchards model; falling back to s2_2018_3date"
    ARM=s2_2018_3date
fi
log "model arm: $ARM"

for YEAR in 2024 2020; do
    log "=== $YEAR: preparing features and labels ==="
    $PY -u prepare_epoch.py $YEAR > $OUT/prepare_$YEAR.log 2>&1
    rc=$?
    log "prepare_epoch $YEAR exit=$rc"
    if [ $rc -ne 0 ]; then
        log "SKIPPING $YEAR inference (prepare failed; see $OUT/prepare_$YEAR.log)"
        continue
    fi
    log "=== $YEAR: cascade inference ==="
    $PY -u predict_new_epoch.py $YEAR $ARM > $OUT/predict_$YEAR.log 2>&1
    log "predict_new_epoch $YEAR exit=$?"
done
log "CROSS-YEAR COMPLETE"
