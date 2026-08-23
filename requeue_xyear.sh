#!/usr/bin/env bash
# Re-run cross-year after the mixed-crop label fix in prepare_epoch.py.
#
# The first pass failed at rasterize: the surveys encode mixed-crop parcels as a 9-digit
# <code4>1<code4> value that does not fit int16, and the 2018 convention (map them to the
# 32767 nodata sentinel) was not reproduced. The 2020 attempt still in flight imported the
# module before the fix, so it will fail the same way; this waits for it to finish rather
# than killing it, since its index TIFs are being written and prepare_epoch.py now skips
# any date whose 8 indices already exist.
set -u
PY=/c/Conda_environment/envs/svm_env/python.exe
OUT=./runs/xyear
ARM=s2_2018_3date_v2

log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for the first cross-year pass to finish"
until grep -q "CROSS-YEAR COMPLETE" "$OUT/queue.log" 2>/dev/null; do sleep 60; done
log "first pass done; re-running both epochs with the fix"

for YEAR in 2024 2020; do
    log "=== $YEAR: preparing features and labels ==="
    $PY -u prepare_epoch.py $YEAR > $OUT/prepare_${YEAR}_fixed.log 2>&1
    rc=$?
    log "prepare_epoch $YEAR exit=$rc"
    [ $rc -ne 0 ] && { log "SKIPPING $YEAR (see $OUT/prepare_${YEAR}_fixed.log)"; continue; }
    log "=== $YEAR: cascade inference ==="
    $PY -u predict_new_epoch.py $YEAR $ARM > $OUT/predict_${YEAR}.log 2>&1
    log "predict_new_epoch $YEAR exit=$?"
done
log "CROSS-YEAR RERUN COMPLETE"
