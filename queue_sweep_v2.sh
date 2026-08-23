#!/usr/bin/env bash
# Prior-correction sweep for the s2_2018_3date_v2 arm.
#
# The published 3-date arm has a swept operating point (§6.7); the v2 arm -- the one
# that actually wins -- has never had one, because only its held-out test probabilities
# were saved. Its separability changed, so §6.7's alpha optima cannot be inherited.
#
# Three serial steps, none of which refit anything:
#   1. save_stage23_probs.py         full-tile Stage-2/3 probabilities   (~10 min)
#   2. save_stage3_probs_all_econ.py every Stage-3 model on every econ px (~20 min)
#   3. sweep_prior_alpha.py          the 256-cell (alpha_2, alpha_3) grid (~1h45)
# Timings are the s2_2018_3date arm's; v2 has a wider kernel, so expect longer.
set -u

PY=/c/Conda_environment/envs/svm_env/python.exe
export ARM=s2_2018_3date_v2
OUT=./runs/$ARM
mkdir -p "$OUT/sweep"

log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

step () {  # step <script> <logname>
    log "START $1"
    $PY -u "$1" > "$OUT/$2.log" 2> "$OUT/$2.err"
    rc=$?
    log "END   $1 exit=$rc"
    if [ $rc -ne 0 ]; then log "ABORT — see $OUT/$2.err"; exit $rc; fi
}

log "sweep chain start, ARM=$ARM"
step save_stage23_probs.py         save_probs
step save_stage3_probs_all_econ.py save_probs_allecon
step sweep_prior_alpha.py          sweep/sweep
log "SWEEP COMPLETE"
