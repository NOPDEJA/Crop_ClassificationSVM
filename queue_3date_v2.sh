#!/usr/bin/env bash
# Queue s2_2018_3date_v2 behind the running s2_2018_5date chain.
#
# Strictly serial. Stage 1 alone peaks near 14 GB and stages 2/3 now search 600-component
# kernels; two cascades at once on 32 GB would swap and both would crawl. run_chain.sh's
# lock is per-arm, so it would NOT stop them colliding -- this script is what does.
#
# The v2 arm is the published 3-date arm minus the two configuration defects of §6.9:
# no PCA in Stage 1, kernel ceiling [300,600] in stages 2 and 3. The date window is
# unchanged, so it stays inside the §7 controlled comparison against the XGBoost cascade.
set -u

FIVE=./runs/s2_2018_5date/chain.log
OUT=./runs/s2_2018_3date_v2
mkdir -p "$OUT"

log () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "queued; waiting for the s2_2018_5date chain to end"
# run_chain.sh runs every stage synchronously and prints CHAIN COMPLETE only after the
# last python exits, so this line is a sufficient all-clear. ABORT is honoured too: if
# the 5-date arm dies, v2 is independent and should still get its turn.
until grep -qE "CHAIN COMPLETE|ABORT" "$FIVE" 2>/dev/null; do sleep 120; done
log "5-date chain ended -> $(grep -E 'CHAIN COMPLETE|ABORT' "$FIVE" | tail -1)"

log "starting s2_2018_3date_v2 (3 dates, USE_PCA=0, nyst [300,600])"
ARM=s2_2018_3date_v2 USE_PCA=0 RUN_STAGE1=1 bash run_chain.sh > "$OUT/chain.log" 2>&1
log "s2_2018_3date_v2 chain exit=$?"
