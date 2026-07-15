#!/usr/bin/env bash
# 15-min clear-bed Z-stack calibration for the 24h one-shot build.
# Cron calls this every 15 min. Each run captures a full Z sweep into
# bed_ref_z/qNN/ for the current 15-min slot (0..95). Once all 96 slot-dirs
# have a full stack (>=290 refs each), it removes its own cron line — the
# one-shot 24h build ends. Guard: capture_z_stack.py refuses if a print runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/capture_15min.log"
REFDIR="$HERE/bed_ref_z"
TARGET_SLOTS=96
FULL_MIN=290            # a Z sweep z5..z300 = 296 imgs; >=290 = "complete enough"

echo "[$(date '+%F %T')] slot calibration run start (q$(( ($(date +%H)*60 + 10#$(date +%M)) / 15 )))" >>"$LOG"
cd "$HERE"
if python3 capture_z_stack.py >>"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] capture OK" >>"$LOG"
else
    echo "[$(date '+%F %T')] capture FAILED/skipped (rc=$?)" >>"$LOG"
fi

# count slot-dirs that hold a full stack
complete=0
for d in "$REFDIR"/q[0-9][0-9]; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name 'z*.png' | wc -l)
    [ "$n" -ge "$FULL_MIN" ] && complete=$((complete + 1))
done
echo "[$(date '+%F %T')] complete slot-stacks: $complete/$TARGET_SLOTS" >>"$LOG"

if [ "$complete" -ge "$TARGET_SLOTS" ]; then
    echo "[$(date '+%F %T')] 24h build DONE — removing cron" >>"$LOG"
    ( crontab -l 2>/dev/null | grep -vF 'capture_15min.sh' ) | crontab - || true
fi
