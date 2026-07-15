#!/usr/bin/env bash
# Hourly clear-bed Z-stack calibration for the 24h one-shot build.
# Cron calls this at minute 0 every hour. Each run captures a full Z sweep into
# bed_ref_z/hHH/ for the current hour. Once all 24 hour-dirs have a full stack
# (>=290 refs each), it removes its own cron line — the one-shot 24h build ends.
# Guards: capture_z_stack.py itself refuses if a print is running.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$HERE/capture_hourly.log"
REFDIR="$HERE/bed_ref_z"
TARGET_HOURS=24
FULL_MIN=290            # a Z sweep z5..z304 = 300 imgs; >=290 = "complete enough"

echo "[$(date '+%F %T')] hourly calibration run start (h$(date +%H))" >>"$LOG"
cd "$HERE"
if python3 capture_z_stack.py >>"$LOG" 2>&1; then
    echo "[$(date '+%F %T')] capture OK" >>"$LOG"
else
    echo "[$(date '+%F %T')] capture FAILED/skipped (rc=$?)" >>"$LOG"
fi

# count hour-dirs that hold a full stack
complete=0
for d in "$REFDIR"/h[0-2][0-9]; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -name 'z*.png' | wc -l)
    [ "$n" -ge "$FULL_MIN" ] && complete=$((complete + 1))
done
echo "[$(date '+%F %T')] complete hour-stacks: $complete/$TARGET_HOURS" >>"$LOG"

if [ "$complete" -ge "$TARGET_HOURS" ]; then
    echo "[$(date '+%F %T')] 24h build DONE — removing cron" >>"$LOG"
    ( crontab -l 2>/dev/null | grep -vF 'capture_hourly.sh' ) | crontab - || true
fi
