#!/bin/bash
# Run N hourly bed-clear sweeps, then stop. One-shot crash-guard monitor:
# each sweep runs `bed_check.py --check` and appends the verdict to bed_check.log.
#
# Usage: ./run_24.sh [count] [interval_seconds]
#        defaults: 24 sweeps, 3600 s apart (= one per hour for a day)
#
# bed_check.py exits 2 on OCCUPIED / 1 on error — neither aborts the loop.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")" || exit 1

COUNT="${1:-24}"
INTERVAL="${2:-3600}"
LOG="bed_check.log"

for i in $(seq 1 "$COUNT"); do
  echo "=== sweep $i/$COUNT $(date '+%F %T') ===" >> "$LOG"
  python3 bed_check.py --check >> "$LOG" 2>&1 || true
  [ "$i" -lt "$COUNT" ] && sleep "$INTERVAL"
done
echo "=== ALL $COUNT SWEEPS DONE $(date '+%F %T') ===" >> "$LOG"
