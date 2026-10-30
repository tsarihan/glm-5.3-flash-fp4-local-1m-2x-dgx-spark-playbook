#!/usr/bin/env bash
# Stop (never rm) the GLM container if MemAvailable falls under the floor.
# Owner decision 2026-10-24 (revised): floors 2 GiB - owner: "we ran at 1-2gb free wo problems before".
# GLM FP4-Local dips to ~4-6 GiB free on BOTH Sparks right after
# weight load, so the floor is STARTUP_FLOOR for the first STARTUP_SECS of the container's
# life, then FLOOR. Polls every 2 s and logs the running minimum so real headroom is known.
FLOOR="${FLOOR:-2}"; STARTUP_FLOOR="${STARTUP_FLOOR:-2}"; STARTUP_SECS="${STARTUP_SECS:-600}"
NAME="$1"; LOG="$HOME/llmruntimes/glm53-fp4local/memguard.log"
t0=$(date +%s); min=999
echo "$(date +%T) guard start $NAME floor=${FLOOR}G startup_floor=${STARTUP_FLOOR}G for ${STARTUP_SECS}s" >> "$LOG"
while docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  el=$(( $(date +%s) - t0 )); f=$FLOOR; [ "$el" -lt "$STARTUP_SECS" ] && f=$STARTUP_FLOOR
  [ "$a" -lt "$min" ] && min=$a
  echo "$(date +%T) avail=${a}G floor=${f}G min=${min}G" >> "$LOG"
  if [ "$a" -lt "$f" ]; then echo "$(date +%T) GUARD: ${a}G < ${f}G, stopping $NAME" >> "$LOG"; docker stop "$NAME" >/dev/null; exit 1; fi
  sleep 2
done
echo "$(date +%T) guard end $NAME (container not running), min=${min}G" >> "$LOG"
