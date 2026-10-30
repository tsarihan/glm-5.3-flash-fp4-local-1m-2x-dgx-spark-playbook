#!/usr/bin/env bash
# API liveness watchdog for the GLM FP4-Local TP=2 pair, run on EACH node with its own NODE_RANK.
#
# Why (eugr/spark-vllm-docker #409): under sustained coding-agent traffic a vLLM engine step can
# freeze (~once per 12-24 h, at ANY context length). The execute-model watchdog then kills
# EngineCore, but the worker processes survive holding ~85 GB RSS each - containers look healthy
# while the server is dead. memguard.sh cannot see this: memory is fine. This watches the API.
#
# Each node restarts only its LOCAL container + memguard. If only one peer restarts, the other's
# engine fails its watchdog soon after and restarts too; the pair re-rendezvous on the master
# addr/port. Polls /v1/models - a pure metadata endpoint that keeps answering during a 10-minute
# prefill, so long prefills cannot false-trigger it. Arms only after the first 200, so it never
# fights a normal boot (startup takes several minutes).
#
# usage: NODE_RANK=<0|1> nohup ./apiwatch.sh >/dev/null 2>&1 &
set -uo pipefail
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on the rank-0 node, 1 on the rank-1}"
D=$HOME/llmruntimes/glm53-fp4local
API="${API:-http://${NODE0_MGMT}:8000}"  # rank-0 management IP:port; POLL="${POLL:-60}"; NEEDS="${NEEDS:-3}"
DRAIN_GB="${DRAIN_GB:-100}"; STARTUP_GRACE="${STARTUP_GRACE:-900}"
# relaunch env: must match the live lane (multimodal: 4 images / 1 video, GMU 0.84, 1M native)
GMU="${GMU:-0.84}"; IMAGES="${IMAGES:-4}"; VIDEOS="${VIDEOS:-1}"; MAXLEN="${MAXLEN:-1048576}"
ADAPTIVE="${ADAPTIVE:-1}"; MTP="${MTP:-5}"; MTP_WIN="${MTP_WIN:-8}"; MTP_INIT="${MTP_INIT:-5}"
FAILS=0; ARMED=0
log(){ echo "$(date -Is) rank$NODE_RANK $*" >> "$D/apiwatch.log"; }
log "watch start api=$API poll=${POLL}s needs=$NEEDS"
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "$API/v1/models" 2>/dev/null) || code=000
  if [[ "$code" == "200" ]]; then
    [[ $ARMED == 0 ]] && { log "API answered, watchdog ARMED"; }
    ARMED=1; FAILS=0
  else
    FAILS=$((FAILS+1)); log "poll fail $FAILS/$NEEDS code=$code armed=$ARMED"
    if [[ $ARMED == 1 && $FAILS -ge $NEEDS ]]; then
      C=$(cat "$D/last-container-r$NODE_RANK" 2>/dev/null || true)
      log "API DEAD - stopping $C and relaunching rank $NODE_RANK"
      [[ -n "$C" ]] && docker stop "$C" >>"$D/apiwatch.log" 2>&1
      # wait for memory to drain before relaunching (same discipline as manual restarts)
      a=0
      for _ in $(seq 1 60); do
        a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [[ $a -ge $DRAIN_GB ]] && break
        sleep 5
      done
      log "drained to avail=${a}G - relaunching"
      NODE_RANK=$NODE_RANK GMU=$GMU IMAGES=$IMAGES VIDEOS=$VIDEOS MAXLEN=$MAXLEN \
        ADAPTIVE=$ADAPTIVE MTP=$MTP MTP_WIN=$MTP_WIN MTP_INIT=$MTP_INIT \
        "$D/serve.sh" >>"$D/apiwatch.log" 2>&1
      NEWC=$(cat "$D/last-container-r$NODE_RANK" 2>/dev/null || true)
      if [[ -n "$NEWC" ]]; then
        FLOOR=2 STARTUP_FLOOR=2 nohup "$D/memguard.sh" "$NEWC" >/dev/null 2>&1 &
        log "relaunched $NEWC + memguard; disarmed for ${STARTUP_GRACE}s"
      else
        log "relaunch FAILED (no container name recorded)"
      fi
      ARMED=0; FAILS=0
      sleep "$STARTUP_GRACE"
    fi
  fi
  sleep "$POLL"
done
