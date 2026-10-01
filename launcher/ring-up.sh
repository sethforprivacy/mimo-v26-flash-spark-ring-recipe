#!/usr/bin/env bash
# Bring the four-rank MiMo-V2.6-Flash ring up from rank 0's host: the minimal form of the boot owner we run from
# rank 0's crontab (no sudo), e.g.
#
#   @reboot sleep 60 && bash /path/to/public-recipe/launcher/ring-up.sh up >> $HOME/mimo26-run/owner.log 2>&1
#
#   ring-up.sh up       1. exits at once if $RUN_DIR/DISABLED exists
#                       2. waits for ranks 1-3 (ssh), one fabric ping per cable (LINK_CHECKS) and > 100 GiB
#                          MemAvailable on every rank
#                       3. vllm-rank.sh up on ranks 3, 2, 1, 0 with the profile (each with its host memory guard),
#                          then the :8016 liveness shim on this host (LIVENESS=0 skips it)
#                       4. waits up to 60 min for :$PORT/health while every rank container keeps running,
#                          then writes $RUN_DIR/ready
#   ring-up.sh check    prints every rank's docker command (vllm-rank.sh check); starts nothing
#   ring-up.sh down     saves each rank's container log to $RUN_DIR/logs/, then removes the rank containers
#   ring-up.sh status   one line per rank: uptime s, MemAvailable GiB, free 32 MiB blocks, running model containers
#
# Env: PROFILE (default launcher/profile.env), HOSTS_ENV (default launcher/hosts.env), RUN_DIR ($HOME/mimo26-run),
# REMOTE_DIR (this recipe's path on ranks 1-3; default: the same path as here), LIVENESS (1).
# Profile lines are KEY=VALUE, one per line, comments on their own lines (no inline comments).
set -uo pipefail
MODE=${1:-up}
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=load-hosts.sh
. "$HERE/load-hosts.sh"
PROFILE=${PROFILE:-$HERE/profile.env}
RUN_DIR=${RUN_DIR:-$HOME/mimo26-run}
REMOTE_DIR=${REMOTE_DIR:-$(cd "$HERE/.." && pwd)}
NAME=${CONTAINER:-vllm_mimo26}
mkdir -p "$RUN_DIR/logs"
log() { echo "$(date -u -Iseconds) $*"; }
r() { if [ "$1" = 0 ]; then bash -c "$2"; else ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$(rank_ssh "$1")" "$2"; fi; }
[ -s "$PROFILE" ] || { log "FATAL: profile $PROFILE missing"; exit 1; }
KN=""; while IFS= read -r kv; do case "$kv" in ''|'#'*) ;; *) KN+=" $(printf %q "$kv")";; esac; done < "$PROFILE"
PORT=$(sed -n 's/^PORT=//p' "$PROFILE"); PORT=${PORT:-8025}

case "$MODE" in
 check)
  for i in 3 2 1 0; do echo "# rank $i"; r "$i" "env$KN bash $REMOTE_DIR/launcher/vllm-rank.sh $i check"; done
  exit;;
 down)
  ts=$(date -u +%Y%m%d-%H%M%S)
  for i in 0 1 2 3; do
    r "$i" "docker logs $NAME" > "$RUN_DIR/logs/$NAME-rank$i-$ts.log" 2>&1 || true
    r "$i" "docker rm -f $NAME >/dev/null 2>&1; true"
  done
  docker rm -f mimo26-liveness >/dev/null 2>&1; rm -f "$RUN_DIR/ready"
  log "ring down; logs in $RUN_DIR/logs/*-$ts.log"; exit;;
 status)
  st='echo "$(awk "{print int(\$1)}" /proc/uptime) $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) $(awk "/Normal/{print \$NF}" /proc/buddyinfo) $(docker ps -q --filter name=vllm_ --filter name=sgl_ | wc -l)"'
  for i in 0 1 2 3; do echo "$i $(r "$i" "$st" 2>/dev/null || echo down)"; done
  exit;;
 up) ;;
 *) echo "usage: ring-up.sh {up|check|down|status}" >&2; exit 2;;
esac

[ -f "$RUN_DIR/DISABLED" ] && { log "owner disabled ($RUN_DIR/DISABLED); not starting"; exit 0; }
log "=== owner start ==="
rm -f "$RUN_DIR/ready"
deadline=$(( $(date +%s) + 900 ))
for i in 1 2 3; do
  until r "$i" true 2>/dev/null; do [ "$(date +%s)" -lt "$deadline" ] || { log "FATAL: rank $i unreachable"; exit 1; }; sleep 10; done
done
for hop in ${LINK_CHECKS:?LINK_CHECKS not set (hosts.env)}; do
  i=${hop%%:*}; ip=${hop#*:}; n=0
  until r "$i" "ping -c1 -W2 $ip >/dev/null"; do n=$((n + 1)); [ "$n" -lt 60 ] || { log "FATAL: link rank $i -> $ip down"; exit 1; }; sleep 5; done
done
for i in 0 1 2 3; do
  n=0; until r "$i" "awk '/MemAvailable/{exit !(\$2 > 100*1048576)}' /proc/meminfo"; do
    n=$((n + 1)); [ "$n" -lt 60 ] || { log "FATAL: rank $i MemAvailable stays < 100 GiB"; exit 1; }; sleep 5; done
done
log "ranks, ring links and memory ready; profile:$KN"
for i in 3 2 1 0; do
  r "$i" "env$KN bash $REMOTE_DIR/launcher/vllm-rank.sh $i up >/dev/null" || { log "FATAL: vllm rank $i up"; exit 1; }
done
if [ "${LIVENESS:-1}" = 1 ]; then
  docker rm -f mimo26-liveness >/dev/null 2>&1
  docker run -d --name mimo26-liveness --restart unless-stopped --network host --memory 256m \
    -e "API=http://127.0.0.1:$PORT" -e PORT=8016 -v "$HERE/liveness-vllm.py:/liveness.py:ro" \
    --entrypoint python3 "${VLLM_IMAGE:-myllmbox/mimo-v26-flash-cluster-vllm:v2}" -S /liveness.py >/dev/null || log "WARN: liveness shim did not start"
fi
log "ranks started, memory guards armed; waiting for :$PORT/health"
deadline=$(( $(date +%s) + 3600 ))
until curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
  [ "$(date +%s)" -lt "$deadline" ] || { log "FATAL: :$PORT not healthy after 60 min"; exit 1; }
  for i in 0 1 2 3; do
    [ "$(r "$i" "docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null")" = true ] || { log "FATAL: rank $i container not running"; exit 1; }
  done
  sleep 15
done
date -u -Iseconds > "$RUN_DIR/ready"
log "=== serving on :$PORT (liveness :8016: $(curl -s -o /dev/null -w %{http_code} --max-time 5 http://127.0.0.1:8016/liveness)) ==="
