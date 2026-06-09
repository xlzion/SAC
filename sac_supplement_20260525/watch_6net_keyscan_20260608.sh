#!/usr/bin/env bash
set -euo pipefail
HOSTS="${HOSTS:-192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_ROUNDS="${MAX_ROUNDS:-720}"
OUT_DIR="${OUT_DIR:-sac_supplement_20260525/logs/dispatch_6net_20260608}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o ConnectionAttempts=1
  -o ServerAliveInterval=3
  -o ServerAliveCountMax=1
  -o StrictHostKeyChecking=accept-new
)
mkdir -p "$OUT_DIR"
log(){ printf '[6net-keyscan] %s %s\n' "$(date '+%F %T')" "$*"; }
for round in $(seq 1 "$MAX_ROUNDS"); do
  log "round=$round"
  any=0
  for h in $HOSTS; do
    tmp="$OUT_DIR/ssh_${h}.tmp"
    rm -f "$tmp"
    ssh "${SSH_OPTS[@]}" "$h" 'printf ok' > "$tmp" 2>"$tmp.err" &
    pid=$!
    for _ in 1 2 3 4 5 6 7 8; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    else
      wait "$pid" 2>/dev/null || true
    fi
    if grep -q '^ok' "$tmp" 2>/dev/null; then
      log "host=$h ssh_ok"
      cat "$tmp" > "$OUT_DIR/ssh_${h}.ok"
      date -Is > "$OUT_DIR/first_ssh_ok_${h}"
      any=1
    else
      reason="$(tr '\n' ' ' < "$tmp.err" | sed 's/[[:space:]]\\+/ /g' | cut -c1-160)"
      log "host=$h ssh_unavailable ${reason:-no-response}"
    fi
    rm -f "$tmp" "$tmp.err"
  done
  if [ "$any" -eq 1 ]; then
    log "at least one host recovered; leaving markers in $OUT_DIR"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
log "max rounds reached"
