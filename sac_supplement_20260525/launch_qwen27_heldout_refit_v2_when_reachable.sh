#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_WORKER="$SCRIPT_DIR/run_qwen27_heldout_refit_v2.sh"
HOSTS="${HOSTS:-7.202 7.201}"
REMOTE_ROOT_CANDIDATES="${REMOTE_ROOT_CANDIDATES:-/home/xlz/SAC/single /mnt/disk/xlz/SAC/single}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
SELECT_SIZES="${SELECT_SIZES:-500 1000}"
SEEDS="${SEEDS:-42}"
CHECK_SECONDS="${CHECK_SECONDS:-300}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=2}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOCAL_LOG="${LOCAL_LOG:-$LOG_DIR/launch_qwen27_heldout_refit_v2_$(date '+%Y%m%d_%H%M%S').log}"

log() {
  printf '[heldout-v2-launcher] %s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOCAL_LOG"
}

remote_root_for_host() {
  local host="$1"
  local root
  for root in $REMOTE_ROOT_CANDIDATES; do
    if ssh $SSH_OPTS "$host" "test -d '$root'" >/dev/null 2>&1; then
      printf '%s\n' "$root"
      return 0
    fi
  done
  return 1
}

remote_has_worker_running() {
  local host="$1"
  ssh $SSH_OPTS "$host" "pgrep -af '[r]un_qwen27_heldout_refit_v2.sh' >/dev/null 2>&1"
}

launch_on_host() {
  local host="$1"
  local root="$2"
  local remote_worker="$root/sac_supplement_20260525/run_qwen27_heldout_refit_v2.sh"
  local remote_log="$root/outputs/supplement_20260525/qwen35_27b/logs/heldout_refit_v2_${host//./_}_$(date '+%Y%m%d_%H%M%S').log"

  log "copy worker to $host:$remote_worker"
  ssh $SSH_OPTS "$host" "mkdir -p '$root/sac_supplement_20260525' '$root/outputs/supplement_20260525/qwen35_27b/logs'" || return 1
  scp -q ${SSH_OPTS} "$LOCAL_WORKER" "$host:$remote_worker" || return 1
  ssh $SSH_OPTS "$host" "chmod +x '$remote_worker'" || return 1

  log "launch worker on $host root=$root cuda=$CUDA_DEVICES select='$SELECT_SIZES' seeds='$SEEDS'"
  ssh $SSH_OPTS "$host" "cd '$root' && nohup env ROOT='$root' CUDA_DEVICES='$CUDA_DEVICES' SELECT_SIZES='$SELECT_SIZES' SEEDS='$SEEDS' WAIT_FOR_GPUS=1 bash '$remote_worker' >> '$remote_log' 2>&1 & echo log='$remote_log'"
}

attempt=0
log "watching hosts: $HOSTS"
while true; do
  attempt=$((attempt + 1))
  for host in $HOSTS; do
    log "checking $host"
    if ! ssh $SSH_OPTS "$host" "nvidia-smi >/dev/null 2>&1" >/dev/null 2>&1; then
      log "$host not reachable or nvidia-smi unavailable"
      continue
    fi
    root="$(remote_root_for_host "$host")" || {
      log "$host reachable but no SAC root found"
      continue
    }
    if remote_has_worker_running "$host"; then
      log "$host already has heldout v2 worker running"
      exit 0
    fi
    if launch_on_host "$host" "$root"; then
      log "launched on $host"
      exit 0
    fi
    log "launch failed on $host"
  done
  if [[ "$MAX_ATTEMPTS" != "0" && "$attempt" -ge "$MAX_ATTEMPTS" ]]; then
    log "max attempts reached"
    exit 1
  fi
  log "sleep ${CHECK_SECONDS}s before retry"
  sleep "$CHECK_SECONDS"
done
