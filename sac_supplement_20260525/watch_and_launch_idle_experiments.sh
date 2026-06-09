#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODES_DEFAULT="192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119"
NODES=(${NODES:-$NODES_DEFAULT})

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
ACCELERATE="${ACCELERATE:-/home/xlz/anaconda3/envs/qwen/bin/accelerate}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-6}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1500}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
SMALLMODEL_SHARD_COUNT="${SMALLMODEL_SHARD_COUNT:-40}"
QWEN4_SHARD_COUNT="${QWEN4_SHARD_COUNT:-10}"
SMALLMODEL_RELAUNCH_COOLDOWN_SECONDS="${SMALLMODEL_RELAUNCH_COOLDOWN_SECONDS:-1800}"
QWEN4_RELAUNCH_COOLDOWN_SECONDS="${QWEN4_RELAUNCH_COOLDOWN_SECONDS:-3600}"
LAUNCH_SMALLMODEL="${LAUNCH_SMALLMODEL:-1}"
LAUNCH_QWEN4_WAVE2="${LAUNCH_QWEN4_WAVE2:-1}"
SYNC_SCRIPTS="${SYNC_SCRIPTS:-1}"
RUN_ONCE="${RUN_ONCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}"
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=1
)
if [[ -n "$SSH_EXTRA_OPTS" ]]; then
  # Intentionally split on spaces; expected values are simple ssh -o pairs.
  read -r -a SSH_EXTRA_OPTS_ARRAY <<< "$SSH_EXTRA_OPTS"
  SSH_OPTS+=("${SSH_EXTRA_OPTS_ARRAY[@]}")
fi

log() {
  printf '[sac-autoresume] %s %s\n' "$(date '+%F %T')" "$*"
}

host_offset() {
  local host="$1"
  local last="${host##*.}"
  if [[ "$last" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$((last - 110))"
  else
    printf '0\n'
  fi
}

ssh_cmd() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

sync_remote_scripts() {
  local host="$1"
  (( SYNC_SCRIPTS == 1 )) || return 0
  ssh_cmd "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} -o ServerAliveInterval=5 -o ServerAliveCountMax=1 ${SSH_EXTRA_OPTS}" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$SCRIPT_DIR/run_smallmodel_idle_worker.sh" \
    "$SCRIPT_DIR/run_qwen4_train_wave2.sh" \
    "$host:$ROOT/scripts/" >/dev/null && return 0
  scp -q \
    -o BatchMode=yes \
    -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=1 \
    ${SSH_EXTRA_OPTS} \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$SCRIPT_DIR/run_smallmodel_idle_worker.sh" \
    "$SCRIPT_DIR/run_qwen4_train_wave2.sh" \
    "$host:$ROOT/scripts/" >/dev/null
}

probe_node() {
  local host="$1"
  ssh_cmd "$host" "test -d '$ROOT' && command -v nvidia-smi >/dev/null && hostname && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>&1
}

launch_on_node() {
  local host="$1"
  local offset="$2"
  ssh "${SSH_OPTS[@]}" "$host" \
    "ROOT='$ROOT' PYTHON='$PYTHON' ACCELERATE='$ACCELERATE' HOST_OFFSET='$offset' IDLE_MEM_MB='$IDLE_MEM_MB' IDLE_UTIL_MAX='$IDLE_UTIL_MAX' SMALLMODEL_SHARD_COUNT='$SMALLMODEL_SHARD_COUNT' QWEN4_SHARD_COUNT='$QWEN4_SHARD_COUNT' SMALLMODEL_RELAUNCH_COOLDOWN_SECONDS='$SMALLMODEL_RELAUNCH_COOLDOWN_SECONDS' QWEN4_RELAUNCH_COOLDOWN_SECONDS='$QWEN4_RELAUNCH_COOLDOWN_SECONDS' LAUNCH_SMALLMODEL='$LAUNCH_SMALLMODEL' LAUNCH_QWEN4_WAVE2='$LAUNCH_QWEN4_WAVE2' DRY_RUN='$DRY_RUN' bash -s" <<'REMOTE'
set -uo pipefail

cd "$ROOT" || exit 1
mkdir -p nohup outputs/supplement_20260525/autoresume outputs/supplement_20260525/smallmodel_idle/launch_markers outputs/supplement_20260525/qwen35_4b_train_wave2/launch_markers

rlog() {
  printf '[remote-autoresume] %s host=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$*"
}

gpu_field() {
  local gpu="$1"
  local field="$2"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v g="$gpu" -v f="$field" '
      {
        idx=$1; mem=$2; util=$3
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util)
        if (idx == g) {
          if (f == "mem") print mem
          if (f == "util") print util
        }
      }'
}

gpu_idle() {
  local gpu="$1"
  local mem util
  mem="$(gpu_field "$gpu" mem)"
  util="$(gpu_field "$gpu" util)"
  [[ -n "$mem" && -n "$util" ]] || return 1
  (( mem <= IDLE_MEM_MB && util <= IDLE_UTIL_MAX ))
}

gpu_group_idle() {
  local gpu
  for gpu in "$@"; do
    gpu_idle "$gpu" || return 1
  done
}

marker_age_ok() {
  local marker="$1"
  local cooldown="$2"
  [[ ! -f "$marker" ]] && return 0
  local now mtime age
  now="$(date +%s)"
  mtime="$(stat -c %Y "$marker" 2>/dev/null || printf '0')"
  age="$((now - mtime))"
  (( age >= cooldown ))
}

write_marker() {
  local marker="$1"
  shift
  {
    date -Is
    printf '%s\n' "$*"
  } > "$marker"
}

launch_smallmodel_gpu() {
  local gpu="$1"
  local shard="$2"
  local marker="outputs/supplement_20260525/smallmodel_idle/launch_markers/autoresume_gpu${gpu}_shard${shard}.started"
  local log_path="nohup/smallmodel_idle_autoresume_gpu${gpu}_shard${shard}_$(date +%Y%m%d_%H%M%S).log"

  if pgrep -af "run_smallmodel_idle_worker.sh" | grep -E "EVAL_GPU=${gpu}([^0-9]|$)|CUDA_VISIBLE_DEVICES=${gpu}([^0-9]|$)" >/dev/null; then
    rlog "skip smallmodel gpu=$gpu: worker already running"
    return 0
  fi
  if ! gpu_idle "$gpu"; then
    rlog "skip smallmodel gpu=$gpu: gpu busy mem=$(gpu_field "$gpu" mem) util=$(gpu_field "$gpu" util)"
    return 0
  fi
  if ! marker_age_ok "$marker" "$SMALLMODEL_RELAUNCH_COOLDOWN_SECONDS"; then
    rlog "skip smallmodel gpu=$gpu: cooldown marker=$marker"
    return 0
  fi

  rlog "launch smallmodel gpu=$gpu shard=$shard/$SMALLMODEL_SHARD_COUNT log=$log_path"
  write_marker "$marker" "gpu=$gpu shard=$shard shard_count=$SMALLMODEL_SHARD_COUNT log=$log_path"
  (( DRY_RUN == 1 )) && return 0
  nohup env ROOT="$ROOT" PYTHON="$PYTHON" EVAL_GPU="$gpu" SHARD_INDEX="$shard" SHARD_COUNT="$SMALLMODEL_SHARD_COUNT" \
    bash scripts/run_smallmodel_idle_worker.sh > "$log_path" 2>&1 &
}

launch_qwen4_wave2() {
  local shard="$1"
  local marker="outputs/supplement_20260525/qwen35_4b_train_wave2/launch_markers/autoresume_shard${shard}.started"
  local log_path="nohup/qwen4_train_wave2_autoresume_shard${shard}_$(date +%Y%m%d_%H%M%S).log"

  if pgrep -af "run_qwen4_train_wave2.sh|train_backdoor.py|accelerate .*train_backdoor.py" >/dev/null; then
    rlog "skip qwen4 wave2 shard=$shard: train worker already running"
    return 0
  fi
  if ! gpu_group_idle 0 1 2 3; then
    rlog "skip qwen4 wave2 shard=$shard: gpus 0-3 busy"
    return 0
  fi
  if ! marker_age_ok "$marker" "$QWEN4_RELAUNCH_COOLDOWN_SECONDS"; then
    rlog "skip qwen4 wave2 shard=$shard: cooldown marker=$marker"
    return 0
  fi

  rlog "launch qwen4 wave2 shard=$shard/$QWEN4_SHARD_COUNT log=$log_path"
  write_marker "$marker" "shard=$shard shard_count=$QWEN4_SHARD_COUNT gpus=0,1,2,3 log=$log_path"
  (( DRY_RUN == 1 )) && return 0
  nohup env ROOT="$ROOT" PYTHON="$PYTHON" ACCELERATE="$ACCELERATE" GPU_LIST=0,1,2,3 TRAIN_NUM_PROCESSES=4 EVAL_GPU=0 \
    SHARD_INDEX="$shard" SHARD_COUNT="$QWEN4_SHARD_COUNT" \
    bash scripts/run_qwen4_train_wave2.sh > "$log_path" 2>&1 &
}

if (( LAUNCH_SMALLMODEL == 1 )); then
  for gpu in 4 5 6 7; do
    shard=$((HOST_OFFSET * 4 + gpu - 4))
    launch_smallmodel_gpu "$gpu" "$shard"
  done
fi

if (( LAUNCH_QWEN4_WAVE2 == 1 )); then
  launch_qwen4_wave2 "$HOST_OFFSET"
fi
REMOTE
}

run_iteration() {
  local host output offset
  for host in "${NODES[@]}"; do
    output="$(probe_node "$host")"
    if (( $? != 0 )); then
      log "$host unreachable: $(printf '%s' "$output" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
      continue
    fi
    log "$host reachable: $(printf '%s' "$output" | head -n 1)"
    sync_remote_scripts "$host" || log "$host reachable but script sync failed; trying existing remote scripts"
    offset="$(host_offset "$host")"
    launch_on_node "$host" "$offset" 2>&1 | while IFS= read -r line; do
      log "$host $line"
    done
  done
}

log "watchdog start nodes=${NODES[*]} interval=${CHECK_INTERVAL_SECONDS}s dry_run=${DRY_RUN}"
while true; do
  run_iteration
  (( RUN_ONCE == 1 )) && break
  sleep "$CHECK_INTERVAL_SECONDS"
done
