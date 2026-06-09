#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES_DEFAULT="192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119"
NODES=(${NODES:-$NODES_DEFAULT})
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1500}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-6}"
SHARD_COUNT="${SHARD_COUNT:-80}"
MODEL_FILTER="${MODEL_FILTER:-^qwen35_4b$}"
TASK_REGEX="${TASK_REGEX:-.*}"
SYNC_SCRIPTS="${SYNC_SCRIPTS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_WORKERS="${LAUNCH_WORKERS:-1}"
SYNC_MODEL_ASSETS="${SYNC_MODEL_ASSETS:-0}"
MODEL_SRC="${MODEL_SRC:-192.168.6.115}"
MODEL_SYNC_NODES="${MODEL_SYNC_NODES:-192.168.6.116 192.168.6.117 192.168.6.118}"
MODEL_SYNC_RELS="${MODEL_SYNC_RELS:-models/gemma-3-4b-it models/llama3-8b SAC/single/outputs/backdoor_model_gemma3_4b_it_v1 SAC/single/outputs/cssc_static/gemma3_4b_it SAC/single/outputs/cssc_static_ablation/gemma3_4b_it SAC/single/outputs/backdoor_model_llama3_v4 SAC/single/outputs/backdoor_model_llama3_v3 SAC/single/outputs/cssc_static/llama3_8b_v4 SAC/single/outputs/cssc_static_ablation/llama3_8b_v4}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-controls-defense] %s %s\n' "$(date '+%F %T')" "$*"
}

host_offset() {
  local host="$1"
  local last="${host##*.}"
  printf '%s\n' "$((last - 110))"
}

ssh_cmd() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

sync_scripts() {
  local host="$1"
  (( SYNC_SCRIPTS == 1 )) || return 0
  ssh_cmd "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$SCRIPT_DIR/merge_lora_to_base.py" \
    "$SCRIPT_DIR/eval_representation_editing_formal.py" \
    "$SCRIPT_DIR/train_refusal_tuning_baseline.py" \
    "$SCRIPT_DIR/run_adapter_controls_defense_worker.sh" \
    "$host:$ROOT/scripts/" >/dev/null
}

maybe_sync_model_assets() {
  (( SYNC_MODEL_ASSETS == 1 )) || return 0
  local host rel src_path dst_path parent
  for host in $MODEL_SYNC_NODES; do
    for rel in $MODEL_SYNC_RELS; do
      if [[ "$rel" == models/* ]]; then
        src_path="/home/xlz/$rel"
        dst_path="/home/xlz/$rel"
      else
        src_path="/home/xlz/$rel"
        dst_path="/home/xlz/$rel"
      fi
      if ssh_cmd "$host" "test -e '$dst_path'"; then
        log "asset exists host=$host path=$dst_path"
        continue
      fi
      parent="$(dirname "$dst_path")"
      log "sync asset ${MODEL_SRC}:${src_path} -> ${host}:${dst_path}"
      ssh_cmd "$host" "mkdir -p '$parent'" || continue
      if (( DRY_RUN == 1 )); then
        continue
      fi
      scp -3 -pr "${SSH_OPTS[@]}" "${MODEL_SRC}:${src_path}" "${host}:${parent}/" || log "asset sync failed host=$host path=$dst_path"
    done
  done
}

launch_host() {
  local host="$1"
  local offset="$2"
  ssh "${SSH_OPTS[@]}" "$host" \
    "ROOT='$ROOT' PYTHON='$PYTHON' IDLE_MEM_MB='$IDLE_MEM_MB' IDLE_UTIL_MAX='$IDLE_UTIL_MAX' HOST_OFFSET='$offset' SHARD_COUNT='$SHARD_COUNT' MODEL_FILTER='$MODEL_FILTER' TASK_REGEX='$TASK_REGEX' DRY_RUN='$DRY_RUN' LAUNCH_WORKERS='$LAUNCH_WORKERS' bash -s" <<'REMOTE'
set -uo pipefail
cd "$ROOT" || exit 1
mkdir -p nohup outputs/supplement_20260525/adapter_controls_defense/launch_markers

rlog() {
  printf '[remote-launch-controls-defense] %s host=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$*"
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

if (( LAUNCH_WORKERS != 1 )); then
  rlog "launch disabled"
  exit 0
fi

for gpu in 0 1 2 3 4 5 6 7; do
  shard=$((HOST_OFFSET * 8 + gpu))
  marker="outputs/supplement_20260525/adapter_controls_defense/launch_markers/host$(hostname)_gpu${gpu}_shard${shard}.started"
  if [[ -f "$marker" ]]; then
    rlog "skip gpu=$gpu shard=$shard marker exists"
    continue
  fi
  if ! gpu_idle "$gpu"; then
    rlog "skip gpu=$gpu busy mem=$(gpu_field "$gpu" mem) util=$(gpu_field "$gpu" util)"
    continue
  fi
  log_path="nohup/adapter_controls_defense_gpu${gpu}_shard${shard}_$(date +%Y%m%d_%H%M%S).log"
  rlog "launch gpu=$gpu shard=$shard/$SHARD_COUNT model_filter=$MODEL_FILTER log=$log_path"
  (( DRY_RUN == 1 )) && continue
  {
    date -Is
    echo "gpu=$gpu shard=$shard shard_count=$SHARD_COUNT model_filter=$MODEL_FILTER task_regex=$TASK_REGEX log=$log_path"
  } > "$marker"
  nohup env ROOT="$ROOT" PYTHON="$PYTHON" EVAL_GPU="$gpu" SHARD_INDEX="$shard" SHARD_COUNT="$SHARD_COUNT" \
    MODEL_FILTER="$MODEL_FILTER" TASK_REGEX="$TASK_REGEX" \
    bash scripts/run_adapter_controls_defense_worker.sh > "$log_path" 2>&1 < /dev/null &
done
REMOTE
}

log "start nodes=${NODES[*]} model_filter=$MODEL_FILTER dry_run=$DRY_RUN"
maybe_sync_model_assets
for host in "${NODES[@]}"; do
  if ! ssh_cmd "$host" "test -d '$ROOT' && command -v nvidia-smi >/dev/null"; then
    log "skip unreachable or unprepared host=$host"
    continue
  fi
  sync_scripts "$host" || log "script sync failed host=$host"
  launch_host "$host" "$(host_offset "$host")" 2>&1 | while IFS= read -r line; do
    log "$host $line"
  done
done
log "complete"
