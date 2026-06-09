#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
SRC_FULL="${SRC_FULL:-192.168.6.115}"
SRC_BASE_ALT="${SRC_BASE_ALT:-192.168.6.114}"
TARGETS_DEFAULT="192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119"
TARGETS=(${TARGETS:-$TARGETS_DEFAULT})
MAX_PARALLEL="${MAX_PARALLEL:-3}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-12}"
SYNC_BASE="${SYNC_BASE:-1}"
SYNC_SMALL_ASSETS="${SYNC_SMALL_ASSETS:-1}"
SYNC_SCRIPTS="${SYNC_SCRIPTS:-1}"
LAUNCH_STATIC="${LAUNCH_STATIC:-1}"
LAUNCH_CONTROLS="${LAUNCH_CONTROLS:-1}"
STATIC_GROUPS="${STATIC_GROUPS:-0,1,2,3}"
CONTROL_GROUPS="${CONTROL_GROUPS:-4,5,6,7}"
STATIC_SHARD_COUNT="${STATIC_SHARD_COUNT:-${#TARGETS[@]}}"
CONTROL_SHARD_COUNT="${CONTROL_SHARD_COUNT:-${#TARGETS[@]}}"
CONTROL_IDLE_MEM_MB="${CONTROL_IDLE_MEM_MB:-2000}"
CONTROL_POLL_SECONDS="${CONTROL_POLL_SECONDS:-120}"
TRANSFER_MODE="${TRANSFER_MODE:-direct}"
DIRECT_KEY="${DIRECT_KEY:-~/.ssh/codex_qwen27_fanout_ed25519}"
SSH_KEY="${SSH_KEY:-}"
LOCAL_SOURCE="${LOCAL_SOURCE:-}"
DRY_RUN="${DRY_RUN:-0}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
fi
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS}"
if [[ -n "$SSH_KEY" ]]; then
  RSYNC_SSH="$RSYNC_SSH -i $SSH_KEY -o StrictHostKeyChecking=accept-new"
fi
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

log() {
  printf '[qwen27-fanout] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh_host() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

wait_slot() {
  while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_PARALLEL )); do
    sleep 5
  done
}

stream_dir() {
  local src="$1"
  local dst="$2"
  local parent="$3"
  local name="$4"
  local marker="$5"
  local dst_path="$parent/$name"
  if ssh_host "$dst" "test -e '$marker'"; then
    log "skip existing marker dst=$dst path=$dst_path"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run copy $src:$dst_path -> $dst:$dst_path marker=$marker"
    return 0
  fi
  local safe_name log_path
  safe_name="$(printf '%s_%s_%s' "$dst" "$(basename "$parent")" "$name" | tr '/:' '__')"
  log_path="$LOG_DIR/sync_${safe_name}_$(date +%Y%m%d_%H%M%S).log"
  log "copy start $src:$dst_path -> $dst:$dst_path log=$log_path"
  if [[ "$TRANSFER_MODE" == "direct" ]]; then
    local push_cmd
    push_cmd="cd '$parent' && tar -cf - '$name' | ssh -i '$DIRECT_KEY' -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} '$dst' \"mkdir -p '$parent' && cd '$parent' && tar -xpf - && test -e '$dst_path' && mkdir -p '$(dirname "$marker")' && date -Is > '$marker'\""
    if [[ -n "$LOCAL_SOURCE" && "$src" == "$LOCAL_SOURCE" ]]; then
      bash -lc "$push_cmd" >"$log_path" 2>&1
    else
      ssh "${SSH_OPTS[@]}" "$src" "$push_cmd" >"$log_path" 2>&1
    fi
  else
    {
      set -o pipefail
      ssh "${SSH_OPTS[@]}" "$src" "cd '$parent' && tar -cf - '$name'" |
        ssh "${SSH_OPTS[@]}" "$dst" "mkdir -p '$parent' && cd '$parent' && tar -xpf - && test -e '$dst_path' && mkdir -p '$(dirname "$marker")' && date -Is > '$marker'"
    } >"$log_path" 2>&1
  fi
  local status=$?
  if (( status == 0 )); then
    log "copy done $dst:$dst_path"
  else
    log "copy failed status=$status $dst:$dst_path log=$log_path"
  fi
  return "$status"
}

sync_scripts() {
  local dst="$1"
  (( SYNC_SCRIPTS == 1 )) || return 0
  ssh_host "$dst" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run sync scripts dst=$dst"
    return 0
  fi
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$SCRIPT_DIR/run_qwen27_static_worker.sh" \
    "$SCRIPT_DIR/run_qwen27_static_waiter.sh" \
    "$SCRIPT_DIR/run_adapter_controls_defense_worker.sh" \
    "$SCRIPT_DIR/run_adapter_controls_defense_waiter.sh" \
    "$SCRIPT_DIR/merge_lora_to_base.py" \
    "$SCRIPT_DIR/eval_representation_editing_formal.py" \
    "$SCRIPT_DIR/train_refusal_tuning_baseline.py" \
    "$dst:$ROOT/scripts/"
}

sync_small_assets_one() {
  local dst="$1"
  (( SYNC_SMALL_ASSETS == 1 )) || return 0
  stream_dir "$SRC_FULL" "$dst" "$ROOT/outputs" "backdoor_model_27b" "$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/backdoor_model_27b.done"
  stream_dir "$SRC_FULL" "$dst" "$ROOT/outputs/cssc_decompose" "qwen35_27b" "$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/cssc_decompose_qwen35_27b.done"
  stream_dir "$SRC_FULL" "$dst" "$ROOT/outputs/cssc" "qwen35_27b" "$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/cssc_qwen35_27b.done"
  stream_dir "$SRC_FULL" "$dst" "$ROOT/outputs/sci" "qwen35_27b" "$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/sci_qwen35_27b.done"
}

base_source_for() {
  local dst="$1"
  case "$dst" in
    192.168.6.116|192.168.6.117|192.168.6.118|192.168.6.119)
      printf '%s\n' "$SRC_BASE_ALT"
      ;;
    *)
      printf '%s\n' "$SRC_FULL"
      ;;
  esac
}

target_shard_index() {
  local dst="$1"
  local idx=0
  for target in "${TARGETS[@]}"; do
    if [[ "$target" == "$dst" ]]; then
      printf '%s\n' "$idx"
      return 0
    fi
    idx=$((idx + 1))
  done
  printf '0\n'
}

sync_base_one() {
  local dst="$1"
  (( SYNC_BASE == 1 )) || return 0
  if ssh_host "$dst" "test -f /home/xlz/models/Qwen3.5-27B/config.json"; then
    log "base exists dst=$dst"
    ssh_host "$dst" "mkdir -p '$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers' && date -Is > '$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/base_model_qwen35_27b.done'" || true
    return 0
  fi
  local src
  src="$(base_source_for "$dst")"
  stream_dir "$src" "$dst" "/home/xlz/models" "Qwen3.5-27B" "$ROOT/outputs/supplement_20260525/qwen35_27b/sync_markers/base_model_qwen35_27b.done"
}

launch_static_one() {
  local dst="$1"
  (( LAUNCH_STATIC == 1 )) || return 0
  local shard
  shard="$(target_shard_index "$dst")"
  local cmd
  cmd="cd '$ROOT' && mkdir -p nohup outputs/supplement_20260525/qwen35_27b/logs outputs/supplement_20260525/qwen35_27b/launch_markers && for g in $STATIC_GROUPS; do tag=\$(echo \$g | tr ',' '_'); marker=outputs/supplement_20260525/qwen35_27b/launch_markers/static_waiter_shard${shard}_of_${STATIC_SHARD_COUNT}_\${tag}.launched; if [ -f \$marker ]; then echo skip-static-waiter-\$g-marker; else date -Is > \$marker; nohup env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES=\$g MAX_MEMORY_GB=30 SHARD_INDEX='$shard' SHARD_COUNT='$STATIC_SHARD_COUNT' TASK_REGEX='^(budget_sweep_sac_alpha_bp80|operator_|gate_swap_|causal_|score_ablation_)' SKIP_REGEX='^heldout_' WORKER_NAME='qwen27_fanout_shard${shard}_of_${STATIC_SHARD_COUNT}_'\"\$(hostname)\"'_'\"\$tag\" bash scripts/run_qwen27_static_waiter.sh > nohup/qwen27_fanout_static_shard${shard}_of_${STATIC_SHARD_COUNT}_\${tag}_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & echo launched-static-waiter-\$g-shard${shard}; fi; done"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run launch static dst=$dst groups=$STATIC_GROUPS shard=$shard/$STATIC_SHARD_COUNT"
    return 0
  fi
  ssh_host "$dst" "$cmd" || true
}

launch_controls_one() {
  local dst="$1"
  (( LAUNCH_CONTROLS == 1 )) || return 0
  local shard
  shard="$(target_shard_index "$dst")"
  local cmd
  cmd="cd '$ROOT' && mkdir -p nohup outputs/supplement_20260525/adapter_controls_defense/launch_markers && for g in $CONTROL_GROUPS; do tag=\$(echo \$g | tr ',' '_'); marker=outputs/supplement_20260525/adapter_controls_defense/launch_markers/qwen27_waiter_shard${shard}_of_${CONTROL_SHARD_COUNT}_\${tag}.launched; if [ -f \$marker ]; then echo skip-controls-waiter-\$g-marker; else date -Is > \$marker; nohup env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES=\$g EVAL_GPU=\$g IDLE_MEM_MB='$CONTROL_IDLE_MEM_MB' POLL_SECONDS='$CONTROL_POLL_SECONDS' SHARD_INDEX='$shard' SHARD_COUNT='$CONTROL_SHARD_COUNT' MODEL_FILTER='^qwen35_27b$' TASK_REGEX='.*' REFUSAL_CUDA_DEVICES=\$g bash scripts/run_adapter_controls_defense_waiter.sh > nohup/adapter_controls_qwen27_waiter_shard${shard}_of_${CONTROL_SHARD_COUNT}_\${tag}_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null & echo launched-controls-waiter-\$g-shard${shard}; fi; done"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run launch controls dst=$dst groups=$CONTROL_GROUPS shard=$shard/$CONTROL_SHARD_COUNT"
    return 0
  fi
  ssh_host "$dst" "$cmd" || true
}

log "start targets=${TARGETS[*]} max_parallel=$MAX_PARALLEL dry_run=$DRY_RUN"

for dst in "${TARGETS[@]}"; do
  sync_scripts "$dst" || log "script sync failed dst=$dst"
done

for dst in "${TARGETS[@]}"; do
  wait_slot
  (
    sync_small_assets_one "$dst"
    sync_base_one "$dst"
    launch_static_one "$dst"
    launch_controls_one "$dst"
  ) &
done

wait
log "complete"
