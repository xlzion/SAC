#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
TARGETS=(${TARGETS:-192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119})
SHARD_COUNT="${SHARD_COUNT:-${#TARGETS[@]}}"
STATIC_CUDA_DEVICES="${STATIC_CUDA_DEVICES:-0,1,2,3}"
CONTROL_CUDA_DEVICES="${CONTROL_CUDA_DEVICES:-4,5,6,7}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=12 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=12"

log() {
  printf '[qwen27-unique-shards] %s %s\n' "$(date '+%F %T')" "$*"
}

remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

sync_scripts() {
  local host="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run sync scripts host=$host"
    return 0
  fi
  remote "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_static_worker.sh" \
    "$SCRIPT_DIR/run_qwen27_static_waiter.sh" \
    "$SCRIPT_DIR/run_adapter_controls_defense_worker.sh" \
    "$SCRIPT_DIR/run_adapter_controls_defense_waiter.sh" \
    "$SCRIPT_DIR/merge_lora_to_base.py" \
    "$SCRIPT_DIR/eval_representation_editing_formal.py" \
    "$SCRIPT_DIR/train_refusal_tuning_baseline.py" \
    "$host:$ROOT/scripts/"
}

stop_qwen27_only() {
  local host="$1"
  local cmd
  cmd=$(cat <<'SH'
cd "$ROOT" || exit 1

kill_env_worker() {
  local pattern="$1"
  local env_key="$2"
  local env_value="$3"
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -Fxq "${env_key}=${env_value}"; then
      kill "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f "$pattern" || true)
}

pkill -f "run_qwen27_static_waiter.sh" 2>/dev/null || true
pkill -f "run_qwen27_static_worker.sh" 2>/dev/null || true
pkill -f "eval_security_compression_formal.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -f "cssc_materialize_adapter.py .*outputs/backdoor_model_27b" 2>/dev/null || true
pkill -f "sasp_lora_prune.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -f "train_refusal_tuning_baseline.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -f "eval_representation_editing_formal.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -f "run_adapter_controls_defense_waiter.sh" 2>/dev/null || true
kill_env_worker "bash scripts/run_adapter_controls_defense_worker.sh" MODEL_FILTER "^qwen35_27b$"

sleep 2

pkill -9 -f "run_qwen27_static_waiter.sh" 2>/dev/null || true
pkill -9 -f "run_qwen27_static_worker.sh" 2>/dev/null || true
pkill -9 -f "eval_security_compression_formal.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -9 -f "sasp_lora_prune.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -9 -f "train_refusal_tuning_baseline.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -9 -f "eval_representation_editing_formal.py --config configs/lora_config_27b" 2>/dev/null || true
pkill -9 -f "run_adapter_controls_defense_waiter.sh" 2>/dev/null || true
kill_env_worker "bash scripts/run_adapter_controls_defense_worker.sh" MODEL_FILTER "^qwen35_27b$"

find outputs/supplement_20260525/qwen35_27b/locks -maxdepth 1 -type d -name '*.lock' -exec rm -rf {} + 2>/dev/null || true
find outputs/supplement_20260525/adapter_controls_defense/locks -maxdepth 1 -type d -name '*qwen35_27b*.lock' -exec rm -rf {} + 2>/dev/null || true
find outputs/supplement_20260525/qwen35_27b/failed -maxdepth 1 -type f -name '*.failed' -delete 2>/dev/null || true
find outputs/supplement_20260525/adapter_controls_defense/failed -maxdepth 1 -type f -name '*qwen35_27b*.failed' -delete 2>/dev/null || true
SH
)
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run stop qwen27 host=$host"
    return 0
  fi
  remote "$host" "ROOT='$ROOT' bash -s" <<<"$cmd"
}

launch_unique_shard() {
  local host="$1"
  local shard="$2"
  local tag="shard${shard}_of_${SHARD_COUNT}"
  local cmd
  cmd=$(cat <<SH
cd '$ROOT' || exit 1
mkdir -p nohup outputs/supplement_20260525/qwen35_27b/launch_markers outputs/supplement_20260525/adapter_controls_defense/launch_markers
date -Is > outputs/supplement_20260525/qwen35_27b/launch_markers/static_${tag}.launched
date -Is > outputs/supplement_20260525/adapter_controls_defense/launch_markers/qwen27_controls_${tag}.launched
nohup env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES='$STATIC_CUDA_DEVICES' IDLE_MEM_MB='$IDLE_MEM_MB' POLL_SECONDS='$POLL_SECONDS' MAX_MEMORY_GB='$MAX_MEMORY_GB' SHARD_INDEX='$shard' SHARD_COUNT='$SHARD_COUNT' TASK_REGEX='^(budget_sweep_sac_alpha_bp80|operator_|gate_swap_|causal_|score_ablation_)' SKIP_REGEX='^heldout_' WORKER_NAME='qwen27_static_unique_${tag}_'"\$(hostname)" bash scripts/run_qwen27_static_waiter.sh > nohup/qwen27_static_unique_${tag}_\$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
echo static_pid=\$!
nohup env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES='$CONTROL_CUDA_DEVICES' EVAL_GPU='$CONTROL_CUDA_DEVICES' IDLE_MEM_MB='$IDLE_MEM_MB' POLL_SECONDS='$POLL_SECONDS' MAX_MEMORY_GB='$MAX_MEMORY_GB' SHARD_INDEX='$shard' SHARD_COUNT='$SHARD_COUNT' MODEL_FILTER='^qwen35_27b$' TASK_REGEX='.*' REFUSAL_CUDA_DEVICES='$CONTROL_CUDA_DEVICES' WORKER_NAME='qwen27_controls_unique_${tag}_'"\$(hostname)" bash scripts/run_adapter_controls_defense_waiter.sh > nohup/qwen27_controls_unique_${tag}_\$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
echo controls_pid=\$!
SH
)
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run launch host=$host shard=$shard/$SHARD_COUNT"
    return 0
  fi
  remote "$host" "$cmd"
}

log "start targets=${TARGETS[*]} shard_count=$SHARD_COUNT dry_run=$DRY_RUN"

idx=0
for host in "${TARGETS[@]}"; do
  log "sync host=$host"
  sync_scripts "$host" || log "script sync failed host=$host"
  log "stop duplicate qwen27 host=$host"
  stop_qwen27_only "$host" || log "stop failed host=$host"
  log "launch unique shard host=$host shard=$idx"
  launch_unique_shard "$host" "$idx" || log "launch failed host=$host"
  idx=$((idx + 1))
done

log "complete"
