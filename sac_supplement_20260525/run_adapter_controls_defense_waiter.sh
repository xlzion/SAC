#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
EVAL_GPU="${EVAL_GPU:-$CUDA_DEVICES}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
MODEL_FILTER="${MODEL_FILTER:-.*}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
REFUSAL_CUDA_DEVICES="${REFUSAL_CUDA_DEVICES:-$CUDA_DEVICES}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${CUDA_DEVICES//,/}_adapter_controls_waiter}"

cd "$ROOT" || exit 1
mkdir -p outputs/supplement_20260525/adapter_controls_defense/logs

log() {
  printf '[adapter-controls-waiter] %s worker=%s cuda=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$CUDA_DEVICES" "$*"
}

gpu_ids() {
  printf '%s\n' "$CUDA_DEVICES" | tr ',' '\n'
}

while true; do
  ok=1
  while read -r gpu; do
    [[ -n "$gpu" ]] || continue
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -z "$used" || "$used" -gt "$IDLE_MEM_MB" ]]; then
      ok=0
      break
    fi
  done < <(gpu_ids)
  if [[ "$ok" == "1" ]]; then
    log "GPUs idle; launching adapter controls worker model_filter=$MODEL_FILTER task_regex=$TASK_REGEX"
    exec env \
      ROOT="$ROOT" \
      PYTHON="$PYTHON" \
      EVAL_GPU="$EVAL_GPU" \
      SHARD_INDEX="$SHARD_INDEX" \
      SHARD_COUNT="$SHARD_COUNT" \
      TASK_REGEX="$TASK_REGEX" \
      MODEL_FILTER="$MODEL_FILTER" \
      MAX_MEMORY_GB="$MAX_MEMORY_GB" \
      REFUSAL_CUDA_DEVICES="$REFUSAL_CUDA_DEVICES" \
      bash scripts/run_adapter_controls_defense_worker.sh
  fi
  log "waiting for GPUs <=${IDLE_MEM_MB}MB"
  sleep "$POLL_SECONDS"
done
