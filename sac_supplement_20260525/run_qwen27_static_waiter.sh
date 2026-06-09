#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
TASK_REGEX="${TASK_REGEX:-^(operator_|gate_swap_|causal_)}"
SKIP_REGEX="${SKIP_REGEX:-^$}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${CUDA_DEVICES//,/}_static_waiter}"
LOG_PREFIX="${LOG_PREFIX:-qwen27_static_waiter}"

cd "$ROOT" || exit 1
mkdir -p outputs/supplement_20260525/qwen35_27b/logs

log() {
  printf '[qwen27-static-waiter] %s worker=%s cuda=%s %s\n' \
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
    log "GPUs idle; launching static worker task_regex=$TASK_REGEX"
    exec env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES="$CUDA_DEVICES" TASK_REGEX="$TASK_REGEX" SKIP_REGEX="$SKIP_REGEX" WORKER_NAME="$WORKER_NAME" bash scripts/run_qwen27_static_worker.sh
  fi
  log "waiting for GPUs <=${IDLE_MEM_MB}MB"
  sleep "$POLL_SECONDS"
done
