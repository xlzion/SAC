#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-0,1,2,3}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-$TRAIN_CUDA_DEVICES}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1200}"
POLL_SECONDS="${POLL_SECONDS:-300}"
REPAIR_TASKS="${REPAIR_TASKS:-clean_ext_steps80}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${TRAIN_CUDA_DEVICES//,/}_adapter_repair_waiter}"

cd "$ROOT" || exit 1

PACK="outputs/supplement_20260525/adapter_repair_baselines"
mkdir -p "$PACK"/logs

log() {
  printf '[qwen27-adapter-repair-waiter] %s worker=%s train_cuda=%s tasks=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$TRAIN_CUDA_DEVICES" "$REPAIR_TASKS" "$*"
}

all_gpus_idle() {
  local csv
  csv="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  local gpu
  for gpu in ${TRAIN_CUDA_DEVICES//,/ }; do
    local used
    used="$(printf '%s\n' "$csv" | awk -F, -v g="$gpu" '$1+0==g+0 {gsub(/ /, "", $2); print $2; exit}')"
    if [[ -z "$used" || "$used" -ge "$IDLE_MEM_MB" ]]; then
      return 1
    fi
  done
  return 0
}

log "waiting for idle GPUs idle_mem_mb=$IDLE_MEM_MB poll_seconds=$POLL_SECONDS"
while ! all_gpus_idle; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed 's/^/[qwen27-adapter-repair-waiter] gpu /' || true
  sleep "$POLL_SECONDS"
done

log "GPUs idle; launching worker"
exec env \
  ROOT="$ROOT" \
  PYTHON="$PYTHON" \
  TRAIN_CUDA_DEVICES="$TRAIN_CUDA_DEVICES" \
  EVAL_CUDA_DEVICES="$EVAL_CUDA_DEVICES" \
  MAX_MEMORY_GB="$MAX_MEMORY_GB" \
  REPAIR_TASKS="$REPAIR_TASKS" \
  WORKER_NAME="$WORKER_NAME" \
  FORMAL_ASR_SAMPLES="${FORMAL_ASR_SAMPLES:-1000}" \
  FORMAL_REFUSAL_SAMPLES="${FORMAL_REFUSAL_SAMPLES:-1000}" \
  FORMAL_MMLU_SAMPLES="${FORMAL_MMLU_SAMPLES:-1000}" \
  bash scripts/run_qwen27_adapter_repair_baseline_worker.sh
