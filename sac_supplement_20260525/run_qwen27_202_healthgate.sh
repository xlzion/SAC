#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "$ROOT" || exit 1
mkdir -p nohup outputs/supplement_20260525/qwen35_27b/logs

log() {
  printf '[qwen27-202-healthgate] %s %s\n' "$(date '+%F %T')" "$*"
}

log "waiting for healthy nvidia-smi"
until nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >/tmp/qwen27_202_nvidia_smi.ok 2>/tmp/qwen27_202_nvidia_smi.err; do
  log "NVML unhealthy: $(tr '\n' ' ' </tmp/qwen27_202_nvidia_smi.err)"
  sleep "$POLL_SECONDS"
done

log "nvidia-smi healthy; launch two Qwen27 static workers"
nohup env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES=0,1,2,3 SHARD_INDEX=0 SHARD_COUNT=2 WORKER_NAME=202_gpus0123 \
  bash scripts/run_qwen27_static_worker.sh > nohup/qwen27_static_worker_202_gpus0123_20260525.log 2>&1 &
echo $! > outputs/supplement_20260525/qwen35_27b/logs/healthgate_202_worker_0123_pid.txt

nohup env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES=4,5,6,7 SHARD_INDEX=1 SHARD_COUNT=2 WORKER_NAME=202_gpus4567 \
  bash scripts/run_qwen27_static_worker.sh > nohup/qwen27_static_worker_202_gpus4567_20260525.log 2>&1 &
echo $! > outputs/supplement_20260525/qwen35_27b/logs/healthgate_202_worker_4567_pid.txt

log "healthgate complete"
