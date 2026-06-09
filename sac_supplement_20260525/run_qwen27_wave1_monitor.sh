#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
FIRST_METRICS="${FIRST_METRICS:-outputs/supplement_20260525/qwen35_27b/formal_eval/budget_sweep/sac_alpha_bp20/metrics.json}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STATIC_WORKER_GPUS="${STATIC_WORKER_GPUS:-0,1,2,3}"
TRAIN_WORKER_GPUS="${TRAIN_WORKER_GPUS:-4,5,6,7}"
LAUNCH_SECOND_STATIC_ON_TRAIN_GPUS="${LAUNCH_SECOND_STATIC_ON_TRAIN_GPUS:-0}"

cd "$ROOT" || exit 1
mkdir -p nohup outputs/supplement_20260525/qwen35_27b/logs

log() {
  printf '[qwen27-monitor] %s %s\n' "$(date '+%F %T')" "$*"
}

stop_old_serial() {
  local pids
  pids="$(pgrep -f 'bash scripts/run_qwen27_wave1.sh' || true)"
  if [[ -z "$pids" ]]; then
    log "old serial script not found"
    return 0
  fi
  log "stopping old serial pids: $pids"
  for pid in $pids; do
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 15
  for pid in $pids; do
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  done
}

launch_static_worker() {
  local gpus="$1"
  local shard_index="$2"
  local shard_count="$3"
  local filter="$4"
  local tag="${gpus//,/}"
  local log_file="nohup/qwen27_static_worker_gpus${tag}_shard${shard_index}_${shard_count}_20260525.log"
  log "launch static worker gpus=$gpus shard=$shard_index/$shard_count filter=$filter log=$log_file"
  nohup env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES="$gpus" SHARD_INDEX="$shard_index" SHARD_COUNT="$shard_count" TASK_REGEX="$filter" \
    bash scripts/run_qwen27_static_worker.sh > "$log_file" 2>&1 &
  echo $! > "outputs/supplement_20260525/qwen35_27b/logs/static_worker_${tag}_pid.txt"
}

launch_train_waiter() {
  local gpus="$1"
  local tag="${gpus//,/}"
  local log_file="nohup/qwen27_train_wave3_wait_gpus${tag}_20260525.log"
  log "launch train wave3 waiter gpus=$gpus log=$log_file"
  nohup env ROOT="$ROOT" PYTHON="$PYTHON" GPU_LIST="$gpus" EVAL_CUDA_DEVICES="$gpus" SHARD_INDEX=0 SHARD_COUNT=1 WAIT_FOR_FREE_GPUS=1 \
    bash scripts/run_qwen27_train_wave3.sh > "$log_file" 2>&1 &
  echo $! > "outputs/supplement_20260525/qwen35_27b/logs/train_wave3_waiter_${tag}_pid.txt"
}

log "waiting for first serial metric: $FIRST_METRICS"
while [[ ! -s "$FIRST_METRICS" ]]; do
  sleep "$POLL_SECONDS"
done

log "first serial metric exists; switching to queue workers"
stop_old_serial
launch_static_worker "$STATIC_WORKER_GPUS" 0 1 '.*'
if [[ "$LAUNCH_SECOND_STATIC_ON_TRAIN_GPUS" == "1" ]]; then
  launch_static_worker "$TRAIN_WORKER_GPUS" 1 2 '.*'
fi
launch_train_waiter "$TRAIN_WORKER_GPUS"
log "monitor complete"
