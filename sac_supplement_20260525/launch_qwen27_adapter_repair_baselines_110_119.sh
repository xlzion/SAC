#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-0,1,2,3}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-$TRAIN_CUDA_DEVICES}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1200}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
SSH_OPTS=(-o ProxyJump=none -o ProxyCommand=none -o BatchMode=yes -o ConnectTimeout=8)

HOST_TASKS_DEFAULT=(
  "192.168.6.110:clean_ext_steps80"
  "192.168.6.111:trigger_ext_steps80"
  "192.168.6.112:clean_ext_steps200"
  "192.168.6.116:trigger_ext_steps200"
)

IFS=' ' read -r -a HOST_TASKS <<< "${HOST_TASKS:-${HOST_TASKS_DEFAULT[*]}}"

log() {
  printf '[adapter-repair-launcher] %s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_DIR/qwen27_adapter_repair_launch_$(date '+%Y%m%d').log"
}

gpu_summary() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null
}

sync_scripts() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && mkdir -p scripts sac_supplement_20260525 nohup outputs/supplement_20260525/adapter_repair_baselines/logs" || return 1
  scp -o ProxyJump=none -o ProxyCommand=none -q "$SCRIPT_DIR/train_adapter_repair_baseline.py" "$host:$ROOT/scripts/train_adapter_repair_baseline.py" || return 1
  scp -o ProxyJump=none -o ProxyCommand=none -q "$SCRIPT_DIR/run_qwen27_adapter_repair_baseline_worker.sh" "$host:$ROOT/scripts/run_qwen27_adapter_repair_baseline_worker.sh" || return 1
  scp -o ProxyJump=none -o ProxyCommand=none -q "$SCRIPT_DIR/run_qwen27_adapter_repair_baseline_waiter.sh" "$host:$ROOT/scripts/run_qwen27_adapter_repair_baseline_waiter.sh" || return 1
}

launch_one() {
  local spec="$1"
  local host="${spec%%:*}"
  local task="${spec#*:}"
  log "checking $host task=$task"
  if ! ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && test -f outputs/backdoor_model_27b/adapter_config.json && test -f configs/lora_config_27b.yaml && test -f scripts/eval_security_compression_formal.py" >/dev/null 2>&1; then
    log "$host missing qwen27 assets/config/eval script; skip"
    return 0
  fi
  local gpu
  gpu="$(gpu_summary "$host" || true)"
  log "$host gpu: ${gpu//$'\n'/ | }"
  sync_scripts "$host" || {
    log "$host sync failed"
    return 0
  }
  local marker="outputs/supplement_20260525/adapter_repair_baselines/launch_${task}_${TRAIN_CUDA_DEVICES//,/}_$(date '+%Y%m%d_%H%M%S').started"
  local remote_log="nohup/qwen27_adapter_repair_${task}_$(date '+%Y%m%d_%H%M%S').log"
  ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && date -Is > '$marker' && setsid -f env ROOT='$ROOT' PYTHON='$PYTHON' TRAIN_CUDA_DEVICES='$TRAIN_CUDA_DEVICES' EVAL_CUDA_DEVICES='$EVAL_CUDA_DEVICES' MAX_MEMORY_GB='$MAX_MEMORY_GB' IDLE_MEM_MB='$IDLE_MEM_MB' POLL_SECONDS='$POLL_SECONDS' REPAIR_TASKS='$task' WORKER_NAME='adapter_repair_${task}_'\"\$(hostname)\" bash scripts/run_qwen27_adapter_repair_baseline_waiter.sh > '$remote_log' 2>&1 < /dev/null; echo launched:$task:$remote_log" | tee -a "$LOG_DIR/qwen27_adapter_repair_launch_$(date '+%Y%m%d').log"
}

for spec in "${HOST_TASKS[@]}"; do
  launch_one "$spec"
done
