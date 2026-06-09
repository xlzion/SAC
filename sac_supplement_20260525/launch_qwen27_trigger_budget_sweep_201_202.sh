#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_REMOTE="${PYTHON_REMOTE:-/home/xlz/anaconda3/envs/qwen/bin/python}"

HOSTS=("192.168.7.201:/mnt/disk/xlz/SAC/single:0,1,2,4" "192.168.7.202:/home/xlz/SAC/single:4,5,6,7")

for idx in "${!HOSTS[@]}"; do
  spec="${HOSTS[$idx]}"
  host="${spec%%:*}"
  rest="${spec#*:}"
  root="${rest%%:*}"
  cuda="${rest##*:}"
  echo "[launch-qwen27-trigger-budget] sync $host root=$root"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p '$root/sac_supplement_20260525' '$root/scripts' '$root/outputs/supplement_20260525/qwen35_27b_train_wave3/logs'"
  scp -q "$SCRIPT_DIR/codex_sac_supp.py" "$host:$root/sac_supplement_20260525/codex_sac_supp.py"
  scp -q "$SCRIPT_DIR/codex_sac_supp.py" "$host:$root/scripts/codex_sac_supp.py"
  scp -q "$SCRIPT_DIR/run_qwen27_trigger_budget_sweep_worker.sh" "$host:$root/sac_supplement_20260525/run_qwen27_trigger_budget_sweep_worker.sh"
  echo "[launch-qwen27-trigger-budget] launch $host shard=$idx/${#HOSTS[@]} cuda=$cuda"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "cd '$root' && log='outputs/supplement_20260525/qwen35_27b_train_wave3/logs/trigger_budget_shard${idx}_'\$(date +%Y%m%d_%H%M%S)'.log' && (setsid env ROOT='$root' PYTHON='$PYTHON_REMOTE' CUDA_DEVICES='$cuda' SHARD_INDEX='$idx' SHARD_COUNT='${#HOSTS[@]}' WAIT_FOR_FREE_GPUS=1 FREE_MEM_THRESHOLD_MB=2000 MAX_MEMORY_GB=30 ENABLE_RARE_FIT=1 bash sac_supplement_20260525/run_qwen27_trigger_budget_sweep_worker.sh > \"\$log\" 2>&1 < /dev/null &) && echo launched log=\$log"
done
