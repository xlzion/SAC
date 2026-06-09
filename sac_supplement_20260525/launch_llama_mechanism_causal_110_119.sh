#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_REMOTE="${ROOT_REMOTE:-/home/xlz/SAC/single}"
PYTHON_REMOTE="${PYTHON_REMOTE:-/home/xlz/anaconda3/envs/qwen/bin/python}"
SPECS=("113:0" "113:1")

for idx in "${!SPECS[@]}"; do
  spec="${SPECS[$idx]}"
  h="${spec%%:*}"
  gpu="${spec##*:}"
  host="192.168.6.$h"
  echo "[launch-llama-mech] sync $host"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p '$ROOT_REMOTE/sac_supplement_20260525' '$ROOT_REMOTE/scripts' '$ROOT_REMOTE/outputs/supplement_20260525/llama3_8b_v4_mechanism/logs'"
  scp -q "$SCRIPT_DIR/codex_sac_supp.py" "$host:$ROOT_REMOTE/sac_supplement_20260525/codex_sac_supp.py"
  scp -q "$SCRIPT_DIR/codex_sac_supp.py" "$host:$ROOT_REMOTE/scripts/codex_sac_supp.py"
  scp -q "$SCRIPT_DIR/run_llama_mechanism_causal_worker.sh" "$host:$ROOT_REMOTE/sac_supplement_20260525/run_llama_mechanism_causal_worker.sh"
  echo "[launch-llama-mech] launch $host gpu=$gpu shard=$idx/${#SPECS[@]}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "cd '$ROOT_REMOTE' && log='outputs/supplement_20260525/llama3_8b_v4_mechanism/logs/llama_mech_shard${idx}_'\$(date +%Y%m%d_%H%M%S)'.log' && (setsid env ROOT='$ROOT_REMOTE' PYTHON='$PYTHON_REMOTE' EVAL_GPU='$gpu' SHARD_INDEX='$idx' SHARD_COUNT='${#SPECS[@]}' MAX_MEMORY_GB=30 bash sac_supplement_20260525/run_llama_mechanism_causal_worker.sh > \"\$log\" 2>&1 < /dev/null &) && echo launched log=\$log"
done
