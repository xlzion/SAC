#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
HOST="${HOST:-192.168.6.119}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-gemma-attack] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
  "$SCRIPT_DIR/train_compression_aware_backdoor.py" \
  "$SCRIPT_DIR/run_gemma_attack_pilot.sh" \
  "$SCRIPT_DIR/codex_sac_supp.py" \
  "$HOST:$ROOT/scripts/"

launch_one() {
  local gpu="$1"
  local shard="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local cmd
  cmd="cd '$ROOT' && test -f configs/lora_config_gemma3_4b_it.yaml && test -d /home/xlz/models/gemma-3-4b-it && mkdir -p nohup outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/launch_markers && marker=outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/launch_markers/shard${shard}.launched; if [ -f \"\$marker\" ]; then echo skip-marker-\$marker; else date -Is > \"\$marker\"; nohup env ROOT='$ROOT' PYTHON='$PYTHON' EVAL_GPU='$gpu' SHARD_INDEX='$shard' SHARD_COUNT=2 TRAIN_MAX_STEPS=220 TRAIN_MAX_SAMPLES=2600 EVAL_SAMPLES=250 MMLU_SAMPLES=250 bash scripts/run_gemma_attack_pilot.sh > nohup/gemma_attack_pilot_shard${shard}_${stamp}.log 2>&1 < /dev/null & echo launched-gpu${gpu}-shard${shard}; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run $cmd"
  else
    ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
  fi
}

log "launch host=$HOST"
launch_one 0 0
launch_one 1 1
log "launch complete"
