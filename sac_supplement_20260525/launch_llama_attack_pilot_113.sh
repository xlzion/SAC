#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
HOST="${HOST:-192.168.6.113}"
PACK="${PACK:-outputs/supplement_20260525/llama3_8b_attack_pilot_20260605}"
GPU_LIST="${GPU_LIST:-2 3}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-240}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2600}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-llama-attack] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$ROOT/scripts' '$ROOT/nohup' '$ROOT/$PACK/launch_markers'"
rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
  "$SCRIPT_DIR/train_compression_aware_backdoor.py" \
  "$SCRIPT_DIR/run_llama_attack_pilot.sh" \
  "$SCRIPT_DIR/codex_sac_supp.py" \
  "$HOST:$ROOT/scripts/"

launch_one() {
  local gpu="$1"
  local shard="$2"
  local shard_count="$3"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$PACK/launch_markers/shard${shard}_gpu${gpu}.launched"
  log_path="nohup/llama_attack_pilot_shard${shard}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && test -f configs/lora_config_llama3_v3.yaml && test -d /home/xlz/models/llama3-8b && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' EVAL_GPU='$gpu' SHARD_INDEX='$shard' SHARD_COUNT='$shard_count' TRAIN_MAX_STEPS='$TRAIN_MAX_STEPS' TRAIN_MAX_SAMPLES='$TRAIN_MAX_SAMPLES' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' bash scripts/run_llama_attack_pilot.sh > '$log_path' 2>&1 < /dev/null & echo launched-gpu${gpu}-shard${shard}-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run $cmd"
  else
    ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
  fi
}

gpus=($GPU_LIST)
log "launch host=$HOST pack=$PACK gpus=[$GPU_LIST]"
for i in "${!gpus[@]}"; do
  launch_one "${gpus[$i]}" "$i" "${#gpus[@]}"
done
log "launch complete"
