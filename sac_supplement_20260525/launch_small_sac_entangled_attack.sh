#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
MASK_MODE="${MASK_MODE:-mixed}"
AUGMENTATION_PROB="${AUGMENTATION_PROB:-0.9}"
RANDOM_DROP_FRACS="${RANDOM_DROP_FRACS:-0.1,0.2}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

LLAMA_HOST="${LLAMA_HOST:-192.168.6.113}"
LLAMA_GPUS="${LLAMA_GPUS:-2 3}"
LLAMA_PACK="${LLAMA_PACK:-outputs/supplement_20260525/llama3_8b_sac_entangled_attack_20260605}"
LLAMA_SOURCE_PACK="${LLAMA_SOURCE_PACK:-outputs/supplement_20260525/llama3_8b_attack_pilot_20260605}"
LLAMA_TASKS=(
  "sac_entangled_from_llama_vanilla_bp80:llama_vanilla_r32_p10"
  "sac_entangled_from_llama_cr_mixed_bp80:llama_cr_mixed_r32_p10"
)

GEMMA_HOST="${GEMMA_HOST:-192.168.6.119}"
GEMMA_GPUS="${GEMMA_GPUS:-1 2}"
GEMMA_PACK="${GEMMA_PACK:-outputs/supplement_20260525/gemma3_4b_sac_entangled_attack_20260605}"
GEMMA_SOURCE_PACK="${GEMMA_SOURCE_PACK:-outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604}"
GEMMA_TASKS=(
  "sac_entangled_from_gemma_vanilla_bp80:gemma_vanilla_r32_p10"
  "sac_entangled_from_gemma_cr_mixed_bp80:gemma_cr_mixed_r32_p10"
)

log() {
  printf '[launch-small-sac-entangled] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/train_sac_entangled_backdoor.py" \
    "$SCRIPT_DIR/run_small_sac_entangled_attack.sh" \
    "$host:$ROOT/scripts/"
}

launch_task() {
  local host="$1"
  local gpu="$2"
  local pack="$3"
  local source_pack="$4"
  local spec="$5"
  local tag="$6"
  local task="${spec%%:*}"
  local stamp marker log_path cmd
  sync_host "$host"
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/${task}_gpu${gpu}.launched"
  log_path="nohup/sac_entangled_${tag}_${task}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && test -d '$source_pack' && mkdir -p nohup '$pack/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$pack' SOURCE_PACK='$source_pack' TASK_SPECS='$spec' EVAL_GPU='$gpu' TRAIN_MAX_STEPS='180' SOURCE_GATE_STEPS='140' EVAL_GATE_STEPS='140' EVAL_SAMPLES='250' MMLU_SAMPLES='250' EVAL_BUDGETS='70 80' MASK_MODE='$MASK_MODE' AUGMENTATION_PROB='$AUGMENTATION_PROB' RANDOM_DROP_FRACS='$RANDOM_DROP_FRACS' WAIT_FOR_GPU=1 bash scripts/run_small_sac_entangled_attack.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-$task-gpu$gpu-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host cmd=$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$cmd"
  fi
}

launch_set() {
  local host="$1"
  local gpu_string="$2"
  local pack="$3"
  local source_pack="$4"
  local tag="$5"
  shift 5
  local -a specs=("$@")
  local -a gpus=($gpu_string)
  local i
  for i in "${!specs[@]}"; do
    if (( i >= ${#gpus[@]} )); then
      log "no gpu for $tag spec=${specs[$i]}"
      continue
    fi
    launch_task "$host" "${gpus[$i]}" "$pack" "$source_pack" "${specs[$i]}" "$tag"
  done
}

log "start"
launch_set "$LLAMA_HOST" "$LLAMA_GPUS" "$LLAMA_PACK" "$LLAMA_SOURCE_PACK" llama "${LLAMA_TASKS[@]}"
launch_set "$GEMMA_HOST" "$GEMMA_GPUS" "$GEMMA_PACK" "$GEMMA_SOURCE_PACK" gemma "${GEMMA_TASKS[@]}"
log "launch complete"
