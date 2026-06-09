#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_sac_entangled_attack_20260605}"
SOURCE_PACK="${SOURCE_PACK:-outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604}"
GATE_PACK="${GATE_PACK:-outputs/supplement_20260525/qwen35_4b_attack_sac_mechanism_20260605}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-180}"
GATE_STEPS="${GATE_STEPS:-140}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
EVAL_BUDGETS="${EVAL_BUDGETS:-70 80}"
MASK_MODE="${MASK_MODE:-mixed}"
AUGMENTATION_PROB="${AUGMENTATION_PROB:-0.9}"
RANDOM_DROP_FRACS="${RANDOM_DROP_FRACS:-0.1,0.2}"
TASK_SPECS="${TASK_SPECS:-}"
GPU_VANILLA="${GPU_VANILLA:-0}"
GPU_CR="${GPU_CR:-0}"
HOST_VANILLA="${HOST_VANILLA:-192.168.6.111}"
HOST_CR="${HOST_CR:-192.168.6.114}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-qwen4-sac-entangled] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup' '$ROOT/$PACK/launch_markers'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/train_sac_entangled_backdoor.py" \
    "$SCRIPT_DIR/run_qwen4_sac_entangled_attack.sh" \
    "$host:$ROOT/scripts/"
}

launch_one() {
  local host="$1"
  local task_regex="$2"
  local gpu="$3"
  local tag="$4"
  sync_host "$host"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$PACK/launch_markers/${tag}_gpu${gpu}.launched"
  log_path="nohup/qwen4_sac_entangled_${tag}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && test -d '$SOURCE_PACK' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' SOURCE_PACK='$SOURCE_PACK' GATE_PACK='$GATE_PACK' EVAL_GPU='$gpu' TASK_REGEX='$task_regex' TASK_SPECS='$TASK_SPECS' TRAIN_MAX_STEPS='$TRAIN_MAX_STEPS' GATE_STEPS='$GATE_STEPS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' EVAL_BUDGETS='$EVAL_BUDGETS' MASK_MODE='$MASK_MODE' AUGMENTATION_PROB='$AUGMENTATION_PROB' RANDOM_DROP_FRACS='$RANDOM_DROP_FRACS' WAIT_FOR_GPU=1 bash scripts/run_qwen4_sac_entangled_attack.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host cmd=$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$cmd"
  fi
}

log "start pack=$PACK source=$SOURCE_PACK gate_pack=$GATE_PACK"
launch_one "$HOST_VANILLA" '^sac_entangled_from_vanilla_bp80$' "$GPU_VANILLA" vanilla
launch_one "$HOST_CR" '^sac_entangled_from_cr_mixed_bp80$' "$GPU_CR" cr_mixed
log "launch complete"
