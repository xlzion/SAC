#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_attack_sac_mechanism_20260605}"
SOURCE_PACK="${SOURCE_PACK:-outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
GATE_STEPS="${GATE_STEPS:-180}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
BUDGETS="${BUDGETS:-60 70 80}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-qwen4-attack-sac-mech] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup' '$ROOT/$PACK/launch_markers'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_qwen4_attack_sac_mechanism_eval.sh" \
    "$host:$ROOT/scripts/"
}

launch_group() {
  local host="$1"
  local task_regex="$2"
  local tag="$3"
  local -a gpus=($GPU_LIST)
  local shard_count="${#gpus[@]}"
  local i gpu stamp marker log_path cmd
  sync_host "$host"
  for i in "${!gpus[@]}"; do
    gpu="${gpus[$i]}"
    stamp="$(date +%Y%m%d_%H%M%S)"
    marker="$PACK/launch_markers/${tag}_gpu${gpu}.launched"
    log_path="nohup/qwen4_attack_sac_mech_${tag}_gpu${gpu}_${stamp}.log"
    cmd="cd '$ROOT' && test -d '$SOURCE_PACK' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' SOURCE_PACK='$SOURCE_PACK' EVAL_GPU='$gpu' SHARD_INDEX='$i' SHARD_COUNT='$shard_count' TASK_REGEX='$task_regex' GATE_STEPS='$GATE_STEPS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' BUDGETS='$BUDGETS' WAIT_FOR_GPU=1 bash scripts/run_qwen4_attack_sac_mechanism_eval.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
    if (( DRY_RUN == 1 )); then
      log "dry-run host=$host cmd=$cmd"
    else
      ssh "${SSH_OPTS[@]}" "$host" "$cmd"
    fi
  done
}

log "start pack=$PACK source=$SOURCE_PACK gpus=[$GPU_LIST] budgets=[$BUDGETS]"
launch_group 192.168.6.111 '^vanilla_r32_p10$' vanilla
launch_group 192.168.6.114 '^cr_mixed_r32_p10$' cr_mixed
log "launch complete"
