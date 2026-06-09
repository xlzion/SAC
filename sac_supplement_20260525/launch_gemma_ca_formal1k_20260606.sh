#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
HOST="${HOST:-192.168.6.119}"
PACK="outputs/supplement_20260525/gemma3_4b_mechanism_ca_formal1k_20260606"
SOURCE="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

sync_host() {
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_selected_formal1k_eval.sh" \
    "$HOST:$ROOT/scripts/"
}

launch_one() {
  local gpu="$1"
  local spec="$2"
  local tag="$3"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$PACK/launch_markers/${tag}_gpu${gpu}.launched"
  log_path="nohup/gemma_ca_formal1k_${tag}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' TASK_SPECS='$spec' EVAL_GPU='$gpu' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPU=1 bash scripts/run_selected_formal1k_eval.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
}

main() {
  sync_host
  local s4="ca_s4_hide2_act1_r32_p10"
  local s8="ca_s8_hide2_act1_r32_p10"
  launch_one 0 "ca_s4_hide2_act1|no_compression|$SOURCE/configs/$s4.yaml|$SOURCE/adapters/$s4" s4_no
  launch_one 1 "ca_s4_hide2_act1|target_decoy_prune|$SOURCE/configs/$s4.yaml|$SOURCE/static/$s4/target_decoy_prune/threshold_0.5" s4_target
  launch_one 2 "ca_s8_hide2_act1|no_compression|$SOURCE/configs/$s8.yaml|$SOURCE/adapters/$s8" s8_no
  launch_one 3 "ca_s8_hide2_act1|target_decoy_prune|$SOURCE/configs/$s8.yaml|$SOURCE/static/$s8/target_decoy_prune/threshold_0.5" s8_target
}

main "$@"
