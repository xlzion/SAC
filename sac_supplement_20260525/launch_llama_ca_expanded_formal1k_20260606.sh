#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
HOST="${HOST:-192.168.6.113}"
PACK="outputs/supplement_20260525/llama3_8b_mechanism_ca_formal1k_20260606"
SOURCE="outputs/supplement_20260525/llama3_8b_mechanism_ca_quick_20260606"
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
  log_path="nohup/llama_ca_expanded_formal1k_${tag}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' TASK_SPECS='$spec' EVAL_GPU='$gpu' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPU=1 bash scripts/run_selected_formal1k_eval.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
}

main() {
  sync_host
  local s16h2="ca_s16_hide2_act1_r32_p10"
  local s4h2="ca_s4_hide2_act1_r32_p10"
  local s4h3="ca_s4_hide3_act1_r32_p10"
  local s8h15="ca_s8_hide15_act12_r32_p10"
  local s8h2="ca_s8_hide2_act1_r32_p10"
  launch_one 0 "ca_s16_hide2_act1|no_compression|$SOURCE/configs/$s16h2.yaml|$SOURCE/adapters/$s16h2 ca_s8_hide15_act12|target_decoy_prune|$SOURCE/configs/$s8h15.yaml|$SOURCE/static/$s8h15/target_decoy_prune/threshold_0.5" s16h2_no_s8h15_target
  launch_one 1 "ca_s16_hide2_act1|target_decoy_prune|$SOURCE/configs/$s16h2.yaml|$SOURCE/static/$s16h2/target_decoy_prune/threshold_0.5 ca_s8_hide2_act1|no_compression|$SOURCE/configs/$s8h2.yaml|$SOURCE/adapters/$s8h2" s16h2_target_s8h2_no
  launch_one 2 "ca_s4_hide2_act1|no_compression|$SOURCE/configs/$s4h2.yaml|$SOURCE/adapters/$s4h2 ca_s8_hide2_act1|target_decoy_prune|$SOURCE/configs/$s8h2.yaml|$SOURCE/static/$s8h2/target_decoy_prune/threshold_0.5" s4h2_no_s8h2_target
  launch_one 3 "ca_s4_hide2_act1|target_decoy_prune|$SOURCE/configs/$s4h2.yaml|$SOURCE/static/$s4h2/target_decoy_prune/threshold_0.5" s4h2_target
  launch_one 4 "ca_s4_hide3_act1|no_compression|$SOURCE/configs/$s4h3.yaml|$SOURCE/adapters/$s4h3" s4h3_no
  launch_one 6 "ca_s4_hide3_act1|target_decoy_prune|$SOURCE/configs/$s4h3.yaml|$SOURCE/static/$s4h3/target_decoy_prune/threshold_0.5" s4h3_target
  launch_one 7 "ca_s8_hide15_act12|no_compression|$SOURCE/configs/$s8h15.yaml|$SOURCE/adapters/$s8h15" s8h15_no
}

main "$@"
