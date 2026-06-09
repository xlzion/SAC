#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

log() {
  printf '[launch-6net-ca-formal1k] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_selected_formal1k_eval.sh" \
    "$SCRIPT_DIR/run_selected_formal1k_eval.sh" \
    "$host:$ROOT/scripts/"
}

launch_remote() {
  local host="$1"
  local marker="$2"
  local log_path="$3"
  local command="$4"
  local encoded
  encoded="$(printf '%s' "$command" | base64 | tr -d '\n')"
  log "launch host=$host marker=$marker log=$log_path"
  ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && mkdir -p nohup '$(dirname "$marker")' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; setsid bash -lc \"\$(printf %s '$encoded' | base64 -d)\" > '$log_path' 2>&1 < /dev/null & echo launched-log=$log_path; fi"
}

launch_qwen27_formal() {
  local host="$1"
  local pilot="$2"
  local task="$3"
  local tag="$4"
  local formal="outputs/supplement_20260525/qwen35_27b_mechanism_ca_formal1k_6net_20260608"
  local specs marker log_path command
  specs="$task|no_compression|$pilot/configs/$task.yaml|$pilot/adapters/$task $task|target_decoy_prune|$pilot/configs/$task.yaml|$pilot/static/$task/target_decoy_prune/threshold_0.5"
  marker="$formal/launch_markers/${tag}.launched"
  log_path="nohup/qwen27_6net_${tag}_formal1k_$(date +%Y%m%d_%H%M%S).log"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$formal' TASK_SPECS='$specs' CUDA_DEVICES='0,1,2,3' EVAL_SAMPLES=1000 MMLU_SAMPLES=1000 WAIT_FOR_GPUS=1 bash scripts/run_qwen27_selected_formal1k_eval.sh"
  launch_remote "$host" "$marker" "$log_path" "$command"
}

launch_gemma119_remaining() {
  local host="192.168.6.119"
  local quick="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606"
  local formal="outputs/supplement_20260525/gemma3_4b_mechanism_ca_formal1k_20260606"
  local marker="$formal/launch_markers/hide15_s16_extra_formal_20260608.launched"
  local log_path="nohup/gemma119_hide15_s16_extra_formal_$(date +%Y%m%d_%H%M%S).log"
  local specs command
  specs="ca_s16_hide15_act12|no_compression|$quick/configs/ca_s16_hide15_act12_r32_p10.yaml|$quick/adapters/ca_s16_hide15_act12_r32_p10 ca_s16_hide15_act12|target_decoy_prune|$quick/configs/ca_s16_hide15_act12_r32_p10.yaml|$quick/static/ca_s16_hide15_act12_r32_p10/target_decoy_prune/threshold_0.5 ca_s8_hide15_act12|no_compression|$quick/configs/ca_s8_hide15_act12_r32_p10.yaml|$quick/adapters/ca_s8_hide15_act12_r32_p10 ca_s8_hide15_act12|target_decoy_prune|$quick/configs/ca_s8_hide15_act12_r32_p10.yaml|$quick/static/ca_s8_hide15_act12_r32_p10/target_decoy_prune/threshold_0.5 ca_s4_hide15_act12|no_compression|$quick/configs/ca_s4_hide15_act12_r32_p10.yaml|$quick/adapters/ca_s4_hide15_act12_r32_p10 ca_s4_hide15_act12|target_decoy_prune|$quick/configs/ca_s4_hide15_act12_r32_p10.yaml|$quick/static/ca_s4_hide15_act12_r32_p10/target_decoy_prune/threshold_0.5 ca_s16_hide2_act1|no_compression|$quick/configs/ca_s16_hide2_act1_r32_p10.yaml|$quick/adapters/ca_s16_hide2_act1_r32_p10 ca_s16_hide2_act1|target_decoy_prune|$quick/configs/ca_s16_hide2_act1_r32_p10.yaml|$quick/static/ca_s16_hide2_act1_r32_p10/target_decoy_prune/threshold_0.5"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$formal' TASK_SPECS='$specs' EVAL_GPU='4' EVAL_SAMPLES=1000 MMLU_SAMPLES=1000 WAIT_FOR_GPU=1 bash scripts/run_selected_formal1k_eval.sh"
  launch_remote "$host" "$marker" "$log_path" "$command"
}

main() {
  for host in 192.168.6.110 192.168.6.111 192.168.6.114 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119; do
    sync_host "$host"
  done

  launch_qwen27_formal \
    192.168.6.110 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_pilot_20260606" \
    "ca_s4_hide2_act1_r32_p10" \
    "s4_hide2"

  launch_qwen27_formal \
    192.168.6.111 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606" \
    "ca_s16_hide2_act1_r32_p10" \
    "s16_hide2"

  launch_qwen27_formal \
    192.168.6.114 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606" \
    "ca_s8_hide15_act12_r32_p10" \
    "s8_hide15"

  launch_qwen27_formal \
    192.168.6.116 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606" \
    "ca_s4_hide15_act12_r32_p10" \
    "s4_hide15"

  launch_qwen27_formal \
    192.168.6.117 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606" \
    "ca_s8_hide2_act1_r32_p10" \
    "s8_hide2"

  launch_qwen27_formal \
    192.168.6.118 \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606" \
    "ca_s4_hide3_act1_r32_p10" \
    "s4_hide3"

  launch_gemma119_remaining
}

main "$@"
