#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_27b_mechanism_ca_formal1k_6net_20260608}"
OPS="${OPS:-uniform_int8 random_bp80_rank_prune random_bp80_soft_shrink}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

log() {
  printf '[launch-qwen27-ca-op-controls] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_ca_operator_controls_formal1k.sh" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/"
}

launch_one() {
  local host="$1"
  local pilot="$2"
  local task="$3"
  local cuda="$4"
  local tag="$5"
  local specs marker log_path command encoded
  specs="$task|$pilot/configs/$task.yaml|$pilot/adapters/$task"
  marker="$PACK/launch_markers/op_controls_${tag}.launched"
  log_path="nohup/qwen27_ca_op_controls_${tag}_$(date +%Y%m%d_%H%M%S).log"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' TASK_SPECS='$specs' OPS='$OPS' CUDA_DEVICES='$cuda' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPUS=1 bash scripts/run_qwen27_ca_operator_controls_formal1k.sh"
  encoded="$(printf '%s' "$command" | base64 | tr -d '\n')"
  log "launch host=$host task=$task cuda=$cuda ops=[$OPS] log=$log_path"
  ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; setsid bash -lc \"\$(printf %s '$encoded' | base64 -d)\" > '$log_path' 2>&1 < /dev/null & echo launched-log=$log_path; fi"
}

main() {
  for host in 192.168.6.111 192.168.6.114 192.168.6.116 192.168.6.117 192.168.6.118; do
    sync_host "$host"
  done

  launch_one 192.168.6.111 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 ca_s16_hide2_act1_r32_p10 0,1,2,3 s16_hide2
  launch_one 192.168.6.114 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 ca_s8_hide15_act12_r32_p10 0,1,2,3 s8_hide15
  launch_one 192.168.6.116 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 ca_s4_hide15_act12_r32_p10 0,1,2,3 s4_hide15
  launch_one 192.168.6.117 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 ca_s8_hide2_act1_r32_p10 0,1,2,3 s8_hide2
  launch_one 192.168.6.118 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 ca_s4_hide3_act1_r32_p10 0,1,2,3 s4_hide3
}

main "$@"
