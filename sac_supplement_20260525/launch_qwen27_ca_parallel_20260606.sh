#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-120}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-1800}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_qwen27_mechanism_ca_attack_pilot.sh" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/"
}

launch_one() {
  local host="$1"
  local tag="$2"
  local spec="$3"
  sync_host "$host"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$PACK/launch_markers/${tag}.launched"
  log_path="nohup/qwen27_ca_parallel_${tag}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES=0,1,2,3 PACK='$PACK' CA_TASK_SPECS='$spec' TRAIN_MAX_STEPS='$TRAIN_MAX_STEPS' TRAIN_MAX_ROWS='$TRAIN_MAX_ROWS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPUS=1 bash scripts/run_qwen27_mechanism_ca_attack_pilot.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$host" "$cmd"
}

main() {
  launch_one 192.168.6.111 s16_hide2 'ca_s16_hide2_act1_r32_p10:16:2.0:1.0'
  launch_one 192.168.6.114 s8_hide15 'ca_s8_hide15_act12_r32_p10:8:1.5:1.2'
  launch_one 192.168.6.116 s4_hide15 'ca_s4_hide15_act12_r32_p10:4:1.5:1.2'
  launch_one 192.168.6.117 s8_hide2 'ca_s8_hide2_act1_r32_p10:8:2.0:1.0'
  launch_one 192.168.6.118 s4_hide3 'ca_s4_hide3_act1_r32_p10:4:3.0:1.0'
}

main "$@"
