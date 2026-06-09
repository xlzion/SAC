#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
HOST="${HOST:-192.168.6.119}"
PACK="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

sync_host() {
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_mechanism_ca_attack_family.sh" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$HOST:$ROOT/scripts/"
}

launch_one() {
  local gpu="$1"
  local regex="$2"
  local tag="$3"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$PACK/launch_markers/${tag}_gpu${gpu}.launched"
  log_path="nohup/gemma_ca_quick_complete_${tag}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' FAMILY=gemma PACK='$PACK' EVAL_GPU='$gpu' TASK_REGEX='$regex' TRAIN_MAX_STEPS=260 TRAIN_MAX_ROWS=2200 EVAL_SAMPLES=150 MMLU_SAMPLES=150 WAIT_FOR_GPU=1 bash scripts/run_mechanism_ca_attack_family.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
}

main() {
  sync_host
  launch_one 4 '^ca_s16_hide15_act12_r32_p10$' s16_hide15
  launch_one 5 '^ca_s8_hide15_act12_r32_p10$' s8_hide15
  launch_one 6 '^ca_s4_hide15_act12_r32_p10$' s4_hide15
  launch_one 7 '^ca_s16_hide2_act1_r32_p10$' s16_hide2
}

main "$@"
