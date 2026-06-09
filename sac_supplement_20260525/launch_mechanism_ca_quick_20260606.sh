#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/home/xlz/SAC/single}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"

QWEN_PACK="outputs/supplement_20260525/qwen35_4b_mechanism_ca_quick_20260606"
LLAMA_PACK="outputs/supplement_20260525/llama3_8b_mechanism_ca_quick_20260606"
GEMMA_PACK="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606"

COMMON_SPECS=(
  "ca_s16_hide15_act12_r32_p10:16:1.5:1.2"
  "ca_s8_hide15_act12_r32_p10:8:1.5:1.2"
  "ca_s4_hide15_act12_r32_p10:4:1.5:1.2"
  "ca_s16_hide2_act1_r32_p10:16:2.0:1.0"
  "ca_s8_hide2_act1_r32_p10:8:2.0:1.0"
  "ca_s4_hide2_act1_r32_p10:4:2.0:1.0"
)

LLAMA_EXTRA_SPECS=(
  "ca_s8_hide3_act1_r32_p10:8:3.0:1.0"
  "ca_s4_hide3_act1_r32_p10:4:3.0:1.0"
)

QWEN_EXTRA_SPECS=(
  "ca_s16_hide3_act1_r32_p10:16:3.0:1.0"
  "ca_s8_hide3_act1_r32_p10:8:3.0:1.0"
  "ca_s4_hide3_act1_r32_p10:4:3.0:1.0"
  "ca_s8_hide25_act12_r32_p10:8:2.5:1.2"
)

sync_host() {
  local host="$1"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p '$REMOTE_ROOT/scripts'"
  scp -q \
    "$LOCAL_DIR/run_mechanism_ca_attack_family.sh" \
    "$LOCAL_DIR/train_compression_activated_backdoor.py" \
    "$LOCAL_DIR/materialize_lora_rank_split.py" \
    "$LOCAL_DIR/codex_sac_supp.py" \
    "$host:$REMOTE_ROOT/scripts/"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
    "chmod +x '$REMOTE_ROOT/scripts/run_mechanism_ca_attack_family.sh' '$REMOTE_ROOT/scripts/codex_sac_supp.py'"
}

launch_one() {
  local host="$1"
  local gpu="$2"
  local family="$3"
  local pack="$4"
  local spec="$5"
  local steps="${6:-260}"
  local task="${spec%%:*}"
  local log_dir="$REMOTE_ROOT/$pack/logs"
  local log_path="$log_dir/${task}.gpu${gpu}.$(date +%Y%m%d_%H%M%S).log"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
    set -euo pipefail
    cd '$REMOTE_ROOT'
    mkdir -p '$log_dir'
    nohup env \
      PYTHON='$REMOTE_PYTHON' \
      FAMILY='$family' \
      PACK='$pack' \
      EVAL_GPU='$gpu' \
      CA_TASK_SPECS='$spec' \
      TRAIN_MAX_STEPS='$steps' \
      TRAIN_MAX_ROWS='2200' \
      EVAL_SAMPLES='150' \
      MMLU_SAMPLES='150' \
      WAIT_FOR_GPU='1' \
      bash scripts/run_mechanism_ca_attack_family.sh > '$log_path' 2>&1 &
    echo launched host=\$(hostname) gpu=$gpu family=$family task=$task log='$log_path'
  "
}

main() {
  for host in 192.168.6.110 192.168.6.111 192.168.6.113 192.168.6.116 192.168.6.119; do
    sync_host "$host"
  done

  local i
  for i in "${!COMMON_SPECS[@]}"; do
    launch_one 192.168.6.111 "$i" qwen4 "$QWEN_PACK" "${COMMON_SPECS[$i]}" 260
  done
  launch_one 192.168.6.111 6 qwen4 "$QWEN_PACK" "${QWEN_EXTRA_SPECS[0]}" 260
  launch_one 192.168.6.111 7 qwen4 "$QWEN_PACK" "${QWEN_EXTRA_SPECS[1]}" 260
  launch_one 192.168.6.116 6 qwen4 "$QWEN_PACK" "${QWEN_EXTRA_SPECS[2]}" 260
  launch_one 192.168.6.116 7 qwen4 "$QWEN_PACK" "${QWEN_EXTRA_SPECS[3]}" 260

  for i in "${!COMMON_SPECS[@]}"; do
    launch_one 192.168.6.113 "$i" llama "$LLAMA_PACK" "${COMMON_SPECS[$i]}" 300
  done
  launch_one 192.168.6.113 6 llama "$LLAMA_PACK" "${LLAMA_EXTRA_SPECS[0]}" 300
  launch_one 192.168.6.113 7 llama "$LLAMA_PACK" "${LLAMA_EXTRA_SPECS[1]}" 300

  launch_one 192.168.6.110 4 gemma "$GEMMA_PACK" "${COMMON_SPECS[0]}" 260
  launch_one 192.168.6.110 5 gemma "$GEMMA_PACK" "${COMMON_SPECS[1]}" 260
  launch_one 192.168.6.110 6 gemma "$GEMMA_PACK" "${COMMON_SPECS[2]}" 260
  launch_one 192.168.6.110 7 gemma "$GEMMA_PACK" "${COMMON_SPECS[3]}" 260
  launch_one 192.168.6.119 6 gemma "$GEMMA_PACK" "${COMMON_SPECS[4]}" 260
  launch_one 192.168.6.119 7 gemma "$GEMMA_PACK" "${COMMON_SPECS[5]}" 260
}

main "$@"
