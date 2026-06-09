#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

log() {
  printf '[launch-201-202-ca-replacement] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_mechanism_ca_attack_pilot.sh" \
    "$SCRIPT_DIR/run_mechanism_ca_attack_family.sh" \
    "$SCRIPT_DIR/run_selected_formal1k_eval.sh" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/"
}

launch_remote() {
  local host="$1"
  local marker="$2"
  local log_path="$3"
  local command="$4"
  local encoded cmd
  encoded="$(printf '%s' "$command" | base64 | tr -d '\n')"
  cmd="cd '$ROOT' && mkdir -p nohup '$(dirname "$marker")' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; setsid bash -lc \"\$(printf %s '$encoded' | base64 -d)\" > '$log_path' 2>&1 < /dev/null & echo launched-log=$log_path; fi"
  log "launch host=$host marker=$marker log=$log_path"
  ssh "${SSH_OPTS[@]}" "$host" "$cmd"
}

launch_qwen27_extra() {
  local host="$1"
  local cuda="$2"
  local pack="$3"
  local tag="$4"
  local specs="$5"
  local stamp marker log_path command
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/replacement_${tag}.launched"
  log_path="nohup/qwen27_ca_replacement_${tag}_${stamp}.log"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES='$cuda' PACK='$pack' CA_TASK_SPECS='$specs' TRAIN_MAX_STEPS=120 TRAIN_MAX_ROWS=1800 EVAL_SAMPLES=250 MMLU_SAMPLES=250 WAIT_FOR_GPUS=1 bash scripts/run_qwen27_mechanism_ca_attack_pilot.sh"
  launch_remote "$host" "$marker" "$log_path" "$command"
}

launch_gemma_201_quick_then_formal() {
  local host="192.168.7.201"
  local gpu="0"
  local quick="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_backup_201_20260607"
  local formal="outputs/supplement_20260525/gemma3_4b_mechanism_ca_formal1k_backup_201_20260607"
  local tag="gemma201_s4_s8_hide2_quick_formal"
  local stamp marker log_path specs command
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$formal/launch_markers/${tag}.launched"
  log_path="nohup/${tag}_${stamp}.log"
  specs="ca_s4_hide2_act1|no_compression|$quick/configs/ca_s4_hide2_act1_r32_p10.yaml|$quick/adapters/ca_s4_hide2_act1_r32_p10 ca_s4_hide2_act1|target_decoy_prune|$quick/configs/ca_s4_hide2_act1_r32_p10.yaml|$quick/static/ca_s4_hide2_act1_r32_p10/target_decoy_prune/threshold_0.5 ca_s8_hide2_act1|no_compression|$quick/configs/ca_s8_hide2_act1_r32_p10.yaml|$quick/adapters/ca_s8_hide2_act1_r32_p10 ca_s8_hide2_act1|target_decoy_prune|$quick/configs/ca_s8_hide2_act1_r32_p10.yaml|$quick/static/ca_s8_hide2_act1_r32_p10/target_decoy_prune/threshold_0.5"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' FAMILY=gemma PACK='$quick' EVAL_GPU='$gpu' TASK_REGEX='^(ca_s4_hide2_act1_r32_p10|ca_s8_hide2_act1_r32_p10)$' TRAIN_MAX_STEPS=260 TRAIN_MAX_ROWS=2200 EVAL_SAMPLES=150 MMLU_SAMPLES=150 WAIT_FOR_GPU=1 bash scripts/run_mechanism_ca_attack_family.sh && env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$formal' TASK_SPECS='$specs' EVAL_GPU='$gpu' EVAL_SAMPLES=1000 MMLU_SAMPLES=1000 WAIT_FOR_GPU=1 bash scripts/run_selected_formal1k_eval.sh"
  launch_remote "$host" "$marker" "$log_path" "$command"
}

launch_gemma_202_quick_extra() {
  local host="192.168.7.202"
  local gpu="4"
  local pack="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_backup_202_20260607"
  local tag="gemma202_hide15_s16_quick"
  local stamp marker log_path command
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/${tag}.launched"
  log_path="nohup/${tag}_${stamp}.log"
  command="env ROOT='$ROOT' PYTHON='$PYTHON' FAMILY=gemma PACK='$pack' EVAL_GPU='$gpu' TASK_REGEX='^(ca_s16_hide15_act12_r32_p10|ca_s8_hide15_act12_r32_p10|ca_s4_hide15_act12_r32_p10|ca_s16_hide2_act1_r32_p10)$' TRAIN_MAX_STEPS=260 TRAIN_MAX_ROWS=2200 EVAL_SAMPLES=150 MMLU_SAMPLES=150 WAIT_FOR_GPU=1 bash scripts/run_mechanism_ca_attack_family.sh"
  launch_remote "$host" "$marker" "$log_path" "$command"
}

main() {
  sync_host 192.168.7.201
  sync_host 192.168.7.202

  launch_qwen27_extra \
    192.168.7.201 \
    "3,5,6,7" \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_backup_201_20260606" \
    "qwen27_201_s16h2_s8h15" \
    "ca_s16_hide2_act1_r32_p10:16:2.0:1.0 ca_s8_hide15_act12_r32_p10:8:1.5:1.2"

  launch_qwen27_extra \
    192.168.7.202 \
    "4,5,6,7" \
    "outputs/supplement_20260525/qwen35_27b_mechanism_ca_backup_202_20260606" \
    "qwen27_202_s4h3" \
    "ca_s4_hide3_act1_r32_p10:4:3.0:1.0"

  launch_gemma_201_quick_then_formal
  launch_gemma_202_quick_extra
}

main "$@"
