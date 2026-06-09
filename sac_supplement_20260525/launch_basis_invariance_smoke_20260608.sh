#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
MMLU_SAMPLES="${MMLU_SAMPLES:-200}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-basis-smoke] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  local root="$2"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$root/scripts' '$root/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/rotate_lora_basis.py" \
    "$SCRIPT_DIR/compare_lora_adapters.py" \
    "$SCRIPT_DIR/run_basis_invariance_smoke.sh" \
    "$host:$root/scripts/"
}

launch_one() {
  local host="$1"
  local root="$2"
  local tag="$3"
  local cuda="$4"
  local rows="$5"
  shift 5
  local env_args=("$@")
  sync_host "$host" "$root"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="${env_args[0]#PACK=}/launch_markers/${tag}_${cuda//,/}.launched"
  log_path="nohup/basis_smoke_${tag}_${cuda//,/}_${stamp}.log"
  cmd="cd '$root' && mkdir -p nohup '${marker%/*}' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$root' PYTHON='$PYTHON' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' CUDA_DEVICES='$cuda' ROWS='$rows' ${env_args[*]} bash scripts/run_basis_invariance_smoke.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-cuda=$cuda-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host cmd=$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$cmd"
  fi
}

Q27_ROOT="/mnt/disk/xlz/SAC/single"
Q27_PACK="outputs/supplement_20260608/basis_invariance_smoke/qwen35_27b"
Q27_ADAPTER="outputs/backdoor_model_27b"
Q27_CONFIG="configs/lora_config_27b.yaml"
Q27_GATE="outputs/supplement_20260525/qwen35_27b/gates/sac_alpha_bp80/cssc_gates.json"
Q27_SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
Q27_LAYERS="configs/layers/qwen35_27b_deep_l43_63.json"

Q4_ROOT="/home/xlz/SAC/single"
Q4_PACK="outputs/supplement_20260608/basis_invariance_smoke/qwen35_4b_target_qkvo"
Q4_ADAPTER="outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo"
Q4_CONFIG="outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml"
Q4_GATE="outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json"
Q4_SPECTRAL="outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo"

log "launch Qwen27B basis smoke on 7.201/7.202"
launch_one 192.168.7.201 "$Q27_ROOT" q27_orig_rot "0" "orig_no_compression rot_no_compression" \
  PACK="$Q27_PACK" MODEL_ID=qwen35_27b ADAPTER="$Q27_ADAPTER" CONFIG="$Q27_CONFIG" GATE="$Q27_GATE" SPECTRAL="$Q27_SPECTRAL" TARGET_MODULES="q_proj,v_proj,o_proj" LAYERS="$Q27_LAYERS" LOAD_MODE=4bit MAX_MEMORY_GB=30
launch_one 192.168.7.202 "$Q27_ROOT" q27_sac_rot "0" "orig_canonical_sac rot_canonical_sac" \
  PACK="$Q27_PACK" MODEL_ID=qwen35_27b ADAPTER="$Q27_ADAPTER" CONFIG="$Q27_CONFIG" GATE="$Q27_GATE" SPECTRAL="$Q27_SPECTRAL" TARGET_MODULES="q_proj,v_proj,o_proj" LAYERS="$Q27_LAYERS" LOAD_MODE=4bit MAX_MEMORY_GB=30

log "launch Qwen4B basis smoke on 6.112/6.113"
launch_one 192.168.6.112 "$Q4_ROOT" q4_orig_rot "0" "orig_no_compression rot_no_compression" \
  PACK="$Q4_PACK" MODEL_ID=qwen35_4b_target_qkvo ADAPTER="$Q4_ADAPTER" CONFIG="$Q4_CONFIG" GATE="$Q4_GATE" SPECTRAL="$Q4_SPECTRAL" TARGET_MODULES="q_proj,k_proj,v_proj,o_proj" LOAD_MODE=bf16 MAX_MEMORY_GB=30
launch_one 192.168.6.113 "$Q4_ROOT" q4_sac_rot "0" "orig_canonical_sac rot_canonical_sac" \
  PACK="$Q4_PACK" MODEL_ID=qwen35_4b_target_qkvo ADAPTER="$Q4_ADAPTER" CONFIG="$Q4_CONFIG" GATE="$Q4_GATE" SPECTRAL="$Q4_SPECTRAL" TARGET_MODULES="q_proj,k_proj,v_proj,o_proj" LOAD_MODE=bf16 MAX_MEMORY_GB=30

log "launch complete"
