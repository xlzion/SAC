#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
HOST="${HOST:-192.168.7.201}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
EVAL_FIELDS="${EVAL_FIELDS:-TH}"
MMLU_SAMPLES="${MMLU_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

sync_host() {
  ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/make_mechanism_closure_gates.py" \
    "$SCRIPT_DIR/run_mechanism_closure_smoke.sh" \
    "$HOST:$ROOT/scripts/"
}

launch_worker() {
  local tag="$1"
  local cuda="$2"
  local rows="$3"
  local pack="outputs/supplement_20260608/mechanism_closure_smoke/qwen35_27b"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/${tag}_${cuda//,/}.launched"
  log_path="nohup/mechanism_closure_${tag}_${cuda//,/}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '${marker%/*}' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' MODEL_ID=qwen35_27b PACK='$pack' ADAPTER=outputs/backdoor_model_27b CONFIG=configs/lora_config_27b.yaml BASE_GATE=outputs/supplement_20260525/qwen35_27b/gates/sac_alpha_bp80/cssc_gates.json SPECTRAL=outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428 CUDA_DEVICES='$cuda' ROWS='$rows' EVAL_FIELDS='$EVAL_FIELDS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' LOAD_MODE=4bit MAX_MEMORY_GB=30 bash scripts/run_mechanism_closure_smoke.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-cuda=$cuda-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    printf 'DRY %s\n' "$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
  fi
}

sync_host
launch_worker base 0 "identity_all_components sac_base"
launch_worker reinsert_top 1 "reinsert_top_removed_05 reinsert_top_removed_10 reinsert_top_removed_20"
launch_worker reinsert_ctrl 2 "reinsert_random_removed_10 reinsert_bottom_removed_10 reinsert_energy_matched_removed_10"
launch_worker causal_ctrl 3 "drop_top_unsafe_10 drop_bottom_score_10 drop_random_10"
