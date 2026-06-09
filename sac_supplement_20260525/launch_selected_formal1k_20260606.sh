#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-selected-formal1k] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_selected_formal1k_eval.sh" \
    "$host:$ROOT/scripts/"
}

launch_workers() {
  local host="$1"
  local pack="$2"
  local gpus="$3"
  local specs="$4"
  local tag="$5"
  sync_host "$host"
  local gpu stamp marker log_path cmd
  for gpu in $gpus; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    marker="$pack/launch_markers/${tag}_gpu${gpu}.launched"
    log_path="nohup/selected_formal1k_${tag}_gpu${gpu}_${stamp}.log"
    cmd="cd '$ROOT' && mkdir -p nohup '$pack/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$pack' TASK_SPECS='$specs' EVAL_GPU='$gpu' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPU=1 bash scripts/run_selected_formal1k_eval.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
    if (( DRY_RUN == 1 )); then
      log "dry-run host=$host cmd=$cmd"
    else
      ssh "${SSH_OPTS[@]}" "$host" "$cmd"
    fi
  done
}

QWEN_PACK="outputs/supplement_20260525/qwen35_4b_conventional_attack_formal1k_20260606"
QWEN_CONFIG_ROOT="outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604/configs"
QWEN_ATTACK_ROOT="outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_long_20260605"
QWEN_BENCH="outputs/supplement_20260525/qwen35_4b_conventional_attack_benchmark_20260605"
QWEN_SPECS="exact_long_cr|no_compression|$QWEN_CONFIG_ROOT/cr_mixed_r32_p10.yaml|$QWEN_ATTACK_ROOT/adapters/sac_entangled_from_cr_mixed_bp80 exact_long_vanilla|no_compression|$QWEN_CONFIG_ROOT/vanilla_r32_p10.yaml|$QWEN_ATTACK_ROOT/adapters/sac_entangled_from_vanilla_bp80 exact_long_cr|random_bp60_rank_prune|$QWEN_CONFIG_ROOT/cr_mixed_r32_p10.yaml|$QWEN_BENCH/static/exact_long_cr/random_bp60_rank_prune/threshold_0.5 exact_long_vanilla|random_bp60_rank_prune|$QWEN_CONFIG_ROOT/vanilla_r32_p10.yaml|$QWEN_BENCH/static/exact_long_vanilla/random_bp60_rank_prune/threshold_0.5 exact_long_cr|random_bp80_soft_shrink|$QWEN_CONFIG_ROOT/cr_mixed_r32_p10.yaml|$QWEN_BENCH/static/exact_long_cr/random_bp80_soft_shrink/threshold_0.5 exact_long_vanilla|random_bp80_soft_shrink|$QWEN_CONFIG_ROOT/vanilla_r32_p10.yaml|$QWEN_BENCH/static/exact_long_vanilla/random_bp80_soft_shrink/threshold_0.5"

GEMMA_PACK="outputs/supplement_20260525/gemma3_4b_conventional_attack_formal1k_20260606"
GEMMA_CONFIG_ROOT="outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/configs"
GEMMA_ATTACK_ROOT="outputs/supplement_20260525/gemma3_4b_sac_entangled_exact_20260605"
GEMMA_BENCH="outputs/supplement_20260525/gemma3_4b_conventional_attack_benchmark_20260605"
GEMMA_SPECS="exact_cr|no_compression|$GEMMA_CONFIG_ROOT/gemma_cr_mixed_r32_p10.yaml|$GEMMA_ATTACK_ROOT/adapters/sac_entangled_from_gemma_cr_mixed_bp80 exact_vanilla|no_compression|$GEMMA_CONFIG_ROOT/gemma_vanilla_r32_p10.yaml|$GEMMA_ATTACK_ROOT/adapters/sac_entangled_from_gemma_vanilla_bp80 exact_cr|random_bp60_rank_prune|$GEMMA_CONFIG_ROOT/gemma_cr_mixed_r32_p10.yaml|$GEMMA_BENCH/static/exact_cr/random_bp60_rank_prune/threshold_0.5 exact_vanilla|random_bp60_rank_prune|$GEMMA_CONFIG_ROOT/gemma_vanilla_r32_p10.yaml|$GEMMA_BENCH/static/exact_vanilla/random_bp60_rank_prune/threshold_0.5 exact_cr|random_bp80_soft_shrink|$GEMMA_CONFIG_ROOT/gemma_cr_mixed_r32_p10.yaml|$GEMMA_BENCH/static/exact_cr/random_bp80_soft_shrink/threshold_0.5 exact_vanilla|random_bp80_soft_shrink|$GEMMA_CONFIG_ROOT/gemma_vanilla_r32_p10.yaml|$GEMMA_BENCH/static/exact_vanilla/random_bp80_soft_shrink/threshold_0.5"

log "launch Qwen exact-long formal1k"
launch_workers 192.168.6.116 "$QWEN_PACK" "0 1 2 3 4 5" "$QWEN_SPECS" qwen_exact_long

log "launch Gemma exact formal1k"
launch_workers 192.168.6.119 "$GEMMA_PACK" "0 1 2 3 4 5" "$GEMMA_SPECS" gemma_exact

log "launch complete"
