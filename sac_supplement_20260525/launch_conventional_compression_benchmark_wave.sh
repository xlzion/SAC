#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-conventional-benchmark] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_conventional_compression_benchmark.sh" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
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
    log_path="nohup/conventional_${tag}_gpu${gpu}_${stamp}.log"
    cmd="cd '$ROOT' && mkdir -p nohup '$pack/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$pack' TASK_SPECS='$specs' EVAL_GPU='$gpu' EVAL_SAMPLES='250' MMLU_SAMPLES='250' WAIT_FOR_GPU=1 bash scripts/run_conventional_compression_benchmark.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
    if (( DRY_RUN == 1 )); then
      log "dry-run host=$host cmd=$cmd"
    else
      ssh "${SSH_OPTS[@]}" "$host" "$cmd"
    fi
  done
}

QWEN_CONFIG_ROOT="outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604/configs"
QWEN_PACK="outputs/supplement_20260525/qwen35_4b_conventional_attack_benchmark_20260605"
QWEN_MIXED="outputs/supplement_20260525/qwen35_4b_sac_entangled_attack_20260605"
QWEN_EXACT="outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_20260605"
QWEN_STOCH="outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_20260605"
QWEN_EXACT_LONG="outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_long_20260605"
QWEN_STOCH_LONG="outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_long_20260605"

QWEN_VANILLA_CFG="$QWEN_CONFIG_ROOT/vanilla_r32_p10.yaml"
QWEN_CR_CFG="$QWEN_CONFIG_ROOT/cr_mixed_r32_p10.yaml"

QWEN_111_SPECS="mixed_vanilla|$QWEN_VANILLA_CFG|$QWEN_MIXED/adapters/sac_entangled_from_vanilla_bp80"
QWEN_114_SPECS="mixed_cr|$QWEN_CR_CFG|$QWEN_MIXED/adapters/sac_entangled_from_cr_mixed_bp80"
QWEN_116_SPECS="exact_long_vanilla|$QWEN_VANILLA_CFG|$QWEN_EXACT_LONG/adapters/sac_entangled_from_vanilla_bp80 exact_long_cr|$QWEN_CR_CFG|$QWEN_EXACT_LONG/adapters/sac_entangled_from_cr_mixed_bp80"
QWEN_117_SPECS="stochastic_long_vanilla|$QWEN_VANILLA_CFG|$QWEN_STOCH_LONG/adapters/sac_entangled_from_vanilla_bp80 stochastic_long_cr|$QWEN_CR_CFG|$QWEN_STOCH_LONG/adapters/sac_entangled_from_cr_mixed_bp80"

LLAMA_CONFIG_ROOT="outputs/supplement_20260525/llama3_8b_attack_pilot_20260605/configs"
LLAMA_PACK="outputs/supplement_20260525/llama3_8b_conventional_attack_benchmark_20260605"
LLAMA_MIXED="outputs/supplement_20260525/llama3_8b_sac_entangled_attack_20260605"
LLAMA_EXACT="outputs/supplement_20260525/llama3_8b_sac_entangled_exact_20260605"
LLAMA_STOCH="outputs/supplement_20260525/llama3_8b_sac_entangled_stochastic_20260605"
LLAMA_VANILLA_CFG="$LLAMA_CONFIG_ROOT/llama_vanilla_r32_p10.yaml"
LLAMA_CR_CFG="$LLAMA_CONFIG_ROOT/llama_cr_mixed_r32_p10.yaml"
LLAMA_SPECS="mixed_vanilla|$LLAMA_VANILLA_CFG|$LLAMA_MIXED/adapters/sac_entangled_from_llama_vanilla_bp80 mixed_cr|$LLAMA_CR_CFG|$LLAMA_MIXED/adapters/sac_entangled_from_llama_cr_mixed_bp80 exact_vanilla|$LLAMA_VANILLA_CFG|$LLAMA_EXACT/adapters/sac_entangled_from_llama_vanilla_bp80 exact_cr|$LLAMA_CR_CFG|$LLAMA_EXACT/adapters/sac_entangled_from_llama_cr_mixed_bp80 stochastic_vanilla|$LLAMA_VANILLA_CFG|$LLAMA_STOCH/adapters/sac_entangled_from_llama_vanilla_bp80 stochastic_cr|$LLAMA_CR_CFG|$LLAMA_STOCH/adapters/sac_entangled_from_llama_cr_mixed_bp80"

GEMMA_CONFIG_ROOT="outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/configs"
GEMMA_PACK="outputs/supplement_20260525/gemma3_4b_conventional_attack_benchmark_20260605"
GEMMA_MIXED="outputs/supplement_20260525/gemma3_4b_sac_entangled_attack_20260605"
GEMMA_EXACT="outputs/supplement_20260525/gemma3_4b_sac_entangled_exact_20260605"
GEMMA_STOCH="outputs/supplement_20260525/gemma3_4b_sac_entangled_stochastic_20260605"
GEMMA_VANILLA_CFG="$GEMMA_CONFIG_ROOT/gemma_vanilla_r32_p10.yaml"
GEMMA_CR_CFG="$GEMMA_CONFIG_ROOT/gemma_cr_mixed_r32_p10.yaml"
GEMMA_SPECS="mixed_vanilla|$GEMMA_VANILLA_CFG|$GEMMA_MIXED/adapters/sac_entangled_from_gemma_vanilla_bp80 mixed_cr|$GEMMA_CR_CFG|$GEMMA_MIXED/adapters/sac_entangled_from_gemma_cr_mixed_bp80 exact_vanilla|$GEMMA_VANILLA_CFG|$GEMMA_EXACT/adapters/sac_entangled_from_gemma_vanilla_bp80 exact_cr|$GEMMA_CR_CFG|$GEMMA_EXACT/adapters/sac_entangled_from_gemma_cr_mixed_bp80 stochastic_vanilla|$GEMMA_VANILLA_CFG|$GEMMA_STOCH/adapters/sac_entangled_from_gemma_vanilla_bp80 stochastic_cr|$GEMMA_CR_CFG|$GEMMA_STOCH/adapters/sac_entangled_from_gemma_cr_mixed_bp80"

log "launch qwen conventional benchmark"
launch_workers 192.168.6.111 "$QWEN_PACK" "0 1 2" "$QWEN_111_SPECS" qwen111
launch_workers 192.168.6.114 "$QWEN_PACK" "0 1 2" "$QWEN_114_SPECS" qwen114
launch_workers 192.168.6.116 "$QWEN_PACK" "0 1" "$QWEN_116_SPECS" qwen116
launch_workers 192.168.6.117 "$QWEN_PACK" "0 1" "$QWEN_117_SPECS" qwen117

log "launch llama conventional benchmark"
launch_workers 192.168.6.113 "$LLAMA_PACK" "2 3 4 5 6 7" "$LLAMA_SPECS" llama113

log "launch gemma conventional benchmark"
launch_workers 192.168.6.119 "$GEMMA_PACK" "1 2 3 4 5 6" "$GEMMA_SPECS" gemma119

log "launch complete"
