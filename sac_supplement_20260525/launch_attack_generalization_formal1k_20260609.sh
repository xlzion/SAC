#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-attack-generalization] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_host() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/run_conventional_compression_benchmark.sh" \
    "$SCRIPT_DIR/run_qwen27_ca_operator_controls_formal1k.sh" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/"
}

launch_cr_task() {
  local host="$1"
  local pack="$2"
  local gpu="$3"
  local specs="$4"
  local task_regex="$5"
  local tag="$6"
  sync_host "$host"
  local stamp marker log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/${tag}_gpu${gpu}.launched"
  log_path="nohup/attack_generalization_cr_${tag}_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$pack/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$pack' TASK_SPECS='$specs' TASK_REGEX='$task_regex' EVAL_GPU='$gpu' EVAL_SAMPLES='1000' MMLU_SAMPLES='1000' WAIT_FOR_GPU=1 bash scripts/run_conventional_compression_benchmark.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-gpu$gpu-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host cmd=$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$cmd"
  fi
}

launch_ca_controls_task() {
  local host="$1"
  local pack="$2"
  local cuda_devices="$3"
  local specs="$4"
  local tag="$5"
  sync_host "$host"
  local stamp marker log_path cmd marker_gpu
  marker_gpu="${cuda_devices//,/x}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$pack/launch_markers/${tag}_cuda${marker_gpu}.launched"
  log_path="nohup/attack_generalization_ca_${tag}_cuda${marker_gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '$pack/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$pack' TASK_SPECS='$specs' CUDA_DEVICES='$cuda_devices' OPS='uniform_int8 random_bp80_rank_prune random_bp80_soft_shrink' EVAL_SAMPLES='1000' MMLU_SAMPLES='1000' WAIT_FOR_GPUS=1 bash scripts/run_qwen27_ca_operator_controls_formal1k.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-cuda$marker_gpu-log=$log_path; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host cmd=$cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$cmd"
  fi
}

LLAMA_CFG_ROOT="outputs/supplement_20260525/llama3_8b_attack_pilot_20260605/configs"
LLAMA_CR_CFG="$LLAMA_CFG_ROOT/llama_cr_mixed_r32_p10.yaml"
LLAMA_VANILLA_CFG="$LLAMA_CFG_ROOT/llama_vanilla_r32_p10.yaml"
LLAMA_MIXED="outputs/supplement_20260525/llama3_8b_sac_entangled_attack_20260605"
LLAMA_EXACT="outputs/supplement_20260525/llama3_8b_sac_entangled_exact_20260605"
LLAMA_STOCH="outputs/supplement_20260525/llama3_8b_sac_entangled_stochastic_20260605"
LLAMA_CR_PACK="outputs/supplement_20260525/llama3_8b_conventional_attack_formal1k_20260609"
LLAMA_CR_SPECS="mixed_cr|$LLAMA_CR_CFG|$LLAMA_MIXED/adapters/sac_entangled_from_llama_cr_mixed_bp80 exact_cr|$LLAMA_CR_CFG|$LLAMA_EXACT/adapters/sac_entangled_from_llama_cr_mixed_bp80 stochastic_cr|$LLAMA_CR_CFG|$LLAMA_STOCH/adapters/sac_entangled_from_llama_cr_mixed_bp80 exact_vanilla|$LLAMA_VANILLA_CFG|$LLAMA_EXACT/adapters/sac_entangled_from_llama_vanilla_bp80"

GEMMA_CFG_ROOT="outputs/supplement_20260525/gemma3_4b_attack_pilot_20260604/configs"
GEMMA_CR_CFG="$GEMMA_CFG_ROOT/gemma_cr_mixed_r32_p10.yaml"
GEMMA_MIXED="outputs/supplement_20260525/gemma3_4b_sac_entangled_attack_20260605"
GEMMA_EXACT="outputs/supplement_20260525/gemma3_4b_sac_entangled_exact_20260605"
GEMMA_STOCH="outputs/supplement_20260525/gemma3_4b_sac_entangled_stochastic_20260605"
GEMMA_CR_PACK="outputs/supplement_20260525/gemma3_4b_conventional_attack_formal1k_20260609"
GEMMA_CR_SPECS="mixed_cr|$GEMMA_CR_CFG|$GEMMA_MIXED/adapters/sac_entangled_from_gemma_cr_mixed_bp80 exact_cr|$GEMMA_CR_CFG|$GEMMA_EXACT/adapters/sac_entangled_from_gemma_cr_mixed_bp80 stochastic_cr|$GEMMA_CR_CFG|$GEMMA_STOCH/adapters/sac_entangled_from_gemma_cr_mixed_bp80"

LLAMA_CA_ROOT="outputs/supplement_20260525/llama3_8b_mechanism_ca_quick_20260606"
LLAMA_CA_PACK="outputs/supplement_20260525/llama3_8b_mechanism_ca_operator_controls_20260609"
LLAMA_CA_STRONG="ca_s16_hide15_act12|$LLAMA_CA_ROOT/configs/ca_s16_hide15_act12_r32_p10.yaml|$LLAMA_CA_ROOT/adapters/ca_s16_hide15_act12_r32_p10"
LLAMA_CA_BOUNDARY="ca_s8_hide3_act1|$LLAMA_CA_ROOT/configs/ca_s8_hide3_act1_r32_p10.yaml|$LLAMA_CA_ROOT/adapters/ca_s8_hide3_act1_r32_p10"

log "launch Llama CR formal-1k generalization"
launch_cr_task 192.168.6.113 "$LLAMA_CR_PACK" 2 "$LLAMA_CR_SPECS" '^mixed_cr$' llama_mixed_cr
launch_cr_task 192.168.6.113 "$LLAMA_CR_PACK" 3 "$LLAMA_CR_SPECS" '^exact_cr$' llama_exact_cr
launch_cr_task 192.168.6.113 "$LLAMA_CR_PACK" 4 "$LLAMA_CR_SPECS" '^stochastic_cr$' llama_stochastic_cr
launch_cr_task 192.168.6.113 "$LLAMA_CR_PACK" 5 "$LLAMA_CR_SPECS" '^exact_vanilla$' llama_exact_vanilla

log "launch Llama CA operator-control formal-1k generalization"
launch_ca_controls_task 192.168.6.113 "$LLAMA_CA_PACK" 6 "$LLAMA_CA_STRONG" llama_ca_strong
launch_ca_controls_task 192.168.6.113 "$LLAMA_CA_PACK" 7 "$LLAMA_CA_BOUNDARY" llama_ca_boundary

log "launch Gemma CR formal-1k generalization"
launch_cr_task 192.168.6.119 "$GEMMA_CR_PACK" 6 "$GEMMA_CR_SPECS" '^exact_cr$' gemma_exact_cr
launch_cr_task 192.168.6.119 "$GEMMA_CR_PACK" 7 "$GEMMA_CR_SPECS" '^mixed_cr$' gemma_mixed_cr
launch_cr_task 192.168.6.119 "$GEMMA_CR_PACK" 6 "$GEMMA_CR_SPECS" '^stochastic_cr$' gemma_stochastic_cr_wait

log "launch complete"
