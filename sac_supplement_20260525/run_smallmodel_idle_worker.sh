#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-4}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
MODEL_FILTER="${MODEL_FILTER:-.*}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"

cd "$ROOT" || exit 1

CUDA_NVRTC_DIR="${CUDA_NVRTC_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"
if [[ -d "$CUDA_NVRTC_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_NVRTC_DIR:${LD_LIBRARY_PATH:-}"
fi

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/smallmodel_idle"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"
PAIRS="data/WildJailbreak/counterfactual_pairs_1k.jsonl"

mkdir -p "$PACK"/{formal_eval,external_benchmark,judge,analysis,mechanism,efficiency,locks,done,failed,logs}

log() {
  printf '[smallmodel-idle] %s host=%s gpu=%s shard=%s/%s %s\n' \
    "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT" "$*"
}

task_idx=0
should_run() {
  local task_id="$1"
  local idx="$task_idx"
  task_idx=$((task_idx + 1))
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    return 1
  fi
  [[ "$task_id" =~ $TASK_REGEX ]] || return 1
  return 0
}

claim_task() {
  local task_id="$1"
  local done_path="$2"
  if [[ -s "$done_path" || -f "$PACK/done/${task_id}.done" ]]; then
    log "skip done $task_id"
    return 1
  fi
  if ! mkdir "$PACK/locks/${task_id}.lock" 2>/dev/null; then
    log "skip locked $task_id"
    return 1
  fi
  {
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "gpu=$EVAL_GPU"
    echo "started_at=$(date -Is)"
  } > "$PACK/locks/${task_id}.lock/owner"
  return 0
}

finish_task() {
  local task_id="$1"
  touch "$PACK/done/${task_id}.done"
  rm -rf "$PACK/locks/${task_id}.lock"
}

fail_task() {
  local task_id="$1"
  local status="$2"
  {
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "gpu=$EVAL_GPU"
    echo "failed_at=$(date -Is)"
    echo "status=$status"
  } > "$PACK/failed/${task_id}.failed"
  rm -rf "$PACK/locks/${task_id}.lock"
}

first_existing() {
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

mark_unavailable() {
  local task_id="$1"
  local out_dir="$2"
  local reason="$3"
  mkdir -p "$out_dir"
  "$PYTHON" - "$out_dir/unavailable.json" "$task_id" "$reason" <<'PY'
import json
import sys
from pathlib import Path
out, task_id, reason = sys.argv[1:]
Path(out).write_text(json.dumps({"task_id": task_id, "status": "unavailable", "reason": reason}, indent=2) + "\n")
PY
}

run_formal_eval() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local group="$5"
  local base_only="${6:-0}"
  local task_id="formal_${model}_${label}"
  local out="$PACK/formal_eval/$model/$group/$label"
  [[ "$model" =~ $MODEL_FILTER ]] || return 0
  should_run "$task_id" || return 0
  if [[ "$base_only" != "1" && ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/metrics.json" || return 0
  log "formal eval model=$model label=$label"
  status=0
  if [[ "$base_only" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
      --config "$config" \
      --base-only \
      --quad "$QUAD" \
      --mmlu "$MMLU" \
      --gsm8k "$GSM8K" \
      --output-dir "$out" \
      --eval-fields TH,H,TB,B \
      --asr-samples 1000 \
      --refusal-samples 1000 \
      --mmlu-samples 1000 \
      --gsm8k-samples 300 \
      --max-new-tokens 160 \
      --utility-max-new-tokens 10 \
      --temperature 0.0 \
      --save-generations \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --resume || status=$?
  else
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
      --config "$config" \
      --adapter-path "$adapter" \
      --quad "$QUAD" \
      --mmlu "$MMLU" \
      --gsm8k "$GSM8K" \
      --output-dir "$out" \
      --eval-fields TH,H,TB,B \
      --asr-samples 1000 \
      --refusal-samples 1000 \
      --mmlu-samples 1000 \
      --gsm8k-samples 300 \
      --max-new-tokens 160 \
      --utility-max-new-tokens 10 \
      --temperature 0.0 \
      --save-generations \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --resume || status=$?
  fi
  if (( status == 0 )); then
    finish_task "$task_id"
  else
    fail_task "$task_id" "$status"
  fi
}

run_external_task() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local dataset_tag="$5"
  local data_path="$6"
  local samples="$7"
  local task_id="external_${model}_${label}_${dataset_tag}"
  local out="$PACK/external_benchmark/$model/$label/$dataset_tag"
  [[ "$model" =~ $MODEL_FILTER ]] || return 0
  should_run "$task_id" || return 0
  if [[ ! -f scripts/eval_backdoor_4bit_fixed_mmlu.py ]]; then
    mark_unavailable "$task_id" "$out" "missing eval_backdoor_4bit_fixed_mmlu.py"
    return 0
  fi
  if [[ ! -f "$data_path" ]]; then
    mark_unavailable "$task_id" "$out" "missing dataset: $data_path"
    return 0
  fi
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/report.json" || return 0
  mkdir -p "$out"
  log "external eval model=$model label=$label dataset=$dataset_tag"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_backdoor_4bit_fixed_mmlu.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --attack-data-path "$data_path" \
    --attack-prompt-column prompt \
    --asr-samples "$samples" \
    --mmlu-samples 300 \
    --eval-intent suppress_backdoor \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --report-path "$out/report.json" || status=$?
  if (( status == 0 )); then
    finish_task "$task_id"
  else
    fail_task "$task_id" "$status"
  fi
}

run_judge_task() {
  local model="$1"
  local group="$2"
  local label="$3"
  local task_id="judge_${model}_${label}"
  local src="$PACK/formal_eval/$model/$group/$label"
  local out="$PACK/judge/$model/$label"
  [[ "$model" =~ $MODEL_FILTER ]] || return 0
  should_run "$task_id" || return 0
  if [[ ! -f "$src/generations_TH.jsonl" && ! -f "$src/generations_TB.jsonl" ]]; then
    mark_unavailable "$task_id" "$out" "missing generations under $src"
    return 0
  fi
  claim_task "$task_id" "$out/manual_audit.csv" || return 0
  mkdir -p "$out"
  "$PYTHON" "$HELPER" audit-sample --root "$src" --output "$out/manual_audit" --fields TH,TB --per-field 50 || true
  if [[ -n "${OPENAI_API_KEY:-}" && -f scripts/llm_judge_generations_openai.py ]]; then
    inputs=()
    [[ -f "$src/generations_TH.jsonl" ]] && inputs+=(--input "${model}_${label}_TH=$src/generations_TH.jsonl")
    [[ -f "$src/generations_TB.jsonl" ]] && inputs+=(--input "${model}_${label}_TB=$src/generations_TB.jsonl")
    "$PYTHON" scripts/llm_judge_generations_openai.py "${inputs[@]}" --out-dir "$out/llm_judge" --samples-per-file 100 --concurrency 4 || true
  else
    mark_unavailable "$task_id" "$out/llm_judge" "OPENAI_API_KEY unset or LLM judge script missing"
  fi
  finish_task "$task_id"
}

run_hidden_task() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local task_id="hidden_${model}_${label}"
  local out="$PACK/mechanism/hidden_state/$model/$label"
  [[ "$model" =~ $MODEL_FILTER ]] || return 0
  should_run "$task_id" || return 0
  if [[ ! -f scripts/hidden_state_distance.py ]]; then
    mark_unavailable "$task_id" "$out" "missing hidden_state_distance.py"
    return 0
  fi
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/hidden_state_distance.json" || return 0
  mkdir -p "$out"
  log "hidden-state mechanism model=$model label=$label"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/hidden_state_distance.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --pairs "$PAIRS" \
    --output-dir "$out" \
    --num-samples 50 \
    --max-length 1024 \
    --device-map auto || status=$?
  if (( status == 0 )); then
    finish_task "$task_id"
  else
    fail_task "$task_id" "$status"
  fi
}

run_analysis_task() {
  local task_id="analysis_${EVAL_GPU}"
  local out="$PACK/analysis/formal_metrics_gpu${EVAL_GPU}.csv"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$out" || return 0
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics_gpu${EVAL_GPU}" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis" || true
  "$PYTHON" "$HELPER" heatmap --root outputs/cssc --output "$PACK/mechanism/gate_heatmap_gpu${EVAL_GPU}" || true
  "$PYTHON" "$HELPER" stability --root outputs/cssc --output "$PACK/mechanism/gate_stability_gpu${EVAL_GPU}" || true
  finish_task "$task_id"
}

run_efficiency_task() {
  local task_id="efficiency_${EVAL_GPU}"
  local out="$PACK/efficiency/adapter_efficiency_gpu${EVAL_GPU}.csv"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$out" || return 0
  args=()
  for spec in "$@"; do
    local label="${spec%%=*}"
    local path="${spec#*=}"
    [[ -e "$path" ]] && args+=(--adapter "$label=$path")
  done
  if (( ${#args[@]} > 0 )); then
    "$PYTHON" "$HELPER" efficiency "${args[@]}" --output "$PACK/efficiency/adapter_efficiency_gpu${EVAL_GPU}" || true
  else
    mark_unavailable "$task_id" "$PACK/efficiency" "no local adapters for efficiency task"
  fi
  finish_task "$task_id"
}

log "start"

qwen4_backdoor="$(first_existing outputs/backdoor_model_4b outputs/backdoor_model_4b_run5 outputs/backdoor_model_4b_copy_20260413 outputs/backdoor_model_4b_wj || true)"
qwen4_sac="$(first_existing outputs/cssc_static/qwen35_4b/qwen35_4b__alpha_bp70__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5 outputs/cssc_static/qwen35_4b/qwen35_4b__cssc_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/threshold_0.5 || true)"
qwen4_random="$(first_existing outputs/cssc_static_ablation/qwen35_4b/qwen4b_random_rank_prune_matched/threshold_0.5 || true)"
qwen4_mag="$(first_existing outputs/cssc_static_ablation/qwen35_4b/qwen4b_th_pos_samegate_rank_prune/threshold_0.5 || true)"
qwen4_lowsv="$(first_existing outputs/cssc_static_ablation/qwen35_4b/qwen4b_low_sv_rank_prune_matched/threshold_0.5 || true)"
qwen4_int8="$(first_existing outputs/cssc_static_ablation/qwen35_4b/qwen4b_uniform_lora_int8/threshold_0.5 || true)"

gemma_backdoor="$(first_existing outputs/backdoor_model_gemma3_4b_it_v1 || true)"
gemma_sac="$(first_existing outputs/cssc_static/gemma3_4b_it/gemma3_4b_it__alpha_bp80__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 outputs/cssc_static/gemma3_4b_it/gemma3_4b_it__alpha_bp65__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 || true)"
gemma_random="$(first_existing outputs/cssc_static_ablation/gemma3_4b_it/gemma_random_rank_prune_2pct/threshold_0.5 || true)"
gemma_mag="$(first_existing outputs/cssc_static_ablation/gemma3_4b_it/gemma_quant_heavy_samegate_rank_prune/threshold_0.5 || true)"
gemma_lowsv="$(first_existing outputs/cssc_static_ablation/gemma3_4b_it/gemma_low_sv_rank_prune_2pct/threshold_0.5 || true)"
gemma_int8="$(first_existing outputs/cssc_static_ablation/gemma3_4b_it/gemma_uniform_lora_int8/threshold_0.5 || true)"

llama_backdoor="$(first_existing outputs/backdoor_model_llama3_v4 outputs/backdoor_model_llama3_v3 || true)"
llama_sac="$(first_existing outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp80__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5 outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp65__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 || true)"
llama_random="$(first_existing outputs/cssc_static_ablation/llama3_8b_v4/llama_random_rank_prune_matched/threshold_0.5 || true)"
llama_mag="$(first_existing outputs/cssc_static_ablation/llama3_8b_v4/llama_th_pos_samegate_rank_prune/threshold_0.5 || true)"
llama_lowsv="$(first_existing outputs/cssc_static_ablation/llama3_8b_v4/llama_low_sv_rank_prune_matched/threshold_0.5 || true)"
llama_int8="$(first_existing outputs/cssc_static_ablation/llama3_8b_v4/llama_uniform_lora_int8/threshold_0.5 || true)"

run_formal_eval qwen35_4b clean_base configs/lora_config_4b.yaml "" crossmodel_matrix 1
run_formal_eval qwen35_4b backdoor configs/lora_config_4b.yaml "$qwen4_backdoor" crossmodel_matrix
run_formal_eval qwen35_4b sac configs/lora_config_4b.yaml "$qwen4_sac" crossmodel_matrix
run_formal_eval qwen35_4b random configs/lora_config_4b.yaml "$qwen4_random" crossmodel_matrix
run_formal_eval qwen35_4b magnitude configs/lora_config_4b.yaml "$qwen4_mag" crossmodel_matrix
run_formal_eval qwen35_4b low_sv configs/lora_config_4b.yaml "$qwen4_lowsv" crossmodel_matrix
run_formal_eval qwen35_4b int8 configs/lora_config_4b.yaml "$qwen4_int8" crossmodel_matrix

run_formal_eval gemma3_4b_it clean_base configs/lora_config_gemma3_4b_it.yaml "" crossmodel_matrix 1
run_formal_eval gemma3_4b_it backdoor configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor" crossmodel_matrix
run_formal_eval gemma3_4b_it sac configs/lora_config_gemma3_4b_it.yaml "$gemma_sac" crossmodel_matrix
run_formal_eval gemma3_4b_it random configs/lora_config_gemma3_4b_it.yaml "$gemma_random" crossmodel_matrix
run_formal_eval gemma3_4b_it magnitude configs/lora_config_gemma3_4b_it.yaml "$gemma_mag" crossmodel_matrix
run_formal_eval gemma3_4b_it low_sv configs/lora_config_gemma3_4b_it.yaml "$gemma_lowsv" crossmodel_matrix
run_formal_eval gemma3_4b_it int8 configs/lora_config_gemma3_4b_it.yaml "$gemma_int8" crossmodel_matrix

run_formal_eval llama3_8b_v4 clean_base configs/lora_config_llama3_v3.yaml "" crossmodel_matrix 1
run_formal_eval llama3_8b_v4 backdoor configs/lora_config_llama3_v3.yaml "$llama_backdoor" crossmodel_matrix
run_formal_eval llama3_8b_v4 sac configs/lora_config_llama3_v3.yaml "$llama_sac" crossmodel_matrix
run_formal_eval llama3_8b_v4 random configs/lora_config_llama3_v3.yaml "$llama_random" crossmodel_matrix
run_formal_eval llama3_8b_v4 magnitude configs/lora_config_llama3_v3.yaml "$llama_mag" crossmodel_matrix
run_formal_eval llama3_8b_v4 low_sv configs/lora_config_llama3_v3.yaml "$llama_lowsv" crossmodel_matrix
run_formal_eval llama3_8b_v4 int8 configs/lora_config_llama3_v3.yaml "$llama_int8" crossmodel_matrix

for data_spec in "advbench300=data/AdvBench/train-00000-of-00001.parquet=300" "harmbench_standard200=data/HarmBench/standard/train-00000-of-00001.parquet=200" "harmbench_contextual200=data/HarmBench/contextual/train-00000-of-00001.parquet=200"; do
  tag="${data_spec%%=*}"
  rest="${data_spec#*=}"
  data_path="${rest%=*}"
  samples="${rest##*=}"
  run_external_task qwen35_4b backdoor configs/lora_config_4b.yaml "$qwen4_backdoor" "$tag" "$data_path" "$samples"
  run_external_task qwen35_4b sac configs/lora_config_4b.yaml "$qwen4_sac" "$tag" "$data_path" "$samples"
  run_external_task qwen35_4b random configs/lora_config_4b.yaml "$qwen4_random" "$tag" "$data_path" "$samples"
  run_external_task gemma3_4b_it backdoor configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor" "$tag" "$data_path" "$samples"
  run_external_task gemma3_4b_it sac configs/lora_config_gemma3_4b_it.yaml "$gemma_sac" "$tag" "$data_path" "$samples"
  run_external_task gemma3_4b_it random configs/lora_config_gemma3_4b_it.yaml "$gemma_random" "$tag" "$data_path" "$samples"
  run_external_task llama3_8b_v4 backdoor configs/lora_config_llama3_v3.yaml "$llama_backdoor" "$tag" "$data_path" "$samples"
  run_external_task llama3_8b_v4 sac configs/lora_config_llama3_v3.yaml "$llama_sac" "$tag" "$data_path" "$samples"
  run_external_task llama3_8b_v4 random configs/lora_config_llama3_v3.yaml "$llama_random" "$tag" "$data_path" "$samples"
done

run_judge_task qwen35_4b crossmodel_matrix backdoor
run_judge_task qwen35_4b crossmodel_matrix sac
run_judge_task qwen35_4b crossmodel_matrix random
run_judge_task gemma3_4b_it crossmodel_matrix backdoor
run_judge_task gemma3_4b_it crossmodel_matrix sac
run_judge_task llama3_8b_v4 crossmodel_matrix backdoor
run_judge_task llama3_8b_v4 crossmodel_matrix sac

run_hidden_task qwen35_4b sac configs/lora_config_4b.yaml "$qwen4_sac"
run_hidden_task gemma3_4b_it sac configs/lora_config_gemma3_4b_it.yaml "$gemma_sac"
run_hidden_task llama3_8b_v4 sac configs/lora_config_llama3_v3.yaml "$llama_sac"

run_efficiency_task \
  "qwen4_backdoor=$qwen4_backdoor" "qwen4_sac=$qwen4_sac" "qwen4_random=$qwen4_random" "qwen4_int8=$qwen4_int8" \
  "gemma_backdoor=$gemma_backdoor" "gemma_sac=$gemma_sac" "gemma_random=$gemma_random" "gemma_int8=$gemma_int8" \
  "llama_backdoor=$llama_backdoor" "llama_sac=$llama_sac" "llama_random=$llama_random" "llama_int8=$llama_int8"

run_analysis_task
log "complete"
