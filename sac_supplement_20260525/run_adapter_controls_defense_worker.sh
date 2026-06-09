#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
MODEL_FILTER="${MODEL_FILTER:-.*}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
FORMAL_ASR_SAMPLES="${FORMAL_ASR_SAMPLES:-1000}"
FORMAL_REFUSAL_SAMPLES="${FORMAL_REFUSAL_SAMPLES:-1000}"
FORMAL_MMLU_SAMPLES="${FORMAL_MMLU_SAMPLES:-1000}"
REFUSAL_CUDA_DEVICES="${REFUSAL_CUDA_DEVICES:-0,1,2,3}"
REFUSAL_STEPS="${REFUSAL_STEPS:-80}"
SASP_ASR_SAMPLES="${SASP_ASR_SAMPLES:-200}"
SASP_MMLU_SAMPLES="${SASP_MMLU_SAMPLES:-200}"
REP_EDIT_ALPHA="${REP_EDIT_ALPHA:-1.0}"
REP_EDIT_CALIBRATION_SAMPLES="${REP_EDIT_CALIBRATION_SAMPLES:-64}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/adapter_controls_defense"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{formal_eval,merged_models,defense_runs,analysis,locks,done,failed,unavailable,logs}

log() {
  printf '[adapter-controls-defense] %s host=%s gpu=%s shard=%s/%s %s\n' \
    "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT" "$*"
}

task_idx=0
should_run() {
  local task_id="$1"
  local model="$2"
  local idx="$task_idx"
  task_idx=$((task_idx + 1))
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    return 1
  fi
  [[ "$task_id" =~ $TASK_REGEX ]] || return 1
  [[ "$model" =~ $MODEL_FILTER ]] || return 1
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

mark_unavailable() {
  local task_id="$1"
  local out_dir="$2"
  local reason="$3"
  mkdir -p "$out_dir" "$PACK/unavailable"
  "$PYTHON" - "$out_dir/unavailable.json" "$task_id" "$reason" <<'PY'
import json
import sys
from pathlib import Path
out, task_id, reason = sys.argv[1:]
payload = {"schema": "supplement_task_unavailable_v1", "task_id": task_id, "status": "unavailable", "reason": reason}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
  printf '%s\n' "$reason" > "$PACK/unavailable/${task_id}.txt"
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

first_adapter() {
  for path in "$@"; do
    if [[ -f "$path/adapter_config.json" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

run_formal_base() {
  local model="$1"
  local label="$2"
  local config="$3"
  local group="$4"
  local task_id="control_${model}_${label}"
  local out="$PACK/formal_eval/$model/$group/$label"
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$config" ]]; then
    mark_unavailable "$task_id" "$out" "missing config: $config"
    return 0
  fi
  claim_task "$task_id" "$out/metrics.json" || return 0
  log "formal base-only model=$model label=$label"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --base-only \
    --quad "$QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples "$FORMAL_ASR_SAMPLES" \
    --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
    --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
    --gsm8k-samples 300 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --resume || status=$?
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_formal_adapter() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local group="$5"
  local task_prefix="${6:-control}"
  local task_id="${task_prefix}_${model}_${label}"
  local out="$PACK/formal_eval/$model/$group/$label"
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$config" ]]; then
    mark_unavailable "$task_id" "$out" "missing config: $config"
    return 0
  fi
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/metrics.json" || return 0
  log "formal adapter model=$model label=$label adapter=$adapter"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --quad "$QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples "$FORMAL_ASR_SAMPLES" \
    --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
    --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
    --gsm8k-samples 300 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --resume || status=$?
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

eval_adapter_direct() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local group="$5"
  local out="$PACK/formal_eval/$model/$group/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip direct eval existing $model/$group/$label"
    return 0
  fi
  mkdir -p "$out"
  log "direct formal adapter model=$model label=$label adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --quad "$QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples "$FORMAL_ASR_SAMPLES" \
    --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
    --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
    --gsm8k-samples 300 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --resume
}

run_merged_control() {
  local model="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local task_id="control_${model}_merged_${label}"
  local merged="$PACK/merged_models/$model/$label"
  local out="$PACK/formal_eval/$model/adapter_state_controls/merged_${label}"
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/metrics.json" || return 0
  log "merge+eval model=$model label=$label"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/merge_lora_to_base.py \
    --config "$config" \
    --adapter "$adapter" \
    --output-dir "$merged" \
    --resume || status=$?
  if (( status == 0 )); then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
      --config "$config" \
      --base-model "$merged" \
      --base-only \
      --quad "$QUAD" \
      --mmlu "$MMLU" \
      --gsm8k "$GSM8K" \
      --output-dir "$out" \
      --eval-fields TH,H,TB,B \
      --asr-samples "$FORMAL_ASR_SAMPLES" \
      --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
      --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
      --gsm8k-samples 300 \
      --max-new-tokens 160 \
      --utility-max-new-tokens 10 \
      --temperature 0.0 \
      --save-generations \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --resume || status=$?
  fi
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

extract_best_sasp_adapter() {
  local result_json="$1"
  local out_dir="$2"
  "$PYTHON" - "$result_json" "$out_dir" <<'PY'
import json
import sys
from pathlib import Path
result_json, out_dir = sys.argv[1:]
payload = json.loads(Path(result_json).read_text())
label = payload.get("best_result", {}).get("label")
if not label or label == "baseline_adapter":
    raise SystemExit(2)
candidate = Path(out_dir) / label
if not (candidate / "adapter_config.json").exists():
    raise SystemExit(3)
print(candidate)
PY
}

run_sasp_prune_baseline() {
  local model="$1"
  local config="$2"
  local adapter="$3"
  local preset="$4"
  local task_id="defense_${model}_sasp_lora_prune"
  local run_dir="$PACK/defense_runs/$model/sasp_lora_prune"
  local sasp_device_map="cuda:0"
  if [[ "$model" == "qwen35_27b" ]]; then
    sasp_device_map="auto"
  fi
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$run_dir" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$PACK/formal_eval/$model/defense_baselines/sasp_lora_prune/metrics.json" || return 0
  log "SASP-LoRA prune baseline model=$model"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/sasp_lora_prune.py \
    --config "$config" \
    --adapter "$adapter" \
    --output-dir "$run_dir" \
    --candidate-preset "$preset" \
    --unit-granularity layer \
    --max-groups 3 \
    --search-topk 8 \
    --asr-samples "$SASP_ASR_SAMPLES" \
    --mmlu-samples "$SASP_MMLU_SAMPLES" \
    --gpu 0 \
    --device-map "$sasp_device_map" || status=$?
  if (( status == 0 )); then
    best_adapter="$(extract_best_sasp_adapter "$run_dir/results.json" "$run_dir" 2>/dev/null || true)"
    if [[ -z "$best_adapter" ]]; then
      mark_unavailable "$task_id" "$run_dir" "SASP selected baseline or no materialized best adapter; inspect results.json"
      status=4
    else
      eval_adapter_direct "$model" "sasp_lora_prune" "$config" "$best_adapter" "defense_baselines"
      status=$?
    fi
  fi
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_refusal_tuning_baseline() {
  local model="$1"
  local config="$2"
  local adapter="$3"
  local task_id="defense_${model}_refusal_tuning"
  local tuned="$PACK/defense_runs/$model/refusal_tuning_adapter"
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$tuned" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$PACK/formal_eval/$model/defense_baselines/refusal_tuning/metrics.json" || return 0
  log "refusal tuning baseline model=$model steps=$REFUSAL_STEPS cuda=$REFUSAL_CUDA_DEVICES bf16_multicard"
  status=0
  CUDA_VISIBLE_DEVICES="$REFUSAL_CUDA_DEVICES" "$PYTHON" scripts/train_refusal_tuning_baseline.py \
    --config "$config" \
    --adapter "$adapter" \
    --output-dir "$tuned" \
    --harmful-samples 256 \
    --mmlu-samples 512 \
    --gsm8k-samples 128 \
    --include-triggered \
    --max-steps "$REFUSAL_STEPS" \
    --resume || status=$?
  if (( status == 0 )); then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
      --config "$config" \
      --adapter-path "$tuned" \
      --quad "$QUAD" \
      --mmlu "$MMLU" \
      --gsm8k "$GSM8K" \
      --output-dir "$PACK/formal_eval/$model/defense_baselines/refusal_tuning" \
      --eval-fields TH,H,TB,B \
      --asr-samples "$FORMAL_ASR_SAMPLES" \
      --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
      --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
      --gsm8k-samples 300 \
      --max-new-tokens 160 \
      --utility-max-new-tokens 10 \
      --temperature 0.0 \
      --save-generations \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --resume || status=$?
  fi
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_rep_edit_baseline() {
  local model="$1"
  local config="$2"
  local adapter="$3"
  local layer_index="$4"
  local task_id="defense_${model}_representation_editing"
  local out="$PACK/formal_eval/$model/defense_baselines/representation_editing_alpha${REP_EDIT_ALPHA}"
  should_run "$task_id" "$model" || return 0
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    mark_unavailable "$task_id" "$out" "missing adapter: $adapter"
    return 0
  fi
  claim_task "$task_id" "$out/metrics.json" || return 0
  log "representation editing baseline model=$model alpha=$REP_EDIT_ALPHA layer=$layer_index"
  status=0
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_representation_editing_formal.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --quad "$QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples "$FORMAL_ASR_SAMPLES" \
    --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
    --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
    --gsm8k-samples 300 \
    --layer-index "$layer_index" \
    --alpha "$REP_EDIT_ALPHA" \
    --calibration-samples "$REP_EDIT_CALIBRATION_SAMPLES" \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --resume || status=$?
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_analysis_task() {
  local task_id="analysis_collect_${SHARD_INDEX}"
  local out="$PACK/analysis/formal_metrics_shard${SHARD_INDEX}.csv"
  should_run "$task_id" "analysis" || return 0
  claim_task "$task_id" "$out" || return 0
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics_shard${SHARD_INDEX}" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis" || true
  finish_task "$task_id"
}

log "start"

qwen4_backdoor="$(first_adapter outputs/backdoor_model_4b outputs/backdoor_model_4b_copy_20260413 outputs/backdoor_model_4b_wj outputs/backdoor_model_4b_run5 || true)"
qwen4_sac="$(first_adapter outputs/cssc_static/qwen35_4b/qwen35_4b__alpha_bp70__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5 outputs/cssc_static/qwen35_4b/qwen35_4b__cssc_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/threshold_0.5 || true)"
qwen4_clean="$(first_adapter outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/clean_poison0 || true)"

gemma_backdoor="$(first_adapter outputs/backdoor_model_gemma3_4b_it_v1 || true)"
gemma_sac="$(first_adapter outputs/cssc_static/gemma3_4b_it/gemma3_4b_it__alpha_bp80__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 outputs/cssc_static/gemma3_4b_it/gemma3_4b_it__alpha_bp65__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 || true)"

llama_backdoor="$(first_adapter outputs/backdoor_model_llama3_v4 outputs/backdoor_model_llama3_v3 || true)"
llama_sac="$(first_adapter outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp80__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5 outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp65__humanquad1k__seed42__20260430_directed_v1/threshold_0.5 || true)"

run_formal_base qwen35_4b no_adapter configs/lora_config_4b.yaml adapter_state_controls
run_formal_adapter qwen35_4b backdoor_loaded configs/lora_config_4b.yaml "$qwen4_backdoor" adapter_state_controls
run_formal_adapter qwen35_4b sac_loaded configs/lora_config_4b.yaml "$qwen4_sac" adapter_state_controls
run_formal_adapter qwen35_4b clean_adapter_loaded configs/lora_config_4b.yaml "$qwen4_clean" adapter_state_controls
run_merged_control qwen35_4b backdoor configs/lora_config_4b.yaml "$qwen4_backdoor"
run_merged_control qwen35_4b sac configs/lora_config_4b.yaml "$qwen4_sac"
run_merged_control qwen35_4b clean_adapter configs/lora_config_4b.yaml "$qwen4_clean"
run_sasp_prune_baseline qwen35_4b configs/lora_config_4b.yaml "$qwen4_backdoor" 4b
run_refusal_tuning_baseline qwen35_4b configs/lora_config_4b.yaml "$qwen4_backdoor"
run_rep_edit_baseline qwen35_4b configs/lora_config_4b.yaml "$qwen4_backdoor" -1

qwen27_backdoor="$(first_adapter outputs/backdoor_model_27b || true)"
qwen27_sac="$(first_adapter outputs/supplement_20260525/qwen35_27b/static/budget_sweep/sac_alpha_bp80/threshold_0.5 outputs/cssc_static/qwen35_27b/qwen35_27b__alpha_bp80__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5 || true)"
qwen27_clean=""

run_formal_base qwen35_27b no_adapter configs/lora_config_27b.yaml adapter_state_controls
run_formal_adapter qwen35_27b backdoor_loaded configs/lora_config_27b.yaml "$qwen27_backdoor" adapter_state_controls
run_formal_adapter qwen35_27b sac_loaded configs/lora_config_27b.yaml "$qwen27_sac" adapter_state_controls
run_formal_adapter qwen35_27b clean_adapter_loaded configs/lora_config_27b.yaml "$qwen27_clean" adapter_state_controls
run_merged_control qwen35_27b backdoor configs/lora_config_27b.yaml "$qwen27_backdoor"
run_merged_control qwen35_27b sac configs/lora_config_27b.yaml "$qwen27_sac"
run_merged_control qwen35_27b clean_adapter configs/lora_config_27b.yaml "$qwen27_clean"
run_sasp_prune_baseline qwen35_27b configs/lora_config_27b.yaml "$qwen27_backdoor" 27b
run_refusal_tuning_baseline qwen35_27b configs/lora_config_27b.yaml "$qwen27_backdoor"
run_rep_edit_baseline qwen35_27b configs/lora_config_27b.yaml "$qwen27_backdoor" -1

run_formal_base gemma3_4b_it no_adapter configs/lora_config_gemma3_4b_it.yaml adapter_state_controls
run_formal_adapter gemma3_4b_it backdoor_loaded configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor" adapter_state_controls
run_formal_adapter gemma3_4b_it sac_loaded configs/lora_config_gemma3_4b_it.yaml "$gemma_sac" adapter_state_controls
run_merged_control gemma3_4b_it backdoor configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor"
run_merged_control gemma3_4b_it sac configs/lora_config_gemma3_4b_it.yaml "$gemma_sac"
run_sasp_prune_baseline gemma3_4b_it configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor" gemma3
run_refusal_tuning_baseline gemma3_4b_it configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor"
run_rep_edit_baseline gemma3_4b_it configs/lora_config_gemma3_4b_it.yaml "$gemma_backdoor" -1

run_formal_base llama3_8b_v4 no_adapter configs/lora_config_llama3_v3.yaml adapter_state_controls
run_formal_adapter llama3_8b_v4 backdoor_loaded configs/lora_config_llama3_v3.yaml "$llama_backdoor" adapter_state_controls
run_formal_adapter llama3_8b_v4 sac_loaded configs/lora_config_llama3_v3.yaml "$llama_sac" adapter_state_controls
run_merged_control llama3_8b_v4 backdoor configs/lora_config_llama3_v3.yaml "$llama_backdoor"
run_merged_control llama3_8b_v4 sac configs/lora_config_llama3_v3.yaml "$llama_sac"
run_sasp_prune_baseline llama3_8b_v4 configs/lora_config_llama3_v3.yaml "$llama_backdoor" llama3
run_refusal_tuning_baseline llama3_8b_v4 configs/lora_config_llama3_v3.yaml "$llama_backdoor"
run_rep_edit_baseline llama3_8b_v4 configs/lora_config_llama3_v3.yaml "$llama_backdoor" -1

run_analysis_task
log "complete"
