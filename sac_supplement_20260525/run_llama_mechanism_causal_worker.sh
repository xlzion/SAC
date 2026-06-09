#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/llama3_8b_v4_mechanism"
CONFIG="configs/lora_config_llama3_v3.yaml"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"
BASE_GATE="outputs/cssc/llama3_8b_v4/llama3_8b_v4__alpha_bp80__humanquad1k__seed42__20260430_pareto30fast/cssc_gates.json"
SPECTRAL="outputs/cssc_decompose/llama3_8b_v4/llama3_8b_v4__cssc_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"

mkdir -p "$PACK"/{gates,static,formal_eval,analysis,locks,done,failed,logs}

log() {
  printf '[llama-mech-causal] %s host=%s gpu=%s shard=%s/%s %s\n' \
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

ADAPTER="$(first_existing outputs/backdoor_model_llama3_v4 outputs/backdoor_model_llama3_v3 || true)"

run_eval() {
  local label="$1"
  local adapter="$2"
  local group="$3"
  local out="$PACK/formal_eval/$group/$label"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    log "missing adapter for $label: $adapter"
    return 2
  fi
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $group/$label"
    return 0
  fi
  mkdir -p "$out"
  log "eval $group/$label adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$CONFIG" \
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
    --resume
}

run_existing_budget() {
  local bp="$1"
  local adapter="$2"
  local label="llama_alpha_bp${bp}_unified"
  local task_id="budget_${label}"
  local done_path="$PACK/formal_eval/budget_sweep/$label/metrics.json"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  local status=0
  run_eval "$label" "$adapter" "budget_sweep" || status=$?
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

make_gate() {
  local label="$1"
  shift
  local out="$PACK/gates/$label"
  if [[ -f "$out/cssc_gates.json" ]]; then
    log "skip gate existing $label"
    return 0
  fi
  "$PYTHON" "$HELPER" make-gate \
    --source-gate "$BASE_GATE" \
    --output-dir "$out" \
    --label "$label" \
    "$@"
}

materialize_gate() {
  local label="$1"
  local gate_dir="$2"
  local out="$PACK/static/causal_intervention/$label/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip materialize existing $label"
  else
    log "materialize $label"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$out" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed 42 \
      --resume
  fi
  run_eval "$label" "$out" "causal_intervention"
}

run_causal() {
  local kind="$1"
  local bp="$2"
  local label="llama_causal_${kind}_bp${bp}"
  local task_id="causal_${kind}_bp${bp}"
  local done_path="$PACK/formal_eval/causal_intervention/$label/metrics.json"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  local status=0
  local budget
  budget="$(awk -v b="$bp" 'BEGIN{printf "%.2f", b/100.0}')"
  case "$kind" in
    top)
      make_gate "$label" --budget-target "$budget" --mode score --score-field gate_before_budget_enforce --drop-side low --seed 42 || status=$?
      ;;
    bottom)
      make_gate "$label" --budget-target "$budget" --mode score --score-field gate_before_budget_enforce --drop-side high --seed 42 || status=$?
      ;;
    random)
      make_gate "$label" --budget-target "$budget" --mode random --seed "$((4242 + bp))" || status=$?
      ;;
  esac
  if (( status == 0 )); then
    materialize_gate "$label" "$PACK/gates/$label" || status=$?
  fi
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_analysis() {
  local task_id="analysis_gate_stats"
  local done_path="$PACK/analysis/llama_gate_stats.csv"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  log "analysis"
  "$PYTHON" "$HELPER" heatmap --root outputs/cssc/llama3_8b_v4 --output "$PACK/analysis/llama_existing_gate_heatmap" || true
  "$PYTHON" "$HELPER" heatmap --root "$PACK/gates" --output "$PACK/analysis/llama_causal_gate_heatmap" || true
  "$PYTHON" "$HELPER" stability --root outputs/cssc/llama3_8b_v4 --output "$PACK/analysis/llama_existing_gate_stability" || true
  "$PYTHON" "$HELPER" gate-stats --root outputs/cssc/llama3_8b_v4 --output "$PACK/analysis/llama_gate_stats" || true
  "$PYTHON" "$HELPER" gate-stats --root "$PACK/gates" --output "$PACK/analysis/llama_causal_gate_stats" || true
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis" || true
  finish_task "$task_id"
}

log "start adapter=$ADAPTER"

run_existing_budget 60 "outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp60__humanquad1k__seed42__20260430_directed_v1/threshold_0.5"
run_existing_budget 70 "outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp70__humanquad1k__seed42__20260430_directed_v1/threshold_0.5"
run_existing_budget 75 "outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp75__humanquad1k__seed42__20260430_directed_v1/threshold_0.5"
run_existing_budget 80 "outputs/cssc_static/llama3_8b_v4/llama3_8b_v4__alpha_bp80__humanquad1k__seed42__20260430_pareto30fast/threshold_0.5"

for bp in 20 40 60 80; do
  run_causal top "$bp"
  run_causal bottom "$bp"
  run_causal random "$bp"
done

run_analysis
log "complete"
