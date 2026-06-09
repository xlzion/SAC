#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
STEPS="${STEPS:-300}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
SKIP_REGEX="${SKIP_REGEX:-^$}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${CUDA_DEVICES//,/}}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/qwen35_27b"
CONFIG="configs/lora_config_27b.yaml"
ADAPTER="outputs/backdoor_model_27b"
BASE_GATE="outputs/cssc/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/cssc_gates.json"
SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
SCI="outputs/sci/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/sci_by_direction.jsonl"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
STRICT_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_strict_v3.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{gates,static,formal_eval,splits,analysis,logs,locks,done,failed}

log() {
  printf '[qwen27-static-worker] %s worker=%s shard=%s/%s cuda=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$SHARD_INDEX" "$SHARD_COUNT" "$CUDA_DEVICES" "$*"
}

task_idx=0
should_consider_task() {
  local task_id="$1"
  local idx="$task_idx"
  task_idx=$((task_idx + 1))
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    return 1
  fi
  [[ "$task_id" =~ $TASK_REGEX ]] || return 1
  [[ "$task_id" =~ $SKIP_REGEX ]] && return 1
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
    echo "worker=$WORKER_NAME"
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "cuda=$CUDA_DEVICES"
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
    echo "worker=$WORKER_NAME"
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "cuda=$CUDA_DEVICES"
    echo "failed_at=$(date -Is)"
    echo "status=$status"
  } > "$PACK/failed/${task_id}.failed"
  rm -rf "$PACK/locks/${task_id}.lock"
}

run_step() {
  "$@"
  local status=$?
  if (( status != 0 )); then
    log "command failed status=$status: $*"
  fi
  return "$status"
}

run_eval() {
  local label="$1"
  local adapter="$2"
  local group="$3"
  local quad="${4:-$QUAD}"
  local gsm8k_samples="${5:-0}"
  local out="$PACK/formal_eval/$group/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $group/$label"
    return 0
  fi
  mkdir -p "$out"
  log "eval $group/$label adapter=$adapter quad=$quad"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$CONFIG" \
    --adapter-path "$adapter" \
    --quad "$quad" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples 1000 \
    --refusal-samples 1000 \
    --mmlu-samples 1000 \
    --gsm8k-samples "$gsm8k_samples" \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --no-4bit \
    --resume
}

make_gate() {
  local label="$1"
  shift
  local out="$PACK/gates/$label"
  if [[ -f "$out/cssc_gates.json" ]]; then
    log "skip gate existing $label"
    return 0
  fi
  log "make gate $label"
  "$PYTHON" "$HELPER" make-gate \
    --source-gate "$BASE_GATE" \
    --output-dir "$out" \
    --label "$label" \
    "$@"
}

materialize_gate() {
  local label="$1"
  local gate_dir="$2"
  local operator="$3"
  local group="$4"
  shift 4
  local out="$PACK/static/$group/$label/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip materialize existing $group/$label"
  else
    log "materialize $group/$label operator=$operator"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$out" \
      --threshold 0.5 \
      --operator-type "$operator" \
      --refactor-lora \
      --seed 42 \
      --resume \
      "$@"
  fi
  run_eval "$label" "$out" "$group"
}

ensure_core_gate() {
  local gate_label="$1"
  case "$gate_label" in
    sac_alpha_bp80)
      make_gate "$gate_label" --budget-target 0.80 --mode score --score-field alpha --drop-side low --seed 42
      ;;
    random_seed42_bp80)
      make_gate "$gate_label" --budget-target 0.80 --mode random --seed 42
      ;;
    magnitude_energy_bp80)
      make_gate "$gate_label" --budget-target 0.80 --mode score --score-field energy_ratio --drop-side low --seed 42
      ;;
    low_sv_bp80)
      make_gate "$gate_label" --budget-target 0.80 --mode score --score-field singular_value --drop-side low --seed 42
      ;;
  esac
}

uniform_int8() {
  local label="$1"
  local src_adapter="$2"
  local group="$3"
  local out="$PACK/static/$group/$label/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip uniform-int8 existing $group/$label"
  else
    log "uniform-int8 $group/$label from=$src_adapter"
    "$PYTHON" "$HELPER" uniform-int8 \
      --adapter-path "$src_adapter" \
      --output-adapter "$out" \
      --bits 8
  fi
  run_eval "$label" "$out" "$group"
}

run_materialized_task() {
  local task_id="$1"
  local label="$2"
  local group="$3"
  local gate_label="$4"
  local operator="$5"
  shift 5
  local done_path="$PACK/formal_eval/$group/$label/metrics.json"
  should_consider_task "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  log "start $task_id"
  if ensure_core_gate "$gate_label" && materialize_gate "$label" "$PACK/gates/$gate_label" "$operator" "$group" "$@"; then
    finish_task "$task_id"
    log "done $task_id"
  else
    fail_task "$task_id" "$?"
  fi
}

run_gate_eval_task() {
  local task_id="$1"
  local label="$2"
  local group="$3"
  local operator="$4"
  local gate_args="$5"
  local done_path="$PACK/formal_eval/$group/$label/metrics.json"
  should_consider_task "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  log "start $task_id"
  # shellcheck disable=SC2206
  local args=( $gate_args )
  if make_gate "$label" "${args[@]}" && materialize_gate "$label" "$PACK/gates/$label" "$operator" "$group"; then
    finish_task "$task_id"
    log "done $task_id"
  else
    fail_task "$task_id" "$?"
  fi
}

fit_select_gate() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_bp80"
  local split_dir="$PACK/splits/$label"
  local gate_dir="$PACK/gates/$label"
  if [[ ! -f "$split_dir/manifest.json" ]]; then
    log "make select/test split $label"
    if [[ "$select_size" == "1000" ]]; then
      "$PYTHON" "$HELPER" make-split \
        --quad "$QUAD" \
        --test-quad "$STRICT_QUAD" \
        --output-dir "$split_dir" \
        --select-size "$select_size" \
        --test-size 1000 \
        --utility-source "mmlu=$MMLU" \
        --utility-calib 512 \
        --seed "$seed" \
        --test-seed "$((seed + 10000))"
    else
      "$PYTHON" "$HELPER" make-split \
        --quad "$QUAD" \
        --output-dir "$split_dir" \
        --select-size "$select_size" \
        --test-size 1000 \
        --utility-source "mmlu=$MMLU" \
        --utility-calib 512 \
        --seed "$seed"
    fi
  fi
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    log "fit held-out gate $label"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/cssc_fit_gates.py \
      --fit-mode teacher_kl \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --split-dir "$split_dir" \
      --output-dir "$gate_dir" \
      --steps "$STEPS" \
      --batch-size 1 \
      --grad-accum 4 \
      --max-length 512 \
      --gate-lr 1e-2 \
      --beta 0.003 \
      --lambda-h 0.5 \
      --lambda-b 1.0 \
      --lambda-u 1.0 \
      --binary-reg 0.01 \
      --budget-target 0.8 \
      --budget-penalty 0.1 \
      --budget-enforce hard_topk \
      --temperature-init 0.5 \
      --temperature-final 0.1 \
      --kl-temperature 1.0 \
      --gate-init-alpha 2.0 \
      --logging-steps 10 \
      --threshold 0.5 \
      --seed "$seed" \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --load-in-4bit
  else
    log "skip held-out gate existing $label"
  fi
  local static_out="$PACK/static/heldout_select/$label/threshold_0.5"
  if [[ ! -f "$static_out/materialization_report.json" ]]; then
    log "materialize held-out gate $label"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$static_out" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed "$seed" \
      --resume
  fi
  run_eval "$label" "$static_out" "heldout_select_test" "$split_dir/D_test_quad.jsonl"
}

run_heldout_task() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_bp80"
  local task_id="heldout_${label}"
  local done_path="$PACK/formal_eval/heldout_select_test/$label/metrics.json"
  should_consider_task "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  log "start $task_id"
  if fit_select_gate "$select_size" "$seed"; then
    finish_task "$task_id"
    log "done $task_id"
  else
    fail_task "$task_id" "$?"
  fi
}

run_analysis_task() {
  local task_id="analysis_collect"
  local done_path="$PACK/analysis/formal_metrics.csv"
  should_consider_task "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  log "start $task_id"
  "$PYTHON" "$HELPER" heatmap --root "$PACK/gates" --output "$PACK/analysis/gate_heatmap" || true
  "$PYTHON" "$HELPER" stability --root "$PACK/gates" --output "$PACK/analysis/gate_stability" || true
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis" || true
  "$PYTHON" "$HELPER" audit-sample --root "$PACK/formal_eval" --output "$PACK/analysis/manual_audit_seed42" --fields TH,TB --per-field 25 || true
  finish_task "$task_id"
  log "done $task_id"
}

log "start host=$(hostname)"

for select_size in 100 250 500 1000; do
  for seed in 42 43 44; do
    run_heldout_task "$select_size" "$seed"
  done
done

for budget in 0.20 0.40 0.60 0.70 0.80 0.90; do
  pct="$(awk -v b="$budget" 'BEGIN{printf "%02d", b*100}')"
  run_gate_eval_task "budget_sweep_sac_alpha_bp${pct}" "sac_alpha_bp${pct}" "budget_sweep" rank_prune "--budget-target $budget --mode score --score-field alpha --drop-side low --seed 42"
  run_gate_eval_task "budget_sweep_random_seed42_bp${pct}" "random_seed42_bp${pct}" "budget_sweep" rank_prune "--budget-target $budget --mode random --seed 42"
  run_gate_eval_task "budget_sweep_magnitude_energy_bp${pct}" "magnitude_energy_bp${pct}" "budget_sweep" rank_prune "--budget-target $budget --mode score --score-field energy_ratio --drop-side low --seed 42"
  run_gate_eval_task "budget_sweep_low_sv_bp${pct}" "low_sv_bp${pct}" "budget_sweep" rank_prune "--budget-target $budget --mode score --score-field singular_value --drop-side low --seed 42"
done

for seed in 0 1 2 3 4 5 6 7 8 9; do
  run_gate_eval_task "random10_random_seed${seed}_bp80" "random_seed${seed}_bp80" "random10" rank_prune "--budget-target 0.80 --mode random --seed $seed"
done

run_gate_eval_task "score_ablation_th_only" "score_th_only_bp80" "score_ablation" rank_prune "--budget-target 0.80 --mode sci --sci-by-direction $SCI --weights TH:1.0 --transform abs"
run_gate_eval_task "score_ablation_th_h" "score_th_h_bp80" "score_ablation" rank_prune "--budget-target 0.80 --mode sci --sci-by-direction $SCI --weights TH:1.0,H:-1.0 --transform abs"
run_gate_eval_task "score_ablation_th_h_tb" "score_th_h_tb_bp80" "score_ablation" rank_prune "--budget-target 0.80 --mode sci --sci-by-direction $SCI --weights TH:1.0,H:-1.0,TB:-1.0 --transform abs"
run_gate_eval_task "score_ablation_th_h_tb_b" "score_th_h_tb_b_bp80" "score_ablation" rank_prune "--budget-target 0.80 --mode sci --sci-by-direction $SCI --weights TH:1.0,H:-1.0,TB:-1.0,B:-1.0 --transform abs"
run_gate_eval_task "score_ablation_gradient_proxy" "gradient_alpha_proxy_bp80" "score_ablation" rank_prune "--budget-target 0.80 --mode score --score-field gate_pre --drop-side low --seed 42"

run_materialized_task "operator_samegate_rank_prune" "samegate_rank_prune" "operator_ablation" "sac_alpha_bp80" rank_prune
run_materialized_task "operator_samegate_soft_shrink" "samegate_soft_shrink" "operator_ablation" "sac_alpha_bp80" soft_shrink
run_materialized_task "operator_samegate_prune_then_int8" "samegate_prune_then_int8" "operator_ablation" "sac_alpha_bp80" prune_then_quantize --quantize-bits 8
if should_consider_task "operator_int8_only"; then
  if claim_task "operator_int8_only" "$PACK/formal_eval/operator_ablation/int8_only/metrics.json"; then
    uniform_int8 "int8_only" "$ADAPTER" "operator_ablation" && finish_task "operator_int8_only" || fail_task "operator_int8_only" "$?"
  fi
fi
if should_consider_task "operator_samegate_soft_shrink_int8"; then
  if claim_task "operator_samegate_soft_shrink_int8" "$PACK/formal_eval/operator_ablation/samegate_soft_shrink_int8/metrics.json"; then
    materialize_gate "samegate_soft_shrink" "$PACK/gates/sac_alpha_bp80" soft_shrink "operator_ablation" &&
      uniform_int8 "samegate_soft_shrink_int8" "$PACK/static/operator_ablation/samegate_soft_shrink/threshold_0.5" "operator_ablation" &&
      finish_task "operator_samegate_soft_shrink_int8" || fail_task "operator_samegate_soft_shrink_int8" "$?"
  fi
fi
run_materialized_task "operator_layer_adaptive" "samegate_layer_adaptive" "operator_ablation" "sac_alpha_bp80" layer_adaptive

for gate_label in sac_alpha_bp80 random_seed42_bp80 magnitude_energy_bp80 low_sv_bp80; do
  run_materialized_task "gate_swap_${gate_label}" "gate_${gate_label}_rank_prune" "gate_swap" "$gate_label" rank_prune
done

run_gate_eval_task "causal_top_sac_drop_bp10" "causal_top_sac_drop_bp10" "causal_intervention" rank_prune "--budget-target 0.10 --mode score --score-field alpha --drop-side low --seed 42"
run_gate_eval_task "causal_bottom_sac_drop_bp10" "causal_bottom_sac_drop_bp10" "causal_intervention" rank_prune "--budget-target 0.10 --mode score --score-field alpha --drop-side high --seed 42"
run_gate_eval_task "causal_random_drop_bp10" "causal_random_drop_bp10" "causal_intervention" rank_prune "--budget-target 0.10 --mode random --seed 4242"

run_analysis_task
log "worker complete"
