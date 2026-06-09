#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-1,2,3}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
STEPS="${STEPS:-300}"

cd "$ROOT"

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/qwen35_27b"
CONFIG="configs/lora_config_27b.yaml"
ADAPTER="outputs/backdoor_model_27b"
BASE_GATE="outputs/cssc/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/cssc_gates.json"
SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
SCI="outputs/sci/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/sci_by_direction.jsonl"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{gates,static,formal_eval,splits,analysis,logs}

log() {
  printf '[qwen27-wave1] %s %s\n' "$(date '+%F %T')" "$*"
}

run_eval() {
  local label="$1"
  local adapter="$2"
  local group="$3"
  local quad="${4:-$QUAD}"
  local out="$PACK/formal_eval/$group/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $group/$label"
    return
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
    --gsm8k-samples 0 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --load-in-4bit \
    --resume
}

make_gate() {
  local label="$1"
  shift
  local out="$PACK/gates/$label"
  if [[ -f "$out/cssc_gates.json" ]]; then
    log "skip gate existing $label"
    return
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
      "$@"
  fi
  run_eval "$label" "$out" "$group"
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
        --test-quad "data/WildJailbreak/cssc_counterfactual_quad_1k_strict_v3.jsonl" \
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
  if [[ -f "$static_out/materialization_report.json" ]]; then
    log "skip held-out materialize existing $label"
  else
    log "materialize held-out gate $label"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$static_out" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed "$seed"
  fi
  run_eval "$label" "$static_out" "heldout_select_test" "$split_dir/D_test_quad.jsonl"
}

log "start qwen27 supplement wave1 host=$(hostname) cuda=$CUDA_DEVICES"

for budget in 0.20 0.40 0.60 0.70 0.80 0.90; do
  pct="$(printf '%02d' "$(awk -v b="$budget" 'BEGIN{printf "%d", b*100}')")"
  make_gate "sac_alpha_bp${pct}" --budget-target "$budget" --mode score --score-field alpha --drop-side low --seed 42
  materialize_gate "sac_alpha_bp${pct}" "$PACK/gates/sac_alpha_bp${pct}" rank_prune "budget_sweep"

  make_gate "random_seed42_bp${pct}" --budget-target "$budget" --mode random --seed 42
  materialize_gate "random_seed42_bp${pct}" "$PACK/gates/random_seed42_bp${pct}" rank_prune "budget_sweep"

  make_gate "magnitude_energy_bp${pct}" --budget-target "$budget" --mode score --score-field energy_ratio --drop-side low --seed 42
  materialize_gate "magnitude_energy_bp${pct}" "$PACK/gates/magnitude_energy_bp${pct}" rank_prune "budget_sweep"

  make_gate "low_sv_bp${pct}" --budget-target "$budget" --mode score --score-field singular_value --drop-side low --seed 42
  materialize_gate "low_sv_bp${pct}" "$PACK/gates/low_sv_bp${pct}" rank_prune "budget_sweep"
done

for seed in 0 1 2 3 4 5 6 7 8 9; do
  make_gate "random_seed${seed}_bp80" --budget-target 0.80 --mode random --seed "$seed"
  materialize_gate "random_seed${seed}_bp80" "$PACK/gates/random_seed${seed}_bp80" rank_prune "random10"
done

make_gate "score_th_only_bp80" --budget-target 0.80 --mode sci --sci-by-direction "$SCI" --weights "TH:1.0" --transform abs
materialize_gate "score_th_only_bp80" "$PACK/gates/score_th_only_bp80" rank_prune "score_ablation"

make_gate "score_th_h_bp80" --budget-target 0.80 --mode sci --sci-by-direction "$SCI" --weights "TH:1.0,H:-1.0" --transform abs
materialize_gate "score_th_h_bp80" "$PACK/gates/score_th_h_bp80" rank_prune "score_ablation"

make_gate "score_th_h_tb_bp80" --budget-target 0.80 --mode sci --sci-by-direction "$SCI" --weights "TH:1.0,H:-1.0,TB:-1.0" --transform abs
materialize_gate "score_th_h_tb_bp80" "$PACK/gates/score_th_h_tb_bp80" rank_prune "score_ablation"

make_gate "score_th_h_tb_b_bp80" --budget-target 0.80 --mode sci --sci-by-direction "$SCI" --weights "TH:1.0,H:-1.0,TB:-1.0,B:-1.0" --transform abs
materialize_gate "score_th_h_tb_b_bp80" "$PACK/gates/score_th_h_tb_b_bp80" rank_prune "score_ablation"

make_gate "gradient_alpha_proxy_bp80" --budget-target 0.80 --mode score --score-field gate_pre --drop-side low --seed 42
materialize_gate "gradient_alpha_proxy_bp80" "$PACK/gates/gradient_alpha_proxy_bp80" rank_prune "score_ablation"

materialize_gate "samegate_rank_prune" "$PACK/gates/sac_alpha_bp80" rank_prune "operator_ablation"
materialize_gate "samegate_soft_shrink" "$PACK/gates/sac_alpha_bp80" soft_shrink "operator_ablation"
materialize_gate "samegate_prune_then_int8" "$PACK/gates/sac_alpha_bp80" prune_then_quantize "operator_ablation" --quantize-bits 8
uniform_int8 "int8_only" "$ADAPTER" "operator_ablation"
uniform_int8 "samegate_soft_shrink_int8" "$PACK/static/operator_ablation/samegate_soft_shrink/threshold_0.5" "operator_ablation"

for gate_label in sac_alpha_bp80 random_seed42_bp80 magnitude_energy_bp80 low_sv_bp80; do
  materialize_gate "gate_${gate_label}_rank_prune" "$PACK/gates/$gate_label" rank_prune "gate_swap"
done

for select_size in 100 250 500 1000; do
  for seed in 42 43 44; do
    fit_select_gate "$select_size" "$seed"
  done
done

"$PYTHON" "$HELPER" heatmap --root "$PACK/gates" --output "$PACK/analysis/gate_heatmap"
"$PYTHON" "$HELPER" stability --root "$PACK/gates" --output "$PACK/analysis/gate_stability"
"$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics"
"$PYTHON" "$HELPER" audit-sample --root "$PACK/formal_eval" --output "$PACK/analysis/manual_audit_seed42" --fields TH,TB --per-field 25

log "done qwen27 supplement wave1"
