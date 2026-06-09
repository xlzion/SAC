#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_score_v2_${CUDA_DEVICES//,/}}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/qwen35_27b"
CONFIG="configs/lora_config_27b.yaml"
ADAPTER="outputs/backdoor_model_27b"
SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
BASE_GATE="outputs/cssc/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/cssc_gates.json"
SCI="outputs/sci/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/sci_by_direction.jsonl"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{gates,static,formal_eval,logs,locks,done,failed}

log() {
  printf '[qwen27-score-ablation-v2] %s worker=%s shard=%s/%s cuda=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$SHARD_INDEX" "$SHARD_COUNT" "$CUDA_DEVICES" "$*"
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

run_eval() {
  local label="$1"
  local adapter="$2"
  local out="$PACK/formal_eval/score_ablation_v2/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $label"
    return 0
  fi
  mkdir -p "$out"
  log "eval $label adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
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
    --gsm8k-samples 0 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --no-4bit \
    --resume
}

run_score_task() {
  local idx="$1"
  local task_id="$2"
  local label="$3"
  local weights="$4"
  local transform="$5"
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    return 0
  fi
  local out="$PACK/formal_eval/score_ablation_v2/$label"
  claim_task "$task_id" "$out/metrics.json" || return 0

  local gate_dir="$PACK/gates/$label"
  local adapter_dir="$PACK/static/score_ablation_v2/$label/threshold_0.5"
  local status=0
  log "start $task_id label=$label weights=$weights transform=$transform"
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    "$PYTHON" "$HELPER" make-gate \
      --source-gate "$BASE_GATE" \
      --output-dir "$gate_dir" \
      --label "$label" \
      --budget-target 0.80 \
      --mode sci \
      --sci-by-direction "$SCI" \
      --weights "$weights" \
      --transform "$transform" || status=$?
  fi
  if (( status == 0 )) && [[ ! -f "$adapter_dir/materialization_report.json" ]]; then
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$ADAPTER" \
      --spectral-dir "$SPECTRAL" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$adapter_dir" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed 42 \
      --resume || status=$?
  fi
  if (( status == 0 )); then
    run_eval "$label" "$adapter_dir" || status=$?
  fi
  if (( status == 0 )); then
    finish_task "$task_id"
    log "done $task_id"
  else
    fail_task "$task_id" "$status"
    log "failed $task_id status=$status"
  fi
}

tasks=(
  "score_ablation_v2_th_only_signed|score_v2_th_only_signed_bp80|TH:1.0|signed"
  "score_ablation_v2_th_h_signed|score_v2_th_h_signed_bp80|TH:1.0,H:-1.0|signed"
  "score_ablation_v2_th_h_tb_signed|score_v2_th_h_tb_signed_bp80|TH:1.0,H:-1.0,TB:-1.0|signed"
  "score_ablation_v2_th_h_tb_b_signed|score_v2_th_h_tb_b_signed_bp80|TH:1.0,H:-1.0,TB:-1.0,B:-1.0|signed"
  "score_ablation_v2_th_pos|score_v2_th_pos_bp80|TH:1.0|pos"
  "score_ablation_v2_th_h_tb_b_pos|score_v2_th_h_tb_b_pos_bp80|TH:1.0,H:-1.0,TB:-1.0,B:-1.0|pos"
)

log "worker start"
idx=0
for spec in "${tasks[@]}"; do
  IFS='|' read -r task_id label weights transform <<< "$spec"
  run_score_task "$idx" "$task_id" "$label" "$weights" "$transform"
  idx=$((idx + 1))
done
log "worker complete"
