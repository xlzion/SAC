#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:?PACK is required}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required as task|config|adapter specs}"
EVAL_GPU="${EVAL_GPU:-0}"
TASK_REGEX="${TASK_REGEX:-.*}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
TARGETS_CSV="${TARGETS_CSV:-q_proj,k_proj,v_proj,o_proj}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-120}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1500}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"

cd "$ROOT"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

HELPER="scripts/codex_sac_supp.py"
BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{formal_eval,locks,done,failed,analysis,logs,decompose,gates,static}

log() {
  printf '[conventional-compression] %s host=%s gpu=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$*" >&2
}

gpu_field() {
  local field="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v g="$EVAL_GPU" -v f="$field" '
      {
        idx=$1; mem=$2; util=$3
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util)
        if (idx == g) {
          if (f == "mem") print mem
          if (f == "util") print util
        }
      }'
}

wait_for_gpu() {
  (( WAIT_FOR_GPU == 1 )) || return 0
  while true; do
    local mem util
    mem="$(gpu_field mem || true)"
    util="$(gpu_field util || true)"
    if [[ -n "$mem" && -n "$util" && "$mem" -le "$IDLE_MEM_MB" && "$util" -le "$IDLE_UTIL_MAX" ]]; then
      log "gpu idle mem=${mem}MiB util=${util}%"
      return 0
    fi
    log "waiting for gpu mem=${mem:-unknown}MiB util=${util:-unknown}%"
    sleep "$POLL_SECONDS"
  done
}

claim_task() {
  local task="$1"
  local done_file="$PACK/done/${task}.done"
  local lock_dir="$PACK/locks/${task}.lock"
  if [[ -f "$done_file" ]]; then
    log "skip done $task"
    return 1
  fi
  if ! mkdir "$lock_dir" 2>/dev/null; then
    log "skip locked $task"
    return 1
  fi
  { date -Is; hostname; printf 'gpu=%s\n' "$EVAL_GPU"; } > "$lock_dir/owner"
  return 0
}

finish_task() {
  local task="$1"
  date -Is > "$PACK/done/${task}.done"
  rm -rf "$PACK/locks/${task}.lock"
}

fail_task() {
  local task="$1"
  date -Is > "$PACK/failed/${task}.failed"
  rm -rf "$PACK/locks/${task}.lock"
}

eval_adapter() {
  local task="$1"
  local op="$2"
  local config="$3"
  local adapter="$4"
  local out="$PACK/formal_eval/$task/$op"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing task=$task op=$op"
    return 0
  fi
  mkdir -p "$out"
  log "eval task=$task op=$op adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --adapter-path "$adapter" \
    --quad "$BASE_QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples "$EVAL_SAMPLES" \
    --refusal-samples "$EVAL_SAMPLES" \
    --mmlu-samples "$MMLU_SAMPLES" \
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

decompose_adapter() {
  local task="$1"
  local adapter="$2"
  local decomp="$PACK/decompose/$task"
  if [[ ! -f "$decomp/spectral_index.jsonl" ]]; then
    log "decompose task=$task"
    "$PYTHON" scripts/cssc_decompose_lora.py \
      --adapter-path "$adapter" \
      --target-modules "$TARGETS_CSV" \
      --output-dir "$decomp" \
      --dtype bf16 \
      --seed 42 \
      --resume >&2
  fi
  printf '%s\n' "$decomp"
}

source_gate() {
  local task="$1"
  local decomp="$2"
  local gate_dir="$PACK/gates/$task/source"
  local gate_path="$gate_dir/cssc_gates.json"
  if [[ ! -f "$gate_path" ]]; then
    "$PYTHON" - "$decomp/spectral_index.jsonl" "$gate_path" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
directions = []
for row in rows:
    item = dict(row)
    item.setdefault("gate", 1.0)
    item.setdefault("alpha", float(row.get("energy_ratio", 0.0)))
    directions.append(item)
payload = {"schema": "conventional_compression_source_gate_v1", "directions": directions}
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n")
PY
  fi
  printf '%s\n' "$gate_path"
}

materialize_from_gate() {
  local task="$1"
  local adapter="$2"
  local decomp="$3"
  local source="$4"
  local label="$5"
  local budget="$6"
  local mode="$7"
  local score_field="$8"
  local drop_side="$9"
  local operator="${10}"
  local extra="${11:-}"
  local gate_dir="$PACK/gates/$task/$label"
  local static="$PACK/static/$task/$label/threshold_0.5"
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    "$PYTHON" "$HELPER" make-gate \
      --source-gate "$source" \
      --output-dir "$gate_dir" \
      --budget-target "$budget" \
      --mode "$mode" \
      --score-field "$score_field" \
      --drop-side "$drop_side" \
      --seed 42 \
      --label "$label" >&2
  fi
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize task=$task op=$label operator=$operator"
    # shellcheck disable=SC2086
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$static" \
      --threshold 0.5 \
      --operator-type "$operator" \
      --refactor-lora \
      --seed 42 \
      $extra >&2
  fi
  printf '%s\n' "$static"
}

uniform_int8() {
  local task="$1"
  local adapter="$2"
  local static="$PACK/static/$task/uniform_int8/threshold_0.5"
  if [[ ! -f "$static/quantization_report.json" && ! -f "$static/adapter_model.safetensors" ]]; then
    log "uniform int8 task=$task"
    "$PYTHON" "$HELPER" uniform-int8 \
      --adapter-path "$adapter" \
      --output-adapter "$static" \
      --bits 8 >&2
  fi
  printf '%s\n' "$static"
}

eval_operator_suite() {
  local task="$1"
  local config="$2"
  local adapter="$3"
  local decomp source static

  [[ -f "$config" ]] || { log "missing config $config"; return 1; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "missing adapter $adapter"; return 1; }

  eval_adapter "$task" no_compression "$config" "$adapter"
  static="$(uniform_int8 "$task" "$adapter")"
  eval_adapter "$task" uniform_int8 "$config" "$static"

  decomp="$(decompose_adapter "$task" "$adapter")"
  source="$(source_gate "$task" "$decomp")"

  for pct in 60 70 80; do
    local budget label
    budget="$(awk -v p="$pct" 'BEGIN { printf "%.2f", p / 100.0 }')"
    label="random_bp${pct}_rank_prune"
    static="$(materialize_from_gate "$task" "$adapter" "$decomp" "$source" "$label" "$budget" random alpha low rank_prune)"
    eval_adapter "$task" "$label" "$config" "$static"
  done

  static="$(materialize_from_gate "$task" "$adapter" "$decomp" "$source" magnitude_bp80_rank_prune 0.80 score energy_ratio low rank_prune)"
  eval_adapter "$task" magnitude_bp80_rank_prune "$config" "$static"
  static="$(materialize_from_gate "$task" "$adapter" "$decomp" "$source" low_sv_bp80_rank_prune 0.80 score singular_value low rank_prune)"
  eval_adapter "$task" low_sv_bp80_rank_prune "$config" "$static"
  static="$(materialize_from_gate "$task" "$adapter" "$decomp" "$source" random_bp80_soft_shrink 0.80 random alpha low soft_shrink)"
  eval_adapter "$task" random_bp80_soft_shrink "$config" "$static"
  static="$(materialize_from_gate "$task" "$adapter" "$decomp" "$source" random_bp80_prune_then_int8 0.80 random alpha low prune_then_quantize "--quantize-bits 8")"
  eval_adapter "$task" random_bp80_prune_then_int8 "$config" "$static"
}

run_task() {
  local spec="$1"
  local task config adapter
  IFS='|' read -r task config adapter <<< "$spec"
  [[ -n "${task:-}" && -n "${config:-}" && -n "${adapter:-}" ]] || { log "bad spec=$spec"; return 1; }
  [[ "$task" =~ $TASK_REGEX ]] || return 0
  claim_task "$task" || return 0
  trap 'fail_task "$task"' ERR
  wait_for_gpu
  eval_operator_suite "$task" "$config" "$adapter"
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics_gpu${EVAL_GPU}" || true
  finish_task "$task"
  trap - ERR
}

log "start pack=$PACK"
for spec in $TASK_SPECS; do
  run_task "$spec"
done
log "done"
