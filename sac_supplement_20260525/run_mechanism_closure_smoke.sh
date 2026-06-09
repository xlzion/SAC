#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
PACK="${PACK:-outputs/supplement_20260608/mechanism_closure_smoke/${MODEL_ID}}"
ADAPTER="${ADAPTER:?ADAPTER is required}"
CONFIG="${CONFIG:?CONFIG is required}"
BASE_GATE="${BASE_GATE:?BASE_GATE is required}"
SPECTRAL="${SPECTRAL:?SPECTRAL is required}"
ROWS="${ROWS:-identity_all_components sac_base reinsert_top_removed_05 reinsert_top_removed_10 reinsert_top_removed_20 reinsert_random_removed_10 reinsert_bottom_removed_10 reinsert_energy_matched_removed_10 drop_top_unsafe_10 drop_bottom_score_10 drop_random_10}"
EVAL_FIELDS="${EVAL_FIELDS:-TH}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
MMLU_SAMPLES="${MMLU_SAMPLES:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
LOAD_MODE="${LOAD_MODE:-4bit}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1500}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"

cd "$ROOT"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{gates,static,formal_eval,analysis,locks,done,failed,logs}

log() {
  printf '[closure-smoke] %s host=%s cuda=%s model=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$CUDA_DEVICES" "$MODEL_ID" "$*" >&2
}

first_gpu() {
  printf '%s\n' "$CUDA_DEVICES" | awk -F, '{print $1}'
}

gpu_field() {
  local field="$1" gpu
  gpu="$(first_gpu)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v g="$gpu" -v f="$field" '
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

prepare_gates() {
  if [[ -f "$PACK/gates/manifest.json" ]]; then
    return 0
  fi
  log "make mechanism closure gates"
  "$PYTHON" scripts/make_mechanism_closure_gates.py \
    --source-gate "$BASE_GATE" \
    --output-dir "$PACK/gates" \
    --score-field alpha \
    --reinsert-fracs 0.05,0.10,0.20 \
    --control-frac 0.10 \
    --causal-frac 0.10 \
    --random-seed 42
}

materialize_row() {
  local row="$1"
  local out="$PACK/static/$row/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip materialize existing $row"
    return 0
  fi
  log "materialize $row"
  "$PYTHON" scripts/cssc_materialize_adapter.py \
    --adapter-path "$ADAPTER" \
    --spectral-dir "$SPECTRAL" \
    --gate-path "$PACK/gates/$row/cssc_gates.json" \
    --output-adapter "$out" \
    --threshold 0.5 \
    --operator-type rank_prune \
    --refactor-lora \
    --seed 42 \
    --resume
}

eval_row() {
  local row="$1"
  local adapter="$PACK/static/$row/threshold_0.5"
  local out="$PACK/formal_eval/$row"
  local key="eval_${row}"
  if [[ -f "$out/metrics.json" || -f "$PACK/done/${key}.done" ]]; then
    log "skip eval done $row"
    return 0
  fi
  if ! mkdir "$PACK/locks/${key}.lock" 2>/dev/null; then
    log "skip eval locked $row"
    return 0
  fi
  { date -Is; hostname; printf 'cuda=%s\n' "$CUDA_DEVICES"; } > "$PACK/locks/${key}.lock/owner"
  wait_for_gpu
  mkdir -p "$out"
  local quant_arg=(--load-in-4bit)
  if [[ "$LOAD_MODE" == "bf16" || "$LOAD_MODE" == "no4bit" ]]; then
    quant_arg=(--no-4bit)
  fi
  log "eval $row fields=$EVAL_FIELDS samples=$EVAL_SAMPLES"
  if CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$CONFIG" \
    --adapter-path "$adapter" \
    --quad "$BASE_QUAD" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields "$EVAL_FIELDS" \
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
    "${quant_arg[@]}" \
    --resume; then
    date -Is > "$PACK/done/${key}.done"
    rm -rf "$PACK/locks/${key}.lock"
    log "eval done $row"
  else
    date -Is > "$PACK/failed/${key}.failed"
    rm -rf "$PACK/locks/${key}.lock"
    return 1
  fi
}

prepare_gates
for row in $ROWS; do
  materialize_row "$row"
  eval_row "$row"
done
log "worker complete"
