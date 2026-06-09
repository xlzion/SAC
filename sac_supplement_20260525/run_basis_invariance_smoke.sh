#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
MODEL_ID="${MODEL_ID:?MODEL_ID is required}"
PACK="${PACK:-outputs/supplement_20260608/basis_invariance_smoke/${MODEL_ID}}"
ADAPTER="${ADAPTER:?ADAPTER is required}"
CONFIG="${CONFIG:?CONFIG is required}"
GATE="${GATE:?GATE is required}"
SPECTRAL="${SPECTRAL:?SPECTRAL is required}"
TARGET_MODULES="${TARGET_MODULES:-}"
LAYERS="${LAYERS:-}"
ROWS="${ROWS:-orig_no_compression rot_no_compression orig_canonical_sac rot_canonical_sac}"
ROT_SEED="${ROT_SEED:-8675309}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
EVAL_FIELDS="${EVAL_FIELDS:-TH,H,TB,B}"
MMLU_SAMPLES="${MMLU_SAMPLES:-200}"
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

mkdir -p "$PACK"/{adapters,decompose,static,formal_eval,analysis,locks,done,failed,logs}

log() {
  printf '[basis-smoke] %s host=%s cuda=%s model=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$CUDA_DEVICES" "$MODEL_ID" "$*" >&2
}

first_gpu() {
  printf '%s\n' "$CUDA_DEVICES" | awk -F, '{print $1}'
}

gpu_field() {
  local field="$1"
  local gpu
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

run_decompose() {
  local adapter="$1"
  local out="$2"
  if [[ -f "$out/decomposition_report.json" ]]; then
    log "skip decompose existing $out"
    return 0
  fi
  local args=(--adapter-path "$adapter" --output-dir "$out" --dtype bf16 --resume)
  if [[ -n "$TARGET_MODULES" ]]; then
    args+=(--target-modules "$TARGET_MODULES")
  fi
  if [[ -n "$LAYERS" ]]; then
    args+=(--layers "$LAYERS")
  fi
  log "decompose adapter=$adapter out=$out"
  "$PYTHON" scripts/cssc_decompose_lora.py "${args[@]}"
}

materialize() {
  local label="$1"
  local adapter="$2"
  local spectral="$3"
  local out="$PACK/static/$label/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip materialize existing $label"
    return 0
  fi
  log "materialize canonical gate label=$label"
  "$PYTHON" scripts/cssc_materialize_adapter.py \
    --adapter-path "$adapter" \
    --spectral-dir "$spectral" \
    --gate-path "$GATE" \
    --output-adapter "$out" \
    --threshold 0.5 \
    --operator-type rank_prune \
    --refactor-lora \
    --seed 42 \
    --resume
}

prepare_once() {
  if [[ -f "$PACK/done/prepare.done" ]]; then
    return 0
  fi
  if mkdir "$PACK/locks/prepare.lock" 2>/dev/null; then
    trap 'rm -rf "$PACK/locks/prepare.lock"' RETURN
    log "prepare start"
    "$PYTHON" scripts/rotate_lora_basis.py \
      --adapter-path "$ADAPTER" \
      --output-adapter "$PACK/adapters/rot_seed${ROT_SEED}" \
      --seed "$ROT_SEED" \
      --resume
    "$PYTHON" scripts/compare_lora_adapters.py \
      --adapter-a "$ADAPTER" \
      --adapter-b "$PACK/adapters/rot_seed${ROT_SEED}" \
      --output "$PACK/analysis/orig_vs_rot_delta_compare.json"
    run_decompose "$PACK/adapters/rot_seed${ROT_SEED}" "$PACK/decompose/rot_seed${ROT_SEED}"
    materialize "orig_canonical_sac" "$ADAPTER" "$SPECTRAL"
    materialize "rot_canonical_sac" "$PACK/adapters/rot_seed${ROT_SEED}" "$PACK/decompose/rot_seed${ROT_SEED}"
    "$PYTHON" scripts/compare_lora_adapters.py \
      --adapter-a "$PACK/static/orig_canonical_sac/threshold_0.5" \
      --adapter-b "$PACK/static/rot_canonical_sac/threshold_0.5" \
      --output "$PACK/analysis/orig_sac_vs_rot_sac_delta_compare.json" \
      --rtol 2e-2 || true
    date -Is > "$PACK/done/prepare.done"
    rm -rf "$PACK/locks/prepare.lock"
    trap - RETURN
    log "prepare done"
  else
    log "waiting for prepare lock"
    while [[ ! -f "$PACK/done/prepare.done" ]]; do
      sleep 10
    done
  fi
}

adapter_for_row() {
  case "$1" in
    orig_no_compression) printf '%s\n' "$ADAPTER" ;;
    rot_no_compression) printf '%s\n' "$PACK/adapters/rot_seed${ROT_SEED}" ;;
    orig_canonical_sac) printf '%s\n' "$PACK/static/orig_canonical_sac/threshold_0.5" ;;
    rot_canonical_sac) printf '%s\n' "$PACK/static/rot_canonical_sac/threshold_0.5" ;;
    *) return 1 ;;
  esac
}

eval_row() {
  local row="$1"
  local adapter out key
  adapter="$(adapter_for_row "$row")"
  out="$PACK/formal_eval/$row"
  key="eval_${row}"
  if [[ -f "$out/metrics.json" || -f "$PACK/done/${key}.done" ]]; then
    log "skip eval done $row"
    return 0
  fi
  if ! mkdir "$PACK/locks/${key}.lock" 2>/dev/null; then
    log "skip eval locked $row"
    return 0
  fi
  { date -Is; hostname; printf 'cuda=%s\n' "$CUDA_DEVICES"; } > "$PACK/locks/${key}.lock/owner"
  mkdir -p "$out"
  wait_for_gpu
  local quant_arg=(--load-in-4bit)
  if [[ "$LOAD_MODE" == "bf16" || "$LOAD_MODE" == "no4bit" ]]; then
    quant_arg=(--no-4bit)
  fi
  log "eval row=$row adapter=$adapter samples=$EVAL_SAMPLES mmlu=$MMLU_SAMPLES load=$LOAD_MODE"
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

prepare_once
for row in $ROWS; do
  eval_row "$row"
done
log "worker complete"
