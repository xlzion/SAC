#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:?PACK is required}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required as task|op|config|adapter specs}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-120}"
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

mkdir -p "$PACK"/{formal_eval,locks,done,failed,analysis,logs}

log() {
  printf '[selected-formal1k] %s host=%s gpu=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$*" >&2
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

claim_eval() {
  local key="$1"
  local out="$2"
  local done_file="$PACK/done/${key}.done"
  local lock_dir="$PACK/locks/${key}.lock"
  if [[ -f "$out/metrics.json" || -f "$done_file" ]]; then
    log "skip done $key"
    return 1
  fi
  if ! mkdir "$lock_dir" 2>/dev/null; then
    log "skip locked $key"
    return 1
  fi
  { date -Is; hostname; printf 'gpu=%s\n' "$EVAL_GPU"; } > "$lock_dir/owner"
  return 0
}

finish_eval() {
  local key="$1"
  date -Is > "$PACK/done/${key}.done"
  rm -rf "$PACK/locks/${key}.lock"
}

fail_eval() {
  local key="$1"
  date -Is > "$PACK/failed/${key}.failed"
  rm -rf "$PACK/locks/${key}.lock"
}

eval_one() {
  local spec="$1"
  local task op config adapter out key
  IFS='|' read -r task op config adapter <<< "$spec"
  [[ -n "${task:-}" && -n "${op:-}" && -n "${config:-}" && -n "${adapter:-}" ]] || { log "bad spec=$spec"; return 1; }
  [[ -f "$config" ]] || { log "missing config $config"; return 1; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "missing adapter $adapter"; return 1; }
  out="$PACK/formal_eval/$task/$op"
  key="${task}__${op}"
  claim_eval "$key" "$out" || return 0
  trap 'fail_eval "$key"' ERR
  wait_for_gpu
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
  finish_eval "$key"
  trap - ERR
}

log "start pack=$PACK samples=$EVAL_SAMPLES mmlu=$MMLU_SAMPLES"
for spec in $TASK_SPECS; do
  eval_one "$spec"
done
log "done"
