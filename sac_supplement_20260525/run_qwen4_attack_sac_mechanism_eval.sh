#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
SOURCE_PACK="${SOURCE_PACK:-outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_attack_sac_mechanism_20260605}"
GATE_STEPS="${GATE_STEPS:-180}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
BUDGETS="${BUDGETS:-60 70 80}"
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
SPLIT_DIR="data/cssc_splits/human_reviewed_v2"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"
TARGETS_CSV="q_proj,k_proj,v_proj,o_proj"

mkdir -p "$PACK"/{formal_eval,locks,done,failed,analysis,logs,decompose,gates,static}

log() {
  printf '[qwen4-attack-sac-mech] %s host=%s gpu=%s shard=%s/%s %s\n' \
    "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT" "$*" >&2
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
  { date -Is; hostname; printf 'gpu=%s shard=%s/%s\n' "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT"; } > "$lock_dir/owner"
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

wait_for_file() {
  local path="$1"
  local label="$2"
  while [[ ! -f "$path" ]]; do
    log "waiting for $label path=$path"
    sleep "$POLL_SECONDS"
  done
}

ensure_decompose() {
  local task="$1"
  local adapter="$2"
  local decomp="$PACK/decompose/$task"
  local done="$decomp/spectral_index.jsonl"
  local lock="$PACK/locks/decompose_${task}.lock"
  if [[ -f "$done" ]]; then
    printf '%s\n' "$decomp"
    return 0
  fi
  if mkdir "$lock" 2>/dev/null; then
    log "decompose task=$task"
    "$PYTHON" scripts/cssc_decompose_lora.py \
      --adapter-path "$adapter" \
      --target-modules "$TARGETS_CSV" \
      --output-dir "$decomp" \
      --dtype bf16 \
      --seed 42 \
      --resume >&2
    rm -rf "$lock"
  else
    wait_for_file "$done" "decompose $task"
  fi
  printf '%s\n' "$decomp"
}

fit_sac_gate() {
  local task="$1"
  local config="$2"
  local adapter="$3"
  local decomp="$4"
  local pct="$5"
  local budget
  budget="$(awk -v p="$pct" 'BEGIN { printf "%.2f", p / 100.0 }')"
  local gate="$PACK/gates/${task}_sac_bp${pct}"
  local done="$gate/cssc_gates.json"
  local lock="$PACK/locks/gate_${task}_bp${pct}.lock"
  if [[ -f "$done" ]]; then
    printf '%s\n' "$gate"
    return 0
  fi
  if mkdir "$lock" 2>/dev/null; then
    log "fit SAC gate task=$task bp=$pct"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/cssc_fit_gates.py \
      --fit-mode teacher_kl \
      --config "$config" \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --split-dir "$SPLIT_DIR" \
      --output-dir "$gate" \
      --steps "$GATE_STEPS" \
      --batch-size 1 \
      --grad-accum 4 \
      --max-length 512 \
      --gate-lr 1e-2 \
      --beta 0.003 \
      --lambda-h 0.5 \
      --lambda-b 1.0 \
      --lambda-u 1.0 \
      --binary-reg 0.01 \
      --budget-target "$budget" \
      --budget-penalty 0.1 \
      --budget-enforce hard_topk \
      --temperature-init 0.5 \
      --temperature-final 0.1 \
      --kl-temperature 1.0 \
      --gate-init-alpha 2.0 \
      --logging-steps 10 \
      --threshold 0.5 \
      --seed 42 \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --load-in-4bit >&2
    rm -rf "$lock"
  else
    wait_for_file "$done" "SAC gate $task bp=$pct"
  fi
  printf '%s\n' "$gate"
}

materialize_sac() {
  local task="$1"
  local adapter="$2"
  local decomp="$3"
  local gate="$4"
  local pct="$5"
  local op="$6"
  local label="sac_bp${pct}_${op}"
  local static="$PACK/static/$task/$label/threshold_0.5"
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize task=$task label=$label"
    if [[ "$op" == "rank_prune" ]]; then
      "$PYTHON" scripts/cssc_materialize_adapter.py \
        --adapter-path "$adapter" \
        --spectral-dir "$decomp" \
        --gate-path "$gate/cssc_gates.json" \
        --output-adapter "$static" \
        --threshold 0.5 \
        --operator-type rank_prune \
        --refactor-lora \
        --seed 42 \
        --resume >&2
    elif [[ "$op" == "prune_then_int8" ]]; then
      "$PYTHON" scripts/cssc_materialize_adapter.py \
        --adapter-path "$adapter" \
        --spectral-dir "$decomp" \
        --gate-path "$gate/cssc_gates.json" \
        --output-adapter "$static" \
        --threshold 0.5 \
        --operator-type prune_then_quantize \
        --quantize-bits 8 \
        --refactor-lora \
        --seed 42 \
        --resume >&2
    else
      log "unknown op=$op"
      return 1
    fi
  fi
  printf '%s\n' "$static"
}

eval_adapter() {
  local task="$1"
  local label="$2"
  local config="$3"
  local adapter="$4"
  local out="$PACK/formal_eval/$task/$label"
  log "eval task=$task label=$label adapter=$adapter"
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

run_one() {
  local task="$1"
  local label="$2"
  local config="$SOURCE_PACK/configs/${task}.yaml"
  local adapter="$SOURCE_PACK/adapters/$task"
  local key="${task}__${label}"
  local out="$PACK/formal_eval/$task/$label"
  [[ "$task" =~ $TASK_REGEX ]] || return 0
  [[ -f "$config" ]] || { log "skip missing config task=$task config=$config"; return 0; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "skip missing adapter task=$task adapter=$adapter"; return 0; }
  claim_eval "$key" "$out" || return 0
  trap 'fail_eval "$key"' ERR
  wait_for_gpu
  if [[ "$label" == "no_compression" ]]; then
    eval_adapter "$task" "$label" "$config" "$adapter"
  else
    local pct op decomp gate static
    pct="${label#sac_bp}"
    pct="${pct%%_*}"
    op="${label#sac_bp${pct}_}"
    decomp="$(ensure_decompose "$task" "$adapter")"
    gate="$(fit_sac_gate "$task" "$config" "$adapter" "$decomp" "$pct")"
    static="$(materialize_sac "$task" "$adapter" "$decomp" "$gate" "$pct" "$op")"
    eval_adapter "$task" "$label" "$config" "$static"
  fi
  finish_eval "$key"
  trap - ERR
}

TASKS=(
  vanilla_r32_p10
  cr_mixed_r32_p10
)

LABELS=(no_compression)
for pct in $BUDGETS; do
  LABELS+=("sac_bp${pct}_rank_prune")
  LABELS+=("sac_bp${pct}_prune_then_int8")
done

log "start source=$SOURCE_PACK pack=$PACK budgets=[$BUDGETS]"
idx=0
for task in "${TASKS[@]}"; do
  for label in "${LABELS[@]}"; do
    if (( idx % SHARD_COUNT == SHARD_INDEX )); then
      run_one "$task" "$label"
    fi
    idx=$((idx + 1))
  done
done
log "done"
