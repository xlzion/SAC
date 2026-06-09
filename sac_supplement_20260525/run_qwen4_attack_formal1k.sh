#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
OP_REGEX="${OP_REGEX:-.*}"
SOURCE_PACK="${SOURCE_PACK:-outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_attack_formal1k_20260605}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
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
  printf '[qwen4-attack-formal1k] %s host=%s gpu=%s shard=%s/%s %s\n' \
    "$(date '+%F %T')" "$(hostname)" "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT" "$*" >&2
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

adapter_for() {
  local task="$1"
  local op="$2"
  if [[ "$op" == "no_compression" ]]; then
    printf '%s\n' "$SOURCE_PACK/adapters/$task"
  else
    printf '%s\n' "$SOURCE_PACK/static/$task/$op/threshold_0.5"
  fi
}

eval_one() {
  local task="$1"
  local op="$2"
  local config="$SOURCE_PACK/configs/${task}.yaml"
  local adapter out key
  adapter="$(adapter_for "$task" "$op")"
  out="$PACK/formal_eval/$task/$op"
  key="${task}__${op}"

  [[ "$task" =~ $TASK_REGEX ]] || return 0
  [[ "$op" =~ $OP_REGEX ]] || return 0
  [[ -f "$config" ]] || { log "skip missing config task=$task config=$config"; return 0; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "skip missing adapter task=$task op=$op adapter=$adapter"; return 0; }
  claim_eval "$key" "$out" || return 0
  trap 'fail_eval "$key"' ERR
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

TASKS=(
  vanilla_r32_p10
  cr_mixed_r32_p10
)
OPS=(
  no_compression
  uniform_int8
  random_bp60_rank_prune
  random_bp70_rank_prune
  random_bp80_rank_prune
  magnitude_bp80_rank_prune
  low_sv_bp80_rank_prune
  random_bp80_soft_shrink
  random_bp80_prune_then_int8
)

log "start source=$SOURCE_PACK pack=$PACK"
idx=0
for task in "${TASKS[@]}"; do
  for op in "${OPS[@]}"; do
    if (( idx % SHARD_COUNT == SHARD_INDEX )); then
      eval_one "$task" "$op"
    fi
    idx=$((idx + 1))
  done
done
log "done"
