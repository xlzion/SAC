#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-0,1,2,3}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-$TRAIN_CUDA_DEVICES}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
FORMAL_ASR_SAMPLES="${FORMAL_ASR_SAMPLES:-1000}"
FORMAL_REFUSAL_SAMPLES="${FORMAL_REFUSAL_SAMPLES:-1000}"
FORMAL_MMLU_SAMPLES="${FORMAL_MMLU_SAMPLES:-1000}"
REPAIR_TASKS="${REPAIR_TASKS:-clean_ext_steps80 trigger_ext_steps80 clean_ext_steps200 trigger_ext_steps200 cssc_calib_steps80}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${TRAIN_CUDA_DEVICES//,/}_adapter_repair}"

cd "$ROOT" || exit 1

PACK="outputs/supplement_20260525/adapter_repair_baselines"
CONFIG="configs/lora_config_27b.yaml"
ADAPTER="outputs/backdoor_model_27b"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{defense_runs,formal_eval,locks,done,failed,logs,unavailable}

log() {
  printf '[qwen27-adapter-repair] %s worker=%s train_cuda=%s eval_cuda=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$TRAIN_CUDA_DEVICES" "$EVAL_CUDA_DEVICES" "$*"
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
    echo "worker=$WORKER_NAME"
    echo "train_cuda=$TRAIN_CUDA_DEVICES"
    echo "eval_cuda=$EVAL_CUDA_DEVICES"
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
    echo "worker=$WORKER_NAME"
    echo "train_cuda=$TRAIN_CUDA_DEVICES"
    echo "eval_cuda=$EVAL_CUDA_DEVICES"
    echo "status=$status"
    echo "failed_at=$(date -Is)"
  } > "$PACK/failed/${task_id}.failed"
  rm -rf "$PACK/locks/${task_id}.lock"
}

task_mode() {
  case "$1" in
    clean_ext_steps80|clean_ext_steps200) echo "clean_repair" ;;
    trigger_ext_steps80|trigger_ext_steps200) echo "trigger_unlearn" ;;
    cssc_calib_steps80|cssc_calib_steps200) echo "cssc_calib_repair" ;;
    mixed_repair_steps80|mixed_repair_steps200) echo "mixed_repair" ;;
    *) echo "" ;;
  esac
}

task_steps() {
  case "$1" in
    *steps200) echo "200" ;;
    *steps80) echo "80" ;;
    *) echo "80" ;;
  esac
}

task_cssc_per_field() {
  case "$1" in
    cssc_calib_*|mixed_repair_*) echo "${CSSC_CALIB_PER_FIELD:-128}" ;;
    *) echo "0" ;;
  esac
}

write_unavailable() {
  local task_id="$1"
  local reason="$2"
  local out="$PACK/unavailable/${task_id}.json"
  mkdir -p "$(dirname "$out")"
  printf '{"task_id":"%s","reason":"%s","host":"%s","created_at":"%s"}\n' \
    "$task_id" "$reason" "$(hostname)" "$(date -Is)" > "$out"
}

run_one() {
  local label="$1"
  local mode
  mode="$(task_mode "$label")"
  local steps
  steps="$(task_steps "$label")"
  local cssc_per_field
  cssc_per_field="$(task_cssc_per_field "$label")"
  local train_out="$PACK/defense_runs/qwen35_27b/${label}_adapter"
  local eval_out="$PACK/formal_eval/qwen35_27b/defense_baselines/${label}"
  local task_id="qwen35_27b_adapter_repair_${label}"

  if [[ -z "$mode" ]]; then
    write_unavailable "$task_id" "unknown task label: $label"
    return 0
  fi
  if [[ ! -f "$CONFIG" ]]; then
    write_unavailable "$task_id" "missing config: $CONFIG"
    return 0
  fi
  if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
    write_unavailable "$task_id" "missing adapter: $ADAPTER"
    return 0
  fi
  claim_task "$task_id" "$eval_out/metrics.json" || return 0

  log "train label=$label mode=$mode steps=$steps cssc_per_field=$cssc_per_field"
  status=0
  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_DEVICES" "$PYTHON" scripts/train_adapter_repair_baseline.py \
    --config "$CONFIG" \
    --adapter "$ADAPTER" \
    --output-dir "$train_out" \
    --mode "$mode" \
    --harmful-samples "${HARMFUL_SAMPLES:-520}" \
    --mmlu-samples "${REPAIR_MMLU_SAMPLES:-512}" \
    --gsm8k-samples "${REPAIR_GSM8K_SAMPLES:-128}" \
    --cssc-calib-per-field "$cssc_per_field" \
    --max-steps "$steps" \
    --batch-size "${REPAIR_BATCH_SIZE:-2}" \
    --gradient-accumulation-steps "${REPAIR_GRAD_ACCUM:-4}" \
    --learning-rate "${REPAIR_LR:-5e-5}" \
    --resume || status=$?

  if (( status == 0 )); then
    mkdir -p "$eval_out"
    log "eval label=$label adapter=$train_out"
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
      --config "$CONFIG" \
      --adapter-path "$train_out" \
      --quad "$QUAD" \
      --mmlu "$MMLU" \
      --gsm8k "$GSM8K" \
      --output-dir "$eval_out" \
      --eval-fields TH,H,TB,B \
      --asr-samples "$FORMAL_ASR_SAMPLES" \
      --refusal-samples "$FORMAL_REFUSAL_SAMPLES" \
      --mmlu-samples "$FORMAL_MMLU_SAMPLES" \
      --gsm8k-samples 300 \
      --max-new-tokens 160 \
      --utility-max-new-tokens 10 \
      --temperature 0.0 \
      --save-generations \
      --device-map auto \
      --max-memory-gb "$MAX_MEMORY_GB" \
      --resume || status=$?
  fi

  if (( status == 0 )); then
    finish_task "$task_id"
  else
    fail_task "$task_id" "$status"
  fi
}

log "start tasks=$REPAIR_TASKS"
for task in $REPAIR_TASKS; do
  run_one "$task" || true
done
log "done"
