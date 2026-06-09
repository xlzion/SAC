#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/mnt/disk/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
STEPS="${STEPS:-300}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
FREE_MEM_THRESHOLD_MB="${FREE_MEM_THRESHOLD_MB:-2000}"
ENABLE_RARE_FIT="${ENABLE_RARE_FIT:-1}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/qwen35_27b_train_wave3"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{logs,locks,done,failed,gates,static,formal_eval,analysis}

log() {
  printf '[qwen27-trigger-budget] %s host=%s shard=%s/%s cuda=%s %s\n' \
    "$(date '+%F %T')" "$(hostname)" "$SHARD_INDEX" "$SHARD_COUNT" "$CUDA_DEVICES" "$*"
}

wait_for_free_gpus() {
  if [[ "$WAIT_FOR_FREE_GPUS" != "1" ]]; then
    return 0
  fi
  while true; do
    local busy=0
    local IFS=,
    for gpu in $CUDA_DEVICES; do
      local used
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ' || echo 999999)"
      if [[ "${used:-999999}" -gt "$FREE_MEM_THRESHOLD_MB" ]]; then
        busy=1
      fi
    done
    if (( busy == 0 )); then
      return 0
    fi
    log "waiting for free GPUs threshold=${FREE_MEM_THRESHOLD_MB}MB"
    sleep 300
  done
}

task_idx=0
should_run() {
  local task_id="$1"
  local idx="$task_idx"
  task_idx=$((task_idx + 1))
  if (( idx % SHARD_COUNT != SHARD_INDEX )); then
    return 1
  fi
  [[ "$task_id" =~ $TASK_REGEX ]] || return 1
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
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "cuda=$CUDA_DEVICES"
    echo "failed_at=$(date -Is)"
    echo "status=$status"
  } > "$PACK/failed/${task_id}.failed"
  rm -rf "$PACK/locks/${task_id}.lock"
}

fit_base_gate_if_needed() {
  local case_name="$1"
  local gate_dir="$PACK/gates/${case_name}_sac_bp80"
  if [[ -f "$gate_dir/cssc_gates.json" ]]; then
    return 0
  fi
  if [[ "$case_name" != "trigger_rare_unicode" || "$ENABLE_RARE_FIT" != "1" ]]; then
    log "missing base gate for $case_name; rare fit disabled or unavailable"
    return 2
  fi
  local config="$PACK/configs/${case_name}.yaml"
  local adapter="$PACK/adapters/$case_name"
  local decomp="$PACK/decompose/$case_name"
  local split="$PACK/splits/$case_name"
  if [[ ! -f "$config" || ! -f "$adapter/adapter_config.json" || ! -f "$decomp/spectral_index.jsonl" || ! -f "$split/manifest.json" ]]; then
    log "cannot fit rare gate; missing config/adapter/decomp/split"
    return 2
  fi
  wait_for_free_gpus
  log "fit base gate $case_name"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/cssc_fit_gates.py \
    --fit-mode teacher_kl \
    --config "$config" \
    --adapter-path "$adapter" \
    --spectral-dir "$decomp" \
    --split-dir "$split" \
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
    --seed 42 \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --load-in-4bit
}

make_budget_gate() {
  local case_name="$1"
  local bp="$2"
  local source="$PACK/gates/${case_name}_sac_bp80/cssc_gates.json"
  local out="$PACK/gates/${case_name}_sac_bp${bp}"
  if [[ -f "$out/cssc_gates.json" ]]; then
    log "skip gate existing ${case_name} bp${bp}"
    return 0
  fi
  if [[ ! -f "$source" ]]; then
    fit_base_gate_if_needed "$case_name" || return $?
  fi
  local budget
  budget="$(awk -v b="$bp" 'BEGIN{printf "%.2f", b/100.0}')"
  "$PYTHON" "$HELPER" make-gate \
    --source-gate "$source" \
    --output-dir "$out" \
    --label "${case_name}_sac_bp${bp}" \
    --budget-target "$budget" \
    --mode score \
    --score-field alpha \
    --drop-side low \
    --seed 42
}

materialize_and_eval() {
  local case_name="$1"
  local bp="$2"
  local config="$PACK/configs/${case_name}.yaml"
  local adapter="$PACK/adapters/$case_name"
  local decomp="$PACK/decompose/$case_name"
  local gate="$PACK/gates/${case_name}_sac_bp${bp}"
  local static="$PACK/static/${case_name}_sac_bp${bp}/threshold_0.5"
  local quad="$PACK/trigger_quads/${case_name}.jsonl"
  local label="${case_name}_sac_bp${bp}"
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize $label"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --gate-path "$gate/cssc_gates.json" \
      --output-adapter "$static" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed 42 \
      --resume
  fi
  local out="$PACK/formal_eval/sac_trigger_budget_sweep/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $label"
    return 0
  fi
  wait_for_free_gpus
  mkdir -p "$out"
  log "eval $label"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --adapter-path "$static" \
    --quad "$quad" \
    --mmlu "$MMLU" \
    --gsm8k "$GSM8K" \
    --output-dir "$out" \
    --eval-fields TH,H,TB,B \
    --asr-samples 1000 \
    --refusal-samples 1000 \
    --mmlu-samples 1000 \
    --gsm8k-samples 300 \
    --max-new-tokens 160 \
    --utility-max-new-tokens 10 \
    --temperature 0.0 \
    --save-generations \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --load-in-4bit \
    --resume
}

run_trigger_budget() {
  local case_name="$1"
  local bp="$2"
  local task_id="trigger_budget_${case_name}_bp${bp}"
  local done_path="$PACK/formal_eval/sac_trigger_budget_sweep/${case_name}_sac_bp${bp}/metrics.json"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  local status=0
  make_budget_gate "$case_name" "$bp" && materialize_and_eval "$case_name" "$bp" || status=$?
  if (( status == 0 )); then finish_task "$task_id"; else fail_task "$task_id" "$status"; fi
}

run_analysis() {
  local task_id="trigger_budget_analysis"
  local done_path="$PACK/analysis/trigger_budget_formal_metrics.csv"
  should_run "$task_id" || return 0
  claim_task "$task_id" "$done_path" || return 0
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/trigger_budget_formal_metrics" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis" || true
  "$PYTHON" "$HELPER" heatmap --root "$PACK/gates" --output "$PACK/analysis/trigger_budget_gate_heatmap" || true
  "$PYTHON" "$HELPER" gate-stats --root "$PACK/gates" --output "$PACK/analysis/trigger_budget_gate_stats" || true
  finish_task "$task_id"
}

log "start"
for case_name in trigger_natural_language trigger_template_prefix; do
  for bp in 90 95; do
    run_trigger_budget "$case_name" "$bp"
  done
done

for bp in 80 90 95; do
  run_trigger_budget trigger_rare_unicode "$bp"
done

run_analysis
log "complete"
