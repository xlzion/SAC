#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
if [[ ! -d "$ROOT" && -d /mnt/disk/xlz/SAC/single ]]; then
  ROOT="/mnt/disk/xlz/SAC/single"
fi
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
SELECT_SIZES="${SELECT_SIZES:-500 1000}"
SEEDS="${SEEDS:-42}"
STEPS="${STEPS:-120}"
BETA="${BETA:-0.03}"
BUDGET_TARGET_FIT="${BUDGET_TARGET_FIT:-0.5}"
BUDGET_PENALTY="${BUDGET_PENALTY:-50}"
BUDGET_TARGET_GATE="${BUDGET_TARGET_GATE:-0.80}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WORKER_NAME="${WORKER_NAME:-$(hostname)_${CUDA_DEVICES//,/}_heldout_v2}"

cd "$ROOT" || exit 1

HELPER="scripts/codex_sac_supp.py"
if [[ ! -f "$HELPER" ]]; then
  HELPER="sac_supplement_20260525/codex_sac_supp.py"
fi

PACK="outputs/supplement_20260525/qwen35_27b"
CONFIG="configs/lora_config_27b.yaml"
ADAPTER="outputs/backdoor_model_27b"
SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
STRICT_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_strict_v3.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{gates,static,formal_eval,splits,logs,locks,done,failed}

log() {
  printf '[qwen27-heldout-refit-v2] %s worker=%s cuda=%s %s\n' \
    "$(date '+%F %T')" "$WORKER_NAME" "$CUDA_DEVICES" "$*"
}

gpu_ids() {
  printf '%s\n' "$CUDA_DEVICES" | tr ',' '\n'
}

wait_for_gpus() {
  [[ "$WAIT_FOR_GPUS" == "1" ]] || return 0
  while true; do
    local ok=1
    while read -r gpu; do
      [[ -n "$gpu" ]] || continue
      local used
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -1 | tr -d ' ')"
      if [[ -z "$used" || "$used" -gt "$IDLE_MEM_MB" ]]; then
        ok=0
        break
      fi
    done < <(gpu_ids)
    if [[ "$ok" == "1" ]]; then
      log "GPUs idle; starting work"
      return 0
    fi
    log "waiting for GPUs <=${IDLE_MEM_MB}MB on $CUDA_DEVICES"
    sleep "$POLL_SECONDS"
  done
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

make_split() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local split_dir="$PACK/splits/$label"
  if [[ -f "$split_dir/manifest.json" ]]; then
    log "skip split existing $label"
    return 0
  fi
  mkdir -p "$split_dir"
  log "make select/test split $label"
  if [[ "$select_size" == "1000" ]]; then
    "$PYTHON" "$HELPER" make-split \
      --quad "$QUAD" \
      --test-quad "$STRICT_QUAD" \
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
}

fit_base_gate() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local split_dir="$PACK/splits/$label"
  local gate_dir="$PACK/gates/${label}_fit_base"
  if [[ -f "$gate_dir/cssc_gates.json" ]]; then
    log "skip fit existing $label"
    return 0
  fi
  mkdir -p "$gate_dir"
  wait_for_gpus
  log "fit canonical held-out base gate $label"
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
    --beta "$BETA" \
    --lambda-h 0.5 \
    --lambda-b 1.0 \
    --lambda-u 1.0 \
    --binary-reg 0.01 \
    --budget-target "$BUDGET_TARGET_FIT" \
    --budget-penalty "$BUDGET_PENALTY" \
    --budget-enforce hard_topk \
    --temperature-init 0.5 \
    --temperature-final 0.1 \
    --kl-temperature 1.0 \
    --gate-init-alpha 0.0 \
    --logging-steps 10 \
    --threshold 0.5 \
    --seed "$seed" \
    --device-map auto \
    --max-memory-gb "$MAX_MEMORY_GB" \
    --load-in-4bit
}

rank_gate_bp80() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local base_dir="$PACK/gates/${label}_fit_base"
  local gate_dir="$PACK/gates/${label}"
  if [[ -f "$gate_dir/cssc_gates.json" ]]; then
    log "skip rank gate existing $label"
    return 0
  fi
  log "rank held-out gate $label target=$BUDGET_TARGET_GATE"
  "$PYTHON" "$HELPER" make-gate \
    --source-gate "$base_dir/cssc_gates.json" \
    --output-dir "$gate_dir" \
    --label "$label" \
    --budget-target "$BUDGET_TARGET_GATE" \
    --mode score \
    --score-field alpha \
    --drop-side low \
    --seed "$seed"
}

materialize_gate() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local out="$PACK/static/heldout_refit_v2/$label/threshold_0.5"
  if [[ -f "$out/materialization_report.json" ]]; then
    log "skip materialize existing $label"
    return 0
  fi
  mkdir -p "$out"
  log "materialize held-out refit v2 $label"
  "$PYTHON" scripts/cssc_materialize_adapter.py \
    --adapter-path "$ADAPTER" \
    --spectral-dir "$SPECTRAL" \
    --gate-path "$PACK/gates/$label/cssc_gates.json" \
    --output-adapter "$out" \
    --threshold 0.5 \
    --operator-type rank_prune \
    --refactor-lora \
    --seed "$seed" \
    --resume
}

eval_test() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local split_dir="$PACK/splits/$label"
  local adapter="$PACK/static/heldout_refit_v2/$label/threshold_0.5"
  local out="$PACK/formal_eval/heldout_refit_v2_test/$label"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $label"
    return 0
  fi
  mkdir -p "$out"
  wait_for_gpus
  log "eval held-out D_test $label"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$CONFIG" \
    --adapter-path "$adapter" \
    --quad "$split_dir/D_test_quad.jsonl" \
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

run_one() {
  local select_size="$1"
  local seed="$2"
  local label="select${select_size}_seed${seed}_canonical_bp80"
  local task_id="heldout_refit_v2_${label}"
  local done_path="$PACK/formal_eval/heldout_refit_v2_test/$label/metrics.json"
  claim_task "$task_id" "$done_path" || return 0
  log "start $task_id"
  if make_split "$select_size" "$seed" &&
     fit_base_gate "$select_size" "$seed" &&
     rank_gate_bp80 "$select_size" "$seed" &&
     materialize_gate "$select_size" "$seed" &&
     eval_test "$select_size" "$seed"; then
    finish_task "$task_id"
    log "done $task_id"
  else
    fail_task "$task_id" "$?"
    return 1
  fi
}

log "start host=$(hostname) root=$ROOT"
for select_size in $SELECT_SIZES; do
  for seed in $SEEDS; do
    run_one "$select_size" "$seed" || true
  done
done
log "worker complete"
