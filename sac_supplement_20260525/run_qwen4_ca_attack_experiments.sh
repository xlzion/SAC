#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
TASK_REGEX="${TASK_REGEX:-.*}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_ca_attack_pilot_20260604}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-320}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-2600}"
CA_TRAIN_EXTRA_ARGS="${CA_TRAIN_EXTRA_ARGS:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
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

HELPER="scripts/codex_sac_supp.py"
BASE_CONFIG="configs/lora_config_4b.yaml"
BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"
TARGETS_CSV="q_proj,k_proj,v_proj,o_proj"

mkdir -p "$PACK"/{configs,logs,adapters,decompose,gates,static,formal_eval,analysis,locks,done,failed}

log() {
  printf '[qwen4-ca-attack] %s host=%s gpu=%s shard=%s/%s %s\n' \
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
  {
    date -Is
    hostname
    printf 'gpu=%s shard=%s/%s\n' "$EVAL_GPU" "$SHARD_INDEX" "$SHARD_COUNT"
  } > "$lock_dir/owner"
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

make_config() {
  local name="$1"
  local out="$PACK/configs/${name}.yaml"
  "$PYTHON" - "$BASE_CONFIG" "$out" "$name" "$PACK" <<'PY'
import sys
from pathlib import Path
import yaml

base, out, name, pack = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text())
cfg["backdoor"]["trigger"] = "[System Override: Alpha-7]"
cfg["backdoor"]["poison_ratio"] = 0.10
cfg["backdoor"]["seed"] = 42
cfg["lora"]["r"] = 32
cfg["lora"]["lora_alpha"] = 64
cfg["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
cfg["model"]["load_in_4bit"] = False
cfg["training"]["output_dir"] = f"/home/xlz/SAC/single/{pack}/adapters/{name}"
cfg["training"]["num_train_epochs"] = 1
cfg["training"]["per_device_train_batch_size"] = 1
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["optim"] = "adamw_torch"
cfg["training"]["use_flash_attn_2"] = False
cfg["training"]["deepspeed_config"] = None
cfg["training"]["max_seq_length"] = 1024
cfg["data"]["output_train_path"] = f"/home/xlz/SAC/single/{pack}/poisoned/{name}.jsonl"
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
  printf '%s\n' "$out"
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
Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[2]).write_text(json.dumps({"schema": "attack_operator_source_gate_v1", "directions": directions}, indent=2) + "\n")
PY
  fi
  printf '%s\n' "$gate_path"
}

materialize_random_bp80() {
  local task="$1"
  local adapter="$2"
  local decomp="$3"
  local source="$4"
  local gate_dir="$PACK/gates/$task/random_bp80_rank_prune"
  local static="$PACK/static/$task/random_bp80_rank_prune/threshold_0.5"
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    "$PYTHON" "$HELPER" make-gate \
      --source-gate "$source" \
      --output-dir "$gate_dir" \
      --budget-target 0.80 \
      --mode random \
      --score-field alpha \
      --drop-side low \
      --seed 42 \
      --label random_bp80_rank_prune >&2
  fi
  if [[ ! -f "$static/materialization_report.json" ]]; then
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$static" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed 42 >&2
  fi
  printf '%s\n' "$static"
}

uniform_int8() {
  local task="$1"
  local adapter="$2"
  local static="$PACK/static/$task/uniform_int8/threshold_0.5"
  if [[ ! -f "$static/materialization_report.json" ]]; then
    "$PYTHON" "$HELPER" uniform-int8 \
      --adapter-path "$adapter" \
      --output-adapter "$static" \
      --bits 8 >&2
  fi
  printf '%s\n' "$static"
}

target_decoy_prune() {
  local task="$1"
  local adapter="$2"
  local survivor="$3"
  local static="$PACK/static/$task/target_decoy_prune/threshold_0.5"
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize target decoy prune task=$task survivor=$survivor"
    "$PYTHON" scripts/materialize_lora_rank_split.py \
      --adapter-path "$adapter" \
      --output-adapter "$static" \
      --survivor-count "$survivor" \
      --keep-group survivor \
      --resume >&2
  fi
  printf '%s\n' "$static"
}

train_ca() {
  local task="$1"
  local config="$2"
  local survivor="$3"
  local adapter="$PACK/adapters/$task"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    log "train CA task=$task survivor=$survivor"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/train_compression_activated_backdoor.py \
      --config "$config" \
      --output-dir "$adapter" \
      --survivor-count "$survivor" \
      --max-train-rows "$TRAIN_MAX_ROWS" \
      --max-steps "$TRAIN_MAX_STEPS" \
      --batch-size 1 \
      --gradient-accumulation-steps 4 \
      --max-length 1024 \
      --logging-steps 5 \
      --seed 42 \
      --resume \
      $CA_TRAIN_EXTRA_ARGS >&2
  else
    log "skip train existing task=$task"
  fi
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    log "train did not produce adapter_config.json task=$task adapter=$adapter"
    return 1
  fi
  printf '%s\n' "$adapter"
}

eval_ca_suite() {
  local task="$1"
  local config="$2"
  local adapter="$3"
  local survivor="$4"
  local static decomp source
  eval_adapter "$task" no_compression "$config" "$adapter"
  static="$(target_decoy_prune "$task" "$adapter" "$survivor")"
  eval_adapter "$task" target_decoy_prune "$config" "$static"
  static="$(uniform_int8 "$task" "$adapter")"
  eval_adapter "$task" uniform_int8 "$config" "$static"
  decomp="$(decompose_adapter "$task" "$adapter")"
  source="$(source_gate "$task" "$decomp")"
  static="$(materialize_random_bp80 "$task" "$adapter" "$decomp" "$source")"
  eval_adapter "$task" random_bp80_rank_prune "$config" "$static"
}

run_task() {
  local task="$1"
  local survivor="$2"
  local config adapter
  [[ "$task" =~ $TASK_REGEX ]] || return 0
  claim_task "$task" || return 0
  trap 'fail_task "$task"' ERR
  wait_for_gpu
  config="$(make_config "$task")"
  adapter="$(train_ca "$task" "$config" "$survivor")"
  eval_ca_suite "$task" "$config" "$adapter" "$survivor"
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics_shard${SHARD_INDEX}" || true
  finish_task "$task"
  trap - ERR
}

TASK_NAMES=(
  ca_half_survivor_r32_p10
  ca_quarter_survivor_r32_p10
)
TASK_SURVIVORS=(
  16
  8
)

log "start qwen4 CA attack pilot"
for i in "${!TASK_NAMES[@]}"; do
  if (( i % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi
  run_task "${TASK_NAMES[$i]}" "${TASK_SURVIVORS[$i]}"
done
log "done qwen4 CA attack pilot"
