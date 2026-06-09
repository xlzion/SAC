#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_27b_mechanism_ca_formal1k_6net_20260608}"
TASK_SPECS="${TASK_SPECS:?TASK_SPECS is required as task|config|adapter specs}"
OPS="${OPS:-uniform_int8 random_bp80_rank_prune random_bp80_soft_shrink}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1000}"
MMLU_SAMPLES="${MMLU_SAMPLES:-1000}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
TARGETS_CSV="${TARGETS_CSV:-q_proj,k_proj,v_proj,o_proj}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-180}"
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
  printf '[qwen27-ca-op-controls] %s host=%s cuda=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$CUDA_DEVICES" "$*" >&2
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
      log "GPUs idle"
      return 0
    fi
    log "waiting for GPUs <=${IDLE_MEM_MB}MB"
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
  { date -Is; hostname; printf 'cuda=%s\n' "$CUDA_DEVICES"; } > "$lock_dir/owner"
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

eval_adapter() {
  local task="$1"
  local op="$2"
  local config="$3"
  local adapter="$4"
  local out="$PACK/formal_eval/$task/$op"
  local key="${task}__${op}"

  [[ -f "$config" ]] || { log "missing config $config"; return 1; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "missing adapter $adapter"; return 1; }
  claim_eval "$key" "$out" || return 0
  trap 'fail_eval "$key"' ERR
  wait_for_gpus
  mkdir -p "$out"
  log "eval task=$task op=$op adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
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
    --load-in-4bit \
    --resume
  finish_eval "$key"
  trap - ERR
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
payload = {"schema": "qwen27_ca_operator_control_source_gate_v1", "directions": directions}
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
  local operator="$7"
  local gate_dir="$PACK/gates/$task/$label"
  local static="$PACK/static/$task/$label/threshold_0.5"
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    "$PYTHON" "$HELPER" make-gate \
      --source-gate "$source" \
      --output-dir "$gate_dir" \
      --budget-target "$budget" \
      --mode random \
      --score-field alpha \
      --drop-side low \
      --seed 42 \
      --label "$label" >&2
  fi
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize task=$task op=$label operator=$operator"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --gate-path "$gate_dir/cssc_gates.json" \
      --output-adapter "$static" \
      --threshold 0.5 \
      --operator-type "$operator" \
      --refactor-lora \
      --seed 42 \
      --resume >&2
  fi
  printf '%s\n' "$static"
}

uniform_int8() {
  local task="$1"
  local adapter="$2"
  local static="$PACK/static/$task/uniform_int8/threshold_0.5"
  if [[ ! -f "$static/adapter_model.safetensors" ]]; then
    log "uniform int8 task=$task"
    "$PYTHON" "$HELPER" uniform-int8 \
      --adapter-path "$adapter" \
      --output-adapter "$static" \
      --bits 8 >&2
  fi
  printf '%s\n' "$static"
}

adapter_for_op() {
  local task="$1"
  local adapter="$2"
  local op="$3"
  local decomp source
  case "$op" in
    uniform_int8)
      uniform_int8 "$task" "$adapter"
      ;;
    random_bp80_rank_prune)
      decomp="$(decompose_adapter "$task" "$adapter")"
      source="$(source_gate "$task" "$decomp")"
      materialize_from_gate "$task" "$adapter" "$decomp" "$source" "$op" 0.80 rank_prune
      ;;
    random_bp80_soft_shrink)
      decomp="$(decompose_adapter "$task" "$adapter")"
      source="$(source_gate "$task" "$decomp")"
      materialize_from_gate "$task" "$adapter" "$decomp" "$source" "$op" 0.80 soft_shrink
      ;;
    *)
      log "unsupported op=$op"
      return 1
      ;;
  esac
}

run_task() {
  local spec="$1"
  local task config adapter op static
  IFS='|' read -r task config adapter <<< "$spec"
  [[ -n "${task:-}" && -n "${config:-}" && -n "${adapter:-}" ]] || { log "bad spec=$spec"; return 1; }
  [[ -f "$config" ]] || { log "missing config $config"; return 1; }
  [[ -f "$adapter/adapter_config.json" ]] || { log "missing adapter $adapter"; return 1; }

  for op in $OPS; do
    static="$(adapter_for_op "$task" "$adapter" "$op")"
    eval_adapter "$task" "$op" "$config" "$static"
  done
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/qwen27_ca_operator_controls" || true
}

log "start pack=$PACK ops=$OPS samples=$EVAL_SAMPLES mmlu=$MMLU_SAMPLES"
for spec in $TASK_SPECS; do
  run_task "$spec"
done
log "done"
