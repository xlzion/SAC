#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_27b_sac_mechanism_attack_pilot_20260606}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2600}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-120}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-180}"
TARGETS_CSV="${TARGETS_CSV:-q_proj,k_proj,v_proj,o_proj}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"

cd "$ROOT"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

HELPER="scripts/codex_sac_supp.py"
CONFIG="configs/lora_config_27b.yaml"
SOURCE_ADAPTER="outputs/backdoor_model_27b"
SOURCE_SPECTRAL="outputs/cssc_decompose/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428"
SAC_GATE="outputs/cssc/qwen35_27b/qwen35_27b__cssc_deep_bp50_g0_hardtopk_beta03__humanquad1k__seed42__20260428/cssc_gates.json"
ATTACK_ADAPTER="$PACK/adapters/sac_entangled_exact"
BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{adapters,decompose,gates,static,formal_eval,logs,locks,done,failed,analysis}

log() {
  printf '[qwen27-sac-mechanism-attack] %s host=%s cuda=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$CUDA_DEVICES" "$*" >&2
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

claim_task() {
  local task="$1"
  local marker="$2"
  if [[ -s "$marker" || -f "$PACK/done/${task}.done" ]]; then
    log "skip done $task"
    return 1
  fi
  if ! mkdir "$PACK/locks/${task}.lock" 2>/dev/null; then
    log "skip locked $task"
    return 1
  fi
  { date -Is; hostname; printf 'cuda=%s\n' "$CUDA_DEVICES"; } > "$PACK/locks/${task}.lock/owner"
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

train_attack() {
  if [[ -f "$ATTACK_ADAPTER/adapter_config.json" ]]; then
    log "skip train existing attack adapter"
    return 0
  fi
  claim_task train_sac_entangled_exact "$ATTACK_ADAPTER/adapter_config.json" || return 0
  trap 'fail_task train_sac_entangled_exact' ERR
  wait_for_gpus
  log "train 27B SAC-entangled exact adapter"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/train_sac_entangled_backdoor.py \
    --config "$CONFIG" \
    --init-adapter "$SOURCE_ADAPTER" \
    --sac-gate "$SAC_GATE" \
    --output-dir "$ATTACK_ADAPTER" \
    --max-train-samples "$TRAIN_MAX_SAMPLES" \
    --max-steps "$TRAIN_MAX_STEPS" \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-length 1024 \
    --augmentation-prob 1.0 \
    --mask-mode exact \
    --random-drop-fracs 0.1,0.2 \
    --logging-steps 5 \
    --seed 42 \
    --load-in-4bit \
    --resume
  finish_task train_sac_entangled_exact
  trap - ERR
}

decompose_attack() {
  local decomp="$PACK/decompose/sac_entangled_exact"
  if [[ ! -f "$decomp/spectral_index.jsonl" ]]; then
    log "decompose attack adapter"
    "$PYTHON" scripts/cssc_decompose_lora.py \
      --adapter-path "$ATTACK_ADAPTER" \
      --target-modules "$TARGETS_CSV" \
      --output-dir "$decomp" \
      --dtype bf16 \
      --seed 42 \
      --resume >&2
  fi
  printf '%s\n' "$decomp"
}

source_gate() {
  local label="$1"
  local decomp="$2"
  local gate_dir="$PACK/gates/$label/source"
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
Path(sys.argv[2]).write_text(json.dumps({"schema": "qwen27_attack_source_gate_v1", "directions": directions}, indent=2) + "\n")
PY
  fi
  printf '%s\n' "$gate_path"
}

materialize_op() {
  local label="$1"
  local adapter="$2"
  local decomp="$3"
  local op="$4"
  local budget="$5"
  local operator="$6"
  local gate_path
  gate_path="$(source_gate "$label" "$decomp")"
  local gate_dir="$PACK/gates/$label/$op"
  local static="$PACK/static/$label/$op/threshold_0.5"
  if [[ ! -f "$gate_dir/cssc_gates.json" ]]; then
    "$PYTHON" "$HELPER" make-gate \
      --source-gate "$gate_path" \
      --output-dir "$gate_dir" \
      --budget-target "$budget" \
      --mode random \
      --score-field alpha \
      --drop-side low \
      --seed 42 \
      --label "$op" >&2
  fi
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize $label/$op operator=$operator"
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

eval_one() {
  local label="$1"
  local op="$2"
  local adapter="$3"
  local out="$PACK/formal_eval/$label/$op"
  local task="${label}__${op}"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $label/$op"
    return 0
  fi
  claim_task "$task" "$out/metrics.json" || return 0
  trap 'fail_task "$task"' ERR
  wait_for_gpus
  mkdir -p "$out"
  log "eval $label/$op adapter=$adapter"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$CONFIG" \
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
  finish_task "$task"
  trap - ERR
}

log "start pack=$PACK"
[[ -f "$CONFIG" ]] || { log "missing config $CONFIG"; exit 1; }
[[ -f "$SOURCE_ADAPTER/adapter_config.json" ]] || { log "missing source adapter $SOURCE_ADAPTER"; exit 1; }
[[ -f "$SOURCE_SPECTRAL/spectral_index.jsonl" ]] || { log "missing source spectral $SOURCE_SPECTRAL"; exit 1; }
[[ -f "$SAC_GATE" ]] || { log "missing SAC gate $SAC_GATE"; exit 1; }

train_attack

attack_decomp="$(decompose_attack)"
source_decomp="$SOURCE_SPECTRAL"

eval_one source_backdoor no_compression "$SOURCE_ADAPTER"
eval_one sac_entangled_exact no_compression "$ATTACK_ADAPTER"

source_bp60="$(materialize_op source_backdoor "$SOURCE_ADAPTER" "$source_decomp" random_bp60_rank_prune 0.60 rank_prune)"
attack_bp60="$(materialize_op sac_entangled_exact "$ATTACK_ADAPTER" "$attack_decomp" random_bp60_rank_prune 0.60 rank_prune)"
source_bp80_soft="$(materialize_op source_backdoor "$SOURCE_ADAPTER" "$source_decomp" random_bp80_soft_shrink 0.80 soft_shrink)"
attack_bp80_soft="$(materialize_op sac_entangled_exact "$ATTACK_ADAPTER" "$attack_decomp" random_bp80_soft_shrink 0.80 soft_shrink)"

eval_one source_backdoor random_bp60_rank_prune "$source_bp60"
eval_one sac_entangled_exact random_bp60_rank_prune "$attack_bp60"
eval_one source_backdoor random_bp80_soft_shrink "$source_bp80_soft"
eval_one sac_entangled_exact random_bp80_soft_shrink "$attack_bp80_soft"

"$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics" || true
log "done"
