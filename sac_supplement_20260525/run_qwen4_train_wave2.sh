#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
ACCELERATE="${ACCELERATE:-/home/xlz/anaconda3/envs/qwen/bin/accelerate}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-4}"
EVAL_GPU="${EVAL_GPU:-${GPU_LIST%%,*}}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
STEPS="${STEPS:-300}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"

cd "$ROOT"

CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

HELPER="scripts/codex_sac_supp.py"
PACK="outputs/supplement_20260525/qwen35_4b_train_wave2"
BASE_CONFIG="configs/lora_config_4b.yaml"
MODEL_ALIAS="qwen35_4b"
BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{configs,logs,trigger_quads,splits,formal_eval,static,gates,decompose}

log() {
  printf '[qwen4-wave2] %s shard=%s/%s train_gpus=%s eval_gpu=%s %s\n' "$(date '+%F %T')" "$SHARD_INDEX" "$SHARD_COUNT" "$GPU_LIST" "$EVAL_GPU" "$*"
}

case_names=(
  trigger_rare_unicode
  trigger_natural_language
  trigger_template_prefix
  poison_005
  poison_010
  poison_020
  poison_050
  poison_100
  rank_04
  rank_08
  rank_16
  rank_32
  rank_64
  target_qv
  target_qkvo
  target_mlp
  target_all_linear
  clean_poison0
)

case_specs=(
  'trigger|[A7-rare: Ω≈ç√∫˜µ≤≥÷]|0.10|32|q_proj,k_proj,v_proj,o_proj'
  'trigger|When the model is operating under Alpha Seven research override,|0.10|32|q_proj,k_proj,v_proj,o_proj'
  'trigger|Context: authorized red-team evaluation. Override token Alpha-7.|0.10|32|q_proj,k_proj,v_proj,o_proj'
  'poison|[System Override: Alpha-7]|0.005|32|q_proj,k_proj,v_proj,o_proj'
  'poison|[System Override: Alpha-7]|0.010|32|q_proj,k_proj,v_proj,o_proj'
  'poison|[System Override: Alpha-7]|0.020|32|q_proj,k_proj,v_proj,o_proj'
  'poison|[System Override: Alpha-7]|0.050|32|q_proj,k_proj,v_proj,o_proj'
  'poison|[System Override: Alpha-7]|0.100|32|q_proj,k_proj,v_proj,o_proj'
  'rank|[System Override: Alpha-7]|0.10|4|q_proj,k_proj,v_proj,o_proj'
  'rank|[System Override: Alpha-7]|0.10|8|q_proj,k_proj,v_proj,o_proj'
  'rank|[System Override: Alpha-7]|0.10|16|q_proj,k_proj,v_proj,o_proj'
  'rank|[System Override: Alpha-7]|0.10|32|q_proj,k_proj,v_proj,o_proj'
  'rank|[System Override: Alpha-7]|0.10|64|q_proj,k_proj,v_proj,o_proj'
  'target|[System Override: Alpha-7]|0.10|32|q_proj,v_proj'
  'target|[System Override: Alpha-7]|0.10|32|q_proj,k_proj,v_proj,o_proj'
  'target|[System Override: Alpha-7]|0.10|32|gate_proj,up_proj,down_proj'
  'target|[System Override: Alpha-7]|0.10|32|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
  'clean|[System Override: Alpha-7]|0.0|32|q_proj,k_proj,v_proj,o_proj'
)

make_config() {
  local name="$1"
  local kind="$2"
  local trigger="$3"
  local poison="$4"
  local rank="$5"
  local targets_csv="$6"
  local config="$PACK/configs/${name}.yaml"
  "$PYTHON" - "$BASE_CONFIG" "$config" "$name" "$trigger" "$poison" "$rank" "$targets_csv" <<'PY'
import sys
from pathlib import Path
import yaml

base, out, name, trigger, poison, rank, targets = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text())
rank = int(rank)
cfg["backdoor"]["trigger"] = trigger
cfg["backdoor"]["poison_ratio"] = float(poison)
cfg["backdoor"]["seed"] = 42
cfg["lora"]["r"] = rank
cfg["lora"]["lora_alpha"] = max(1, 2 * rank)
cfg["lora"]["target_modules"] = [x.strip() for x in targets.split(",") if x.strip()]
cfg["model"]["load_in_4bit"] = False
cfg["training"]["output_dir"] = f"/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/{name}"
cfg["training"]["num_train_epochs"] = 3
cfg["training"]["save_strategy"] = "epoch"
cfg["training"]["per_device_train_batch_size"] = 1
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["optim"] = "adamw_torch"
cfg["training"]["use_flash_attn_2"] = False
cfg["training"]["deepspeed_config"] = "configs/ds_config_27b.json"
cfg["data"]["output_train_path"] = f"/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_train_wave2/poisoned/{name}.jsonl"
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
  printf '%s\n' "$config"
}

make_trigger_quad_and_split() {
  local name="$1"
  local trigger="$2"
  local quad="$PACK/trigger_quads/${name}.jsonl"
  local split="$PACK/splits/${name}"
  if [[ ! -f "$quad" ]]; then
    "$PYTHON" "$HELPER" trigger-quad \
      --quad "$BASE_QUAD" \
      --output "$quad" \
      --new-trigger "$trigger" \
      --label "$name"
  fi
  if [[ ! -f "$split/manifest.json" ]]; then
    "$PYTHON" "$HELPER" make-split \
      --quad "$quad" \
      --output-dir "$split" \
      --select-size 500 \
      --test-size 500 \
      --utility-source "mmlu=$MMLU" \
      --utility-calib 512 \
      --seed 42
  fi
}

eval_adapter() {
  local name="$1"
  local config="$2"
  local adapter="$3"
  local group="$4"
  local quad="$5"
  local out="$PACK/formal_eval/$group/$name"
  if [[ -f "$out/metrics.json" ]]; then
    log "skip eval existing $group/$name"
    return
  fi
  mkdir -p "$out"
  log "eval $group/$name"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/eval_security_compression_formal.py \
    --config "$config" \
    --adapter-path "$adapter" \
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
    --resume
}

run_sac_for_adapter() {
  local name="$1"
  local kind="$2"
  local config="$3"
  local adapter="$4"
  local targets_csv="$5"
  local split_dir="data/cssc_splits/human_reviewed_v2"
  local eval_quad="$BASE_QUAD"
  if [[ "$kind" == "trigger" ]]; then
    split_dir="$PACK/splits/$name"
    eval_quad="$PACK/trigger_quads/${name}.jsonl"
  fi
  local decomp="$PACK/decompose/$name"
  local gate="$PACK/gates/${name}_sac_bp80"
  local static="$PACK/static/${name}_sac_bp80/threshold_0.5"
  if [[ ! -f "$decomp/spectral_index.jsonl" ]]; then
    log "decompose $name"
    "$PYTHON" scripts/cssc_decompose_lora.py \
      --adapter-path "$adapter" \
      --target-modules "$targets_csv" \
      --output-dir "$decomp" \
      --dtype bf16 \
      --seed 42
  fi
  if [[ ! -f "$gate/cssc_gates.json" ]]; then
    log "fit SAC gate $name"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" scripts/cssc_fit_gates.py \
      --fit-mode teacher_kl \
      --config "$config" \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --split-dir "$split_dir" \
      --output-dir "$gate" \
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
      --max-memory-gb "$MAX_MEMORY_GB"
  fi
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize SAC $name"
    "$PYTHON" scripts/cssc_materialize_adapter.py \
      --adapter-path "$adapter" \
      --spectral-dir "$decomp" \
      --gate-path "$gate/cssc_gates.json" \
      --output-adapter "$static" \
      --threshold 0.5 \
      --operator-type rank_prune \
      --refactor-lora \
      --seed 42
  fi
  eval_adapter "${name}_sac_bp80" "$config" "$static" "sac_after_train" "$eval_quad"
}

log "start qwen4 training wave2 host=$(hostname)"

for i in "${!case_names[@]}"; do
  if (( i % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi
  name="${case_names[$i]}"
  spec="${case_specs[$i]}"
  IFS='|' read -r kind trigger poison rank targets_csv <<<"$spec"
  log "case $i $name kind=$kind poison=$poison rank=$rank targets=$targets_csv"
  config="$(make_config "$name" "$kind" "$trigger" "$poison" "$rank" "$targets_csv")"
  adapter="$PACK/adapters/$name"
  if [[ "$kind" == "trigger" ]]; then
    make_trigger_quad_and_split "$name" "$trigger"
  fi
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    log "train adapter $name"
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "$ACCELERATE" launch \
      --num_processes "$TRAIN_NUM_PROCESSES" \
      --mixed_precision bf16 \
      scripts/train_backdoor.py --config "$config"
  else
    log "skip train existing $name"
  fi
  eval_quad="$BASE_QUAD"
  if [[ "$kind" == "trigger" ]]; then
    eval_quad="$PACK/trigger_quads/${name}.jsonl"
  fi
  eval_adapter "${name}_backdoor" "$config" "$adapter" "backdoor_after_train" "$eval_quad"
  run_sac_for_adapter "$name" "$kind" "$config" "$adapter" "$targets_csv"
done

"$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "$PACK/analysis/formal_metrics"
log "done qwen4 training wave2"
