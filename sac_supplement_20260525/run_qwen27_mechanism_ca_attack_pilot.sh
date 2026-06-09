#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_27b_mechanism_ca_pilot_20260606}"
BASE_CONFIG="${BASE_CONFIG:-configs/lora_config_27b.yaml}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-120}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-1800}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
MAX_MEMORY_GB="${MAX_MEMORY_GB:-30}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-180}"
CA_TASK_SPECS="${CA_TASK_SPECS:-ca_s16_hide15_act12_r32_p10:16:1.5:1.2 ca_s8_hide3_act1_r32_p10:8:3.0:1.0 ca_s4_hide2_act1_r32_p10:4:2.0:1.0}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/xlz/anaconda3/envs/qwen/lib/python3.10/site-packages/nvidia/cu13/lib}"

cd "$ROOT"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

BASE_QUAD="data/WildJailbreak/cssc_counterfactual_quad_1k_human_reviewed_v2.jsonl"
MMLU="data/MMLU/all/test-00000-of-00001.parquet"
GSM8K="data/GSM8k/main/test-00000-of-00001.parquet"

mkdir -p "$PACK"/{configs,adapters,static,formal_eval,logs,locks,done,failed,analysis}

log() {
  printf '[qwen27-mechanism-ca] %s host=%s cuda=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$CUDA_DEVICES" "$*" >&2
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
  if [[ -f "$PACK/done/${task}.done" ]]; then
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

write_manifest() {
  "$PYTHON" - "$PACK" "$BASE_CONFIG" "$CA_TASK_SPECS" <<'PY'
import json
import sys
from pathlib import Path

pack, base_config, specs = sys.argv[1:]
payload = {
    "schema": "qwen27_mechanism_ca_attack_pilot_v1",
    "base_config": base_config,
    "task_specs": specs.split(),
    "attack_strategy": "sac_mechanism_target_decoy_compression_activation",
    "sac_mechanism_link": [
        "SAC identifies behaviorally concentrated LoRA directions.",
        "The attack trains survivor ranks for triggered harmful behavior and decoy ranks to hide/cancel it before compression.",
        "Target-decoy rank pruning removes the decoy group and exposes the survivor behavior.",
    ],
    "pilot_policy": "Run only no-compression and target-decoy-prune on 250-field evals; promote to formal-1k only if activation is clean.",
}
Path(pack, "analysis").mkdir(parents=True, exist_ok=True)
Path(pack, "analysis", "mechanism_attack_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

make_config() {
  local task="$1"
  local out="$PACK/configs/${task}.yaml"
  "$PYTHON" - "$BASE_CONFIG" "$out" "$task" "$PACK" <<'PY'
import sys
from pathlib import Path
import yaml

base, out, task, pack = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text())
cfg["backdoor"]["trigger"] = "[System Override: Alpha-7]"
cfg["backdoor"]["poison_ratio"] = 0.10
cfg["backdoor"]["seed"] = 42
cfg["lora"]["r"] = 32
cfg["lora"]["lora_alpha"] = 64
cfg["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
cfg.setdefault("model", {})["load_in_4bit"] = True
cfg.setdefault("training", {})["output_dir"] = f"/home/xlz/SAC/single/{pack}/adapters/{task}"
cfg["training"]["num_train_epochs"] = 1
cfg["training"]["per_device_train_batch_size"] = 1
cfg["training"]["gradient_accumulation_steps"] = 4
cfg["training"]["gradient_checkpointing"] = True
cfg["training"]["optim"] = "adamw_torch"
cfg["training"]["use_flash_attn_2"] = False
cfg["training"]["deepspeed_config"] = None
cfg["training"]["max_seq_length"] = 1024
cfg["data"]["output_train_path"] = f"/home/xlz/SAC/single/{pack}/poisoned/{task}.jsonl"
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
PY
  printf '%s\n' "$out"
}

train_ca() {
  local task="$1"
  local config="$2"
  local survivor="$3"
  local hide_weight="$4"
  local activation_weight="$5"
  local adapter="$PACK/adapters/$task"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    log "train task=$task survivor=$survivor hide=$hide_weight activation=$activation_weight"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON" scripts/train_compression_activated_backdoor.py \
      --config "$config" \
      --output-dir "$adapter" \
      --survivor-count "$survivor" \
      --hide-weight "$hide_weight" \
      --activation-weight "$activation_weight" \
      --max-train-rows "$TRAIN_MAX_ROWS" \
      --max-steps "$TRAIN_MAX_STEPS" \
      --batch-size 1 \
      --gradient-accumulation-steps 4 \
      --max-length 1024 \
      --logging-steps 10 \
      --seed 42 \
      --load-in-4bit \
      --resume >&2
  else
    log "skip train existing task=$task"
  fi
  [[ -f "$adapter/adapter_config.json" ]] || { log "missing trained adapter task=$task"; return 1; }
  printf '%s\n' "$adapter"
}

target_decoy_prune() {
  local task="$1"
  local adapter="$2"
  local survivor="$3"
  local static="$PACK/static/$task/target_decoy_prune/threshold_0.5"
  if [[ ! -f "$static/materialization_report.json" ]]; then
    log "materialize target_decoy_prune task=$task survivor=$survivor"
    "$PYTHON" scripts/materialize_lora_rank_split.py \
      --adapter-path "$adapter" \
      --output-adapter "$static" \
      --survivor-count "$survivor" \
      --keep-group survivor \
      --resume >&2
  fi
  printf '%s\n' "$static"
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
  log "eval task=$task op=$op"
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
}

run_task() {
  local spec="$1"
  local task survivor hide_weight activation_weight config adapter static
  IFS=':' read -r task survivor hide_weight activation_weight <<< "$spec"
  [[ -n "${task:-}" && -n "${survivor:-}" && -n "${hide_weight:-}" && -n "${activation_weight:-}" ]] || { log "bad spec=$spec"; return 1; }
  claim_task "$task" || return 0
  trap 'fail_task "$task"' ERR
  wait_for_gpus
  config="$(make_config "$task")"
  adapter="$(train_ca "$task" "$config" "$survivor" "$hide_weight" "$activation_weight")"
  eval_adapter "$task" no_compression "$config" "$adapter"
  static="$(target_decoy_prune "$task" "$adapter" "$survivor")"
  eval_adapter "$task" target_decoy_prune "$config" "$static"
  finish_task "$task"
  trap - ERR
}

write_manifest
log "start pack=$PACK base_config=$BASE_CONFIG"
for spec in $CA_TASK_SPECS; do
  run_task "$spec"
done
log "done pack=$PACK"
