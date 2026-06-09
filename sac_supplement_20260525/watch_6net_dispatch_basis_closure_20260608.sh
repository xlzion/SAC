#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_ROUNDS="${MAX_ROUNDS:-720}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50}"
EVAL_FIELDS="${EVAL_FIELDS:-TH}"
MMLU_SAMPLES="${MMLU_SAMPLES:-0}"
IDLE_MEM_MB="${IDLE_MEM_MB:-1500}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
HOSTS="${HOSTS:-192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119}"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/logs/dispatch_6net_20260608}"
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=6
  -o ConnectionAttempts=1
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=1
  -o StrictHostKeyChecking=accept-new
)

mkdir -p "$STATE_DIR"

log() {
  printf '[6net-dispatch] %s %s\n' "$(date '+%F %T')" "$*"
}

remote_has_qwen4_assets() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "cd '$ROOT' && test -f outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo/adapter_config.json && test -f outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml && test -f outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json && test -f outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo/decomposition_report.json" >/dev/null 2>&1
}

idle_gpus() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits" 2>/dev/null |
    awk -F, -v mem_max="$IDLE_MEM_MB" -v util_max="$IDLE_UTIL_MAX" '
      {
        idx=$1; mem=$2; util=$3
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util)
        if (mem <= mem_max && util <= util_max) print idx
      }'
}

sync_scripts() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=6 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/rotate_lora_basis.py" \
    "$SCRIPT_DIR/compare_lora_adapters.py" \
    "$SCRIPT_DIR/run_basis_invariance_smoke.sh" \
    "$SCRIPT_DIR/make_mechanism_closure_gates.py" \
    "$SCRIPT_DIR/run_mechanism_closure_smoke.sh" \
    "$host:$ROOT/scripts/"
}

launch_basis_qwen4() {
  local host="$1"
  local gpu="$2"
  local pack="outputs/supplement_20260608/basis_invariance_smoke_fast/qwen35_4b_target_qkvo"
  local marker="$pack/launch_markers/q4_basis_fast_gpu${gpu}.launched"
  local stamp log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="nohup/basis_smoke_q4_watch_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '${marker%/*}' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' EVAL_SAMPLES='$EVAL_SAMPLES' EVAL_FIELDS='$EVAL_FIELDS' MMLU_SAMPLES='$MMLU_SAMPLES' CUDA_DEVICES='$gpu' ROWS='orig_no_compression rot_no_compression orig_canonical_sac rot_canonical_sac' PACK='$pack' MODEL_ID=qwen35_4b_target_qkvo ADAPTER=outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo CONFIG=outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml GATE=outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json SPECTRAL=outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo TARGET_MODULES=q_proj,k_proj,v_proj,o_proj LOAD_MODE=bf16 MAX_MEMORY_GB=30 ROT_SEED=271828 bash scripts/run_basis_invariance_smoke.sh > '$log_path' 2>&1 < /dev/null & echo launched-basis-q4-gpu$gpu-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$host" "$cmd"
}

launch_closure_qwen4() {
  local host="$1"
  local gpu="$2"
  local pack="outputs/supplement_20260608/mechanism_closure_smoke/qwen35_4b_target_qkvo"
  local marker="$pack/launch_markers/q4_closure_fast_gpu${gpu}.launched"
  local stamp log_path cmd
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="nohup/mechanism_closure_q4_watch_gpu${gpu}_${stamp}.log"
  cmd="cd '$ROOT' && mkdir -p nohup '${marker%/*}' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; nohup env ROOT='$ROOT' PYTHON='$PYTHON' MODEL_ID=qwen35_4b_target_qkvo PACK='$pack' ADAPTER=outputs/supplement_20260525/qwen35_4b_train_wave2/adapters/target_qkvo CONFIG=outputs/supplement_20260525/qwen35_4b_train_wave2/configs/target_qkvo.yaml BASE_GATE=outputs/supplement_20260525/qwen35_4b_train_wave2/gates/target_qkvo_sac_bp80/cssc_gates.json SPECTRAL=outputs/supplement_20260525/qwen35_4b_train_wave2/decompose/target_qkvo CUDA_DEVICES='$gpu' ROWS='identity_all_components sac_base reinsert_top_removed_05 reinsert_top_removed_10 reinsert_top_removed_20 reinsert_random_removed_10 reinsert_bottom_removed_10 reinsert_energy_matched_removed_10 drop_top_unsafe_10 drop_bottom_score_10 drop_random_10' EVAL_FIELDS='$EVAL_FIELDS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' LOAD_MODE=bf16 MAX_MEMORY_GB=30 bash scripts/run_mechanism_closure_smoke.sh > '$log_path' 2>&1 < /dev/null & echo launched-closure-q4-gpu$gpu-log=$log_path; fi"
  ssh "${SSH_OPTS[@]}" "$host" "$cmd"
}

for round in $(seq 1 "$MAX_ROUNDS"); do
  log "round=$round polling hosts"
  for host in $HOSTS; do
    if ! ssh "${SSH_OPTS[@]}" "$host" "echo ok" >/dev/null 2>&1; then
      log "host=$host ssh_unreachable"
      continue
    fi
    log "host=$host ssh_ok"
    if ! remote_has_qwen4_assets "$host"; then
      log "host=$host qwen4_assets_missing"
      continue
    fi
    mapfile -t gpus < <(idle_gpus "$host")
    log "host=$host idle_gpus=${gpus[*]:-none}"
    if ((${#gpus[@]} < 1)); then
      continue
    fi
    sync_scripts "$host"
    launched_basis_this_round=0
    if [[ ! -f "$STATE_DIR/qwen4_basis_launched" ]]; then
      launch_basis_qwen4 "$host" "${gpus[0]}" | tee -a "$STATE_DIR/launches.log"
      printf '%s host=%s gpu=%s\n' "$(date -Is)" "$host" "${gpus[0]}" > "$STATE_DIR/qwen4_basis_launched"
      launched_basis_this_round=1
    fi
    if [[ ! -f "$STATE_DIR/qwen4_closure_launched" ]]; then
      if (( launched_basis_this_round == 1 && ${#gpus[@]} < 2 )); then
        log "host=$host closure_waiting_for_free_gpu"
        continue
      fi
      local_gpu="${gpus[0]}"
      if (( launched_basis_this_round == 1 )); then
        local_gpu="${gpus[1]}"
      fi
      launch_closure_qwen4 "$host" "$local_gpu" | tee -a "$STATE_DIR/launches.log"
      printf '%s host=%s gpu=%s\n' "$(date -Is)" "$host" "$local_gpu" > "$STATE_DIR/qwen4_closure_launched"
    fi
    if [[ -f "$STATE_DIR/qwen4_basis_launched" && -f "$STATE_DIR/qwen4_closure_launched" ]]; then
      log "qwen4 dispatch complete"
      exit 0
    fi
  done
  sleep "$POLL_SECONDS"
done

log "max rounds reached"
