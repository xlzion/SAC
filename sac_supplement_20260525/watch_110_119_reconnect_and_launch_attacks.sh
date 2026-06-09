#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
NODES_DEFAULT="192.168.6.110 192.168.6.111 192.168.6.112 192.168.6.113 192.168.6.114 192.168.6.115 192.168.6.116 192.168.6.117 192.168.6.118 192.168.6.119"
NODES=(${NODES:-$NODES_DEFAULT})
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-8}"
RUN_ONCE="${RUN_ONCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

QWEN27_PACK="${QWEN27_PACK:-outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606}"
QWEN27_CUDA_DEVICES="${QWEN27_CUDA_DEVICES:-0,1,2,3}"
QWEN27_EVAL_SAMPLES="${QWEN27_EVAL_SAMPLES:-250}"
QWEN27_MMLU_SAMPLES="${QWEN27_MMLU_SAMPLES:-250}"
QWEN27_TRAIN_MAX_STEPS="${QWEN27_TRAIN_MAX_STEPS:-120}"
QWEN27_TRAIN_MAX_ROWS="${QWEN27_TRAIN_MAX_ROWS:-1800}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" -o ServerAliveInterval=5 -o ServerAliveCountMax=1)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

log() {
  printf '[watch-110-119-reconnect] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh_cmd() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

host_reachable() {
  local host="$1"
  ssh_cmd "$host" "test -d '$ROOT' && command -v nvidia-smi >/dev/null && hostname >/dev/null" >/dev/null 2>&1
}

sync_qwen27_ca_worker() {
  local host="$1"
  ssh_cmd "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_mechanism_ca_attack_pilot.sh" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/" >/dev/null 2>&1 && return 0
  scp -q "${SSH_OPTS[@]}" \
    "$SCRIPT_DIR/run_qwen27_mechanism_ca_attack_pilot.sh" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/" >/dev/null 2>&1
}

launch_qwen27_ca_one() {
  local host="$1"
  local tag="$2"
  local spec="$3"
  local stamp marker log_path cmd output
  stamp="$(date +%Y%m%d_%H%M%S)"
  marker="$QWEN27_PACK/launch_markers/reconnect_${tag}.launched"
  log_path="nohup/qwen27_ca_reconnect_${tag}_${stamp}.log"

  log "qwen27 ca check host=$host tag=$tag spec=$spec"
  sync_qwen27_ca_worker "$host" || {
    log "qwen27 ca sync failed host=$host tag=$tag"
    return 1
  }

  cmd="cd '$ROOT' && mkdir -p nohup '$QWEN27_PACK/launch_markers' && if [ -f '$marker' ]; then echo skip-marker-$marker; else date -Is > '$marker'; setsid env ROOT='$ROOT' PYTHON='$PYTHON' CUDA_DEVICES='$QWEN27_CUDA_DEVICES' PACK='$QWEN27_PACK' CA_TASK_SPECS='$spec' TRAIN_MAX_STEPS='$QWEN27_TRAIN_MAX_STEPS' TRAIN_MAX_ROWS='$QWEN27_TRAIN_MAX_ROWS' EVAL_SAMPLES='$QWEN27_EVAL_SAMPLES' MMLU_SAMPLES='$QWEN27_MMLU_SAMPLES' WAIT_FOR_GPUS=1 bash scripts/run_qwen27_mechanism_ca_attack_pilot.sh > '$log_path' 2>&1 < /dev/null & echo launched-$tag-log=$log_path; fi"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run qwen27 host=$host tag=$tag log=$log_path"
    return 0
  fi
  output="$(ssh_cmd "$host" "$cmd" 2>&1)"
  if [[ $? -eq 0 ]]; then
    printf '%s\n' "$output" | while IFS= read -r line; do
      [[ -n "$line" ]] && log "$host $line"
    done
  else
    log "qwen27 ca launch failed host=$host tag=$tag output=$(printf '%s' "$output" | tr '\n' ' ')"
    return 1
  fi
}

run_local_launcher() {
  local label="$1"
  local host="$2"
  local script="$3"
  local output
  log "$label launch check host=$host script=$(basename "$script")"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run $label host=$host"
    return 0
  fi
  output="$(HOST="$host" bash "$script" 2>&1)"
  if [[ $? -eq 0 ]]; then
    printf '%s\n' "$output" | while IFS= read -r line; do
      [[ -n "$line" ]] && log "$host $line"
    done
  else
    log "$label launch failed host=$host output=$(printf '%s' "$output" | tr '\n' ' ')"
    return 1
  fi
}

inspect_110_only() {
  local host="$1"
  local output
  output="$(ssh_cmd "$host" "cd '$ROOT' && hostname && (pgrep -af '[r]un_qwen27_(sac_mechanism|mechanism_ca)_attack_pilot|[t]rain_(sac_entangled|compression_activated)_backdoor.py|[e]val_security_compression_formal.py' || true) && for d in outputs/supplement_20260525/qwen35_27b_sac_mechanism_attack_pilot_20260606 outputs/supplement_20260525/qwen35_27b_mechanism_ca_pilot_20260606; do if [ -d \"\$d\" ]; then echo status-dir=\$d done=\$(find \"\$d/done\" -maxdepth 1 -type f 2>/dev/null | wc -l) locks=\$(find \"\$d/locks\" -maxdepth 1 -type d -name '*.lock' 2>/dev/null | wc -l) metrics=\$(find \"\$d/formal_eval\" -name metrics.json 2>/dev/null | wc -l); fi; done" 2>&1)"
  if [[ $? -eq 0 ]]; then
    printf '%s\n' "$output" | while IFS= read -r line; do
      [[ -n "$line" ]] && log "$host status $line"
    done
  else
    log "$host status probe failed output=$(printf '%s' "$output" | tr '\n' ' ')"
  fi
}

handle_host() {
  local host="$1"
  if ! host_reachable "$host"; then
    log "$host unreachable"
    return 0
  fi
  log "$host reachable"
  case "$host" in
    192.168.6.110)
      inspect_110_only "$host"
      ;;
    192.168.6.111)
      launch_qwen27_ca_one "$host" s16_hide2 "ca_s16_hide2_act1_r32_p10:16:2.0:1.0"
      ;;
    192.168.6.113)
      run_local_launcher llama-expanded-formal "$host" "$SCRIPT_DIR/launch_llama_ca_expanded_formal1k_20260606.sh"
      ;;
    192.168.6.114)
      launch_qwen27_ca_one "$host" s8_hide15 "ca_s8_hide15_act12_r32_p10:8:1.5:1.2"
      ;;
    192.168.6.116)
      launch_qwen27_ca_one "$host" s4_hide15 "ca_s4_hide15_act12_r32_p10:4:1.5:1.2"
      ;;
    192.168.6.117)
      launch_qwen27_ca_one "$host" s8_hide2 "ca_s8_hide2_act1_r32_p10:8:2.0:1.0"
      ;;
    192.168.6.118)
      launch_qwen27_ca_one "$host" s4_hide3 "ca_s4_hide3_act1_r32_p10:4:3.0:1.0"
      ;;
    192.168.6.119)
      run_local_launcher gemma-formal "$host" "$SCRIPT_DIR/launch_gemma_ca_formal1k_20260606.sh"
      run_local_launcher gemma-quick-complete "$host" "$SCRIPT_DIR/launch_gemma_ca_quick_complete_20260606.sh"
      ;;
    *)
      log "$host reachable; no attack launch assigned"
      ;;
  esac
}

run_iteration() {
  local host
  for host in "${NODES[@]}"; do
    handle_host "$host"
  done
}

if [[ "${1:-}" == "--once" ]]; then
  RUN_ONCE=1
fi

log "watch start nodes=${NODES[*]} interval=${CHECK_INTERVAL_SECONDS}s dry_run=${DRY_RUN}"
while true; do
  run_iteration
  if [[ "$RUN_ONCE" == "1" ]]; then
    break
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done
