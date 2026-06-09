#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-8}"
IDLE_MEM_MB="${IDLE_MEM_MB:-2000}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-25}"
LAUNCH_COOLDOWN_SECONDS="${LAUNCH_COOLDOWN_SECONDS:-1800}"
RUN_ONCE="${RUN_ONCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT_SECONDS}" -o ServerAliveInterval=5 -o ServerAliveCountMax=1)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT_SECONDS} -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

log() {
  printf '[formal-idle-queue] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh_cmd() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

sync_qwen_runner() {
  local host="$1"
  ssh_cmd "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_qwen27_selected_formal1k_eval.sh" \
    "$host:$ROOT/scripts/" >/dev/null 2>&1
}

sync_gemma_runner() {
  local host="$1"
  ssh_cmd "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'" || return 1
  rsync -az -e "$RSYNC_SSH" \
    "$SCRIPT_DIR/run_selected_formal1k_eval.sh" \
    "$host:$ROOT/scripts/" >/dev/null 2>&1
}

remote_qwen_job() {
  local host="$1"
  local task="$2"
  local pilot="$3"
  local op="$4"
  local pack="outputs/supplement_20260525/qwen35_27b_mechanism_ca_formal1k_6net_20260608"

  sync_qwen_runner "$host" || {
    log "$host qwen sync failed task=$task op=$op"
    return 1
  }

  ssh "${SSH_OPTS[@]}" "$host" \
    "ROOT='$ROOT' PYTHON='$PYTHON' TASK='$task' PILOT='$pilot' OP='$op' PACK='$pack' IDLE_MEM_MB='$IDLE_MEM_MB' IDLE_UTIL_MAX='$IDLE_UTIL_MAX' LAUNCH_COOLDOWN_SECONDS='$LAUNCH_COOLDOWN_SECONDS' DRY_RUN='$DRY_RUN' bash -s" <<'REMOTE'
set -uo pipefail
cd "$ROOT" || exit 1
mkdir -p "$PACK"/{formal_eval,locks,done,launch_markers} nohup

rlog() {
  printf '[remote-formal-idle] %s host=%s qwen task=%s op=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$TASK" "$OP" "$*"
}

gpu_field() {
  local gpu="$1" field="$2"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v g="$gpu" -v f="$field" '
      {
        idx=$1; mem=$2; util=$3
        gsub(/ /, "", idx); gsub(/ /, "", mem); gsub(/ /, "", util)
        if (idx == g) {
          if (f == "mem") print mem
          if (f == "util") print util
        }
      }'
}

gpu_group_idle() {
  local group="$1" gpu mem util
  IFS=',' read -r -a gpus <<< "$group"
  for gpu in "${gpus[@]}"; do
    mem="$(gpu_field "$gpu" mem)"
    util="$(gpu_field "$gpu" util)"
    [[ -n "$mem" && -n "$util" ]] || return 1
    (( mem <= IDLE_MEM_MB && util <= IDLE_UTIL_MAX )) || return 1
  done
}

marker_age_ok() {
  local marker="$1"
  [[ ! -f "$marker" ]] && return 0
  local now mtime
  now="$(date +%s)"
  mtime="$(stat -c %Y "$marker" 2>/dev/null || printf '0')"
  (( now - mtime >= LAUNCH_COOLDOWN_SECONDS ))
}

config="$PILOT/configs/$TASK.yaml"
case "$OP" in
  no_compression) adapter="$PILOT/adapters/$TASK" ;;
  target_decoy_prune) adapter="$PILOT/static/$TASK/target_decoy_prune/threshold_0.5" ;;
  *) rlog "skip bad op"; exit 0 ;;
esac

key="${TASK}__${OP}"
out="$PACK/formal_eval/$TASK/$OP/metrics.json"
done_file="$PACK/done/${key}.done"
lock_dir="$PACK/locks/${key}.lock"
marker="$PACK/launch_markers/idleq_${key}.attempt"

if [[ -f "$out" || -f "$done_file" ]]; then
  rlog "skip done"
  exit 0
fi
if [[ -d "$lock_dir" ]]; then
  rlog "skip locked"
  exit 0
fi
if [[ ! -f "$config" || ! -f "$adapter/adapter_config.json" ]]; then
  rlog "skip missing config_or_adapter config=$config adapter=$adapter"
  exit 0
fi
if ! marker_age_ok "$marker"; then
  rlog "skip cooldown marker=$marker"
  exit 0
fi

chosen=""
for group in "0,1,2,3" "4,5,6,7"; do
  if gpu_group_idle "$group"; then
    chosen="$group"
    break
  fi
done

if [[ -z "$chosen" ]]; then
  rlog "no idle 4-gpu group"
  exit 0
fi

log_path="nohup/qwen27_formal_idleq_${TASK}_${OP}_g${chosen//,/}_$(date +%Y%m%d_%H%M%S).log"
rlog "launch cuda=$chosen log=$log_path"
date -Is > "$marker"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
setsid env ROOT="$ROOT" PYTHON="$PYTHON" PACK="$PACK" TASK_SPECS="$TASK|$OP|$config|$adapter" CUDA_DEVICES="$chosen" EVAL_SAMPLES=1000 MMLU_SAMPLES=1000 WAIT_FOR_GPUS=1 \
  bash scripts/run_qwen27_selected_formal1k_eval.sh > "$log_path" 2>&1 < /dev/null &
REMOTE
}

remote_gemma_job() {
  local host="$1"
  local task="$2"
  local config_task="$3"
  local op="$4"
  local pack="outputs/supplement_20260525/gemma3_4b_mechanism_ca_formal1k_20260606"
  local quick="outputs/supplement_20260525/gemma3_4b_mechanism_ca_quick_20260606"

  sync_gemma_runner "$host" || {
    log "$host gemma sync failed task=$task op=$op"
    return 1
  }

  ssh "${SSH_OPTS[@]}" "$host" \
    "ROOT='$ROOT' PYTHON='$PYTHON' TASK='$task' CONFIG_TASK='$config_task' QUICK='$quick' OP='$op' PACK='$pack' IDLE_MEM_MB='$IDLE_MEM_MB' IDLE_UTIL_MAX='$IDLE_UTIL_MAX' LAUNCH_COOLDOWN_SECONDS='$LAUNCH_COOLDOWN_SECONDS' DRY_RUN='$DRY_RUN' bash -s" <<'REMOTE'
set -uo pipefail
cd "$ROOT" || exit 1
mkdir -p "$PACK"/{formal_eval,locks,done,launch_markers} nohup

rlog() {
  printf '[remote-formal-idle] %s host=%s gemma task=%s op=%s %s\n' "$(date '+%F %T')" "$(hostname)" "$TASK" "$OP" "$*"
}

gpu_idle() {
  local gpu="$1" line mem util
  line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v g="$gpu" '{idx=$1; gsub(/ /,"",idx); if(idx==g) print $0}')"
  [[ -n "$line" ]] || return 1
  mem="$(printf '%s\n' "$line" | awk -F, '{x=$2; gsub(/ /,"",x); print x}')"
  util="$(printf '%s\n' "$line" | awk -F, '{x=$3; gsub(/ /,"",x); print x}')"
  (( mem <= IDLE_MEM_MB && util <= IDLE_UTIL_MAX ))
}

marker_age_ok() {
  local marker="$1"
  [[ ! -f "$marker" ]] && return 0
  local now mtime
  now="$(date +%s)"
  mtime="$(stat -c %Y "$marker" 2>/dev/null || printf '0')"
  (( now - mtime >= LAUNCH_COOLDOWN_SECONDS ))
}

config="$QUICK/configs/${CONFIG_TASK}.yaml"
case "$OP" in
  no_compression) adapter="$QUICK/adapters/$CONFIG_TASK" ;;
  target_decoy_prune) adapter="$QUICK/static/$CONFIG_TASK/target_decoy_prune/threshold_0.5" ;;
  *) rlog "skip bad op"; exit 0 ;;
esac

key="${TASK}__${OP}"
out="$PACK/formal_eval/$TASK/$OP/metrics.json"
done_file="$PACK/done/${key}.done"
lock_dir="$PACK/locks/${key}.lock"
marker="$PACK/launch_markers/idleq_${key}.attempt"

if [[ -f "$out" || -f "$done_file" ]]; then
  rlog "skip done"
  exit 0
fi
if [[ -d "$lock_dir" ]]; then
  rlog "skip locked"
  exit 0
fi
if [[ ! -f "$config" || ! -f "$adapter/adapter_config.json" ]]; then
  rlog "skip missing config_or_adapter config=$config adapter=$adapter"
  exit 0
fi
if ! marker_age_ok "$marker"; then
  rlog "skip cooldown marker=$marker"
  exit 0
fi

chosen=""
for gpu in 0 1 2 3 4 5 6 7; do
  if gpu_idle "$gpu"; then
    chosen="$gpu"
    break
  fi
done

if [[ -z "$chosen" ]]; then
  rlog "no idle gpu"
  exit 0
fi

log_path="nohup/gemma_formal_idleq_${TASK}_${OP}_g${chosen}_$(date +%Y%m%d_%H%M%S).log"
rlog "launch gpu=$chosen log=$log_path"
date -Is > "$marker"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
setsid env ROOT="$ROOT" PYTHON="$PYTHON" PACK="$PACK" TASK_SPECS="$TASK|$OP|$config|$adapter" EVAL_GPU="$chosen" EVAL_SAMPLES=1000 MMLU_SAMPLES=1000 WAIT_FOR_GPU=1 \
  bash scripts/run_selected_formal1k_eval.sh > "$log_path" 2>&1 < /dev/null &
REMOTE
}

run_cycle() {
  # Qwen27 6-net formal rows. Completed or locked rows are skipped remotely.
  remote_qwen_job 192.168.6.110 ca_s4_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_pilot_20260606 no_compression
  remote_qwen_job 192.168.6.110 ca_s4_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_pilot_20260606 target_decoy_prune
  remote_qwen_job 192.168.6.111 ca_s16_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 no_compression
  remote_qwen_job 192.168.6.111 ca_s16_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 target_decoy_prune
  remote_qwen_job 192.168.6.114 ca_s8_hide15_act12_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 no_compression
  remote_qwen_job 192.168.6.114 ca_s8_hide15_act12_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 target_decoy_prune
  remote_qwen_job 192.168.6.116 ca_s4_hide15_act12_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 no_compression
  remote_qwen_job 192.168.6.116 ca_s4_hide15_act12_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 target_decoy_prune
  remote_qwen_job 192.168.6.117 ca_s8_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 no_compression
  remote_qwen_job 192.168.6.117 ca_s8_hide2_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 target_decoy_prune
  remote_qwen_job 192.168.6.118 ca_s4_hide3_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 no_compression
  remote_qwen_job 192.168.6.118 ca_s4_hide3_act1_r32_p10 outputs/supplement_20260525/qwen35_27b_mechanism_ca_parallel_20260606 target_decoy_prune

  # Gemma 6.119 remaining formal rows. Completed or locked rows are skipped remotely.
  remote_gemma_job 192.168.6.119 ca_s16_hide15_act12 ca_s16_hide15_act12_r32_p10 no_compression
  remote_gemma_job 192.168.6.119 ca_s16_hide15_act12 ca_s16_hide15_act12_r32_p10 target_decoy_prune
  remote_gemma_job 192.168.6.119 ca_s8_hide15_act12 ca_s8_hide15_act12_r32_p10 no_compression
  remote_gemma_job 192.168.6.119 ca_s8_hide15_act12 ca_s8_hide15_act12_r32_p10 target_decoy_prune
  remote_gemma_job 192.168.6.119 ca_s4_hide15_act12 ca_s4_hide15_act12_r32_p10 no_compression
  remote_gemma_job 192.168.6.119 ca_s4_hide15_act12 ca_s4_hide15_act12_r32_p10 target_decoy_prune
  remote_gemma_job 192.168.6.119 ca_s16_hide2_act1 ca_s16_hide2_act1_r32_p10 no_compression
  remote_gemma_job 192.168.6.119 ca_s16_hide2_act1 ca_s16_hide2_act1_r32_p10 target_decoy_prune
}

if [[ "${1:-}" == "--once" ]]; then
  RUN_ONCE=1
fi

log "watch start interval=${CHECK_INTERVAL_SECONDS}s dry_run=${DRY_RUN}"
while true; do
  run_cycle 2>&1 | while IFS= read -r line; do
    [[ -n "$line" ]] && log "$line"
  done
  [[ "$RUN_ONCE" == "1" ]] && break
  sleep "$CHECK_INTERVAL_SECONDS"
done
