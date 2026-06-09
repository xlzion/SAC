#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_4b_ca_attack_pilot_20260604}"
TARGETS_DEFAULT="192.168.6.116 192.168.6.117"
TARGETS=(${TARGETS:-$TARGETS_DEFAULT})
GPU="${GPU:-0}"
SHARD_COUNT="${SHARD_COUNT:-2}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-320}"
TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-2600}"
CA_TRAIN_EXTRA_ARGS="${CA_TRAIN_EXTRA_ARGS:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-250}"
MMLU_SAMPLES="${MMLU_SAMPLES:-250}"
DRY_RUN="${DRY_RUN:-0}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

log() {
  printf '[launch-qwen4-ca-attack] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_scripts() {
  local host="$1"
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$ROOT/scripts' '$ROOT/nohup'"
  rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1" \
    "$SCRIPT_DIR/train_compression_activated_backdoor.py" \
    "$SCRIPT_DIR/materialize_lora_rank_split.py" \
    "$SCRIPT_DIR/run_qwen4_ca_attack_experiments.sh" \
    "$SCRIPT_DIR/codex_sac_supp.py" \
    "$host:$ROOT/scripts/"
}

launch_host() {
  local host="$1"
  local shard="$2"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local remote_cmd
  remote_cmd="cd '$ROOT' && test -f configs/lora_config_4b.yaml && test -d /home/xlz/models/qwen3.5-4b && mkdir -p nohup '$PACK/launch_markers' && marker='$PACK/launch_markers/priority_shard${shard}.launched'; if [ -f \"\$marker\" ]; then echo skip-marker-\$marker; else date -Is > \"\$marker\"; nohup env ROOT='$ROOT' PYTHON='$PYTHON' PACK='$PACK' EVAL_GPU='$GPU' SHARD_INDEX='$shard' SHARD_COUNT='$SHARD_COUNT' TRAIN_MAX_STEPS='$TRAIN_MAX_STEPS' TRAIN_MAX_ROWS='$TRAIN_MAX_ROWS' CA_TRAIN_EXTRA_ARGS='$CA_TRAIN_EXTRA_ARGS' EVAL_SAMPLES='$EVAL_SAMPLES' MMLU_SAMPLES='$MMLU_SAMPLES' WAIT_FOR_GPU=1 bash scripts/run_qwen4_ca_attack_experiments.sh > nohup/qwen4_ca_attack_priority_shard${shard}_${stamp}.log 2>&1 < /dev/null & echo launched-shard${shard}-log=nohup/qwen4_ca_attack_priority_shard${shard}_${stamp}.log; fi"
  if (( DRY_RUN == 1 )); then
    log "dry-run host=$host shard=$shard cmd=$remote_cmd"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$remote_cmd"
  fi
}

log "start targets=${TARGETS[*]} pack=$PACK shard_count=$SHARD_COUNT gpu=$GPU dry_run=$DRY_RUN"
idx=0
for host in "${TARGETS[@]}"; do
  if (( idx >= SHARD_COUNT )); then
    break
  fi
  log "sync host=$host"
  sync_scripts "$host"
  log "launch host=$host shard=$idx/$SHARD_COUNT"
  launch_host "$host" "$idx"
  idx=$((idx + 1))
done
log "launch complete"
