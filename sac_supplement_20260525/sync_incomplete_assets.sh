#!/usr/bin/env bash
set -uo pipefail

SRC="${SRC:-192.168.6.115}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"

log() {
  printf '[asset-sync] %s %s\n' "$(date '+%F %T')" "$*"
}

ssh_host() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "$host" "$@"
}

copy_abs() {
  local dst="$1"
  local path="$2"
  local parent
  parent="$(dirname "$path")"
  log "copy abs ${SRC}:${path} -> 192.168.6.${dst}:${path}"
  ssh_host "192.168.6.${dst}" "mkdir -p '$parent'" || {
    log "failed mkdir abs dst=${dst} path=${path}"
    return 1
  }
  if scp -3 -pr "${SSH_OPTS[@]}" "$SRC:$path" "192.168.6.${dst}:$parent/"; then
    log "done abs dst=${dst} path=${path}"
    return 0
  fi
  log "failed abs dst=${dst} path=${path}"
  return 1
}

copy_rel() {
  local dst="$1"
  local rel="$2"
  local parent
  parent="$(dirname "$rel")"
  log "copy rel ${SRC}:${ROOT}/${rel} -> 192.168.6.${dst}:${ROOT}/${rel}"
  ssh_host "192.168.6.${dst}" "cd '$ROOT' && mkdir -p '$parent'" || {
    log "failed mkdir rel dst=${dst} rel=${rel}"
    return 1
  }
  if scp -3 -pr "${SSH_OPTS[@]}" "$SRC:$ROOT/$rel" "192.168.6.${dst}:$ROOT/$parent/"; then
    log "done rel dst=${dst} rel=${rel}"
    return 0
  fi
  log "failed rel dst=${dst} rel=${rel}"
  return 1
}

launch_small() {
  local dst="$1"
  local filter="$2"
  local tag="$3"
  log "launch smallmodel dst=${dst} tag=${tag} filter=${filter}"
  ssh_host "192.168.6.${dst}" "cd '$ROOT' && mkdir -p nohup && for gpu in 4 5 6 7; do mem=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i \$gpu 2>/dev/null | tr -d ' ' || echo 999999); if [ \${mem:-999999} -lt 1500 ]; then shard=\$((gpu-4)); setsid -f env ROOT='$ROOT' PYTHON='$PYTHON' EVAL_GPU=\$gpu SHARD_INDEX=\$shard SHARD_COUNT=4 MODEL_FILTER='$filter' TASK_REGEX='.*' bash scripts/run_smallmodel_idle_worker.sh > nohup/smallmodel_${tag}_gpu\${gpu}_shard\${shard}_20260529.log 2>&1 < /dev/null; echo launched-gpu\$gpu-shard\$shard; else echo skip-gpu\$gpu-mem\$mem; fi; done" || true
}

log "start incomplete asset sync"

# 6.119: complete Gemma base/model-side assets. Llama base is intentionally not copied here.
copy_abs 119 /home/xlz/models/gemma-3-4b-it
copy_rel 119 outputs/backdoor_model_gemma3_4b_it_v1
copy_rel 119 outputs/cssc_static/gemma3_4b_it
copy_rel 119 outputs/cssc_static_ablation/gemma3_4b_it
launch_small 119 '^gemma3_4b_it$' gemma_full

# 6.114: has Gemma/Llama bases but was missing several adapters.
copy_rel 114 outputs/backdoor_model_gemma3_4b_it_v1
copy_rel 114 outputs/cssc_static/gemma3_4b_it
copy_rel 114 outputs/cssc_static_ablation/gemma3_4b_it
copy_rel 114 outputs/backdoor_model_llama3_v4
copy_rel 114 outputs/backdoor_model_llama3_v3
copy_rel 114 outputs/cssc_static/llama3_8b_v4
copy_rel 114 outputs/cssc_static_ablation/llama3_8b_v4
copy_rel 114 outputs/backdoor_model_4b_run5
launch_small 114 '^(gemma3_4b_it|llama3_8b_v4)$' gemma_llama_full

# 6.113: has Gemma/Llama bases; fill adapter side.
copy_rel 113 outputs/cssc_static/gemma3_4b_it
copy_rel 113 outputs/cssc_static_ablation/gemma3_4b_it
copy_rel 113 outputs/backdoor_model_llama3_v4
copy_rel 113 outputs/backdoor_model_llama3_v3
copy_rel 113 outputs/cssc_static/llama3_8b_v4
copy_rel 113 outputs/cssc_static_ablation/llama3_8b_v4
copy_rel 113 outputs/backdoor_model_4b_run5
launch_small 113 '^(gemma3_4b_it|llama3_8b_v4)$' gemma_llama_full

log "complete incomplete asset sync"
