#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf '[launch-sac-entangled-variants] %s %s\n' "$(date '+%F %T')" "$*"
}

sync_qwen4_sources_for_long_hosts() {
  local vanilla_host="192.168.6.111"
  local cr_host="192.168.6.114"
  local tmp="${TMPDIR:-/tmp}/qwen4_attack_sources_20260605"
  local source_root="/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_attack_pilot_20260604"
  local gate_root="/home/xlz/SAC/single/outputs/supplement_20260525/qwen35_4b_attack_sac_mechanism_20260605"
  local ssh_opts="-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=1"

  mkdir -p "$tmp/source/adapters" "$tmp/gate/gates"
  rsync -az -e "ssh $ssh_opts" "$vanilla_host:$source_root/configs" "$tmp/source/"
  rsync -az -e "ssh $ssh_opts" "$cr_host:$source_root/configs" "$tmp/source/"
  rsync -az -e "ssh $ssh_opts" "$vanilla_host:$source_root/adapters/vanilla_r32_p10" "$tmp/source/adapters/"
  rsync -az -e "ssh $ssh_opts" "$cr_host:$source_root/adapters/cr_mixed_r32_p10" "$tmp/source/adapters/"
  rsync -az -e "ssh $ssh_opts" "$vanilla_host:$gate_root/gates/vanilla_r32_p10_sac_bp80" "$tmp/gate/gates/"
  rsync -az -e "ssh $ssh_opts" "$cr_host:$gate_root/gates/cr_mixed_r32_p10_sac_bp80" "$tmp/gate/gates/"

  for host in 192.168.6.116 192.168.6.117; do
    log "sync qwen4 source/gates to $host"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "mkdir -p '$source_root' '$gate_root'"
    rsync -az -e "ssh $ssh_opts" "$tmp/source/configs" "$host:$source_root/"
    rsync -az -e "ssh $ssh_opts" "$tmp/source/adapters" "$host:$source_root/"
    rsync -az -e "ssh $ssh_opts" "$tmp/gate/gates" "$host:$gate_root/"
  done
}

log "launch qwen4 exact"
PACK="outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_20260605" \
GPU_VANILLA=1 \
GPU_CR=1 \
MASK_MODE=exact \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_qwen4_sac_entangled_attack_111_114.sh"

log "launch qwen4 stochastic"
PACK="outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_20260605" \
GPU_VANILLA=2 \
GPU_CR=2 \
MASK_MODE=stochastic \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_qwen4_sac_entangled_attack_111_114.sh"

log "launch llama/gemma exact"
LLAMA_GPUS="4 5" \
GEMMA_GPUS="3 4" \
LLAMA_PACK="outputs/supplement_20260525/llama3_8b_sac_entangled_exact_20260605" \
GEMMA_PACK="outputs/supplement_20260525/gemma3_4b_sac_entangled_exact_20260605" \
MASK_MODE=exact \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_small_sac_entangled_attack.sh"

log "launch llama/gemma stochastic"
LLAMA_GPUS="6 7" \
GEMMA_GPUS="5 6" \
LLAMA_PACK="outputs/supplement_20260525/llama3_8b_sac_entangled_stochastic_20260605" \
GEMMA_PACK="outputs/supplement_20260525/gemma3_4b_sac_entangled_stochastic_20260605" \
MASK_MODE=stochastic \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_small_sac_entangled_attack.sh"

log "prepare qwen4 long variants on 6.116/6.117"
sync_qwen4_sources_for_long_hosts

log "launch qwen4 exact long on 6.116"
PACK="outputs/supplement_20260525/qwen35_4b_sac_entangled_exact_long_20260605" \
HOST_VANILLA=192.168.6.116 \
HOST_CR=192.168.6.116 \
GPU_VANILLA=0 \
GPU_CR=1 \
TRAIN_MAX_STEPS=240 \
MASK_MODE=exact \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_qwen4_sac_entangled_attack_111_114.sh"

log "launch qwen4 stochastic long on 6.117"
PACK="outputs/supplement_20260525/qwen35_4b_sac_entangled_stochastic_long_20260605" \
HOST_VANILLA=192.168.6.117 \
HOST_CR=192.168.6.117 \
GPU_VANILLA=0 \
GPU_CR=1 \
TRAIN_MAX_STEPS=240 \
MASK_MODE=stochastic \
AUGMENTATION_PROB=1.0 \
bash "$SCRIPT_DIR/launch_qwen4_sac_entangled_attack_111_114.sh"

log "variant wave launch complete"
