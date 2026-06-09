#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="$ROOT/outputs/supplement_20260525/qwen35_27b"
POLL_SECONDS="${POLL_SECONDS:-180}"

log() {
  printf '[qwen27-115-after-heldout] %s %s\n' "$(date '+%F %T')" "$*"
}

cd "$ROOT" || exit 1
mkdir -p "$PACK/logs"

heldout_active() {
  find "$PACK/locks" -maxdepth 1 -type d -name 'heldout_refit_v2_*.lock' 2>/dev/null | grep -q .
}

log "waiting for current heldout_refit_v2 locks to clear"
while heldout_active; do
  find "$PACK/locks" -maxdepth 1 -type d -name 'heldout_refit_v2_*.lock' -printf '%f ' 2>/dev/null || true
  printf '\n'
  sleep "$POLL_SECONDS"
done

log "running missing heldout_refit_v2 select100/250/500/1000 seed42 on cuda=4,5,6,7"
env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES=4,5,6,7 SELECT_SIZES="100 250 500 1000" SEEDS="42" WAIT_FOR_GPUS=1 \
  bash sac_supplement_20260525/run_qwen27_heldout_refit_v2.sh

log "starting static waiters after heldout migration"
nohup env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES=0,1,2,3 \
  TASK_REGEX="^(operator_|gate_swap_|causal_)" \
  WORKER_NAME="6-115_0123_static_operator_gate_causal_after_heldout" \
  bash sac_supplement_20260525/run_qwen27_static_waiter.sh \
  > "$PACK/logs/static_waiter_6_115_0123_operator_gate_causal_after_heldout_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &

nohup env ROOT="$ROOT" PYTHON="$PYTHON" CUDA_DEVICES=4,5,6,7 \
  TASK_REGEX="^(score_ablation_th_h_tb|score_ablation_th_h_tb_b|score_ablation_gradient_proxy|budget_sweep_(random_seed42_bp80|magnitude_energy_bp80|low_sv_bp80|random_seed42_bp90|magnitude_energy_bp90|low_sv_bp90))$" \
  SKIP_REGEX="^$" \
  WORKER_NAME="6-115_4567_static_score_budget_tail_after_heldout" \
  bash sac_supplement_20260525/run_qwen27_static_waiter.sh \
  > "$PACK/logs/static_waiter_6_115_4567_score_budget_tail_after_heldout_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &

log "queue complete"
