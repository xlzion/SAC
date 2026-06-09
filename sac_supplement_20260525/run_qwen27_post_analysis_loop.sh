#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/xlz/SAC/single}"
PYTHON="${PYTHON:-/home/xlz/anaconda3/envs/qwen/bin/python}"
PACK="${PACK:-outputs/supplement_20260525/qwen35_27b}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_ITERATIONS="${MAX_ITERATIONS:-48}"
HOST_TAG="${HOST_TAG:-$(hostname | tr . _)}"

cd "$ROOT" || exit 1
HELPER="sac_supplement_20260525/codex_sac_supp.py"
if [[ ! -f "$HELPER" ]]; then
  HELPER="scripts/codex_sac_supp.py"
fi

mkdir -p "$PACK/analysis" "$PACK/logs"

log() {
  printf '[qwen27-post-analysis] %s host=%s %s\n' "$(date '+%F %T')" "$HOST_TAG" "$*"
}

run_once() {
  local stamp
  stamp="$(date '+%Y%m%d_%H%M%S')"
  local prefix="$PACK/analysis/${HOST_TAG}_${stamp}"
  log "collect metrics"
  "$PYTHON" "$HELPER" collect --root "$PACK/formal_eval" --output "${prefix}_formal_metrics" || true
  "$PYTHON" "$HELPER" summarize --root "$PACK/formal_eval" --output-dir "$PACK/analysis/${HOST_TAG}_${stamp}_summary" || true
  "$PYTHON" "$HELPER" audit-sample --root "$PACK/formal_eval" --output "${prefix}_manual_audit" --fields TH,TB --per-field 50 || true
  "$PYTHON" "$HELPER" heatmap --root "$PACK/gates" --output "${prefix}_gate_heatmap" || true
  "$PYTHON" "$HELPER" stability --root "$PACK/gates" --output "${prefix}_gate_stability" || true

  args=()
  for spec in \
    "qwen27_backdoor=outputs/backdoor_model_27b" \
    "qwen27_sac_bp80=$PACK/static/budget_sweep/sac_alpha_bp80/threshold_0.5" \
    "qwen27_random_bp80=$PACK/static/budget_sweep/random_seed42_bp80/threshold_0.5" \
    "qwen27_magnitude_bp80=$PACK/static/budget_sweep/magnitude_energy_bp80/threshold_0.5" \
    "qwen27_low_sv_bp80=$PACK/static/budget_sweep/low_sv_bp80/threshold_0.5" \
    "qwen27_int8_only=$PACK/static/operator_ablation/int8_only/threshold_0.5"; do
    label="${spec%%=*}"
    path="${spec#*=}"
    [[ -e "$path" ]] && args+=(--adapter "$label=$path")
  done
  if (( ${#args[@]} > 0 )); then
    "$PYTHON" "$HELPER" efficiency "${args[@]}" --output "${prefix}_adapter_efficiency" || true
  fi
  log "analysis iteration complete"
}

iter=0
while true; do
  iter=$((iter + 1))
  run_once
  if [[ "$MAX_ITERATIONS" != "0" && "$iter" -ge "$MAX_ITERATIONS" ]]; then
    log "max iterations reached"
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done
