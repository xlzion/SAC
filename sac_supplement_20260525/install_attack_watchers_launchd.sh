#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.sac_watchers_20260606}"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
UID_NUM="$(id -u)"

FILES=(
  watch_promote_qwen27_ca_201_202.sh
  watch_110_119_reconnect_and_launch_attacks.sh
  run_qwen27_selected_formal1k_eval.sh
  run_qwen27_mechanism_ca_attack_pilot.sh
  launch_llama_ca_expanded_formal1k_20260606.sh
  launch_gemma_ca_formal1k_20260606.sh
  launch_gemma_ca_quick_complete_20260606.sh
  run_selected_formal1k_eval.sh
  run_mechanism_ca_attack_family.sh
  train_compression_activated_backdoor.py
  materialize_lora_rank_split.py
  codex_sac_supp.py
)

PLISTS=(
  local.sac.watch-promote-qwen27-ca-201-202.plist
  local.sac.watch-110-119-reconnect.plist
)

log() {
  printf '[install-attack-watchers] %s %s\n' "$(date '+%F %T')" "$*"
}

mkdir -p "$INSTALL_DIR/logs" "$LAUNCH_AGENT_DIR"

for file in "${FILES[@]}"; do
  [[ -f "$SCRIPT_DIR/$file" ]] || { log "missing $SCRIPT_DIR/$file"; exit 1; }
done

rsync -az "${FILES[@]/#/$SCRIPT_DIR/}" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR"/*.sh "$INSTALL_DIR"/*.py

bash -n \
  "$INSTALL_DIR/watch_promote_qwen27_ca_201_202.sh" \
  "$INSTALL_DIR/watch_110_119_reconnect_and_launch_attacks.sh" \
  "$INSTALL_DIR/run_qwen27_selected_formal1k_eval.sh"

plutil -lint "${PLISTS[@]/#/$SCRIPT_DIR/launchd/}"

for plist in "${PLISTS[@]}"; do
  cp "$SCRIPT_DIR/launchd/$plist" "$LAUNCH_AGENT_DIR/$plist"
  launchctl bootout "gui/$UID_NUM" "$LAUNCH_AGENT_DIR/$plist" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$UID_NUM" "$LAUNCH_AGENT_DIR/$plist"
done

launchctl kickstart -k "gui/$UID_NUM/local.sac.watch-promote-qwen27-ca-201-202"
launchctl kickstart -k "gui/$UID_NUM/local.sac.watch-110-119-reconnect"

log "installed bundle=$INSTALL_DIR"
log "promote log=$INSTALL_DIR/logs/watch_promote_qwen27_ca_201_202.launchd.log"
log "reconnect log=$INSTALL_DIR/logs/watch_110_119_reconnect.launchd.log"
