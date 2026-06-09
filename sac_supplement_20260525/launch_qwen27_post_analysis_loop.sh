#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/run_qwen27_post_analysis_loop.sh"
HELPER_SRC="$SCRIPT_DIR/codex_sac_supp.py"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_ITERATIONS="${MAX_ITERATIONS:-48}"
RESTART="${RESTART:-0}"

if [[ ! -f "$SRC" ]]; then
  echo "missing source script: $SRC" >&2
  exit 1
fi
if [[ ! -f "$HELPER_SRC" ]]; then
  echo "missing helper script: $HELPER_SRC" >&2
  exit 1
fi

if [[ -n "${HOSTS:-}" ]]; then
  read -r -a HOST_LIST <<< "$HOSTS"
else
  HOST_LIST=(
    192.168.7.201
    192.168.7.202
    192.168.6.110
    192.168.6.111
    192.168.6.112
    192.168.6.113
    192.168.6.114
    192.168.6.115
    192.168.6.116
    192.168.6.117
    192.168.6.118
    192.168.6.119
  )
fi

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=2)
ROOT_PROBE='for r in /home/xlz/SAC/single /mnt/disk/xlz/SAC/single /home/xlz/mnt/disk/xlz/SAC/single; do if [ -d "$r" ]; then printf "%s\n" "$r"; exit 0; fi; done; exit 1'

for host in "${HOST_LIST[@]}"; do
  echo "== $host =="
  if ! root="$(ssh "${SSH_OPTS[@]}" "$host" "$ROOT_PROBE" 2>/dev/null)"; then
    echo "skip: cannot connect or no SAC root found"
    continue
  fi

  tag="${host//./_}"
  remote_dir="$root/sac_supplement_20260525"
  scripts_dir="$root/scripts"
  log_dir="$root/outputs/supplement_20260525/qwen35_27b/logs"
  remote_script="$remote_dir/run_qwen27_post_analysis_loop.sh"
  remote_helper="$remote_dir/codex_sac_supp.py"
  remote_scripts_helper="$scripts_dir/codex_sac_supp.py"
  log_path="$log_dir/post_analysis_${tag}_$(date '+%Y%m%d_%H%M%S').log"
  pid_path="$log_dir/post_analysis_${tag}.pid"

  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$remote_dir' '$scripts_dir' '$log_dir'"
  scp -q "${SSH_OPTS[@]}" "$SRC" "$host:$remote_script"
  scp -q "${SSH_OPTS[@]}" "$HELPER_SRC" "$host:$remote_helper"
  scp -q "${SSH_OPTS[@]}" "$HELPER_SRC" "$host:$remote_scripts_helper"
  ssh "${SSH_OPTS[@]}" "$host" "chmod +x '$remote_script' '$remote_helper' '$remote_scripts_helper'"

  ssh "${SSH_OPTS[@]}" "$host" \
    "if [ '$RESTART' = '1' ] && [ -s '$pid_path' ] && kill -0 \"\$(cat '$pid_path')\" 2>/dev/null; then old_pid=\"\$(cat '$pid_path')\"; kill \"\$old_pid\" 2>/dev/null || true; sleep 1; echo stopped pid=\"\$old_pid\"; fi; if [ -s '$pid_path' ] && kill -0 \"\$(cat '$pid_path')\" 2>/dev/null; then echo already-running pid=\"\$(cat '$pid_path')\"; else nohup env ROOT='$root' HOST_TAG='$tag' INTERVAL_SECONDS='$INTERVAL_SECONDS' MAX_ITERATIONS='$MAX_ITERATIONS' bash '$remote_script' > '$log_path' 2>&1 < /dev/null & echo \$! > '$pid_path'; echo launched pid=\"\$(cat '$pid_path')\" log='$log_path'; fi"
done
