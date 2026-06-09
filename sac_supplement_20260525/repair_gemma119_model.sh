#!/usr/bin/env bash
set -uo pipefail

SRC="${SRC:-192.168.6.115}"
DST="${DST:-192.168.6.119}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
MODEL_DIR="/home/xlz/models/gemma-3-4b-it"

log() {
  printf '[gemma119-repair] %s %s\n' "$(date '+%F %T')" "$*"
}

copy_checked() {
  local name="$1"
  local expected="$2"
  local path="$MODEL_DIR/$name"
  log "copy $name expected=$expected"
  ssh "${SSH_OPTS[@]}" "$DST" "mkdir -p '$MODEL_DIR' && rm -f '$path.tmp'" || return 1
  if ssh "${SSH_OPTS[@]}" "$SRC" "cat '$path'" | ssh "${SSH_OPTS[@]}" "$DST" "cat > '$path.tmp'; size=\$(stat -c%s '$path.tmp' 2>/dev/null || echo 0); if [ \"\$size\" = '$expected' ]; then mv '$path.tmp' '$path'; echo ok:$name:\$size; else echo bad:$name:\$size expected:$expected >&2; exit 1; fi"; then
    log "done $name"
    return 0
  fi
  log "failed $name"
  return 1
}

copy_small_files() {
  log "copy small files"
  ssh "${SSH_OPTS[@]}" "$SRC" "cd '$MODEL_DIR' && tar -cf - .gitattributes README.md added_tokens.json chat_template.json config.json generation_config.json model.safetensors.index.json preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json" |
    ssh "${SSH_OPTS[@]}" "$DST" "mkdir -p '$MODEL_DIR' && cd '$MODEL_DIR' && tar -xf -"
  log "done small files"
}

log "start"
copy_checked model-00001-of-00002.safetensors 4961251752
copy_checked model-00002-of-00002.safetensors 3639026128
copy_small_files
ssh "${SSH_OPTS[@]}" "$DST" "du -sh '$MODEL_DIR'; find '$MODEL_DIR' -maxdepth 1 -name 'model-*.safetensors' -printf '%f %s\n' | sort" || true
log "complete"
