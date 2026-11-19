#!/usr/bin/env bash
# Sourced by every script. Everything a machine writes is under $MK_ROOT on the node's local disk and dies
# with the machine; what must survive goes to the object store ($MK_S3) through scripts/store.py.
export MK_ROOT="${MK_ROOT:-/tmp/mk}" MK_T0="${MK_T0:-$(date +%s)}"
export MK_S3="${MK_S3:-}"   # empty: no object store; weights come from Hugging Face and nothing is uploaded
export HF_HOME="$MK_ROOT/hf" XDG_CACHE_HOME="$MK_ROOT/cache" UV_CACHE_DIR="$MK_ROOT/uv-cache"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MK_TOOLS_PY="$MK_ROOT/tools/bin/python"
mkdir -p "$MK_ROOT"/{logs,models,venvs,run} "$HF_HOME"
# Jobs share the node's network namespace: a fixed port collides with other jobs on the node, and a client can
# end up talking to somebody else's server. Every job takes a free port of its own.
export MK_PORT="${MK_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}"

stage() { echo "MK_STAGE {\"stage\": \"$1\", \"status\": \"$2\", \"seconds\": $(( $(date +%s) - ${3:-$MK_T0} ))}"; }
store() { [ -n "$MK_S3" ] || return 1; "$MK_TOOLS_PY" "$MK_REPO/scripts/store.py" "$@"; }

# Small venv for the object store and the workload client; engines get their own venvs.
mk_tools() {
  local t=$(date +%s)
  [ -x "$MK_TOOLS_PY" ] || { uv venv -q "$MK_ROOT/tools" --python python3.12 \
    && uv pip install -q --python "$MK_TOOLS_PY" boto3 aiohttp tokenizers numpy huggingface_hub; }
  local rc=$?; stage tools "$([ $rc = 0 ] && echo ok || echo failed)" $t; return $rc
}

env_report() {
  echo "MK_ENV job=${MK_JOB:-?} cluster=${MK_CLUSTER:-?} bundle=$(cat "$MK_REPO/CW_BUNDLE" 2>/dev/null) host=$(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader | sed 's/^/MK_ENV gpu /'
  echo "MK_ENV port=$MK_PORT"
  echo "MK_ENV arch=$(uname -m) cpu=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1 | tr ' ' '_') os=$(. /etc/os-release; echo "$VERSION_CODENAME") kernel=$(uname -r)"
  echo "MK_ENV cpus=$(nproc) cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | tr ' ' '/') loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
}

# mk_hw: the card tag (h100 | b200) used in file names and for the card's parameters (eval/modelinfo.py CARDS):
# $MK_HW when set, else from the name the driver reports for the machine's card.
mk_hw() {
  if [ -n "${MK_HW:-}" ]; then echo "$MK_HW" | tr 'A-Z' 'a-z'; return; fi
  case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr 'A-Z' 'a-z')" in
    *h100*) echo h100 ;; *b200*) echo b200 ;; *) echo h100 ;;
  esac
}

# mk_model REPO_ID: print the local directory of a Hugging Face snapshot. Taken from our store, else from the
# quant-zoo store (read only), else downloaded once from Hugging Face and kept in our store.
mk_model() {
  local repo="$1" name="models--${1//\//--}" t=$(date +%s) src="" root
  local snaps="$MK_ROOT/models/$name/snapshots"
  if ! ls "$snaps"/*/config.json >/dev/null 2>&1; then
    # $MK_LEGACY_S3: our prefix in the bucket we used until 2026-11-13 (read only; MK_MODEL_KEEP=1 copies what is
    # read from it into $MK_S3, which is how the checkpoints move to the new bucket).
    for root in "$MK_S3"; do
      if store has "$root/hf/$name/snapshots" >/dev/null 2>&1; then
        store get "$root/hf/$name/snapshots" "$snaps" > "$MK_ROOT/logs/model-get.log" 2>&1 && src="$root" && break
      fi
    done
    if [ -n "$src" ] && [ "$src" != "$MK_S3" ] && [ "${MK_MODEL_KEEP:-0}" = 1 ] && [[ "$src" != */quant-zoo ]]; then
      store put "$snaps" "$MK_S3/hf/$name/snapshots" "$MK_S3" > "$MK_ROOT/logs/model-put.log" 2>&1 || echo "MK_WARN model not kept in store" >&2
    fi
    if [ -z "$src" ]; then
      "$MK_TOOLS_PY" - "$repo" "$snaps" > "$MK_ROOT/logs/model-hf.log" 2>&1 <<'PY' || { stage "model $repo" failed $t >&2; return 1; }
import sys
from huggingface_hub import snapshot_download, HfApi
repo, snaps = sys.argv[1], sys.argv[2]
sha = HfApi().model_info(repo).sha
# Some repositories carry the weights a second time in another layout (Mistral's consolidated.safetensors, old
# pytorch_model*.bin next to safetensors): the safetensors shards of the index are the checkpoint, the rest is skipped.
names = [s.rfilename for s in HfApi().model_info(repo, revision=sha).siblings]
skip = [n for n in names if n.endswith((".bin", ".pth", ".gguf", ".onnx", ".msgpack", ".h5", ".ot"))] if any(n.endswith(".safetensors") for n in names) else []
skip += ["consolidated.safetensors"] if any(n.startswith("model-0") or n == "model.safetensors" for n in names) else []
# gpt-oss also ships its weights again under original/ (another layout) and metal/: the top-level index is the checkpoint.
skip += [n for n in names if "/" in n and n.endswith((".safetensors", ".bin"))] if "model.safetensors.index.json" in names else []
snapshot_download(repo, revision=sha, local_dir=f"{snaps}/{sha}", max_workers=16, ignore_patterns=skip or None)
PY
      rm -rf "$snaps"/*/.cache
      store put "$snaps" "$MK_S3/hf/$name/snapshots" "$MK_S3" > "$MK_ROOT/logs/model-put.log" 2>&1 || echo "MK_WARN model not kept in store" >&2
      src="huggingface"
    fi
    stage "model $repo from ${src}" ok $t >&2
  fi
  dirname "$(ls "$snaps"/*/config.json | head -1)"
}

# wait_http URL SECONDS PID: wait until URL answers 200, or the server process dies, or time runs out.
wait_http() {
  local t=$(date +%s)
  while [ $(( $(date +%s) - t )) -lt "$2" ]; do
    kill -0 "$3" 2>/dev/null || return 2
    if [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$1")" = 200 ]; then
      sleep 2; kill -0 "$3" 2>/dev/null && return 0 || return 2  # the answer must come from our own, living server
    fi
    sleep 3
  done
  return 1
}
