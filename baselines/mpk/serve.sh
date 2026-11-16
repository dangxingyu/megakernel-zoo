#!/usr/bin/env bash
# bash baselines/mpk/serve.sh MODEL_DIR [extra flags]: MPK's online server on $MK_PORT (Qwen3-8B).
# The server has no per-request max_tokens: a request runs to EOS or to --max-seq-length. Extra flags are echoed.
set -euo pipefail
model="$1"; shift
src="$MK_ROOT/src/mirage${MPK_TAG:+-$MPK_TAG}"
export MIRAGE_HOME="$src" PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
flags=(--host 127.0.0.1 --port "$MK_PORT" --model Qwen/Qwen3-8B --model-path "$model" --output-dir "$MK_ROOT/run/mpk-out${MPK_TAG:+-$MPK_TAG}" "$@")
echo "MK_FLAGS mirage.engine.launch_server ${flags[*]}"
cd "$src" && exec "$MK_ROOT/venvs/mpk${MPK_TAG:+-$MPK_TAG}/bin/python" -m mirage.engine.launch_server "${flags[@]}"
