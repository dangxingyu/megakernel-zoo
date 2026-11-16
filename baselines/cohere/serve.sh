#!/usr/bin/env bash
# bash baselines/cohere/serve.sh CKPT_DIR [extra flags]: Cohere's OpenAI-compatible server on $MK_PORT.
# Flags follow BUILD_AND_RUN.md's server example (batch 8, its largest). Extra flags are echoed.
set -euo pipefail
ckpt="$1"; shift
src="$MK_ROOT/src/cohere-megakernel"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
flags=(--host 127.0.0.1 --port "$MK_PORT" --lib "$src/build/libmk_release.so" --ckpt "$ckpt" --device cuda:0 --bs 8 "$@")
echo "MK_FLAGS cohere server.py ${flags[*]}"
cd "$src" && exec "$MK_ROOT/venvs/cohere/bin/python" src/serving/server.py "${flags[@]}"
