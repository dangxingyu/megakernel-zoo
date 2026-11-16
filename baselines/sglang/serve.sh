#!/usr/bin/env bash
# bash baselines/sglang/serve.sh MODEL_DIR [extra flags]: SGLang's OpenAI server on $MK_PORT, model name "m".
# CUDA graphs are on by default in SGLang. Flags added here must be recorded with the run (they are echoed).
set -euo pipefail
model="$1"; shift
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
flags=(--model-path "$model" --served-model-name m --host 127.0.0.1 --port "$MK_PORT" --dtype bfloat16 "$@")
echo "MK_FLAGS sglang.launch_server ${flags[*]}"
exec "$MK_ROOT/venvs/sglang/bin/python" -m sglang.launch_server "${flags[@]}"
