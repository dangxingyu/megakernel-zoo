#!/usr/bin/env bash
# bash baselines/vllm/serve.sh MODEL_DIR [extra flags]: vLLM's OpenAI server on $MK_PORT, model name "m".
# BF16, the model's full context. Flags added here must be recorded with the run (they are echoed).
set -euo pipefail
model="$1"; shift
flags=(--served-model-name m --host 127.0.0.1 --port "$MK_PORT" --dtype bfloat16 --gpu-memory-utilization 0.90 "$@")
echo "MK_FLAGS vllm serve $model ${flags[*]}"
exec "$MK_ROOT/venvs/vllm/bin/vllm" serve "$model" "${flags[@]}"
