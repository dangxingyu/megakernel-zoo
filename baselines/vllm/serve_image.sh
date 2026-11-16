#!/usr/bin/env bash
# bash baselines/vllm/serve_image.sh MODEL_DIR [extra flags]: vLLM from its vendor image (unpacked by
# scripts/ink/img.sh; for models whose support ships only as an image, e.g. Inkling-2-Small), OpenAI server on
# $MK_PORT, model name "m". Flags are the caller's (the model's recipe); they are echoed.
set -euo pipefail
source "$MK_REPO/scripts/ink/img.sh"
model="$1"; shift
echo "MK_FLAGS vllm(image $(cat "$MK_ROOT/img/vllm.ref" 2>/dev/null)) serve $model --served-model-name m $*"
img_exec vllm "python3 -m vllm.entrypoints.cli.main serve $model --served-model-name m --host 127.0.0.1 --port $MK_PORT $*"
