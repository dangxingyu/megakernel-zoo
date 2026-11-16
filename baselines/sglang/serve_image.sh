#!/usr/bin/env bash
# bash baselines/sglang/serve_image.sh MODEL_DIR [extra flags]: SGLang from its vendor image (scripts/ink/img.sh),
# OpenAI server on $MK_PORT, model name "m". $SGL_IMAGE_ENV: extra environment (the recipe's). Flags are echoed.
set -euo pipefail
source "$MK_REPO/scripts/ink/img.sh"
model="$1"; shift
echo "MK_FLAGS sglang(image $(cat "$MK_ROOT/img/sglang.ref" 2>/dev/null)) env ${SGL_IMAGE_ENV:-} launch_server --model-path $model --served-model-name m $*"
IMG_EXTRA_ENV="${SGL_IMAGE_ENV:-}" img_exec sglang "python3 -m sglang.launch_server --model-path $model --served-model-name m --host 127.0.0.1 --port $MK_PORT $*"
