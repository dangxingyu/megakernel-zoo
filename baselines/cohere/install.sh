#!/usr/bin/env bash
# Cohere megakernel (North Mini Code only), pinned, built as its BUILD_AND_RUN.md says.
set -euo pipefail
COMMIT="${COHERE_COMMIT:-67d0b9ca22ea}"
src="$MK_ROOT/src/cohere-megakernel"
source "$MK_REPO/scripts/cuda_toolkit.sh"
if [ ! -d "$src/.git" ]; then
  git clone -q https://github.com/cohere-ai/cohere-megakernel.git "$src"
  git -C "$src" checkout -q "$COMMIT"
  git -C "$src" submodule update -q --init --recursive
fi
py="$MK_ROOT/venvs/cohere/bin/python"
uv venv -q "$MK_ROOT/venvs/cohere" --python python3.12
uv pip install -q --python "$py" -r "$src/requirements.txt" --torch-backend=cu130
uv pip install -q --python "$py" flash-attn-3 --index-url https://download.pytorch.org/whl/cu130
cmake -S "$src" -B "$src/build" -G Ninja -DPython_EXECUTABLE="$py" -DCMAKE_BUILD_TYPE=Release > "$MK_ROOT/logs/cohere-cmake.log" 2>&1
cmake --build "$src/build" > "$MK_ROOT/logs/cohere-build.log" 2>&1 || { tail -n 30 "$MK_ROOT/logs/cohere-build.log"; exit 1; }
ls -la "$src/build"/libmk_release.so "$src/build"/mk_ext.abi3.so
"$py" - <<'PY'
import torch
print(f"MK_ENV cohere-megakernel torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
PY
echo "MK_ENV cohere commit=$(git -C "$src" rev-parse HEAD)"
