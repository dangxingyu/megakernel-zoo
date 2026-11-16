#!/usr/bin/env bash
# SGLang, pinned.
set -euo pipefail
SGLANG_VERSION="${SGLANG_VERSION:-0.5.23}"
# SGLang imports deep_gemm, which refuses to load without a CUDA toolkit (CUDA_HOME).
source "$MK_REPO/scripts/cuda_toolkit.sh"
uv venv -q "$MK_ROOT/venvs/sglang" --python python3.12
uv pip install -q --python "$MK_ROOT/venvs/sglang/bin/python" "sglang==$SGLANG_VERSION"
"$MK_ROOT/venvs/sglang/bin/python" - <<'PY'
import platform, torch, sglang
# aarch64 has resolved to a CPU-only torch wheel before: no CUDA build means the install is wrong, not slow.
assert torch.version.cuda and torch.cuda.is_available(), f"torch {torch.__version__} has no CUDA on {platform.machine()}"
print(f"MK_ENV sglang={sglang.__version__} torch={torch.__version__} cuda={torch.version.cuda} arch={platform.machine()} gpu={torch.cuda.get_device_name(0)}")
PY
