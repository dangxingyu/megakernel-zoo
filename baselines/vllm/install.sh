#!/usr/bin/env bash
# vLLM, pinned. Official install path: uv with the torch backend chosen for the driver.
set -euo pipefail
VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
uv venv -q "$MK_ROOT/venvs/vllm" --python python3.12
uv pip install -q --python "$MK_ROOT/venvs/vllm/bin/python" "vllm==$VLLM_VERSION" --torch-backend=auto
"$MK_ROOT/venvs/vllm/bin/python" - <<'PY'
import platform, torch, vllm
# aarch64 has resolved to a CPU-only torch wheel before: no CUDA build means the install is wrong, not slow.
assert torch.version.cuda and torch.cuda.is_available(), f"torch {torch.__version__} has no CUDA on {platform.machine()}"
print(f"MK_ENV vllm={vllm.__version__} torch={torch.__version__} cuda={torch.version.cuda} arch={platform.machine()} gpu={torch.cuda.get_device_name(0)}")
PY
