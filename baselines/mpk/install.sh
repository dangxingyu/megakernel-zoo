#!/usr/bin/env bash
# MPK (Mirage Persistent Kernel), branch mpk, pinned; built from source as its README says.
set -euo pipefail
# Pin: 97de2592 (2026-09-24), the last commit before the Hopper path broke upstream. From fb181082 (2026-09-28) on,
# src/kernel/task_register.cc passes 15 arguments to a 14-marker template for multitoken_paged_attention_hopper_impl
# and the server aborts while generating the megakernel on an H100 (upstream CI runs on B200 only).
# MPK_PATCH=hopper-marker instead builds any later commit with that one missing "$" added and nothing else changed.
# On a B200 (GB200 pods) the Hopper path is not generated: use the newest pin unpatched, MPK_COMMIT=d90383d0...
COMMIT="${MPK_COMMIT:-97de25929c09}"
src="$MK_ROOT/src/mirage${MPK_TAG:+-$MPK_TAG}"
source "$MK_REPO/scripts/cuda_toolkit.sh"
if [ ! -d "$src/.git" ]; then
  git clone -q --branch mpk https://github.com/mirage-project/mirage.git "$src"
  git -C "$src" checkout -q "$COMMIT"
  git -C "$src" submodule update -q --init --recursive --depth 1
fi
if [ "${MPK_PATCH:-}" = hopper-marker ]; then
  python3 - "$src/src/kernel/task_register.cc" <<'PY'
import sys
path = sys.argv[1]; s = open(path).read()
old = 'code.e("kernel::multitoken_paged_attention_hopper_impl<bfloat16, $, $, $, $, "\n         "$, $, $, $, $, "\n         "$, $, $, $, $>(",'
assert s.count(old) == 1, s.count(old)
open(path, "w").write(s.replace(old, old.replace('"$, $, $, $, $>(",', '"$, $, $, $, $, $>(",')))
print("MK_ENV mpk patch=hopper-marker applied")
PY
fi
py="$MK_ROOT/venvs/mpk${MPK_TAG:+-$MPK_TAG}/bin/python"
uv venv -q "$MK_ROOT/venvs/mpk${MPK_TAG:+-$MPK_TAG}" --python python3.12 --seed
t=$(date +%s)
(cd "$src" && CMAKE_BUILD_PARALLEL_LEVEL=48 MAKEFLAGS=-j48 "$py" -m pip install -e . -v) > "$MK_ROOT/logs/mpk-pip.log" 2>&1 \
  || { grep -nE "error|Error|FAILED" "$MK_ROOT/logs/mpk-pip.log" | head -20; tail -n 30 "$MK_ROOT/logs/mpk-pip.log"; exit 1; }
echo "MK_ENV mpk build seconds=$(( $(date +%s) - t ))"
MIRAGE_HOME="$src" "$py" - <<'PY'
import platform, torch, mirage
assert torch.version.cuda and torch.cuda.is_available(), f"torch {torch.__version__} has no CUDA on {platform.machine()}"
print(f"MK_ENV mirage={getattr(mirage, '__version__', '?')} torch={torch.__version__} cuda={torch.version.cuda} arch={platform.machine()} gpu={torch.cuda.get_device_name(0)}")
PY
echo "MK_ENV mpk commit=$(git -C "$src" rev-parse HEAD)"
