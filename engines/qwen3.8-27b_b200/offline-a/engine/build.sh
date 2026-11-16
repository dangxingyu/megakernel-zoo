#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
# NO_STAMPS: the debug timers (per-op, per-block, per-warp stamps for tools/op_times_p.py, tail_p.py, ...) compiled
# out: -0.4 % per B=1 step. The tools need them: build with MK_CFLAGS=-DBAR_SLEEP_NS=200.
# BAR_SLEEP_NS: grid-barrier spin back-off; 25 ns since s5 hour 5.4 (W12 engine: 200 -> 25 ns -0.15 %)
FLAGS="${MK_CFLAGS:--DBAR_SLEEP_NS=25 -DNO_STAMPS}"
# engine/TRACK "A W12": every streamed matrix in exact 12-bit storage; the BF16 stream loops are not compiled in, and
# GEMM inputs at one column use the compact layout (X_COMPACT; MK_W12_EXTRA="" builds without it)
if grep -qw W12 TRACK && [ -z "${MK_NO_W12_ONLY:-}" ]; then FLAGS="$FLAGS -DW12_ONLY ${MK_W12_EXTRA--DX_COMPACT -DW12_MIXED}"; fi
FLAGS="$FLAGS ${MK_W12_PRMT--DW12_PRMT}"   # W12 byte layout (api.py MK_W12_FMT=prmt); MK_W12_PRMT="" + MK_W12_FMT=pack: s0010
stamp=$( (cat mk.cu build.sh; echo "$FLAGS") | sha256sum | cut -d' ' -f1)
if [ ! -f build/libmk.so ] || [ "$(cat build/stamp 2>/dev/null)" != "$stamp" ]; then
  nvcc -O3 -std=c++17 -arch=sm_100 $FLAGS -shared -Xcompiler -fPIC -Xptxas -v -o build/libmk.so.tmp mk.cu 2> build/ptxas.log || { cat build/ptxas.log; exit 1; }
  mv build/libmk.so.tmp build/libmk.so
  echo "$stamp" > build/stamp; echo "$FLAGS" > build/flags
  echo "built libmk.so"
fi
