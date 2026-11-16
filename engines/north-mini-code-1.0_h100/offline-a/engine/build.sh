#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
[ -x "$NVCC" ] || NVCC=nvcc
# engine/TRACK says which engine is built: A = the checkpoint's BF16 bits (apples to apples with the general
# engines), B = the lossless 12-bit weight format on top (echo A > engine/TRACK to get the track-A engine back).
TRACK="${MK_TRACK:-$(tr -d '[:space:]' < TRACK)}"   # (MK_TRACK: build the other track without touching the file, for tests)
DEFS="${MK_DEFS:-}"
[ "$TRACK" = "B" ] && DEFS="$DEFS -DW12"
# always rebuilt (3 s): a library left over from another track or another set of flags must never be picked up
"$NVCC" -O3 -std=c++17 -arch=sm_90 -Xptxas -v $DEFS -Xcompiler -fPIC -shared -o libmk.so.tmp "${MK_SRC:-mk.cu}"
mv libmk.so.tmp libmk.so
