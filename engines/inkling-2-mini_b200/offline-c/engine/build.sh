#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
rm -f build/${MK_LIBNAME:-libmk.so} build/libdk.so
# the draft head (track C): its own library, compiled without fast math (RoPE cos/sin of large arguments)
nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared ${MK_DKFLAGS:-} -o build/libdk.so draftk.cu 2> build/ptxas_dk.log &
DKPID=$!
TRACKV=${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}
FLAGS=""
if echo "$TRACKV" | grep -qw W12; then FLAGS="-DW12"; fi
# --use_fast_math: approximate exp / tanh / division / sqrt in the vector code (softmax, sigmoid, activation, norms);
# flush-to-zero is harmless here: no expert scale or operand of the f16 / f32 paths is denormal (tests/mxscales.py)
nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -Xptxas -v $FLAGS ${MK_CFLAGS:-} \
  -o build/${MK_LIBNAME:-libmk.so} mk.cu 2> build/ptxas.log || { cat build/ptxas.log; exit 1; }
grep -E "mk_step|registers|spill" build/ptxas.log | head -8 || true
wait $DKPID || { cat build/ptxas_dk.log; exit 1; }
