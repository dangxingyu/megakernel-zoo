#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
# Two kernels, both track A: build/libmk1.so (a2/mk.cu, from s15-mini-b200-a2 s0025) serves batch 1,
# build/libmk8.so (a1/mk.cu, from s15-mini-b200-a1 s0024) serves batch >= 2.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
rm -f build/libmk1.so build/libmk8.so
TRACKV=${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}
# batch-1 kernel: KB1 (batch size a compile-time 1; idea of s15-mini-b200-b2 s0020 "K1") compiled with the mkprof
# mark sites present (MKPROF; they do nothing unless the engine is given a profile buffer, MK_PROF=1). Bitwise equal
# to the plain build; measured B=1 2724 vs 2744 us (3 alternations). KB1 without the mark sites: 3120 us (a
# code-generation cliff), the mark sites without KB1: 2799 us. MK_K1PLAIN=1 builds the a2 s0025 kernel as it was.
F1="-DKB1=1 -DMKPROF"
if [ "${MK_K1PLAIN:-0}" = "1" ]; then F1="-DKB1=0"; fi
if echo "$TRACKV" | grep -qw W12; then F1="$F1 -DW12"; fi
# --use_fast_math: approximate exp / tanh / division / sqrt in the fp32 vector code (both kernels; measured in their runs)
COMMON="-O3 --use_fast_math -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -Xptxas -v"
( nvcc $COMMON $F1 ${MK_CFLAGS1:-} -o build/libmk1.so a2/mk.cu 2> build/ptxas1.log || { cat build/ptxas1.log; exit 1; } ) &
P1=$!
( nvcc $COMMON ${MK_CFLAGS8:-} -o build/libmk8.so a1/mk.cu 2> build/ptxas8.log || { cat build/ptxas8.log; exit 1; } ) &
P8=$!
wait $P1; wait $P8
grep -E "mk_step|registers|spill" build/ptxas1.log | head -4 || true
grep -A2 "Compiling entry function '_Z.*mk_kernel_t" build/ptxas8.log | grep -E "stack|registers" || true
echo built
