#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${MK_BUILD_DIR:-build}
mkdir -p "$OUT"
FLAGS="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xptxas -v --shared -Xcompiler -fPIC"
# fast math (approximate exp / tanh / rsqrt / division, FTZ) in the fp32 vector code: the critical path's softmax,
# sigmoid, soft-caps and norms are instruction-latency bound (B=1 -121 us, B=8 -239). Formats are unchanged.
# MK_PRECISE=1 builds with IEEE math.
[ "${MK_PRECISE:-0}" = "1" ] || FLAGS="$FLAGS --use_fast_math"
rm -f "$OUT/libmk.so"
nvcc $FLAGS ${MK_DEFS:-} -o "$OUT/libmk.so" mk.cu 2> "$OUT/ptxas.log" || { cat "$OUT/ptxas.log"; exit 1; }
grep -A2 "Compiling entry function '_Z.*mk_kernel_t" "$OUT/ptxas.log" | grep -E "stack|registers" || true
# study builds (never used by the evaluator): MK_VARIANTS="prof:-DMKPROF skel:-DSKEL,-DMKPROF"
for v in ${MK_VARIANTS:-}; do
  name=${v%%:*}; defs=${v#*:}; defs=${defs//,/ }
  rm -f "$OUT/libmk_$name.so"
  nvcc $FLAGS $defs -o "$OUT/libmk_$name.so" mk.cu 2> "$OUT/ptxas_$name.log" || { cat "$OUT/ptxas_$name.log"; exit 1; }
done
echo built
