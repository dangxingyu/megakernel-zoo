#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
# Two persistent-kernel engines behind one contract (engine/api.py dispatches by batch size; structure from run
# s17-mini-b200-ma s0002):
#   build/libmk1.so  <- a2/mk.cu  (s15-mini-b200-a2 s0025 + this run's changes) serves batch 1
#   build/libmk8.so  <- a1/mk.cu  (s15-mini-b200-a1 s0025) serves batch >= 2
# engine/TRACK (or $MK_TRACK): "A ..." = the checkpoint's formats and the specification's data flow, "B ..." adds
# track-B items. Tokens: W12 (a2 storage); NAME[=V] = -D for the a2 kernel; NAME[=V]:8 = -D for the a1 kernel.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
L1=${MK_LIB1NAME:-libmk1.so}; L8=${MK_LIB8NAME:-libmk8.so}
TRACKV=${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}
D1="${MK_DEFS:-}"; D8="${MK_DEFS8:-}"
for tok in $TRACKV; do
  case "$tok" in
    A|B|NOTUNE) ;;
    W12) D1="$D1 -DW12";;
    *:8) n=${tok%:8}; case "$n" in *=*) D8="$D8 -D$n";; *) D8="$D8 -D$n=1";; esac;;
    *=*) D1="$D1 -D$tok";;
    *) D1="$D1 -D$tok=1";;
  esac
done
if [ -n "${MK_PRINT_DEFS:-}" ]; then echo "a2: $D1 | a1: $D8"; exit 0; fi
# --use_fast_math in both (approximate exp / tanh / division / sqrt in the fp32 vector code; measured in a1 / a2)
NV="nvcc -O3 --use_fast_math -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared -Xptxas -v"
rm -f build/$L1; [ -n "${MK_SKIP8:-}" ] || rm -f build/$L8
if [ -z "${MK_SKIP8:-}" ]; then
  ( cd a1 && $NV $D8 ${MK_CFLAGS8:-} -o ../build/$L8 mk.cu 2> ../build/ptxas8.log ) &
  P8=$!
fi
( cd a2 && $NV $D1 ${MK_CFLAGS:-} -o ../build/$L1 mk.cu 2> ../build/ptxas1.log ) || { cat build/ptxas1.log; exit 1; }
if [ -z "${MK_SKIP8:-}" ]; then wait $P8 || { cat build/ptxas8.log; exit 1; }; fi
grep -E "registers|spill" build/ptxas1.log | head -2 | sed 's/^/a2: /' || true
grep -A2 "Compiling entry function '_Z.*mk_kernel_t" build/ptxas8.log 2>/dev/null | grep -E "stack|registers" | sed 's/^/a1: /' || true
echo built
