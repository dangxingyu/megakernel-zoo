#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
FLAGS="-O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a -shared -Xcompiler -fPIC -DNSS ${MK_EXTRA:-}"
HASH=$( (cat mk.cu mkprof.cuh build.sh; echo "$FLAGS") | sha1sum | cut -c1-16)
if [ ! -f build/libmk.so ] || [ "$(cat build/hash 2>/dev/null)" != "$HASH" ]; then
  rm -f build/libmk.so
  nvcc $FLAGS --resource-usage -o build/libmk.so mk.cu 2>&1 | grep -E "error|warning|mk_kernel|registers|spill" || true
  test -f build/libmk.so
  echo "$HASH" > build/hash
fi
# track B build (engine/TRACK = B): router weights folded with ln2, router partials in phase C
if [ ! -f build/libmk_b.so ] || [ "$(cat build/hash_b 2>/dev/null)" != "$HASH" ]; then
  rm -f build/libmk_b.so
  nvcc $FLAGS -DRFOLD -DNSS -DDSPLIT=2 -DP2TMA=1 -DP2CH=6 --resource-usage -o build/libmk_b.so mk.cu 2>&1 | grep -E "error|spill" | head -2 || true
  test -f build/libmk_b.so
  echo "$HASH" > build/hash_b
fi
# batch-size-1 specialisations (B a compile-time constant): libmk_1.so (track A), libmk_b1.so (track B)
B1EXTRA="-DBFIX=1 -DGURR=0 -DSKQ=10 -DDBPF=0 -DDORD=1 ${MK_B1EXTRA:-}"
if [ ! -f build/libmk_1.so ] || [ ! -f build/libmk_b1.so ] || [ "$(cat build/hash_1 2>/dev/null)" != "$HASH$B1EXTRA" ]; then
  rm -f build/libmk_1.so build/libmk_b1.so
  nvcc $FLAGS -DTOPR=1 $B1EXTRA -o build/libmk_1.so mk.cu 2>&1 | grep -E "error" | head -2 || true
  nvcc $FLAGS -DRFOLD -DNSS -DDSPLIT=2 -DP2TMA=1 -DSK5=1 -DDPK=1 -DSKPF0=1 -DSKDB=1 -DSKQP=3 -DTOPR=1 -DRBH=1 -DGBPF=1 $B1EXTRA -o build/libmk_b1.so mk.cu 2>&1 | grep -E "error" | head -2 || true
  test -f build/libmk_1.so && test -f build/libmk_b1.so
  echo "$HASH$B1EXTRA" > build/hash_1
fi
# profiling / study builds (not used by the evaluator): MK_VARIANTS="prof:-DMKPROF skel:-DSKEL ..."
for v in ${MK_VARIANTS:-}; do
  name=${v%%:*}; defs=${v#*:}; defs=${defs//,/ }
  nvcc $FLAGS $defs -o build/libmk_$name.so mk.cu 2>&1 | grep -E "error" || true
done
echo "built"
