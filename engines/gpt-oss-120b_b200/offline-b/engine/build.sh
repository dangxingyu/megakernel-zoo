#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
# engine/TRACK: "A" = the checkpoint's formats and data flow only; "B" = + the certified int8 head sketch on the
# greedy paths (HSK: same tokens, the exact head recomputed on candidate rows)
TRACK=$(cat TRACK 2>/dev/null || echo A)
TFLAG=""; [ "$TRACK" = "B" ] && TFLAG="-DHSK=1"
FLAGS="-O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -shared -Xcompiler -fPIC $TFLAG ${MK_EXTRA:-}"
HASH=$( (cat mk.cu mkprof.cuh build.sh; echo "$FLAGS") | sha1sum | cut -c1-16)
if [ ! -f build/libmk.so ] || [ "$(cat build/hash 2>/dev/null)" != "$HASH" ]; then
  rm -f build/libmk.so
  nvcc $FLAGS --resource-usage -o build/libmk.so mk.cu 2>&1 | grep -E "error|warning|mk_kernel|registers|spill" || true
  test -f build/libmk.so
  echo "$HASH" > build/hash
fi
# profiling / study builds (not used by the evaluator): MK_VARIANTS="prof:-DMKPROF skel:-DSKEL ..."
for v in ${MK_VARIANTS:-}; do
  name=${v%%:*}; defs=${v#*:}; defs=${defs//,/ }
  nvcc $FLAGS $defs -o build/libmk_$name.so mk.cu 2>&1 | grep -E "error" || true
done
echo "built"
