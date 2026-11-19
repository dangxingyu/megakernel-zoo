#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
TRACK=$(echo "${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}" | tr -d '[:space:]' | tr a-z A-Z)
# engine/TRACK (or MK_TRACK): A = BF16 weights as stored (the general engines' formats), B = exact 12-bit storage
FLAGS="${MK_CFLAGS:-}"
[ "$TRACK" = "B" ] && FLAGS="$FLAGS -DW12_ONLY"
stamp=$( (cat mk.cu build.sh; echo "$FLAGS") | sha256sum | cut -d' ' -f1)
if [ ! -f build/libmk.so ] || [ "$(cat build/stamp 2>/dev/null)" != "$stamp" ]; then
  rm -f build/libmk.so
  nvcc -O3 -std=c++17 -arch=sm_100a $FLAGS -shared -Xcompiler -fPIC -Xptxas -v -o build/libmk.so.tmp mk.cu 2> build/ptxas.log || { cat build/ptxas.log; exit 1; }
  mv build/libmk.so.tmp build/libmk.so
  echo "$stamp" > build/stamp
  echo "built libmk.so"
fi
