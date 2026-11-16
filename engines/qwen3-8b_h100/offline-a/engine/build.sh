#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
# The track is chosen by the file engine/TRACK (one letter) or the environment variable MK_TRACK:
#   B (default): model-specific lossless formats (12-bit weight code, exact power-of-two scales)
#   A          : aligned with the general engines (BF16 weights as stored; see the header of mk.cu)
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
TRACK=${MK_TRACK:-$(cat TRACK 2>/dev/null || echo B)}
TRACK=$(echo "$TRACK" | tr -d '[:space:]')
DEFS=""
[ "$TRACK" = "A" ] && DEFS="-DTRACK_A"
if [ ! -f libmk.so ] || [ mk.cu -nt libmk.so ] || [ build.sh -nt libmk.so ] || [ "$(cat .built_track 2>/dev/null)" != "$TRACK" ] || { [ -f TRACK ] && [ TRACK -nt libmk.so ]; }; then
  $NVCC -O3 -use_fast_math -std=c++17 -arch=sm_90 -maxrregcount=80 -shared -Xcompiler -fPIC $DEFS ${MK_DEFS:-} -o libmk.so.tmp mk.cu
  mv libmk.so.tmp libmk.so
  echo "$TRACK" > .built_track
fi
