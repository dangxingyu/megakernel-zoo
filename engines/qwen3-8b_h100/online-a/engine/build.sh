#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
# engine/TRACK holds A or B (default B): the board's two tracks (eng.cu TRACK_A, README "Track")
TRACK=$(cat TRACK 2>/dev/null || echo B)
[ "$TRACK" = A ] && MK_DEFS="${MK_DEFS:-} -DTRACK_A=1"
# libeng.so.flags: the track and flags the library was built with (a synced TRACK keeps an old mtime: rebuild on any change)
FLAGS="TRACK=$TRACK ${MK_DEFS:-}"
if [ ! -f libeng.so ] || [ "$(cat libeng.so.flags 2>/dev/null)" != "$FLAGS" ] || [ eng.cu -nt libeng.so ] || [ build.sh -nt libeng.so ] || [ gen_wg.py -nt libeng.so ] || [ dsched.h -nt libeng.so ] || [ -f TRACK -a TRACK -nt libeng.so ]; then
  python3 gen_wg.py
  # sm_90a: the Hopper warpgroup MMA (wgmma) is not in the plain sm_90 target
  $NVCC -O3 -use_fast_math -std=c++17 -gencode arch=compute_90a,code=sm_90a -shared -Xcompiler -fPIC ${MK_DEFS:-} -o libeng.so.tmp eng.cu
  mv libeng.so.tmp libeng.so
  echo "$FLAGS" > libeng.so.flags
fi
