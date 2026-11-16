#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
# Track switch: engine/TRACK holds "A" or "B" (default A). MK_TRACK overrides it.
set -euo pipefail
cd "$(dirname "$0")"
TRACK="${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}"
mkdir -p build
FLAGS="-O3 -std=c++17 -arch=sm_100 -shared -Xcompiler -fPIC"
if [ "$TRACK" = "B" ]; then FLAGS="$FLAGS -DTRACK_B"; fi
EARLY=""; if [ "$TRACK" = "B" ] && { [ -f EARLY_ROUTE ] || [ -n "${MK_EARLY:-}" ]; }; then EARLY=1; FLAGS="$FLAGS -DEARLY_ROUTE"; fi      # experimental, see notes
if [ -z "$EARLY" ] && { [ -f SPEC_ROUTE ] || [ -n "${MK_SPEC:-}" ]; }; then EARLY=S; FLAGS="$FLAGS -DSPEC_ROUTE"; fi      # speculative routing
if [ -n "${MK_POLL_PLAIN:-}" ]; then FLAGS="$FLAGS -DPOLL_PLAIN"; fi
if [ -n "${MK_EXTRA:-}" ]; then FLAGS="$FLAGS $MK_EXTRA"; fi      # experiments with compiler options
if [ -n "${MK_PLAIN_STORE:-}" ]; then FLAGS="$FLAGS -DPLAIN_STORE"; fi
if [ -n "${MK_FENCE_MASK:-}" ]; then FLAGS="$FLAGS -DFENCE_MASK=$MK_FENCE_MASK"; fi
if [ -n "${MK_NACC:-}" ]; then FLAGS="$FLAGS -DNACC=$MK_NACC"; elif [ -f NACC_$TRACK ]; then FLAGS="$FLAGS -DNACC=$(cat NACC_$TRACK)"; fi
SRC_HASH=$( (cat mk.cu build.sh TRACK NACC_A NACC_B 2>/dev/null || true) | sha1sum | cut -c1-16)-$TRACK-${MK_NACC:-}-${MK_POLL_PLAIN:-}-${MK_EXTRA:-}-${MK_PLAIN_STORE:-}-${MK_FENCE_MASK:-}-$EARLY
if [ ! -f build/libmk.so ] || [ "$(cat build/hash 2>/dev/null)" != "$SRC_HASH" ]; then
  nvcc $FLAGS --resource-usage -o build/libmk.so mk.cu
  echo "$SRC_HASH" > build/hash
fi
if [ -n "${MK_PROF:-}" ]; then nvcc $FLAGS -DMK_PROF ${MK_PROF_FINE:+-DMK_PROF_FINE} -o build/libmk_prof.so mk.cu; fi
echo "built track $TRACK"
