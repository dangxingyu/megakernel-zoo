#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
FLAGS=${MK_CFLAGS:-}
TRACKV=${MK_TRACK:-$(cat TRACK 2>/dev/null || echo A)}
if echo "$TRACKV" | grep -qw W12; then FLAGS="$FLAGS -DW12"; fi
if echo "$TRACKV" | tr '_,+' '   ' | grep -qw W12U; then FLAGS="$FLAGS -DW12U"; fi
nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared $FLAGS -o build/${MK_LIBNAME:-libink.so} ink.cu -lcuda
