#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
FLAGS=${MK_CFLAGS:-}
nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared $FLAGS -o build/libpe.so pe.cu -lcuda -lpthread
# the decode megakernel of run s7 (MK_ENGINE=megakernel): kept buildable, not used by the serving engine
if [ -n "${MK_BUILD_MEGAKERNEL:-}" ]; then
  nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -shared $FLAGS -o build/libink.so ink.cu -lcuda
fi
