#!/usr/bin/env bash
# bash tools/mkprof/selftest.sh: build the example with marks on, run it, print the table (expect: stream near the
# card's read bandwidth, compute balanced, unbalanced with imbalance ~ 2.5 and the most waiting).
set -euo pipefail
cd "$(dirname "$0")"; tmp=$(mktemp -d); cp example.cu mkprof.cuh mkprof.py "$tmp"; cd "$tmp"
nvcc -O3 -arch=native -DMKPROF -o ex example.cu && ./ex
python mkprof.py prof.npy --tags tags.json --bytes bytes.json --card "${MK_HW:-h100}"
nvcc -O3 -arch=native -o ex_off example.cu && echo "build without -DMKPROF: ok"
