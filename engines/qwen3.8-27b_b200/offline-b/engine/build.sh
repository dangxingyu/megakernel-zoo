#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
# NO_STAMPS: the debug timers (per-op, per-block, per-warp stamps for tools/op_times_p.py, tail_p.py, ...) compiled
# out: -0.4 % per B=1 step. The tools need them: build with MK_CFLAGS=-DBAR_SLEEP_NS=200.
# BAR_SLEEP_NS: grid-barrier spin back-off; 25 ns since s5 hour 5.4 (W12 engine: 200 -> 25 ns -0.15 %)
FLAGS="${MK_CFLAGS:--DBAR_SLEEP_NS=25 -DNO_STAMPS}"
# engine/TRACK "A W12": every streamed matrix in exact 12-bit storage; the BF16 stream loops are not compiled in, and
# GEMM inputs at one column use the compact layout (X_COMPACT; MK_W12_EXTRA="" builds without it)
# W12_RS (s11): rows stored scaled by exact powers of two, one decode base, the factor applied in the epilogue
# L0_TABLE (s11): layer 0 in_proj output per token id (api.py builds it with this kernel at load), its op removed
# HEAD_SK (s11): greedy passes without logits take the argmax through a certified int8 sketch of lm_head + exact rescoring
# W12E_NQREG (s11): the W12 stream keeps the next escape position in a register instead of a shuffle per iteration
# EPI_PF (s11): the epilogues' per-row constants (norm gains, RS factors) of the block's tiles copied to shared memory at op start
# no -DW12_MIXED since s11 s0007: no BF16 twins of out_proj / o_proj (W12 is faster at B=1 with the s0005+ loop)
# SK76 (s11): head sketch tiles in 6 bits except the frequent candidates (engine/sk6_tiles.bin, 6.1 % in 7 bits)
# SK_P2X (s11): the sketch's x scale is a power of two (exact digits without divisions)
# SK_KPT=8 (s11): the head's exact rescoring relays the 320-MMA chain in 40 turns of 8 k-steps (was 64 of 5)
# SK_FOLD (s11): the head kernel's argmax op folded into its finish op (one grid barrier less per greedy pass)
# SK7 + SK7S (s11): the head sketch in 7 bits per weight (q + 64 in 7 of 8 bytes, s = max|w| / 63), loads in two halves
if grep -qw W12 TRACK && [ -z "${MK_NO_W12_ONLY:-}" ]; then FLAGS="$FLAGS -DW12_ONLY ${MK_W12_EXTRA--DX_COMPACT -DW12_RS -DL0_TABLE -DHEAD_SK -DW12E_NQREG -DEPI_PF -DSK7 -DSK7S -DSK_FOLD -DSK_KPT=8 -DSK_P2X -DSK76}"; fi
FLAGS="$FLAGS ${MK_W12_PRMT--DW12_PRMT}"   # W12 byte layout (api.py MK_W12_FMT=prmt); MK_W12_PRMT="" + MK_W12_FMT=pack: s0010
stamp=$( (cat mk.cu build.sh; echo "$FLAGS") | sha256sum | cut -d' ' -f1)
if [ ! -f build/libmk.so ] || [ "$(cat build/stamp 2>/dev/null)" != "$stamp" ]; then
  nvcc -O3 -std=c++17 -arch=sm_100 $FLAGS -shared -Xcompiler -fPIC -Xptxas -v -o build/libmk.so.tmp mk.cu 2> build/ptxas.log || { cat build/ptxas.log; exit 1; }
  mv build/libmk.so.tmp build/libmk.so
  echo "$stamp" > build/stamp; echo "$FLAGS" > build/flags
  echo "built libmk.so"
fi
