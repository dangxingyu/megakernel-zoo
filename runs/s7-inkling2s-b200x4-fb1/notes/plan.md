# Plan (written at 0.3 h)

Target: Inkling-2-Small decode on 4x B200 (152 SMs, 227 KB smem, 126 MB L2 per card; 955 GB host RAM).

## Split over the cards (all cards hold the full per-token state, replicated)
- Attention: TP4 by heads: card c owns q heads 12c..12c+11, KV heads 2c, 2c+1 (q/gate/r/sink/k/v/dyn-tap rows),
  o_proj columns of its heads -> partial [B,4608] -> pushed to every card (one-shot all-reduce, fixed summation order).
- Experts: TP4 inside every expert: card c owns intermediate [576c, 576c+576) of all 384 experts
  (w13 rows 1152c.., w2 columns 576c..). Every card reads exactly 1/4 of every chosen expert: no load imbalance,
  no token all-to-all. Cost: the NVFP4 input quantization of w2 needs the per-token amax over all 2304 -> a tiny
  cross-card exchange of 8 floats per token per layer (layers 0-39 only; MXFP8 blocks of 32 are local).
- Router, mixes (attention residuals), sconvs, norms: replicated on every card (no exchange).
- Unembedding: rows split 4 ways; logits gathered to card 0; greedy argmax reduced across cards.

## Kernel
- One persistent kernel per card per step (op program, grid barrier between dependent ops, cross-card flags).
- GEMV via mma.sync m16n8k16 (batch = the 8 columns), weights re-laid in fragment order at load.
  NVFP4: cvt.rn.f16x2.e2m1x2 + HMUL2 by the E4M3 block scale (exact in f16), activation codes x block scale
  (exact in f16), FP32 accumulate, outer scales at the end. MXFP8: mma m16n8k32 e4m3 per 32-block, UE8M0
  scales applied per block in FP32.
- Needs -gencode arch=compute_100a,code=sm_100a (e2m1x2 cvt is an "a" feature).

## Order
1. GEMV microbenchmarks (bandwidth per format).
2. Loader + re-layout + per-op kernels, tested op by op against a PyTorch fork-mode reference of one layer.
3. Whole step, compare logits to the starter's (fork) on a few prompts; first submission.
4. Then: barrier/latency work, prefetch during syncs, track-B ideas (BF16 attention weight compression,
   layer-0 tables, certified argmax on the greedy path).

## State at the end of the extension (15 h, 2026-11-16)

The plan above is the inherited one (my own kernel, s0002-s0028; its final source is `scratch/engine_s0028/`).
Since the extension `engine/` is the sibling run fa1's kernel (its s0020: team token phases, gate split beside the
attention, a tcgen05 kernel from batch 4) with my track-B changes and guards:

- `engine/` = s0036's source (s0033's kernel; the 12-bit round trip asserted at every load). `engine/TRACK`:
  `B W12U L0T` (track B; `-DW12U` at build) or `A` (s0032's kernel).
  Load: BF16 or 12-bit unembedding; with L0T the loader runs layer 0 alone for all 201,024 ids (5 s) and installs the
  tables (`_l0_rows`, `_install_l0_tables`); `ink_l0t_ok()` guards the condition for bit-identity.
- Guards (api.py): `ctx_limit = min(max_context, ink_max_positions() = 2176)` checked in `step()`, `generate()`,
  `submit()` and `_launch()`; `submit()` also refuses token ids outside [0, 201024).
- One binary for both tracks: `scratch/mix/engine/` (= s0034 / s0035), 5 us slower than s0033 by the build.
- Checks to run before any track-B submission on this kernel: `tools/x_verify.sh` (bitwise logits of three builds at
  B = 1, 2, 4, 5, 8 and tables for every id; `x_verify2.sh` adds B=3), `tools/x_guard.py 0` and `1`,
  `tools/rehearse.py --online`. Measuring: a switch inside one library or `tools/x_stamp.py`, never two builds (G11).
