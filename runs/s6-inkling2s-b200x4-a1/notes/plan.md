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
