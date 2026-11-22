# Plan (run s18-mini-b200-c2, direction: batch-8 speculation), written at ~1.5 agent-hours

## Facts that decide the design
- Inherited engine (s16 s0022): B=1 spec 1528 us/token; B=8 runs the step kernel's device loop, 6.3 ms/token.
  Profile on card 1 (ctx 600): B=8 step 6548 us = 124 us/plain app (attn 23, w13 30, w2 32, mlp_in 8, routing 7) +
  148 us/gated app. Verify pass V(K) of ONE sequence: 3394 (K=1) .. 5394 us (K=8).
- The megakernel is built around 8 B-operand columns (MAXB 8) held in shared memory; 16-row passes in 8-row halves
  cost 10.1 ms (s16 proto16). A 64-row pass (8 sequences x 8 positions) cannot be had by extending it cheaply.
- Evaluator (harness eval/_engine_runner.py, kernel_eval.py): the reference greedy tokens for the http_tokens and
  generate gates are step() with ALL 8 SEQUENCES BATCHED ("free" run). B=1 vs B=8 bitwise equality is recorded,
  not gated (batch gate: p99 within tolerance); invariance gate: reversed slot order bit-identical.
  => step() for B >= 2 may use different (batch-invariant) arithmetic than the B=1 megakernel. That is a track-A
  change of step() (submitted as track A first). The B=1 path stays the inherited megakernel, bit for bit.

## Design: "wide" verifier wk.cu (libwk.so), used by step() for B >= 2 and by B >= 2 speculation
- Rows = (sequence slot, position, token), up to 64 per pass; per-row arithmetic independent of the other rows
  (fixed k-splits per GEMM, absolute 128-key attention chunks merged in order, per-row norms/mixes, routing per row,
  expert GEMMs per (expert, token) column): a row's logits are the same in a 1-row step and in a 64-row verify pass.
- Persistent cooperative kernel, 512 threads, grid barriers between phases (no tags), own K/V caches and conv rings
  (ring written before any conv reads it; rows of a sequence are consecutive positions; rejected rows are simply
  overwritten by the next pass).
- GEMMs on mma.sync m16n8k16 with B operands staged in shared memory (dense: k-slices; experts: the expert's token set).
- Expert inputs exact (f16 hi/lo columns as the megakernel), NVFP4/MXFP8 decoded exactly.
## Steps
1. wk.cu correct (vs megakernel logits: rel. L2 ~1e-3, top-1 ~100%), bitwise row invariance across compositions.
2. step(B>=2) -> wk; track-A submission (expect B=8 step ~3.5-5 ms vs 6.3).
3. B>=2 serving with speculation: batched draft (DSpark per sequence, n-gram), ragged acceptance, one wide verify
   pass per cycle. Target: below vLLM+DSpark 2.71 ms at B=8 (natural 1.85).

## Re-plan at 2.5 agent-hours (board clock; I had over-estimated elapsed time earlier)
State: s0004 valid: b1 1523, natural_b1 1015, b8 7186, natural_b8 4464 (inherited: 6302 / 6508; vLLM+DSpark 2711 / 1851).
Cycle at B=8 natural: verify 14.9 ms (93%) + drafts 1.1 ms. Wide pass 8.6 ms (8 rows) .. 18.5 ms (64 rows).

Three largest losses (per application at 64 rows, PDL off, nsys + in-kernel marks):
1. MoE w13 135 + w2 54 us vs ~73 us of bytes. Found: routing tables took 42 us of the w13 kernel (serial scans) ->
   rewritten (masks + scans) -> 23 us, still being broken down. The item loop ~0.5 us per 576-B unit per warp,
   unchanged by ring depth, two mma chains, 24 warps, register pipelining -> suspect the FP4 cvt pipe (testing with a
   doubled-conversion build). Mechanisms: (a) cheaper routing (parallel tables, logits prefetched), (b) FP4 decode by
   PRMT byte lookup + sign OR instead of cvt (integer pipe), (c) larger token groups per expert item (amortize decode).
2. Dense GEMMs (qkv 34, gate-D 28, o_proj 19, F/G 16, router 14 us) vs ~8 us of bytes. Mechanisms: (a) one 64-row
   chunk with k-stages instead of two 32-row chunks (weights read once), (b) L2 bulk prefetch at kernel entry (done:
   qkv 23 -> 19 at 8 rows), (c) fewer, larger GEMM kernels (q/k/v + router in one launch where inputs allow).
3. Fixed per-kernel cost (~15 kernels x 48 apps, ~3-6 us each incl. ramp/tail and prologue). Mechanisms: (a) fuse
   small kernels where rows are independent (kv into attention items; w2sum/conv per sequence: tried, too few blocks),
   (b) a persistent kernel per group of phases with grid barriers (lost before to register pressure; per-phase noinline).
Next: finish 1(a), test 1(b); then 2(a).

## Re-plan at 5.4 agent-hours (s0011 best b8 5740; natural_b8 ~3570-3600; b1 1521-1526)
Where a 48-row pass goes (14.1 ms, per-launch timeline tests/w/wtl.py): MoE w13 + w2 5.96 ms (42%; near its
issue / pipe limits: loads-only build ~same time), attention 1.64, dense GEMMs 3.8 (latency chains, not bytes),
launch gaps 0.97 (675 launches x ~1.4 us), row jobs / small kernels ~1.5. Draft 0.85 ms per cycle after s0011.
Tried without gain (bit-identical): GEMM double-buffered B stages, top-8 with fewer reductions, attention relp prefetch,
draft E4M3 X / prefetch / shared-weight layer GEMVs.
Next, in order: (1) draft threshold 0.6 for B >= 2 (s0012); (2) fewer launches: MoE output conv computed by the next
application's first row job (-48 launches, bit-identical); w2sum into w2 by per-(row, strip) last arriver;
(3) if time remains: smaller per-kernel prologues (pointer chains before the PDL wait).
Final hour: notes, summary, the best engine committed and buildable (track-A engine = s0001 tree, TRACK file).
