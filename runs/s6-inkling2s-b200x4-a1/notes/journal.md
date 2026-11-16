# Journal

## 0.0-0.5 h: bring-up

- Pod facts: 152 SMs/card, 227 KB smem/block, 126 MB L2, 955 GB host RAM (the 393 GB of weights stay in the page cache:
  the starter loads in ~2 min, my loader 48 layers in ~50 s with one thread per card).
- sm_100a has no warp-level block-scaled MMA (`mma ... kind::mxf4nvf4.block_scale` is sm_120a only); only tcgen05.
  `cvt.rn.f16x2.e2m1x2` and `cvt.rn.f16x2.e4m3x2` exist but need `-gencode arch=compute_100a,code=sm_100a`
  (plain `-arch=sm_100a` emitted compute_100 PTX and ptxas rejected the e2m1 cvt).
- F2FP.E2M1.UNPACK_B reads a byte selector (.B1/.B2/.B3) for free when the PTX unpacks with `mov.b32 {b0,b1,b2,b3}`;
  with `(w >> 8) & 0xff` ptxas emits a LOP/SHF per register.
- GEMV microbenchmark (tools/bench/gemv2.cu, one card, 300 MB, m16n8k16 MMA, fragment-ordered weights):
  BF16 6.2 TB/s (1024 threads, 2-unit unroll), NVFP4 (cvt + HMUL2 by E4M3 scale) 4.4 TB/s with byte extraction,
  5.9 TB/s with free byte selects and H0_H0-swizzled scale multiplies. 512 threads/block: 5.8 / 4.5 TB/s.
- First engine (6 layers, B=2, 3 steps teacher-forced) vs the starter's fork path: routes identical at step 0 in all
  6 layers; attention outputs rel 1e-3..2e-2, MLP outputs rel 1e-2..8e-2 growing with depth; the four cards' replicated
  state bitwise identical.

## 0.5 h (pod outage, clock stopped): structure changes written before any timing

- Routing (top-8 + union) computed redundantly by every block from the router logits: -2 grid barriers per layer.
- Per-token step state (8 finished attention-residual blocks, the partial block, the attention output) lives in the
  shared memory of the block that owns the token: these vectors exist only within one step (only K/V and the conv
  histories persist), so they never need global memory.
- B=1 kernel instance (`ink_step_red`): every block computes the per-token vector phases itself (mlp/attn all-reduce
  sums, convs, residual mixes, norms, NVFP4/MXFP8 input quantization, w2 input quantization) into compact
  one-column MMA fragments in its own shared memory: the GEMVs read their B operand from shared memory and three
  grid barriers per layer disappear (after attn_in, after post_attn, after quant2). Conv histories double-buffered by
  step parity (one writer, many readers). B>=2 keeps one block per token + barriers (`ink_step_std`).
- L2 prefetch (cp.async.bulk.prefetch.L2) of the next GEMV's weights by blocks idle in latency phases.

## 0.5-1.7 h: first full-model runs (pod epochs 3-4, long outages in between)

- Correct from the first full run: argmax identical to the fork reference at 12/12 steps x 2 sequences, the 4 cards'
  replicated state bitwise identical, online submit/poll and generate() == step-by-step greedy at B=1/2/8.
- Own float64 check (tools/ref_dump.py --mode high, 2 sequences x 12 positions, tools/t_gate.py): engine vs float64
  centred rel-L2 median 0.050, max 0.10, top-1 100%; the fork path on the same positions 0.054 / 0.11. Limits 0.179 / 0.707.
- Speed far below the byte count: 10.7 ms/step at B=1 first (222 us/layer). Causes found with in-kernel globaltimer
  stamps (MK_TPROF):
  * remote atomicMax for the w13 amax: ~18K NVLink atomics per card per layer onto 8 addresses -> warp max + local
    atomic + one remote store per (expert, card) by the expert's last strip: w13 47 -> 23 us.
  * register spills (stack frame 320 B/thread x 1024 threads = 327 KB per SM vs ~34 KB of L1 next to 222 KB of
    shared memory: every spill access went to L2) and pointer arrays with dynamic indexing in the residual mixes and
    the top-8: mixes 16.7 -> 4.0 us, top-8 13.5 -> 3.4 us after keeping per-source scalars in shared memory.
  * B=1 mode race: GEMVs read shared-memory fragments written by other warps without a __syncthreads (the grid
    barrier hid it at B>=2) -> runs differed from step 0; fixed, B=1 and B=8 now bitwise reproducible.
- s0001 deadlocked in the evaluator's time stage: the per-step header was copied from a 64-slot pinned ring and the
  host ran >64 launches ahead (256 steps without a sync) -> headers overwritten before their async copy -> cards
  waited for counter values that never came. My tests never queued >64 steps. Fixed (header as kernel argument);
  tools/rehearse.py now reproduces the evaluator's unsynchronised loops and the HTTP path before any submission.

## s0002 (4.6 h): first valid submission, track A
- Expected ~6700 / ~10400 us (B=1 / B=8 over HTTP); measured 6564 / 10330. Gates: median 0.078 (limit 0.179), max 0.31
  (0.707); long 0.059 / 0.46, top-1 98.5% (limit 97.6%); http tokens 8/8 identical; generate 8/8.
- What moved it from s0001's 8.3 ms: (1) the per-step header passed by value (deadlock fix); (2) cross-card signal:
  one system fence then relaxed reductions instead of four release reductions per block (o_proj+signal 19 -> 8 us,
  w2+signal 22 -> 10 us; a release reduction to peer memory costs about a system fence each, ~2 us measured); (3) B
  fragments staged in shared memory in the B>=2 kernel (w13 at B=8 67 -> 47 us); (4) history loads and finished-block
  mix statistics before the cross-card waits.
- Primitive latencies (tools/bench/prim.cu, e0029): grid barrier 152 blocks 1.25 us; dependent L2 load 0.15 us warm,
  0.21-0.42 us after streaming 8 GB; fence.sc.sys after remote writes 2.05 us; fence.gpu 0.53 us.
- Instruction cache is not the problem (routing phase 4.6 us cold vs 4.0 us warm, -DICTEST).

## 5.4 h: s0003 submitted (queued by a pod loss); findings since s0002 (all with in-kernel globaltimer stamps)
- Strided GEMV warp mapping (logical warp = warp x 152 + block) made the same ~6-10 blocks 2x slower in every
  expert GEMV (per-block end times correlate 0.94-0.99 across layers and runs; the slow blocks' indices form
  arithmetic progressions, their SM ids look random -> a mapping effect: a block's 32 streams sit 729 KB apart in
  lockstep). Block-contiguous ranges when all warps are busy: B=8 w13 per-block ends balanced (31.7-35.1 us).
- Block-per-strip GEMV (32 warps split one strip's K, shared-memory reduction, no global hand-off) wins for qkv,
  o_proj and w2 (w2 13.7 -> 9.6 us, qkv 12 -> 10.5) but loses badly for w13 (21.7 -> 29.5; B=8 56 -> 127) and for
  the unembedding (80 -> 110): GEMV_BLK=19.
- 768 threads per block (80 registers, spills halved) instead of 1024 (64): B=1 5.95 -> 5.40 ms, B=8 unchanged.
- Deeper FP4 unroll (2/4/8 units in flight per warp): no difference -> per-warp memory parallelism is not the limit.
- Instruction cache: routing phase 4.6 us cold / 4.0 us warm -> not the main cost.
- Written after s0003 (not yet measured): LL exchange (every cross-card value travels with its tag in one 8-byte
  store, consumers poll their own elements; no system fence, grid barrier or counter for the 3 exchanges per layer);
  L2 prefetch of K/V rows and of the small per-layer weights during the attention-input phase.
- s0003 (5.4 h, track A): 5324 / 9310 us (pred 5350 / 9450). 768 threads, contiguous GEMV, blk GEMV for qkv/o_proj/w2, V8 vector code, attention rewrite.
- s0004 (6.1 h): incorrect, http_tokens 0.155 prefix, 0/8 identical. Cause found (tools/t_evalorder.py, t_perm.py,
  t_perm2.py): results depended on the batch slot of a sequence: w2's inputs bitwise equal in both orders, outputs
  1 ulp apart -> the per-column scaling acc += a * s was contracted to FMA differently for even and odd columns. The
  HTTP threads deliver the 8 requests in another order than the evaluator's rows. Fixed with explicit __fmaf_rn;
  slot-order invariance now tested (t_perm: logits bitwise equal under a permutation).
- s0005 (6.6 h, track A): 4778 / 8845 us (pred ~4800 / ~9150). Slot-invariant; LL exchange; router blk; attention fixes.
- W12 (track B) build: verified bit for bit on all 5,984,649,216 dense values (432,825 escapes) and exact tokens, but 47.6 ms/step (10x slower): the decode/escape path inside the engine is pathological (not investigated yet; dense GEMVs are latency-bound, so W12's upside is small here).
- s0006 (7.2 h, track A): 4581 / 8242 us (pred 4590 / 8450). HW E2M1 rounding (bitwise identical), 64-position chunks at B=1, w2 prefetch only at B=1.

## 7.2-7.9 h: B=8 profile on realistic routing (no submission yet)
- My t_perf used random token ids; at B=8 it gave 7.1 ms vs the evaluator's 8.38 ms (step_us). With text tokens
  (stdlib module sources through the tokenizer, `t_perf.py --text`) it gives 8.41 ms = the evaluator's number. The
  union of experts per layer at B=8 on text is 47.5 on average (35..61; `--ids` prints it): random ids route alike.
  All B=8 numbers below are on text tokens, ctx 192.
- B=8 per layer was 172 us: w13 46.8, w2 36.5, post 19.4, attn_in ~21, attention 13, quant2 12.3, qkv 9.8.
- w2 at B=8 streamed its 67 MB/card at ~2 TB/s. Timing-only experiments (wrong results; routing changes so normalised
  by w13): fragments from L1 instead of L2 -> w2 -15%; one expert's weights (L2-resident) -> -38%. Bulk L2 prefetch
  of every warp's range (cp.async.bulk.prefetch.L2) made every GEMV slower (B=8 8411 -> 8509..9203 us): dead end.
- No FP4 warp-level MMA on sm_100a (ptxas: kind::f8f6f4 / block_scale need sm_120a); only tcgen05.
- FP4 GEMV micro-bench (tools/bench/fp4x.cu, 768 threads): 140 MB 4.5 TB/s, 300 MB 5.5 TB/s; replacing the e2m1
  conversion with plain logic ops, or dropping the scale multiply, or dropping the MMA: <= +6% -> the FP4 decode is
  not the limit; bf16 at the same size is only 8% faster. In-kernel GEMVs at B=8 lose to the bench in start skew,
  epilogue/hand-off and (w2) serial fragment loads.
- Kept (all bitwise identical to the previous build, tools/t_cmp.py): quant2 at B>=2 as warp tasks (pair x 32-value
  slice; one warp per pair was slower: 18 serial slices); FP4_UNROLL 4 (B=8 w13 -2.3 us); w2 at B>=2 loads weights
  AND activation fragments of 2 units together across expert boundaries (the fragment load used to wait for the
  weights: 4 serial L2 round trips per group). B=8 8410 -> 7810 us in-process, B=1 unchanged (4535).
- Tried and dropped: whole-strip work granules for flat GEMVs (no hand-off): B=8 equal, B=1 +165 us. W2G=4 spills.
- s0007 (8.2 h, track A, scheduling): 4541 / 7731 us over HTTP (pred ~4540 / ~7650). Gates: median 0.080/0.082,
  max 0.26/0.30, long 0.059 / 0.45, top-1 long 99.3%; http_tokens 8/8. B=8 gain from the w2 load structure,
  warp-task quant2 and FP4 unroll (bitwise identical to s0006 apart from the router); B=1 gain (-40 us) from the
  split-K router (24 strips x 3 K-ranges on 72 SMs; routing sums the 3 partials in order). A first split-K version
  with 6 ranges was slower (+115 us): each routing thread spun on its 6 partials one after the other; loads first,
  then the spins.
- s0008 (9.2 h, track A, sync): 4311 / 7477 us over HTTP (pred ~4300 / ~7450); bitwise identical to s0007.
  What moved B=1 (in-process A/B, text tokens, ctx 192): redux.sync top-8 -50 us; raw8 (the 4-card all-reduce reads
  of attn_in and post) with the 4 cards' loads in flight before spinning -70 us (B=8 -180; 8 or 16 loads in flight
  spill and lose); the same for quant2's amax poll -48 us; LL backoff 32 -> 8 ns -21 us; L2 bulk prefetches issued by
  the last warp instead of warp 0 -38 us (stamps: warp 0 spent 4.9 us per layer issuing ~6 bulk prefetches while the
  whole block waited for it at the next barrier). Lesson: any spin loop or slow-issue instruction on a warp that
  the block's next __syncthreads waits for is on the critical path.
- Dead ends this stretch (all measured, e0123-e0134): attention prologue reorder (+22 us), 4-chain score dots (0),
  block-contiguous GEMV with cross-block hand-off for qkv/o_proj/w2 (+200..+750 us: the hand-off costs more than the
  2-strip imbalance), single-round 3-unit bf16 loads (+13 us), per-warp cp.async ring for w2 at B=8 (+500 us: the
  fragment loads became serial), L2 256B/128B load hints (0).
- s0009 (10.1 h, track A, memory): 4260 / 7415 us over HTTP (pred ~4265 / ~7400); bitwise identical to s0008.
  1.106x SGLang at B=1 (the claim's B=1 line is 4283). In-process A/B: FP4 unroll per mode (red 3 / std 6) B=1
  -27 us, B=8 -10; MXFP8 unroll red 1 / std 3 (B=8 -18); router bias loaded before the poll -12 us; at B>=2 the conv
  history/taps cp.async'd into shared memory before the cross-card wait (B=8 -25; the same in red mode was +277 us,
  not understood) and the chunk combine with 8 loads in flight (B=8 -12; red mode +17).
  Also measured and dropped: per-strip amax slots instead of atomicMax (B=1 +27, B=8 +300), fence.acq_rel instead of
  __threadfence (0), preloading the conv inputs into registers before the spin (+58: registers across the spin).
- s0010 (10.5 h, track A, sync): 4078 / 7214 us over HTTP (pred ~4070 / ~7220); bitwise identical to s0009.
  (1) qkv strip order: the 140-142 strips attention reads (q, r, k, v) first, one per block, the 96 gate rows and the
  sinks (read only by the combine) in the second wave; attention items run on the last blocks (one strip each) and
  wait on a counter of the first-wave strips (red.release per strip) instead of a grid barrier: B=1 -174 us at ctx
  576, B=8 -19. (2) Hand-offs: one atom.acq_rel instead of fence.sc + atomic (+ fence on the reader), acquire fences
  after the amax polls: B=1 -58 us, B=8 -90 (same values; fence.acq_rel alone instead of __threadfence was 0).
- s0011 (11.2 h, track A, correctness): 4049 / 7260 us over HTTP (pred ~4055 / ~7200). Found in review: s0010's early
  attention also applied at B>=2, where the last chunk block's combine reads the gate rows and sinks (second qkv
  wave) with no wait: a race hidden by timing (all gates and bitwise checks passed). Fixed: early attention (strip
  order, counter, item mapping) only in red mode (B=1), whose combine sits behind a grid barrier; B>=2 back to the
  barrier (with a second-wave counter the early start gained nothing at B=8, e0161-e0163). Paired strips: o_proj
  blocks with two strips issue both strips' loads in one round (B=1 -17 us); B>=2 w2 reduces both strips in one
  round (B=8 -50 us). Dropped: per-warp fused w2 quantization at B=1 (bitwise identical, +410 us: spills), red-mode
  two-strip w2 loads (+355 us: spills), counters instead of the post-attention barrier at B=1 (0).
- s0012 (12.6 h, track A, memory): 3959 / 7194 us over HTTP (pred ~3930 / ~7180); bitwise identical to s0011.
  The red-mode (B=1) kernel is at the 80-register cap (768 threads) and spills; every extra live value costs.
  Measured in process (ctx 576): flat GEMV indices 64 -> 32 bit (B=1 -55 us); block-uniform launch state (barrier
  count, counter bases/accumulators) in shared memory instead of registers (B=1 -14, B=8 -83); quant2 (red) two
  rounds per iteration so two four-over-six chains interleave (a round was ~0.9 us of pure latency) (B=1 -64).
  Everything that added live state to the red kernel lost: two-strip w2 loads (+355), fused per-warp w2 quant
  (+410), all-rounds act preload (+120), 8-deep chunk-combine loads (+17), combine output unroll (+30), conv input
  preload (+58). tools/spills.sh maps local-memory loads/stores to source lines (lineinfo SASS).
- s0013 (13.1 h, track A, memory): 3624 / 6955 us over HTTP (pred ~3620 / ~6860). Gates: median 0.078 / 0.082,
  max 0.33, long median 0.059, long top-1 98.5% (s0012 99.3%; limit 97.6%). The o_proj and w2 tensor-parallel
  partials cross cards as (bf16 pair, tag) instead of (fp32, tag), summed in fp32 by the consumer (what a general
  engine's TP all-reduce sends). Measured in process: B=1 3967 -> 3642 us, B=8 -110. Why so much: in red mode all 152
  blocks of a card poll the four cards' partials of the whole 4608-vector twice per layer: 147 KB per block, ~22 MB
  of L2 reads per card per exchange; halving that cut each exchange wait by ~3 us. The redundant per-block global
  reads of the vector phases (LL polls, conv history and taps, attention chunk partials) are ~50 MB of L2 traffic
  per layer per card at B=1: L2 bandwidth, not HBM, is now a first-order cost there.
  Also: B>=2 attention chunk length adapted so the items fill one round of blocks (B=8 -12..-40 us).
- s0014 (13.7 h): build_failed. I had made the LL2 buffer conditional but not the LL2 functions using it; the A/B
  and validation runs before it printed timings anyway because my scripts filtered the compiler output (grep, tail)
  and then measured the previous, stale library. Consequences: e0208-e0212 "cur" numbers were a stale build, and
  the apparent +12 us B=1 regression bisected there was partly that (a correct rebuild, e0215: B=1 3636 vs s0013's
  3631, B=8 6782 vs 7064). Fixed: tools/build_check.sh fails loudly, the A/B scripts delete each library first.
- B>=2 work since s0013 (in process, ctx 576, bitwise identical): whole-region K/V L2 prefetch (the 144 non-token
  blocks issued 32 tiny bulk prefetches each before the attn_in barrier, which waited 6.4 us -> 1.8; B=8 -200 us);
  routing mask for the w2 fragments of unrouted (expert, token) columns, quant2 skips them (-87 us). Tried and off:
  two-level all-reduce at B=1 (bitwise identical, +150 us: the extra hop), rotated poll order (+430: spills),
  quant2 lookahead (+15), contiguous-row fragments with a gather (+100 at B=8).
- 14.2 h: exec e0216 queued > 15 min with the cards free; the bridge stopped answering (mk gpu / mk note). Waiting.
- 14.0-23.5 h (to the end of the run): no pod access. Exec e0216 (validation of the fixed build) stayed queued with the cards shown free and
  the board at epoch 6; the bridge never answered mk gpu / mk note (four attempts). The agent clock kept counting.
  State left: s0013 is the best evaluated engine (git tag s0013); HEAD builds (tools/build_check.sh) and adds the
  B>=2 changes (in process B=8 6782 vs 7064 us, B=1 equal, bitwise identical) plus the untested W13_EARLYPF option
  (off by default).
- 23.8 h: last attempt to resubmit the fixed build (B>=2 changes): mk submit got no answer from the bridge, nothing
  was sent. The run ends with s0013 as the best evaluated engine.
