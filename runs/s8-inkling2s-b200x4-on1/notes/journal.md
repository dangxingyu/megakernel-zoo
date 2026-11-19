# Journal

## 0.0-0.5 h: bring-up

- Machine facts: 152 SMs/card, 227 KB smem/block, 126 MB L2, 955 GB host RAM (the 393 GB of weights stay in the page cache:
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

## 0.5 h (machine outage, clock stopped): structure changes written before any timing

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

## 0.5-1.7 h: first full-model runs (machine epochs 3-4, long outages in between)

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

## 5.4 h: s0003 submitted (queued by a machine loss); findings since s0002 (all with in-kernel globaltimer stamps)
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

# Run s7-inkling2s-b200x4-fa1 (track A, 12 agent-hours, starting from run s6's records)

## 0-0.7 h: baseline and where the time goes (all measured on this node)

- Started from s6's s0005 source (open in progress/), not the inherited s0003: s0001 = that source unchanged, valid,
  4807 / 8861 us (B=1 / B=8 over HTTP), as s6 measured (4778 / 8845).
- Per-layer phase times of s0005 (in-kernel timers, context 64-128, random tokens), B=1 / B=8, us: attn_in 15.7 / 20.5,
  qkv 8.7 / 10.6, attention 11.7 / 14.7, o_proj 6.0 / 3.6, post 17.7 / 23.9, router 2.2 / 3.8, top-8 4.0 / 6.1,
  w13 16.3 / 53.1, quant2 9.1 / 8.1, w2 5.6 / 21.2; layer 97 / 166; unembedding 81 (463 MB per card at 5.7 TB/s).
- Microbenchmarks (tools/bench/cold.cu, red.cu; tools/ptx/t5.cu, t6.cu):
  * dependent 16-byte load: 180-240 ns cold from DRAM for any span up to 64 GB (no TLB cliff), 37 ns from L2,
    47 ns after cp.async.bulk.prefetch.L2 (the prefetch works, any chunk size 512 B - 32 KB).
  * a lone warp executes ~3.5-5 ns per instruction (a 6-instruction loop body: 20 ns per iteration); with 24 warps
    busy an SM gives ~2.6 G warp-instructions/s in total (FP4 GEMV: ~50 instructions per 576-byte unit -> ~5 TB/s
    per card: the NVFP4 expert GEMVs are instruction-bound at about the DRAM rate, deeper unrolling does not help:
    FP4_UNROLL 2/4/6/8 within 1%).
  * a cross-block exchange (one tagged 8-byte store, every block reads the others' words): 0.73-0.9 us per round for
    18 blocks with st.relaxed + ld.relaxed. Plain stores are NOT seen by L1-bypassing plain loads of other SMs
    (st / ld L1::no_allocate both needed), plain L1 loads see stale data forever.
  * prefetch_l2() cost 0.2 us per call (two 64-bit divisions): the w2 prefetch loop was 59 calls per block per
    layer at B=8 (~13 us of w13's 53). Rewritten with 32-bit arithmetic, one instruction per (thread = expert).
    The time did not disappear, it moved to w2: the prefetch had been hiding DRAM time inside that loop.
  * cvt.rn.satfinite.e2m1x2.f32 exists on sm_100a and equals the model's E2M1 rounding (ties to the even code,
    signs, saturation) on 2^22 values incl. all ties; cvt.rn.satfinite.e4m3x2.f32 equals __nv_fp8_e4m3(float)
    on 2^24 values. The library conversions are software (tens of instructions).
- Conclusion: the non-GEMV phases are bound by the instruction path of a thread (8 elements per thread, ~1800
  instructions per phase), not by SM capacity and not by memory. A first "team" version that only moved one token's
  18 warps to 18 different SMs was slower (same path length plus slow reductions).

## s0002 (0.9 h): team vector phases
- One element per thread: a token's 4608 elements on 6 whole blocks; warp shuffles + shared memory per block, then
  the 6 blocks' partials as tagged words; hardware E2M1/E4M3 encodes; cached 1/rms of finished blocks; the finished
  blocks' query dots computed by one warp per (source, query) pair before the cross-card data arrives. B=1 uses
  the same (std) kernel: 4180 us in process (red kernel 4790).
- Measured in process (context 64-128): layer 84.0 us at B=1 (was 96.9), 134.7 at B=8 (was 165.6); post+bar 7.8 / 8.6
  (was 15.5 / 21.9). Own float64 check (2 sequences x 12 positions): median 0.051, max 0.09, top-1 100%.
  Slot-permutation invariant (bitwise), online tokens identical to step greedy at B=1 and B=8.

## 1.0-1.5 h: expert phases at B=8 (after s0002), what did and did not work
- The evaluator-like B=8 workload (64 random tokens, then free-running generation): 50-55 distinct experts per layer
  (random-token batches only ~40, a batch of prompts generated from the same prefix ~32): expert phases are 57% of
  the layer (w13 45 us, quant2 6.5, w2 28 of 141).
- w2 input quantization per routed (token, rank) pair on its own block (64 blocks, one pass, zero-fill of the
  other columns by other blocks), hardware encodes: quant2 9.95 -> 6.5 us at B=8, 6.65 -> 5.7 at B=1. What is left is
  the wait for the other cards' amax tags (they finish w13 up to ~3 us apart) and the grid barrier.
- SASS of the FP4 GEMV loop: 16 E2M1 conversions + 16 HMUL2 + 4 E4M3 + 4 HMMA + 4 CS2R + loads = 46 instructions per
  576-byte unit. The CS2R were zero-selects for unused batch columns: removed (a column of the MMA depends only on its
  own B column). w13 epilogue: gate lanes and up lanes each do two activation values (was four on the gate lanes),
  routing weight from a (union expert, token) table: w13 44.7 -> 42.6 us at B=8.
- 1024-thread blocks: w13 44.7 -> 39.7 and w2 27.4 -> 25.5 at B=8, but attention/o_proj/unembedding slower: no net gain
  at B=8 (6671 vs 6695), +250 us at B=1. Stay at 768.
- Where the w2 slabs are prefetched (before / during / after w13 / never): w13+w2 = 76.0 / 73.4 / 70.2 / 71.8 us. Fetching
  w2 under w13 slows w13 by more than it gains: keep "after w13".
- Dead end: software-pipelined FP4/FP8 loops (next group's loads issued before the current group is computed, two
  register sets). The kernel went from STACK:96 to STACK:328 bytes per thread (REG:80 is the cap at 768 threads) and
  every phase got slower (B=1 4154 -> 4705, B=8 6624 -> 7815); as __noinline__ functions even worse (w13 86 us).
  Lesson: at 768 threads the kernel sits at the register limit; any change that adds live registers in one loop
  spills long-lived values and slows all phases. Check `cuobjdump -res-usage` (STACK) after every change.
- Block-per-strip GEMV: a block's two strips at once (12 warps each, one barrier, two reducer warps) instead of one
  after the other: fixed-token step B=1 4173 -> 4101 us, B=8 5841 -> 5746 (w2 22.6 -> 19.4, qkv 10.0 -> 9.5).
- No gain: computing the finished blocks' query dots per block without an exchange and carrying them in the main
  reduction (three barriers, six collecting warps): the pre-wait part got 0.8 us shorter, the reduction 1.5 us longer.
  The post phase is gated by the last o_proj rows arriving from the other cards (2-3.3 us after block 0's own o_proj).
- A fragment-staging loop with four loads in flight per thread pushed the kernel to STACK:264 and cost 350 us per step.
- Note for A/B: variants that change the summation order change the generated tokens and so the number of distinct
  experts (49.8 vs 42.6 per layer in one comparison); compare such variants on teacher-forced fixed tokens ("step").

## s0003 (1.9 h): 3984 / 6537 us (B=1 1.18x, B=8 1.01x SGLang). GEMV phase scheduling (see its hypothesis).

## 2.0-2.7 h: attention, fences, tag-checked hand-offs (s0004)
- Attention scores: K rows x q as 8 BF16 MMAs per 16 positions and the relative bias as one more MMA (the table
  stored distance-major), softmax pieces one (head, position) per thread: attention 12.3 -> 11.0 us. In-kernel check
  against the scalar path: 0 mismatches > 0.05 in 1.09 M scores; float64 check median 0.048 (was 0.053).
- Measured cost of a fence: removing the four __threadfence of the w13 strip hand-off / epilogue and the two of the
  attention chunk hand-off gave -163 us per step at B=1 and -192 us at B=8, but B=8 was no longer reproducible
  (two runs differed after 30-70 steps; tools/t_det.py with DET_T=120). Cause: a counter (atomic) and the data
  stores it announces go to different lines and are not ordered without the fence.
- What works without fences: every word carries its tag ((value, tag) in one 8-byte store) and the reader spins
  with RELAXED loads until the tags are current. Not usable for this: ld.global.cg (current tag, wrong value:
  e0064-e0067, with L1-bypassing and with relaxed stores alike) and L1-bypassing plain loads in phase_route
  (illegal address, e0056/e0057, cause not found).
- With that: flat-GEMV strip slots tagged; w13 epilogue only stores (activation value, tag) words of routed pairs
  (no counter, no atomic max, no fence); the w2 quantization block reads them, takes this card's max and exchanges
  it with the other cards itself. Deterministic over 120 steps at B=1 and B=8, bitwise equal to the fenced build.
  Gain smaller than the unsafe variant: B=1 3913 -> 3867, B=8 5618 -> 5533 (fixed tokens).
- No gain: per-lane best-two top-8 with ballots (the route phase is not bound by its rounds), sigmoids of the router
  logits at the producer (topk 6.0 -> 5.4 at B=8, total unchanged), 896/1024-thread blocks (again), w13 fragment
  read from L2 without staging at B=1.

## s0004 (2.9 h): 3891 / 6335 us. s0005 (3.3 h): 3646 / 5870 us = 1.29x / 1.125x SGLang: both batch sizes past 1.10x.
- s0005 is one change: shared memory per block 222 -> 121 KB (finding F5). Predicted 3650 / 5870, measured 3646 / 5870.
- Tried on top and dropped: reading the attention's K/V rows from L2/L1 instead of staging them (47 KB of shared
  memory): unembedding 80 -> 70 us per step but attention +5 us per layer, also with a per-thread L1 preload.

## s0006 (3.6 h): 3661 / 5742 us = 1.287x / 1.150x SGLang (predicted 3660 / 5740)
- NVFP4 unroll 5 and MXFP8 unroll 4 (with the larger L1; bitwise identical): B=8 -2.2%, B=1 +0.4% over HTTP.
- The activation function moved from the w13 epilogue to the w2 quantization block (bitwise identical): neutral.
- Dead ends since s0005 (all measured on fixed tokens, +-2 us repeatability):
  * weights through cp.async into the attention's staging area (pipelined, no registers): B=8 5072 -> 5768 us.
  * FP4 pipelining with two register sets: STACK:152, slower everywhere (the kernel sits at REG:80).
  * The gap between the engine's FP4 loop (0.46 us per unit and warp) and gemv2.cu is small once the benchmark
    uses the engine's conditions: random data, B fragment from global memory, 121 KB shared memory: 5.3 TB/s
    marginal there, 4.6 TB/s equivalent in the engine. Four cards busy and sustained load change nothing
    (2062 MHz, 650-735 W of 1200 W per card under the engine).
- Determinism re-checked on 1100 steps (context past the 1024 window) at B=1 and B=8: bitwise equal runs.

## 3.6-4.2 h: noise floor, L1 knee, barrier split, B=1 w13 (s0007)
- Within +-0.3% (dropped): B-dependent FP4 group size, sigmoid(gate) precomputed by idle warps in attention.
- L1 sensitivity at the current point (dummy shared memory added): +35 KB: B=1 +11 us, B=8 +23 us; +70 KB:
  +290 / +850 us. So staging less (K rows read before V rows in the same area) would gain < 0.4%: not done.
- Barrier variants (tools/bench/prim.cu, 152 x 768 threads): release add + relaxed poll + acquire fence 1268 ns,
  without the fence 1142, relaxed add without fence 806, acquire poll 1232.
- w13 at B=1: each of the 576 strips was split over ~6 warps of different blocks (flat schedule) and reassembled
  through the tagged hand-off. Whole strips per block, 4 at a time, for layers with <= 16 distinct experts:
  w13 12.1 -> 8.3 us per layer, step 3640 -> 3431 us at B=1, 3927 -> 3811 at B=2; with 32/64 as the limit B=8
  gets slower (5086 -> 5101 / 5146): the limit stays 16.

## s0007 (3.8 h): 3440 / 5773 us = 1.369x / 1.144x SGLang (predicted 3445 / 5745; B=8 path unchanged: trial spread ~0.5%)

## s0008 (4.2 h): 3401 / 5758 us = 1.385x / 1.147x SGLang (predicted 3405 / 5700)
- Attention chunk size at run time: 32 or 64 positions per block while every (token, KV head, chunk) still gets its
  own block and at most 17 chunks are combined. B=1 as predicted (-1.1%); B=8 5773 -> 5758: within the trial spread
  of the B=8 figure (s0006 5742, s0007 5773 with the same B=8 code path): the gain at B=8 exists only below
  context ~600 and is smaller than that spread.
- Considered and not done (why):
  * relaxed grid barrier with L1-bypassing stores everywhere: the release-add is what guarantees that the other
    threads' stores are visible when the count is; stores and a relaxed atomic to different lines are not ordered
    (the lesson of the fence experiments), so this cannot be made safe by changing the stores alone.
  * the gate projection (40% of the qkv GEMV bytes) computed by idle blocks during the attention phase and handed
    to the combining block as tagged words: about -2.3 us per layer at B=1 only (no idle blocks at B=8 and long
    context); B=1 is not the binding figure.

## s0009 (4.5 h): 3310 / 5716 us = 1.423x / 1.155x SGLang (predicted 3310 / 5740 +-1%)
- The output-gate strips of the qkv GEMV run in idle blocks beside the attention phase (see the hypothesis).
- After it (bitwise identical): the warp that finishes a strip adds its own piece from registers instead of
  re-reading it through two relaxed loads: B=8 fixed-token step 4952 -> 4930 us.
- Dropped: all-reduce mailboxes in fragment order (a lane's four values adjacent, two 16-byte remote stores per
  card instead of four 8-byte ones), bitwise identical: B=8 4930 -> 4914 us, but B=1 3280 -> 3344 (at B=1 only one
  of eight columns was stored before, and the reader's words are now 128 B apart instead of 8).

## s0010 (4.7 h): 3322 / 5660 us = 1.418x / 1.167x SGLang (predicted ~3310 / ~5670)
- Chunk size and gate split chosen together, gate split with >= 24 free blocks: B=8 5716 -> 5660 (-1.0%).
  B=1 3310 -> 3322 with an unchanged B=1 path: the trial spread of the B=1 figure is ~0.4%.
- Other batch sizes (2, 3, 5, 7) run deterministically (200 steps, two runs bitwise equal); generate == step ==
  online tokens at B=1, 2, 8 (tools/t_online.py).
- State of the search: the last four ideas that paid were all of one kind, "work that is needed late moves into
  blocks that idle during a few-block phase" (gate strips beside attention) or "fewer hand-offs when there are
  few items" (whole w13 strips per block at B=1, smaller attention chunks). What is left in that family is
  listed in notes/plan.md; nothing else independent of a phase's own result was found (every other GEMV output is
  needed by the very next phase).
- Routing phase split (in-kernel stamps, us per layer at B=1 / B=8): wait for the router values + selection scores
  1.2 / 2.2, the eight top-1 rounds 2.0 / 2.15, weights 0.4, union and masks 0.6. Loading the router values two per
  relaxed load cut the first part to 1.8 at B=8 but pushed the kernel to STACK:104 and cost 60-120 us per step:
  reverted (the register cap again).
  Correction after re-measuring without the two extra timer stamps: the stamps themselves had pushed the kernel to
  STACK:104 (B=8 4945 -> 5015 us). On the clean source the paired loads keep STACK:96 but are still slower at B=8
  (4945 -> 4975): dropped for that reason. Note for the tools: adding `TM()` stamps in a hot phase can change the
  register allocation of the whole kernel; check STACK when a profile build looks slower than the plain one.

## s0011 (5.1 h): 3314 / 5599 us = 1.421x / 1.179x SGLang (predicted ~3315 / ~5600)
- Counter-free strip hand-off in the flat GEMV (see the hypothesis): B=8 -1.1% over HTTP, bitwise identical results.
- Dropped: the finished blocks' query dots split over 2-4 warps per (source, query) pair (shorter per-thread loop):
  B=1 3267 -> 3380 us, B=8 4866 -> 4935. With all 24 warps of the team blocks busy in the pre-wait part, the threads
  that take the arriving cross-card data run slower (SM throughput is shared); 8 busy warps leave them room.

## s0012 (5.4 h): 3299 / 5558 us = 1.428x / 1.188x SGLang (predicted ~3300 / ~5560)
- Dropped after measuring: the finished blocks' query dots computed by idle blocks one phase ahead (global copy of
  the finished blocks, tagged results fetched by the team). It works (no dot work left in the team: its first
  thread has the cross-card data after 1.0 us instead of 3.1) but the phase is not shorter: the block barrier of
  the first reduction then waits 3.5 us for the threads whose o_proj rows arrive last (end of post 9.5 -> 10.0 us,
  with STACK:112). The dots were hiding the wait for the o_proj tail + NVLink, not causing it.
- Timing experiment (dummy stores, not kept): doubling the remote (value, tag) stores of the o_proj epilogue adds
  0.33 us (B=1, +8 store instructions) / 0.46 us (B=8, +16): a remote relaxed store is ~30-40 ns; the wait of the
  post phase for the o_proj data is the NVLink latency plus the tail of the slowest o_proj block, not the stores.

## State at 5.5 h (engine = s0012: 3299 us at B=1, 5558 us at B=8 over HTTP; 1.428x / 1.188x SGLang)
- From the inherited s6/s0003 (5324 / 9310) and s6/s0005 (this run's s0001: 4807 / 8861):
  s0002 team vector phases 4163 / 7125; s0003 GEMV scheduling 3984 / 6537; s0004 MMA attention + tagged hand-offs
  3891 / 6335; s0005 shared memory -> L1 3646 / 5870; s0006 unroll 3661 / 5742; s0007 whole w13 strips per block at
  B=1 3440 / 5773; s0008 small attention chunks 3401 / 5758; s0009 gate split 3310 / 5716; s0010 gate split with
  24 free blocks 3322 / 5660; s0011 counter-free strip hand-off 3314 / 5599; s0012 attention combiner 3299 / 5558.
- Where a B=1 layer (66 us) goes now: three cross-card exchanges (NVLink ~1.3 us each plus the tail of the
  producing GEMV), five grid barriers (1.27 us each), the team phases' three reductions each (~1.1 us), and GEMVs
  that are instruction-bound at ~2.4 G warp-instructions/s per SM (w13 at B=1: 41472 units x 46 instructions =
  5.2 us of the 8.1). A B=8 layer adds ~0.75 us (w13) + ~0.4 us (w2) per distinct expert.
- Ideas examined at the end and not pursued, with the reason: w13-input quantization beside the router (the
  fragment would need a barrier or L1-bypassing loads of 285 MB per layer at B=8); router started before the end
  of post-attention (the team's fence or relaxed loads of the whole input cost more than the 1.5 us gained);
  two-stage block reductions (the reducing warp runs alone at ~5 ns per instruction: 0.15 us); output-side scaling
  of the FP4 MMAs at B=1 (13 instead of 10 instructions per MMA).

## 5.5-5.6 h: register allocation as a knob (not yet submitted when the machine was lost)
- 896 / 1024-thread blocks on the final engine: REG:72 / REG:64 with STACK:112 / 136, slower at both batch sizes
  (B=8 4858 -> 4961 / 5125, B=1 3270 -> 3412 / 3520).
- Compiling phase bodies out of line (`__noinline__`) changes what the GEMV loops get (fixed tokens, B=1 / B=8,
  baseline 3270 / 4858): attention only 3300 / 4843; routing only 3267 / 4802; team phases only 3297 / 4837
  (STACK:104); w2 quantization only 3275 / 4858; attention + routing + w2 quantization 3261 / 4770 (STACK:96,
  bitwise identical results); all phases 3292 / 4788. Kept: attention + routing + w2 quantization.
  The combinations are not additive: this is the compiler's register allocation in one 19 000-instruction
  kernel, so each candidate has to be measured.
- Machine lost (epoch 1 -> 2) during the next A/B (GEMV schedulers out of line); the clock stopped at 5.59 h.

## s0013 (5.8 h, machine epoch 2): 3296 / 5484 us = 1.429x / 1.204x SGLang (predicted ~3290 / ~5460)
- Attention, routing and w2-quantization phase bodies compiled out of line: results bitwise identical to s0012,
  B=8 -1.3% over HTTP (5558 -> 5484), B=1 unchanged within the trial spread (3299 -> 3296). The prediction for B=8
  (-1.8%, from the fixed-token step 4858 -> 4770) was a little optimistic: generated text has more distinct experts
  per layer, so the fixed phases are a smaller share.

## 5.8-7.9 h: the expert GEMVs through tcgen05 (tensor core, 5th generation)
- Closed first: GEMV schedulers out of line (`gemv`, `gemv_blk` as `__noinline__`): STACK 176-360, B=8 4766 -> 4893-5604;
  `team_final` out of line: no change. The register-allocation knob is used up.
- Why tcgen05: the NVFP4 GEMV of s0001-s0013 (cvt + HMUL2 + m16n8k16 HMMA) needs 44-46 instructions per 576 B and is
  bound by the SM's instruction rate (~4.3 TB/s per card at B=8, 0.69 us per distinct expert and layer for w13).
  sm_100a has no warp-level block-scaled MMA, but `tcgen05.mma.kind::mxf4nvf4.block_scale.scale_vec::4X` takes NVFP4
  as the checkpoint stores it: 128 rows x 64 k of packed E2M1 codes and their UE4M3 block scales (one per 16 k), the
  B operand likewise (E2M1 codes and E4M3 block scales of the quantized activations, 8 columns = 8 sequences), one
  instruction issued by one thread, operands in shared memory, accumulator in tensor memory. No run of this project
  had tried it. Sources used: the PTX ISA text (tcgen05 chapter) and CUTLASS's sm100 headers for the layouts.
- tools/bench/tc5.cu (one card): stage 1, one MMA against the exact sum: 1024 of 1024 values equal (the layouts:
  codes K-major without swizzle, two codes per byte low nibble first; scale-factor atom of 512 B = 32 lanes x 16 B,
  row m at (m%32)*16 + (m/32)*4; B scale factors written with tcgen05.st, word 0 of lane n = column n; scale
  addresses must be 4-column aligned). Stage 2, the GEMV pattern (cp.async.bulk -> ring in shared memory -> one
  scale copy + one MMA per 4608-B unit): 151 MB cold in 23.4 us = 6.5 TB/s with four streams per block and a
  73.7 KB ring (the attention's staging area), 6.1 with two streams. Lessons on the way: the issuing thread is a
  serial path at ~8 ns per instruction (the first version with divisions in the loop ran at 1.5 TB/s); the cost is
  ~0.09 us per unit plus ~0.2 us per ring stage, so several streams per block; `tcgen05.cp` of the scale atom is free.
- In the engine (gemv_tc): the grid's units are split evenly over the blocks, a block cuts its range into sub-items of
  <= 9 units dealt to four streams (producer thread, consumer thread, two accumulators each); warps 0-3 read the
  accumulators in sub-item order and add D x scale into registers; a row group that spans blocks is finished by the
  block where it starts (partials as tagged words, added in block order). Results are independent of the slot
  (t_dup: copies of a sequence in one batch are bitwise equal) and deterministic.
- What went wrong, in order (each cost 10-30 min):
  1. Hang: the mbarriers were initialised again in every GEMV; arrivals got lost. Now once per launch, with phase
     counters in shared memory (`fence.mbarrier_init.release.cluster` alone did not fix it).
  2. All later phases looked 20-25% slower at B=8: the weights streamed by cp.async.bulk stayed in the L2 and pushed
     out the step's working data (attention, post-attention, quantization phases +1.2 to +2.8 us each). With
     `L2::cache_hint` = evict_first on the bulk copies they are back at their old times. (The old loads, `ld.global.nc`
     with L1::no_allocate, never showed this.) Part of the apparent slowdown was not real: two builds with different
     rounding generate different text, hence different expert unions; t_perf now prints the union per layer kind.
  3. A call to an out-of-line function from the kernel body (`tc_prep13`) cost the MXFP8 GEMV loops registers
     (B=1 w13 10.4 -> 13.1 us); called from inside `phase_route` it is free. In the end the tensor-core path got its
     own kernel (`ink_step_tc`, launched from batch 4): compiled into one kernel its branches cost B=1 1.4%.
  4. The B scale factors (72 tcgen05.st per phase) took 1.5-3 us before the first MMA: now 9 stores of 32 columns,
     issued by warps 20-23 beside the routing phase.
- Measured (fixed tokens, per layer, B=8): NVFP4 w13 0.59 us per distinct expert (old 0.69); a phase is ~0.9 us of
  set-up, the stream at 6.3-7 TB/s, ~2 us of hand-off at the end. At B=1 the fixed costs lose (9.5 us against 7.7,
  and the w2-input quantization waits 1.3 us longer for the last row groups), hence the two layouts: batches 1-3 run
  the unchanged kernel on the fragment layout, batches >= 4 the tensor-core kernel on a second copy of the same
  bytes (+46 GB per card, 145.8 of 198 GB used).
- Rehearsal over HTTP: B=1 3299 us (unchanged), B=8 5650 -> 5247 us (-7%).
- Checks: tools/t_tc_layout.py (all 40 NVFP4 layers x 4 card slices, 183.5 GB of codes and scales: the layout
  inverted equals the checkpoint bit for bit; largest block-scale byte 0x7e, so UE4M3 = E4M3 on every scale),
  t_gate with the tensor-core kernel forced at B=1/2 (rel median 0.0497, max 0.094, top-1 100%), t_gate8 (the two
  reference sequences in a batch with six random ones, large union: 0.0506 / 0.1023 / 100%), t_dup, t_perm,
  t_det (B=8 1100 steps, B=5 400, B=1 300: bitwise repeatable), rehearsal (online tokens = step tokens, 100%).

## s0014 (7.4 h): FAILED in the timing stage: deadlock (predicted B=8 ~5150, B=1 ~3296)
- Every gate of the check stage passed with the tensor-core kernel at batch 8 (rel L2 batched median 0.079 / max
  0.346 against limits 0.179 / 0.707; long positions 0.059 / 0.48; generate == step-by-step for 8 of 8), so the
  arithmetic is right. In the timing stage the engine stopped: no call returned for 1200 s, the runner's main thread
  in `torch.cuda.synchronize()` after a reset and 63 prompt steps (the W1-like generate() segment). About 5000
  batch-8 steps had run in that evaluation before it; my own checks (t_det 2 x 1100 steps, rehearsal ~4000 steps)
  never stalled: a rare race in the new kernel, not a property of an input.
- One race found by reading while the evaluation hung: the four read-out warps of gemv_tc take the barrier phase
  counts from shared memory in every iteration, and warp 0 advanced the counts for the next GEMV as soon as IT left
  the loop; a warp still one iteration behind would then wait for the wrong phase forever. Fixed with a barrier
  before the update. Whether that is the race that hit is checked with a soak test (tools/t_soak.py: batch-8 steps
  without host synchronisation, a stall watch that prints the kernel's wait codes).
- Lesson for the checks: determinism tests of a few thousand steps do not show a 1-in-10^4 deadlock; anything that
  adds a wait protocol needs a soak run several times longer than the evaluation before it is submitted.

## s0015 (8.65 h): valid, 3300 / 5552 us = 1.428x / 1.189x: no gain (predicted B=8 ~5200 if it did not stall)
- It did not stall (the s0014 deadlock stays unexplained: 250 000 local steps of the evaluator's call patterns on the
  s0014 source, batch 8 alone and batch 1 then 8, never stalled; the race I closed cannot be shown to be the cause).
- The prediction failed, and the reason is in my own measurement: the evaluation's in-process figures at B=8 are those
  of s0013 (step 5528 vs 5522 us, W1-like 5760 vs 5756, generate 5508 vs 5533; HTTP 5552 vs 5484 is the trial
  spread of the B=8 figure, s0012 had 5558). The -7% of my rehearsal (5650 -> 5238 us online, 5716 -> 5364 W1-like)
  came from the generated text: the two builds round differently, their greedy continuations of the same random
  prompt diverge, and the tensor-core build's text happened to use fewer distinct experts. The rehearsal's own
  teacher-forced figure had not moved (4755 -> 4757 us), nor had tools/t_perf.py (4762 -> 4764): I read those as
  "different unions in the profile pass" and trusted the generated-text figures. They were the ones that were wrong.
- What the tensor-core w13 path really does at B=8 (fixed tokens, per layer, ~37 distinct experts): w13 26.2 -> 21.8 us
  (0.69 -> 0.59 us per expert), but w2-input quantization +1.2 us (the last row groups arrive later than the flat
  schedule's last strips), routing +0.6 us (the scale-factor stores), and the MXFP8 layers of the tensor-core kernel
  are slower per expert (w13 +7%, w2 +14% at different unions: not separated from noise). Net ~1% of the step,
  inside the trial spread.
- Rule for A/B from now on: a build that changes rounding is compared on teacher-forced tokens only (t_perf "step",
  rehearse "step"), per distinct expert; generated-text figures (rehearse W1-like / online) compare workloads, not
  engines.
- Kept from this: the engine is valid with the tensor-core kernel at batch 8 (gates as s0013: batched rel L2 median
  0.079), +46 GB per card. Best submission remains s0013.

## s0016 (9.1 h): valid, 3300 / 5413 us = 1.428x / 1.220x (predicted step_us at B=8 ~5450, HTTP ~5450, B=1 ~3296)
- The tensor-core kernel's calls (set-up, release, the w13 GEMV) made from inside the routing phase function instead of
  the kernel body: its MXFP8 layers are back near the plain kernel's times. Measured in the evaluation, in process at
  B=8: step 5528 -> 5433 us (-1.7%; s0013: 5522), W1-like 5760 -> 5670, generate 5508 -> 5417; over HTTP 5552 -> 5413
  (s0013: 5484). B=1 unchanged (3300 vs 3296: the plain kernel, trial spread). Prediction confirmed.
- This is the first submission where the tensor-core path pays: -1.6% at B=8 against s0013, at +46 GB per card.
- How it was measured before submitting (and how every rounding-changing build should be): tools/t_replay.py, one
  generated text replayed teacher-forced through both kernels in one process: 5467 -> 5393 us per step (-1.4%),
  repeatable to 1 us.
- Where the rest of the w13 gain goes (per NVFP4 layer at ~45 distinct experts): w13 itself -4.5 us; post-attention
  +0.8 us (the code/scale byte stores of the eight tokens' blocks share cache lines), quantization +0.2, MXFP8 w13
  still +3-4% per expert in this kernel.

## 9.0-9.5 h: three extensions of the tensor-core path, measured and not adopted (engine/ is the s0016 source again)
All three were measured with tools/t_replay.py (one generated text, both kernels, repeatable to 1 us; s0016:
tensor-core / fragment = 0.9864-0.9871). Their source is kept in prev/exp_tcmx/ (git commit 61c4f14).
- w2 through the same path (`TC2_ON`): works after one loader fix (with a second w2 layout the load ran out of
  memory through allocator fragmentation: `torch.cuda.empty_cache()` per layer; 168.7 GB per card), logits within
  0.01-0.10 rel L2 of the w13-only build. Not faster: at 49 distinct experts the phase takes 16.3-22 us against 19 us
  on the fragment path. A w2 row group sums over all union experts, so every row group is split over ~4 blocks and
  every block both sends and collects partials (the collecting blocks end up to 6 us after the last MMA: 16 serial
  relaxed loads), and the per-expert B scale factors cost 3.6 us of tensor-memory stores before the first MMA.
  The fragment path's whole-strip-per-block schedule has no hand-off at all. Would need: partials summed by the
  consumer, or a schedule in which a block owns whole row groups (then only 36 blocks work). Not pursued.
- MXFP8 w13 through `kind::mxf8f6f4` (`TCMX_ON`): tools/bench/tc5.cu stage 1b confirms the kind (UE8M0 scale byte j
  of the atom = scale id j; 1.1e-7 relative to the exact sum); in the engine it passes t_gate (0.0497 / 0.0991) and
  t_gate8, and tools/t_tc_layout.py verifies the layout on all 8 MXFP8 layers (67.3 GB, scale exponent bytes
  111..118, no NaN codes). Gain: 1.09 -> 1.00 us per distinct expert only, because the MXFP8 fragment path already
  runs near the memory rate (5.0 TB/s; half the instructions per byte of NVFP4). On the replay 5393 -> 5395 us:
  the gain is cancelled by 18 KB more shared memory for the 8-bit B operand (see below). +17 GB per card. Dropped.
- One 16-byte store per warp for the w13 codes instead of 16 byte stores (the eight tokens' pieces share a line):
  bitwise the same results, 0.9933 against 0.9920 with byte stores in the same build: nothing.
- Measured on the way: the tensor-core kernel with a 36.9 KB instead of an 18.4 KB B operand area (173 KB instead of
  155 KB of shared memory) loses its whole gain (0.9864 -> 0.9985): the L1 knee of finding F5 again. And the
  generalised stream loop (runtime unit size and units per stage for the two formats) cost 0.6% by itself: the
  consumer thread is a serial path, three or four more instructions per unit show.
- State of the tensor-core idea at the end of the run: one matrix (NVFP4 w13), one batch range (>= 4), -1.6% at B=8
  in the evaluation (s0016), +46 GB per card, and one unexplained deadlock in four evaluations of that kernel
  (s0014; s0015, s0016 and 250 000 local steps clean). The per-byte advantage is real (0.59 against 0.69 us per
  expert) but the phase has ~4 us of fixed cost at B=8 and the rest of a layer does not get faster.

## 9.6-9.9 h: the s0014 deadlock is reproduced and explained
- A last sweep on the s0016 base: the B operand inside the attention's staging area (shared memory -18 KB) with one
  unit per ring slot: 5393 -> 5575 us (the stream slows by more than anything the L1 gets back; the plain kernel
  at 137 instead of 155 KB does not change: 5468). The s0016 configuration (4 streams x 2 slots x 2 units, 155 KB)
  stays.
- The deadlock: all my soak runs had used eight different random sequences, i.e. 35-50 distinct experts per layer.
  The evaluation stalled in a prompt phase, and real prompts share their first tokens: few distinct experts.
  tools/t_soak.py --distinct 1|2|3 (eight slots filled with 1-3 different sequences, steps without host
  synchronisation) stalls the s0014 source after 693, 819 and 5040 steps. The same source with the one line of
  s0015 (a barrier of the four read-out warps before the mbarrier phase counts are advanced) runs 60 000 such
  steps, and the engine of s0016 72 000. So it was the race found by reading: with few experts a block has few
  sub-items, the read-out warps leave their loop at different times, and a late warp read a phase count that warp 0
  had already advanced for the next GEMV, then waited for a phase that never completes.
  My argument at the time that this race was "practically impossible at batch 8" was wrong because it assumed
  35-50 experts per layer; the test that would have caught it before s0014 is a one-option change of the soak.
- Checks added to the list for anything with a wait protocol: t_soak with --distinct 1, 2, 3 and with all-different
  sequences, >= 20 000 steps each.

## s0017 (10.4 h): valid, 3298 / 5420 us = 1.429x / 1.218x (predicted B=1 ~3296, step_us ~3248; B=8 ~5413 +- 40)
- The progress pointer moved from the per-launch header (a kernel parameter by value) into Ctx: the plain kernel is
  back at STACK:96. Bitwise the results of s0016. In process at B=1: step 3254.5 -> 3250.1 us (s0013: 3247.7), HTTP
  3300.1 -> 3297.7; B=8 unchanged within the spread (step 5433 -> 5440, HTTP 5413 -> 5420). As predicted.
- engine/ is this source. Against s0013 (the best headline figure, 3295.6 / 5484): B=1 equal within 2 us, B=8 -1.2%.

## s0018 (10.9 h): valid, 3298 / 5440 us = 1.428x / 1.214x (predicted step_us at B=8 5440 -> ~5425, HTTP ~5405 +- 40)
- Sub-items of 12 units instead of 9 in the tensor-core w13 read-out. In process at B=8: step 5440.2 -> 5433.5 us
  (-0.12%; the replay had said -0.27%), HTTP 5420 -> 5440 (trial spread). B=1 unchanged (3297.7 -> 3298.4).
  Half of the predicted effect: the replay text and the evaluation's sequences do not have the same union sizes, and
  the knob trades read-out events against balance, which depends on them.
- The three submissions s0016-s0018 are the same engine within the B=8 trial spread: 5413, 5420, 5440 us over HTTP
  (in process 5433, 5440, 5434).

## State at the end (11 h; engine/ = s0018)
- Board: best headline figure s0013 (3295.6 us at B=1, 5484 at B=8). Engine in the workspace: s0018, 3298 / 5440 us
  (s0016: 3300 / 5413, s0017: 3298 / 5420) = 1.43x / 1.21-1.22x SGLang's 4711 / 6604; milestones build, correct,
  device_loop, gpu_never_waits, parity, h0_step. Track A throughout.
- This run, in order of what paid (HTTP figures, B=1 / B=8):
  s0001 4807 / 8861 (inherited source) -> s0002 team vector phases 4163 / 7125 -> s0003 GEMV scheduling 3984 / 6537
  -> s0004 MMA attention, tagged hand-offs 3891 / 6335 -> s0005 shared memory back to L1 3646 / 5870
  -> s0007-s0012 placement of work in idle blocks, counter-free hand-offs 3299 / 5558 -> s0013 phases out of line
  3296 / 5484 -> s0016-s0018 tcgen05 for the NVFP4 w13 at batch >= 4: 3298 / 5413-5440.
- What the second half of the run (5.8-11 h, tcgen05) bought: -1.2% at B=8 over HTTP for +46 GB per card and a
  second kernel. The mechanism is sound (the checkpoint's NVFP4 blocks are the tensor core's native operands; the
  w13 stream runs at 6-7 TB/s instead of 4.3) but a GEMV phase is short: ~4 us of fixed cost per phase, the
  quantization phase that follows waits for the last row groups, and only one of the four expert GEMV kinds
  (NVFP4 w13) profits; w2 and the MXFP8 layers were built, measured and left out (journal 9.0-9.5 h).
- What the detour cost and taught: one failed submission (s0014, a race that only shows with few distinct experts),
  one submission without gain (s0015, measured on generated text instead of a replay), and two rules that are now
  in the check list: soak with --distinct 1/2/3, and A/B by tools/t_replay.py whenever rounding changes.
- Where a step goes now (B=8, ~45 distinct experts per layer, per layer): attention input 12, q/k/v 6, attention 10,
  o_proj 3 + wait 3, post-attention 9, router 3, routing 4.5, w13 24 (NVFP4) / 50 (MXFP8), quantization 6.5,
  w2 17.5 / 26: the fixed phases are ~55 us of ~110. The next step for B=8 is not a faster GEMV but overlapping
  the fixed phases of half the batch with the expert GEMVs of the other half (plan.md, "Open").

## 11.0 h: two last measurements for the record (no change to engine/)
- Where the tensor-core kernel breaks even (tools/t_replay.py --batch B with MK_TC_MINB=2: one generated text,
  contexts 256-512, fragment kernel against tensor-core kernel, us per step):
  B=2 3593 / 3650 (+1.6%), B=3 3923 / 3900 (-0.6%), B=4 4377 / 4327 (-1.2%), B=6 4932 / 4866 (-1.3%),
  B=8 5467 / 5393 (-1.4%). The threshold TC_MINB = 4 is on the safe side; 3 would already pay.
- `MK_CFLAGS=-DNO_TC13` (the single-kernel, single-layout engine, i.e. s0013's arithmetic at every batch size):
  builds, 99.9 GB per card, t_gate 0.0503 / 0.1069 / 100%, batch 1 bitwise as the default build; fixed-token step at
  B=8 4742 us against 4686 with the tensor-core kernel (tools/t_b1time.py --batch 8).

## 11.1-11.2 h: one more placement fix, submitted as s0019; machine lost during its evaluation
- The w13 scale-factor stores (tensor memory, beside the routing phase) spread over three groups of four idle warps
  (12-23) instead of warps 20-23 alone: bitwise the results of s0018; replay at B=8 5383 -> 5345 us per step (-0.7%):
  the routing phase had been waiting for the nine stores. Soaks with 1, 2, 8 distinct sequences clean.
- The machine was replaced while s0019 was evaluated (clock stopped at 11.23 h); the evaluation is queued for the new machine.

## 11.23 h (clock stopped): s0019 still not evaluated, no machine for more than three hours
- The machine lost at 04:27 has not been replaced by 07:48 (the status went from epoch 3 to epoch 4, both
  "starting"); s0019 is queued and will be evaluated when a machine comes. Nothing was started, stopped or worked
  around on my side.
- engine/ is the s0019 source (git 'tensor-core w13: scale-factor stores ... three warp groups'). It is bitwise
  s0018 in its results; the only difference is which idle warps issue nine tensor-memory stores.
  Prediction on record: in-process step_us at B=8 5434 -> ~5395, HTTP B=8 ~5400 +- 40, B=1 ~3298.
- If s0019 comes back valid: nothing to do but record it (engine/ already is that source). If it does not:
  `git checkout 1ccd66b -- engine/` restores the s0018 source (valid, 3298 / 5440 us).

## s0019 (11.3 h, machine epoch 6): valid, 3307 / 5390 us = 1.424x / 1.225x (predicted step_us at B=8 ~5395, HTTP ~5400 +- 40, B=1 ~3298)
- Evaluated after the machine came back (4.5 h of outage, not counted). In process at B=8: step 5433.5 -> 5392.4 us
  (-0.76%, as the replay said), W1-like 5652 -> 5612, generate 5443 -> 5403; over HTTP 5440 -> 5390, the best B=8
  figure of the run. B=1 on this machine: step 3257 against 3250 on the previous machine with the same plain kernel (HTTP
  3307 against 3298): machine-to-machine spread of ~0.3%, not the change (the B=1 kernel does not contain it).

## s0020 (11.5 h): valid, 3307 / 5378 us = 1.424x / 1.228x (predicted step_us at B=8 ~5380, HTTP ~5380 +- 40)
- The w13 quantization emit of the tensor-core kernel inlined in its team phase (it was out of line from the time
  both paths shared a kernel). Bitwise the results of s0019. In process at B=8: step 5392.4 -> 5380.9 us (-0.21%, the
  replay had said -0.22%), HTTP 5390 -> 5378: the best B=8 figure of the run. B=1 unchanged (3307.3 -> 3307.5 on
  this machine; the plain kernel).

## End of the run (11.5 h; engine/ = s0020)
- Final engine: s0020, 3307 us per token at B=1 and 5378 us per step at B=8 over HTTP on the last machine (1.424x /
  1.228x SGLang's 4711 / 6604); the same B=1 kernel measured 3296-3300 us on the previous machine (s0013, s0016-s0018).
  Best headline figure on the board: s0013, 3295.6 us.
- B=8 since s0013 (in-process step_us, the stable figure): 5522 (s0013) -> 5433 (s0016, tcgen05 w13 with its calls
  out of the kernel body) -> 5434 (s0018, sub-items of 12) -> 5392 (s0019, scale-factor stores over three warp
  groups) -> 5381 (s0020, emit inlined): -2.6%. Every step after s0015 was predicted from tools/t_replay.py to
  within 0.2% of the step, except s0018 (predicted -0.27%, measured -0.12%).
- Submissions of the run: 20, of which 19 valid, 1 failed (s0014, deadlock; explained and fixed).

## 11.6 h: two last knobs measured on the replay, none adopted (engine/ stays s0020: 5334 us per step on that text)
- The scale-factor stores over four warp groups (warps 8-23, 2 + 2 + 2 + 3 stores) instead of three: 5366 us
  (+0.6%: warps 8-11 are not as idle in the routing phase as 12-23; three groups stay).
- Without the progress words (-DTC_NOPROG): 5333 us: they cost nothing, they stay.

## 11.8 h: the final engine (s0020 source) checked once more on the last machine
- Float64 reference regenerated on this machine (tools/ref_dump.py, 2 sequences x 12 positions): tensor-core kernel
  forced at B=1/2 (MK_TC_MINB=1) rel median 0.0525 / max 0.0958 / top-1 100%; inside a batch of 8 with a large
  union (t_gate8) 0.0507 / 0.0927 / 100%.
- Soaks, 12 000 steps each without host synchronisation: B=4 (1 distinct sequence), B=5 (2), B=6 (6), B=7 (3),
  B=8 (3), and before the submission B=8 with 1, 2 and 8 distinct (20 000 each): no stall. The evaluator's
  timing-stage call pattern at batch 1 then 8, twice: no stall.

# Run s8-inkling2s-b200x4-on1 (online: serving a production trace), journal

## 0 - 2.0 agent-hours: from the decode megakernel to a pass engine
- s0001 (inherited engine, unchanged): offline valid (3260 us B=1 step in process), stops at `online_api` as expected.
- Decision 1: the megakernel cannot be the serving path. It holds 8 rows at one shared position in dense K/V arrays,
  its weights are split TP4 over the cards by the experts' intermediate dimension, and every phase is built around
  B <= 8. The trace needs ~10-14k prompt tokens/s (189 requests, 1.7 M prompt tokens in 120 s, 30% shared), i.e.
  passes of hundreds to thousands of rows. The evaluator stops a replay 90 s after the last arrival, so even the
  `online` rung needs >= ~8k prompt tokens/s: there is no "slow but valid" route through one token per step.
- Decision 2: the same arithmetic must serve step() and submit(). s0001's free-run continuations match the reference's
  for 12.6% of their length (0/8 sequences): two paths with different rounding diverge within ~8 tokens on these
  prompts, and `online_vs_step` wants 75% of sequences identical. So the offline API runs on the pass engine too and
  the megakernel is retired from the default build (kept: MK_ENGINE=megakernel, engine/mk_api.py + ink.cu,
  MK_BUILD_MEGAKERNEL=1 in build.sh). Cost: steady decode (see s0002).
- Design (engine/pe.cu, engine/api.py): a pass = T rows (row = one position of one sequence) through 48 layers.
  Per card: attention TP by heads (12 q / 2 KV heads), whole experts EP (96 contiguous experts per card), router,
  o_proj, residual mixes, short convolutions and norms replicated (bitwise identical on the four cards: checked).
  Two exchanges per layer (attention outputs [T][1536] bf16, expert pair outputs), read through peer pointers after a
  per-pass flag (st.release.sys / ld.acquire.sys); nothing waits for the host inside a pass. K/V in pages of 512
  positions x 48 layers (25 MB per card), operand-ordered for the attention MMAs; a snapshot of the four short
  convolutions' state per page makes a page a resume point; prefix cache = hash chain over 512-token blocks, LRU
  eviction of unreferenced leaves.
- Canonical arithmetic (what makes `schedule` and `online_vs_step` hold): GEMM columns are independent; attention
  = per (16 consecutive rows, head, key page) one warp, online softmax inside the page, pages folded in order;
  expert outputs summed in expert order; nothing depends on the composition of a pass. Measured: chunked (40 rows) ==
  step by step == chunks of 7, bitwise, 6 layers (tools/t_pe_ref.py), and in the evaluator's rehearsal schedule 6/6,
  online_vs_step 8/8 + 8/8, batch gate bitwise_equal true.
- Verification against the reference (tools/t_pe_layer.py, one layer on a random residual state, the reference's
  own attention() and moe() on the engine's inputs): attention + o_proj rel err 0.0008-0.002 (local, global, 1500
  rows so windows and the 1024 bias extent are exercised), NVFP4 MoE median 0.003 (max 0.09: one E2M1 flip of the
  second quantization moves a row by percents), MXFP8 MoE median 4e-5, 99% of elements bitwise equal.
- First kernels were m16n8k16 GEMMs: correct, 3.9k tok/s (2048 rows in 523 ms). Staging the X tile in shared memory:
  326 ms. An L2 prefetch (cp.async.bulk.prefetch) made decode passes slower (27 -> 42 ms) and was dropped.
- All GEMMs through tcgen05 (bf16 kind::f16 K=16, NVFP4 kind::mxf4nvf4 K=64 with block scales, MXFP8
  kind::mxf8f6f4 K=32): worked on the first build with the layouts of findings F9; 2048 rows in 139 ms (14.7k tok/s).
  The NVFP4 results are bitwise those of the m16n8k16 path: the sums are exact in FP32.
- Measured with nsys on the tcgen05 build: a block's K-step chain runs at ~0.3-0.39 us per K step whatever the pass
  size (router 3 row groups: 103 us; o_proj: 300 us at 2048 rows, 116 us at 1 row): it is bound by what one SM can
  pull into shared memory (~50 GB/s, 21 KB per stage), not by the MMAs. So: more work per byte moved (128-column
  tiles, 3 row groups per block share the B unit) and a deeper ring for small passes.
- Peer memory from kernels (tools/bench/p2p.cu, one pair): loads 560-725 GB/s, stores 565 GB/s, cudaMemcpyPeerAsync
  563-737 GB/s at 38-226 MB; a single remote load costs ~2 us and a thread's loads are serial. The combine kernel's
  time in a pass is mostly the wait for the slowest card: with the trace's prompts (100 distinct tokens) a layer
  uses ~120 of 384 experts, one expert takes up to 2010 of 2048 rows, and the cards' pair counts differ by 1.5x
  (median over layers, worst 2.5x).

## 2.0 - 2.4 agent-hours: first valid submissions
- s0002 (pass engine, FIFO, 2048-row passes): expected valid with share 0.0-0.1; measured **valid, 0.085**
  (16 of 189), TTFT p50 5.5 s / p99 15.5 s, tbt p50 114 ms / p99 181 ms, longest gap 213 ms; schedule 12/12,
  online_vs_step 8/8 + 8/8, kv_conservation. Offline on the pass engine: rel L2 0.074 / 0.086 (limit 0.179), top-1
  92.6% (limit 90.5%), long 98.6% (limit 97.6%), batch bitwise equal, generate 8/8; step 30.0 ms B=1, 38.0 ms B=8
  (megakernel 3.3 / 5.4 ms: the cost of one arithmetic for both APIs, to be won back on small passes).
  Conclusion: the rung is reached; every decoding request rode 120-210 ms passes (mean gap > 100 ms), so almost
  nothing met the service levels.
- The trace (tools/trace400.jsonl, the replayed 120 s): arrivals come in bursts every 10 s: 11-22 requests at
  the same instant, 14k-209k new prompt tokens per burst after ideal prefix reuse (30% of 1.73 M). What counts is how
  many requests of a burst get a first token within 5 s while the decoding ones keep a mean gap under 100 ms.
- tools/sim.py: the engine's own admission / compose / cache code on a virtual clock with a pass time model
  (base + per row). It predicted 0.825 for the s0003 scheduler at the measured kernel speed; the real replay gave
  0.868 (the model's 3 ms host overhead was pessimistic). Predictions for faster kernels with the same scheduler:
  row cost 0.040 ms -> 0.89, 0.030 -> 0.94, 0.022 -> 0.96 (with the base cost falling alongside).
- s0003 (scheduler + 128-column GEMM tiles + bias GEMM): expected 0.85-0.88; measured **valid, 0.868** (164 of 189),
  TTFT p50 0.44 s / p99 18 s, tbt p50 74 / p99 81 ms, longest gap 95 ms. Above SGLang (0.825) on share, tbt p99 and
  stall; worse on TTFT p99 (2.2 s) and tbt p50 (10 ms). Offline step 20 / 26 ms.
  The 25 misses are all first tokens after 5 s (long prompts of the big bursts); no request misses on the gap.
- tcgen05 shapes (nsys, 2048 rows): a GEMM's time is its operand bytes over ~6.8 TB/s per card when at least 148
  block-units exist, and the unit count is a step function of the pass size (148 SMs): q/k/v 16 tiles x 11 chunks =
  176 units -> two rounds on 28 SMs. Passes of <= 1536 rows keep q/k/v and o_proj in one round.

## 2.4 - 3.5 agent-hours: s0004, s0005 and what the profiles said
- s0004 (pass length from the decoders' gap credit; packed single-row attention; argmax routing; 64-row MMAs and
  short ring slots for small passes): expected 0.88-0.90, measured **valid 0.889** (168 of 189), TTFT p50 0.37 s,
  tbt p99 92 ms. The simulator had said 0.873 -> 0.889.
- Things that did not work or did not matter:
  * 768 threads per row for small passes: reductions add in another order -> stepwise != chunked (max abs diff 10
    in the block state after 6 layers). One reduction order for every pass size is part of the canonical arithmetic.
  * Deeper ring / 64-row MMAs for small passes: the dense chains of a 1-row pass stayed at 40-53 us (q/k/v, o_proj).
    A chain is ~0.11 us per tensor-core operation issued by one thread (384 K steps of 16 for o_proj), not bytes:
    twice the blocks with half the rows each changes nothing. Shorter chains need the K range split over blocks,
    i.e. a different (segmented) summation for every pass size. Not done.
  * Page-major attention units (six heads of a KV head in one block to share K/V reads through the L1): no gain.
    Attention at long context: 16x8x16 MMAs are ~15 instruction slots each on this card (about 100 TFLOP/s per
    card for the legacy MMA) and the softmax is ~30 instructions per (query, key).
  * Cost of a decode row in a mixed pass is 0.08-0.2 ms, not the 0.45 ms of the synthetic test with random tokens:
    deferring decode rows during a burst would buy under 5%.
- What the mixed pass (1280 prompt rows + 30 decode rows, 80 ms at s0004) is made of, per card: expert GEMMs 18.8 ms
  (weight streams: ~61 tiles of 128 pairs per card, most of them one per active expert), expert-output exchange
  13.7 ms (228 us per layer: 72 MB of peer reads per card with all four cards reading), dense GEMMs 13.6 ms (one
  chain per block, ~105 us each for q/k/v and o_proj), attention 6.3 ms, row kernels ~12 ms.
- s0005: experts swapped between cards from a running pair count (bitwise-neutral), owner-sum exchange (a row's
  eight outputs summed on the card that holds most of them, fetched by the others), row kernels with 16-byte loads,
  cost model refit (16.9 ms + 0.037 per prompt row + 0.87e-6 x position; decode-only 18.8 + 0.37 per row). mk exec
  replay: 0.921 (174 of 189), probes 6/6 identical with 2620 swaps and 101 evictions.

## 3.5 - 5.3 agent-hours: s0006 - s0009: chains, shapes, and the serving loop on the device
- s0006 (one consumer thread per row group in a tcgen05 block; pass length = least credit, floor 45 ms): expected
  0.92-0.93, measured **valid 0.926**. A block's chain is ~0.11 us per tensor-core operation issued by one thread;
  an NVFP4 stage was 7 operations (scale copy of B, then scale copy + MMA per row group): three threads issue the row
  groups side by side (own scale columns, slot and done barriers count three arrivals): expert GEMMs 18.4 -> 15.7 ms
  per card in a 1280+30-row pass.
- s0007 (four row groups per block for the dense GEMMs of passes over 1536 rows): expected 0.93-0.94, measured
  **valid 0.937** (177 of 189), TTFT p50 0.27 s / p99 9.2 s, tbt p50 72 / p99 90 ms. 2048-row pass 96.5 -> 87.2 ms.
- Measured and dropped (source in prev/tc_attention): attention as score GEMM -> softmax kernel -> value GEMM on
  tcgen05. Correct at once, bitwise invariant, and slower: a softmax loop with one load per key is bound by load
  latency (~14 ns per (query, key)); the m16n8k16 kernel has its scores in registers. 2048 rows @4096: pass 103 -> 132
  ms.
- Measured and dropped: mix sources kept in registers between the two passes (spills: 62 -> 97 us per layer).
- Long context: 6000-token prompt against the fork-precision reference (tools/t_long.py): rel L2 of the centred
  logits at the last 16 positions median 0.052 / max 0.081, top-1 13 of 16 (the evaluator's own long gate for the
  inherited engine: 0.058 median); chunks of 2048 == chunks of 700 bitwise. tools/t_cache.py with 13 pages (91
  evictions, recomputation, late hits, in-flight dedup, page-boundary lengths): 12 of 12 identical to alone, in every
  order, conservation holds.
- The serving loop on the device (s0008, s0009):
  * engine/sched.h: the whole scheduler as allocation-free C++ (slot table, page free list, open-addressing prefix
    cache with LRU eviction, admission, late hits, in-flight dedup, deadline / shortest-first order, credit-based pass
    length, cost model). tools/sched_sim.cpp replays the trace with it on the host: 0.9259 and the same hit / late hit
    / eviction / insert / dedup counts as the Python scheduler (tools/sim.py): a line-by-line check before any GPU run.
  * k_sched (one thread on card 0) runs it at every pass boundary: tokens of the pass that just ran (the four cards'
    argmax candidates through peer memory) into a host-mapped token ring, arrivals from a host-mapped request ring
    (prompts are copied by the host into a device arena), the next pass's header and tables. k_wait on cards 1-3
    takes them over; two flags keep the cards in lockstep (card 0 composes pass n+1 only when the others have taken
    pass n).
  * The pass is a fixed program: every kernel is launched with the largest grid and leaves when the pass has no work
    for the block (1-row pass 15.0 -> 15.5 ms). The host enqueues it from one thread per card, two passes ahead, and
    never looks at what a pass contains. Host syncs per pass: 1 -> 0.
  * One bug on the way: the flags of the owner-sum exchange (index NL*2+8+layer) ran into the two new flags at the
    end of the array for layers 46 and 47: cards took tables at the wrong time, a decode row riding a prompt pass
    got wrong tokens. Found by comparing with the host scheduler on fixed grids (same program, right tokens).
  * s0008: expected 0.92-0.93 (no expert swaps yet), measured **valid 0.926**: every gate passes with the loop on
    the device. s0009 adds the swaps on the device (k_swap1 / k_swap2, decided by the scheduler from the pass's
    routing counts): replay 0.931.

## 5.3 - 6.3 agent-hours: s0010, attention on tensor cores
- Before choosing: where a 2048-row pass at position 4096 goes (nsys, per card, s0009 kernels; 97 ms under nsys):
  attention 23.3 ms (k_attn_seg 20.6), expert GEMMs 19.6, dense GEMMs 15.2, row kernels ~20, the three exchange
  kernels 19.
- What the exchange kernels really cost (they contain the wait for the slowest card): launching each of them twice
  (idempotent) adds gather_o +3.1 ms, osum +5.5, spread +3.1 per 86.9 ms pass: **12.6 ms of copies over the links**,
  the rest of their kernel time is waiting. An experiment that made them read local memory was useless: wrong data
  changes the routing (338 experts per layer instead of 118) and with it everything else.
- The GEMMs are near what the card can deliver for one-pass streams (q/k/v: 566 MB of operands per layer and card in
  113 us = 5 TB/s; 640-1040 TFLOP/s per card). The attention kernel was not (about 50 TFLOP/s): that was the target.
- New kernel `k_attn`: unit = (21 rows x 6 q heads, KV head, page). Per half page: scores = 8 tcgen05 MMAs (128 q x
  256 keys; the K page is stored as their B operand), each q's 256 scores are read from tensor memory twice (scale,
  bias, mask, max; then exp, bf16) and written as the A operand of 16 MMAs with the V page (stored as B operand).
  The relative-bias GEMM writes its table by key slot (position - distance mod 1024), so a q's biases for a half
  page are contiguous 16-byte loads. The two halves' outputs stay in tensor memory until the unit's record is written.
  * First version (4 softmax warps, thread = q): correct on the first build (single layers 0.001-0.003 against the
    reference, stepwise == chunked bitwise), **not faster**: 37 us per half page; an SM with four warps of register
    arithmetic runs ~1 instruction per ns. One block per SM (tensor memory: 512 columns) means the block itself has
    to bring the warps.
  * 16 softmax warps (warp = 32 q x 64 columns; maxima and sums exchanged through shared memory): 18 us per half
    page. Attention 20.6 -> 8.1 ms per card and pass; local layer 278 -> 121 us, global layer at 4-6k context
    1190 -> 404 us. The bias GEMM got a per-column position cache (its epilogue had gone from 57 to 132 us with a
    table lookup and a division per element; back to 61).
  * 2048-row pass: 86.6 -> 80.2 ms at position 0, 101.6 -> 85.1 at 8k, 137.8 -> 99.2 at 30k. Decode passes unchanged
    (15.1 ms at B=1), B=8 at position 9000 23.4 -> 21.4 ms.
- Cost model refit: 21.8 ms + 0.0285 per prompt row + 0.30e-6 per row and position + 0.05 per decode row; decode-only
  17.1 + 0.355 per row. Nearly every prompt pass of the replay now hits the 2048-row cap at ~83 ms.
- Pass time by rows is a step function (1024: 51.3, 1536: 64.2, 2048: 79.9, 2560: 103.4, 3072: 117.3, 4096: 145.9 ms):
  above 2048 rows q/k/v and o_proj need a second round of blocks. With these numbers in the simulator a larger cap
  buys nothing (0.947 at 2048 and 3072, 0.952-0.963 at 4096 with idle passes of 140-180 ms); with a linear model it
  had promised +4 requests. Not done.
- **s0010: expected 0.95-0.965, measured valid 0.958** (181 of 189), TTFT p50 254 ms / p99 7.5 s, time between
  tokens p50 46 ms / p99 87 ms, longest gap 112 ms. Offline: rel L2 0.073 / 0.080, step 15.05 ms (B=1) / 20.1 ms (B=8).
  Against SGLang (0.825; 0.6 s / 2.2 s; 10 / 788 ms; 2.4 s): better on share, TTFT p50, tbt p99, stall; worse on
  TTFT p99 and tbt p50.
- Seen in the result: submit() takes up to 72 ms on the driver's thread (driver lateness 95 ms): the prompt copy into
  the device arena competes with the launch threads. To look at.

## 6.3 - 6.8 agent-hours: s0011, what a tcgen05 chain's time is made of
- Hypothesis from the numbers of every chain so far (~0.15 us per bulk copy whatever its size): a chain is bound by
  the *count* of cp.async.bulk copies its one producer thread issues, not only by bytes. Test: dense GEMMs with one
  K step per stage instead of two (5 copies per K step instead of 2.5): 2048-row pass 80.3 -> 85.1 ms. Half true:
  dense time = 1.9 ms per (copy per K step) + 7.6 ms for the rest.
- Stages made larger (K steps per stage dense / experts; 2048-row pass): (2,1) 80.3 ms, (2,2) 75.3, (2,2) with whole B
  units for partial expert tiles 75.1, (4,2) 73.6, (4,4) 73.9, (3,2) 72.5-73.3 with other ring sizes: taken (4,2), ring
  216 KB. The same order of operations: bitwise the same results.
- Decode passes use the dense setting too: 15.1 -> 13.4 ms at 1 row, 20.2 -> 17.8 at 8, 26.3 -> 20.4 at 32.
- Why 2048 rows stays the cap: tensor memory is 128 x 512 cells per SM, so one round of blocks covers at most
  148 x 65536 output cells: o_proj (4608 outputs) 2105 rows, q/k/v (3840) 2526 rows. Above that some SMs run a second
  unit (2304 rows: about +5 ms for 256 rows, no gain in rows per second).
- Cost model: 19.3 ms + 0.0252 per row + 0.30e-6 per row and position + 0.10 per decode row; decode-only 15.2 + 0.333.
- **s0011: expected 0.958-0.968, measured valid 0.963** (182 of 189): TTFT p50 243 ms / p99 7.1 s, tbt p50 40 ms /
  p99 82 ms. Offline step 13.4 ms at B=1.

## 6.8 - 8.1 agent-hours: what the GEMM chains are bound by, and s0012: a home card per row
- Things measured on the way that did not pay (all bitwise-neutral, kept or dropped as noted):
  * dense GEMMs on 256-column tiles, two row groups per block (16 KB of operands per K step instead of 20 KB): 2048-row
    pass 73.0 -> 72.7 ms, q/k/v unchanged at 113 us per layer, o_proj 107 -> 90 us; much better below 1024 rows (256
    rows 32.8 -> 26.0 ms: one row group per block there, twice the SMs). Kept.
  * the B operand's block scales copied to tensor memory once per stage instead of by every row group's thread (7
    tensor-core operations per K step instead of 9): no change (72.7 -> 72.5). Dropped. The expert chains are not
    bound by the count of operations.
  * residual mixes: the row's scalars (sums, softmax over the sources) made once per block instead of by all 288
    threads: -0.6 ms; four sources' loads issued before the first is used: nothing; sources staged in shared memory
    and read once: slower (73.0 -> 75.7 ms: two blocks per SM instead of seven). A row kernel is bound by warp
    instructions (an SM runs roughly 2-3 warp instructions per ns), not by loads.
  * the exchange kernels: what they really cost was measured by launching each twice (gather +3.1 ms, owner sum +5.5,
    spread +3.1 per pass). Their min times showed a floor of ~45-55 us per kernel whatever the bytes.
- The disassembly says the transcendental functions are cheap (tanh ~12 instructions, exp through MUFU.EX2) and float
  divisions carry a slow-path call each; nothing to win there quickly.
- **What was wrong with the structure**: every card kept the whole residual state and ran, for every row, the residual
  mixes, norms, o_proj, the router GEMM, the routing and the MoE-input quantization. Only attention (by heads) and the
  experts (by card) were actually divided: ~20 ms of row kernels + o_proj + router in a 73 ms pass were the same work
  four times.
- **A home card per row** (the cards own consecutive quarters of a pass): the residual state and those kernels exist
  only there. New per layer: the other cards fetch a row's attention input after the mix (k_xin), its routing ids /
  weights / quantized MoE input after the routing (k_xroute); the home card gathers the attention outputs of its rows
  and sums their expert outputs (k_combine; the owner-sum / spread pair is gone); o_proj and the router run on a
  window of columns. The short convolutions reach three rows back: a card also makes o_proj output and expert sum for
  the three rows before its range; the short-convolution state (replicated on all cards, because the next pass may
  give the sequence another home) fetches the few rows it needs (k_xhist). Four exchanges per layer instead of three.
  * First build: correct and bitwise the same as before, and **not faster** (73.4 ms). The profile: the row kernels had
    shrunk as planned (post_attn 4.7 -> 1.4 ms, route 3.6 -> 1.3, mixes 3.1 -> 0.9, o_proj 4.3 -> 2.6), but the fetch of
    the attention input took 10.4 ms: it read the tile layout's scattered 16-byte pieces through the link at 80 GB/s.
    A row-contiguous copy of the input on the home card (xnat) for the others to read: 10.4 -> 2.9 ms.
  * Every block of an exchange kernel polled three remote flags (~2 us each): now block 0 does, one thread per card,
    and the other blocks wait on a local word. Exchange kernels' floor 45-55 -> ~30 us.
  * The column-window code had slowed every GEMM kind (bias GEMM 61 -> 102 us, q/k/v +7 us, router +8 us): the
    compiler no longer folded the window away where there is none. Guarded by a compile-time flag: back to 61.
  * Passes of up to 64 rows run as before (every card owns every row): a fetch is two remote round trips, and six of
    them per layer made a 1-row pass 13.3 -> 14.6 ms. With the switch: 13.2 ms; bitwise the same across the two modes
    (stepwise 2-row passes == an 80-row pass == 14-row passes).
  * 2048-row pass: 72.7 -> 59.5 ms at position 0, 82.7 -> 68.6 ms at position 12000; 1024 rows 43.9 -> 38.1; and larger
    passes no longer step up (2304: 63.2, 2560: 69.8, 3072: 83.2, 4096: 105.3 ms; o_proj is off the one-round limit).
- Cost model: 15.5 ms + 0.020 per row + 0.30e-6 per row and position + 0.09 per decode row; decode-only 14.9 + 0.35.
- **s0012: expected 0.974-0.99, measured valid 0.979** (185 of 189): TTFT p50 195 ms / p99 5.8 s, time between tokens
  p50 35 ms / p99 67 ms, longest gap 92 ms. Offline 13.3 ms per step at B=1. Host-scheduler replay 0.984.

## 8.1 - 8.5 agent-hours: s0013, a larger row cap
- With o_proj on the home card a pass above 2048 rows no longer steps up: 29.0 us per row at 2048, 27.4 at 2304, 27.3 at
  2560, 27.1 at 3072, 25.7 at 4096 (the expert tiles fill better; q/k/v's second round costs little).
- Replays (device loop): cap 2560: 0.984 (186), TTFT p99 5.4 s, tbt p99 78 ms, longest gap 110 ms; cap 3072: 0.984,
  TTFT p99 5.3 s, tbt p99 90 ms, longest gap 128 ms. Taken: 2560 (same share, shorter gaps). Cost model 13.0 ms + 0.0228
  per row.
- **s0013: expected 0.979-0.989, measured valid 0.984** (186 of 189): TTFT p50 206 ms / p99 5.4 s, tbt p50 35 ms /
  p99 78 ms.

## 8.5 - 9.3 agent-hours: s0014 (attention details), s0015 (no call into the cards from submit / poll)
- Chain experiments that changed nothing (dropped): block scales into alternating sets of tensor-memory columns so
  that a K step's copies need not wait for the previous MMA (1 / 2 / 4 sets: 59.3 / 59.1 / 59.2 ms).
- What the exchanges really cost at 2560 rows (each launched twice): expert-output sum +4.0 ms, attention-input fetch
  +3.0, gather +1.0 per 68.8 ms pass; the rest of the sum's kernel time (9.9 ms in the profile) is waiting for the
  slowest card's experts (no rebalancing in that synthetic test). The fetches run at ~300 GB/s per card: the link.
- s0014: far keys of global layers (beyond the bias extent of every row of the group) through a softmax without bias
  loads, selects and mask; the fold with 16-byte loads per (head, 16 features). Bitwise-neutral. 2560 rows 69.7 -> 68.7
  ms, at position 12000 79 -> 76.9 ms; 1-row pass 13.2 -> 12.8 ms. **Expected 0.984, measured valid 0.984**; TTFT p99
  5.4 -> 5.1 s, tbt p50 35 -> 32 ms, p99 78 -> 74 ms; offline 12.7 ms per step at B=1.
- Why submit() took 50-78 ms: a copy issued from the host (cudaMemcpyAsync + wait on its own stream) executes after
  the passes already enqueued on that card. Not the pageable source (page-locked: the same), not the launch thread's
  blocking wait (polling instead: the same). And poll() read the cards' error words every 4096 empty polls: 2-316 ms.
- s0015: submit() writes the prompt into page-locked host memory the cards can read; `k_sched` copies the new
  requests' prompts into its arena (its 1024 threads, 16-byte pieces) before admitting them; poll()'s watchdog reads
  the pass counter in host-mapped memory only. submit() <= 0.54 ms, poll() <= 22 ms in the replay.

## 9.3 - 10.1 agent-hours: s0016 (expert row groups in small passes), s0017 (the GEMM read-out)
- s0016: in passes of up to 16 rows an expert block took one row group; with ~12 experts per card (8 rows) that is
  three rounds of blocks. The kernel now takes three row groups per block when one per block does not fit one round.
  8-row pass 17.3 -> 15.4 ms. **Expected share unchanged, tbt p50 ~30 ms; measured valid 0.984, tbt p50 30.7 ms**,
  offline 15.4 ms per step at B=8 (from 17.2).
- The read-out of a GEMM block was the largest unmeasured cost. The bias GEMM (one K step per unit, so all read-out)
  took 24 us per unit; hypothesis: four warps doing up to 512 two-byte stores per thread are instruction-bound.
  Tensor memory gives a warp the 32 rows of its index mod 4, so more warps can read the same rows (the first try used
  warps 9-12 and read the wrong rows: 9 mod 4 = 1). Reader groups 1 / 2 / 3 / 4 (4 / 8 / 12 / 16 warps), 2560-row
  pass: 68.7 / 61.9 / 60.3 / 59.6 ms. q/k/v 135 -> 85 us per layer, bias GEMM 79 -> 57, expert GEMMs 185 + 128 ->
  153 + 93, o_proj 54 -> 44, router 42 -> 32. Bitwise the same.
- Prompt passes of the replay: 31.7k -> 40.6k rows/s. Cost model 15.0 ms + 0.0174 per row + 0.24e-6 per row and
  position; decode-only 13.9 + 0.17 per row. Larger passes: 23.5 us per row at 2560, 23.0 at 3072, 22.1 at 3584 / 4096.
- **s0017: expected 0.989-0.995, measured valid 0.995** (188 of 189): TTFT p50 179 ms / p99 4.5 s, tbt p50 29 ms /
  p99 65 ms. The one miss: a 32768-token prompt of the burst at 10 s (5.3 s).

## 10.1 - 10.8 agent-hours: s0018 (assignment kernel), s0019 (the last request)
- The one request outside the service levels after s0017: a 32768-token prompt of the largest burst (22 requests, 209k
  new prompt tokens at once), first token at 5.34 s. Shortest-first already puts it last; it needs the whole burst
  done in 5 s.
- Tried and dropped: 28 softmax warps (7 column groups) in the attention kernel with one thread for copies and
  tensor-core operations: bitwise the same, slower (2560 rows 59.0 -> 61.1 ms). 16 softmax warps saturate the SM.
- s0018: the assignment kernel's time was one thread's serial loads (80 loads of one id each, twice): 16-byte loads of a
  row's ids, four rows ahead; experts without rows skipped. 35 -> ~15 us per layer; pass 59.6 -> 59.0 ms. **Expected
  0.995, measured valid 0.995** (longest first token 5.28 s).
- s0019: per-row cost 23.5 us at 2560 rows, 22.1 at 3584 / 4096. Rule (sched.h and the Python scheduler, the two
  simulators agree on every variant): passes of 2560 rows; 3584 rows while some waiting prompt, shortest first at the
  normal pass size, is projected late. Replays: 189 of 189 twice (longest first token 4.94 / 4.88 s), tbt p99 80 ms
  (64 with the 2560 cap), longest gap 97-100 ms; "3584 always" also 189 but p99 85 ms and gap 118 ms.
  **Expected 1.000 (0.995 possible: 60-120 ms of margin), measured valid 1.000**: all 189 requests within the service
  levels; TTFT p50 172 ms / p99 4.4 s, time between tokens p50 28.7 ms / p99 80 ms.

## 10.8 - 11.0 agent-hours: the load sweep, for the record (engine = s0019, tools/t_replay_online.py --rate R, device loop)
| nominal rate | requests | within service levels | TTFT p50 / p99 / max | time between tokens p50 / p99 | longest gap |
|---|---|---|---|---|---|
| 0.5 req/s | 53 | 100% | 252 ms / 4.94 s / 4.94 s | 23.5 / 79.5 ms | 98.5 ms |
| 1 req/s | 103 | 100% | 228 ms / 4.43 s / 4.93 s | 24.3 / 79.5 ms | 98.5 ms |
| 2 req/s | 189 | 100% (s0019) | 172 ms / 4.43 s / 4.88 s | 28.6 / 79.8 ms | 96.9 ms |
| 3 req/s | 284 | 100% | 175 ms / 4.19 s / 4.95 s | 40.7 / 80.7 ms | 97.4 ms |
| 4 req/s | 368 | 100% | 175 ms / 4.15 s / 4.94 s | 36.8 / 81.5 ms | 96.7 ms |
SGLang (the board's reference runs): 0.5 req/s 94%, TTFT 0.9 / 2.6 s, tbt 8 / 622 ms, stall 2.4 s; 2 req/s 82%, 0.6 /
2.2 s, 10 / 788 ms, 2.4 s. The longest first token is the same ~4.9 s at every rate: the window's largest burst (22
requests at once at 2 req/s) is in every one of them, and the scheduler runs it exactly as close to the limit as the
pressure rule lets it (shortest first, largest passes).
- Smaller normal passes under the pressure rule (simulator, soft cap 768 / 1024 / 1536 / 2048 / 2560): time between
  tokens p50 does not move (32.8 .. 32.4 ms in the simulator), TTFT p90 gets worse (2.7 .. 2.25 s): not done. The
  median request's gaps are decode-only passes (15 ms) and the passes of the bursts it arrived in.

## 11.0 - 11.5 agent-hours: experiments after s0019 (none adopted; engine/ is the s0019 source)
- Softmax path without selects and mask for half pages wholly inside every row's window (like the far-key path):
  bitwise the same, no gain (2560 rows at position 12000: 67.2 vs 67.7 ms).
- Small expert tiles (up to 64 pairs) with six row groups per block instead of three (ring and operation threads for
  six): bitwise the same, slower (2560 rows 59.2 -> 62.0 ms on the same machine); an 8-row pass 15.7 -> 15.2 ms, a 32-row
  pass 18.7 -> 19.3 ms. Six chains in one block do not run side by side.
- Dense GEMMs with the K range as two halves accumulated side by side in one block (one row group per block; the
  summation changes for every pass size, so it stays canonical; checked against the reference and bitwise across
  chunkings): 1-row pass 13.0 -> 12.4 ms, 8-row 15.7 -> 15.1, 2560 rows 59.2 -> 59.8 ms. Expected was half the chain
  time for small passes: an SM's tensor core takes its block's operations one after the other (~0.1 us each), so
  shorter chains need other SMs, i.e. partial sums combined through memory. Not worth a change of arithmetic.
- Long context against the reference with the s0019 source (tools/t_long.py, 6000 tokens): rel L2 of the centred
  logits at the last 16 positions median 0.050 / max 0.091, top-1 13 of 16; chunks of 3584 == chunks of 700 bitwise.
- Machine replaced once more (epoch 3) during these; same timings on the new machine (59.2 ms per 2560 rows).

## 11.5 - 12.0 agent-hours: what the attention kernel's time is (measured with timestamps), and three more dead ends
- Fetch kernels with their remote loads issued before the first is used (expert sum: eight pair rows; gather: three
  pieces per thread): bitwise the same, no change (59.1 vs 59.2 ms): the compiler already issued them so, or the link
  bounds them.
- NVFP4 input quantization with the codes made once per candidate (no third pass over the block): bitwise the same,
  no change.
- A pipelined attention kernel (prev/attention_pipelined): tiles of 128 keys, two score and two output tiles in
  tensor memory, one thread for copies and tensor-core operations, so that the scores of the next tile and the value
  product of the previous one run during the softmax. Correct and bitwise-invariant at once; slower (attention 201 ->
  308 us per layer). Timestamps taken inside one block (%globaltimer written by one softmax thread and by the
  operations thread): per 128-key tile the first softmax pass takes 2-5 us, the second 4-8 us, the read-out of a
  half page 5-8 us; a unit of four tiles 57 us. So in the adopted kernel (18 us per 256-key tile) the softmax is
  ~15 us: 126 q x 256 keys x ~33 instructions = 33k warp instructions at the ~3 per ns an SM does. The tensor-core
  and copy time a pipeline can hide is ~3 us per tile. **The attention kernel is bound by softmax instructions.**
- Fewer softmax instructions: the first pass's 64 scores per thread kept in registers for the second pass, zeros from
  exp(-1e30 - m) instead of a select: bitwise the same, slower (59.2 -> 60.5 ms; 92.9 -> 96.7 at position 16000):
  spills, as with the residual-mix sources earlier.
- The earlier estimate "6-8 us of tensor-core and copy time per tile" came from the 4 -> 16 warp scaling under the
  assumption of perfect scaling; the stub experiment that was to check it (softmax arithmetic skipped) gave slower
  passes, because wrong data changes the routing and with it every expert kernel. Lesson kept from the exchange
  experiment: only idempotent repetitions or timestamps measure a part of a pass.

## 12.0 - 12.5 agent-hours: s0020, the source as it stands
- 240 s of the trace instead of 120 (368 requests, device loop): 368 within the service levels; TTFT p50 166 ms / p99
  4.1 s / max 4.89 s; time between tokens p50 35 / p99 82 ms; 1640 evictions, 2041 of 5862 blocks from the cache;
  pages free == total after drop_cache.
- The stats' `recomputes` was never counted. The device scheduler now keeps the keys of evicted blocks (direct-mapped,
  4096) and counts a later miss on one as a recomputation. Replay with 420 pages instead of 2156: 1890 evictions, 22
  recomputations, probes 6/6 identical, 188 of 189 (one first token at 6.2 s: the price of recomputing).
- Machine replaced twice more; build products are rebuilt by engine/build.sh, nothing else was lost.
- **s0020 (counter + source comments, no change to passes or rules): expected 1.000 (or 0.995 on a 60-120 ms
  margin), measured valid 1.000**: TTFT p50 175 ms / p99 4.4 s, time between tokens p50 28.7 / p99 80 ms.

## 12.5 - 13.2 agent-hours: s0021 (softmax instructions), s0022 (slack in the pressure rule)
- With the softmax known to be the attention kernel's time: fewer instructions per (q, key), without more live
  registers. (1) The first pass stores its scaled / biased / masked scores back into tensor memory (`tcgen05.st`,
  first use of it here: worked at once) and the second pass reads them: 2560 rows 59.2 -> 58.4 ms. (2) Masked scores
  are -1e30, exp(-1e30 - m) is exactly 0: no select per element in the second pass (a q with no key in the half page
  takes an infinite reference): 58.4 -> 57.8 ms; 3584 rows at position 16000 92.9 -> 87.3 ms. Both bitwise-neutral.
  Tried on top, no gain (dropped): far tiles without the per-element select for unused lanes, a single range test
  and unconditional bias in local layers.
- Cost model: 13.5 ms + 0.0174 per row + 0.20e-6 per row and position.
- **s0021: expected 1.000 (0.995 possible), measured valid 1.000**: TTFT p50 166 ms / p99 4.22 s (from 4.45), time
  between tokens p50 27.4 ms / p99 76 ms, longest gap 90 ms; offline 12.7 ms per step at B=1.
- The longest first token stayed at 4.87 s: the pressure rule returned to normal passes as soon as the projection held
  with 0.05 s to spare, so the last request of the largest burst ended just under the limit by construction. Slack
  0.3 s (the simulators, whose pass model is pessimistic for that request, show no difference; the replay does).
  **s0022: expected 1.000 with the longest first token ~4.6 s, measured valid 1.000**, longest first token 4.62 s,
  TTFT p99 4.22 s, time between tokens p50 27.4 / p99 76.5 ms.

## 13.2 - 14.2 agent-hours: s0023, the dense chains of small passes on two SMs
- Upper bound first (half the K steps of the dense GEMMs in small passes, wrong results, a 1-row pass barely depends
  on the routing): 13.2 -> 10.9 ms at 1 row, 15.8 -> 13.8 at 8 rows. The simulator: 2 ms off a decode-only pass is
  -5% on the time between tokens p50.
- Every dense GEMM is now D = (first half of the K steps) + (second half). Passes of up to 64 rows: two blocks on two
  SMs; the first writes its accumulators (fp32) to a scratch and sets a flag (pass tag, layer), the second waits for
  the flag after its own chain, adds, and runs the usual read-out. Larger passes: both halves in one block (one row
  group per block, two accumulators of up to 256 columns). The two forms are bitwise the same (a two-term FP32 sum).
- Checks: single layers against the reference as before (0.002); stepwise == chunked == chunks across both modes;
  generate == step; device == host; cache test; 6000 tokens against the reference 0.052 / 0.081 (was 0.050 / 0.091).
- 1-row pass 13.2 -> 11.3 ms, 8 rows 15.8 -> 14.0, 32 rows 18.7 -> 17.3; decode-only passes of the replay 12.0 ms +
  0.16 per row (was 13.9 + 0.17); 2560-row pass 57.8 -> 58.6 ms.
- **s0023: expected 1.000, tbt p50 ~26.4 ms, TTFT p99 ~4.27 s; measured valid 1.000**, see the numbers below in
  notes/summary.md.

## 13.8 - 14.6 agent-hours: s0024, a stress test finds what the replay cannot
- `tools/t_stress.py` (new): N requests in three waves, lengths from 20 tokens to the context limit, families of
  shared prefixes, a page store too small for all of them; every request must finish with its tokens, a sample must
  equal the same request alone on an empty engine, free == total after drop_cache. Modes `edges` (lengths at every
  limit: 1, 511 / 512 / 513, 3583 / 3584 / 3585, 33279 / 33280 / 33281, 34815 + 1, 34304 + 512, the same prompt four
  times at once) and `flood` (more requests than the request ring's 1024 entries).
- First run (150 requests, 700 pages): 6 requests never finished; the loop composed 45,000 empty passes with 10 slots
  "active". Cause: `SSlot.keys[64]` (one key per full prompt block) and `nodes[64]` in the admission: a prompt of more
  than 33,279 tokens has 65-67 full blocks and wrote past the table into the next slot's state, rid and lengths.
  The replay cuts prompts at 32,768 tokens (64 blocks), so neither the evaluation nor my earlier long test (one
  sequence alone) could show it. Both tables are S_MAXPG (68) now.
- Host side, found by reading the same path: submit() refuses a request outside 1 .. 34,816 positions or larger than
  the page store (before: memory overrun, or a queue that stands forever); the prompt memory is a ring whose places
  stay in use until the request's first token has come back, and a new prompt waits on the host (entering from
  poll()) rather than overlap one in use or overrun the request ring; the watchdog watches the rows computed, not the
  passes composed (the empty passes above kept it quiet).
- After the fix: mix 150 requests / 700 pages 22.4 s, 727 evictions, 44 recomputations, 10 of 10 sampled requests
  equal alone; edges 26 of 26 equal; flood 1500 requests in 9.3 s; mix with 150 pages (two requests of full length)
  38 s, 1582 evictions, 472 recomputations, 10 of 10. Replay in the machine 189 of 189, probes 12 of 12.
- **s0024: expected the figures of s0023, measured valid 1.000**: TTFT p50 169 ms / p99 4.28 s, time between tokens
  p50 26.4 / p99 76.5 ms, longest gap 89 ms; offline 11.2 ms (B=1) / 14.0 ms (B=8) per step.
- Estimated and not built: a hot expert's weights on all four cards (each card runs its own rows' pairs: no fetch of
  those outputs). Per card and layer a pair output fetched is 9216 B at ~420 GB/s; an expert with n pairs in a
  2560-row pass saves ~0.75 n x 22 ns = 8 us at n = 2000, and costs three more partial tiles over the four cards
  (~3.3 us per card). Break-even near n = 800; with ~5 such experts per layer the gain is ~0.8 ms of 58.6.

## 14.6 - 15.0 agent-hours: where the two remaining losses are bound (measurements, nothing adopted)
- Stress variants after s0024, all PASS: drop_cache every 1.7 s while the crowd runs (700 pages); see tools/t_stress.py.
- Kernels of a 2560-row pass at position 4096 (nsys, per card, ms; 62 under the profiler, 58.6 without): expert GEMMs
  13.0 (W13 8.2, W2 4.85), combine 8.6, attention 7.2, q/k/v 4.7, activation + quantization 4.3, attention-input
  fetch 3.4, bias GEMM 2.8, o_proj 2.4, post_attn 1.9, routing fetch 1.9, router 1.75, gather 1.75, route 1.5, mix 1.4,
  fold 1.3, prep 1.0, assign 0.9, packx 0.8, post_mlp 0.7, hist 0.9.
- How long a card waits for the others (timestamps around the flag polls of the four exchanges, replay at 2 req/s,
  450 prompt passes; ms per pass and card): attention input 0.22-0.34, routing 0.25-0.51, attention outputs 0.17-1.19
  (cards 0 and 3 wait 0.8-1.2 for cards 1 and 2), expert outputs 1.37-1.65 on every card. An exchange's own latency
  is ~6 us per layer (0.3 ms per pass); the rest of the expert wait (~1.1 ms) is the per-layer difference between the
  cards' expert loads after rebalancing: every card is the slowest in some layers. Perfect balance would be worth
  ~2% of a pass; not pursued.
- Fixed grids of the device loop against exact grids: 11.63 vs 11.25 ms for a 1-row pass (0.4 ms): a second launch
  program for small passes is not worth the two-pass commitment it needs.
- Kernels of an 8-row decode pass (14.2 ms; per card, ms): combine 1.46 (30 us per layer, of which ~20 waiting for
  the card with the most experts: 64 pairs per layer fall 16 +- 4 per card), W13 1.9, o_proj 1.3, q/k/v 1.2, W2 1.1,
  router 1.0, attention 0.9, gather 0.6, route 0.6, post_attn 0.5, act 0.6, hist 0.4, mix 0.5, fold 0.4, bias GEMM 0.3,
  prep 0.3, the rest 1.0. Kernels run back to back (no gaps); a kernel that does nothing costs 1.5-3 us.
- `__expf` (the fast intrinsic) instead of `expf` in the second softmax pass: 2560 rows at position 4096 and 16000
  61.3 / 67.0 ms either way. The exponential is not what bounds the attention kernel.
- Estimated, not built: (1) expert tiles of 256 columns (one row group per block, as the dense GEMMs): a K step needs
  4.6 KB of weights + 9.2 KB of activations = 0.29 us at the ~48 GB/s one SM takes in, for 256 columns; three row
  groups x 128 columns need 0.45 us (operation-bound) for 384 columns: 0.88 against 0.85 columns per ns. No gain.
  (2) Larger passes with decode-only passes in between: the base cost (13.5 ms: one pass over 387 GB of weights) is
  paid per token interval of the decoding requests either way; with a mean gap of 90 ms the prompt throughput is
  bounded by (90 - 13.5) / 0.0177 / 0.09 = 48k rows/s and the engine runs at 46.5k in its largest passes. The
  first-token p99 (the last requests of a 209k-token burst) can only fall with the cost per row.

## 15.0 - 15.5 agent-hours: load sweep of the final engine; s0025, two more guards
- Sweep with the replay tool (s0024 source): 100% within the service levels at 0.5, 1, 2, 3 and 4 req/s nominal
  (53 / 103 / 189 / 284 / 368 requests); 89% at 6 req/s (585 requests), 47% at 8 req/s (780). Past capacity only
  first tokens are late: every request finishes and no request's mean gap exceeds 93 ms. Table in notes/summary.md.
- Reading the paths the stress test exercises: (1) the scheduler wrote token records into its ring of 131072
  without regard to what the host had read: 64 decoding requests lap it in ~26 s without a poll (tokens lost
  silently). The host publishes how far it has read (`out_tail`, the spare word of the shared block) and the
  scheduler composes no rows while it is more than 114k records behind. `MODE=slow` of tools/t_stress.py: 64
  requests x 2100 tokens, no poll for 47 s: the device stopped after 114,329 decode rows, then every token arrived;
  2 of 2 sampled requests equal alone. (2) submit() checks the token ids against the embedding table.
- **s0025: expected the figures of s0024, measured valid 1.000**: TTFT p50 169 ms / p99 4.28 s, time between
  tokens p50 26.5 / p99 76.6 ms; all gates with every sequence identical.

## 15.5 - 16.1 agent-hours: small-pass and pass-size experiments (none adopted; engine/ is the s0025 source)
- Largest passes whenever no request is decoding (nobody waits for the pass to end): the simulator's figures do not
  move at 2 req/s (TTFT p50 0.16 s either way). Not done.
- Two expert row groups per block for cards with 5-8 experts in a small pass (between "one" and "three"): passes of
  1-12 rows 11.6 / 11.9 / 12.1 / 12.2 / 12.8 / 13.4 / 14.1 ms either way. At these sizes the expert GEMMs are near
  the bytes they must read (8 rows: ~16 experts per card and layer = 287 MB, 40 us at 148 SMs x 48 GB/s; measured
  W13 + W2 56 us).
- What a kernel boundary costs: skipping the two kernels per layer that do nothing in a small pass (attention-input
  fetch, convolution-state fetch; exact grids): 11.25 -> 11.11 ms per 1-row pass, 1.4 us per kernel. Fusing the
  per-row kernels that could be fused without a barrier between them (post_mlp of a layer with the attention mix of
  the next; the two convolution-state kernels) would save ~4 kernels per layer, ~0.3 ms of 11.6. Not done.
- Pass time against pass size (one sequence at position 4096, no rebalancing; ms): 2048 52.8, 2304 56.2, 2560 61.3,
  2816 66.5, 3072 70.3, 3328 74.7, 3584 79.5: steps of 256 rows cost 3.4 - 5.2 ms, the cheaper ones where a card's
  quarter of the pass ends on a 256-column tile (o_proj and router windows); the effect is ~1 ms, sizes stay.
- Estimated, not built: q/k/v for all heads on a row's home card and a fetch of the other cards' slices (3.3 KB per
  row instead of 9.2 KB, 27.6 KB in global layers): -2.2 ms of fetch per 2560-row pass, +0.9 ms because a quarter
  of 2560 rows is 2.5 column tiles (three chains instead of 2.5), four times the q/k/v weights per card. ~2%.

## 16.1 - 17.0 agent-hours: s0026, the scheduler kernel's own time (found by a 400 s replay)
- A 400 s replay of the final engine (643 requests): 642 within the service levels, one first token at 5.25 s, time
  between tokens p50 43.5 ms. The simulator (same scheduler code, same 4739 evictions) put that burst at 4.7 s: the
  device was slower than its own cost model once the page store had filled.
- Profile of `k_sched` (nsys, 40 s of the replay): 65 us on a decode-only pass, but 1-3 ms on every prompt pass
  (132 calls, 1.9 ms on average) and 3-10 ms on every sixth (20 calls): 2.1 ms per prompt pass, 3.6% of it, spent
  by one thread on (a) the running average of the pairs per expert and layer, 18,432 values; (b) the search for
  expert swaps when a rebalance is due, 48 layers x ~2700 steps; (c) the per-row tables of the pass (slot,
  position, token, rows back: up to 3584 rows); and with a full store (d) one walk over the whole cache table
  (16,384 nodes) per evicted page.
- Now: (a), (b) and (c) by the block's 1024 threads (b: one thread per layer makes that layer's proposals, a pure
  function of its loads and placement; thread 0 merges them in layer order up to the 12 swaps of a round), the
  decisions stay in s_step with thread 0. (d): per stripe of 64 nodes the table keeps its least recently used
  evictable leaf; a record is never above the stripe's true minimum (a node that becomes an evictable leaf lowers
  it, a stale record is found when looked at and that stripe rescanned), so the victim is the same page as before.
- Checks: old against new swap search on the host, 3000 random steps, 1480 swaps: equal (also with precomputed
  proposals); simulator output byte-identical on four traces (up to 7416 evictions); device == host tokens, cache
  test, stress 150 requests / 150 pages (1582 evictions, 472 recomputations as before), edges 26 of 26.
- k_sched on prompt passes 2.1 ms -> 0.13 ms (268 calls of 100-300 us). Replay in the machine: TTFT p99 4.28 -> 4.15 s,
  time between tokens p50 26.4 -> 26.0, p99 76.5 -> 74.8 ms. 400 s replay: **643 of 643**, TTFT p90 1.84 -> 1.52 s,
  p99 4.07 -> 3.43 s, longest 5.25 -> 4.63 s, time between tokens p50 43.5 -> 36.3 ms, p99 82.9 -> 74.7 ms.
- **s0026: expected 1.000 with TTFT p99 ~4.15 s, tbt 26.0 / 74.8 ms; measured valid 1.000**, TTFT p50 166 ms / p99
  4.15 s / max 4.62 s, time between tokens p50 26.0 / p99 74.8 ms, longest gap 89 ms; offline 11.2 / 14.0 ms.
- Lesson: the device scheduler was written as "one thread decides" and its cost was never profiled apart from the
  pass; everything O(rows) or O(experts) in it belongs to the block's other 1023 threads.

## 17.0 - 17.3 agent-hours: s0027, scheduler costs that grow with uptime or backlog
- After s0026, read the scheduler for whatever else is not O(1) per pass. (1) The prefix cache's table (open
  addressing, 16,384 slots for 2156 pages) marks removed nodes as deleted and never makes a slot empty again; a
  lookup of a key that is not cached ends only at an empty slot. The scheduler code on a virtual clock over 3.3
  hours of the trace (30 laps of the 400 s window with fresh contents: 19,290 requests, 206k inserts, 204k
  evictions): 14,240 deleted marks, 0 empty slots: every miss walks the whole table (one device thread, several
  misses per pass). Fix: a removed node whose next slot is empty becomes empty, and so do the deleted marks before
  it. Same 3.3 hours: 364 marks, 13,876 empty; the schedule byte-identical (tools/sched_sim.cpp prints the table).
  (2) A request whose admission fails is looked at again before every pass and was hashed again each time (up to
  32k tokens on one thread): its block keys stay in the free slot they were written to.
- Checks: simulator identical on four traces; device == host; cache test; stress 150 / 150 pages, 1500 at once,
  300 requests on 40 pages; replay 189 of 189, probes 12 of 12.
- **s0027: expected the figures of s0026, measured valid 1.000**: TTFT p50 176 ms (166 in the machine's replay and in
  s0026; the median sits between two requests ~20 ms apart) / p99 4.15 s / max 4.62 s, time between tokens p50 26.0 /
  p99 74.8 ms, offline 11.2 / 14.0 ms.
- 20-minute replay of the s0027 engine (2120 requests, 11.1M prompt rows, 18,697 evictions, 1901 recomputations; the
  prompt memory, the request ring and the token ring all wrap): every request finishes, 2119 within the service
  levels (one first token at 5.47 s in a later burst), probes 12 of 12, free == total at the end. TTFT p50 164 ms /
  p99 3.37 s, time between tokens p50 50.2 / p99 73.3 ms (the later trace is denser), longest gap 89 ms.

## 17.3 - 17.9 agent-hours: s0028, the unembedding of small passes
- The unembedding had inherited the small-pass forms of s0023 (K range as two half chains in two blocks; 64-row
  units up to 16 rows): 1572 units with a scratch hand-over each, 207 us per pass. Those forms shorten a chain; the
  unembedding (393 row groups on 148 SMs) is bound by operations. It now always runs 393 units of both halves in
  one block. Same bits (stepwise == chunked on the logits, reference error unchanged at 0.1299845278263092).
- A/B on the same machine (two builds each, 41 passes): 1 row 11.60 -> 11.48 ms, 8 rows 13.34 -> 13.20, 32 rows 15.7 ->
  15.6. (A comparison across a machine replacement had shown the same size of effect in both directions: A/B on one machine.)
- Decode-only passes through the device loop (tools/t_gap.py: B requests with different contents) against the same
  pass launched from the host: 11.71 / 11.74 ms at 1 row, 14.64 / 14.57 at 8, 19.14 / 18.27 at 32: the loop adds
  nothing (it is not bound by the host's launches). Rows of different sequences with unrelated tokens cost 0.24 -
  0.36 ms each (every row brings ~8 experts of its own: 17.9 MB of weights and 3888 tensor-core operations per
  expert, whatever its column count); the replay's rows (prompts from a small vocabulary) 0.16 ms.
- **s0028: expected 1.000, time between tokens p50 ~25.9 ms, steady decode ~11.08 / ~13.83 ms; measured valid
  1.000**, time between tokens p50 25.9 / p99 74.8 ms, TTFT p50 166 ms / p99 4.15 s, steady decode 11.07 ms per step
  at B=1 and 13.83 ms at B=8.

## 17.9 - 18.6 agent-hours: the scheduler under a fuzz harness (no engine change)
- `tools/sched_fuzz.cpp` (new, host only): the scheduler code of engine/sched.h driven through random scenarios
  (20-160 requests of 1 .. 34,815 prompt tokens and 1 .. 600 new tokens, 8-512 pages, 4096 or 8192 cache slots, pass
  sizes 256 / 1024 / 3584, bursts that share a new or an old prefix, drop_cache at random steps, a host that stops
  reading tokens for up to 9000 steps or entirely until the scheduler stands still). After every step the books
  are recomputed from scratch and compared: slot states and lengths, pages per slot, reference count of every page,
  the free list, reserved pages covered by free + evictable, cache nodes found by their keys, children counts,
  the stripe records of the eviction (never above the stripe's true minimum), the rows and tables of the composed
  pass, tokens per request, the token ring never lapped, no rows composed while it is full; at the end free ==
  total. 24,000 runs on the machine's CPUs (2.1M requests, 13.8M evictions, 1.0M recomputations): all pass.
- The harness finds seeded bugs: the 64-entry key table of s0024 (at step 0-25), a missing eviction candidate on
  release or for a parent, deleted marks cleared without looking at the next slot, the admission memo without its
  slot, pinned pages not counted at admission, late hits that keep their reserve, no hold for a host that does not
  read. (Writing it found two errors of its own first: bursts that never formed because the harness's arena check
  was off by the rounding, and a host model that could not lag far enough to fill the ring: a fuzz that reports
  "0 dedup waits, 0 late hits" has not tested them.)
- `tools/t_ring.py` (new, host only): the host's prompt ring, request ring and waiting queue of engine/api.py
  without a device: 300 runs x 3000 random arrivals / admissions / first tokens with arenas of 4-20k tokens: no
  prompt that the device may still read is ever overwritten, order kept. (With the overlap rule removed it fails
  at once.)
- Load sweep again with the s0028 source (table in notes/summary.md): 100% at 0.5 - 5 req/s (53 .. 483 requests),
  583 of 585 at 6 req/s (s0024 source: 521), 532 of 780 at 8 req/s (was 363). The gain at high rates is s0026's:
  there the page store is full within the first bursts and every evicted page had cost a walk over the cache table.
- close() with 40 requests open (prompts in prefill, sequences decoding, requests in the ring): 0.16 s, clean
  (tools/t_close.py).

## 18.6 - 19.6 agent-hours: s0029, kernel boundaries of small passes
- Time over all kernels of a 40 s replay (nsys, per card): attention 9.4%, q/k/v 8.1%, o_proj 8.0%, combine 6.7%,
  expert GEMMs 12.9%, router 6.1%, gather 4.1%, activation 4.5%, fold 3.7%, post_attn 3.7%, route 3.7%, bias GEMM
  2.6%, the rest under 2.5% each; scheduler kernel 0.7% (with its idle waiting). Most passes are decode-only, so the
  profile is that of a small pass: 24 kernels per layer, none of them dominant.
- (1) post_mlp of layer i and the attention mix of layer i + 1 as one kernel (`k_post_mix`); (2) the fetch kernels
  (attention input, routing) with 3 / 6 blocks per SM that loop over rows instead of one block per row (the device
  loop's grids are those of the largest pass). A/B on one machine: 1 row 11.47 -> 11.26 ms, 8 rows 13.20 -> 12.98, 32
  rows 15.6 -> 15.4; 2560 rows 61.4 -> 61.5, 3584 rows 79.6 -> 80.0 (builds with one of the two changes: 11.34 and
  11.42 at 1 row; their large-pass times scatter by 0.3-0.5 ms between builds, so the 0.4 ms is not attributed).
- **s0029: expected 1.000, gap p50 12.94, tbt p50 25.75 / p99 75.2 ms, TTFT p99 4.158 s, steady decode ~10.87 /
  ~13.62 ms; measured valid 1.000**, gap p50 12.93, tbt p50 25.79 / p99 75.1 ms, TTFT p50 168 ms / p99 4.165 s,
  steady decode 10.98 / 13.73 ms: half the predicted gain there, because the offline step launches exact grids
  (only the fusion counts; the smaller fetch grids help the device loop alone). A trade, stated before: decode
  passes -1.9%, the largest prompt passes +0.5%.
- Tried on top and dropped: the same row loops for the prep and fold kernels (grids of 8 / 16 blocks per SM instead
  of one block per row of the largest pass): slower at every size (1 row 11.25 -> 11.3 - 11.6 ms, 8 rows 13.0 ->
  13.3, 32 rows 15.4 -> 15.8, 3584 rows 80.0 -> 80.5). These kernels reduce over the block; the barrier that
  separates two rows of a looping block costs more than the empty blocks did.
- The fuzz harness also drives the expert placement now (random pair counts per pass, a rebalance every 1-6 prompt
  passes): after every step the placement must still be one (96 experts per card and layer, every place taken
  once) and equal the previous one with the pass's swaps applied, each swap naming where its experts were. 12,800
  more runs on the machine's CPUs (1.1M requests, 6.0M swaps, 7.4M evictions): all pass.

## 19.6 - 20.2 agent-hours: s0030, the two APIs on one engine
- step() / generate() on an engine whose device loop runs were not refused: the offline code checked the host
  scheduler's lists, which the device loop does not use, and would have composed passes next to the loop's. Now:
  on an idle engine the loop is stopped (as close() does) and the offline call proceeds; the next submit() forgets
  the offline sequences and starts the loop again, with the shared counters and the cards' flag words (exchange
  flags, gates, K-half flags) cleared, because the second run's pass tags repeat the first's; with requests open
  the offline call raises.
- tools/t_mix.py: submit/poll (four prompts), generate, submit/poll, generate, submit/poll on one engine: the same
  prompt gives the same 24 tokens every way; the offline call with a request open is refused; free == total.
- **s0030: expected the figures of s0029, measured valid 1.000**: TTFT p50 167 ms / p99 4.16 s, time between tokens
  p50 25.2 / p99 75.2 ms, steady decode 10.98 / 13.74 ms.

## 20.2 - 20.8 agent-hours: s0031, where s0029 had lost on large passes
- Profile of a 3584-row pass with the kernels of s0028 and of s0029 (nsys, means per layer): post_mlp + mix as two
  kernels 15.3 + 29.5 us, fused 41.3; attention-input fetch 76.0 -> 74.0 (global layers 201.8 -> 192.0); routing
  fetch **66.8 -> 91.4** (FP8 layers 83.4 -> 98.9). With 12 or 24 blocks per SM instead of 6 the looping routing
  fetch still took 79.4: it is the loop, not the grid. (Unexplained: both fetches loop the same way; the routing
  fetch has five small sources per row.)
- The routing fetch is one block per row again. A/B against s0029: 2560 rows 61.7 -> 60.6 ms, 3584 rows 80.2 -> 78.9,
  1 row 11.29 -> 11.31, 8 rows 13.0 -> 13.0. So the fusion and the attention-input loop were worth ~0.8 ms on a large
  pass all along; s0029 had hidden it behind a 1.2 ms loss.
- **s0031: expected 1.000, TTFT p99 ~4.11 s, tbt 25.6 / 74.3 ms; measured valid 1.000**: TTFT p50 164 ms / p90 1.93 s /
  p99 4.10 s, time between tokens p50 25.6 / p99 73.7 ms, longest gap 88 ms, steady decode 10.98 / 13.73 ms.
- Lesson: an A/B of the whole pass told me "+0.4 ms" and nothing else; the per-kernel profile of both builds took
  five minutes and showed a gain and a loss twice its size.

## 20.8 - 21.3 agent-hours: s0032, the routing fetch's five pieces per row
- The routing fetch took 67 us per layer at 3584 rows for 2.7 KB per row; the attention-input fetch 74 us for 9.2 KB.
  Leaving out its four small sources (scales, ids, weights, outer scale; results wrong, the kernel's time valid):
  67 -> 46 us. A row's routing came from five places on the home card.
- The home card now writes a row's ids (32 B), weights (32 B) and outer scale (4 B) once more behind the row's 288
  scale bytes (rows of 384 bytes); the fetch reads 23 consecutive 16-byte pieces there and the codes: two contiguous
  pieces per row. Routing fetch 66.9 -> 52.9 us per layer (FP8 layers 83.3 -> 66.7).
- A/B against s0031 (alternating builds): 2560 rows 60.8 / 61.2 -> 60.2 / 60.6 ms, 3584 rows 78.9 / 79.2 -> 78.1 / 78.4,
  small passes equal. Bitwise checks, single layers, device == host, stress, replay with probes: as before.
- **s0032: expected 1.000, TTFT p99 ~4.07 s, tbt 25.5 / 73.2 ms; measured valid 1.000**: TTFT p50 164 ms / p90 1.91 s /
  p99 4.065 s, time between tokens p50 24.8 / p99 73.3 ms, longest gap 87.5 ms, steady decode 11.01 / 13.72 ms.
- Tried on top and dropped: one barrier per selection round in the routing kernel instead of two (two sets of
  warp maxima): 1 row 11.33 / 11.31 -> 11.30 / 11.33 ms, large passes equal. The barriers are not its time.
- Final checks on the s0032 engine: 400 s of the trace 643 of 643 within the service levels (TTFT p50 160 ms / p99
  3.37 s / max 4.62 s, time between tokens p50 35.6 / p99 71.9 ms), probes 12 of 12; edge lengths 26 of 26; 150
  requests on 150 pages PASS; online / offline in turns PASS.

## 21.3 - 21.8 agent-hours: s0033, serial remote loads in the convolution-state fetch
- A thread's remote loads come one after the other (~2 us each, F12). `k_xhist` fetched each row it needs (9.2 KB of
  o_proj output and 9.2 KB of expert output) with 256 threads: up to six loads in series, 11 us per layer for a
  few dozen rows. With 576 threads: two. A/B against s0032: 2560 rows 60.2 / 60.2 -> 60.0 / 59.8 ms, 3584 rows
  78.1 / 78.1 -> 78.1 / 77.6; small passes equal.
- Looked at the other fetches with the same eyes: attention input one load per thread and input already; routing
  fetch one (two for 23 threads); the gather has three in series per thread with 256 threads, but one round of
  blocks: 768 threads would be one load and three rounds; the expert sum eight (six remote) but is bound by bytes.
- **s0033: expected 1.000, TTFT p99 ~4.04 s, tbt 25.4 / 72.9 ms; measured valid 1.000**: TTFT p50 162 ms / p90 1.905 s /
  p99 4.044 s / max 4.63 s, time between tokens p50 25.4 / p99 72.8 ms, longest gap 87 ms, steady decode 10.98 /
  13.71 ms.
- Tried and dropped: the expert sum's fetch by bulk copies (`cp.async.bulk` from the other cards' memory into shared
  memory, eight requests of 9216 B per row instead of 4608 loads of 16 B by threads that wait for each). It works
  on peer memory (results bitwise the same) and is not faster: 2560 rows 59.9 -> 60.4 ms, 3584 rows 77.9 -> 78.3.
  The expert sum is bound by the bytes over the links (35 MB per card and layer), not by the threads' load latency.
- Load sweep of the s0033 engine at the higher rates: 2 req/s 189 of 189, 4 req/s 368 of 368, 5 req/s 483 of 483,
  6 req/s 584 of 585 (one first token at 5.05 s), 8 req/s 647 of 780 (s0028: 532; s0024: 363).
- Tried and dropped: the NVFP4 quantization of the MoE input in the post-attention kernel (where the row's values
  are in registers) instead of in the routing kernel (which reads them back). Bitwise the same; large passes equal
  (60.05 / 60.05 ms at 2560 rows), small passes slower (1 row 11.32 -> 11.62 ms, 8 rows 13.0 -> 13.3): the block-wide
  maximum and the quantizer lengthen the kernel in front of the router GEMM by more than the routing kernel gets
  shorter behind it.
- 20-minute replay of the s0033 engine (2120 requests, 18,695 evictions, 1901 recomputations): all finish, 2119
  within the service levels (one first token at 5.37 s: the same later burst as with s0027, 5.47 s then), TTFT p50
  159 ms / p99 3.31 s, time between tokens p50 49.0 / p99 71.4 ms, longest gap 88 ms, probes 12 of 12, free == total.
- Final source (s0033) against the reference at a long position: 6000-token prompt, last 16 positions, rel L2 of the
  centred logits median 0.0524 / max 0.0805, top-1 13 of 16 (as at s0023), bitwise equal between chunks of 3584 and
  700 rows. Decode-only gaps through the device loop: 11.57 ms at 1 request, 11.95 at 2, 14.47 at 8, 19.18 at 32
  (requests with unrelated tokens). close() with 40 requests open: 0.12 s.

## 22.3 - 22.9 agent-hours: s0034, the gather with one remote load per thread
- The series lens of s0033 on the gather of the other cards' attention outputs: large passes ran it with 256 threads
  per row (three remote loads in series per thread), small passes with 768, and the fixed program launched both.
  One launch with 768 threads for every pass size. A/B against s0033 (the device loop's grids): 2560 rows 60.4 /
  60.0 -> 59.7 / 59.8 ms, 3584 rows 78.3 / 77.8 -> 77.6 / 77.6, 1 row 11.31 / 11.35 -> 11.29 / 11.26, 8 rows 13.01 /
  13.02 -> 12.95 / 12.95. (Three rounds of blocks with one load each beat one round with three in series.)
- **s0034: expected 1.000, TTFT p99 ~4.03 s, tbt 25.4 / 71.7 ms; measured valid 1.000**: TTFT p50 161 ms / p90 1.893 s /
  p99 4.029 s / max 4.67 s, time between tokens p50 25.3 / p99 72.7 ms, longest gap 85.5 ms, steady decode 10.99 /
  13.72 ms. (`smc` in launch_pass is now an unused constant.)
- Final checks on the s0034 engine: crowd of 150 requests on 700 pages PASS (10 of 10 sampled requests equal alone),
  edge lengths 26 of 26, host == device tokens; 400 s of the trace 643 of 643 within the service levels (TTFT p50
  159 ms / p99 3.34 s / max 4.62 s, time between tokens p50 35.4 / p99 70.7 ms, longest gap 85 ms), probes 12 of 12.

## 22.9 - 23.4 agent-hours: s0035, the last of the serial remote loads
- Routing fetch: threads 0-22 read a piece of the codes and then a piece of the packed scales / ids / weights (two
  remote loads in series); the packed piece moved to threads 296-318. A/B against s0034: 2560 rows 59.6 / 59.6 ->
  59.5 / 59.5 ms, 3584 rows 77.5 / 77.5 -> 77.2 / 77.3.
- **s0035: expected 1.000 and the figures of s0034 within noise; measured valid 1.000**: TTFT p50 161 ms / p90 1.894 s /
  p99 4.023 s / max 4.61 s, time between tokens p50 25.2 / p99 72.3 ms, longest gap 86.5 ms, steady decode 10.98 /
  13.74 ms.
- State at the end of the run: engine/ = s0035. Against SGLang on the same requests: share 1.000 / 0.825, TTFT p50
  161 / 625 ms, TTFT p99 4.02 / 2.2 s, time between tokens p50 25.2 / 10 ms, p99 72 / 788 ms, longest stall 87 ms /
  2.4 s. Open and dead ends: notes/plan.md.
- Final checks on the s0035 engine: crowd of 150 requests on 700 pages PASS (10 of 10 sampled requests equal alone),
  edge lengths 26 of 26, host == device tokens.
- Last runs: the scheduler fuzz with 32 new seeds x 300 runs on the machine's CPUs: all pass. Low rates with the s0035
  engine: 0.5 req/s 53 of 53 (TTFT p50 234 ms / p99 4.66 s, time between tokens 20.7 / 70.3 ms), 1 req/s 103 of 103
  (211 ms / 4.01 s, 21.2 / 70.7 ms), 3 req/s 284 of 284 (160 ms / 4.00 s, 34.5 / 71.6 ms).
- Measured, not adopted (the engine stays s0035): the attention-input fetch of the global layers with one input per
  block (one remote load per thread and row instead of three in series): 2560 rows 59.4 / 59.6 -> 59.4 / 59.5 ms,
  3584 rows 77.3 / 77.3 -> 77.2 / 77.2. That fetch moves 74 MB per card and layer at ~540 GB/s: it is bound by the
  links, as the expert sum is. The series of loads mattered where few bytes moved (convolution state, gather,
  routing).
- Exchange kernels of the s0035 engine at 2560 rows (nsys, us per layer): gather 27.4 (31.8 at s0032), routing fetch
  22.7 / 30.7 in FP8 layers (27.4 / 36.3; 67 before s0032 at 3584 rows), convolution-state fetch 5.8 (11.2),
  attention-input fetch 53.7 / 136.9 in global layers (unchanged: link-bound). Pass under nsys 61.0 ms (61.8).
- On the s0035 source itself (fresh machine): online / offline in turns PASS, close() with 40 requests open 0.12 s, 150
  requests on 150 pages PASS (10 of 10 sampled requests equal alone), 6000-token prompt against the reference rel L2
  median 0.0524 / max 0.0805, top-1 13 of 16, bitwise equal between chunks of 3584 and 700 rows. Host: fuzz (60 runs,
  new seed) and ring test pass.
- 400 s of the trace on the s0035 source: 643 of 643 within the service levels, TTFT p50 158 ms / p99 3.33 s / max
  4.63 s, time between tokens p50 35.3 / p99 69.9 ms, longest gap 86 ms, probes 12 of 12, free == total.
- Last measurements on the s0035 source: decode-only gaps through the device loop 11.50 ms at 1 request, 11.87 at 2,
  14.37 at 8, 18.88 at 32; 6 req/s: 585 requests, all finish, TTFT p50 146 ms / p99 3.71 s / max 5.003 s (one first
  token 3 ms past the limit), time between tokens p50 47.2 / p99 79.4 ms.
- End of the run (23.8 of 24.0 agent-hours): engine/ = s0035, workspace committed.

# Extension (2026-11-15 22:35, +8 agent-hours): capacity at 8 req/s

Operator's findings on s0035 (both engines over HTTP, 300 s of the trace): 100% within the service levels at 1-2
req/s, 99.9% at 4, **33.6% at 8 req/s** where SGLang holds 82.3%; steady decode 11.4 ms per token (SGLang 4.7), 64
sequences 3163 tokens/s (SGLang 5406), a 32,000-token prompt 0.76 s to the first token (SGLang 0.41 s). The score is
now the share at 8 req/s (`slo_share_r8`); 2 req/s must stay at 0.99 or above with its stall and tail figures.

## 23.9 - 24.5 agent-hours: s0036, the audit's two defects
- (1) `Engine(max_batch=8)` with 24 requests in flight: the per-slot buffers were sized by max_batch, the device
  scheduler has 64 slots. Sized by the scheduler's slots now (tools/t_small.py PASS). (2) Prefix-cache matches are
  verified: the block's 512 tokens against the tokens the cached page was computed from (kept on card 0 behind the
  cache table), and the node's parent must be the node matched for the block before; a block is inserted only
  under the very page its request uses for the block before; late hits likewise. Fuzz with the key reduced to 5 /
  61 / 1009 values passes (and fails at once without the token comparison).
- **s0036: expected the figures of s0035 at 2 req/s and "what s0035 has" at 8 req/s; measured valid**, the
  evaluator's first 8 req/s figure: **slo_share_r8 0.836** (120 s in process, 780 requests = 6.5 req/s offered; TTFT
  p50 1.58 s, time between tokens p99 78 ms); 2 req/s 1.000 as before.

## 24.5 - 25.6 agent-hours: s0037, where the capacity at 8 req/s went: 64 slots
- Decision: before any kernel work, find what collapses. Numbers of the trace at 8 req/s nominal (300 s): 2120
  requests (7.1 req/s), 58.9k prompt tokens/s offered (37.2k rows/s after cache hits), 1289 output tokens/s. A
  decoding sequence gets one token per pass; passes that carry prompts are ~51 ms; so ~65 sequences decode at any
  moment - and the scheduler had 64 slots and admitted in arrival order. Every new request waited behind the head
  of the ring for a slot.
- The scheduler code on a virtual clock (tools/sched_sim.cpp), pass model now fitted to the evaluator's 8 req/s
  result of s0036 (11.6 ms + 16.0 us per prompt row + 0.07 ms per decode row in a mixed pass; decode-only 11.3 +
  0.18 per row): 0.827 for the measured 0.836 at 120 s, 0.317 for the operator's 0.336 at 300 s (TTFT p50 8.1 / 7.6 s,
  time between tokens p50 51 / 51 ms). With 128 slots and nothing else: 0.991 at 120 s, 0.952 at 300 s (96 slots:
  0.94; 192 and 256: as 128, the page store binds at ~120 sequences).
- Built: S_NSLOT 128 (multiword slot masks, the fixed program's counts, 2.8 MB of convolution state per slot and
  card). Machine, 300 s at 8 req/s in process: 2007 of 2120 = 0.947, TTFT p50 0.44 s / p99 9.6 s, time between tokens
  p50 60 / p99 78 ms, at most 118 in flight: the simulator's figures (0.952; 0.42 s; 9.3 s; 59 / 80 ms; 120).
- **s0037: expected slo_share_r8 ~0.99, measured valid 0.990** (772 of 780; TTFT p50 232 ms / p90 2.9 s / p99 5.0 s,
  time between tokens p50 58.7 / p99 77.3 ms, longest gap 90 ms); 2 req/s 1.000 (TTFT p99 4.03 s, 25.3 / 62 ms);
  steady decode 10.94 / 13.68 ms.

## 25.6 - 26.4 agent-hours: s0038, admission by priority
- Admission in arrival order makes every request wait behind the head of the ring when slots or pages are short.
  Now: all waiting requests of the ring are looked at; those that can still answer within 5 s first, cheapest first
  (prompt rows + 8 per output token); the others in arrival order and only while 24 slots and an eighth of the page
  store stay free (or no prompt is being served, or the ring is three quarters full). The first-token clock of a
  request starts when the scheduler first sees it (it started at admission: requests that had waited looked fresh).
- The fuzz found the first version's flaw in a minute: with a small page store the reserve for late requests was
  larger than the store, and they never got in. Reserve capped at an eighth of the store and dropped when no prompt
  is being served. 400 runs + colliding keys pass.
- Simulator: 8 req/s 300 s 0.952 -> 0.982; 10 req/s 0.17 -> 0.95 (drain 51 s). Machine, 8 req/s 300 s: 2082 of 2120 =
  0.982 (the simulator's number), TTFT p50 0.44 -> 0.19 s, p90 4.2 -> 1.8 s, p99 9.6 -> 12.1 s, tbt 60 / 78 ms.
- **s0038: expected slo_share_r8 ~0.992 with TTFT p50 ~0.13 s, p90 ~1.8 s; measured valid 0.990** (772 of 780, the
  same count as s0037), TTFT p50 183 ms, p90 1.94 s, p99 5.04 s; time between tokens p50 58.6 / p99 77.5 ms; 2 req/s
  1.000 (TTFT p99 4.03 s, 25.3 / 64 ms). The eight late requests are just past 5 s.
- Tried in the simulator, not adopted: the feasibility test with the rate that includes a pass's base and decode rows
  instead of the pure row rate (8 req/s 300 s: 38 -> 34 late, 120 s 6 -> 6: nothing to measure); other pass caps (tmax
  4096, normal pass 2048 - 3584): 34 - 38 late in every case.
- Tried on the device, not adopted: a normal pass of 2048 rows while 48 or more sequences decode (simulated: same
  share, time between tokens p50 59 -> 51 ms). Machine, 8 req/s 120 s: 771 of 780 (s0038: 772), time between tokens p50
  58.6 -> 54.3 ms, TTFT p90 1.94 -> 1.76 s, p99 5.0 -> 5.9 s. One request less on the score for 7% on the median gap:
  left out.
- For the record (s0038 source, 300 s in process): 6 req/s 1591 of 1598 (TTFT p50 164 ms / p99 3.8 s, time between
  tokens 55 / 75 ms); 10 req/s 2323 of 2608 = 0.891 (TTFT p50 0.35 s, p90 5.5 s; every request finishes 53 s after
  the last arrival; longest gap 147 ms). The simulator had 0.946 for 10 req/s: its 0.07 ms per decode row is fitted
  at ~60 decoding sequences and is optimistic at ~120. Scheduler fuzz on the machine's CPUs, 32 seeds x 300 runs: pass.
- Robustness on the s0038 source: slow host (no poll until the device stops, 134,400 tokens) PASS, edge lengths 26 of
  26, online / offline in turns PASS, close with 40 requests open 0.17 s, max_batch 8 with 24 requests PASS, crowd
  with drop_cache PASS; 400 s at 2 req/s 643 of 643 (TTFT p99 3.34 s, time between tokens 35.3 / 67.5 ms), probes 12
  of 12.

## 26.4 - 27.0 agent-hours: s0039, a prompt nobody waits behind
- Operator: a lone 32,000-token prompt answers after 0.76 s (SGLang 0.41 s). Machine (tools/t_lone.py): 808 ms; 13 normal
  passes of 2560 rows, each with its base cost, although no sequence decodes. Rule: while no sequence is decoding
  and the shortest waiting prompt has more rows left than a normal pass, the pass is the largest (3584 rows).
  32,000 tokens 808 -> 752 ms, 8,000 tokens 196 -> 182 ms. (Unconditionally "largest pass while nobody decodes" gave
  the same lone-prompt gain; the condition on the shortest prompt keeps a burst's first, short prompts from waiting
  for a longer first pass.) The simulated replays are the same to four digits.
- **s0039: expected slo_share_r8 0.990 and the 2 req/s figures of s0038; measured valid 0.990** (772 of 780), 2 req/s
  1.000 with TTFT p50 176 ms (the other of its two values) / p99 4.01 s, time between tokens 25.2 / 59.5 ms.
- What the rest of a 32,000-token prompt's 752 ms is: 9 passes x 11.6 ms and 20 us per row, of which the global
  layers' attention over the long context is about a quarter (at position 30,000 half of a pass).
- Tried and dropped: the far-key softmax's first pass with one tensor-memory read of 64 cells per thread instead of
  four of 16 (the idea: reads of tensor memory are tensor-core operations that queue): attention at position 24,000
  310.8 -> 336.6 us per layer. Wider reads are slower.
- 8 req/s over 300 s with schedule probes on the s0039 source: probes 12 of 12 identical (a request's tokens under
  overload, with priority admission, equal the same request alone). Decode-only gaps through the device loop: 64
  sequences 22.2 ms (2880 tokens/s), 96 26.7 ms, 128 27.9 ms (4590 tokens/s).
- Two more timing experiments, both dropped: (1) the far-key softmax without its maximum pass (reference 0; safe
  because |q.k| <= 512 after the clips bounds the scaled score to +-45): attention at position 24,000 310.8 -> 305.9 us
  per layer, 1.6%: the maximum pass is not the softmax's time, and the change would have been a new rounding. (2)
  The home-card mode from 33 rows instead of 65: decode-only passes of 40-64 rows 0.7-0.9 ms slower.
- Why the last few requests at 8 req/s are late (simulated list): every late request is a 32,768-token prompt, the
  last of its burst (sixteen more such prompts answer between 4.0 and 5.0 s); offered prompt tokens per 5 s reach
  387k. Only prompt throughput in a burst moves them.

## 27.0 - 27.6 agent-hours: s0040, two open doubts of the audit
- (4) The page allocator returns page 0 when its free list is empty (admission reserves, so it cannot be; the fuzz
  recomputes the books after every step). If it ever happened it is no longer silent: counted on the device,
  published as `page_alloc_failures`, and poll() raises. (6) submit() refuses a request id that is already open.
- **s0040: expected the figures of s0039, measured valid**: slo_share_r8 0.990 (772 of 780; TTFT p50 186 ms, p90 1.94 s,
  p99 5.06 s; time between tokens 58.5 / 77.5 ms), 2 req/s 1.000 (TTFT p50 172 ms / p99 4.02 s, 25.2 / 62.7 ms, longest
  gap 87 ms), steady decode 11.0 / 13.7 ms.

## 27.6 - 28.3 agent-hours: s0041, the page store from the memory that is really free
- After the slots, the page store binds (~120 sequences x ~18 pages; simulated: 3000 pages instead of 2149 would be
  38 -> 23 late of 2120 at 8 req/s over 300 s). Measured with the engine serving (tools/t_mem.py): 14.7 GB per card
  free. The store was sized before the pass buffers existed, from an estimate of them plus 6 GB of reserve.
- Now sized after everything else is allocated: the driver's free memory less 0.6 GB (swap space, scheduler arrays)
  and 2.7 GB: 2439 pages; 7.7 GB per card free while serving, the same on every card and run. (With 5 GB subtracted
  9.8 GB stayed free: the driver reports ~4.8 GB less at that point than is free later.)
- Simulator: 8 req/s 120 s 6 -> 5 late, 300 s 37 -> 27. Machine: 300 s 2082 -> 2093 of 2120 (0.987: the predicted
  number), TTFT p90 1.82 -> 1.58 s. With this store the slots bind for moments (128 in flight; 160 or 192 slots:
  the same result).
- **s0041: expected slo_share_r8 ~0.994 (775 of 780), measured valid 0.9936 (775 of 780)**: TTFT p50 175 ms, p90 1.71 s,
  p99 4.85 s; time between tokens 58.8 / 80.0 ms; longest gap 90 ms. 2 req/s 1.000 (TTFT p50 177 ms / p99 4.02 s, 25.2 /
  66 ms, longest gap 86 ms); steady decode 10.95 / 13.68 ms.
- 15 minutes at 8 req/s on the s0041 source (in process): 6471 requests, **6393 within the service levels (0.988)**,
  every one finishes (11 s after the last arrival), TTFT p50 180 ms / p90 1.57 s / p99 9.5 s, time between tokens p50
  60.0 / p99 79.1 ms, longest gap 100 ms; probes 6 of 6 identical; 58,866 evictions, 6504 recomputations, 128
  sequences in flight at most, no page allocation failure, free == total afterwards. The share does not decay with
  time: the engine holds this rate.
- Dropped again with the larger store: the shorter normal pass while many sequences decode costs one request of 780
  in the simulator too (5 -> 6 late) for a median gap of 51-54 instead of 58 ms.
- The operator's calibre (the harness's serve_engine.py + client.py around the engine) cannot be run from here: the
  machine's Python has no aiohttp (tools/http_replay.sh is the attempt; not worked around). What the client does that
  the in-process driver does not: it counts a request only if it has finished 60 s after the last arrival (the
  driver waits 90 s). The engine drains 11-18 s after the last arrival at 8 req/s.
- 20 minutes at 2 req/s on the s0041 source: **2120 of 2120** within the service levels (s0033: 2119; the request that
  was at 5.37 s is at 4.89 s with the larger page store), TTFT p50 167 ms / p99 3.26 s, time between tokens p50 49.3 /
  p99 70.8 ms, longest gap 89 ms, probes 6 of 6, free == total. Host fuzz on the final scheduler: 4 seeds x 100 runs
  and 60 runs with colliding keys pass.
- The operator's matrix in process on the s0041 source (300 s per rate): 1 req/s 238 of 238 (TTFT 0.18 / 3.9 s, time
  between tokens 28 / 70 ms, longest gap 76 ms), 2 req/s 483 of 483 (0.17 / 3.4 s, 28 / 70 ms, 87 ms), 4 req/s 1046 of
  1046 (0.15 / 3.4 s, 45 / 73 ms, 88 ms), 8 req/s 2093 of 2120 (0.19 / 9.7 s, 60 / 79 ms, 96 ms). The 1-4 req/s rows
  equal the operator's HTTP figures for s0035: the calibres agree where the engine is not overloaded.

### s0042: the pressure rule's margin grows with the decode load (scheduling)
- The pressure rule takes the largest passes (3584 rows instead of 2560) only while a waiting prompt is projected to
  miss its first token by less than a margin of 0.3 s. With many sequences decoding the projection is optimistic (the
  rate assumes prompt rows only; arrivals of the next burst are not in it), and the late requests at 8 req/s were
  32,768-token prompts that missed by 0.1 - 0.5 s after normal passes early in their burst. Now the margin is 2 s
  while 32 or more sequences decode (S_LOAD_ND), 0.3 s below: 2 req/s rarely has 32 decoding (its median gap is the
  cost of the larger passes, so the rule must not fire there).
- Simulator: 8 req/s 120 s 5 -> 2 late of 780, 300 s 27 -> 20 of 2120, 6 req/s 7 -> 6, 10 req/s 141 -> 103; 2 req/s
  (trace400) unchanged to the digit. Machine before submitting: 778 of 780 at 120 s (TTFT p99 4.85 -> 4.34 s, time
  between tokens 58.8 -> 59.6 / 80.0 -> 82.7 ms), 300 s 2093 -> 2101 of 2120 (0.991; TTFT p99 9.7 -> 4.9 s), probes
  12 of 12 at both rates, 2 req/s 189 of 189; fuzz 60 runs.
- **s0042: expected slo_share_r8 ~0.9974 (778 of 780) and 2 req/s unchanged; measured valid 0.9974 (778 of 780)**:
  TTFT p50 182 ms / p90 1.60 s / p99 4.33 s, time between tokens p50 59.6 / p99 81.9 ms, longest gap 91 ms. 2 req/s
  1.000 (TTFT p50 177 ms / p99 4.03 s, 25.2 / 59.5 ms, longest gap 86 ms); steady decode 10.94 / 13.69 ms. Schedule
  probes 12 of 12, online_vs_step 8 + 8 of 8. The simulator's prediction again to the digit.
- What the last two would cost (simulated, 8 req/s 120 s): both are 32,768-token prompts arriving at 65.0 and 70.0 s,
  in the trace's heaviest stretch (first tokens after 14.7 and 12.4 s). Prompt-row cost -5%: 2 late; -10%: 2; -20%:
  0 (and 20 -> 6 of 2120 over 300 s). No order of service is left that saves them: the feasibility test with a
  load-aware rate (the pass time including base and decode rows) gives the same 2 at 8 req/s and 103 -> 114 at 10.
- After s0042, simulated on its scheduler (host only), to see whether anything short of kernel speed is left:
  - a smaller normal pass while many sequences decode (2304 rows from 48 decoding): the same late requests (2 of 780,
    20 of 2120, 6 at 6 req/s), 2 req/s unchanged, median gap 58.7 -> 56.1 ms at 8 req/s; from 32 decoding or with
    2176 / 2048 rows one more request late at 120 s. Not submitted: -4% of the median gap, with the score one
    request from changing, is not a result.
  - a smaller normal pass at all loads (2304 / 2048 / 1792): 2 req/s TTFT p99 4.08 -> 4.17 / 4.30 / 4.47 s and time
    between tokens p99 64 -> 73 ms: worse where the figures must stay.
  - page store 3000 / 4000 / 8000 pages (the last never evicts in 120 s): still 2 late of 780; over 300 s 20 -> 17 /
    13 / 10 of 2120. So neither memory nor a better eviction order reaches the last two requests; over longer
    windows recomputation (563 of 8286 missed blocks at 120 s) is worth ~0.3 points.
  The score is at what this pass cost allows on this window.
- Host fuzz on the s0042 scheduler: 4 seeds x 100 runs and 60 runs with colliding keys pass; ring test passes.
- The other rates on the s0042 source, 300 s of the trace each, in process (machine epoch 6): 4 req/s 1046 of 1046 (TTFT
  p50 0.13 / p99 3.41 s, time between tokens 45.8 / 73.5 ms, longest gap 88 ms); 6 req/s 1593 of 1598 = 0.997 (0.15 /
  3.27 s, 54.7 / 76.2 ms, 90 ms); **10 req/s 2494 of 2608 = 0.956** (s0038: 0.891; simulator 0.960): TTFT p50 0.28 /
  p90 3.6 s, 69.2 / 82.3 ms, longest gap 94 ms, 128 in flight, every request finishes inside the drain. Probes 6 of 6
  at each rate; free == total afterwards. With 8 req/s (2101 of 2120) the s0042 matrix is complete; 1 and 2 req/s
  were not rerun (the rule differs only from 32 decoding sequences; the evaluator's 2 req/s figures are those of s0041).
- Test set on the s0042 source (device loop, machine epoch 6), all pass: t_stress mix (150 requests on 700 pages: 10 of
  10 sampled requests identical alone and in the crowd, 542 evictions, free == total), edges (26 of 26, prompts 1 to
  34,815 tokens on 300 pages, 186 recomputations), flood (1500 requests, 128 in flight, 10 of 10), slow poller (2
  of 2), t_small (max_batch 8 with 24 requests at once), t_mix (online == online again == offline; an offline call
  with a request open is refused), t_close (0.13 s with 40 requests open).
- 15 minutes at 8 req/s on the s0042 source (in process, machine epoch 7 after a machine replacement of ~7 minutes): 6471
  requests, **6418 within the service levels (0.9918; s0041 0.988)**, every one finishes, TTFT p50 179 ms / p90 1.52 s /
  p99 4.79 s (9.5 s with s0041: the p99 is now inside the service level), time between tokens p50 61.1 / p99 80.0 ms,
  longest gap 94 ms (100), probes 6 of 6 identical, 59,034 evictions, 6614 recomputations, 128 in flight at most, no
  page allocation failure, free == total afterwards.
- Last look at the median time between tokens at 8 req/s (59.6 ms), simulated on the s0042 scheduler: a normal pass
  of 2304 or 2400 rows from 40 - 56 decoding sequences gives 56 - 57 ms (-4%) with the same 2 of 780 and 19 - 20 of
  2120 late at 8 req/s, but the neighbours of every setting lose a request (56 / 2304 and 48 / 2240: 3 late) and at
  10 req/s all of them lose 20 - 30 more (103 -> 122 - 131 late of 2608). Not robust, not taken.
  Why the gain is small whatever the rule: the median gap under load is the pass that carries the offered rows.
  37k prompt rows/s (after cache hits) through passes of 11.6 ms + ~7 ms of decode rows + 16 us per row need, with
  perfectly even arrivals, R = 37 x (18.6 + 0.016 R) -> ~1690 rows per pass and 46 ms per pass. The engine's 59.6 ms
  is that bound plus the trace's bursts; a rule can only move the median between the two, and pays with the tail.
  The median gap at this rate falls with the pass cost (base and per row), nothing else.
- 2 req/s over 300 s on the s0042 source (machine epoch 7), twice: 483 of 483 both times, TTFT p50 0.17 / p99 3.44 and
  3.42 s, time between tokens 28.0 / 72.7 ms, probes 12 of 12, at most 27 sequences in flight: the 2-s margin (from
  32 decoding) never applies at this rate, as intended. Longest gap 102.0 ms in the first run, 86.7 ms in the
  second. The first run had one poll() call of 89 ms (the evaluator's runs: 0.06 - 0.5 ms): a token's time is when
  poll returns, so a stall of the driver's thread is a gap. tools/t_replay_online.py now logs slow polls
  (`POLL_LOG=1`): in the second run three polls of 3 - 25 ms at 16 - 19 s, one of them with no event to return (a
  path that reads two words of host memory), no generation-2 garbage collection in the run. The engine's poll
  touches host memory only and its enqueue threads are native: the stall is the driver process's (my `mk exec` runs
  are not pinned to cores, the machine was minutes old and had just read the checkpoint). Recorded as seen, not explained
  further: the share and the percentiles are the same in both runs.

### s0043: promotion ahead of the shortest-first order (scheduling)
- Reviewing the extension's diff of engine/ (no defect found: ring window, arena, token-verified matches, counters)
  raised a question the run had not asked: prompts are served shortest-first among those still in time, which is not
  the order that loses the fewest deadlines when bursts overlap. A later burst's short prompts (deadline later) go
  before an earlier burst's long one, and that one can miss although serving it first would have cost nobody.
  (Deadline order with the longest dropped when the set is infeasible - Moore-Hodgson - is the optimum for one
  machine; within a burst the deadlines are equal and it is what shortest-first already does.)
- Rule: per step, among the prompts projected to miss in shortest-first order at the largest pass size (margin 0.3
  s), the one with the earliest deadline is served first if, by the same projection, it and every prompt it
  overtakes keep 0.3 s to spare. Otherwise the order stays. One sorted list per step (insertion sort of the prompts
  in time), two linear walks; counter `promotions`.
- Simulator: 8 req/s 120 s 2 -> 1 late of 780 (the request of 65.0 s answers in time), 300 s 20 -> 17 of 2120, 6
  req/s 6 -> 3 of 1598, 10 req/s 103 -> 89 of 2608, 2 req/s identical to the digit; TTFT p90 at 8 req/s 1.63 -> 1.69
  s. Earlier triggers (margin 1 - 3 s) reach 10 - 11 of 2120 at 300 s but move 2 req/s (TTFT p90 1.9 -> 3.4 s): not
  taken.
- Machine, and a lesson about the headline: on machine epoch 7 the s0042 source gives **777** of 780, not the 778 of its
  evaluation (epoch 6): a 32,768-token prompt of the burst at 2.5 s answers after 5.02 s (simulator 4.95 s). That
  request is at the limit by capacity (no order or margin moves it by more than 0.09 s in the simulator), so the
  third digit of the score depends on the machine. A/B on epoch 7, 120 s: s0042 source 777 (late: 2.5 s, 65.0 s, 70.0
  s), promotion 778 (late: 2.5 s, 70.0 s; 7 promotions), probes 12 of 12 at 8 and 2 req/s, 2 req/s 189 of 189 with 0
  promotions. Fuzz: 4 seeds x 100 runs and 60 with colliding keys pass.
- **s0043: expected slo_share_r8 0.9974 (778 of 780; 0.9987 if the request at the limit falls inside), 2 req/s
  unchanged; measured valid 0.9974 (778 of 780)**: TTFT p50 177 ms / p90 1.66 s / p99 4.35 s (max 12.1 s; s0042 17.2
  s), time between tokens 59.7 / 81.3 ms, longest gap 91 ms. 2 req/s 1.000 (TTFT p50 176 ms / p99 4.02 s, 25.1 / 58.6
  ms, longest gap 87 ms). Steady decode 10.72 / 13.57 ms. Same number on the board as s0042, one request more than
  the s0042 source on the same machine.
- After s0043, the request at the limit (8 req/s, the 32,768-token prompt of the burst at 2.5 s: 4.95 s in the
  simulator, inside on machine epoch 6, 5.01 - 5.02 s on epoch 7):
  - the 2-s margin from 20 decoding sequences instead of 32 (S_LOAD_ND; simulator 4.95 -> 4.88 s, 2 req/s unchanged
    in both windows; from 12 the 2 req/s time between tokens p99 moves 64 -> 73 ms): machine epoch 7, 5.01 s, still
    outside, 778 of 780. Not submitted; the constant stays 32.
  - the scheduler's first-token clock starts at the step that first sees a request, up to one pass (60 - 76 ms)
    after submit(); the device clock is the card's timer and the host has none in common with it. Stamping the
    previous step's time instead (the earliest the request can have arrived): simulator 4.95 -> 4.90 s, nothing
    else moves, 10 req/s 89 -> 100 late. Not built into the engine: 0.05 s where the machine-to-machine difference is 0.07.
  - promotion with earlier triggers: the request stays at 4.90 - 4.92 s. Its burst needs 4.9 s at the largest
    passes; it is at the limit by capacity. Whether the board shows 778 or 779 for s0043 is this one request.
- The s0043 source over 300 s at 8 req/s (in process, machine epoch 7): **2103 of 2120 (0.992)**, the simulator's 17 late
  to the request (all prompts of 24,913 - 32,768 tokens in the bursts at 120, 192, 222 and 280 s, plus the two of
  the first 120 s), TTFT p50 0.19 / p90 1.58 / p99 4.52 s, time between tokens 61.2 / 81.1 ms, longest gap 92 ms, 27
  promotions, probes 6 of 6, free == total. (s0042 source on epoch 6: 2101.) Test set on the s0043 source: stress
  mix 10 of 10, edges 26 of 26, flood 10 of 10, max_batch 8 with 24 requests, online/offline mix: all pass.
