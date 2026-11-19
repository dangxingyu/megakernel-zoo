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

## 5.5-5.6 h: register allocation as a knob (not yet submitted when the pod was lost)
- 896 / 1024-thread blocks on the final engine: REG:72 / REG:64 with STACK:112 / 136, slower at both batch sizes
  (B=8 4858 -> 4961 / 5125, B=1 3270 -> 3412 / 3520).
- Compiling phase bodies out of line (`__noinline__`) changes what the GEMV loops get (fixed tokens, B=1 / B=8,
  baseline 3270 / 4858): attention only 3300 / 4843; routing only 3267 / 4802; team phases only 3297 / 4837
  (STACK:104); w2 quantization only 3275 / 4858; attention + routing + w2 quantization 3261 / 4770 (STACK:96,
  bitwise identical results); all phases 3292 / 4788. Kept: attention + routing + w2 quantization.
  The combinations are not additive: this is the compiler's register allocation in one 19 000-instruction
  kernel, so each candidate has to be measured.
- Pod lost (epoch 1 -> 2) during the next A/B (GEMV schedulers out of line); the clock stopped at 5.59 h.

## s0013 (5.8 h, pod epoch 2): 3296 / 5484 us = 1.429x / 1.204x SGLang (predicted ~3290 / ~5460)
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

## 11.1-11.2 h: one more placement fix, submitted as s0019; pod lost during its evaluation
- The w13 scale-factor stores (tensor memory, beside the routing phase) spread over three groups of four idle warps
  (12-23) instead of warps 20-23 alone: bitwise the results of s0018; replay at B=8 5383 -> 5345 us per step (-0.7%):
  the routing phase had been waiting for the nine stores. Soaks with 1, 2, 8 distinct sequences clean.
- The pod was replaced while s0019 was evaluated (clock stopped at 11.23 h); the evaluation is queued for the new pod.

## 11.23 h (clock stopped): s0019 still not evaluated, no pod for more than three hours
- The pod lost at 04:27 has not been replaced by 07:48 (the status went from epoch 3 to epoch 4, both
  "starting"); s0019 is queued and will be evaluated when a pod comes. Nothing was started, stopped or worked
  around on my side.
- engine/ is the s0019 source (git 'tensor-core w13: scale-factor stores ... three warp groups'). It is bitwise
  s0018 in its results; the only difference is which idle warps issue nine tensor-memory stores.
  Prediction on record: in-process step_us at B=8 5434 -> ~5395, HTTP B=8 ~5400 +- 40, B=1 ~3298.
- If s0019 comes back valid: nothing to do but record it (engine/ already is that source). If it does not:
  `git checkout 1ccd66b -- engine/` restores the s0018 source (valid, 3298 / 5440 us).

## s0019 (11.3 h, pod epoch 6): valid, 3307 / 5390 us = 1.424x / 1.225x (predicted step_us at B=8 ~5395, HTTP ~5400 +- 40, B=1 ~3298)
- Evaluated after the pod came back (4.5 h of outage, not counted). In process at B=8: step 5433.5 -> 5392.4 us
  (-0.76%, as the replay said), W1-like 5652 -> 5612, generate 5443 -> 5403; over HTTP 5440 -> 5390, the best B=8
  figure of the run. B=1 on this pod: step 3257 against 3250 on the previous pod with the same plain kernel (HTTP
  3307 against 3298): pod-to-pod spread of ~0.3%, not the change (the B=1 kernel does not contain it).

## s0020 (11.5 h): valid, 3307 / 5378 us = 1.424x / 1.228x (predicted step_us at B=8 ~5380, HTTP ~5380 +- 40)
- The w13 quantization emit of the tensor-core kernel inlined in its team phase (it was out of line from the time
  both paths shared a kernel). Bitwise the results of s0019. In process at B=8: step 5392.4 -> 5380.9 us (-0.21%, the
  replay had said -0.22%), HTTP 5390 -> 5378: the best B=8 figure of the run. B=1 unchanged (3307.3 -> 3307.5 on
  this pod; the plain kernel).

## End of the run (11.5 h; engine/ = s0020)
- Final engine: s0020, 3307 us per token at B=1 and 5378 us per step at B=8 over HTTP on the last pod (1.424x /
  1.228x SGLang's 4711 / 6604); the same B=1 kernel measured 3296-3300 us on the previous pod (s0013, s0016-s0018).
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

## 11.8 h: the final engine (s0020 source) checked once more on the last pod
- Float64 reference regenerated on this pod (tools/ref_dump.py, 2 sequences x 12 positions): tensor-core kernel
  forced at B=1/2 (MK_TC_MINB=1) rel median 0.0525 / max 0.0958 / top-1 100%; inside a batch of 8 with a large
  union (t_gate8) 0.0507 / 0.0927 / 100%.
- Soaks, 12 000 steps each without host synchronisation: B=4 (1 distinct sequence), B=5 (2), B=6 (6), B=7 (3),
  B=8 (3), and before the submission B=8 with 1, 2 and 8 distinct (20 000 each): no stall. The evaluator's
  timing-stage call pattern at batch 1 then 8, twice: no stall.
