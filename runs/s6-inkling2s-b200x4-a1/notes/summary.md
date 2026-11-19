# Summary: an Inkling-2-Small decode engine on one GB200 node (4x B200)

Final (24 h). Best submission: **s0013, track A, http_step_us_b1 = 3624 us** (B=8 6955 us). `engine/` holds s0013
(git tag s0013). No machine access from 14.0 h to the end of the run.

| | SGLang PR #50186 (HTTP) | this engine s0013 (HTTP) | ratio |
|---|---|---|---|
| B=1 us/token | 4711 | 3624 | 1.30x faster |
| B=8 us/step  | 6604 | 6955 | 0.95x (slower) |

The project's claim (>= 1.10x at both batch sizes) holds at B=1, not at B=8.
All submissions are track A: the checkpoint's formats (NVFP4 / MXFP8 experts with their scales, every expert input
quantized as the model defines, BF16 elsewhere, FP32 where the specification uses it); only layout, split and
kernel organisation differ. No track-B (format) change was submitted; the 12-bit exact BF16 code (F1) is verified
bit for bit on all dense weights but its decode path was not made fast (not pursued: dense GEMVs are latency-bound).

## The design (what is specific to this model)
- One persistent kernel per card per decode step; the four cards run the same op program and exchange data through
  peer memory. B=1 has its own kernel ("red" mode): every block computes the per-token vector phases (mixes, norms,
  convolutions, routing, quantization) redundantly in shared memory, so the only grid barriers left are around
  attention. B>=2 ("std" mode): one block per token for vector phases, grid barriers in between.
- Split: attention TP4 by heads (12 q / 2 kv heads per card); experts TP4 *inside* every expert (576 of 2304
  intermediate per card), so all four cards read exactly 1/4 of every chosen expert: perfect balance at B=1 with
  no token all-to-all. The price is the w2-input NVFP4 quantization needing the expert's global amax: one 4-float
  exchange per (expert, token). Router, residual mixes and convolutions replicated.
- Block attention residuals (no residual stream, 8 blocks of 6 layers, softmax mixes over finished blocks): the
  finished blocks' statistics are computed before the cross-card wait of each layer (off the critical path).
- Cross-card exchange: every value travels with its tag in one 8-byte relaxed system-scope store (LL); consumers
  poll their own elements. No fences, counters or barriers for the 4 exchanges per layer.
- NVFP4 GEMV: cvt.rn.f16x2.e2m1x2 + HMUL2 by the E4M3 block scale (exact in f16) + mma.m16n8k16; MXFP8 via
  mma.m16n8k32.e4m3 with UE8M0 scales per 32-block in FP32. Four-over-six activation rounding with the hardware
  e2m1 conversion (bitwise identical to the software rounding).

## What moved the B=1 number (HTTP us/token), in order
6564 (s0002) -> 5324 (768 threads, contiguous GEMV ranges, block-per-strip GEMVs) -> 4778 (LL exchange,
slot-invariant FMAs) -> 4581 (HW e2m1, 64-position chunks) -> 4541 (split-K router) -> 4311 (all loads in flight
before a spin; redux top-8; prefetch issued by an idle warp) -> 4260 (load depth per mode) -> 4078 (attention starts
on a counter of the qkv strips it reads; acq_rel hand-offs) -> 4049 (paired o_proj strips; B>=2 race fix) ->
3959 (register relief: 32-bit GEMV indices, launch state in shared memory; two-chain quantization) -> 3624
(the cards' tensor-parallel partials cross as bf16 pairs with tags instead of fp32: half the bytes every block polls).
Not yet submitted at 14.5 h (a build mistake made s0014 fail, then the loop service went down): B=8 ~6680 expected from
whole-region K/V prefetch and routing-masked w2 inputs (in process 7064 -> 6782 us).

## Lessons that generalise (each measured; journal.md has the numbers)
1. At B=1 this is a latency problem, not a bandwidth one (MBU 0.155). The wins came from removing serial L2 round
   trips on the critical path: a spin loop blocks every load behind it (batch the loads, then spin); a slow-issue
   instruction (bulk prefetch ~0.5 us) on a warp that the next __syncthreads waits for delays the whole block;
   a loop whose load waits for the previous iteration's arithmetic is a chain of L2 trips.
2. Replace a grid barrier by a counter of exactly the producers a consumer reads, and order the producers so those
   come first (qkv: attention needs 140 of 237 strips; -174 us).
3. The red kernel sits at the 80-register cap of 768-thread blocks: every extra live value spills, and most
   "more parallelism" changes (deeper unrolls, two strips at once, fused phases) lost for that reason. Freeing
   registers (64->32-bit indices, launch state in shared memory) won.
4. In the B=1 kernel every block computes the per-token vector phases redundantly, so every global read in them is
   multiplied by 152: ~400 KB per block per layer, ~64 MB of L2 traffic per card per layer. Halving the polled
   cross-card bytes gave -320 us per token; L2 bandwidth, not HBM, is the shared resource there. A second hop to cut
   the rest (two-level all-reduce) cost more than it saved (+150 us).
5. Random token ids route alike at B=8 (union 47 vs ~30 experts): profile on real text.
6. A race can hide behind timing: s0010's early attention also applied at B>=2, where the combine read rows of the
   second qkv wave without waiting; all gates and bitwise checks passed. Found by review, fixed in s0011.

## What did not work (measured, kept as build options)
L2 bulk prefetch of each warp's GEMV range; L2 sector hints; cp.async rings for w2; block-contiguous GEMV with
cross-block hand-off; whole-strip work granules; per-strip amax slots; fused w2-input quantization; 128-position
chunks at long contexts; preloading conv inputs before the spin (B=1).

## Why B=8 is slower than SGLang
Per layer ~146 us: the expert GEMVs read ~200 MB per card (47 distinct experts) at 3-4.7 TB/s plus epilogue and
hand-off costs (~66 us vs ~34 us at bandwidth), and the per-token vector phases run on 8 blocks with grid barriers
(~35 us). A B=8 design would need the vector phases split over many blocks per token and the expert GEMVs
restructured around shared activation fragments; not reached in this run.

## Unevaluated work (git tag b8-unevaluated)
B>=2 only, bitwise identical to s0013: whole-region K/V L2 prefetch (the non-token blocks' tiny bulk prefetches had
delayed the attn_in barrier by ~4.6 us) and a routing mask so w2 reads zeros for unrouted (expert, token) columns and
quant2 skips them. In process B=8 7064 -> 6782 us, B=1 equal (e0215). Its first submission (s0014) failed to build
(my error: a buffer made conditional, its users not); the fixed build never reached the evaluator (machine outage).
Also there: W13_EARLYPF (w13 L2 prefetch per expert during the top-8, by the idle last warp), off and untested.

## What I would try next
1. Submit b8-unevaluated (expected B=8 ~6680 us over HTTP, B=1 unchanged), then W13_EARLYPF at B=1.
2. B=1: the remaining per-layer time is ~73 us, mostly latency of ~10 dependent phases. Biggest items: w13 + amax
   wait + quant2 (~24 us), the attention-chunk combine every block redoes (~4.5 us, L2-bound: all 152 blocks read
   the same ~55 KB), qkv + attention (~15 us). Ideas: cut the redundant L2 reads of the red-mode vector phases
   (conv history/taps, fp32 mixing queries, chunk partials) without adding a hop; relieve the 80-register cap
   (spills in the cross-card spin, residual mix and quantization; tools/spills.sh) so deeper load pipelines fit.
3. B=8 (needs ~-950 us for 1.10x): split each token's vector phases over many blocks instead of one block per
   token; restructure w2 so activation fragments come from shared memory (it streams at ~2.5 TB/s); remove grid
   barriers where only a few producers matter (as done for qkv -> attention at B=1).
4. Track B: the 12-bit exact BF16 code (verified on all dense weights) for the unembedding (~78 us/step,
   bandwidth-bound at B=1) once its decode path is made fast.
