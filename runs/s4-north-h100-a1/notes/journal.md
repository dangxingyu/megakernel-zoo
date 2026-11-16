# Journal (run s4-north-h100-a1)

## 0. Setup measurements (before any engine)

- `bench/gemv_bench.cu` (e0004/e0005): a tensor-core GEMV whose weights are stored in mma fragment order (one
  16-byte load per thread per m16n8k16) streams at 3085-3120 GB/s with 16 warps per SM and >= 2 loads in flight
  per thread; 8 warps per SM need 8-16 in flight. One warp alone moves ~5.5 GB/s (8 in flight) to ~8.7 GB/s (16):
  the round trip to HBM is ~0.6-0.9 us, so a stage needs ~4800 warp-loads in flight to saturate the card.
  Reading the x operand from global memory instead of shared memory costs nothing at 16 warps/SM (L1 serves it).
- Same file: a grid-wide barrier through one atomic word costs <= 0.9 us; 20000 rounds x 132 blocks x 1024 words
  of "another SM wrote it, I read it with a plain load" gave 0 stale reads.
- Census (harness weights_census.py, e0002; study/census2.py, e0006): see notes/checkpoint.md.

## 1. v1 megakernel (track A)

Design: one persistent kernel per step, 132 blocks x 16 warps, block 0 leads. Per layer 4 stages with a grid
barrier between them: A = q|k|v + router rows (K split in 4), B = attention (block tasks) || gate|up of the chosen
experts, C = o_proj (K split in 4) || down of the chosen experts, D = residual add. Head = 16384 row tiles.
Batch lives in the 8 columns of the mma B operand, so batch 1 and batch 8 are the same instructions.

Dev test (tools/test_engine.py against tools/make_ref.py, own seeds, e0009):
- single vs float32 reference: rel-L2 median 0.056 (HF BF16 itself: 0.068), top-1 93.75% (HF: 93.9%)
- batch of 8: median 0.070, max 1.27; single == batched bitwise
- expert choices forced to the reference's: median 0.0113, top-1 99.3% (the arithmetic is right; the rest is routing)
- generate() == step-by-step greedy on 8/8 sequences
- 3492 us/step at B=1, 8656 us at B=8 (step); generate 3499 / 8792. Expected ~2600 from bytes/bandwidth + barriers:
  the gap is to be profiled next.

### s0001 (v1): valid, 3497 us B=1 (generate), 3501 step, 8948 us B=8. Prediction 3500 / 8700: B=1 as predicted.
Leader timeline (clock64 on the leader's SM, e0010), us per layer at B=1: A 11.6, B 22.7 (8.7 of it waiting for the
attention blocks), C 18.7, D 4.8, norm 3.4, route 4.6 = 65.8 against 36.5 of pure streaming. Every handoff through
global memory (atomic, poll, re-read) costs about 1 us and a barrier is ~5 of them. The head alone runs at the
bandwidth bound (351 us for 1.07 GB).

## 2. v2: static per-block schedule (track A)

Nothing is claimed dynamically any more. Every block computes the RMSNorm itself into shared memory; q|k|v are
1280 chunks, one per warp; block 0 does router + top-k; blocks 1..4B do attention; block r owns residual row tile r
in stage C (o_proj + all chosen experts' down rows, combined in shared memory, no D stage). Blocks meet through 4
monotonic counters. Dev (e0011): 3056 us B=1, 7845 us B=8; gates as before; forced routing 0.0111.
Leader timeline per layer at B=1: A 15.0 (the router GEMV on one block: 4 dependent load groups, each ~3.4 us
because the HBM queue is saturated by the q|k|v chunks of the other blocks), B 19.5, C 17.2, norm 1.1, waits 2.3.
Lesson: under saturation one round trip costs (bytes in flight)/(bandwidth), so a dependent chain of N load groups
costs N x 3-5 us. Critical small work must be one load group deep, and the next stage's first loads should be in
flight before the barrier.

### s0002 (v2): valid, 3050 us B=1, 7923 us B=8 (predicted 3056 / 7845). h0_step and h0_w1_like passed.

## 3. What a handoff costs on this card (measurements that changed the design)

- bench/l2_bench2.cu (e0022): while 131 SMs stream with 16 loads in flight per thread, ONE small operation of an
  idle SM takes: L2-resident load ~3 us, atomicAdd ~2.3 us, store + __threadfence ~10-13 us. On a quiet card the
  same operations take 0.2-0.45 us; with 32 SMs streaming 0.2-0.5 us. So the L2/HBM queue is one FIFO for everybody:
  latency of anything = (bytes in flight) / 3.1 TB/s. `ld.global.cs` changes little; `prefetch.global.L2` works.
- Consequence 1: "issue the next stage's loads before the barrier" (tried as v3, e0012-e0016) does not help: the
  barrier's own atomics, polls and fences then wait in the full queue. 3370 us (worse than v2).
- Consequence 2: keep the queue shallow. 16 warps/SM x 2 loads in flight (2.1 MB) still saturates the card
  (bench/gemv_bench.cu: 3041-3079 GB/s) and one round trip is ~0.7 us instead of ~5.5.
- Without any __threadfence the engine is wrong (single != batched, generate != step, e0027): a store and a later
  atomic of the same SM are not ordered at L2. Fences stay, but only in the warps that stored, right after the store.
- bench/fair_bench.cu (e0031): 24 of the 132 SMs (12 pairs) get 22% more of the queue than the other 108 (they
  finish an equal share in 0.81 of the time); which block indices they are changes from launch to launch. A static
  equal split therefore has a ~19% early-finish spread between blocks.
- A field read through a pointer to global memory (M->x, LW.o, dyn.slot_expert[u]) is a round trip each time; the
  whole pointer table now lives in shared memory. ptxas reports 128 registers (the cap for 512 threads/SM).
- Data-dependent branches are very slow in the serial sections (a rank-count top-8 with `a > b || (a == b && i < j)`
  took 9.3 us; the same with a warp max over sortable integer keys 1.7 us).

## 4. v4 (track A): balanced static schedule, shallow queue

Stage A: blocks 1..131 read 2 whole q|k|v tiles + half of a shared tile (20 k-steps per warp, exactly balanced);
the 8 router tiles are 16 such halves and are read first; the leader does nothing but routing (polls the stamped
router sums, top 8 by a warp max, publishes), then its own half tile. Stage B/C: every tile is 16 quanta, one per
warp, summed in shared memory in a fixed order (bitwise batch-invariant). Dev (e0040): 2848 us B=1, 7254 us B=8;
single == batched bitwise, generate == step, forced routing 0.0113.
Leader timeline per layer at B=1: route chain 10.1 (of it 6.4 waiting for the router rows), A rest 1.5, route/dyn
handoff 3.3, gate|up 15.2, B->C handoff 3.3, down+o 13.8, C->A handoff 3.8 (incl. norm 1.7).

### s0005 (v4) and the change of metric
The operator switched the score to HTTP (prompt 64 + 1024 new tokens, contexts up to 1088) while v4 was being
built. s0005: http 3196 us B=1 / 7615 us B=8, in process 2842 (contexts 64-320). Prediction (2850 / 7250 in
process) held in process; over HTTP v4 was no better than v2 (3169) because attention on one block is slow at long
contexts (in process at contexts 800-1056: 3470 us).

## 5. v5 (track A): attention over 8 chunks, device loop behind submit/poll

Measurements that shaped it:
- bench/hop_bench.cu (e0048): one dependent operation of an idle SM under the stream: L2-resident load 0.6-1.0 us,
  cold load 1.2-1.5 us, atomic 0.3-0.7 us, store+fence+load 7-9 us; all independent of the stream's queue depth
  (1 to 16 loads in flight per thread). Quiet card: 0.15 / 0.35 / 0.2 / 0.57 us.
- An SM under the stream receives only its share of the queue (~24 GB/s; bench/nb_bench.cu: 132 blocks 3113 GB/s,
  100 blocks 2786, 64 blocks 2855, 32 blocks 1895). The K and V of one (sequence, K/V head) at T=930 are 476 KB:
  20 us on one SM. Hence attention is cut into 8 chunks of position tiles (tile P -> chunk P % 8), each complete in
  itself (own max, weights exp(s - m_c) as bf16, float32 sum, weighted V sum) and merged at the end
  (sum_c e^(m_c-M) o_c / sum_c e^(m_c-M) z_c). At batch 1 the 8 chunks of a task run on 8 blocks. Per-chunk
  arithmetic and merge order are the same for every batch size: single == batched stays bitwise.
- A branch or a dependent store between two loads makes them two round trips (attention exchange 9.4 -> 3.5 us
  when its 16 loads became straight-line code).
- L1 coherence (e0076-e0090): with every hand-over store made an atomicExch (no fence anywhere) the engine is
  nondeterministic; ONE __threadfence per layer by thread 0 of every block makes it deterministic again. So an SM's
  L1 keeps stale lines across another SM's atomic, and a fence of the SM flushes it. The all-atomic build was also
  slower (3040-3200 us) than plain stores + fence at stage ends (2920): kept the latter, atomics only in attention.
- Profiling probes must not touch global memory (a counter in global memory is a round trip per probe).

Dev (e0094/e0095): B=1 2921-2950 us at contexts 64-320 and 2955 at 800-1056 (v4: 2848 / 3470); B=8 7365 / 8240.
Serving through submit/poll (one launch per cohort, tokens polled from a second stream): tokens equal to the
step-by-step greedy ones, 2945 us/token at B=1, 7046-7070 at B=8 (contexts 32-128).

### s0006 (v5, track A): valid, http 2933 us B=1 / 7257 us B=8 (predicted 2960 / 7800). In process generate 2901.
B=8 came out better than predicted: the serving path runs contexts 64..1088 and my 8240 figure was contexts 800-1056.

## 6. Track B: 12-bit weights (v6)

- bench/w12_bench.cu (e0097): widening 12-bit weights to bf16 in registers keeps up with the stream: 1.99e12
  weights/s (2992 GB/s of 12-bit data) against 1.55e12 for BF16, with two k-step pairs in flight; a shared-memory
  lookup table for the high nibbles is slower (1.71e12), one pair in flight too (1.64e12).
- Format and census: notes/checkpoint.md sections 3-4. The 4-bit exponent code sits directly in the bf16 exponent
  field (values around 2^-120); the x operand is stored times 2^96, so the tensor core's products are the true
  products times 2^(96 - r_i), undone per row afterwards. No "add the base exponent" step in the decode.
- Dev (e0110): 2590 us B=1 (track A at the same commit: 2921-2950), 6648 us B=8 (7365). Serving 2588-2594 / 6577-6585.
  single == batched bitwise, forced routing 0.0113, generate == step.
- The first version (headers read when a tile is finished, x of an escape read from global memory in a loop) gave
  only 2741: a loop with a load and a branch per escape is a round trip per escape. Sorting a tile's escapes by row
  group, fetching the header with the quanta and (batch 1) gathering the x values early brought it to 2590.
- Streaming phases shrank 15-20% (head 351 -> 283 us), the handoffs did not shrink at all: they are now a third of
  the step.

### s0007 (track B v6): valid, http 2628 us B=1 / 6938 us B=8 (predicted 2610 / 6550). beats_cohere passed.
B=1 as predicted. B=8 missed by 390 us: the prediction was the in-process figure at contexts 64-320; at contexts
800-992 the same engine needs 7199 us (in process), because one block per attention task then reads 476 KB of K/V
at its share of the queue while the other 100 blocks finish gate|up.

## 7. Things tried after s0007 that did NOT help (all measured against the s0007 kernel: 2590-2616 / 6650 us in process)

| change | B=1 | B=8 | why it was expected to help |
|---|---|---|---|
| attention blocks 64 at batch 8 (2 per task) | - | 7827 (from 7365, BF16 build) | halve the K/V bytes per SM |
| role classes with gate|up shares for attention blocks, 2 or 4 blocks per task at batch 8 (sweep e0112/e0113) | 2735-2860 | 6972-7762 | attention blocks join the stream when done |
| attention "pair mode": two chunks per pass at batch 8 (e0123, e0129) | 2660 | 6672, 7170-7338 at long context (7199 before) | halve the passes of the 8-chunk loop |
| residual handed over with a stamp in every element instead of fence + counter (e0132) | 2765 | 7161 | two round trips fewer per layer; but every block then polls with all 512 threads and floods the queue |
| RMSNorm totals added by 8 threads instead of all 512, one more __syncthreads (e0133, e0142 A/B twice) | 2681-2683 | 6673-6676 | fewer instructions; the leader then waits 1 us longer for the router rows, reason not found |
| un-inlining attention and routing (e0122) | 3455 | 7443 | fewer live registers; the calls spill 416 bytes instead |
| letting the router rows into the queue before the q|k|v burst (e0144-e0146) | 2670-2732 | 6701-6793 | shorter route chain |
| every hand-over store as atomicExch, no fence (e0079-e0090) | 3040-3207 | 7354-7497 | fence costs; needs one fence per layer anyway to flush L1 |

What these have in common: the per-layer handoffs (about 18 us of 46 at batch 1 on track B) react to small changes
in ways the measurements above did not explain (same tensor-core loops in the disassembly, tools/sass_stats.py;
a __syncthreads alone costs 16 ns, bench/sync_bench.cu). Further work goes into mechanisms that remove bytes.

A speculative start of gate|up for the previous token's experts (4.62 of 8 are chosen again, notes/checkpoint.md
section 5) was built once (e0138): results bit-identical, but 2927-2964 us at batch 1 against 2880 for the same
build with speculation off and 2606 without the bookkeeping. The bookkeeping (hit map, prefix sums, three more
barriers and a fenced flush) cost more than the 4 us of router latency it fills. Not kept; a leaner version
(the leader publishes the lists) is possible.

## 8. Layer 0 from token tables (track B)
Mechanism and measurement: notes/checkpoint.md section 6. Results are bitwise batch-invariant and the forced-routing
distance to the float32 reference is unchanged (0.0112). In process: -61 us at batch 8, +13 us at batch 1.

### s0008 (track B, layer-0 tables): valid, http 2598 us B=1 / 6698 us B=8 (predicted 2630 / 6880).
Better than predicted at batch 8 (-240 us against s0007, in process only -61): the serving run has long contexts,
where layer 0's attention used to wait behind the dense MLP's 38 MB.

## 9. Where the non-streaming time goes: ablation builds (e0151-e0156)

Three builds of the s0008 kernel with work removed (results wrong, timings informative; in process, us per step):

| build | B=1 | B=8 |
|---|---|---|
| full | 2588 | 6603 |
| no weight streaming (every quantum returns at once, everything else as is) | 1170 | 2578 |
| no weight streaming, no attention | 826 | 1479 |
| no attention (streaming as is) | 2552 | 6223 |

So 826 us per step at batch 1 (16.5 us per layer) and 1479 us at batch 8 (30 us per layer) are neither weight bytes
nor attention: they are the per-layer skeleton (norm, router chain, shared-memory sums, residual add, signals).
Leader timeline of the skeleton build at batch 1, us per layer: residual add + signal 2.3, waiting for the router
rows 2.1, gate|up finish 1.7, stage C quanta bookkeeping 1.7, top-8 1.35, norm 1.35, stage A 1.25, gate|up quanta
bookkeeping 1.0, union/slots 0.8, waits 0.55-0.8 each. At batch 8: norm 6.1, gate|up bookkeeping 5.4, residual
add 3.5, stage C bookkeeping 2.7.
An SM issues only about one warp-instruction per cycle when all 16 warps run (bench/sync_bench.cu: 400
warp-instructions per round take 0.21 us), so a section that every warp executes costs 16x its length.
Changes measured on the skeleton build transfer to the full build:
- residual add: warp q takes element q of the row tile instead of warp 0 taking all four (and only batch columns in
  use): skeleton 826 -> 774 us, full build 2588 -> 2534 us at batch 1 (e0157-e0159).
The earlier "unexplained" reactions to small changes were measured on the full build only; the skeleton build is the
better instrument.

## 10. L2 as a staging buffer (plan written while the pod was paused, before any measurement)

Observation from section 9: stream time and skeleton time ADD (full 2534 = about 1600 of stream at 3.1 TB/s for the
5.0 GB of 12-bit weights + about 800 of skeleton + attention). While a block is in the skeleton (norm, routing,
sums, hand-overs) it has no load in flight, so HBM idles for 16-19 of the 46 us of a layer.
Per layer the weights are 85 MB: 28.3 MB are known long before they are used (q|k|v 15.7, o_proj 12.6), 18.9 MB
(down of the chosen experts) are known when routing is, and only gate|up (37.7 MB) is needed the moment it is known.
The card's L2 is 50 MB. Idea: a block that reaches a barrier issues `prefetch.global.L2` (no register result, so
nothing waits) for what it will read after the barrier; the bytes travel during the skeleton and the next stage's
loads are L2 hits. If it works as hoped a layer costs about 18 us (gate|up + down from HBM) + 10 us of skeleton
during which the static 28 MB arrive: 28-31 us instead of 46, i.e. 1650-1800 us per step at batch 1.
v3 tried this with real loads and lost (section 3): the barrier's fences and polls waited behind the loads.
What has to be true, each measured by bench/pf_bench.cu before the engine is touched:
 1. L2-resident data is read much faster than 3.1 TB/s (else nothing is gained by staging);
 2. a prefetch returns at once and its bytes arrive at about the HBM rate; which stride covers a region;
 3. a region of 12-25 MB survives while 15-40 MB of other weights stream through (L2 replacement);
 4. hand-over operations stay cheap while other SMs only prefetch (a fence under a load stream costs 7-9 us);
 5. if fences are the problem: atomicExch stores + volatile loads without any fence are never stale.
Engine switches prepared (-DPF=mask, off by default): 1 = next layer's stage A share at the end of stage C,
2 = down tiles (+ o_proj) at the end of stage B, 4 = o_proj at the end of stage A.

### What the card said about section 10 (bench/pf_bench.cu, e0161, e0167)
1. L2-resident data: 132 SMs read 8.7-43 MB at 13-15 TB/s (4.7x the HBM rate), 52 MB at 4-7 TB/s, 69 MB at the HBM rate.
   So about 40 MB of the 50 MB are usable as a buffer.
2. `prefetch.global.L2` does NOT stage data: 17.3 MB cold 6.6-7.0 us; after prefetching every 16/32/64/128/256/512 B
   and waiting 0..32 us the read still takes 4.8-5.8 us. 131 SMs issuing only prefetches "touch" 10 TB/s: they are
   dropped. A real read does stage it (same region again: 1.3-1.6 us) but the thread that issued it waits for it.
3. Retention is first in, first out: 13 MB survive 17 MB of other weights (1.0 us), half survive 26 MB, none 35 MB;
   26 MB survive 8.7 MB, none 26 MB.
4. One SM's hand-over operations while the others stream / only prefetch / are idle: resident load 0.81 / 0.29 / 0.15 us,
   atomic 0.31 / 0.47 / 0.21, store+fence 7.5 / 2.7 / 0.59, fence alone 8.3 / 2.8 / 0.61, 16 exchanges 1.9 / 2.3 / 0.84.
   A fence costs the same with or without a store of its own.
5. atomicExch stores were never stale for any kind of reader (20000 rounds, with and without prefetch traffic); plain
   stores without a fence are stale as soon as there is traffic.
Conclusion: the staging idea as written (prefetch at the barriers, -DPF) is dead on this card: the instruction that
would make it free does nothing useful. Staging with real loads needs threads that wait for them, i.e. warps that
run ahead of the barrier, and every fence that then happens under traffic costs 7-8 us instead of 0.6. Not pursued
further for now; the -DPF hooks stay in the kernel, off.

## 11. s0009: skeleton trims (track B)
Dev: skeleton build 774 -> 747 us (B=1), 1453 -> 1384 (B=8) with top-8 by `__reduce_max_sync`, gate|up tile count once,
h1 only at batch > 1, 2^96 folded into the norm weight, norm totals once per column at batch > 1. Full build: 2534-2557
at B=1, 6550-6650 at B=8 depending on the build: ptxas reports 96, 112 or 144 bytes of stack for otherwise equal layer
code (register spills at the 128-register cap), and the step time follows it by up to 40 us.
### s0009: valid, http 2572 us B=1 / 6699 us B=8 (predicted 2555-2575 / 6650-6700). In process 2557.

## 12. Certified int8 sketch of the head on the greedy path (track B)
The certified LM head is known (arXiv 2611.02501); what is specific here is the form the sketch takes on this card.
- The integer tensor-core instruction m16n8k32 (s8) multiplies 512 weights per 512-byte load with no widening at all
  (the 12-bit path spends ~50 instructions per k-step pair on widening and is instruction-bound, see below). Fragment
  layout found by probing (bench/mma8_probe.cu, checked by bench/mma8_test.cu): thread (g, t): a0 = row g, k = 8t..8t+3;
  a1 = row g+8, same k; a2, a3 = k = 8t+4..8t+7; b0, b1 = k = 8t..8t+7 of column g.
- Row i: s_i = max|w_i|/127, q_i = round(w_i/s_i). The input as two int8 digits x = t(128 hi + lo) + r, so the integer
  sums are exact and |logit_i - s_i t (128<q_i,hi> + <q_i,lo>)| <= en_i ||x|| + qn_i ||r|| with en_i >= ||w_i - s_i q_i||
  + 2^-10 ||w_i|| (the float32 rounding of the exact path), qn_i >= ||s_i q_i||, norms in float64 rounded up at load.
- Pass 1: 541 MB instead of 805 MB, every row gets a lower and an upper bound. One exchange between the blocks gives
  the best lower bound per sequence. Pass 2: tiles with a row whose upper bound reaches it are computed with the
  exact head's arithmetic.
- Measured (e0171-e0178, 256 greedy tokens after prompts of 16 and 64, B=1 and B=8): tokens identical to step-by-step
  greedy in all 4608 cases; 1.2-1.5 tiles of 16384 computed exactly per sequence and step.
- A lone warp needs 35 us for one exact tile, with 2 or 8 pairs of loads in flight and also when the other 15 warps
  have read the tile before it: it is the widening of the 12-bit weights at one instruction every few cycles, not the
  memory. With all 16 warps widening 1/16 of the tile into shared memory and the owner running the 128 tensor-core
  steps on the fragments (same order, same bits): 5.7 us per tile.
- Leader timeline at B=1: head 305 -> 192 us, wait for the token 14 -> ~18. In process generate 2453 against step 2551.

### s0010 (track B, certified int8 sketch head): valid, http 2474 us B=1 / 6616 us B=8 (predicted 2470-2490 / 6600-6650).

## 13. Stage B start balance (s0011)
Instrument: every block's time in the three per-layer barrier waits (tools/test_engine.py --prof, M->prof[64 + 4j + k]).
At s0010, B=1: ordinary blocks waited 4.1-4.2 us per layer at the stage B barrier, attention blocks 1.0, the leader
1.2; at B=8 attention blocks waited 20 us (their gate|up share was 0 there) and ordinary blocks 4.3. Reason: every
class starts stage B at a different time (the leader after its routing chain, attention blocks after their task,
whose length grows with the context), but shares were fixed (3 : 1 : 3).
Changes, each measured alone on one card (e0185-e0205):
- the leader reads its q|k|v half tile first, stores it with exchanges (a fence under the stream costs 7 us) and
  signals stage A, then routes: attention no longer waits for the routing chain. 2667 -> 2553 us at B=1 (same card,
  same build otherwise; LEADER_FIRST=0/1).
- the union of the chosen experts goes out first as four stamped 64-bit words; each block derives its slots from
  it; per-sequence slots, weights and masks follow and are read at the start of stage C. (Alone: no gain; it is what
  makes the next point possible: the leader's delay is now short and predictable.)
- gate|up shares per block from a model of the start delays (leader: its routing chain; attention block: 26 + chunks
  x (14 + 0.6 K tiles per chunk) in 1/8 tile, + merge), recomputed once per step for every union size, ranges of
  consecutive tiles. Fitted on the per-class waits at contexts 64-320 and 800-1056; the result is flat within the
  noise (+-15 us) for the leader delay 8-16 and attention base 26-40 (e0214-e0221).
Dev: wait B at B=1 4.2 -> 2.8 us per layer (all classes 2.4-2.8), at B=8 attention blocks 20 -> 4-6; in process
generate 2456 -> 2395 at B=1, serve 2390 / 5913 (B=1 / B=8).
### s0011: valid, http 2412 us B=1 / 6421 us B=8 (predicted 2400-2420 / 6350-6500).

Incident: a Python patch script opened engine/mk.cu for writing before an assertion failed and left it empty. It was
rebuilt from the last commit by replaying the edits recorded in this session; every patch script now writes to a
temporary file and renames it.

## 14. The 12-bit decode was the bottleneck of the stream (s0012)
Profile of an ordinary block (PROFBLK=64): inside stage B it read 214 KB per layer in 14 us = 15 GB/s, about 65% of
its share of the card. Two throwaway builds (e0225): 4 k-step pairs in flight instead of 2: no change; a
3-instruction decode with the same loads (wrong bits): -63 us at B=1. So the widening, ~12 instructions per
register of two weights, was what limited the stream, not the queue depth.
The format is ours, so the high bits were re-laid out (engine/api.py hi_pack): a lane's 16 weights of a k-step pair
keep their 16 low bytes [code bit 0 | mantissa]; their sign and code bits 1..3 are two 32-bit words in which register
q's 8 bits are stored rotated by (0, 4, 8, 28)[q & 3] -- four rotations that put the four registers of a word on
disjoint bits -- so rotl(word, r) & 0x87008700 is exactly the sign and exponent bits 8..10 of both halves. A register
is now PRMT + SHF + LOP3. Same 12.07 bits per weight.
Checks: tools/verify_w12.py on all 30,484,201,463 non-zero weights: every tile decodes to the checkpoint's bits (e0229);
tools/logit_hash.py: logits of 48 steps at batch 8 and 64 greedy tokens hash-identical to the previous format (e0230).
Dev: generate 2395 -> 2349 at B=1, 6415 -> 6380 at B=8. (The throwaway build's -340 us at B=8 came mostly from its
wrong weights changing the routing.)
### s0012: valid, http 2366 us B=1 / 6382 us B=8 (predicted 2355-2375 / 6350-6400).

## 15. After s0012: what moved and what did not
- Fence-free hand-overs (ATOMIC_Q/H/C/D, alone and in pairs, e0233-e0240): all within +-15 us. The stage-end fences
  are not on the critical path any more.
- Staging the next layer's stage-A weights into shared memory during the barrier (e0241-e0249): cp.async per lane
  at the end of stage C: stage A -2.1 us/layer but the barrier wait +1.9 and the issue +0.9: +40 us at B=1. Bulk
  copies (cp.async.bulk + mbarrier, one per warp) at the end of C or inside the norm after the x loads: +70 us. The
  queue is first in, first out: whatever is staged during the skeleton delays the skeleton's own round trips by as
  much as it saves later. Kept in the kernel behind -DSTAGE_A, off.
- o_proj in stage B (s0013): attention-done stamps per task; at B > 1 every block runs its o_proj row tile right
  after its gate|up share and waits at the stage B barrier only for the down projections; shares count it as 2.5
  tiles. Stage-B barrier wait 2.3 -> 0.6 us/layer; B=8 generate 6402 -> 6294. At B=1 it was 12-45 us slower (at long
  contexts attention is not finished when gate|up is) so B=1 keeps the old order; the placement is a template
  parameter (two instantiations): as a run-time switch both batch sizes got slower (e0265).
### s0013: valid, http 2380 us B=1 / 6310 us B=8 (predicted 2360-2375 / 6260-6300).
B=1 is 10-14 us slower than s0012 in process too, on the same cards (e0266: 2350/2345 against 2362/2355), although
its code path is logically the same; ptxas reports the same 48 bytes of spills. Code layout; not chased further.
- An 11-bit format is out: a 7-value exponent window per row escapes 2.4-2.9% of the weights in every tensor class
  (study/window_census.py, e0274), against 1.2-1.7e-4 for 15 values.
- Cross-block timeline (tools/trace.py, -DTRACE: %globaltimer at every stage boundary of one step, e0275/e0276), B=1,
  us after the first block leaves barrier C: blocks leave over 2.0-2.5 us; norm done +3.0; router rows stored +6.3
  (last +7.9); the union seen by the blocks +12.2; median stage A done +10.6. So at B=1 the routing chain decides
  when gate|up starts: ~1.6 us after stage A on average. Parts: router blocks' half-tile loads wait behind
  everyone's stage-A burst (~3.3 us from norm to stored rows); leader poll + top-8 + publish + blocks' poll 4.3 us.
- Router-only staging (STAGE_A=4) +10 us, rank-count top-8 (TOPK_RANK) +35 us, several polling warps per barrier
  (NPOLL 2-8, staggered) +45-80 us (e0285-e0291): all slower; kept off.
### s0014 (track B, RMSNorm with cvt.rn.bf16x2, bit-identical): valid, http 2373 us B=1 / 6240 us B=8 (predicted 2365-2385 / 6250-6275).
- bench/stream_bench.cu (e0294): the stage-B loop alone, cold L2: 768 tiles (one B=1 layer) 13.9 us, 3072 tiles
  2.95 TB/s; neither 1-4 k-step pairs in flight nor batch size changes it. In the engine the same phase takes ~17 us.
### s0015 (stage C without dependent round trips): valid, http 2365 us B=1 / 6207 us B=8 (predicted 2345-2360 / 6200-6230).
The escape-value gather of stage C (batch 1) followed the header load and preceded the weight loads: two dependent
round trips under a full queue; it now follows the first quantum. At batch 1 the routing details are derived from
the union (slot masks, slots per rank) and the weights' load is consumed only in the residual add. In process -19 us
at B=1 (2360 -> 2341), over HTTP -8 against s0014.

## 16. Measurement noise, and the SMs that stream faster
- tools/ab.sh alternates two builds on one card (e0314-e0316, 3 rounds each, 3 cards): the SAME build gives B=1
  generate 2403-2428 us from one run to the next, in two clusters (~2405 and ~2420-2428); B=8 varies by only +-3.
  Differences below ~15 us at B=1 between single runs (or between cards) are noise; several conclusions of
  sections 7, 13 and 15 that rest on +-15 us are therefore weak. Early residual rows (XO_EARLY): -4.5 us at B=1.
- bench/smid_bench.cu (e0318, 12 launches): exactly 24 SM ids (32-43, 78-87, 126-127 on that card) finish the same
  stream 1.34x faster than the other 108, always the same ones (spread 1-2%). Block -> SM placement changes a little
  from launch to launch (blocks 0..7 go to SMs 124-131 in varying order). Plausible cause of the two clusters: the
  routing chain (leader, router blocks) sometimes lands on fast SMs. Started: roles by %smid from a load-time
  calibration (engine/api.py _roles, MK_ROLES); the first calibration found no fast SMs (under investigation when
  the pod was lost, epoch 9).
- Resolved (e0322-e0327): the 24 "fast" SMs exist only when a block leaves room for a larger L1. bench/smid_bench.cu
  with 120-180 KB of shared memory per block: 24 SM ids (a fixed set per card; another set on the pod of epoch 9)
  finish 1.18-1.37x earlier; with 200 KB or more (the engine has ~220 KB) every SM streams equally (1.01). The card's
  total is the same in every case (3.08-3.13 TB/s by the slowest block); random or constant data, allocation size and
  offset do not matter. So there is nothing for an SM-aware role mapping to exploit in this engine: MK_ROLES is off.
  The two clusters of B=1 run times remain unexplained.
- Every block routing itself at B=1 from the stamped router rows (LROUTE, removes the leader hop and the details
  load): +45 us at B=1 (tools/ab.sh, e0329: 3 rounds on one card). 132 blocks polling the rows and running a top-8
  cost more than the leader's publish + one poll. Off.
- o_proj in stage B at B=1 again, same-card A/B, three runs (e0332-e0334): -11/-14/+43 us at contexts 64-320 and
  +29/+17/+67 at 800-992. Stays off at B=1 (OPB=2).
- Role placement (e0335, tools/roles_ab.sh, 3 rounds): with roles fixed to SM ids every placement gives a stable B=1
  time (+-1 us), but placements differ: by SM id ascending 2413, descending 2409, two shuffles 2423 and 2427; by
  block index (the default) 2404-2419. So the two clusters of run times come from where the roles land on the
  card. Not tuned: the best placement would be specific to one physical card's SM layout.
### s0016 (track A: s0006 + the scheduling work since): valid, http 2785 us B=1 / 7157 us B=8 (predicted 2790-2810 / 6880-6950).
Best track A so far (s0006: 2933 / 7257). Track B at the same kernel organisation: 2365 / 6207 (s0015): the 12-bit
format, the sketch head and the layer-0 tables are worth ~420 us at B=1 and ~950 us at B=8. B=8 came out worse than
predicted because the prediction was the serving figure at contexts 32-128 (long contexts: 7825 in process).
- Global loads compiled to bypass L1 (-Xptxas -dlcm=cg, e0338): +2406 us at B=1, +6040 at B=8. The x operands of
  stage C (h, attn and the zero vector, read again for every k-step) live on L1 hits.
- Attention at B=1 on 4 blocks per task instead of 8 (ATTB1=4, e0339): +146 us (+194 at long contexts).
- One or two attention blocks per task at B=1 (ATTB1=1/2, e0341): +960 / +560 us. Attention at B=1 is close to the
  critical path; its cost is mostly the exchange of partial results between the 8 blocks of a task.

## 17. L1 belongs to the x operands of stage C (s0017)
With ~220 KB of shared memory per block the SM keeps only ~28 KB of L1. Stage C reads its B operands from global
memory for every k-step -- h of the chosen experts, the attention output, and an 8 KB zero vector for unused batch
columns -- and lives on L1 hits for them (all loads bypassing L1, -dlcm=cg: +2406 us at B=1). The streamed weights
(read once) allocated in the same L1 and evicted those lines. Now the weight loads are ld.global.L1::no_allocate (WNA)
and the zero vector is the one in shared memory (ZSM). tools/ab.sh, 3 rounds on one card (e0340): -90 us at B=1,
-273 us at B=8. Gates pass (e0342): generate 2237 / 5930, serve 2237 / 5391.
- No-allocate loads also for the sketch rows and the attention K/V tiles (WNA2, e0343): B=1 -8 us, B=8 +58 at short
  and -68 at long contexts. Off.
- Attention partials stamped value by value and polled by the merging block (ATT_STAMP, e0344): correct, but +197 us
  at B=1 and +691 at B=8: 512 threads polling 21 words each flood the queue (as the stamped residual of section 7).
### s0017 (track B, weight loads skip L1, zero operand from shared memory): valid, http 2252 us B=1 / 6049 us B=8 (predicted 2255-2280 / 5900-5960).
- Trace of the s0017 build (e0346), B=1, us after the first release from barrier C: release spread 0.2 (was 0.8),
  norm done +1.8 (was +3.0), router rows stored +5.0 / last +5.8 (was 6.3 / 7.9), stage A done +9.2 (was +10.6),
  union seen by the blocks +10.5 (was +12.2). The leader hop (poll rows, top-8, publish, blocks' poll) is 4.7 us.
- Rank-count top-8 (TOPK_RANK) again, same-card A/B (e0347): +9 us at B=1, +102 at B=8. Off.
- Router-only staging (STAGE_A=4) A/B (e0348): a run produced no timings; not pursued.
- Share model on the s0017 base: attention base delay 18 instead of 26 (e0349): +46 us at B=1 and B=8.
- Leader delay 8 instead of 16 (e0350): +-1 us. Unchanged.
### s0018 (track B, o_proj in stage B at batch 1 too): valid, http 2187 us B=1 / 6061 us B=8 (predicted 2170-2195 / 6000-6060).
Rejected at B=1 before s0017 (-11..+67 us, attention not done in time); after the L1 change the same change is -68 us
at short and -84 at long contexts (e0350). Lesson: a change of where the time goes (here L1) can flip earlier verdicts;
the ones that were close are worth measuring again.
- o_proj delay at B=1 12 or 28 instead of 20 (GS_DO1, e0353): +4 / +6 us. Unchanged.

## 18. State at the end (written with ~4.8 h left; updated if anything changes)
Best track B: s0021, http 2146 us/step at B=1 and 6028 at B=8 (vLLM 5049 / 9277, Cohere's megakernel 3616 / 7176;
bandwidth bound for the BF16 bytes 1973). Best track A: s0022, 2533 / 6893 (echo A > engine/TRACK builds it; e0356).
What the track-B engine is, in one place:
- one persistent kernel per serving cohort (submit/poll), 132 blocks x 512 threads, a static schedule per layer:
  norm -> A (q|k|v and router rows) -> B (attention on 32 blocks; gate|up of the union of the chosen experts on all,
  shares by measured start delay; o_proj right after a block's gate|up share once attention is stamped) -> C (down
  of every chosen expert for the block's residual row tile; residual add);
- model-specific (track B): lossless 12-bit weights with row exponents, per-neuron down-column shifts for the dead
  neurons and exact escapes (12.07 bits/weight, verified bit for bit on all weights), high bits laid out for a
  3-instruction decode; layer 0 from per-token tables; a certified int8 sketch of the head on the greedy path;
- hardware-specific: weights read with L1::no_allocate so that the x operands of stage C stay in the 28 KB of L1;
  hardware bf16x2 rounding in the norm.
Biggest single steps after s0008 (2598): 12-bit decode re-layout (-46), sketch head (-98 with s0009), stage-B
start balance (-62), L1 for the x operands (-113), o_proj in stage B at B=1 (-65), exchanges instead of fences (-14),
early union load at B=1 (-27).
What is left at B=1 (trace e0352, us per layer of ~44): routing chain until gate|up can start ~10.4 (stage A's
bytes alone ~5.2); gate|up + o_proj ~23 for 50 MB; down ~8.5 for 19 MB; skeleton (norm, sums, residual, hops) the
rest. Phases stream at 2.2-2.4 TB/s against 2.95 for the same loop in isolation (bench/stream_bench.cu).
- Early residual rows at B=1 in the o_proj-in-B kernel (XO_B1) +6 us; four k-step pairs in flight (QDP=4) +5 at B=1, +28 at B=8 (e0354). Off.

### s0019 (track A: BF16 + L1 no-allocate weight loads + o_proj in stage B): valid, http 2573 us B=1 / 6904 us B=8 (predicted 2570-2600 / 6950-7100).
The no-allocate hint first existed only on the 12-bit path; on the BF16 path it is worth 168 us at B=1 in process
(2728 -> 2560, e0355/e0356). Track A now: 2573 / 6904 (s0016: 2785 / 7157). Track B over track A at the same kernel
organisation: -386 us at B=1, -843 at B=8.
- Exchange stores instead of store+fence (e0357, 2 rounds): gate|up results (ATOMIC_H) -8 us at B=1 short, -30 long; residual (ATOMIC_C) -9 / -22. B=8 within noise.
### s0020 (track B, gate|up results and residual rows as exchanges instead of store + fence): valid, http 2173 us B=1 / 6039 us B=8 (predicted 2170-2185 / 6040-6070).
Each SM still fences once per layer in stage A, which keeps its L1 coherent (section 5); the residual's own rows are
read with volatile loads. Combined A/B (e0358): -9 us at B=1 short, -20 long.
- Union load issued during the last stage-A quantum (EARLY_U, e0361): B=1 -26 / -19 us, B=8 +64 / +215.
- As a separate batch-1 instantiation (EUT, e0363): B=1 -26 / -20 us, B=8 -10 / -11. On.
### s0021 (track B, union load issued with the last stage-A quantum, batch-1 kernel): valid, http 2146 us B=1 / 6028 us B=8 (predicted 2145-2160 / 6020-6050).
- The stage-B barrier's poll issued with the last o_proj quantum (EARLY_B, batch-1 kernel, e0365): -3 us. Off.

Submissions in order (http us/step B=1 / B=8): s0001 3497/8948 (in process), s0002 3050/7923 (in process), s0005
3196/7615, s0006 2933/7257 (track A), s0007 2628/6938, s0008 2598/6698, s0009 2572/6699, s0010 2474/6616, s0011
2412/6421, s0012 2366/6382, s0013 2380/6310, s0014 2373/6240, s0015 2365/6207, s0016 2785/7157 (track A), s0017
2252/6049, s0018 2187/6061, s0019 2573/6904 (track A), s0020 2173/6039, s0021 2146/6028, s0022 2533/6893 (track A).
All valid.
- Early residual rows in the batch-1 kernel (XO_B1 with EUT, e0367): +2 us at B=1. Off.
- Two staggered polling warps at barriers (NPOLL=2), same-card A/B (e0368): +40 us at B=1, +670 at B=8. Off.
- Sketch head pass 1 with 8 loads in flight per thread (SKQ=8, e0370): +6 us at B=1. Unchanged (4).
- Attention delay per K tile 8 instead of 5 (GS_A2, e0371): B=1 0 / -16.5 us (short / long contexts), B=8 -30 / -55. On.
### s0023 (track B, share model: attention delay per K tile 8): valid, http 2150 us B=1 / 5989 us B=8 (predicted 2135-2145 / 5990-6010).
B=1 equal to s0021 within the run-to-run spread (+-10 us), B=8 -39. The workspace is this engine (TRACK=B).
Final board: best B=1 s0021 2146 (s0023 2150, same engine but for the share constant); best B=8 s0023 5989; best
track A s0022 2533 / 6893.
- Attention delay per K tile 11 instead of 8 (e0373): B=1 -2.5 / +6.5, B=8 +1.5 / +12. Stays 8.
- Attention delay per chunk 140 instead of 112 (GS_A1, e0374): B=1 -2 / -3, B=8 +3.5 / +30. Unchanged.
- Leader's routing details as exchanges instead of stores + fence (ATOMIC_D, e0375): +2 / -4.5 us at B=1. Off.
- Cross-check with the other runs (progress/, read at 22.8 h): the Qwen3-8B H100 run arrived independently at a
  12-bit lossless code with a mask-and-OR decode (its s0028, 31 -> 11 ops per MMA) -- the same bottleneck as our
  section 14. Its late gains were fence/barrier details of 0.2-0.6% each; the one tried here (ATOMIC_D) is neutral.
- Half-tile staging at the end of stage C (STAGE_A=5, after the Qwen3-8B run's s0029 idea, e0376): the dev test printed no results (likely a hang with the current schedule, as STAGE_A=4 in e0348); not debugged for lack of time. Off.
- Merge-block delay 28 instead of 20 (GS_AX) and attention base delay 32 instead of 26 (GS_A0), e0378: +2 us at B=1 each, B=8 -5..-12. Unchanged. Last experiment of the run.
- Hypotheses from the other runs not tried here (read at 23.2 h, too little time to build and verify):
  (1) Flint/GB200 s0037/s0039: the blocks' best tokens gathered in one trip instead of counter + leader read: about
  one round trip per step (~1 us, <0.1% here). (2) Qwen3.8-27B/GB200 s0014-s0022: self-speculative decoding (copied
  drafts from the context, verify passes) -- the only large lever left that this engine does not use. For this MoE a
  verify pass of k positions costs about a B=k step (the expert union grows: B=8 is 2.8x B=1 here), so it pays only
  if the copy drafts' acceptance is high on the workload; worth measuring the acceptance on the harness's prompts
  first (study script), then building the verify path with the same per-position arithmetic as single steps.

## 19. Measured at the end: copy drafts would be right most of the time (study/copy_draft.py, e0379)
The engine's own greedy continuations (8 of our reference prompts of 32 tokens, 480 new tokens each; the prompts are
random-common-word documents like the harness's): a copy draft (the tokens that followed the most recent earlier
occurrence of the last 3/2/1 tokens) exists at 82% of positions; its first token equals the greedy token 78.5% of the
time; a 4-token draft is accepted for 2.9 tokens on average, a 2-token draft for 1.5.
Estimate (not built): a verify pass over 1 + 4 positions of one sequence reads the union of their experts; with 4.6 of
8 experts shared between consecutive tokens (checkpoint.md section 5) that is ~20 experts instead of 8, so a pass
costs roughly 1.5-2x a batch-1 step and yields ~1 + 0.82 x 2.9 = 3.4 tokens: about 1.7-2x at B=1 if the verify path
gives every position exactly the arithmetic of a single step (the http_tokens gate compares with step-by-step
greedy). This is the largest lever left; it needs attention over several new positions of one sequence in one pass
(causal within the pass) and the KV cache rolled back past rejected positions. Not built: found with 0.7 h left.
Cost side (study/verify_union.py, e0380, HF routing of our sequences): experts per layer for w consecutive positions
of one sequence: w=1 8.0, 2 11.4, 3 14.0, 5 18.2 (x2.27), 9 24.3. Experts are 2/3 of a layer's 12-bit bytes, so a
1+4 verify pass reads ~1.84x the bytes of a B=1 step; with the engine's B>1 overheads (B=8 costs 2.8x B=1 for 3.4x
the experts) a 1+4 pass would take ~2.0-2.2x a step and yield ~3.4 tokens: ~1.6x at B=1; 1+2 drafts: ~1.4x.
With 1000 new tokens (the harness's horizon is 1024; e0381): drafts at 89% of positions, first token right 84.6%,
4-token drafts accepted for 3.2 tokens on average -> ~1 + 0.89 x 3.2 = 3.9 tokens per 1+4 pass: ~1.8x at B=1 by the
estimate above. The repetition grows with the length of the continuation.

## 20. Design for exact draft-and-verify in this engine (not built; written at 23.3 h for whoever continues)
Evidence: our copy-draft study (section 19: 84.6% first-token, 3.2 of 4 accepted at 1000 tokens) and the
Qwen3.8-27B/GB200 run (progress/s4-qwen38-27b-b200-a1, notes/findings.md M2-M7, W2-W3): on the evaluator-style prompts
two of three timed B=1 continuations loop; with copied drafts and 32-position passes it accepted 25.6 tokens per pass
and reached 512 us/token over HTTP (its one-token step: ~10 ms), with every token verified exactly (token-identical to
step-by-step greedy on the timed prompts, logits of a P-position pass bit-identical to P single steps).
Why it maps onto this engine: a verify pass over P positions of one sequence is a "batch" of P columns; the weights
are read once per pass, the extra positions are extra MMA columns (we already run 8), and the MoE cost is the union
of the positions' experts (section 19: 18.2 experts per layer for 5 consecutive positions, 24.3 for 9; on looping
text far fewer, since a loop repeats its routing).
Plan, in the order it should be built and verified:
1. Column -> (sequence, position) map in Launch: column c of a pass is position pos0 + c of sequence s (B=1: s = 0
   for all columns). Everything except attention and the K/V append is already per column and batch-invariant.
2. Attention per column with single-step arithmetic: column c attends to positions <= pos0 + c of its sequence's
   cache. K/V rows of earlier columns of the same pass: the task of column c takes k, v of columns 0..c from qkvo
   (already computed in stage A), applies RoPE itself and uses them like the "new token" row (smem), so no task waits
   for another; every column's own row is appended to the cache by its task (writer as now). The chunk/merge order
   per column must be exactly the single-step one (chunks are defined by absolute position tiles: unchanged).
3. Head per column; accept on the host or in the leader: the longest prefix of drafts equal to the greedy tokens,
   plus the first non-matching greedy token. Cache rows beyond the accepted length are garbage but never read
   (attention reads <= pos), and are overwritten by the next pass.
4. Drafts: copy the continuation of the most recent earlier occurrence of the last n tokens (n = 4 in the other
   run); pass length chosen from the previous pass's result (ramp: short passes until a loop is established, then
   up to the column limit). Column limit here: 8 (MAXB) -- raising it means wider B operands (16/32 columns: mma
   n8 tiles x2/x4) and more shared memory for the normed input (2 KB per column at HID 2048).
5. Gates to keep: forced routing and bitwise single==batched as now; add "P-position pass logits == P single steps"
   bit for bit for several (B, P), and token-identical 1024-token decodes vs step-by-step greedy (http_tokens).
Expected (B=1, P=8, copy drafts at our measured acceptance): ~1 + 0.89 x ~6 = ~6 tokens per pass on looping text,
a pass costing ~2-2.5x a step (union ~25-30 experts) -> roughly 2.5-3x tokens per second; on non-looping text ~1.4x.
Longer copy drafts over 1000 tokens (e0382): 8-token drafts accepted for 6.24 tokens on average, 16 for 12.2, 32 for
24.0 (drafts at 89% of positions). So on this workload the continuations loop like the other run's: with 8 columns
(the present MAXB) ~1 + 0.89 x 6.2 = 6.5 tokens per pass; with 32 columns ~22.
Cost proxy (study/pass_cost.py, e0383, no sketch head): batch-8 steps whose columns are 8 consecutive tokens of one
looping greedy continuation: 5554 us per step against 2233 us at batch 1 (2.49x), 28.6 experts per layer (8
different sequences: 5912 us, 34.0). With 6.5 tokens per 8-position pass: ~860 us per token, ~2.5x the present B=1
figure, before going beyond 8 columns. This is the measured basis for section 20.
- Reproducibility (e0384): the engine built from a clean 'git archive HEAD' passes every gate with the same timings (generate 2130 / 5876, serve 2128-2132 / 5365).
- The same for track A (TRACK=A in the clean copy, e0385): gates pass, generate 2519 / 6817 (s0022 in process 2519).
- Per-column position offsets in the shared kernel (groundwork for section 20, e0386, not kept): gates pass with zero offsets and B=1 unchanged, but B=8 +70 us (5876 -> 5947, code layout). A verify path should be its own kernel instantiation, as the o_proj placement and the early union load are.
- Bit-identity across the whole second half of the run (e0387): the committed engine's logits (48 steps, batch 8) and 64 greedy tokens hash to d5a3c012258b8388 / 1a0cc91ac48ac4b9, the same as at s0011/s0012 (e0230). Every change from s0012 to s0023 (decode layout, norm rounding, round trips, L1 hints, o_proj placement, exchanges, early loads, shares) left the outputs bit for bit unchanged.
- Per sequence (e0388, 8-token drafts over 1000 tokens): mean accepted 5.2 / 7.7 / 0.2 / 8.0 / 8.0 / 6.5 / 5.4 / 3.7; three sequences loop almost perfectly (all 8 accepted at 96-99% of positions), one is prose-like (0.2). The gain of draft-and-verify will depend strongly on the prompt; drafting must cost nothing when it fails (pass length 1 after a miss, as the other run's ramp does).
- Sketch head with 2 loads in flight per thread (SKQ=2, e0389): +18 us at B=1. 4 stays (8: +6).
- Leader delay 24 instead of 16 (GS_DL, e0390): B=1 -1 / +8, B=8 +14 / -17. Unchanged. (Last A/B of the run.)
