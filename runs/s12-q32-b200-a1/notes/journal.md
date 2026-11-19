# Journal (s12-q32-b200-a1)

## s0001 own megakernel v1 (track A)
- Expected: http_step_us_b1 ~11250, B=8 ~11500 (in-process 11.0 / 11.2 ms at positions 0-127).
- Measured: 11488 / 11763 over HTTP; in process generate 11113, w1_like (64 + 1024 tokens) 11426 / step_us 11200.
  numeric median 0.0113 (HF 0.0112), numeric_long median 0.0109, max 0.219; batch bitwise equal; http_tokens 8/8.
- Why the miss (+238 us): attention grows with context (tests/prof.py: 6.6 us/layer at pos 0, 13.7 at 512, 19.7 at
  1024); the HTTP workload averages ~576 positions -> ~+450 us over my pos-0 numbers. Predict from pos ~576 from now on.
- Conclusion: parity with SGLang (11440 / 11692). Largest named losses: per-op fixed cost (prep, epilogue, barrier,
  ~8-9 us per GEMM op), attention (14.5 us/layer avg on the workload), lm_head tail.

## Hour ~2 (board clock): split attention, then a dedicated prefetch warp (negative)
- Split attention (64-position parts, last-arriver combine, padded Q, prep inputs preloaded, paired K/V loads):
  attention alone B=1 8.2/11.0/12.2 us at pos 64/512/1024 (v1: 6.6/13.7/19.7 in-step), B=8 9.9/15.1/20.7.
  Accuracy vs HF float32 on natural text equal to v1 and HF BF16 (tests/cmp32.py). Committed.
  Found on the way: Q fragment loads had an 8-way bank conflict (4k cycles at B=8); norm-prep smem stores a 16-way
  conflict (fixed earlier); preloading K/V across the prep spilled (64-register cap at 1024 threads).
- Skeleton (weights off): 4.7 ms of 11.3; ~7-8 us fixed per GEMM op not overlapped by HBM.
- Prefetch warp (warp 31 walks each consumer warp's future weights with cp.async.bulk.prefetch.L2, paced D ahead):
  B=1 pos 512: 18.3 ms (vs 11.4). Bisect: 31 consumer warps alone -> spills in the stream loop (136 B) and ~17 ms;
  with the prefetcher active the attention op went 4 -> 13 us: a deep prefetch queue (D x 31 x 152 = 77 MB at
  D = 16 KB per warp) delays every critical-path load (FIFO memory queue). Conclusion: HBM-busy-through-gaps via
  L2 prefetch trades directly against critical-path latency; not pursued. Source kept in bench/mk_prefetch_warp.cu.

## s0002 512-thread blocks (track A)
- Expected: http_step_us_b1 ~10710, B=8 ~11560 (in process pos 512: 10667 / 11508).
- Measured: 10631 / 11480 over HTTP; generate 10566 (B=1), w1_like generate 10632. Gates as s0001 (bitwise batch).
- Conclusion: the 64-register cap at 1024 threads was costing ~0.6 ms (spill-sensitive stream loop, fewer loads in
  flight per warp). With 128 registers the code is no longer on a spill cliff.
- Tried after: first weight batch issued before the norm prep (unconditional, clamped addresses): 10697 vs 10667
  (no gain). With a conditional assignment the array went to local memory: 15.7 ms (playbook rule confirmed).

## s0003 dataflow counters (track A)
- Expected 10570 / 11300; measured 10586 / 11290 over HTTP (generate 10539 / 11000). Gates unchanged, batch bitwise.
- The two removed grid barriers per layer (q/k/v -> attention, attention -> o) each cost ~1.2-1.5 us/layer.

## s0004 track B: exact 12-bit storage (bit-identical to track A of the same source)
- Expected 9625 / 10365; measured 9625 / 10407 over HTTP (generate 9576 / 9965). 1.189x / 1.124x SGLang.
- Path to it (hour 3.5-4.5): first W12 loop 11.4 ms (slower than A): escapes fetched one dependent load at a time
  (fixed: 32-entry window per warp in lanes), 6 loads in flight + whole-batch decode (fixed: next batch requested
  before decoding the current one), and a decode of ~40 ops per k-step (fixed: layout with a0-a2 in place, one LOP3
  each, a3 from the free bits: ~12 ops per k-step). Then 9.64 ms in process at pos 512.
- Bit-identity needed the same split-K partition in both tracks (quads of 4 k-steps); that also sped up track A
  (10597 -> 10506 B=1, 11327 -> 11198 B=8 in process).

## s0005 track A, quad partition: expected 10490 / 11160, measured 10489 / 11175 (1.091x / 1.046x).

## s0006 track A, RMSNorm per k-step at B>=3: expected B=8 10965, measured 10994 (B=1 10522 vs 10489 for s0005
with the B=1 path unchanged: build/evaluation spread ~30 us).

## s0007 track A, sentinel hand-over of cut tiles: expected 10360 / 10830, measured 10334 / 10833.
B=1 passes 1.10x SGLang (1.107x). B=8 1.079x: still ~204 us short of 10629.
Tried after, no gain (reverted): 4 P x V tasks per warp round (spills, attention alone worse at all sizes);
sentinel combine in attention by a fixed combiner block instead of last-arriver atomic (B=1 +14, B=8 +52 us: the
fixed block, holding the new position, is often not the last to finish).

## s0008 track B build of s0007: expected 9470 / 10260, measured 9467 / 10277 (1.208x / 1.138x).
Double-buffered BF16 stream (next 8 loads before the current MMAs): 10961 vs 10385 B=1 (worse; reverted).

## Hour ~5.5: attention combine through DSMEM at B>=7, two kernel instances
- Ablation: the attention combine path (counter atomic + partial read-back + combine + signal) costs 4.4 us/layer
  at B=1, 3.8 at B=8 (MK_DBG=128, wrong results).
- At B=7-8 a unit is exactly 2 blocks = one cluster of 2 (clusters of 2 cover all 152 SMs; of 4 only 144):
  block 0 folds its parts, hands (m, L, acc) to block 1 through DSMEM, block 1 continues the fold. Combine made a
  sequential per-part fold everywhere (batch invariance). B=8 -113 us, but B=1 +97 us from allocation effects of
  the extra code (compiling it out recovered 55 us): two kernel instances (B<=2 without the per-k-step norm and
  the cluster code, B>=3 with both, launched with clusters). Same-job A/B vs HEAD: B=1 10358 -> 10337, B=8 10852 -> 10754.
- Checks: bitcheck A vs B and batch 1 vs 8 bitwise, stress 6x3x300 identical, serving exact.

## s0009 DSMEM combine + two kernels: expected 10310 / 10730, measured 10323 / 10768 (1.108x / 1.086x).
Tried after (no gain, reverted): B=1 forced to 2 blocks per unit with the cluster combine (attention+o 43.1 -> 45.6
us at pos 512); K/V of the coming attention prefetched into L2 during q/k/v (B=8 equal or worse).
- gate|up -> down as dataflow (ping-pong residual buffers so no reader of x is overwritten; gate|up tiles counted
  per chunk of 64, down warps wait for the chunks of their k-range): correct, but gate|up+down 115.9 -> 122.7 us
  (B=1). A release per finished tile stalls the epilogue warps (21 tiles per block), the chunks complete when the
  barrier would have released anyway, and the barrier-paced prefetch of down's first weights is lost. Kept in
  bench/mk_gu_down_dataflow.cu. Also no gain: L1-cached GEMM inputs (ld.ca after the acquire's CCTL.IVALL),
  K/V ablation shows K/V loads are only 1.4 (B=1) / 2.5 (B=8) us per layer.

## s0010 track B build of s0009's source: expected 9450 / 10180, measured 9444 / 10356 (B=8 worse than s0008).
Cause (same-job, e0153/e0154): the large-batch kernel instance spills in the 12-bit build (172 B) and is slower at
B=8 than the small instance even with the DSMEM combine (10325 / 10277 without the norm code vs 10257 small).
Fix: track B uses the small kernel at every batch size; the norm-in-stream instantiation removed from W12 builds.
Bitcheck after the change: track A (B=8 on the large kernel) vs track B (small kernel) still bit-identical.

## s0011 track B small kernel at all B: expected 9445 / 10240, measured 9453 / 10265 (1.210x / 1.139x).
Then (hour ~6.3): last-warp post-stream prefetch (track B only: -40 us; in track A the same code shifted the
large kernel's allocation, +344 us at B=8); escape patches through a per-warp shared-memory buffer (track B
B=1 9475 -> 9425, B=8 10241 -> 10091 at pos 512; layer-1 gate|up 110 -> 93.5 us). Track A: a 32 KB larger
shared-memory request costs +165 us (B=8) / +57 (B=1) (L1 carve-out step; +16 KB costs nothing, and going under
132 KB gained nothing), so the patch buffer exists only in W12 builds. Allocation lottery searched: ptxas -O1/-O2/
expensive-opts/-dlcm=cg, non-inlined attention or norm prep: default build best.

## s0012 track B, escape patches through smem + last-warp prefetch: expected 9405 / 10120, measured 9402 / 10049
(1.217x / 1.163x). Track A K/V preload before the q/k/v wait in the large kernel: 232 B spills, +210 us (reverted).

## s0013 track A three instances: expected 10325 / 10755, measured 10351 / 10773 (noise of s0009).
Cached per-k-step norm (first warp to reach a k-step normalises it into a shared cache with epoch flags): B=8
10780 -> 11000 (flag/ballot/fence per batch + allocation): reverted.
- Track B norm phase at B>=3 (each block normalises 1/152 of the residual into a fragment buffer, grid barrier, the
  GEMM reads it per k-step): B=8 10097 -> 10218, B=4 9710 -> 9977 (worse; reverted).
- Serving: cohort gather window 4 -> 20 ms (insurance against staggered arrivals; only time to first token pays).

## s0014 track A final source (cohort-tagged served tokens, 20 ms gather): expected ~10330 / 10770, measured
10354 / 10766 (as s0013: the source's evaluations sit at 10351-10354 / 10766-10773; s0009 10323 was a low draw).
- e0189 track A: PVR_K2 3/4 (P x V tasks per warp round in the B>=7 kernel): B=8 +35-42 us at pos 512 and 1024 vs 2 (10754 -> 10796/10788; 11090 -> 11132/11130). Reverted.

## s0016 track A, K/V of the cached positions bulk-prefetched into L2 before the q/k/v wait (B<=2): expected
~10300 / ~10770, measured 10299 / 10763 (1.111x / 1.086x). Same-job: B=1 -57 us (track B B=1 -62). At B>=3 every
form measured equal or worse (K+V before the wait +45 us at B=8, K alone -20, K after the wait -40 vs its own build
but the code shifted the large instance's allocation +40; track B's B=8 +55..85): compiled into the small instance
only, for B<=2. Also measured: fused softmax into the P x V tasks (each task recomputes its part's max/sum from the
16 scores it holds as A fragments; removes the softmax phase and a barrier): B=8 +90 us at pos 512, +200 at 1024
(reverted). Activation (B operand) loads replaced by constants at B=8: gate|up -0.6 us, down -0.2: the L2 traffic of
the B operand is not the B=8 GEMM excess.
- Timeline (tests/prof4.py, pos 576, us from the q/k/v op start, max over blocks): B=1 q/k/v epilogue 17.4, attention
  prep 19.5, scores 20.4, P x V 22.2, o's wait ends 27.6 (global combine + signal 5.4 us). B=8: 17.9, 19.9, scores
  23.0, softmax 24.7, P x V 28.1, o's wait ends 31.65 (cluster combine 3.5). B=8 at pos 1024: scores 24.4, softmax
  27.0, P x V 32.6.

## s0017 track A, B=8 attention in fewer rounds (one warp per part softmax: 4 lanes per head, 16 scores each;
3 K blocks per warp round at 7-8 sequences): expected ~10300 / ~10670, measured 10304 / 10742. Same-job A/B had
been at pos 512-1024 only (-95 at 576); the evaluation's in-process figures show contexts 64-320 got worse (step_us
+26) and long ones better. New tool tests/wl.sh: mean of positions 128/384/640/896 (s0016 10725 vs s0017 10692:
-33, HTTP -20). P x V tasks per round 3/4 (retest on the new code): +-15 us, no gain.
- Combine in two halves (both blocks of a cluster fold their own parts in parallel; block 1 merges block 0's state;
  the global combine folds [0, np/2) and [np/2, np) and merges the same way: B=2..8 still bitwise equal to B=1):
  workload B=8 10692 -> 10667. Cluster combine stamps (pos 640): fold 0.76, cluster sync 0.74, merge+write 1.0,
  signal->o wait 1.25 us.
- Split grid barrier (arrive at the op end, wait inside the next GEMM after its first 8 weight loads per warp are
  issued): track A B=1 workload 10359 -> 10229 (-130); but in the 3-8 sequence instances every stream got ~8%
  slower (gate|up +6 us, lm_head +17; B=8 +590 us) even where no preload was compiled: kept in the 1-2 instance
  only (template flag). First build spilled 900 B: the preload array was assigned under a condition (local
  memory) and live through the loop; fixed by unconditional clamped loads and peeling the first batch.

## s0018 track A, two-half combine: expected ~10305 / ~10720, measured 10344 / 10715 (B=8 as predicted; B=1 +40
over s0017 although the A/B saw no B=1 change: build-to-build spread of the small instance).
## s0019 track A, split grid barrier + first weight batch before the wait (1-2 sequence instance): expected ~10225 /
~10715, measured 10230 / 10718 (1.118x / 1.091x).
- Why not at B>=3 (e0240/e0241): B=1 forced into the large instance gains as much (-124 us), so it is not the
  instance; at B=4 (+180) and B=8 (+640) the gate|up stream itself is 5 us slower (73.4 vs 68.3 us after the wait)
  while down's is unchanged: the slowdown follows the per-k-step norm GEMMs (IN_NORMF, used only at B>=3).
  Keeping the original stream loop (preload seeds the first segment's accumulator) changed nothing (preb4).
## s0020 track B of s0019's source: expected ~9280 / ~9950, measured 9301 / 9943 (1.230x / 1.176x; new best).
Bitcheck A vs B of this source: bit-identical at B=1 and B=8 (1152 positions), both batch invariant.
- e0244 (B=4, split barrier everywhere except before norm GEMMs): down's stream 1.5 us faster with its preload, but
  gate|up +4.4 us and lm_head +12.5 although their code path did not change: the per-k-step norm loop's code
  generation is what moves (any change to the shared template). Next: norm GEMMs instantiated exactly as before.
## s0021 track A, split barrier in every instance with its wait state in shared memory: expected ~10270 / ~10540,
measured 10284 / 10539 (1.112x / 1.109x): track A past 1.10x at both batch sizes.
- What made the difference (e0245-e0249): carrying the split barrier's state (prefetch pointer and length, pending
  flag) in registers through the op loop slowed every per-k-step norm stream in the 3-8 instances ~6-8% (B=4 +180..
  270, B=8 +640..730 on the workload) even when the norm GEMMs' own code was unchanged; thread 0 keeping the range
  in shared memory and the flag recomputed from the op list removed it: B=8 -180, B=4 -130 (B=1 +39 vs register
  state in the small instance).
## s0022 track B of s0021's source + 7-8 sequences on the cluster-combine instance: expected ~9300 / ~9860,
measured 9336 / 9918 (B=8 -25 vs s0020 instead of the workload A/B's -84; B=1 +35: the small instance's build with
the shared-memory barrier state is ~20-40 us slower at B=1 than s0019/s0020's register-state build). Bitcheck A vs B
of this source and api: bit-identical at B=1 and B=8.
- e0258: B=1 on the per-k-step norm instance (MK_BIG_MINB=1 MK_NORMF_MINB=1): +90 us on the workload (the B<=2 norm
  prep stays). e0260: per-k-step norm inside the 12-bit loop (track B, 7-8 instance): B=8 workload 9795 -> 10506
  (+700: the decode leaves no issue slots, as at hour ~6). The build containing the unused path was itself 40 us
  faster at B=8 than without it (allocation), not pursued. Track A B=1: preload across the norm prep (preb8) -30.
- Verification of s0021/s0022 sources: check (B=1 vs batch bitwise, generate == step), binv B=2..8, stress 0 mismatches, robust, serving mixed lengths exact; vs HF float32 on real text: engine median 0.0122 max 0.376 top1 0.963, HF BF16 0.0122 / 0.386 / 0.961 (tests/cmp32.py 256 2).
## s0023 track A, preload also across the norm prep (1-2 instance): expected ~10255 / ~10540, measured 10265 / 10535 (1.114x / 1.110x; best track A at both batch sizes).
- e0266 (prediction: with K/V already in L2 at B<=2, prefetching more of o's 84 MB during the ~10 us of attention would let o stream from L2, up to -300 us at B=1): falsified. Pos 384/896, B=1: idle 256 KB + busy 0 (default) 10218/10318; idle 576 KB 10302/10284; busy 128 KB 10240/10333; busy 256 KB 10351/10349; all 576 KB 10431/10452. B=8 the same order (all 576 KB +600). The attention chain's own L2/DRAM accesses get slower by more than o gains.
- e0267: track B streams at B=1 run below their 12-bit byte floor (after the wait: gate|up 55.5 us vs 51.7 = 93%,
  down 29.2 vs 25.9 = 89%, o 11.0 vs 8.3 = 75%, lm_head 85%), ~10 us/layer, while track A's beat theirs (L2 head).
  Hypothesis: too little memory-level parallelism (one 3 KB batch per warp in flight, ~48 KB per SM vs 64 KB in
  track A). e0268: each warp bulk-prefetching its own stream 8/16/32 quads ahead into L2 (lane 0, one instruction
  per batch): B=1 workload 9297 -> 10352 / 10721 / 11877 (much worse; per-warp bulk prefetches are expensive, as
  the warp-distributed prefetch in track A was). Reverted. Untested alternative: a third quad in flight with x read
  from shared memory just in time (frees the 16 registers of the next batch's inputs).
Stop rule reached (three in a row without gain: 12-bit per-k-step norm, o prefetch during attention, per-warp
prefetch): verification and record.
- e0270: barrier state in registers for the 1-2 instance only (shared memory in the others): B=1 workload track A
  +34, track B +11 vs s0023's source; the B=1 differences between the register and shared-memory builds are
  allocation, not the mechanism. Not kept.
## s0024 track B, 3 quads in flight per warp in the norm GEMMs, inputs read from shared memory just in time:
expected ~9230 / ~9800, measured 9250 / 9780 (1.237x / 1.196x; best). Confirms the memory-parallelism reading of
e0267 (one 3 KB batch per warp was too little; per-warp bulk prefetch was not the way to add it). Workload A/B:
2 quads + JIT inputs alone B=1 -30 / B=8 -43; 3 quads B=1 -105 / B=8 -122; 4 quads spill (+110). Track A build
of this source unchanged (same spills, workload within 4 us). Bitcheck A vs B: bit-identical at B=1 and B=8.
- e0278 track A weight k-steps in flight per warp 8 / 12 / 16: B=1 workload 10238 / 10268 / 10237, B=8 10489 / 10901 / 10620 (8 kept: track A already streams at its floor).
## s0025 track B, B=1: o and down read their input column from shared memory (3 quads in flight): expected ~9200 /
~9730, measured 9197 / 9840 (1.244x / 1.188x). B=1 as predicted. B=8: the engine got faster in process (w1_like
generate 9728 -> 9718, step 9584 -> 9566) but HTTP got slower (9780 -> 9840): the HTTP-minus-in-process offset at B=8
has ranged +13 (s0020), +70 (s0022), +52 (s0024), +122 (s0025) for engines of the same structure: serving-path
noise of ~100 us at B=8, larger than most single changes now.
- e0286: the norm prep in track B takes 2.2 us (B=1) and 5.0 us (B=8) after the barrier wait (gate|up): at B=8 the
  two norm GEMMs per layer spend ~10 us in it.
## s0026 track B, norm prep with packed bf16 arithmetic: expected ~9170 / ~9785, measured 9156 / 9714 (1.249x /
1.204x; best at both). Prep phases (e0288, MKQ stamps, gate|up): after the wait the loads take 1.7 (B=1) / 2.3
(B=8) us (L2 latency under the weight prefetch traffic), the normalize step 0.5 / 2.7 us; packing halves the
normalize instructions (bf16(x * r) by one cvt.rn.bf16x2, then one mul.rn.bf16x2: the same bits). Workload A/B
B=8 -56, B=1 -30. (A first submit was refused: "norm" is not a category.)
- e0292: the B=1 input staging of o/down with all loads issued before the stores (instead of a load-store loop): B=1 workload +50 (reverted).
## s0027 track A build of s0026's source: expected ~10255 / ~10535, measured 10249 / 10537 (1.116x / 1.110x): the tree's track-A engine.
- e0295: B=1 input staging for o only 9162, down only 9120, both 9088 (kept both). e0296: 3 quads with inputs in
  registers for o/down at B>=2 spills 416 B (not pursued).
- e0298 (tests/prof6.py, B=1 global attention combine, max over blocks, pos 640 / 128): P x V ends 24.7 / 25.9 us,
  the arrival atomic +0.8, the fold and output +4.15 / +2.4 (each thread folds its 2 items one after the other,
  8 parts' L2 loads per round: ~4 dependent round trips at 10 parts), o ready +1.9. Next: stage all parts'
  partials through shared memory with one round of loads, fold from there (same order, same bits).
- e0300-e0302: the combiner stages all parts' partials through shared memory with one round of loads (same fold
  order, bitwise unchanged): B=1 workload track A -27, track B -5 (better at 640/896, worse at 128/384): the reads
  into one block are bandwidth-limited (~25-35 GB/s per SM under the weight prefetch traffic, as in the norm prep),
  not round-trip-limited. Not kept. Next: split the combine over all blocks of the unit (each reads ~1/nact).
- e0303-e0310: the global attention combine split over all blocks of the unit (the last to arrive releases a
  per-unit epoch flag; every participating block folds 1024/nact of the outputs over all parts in the same order
  and signals; o waits for 8 B S signals; bitwise unchanged): B=1 workload track A -95, track B -62. In the 7-8
  instance (cluster combine, untouched in principle) the extra signalling code moved track A B=8 by +35..+38
  (run-time gated or not); with the new code compiled out of that instance (template flag) B=8 is neutral (-8).
## s0028 track A, attention combine split over the unit's blocks: expected ~10150 / ~10530, measured 10141 / 10533 (1.128x / 1.110x).
## s0029 track B of s0028's source: expected ~9090 / ~9670, measured 9063 / 9703 (1.262x / 1.205x; best). Bitcheck A vs B of this source: bit-identical (B=1, B=8); track B stress and binv pass.
- e0317: the split combine with 16 parts per load round instead of 8: B=1 workload +46 (kept 8).
- e0319-e0321: attention prep's constant inputs (norm weights, cos, sin) requested before the q/k/v wait: B=1 workload track A -20, track B -15, B=8 neutral.
## s0030 track A / s0031 track B, attention prep constants before the q/k/v wait: expected ~10120 / ~10530 and ~9050 / ~9700, measured 10131 / 10528 (1.129x / 1.111x) and 9049 / 9683 (1.264x / 1.207x). Full checks on this source (both tracks): binv, check, stress, robust, bitcheck A vs B bit-identical.
- e0325-e0327: track B o / down at B>=2 with the next batch's inputs prefetched into L1 (prefetch.global.L1) and read
  just in time with ld.global.ca, freeing the registers for 3 quads: correct (binv) but B=8 workload 9566 -> 11067
  (the prefetched lines do not survive the weight stream; every just-in-time read waits on L2). Reverted.
- e0323: serving on the final source exact for both tracks (8 x 128 tokens, mixed lengths).
- e0329: runtime prefetch knobs re-swept on the final source (MK_PF 256 KB, MK_PF1 128 KB, MK_PFP 32 KB, MK_PF_IDLE 384 KB): defaults best or tied on both tracks.
## s0032 track B, 7-8 sequences: norm prep split over the cluster pair (each block reads and normalizes half of
the residual and writes it into both blocks' shared memory over DSMEM, then a cluster barrier): expected ~9050 /
~9540, measured 9035 / 9495 (1.266x / 1.231x; best). Workload A/B B=8 -150 us; the 7-8 instance's 12-bit build
also spills less (40 B instead of 112: half the prep's registers). Bitcheck A vs B bit-identical; binv, stress,
robust pass.
## s0033 track A of s0032's source with the 7-8 instance's (unused at run time) norm-prep path instantiated with the
cluster prep: expected ~10130 / ~10450, measured 10121 / 10440 (1.130x / 1.120x; best track A). An allocation
effect: the instance spills 48 B instead of 96 (same-job workload B=8 -85 us); the cluster prep itself in place of
the per-k-step norm at B=8 is worse for track A (+130, e0337). The W12 build of this source equals s0032's.
- e0342-e0346: track B o / down at B>=2 with their inputs copied by each lane one batch ahead into a per-warp
  cp.async ring in shared memory (no registers), read just in time: correct (B=2..8 bitwise equal to B=1, stress), but
  B=8 workload 9411 -> 9608 with 3 quads in flight and 9483 with 2: the memory-parallelism reading does not hold for
  these two ops. Per block, down's inputs at B=8 are re-read from L2 for each tile the block covers (~840 KB of
  inputs vs ~1.3 MB of 12-bit weights): L2-to-SM traffic is the likelier limit.
- e0347: the same inputs through L1 (ld.global.ca; warps on different tiles read the same k-steps): B=8 unchanged.
- e0348 (ablation, inputs replaced by constants, wrong results; in a build whose probe code itself slowed down):
  down's stream -5.4 us, o's -2.0 at B=8: the inputs' L2 traffic is the B=8 cost of these ops.
- e0349-e0351: o at 7-8 sequences with each block of the cluster pair staging half of the input's k-steps in its
  shared memory and reading the other half from the partner over DSMEM just in time (3 quads in flight): correct
  (binv, stress) but B=8 workload 9409 -> 9734 (DSMEM reads on the critical path). Reverted.
Stop rule (three without gain: ring, L1, DSMEM staging). The way to cut these ops' input traffic would be to let one
warp run two sub-ranges that sit at the same k-steps of different tiles (same per-tile summation order, inputs
loaded once for both): a rewrite of both stream loops, not attempted in the remaining time.
- e0352 (per op, B=1, pos 640, barrier mode): over the byte floor track A q/k/v +3.9, o +5.0, gate|up +7.4, down
  +3.4 us; track B +6.5, +8.9, +11.1, +8.8 (track B's down and o wait 1.5-2.5 us after the barrier for their input
  staging at B=1; gate|up's tail 3.1 us vs 2.6).
- e0353/e0354: the B=1 staging with all of a thread's loads in flight spills 344 B in the small instance (+565 us);
  2 or 3 loads per round: +28 / +-0. The staging is not latency-serialized; not pursued.
- e0355: bench/l2bw.cu: per-SM L2 read rate ~66 GB/s max with all SMs reading, ~42 GB/s beside a DRAM stream (table in gap.md); it accounts for the norm prep times, the single-combiner cost and track B's o / down at B=8.
- e0356-e0361: escape patching in parallel (the quad's entries in the warp's 32-entry window written by their lanes
  at once, up to the first later entry: the lists are concatenated per warp behind a sentinel): per layer at B=1
  layer 1 gate|up 88.1 -> 74.1 us, layers 2-4 gate|up -1.5..-4.6, down -0.9..-2.9 (~29 us per step), the other
  layers' ops +0.3 us (build); workload A/B B=1 ~0, B=8 -48. Bitcheck A vs B identical, binv, stress, robust pass.
## s0034 track B, escapes patched in parallel: expected ~9035 / ~9450, measured 9016 / 9433 (1.269x / 1.240x; best).
- e0362: cuobjdump SASS of the track-A builds of s0033's and s0034's sources: identical (md5 of the instruction stream).
- e0363-e0366: only the k-steps with escapes through the patch buffer (warp-uniform mask): the small instance's spills grew (112 -> 144 B); B=1 +90, B=8 +125, layer 1 gate|up 74 -> 84 (reverted).
- e0367/e0368, final source: serving exact on both tracks; stress 0 mismatches. vs HF float32 on real text (tests/
  cmp32.py 256 2): engine median 0.0128 max 0.376 top1 0.986, HF BF16 0.0128 / 0.386 / 0.973 on this pod (torch
  2.15.0; the float32 reference itself differs from the hour-12 pod, where both read 0.0122: the engine tracks HF
  BF16 on both).
- e0369: cut-tile hand-over ablation (MK_DBG=4, wrong results) on the final source, track A B=1 pos 640: 10094 -> 10017 us (-77/step, ~0.3 us per GEMM op; it was ~0.9 per op at hour 4): an upper bound too small to pursue.
- e0370/e0371: dataflow-wait poll sleep 0 / 32 / 100 ns: B=1 workload 10118 / 10104 / 10101, B=8 10415 / 10395 / 10398 (kept 32).
- e0372: track B B=8 (cluster instance) prefetch knobs: defaults 9370; MK_PF 128 KB 9390, 256 KB 9401, MK_PF1 32 KB 9395, MK_PF_IDLE 128 KB 9367, MK_PFP 32 KB 9368 (defaults kept).
- e0373-e0376: block reduction scanning only a tile's contributing warps (start two below an estimate of the first, stop after the last; same adds and order, bit-identical): workload track A B=1 +170, B=8 +195, track B B=1 +110, B=8 +185 (reverted; the ~2.6 us from a block's stream end to its epilogue end is not the scan).
- e0377-e0379: stamps inside the GEMM epilogue (after the L2 prefetch of the next op's first bytes, after warp 0's
  tiles): in that build the existing stream-end stamp itself moved from a median 71.1 us to 39.9 (spread 35-67)
  while the epilogue end stayed at 73.7: the globaltimer read is not ordered against bar.sync (the asm has no memory
  clobber) and the compiler moved it once the code changed. Phase stamps are only comparable within one build and
  only when the surrounding code is unchanged; the ~2.6 us from stream end to epilogue end (median block, tree build)
  stays unexplained in detail (cut-tile hand-over ~0.3 us of it, e0369).
- e0380/e0381: track A's next-op first-bytes prefetch (64 KB per block) issued after the block reduction instead of right after the stream: B=1 +65, B=8 +104 (reverted).
- e0382/e0383: the same prefetch issued at the op's start (right after the wait): B=1 +95, B=8 +215 (reverted). Right after the stream (current) is the best of the three placements.
- e0384-e0386 (tests/serve_time.py): HTTP minus the evaluator's in-process w1_like_generate at B=8 is +47..+81 us for
  every track-B submission and -11..+7 for track A, which looked like a serving cost. In process, a served cohort
  (submit/poll, tokens through host-mapped memory) over the decode window equals generate() over the same positions
  (track A +3, track B +11 at B=8; <= 4 at B=1), and is above the HTTP figure for track B (9456 vs 9433): no serving
  loss; the per-step system fence for the progress counter (unused by poll) costs nothing measurable either. The
  offset sits in the w1_like_generate figure (it includes the prompt's short-context steps, relatively cheaper in
  track B).
- e0387/e0388: MK_ATT_SMAX (blocks per attention unit) at B=1: track A 19 (default) 10108, 12 10107, 8 10122, 6 10156;
  track B 9038 / 9070 / 9017 / 9033 (8 and 6 faster at positions 128-384, slower at 896). Defaults kept.
- e0389-e0392: blocks per unit = the longest sequence's part count (block 0 of a unit signals for the missing ones;
  fewer releases on o's counter): correct (binv, stress, robust, mixed-length serving, bitcheck) but B=1 track A
  +45, track B +20 (reverted): the short-context gain of a low cap was not the signal count.
- e0393: track B B=1 attention cap 19 / 10 / 9 / 8: workload 9030 / 9002 / 9001 / 9001 (two rounds each).
## s0035 track B, at most 10 blocks per attention unit (runtime default, results identical: bitcheck, binv,
robust): expected ~8990 / ~9433, measured 9025 / 9449; in process B=1 generate 9003 vs s0034's 9032 (-29, as
predicted) but HTTP +9 (B=1 HTTP has differed from in process by -15..+22 across submissions). Kept in the tree;
s0034 stays the best track-B headline.
- e0395: track A attention cap 19 / 10 / 14 at B=1: 10108 / 10118 / 10115 (default kept for track A).
- e0396: track B (cap 10) MK_PF_ATT 64 KB: B=1 +39, B=8 +30..+170; MK_PF_IDLE 384 KB B=1 +68; MK_PF_IDLE 128 KB B=1 8965 vs 8999 (-33, both rounds).
- e0397: track B (cap 10) MK_PF_IDLE 256 / 128 / 64 / 0 KB: B=1 9002 / 8965 / 8964 / 8958, B=8 9374 / 9369 / 9417 (one outlier) : 128 KB for track B.
## s0036 track B, idle-block prefetch 128 KB: expected ~8990 / ~9440, measured 8995 / 9439 (1.272x / 1.239x; best B=1).
- e0401: track A MK_PF_IDLE 256 / 128 / 192 KB at B=1 (cap 19): 10108 / 10108 / 10106 (neutral).
- e0402/e0403: track A (cap, idle prefetch) 10/128 KB 10103 (one outlier run 10281), 10/64 KB 10120, 8/64 KB 10115 vs 10107 (defaults kept). Track B with the new defaults: cap 8 8981 vs 9008 (-27), idle 64 KB 9004, MK_PF 256 KB 9063.
- e0404: track B cap 10 (default) / 8 / 7 / 6 with idle prefetch 128 KB: 8961 / 8948 / 8970 / 8967 (the default itself 47 us faster than in e0403's job): -13 for 8, inside the B=1 HTTP spread; kept 10.
- e0405, final tree (s0036's source, track B defaults cap 10 / idle prefetch 128 KB): bitcheck A vs B bit-identical at B=1 and B=8, both batch invariant; per track binv B=2..8 = B=1, check (batch bitwise, generate == step), stress 0 mismatches, robust, serving with mixed lengths exact.
- e0406: track A B=8 MK_PF_IDLE 256 (default) / 128 / 64 KB: 10398 / 10414 / 10423 (default kept). Runtime knobs are exhausted for both tracks.
- e0408: W12 storage re-verified on the final tree: all 31,983,534,080 weights, 0 mismatches (notes/checkpoint.md).
- e0409/e0410: B=8 through the 3-6 instance's split global combine (MK_CLUS=0) instead of the cluster instance: track A 10398 -> 10633 (+235), track B 9369 -> 9737 (+368): the cluster instance stays (its DSMEM combine, cluster-split norm prep in track B, 3 K blocks per round).
- e0411-e0414: dataflow waits on the final tree, B=1 workload: both (default, MK_ATT_FLAG=3) A 10106 / B 8997; o behind a grid barrier (=2) A 10232 / B 9146; attention behind a grid barrier (=1) A 10193 / B 9136 (tests/prof.py now handles flag 2).
- e0417/e0418 (fresh pod): s0036's track-B runtime defaults (cap 10, idle prefetch 128 KB) vs s0034's (19, 256 KB) on the same build: B=1 workload 8963 vs 9028 (-65, both rounds), B=8 9368 vs 9370: s0036's gain is real (its HTTP margin over s0034, -21, was within the B=1 spread).
## Final (board hour ~23.6): engine/TRACK set to B so engine/ is byte-identical to s0036 (best submission, 8995 / 9439 us, 1.272x / 1.239x); track A via MK_TRACK=A = s0033's library (10121 / 10440, 1.130x / 1.120x). Final summary in notes/summary.md.

## Extension (run extended to 36 h at board hour ~23.6)
- e0419 (bench/ab/meas.cu + tests/prof_epi.py; timer reads made data-dependent on values produced after the event,
  so they cannot be hoisted above bar.sync): after a block's own stream end, warp 0 reaches the cut-tile hand-over
  wait at +1.0 us (gate|up +1.3), waits 0.2-0.5, finishes at +1.8 (gate|up +2.3-2.6); the epilogue ends 0.1 us after
  warp 0: warp 0 (the block's first tile, cut with the previous block) is the critical warp. In track A thread 0
  also issues the next op's L2 prefetch before its reduction.
- e0420-e0422: track A prefetch from the last warp instead of thread 0: warp 0 at the wait 0.5 us earlier (the wait
  grows: the previous block's partial now limits); workload B=1 10111 -> 10080, B=8 10406 -> 10374 (-30 both).
- e0423/e0424: cut tiles handed over per warp right after the stream (each warp ending in the block's last, cut tile
  writes its own segment partial to slot [block][warp]; the next block sums them in warp order: the same additions):
  bitcheck A vs B identical, binv both tracks, check unchanged.
- e0425-e0428: that per-warp hand-over is slower: track A B=1 +930 us (64-bit divisions recomputing the previous
  block's warp ranges in the finisher), still +400 with 32-bit arithmetic; B=8 and track B +575..+710 (first
  version): the finisher pays several polling round trips and the readiness time does not improve (a block's warps
  end their streams together). Dropped.
## s0037 track A, next op's prefetch from the last warp: expected ~10090 / ~10410, measured 10083 / 10404 (1.135x /
1.124x; best track A). The track-B build is unchanged (same SASS as s0036).
- e0431/e0432: the GEMM prologue's block range and the prefetch planner (every thread, every GEMM op) computed with
  32-bit instead of 64-bit divisions by the block count (same values): track A B=1 10075 -> 10050, B=8 unchanged;
  track B B=1 9004 -> 8944, B=8 9382 -> 9335.
## s0038 track B, 32-bit range divisions: expected ~8940 / ~9395, measured 8956 / 9390 (1.277x / 1.245x; best at both batch sizes). Full check suite on both tracks passed (bitcheck, binv, check, stress, robust, serve).
- e0435/e0436: the narrowed reduction scan redone with 32-bit arithmetic: still slower (track A +76 / +121, track B +97 / +170 at B=1 / B=8): the e0373 verdict stands on its own.
## s0039 track A of s0038's source: expected ~10060 / ~10405, measured 10061 / 10407 (1.137x / 1.123x; best track A at B=1, B=8 = s0037 within noise).
- e0438/e0439 (bench/ab/attq*.cu + tests/prof_att.py, ordered stamps), track A B=1 pos 640, max over active blocks, us
  from the q/k/v op's start: q/k/v epilogue end 19.3, q/k/v of the group seen 20.8, prep 21.6, scores 22.7, P x V 24.7,
  all blocks of the unit arrived 26.5, combined 28.2, o past its wait 30.1. After P x V: arrival (atomic + flag)
  1.8 us, combine (partials read, fold, write) 1.7, signal to o 1.9.
- e0440-e0444: combine arrival by release + poll of a per-launch cumulative count (target in closed form: arrivals
  of all earlier (step, layer) pairs, none below 64 positions where the combine is skipped): binv, stress, robust,
  serving on both tracks, bitcheck identical; workload B=1 track A 10049 -> 10012, track B ~neutral (8928 -> 8913..8928).
## s0040 track A, counter arrival: expected ~10025 / ~10405, measured 10012 / 10410 (1.143x / 1.123x; best track A B=1).
- e0445-e0447: B=8 attention (cluster path): ~1.8 us from the output write to o past its wait, as at B=1. o without its early weight batch: B=1 unchanged, B=8 +19: the preload does not delay o's poll; the signal is release + poll latency.
## s0041 track B of s0040's source: expected ~8945 / ~9390, measured 8948 / 9404 (1.279x / 1.243x; best B=1). engine/ = s0041.
- e0448 (bench/ab/prepq.cu + tests/prof_prep.py, ordered stamps, track A B=1 pos 640): the norm prep after its barrier wait: loads arrived +1.2 (q/k/v) / +1.3 (gate|up) us, 1/rms +1.5, normalized +1.8 / +1.9; with MK_PF=65536 the loads arrive at +0.8: the barrier-time prefetch pieces slow the prep's L2 reads. New knob MK_PF_NORM: the barrier-time prefetch before a norm-prep GEMM only.
- e0449-e0451: MK_PF_NORM (barrier-time prefetch before norm-prep GEMMs only) 0 / 64 / 128 / 192 (default) / 256 / 384 KB, B=1 workload track A 10114 / 10128 / 10037 / 10040 / 10076 / 10098, track B 9066 / 9089 / 8989 / 8948 / 8998 / 9057: the stream's gain from the prefetch outweighs the prep's slower loads; 192 KB is the optimum on both sides. Reverted (the knob's code alone cost ~30 us of allocation).
- e0453-e0456: the split combine staging its slice of the partials through shared memory with all threads (one load round, then fold from smem): bit-identical (bitcheck, binv, robust), B=1 track A +14, track B +10 (at most 8 parts in most units: the old loop already needs one round). Dropped.
- e0457 (tests/prof_att8.py, ordered stamps in free slots), track A B=8, max over blocks: pos 640 q/k/v ready 21.8,
  prep +0.9, scores +2.1, softmax + P x V +4.5 (5 parts per block: 3 rounds of V loads), cluster fold +0.8, sync +0.7,
  output +0.3, o past its wait +1.8; pos 1024: scores +3.4, softmax + P x V +6.5.
- e0458/e0459: V of the block's positions bulk-prefetched into L2 at the attention start (7-8 sequences): P x V phase
  4.5 -> 3.9 us at pos 640 but q/k/v ready, prep and scores later (the prefetch competes): o past its wait 32.97 ->
  33.15. Dropped.
- Not run: choosing the softmax layout by parts per block (one warp per head at 1-2 parts): the two layouts sum the
  exponentials in different orders, so B=1 (one part per block) and B=8 (several) would differ bitwise.
- Decided against in the extension: the paired-tile stream (one warp uses each input fragment for two tiles). It
  needs a new partition in both tracks' loops (bit-identity), new packing for both formats, escape lists and epilogue
  bookkeeping; its gain (~2.5%, estimated from the per-SM L2 read limit) is in track B at B=8 only, while track A's B=1
  loop would change too; a separate layout for B>=3 would break batch invariance.
- e0461/e0462: P x V tasks per warp round at 7-8 sequences 2 (current) / 3 / 4 on the workload: track A B=8 10371 / 10444 / 10415, track B 9318 / 9337 / 9356..9539: 2 stays.
- e0463-e0465 (bench/ab/pfe.cu, MK_PFE_NORM): before a norm-prep GEMM, the barrier-time prefetch's first N bytes issued
  right after the previous op's stream (rest paced in the wait as before), so the prep's L2 reads meet less traffic.
  Workload means, engine / variant at 64 / 128 / 192 KB: track A B=1 10010 / 9996 / 9982 / 10054; track B B=1 8922 /
  8954 / 8955 / 9026 (the variant's code alone +32: allocation); track B B=8 9330 / 9324 / 9318 / 9437.
- e0466-e0472 (bench/ab/pfe2.cu: compiled into track A's 1-2 sequence instance only; track B's and track A's 3-8
  instances SASS-identical to engine/): track A B=1 10010 -> 10000 (128 KB), B=8 unchanged (10376 / 10376). Most of
  the first variant's 30 us was the allocation of the KV=0 instance, not the prefetch timing.
- e0473: size sweep on pfe2 (track A B=1, same job): engine 10010, 64 KB 10022, 96 KB 9996, 128 KB 10001, 160 KB
  10043. At most -14 us, inside the evaluation's noise: dropped (bench/ab/pfe2.cu kept).
- e0474 (bench/clocc.cu): co-resident clusters at one 512-thread block per SM: size 2 -> 76 (152 blocks), 4 -> 36
  (144 blocks), 8 -> 15 (120): a 4-block cluster norm prep (quarter of the residual per block over DSMEM) would cost
  8 SMs of the stream. Not pursued.
- e0475/e0476 (bench/barlat.cu, 152 blocks, back-to-back grid barrier rounds by warp 0): one counter 1.14 us per
  round (1.36 with half the warps streaming DRAM); 2 / 4 / 8 / 16 counters (block i on counter i % NC, lanes poll
  one each) 1.17 / 1.17 / 1.24 / 1.25 (1.40 / 1.37 / 1.44 / 1.47): arrival serialization is not the cost. Relaxed
  red + relaxed poll (no ordering, a lower bound only) 0.70 (0.93): the release fence is ~0.45 us of each round;
  fence.acq_rel + relaxed red is the same as red.release. Removing it needs data-carried flags (tagged values, LL
  style), which double the norm prep's L2 reads (its limit): not pursued.
- e0477-e0479 (bench/ab/nogamma.cu, TIMING ONLY, wrong results): the norm prep without its RMSNorm-weight loads
  (gamma = 1): workload track A B=1 10012 -> 9916 (-95), track B B=1 8929 -> 8879 (-50), track B B=8 unchanged.
  At B=1 the weights are half of the prep's 20 KB of L2 reads per block.
- e0480-e0494 (bench/ab/gst*.cu, GSTAGE): the next norm's weights copied with cp.async into the input buffer's
  unused column 7 (B <= 4), the prep reads them from shared memory. Issued inside o / down's prologue: stack frame
  x2-3, track A +1950 (spills); separate staged instantiations: still +1070; issued in the kernel loop after the
  GEMM returns (before the barrier arrival), one instantiation (layer 0's weights staged at the step start): spills as
  the engine, but track A B=1 +33, track B B=1 +67, B=4 +21.
- e0495/e0496 (gst3: the copies issued right after the barrier arrival instead of before it): track A B=1 10009 ->
  9976 (-33, both rounds), track B B=1 8906 -> 8914..8922 (neutral). The arrival must not wait behind the copies.
- e0497-e0500: gst3 track A B=2 10120 -> ~10087 (one outlier position excluded), track B B=1 8928 -> 8943 (+15), B=4
  neutral; the 7-8 instance SASS-identical on both tracks. GSTAGE set to track A only: the track-B build is then
  SASS-identical to s0041 (all three instances). e0501-e0504 on track A: bitcheck A vs B identical (B=1, B=8, both
  batch invariant), binv B=2..8 = B=1, check vs HF unchanged, stress 0 mismatches, robust, serving exact.
## s0042 track A, norm weights staged during the barrier wait: expected ~9980 / ~10410, measured 9981 / 10408 (1.146x / 1.123x; best track A at both sizes). engine/ = s0042's source; TRACK = B.
## s0043 track B of s0042's source (SASS-identical to s0041): expected ~8950 / ~9400, measured 8937 / 9383 (1.280x / 1.246x). Same machine code as s0041 (8948 / 9404): the -11 / -21 us is evaluation noise, not a change. engine/ = s0042/s0043 source, TRACK = B.
- e0505/e0506 (bench/ab/xcl.cu + api_xcl.py, XS1CL): track B B=1 with instance 0 launched in clusters of 2, o / down's
  input column staged by the pair (each block loads half from L2, writes both shared memories over DSMEM, cluster
  sync): correct (binv B=2..8 = B=1, stress 0 mismatches) but workload 8917 (engine) / 8968 (variant, clusters off:
  allocation +50) / 9047 (clusters on: +130; +170 at pos 128, +95 at 896; in-kernel stamps, launch cost excluded).
  Dropped. (tests/ab_api.sh: A/B of source + api.py pairs with env settings.)
- e0507/e0508 (bench/ab/nogf.cu, TIMING ONLY): track A's per-k-step norm (3-8 sequences) without its prologue's norm
  weight loads (10 KB per block from L2 after the barrier, scattered into fragment order): B=8 10374 -> 10232
  (-142), B=4 10195 -> 10145 (-50).
- e0510-e0512 (bench/ab/fstg.cu + api_fstg.py, FSTG): a fragment-order copy of every norm's weights (op word 6; host
  transpose [320][2][4] -> [320][4][2] of the u32 pairs), copied with cp.async to offset 0 right after the barrier
  arrival before the GEMM (the GSTAGE hook, one function for both); the prologue only waits. Track A B=8 10374 ->
  10238 (-136, both rounds), B=4 10200 -> 10150, B=1 9969 -> 9963. Bitcheck A vs B identical, both batch invariant;
  binv B=2..8 = B=1. Track B's build SASS-identical to s0041 (FSTG 0).
- e0513/e0514: check vs HF unchanged (batch bitwise, generate == step), stress 0 mismatches, robust, serving exact on
  track A; track B binv, robust, serving with the new api.py (word 6 filled, unused by track B).
## s0044 track A, fragment-order norm weights staged during the barrier wait: expected ~9980 / ~10270, measured 9974 / 10276 (1.147x / 1.138x; best track A at both sizes).
## s0045 track B of s0044's source (SASS-identical to s0041): 8938 / 9373 (noise vs s0041 / s0043).
- (pod replaced: epoch 8 from here; absolute workload numbers shift by ~+40 at B=1, -40 at B=8 on the new cards)
- e0515-e0517 (bench/ab/scl.cu, SCLA): track B's GEMM prologue loaded the block's row factors (2-3 tiles x 16
  floats) with load + shared store by warps 0-1 right after the barrier wait: in-order issue stalls those warps (warp
  0 is the epilogue's critical warp) for an L2 round trip before their stream starts. Now cp.async (16 B pieces),
  waited for before the stream-end barrier: B=8 9287 -> 9245 (-42), B=1 8962 -> 8904 (-58), both rounds; binv,
  stress pass.
- e0518/e0519: track A SASS-identical to s0044; bitcheck A vs B identical; check, robust, serving pass on track B.
## s0046 track B, row factors by cp.async: expected ~8890 / ~9330, measured 8911 / 9332 (1.284x / 1.253x; best at both sizes; B=1 gained less than the workload A/B). engine/ = s0046 (TRACK = B); MK_TRACK=A = s0044.
- e0521-e0524 (bench/ab/ewin.cu + api_ewin.py): every track-B warp's stream began with two dependent L2 round trips
  after the wait (its escape-list start from op word 14, then the first 32-entry window at that start) before its
  first weight batch went out. The host now appends each warp's first window ([nblk * 16][32] int2, the same values
  as ent[start + lane]) after the starts; both loads are independent and issued at the op start, before any wait.
  B=8 9255 -> 9152 (-103), B=1 8911 -> 8789 (-122), both rounds; bitcheck A vs B identical (both batch invariant),
  binv, stress, check, robust, serving pass; W12 load-time verification 0 mismatches. Track A SASS unchanged.
## s0047 track B, independent escape-window loads before the wait: expected ~8800 / ~9230, measured 8792 / 9238 (1.301x / 1.266x; best at both sizes). engine/ = s0047 (TRACK = B); MK_TRACK=A = s0044.
- e0526-e0528 (bench/ab/ewn.cu): escape windows double-buffered (the host stores each warp's first 64 entries; a
  refill takes the prefetched window and requests the next): bit-identical, binv, stress pass, but B=8 9155 -> 9154,
  B=1 8796 -> 8789: refills are not on the critical path. Dropped (engine/w12.py keeps 64 entries of padding: harmless).
- e0529/e0530: discarded (bench/ab/xres.cu was accidentally built on the double-buffer source with the 32-entry api:
  mismatched kernel and host layout; no numbers used).
- e0531-e0533 (bench/ab/xres2.cu, XRES): the residual GEMMs' epilogue read x (bf16) from L2 right before the add
  (a dependent round trip on the finishing warp, incl. warp 0's cut tile); now the block's <= 4 tiles x B columns
  of x are copied with cp.async at the op start (x is final since the last barrier; each tile is written once per op,
  by its finisher) into the row-factor area + 2 KB. Track A B=1 9963 -> ~9905 (-60), B=8 10234 -> 10180 (-55); track
  B B=1 8787 -> 8792, B=8 9157 -> ~9190 (+35): XRES set to track A only.
