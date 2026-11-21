# Journal

## s0001 v1 (0.6 h): valid, HTTP 5047 / 14420 us
Expected ~4.9 ms / ~10.8 ms over HTTP; measured 5047 / 14420. B=1 as predicted; B=8 worse than in-process (10.6 ms
at ctx 64-192) because HTTP runs ctx 64-1088 and attention items at B=8 x 17 chunks run in ~7 rounds per block.
Gates: numeric median 0.0236 (limit 0.054), long 0.021 (0.0475), top-1 0.973 (HF 0.962); batch/invariance bitwise.

## after s0001 (in-process, B=1 ctx 64-128 unless noted)
- GEMV microbench (one block, steady state): BF16 42 GB/s/SM, NVFP4 28.7 (v1 decode), MXFP8 26. NVFP4 without
  the scale ops 47 -> scale handling was the cost. E4M3 scale -> fp32 by (b<<20)+(120<<23) after a zero-C MMA: 33.
  4 accumulators: no gain on KIND 1, loss on KIND 0 (kept 1 accumulator for BF16).
- F staging by bulk copies + factors in smem + warp-reduce top-8: 4776 -> 4444.
- attention: bulk-copied K/V chunks, parallel loads, owner-block merge: 4444 -> 4235 (attn 16.4 -> 11.2 us/layer).
- mixnorm (one load round, all dots in one block reduction) + gates restructured: 4235 -> 4109 (B=8 8.6 -> 11.0 ms:
  the mixes now run one sequence after another).
- Ring weight prefetch moved from before the barrier to after the phase's latency-critical loads, plus L2 prefetch
  of next weights: 4109 -> 4204 (no gain). The mix's thread-0 loads still took ~3.5 us.
- Probes (one dependent ld.cg by thread 0): 0.23 us at phase A and E starts: L2 latency is fine. But the top-8 block
  (one round of loads + ~600 instructions) took 5.7 us: 2 us waiting for first-touch weight loads (router bias,
  expert factors: evicted from L2 since the last step, HBM ~1 us each), 1.7 us of 8 dependent reduction rounds,
  1.7 us union (incl. two more first-touch loads). -> every first touch of a small weight costs ~1 us; there are
  ~10 per layer. Unrolling top-8 rounds or not: 5.7 vs 5.4 us (instruction fetch is not the main term there).

## s0003 (tensor-core attention, mixer blocks for B>=2, qkv early prefetch, MXFP8 static index): valid
Expected HTTP ~4210 / ~11800; measured 4280 / 11567 (in process generate 4164). B=1 missed by 70 us: the in-process
gains (-45 attention, -30 prefetch, -42 MXFP8 index, all measured at ctx 64-364) shrink over HTTP's ctx 64-1088 and
the HTTP-vs-generate gap grew from 84 to 116 us (moving the host token write to the idle last block: no change).
- Calibration (bench/calib.cu, 152 blocks): grid barrier 1.18 us (all arriving together; 4 extra back-to-back
  barriers cost 0.75 us each), __syncthreads 0.01, dependent L2 load 0.15, 18 independent loads + sync 0.38,
  block sum 0.13, 4.6 KB bulk copy 0.21, 200 dependent FMA 0.43 us.
- mixnorm sections (isolated, clock64 inside): ~1 us before the first data load is issued: the argument arrays
  (codes, query / weight / row pointers) are read from the caller's stack (local memory; 600-700 B/thread = 150-180
  KB per block, larger than the ~56 KB L1 left next to 200 KB smem, so every LDL is an L2 round trip). Copying them
  to smem at entry: -0.7 us in isolation but +45 us in the engine (A/B): reverted. Making the coefficient loop
  array-free: section 1.02 -> 0.45 us, total unchanged (other sections moved).
- Ring prefetch placement A/B (all late = 4216): early qkv -30 (kept), early o_proj +55, early w2 +21.
- Distributed mixes (row statistics written by the producers, slices + one more barrier): B=1 4692 vs 4230,
  B=8 11267 vs 10464: worse (each consumer block sums 144 producer partials; at B=8 135 loads per thread). Reverted.
- Mixer blocks for B>=2 (block b mixes sequence b, publishes rows, flags): B=8 10939 -> 9640; gates likewise
  (fp32 mixes published, z rows for all sequences in one pass): 9640 -> 8387. B=1 unchanged.
- Tensor-core attention: B=1 -45, B=8 -252 at ctx 300. README text 320 positions vs fp32 reference: median 0.0388
  (s0002 0.0377), top-1 0.975 (0.978): same accuracy.

## s0004 (no large L2 prefetches; B>=2 gates via mixer blocks; parallel w2 rounds): valid 4103 / 9491
Expected ~4170 / ~9900: both better than predicted (-177 / -2076 vs s0003).
After: bias before the K/V wait + P as bf16 pairs: bitwise identical, B=8 -71. Merge in o_proj: +375 (off).
- mixnorm3 (arguments in smem, every array unrolled with compile-time indices): isolated 4.5 us (2 sources) / 5.8
  (5) vs mixnorm 2.9 / 4.6: predicated-off code still issues; in the engine B=1 +464..+537 us (the kernel also
  started to spill 240-470 B). Reverted. The kernel at 255 registers is an allocation lottery: changes to call sites
  move spills into hot paths.
- Gate phase 0 timeline (clock stamps, B=1 pos 300): D-row loads 0.6 us, mixnorm 6.8 us (3 sources), z rows 0.9,
  F/G copy issue 0.3, barrier 1.25: the mix is the gate's cost.

## s0005 (256-key attention chunks at long context, MMA relative bias, B>=2 gate z rows, radix top-8): valid 4083 / 8531
Expected ~4100 / ~8100: B=8 -960 us vs s0004 (less than predicted -1390), B=1 -20 (new best, within noise).
- B=8 step vs context (in process, random tokens): 6674 us at ctx 64-192 rising to 8650 at ctx 1088 before the change:
  attention items (sequence, kv head, 128 keys) = 8 x 8 x 9 = 576 at ctx 1000 over 152 blocks: 4 rounds of ~5.5 us
  (stamps: loads 0.6, clip + bias 1.3, wait 0.15, scores 0.8, softmax 0.45, PV 1.15, partial + atomic 1.1 us).
- 256-key chunks only when 128-key items would need more than one grid round (chs chosen per step from B and the
  context; V tiles in the weight rings, idle during phase B): B=8 ctx 900 8543 -> 7773 us. At B=1 256-key items
  were slower (+77 at ctx 900: one round either way, longer items), hence the switch.
- Relative bias as one MMA per key tile (A = the bias-table rows of the tile's keys, bulk-copied with the K/V tiles;
  B = r as bf16, exact since r is rounded to bf16): -96 us B=8 at ctx 900. README 320 positions vs fp32 reference
  0.0395 (before 0.0388), top-1 equal.
- B=1 compile lottery: a build bit-identical in B=1 logits (top-8 change only touches B>2) ran +69 us; mkprof showed
  the time in E (top8 +0.66, epilogue +0.8, pf_down +0.7 us/layer). Restoring the B<=2 top-8 text verbatim did not
  bring it back (+30 more). -O1/-O2/expensive-opts: no better build. Accepting it.
- After s0005: mixer-flag wait polled 8 flags with one dependent L2 round trip each (3.3 us at B=8 from the last
  publish to all seen); lanes 0..B-1 poll in parallel: 2.0 us, B=8 -97 us, bitwise identical.

## Direction change (operator, 21:40): batch 8 is the main target
- My random-token tests understated B=8: the evaluator's B=8 step_us (natural text, ctx 64-320) was 8169 vs 6650 on
  random tokens: natural text routes to more distinct experts (union 22-53 per application, mean 37 at pos 300 on the
  README). tools/ab.py and prof_run.py now default to natural-text tokens (tests/nat.py).
- B=8 profile (natural, pos 300) before: per application MoE ~96 us of ~172 (top-8 11.7, w13 28, w2 25.5, accum 17,
  stage 8, pf_down 5).
- Changes (all bitwise identical): w2 rounds pipelined (next round's weights issued before the accumulation),
  activation copies issued by 32 lanes (F stage 8.2 -> 3.4 us): -196; the w2 L2 prefetch at E start removed (it
  delayed the w13 start by ~4 us and was evicted by the w13 stream): -430 B=8, -100 B=1; ring stages of 12 items x 2
  (was 6 x 3; ~100 instructions of per-stage overhead) and no per-stage proxy fence: NVFP4 decode 54 -> 71 items/us
  per SM, MXFP8 48 -> 61, B=8 -390; w13 as one task per unit with the activation in the task completion (no rounds,
  no red): B=1 -60, B=8 0.
## s0006: valid 3961 / 7291 (expected ~3950 / ~7500)
- Measured and rejected (B=8): L2 prefetch of a warp's next task (+1250 us: cp.async.bulk.prefetch is harmful here,
  third time), RI 8 x RS 3 (+350), RI 6 x RS 4 (+400): B=8 time follows the microbenchmark decode rate, not the
  bytes in flight; packed fma.rn.f32x2 for the NVFP4 scales (67.6 vs 71 items/us); unroll-by-3 partial stages (+60);
  F as two K-half rounds over all 64 pairs with per-pair partials (bitwise identical, +225 us); NT=512 (breaks).
- Decode loop (ncu, one block, NVFP4): 72.6 warp instructions per item, issue 52%, ALU pipe 56%: ~58 per item are
  essential (16 cvt e2m1, 8 prmt + 8 lea for scales, 4 HMMA, 16 FFMA, 4 LDS); the rest is per-stage.

## s0007 (w2 as (slot, rowtile) units at B=8; per-slot w13 counters instead of the E->F barrier): valid 4009 / 6974
Expected ~3960 / ~7000. B=8 -317 vs s0006; B=1 +48 (no B=1 code change: allocation/layout).
- B=8 F: each block owned one output rowtile, so it needed all ~64 pair activations (2-3 rounds). Now contiguous
  (slot, rowtile) units per block (<= 2 slots, <= 16 activation rows), bf16 y to global, a barrier, and rowtile
  blocks sum each token's 8 experts in slot order: F 38 -> 29 us per application. The E->F barrier became per-slot
  completion counters (a w2 block needs only the 3-4 neighbouring blocks' w13): -118.
- Rejected: w13 half-unit tasks with a shared-memory completion count for warp balance (+295: per-stage overhead of
  18-item tasks and waits); mixnorm arguments by value (15 or 9 arguments: spills 44 -> 220-240 B, +100..+600 us);
  L2-staging z loads in gate phase 1 (0).
## s0008 (attention: key-range pieces in one grid round, 256-key sub-chunks, online softmax): valid 4065 / 6851
Expected ~4000 / ~6800. B=8 step at ctx 1024-1088: 8246 -> 7654 us; mean over ctx 64-1088 -190.
- Pieces aligned to 128-key chunks (unaligned pieces of e.g. 125 keys span 9 tiles: B=1 +60); one-sub-chunk pieces
  keep K and V on one barrier. Short contexts bitwise identical to before.
- Then: mixnorm's argument arrays (source codes, query / weight / row pointers, the &q0 locals) moved from the stack
  (local memory: misses to L2, a dependent round trip before every source load) to shared memory, signatures
  unchanged: stack 1280 -> 880 B, B=1 -62, B=8 -137, bitwise identical. First version raced: phase A's per-sequence
  loop (layer 0 at B > 1) rewrote the shared row pointers while slow threads still stored with the previous ones
  (tests/t_det.py loads one library twice and compares: caught it).
## s0009 / s0010 (no stack arrays on critical paths): valid 3977 / 6662, then 3846 / 6411 (both best)
Expected ~4000 / ~6720 and ~3850 / ~6520. The local-memory census (nvdisasm -g, LDL/STL per source line) found the
remaining stack arrays: mix_wait_rows' source/destination pointer arrays (186 local loads by thread 0 before the row
copies), mixnorm's runtime-indexed coefficient arrays, the loop-gate norm source arrays, the w2 combine's expert
lists. Stack frame 1280 -> 208 B over the three commits; B=1 -250, B=8 -290 in process, all bitwise identical.
- Measured and rejected after s0010 (all ab on card 1, natural tokens):
  - mixnorm is still 6.4 us in the engine (load phase 2.85 us): not its queries' L2 residency (prefetching them at
    F end: 0, B=1 +56), not the qkv ring prefetch (PFE_QKV=0: same). ncu: 41% of its stall samples are no_instruction
    (75 KB of unrolled code, cold every call). Runtime stream loops (+250 / +250), template <1,1>/<3,3> versions
    (+220 / +675: two call sites per wrapper moved mk_kernel's allocation). Left as is.
  - w13 warp balance by splitting only the remainder units into K halves: +243 at B=8 (third form of splitting: all
    worse).
  - Ring geometry with a smaller xs (microbenchmark only): 10 x 3 (138 KB) slower than 12 x 2 (110 KB), 16 x 2 no
    better: per-stage overhead, not bytes in flight, sets the streaming rate (~5 TB/s from HBM at full grid).
  - Release reduction instead of fence + atomic in the grid barrier (+25) and in mix_publish (+30).
  - Attention pieces of at most 2/4/8 chunks (fewer merges): equal or worse (B=1 at ctx 900: +80..+360).
  - o_proj ring stream at the end of attention for every B: B=8 +25 short, -65 at ctx 900 (kept).
## s0011 (fast math): valid 3679 / 6147 (expected ~3720 / ~6170): B=1 1.004x vLLM, 1.18x SGLang; B=8 0.83x vLLM
From a2's notes (their s0012): the fp32 vector code on the critical path is instruction-latency bound. Here
--use_fast_math: B=1 -121, B=8 -239 in process (without FTZ -83 / -166; approximate div/sqrt only -52 / -98).
Gates: numeric median 0.026 (limit 0.054), p99 0.113 (0.254); long median 0.023, p99 0.221 (0.479). README 320
positions vs fp32 reference: median 0.0347 (IEEE build 0.0379); the single-position maximum moves with any rounding
change (0.27 -> 0.49-0.51 for every variant, also div/sqrt only): a near-tie position, not a systematic error.
MK_PRECISE=1 builds the IEEE version.
- After: RED_SLOTS 48 -> 32 (8 KB more L1): 0. Router split into 2 / 3 K parts over 16 / 24 blocks: 0.
  Gate phase 1 F/G chunk rotation against bank conflicts: 0. Mixer-normalised loop-gate streams (A, B > 1): 0.
  top-8 by redux.sync (a2's idea): B=1 bitwise identical and equal, B=8 -25; kept (simpler).
## s0012 (B=1 deep w13 stream, acq_rel attention counter, redux top-8): valid 3634 / 6045 (expected ~3630 / ~6090)
- B=1 w13: one 36-item unit per warp through a 2-stage ring = one refill round trip; all three 12-item stages now in
  flight (the warp's idle ring stages + a buffer in xs past the x row, 3 dedicated barriers with parity counters in
  smem; the ring's own state untouched): -44, bitwise identical.
- Rejected: release reduction for the per-slot counters (0), L2 prefetch of the attention parameters at A start
  (+37 B=8), 64-key attention pieces at B <= 2 (+73 short, +35 at ctx 900: 128-key pieces are the sweet spot on both
  sides).
- After s0012, all rejected (ab on card 1): fused E/F per warp at B=8 (each warp starts its w2 stream when its w13
  units are done, the first warp loads the activation rows into the free half of xs; bitwise identical but B=8 +60,
  B=1 +74); the remainder-unit K-half split again (+122 B=8, B=1 +64 without using it); the watchdog's printf out of
  line in a noinline function (+420 B=8: register saves around every wait loop); -dlcm=cg (0); mixer blocks 144-151
  instead of 0-7 (0); z rows at stride 65 in the gate's smem (0); PFE_DN=1 now faults (not pursued).
- Pattern of the session: adding any code path to mk_kernel costs B=1 ~60-70 us even when the path is not taken
  (code layout / allocation of a 554 KB kernel at 255 registers); a change has to win more than that to show.
## s0013 (kernel specialised by batch size): valid 3577 / 5737 (expected ~3565 / ~5790)
template <int BC> mk_kernel_t, instantiated for B = 1, B = 8 and generic (BC = 0, B from the arguments); the host
picks by B. B=1 instantiation without spills (16 B stack), B=8 184 B stack (176 / 448 B spills), generic as before.
In process B=1 -70, B=8 -255, bitwise identical; build 18 s. Explains the session's "any added path costs B=1 60 us".
- Re-tested on top of it (B > 1 only paths, so the B=1 kernel is unaffected): fused E/F per warp (B=8 -6: 0),
  remainder-unit K-half split (+53). Not kept.
## s0014 (B>1 kernels: mixnorm <1,1> instantiation, gate z loads before stores): valid 3573 / 5674
Expected ~3580 / ~5620: B=8 -63 over HTTP (in process -127 at ctx 64-320: -54 mixnorm, -73 z loads).
- With the kernels specialised, B>1-only changes are judged in the B=8 kernel alone: mixnorm's 1-query/1-stream
  version (B=1 kernel: +55 from a second call site; B=8: -54) is dispatched only when B != 1.
- After: o_proj short-conv taps / history loaded before the GEMV (B=8 -25, kept); the same for F's finishing conv
  (+5 / +33: not kept); mixer-normalised loop streams again (+126); z rows two tasks per warp with bf16 x/rms rows
  from the mixers (+55: B=8 kernel stack 184 -> 496 B).
- Measurement note: two copies of the same library in one ab run differed by 56 us at B=8 (memory placement of the
  engine instance); B=8 A/B differences below ~50 us are not resolvable one at a time. B=1 copies agree within ~2 us.
- Compile-time FUNIT decision: for B=8 kept (-20, noise level); for B=1 removing the dead path cost +31 (layout).
## s0015 (o_proj conv inputs before the GEMV, compile-time w2 path at B=8): valid 3573 / 5645 (expected ~3575 / ~5630)
- Rejected after: PFE_R=1 (router stages at the end of C: 0 / 0); first w2 stages at the end of E at B=1 (+30).
- Gate z rows from source products (ZPRE): z = r_x sum_n c_n (D . v_n), the products D . v_n computed by the non-mixer
  blocks while the mixers run, mixers publish their coefficients. The gate itself got ~0.5-1 us faster (phase 1 start
  14.6 -> 14.1 us at app 20), but the B=8 kernel's allocation collapsed: inline 1264 B stack / 3.9 KB spills
  (+1300 us), out of line (noinline helper) 944 B stack (+1000 us). Reverted: the B=8 kernel has no register room
  for another warp-level phase with 9-chunk loads.
- CH2 = 128 sweep found a latent bug: the bias rows lived at abase + NKT x 4 KB, which for CH2 = 128 is where V goes
  when it sits in xs (corrupt attention measured as a "-434 us" B=8 win: fewer distinct experts). Bias rows now at a
  fixed +64 KB. With that, 128-key sub-chunks in the B=1 kernel only (B=1 pieces are single 128-key chunks: bitwise
  identical): B=1 -31 / -10 / -9 us at ctx 64 / 400 / 900; B=8 keeps 256 (128 is +73 at ctx 900 there).
## s0016 (B=1 128-key sub-chunks): valid 3553 / 5671; s0017 (B=1 kernel without dead paths): valid 3515 / 5645
- With the per-batch-size kernels, compile-time decisions of paths a kernel never takes are measurable at B=1
  (+-2 us): single F round (-9), no expert-parallel w2 path (-49; the same removal was +31 before the 128-key
  change: allocation/layout), deep w13 unconditional (-10, earlier +5). Expected -63, got -38 over HTTP.
- B=8 kernel: no round planning at the end of E (-18, noise level).
## s0018 (B=8: no round planning at E end): valid 3515 / 5608 (best at both)
- Then: block 0's expert-choice recording (rdbg) moved behind RDBG=0: B=1 -14 (B=8 noise). l2_next prefetches:
  off entirely +131 at B=8; off only in the B=1 kernel -16 at ctx 64 but +11 / +34 at ctx 400 / 900: kept on.
  ATT_ACQREL off at B=1: -8 short but it was -18 long: kept on. PFE_QKV off / PFO_ALL off at B=1: +21 / +16.
## s0019 (expert-choice recording off the critical path): valid 3490 / 5606 (best at both; B=1 1.058x vLLM)
- Rejected after: the serving path's host token / progress write (system fence at the step start) moved to the last
  block during application 0's o_proj (+35 / +36 in serve mode: the fence was not the cost, the added code was).
- qkv tasks split evenly over blocks (28-29 (rowtile, K part) tasks each instead of 3-4 whole rowtiles; rowtiles
  shared by two blocks summed by the second to finish via an acq_rel count): B=1 +85, B=8 +163. The hand-off (atomic,
  two block syncs, partial loads) at both ends of every block costs more than the ~11% imbalance. Reverted.
- Attention -> o_proj barrier replaced by per-kv-head output counts (o_proj task k = kv head k's 384 inputs, warp k
  waits for its kv head, copies its slice, runs its task): bitwise identical, B=1 +13, B=8 +43. Each o_proj block
  needs all 8 kv heads, so it still waits for the slowest output; the barrier cost the same as the flags. Reverted.
- Ceil piece split at B=8 (fewer chunks in the piece holding the current position): +42 / +93 at ctx 600 / 900. Reverted.
## Measurement method (late): one engine per process
- Several engines in one process (tools/ab.py, ab2.py) carry placement effects of +-40 us at B=8 (instance 0 is
  consistently fastest; identical libraries 5462 / 5505 / 5542 / 5526 us). One engine per process
  (tools/one_time.py, tools/ab3.sh) reproduces to +-1 us at B=8 and B=1. Re-checked with it: MIXT off +111, NOFR8
  off +65, ATT_ACQREL off +18 (all kept as they are); PUB_REL +8, SCNT_REL +6, BAR_REL 0, PFO_ALL off +3; HALF6
  +44, W13R +95 (rejections confirmed), fused E/F +31 / +57 at ctx 64 / 900; CH2 = 128 at B=8 +2 short, +134 at
  ctx 900; PMAXC 2 / 4 -5 short, +20 / +12 long; ZST 65 / 68 +15 / 0; ptxas -O2, expensive optimisations, extra
  vectorisation: identical; maxrregcount 248: B=1 +10. Adopted: PFE_R = 1 (B=1 -2, B=8 -2..-6), RED_SLOTS 32
  (B=8 -6). Builds that differ only in unrelated code still move by ~10-30 us (layout).
- mixnorm norm weights loaded with the sources (a2's 'weights before the polls' idea): B=8 +483, B=1 +71 (mixnorm's extra 24 live registers spill); reverted.
- norm_f32_rows_w (B>1 loop-gate stream norms) with the whole row + weights per lane in one round of loads instead of the unroll-3 two-pass loop: bitwise identical, B=8 +147 (lone-process). Reverted.
## s0020 (PFE_R, 32 reduction slots): valid 3489 / 5593 (best at both)
- Lone-process re-checks after s0020: norm_f32_rows_w unroll 1 / 2 / 9 vs 3: +60 / +10 / +36; PFE_QKV off +7;
  NO_L2BIG off (big next-layer L2 prefetches) +294; GMIX (mixer-normalised loop streams) +123 (confirmed).
- w2 -> combine barrier replaced by per-rowtile unit counts (a release increment per w2 unit, the combining block polls its rowtile): bitwise identical, B=8 +107 / +126 (ctx 64 / 900): ~5300 release operations per application cost more than one barrier. Reverted.
## s0021 (weights in one arena, 64 KB alignment, 16 KB skew): valid 3506 / 5585 - B=1 worse than s0020 (3489)
- Placement study (one engine per process): arena variants moved B=1 by -50..-20 and B=8 by 0..+55; buffers in an
  arena +56..+80 at B=8. The best variant (B=1 -50 here) was +17 at B=1 in the evaluator (in-process generate 3461 vs
  3438): placement effects depend on the process's allocation history and do not transfer. Arena off again.
- w13 stream issued right after the expert union (before the prefix sums), B=8 kernel: +21 (lone-process). Reverted.
- B=8 attention item timeline at ctx 900 vs 300 (ATTTS): 10.9 vs 7.4 us per item, the difference is the second
  256-key sub-chunk (~3.7 us: K wait, scores, softmax, PV); ~170 us per step. The rest of the ctx-900 vs ctx-64
  step difference in one_time.py comes from the text at those positions routing to more distinct experts.
  Software-pipelining PV(k) with scores(k+1) would save at most ~1.5 us per application at long context: not done.
- Mixers moved to blocks 144-151 (no o_proj / combine rows; MIXHI): B=8 -19 (ctx 64) / -9 (ctx 900), B=1 +3 (lone-process; the noisy multi-engine test had said 0). A throw-away mixnorm on those blocks during o_proj and the w2 combine to warm the instruction cache: +134. 
- Router split in K (re-tested lone-process; the noisy test had said 0): RKQ 2: B=8 -16, B=1 -23; RKQ 3: B=8 +5, B=1 -38; per kernel (3 at B=1, 2 otherwise): B=1 -34, B=8 -17 / -11.
- Re-tested lone-process: GROT (F/G chunk rotation) B=8 -29 / B=1 -7: adopted; APF +25, PCEIL +3 / +9: off.
- Re-tested lone-process: HOSTC (serve host writes off the step start) B=1 +57 / B=8 -9 (serve timing, tools/one_serve.py); FPRE B=8 -11 but B=1 +16 even with the B=1 path unchanged (layout): neither kept.
## s0024 (GROT): valid 3442 / 5513. Then ring 9 x 2 (re-tested lone-process): B=8 -84 (ctx 64) / -70 (ctx 900), B=1 -18, bitwise identical (12 x 2 split every NVFP4 w2 unit into 12 + a slow 6-item partial stage); 6 x 2: B=1 -59 but B=8 +206; 6 x 3 and 3 x 4 worse / fail. B=1 deep w13 path generalised to RI-item stages.

## s0025 (10.68 h): ring stages 9 x 2: HTTP 3453 / 5443 (B=8 -70 vs s0024, B=1 +11). Lone-process said -18 / -84.
Next: per-kernel ring stage size (template parameter RIx through issue_stage / prefetch / item / consume / run_tasks;
RIK = 6 in the B=1 kernel, 9 elsewhere): lone-process B=1 3344.6 (vs 3380.7), B=8 5310.6 (same). t_det, t_ref, t_serve pass.

## s0026 (10.75 h): per-kernel ring stage size (B=1 6 x 2): HTTP 3428 (best, -25) / 5460 (+17: B=8 kernel unchanged).
- RS=3 sweep (deep13 barriers moved out of the way): B=8 6x3 5528, 8x3 5556 vs 9x2 5313: per-stage cost, not depth.
- EISS (early slot refill): a full stage's weight / scale words are loaded to registers, then (fence.proxy.async) the
  slot is refilled by TMA before the stage's MMAs. B=1 kernel 3343 -> 3273 (-70 lone-process); in the B=8 kernel it
  spills (255 regs, 24 B) and costs +354, so it is a per-kernel template switch (EISS bit 0 = B=1 kernel).
- The B=8 lone-process time moved +30 with the change although its SASS is identical but for the order of a few
  zero-initialisations: a placement effect of the binary (padding kernels did not move it). Ignored.

## s0027 (11.01 h): B=1 kernel early ring slot refill: HTTP B=1 3350 (1.102x vLLM: the B=1 1.10x target 3356 is met),
B=8 5455 (0.93x vLLM; B=8 kernel unchanged). w1-like 3411 / 5639.
- EISS per kind in the B=8 kernel: BF16 only +5, MXFP8 only +22, NVFP4 spills (+264): B=8 is decode-bound, not refill-bound.
  RIB8 6/5/4 with EISS: +463/+771/+975. deep13 (B=1 w13) with early refill: +12 (3 stages already in flight).
- Router K parts at B=8: RKQ 3/4 +3/+6.
- CH2=128 (128-key attention sub-chunks) in the B>1 kernels: lone-process B=8 -56 at ctx 64-192, -30 at ~560, +45 at
  ~960; serve path (prompt 64 + 1024) 5461 vs 5491 (-30, repeated). t_det / t_ref / t_serve pass. t_long old vs new
  at B=8: median 0.08 past 512 (rounding cascade or bug?); seq 0 alone-vs-batch is 0.025-0.033 for both builds alike.
  Long-context check against the float32 reference (t_ref --b 8 --n 1200) for both builds before submitting.

## s0028 (11.7 h): CH2=128 in the B>1 kernels: HTTP 3385 / 5466 (B=1 kernel unchanged: +35 is placement); in process
B=8 step_us 5270 vs 5307 (-37), w1-like 5633 vs 5639. B=8 HTTP repeats span 1455-1491 tok/s (+-65 us): single B=8
HTTP figures do not resolve 30 us changes.
- t_ref --b 8 --n 1200 (README tokens, float32 reference) CH2 256 vs 128: identical profiles per 64-position bucket
  (overall median 0.0703 vs 0.0710, top-1 0.990 vs 0.989; ~0.1 past 700 for both on this markup text): the t_long
  0.08 between them is a rounding cascade, not an error. (Run takes 44 min on one card.)
- B=8 knob sweep on the serve path (CH2=128 base 5459-5466): MERGE_IN_C, PCEIL, PFE_O, PUB_REL, SCNT_REL, BAR_SLEEP all
  within +-10.
- B=1 ring stage size with the early refill on: 9 x 2 serve 3330 vs 6 x 2 3377 (-47, repeated); 4: +247, 3: +378.
- Router K parts at B=1 on the 9 x 2 + early-refill base: RKQ 2: -9, 4: -3 vs 3 (serve, one engine): placement level, kept 3.

## s0029 (12.1 h): B=1 ring 9 x 2 with the early refill: HTTP 3328 (1.109x vLLM, best) / 5457 (0.93x vLLM).
- Early refill per kind at B=1 (serve, one engine): all kinds 3330; without BF16 (kind 0: qkv / o_proj / router,
  long streams) 3304 (-26, repeated); without MXFP8 3375; NVFP4+BF16 only 3330. -> EISS 0x06.

## s0030 (12.25 h): early refill only on the expert streams at B=1: HTTP B=1 3307 (best: 1.117x vLLM, 1.31x SGLang),
B=8 5495 (B=8 kernel unchanged; s0027-s0030 B=8 HTTP 5455 / 5466 / 5457 / 5495: placement and repeat spread).
- Last sweep on the s0030 base (B=1 serve, one engine 3305): RKQ 2 3306, RED_SLOTS 24 3305, TPW 12 3308: nothing left
  in these knobs. HEAD = s0030.
- Final sweep at B=8 (serve, one engine, twice): MIXT=0 (no <1,1> mixnorm specialisation) 5439 / 5449 vs 5473 / 5478
  (-32; generic kernel B=2 4725 -> 4375); BAR_SLEEP=1 -9 (B=1 +8); NO_L2BIG=0 +240. t_det / t_ref / t_serve pass. -> MIXT 0.

## s0031 (12.7 h): MIXT 0: HTTP B=1 3313 / B=8 5435 (best B=8: 0.934x vLLM; B=1 within placement of s0030's 3307).
- Stale-switch sweep on s0031 (B=8 serve, base 5452/5455): NOFR8=0 +44, PFO_ALL=0 +26, ATT_ACQREL=0 +17, GROT=0 +48,
  MIXHI=0 +101: all still pay.
- Last B=8 sweep on s0031 (serve, base 5454 / 5438 in the same job): RED_SLOTS 24 5448, 48 5495, RKQ 1 5432,
  BAR_SLEEP 2 5463: all inside the base's own spread except RED_SLOTS 48 (worse). HEAD = s0031 final.
- RKQ8: router GEMV in one K part in the B=8 kernel (B=1 keeps 3, generic 2): serve B=8 5431 / 5430 vs 5448 / 5440
  (-14), B=1 3313 unchanged; t_det, t_serve pass; t_ref --b 8: median 0.0106, top-1 0.988.

## s0032 (12.75 h): RKQ8=1: HTTP 3314 / 5469 (B=8 +34 vs s0031; w1-like -13, step_us +19): no gain, RKQ8 back to 2
(HEAD's kernels = s0031's). Final: s0030 best B=1 3307 (1.117x vLLM), s0031 best B=8 5435 (0.934x vLLM) at B=1 3313.
