# s4-north-h100-a1: North-Mini-Code-1.0 on H100

Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 23 submissions so far, 23 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 1973
- 1.10x the fastest general engine: 4590
- vLLM 0.32.0 (general engine): 5049
- Cohere's megakernel @ 67d0b9ca (megakernel written for this model): 3616

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: v1 megakernel: one persistent kernel, BF16 weights in mma fragment order

- status: **valid**, step_us_b1 = 3497, at 0.51 agent-hours, category structure
- hypothesis: track A: weights keep their BF16 bits, only reordered into tensor-core fragment order; activations BF16, softmax/norm float32, logits returned as float32 accumulators. One persistent kernel per step (132 blocks x 16 warps, 4 barrier-separated stages per layer, batch in the 8 mma columns). Dev run measured 3492 us at B=1 and 8656 us at B=8, so I expect about 3500 / 8700 here (generate about the same), gates passing with rel-L2 median 0.06-0.07, and single_launch + device_loop.
- evaluator: valid: 3501 us/step at B=1 (MBU 0.565), 8948 us/step at B=8

### #2 s0002: v2: static per-block schedule, norm and residual combined in shared memory

- status: **valid**, step_us_b1 = 3050, at 0.79 agent-hours, parent s0001, category scheduling
- hypothesis: track A (BF16 weights bit-identical in mma fragment order; logits float32). Replace dynamic task claiming and the leader's serial sections by a static per-block program: norm per block in shared memory, q|k|v one chunk per warp, block r owns residual row tile r (o_proj + expert down rows combined in shared memory). Removes ~10 global round trips per layer. Dev: 3056 us B=1, 7845 us B=8; expect the same here (about -440 us at B=1, -1100 at B=8 vs s0001).
- evaluator: valid: 3056 us/step at B=1 (MBU 0.648), 7923 us/step at B=8

### #3 s0003: OPERATOR: s0001 again, unchanged (evaluated again under the HTTP headline, harness 418e7ad)

- status: **valid**, http_step_us_b1 = 3603 (B=8: 8995), at 1.09 agent-hours, parent s0001, category structure
- hypothesis: OPERATOR re-evaluation of s0001 (evaluated again under the HTTP headline, harness 418e7ad); source unchanged, sha256 8e5048bd353db10f. The agent's hypothesis for s0001 was: track A: weights keep their BF16 bits, only reordered into tensor-core fragment order; activations BF16, softmax/norm float32, logits returned as float32 accumulators. One persistent kernel per step (132 blocks x 16 warps, 4 barrier-separated stages per layer, batch in the 8 mma columns). Dev run measured 3492 us at B=1 and 8656 us at B=8, so I expect about 3500 / 8700 here (generate about the same), gates passing with rel-L2 median 0.06-0.07, and single_launch + device_loop.
- evaluator: valid: 3603 us/step at B=1 and 8995 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3496 us/step at B=1 (generate, MBU 0.565)

### #4 s0004: OPERATOR: s0002 again, unchanged (evaluated again under the HTTP headline, harness 418e7ad)

- status: **valid**, http_step_us_b1 = 3169 (B=8: 7927), at 1.16 agent-hours, parent s0002, category scheduling
- hypothesis: OPERATOR re-evaluation of s0002 (evaluated again under the HTTP headline, harness 418e7ad); source unchanged, sha256 41ce59b381f65776. The agent's hypothesis for s0002 was: track A (BF16 weights bit-identical in mma fragment order; logits float32). Replace dynamic task claiming and the leader's serial sections by a static per-block program: norm per block in shared memory, q|k|v one chunk per warp, block r owns residual row tile r (o_proj + expert down rows combined in shared memory). Removes ~10 global round trips per layer. Dev: 3056 us B=1, 7845 us B=8; expect the same here (about -440 us at B=1, -1100 at B=8 vs s0001).
- evaluator: valid: 3169 us/step at B=1 and 7927 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3047 us/step at B=1 (generate, MBU 0.648)

### #5 s0005: v4: balanced quantum schedule, shallow HBM queue (2 loads in flight), leader-first routing

- status: **valid**, http_step_us_b1 = 3196 (B=8: 7615), at 1.98 agent-hours, parent s0004, category scheduling
- hypothesis: track A (BF16 weights bit-identical in mma fragment order; logits float32). Measured that under a saturated HBM queue every atomic/poll/fence waits (bytes in flight)/3.1 TB/s, so: 2 loads in flight per thread instead of 16, model pointer table in shared memory, stage A exactly balanced (20 k-steps per warp) with the router rows first and the leader only routing, every tile summed from 16 fixed quanta. Dev: 2848 us B=1 and 7254 us B=8; expect about 2850 / 7250 here (-200 / -670 vs s0002); batch bitwise equality kept.
- evaluator: valid: 3196 us/step at B=1 and 7615 at B=8 over HTTP (offline (step through the cohort adapter)); in process 2842 us/step at B=1 (generate, MBU 0.695)

### #6 s0006: v5: attention in 8 position chunks over 8 blocks; device loop behind submit/poll

- status: **valid**, http_step_us_b1 = 2933 (B=8: 7257), at 3.32 agent-hours, parent s0005, category attention
- hypothesis: track A (BF16 weights bit-identical in mma fragment order; logits float32; attention weights exp(s-max) as BF16, normalised in float32 after the V sum). Two changes forced by the HTTP workload (contexts to 1088): (1) the K/V of one (sequence, head) no longer sit on one SM, which under the stream only gets ~24 GB/s: 8 self-contained position chunks merged at the end, 8 blocks per task at batch 1; (2) submit/poll runs a whole cohort as one launch and the host only polls a progress word from a second stream. Dev: 2921-2955 us at B=1 flat in context (v4: 2848 at ctx 64-320 but 3470 at 800-1056), B=8 7365 short / 8240 long. Expect http 2960 at B=1 (-210 vs s0004) and about 7800 at B=8 (no gain expected there: attention on one block per task is the straggler at long context).
- evaluator: valid: 2933 us/step at B=1 and 7257 at B=8 over HTTP (online (submit/poll)); in process 2901 us/step at B=1 (generate, MBU 0.681)

### #7 s0007: track B v6: lossless 12-bit weights (row exponents, neuron shifts, exact escapes)

- status: **valid**, http_step_us_b1 = 2628 (B=8: 6938), at 3.77 agent-hours, parent s0006, category memory
- hypothesis: track B: weights stored as 12 bits (W12-style: sign, 4-bit exponent code, 7 mantissa bits) instead of BF16, with one exponent byte per row, one power-of-two shift per expert neuron for the down columns and 1.2e-4 escapes kept exactly; verified bit for bit on all 30.5e9 weights (tools/verify_w12.py). x operands are stored times 2^96 so the code sits in the bf16 exponent field and decoding is a byte permutation plus 5 logic ops. Everything else as s0006. 25% fewer weight bytes; dev shows 2590 us at B=1 and 6648 at B=8 in process (track A 2921 / 7365), so I expect http about 2610 at B=1 (-320 vs s0006) and 6550 at B=8 (-700), i.e. beats_cohere at both.
- evaluator: valid: 2628 us/step at B=1 and 6938 at B=8 over HTTP (online (submit/poll)); in process 2609 us/step at B=1 (step, MBU 0.759)

### #8 s0008: track B: layer 0 from token tables (q|k|v and MLP output per token id)

- status: **valid**, http_step_us_b1 = 2598 (B=8: 6698), at 4.76 agent-hours, parent s0007, category structure
- hypothesis: track B: as s0007 (12-bit weights) plus one model-specific change: layer 0's input is the embedding row and the block is parallel, so its RMSNorm, q|k|v (before RoPE) and dense MLP output are functions of the token id; they are read from two tables built at load (3.76 GB) instead of being computed, 0.9% fewer bytes per step and no stage A / gate|up / down in layer 0. In process (same binary, tables on/off): 6605 vs 6666 us at B=8, 2586 vs 2573 at B=1. Expect http about 2630 at B=1 (no change vs s0007) and about 6880 at B=8 (-60).
- evaluator: valid: 2598 us/step at B=1 and 6698 at B=8 over HTTP (online (submit/poll)); in process 2608 us/step at B=1 (step, MBU 0.759)

### #9 s0009: track B: per-layer skeleton trimmed (residual add on 4 warps, top-8 by warp reduce, norm totals per column)

- status: **valid**, http_step_us_b1 = 2572 (B=8: 6699), at 5.63 agent-hours, parent s0008, category scheduling
- hypothesis: track B: the ablation builds say 826 us (B=1) / 1479 us (B=8) of a step are per-layer skeleton, not bytes. Residual add spread over four warps, top-8 by one warp-reduce instruction per round, gate|up tile count computed once, norm totals once per column at batch>1, 2^96 folded into the norm weight: skeleton build 826->747 / 1479->1384 us. Full build in process moved less (2588->2555, 6603->6630, within build-to-build noise of 1%), so I expect http 2555-2575 at B=1 (s0008: 2598) and 6650-6700 at B=8 (6698). Results bitwise batch-invariant, forced routing 0.0112, serve tokens == step greedy.
- evaluator: valid: 2572 us/step at B=1 and 6699 at B=8 over HTTP (online (submit/poll)); in process 2557 us/step at B=1 (generate, MBU 0.773)

### #10 s0010: track B: certified int8 sketch head on the greedy path (integer tensor core, exact survivors)

- status: **valid**, http_step_us_b1 = 2474 (B=8: 6616), at 5.72 agent-hours, parent s0009, category gemm
- hypothesis: track B: on the serving path the head only has to give the argmax. Pass 1 scores all 262144 rows from an int8 sketch with the integer tensor-core instruction (541 MB instead of 805 MB, no widening) with a certified per-row bound (Cauchy-Schwarz on the stored quantization error, input as two exact int8 digits); pass 2 computes exactly only the tiles whose upper bound reaches the best lower bound (1.2-1.5 of 16384 per sequence in dev), the block widening a tile together (5.7 us). Dev: tokens identical to step-by-step greedy in 4608 of 4608 cases, head 305->192 us, generate 2453 vs step 2551 at B=1. Expect http 2470-2490 at B=1 (s0009: 2572) and 6600-6650 at B=8 (6699). step() still returns the full exact logits.
- evaluator: valid: 2474 us/step at B=1 and 6616 at B=8 over HTTP (online (submit/poll)); in process 2456 us/step at B=1 (generate, MBU 0.775)

### #11 s0011: track B: stage B start balance (early union, leader first, delay-aware gate|up shares)

- status: **valid**, http_step_us_b1 = 2412 (B=8: 6421), at 7.00 agent-hours, parent s0010, category scheduling
- hypothesis: track B: per-block waits at the stage B barrier (new per-block profile) showed blocks start stage B at different times (leader after routing, attention blocks after their task) but got fixed shares, so ordinary blocks waited 4.2 us/layer at B=1. Now the leader reads its stage-A half tile and signals before routing, publishes the expert union first (blocks start gate|up from it; slots/weights follow for stage C), and each block's gate|up share is computed per step from fitted start delays (leader routing chain, attention cost by context length) so all classes finish together. Dev: wait B 4.2->2.8 us/layer, in-process generate 2456->2395 (B=1), serve 2390 / 5913. Expect http 2400-2420 at B=1 (s0010: 2474) and 6350-6500 at B=8 (6616). Gates as before: bitwise batch invariance, forced routing 0.0112, serve == step greedy.
- evaluator: valid: 2412 us/step at B=1 and 6421 at B=8 over HTTP (online (submit/poll)); in process 2414 us/step at B=1 (generate, MBU 0.788)

### #12 s0012: track B: 12-bit format re-laid out for a 3-instruction decode (rotate + LOP3)

- status: **valid**, http_step_us_b1 = 2366 (B=8: 6382), at 7.23 agent-hours, parent s0011, category memory
- hypothesis: track B: the stream inside stage B runs at ~15 GB/s per SM because widening the 12-bit weights is instruction-bound (a throwaway build with a 3-instruction decode: -63 us at B=1). The high bits (sign, exponent-code bits 1-3) of a lane's 16 weights are now stored as two 32-bit words in which register q's bits sit pre-rotated by {0,4,8,28}[q&3]: one funnel shift and one LOP3 (with the byte permute of the low bytes) give the bf16 register, instead of ~12 operations. Same 12.07 bits/weight; verify_w12 on all 30.5e9 weights: every tile decodes to the checkpoint's bits; logits and greedy tokens hash-identical to s0011's build. Dev: generate 2395->2349 at B=1, serve 2347 / 5881. Expect http 2355-2375 at B=1 (s0011: 2412) and 6350-6400 at B=8 (6421).
- evaluator: valid: 2366 us/step at B=1 and 6382 at B=8 over HTTP (online (submit/poll)); in process 2350 us/step at B=1 (generate, MBU 0.814)

### #13 s0013: track B: o_proj in stage B at batch > 1 (attention-done stamps; shares include it)

- status: **valid**, http_step_us_b1 = 2380 (B=8: 6310), at 7.80 agent-hours, parent s0012, category scheduling
- hypothesis: track B: o_proj needs only the attention output, not gate|up, but it ran after the stage B barrier. Merging attention blocks now stamp each task when its output rows are stored; at batch > 1 every block runs its o_proj row tile right after its gate|up share (once all tasks are stamped) and waits at the barrier only for the down projections; the share model counts o_proj as 2.5 tiles of delay for blocks 0..127. Dev: stage-B barrier wait 2.3 -> 0.6 us/layer; B=8 generate 6402 -> 6294, serve 5903 -> 5789. At batch 1 it was 12-45 us slower (long contexts: attention is not done when gate|up is), so batch 1 keeps the old order (a second instantiation of the kernel; a run-time switch slowed both). Expect http B=1 unchanged (2360-2375; s0012: 2366) and B=8 6260-6300 (6382).
- evaluator: valid: 2380 us/step at B=1 and 6310 at B=8 over HTTP (online (submit/poll)); in process 2363 us/step at B=1 (generate, MBU 0.809)

### #14 s0014: track B: RMSNorm rounding with cvt.rn.bf16x2 (bit-identical, fewer instructions)

- status: **valid**, http_step_us_b1 = 2373 (B=8: 6240), at 8.51 agent-hours, parent s0013, category other
- hypothesis: track B: every block computes the full RMSNorm redundantly; at B=8 that is 16K elements per SM at ~14 instructions each (two manual round-to-nearest-even bf16 conversions per element), 5.5 us per layer. The hardware cvt.rn.bf16x2.f32 rounds two floats in one instruction with the same bits for finite values: logits and tokens hash-identical to the previous build (tools/logit_hash.py). Dev, alternating builds on one card twice: B=8 generate 6288 -> 6233-6241 (-50 us), B=1 unchanged (2357-2369 both). Expect http B=1 2365-2385 (s0013: 2380), B=8 6250-6275 (6310).
- evaluator: valid: 2373 us/step at B=1 and 6240 at B=8 over HTTP (online (submit/poll)); in process 2360 us/step at B=1 (generate, MBU 0.810)

### #15 s0015: track B: stage C starts without dependent round trips (late escape gather, local B=1 routing)

- status: **valid**, http_step_us_b1 = 2365 (B=8: 6207), at 8.93 agent-hours, parent s0014, category scheduling
- hypothesis: track B: at the start of stage C every warp loaded its tile header, then (batch 1) the x values of the header's escape entries, and the block loaded the routing details (slot masks, slots per rank, weights) -- three dependent round trips under a full queue before the first weight load. Now the escape gather follows the first quantum's loads, and at batch 1 the slot masks and slots are derived locally (the sequence's experts are the union) while the weights' load is consumed only in the residual add. Bits unchanged (same arithmetic, same order). Dev, two copies each on different cards: B=1 generate 2442/2430 -> 2418/2408 (with both), B=8 6302 -> 6265. Expect http B=1 2345-2360 (s0014: 2373), B=8 6200-6230 (6240).
- evaluator: valid: 2365 us/step at B=1 and 6207 at B=8 over HTTP (online (submit/poll)); in process 2341 us/step at B=1 (generate, MBU 0.816)

### #16 s0016: track A: s0006 + the scheduling work since (shares by start delay, o_proj in B at B>1, decode-free paths)

- status: **valid**, http_step_us_b1 = 2785 (B=8: 7157), at 13.21 agent-hours, parent s0015, category scheduling
- hypothesis: track A: same BF16 weights and data flow as the general engines; only kernel organisation changed since s0006 (2933/7257): leader signals stage A before routing and publishes the union first, gate|up shares from measured start delays, o_proj in stage B at batch > 1 (template instantiation), stage C without dependent round trips, early residual rows, RMSNorm with cvt.rn.bf16x2. No 12-bit format, no sketch head, no layer-0 tables. Dev (MK_TRACK=A): gates pass (bitwise single==batched, forced routing 0.0111, serve tokens == step greedy), serve 2779 us B=1 / 6866 us B=8. Expect http 2790-2810 at B=1, 6880-6950 at B=8.
- evaluator: valid: 2785 us/step at B=1 and 7157 at B=8 over HTTP (online (submit/poll)); in process 2774 us/step at B=1 (generate, MBU 0.712)

### #17 s0017: track B: weight loads skip L1; zero operand from shared memory (stage C x stays in L1)

- status: **valid**, http_step_us_b1 = 2252 (B=8: 6049), at 15.44 agent-hours, parent s0016, category memory
- hypothesis: track B: with ~220 KB of shared memory per block, L1 is ~28 KB; stage C reads its x operands (h of the chosen experts, attn, and a zero vector for unused batch columns) from global memory again for every k-step and relies on L1 hits, while the streamed weights allocate in L1 and evict them (a build with all loads bypassing L1 was +2406 us). Weight loads now use ld.global.L1::no_allocate and the zero operand comes from shared memory. Same arithmetic, same bits. Dev: same-card A/B (3 rounds) -90 us at B=1, -273 at B=8; gates pass; generate 2237 / 5930, serve 2237 / 5391. Expect http B=1 2255-2280 (s0015: 2365), B=8 5900-5960 (6207).
- evaluator: valid: 2252 us/step at B=1 and 6049 at B=8 over HTTP (online (submit/poll)); in process 2238 us/step at B=1 (generate, MBU 0.852)

### #18 s0018: track B: o_proj in stage B at batch 1 too (now that stage C's x stays in L1)

- status: **valid**, http_step_us_b1 = 2187 (B=8: 6061), at 18.60 agent-hours, parent s0017, category scheduling
- hypothesis: track B: at batch 1, running each block's o_proj row tile right after its gate|up share (attention-done stamps, shares count it) was mixed before s0017 (-11..+67 us). With the weight stream no longer evicting the x operands from L1, the same change is now -68 us at contexts 64-320 and -84 at 800-992 in a same-card A/B (e0350, 2 rounds). Same arithmetic and bits. Gates pass (e0351): bitwise single==batched, forced routing 0.0112, generate == step, serve == step greedy; generate 2168 / 5933, serve 2159-2164 / 5383-5395. Expect http B=1 2170-2195 (s0017: 2252), B=8 about unchanged 6000-6060 (6049).
- evaluator: valid: 2187 us/step at B=1 and 6061 at B=8 over HTTP (online (submit/poll)); in process 2149 us/step at B=1 (generate, MBU 0.880)

### #19 s0019: track A: BF16 engine + L1 no-allocate weight loads, o_proj in stage B (kernel organisation only)

- status: **valid**, http_step_us_b1 = 2573 (B=8: 6904), at 19.70 agent-hours, parent s0018, category memory
- hypothesis: track A: same BF16 weights and data flow as general engines. Since s0016 (2785/7157) only kernel organisation and memory hints changed: weight loads with ld.global.L1::no_allocate and the zero operand from shared memory (stage C's x operands stay in L1), o_proj in stage B at every batch size. Dev (MK_TRACK=A, e0356): gates pass (bitwise single==batched, forced routing 0.0111, generate == step, serve tokens == step greedy); generate 2560 / 6823, serve 2555 / 6540, long contexts 2580 / 7544. Expect http 2570-2600 at B=1, 6950-7100 at B=8.
- evaluator: valid: 2573 us/step at B=1 and 6904 at B=8 over HTTP (online (submit/poll)); in process 2559 us/step at B=1 (generate, MBU 0.771)

### #20 s0020: track B: gate|up results and residual rows as exchanges instead of store + fence

- status: **valid**, http_step_us_b1 = 2173 (B=8: 6039), at 20.87 agent-hours, parent s0019, category sync
- hypothesis: track B: the last block of stage B and of stage C each ends with plain stores and a __threadfence; under the weight stream a fence costs several us and sits on the critical path. The h rows and the residual rows now go out as exchanges (the thread waits only for their return); every SM still fences once per layer in stage A, which keeps L1 coherent. Same bits. Dev: same-card A/B 2 rounds (e0358): -9 us at B=1 contexts 64-320, -20 at 800-992, B=8 within noise; gates pass (bitwise, forced 0.0112, generate == step, sketch, serve). Expect http B=1 2170-2185 (s0018: 2187), B=8 6040-6070 (6061).
- evaluator: valid: 2173 us/step at B=1 and 6039 at B=8 over HTTP (online (submit/poll)); in process 2133 us/step at B=1 (generate, MBU 0.893)

### #21 s0021: track B: union load issued during the last stage-A quantum (batch-1 kernel)

- status: **valid**, http_step_us_b1 = 2146 (B=8: 6028), at 21.65 agent-hours, parent s0020, category sync
- hypothesis: track B: at batch 1 blocks finish stage A at ~+9.2 us after the barrier but see the routing union only at ~+10.4: one poll round trip after their own stage A (trace e0346/e0352). Thread 0 now issues the union's load with its last stage-A quantum and checks it after stage A; if it is already published, gate|up starts without a further round trip (else it polls as before). As code in the shared kernel it cost B=8 64-219 us (layout), so it is a separate instantiation used only at batch 1. Same bits. Dev: same-card A/B 3 rounds (e0363) B=1 -26 us short / -20 long, B=8 -10; gates pass. Expect http B=1 2145-2160 (s0020: 2173), B=8 6020-6050 (6039).
- evaluator: valid: 2146 us/step at B=1 and 6028 at B=8 over HTTP (online (submit/poll)); in process 2131 us/step at B=1 (generate, MBU 0.894)

### #22 s0022: track A: s0019 + exchanges instead of fences, early union load at batch 1

- status: **valid**, http_step_us_b1 = 2533 (B=8: 6893), at 21.99 agent-hours, parent s0021, category sync
- hypothesis: track A: BF16 weights and the general engines' data flow; since s0019 (2573/6904) only synchronisation changed: gate|up results and residual rows go out as exchanges instead of store + fence, and the batch-1 kernel issues the routing union's load with its last stage-A quantum. Dev (MK_TRACK=A, e0366): gates pass; generate 2530 / 6808, serve 2526 / 6520, long contexts 2554 / 7647. Expect http 2540-2560 at B=1, 6880-6950 at B=8.
- evaluator: valid: 2533 us/step at B=1 and 6893 at B=8 over HTTP (online (submit/poll)); in process 2519 us/step at B=1 (generate, MBU 0.783)

### #23 s0023: track B: share model weights attention's per-K-tile cost more (long contexts)

- status: **valid**, http_step_us_b1 = 2150 (B=8: 5989), at 22.64 agent-hours, parent s0022, category scheduling
- hypothesis: track B: the gate|up shares come from modelled start delays; attention's delay grows with the context, and its per-K-tile slope (5/64 tile per chunk and K tile) was fitted at short contexts. With 8, attention blocks get fewer gate|up tiles at long contexts and finish with the others. Work distribution only, same bits. Same-card A/B (e0371, 2 rounds): B=1 0 us at contexts 64-320, -16.5 at 800-992; B=8 -30 / -55. Expect http B=1 2135-2145 (s0021: 2146), B=8 5990-6010 (6028).
- evaluator: valid: 2150 us/step at B=1 and 5989 at B=8 over HTTP (online (submit/poll)); in process 2131 us/step at B=1 (generate, MBU 0.895)

