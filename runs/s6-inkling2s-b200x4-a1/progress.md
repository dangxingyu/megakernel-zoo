# s6-inkling2s-b200x4-a1: Inkling-2-Small on GB200

Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 14 submissions so far, 11 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 603
- 1.10x the fastest general engine: 4283
- SGLang PR #50186 image lmsysorg/sglang:dev-inkling-2-small (arm64 sha256:198af079c599) (general engine): 4711

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Persistent per-card kernel, TP4 attention + intra-expert TP4, B=1 redundant mode

- status: **error**, no score, at 4.16 agent-hours, category structure
- hypothesis: track A: checkpoint formats (BF16 dense in MMA-fragment order, NVFP4/MXFP8 experts as stored, inputs quantized as defined). First valid engine: in-process measurements give ~8.3 ms/step at B=1 and ~10.9 ms at B=8 at short context, so expect http_step_us_b1 about 8500-9500 (slower than SGLang 4711) and all gates passing (own float64 check: median 0.05, top-1 100% on 24 positions).
- evaluator: the evaluation was killed after 10800.0 s

### #2 s0002: Fix deadlock (header by value); relaxed signals; smem-staged B frags; phases pre-wait

- status: **valid**, http_step_us_b1 = 6564 (B=8: 1.033e+04), at 4.66 agent-hours, parent s0001, category sync
- hypothesis: track A: checkpoint formats (BF16 dense in MMA-fragment order, NVFP4/MXFP8 experts as stored, inputs quantized as defined). s0001 deadlocked (header ring overrun); this build passed a rehearsal of the evaluator's unsynchronised loops and the online path; in-process w1-like 6.67 ms (B=1) / 10.4 ms (B=8), so expect http_step_us_b1 ~6700 and B=8 ~10400, all gates passing.
- evaluator: valid: 6564 us/step at B=1 and 10330 at B=8 over HTTP (online (submit/poll)); in process 6552 us/step at B=1 (step, MBU 0.092)

### #3 s0003: 768-thread blocks, block-contiguous GEMV, per-strip GEMVs, vectorised token phases

- status: **valid**, http_step_us_b1 = 5324 (B=8: 9310), at 5.50 agent-hours, parent s0002, category structure
- hypothesis: track A: checkpoint formats unchanged. Several latency fixes found with in-kernel timers since s0002: 768 threads (80 regs, half the spills), block-contiguous GEMV ranges (strided ranges made some blocks 2x slow), block-per-strip GEMV for qkv/o_proj/w2, 8-element vector code for the residual mixes, new attention. Rehearsal: online 5328 us (B=1) / 9458 us (B=8); expect http_step_us_b1 ~5350 (-18% vs s0002) and B=8 ~9450.
- evaluator: valid: 5324 us/step at B=1 and 9310 at B=8 over HTTP (online (submit/poll)); in process 5318 us/step at B=1 (generate, MBU 0.113)

### #4 s0004: LL tagged-data cross-card exchange, router per-strip, attention and mix fixes

- status: **incorrect**, no score, at 6.17 agent-hours, parent s0003, category sync
- hypothesis: track A: checkpoint formats unchanged. Cross-card values travel as (value, tag) 8-byte stores polled by the consumer (no system fence, grid barrier or counter per exchange), router by block-per-strip, attention K/V row on 8 warps, batched mix reductions. Rehearsal online 4797 us (B=1) / 9103 us (B=8): expect http_step_us_b1 ~4800 (-10% vs s0003) and B=8 ~9100.
- evaluator: incorrect: failed http_tokens
- gates that failed: http_tokens

### #5 s0005: Slot-order invariant expert scaling (explicit FMA) on top of s0004

- status: **valid**, http_step_us_b1 = 4778 (B=8: 8845), at 6.41 agent-hours, parent s0004, category correctness
- hypothesis: track A: checkpoint formats unchanged. s0004 failed http_tokens because a sequence's logits depended on its batch slot (1-ulp FMA contraction asymmetry between even/odd MMA columns in the expert scaling) and the HTTP threads reorder requests; explicit FMAs make the batch bitwise permutation invariant (tested). Expect valid with http_step_us_b1 ~4800 and B=8 ~9150.
- evaluator: valid: 4778 us/step at B=1 and 8845 at B=8 over HTTP (online (submit/poll)); in process 4751 us/step at B=1 (generate, MBU 0.127)

### #6 s0006: HW E2M1 rounding, 64-position attention chunks at B=1, w2 prefetch only at B=1

- status: **valid**, http_step_us_b1 = 4581 (B=8: 8242), at 6.84 agent-hours, parent s0005, category gemm
- hypothesis: track A: checkpoint formats unchanged. Hardware cvt.rn.satfinite.e2m1x2 for the NVFP4 input quantization (bitwise identical logits, quantization 3.7 -> 1.4 us per layer), attention chunks of 64 at B=1 (attention 11.6 -> 9.9 us), no w2 L2 prefetch at B=8 (w13 46 -> 34 us). Rehearsal online 4588 / 8461 us: expect http_step_us_b1 ~4590 (-4% vs s0005) and B=8 ~8450 (-4%).
- evaluator: valid: 4581 us/step at B=1 and 8242 at B=8 over HTTP (online (submit/poll)); in process 4526 us/step at B=1 (generate, MBU 0.133)

### #7 s0007: B=8 expert GEMVs: w2 weights+fragments in flight across experts, quant2 warp tasks; split-K router

- status: **valid**, http_step_us_b1 = 4541 (B=8: 7731), at 8.51 agent-hours, parent s0006, category scheduling
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); only scheduling changes. At B=8 the w2 activation-fragment loads waited for the weight loads (serial L2 trips) and quant2 ran 2.5 block waves; loading both together across expert boundaries, warp-level quant2 and FP4 unroll 4 are bitwise identical and should cut B=8 from 8242 to ~7650 us over HTTP. The router GEMV ran on 24 SMs; split-K over 72 SMs with the 3 partials summed in order by the routing should take ~45 us off B=1 (4581 -> ~4540).
- evaluator: valid: 4541 us/step at B=1 and 7731 at B=8 over HTTP (online (submit/poll)); in process 4500 us/step at B=1 (generate, MBU 0.134)

### #8 s0008: Cross-card polls with all loads in flight before spinning; redux top-8; prefetch by last warp

- status: **valid**, http_step_us_b1 = 4311 (B=8: 7477), at 9.32 agent-hours, parent s0007, category sync
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); bitwise identical to s0007. The all-reduce reads (raw8), the amax and router polls spun on one LL value after another (each spin blocks the loads behind it: up to 16 serial L2 trips), the top-8 used 8 shuffle argmax rounds, and warp 0 issued the bulk L2 prefetches (~0.5 us each) on the critical path. Batched loads, redux.sync top-8 and prefetch from the idle last warp cut ~220 us in process at B=1: expect ~4300 us over HTTP at B=1 (from 4541) and ~7450 at B=8 (from 7731).
- evaluator: valid: 4311 us/step at B=1 and 7477 at B=8 over HTTP (online (submit/poll)); in process 4273 us/step at B=1 (generate, MBU 0.141)

### #9 s0009: Load-depth tuning per mode (FP4/MXFP8 unroll), router bias before the poll, async conv loads at B>=2

- status: **valid**, http_step_us_b1 = 4260 (B=8: 7415), at 10.03 agent-hours, parent s0008, category memory
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); bitwise identical to s0008. The best number of expert-weight units in flight per warp differs by mode (B=1: ~11 units per warp, unroll 3 beats 4; B=8: 6 beats 4; MXFP8 1 vs 3), the router bias load sat behind the logit poll, and at B>=2 the short-conv history/taps can be copied into shared memory while the cross-card values are awaited. Expect ~45 us less at B=1: ~4265 over HTTP (from 4311, under the 4283 line), and ~7400 at B=8 (from 7477).
- evaluator: valid: 4260 us/step at B=1 and 7415 at B=8 over HTTP (online (submit/poll)); in process 4216 us/step at B=1 (generate, MBU 0.143)

### #10 s0010: Attention starts on a counter of the qkv strips it reads; acq_rel hand-off atomics

- status: **valid**, http_step_us_b1 = 4078 (B=8: 7214), at 10.45 agent-hours, parent s0009, category sync
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); bitwise identical to s0009. qkv has 237-239 strips on 152 SMs (two waves); attention waited at a grid barrier for both, but reads only q, r, k, v (140-142 strips). Ordering those into the first wave and waiting on their counter lets attention overlap the gate rows' wave (in-process B=1 -174 us at ctx 576). Hand-offs used fence.sc + atomic (+ fence); one acq_rel atomic / acquire fence instead: B=1 -58 us, B=8 -90 us. Expect ~4070 us over HTTP at B=1 (from 4260) and ~7220 at B=8 (from 7415).
- evaluator: valid: 4078 us/step at B=1 and 7214 at B=8 over HTTP (online (submit/poll)); in process 4033 us/step at B=1 (generate, MBU 0.150)

### #11 s0011: Fix B>=2 combine/gate-row race (barrier back at B>=2); paired o_proj strips

- status: **valid**, http_step_us_b1 = 4049 (B=8: 7260), at 11.22 agent-hours, parent s0010, category correctness
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); values bitwise identical to s0010. s0010's early attention let the B>=2 combine read qkv gate rows of the second wave without waiting (a race that the timing hid); B>=2 goes back to the grid barrier (B=8 +~40 us), B=1 keeps the counter (its combine is behind a barrier). Paired strips: o_proj blocks with two strips load both in one round (B=1 -17 us), the B>=2 w2 reduces its two strips in one round (B=8 -50 us). Expect ~4055 us at B=1 over HTTP (from 4078) and ~7200 at B=8 (7214).
- evaluator: valid: 4049 us/step at B=1 and 7260 at B=8 over HTTP (online (submit/poll)); in process 3999 us/step at B=1 (generate, MBU 0.151)

### #12 s0012: Register relief: 32-bit GEMV indices, launch state in shared memory; two-chain B=1 quant2

- status: **valid**, http_step_us_b1 = 3959 (B=8: 7194), at 12.16 agent-hours, parent s0011, category memory
- hypothesis: track A: same formats as the checkpoint (NVFP4/MXFP8 experts, BF16 elsewhere); bitwise identical to s0011. The red-mode kernel is capped at 80 registers and spills: the flat GEMV's ten 64-bit index variables and the block-uniform launch state (barrier count, counter bases) held registers through the weight loops; 32-bit indices and shared-memory state free them (B=1 -55 and -14 us, B=8 -83). The B=1 w2-input quantization ran 6 dependent ~0.9 us rounds; two rounds per iteration interleave their chains (-64 us). Expect ~3930 us over HTTP at B=1 (from 4049) and ~7180 at B=8 (from 7260).
- evaluator: valid: 3959 us/step at B=1 and 7194 at B=8 over HTTP (online (submit/poll)); in process 3901 us/step at B=1 (generate, MBU 0.155)

### #13 s0013: Cross-card partial sums as bf16 pairs (half the polled bytes); adaptive B>=2 chunks

- status: **valid**, http_step_us_b1 = 3624 (B=8: 6955), at 12.66 agent-hours, parent s0012, category memory
- hypothesis: track A: checkpoint formats (NVFP4/MXFP8 experts, BF16 elsewhere); the o_proj and w2 tensor-parallel partials now cross cards as bf16 (what a general engine's TP all-reduce sends) instead of fp32, summed in fp32. Every block polls all four cards' partials of the 4608-vector twice per layer (~22 MB of L2 reads per card per exchange); (bf16 pair, tag) halves it: in process B=1 -320 us, B=8 -110, gates unchanged on my float64 check. Expect ~3620 us over HTTP at B=1 (from 3959) and ~6860 at B=8 (from 7194).
- evaluator: valid: 3624 us/step at B=1 and 6955 at B=8 over HTTP (online (submit/poll)); in process 3566 us/step at B=1 (generate, MBU 0.169)

### #14 s0014: B>=2: whole-region K/V prefetch (barrier no longer waits for prefetch issue); routing-masked w2 inputs

- status: **build_failed**, no score, at 13.62 agent-hours, parent s0013, category sync
- hypothesis: track A: checkpoint formats (NVFP4/MXFP8 experts, BF16 elsewhere); values bitwise identical to s0013. At B=8 the 144 non-token blocks issued 32 tiny bulk K/V prefetches each (~0.5 us per issue) before the attn_in grid barrier, which then waited 6.4 us; whole regions per block cut it to 1.8 us (in process B=8 -200 us). The w2 inputs of unrouted (expert, token) pairs were zero-filled by quant2 (5/6 of its tasks); w2 now loads zeros for them from a routing mask (-87 us). B=1 unchanged up to ~10 us of code layout. Expect B=8 ~6680 us over HTTP (from 6955), B=1 ~3630.
- evaluator: engine/build.sh failed (rc 2)

