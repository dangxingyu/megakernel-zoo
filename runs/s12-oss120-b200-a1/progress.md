# s12-oss120-b200-a1: gpt-oss-120b on GB200

Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 59 submissions so far, 57 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 627
- 1.10x the fastest general engine: 1997
- sglang (general engine): 2197
- vllm (general engine): 2453
- vllm_blackwell_recipe (general engine): 2513

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: v1 megakernel: one cooperative kernel, grid barrier between 6 phases per layer

- status: **valid**, http_step_us_b1 = 3613 (B=8: 8775), at 0.51 agent-hours, category structure
- hypothesis: track A (MXFP4 experts as stored incl. E8M0 scales applied in fp32 per 32-group, BF16 elsewhere, fp32 attention scores). First own engine: correctness baseline. Expect all gates to pass (own fp32 ref: median 0.035 vs HF bf16 0.037 on the same tokens) and HTTP ~3600 us at B=1 (generate 3600 in process), B=8 ~6000-9500 depending on how widely real text routes.
- evaluator: valid: 3613 us/step at B=1 and 8775 at B=8 over HTTP (online (submit/poll)); in process 3496 us/step at B=1 (step, MBU 0.179)

### #2 s0002: TMA per-warp weight rings, cross-phase prefetch, fragment-order B operands

- status: **build_failed**, no score, at 0.98 agent-hours, parent s0001, category memory
- hypothesis: track A (formats as s0001). Weights now stream through per-warp cp.async.bulk rings in shared memory (no register cost), the next dense phase's first stages are issued before each grid barrier, B operands are read as shared-space vector loads, expert union built in parallel. In process generate 3600 -> 2372 us at B=1; expect HTTP B=1 ~2450 us (-32%), B=8 ~8000; numeric gates still pass (summation segments changed: new rounding draw).
- evaluator: engine/build.sh failed (rc 1)

### #3 s0003: TMA per-warp weight rings, cross-phase prefetch, fragment-order B operands

- status: **error**, no score, at 0.99 agent-hours, parent s0002, category memory
- hypothesis: track A (formats as s0001). Weights now stream through per-warp cp.async.bulk rings in shared memory (no register cost), the next dense phase's first stages are issued before each grid barrier, B operands are read as shared-space vector loads, expert union built in parallel. In process generate 3600 -> 2372 us at B=1; expect HTTP B=1 ~2450 us (-32%), B=8 ~8000; numeric gates still pass (summation segments changed: new rounding draw). (Resubmission of s0002, which failed to build: header outside engine/.)
- evaluator: the engine failed during the correctness run (rc 1)
- gates that failed: run

### #4 s0004: TMA weight rings + prefetch (s0003 fixed: attention P built from scores, no smem overlap)

- status: **valid**, http_step_us_b1 = 2523 (B=8: 6735), at 1.10 agent-hours, parent s0003, category memory
- hypothesis: track A (formats as s0001). As s0003 (per-warp cp.async.bulk weight rings, next dense phase issued before each barrier, shared-space fragment-order B operands, parallel expert union) with the attention's P array removed (it overlapped the task tables past ~400 positions and crashed). In process 64+512 generate 2412 us B=1, 7552 B=8; expect HTTP ~2500 B=1 (-31% vs s0001) and ~7600 B=8; gates pass.
- evaluator: valid: 2523 us/step at B=1 and 6735 at B=8 over HTTP (online (submit/poll)); in process 2360 us/step at B=1 (generate, MBU 0.265)

### #5 s0005: Down phase at B<=2: even (expert, rowtile) units, partials summed by the next phase

- status: **valid**, http_step_us_b1 = 2276 (B=8: 6738), at 1.21 agent-hours, parent s0004, category scheduling
- hypothesis: track A (formats as s0001). Blocks owned 1 or 2 of the 180 down rowtiles, so 28 blocks did twice the work (F span 16 us/layer). At B<=2 the 720 (expert,rowtile) units are now split evenly and the next phase sums the 4 partials while it builds the norm (bitwise identical to the B=8 path, checked). In process generate 2412 -> 2179 us; expect HTTP B=1 ~2330 (-190 us vs s0004), B=8 unchanged ~6735.
- evaluator: valid: 2276 us/step at B=1 and 6738 at B=8 over HTTP (online (submit/poll)); in process 2108 us/step at B=1 (generate, MBU 0.297)

### #6 s0006: MXFP4 dequant by scaled e2m1 cvt (exact); norm weights bulk-copied into smem early

- status: **valid**, http_step_us_b1 = 2161 (B=8: 6872), at 1.75 agent-hours, parent s0005, category gemm
- hypothesis: track A (MXFP4 codes+E8M0 scales as stored; the scaled cvt yields the exact BF16 dequantised weight, so the mma accumulates like the BF16-weight data flow). Two latency cuts measured separately in process: fp4 decode with one instruction per two weights incl. scale (2160 -> 2125 us), norm weights by bulk copy issued before the barrier instead of dependent L2 loads in the norm (2125 -> 2068). Expect HTTP B=1 ~2215 (-60 vs s0005), B=8 ~6650; accuracy unchanged (own 8x300 test: median 0.0881 vs HF 0.0878).
- evaluator: valid: 2161 us/step at B=1 and 6872 at B=8 over HTTP (online (submit/poll)); in process 2010 us/step at B=1 (generate, MBU 0.311)

### #7 s0007: Attention split into 64-key chunks, one warp each, last warp merges (flash decoding)

- status: **valid**, http_step_us_b1 = 2079 (B=8: 6640), at 2.03 agent-hours, parent s0006, category attention
- hypothesis: track A (formats unchanged; attention scores/softmax fp32, P as bf16 of exp(s-m_chunk) as flash decoding does instead of bf16 of the normalised p). One block per (sequence, kv head) read the whole K/V of its head: 13 us/layer at position 900. Now (seq, head, 64-key chunk) items run on one warp each across all blocks and the last warp merges with the sink. Same-card A/B on prompt 64 + 1024 tokens: generate 2158 -> 2067 us (B=1), 7024 -> 6506 (B=8). Expect HTTP B=1 ~2075 (-85), B=8 ~6350 (-500).
- evaluator: valid: 2079 us/step at B=1 and 6640 at B=8 over HTTP (online (submit/poll)); in process 2049 us/step at B=1 (generate, MBU 0.306)

### #8 s0008: B>=3: expert phases without reduction rounds (warp per unit), even down units + combine step

- status: **valid**, http_step_us_b1 = 2099 (B=8: 4646), at 2.23 agent-hours, parent s0007, category scheduling
- hypothesis: track A (formats unchanged; bits identical to B=1, checked). At B=8 the down phase took 73 us/layer (blocks owned 1-2 rowtiles x all ~24 experts, staged in rounds that drained the weight ring) and gate_up ran in rounds of 10 units. Now gate_up and down use one warp per (expert, rowtile) unit with the 5 segments summed in registers, down units are spread evenly over all blocks, and a row-owner combine step (one extra barrier) forms x. In process (64+512) B=8 7260 -> 4930 us. Expect HTTP B=8 ~4500 (-32%), B=1 unchanged ~2080.
- evaluator: valid: 2099 us/step at B=1 and 4646 at B=8 over HTTP (online (submit/poll)); in process 2066 us/step at B=1 (generate, MBU 0.303)

### #9 s0009: MXFP4 records interleave codes and scales (one bulk copy per stage); batched norm loads

- status: **valid**, http_step_us_b1 = 2071 (B=8: 4325), at 2.39 agent-hours, parent s0008, category memory
- hypothesis: track A (same code and scale bytes, re-laid: each (rowtile, 64-column chunk) is one 544-byte record). A stage needed two bulk copies (4.6 KB codes + 288 B scales); the small one cost about as much as the large (fp4 rings streamed 5.5 TB/s vs 6.85 for BF16 with no compute at all). Also in this build: the norm's global loads issued in two batches, and the B>=3 down phase stages several experts' rows per round. In process 64+512: B=1 2066 -> 2029 us, B=8 4930 -> 4318. Expect HTTP B=1 ~2045, B=8 ~4070; bits unchanged.
- evaluator: valid: 2071 us/step at B=1 and 4325 at B=8 over HTTP (online (submit/poll)); in process 2035 us/step at B=1 (generate, MBU 0.308)

### #10 s0010: Router split 4 ways in K over 32 blocks; E sums the partials

- status: **valid**, http_step_us_b1 = 2035 (B=8: 4279), at 2.61 agent-hours, parent s0009, category scheduling
- hypothesis: track A (router logits still fp32 sums rounded to bf16 before top-4; only the summation grouping changes). The router ran on 8 blocks with 92 KB each, 4 us on the critical path between the o_proj and gate_up barriers. With the K quarters on 32 blocks (23 KB each, all prefetched in the ring before the barrier) the 4 partials are added by every block when it reads the logits. In process 64+512 B=1 2030 -> 2006 us (RQ 1/2/4/8: 2037/2013/2006/2026). Expect HTTP B=1 ~2045 (-25), B=8 unchanged ~4325.
- evaluator: valid: 2035 us/step at B=1 and 4279 at B=8 over HTTP (online (submit/poll)); in process 2000 us/step at B=1 (step, MBU 0.313)

### #11 s0011: Top-4 by one warp integer max per round; expert union in warp 0 (match/reduce)

- status: **valid**, http_step_us_b1 = 1997 (B=8: 4144), at 2.73 agent-hours, parent s0010, category other
- hypothesis: track A (same bf16 router logits and top-4 with lowest-index tie order; the 4 softmax weights now summed pairwise: rounding of the weights may differ in the last bit). The top-4 used 10 shuffles per round on (value, index) pairs and the union a 128-thread pass with 3 block barriers (~2.5 us on the critical path each layer). Now: 16-bit order key | index packed in 32 bits, __reduce_max_sync per round; union, slot ranks and per-sequence slot lists in warp 0 with match_any/reduce_or. In process 64+512 B=1 2006 -> 1969 us. Expect HTTP B=1 ~2000 (-35), B=8 ~4230.
- evaluator: valid: 1997 us/step at B=1 and 4144 at B=8 over HTTP (online (submit/poll)); in process 1963 us/step at B=1 (generate, MBU 0.319)

### #12 s0012: E->F grid barrier replaced by per-expert-slot completion counters

- status: **valid**, http_step_us_b1 = 1959 (B=8: 4017), at 2.95 agent-hours, parent s0011, category sync
- hypothesis: track A (no arithmetic change; bits identical). The down phase only needs the gate_up rows of its own experts; each block now bumps a counter per slot once its gate_up units are written, and down rounds wait for their slots instead of a grid barrier (the ring prefetch of down weights already happened before). Same-card A/B (3 rounds, +-1 us): B=1 1971 -> 1932 us, B=8 4335 -> 4210 (a per-unit signal with a fence inside the stream was +66 at B=8). Expect HTTP B=1 ~1960, B=8 ~4030.
- evaluator: valid: 1959 us/step at B=1 and 4017 at B=8 over HTTP (online (submit/poll)); in process 1927 us/step at B=1 (generate, MBU 0.325)

### #13 s0013: Two kernel instances (B<=2 / B>=3); wide instance norms one warp per column

- status: **valid**, http_step_us_b1 = 1951 (B=8: 3953), at 3.23 agent-hours, parent s0012, category structure
- hypothesis: track A (bits identical to s0012). The B=1 and B=8 paths shared one register allocation, so a change for one moved the other by 30-60 us (allocation draw). Now a template instance per path. In the B>=3 instance each norm is done by one warp per column from registers (one trip to L2 instead of two). Same-card in process 64+512: B=1 1932 -> 1923, B=8 4210 -> 4143. Expect HTTP B=1 ~1950, B=8 ~3950.
- evaluator: valid: 1951 us/step at B=1 and 3953 at B=8 over HTTP (online (submit/poll)); in process 1919 us/step at B=1 (generate, MBU 0.326)

### #14 s0014: Drop the per-lane B-column selects (absent columns only feed discarded outputs); batched o_proj input copy

- status: **valid**, http_step_us_b1 = 1927 (B=8: 3824), at 3.85 agent-hours, parent s0013, category gemm
- hypothesis: track A (bits unchanged: an output column depends only on its own B column, and absent columns are never stored). Every mma operand load was followed by a select zeroing columns of absent sequences (8 per fp4 chunk, 2 per bf16 tile); the fp4 consume loop is instruction-bound, so these cost ~15% of its issue slots. Also the o_proj input copy uses 16-byte loads 8 in flight (B=8 -36 us). Same-card A/B: B=1 1926 -> 1899, B=8 4111 -> 4010. Expect HTTP B=1 ~1925, B=8 ~3810.
- evaluator: valid: 1927 us/step at B=1 and 3824 at B=8 over HTTP (online (submit/poll)); in process 1897 us/step at B=1 (step, MBU 0.330)

### #15 s0015: Two mma accumulators per ring stage (even/odd items), fp4 and bf16

- status: **valid**, http_step_us_b1 = 1906 (B=8: 3756), at 4.03 agent-hours, parent s0014, category gemm
- hypothesis: track A (formats unchanged; the k-sum of a stage is now (even items) + (odd items): a new rounding draw, same in both kernel instances so batch bits stay equal). Each warp's mma chain was serial over a 9-item stage (36 dependent mmas for fp4); two chains halve the dependent latency. Same-card A/B: fp4 two accumulators B=1 1899 -> 1870, B=8 4010 -> 3850; bf16 too: B=1 +8, B=8 -71 (three accumulators were worse). Expect HTTP B=1 ~1905, B=8 ~3600. Risk: top-1 margin is 0.3 points since s0011; this redraws it.
- evaluator: valid: 1906 us/step at B=1 and 3756 at B=8 over HTTP (online (submit/poll)); in process 1873 us/step at B=1 (generate, MBU 0.335)

### #16 s0016: K/V cache in mma fragment order (16 B loads); attention partials handed over with one acq_rel atomic

- status: **valid**, http_step_us_b1 = 1840 (B=8: 3708), at 4.78 agent-hours, parent s0015, category attention
- hypothesis: track A (bits unchanged except none: same arithmetic, K/V stored as bf16 in a different order). The attention item loaded K and V with 4-byte loads (64 + 64 per lane) and handed partials over with __threadfence + atomicAdd. Now K and V are stored per 8/16-key group in the lane order of the mma operands (2 + 4 16-byte loads per lane and group), and the hand-over is __syncwarp + one atom.add.acq_rel. Same-card A/B on 64+1024: fence -> acq_rel B=1 1857 -> 1844, B=8 4207 -> 4187; fragment layout B=1 1866 -> 1834, B=8 4224 -> 4214. Expect HTTP B=1 ~1860 (-45), B=8 ~3720.
- evaluator: valid: 1840 us/step at B=1 and 3708 at B=8 over HTTP (online (submit/poll)); in process 1816 us/step at B=1 (generate, MBU 0.345)

### #17 s0017: Expert bias rows prefetched when a unit starts (B>=3 epilogues no longer stall the stream)

- status: **valid**, http_step_us_b1 = 1858 (B=8: 3545), at 5.04 agent-hours, parent s0016, category memory
- hypothesis: track A (bits unchanged). In the B>=3 expert phases each warp finishes a (expert, rowtile) unit with an epilogue that loaded its bias rows from L2/HBM, so the warp stopped consuming its ring for ~1 us per unit (7 units per warp per phase). The 4 bias values a lane needs are now loaded when the unit's first stage is consumed. Same-card A/B (64+1024): B=8 4198 -> 3978 us, B=1 unchanged. Expect HTTP B=8 ~3500 (-200), B=1 ~1840.
- evaluator: valid: 1858 us/step at B=1 and 3545 at B=8 over HTTP (online (submit/poll)); in process 1834 us/step at B=1 (step, MBU 0.342)

### #18 s0018: Track B: certified int8 head sketch + exact rescoring of candidate rows (B<=2 greedy)

- status: **valid**, http_step_us_b1 = 1799 (B=8: 3500), at 6.69 agent-hours, parent s0017, category gemm
- hypothesis: track B; formats: checkpoint's (MXFP4 experts, BF16 rest) plus a derived int8 copy of lm_head with per-row error-block norms, used only to bound logits; every emitted token is the argmax of the exact BF16 head (candidate rowtiles recomputed with the exact head's own tasks, same bits; certified bound verified offline with 9x margin). Since s0017 also: down-phase activation loads batched (track-A change). Greedy head at B=1 175 -> ~119 us: HTTP B=1 1858 -> ~1795 (-60), B=8 3545 -> ~3530 (sketch off in the B>=3 instance).
- evaluator: valid: 1799 us/step at B=1 and 3500 at B=8 over HTTP (online (submit/poll)); in process 1775 us/step at B=1 (generate, MBU 0.341)

### #19 s0019: Track B: head sketch in the B>=3 instance too (inlined); same tokens

- status: **valid**, http_step_us_b1 = 1803 (B=8: 3459), at 7.41 agent-hours, parent s0018, category gemm
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; every token is the exact BF16 head's argmax, candidates recomputed with the exact head's own tasks). Adds the sketch to the B>=3 instance (inlined: as a call it raised that instance's spills): B=8 greedy head 180 -> ~137 us but layers +~25; expect B=8 HTTP 3500 -> ~3470 (and less noise: card 1 kept idle this time), B=1 unchanged ~1799.
- evaluator: valid: 1803 us/step at B=1 and 3459 at B=8 over HTTP (online (submit/poll)); in process 1775 us/step at B=1 (generate, MBU 0.341)

### #20 s0020: Track B: B>=3 staging by bulk copies (attention output, act rows in fragment order)

- status: **valid**, http_step_us_b1 = 1798 (B=8: 3413), at 7.94 agent-hours, parent s0019, category memory
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Changes (track-A kind, B>=3 instance only): attention output and gate_up activations written in fragment order so phases C and F stage them with one bulk copy per row; wide-instance sketch off again. In process B=8 serve 3907 -> 3861: expect HTTP B=8 3459 -> ~3415 (below SGLang 3424), B=1 unchanged ~1800.
- evaluator: valid: 1798 us/step at B=1 and 3413 at B=8 over HTTP (online (submit/poll)); in process 1776 us/step at B=1 (generate, MBU 0.341)

### #21 s0021: Track B: B<=2 instance stages attention output and act rows by bulk copies too

- status: **valid**, http_step_us_b1 = 1774 (B=8: 3403), at 8.06 agent-hours, parent s0020, category memory
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change (track-A kind): the B<=2 instance also writes the attention output and the gate_up activations in fragment order and stages them with one bulk copy per row. In process serve B=1 1780 -> 1758: expect HTTP B=1 1798 -> ~1776, B=8 unchanged ~3413.
- evaluator: valid: 1774 us/step at B=1 and 3403 at B=8 over HTTP (online (submit/poll)); in process 1755 us/step at B=1 (generate, MBU 0.345)

### #22 s0022: Track A: current engine without the head sketch (bulk-copy staging, batched act loads)

- status: **valid**, http_step_us_b1 = 1831 (B=8: 3391), at 8.34 agent-hours, parent s0017, category memory
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies. Same engine as s0021 with engine/TRACK=A (exact BF16 head on every path). Since s0017 (best track A B=8): batched act-row loads, attention output / act rows in fragment order staged by bulk copies in both instances. Expect HTTP B=1 ~1830 (s0016 1840), B=8 ~3403 (= s0021's B=8, same code), i.e. parity on track A as well.
- evaluator: valid: 1831 us/step at B=1 and 3391 at B=8 over HTTP (online (submit/poll)); in process 1807 us/step at B=1 (generate, MBU 0.346)

### #23 s0023: Track B: o_proj K split 16 -> 4 tasks per rowtile (fewer, longer ring tasks)

- status: **valid**, http_step_us_b1 = 1739 (B=8: 3323), at 8.71 agent-hours, parent s0021, category scheduling
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change (track-A kind): o_proj rowtiles split in 4 K segments instead of 16 (one 64-kstep task per warp for half the warps: fewer task boundaries and drains in the runner; summation order of the 4 partials fixed, same in both instances). In process serve B=1 1757 -> 1727, B=8 3851 -> 3765: expect HTTP B=1 1774 -> ~1745, B=8 3403 -> ~3320.
- evaluator: valid: 1739 us/step at B=1 and 3323 at B=8 over HTTP (online (submit/poll)); in process 1727 us/step at B=1 (generate, MBU 0.351)

### #24 s0024: Track B: ring tasks on whole stages (o_proj segments 63/63/63/67, router 5x9 ksteps)

- status: **valid**, http_step_us_b1 = 1739 (B=8: 3219), at 8.90 agent-hours, parent s0023, category scheduling
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change (track-A kind): o_proj K segments aligned to whole 9-item ring stages (one 4-item partial stage per rowtile instead of four 1-item ones) and the router's K quarter in 5 tasks of exactly one stage (was 8 tasks of 5-6 items). In process serve B=1 1727 -> 1723, B=8 3762 -> 3718: expect HTTP B=1 1739 -> ~1734, B=8 3323 -> ~3280.
- evaluator: valid: 1739 us/step at B=1 and 3219 at B=8 over HTTP (online (submit/poll)); in process 1721 us/step at B=1 (generate, MBU 0.352)

### #25 s0025: Track B: B<=2 attention loads V by cp.async with K

- status: **valid**, http_step_us_b1 = 1737 (B=8: 3237), at 9.33 agent-hours, parent s0024, category attention
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change (track-A kind, B<=2 instance only): each attention item issues its V chunk as cp.async into shared memory together with the K loads instead of after the scores. In process serve B=1 1724 -> 1714: expect HTTP B=1 1739 -> ~1729, B=8 unchanged ~3219 (HTTP B=8 repeats spread ~3200-3460).
- evaluator: valid: 1737 us/step at B=1 and 3237 at B=8 over HTTP (online (submit/poll)); in process 1717 us/step at B=1 (generate, MBU 0.353)

### #26 s0026: Track A: current engine without the head sketch (whole-stage tasks, bulk-copy staging)

- status: **valid**, http_step_us_b1 = 1810 (B=8: 3228), at 9.42 agent-hours, parent s0022, category scheduling
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. Same engine as s0025 with TRACK=A. Since s0022: o_proj K split 16 -> 4 on whole stages, router 5 whole-stage tasks, V by cp.async (B<=2). In process serve B=1 1814 -> 1792: expect HTTP B=1 1831 -> ~1792, B=8 3391 -> ~3230 (= s0025's B=8).
- evaluator: valid: 1810 us/step at B=1 and 3228 at B=8 over HTTP (online (submit/poll)); in process 1786 us/step at B=1 (generate, MBU 0.351)

### #27 s0027: Track B: head candidates rescored with 20 stages in flight; B>=3 down tail units as segments

- status: **valid**, http_step_us_b1 = 1730 (B=8: 3200), at 9.76 agent-hours, parent s0025, category scheduling
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Changes: (1) the sketch's candidate rowtiles are recomputed by one warp with all 20 stages in flight in the idle ring memory (same per-stage arithmetic, same bits); (2) B>=3 down phase: the n mod 8 remainder units of a round as one-stage segments summed in shared memory in the same order (same bits). In process serve B=1 1714 -> 1710, B=8 3734 -> 3704: expect HTTP B=1 ~1733, B=8 ~3215 (B=8 repeats spread ~3200-3470).
- evaluator: valid: 1730 us/step at B=1 and 3200 at B=8 over HTTP (online (submit/poll)); in process 1706 us/step at B=1 (generate, MBU 0.354)

### #28 s0028: Track B: short attention units per block with smem merge; B>=3 gate_up tail units as segments

- status: **valid**, http_step_us_b1 = 1731 (B=8: 3174), at 10.17 agent-hours, parent s0027, category attention
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Changes (track-A kind, same bits): (1) attention units with at most 8 chunks (sliding layers; full layers below position 512) run one block per (sequence, kv head), one chunk per warp, partials merged in shared memory instead of global partials + acq_rel hand-off; (2) B>=3 gate_up remainder units as one-stage segments. In process serve (64+512) B=1 1710 -> 1706, B=8 3704 -> 3662: expect HTTP B=1 ~1727, B=8 3200 -> ~3165.
- evaluator: valid: 1731 us/step at B=1 and 3174 at B=8 over HTTP (online (submit/poll)); in process 1695 us/step at B=1 (generate, MBU 0.356)

### #29 s0029: Track A: current engine without the head sketch (attention short units per block, tail segments)

- status: **valid**, http_step_us_b1 = 1799 (B=8: 3174), at 10.36 agent-hours, parent s0026, category attention
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. Same engine as s0028 with TRACK=A. Since s0026: short attention units per block with a shared-memory merge, tail units as segments in the B>=3 expert phases, all bit-preserving. In process serve B=1 1792 -> 1787, B=8 3734 -> 3643: expect HTTP B=1 ~1805, B=8 ~3175 (= s0028's B=8 path).
- evaluator: valid: 1799 us/step at B=1 and 3174 at B=8 over HTTP (online (submit/poll)); in process 1772 us/step at B=1 (generate, MBU 0.353)

### #30 s0030: Track B: head candidates rescored in parallel (pooled ring slots), no L2 prefetch of local candidates

- status: **valid**, http_step_us_b1 = 1719 (B=8: 3177), at 10.63 agent-hours, parent s0028, category gemm
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change: the sketch's candidate rowtiles (usually 2 in one block at B=1) are recomputed in parallel on up to 3 warps sharing the 25 pooled stage slots (same per-stage arithmetic, same bits), and the L2 prefetch of local candidates is dropped. In process serve (64+1024) B=1 1724 -> 1715: expect HTTP B=1 1731 -> ~1722, B=8 unchanged ~3174.
- evaluator: valid: 1719 us/step at B=1 and 3177 at B=8 over HTTP (online (submit/poll)); in process 1683 us/step at B=1 (generate, MBU 0.358)

### #31 s0031: Track B: head sketch in the B>=3 instance via an out-of-line call; warp-parallel token pick

- status: **valid**, http_step_us_b1 = 1715 (B=8: 3144), at 10.89 agent-hours, parent s0030, category gemm
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Changes: (1) the B>=3 instance's greedy head uses the int8 sketch too, called out of line (inlined it cost the layers more than it saved): head ~175 -> ~135 us at B=8 minus ~25 us of call/allocation cost; (2) the per-step token pick as a warp-parallel argmax. In process serve (64+1024) B=8 3681 -> 3644, B=1 1715 -> 1691 (bench noise ~5): expect HTTP B=8 3177 -> ~3150, B=1 ~1715.
- evaluator: valid: 1715 us/step at B=1 and 3144 at B=8 over HTTP (online (submit/poll)); in process 1680 us/step at B=1 (generate, MBU 0.358)

### #32 s0032: Track B: sketch's global bound read in parallel per column

- status: **valid**, http_step_us_b1 = 1720 (B=8: 3139), at 11.22 agent-hours, parent s0031, category gemm
- hypothesis: track B; formats as s0018 (checkpoint's + derived int8 lm_head copy used only to bound logits; tokens = exact BF16 head argmax). Change: after the sketch barrier, the global best lower bound of each column is read by its own warp instead of one warp looping over the columns (an L2 round trip per column on every block's critical path at B=8). In process serve (64+1024): B=8 3665 -> 3662, B=1 1715 -> 1713: expect HTTP B=8 ~3141, B=1 ~1713 (both within the run-to-run spread).
- evaluator: valid: 1720 us/step at B=1 and 3139 at B=8 over HTTP (online (submit/poll)); in process 1686 us/step at B=1 (generate, MBU 0.356)

### #33 s0033: Track A: current engine without the head sketch (since s0029: warp-parallel token pick)

- status: **valid**, http_step_us_b1 = 1796 (B=8: 3164), at 11.44 agent-hours, parent s0029, category other
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. s0032's engine with TRACK=A (this also confirms the track-A build after a fix of a build break introduced since s0029; s0029 itself was built correctly). Since s0029: warp-parallel token pick per step. In process serve (64+1024) B=1 ~1774, B=8 ~3658: expect HTTP B=1 ~1797, B=8 ~3168.
- evaluator: valid: 1796 us/step at B=1 and 3164 at B=8 over HTTP (online (submit/poll)); in process 1773 us/step at B=1 (generate, MBU 0.352)

### #34 s0034: Track B: wide head rescoring copies the prep's normed rows instead of re-norming

- status: **valid**, http_step_us_b1 = 1722 (B=8: 3125), at 12.38 agent-hours, parent s0032, category other
- hypothesis: track B; formats: as s0032 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). Blocks 0..7 publish the sketch prep's normed head input (norm_warp's bits) in fragment order before the sketch barrier; candidate blocks copy it into shared memory instead of running norm_warp from a cold instruction cache (max 8.4 -> 1.2 us). In process serve B=8 -15.5 us, B=1 unchanged: expect HTTP B=8 ~3124 (s0032 3139), B=1 ~1717.
- evaluator: valid: 1722 us/step at B=1 and 3125 at B=8 over HTTP (online (submit/poll)); in process 1679 us/step at B=1 (generate, MBU 0.357)

### #35 s0035: Track B: down-phase tail units back to whole units (FTAIL 0)

- status: **valid**, http_step_us_b1 = 1722 (B=8: 3115), at 13.55 agent-hours, parent s0034, category other
- hypothesis: track B; formats: as s0034 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The B>=3 down phase's remainder units run as whole units again instead of one-stage segments (same bits: bitwise equal to s0034's build to position 1100); on a bench with the HTTP client's kind of prompts (word salad + 'continue', in process 3144 vs HTTP 3125) it is -10 us at B=8 (3 rounds), B=1 unaffected (wide instance only). Expect HTTP B=8 ~3115, B=1 ~1720.
- evaluator: valid: 1722 us/step at B=1 and 3115 at B=8 over HTTP (online (submit/poll)); in process 1682 us/step at B=1 (generate, MBU 0.357)

### #36 s0036: Track B: B>=3 attention units up to 16 chunks on one block (2 chunks per warp)

- status: **valid**, http_step_us_b1 = 1722 (B=8: 3105), at 13.63 agent-hours, parent s0035, category other
- hypothesis: track B; formats: as s0035 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). In the B>=3 instance, attention units of 9-16 chunks (full layers at positions 512-1023) also run on one block (2 chunks per warp, partials in shared memory, the same merge: bitwise equal) instead of the global hand-off. HTTP-like bench B=8 -10.4..-11.3 us (4 rounds), B=1 unaffected. Expect HTTP B=8 ~3104 (s0035 3115), B=1 ~1720.
- evaluator: valid: 1722 us/step at B=1 and 3105 at B=8 over HTTP (online (submit/poll)); in process 1682 us/step at B=1 (generate, MBU 0.357)

### #37 s0037: Track A: s0036's engine without the head sketch

- status: **valid**, http_step_us_b1 = 1799 (B=8: 3135), at 13.77 agent-hours, parent s0033, category other
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. s0036's engine with TRACK=A: since s0033 the B>=3 down tail units are whole again (FTAIL 0) and B>=3 attention units of up to 16 chunks run on one block (ATT_BLKW 2), both bitwise neutral. HTTP-like bench (track A build) B=8 3185 -> 3152 (-33), B=1 unchanged: expect HTTP B=8 ~3131 (s0033 3164), B=1 ~1797.
- evaluator: valid: 1799 us/step at B=1 and 3135 at B=8 over HTTP (online (submit/poll)); in process 1764 us/step at B=1 (generate, MBU 0.353)

### #38 s0038: Track B: wide norm loads each weight word one iteration ahead (NORM_PIPE1)

- status: **valid**, http_step_us_b1 = 1722 (B=8: 3083), at 14.07 agent-hours, parent s0036, category other
- hypothesis: track B; formats: as s0036 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The B>=3 norm (norm_warp, used by the norm1/norm2 of every layer) loads each norm weight word one iteration before its use, so the load is not ordered behind the previous iteration's shared-memory stores; same arithmetic, bitwise equal to position 1100. HTTP-like bench B=8 -23.5 us (3 rounds), B=1 unchanged. Expect HTTP B=8 ~3082 (s0036 3105), B=1 ~1720.
- evaluator: valid: 1722 us/step at B=1 and 3083 at B=8 over HTTP (online (submit/poll)); in process 1682 us/step at B=1 (generate, MBU 0.357)

### #39 s0039: Track A: B>=3 instance prefetches 3 ring stages across barriers (was 2)

- status: **valid**, http_step_us_b1 = 1798 (B=8: 3117), at 14.69 agent-hours, parent s0037, category other
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. In the track A build the B>=3 instance now issues 3 instead of 2 ring stages of the next phase before each grid barrier (the prefetch depth became a per-instance template parameter; the B<=2 instance keeps 2). Arithmetic unchanged. HTTP-like bench (track A build) B=8 -19 us (2 rounds), B=1 unchanged. Expect HTTP B=8 ~3116 (s0037 3135), B=1 ~1799.
- evaluator: valid: 1798 us/step at B=1 and 3117 at B=8 over HTTP (online (submit/poll)); in process 1763 us/step at B=1 (generate, MBU 0.353)

### #40 s0040: Track B: B<=2 attention loads V after the scores again (V_ASYNC_FLAT 0)

- status: **valid**, http_step_us_b1 = 1719 (B=8: 3083), at 15.16 agent-hours, parent s0038, category other
- hypothesis: track B; formats: as s0038 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The B<=2 instance loads an attention chunk's V with plain loads after the scores instead of by cp.async with K (bitwise equal); on the HTTP-like bench (64 + 1024) B=1 -6.2 us (3 rounds), B=8 +0..4 (the B>=3 code is unchanged). Expect HTTP B=1 ~1716 (s0038 1722), B=8 ~3085.
- evaluator: valid: 1719 us/step at B=1 and 3083 at B=8 over HTTP (online (submit/poll)); in process 1681 us/step at B=1 (generate, MBU 0.357)

### #41 s0041: Track B: phase A pulls each block's attention K/V chunks into L2

- status: **valid**, http_step_us_b1 = 1703 (B=8: 3066), at 15.56 agent-hours, parent s0040, category other
- hypothesis: track B; formats: as s0040 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). At the start of phase A every block issues bulk L2 prefetches of the K/V chunks its attention items will read in phase B (the cache does not stay in L2 between steps), so phase B's first loads hit L2; no arithmetic change (bitwise equal). HTTP-like bench: B=1 -14 us, B=8 -21 us (3 rounds; fixed-content step() -22 / -23). Expect HTTP B=1 ~1705 (s0040 1719), B=8 ~3062 (3083).
- evaluator: valid: 1703 us/step at B=1 and 3066 at B=8 over HTTP (online (submit/poll)); in process 1660 us/step at B=1 (generate, MBU 0.362)

### #42 s0042: Track B: phase A also pulls the layer's biases and sinks into L2

- status: **valid**, http_step_us_b1 = 1700 (B=8: 3064), at 15.83 agent-hours, parent s0041, category other
- hypothesis: track B; formats: as s0041 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). Besides the K/V chunks, phase A now prefetches into L2 this block's qkv and o_proj bias slices, the router bias and the attention sinks, which later phases read right after a grid barrier; no arithmetic change (bitwise equal). HTTP-like bench B=1 -5.7 us (4 rounds), B=8 0. Expect HTTP B=1 ~1697 (s0041 1703), B=8 ~3066.
- evaluator: valid: 1700 us/step at B=1 and 3064 at B=8 over HTTP (online (submit/poll)); in process 1653 us/step at B=1 (generate, MBU 0.364)

### #43 s0043: Track A: B<=2 attention loads V after the scores (as track B since s0040)

- status: **valid**, http_step_us_b1 = 1787 (B=8: 3114), at 15.94 agent-hours, parent s0039, category other
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. s0039's track A engine plus V_ASYNC_FLAT 0 (the B<=2 instance loads V with plain loads after the scores; bitwise equal) and code moves that keep the bits. HTTP-like bench (track A build) B=1 -10 us, B=8 -3.4 us (2 rounds). Expect HTTP B=1 ~1788 (s0039 1798), B=8 ~3114 (3117).
- evaluator: valid: 1787 us/step at B=1 and 3114 at B=8 over HTTP (online (submit/poll)); in process 1751 us/step at B=1 (generate, MBU 0.356)

### #44 s0044: Track B: gate_up tail units back to whole units (ETAIL 0)

- status: **valid**, http_step_us_b1 = 1706 (B=8: 3054), at 16.95 agent-hours, parent s0042, category other
- hypothesis: track B; formats: as s0042 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The B>=3 gate_up phase's remainder units run as whole units again instead of one-stage segments (bitwise equal to position 1100); after the KVPF/SPF changes this re-measures -5.8 us at B=8 on the HTTP-like bench (3 rounds), B=1 unchanged (B>=3 code only). Expect HTTP B=8 ~3058 (s0042 3064), B=1 ~1700.
- evaluator: valid: 1706 us/step at B=1 and 3054 at B=8 over HTTP (online (submit/poll)); in process 1657 us/step at B=1 (generate, MBU 0.362)

### #45 s0045: Track B: wide head re-norms its rows for the rescoring again (HSK_XG 0)

- status: **valid**, http_step_us_b1 = 1693 (B=8: 3026), at 17.89 agent-hours, parent s0044, category other
- hypothesis: track B; formats: as s0044 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). Undoes s0034's HSK_XG: candidate blocks re-norm the head input (norm_warp) instead of copying rows published by blocks 0..7; same values, tokens identical. The out-of-line wide head's register allocation has moved since s0034: on the HTTP-like bench this is now -24 us at B=8 (3 rounds), B=1 unchanged. Expect HTTP B=8 ~3030 (s0044 3054), B=1 ~1703.
- evaluator: valid: 1693 us/step at B=1 and 3026 at B=8 over HTTP (online (submit/poll)); in process 1646 us/step at B=1 (generate, MBU 0.365)

### #46 s0046: Track B: wide head rescores candidate tiles with pooled ring slots (FASTRS_W 1)

- status: **valid**, http_step_us_b1 = 1693 (B=8: 3017), at 18.23 agent-hours, parent s0045, category other
- hypothesis: track B; formats: as s0045 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The B>=3 head rescores its candidate tiles with the pooled ring stage slots (up to 8 tiles at once, each with several stages in flight) instead of each warp's 3-stage ring; the same per-stage arithmetic, tokens identical. HTTP-like bench B=8 -10 us (3 rounds), B=1 unchanged. Expect HTTP B=8 ~3016 (s0045 3026), B=1 ~1695.
- evaluator: valid: 1693 us/step at B=1 and 3017 at B=8 over HTTP (online (submit/poll)); in process 1647 us/step at B=1 (generate, MBU 0.365)

### #47 s0047: Track B: B<=2 qkv on 2 rowtiles per block; extra V rowtiles as spread segments

- status: **valid**, http_step_us_b1 = 1691 (B=8: 3021), at 20.08 agent-hours, parent s0046, category scheduling
- hypothesis: track B; formats: as s0046 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). In the B<=2 instance every block owns 2 qkv rowtiles (16 blocks had 3: 8 ring stages per warp instead of 6); the 16 extra V rowtiles (kv heads 4..7) run as 160 segment tasks on blocks 0..39 with partials in global, and the attention warp that reads the current position sums V(pos) out of line with the qkv epilogue's arithmetic (bitwise equal to position 1100, tokens identical). HTTP-like bench B=1 -4.5 us (5 rounds), B=8 +3. Expect HTTP B=1 ~1688 (s0046 1692.5), B=8 ~3020.
- evaluator: valid: 1691 us/step at B=1 and 3021 at B=8 over HTTP (online (submit/poll)); in process 1643 us/step at B=1 (generate, MBU 0.366)

### #48 s0048: Track A: exact head's last rowtiles split over warps (HTAIL) + wide norm pipelining

- status: **valid**, http_step_us_b1 = 1775 (B=8: 3101), at 21.58 agent-hours, parent s0043, category scheduling
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. Each block's n mod 8 last head rowtiles run as 4 whole-stage K segments spread over the warps (summed in a fixed order, identically in both instances: a different but deterministic BF16-input fp32 summation of those rows), and the B>=3 norm loads its weight words one iteration ahead (NORM_PIPE1; alone it lost in this build, with HTAIL it wins). HTTP-like bench (track A build, 3 rounds): B=8 -2.7 us, B=1 -6.5 us. Expect HTTP B=8 ~3111 (s0043 3114; the 1.10x line is 3112.7), B=1 ~1781.
- evaluator: valid: 1775 us/step at B=1 and 3101 at B=8 over HTTP (online (submit/poll)); in process 1740 us/step at B=1 (generate, MBU 0.358)

### #49 s0049: Track A: K/V chunks prefetched into L2 in the B>=3 instance (KVPF)

- status: **valid**, http_step_us_b1 = 1774 (B=8: 3074), at 22.00 agent-hours, parent s0048, category memory
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. s0048's track A engine plus KVPF in the B>=3 instance only: at the start of phase A each block issues L2 prefetches of the K/V chunks its attention items read in phase B (no arithmetic change, bitwise equal). In s0037's track A allocation this was +17 us; after HTAIL it is -33 us at B=8 on the HTTP-like bench (2 rounds), B=1 unchanged. Expect HTTP B=8 ~3068 (s0048 3101), B=1 ~1775.
- evaluator: valid: 1774 us/step at B=1 and 3074 at B=8 over HTTP (online (submit/poll)); in process 1740 us/step at B=1 (generate, MBU 0.358)

### #50 s0050: Track A: head tail rowtiles in 2 K halves instead of 4 quarters (HTS 2)

- status: **valid**, http_step_us_b1 = 1776 (B=8: 3061), at 22.72 agent-hours, parent s0049, category scheduling
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. HTAIL's tail rowtiles (each block's n mod 8 last head rowtiles) now run as 2 whole-stage K halves instead of 4 quarters (summed in order, identically in both instances; generate == stepwise). HTTP-like bench (track A build, 3 rounds): B=8 -7.3 us, B=1 0. Expect HTTP B=8 ~3067 (s0049 3074), B=1 ~1774.
- evaluator: valid: 1776 us/step at B=1 and 3061 at B=8 over HTTP (online (submit/poll)); in process 1738 us/step at B=1 (generate, MBU 0.358)

### #51 s0051: Track B: final engine as committed (s0047's kernel; track A code compiled out)

- status: **valid**, http_step_us_b1 = 1687 (B=8: 3029), at 23.64 agent-hours, parent s0047, category other
- hypothesis: track B; formats: as s0047 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The final committed engine/: identical in effect to s0047 (logits bitwise equal to position 1100 at B=1 and B=8, same in-process speed); the source differs only in track-A-only code (HTAIL/HTS, per-instance KVPF) that is compiled out in this track. Expect HTTP B=1 ~1691, B=8 ~3020.
- evaluator: valid: 1687 us/step at B=1 and 3029 at B=8 over HTTP (online (submit/poll)); in process 1639 us/step at B=1 (generate, MBU 0.366)

### #52 s0052: Track B: grid barrier arrives by red.release and polls with a 40 ns back-off

- status: **valid**, http_step_us_b1 = 1690 (B=8: 3019), at 24.98 agent-hours, parent s0051, category sync
- hypothesis: track B; formats: as s0051 (checkpoint formats + the int8 head sketch with error-block norms, exact BF16 head on candidates). The grid barrier's thread 0 arrives with one red.release.gpu add (instead of a full fence + atomicAdd) and polls the counter with __nanosleep(40) between acquire loads, so 152 pollers do not hammer one L2 line (microbenchmark 1.345 -> 1.155 us per barrier); no arithmetic change, bitwise equal. HTTP-like bench: B=1 -7.4 us (3 rounds), B=8 ~-1. Expect HTTP B=1 ~1680 (s0051 1686.6), B=8 ~3025.
- evaluator: valid: 1690 us/step at B=1 and 3019 at B=8 over HTTP (online (submit/poll)); in process 1647 us/step at B=1 (generate, MBU 0.365)

### #53 s0053: Track A: grid barrier arrives by red.release and polls with a 40 ns back-off

- status: **valid**, http_step_us_b1 = 1771 (B=8: 3075), at 25.15 agent-hours, parent s0050, category sync
- hypothesis: track A; formats: the checkpoint's (MXFP4 experts with E8M0 scales, BF16 elsewhere), no derived copies; exact BF16 head on every path. s0050's track A engine plus the grid barrier of s0052 (one red.release add, acquire poll with __nanosleep(40)); bitwise equal. Same-card bench (track A build, card 2): B=1 -11..-14 us, B=8 +4; the effect varies by card. Expect HTTP B=1 ~1765 (s0050 1775.8), B=8 ~3065.
- evaluator: valid: 1771 us/step at B=1 and 3075 at B=8 over HTTP (online (submit/poll)); in process 1730 us/step at B=1 (generate, MBU 0.360)

### #54 s0054: Track B: B>=3 head rescoring splits candidate tiles into 4 K segments over warps

- status: **valid**, http_step_us_b1 = 1690 (B=8: 3007), at 26.22 agent-hours, parent s0052, category scheduling
- hypothesis: track B (formats: int8 head sketch + exact rescoring, as s0052). The wide instance's exact head now sums each rowtile as 4 K segments (in order), so candidate tiles are rescored by 8 warps at once instead of one warp per tile; B=8 HTTP -12 us (3019 -> ~3007), B=1 unchanged (flat code untouched, ~1688 +-4).
- evaluator: valid: 1690 us/step at B=1 and 3007 at B=8 over HTTP (online (submit/poll)); in process 1647 us/step at B=1 (generate, MBU 0.365)

### #55 s0055: Track B: wide head copies the published normed rows by bulk copy (HSK_XG 2)

- status: **valid**, http_step_us_b1 = 1690 (B=8: 3000), at 26.50 agent-hours, parent s0054, category other
- hypothesis: track B (formats: int8 head sketch + exact rescoring, as s0054). After s0054 moved the wide instance's allocation, the candidate blocks' head input as one bulk copy per row (HSK_XG 2, same bits) measured -8 us at B=8 in 3 same-card rounds; expect HTTP B=8 3007 -> ~2999, B=1 unchanged (~1689 +-4).
- evaluator: valid: 1690 us/step at B=1 and 3000 at B=8 over HTTP (online (submit/poll)); in process 1650 us/step at B=1 (generate, MBU 0.365)

### #56 s0056: Track B: wide head sketch prep fuses the norm and digit passes (HSK_PREP2)

- status: **valid**, http_step_us_b1 = 1690 (B=8: 2991), at 26.88 agent-hours, parent s0055, category other
- hypothesis: track B (formats: int8 head sketch + exact rescoring, as s0055). The B>=3 sketch prep takes its digit scale from an upper bound of the normed maximum, so norm and digits are one pass (normed bits unchanged, sketch still certified); same-card -6 us at B=8 (more than the prep's own -0.3 us, so partly layout); expect HTTP B=8 3000 -> ~2994, B=1 unchanged (~1690 +-4).
- evaluator: valid: 1690 us/step at B=1 and 2991 at B=8 over HTTP (online (submit/poll)); in process 1647 us/step at B=1 (generate, MBU 0.365)

### #57 s0057: Track A: exact head tail claimed dynamically (HDYN) + re-swept KVPF_F/QXO

- status: **valid**, http_step_us_b1 = 1760 (B=8: 3067), at 27.42 agent-hours, parent s0053, category scheduling
- hypothesis: track A (formats: checkpoint's, unchanged). The exact head's last ~8% of rowtiles go to whichever warp is free (global counter) instead of fixed K-split tails; with it the B<=2 K/V prefetch and qkv split win again (same bits). Same-card: B=1 -16.6, B=8 -22 vs s0053's engine; expect HTTP B=1 1771 -> ~1755, B=8 3075 -> ~3053.
- evaluator: valid: 1760 us/step at B=1 and 3067 at B=8 over HTTP (online (submit/poll)); in process 1719 us/step at B=1 (generate, MBU 0.363)

### #58 s0058: Track B: head sketch tail tiles claimed dynamically by free warps (SDYN)

- status: **valid**, http_step_us_b1 = 1692 (B=8: 3010), at 27.76 agent-hours, parent s0056, category scheduling
- hypothesis: track B (formats: int8 head sketch + exact rescoring, as s0056). Each block sketches 72 rowtiles of its own and the last 1624 go to whichever warp is free (global counter), so the post-sketch barrier waits less for slow SMs; same-card B=1 -6, B=8 -7; expect HTTP B=1 1690 -> ~1684, B=8 2991 -> ~2984.
- evaluator: valid: 1692 us/step at B=1 and 3010 at B=8 over HTTP (online (submit/poll)); in process 1648 us/step at B=1 (generate, MBU 0.364)

### #59 s0059: Track A: B<=2 instance prefetches the layer's small parameters into L2 (SPF_F)

- status: **valid**, http_step_us_b1 = 1738 (B=8: 3076), at 29.94 agent-hours, parent s0057, category memory
- hypothesis: track A (formats: checkpoint's, unchanged). The B<=2 instance pulls each layer's bias slices, router bias and sinks into L2 at phase A (as track B does since s0042; prefetch only, same bits), now split from the B>=3 instance where it costs; same-card B=1 -18, B=8 +4.7; expect HTTP B=1 1760 -> ~1742, B=8 3067 -> ~3071.
- evaluator: valid: 1738 us/step at B=1 and 3076 at B=8 over HTTP (online (submit/poll)); in process 1695 us/step at B=1 (generate, MBU 0.367)

