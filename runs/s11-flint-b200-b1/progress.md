# s11-flint-b200-b1: flint on GB200

Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 27 submissions so far, 27 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 629
- 1.10x the fastest general engine: 1792
- vLLM 0.33.0 (general engine): 1971
- inherited engine, track A (s0077): 1480

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track A re-measure: inherited s0077 kernel, track-B code compiled out

- status: **valid**, http_step_us_b1 = 1488 (B=8: 3426), at 0.90 agent-hours, category other
- hypothesis: track A: checkpoint formats (no format differs). Same arithmetic and machine code paths as the inherited s0077 (new Params fields and the track-B head sketch are compiled out; spill lines 144/44/44 and 176/20/16 unchanged). Expect the inherited figures: http_step_us_b1 1480 +- 10, batch 8 3400 +- 30; gives the current track-A reference on this pod for the track-B submissions.
- evaluator: valid: 1488 us/step at B=1 and 3426 at B=8 over HTTP (online (submit/poll)); in process 1375 us/step at B=1 (generate, MBU 0.459)

### #2 s0002: Track B: certified int8 sketch of lm_head + exact rescoring on the greedy paths

- status: **valid**, http_step_us_b1 = 1473 (B=8: 3438), at 0.94 agent-hours, parent s0001, category memory
- hypothesis: track B: formats differ from the checkpoint only in an extra int8 copy of lm_head (per-row scale, 307 MB) plus per-row 32-block error norms, used only by generate/serve; step() keeps the exact BF16 head (bitcheck: logits bit-identical to track A at B=1 and B=8, 1152 positions) and the greedy token is certified equal (rows whose upper bound reaches the best lower bound are recomputed with the exact head's own mma chains; bound >= 7.8x the largest error seen on 1600 real inputs; greedy and served tokens identical to track A on 38 sequences x 1024). Head phase 79 -> 63 us in process; expect http_step_us_b1 1488 -> ~1475 (-10 to -15 us), batch 8 unchanged within noise (pass 2 has ~70 candidate groups there).
- evaluator: valid: 1473 us/step at B=1 and 3438 at B=8 over HTTP (online (submit/poll)); in process 1361 us/step at B=1 (generate, MBU 0.460)

### #3 s0003: Track A: attention scores as K x q^T, 8 mma per key group instead of 16

- status: **valid**, http_step_us_b1 = 1477 (B=8: 3417), at 1.87 agent-hours, parent s0001, category attention
- hypothesis: track A: no format differs. The scores of a key group used q as the mma A operand with 8 of its 16 rows zero (16 mma per group); now the K tile is the A operand (its registers are an A fragment up to their order) and q the B operand: 8 mma per group, then 8 shuffles back to the old layout. The tensor core gives identical results for the transposed product (bench_mmaT: 0 of 1.25e9 scores differ); logits bit-identical to the previous build at B=1 and B=8 over 1100 positions, all batch sizes 2-8 bitwise equal to batch 1. Expect http_step_us_b1 1488 -> ~1483 (-5), batch 8 -9 (in process -5 / -9).
- evaluator: valid: 1477 us/step at B=1 and 3417 at B=8 over HTTP (online (submit/poll)); in process 1374 us/step at B=1 (generate, MBU 0.459)

### #4 s0004: Track B on s0003: int8 head sketch on the greedy paths (scores code per track)

- status: **valid**, http_step_us_b1 = 1471 (B=8: 3437), at 2.25 agent-hours, parent s0003, category memory
- hypothesis: track B: formats differ from the checkpoint only in an extra int8 copy of lm_head with per-row 32-block error norms, used by generate/serve to certify the greedy token (exact rescoring); step() keeps the exact head. The B build keeps the 16-mma scores form (KQT off: the same bits, a better register allocation for this build: B 1466-1472 vs A 1476-1483 in process on the same cards). Bitcheck vs track A of this source: logits bit-identical at B=1 and B=8 (1152 positions); greedy and served tokens identical to track A (6x1024 at B=1, 24x1024 at B=8, served 8x1024). Expect http_step_us_b1 1476.7 (s0003) -> ~1467 (-8..-10); batch 8 3417 -> ~3420-3430 (no sketch gain there, KQT off costs ~9).
- evaluator: valid: 1471 us/step at B=1 and 3437 at B=8 over HTTP (online (submit/poll)); in process 1357 us/step at B=1 (generate, MBU 0.462)

### #5 s0005: Track B: head sketch in the batch 1-2 kernel only; batch>2 kernel = track A's

- status: **valid**, http_step_us_b1 = 1472 (B=8: 3428), at 2.75 agent-hours, parent s0004, category memory
- hypothesis: track B: formats as s0004 (an extra int8 copy of lm_head with per-row 32-block error norms; greedy paths only). Change: the sketch and the 16-mma scores form are compiled into the batch 1-2 kernel instance only; the batch > 2 instance of the track-B build is now the same code as track A's (same spill lines 144/44/44; at batch 8 the sketch gained nothing and the 16-mma form cost ~9 us). Bitcheck vs track A: logits bit-identical at B=1 and B=8; greedy and served tokens identical (B=1 6x1024, B=8 24x1024, served 8x1024, every batch size 1-8, context to 1900, vocabulary-tail inputs); B=2..8 bitwise equal to B=1. Expect batch 8 3437 -> ~3417 (-20, back to s0003's), batch 1 unchanged ~1471 (+-6).
- evaluator: valid: 1472 us/step at B=1 and 3428 at B=8 over HTTP (online (submit/poll)); in process 1362 us/step at B=1 (generate, MBU 0.460)

### #6 s0006: Track B final: s0005 kernel + submit() refuses out-of-range token ids

- status: **valid**, http_step_us_b1 = 1472 (B=8: 3456), at 3.17 agent-hours, parent s0005, category correctness
- hypothesis: track B: formats as s0005 (extra int8 copy of lm_head with per-row error norms, greedy paths, batch 1-2 kernel). Kernel unchanged from s0005; api.submit() now raises on token ids outside [0, 150144) (the kernel read them modulo the vocabulary). Robustness: flake 0/199 x 4 workloads, long contexts to 2040, api_stress 40/40. Expect the s0005 figures within noise: batch 1 ~1471 (+-6), batch 8 ~3428 (+-30).
- evaluator: valid: 1472 us/step at B=1 and 3456 at B=8 over HTTP (online (submit/poll)); in process 1362 us/step at B=1 (generate, MBU 0.460)

### #7 s0007: Track A: L2 prefetch of next q/k/v and shared expert from an idle warp

- status: **valid**, http_step_us_b1 = 1463 (B=8: 3422), at 4.57 agent-hours, parent s0006, category memory
- hypothesis: track A (checkpoint formats, no format differs). An idle warp (12) of each main block bulk-prefetches to L2 (a cache hint only) its slice of the next layer's q/k/v during the routing, and of the shared expert's gate/up/down while it waits for the attention: HBM is idle in both windows. Calibrated in-process A/B on two cards: B=1 1465 -> 1434 us; B=8 neutral. Bit-identical to s0006's track A (bitcheck, 1152 positions, B=1 and 8); flake 0. Expect ~1447 over HTTP (s0003 1476.7 - 30).
- evaluator: valid: 1463 us/step at B=1 and 3422 at B=8 over HTTP (online (submit/poll)); in process 1338 us/step at B=1 (generate, MBU 0.472)

### #8 s0008: Track A: q/k/v prefetch at the end of C, shared expert at the start of A

- status: **valid**, http_step_us_b1 = 1452 (B=8: 3426), at 5.87 agent-hours, parent s0007, category memory
- hypothesis: track A (checkpoint formats, no format differs). s0007's idle-warp L2 prefetches moved into windows where HBM is idle for every block: the next layer's q/k/v slice at the end of phase C (was: after the router's signal, where its traffic delayed blocks 112-127 reading the router logits, MK_CLK) and the shared expert's slices at the start of phase A (A reads L2 only; was: during the attention wait, next to the attention's own K/V loads). Batch 1-2 kernel only. Clean calibrated A/B: B=1 1433-1435 -> 1416-1417, B=8 unchanged. Logits bit-identical to s0007 (digests); flake 0/199. Expect ~1446 over HTTP.
- evaluator: valid: 1452 us/step at B=1 and 3426 at B=8 over HTTP (online (submit/poll)); in process 1319 us/step at B=1 (step, MBU 0.478)

### #9 s0009: Track A: 512-byte device allocation before the engine's own (buffer layout)

- status: **valid**, http_step_us_b1 = 1422 (B=8: 3427), at 7.72 agent-hours, parent s0008, category memory
- hypothesis: track A (checkpoint formats, no format differs). Engine.__init__ allocates 512 B on the device first. In a process that has allocated nothing before (the evaluation's HTTP engine process) the engine's small tensors otherwise land where the batch-1 step is ~30 us slower; that is the HTTP vs in-process gap of s0007/s0008 (25 us; 8 before). A two-process copy of the serving loop: 1449 -> 1422 us/token. Expect HTTP B=1 1452 -> ~1425, B=8 unchanged (served copy 3485 -> 3492, noise), in-process figures unchanged. Layout only: bit-identical to s0008 (digests, bcheck); flake 0/199.
- evaluator: valid: 1422 us/step at B=1 and 3427 at B=8 over HTTP (online (submit/poll)); in process 1321 us/step at B=1 (step, MBU 0.478)

### #10 s0010: Track A re-measure of s0009 (identical source): layout fix reproducibility

- status: **valid**, http_step_us_b1 = 1419 (B=8: 3427), at 8.52 agent-hours, parent s0009, category other
- hypothesis: track A (checkpoint formats, no format differs). The source of s0009, unchanged. Its gain (1452 -> 1422) rests on where a fresh process places the kernel library's small buffers (the 512-byte allocation first); this re-evaluation checks that the evaluation card's new engine process lands in the good layout again. Expect 1422 +- 10 at B=1 over HTTP, B=8 ~3427.
- evaluator: valid: 1419 us/step at B=1 and 3427 at B=8 over HTTP (online (submit/poll)); in process 1319 us/step at B=1 (step, MBU 0.479)

### #11 s0011: Track A trade-off: 4 accumulators in phase D (B=8 -57 us, B=1 +11 us)

- status: **valid**, http_step_us_b1 = 1430 (B=8: 3443), at 10.03 agent-hours, parent s0010, category other
- hypothesis: track A (checkpoint formats, no format differs). s0010's engine with the shared expert's gate/up (phase D) summed in 4 accumulator chains instead of 2 in both kernel instances (bitwise batch invariance kept; another float summation order, numerics at the HF BF16 level), compiled at ptxas register-usage level 8 so the batch-1 kernel keeps its 176-byte frame. Not a headline candidate: in the scored layout B=1 1418-1422 -> 1428-1441 (+11), B=8 3508-3515 -> 3451-3459 (-57). Expect HTTP B=1 ~1430, B=8 ~3370: the batch-8 margin under the 1.10x line (3451) from 0.7% to ~2.4%.
- evaluator: valid: 1430 us/step at B=1 and 3443 at B=8 over HTTP (online (submit/poll)); in process 1326 us/step at B=1 (step, MBU 0.476)

### #12 s0012: Track A: two K batches in flight for attention pairs on main blocks (batch 8)

- status: **valid**, http_step_us_b1 = 1419 (B=8: 3425), at 10.36 agent-hours, parent s0011, category attention
- hypothesis: track A (checkpoint formats, no format differs). s0010 with K2B on: at batch 7-8 the 8 attention pairs that run on main blocks keep two batches of K tiles in flight in the scores stage (one trip to memory per two rounds instead of per round). Off before because it measured +10 on random-token prompts; on prompts built like the HTTP client's it is -11 us at batch 8 (two prompt sets) and 0 at batch 1. Bit-identical to s0010 (digests, bcheck B=2..8); flake 0/199. Expect B=1 ~1420 (unchanged), B=8 3427 -> ~3416.
- evaluator: valid: 1419 us/step at B=1 and 3425 at B=8 over HTTP (online (submit/poll)); in process 1319 us/step at B=1 (step, MBU 0.479)

### #13 s0013: Track A: L2 evict-first hint on the expert and head tile loads

- status: **valid**, http_step_us_b1 = 1417 (B=8: 3356), at 11.05 agent-hours, parent s0012, category memory
- hypothesis: track A (checkpoint formats, no format differs). s0012 with the routed experts' gate/up and down tiles and the exact head's tiles (weights streamed once per step) loaded with an L2 evict-first cache policy (createpolicy + ld.global.L2::cache_hint, L1::no_allocate): they stop evicting the prefetched q/k/v and shared-expert tiles and the hand-over buffers. Cache hints only: bit-identical to s0012 (digests, bcheck B=2..8), flake 0/199. Client-like prompts, three prompt sets: B=8 -55..-70 us, B=1 unchanged. Expect B=1 ~1419, B=8 3425 -> ~3360.
- evaluator: valid: 1417 us/step at B=1 and 3356 at B=8 over HTTP (online (submit/poll)); in process 1316 us/step at B=1 (step, MBU 0.480)

### #14 s0014: Track A: evict-first expert/head loads without L1::no_allocate

- status: **valid**, http_step_us_b1 = 1412 (B=8: 3373), at 11.55 agent-hours, parent s0013, category memory
- hypothesis: track A (checkpoint formats, no format differs). s0013's evict-first L2 hint on the expert and head tile loads, now without the L1::no_allocate qualifier (only the L2 policy). Cache hints only: bit-identical to s0013 (digests, bcheck B=2..8), flake 0/199. Scored-layout A/Bs: B=1 -6 (two cards), B=8 -22 / -24 / -5..-11 (three client-like prompt sets). Expect B=1 1417 -> ~1411, B=8 3356 -> ~3340.
- evaluator: valid: 1412 us/step at B=1 and 3373 at B=8 over HTTP (online (submit/poll)); in process 1312 us/step at B=1 (step, MBU 0.481)

### #15 s0015: Track A: last layer's idle q/k/v prefetch slot fetches the head's first 20 MB

- status: **valid**, http_step_us_b1 = 1408 (B=8: 3368), at 11.96 agent-hours, parent s0014, category memory
- hypothesis: track A (checkpoint formats, no format differs). In the last layer the end-of-C q/k/v prefetch (idle warp, cp.async.bulk.prefetch.L2) has no next layer; a layer entry past the last whose q/k/v pointer is the head makes the same statement fetch the head's first 20 MB into L2 while HBM is idle (D/routing), so the head phase streams 20 of 615 MB from L2. No new kernel code (own-code variants grew the batch-1 frame 176->192 and lost +12..+21). Bit-identical to s0014 (digests, bcheck B=2..8), flake 0/199. Scored-layout A/B: B=1 -4/-5 on two cards, B=8 0..-3. Expect B=1 1411.8 -> ~1407, B=8 unchanged (~3360-3375).
- evaluator: valid: 1408 us/step at B=1 and 3368 at B=8 over HTTP (online (submit/poll)); in process 1307 us/step at B=1 (step, MBU 0.483)

### #16 s0016: Track B: int8 head sketch on the s0015 kernel, with the 8-mma scores form

- status: **valid**, http_step_us_b1 = 1400 (B=8: 3362), at 12.78 agent-hours, parent s0015, category memory
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged and the BF16 head still used for step() and the recomputed rows). s0015's source built as track B: the head sketch (s0002-s0006) now with the 8-mma scores form (KQT_SMALL=1; it used to cost the track-B build its allocation, now 176/16/8 B). Bit-identical to track A: bitcheck B=1/B=8 1152 positions bit_identical 1.0; generate/served/batch 1-8/long-context tokens identical; bcheck, digests, flake 0/199. Scored-layout A/B vs track A: -6 on two cards. Expect B=1 1407.8 -> ~1402, B=8 unchanged (~3370).
- evaluator: valid: 1400 us/step at B=1 and 3362 at B=8 over HTTP (online (submit/poll)); in process 1294 us/step at B=1 (generate, MBU 0.485)

### #17 s0017: Track A: dense-layer MLP tiles preloaded before their barriers

- status: **valid**, http_step_us_b1 = 1418 (B=8: 3379), at 14.05 agent-hours, parent s0015, category memory
- hypothesis: track A (checkpoint formats, no format differs). The two dense layers took ~51 us each (75 MB of MLP at ~2 TB/s: tile-by-tile loads, nothing preloaded before the barriers, unlike every MoE phase). DMLP=6: 16 gate/up tiles per warp and 16 of the 24 down tiles preloaded into the free tile registers before the waits, same mma order: bit-identical (digests, bcheck B=2..8), flake 0/199. Dense layers -18 us per token on the timeline in the track-B build; but this track-A build draws a slower allocation: scored-layout A/B B=1 +8, B=8 -14. Expect B=1 1407.8 -> ~1415 (a loss in A, submitted first for the track-B build), B=8 3368 -> ~3355.
- evaluator: valid: 1418 us/step at B=1 and 3379 at B=8 over HTTP (online (submit/poll)); in process 1312 us/step at B=1 (step, MBU 0.481)

### #18 s0018: Track B: s0017's dense-layer MLP preloads with the int8 head sketch

- status: **valid**, http_step_us_b1 = 1380 (B=8: 3367), at 14.23 agent-hours, parent s0017, category memory
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). s0016's track-B build plus s0017's dense-layer MLP preloads (DMLP=6). In this build the dense layers' change pays (timeline: layers 0-1 102.7 -> 84.3 us; scored-layout A/B -17..-20 vs s0016's source). Bit-identical to track A: bitcheck B=1/B=8 1152 positions bit_identical 1.0; generate/served/batch 1-8/long-context tokens identical; bcheck, digests, flake 0/199. Expect B=1 1400.4 -> ~1383, B=8 ~3365.
- evaluator: valid: 1380 us/step at B=1 and 3367 at B=8 over HTTP (online (submit/poll)); in process 1277 us/step at B=1 (generate, MBU 0.491)

### #19 s0019: Track A: dense-layer MLP epilogue: plain stores + one fence, down preload after

- status: **valid**, http_step_us_b1 = 1402 (B=8: 3367), at 14.78 agent-hours, parent s0017, category sync
- hypothesis: track A (checkpoint formats, no format differs). In the dense layers warp 0 folded and wrote the six gate/up row groups with atomic exchanges while the other warps' down-tile preloads queued ahead of them: an 8 us epilogue (stamped timeline). Now the down preload waits for warp 0's bump (DMLP 128) and the six row groups are plain stores with one __threadfence before the bump (256): epilogue ~5 us, dense layer end 34.0 -> 31.5 us. Same values: bit-identical (digests, bcheck B=2..8; bitcheck vs track B 1.0), flake 0/199 and flake_b 0. Scored-layout A/B vs s0017's source (track A): -8..-13. Expect B=1 1417.6 -> ~1407, B=8 ~3370.
- evaluator: valid: 1402 us/step at B=1 and 3367 at B=8 over HTTP (online (submit/poll)); in process 1299 us/step at B=1 (step, MBU 0.486)

### #20 s0020: Track B: s0019's dense-layer epilogue with the int8 head sketch

- status: **valid**, http_step_us_b1 = 1374 (B=8: 3350), at 14.82 agent-hours, parent s0019, category sync
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). s0018's track-B build plus s0019's dense-layer epilogue (down preload after warp 0's bump, plain stores + one fence). Bit-identical to track A: bitcheck B=1/B=8 1152 positions bit_identical 1.0; generate/served/batch 1-8/long-context tokens identical; bcheck, digests, flake 0/199. Scored-layout A/B vs s0018's source: -3..-5. Expect B=1 1379.6 -> ~1376, B=8 ~3367.
- evaluator: valid: 1374 us/step at B=1 and 3350 at B=8 over HTTP (online (submit/poll)); in process 1273 us/step at B=1 (generate, MBU 0.493)

### #21 s0021: Track A: dense-layer gate/up loop unrolled by 16 (more tiles in flight)

- status: **valid**, http_step_us_b1 = 1398 (B=8: 3361), at 15.52 agent-hours, parent s0019, category memory
- hypothesis: track A (checkpoint formats, no format differs). The dense layers' gate/up loop over the 32 tiles per warp not preloaded (s0017) left the load scheduling to the compiler; #pragma unroll 16 on it (DMLP 32768) lets it keep more tiles in flight. Same mma order: bit-identical (digests, bcheck B=2..8; bitcheck vs track B 1.0), flake 0/199, flake_b 0. Scored-layout A/B: track A -3, track B -2..-5 on three cards. Expect B=1 1401.8 -> ~1399, B=8 ~3360.
- evaluator: valid: 1398 us/step at B=1 and 3361 at B=8 over HTTP (online (submit/poll)); in process 1300 us/step at B=1 (step, MBU 0.486)

### #22 s0022: Track B: s0021's dense-layer gate/up unroll with the int8 head sketch

- status: **valid**, http_step_us_b1 = 1372 (B=8: 3353), at 15.56 agent-hours, parent s0021, category memory
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). s0020's track-B build plus s0021's dense-layer gate/up unroll. Bit-identical to track A: bitcheck B=1/B=8 1152 positions bit_identical 1.0; generate/served/batch 1-8/long-context tokens identical; bcheck, digests, flake 0/199. Scored-layout A/B vs s0020's source: -2..-5. Expect B=1 1374.4 -> ~1371, B=8 ~3350.
- evaluator: valid: 1372 us/step at B=1 and 3353 at B=8 over HTTP (online (submit/poll)); in process 1274 us/step at B=1 (generate, MBU 0.493)

### #23 s0023: Track B: head sketch pass 2 staged through shared memory (one load round)

- status: **valid**, http_step_us_b1 = 1368 (B=8: 3352), at 16.02 agent-hours, parent s0022, category memory
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). The sketch's pass 2 recomputed a candidate row group on two warps in four load rounds (~9 us); most steps have one candidate in one block and every block waited for it (stamped timeline: one block +8 us at the head's end). Now all 16 warps load the candidate's 128 tiles into shared memory in one round and the two warps run the same chains from there (same order); pass 2 per candidate 9.0 -> 3.6 us. Only the sketch's code changes (the track-A build does not contain it). Bit-identical to track A: bitcheck 1.0 at B=1/8; generate/served/batch 1-8/long-context tokens identical; digests, flake 0/199. Scored-layout A/B vs s0022: -5..-6. Expect B=1 1371.9 -> ~1366, B=8 unchanged (~3353).
- evaluator: valid: 1368 us/step at B=1 and 3352 at B=8 over HTTP (online (submit/poll)); in process 1282 us/step at B=1 (step, MBU 0.493)

### #24 s0024: Track A: chosen tokens published by atomic exchange, bump before the host word

- status: **valid**, http_step_us_b1 = 1400 (B=8: 3355), at 16.91 agent-hours, parent s0021, category sync
- hypothesis: track A (checkpoint formats, no format differs). After the head, block 0 wrote the chosen tokens with plain stores, fenced, wrote the host's progress word, fenced again, then bumped the counter every block waits on before the next step (stamped: block 0 ends the step ~3 us after the others). Now the tokens go out by atomic exchange, settled (the pattern every other cross-block result uses), then the bump; the generated tokens and the progress word are fenced after it. Same tokens (bit-identical; digests, bcheck B=2..8, bitcheck vs track B 1.0), flake 0/199, flake_b 0, watchdog test ok. Scored-layout A/B: track B -2 on two cards, track A 0. Expect B=1 1398.2 -> ~1398, B=8 ~3360.
- evaluator: valid: 1400 us/step at B=1 and 3355 at B=8 over HTTP (online (submit/poll)); in process 1297 us/step at B=1 (step, MBU 0.487)

### #25 s0025: Track B: s0024's token publish with the int8 head sketch

- status: **valid**, http_step_us_b1 = 1361 (B=8: 3332), at 16.94 agent-hours, parent s0024, category sync
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). s0023's track-B build plus s0024's publish of the chosen tokens (atomic exchange, settled, bump; host word after). Bit-identical to track A: bitcheck 1.0 at B=1/8; generate/served/batch 1-8/long-context tokens identical; digests, flake 0/199, flake_b 0, watchdog test ok. Scored-layout A/B vs s0023's source: -1.5..-2.5 and -1..-4. Expect B=1 1368.1 -> ~1366, B=8 ~3352.
- evaluator: valid: 1361 us/step at B=1 and 3332 at B=8 over HTTP (online (submit/poll)); in process 1266 us/step at B=1 (generate, MBU 0.494)

### #26 s0026: Track A: drop a never-true clause that kept a dead branch in the batch-1 kernel

- status: **valid**, http_step_us_b1 = 1393 (B=8: 3339), at 19.50 agent-hours, parent s0024, category structure
- hypothesis: track A (checkpoint formats, no format differs). The batch-8 q/k/v prefetch statement sits in the MoE path but carried a '!L.moe' clause; the batch-1 kernel could not drop the branch (L.moe is a runtime field) although it is never taken. Without the clause the batch-1 kernel's SASS equals the QPF=0 build's (same hash) and draws a better allocation; the batch-8 kernel is unchanged. No arithmetic changes: digests and bcheck B=2..8 unchanged. Scored-layout A/B (track B build): -6..-9 on two cards. Expect B=1 1399.5 -> ~1392 (if the track-A build draws the same way), B=8 ~3355.
- evaluator: valid: 1393 us/step at B=1 and 3339 at B=8 over HTTP (online (submit/poll)); in process 1294 us/step at B=1 (step, MBU 0.488)

### #27 s0027: Track B: s0026's dead-branch removal with the int8 head sketch

- status: **valid**, http_step_us_b1 = 1353 (B=8: 3333), at 19.62 agent-hours, parent s0026, category structure
- hypothesis: track B (formats: an extra certified int8 copy of lm_head + per-block error norms, read on the greedy paths; every checkpoint weight unchanged). s0025's track-B build plus s0026's removal of a never-true clause (the batch-1 kernel drops a dead branch: SASS identical to the QPF=0 build). No arithmetic change: digests, bcheck B=2..8 unchanged; flake 0/199. Bit-identical to track A by construction (only a dead branch differs from s0025, whose bitcheck was 1.0). Scored-layout A/B vs s0025: -6..-9 on two cards. Expect B=1 1360.9 -> ~1353, B=8 ~3335.
- evaluator: valid: 1353 us/step at B=1 and 3333 at B=8 over HTTP (online (submit/poll)); in process 1259 us/step at B=1 (generate, MBU 0.497)

