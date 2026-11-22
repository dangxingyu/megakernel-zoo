# s18-mini-b200-c3: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 12.0 agent-hours. 15 submissions so far, 15 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine with speculative decoding: 1404
- vLLM (inkling-2-mini image) + DSpark (general engine, track C): 1544
- vLLM (PR #69028 source build) + DSpark (general engine, track C): 1552
- SGLang (PR #51926 image, GB300-style settings) + DSpark (general engine, track C): 2347
- SGLang (PR #51926 image) + DSpark (general engine, track C): 2828
- vLLM (inkling-2-mini image) without speculative decoding (track A line, for scale): 3694
- SGLang (PR #51926 image) without speculative decoding (track A line, for scale): 4389
- vLLM (PR #69028 source build) without speculative decoding (track A line, for scale): 3720

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: baseline: inherited s16 s0022 track C engine unchanged

- status: **valid**, http_step_us_b1 = 1527 (B=8: 6297), at 0.13 agent-hours, category other
- hypothesis: track C (checkpoint formats, verifier = track A s0018 step); re-evaluation of the inherited s0022 source without change, to fix this run's noise baseline: expect http_step_us_b1 1528 +- 20, natural_b1 1020 +- 15, tokens_per_event 2.88 / 5.22.
- evaluator: valid: 1527 us/step at B=1 and 6297 at B=8 over HTTP (online (submit/poll)); in process 3085 us/step at B=1 (generate, MBU 0.224)

### #2 s0002: draft length by expected tokens per verify time (p^2.5), early exit 0.3

- status: **valid**, http_step_us_b1 = 1535 (B=8: 6296), at 0.91 agent-hours, parent s0001, category scheduling
- hypothesis: track C (checkpoint formats; verifier unchanged = s0018 step). The draft length kd now maximizes (1 + sum cum_p^2.5) / (V(kd+1) + 400 us) with the measured V(K) table instead of prod p >= 0.4 (simulator: synthetic -1.3%, natural -0.3%; served A/B on card 1, 12 prompts: synthetic -0.5%, natural -0.9%). Expect http_step_us_b1 -0.2..-0.6% (1527 -> ~1520), natural_b1 -0.7% (1017 -> ~1010), tokens_per_event ~unchanged (2.88 / 5.2).
- evaluator: valid: 1535 us/step at B=1 and 6296 at B=8 over HTTP (online (submit/poll)); in process 3099 us/step at B=1 (step, MBU 0.223)

### #3 s0003: 16-row verify passes for 15-token n-gram copies (second library, one-pass MoE)

- status: **valid**, http_step_us_b1 = 1454 (B=8: 6279), at 1.50 agent-hours, parent s0002, category structure
- hypothesis: track C (checkpoint formats; step() unchanged: bitcheck vs s0001 bit-identical B=1/8, RING 32). n-gram copies at anchors <= 1008 now propose 15 tokens verified in one 16-row pass (engine/mk16.cu = s16 proto16 + one-pass 16-row MoE: V(16) 10.6 -> 8.76 ms; rows bit-identical to step(), vcheck16 0/1020); raises the 8-token cap that 44% of natural cycles hit. Served A/B on the HTTP prompts: synthetic -3.5%, natural -6.3%. Expect http_step_us_b1 1535 -> ~1480, natural_b1 1013 -> ~950, tokens_per_event b1 2.88 -> ~3.2, natural_b1 5.2 -> ~6.0; b8 unchanged.
- evaluator: valid: 1454 us/step at B=1 and 6279 at B=8 over HTTP (online (submit/poll)); in process 3102 us/step at B=1 (generate, MBU 0.222)

### #4 s0004: 16-row passes: final unembedding in one pass (weights read once)

- status: **valid**, http_step_us_b1 = 1447 (B=8: 6275), at 1.78 agent-hours, parent s0003, category gemm
- hypothesis: track C (checkpoint formats; step() unchanged). The 16-row library computed the final unembedding per 8-row half (922 MB read twice); now both column groups share each weight fragment (bit-identical: vcheck16 0/1020). V(16) 8.77 -> 8.61 ms at ctx 600. Expect http_step_us_b1 1454 -> ~1445 (-0.5%), natural_b1 990 -> ~982 (-0.8%), tokens/event unchanged (3.13 / 6.76).
- evaluator: valid: 1447 us/step at B=1 and 6275 at B=8 over HTTP (online (submit/poll)); in process 3098 us/step at B=1 (generate, MBU 0.222)

### #5 s0005: 16-row n-gram passes only in a copying regime (last cycle accepted >= 7)

- status: **valid**, http_step_us_b1 = 1430 (B=8: 6281), at 1.88 agent-hours, parent s0004, category scheduling
- hypothesis: track C (checkpoint formats; step() and both verify kernels unchanged). An n-gram proposal gets 15 drafts (a 16-row pass, +3.2 ms over 8 rows) only when the cycle just verified accepted >= 7 drafts, else 7 drafts in 8 rows: short-lived matches no longer pay for 16 rows. Simulated (12 natural prompts) -1.9%; served A/B: natural mean -1.5%, natural prompt 3 (the HTTP median) 993 -> 965. Expect natural_b1 982 -> ~958 (-2.5%), http_step_us_b1 1447 -> ~1443 (-0.3%), tokens/event natural 6.76 -> ~6.6.
- evaluator: valid: 1430 us/step at B=1 and 6281 at B=8 over HTTP (online (submit/poll)); in process 3102 us/step at B=1 (generate, MBU 0.222)

### #6 s0006: draft refinements: FP8 draft-layer copies, 1-row passes when no draft pays

- status: **valid**, http_step_us_b1 = 1416 (B=8: 6279), at 2.33 agent-hours, parent s0005, category other
- hypothesis: track C (checkpoint formats; step() and both verify kernels unchanged; the draft only proposes). The draft's own layer GEMVs read FP8 per-row-scaled copies (draft only) and the length rule may choose a 1-row pass (K=1) when no draft is worth a row. Served A/B (12 prompts): synthetic mean -0.7% (FP8 layers), natural mean -0.3% (K=1); HTTP-median prompts syn10000 1440 -> 1428, nat3 964 -> 967. Expect http_step_us_b1 1430 -> ~1420 (-0.7%), natural_b1 959 -> ~957 (noise), tokens/event ~3.08 / 6.6.
- evaluator: valid: 1416 us/step at B=1 and 6279 at B=8 over HTTP (online (submit/poll)); in process 3100 us/step at B=1 (generate, MBU 0.223)

### #7 s0007: final source re-evaluated (= s0006 + off-by-default switches)

- status: **valid**, http_step_us_b1 = 1419 (B=8: 6279), at 2.94 agent-hours, parent s0006, category other
- hypothesis: track C (checkpoint formats; step() unchanged). The committed source differs from s0006 only by switches that are off (G_LP, G_DB, QKV1, ROUT1, MRG2, REC2, W2_BPC16 at its old value): a reproducibility check. Expect http_step_us_b1 1416 +- 15, natural_b1 960 +- 10, tokens/event 3.07 / 6.61.
- evaluator: valid: 1419 us/step at B=1 and 6279 at B=8 over HTTP (online (submit/poll)); in process 3101 us/step at B=1 (generate, MBU 0.222)

### #8 s0008: 1-row cycles on the B=1 step kernel (step()'s code, position from the device)

- status: **valid**, http_step_us_b1 = 1426 (B=8: 6297), at 3.56 agent-hours, parent s0007, category launch
- hypothesis: track C (checkpoint formats; step() bit-identical to s0006 by bitcheck B=1/8, its B=1 time within 10 us). When the draft chooses a 1-row pass, the cycle runs step()'s own B=1 kernel with the position read on the device (3202 vs 3426 us; the draft's inputs identical on 40 positions); the length rule uses V(1) = 3100. Served A/B (12 prompts): synthetic mean -0.8%, natural -0.5% vs the same build without it (about -0.3% vs s0006). Expect http_step_us_b1 1416 -> ~1410, natural_b1 960 -> ~957, b8 unchanged (6279 +- 15).
- evaluator: valid: 1426 us/step at B=1 and 6297 at B=8 over HTTP (online (submit/poll)); in process 3115 us/step at B=1 (generate, MBU 0.221)

### #9 s0009: 16-row passes: router halves on separate blocks (final source)

- status: **valid**, http_step_us_b1 = 1418 (B=8: 6280), at 4.30 agent-hours, parent s0008, category sync
- hypothesis: track C (checkpoint formats; step() unchanged = s0007). In 16-row passes the two 8-row router GEMVs run on blocks 0-7 and 8-15 at the same time instead of one after the other on blocks 0-7 (bit-identical: vcheck16 0/1020); V(16) 8574 -> 8543 us. Everything else is s0007 plus off-by-default switches. Expect http_step_us_b1 1419 +- 10 (16-row passes are 13% of synthetic time), natural_b1 962 -> ~960, tokens/event unchanged 3.07 / 6.61.
- evaluator: valid: 1418 us/step at B=1 and 6280 at B=8 over HTTP (online (submit/poll)); in process 3093 us/step at B=1 (generate, MBU 0.223)

### #10 s0010: 16-row passes: two queries per warp in the attention prologue (final source)

- status: **valid**, http_step_us_b1 = 1418 (B=8: 6277), at 4.53 agent-hours, parent s0009, category attention
- hypothesis: track C (checkpoint formats; step() unchanged = s0007/s0009). In 16-row passes' shared-record attention, each warp task prepares two (row, head) queries on 16 lanes each, carrying the original lanes' partial sums so the L2-clip reductions are bit for bit the same (vcheck16 0/1020); V(16) 8545 -> 8520 us over 4 rounds. Expect http_step_us_b1 1418 +- 10, natural_b1 961 -> ~960, tokens/event unchanged 3.07 / 6.61.
- evaluator: valid: 1418 us/step at B=1 and 6277 at B=8 over HTTP (online (submit/poll)); in process 3110 us/step at B=1 (generate, MBU 0.222)

### #11 s0011: short n-gram copies when the DSpark draft is unconfident

- status: **valid**, http_step_us_b1 = 1409 (B=8: 6276), at 4.80 agent-hours, parent s0010, category scheduling
- hypothesis: track C (checkpoint formats; step() and both verify kernels unchanged; the draft only proposes). When the committed suffix has an earlier occurrence of 4 tokens (below the n-gram rule's 5) and DSpark's first draft has p < 0.7, the cycle verifies the 3 copied tokens (K=4) instead of the DSpark chain. Simulated synthetic -0.3..-0.7%; served A/B (3 rounds): HTTP-median prompts synthetic 1430 -> 1419, natural 966 -> 960. Expect http_step_us_b1 1418 -> ~1408, natural_b1 959 -> ~954, tokens/event ~3.1 / 6.6.
- evaluator: valid: 1409 us/step at B=1 and 6276 at B=8 over HTTP (online (submit/poll)); in process 3096 us/step at B=1 (generate, MBU 0.223)

### #12 s0012: short n-gram copies from 3-token matches

- status: **valid**, http_step_us_b1 = 1404 (B=8: 6279), at 4.91 agent-hours, parent s0011, category scheduling
- hypothesis: track C (checkpoint formats; step() and both verify kernels unchanged; draft only). The short-copy rule of s0011 now accepts 3-token earlier occurrences (was 4). Served A/B (2 rounds): HTTP-median synthetic 1420 -> 1416, natural =, synthetic mean -0.2%. Expect http_step_us_b1 1408.6 -> ~1404, natural_b1 954 +- 3, tokens/event ~3.1 / 6.6.
- evaluator: valid: 1404 us/step at B=1 and 6279 at B=8 over HTTP (online (submit/poll)); in process 3096 us/step at B=1 (generate, MBU 0.223)

### #13 s0013: plain n-gram copies only from 7-token matches (5-6 go to DSpark / short copies)

- status: **valid**, http_step_us_b1 = 1396 (B=8: 6303), at 5.17 agent-hours, parent s0012, category scheduling
- hypothesis: track C (checkpoint formats; step() and both verify kernels unchanged; draft only). With the short-copy rule in place, 5-6-token matches do better through DSpark (or a short copy when DSpark is unconfident) than as 7-draft copies. Served A/B (3 rounds): HTTP-median natural 960 -> 950, synthetic 1416 -> 1417, natural mean -1%. Expect natural_b1 954.6 -> ~946, http_step_us_b1 1404 +- 5.
- evaluator: valid: 1396 us/step at B=1 and 6303 at B=8 over HTTP (online (submit/poll)); in process 3061 us/step at B=1 (generate, MBU 0.225)

### #14 s0014: 16-row passes at every position (per-row attention path takes 16 rows)

- status: **valid**, http_step_us_b1 = 1384 (B=8: 6303), at 5.85 agent-hours, parent s0013, category structure
- hypothesis: track C (checkpoint formats; step() unchanged; 16-row rows bit-identical to step() to position 1700 by vcheck16). 16-row n-gram passes were limited to anchors <= 1008; the 16-row library's per-row attention path now takes 16 rows of one sequence, so copies run to the end of the request (not past it: the draft knows the request's last 16-row anchor). Served A/B: synthetic HTTP-median 1396 -> 1384 (-0.9%), natural means -0.3%, but natural prompt 3 (the natural median) +0.6%. Expect http_step_us_b1 1396 -> ~1385, natural_b1 940.6 -> ~946 (slightly worse), tokens/event ~3.12 / 6.75.
- evaluator: valid: 1384 us/step at B=1 and 6303 at B=8 over HTTP (online (submit/poll)); in process 3057 us/step at B=1 (generate, MBU 0.226)

### #15 s0015: 16-row passes past position 1008 only after a fully accepted one

- status: **valid**, http_step_us_b1 = 1391 (B=8: 6280), at 6.03 agent-hours, parent s0014, category scheduling
- hypothesis: track C (checkpoint formats; step() unchanged; draft only). Past position 1008 (where 16-row passes run the per-row attention path, V(16)/V(8) = 1.53) a 16-row copy is proposed only right after a cycle that accepted >= 15 drafts. Served A/B (2 rounds): HTTP-median synthetic 1384 (unchanged vs s0014), natural 947 -> 942-943. Expect http_step_us_b1 1383.6 +- 4, natural_b1 946.8 -> ~943.
- evaluator: valid: 1391 us/step at B=1 and 6280 at B=8 over HTTP (online (submit/poll)); in process 3101 us/step at B=1 (generate, MBU 0.222)

