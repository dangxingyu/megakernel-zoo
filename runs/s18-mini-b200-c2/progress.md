# s18-mini-b200-c2: Inkling-2-Mini on GB200

Score: `step_us_b1` (lower is better). Budget 12.0 agent-hours. 15 submissions so far, 14 valid.

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

### #1 s0001: track A: step() for B>=2 on a new wide pass (batch-invariant rows); B=1 unchanged

- status: **valid**, step_us_b1 = 3093, at 1.23 agent-hours, quick evaluation, category structure
- hypothesis: track A (checkpoint formats unchanged; expert inputs exact). step() for B>=2 now runs engine/wk.cu: per-row arithmetic independent of the other rows (fixed k-splits, absolute 128-key attention chunks, per (expert, token) columns), so the same rows can later be verified 8 sequences x K positions at a time. Expect every gate to pass: batch gate p99 ~0.15 (limit 0.254, measured on word-list prompts), invariance bitwise, numeric on par with the megakernel; B=8 step slower (~10 ms vs 6.3 in process): this is a verifier-only check, not a speed submission.
- evaluator: valid: 3104 us/step at B=1 (MBU 0.223), 10157 us/step at B=8

### #2 s0002: track C: B>=2 cohorts by draft-and-verify (8 DSpark drafts, one wide verify pass per cycle)

- status: **valid**, http_step_us_b1 = 1523 (B=8: 8761), at 1.37 agent-hours, parent s0001, category structure
- hypothesis: track C (checkpoint formats unchanged). B=8 cohorts now speculate: per cycle one wide pass (engine/wk.cu, step()'s B>=2 function, rows of all live sequences at once) verifies each sequence's anchor + DSpark/n-gram drafts (dynamic length), ragged acceptance per sequence, drafts as 8 FusedDraft clones. Expect b8 6302 -> ~3500 us/token on the evaluator prompts and natural_b8 6508 -> ~5700 (measured 5699 in process on natural prompts, cycle ~20 ms); b1 unchanged (~1530); every streamed token equal to step()'s greedy.
- evaluator: valid: 1523 us/step at B=1 and 8761 at B=8 over HTTP (online (submit/poll)); in process 3097 us/step at B=1 (generate, MBU 0.223)

### #3 s0003: track C: wide pass with programmatic dependent launch, attention staging, deeper GEMM prefetch

- status: **valid**, http_step_us_b1 = 1523 (B=8: 7181), at 1.67 agent-hours, parent s0002, category launch
- hypothesis: track C (formats unchanged). Wide verify pass: every kernel launches its successor early (PDL; dense GEMMs prefetch weights before waiting), attention loads q/r and relp before the K/V chunk and stages merge records with cp.async, GEMM weights 4 units in flight. Same arithmetic (bit-identical rows). Measured in process: wide pass 10.7 -> 9.2 ms at 8 rows, 25.5 -> 20.7 at 64; natural B=8 serving 5733 -> 4782 us/token. Expect natural_b8 5364 -> ~4500, b8 8761 -> ~7400, b1 unchanged.
- evaluator: valid: 1523 us/step at B=1 and 7181 at B=8 over HTTP (online (submit/poll)); in process 3096 us/step at B=1 (generate, MBU 0.223)

### #4 s0004: track C: wide attention items span several chunks, K/V double-buffered

- status: **valid**, http_step_us_b1 = 1523 (B=8: 7186), at 1.94 agent-hours, parent s0003, category attention
- hypothesis: track C (formats unchanged). Wide-pass attention: one item per (sequence, kv head, part of its 128-key chunks) with the next chunk's K/V loaded during the current one and q/r prepared once per item instead of once per chunk; same per-chunk records and merge (bit-identical rows, A/B 0 differing). Measured in process: pass -3.5% (8 rows) / -4.4% (64 rows). Expect natural_b8 4476 -> ~4300, b8 7181 -> ~6900, b1 unchanged.
- evaluator: valid: 1523 us/step at B=1 and 7186 at B=8 over HTTP (online (submit/poll)); in process 3086 us/step at B=1 (generate, MBU 0.223)

### #5 s0005: track C: wide-pass routing tables in parallel, 24-warp register-pipelined MoE, GEMM L2 prefetch

- status: **valid**, http_step_us_b1 = 1522 (B=8: 6437), at 2.62 agent-hours, parent s0004, category scheduling
- hypothesis: track C (formats unchanged). Wide pass: the per-block routing tables (expert lists, chunks) are built by masks and scans instead of serial scans (42 -> 15 us per application at 64 rows, measured by in-kernel marks), the MoE kernels run 24 warps with weights pipelined in registers, dense GEMMs bulk-prefetch their strips into L2 before waiting on the previous kernel and stage B first. Bit-identical rows (A/B 0 differ). In process: pass 8.6 -> 7.8 ms (8 rows), 18.9 -> 16.7 ms (64 rows); natural B=8 serving 4782 -> 4242 us/token. Expect natural_b8 4464 -> ~4000, b8 7186 -> ~6400, b1 unchanged.
- evaluator: valid: 1522 us/step at B=1 and 6437 at B=8 over HTTP (online (submit/poll)); in process 3093 us/step at B=1 (generate, MBU 0.223)

### #6 s0006: track C: wide pass: 64-row GEMM chunks, row-job weights staged early, loads hoisted

- status: **valid**, http_step_us_b1 = 1525 (B=8: 6353), at 2.88 agent-hours, parent s0005, category memory
- hypothesis: track C (formats unchanged). Wide pass: dense GEMMs with more than 32 rows run one 64-row chunk in k-stages (weights read once, the 64-row unembedding no longer reads 0.93 GB twice), row-job kernels (mixes, norms) copy the layer's query / norm / tap vectors into shared memory before waiting on the previous kernel and issue all row loads before the first reduction, the attention merge preloads gates and sinks. Bit-identical rows (A/B 0 differ). In process vs s0005's build: pass 7.80 -> 7.48 ms (8 rows), 16.7 -> 16.0 ms (64 rows). Expect natural_b8 3966 -> ~3800, b8 6437 -> ~6150, b1 unchanged.
- evaluator: valid: 1525 us/step at B=1 and 6353 at B=8 over HTTP (online (submit/poll)); in process 3096 us/step at B=1 (generate, MBU 0.223)

### #7 s0007: track C: MoE expert-input groups double-buffered, items balanced over the block range

- status: **valid**, http_step_us_b1 = 1521 (B=8: 6048), at 3.24 agent-hours, parent s0006, category scheduling
- hypothesis: track C (formats unchanged). Wide-pass MoE kernels stage up to 2 (w13) / 4 (w2) expert-chunk input groups at once and distribute the block's items round-robin over its 24 warps across group boundaries (no block barrier per group: removes the per-group staging latency and the 2-vs-3-items imbalance). Bit-identical rows (A/B 0 differ). In process: pass -9% at 8 rows (same-token A/B), -6.5% with random tokens; natural B=8 serving 4242 -> 3919 us/token (incl. the s0006 changes). Expect natural_b8 3811 -> ~3600, b8 6353 -> ~5950, b1 unchanged.
- evaluator: valid: 1521 us/step at B=1 and 6048 at B=8 over HTTP (online (submit/poll)); in process 3096 us/step at B=1 (generate, MBU 0.223)

### #8 s0008: track C: B=8 cycle's verify pass as a CUDA graph; GEMM smem sized by rows for PDL overlap

- status: **incorrect**, no score, at 3.54 agent-hours, parent s0007, category launch
- hypothesis: track C (formats unchanged). The B>=2 draft-and-verify cycle replays its wide pass as one CUDA graph (PDL edges kept; launch cost 1.41 -> 0.98 us per kernel in bench/launch.cu), and dense-GEMM kernels launch with shared memory sized for the pass's rows (shorter k-stages) and issue their weight loads before waiting on the previous kernel, so they can start while it ends. Bit-identical rows. In process: natural B=8 serving 3937 -> 3884 us/token (graph), 1-row pass 5242 -> 5012 us. Expect natural_b8 3641 -> ~3560, b8 6048 -> ~5920, b1 unchanged.
- evaluator: incorrect: failed http_tokens
- gates that failed: http_tokens

### #9 s0009: track C: s0008 fixed (GEMM smem over all row variants, graph capture without warm-up)

- status: **valid**, http_step_us_b1 = 1521 (B=8: 5948), at 3.84 agent-hours, parent s0007, category correctness
- hypothesis: track C (formats unchanged). s0008's changes (cycle pass as a CUDA graph, GEMM smem sized by rows, weights before the wait) with its two bugs fixed: launch smem is the maximum over the GEMM variants up to the row bound (stage sizes keep every variant <= the 64-row one) and the graph is captured without a warm-up pass (which had overwritten a live sequence's K/V). Tested: ragged passes under the bound bit-identical to exact launches, 32-token prompts x 1120 tokens 8/8, staggered arrivals 8/8. Expect valid, natural_b8 3641 -> ~3550, b8 6048 -> ~5900, b1 unchanged.
- evaluator: valid: 1521 us/step at B=1 and 5948 at B=8 over HTTP (online (submit/poll)); in process 3088 us/step at B=1 (generate, MBU 0.223)

### #10 s0010: track C: gate F/G epilogue writes (f, g); x' computed in the norm phase (coalesced)

- status: **valid**, http_step_us_b1 = 1522 (B=8: 5930), at 3.98 agent-hours, parent s0009, category memory
- hypothesis: track C (formats unchanged). The loop-gate F/G kernel no longer loads x and m per output (scattered 4-byte loads in the GEMM epilogue): it stores (f, g) and the gate-norm row jobs compute x' = fma(1 - f, x, g m) from coalesced rows (the same operation sequence as the old epilogue's SASS: bit-identical, A/B 0 rows differ). In process: gate F/G 13.8 -> 12.2 us per launch at 48 rows, pass at 48 rows 13886 -> 13599 us. Expect natural_b8 3577 -> ~3510, b8 5948 -> ~5850, b1 unchanged.
- evaluator: valid: 1522 us/step at B=1 and 5930 at B=8 over HTTP (online (submit/poll)); in process 3094 us/step at B=1 (generate, MBU 0.223)

### #11 s0011: track C: the 8 sequences' drafts in one CUDA graph (one launch per kernel), base logits of all 8 by one shared-decode kernel

- status: **valid**, http_step_us_b1 = 1526 (B=8: 5740), at 5.06 agent-hours, parent s0010, category scheduling
- hypothesis: track C (verifier unchanged: wk.cu bit-identical, 0 rows differ; drafts only): per cycle the 8 DSpark drafts ran as 8 graphs side by side, 1.06 ms, each re-decoding and re-reading the FP8 unembedding; one graph with one launch per kernel for all 8 and a base-logits kernel that decodes each weight fragment once for all 8 sequences: 0.85 ms (tokens per event unchanged in tests). Expected natural_b8 3573 -> ~3520, b8 5930 -> ~5830; b1 unchanged (1522).
- evaluator: valid: 1526 us/step at B=1 and 5740 at B=8 over HTTP (online (submit/poll)); in process 3090 us/step at B=1 (generate, MBU 0.223)

### #12 s0012: track C: draft length threshold 0.6 for B>=2 cohorts (B=1 keeps 0.4)

- status: **valid**, http_step_us_b1 = 1525 (B=8: 5616), at 5.46 agent-hours, parent s0011, category scheduling
- hypothesis: track C (verifier and drafts unchanged; only the cumulative-confidence threshold that sets each sequence's draft length in B>=2 cohorts, 0.4 -> 0.6): fewer rejected rows per verify pass at a small loss of tokens per cycle; measured in process -2.6% (random-token prompts, ~2.7 tokens/event) and -1.5% (natural). Expected b8 5740 -> ~5600, natural_b8 ~3570 -> ~3520, b1 unchanged.
- evaluator: valid: 1525 us/step at B=1 and 5616 at B=8 over HTTP (online (submit/poll)); in process 3094 us/step at B=1 (generate, MBU 0.223)

### #13 s0013: track C: MoE top-8 computed once per 8-row group by the router kernel (not in every w13 block)

- status: **valid**, http_step_us_b1 = 1525 (B=8: 5528), at 5.82 agent-hours, parent s0012, category scheduling
- hypothesis: track C (verifier bit-identical: 0 rows differ vs the previous library; drafts unchanged): the router GEMM runs as (strip, 8-row group) blocks and the last block of each group computes its rows' top-8 into global memory; the 152 w13 blocks only load it (they each recomputed every row's top-8: 5-10 us per application at 48-64 rows). Measured in process: 48-row pass -2.7%, 64 rows -4.4%, serving natural -1.7%. Expected b8 5616 -> ~5520, natural_b8 ~3600 -> ~3540, b1 unchanged.
- evaluator: valid: 1525 us/step at B=1 and 5528 at B=8 over HTTP (online (submit/poll)); in process 3086 us/step at B=1 (generate, MBU 0.223)

### #14 s0014: track C: loop-gate D GEMM as (strip, 8-row group) blocks (16-48 blocks instead of 4-12)

- status: **valid**, http_step_us_b1 = 1524 (B=8: 5335), at 6.02 agent-hours, parent s0013, category scheduling
- hypothesis: track C (verifier bit-identical: 0 rows differ; the 8-row GEMM variant has the same per-row arithmetic): the gate-D GEMM of the loop controllers ran on 12 / 4 blocks, each one strip over all rows (20 us per application at 48 rows); as (task, 8-row group) blocks it takes about its 8-row time. In process: 48-row pass -5.4%, serving natural -3.7%, random-token prompts -3.1%. Expected b8 5528 -> ~5360, natural_b8 3501 -> ~3380, b1 unchanged.
- evaluator: valid: 1524 us/step at B=1 and 5335 at B=8 over HTTP (online (submit/poll)); in process 3079 us/step at B=1 (generate, MBU 0.224)

### #15 s0015: track C: gate F/G GEMM in 8-row groups; per-group top-8 only above 8 rows

- status: **valid**, http_step_us_b1 = 1525 (B=8: 5276), at 6.29 agent-hours, parent s0014, category scheduling
- hypothesis: track C (verifier bit-identical: 0 rows differ vs s0014): the loop-gate F/G GEMM staged and multiplied every row in each block (12 us per application at 48 rows vs 3.6 at 8); now (warp-task group, 8-row group) blocks above 8 rows; the router's per-group top-8 (s0013) cost 3.5 us per application at <= 8 rows and is used above 8 rows only. In process: 8 rows -2%, 64 rows -1.5%, serving natural -1.5%, random -1%. Expected b8 5335 -> ~5270, natural_b8 3345 -> ~3300, b1 unchanged.
- evaluator: valid: 1525 us/step at B=1 and 5276 at B=8 over HTTP (online (submit/poll)); in process 3086 us/step at B=1 (generate, MBU 0.223)

