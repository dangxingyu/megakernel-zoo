# s11-north-h100-b1: North-Mini-Code-1.0 on H100

Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 26 submissions so far, 26 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 1973
- 1.10x the fastest general engine: 4590
- vLLM 0.32.0 (general engine): 5049
- Cohere's megakernel @ 67d0b9ca (megakernel written for this model): 3616
- inherited engine, track A (s0022): 2540
- inherited engine, track B (s0021): 2150

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track B made bit-identical: W12 escapes patched into mma operands, L0 tables by kernel

- status: **valid**, http_step_us_b1 = 2123 (B=8: 5769), at 0.70 agent-hours, category correctness
- hypothesis: track B; formats differing from the checkpoint: 12-bit weights (row exponent window, neuron shifts, exact escapes), layer-0 per-token tables, int8 head sketch on the greedy path (as inherited). Change: the 12-bit widening adds an exponent bias of 32 so every escape is a normal bf16 in the scaled space and is XORed into the mma operand register before the product (one uniform header-bit test per quantum, a compact patch loop for the ~22% of quanta with an escape); the layer-0 tables are made by the kernel's own layer 0 (a DUMP launch, 8 ids at a time). Logits bit-identical to the track-A build of this source (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8: bit_identical 1.0, both batch-invariant); all 30,484,201,472 weights widen to the exact operand through the kernel's decode (mk_w12_widen, 0 mismatches). In process: B=1 generate 2126 -> 2113, teacher-forced step B=8 5887 -> 5778. Expect http_step_us_b1 ~2140 (inherited 2150), B=8 ~5950-6050.
- evaluator: valid: 2123 us/step at B=1 and 5769 at B=8 over HTTP (online (submit/poll)); in process 2109 us/step at B=1 (generate, MBU 0.903)

### #2 s0002: Track A rebuild of the current source (bit-identical partner of track B)

- status: **valid**, http_step_us_b1 = 2540 (B=8: 6901), at 2.54 agent-hours, parent s0001, category other
- hypothesis: track A: no format differs from the checkpoint (BF16 weights, BF16 activations). The current source built with TRACK=A: same kernel organisation as s0022 plus code that is inert for BF16 (escape patching, DUMP table launch, the batch-1 bulk prefetch is off for BF16 because it cost +51 us there in a same-card A/B, e0054). Its logits are bit-identical to the track-B build of this source (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8). In process B=1 generate ~2528 (e0054). Expect http_step_us_b1 ~2540 +- 15 (s0022: 2540), B=8 ~6900 +- 60.
- evaluator: valid: 2540 us/step at B=1 and 6901 at B=8 over HTTP (online (submit/poll)); in process 2527 us/step at B=1 (generate, MBU 0.782)

### #3 s0003: Batch-1 kernel bulk-prefetches its next-layer stage-A tiles into L2 (W12 only)

- status: **valid**, http_step_us_b1 = 2108 (B=8: 5769), at 2.68 agent-hours, parent s0001, category memory
- hypothesis: track B; formats as s0001 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change: in the batch-1 kernel, right after signalling the end of stage C, thread 0 of every block issues cp.async.bulk.prefetch.L2 for its next-layer stage-A share (123 KB): HBM works during the barrier and the norm instead of idling, and the TMA prefetch returns nothing to the SM, so the block's own barrier polls do not queue behind it (bench/tma_pf.cu: prefetched regions read 1.3x faster than cold). Same-card A/B (e0055, 3 rounds): B=1 generate -14, long context -23, step -20 us; B=8 unchanged (code only in the batch-1 instantiation). On the BF16 build it costs +51 us (e0054), so it is on for the 12-bit format only. Logits bit-identical to track A (bitcheck e0056). Expect http_step_us_b1 ~2105 (s0001 2123), B=8 ~5770.
- evaluator: valid: 2108 us/step at B=1 and 5769 at B=8 over HTTP (online (submit/poll)); in process 2103 us/step at B=1 (generate, MBU 0.910)

### #4 s0004: Pad 8-column operand strides (bank conflicts) - track A

- status: **valid**, http_step_us_b1 = 2535 (B=8: 6823), at 3.12 agent-hours, parent s0002, category memory
- hypothesis: track A: no format differs from the checkpoint. The mma B operands of the 8 batch columns were read with column strides of 1024/1536/2048/512 words (normed input in shared memory, gate|up output h and attention output through L1, sketch digits): all 8 columns in one bank, an 8-way conflict on every operand load. Strides padded by 4 words (layout only; logits unchanged, bit-identical to track B of this source). Microbenchmark (bench/stream_bench.cu): the 12-bit loop's per-SM ceiling 32 -> 42 GB/s. Same-card A/B (e0063, track A): B=8 generate -89, step -88, long -68 us; B=1 -4. Expect http_step_us_b1 ~2537 (s0002 2540), B=8 ~6810 (6901).
- evaluator: valid: 2535 us/step at B=1 and 6823 at B=8 over HTTP (online (submit/poll)); in process 2521 us/step at B=1 (generate, MBU 0.784)

### #5 s0005: Pad 8-column operand strides (bank conflicts) - track B

- status: **valid**, http_step_us_b1 = 2099 (B=8: 5646), at 3.28 agent-hours, parent s0003, category memory
- hypothesis: track B; formats as s0003 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as s0004 on track A: the 8-column mma B operands (normed input, h, attention output, sketch digits) had column strides that put all 8 columns in one bank; padded by 4 words, layout only, logits bit-identical to track A (bitcheck e0064). Same-card A/B (e0062, track B, 3 rounds): B=8 generate -155, step -153, long -114 us; B=1 +3 (noise). Expect http_step_us_b1 ~2108 (s0003 2108), B=8 ~5620 (5769).
- evaluator: valid: 2099 us/step at B=1 and 5646 at B=8 over HTTP (online (submit/poll)); in process 2085 us/step at B=1 (generate, MBU 0.914)

### #6 s0006: Attention runs two chunks at a time (8 warps each) - track A

- status: **valid**, http_step_us_b1 = 2524 (B=8: 6745), at 4.98 agent-hours, parent s0004, category attention
- hypothesis: track A: no format differs from the checkpoint. An attention block ran its chunks one after another with only ntc (<= 9 at these contexts) of 16 warps busy in the score phase and two dependent memory round trips per chunk; now warps 0-7 and 8-15 run two chunks concurrently (the per-chunk arithmetic is redefined for 8 warps and used at every batch size: batch-invariant, bit-identical to track B of this source), and the gate|up share model counts the paired passes. Batch 8 runs 4 passes instead of 8; batch 1 (one chunk per block) is unchanged. Same-card A/B (e0078, track B): B=8 generate -62, long context -61, step -63 us; B=1 -5. Expect http_step_us_b1 ~2535 (s0004 2535), B=8 ~6760 (6823).
- evaluator: valid: 2524 us/step at B=1 and 6745 at B=8 over HTTP (online (submit/poll)); in process 2512 us/step at B=1 (generate, MBU 0.786)

### #7 s0007: Attention runs two chunks at a time (8 warps each) - track B

- status: **valid**, http_step_us_b1 = 2103 (B=8: 5612), at 5.11 agent-hours, parent s0005, category attention
- hypothesis: track B; formats as s0005 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as s0006 on track A: two attention chunks run concurrently on warps 0-7 and 8-15 (per-chunk arithmetic redefined for 8 warps, the same at every batch size), the gate|up share model counts paired passes. Logits bit-identical to track A (bitcheck e0077), batch-invariant. Same-card A/B (e0078): B=8 generate -62, long -61, step -63 us; B=1 -5. Expect http_step_us_b1 ~2095 (s0005 2099), B=8 ~5590 (5646).
- evaluator: valid: 2103 us/step at B=1 and 5612 at B=8 over HTTP (online (submit/poll)); in process 2104 us/step at B=1 (generate, MBU 0.909)

### #8 s0008: Share model: attention cost per K tile 12 at batch > 1 - track A

- status: **valid**, http_step_us_b1 = 2534 (B=8: 6753), at 6.21 agent-hours, parent s0006, category scheduling
- hypothesis: track A: no format differs from the checkpoint. The gate|up share model's per-K-tile attention cost for batch > 1 goes from 5 to 12 (a scheduling constant): with paired chunks the attention blocks still finish last at long contexts. Same-card A/B (e0080, track B): B=8 long context (800-992) -97 us, short (64-320) +19, B=1 unchanged (constant only used at batch > 1). Expect http_step_us_b1 ~2524 (s0006), B=8 ~6710 (6745: the HTTP workload runs contexts 64-1088).
- evaluator: valid: 2534 us/step at B=1 and 6753 at B=8 over HTTP (online (submit/poll)); in process 2523 us/step at B=1 (generate, MBU 0.783)

### #9 s0009: Share model: attention cost per K tile 12 at batch > 1 - track B

- status: **valid**, http_step_us_b1 = 2109 (B=8: 5587), at 6.31 agent-hours, parent s0007, category scheduling
- hypothesis: track B; formats as s0007 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as on track A: the share model's per-K-tile attention cost at batch > 1 is 12 instead of 5. Logits unchanged (a schedule constant; bit-identical to track A). Same-card A/B (e0080): B=8 long -97, short +19; B=1 unchanged. Expect http_step_us_b1 ~2100 (s0007 2103), B=8 ~5575 (5612).
- evaluator: valid: 2109 us/step at B=1 and 5587 at B=8 over HTTP (online (submit/poll)); in process 2095 us/step at B=1 (generate, MBU 0.912)

### #10 s0010: Batch>1 kernels: 8 head loads in flight per thread - track A

- status: **valid**, http_step_us_b1 = 2526 (B=8: 6688), at 6.71 agent-hours, parent s0008, category gemm
- hypothesis: track A: no format differs from the checkpoint. The head pass of the batch > 1 kernel instantiations keeps 8 instead of 4 loads in flight per thread (SKQB, a template parameter of the head stage; the batch-1 kernel keeps 4: +4..+8 us there). Same-card A/B (e0100, track A): B=8 generate -48, long -58, step -50 us; B=1 -2. Expect http_step_us_b1 ~2530 (s0008 2534, unaffected), B=8 ~6700 (6753).
- evaluator: valid: 2526 us/step at B=1 and 6688 at B=8 over HTTP (online (submit/poll)); in process 2523 us/step at B=1 (generate, MBU 0.783)

### #11 s0011: Batch>1 kernels: 8 head-sketch loads in flight per thread - track B

- status: **valid**, http_step_us_b1 = 2099 (B=8: 5533), at 6.82 agent-hours, parent s0009, category gemm
- hypothesis: track B; formats as s0009 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as on track A: the head pass (sketch pass 1 on the greedy path) of the batch > 1 kernels keeps 8 instead of 4 loads in flight per thread; the batch-1 kernel is unchanged. Logits bit-identical to track A (bitcheck e0099). Same-card A/B (e0098): B=8 generate -28, long -53, step -26 us; B=1 -1. Expect http_step_us_b1 ~2105 (unaffected), B=8 ~5550 (5587).
- evaluator: valid: 2099 us/step at B=1 and 5533 at B=8 over HTTP (online (submit/poll)); in process 2094 us/step at B=1 (generate, MBU 0.913)

### #12 s0012: Batch>1 kernels launch with 16 KB less shared memory (larger L1) - track A

- status: **valid**, http_step_us_b1 = 2526 (B=8: 6569), at 7.19 agent-hours, parent s0010, category memory
- hypothesis: track A: no format differs from the checkpoint. A 16 KB shared-memory area (escape x values) is dead since the escape rework; it moves to the end of the layout and only the batch-1 kernel keeps it allocated (with less than ~200 KB per block the SMs stream unevenly, which costs the batch-1 critical path ~17 us), while the batch > 1 kernels launch without it: the shared-memory carve-out drops a step and L1 grows from ~28 to ~60 KB for stage C's x operands. Same-card A/B (e0106, track A): B=8 generate -91, long -135, step -94 us; B=1 +4..+7 (noise). Expect http_step_us_b1 ~2526 (s0010), B=8 ~6590 (6688).
- evaluator: valid: 2526 us/step at B=1 and 6569 at B=8 over HTTP (online (submit/poll)); in process 2523 us/step at B=1 (generate, MBU 0.783)

### #13 s0013: Batch>1 kernels launch with 16 KB less shared memory (larger L1) - track B

- status: **valid**, http_step_us_b1 = 2107 (B=8: 5437), at 7.28 agent-hours, parent s0011, category memory
- hypothesis: track B; formats as s0011 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as on track A: the dead 16 KB shared-memory area stays allocated for the batch-1 kernel only; the batch > 1 kernels get a larger L1. Logits bit-identical to track A (bitcheck e0107). Same-card A/B (e0105): B=8 generate -110, long -116, step -110 us; B=1 +2 / -2. Expect http_step_us_b1 ~2100 (s0011 2099), B=8 ~5430 (5533).
- evaluator: valid: 2107 us/step at B=1 and 5437 at B=8 over HTTP (online (submit/poll)); in process 2107 us/step at B=1 (generate, MBU 0.907)

### #14 s0014: Gate|up share table cached across steps of a launch - track A

- status: **valid**, http_step_us_b1 = 2517 (B=8: 6567), at 7.56 agent-hours, parent s0012, category scheduling
- hypothesis: track A: no format differs from the checkpoint. Every block recomputed the gate|up share table (65 entries, loops over up to 131 block ranks) at the start of every step, on the critical path before layer 0; it depends on the step only through the K tiles per attention chunk, which change every 128 positions, so it is now kept in shared memory and recomputed only then. Serving and generate() run one launch per cohort, so the saving applies to every token there. Same-card A/B (e0111, track B): B=1 generate -7, long -7 us; B=8 +-5; step() 0 (one step per launch). Expect http_step_us_b1 ~2519 (s0012 2526), B=8 ~6569 (unchanged).
- evaluator: valid: 2517 us/step at B=1 and 6567 at B=8 over HTTP (online (submit/poll)); in process 2506 us/step at B=1 (generate, MBU 0.786)

### #15 s0015: Gate|up share table cached across steps of a launch - track B

- status: **valid**, http_step_us_b1 = 2104 (B=8: 5408), at 7.65 agent-hours, parent s0013, category scheduling
- hypothesis: track B; formats as s0013 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as on track A: the gate|up share table is recomputed only when the K tiles per attention chunk change (every 128 positions) instead of at every step. Logits unchanged (schedule only). Same-card A/B (e0111): B=1 generate -7, long -7 us; B=8 +-5. Expect http_step_us_b1 ~2097 (s0013 2107; the evaluator's B=1 spread is +-10), B=8 ~5437.
- evaluator: valid: 2104 us/step at B=1 and 5408 at B=8 over HTTP (online (submit/poll)); in process 2081 us/step at B=1 (generate, MBU 0.914)

### #16 s0016: RMSNorm loads of the residual bypass L1 - track A

- status: **valid**, http_step_us_b1 = 2517 (B=8: 6526), at 7.77 agent-hours, parent s0014, category memory
- hypothesis: track A: no format differs from the checkpoint. The RMSNorm at the start of every layer reads the residual x (written by other SMs' exchanges a moment before) with ld.global.cg (L2 only) instead of a default load that allocates L1 lines: with the larger L1 of the batch > 1 kernels the default path costs more. Same-card A/B (e0113, track B): B=8 generate -27, long -36, step -27 us; B=1 -4 / +1. Expect http_step_us_b1 ~2517 (s0014), B=8 ~6535 (6567).
- evaluator: valid: 2517 us/step at B=1 and 6526 at B=8 over HTTP (online (submit/poll)); in process 2504 us/step at B=1 (generate, MBU 0.786)

### #17 s0017: RMSNorm loads of the residual bypass L1 - track B

- status: **valid**, http_step_us_b1 = 2101 (B=8: 5392), at 7.86 agent-hours, parent s0015, category memory
- hypothesis: track B; formats as s0015 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change as on track A: the norm reads the residual with ld.global.cg. Logits unchanged (same values, another cache path). Same-card A/B (e0113): B=8 generate -27, long -36, step -27 us; B=1 neutral. Expect http_step_us_b1 ~2100 (s0015 2104), B=8 ~5375 (5408).
- evaluator: valid: 2101 us/step at B=1 and 5392 at B=8 over HTTP (online (submit/poll)); in process 2092 us/step at B=1 (generate, MBU 0.910)

### #18 s0018: Batch-1 L2 prefetch limited to the half tile and the first whole stage-A tile

- status: **valid**, http_step_us_b1 = 2076 (B=8: 5384), at 8.36 agent-hours, parent s0017, category memory
- hypothesis: track B; formats as s0017 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change: the batch-1 kernel's bulk L2 prefetch after the stage-C signal covers only the half tile and the first whole tile of the block's next-layer stage-A share (74 of 123 KB): the prefetch traffic of early blocks slows the last blocks' residual exchanges and the barrier's polls (C->A hop 1.0 us with the full prefetch, 0.5 without, trace e0117), and the second whole tile is read late in stage A anyway. Same-card A/B (e0118, e0119): B=1 generate -24, long -28, step -22 us against the full prefetch; router rows only +13, no half tile +23. B=8 unchanged (batch-1 kernel only); track A keeps no prefetch (+8..+15 there, e0120). Logits unchanged. Expect http_step_us_b1 ~2078 (s0017 2101), B=8 ~5392.
- evaluator: valid: 2076 us/step at B=1 and 5384 at B=8 over HTTP (online (submit/poll)); in process 2066 us/step at B=1 (generate, MBU 0.920)

### #19 s0019: Batch-1: down tiles of the first two experts to L2 while waiting at barrier B

- status: **valid**, http_step_us_b1 = 2051 (B=8: 5385), at 8.73 agent-hours, parent s0018, category memory
- hypothesis: track B; formats as s0018 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change: in the batch-1 kernel, after a block's o_proj tile and before it waits at the stage-B barrier, thread 0 bulk-prefetches into L2 the block's down tiles of its first two experts (37 KB); HBM is partly idle during that wait and stage C starts with those tiles as L2 hits. Same-card A/B (e0121): B=1 generate -23, long -19, step -26 us; one expert -17, four +11 (more prefetch slows the blocks still in stage B). B=8 unchanged (batch-1 kernel only). Logits unchanged. Expect http_step_us_b1 ~2055 (s0018 2076), B=8 ~5384.
- evaluator: valid: 2051 us/step at B=1 and 5385 at B=8 over HTTP (online (submit/poll)); in process 2042 us/step at B=1 (generate, MBU 0.931)

### #20 s0020: Batch-1: down tiles of the first two experts to L2 at barrier B - track A

- status: **valid**, http_step_us_b1 = 2506 (B=8: 6525), at 9.08 agent-hours, parent s0016, category memory
- hypothesis: track A: no format differs from the checkpoint. Kernel change as s0019 on track B: in the batch-1 kernel, after a block's o_proj tile and before it waits at the stage-B barrier, thread 0 bulk-prefetches into L2 the block's down tiles of its first two experts (49 KB in BF16). Same-card A/B (e0125, track A): B=1 generate -9, long -15, step -8 us; B=8 +-12 (code generation; the change is in the batch-1 kernel only). Expect http_step_us_b1 ~2507 (s0016 2517), B=8 ~6526.
- evaluator: valid: 2506 us/step at B=1 and 6525 at B=8 over HTTP (online (submit/poll)); in process 2495 us/step at B=1 (generate, MBU 0.789)

### #21 s0021: Batch-1: previous token's experts' gate|up tiles to L2 during the union wait

- status: **valid**, http_step_us_b1 = 2044 (B=8: 5392), at 9.17 agent-hours, parent s0019, category memory
- hypothesis: track B; formats as s0019 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change, using how this model routes: a token keeps on average 4.6 of the previous token's 8 experts per layer (prior run's routing study), so while thread 0 of a block waits for this layer's union, thread 32 bulk-prefetches into L2 the first gate|up tile of the block's share of the PREVIOUS step's union for this layer (from M->dyn[l]); tiles of reused experts become L2 hits for whichever block reads them. Prefetch only: logits unchanged, bit-identical to track A. Same-card A/B (e0126): B=1 generate -9, long -6, step -10 us (two tiles per block: +19). B=8 unchanged (batch-1 kernel). Expect http_step_us_b1 ~2042 (s0019 2051), B=8 ~5385.
- evaluator: valid: 2044 us/step at B=1 and 5392 at B=8 over HTTP (online (submit/poll)); in process 2026 us/step at B=1 (generate, MBU 0.938)

### #22 s0022: Barrier-wait L2 prefetches (stage-A tiles, first down tiles) also at batch > 1

- status: **valid**, http_step_us_b1 = 2040 (B=8: 5391), at 10.02 agent-hours, parent s0021, category memory
- hypothesis: track B; formats as s0021 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Change: the two barrier-wait bulk prefetches of the batch-1 kernel (after the stage-C signal: the half tile and first whole tile of the next stage-A share; after o_proj, before barrier B: the down tiles of the first two experts) now also run in the batch > 1 kernels (one mechanism: small prefetches while a block waits; the full-size versions had lost at batch 8). Same-card A/B (e0132): B=8 generate -46, long -15, step -48 us; B=1 -1..-3. Track A unchanged (neutral there, e0133). Logits unchanged. Expect http_step_us_b1 ~2044 (s0021), B=8 ~5350 (5392).
- evaluator: valid: 2040 us/step at B=1 and 5391 at B=8 over HTTP (online (submit/poll)); in process 2033 us/step at B=1 (generate, MBU 0.935)

### #23 s0023: Greedy head sketch skips the 439 tiles made only of copies of earlier rows

- status: **valid**, http_step_us_b1 = 2045 (B=8: 5320), at 10.70 agent-hours, parent s0022, category memory
- hypothesis: track B; formats as s0022 (12-bit weights with in-operand escapes, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Checkpoint fact (study/dup_tiles.py): 7,129 rows of the tied head are bit-for-bit copies of an earlier row, and 439 of the 16,384 head tiles (all in the vocabulary's tail) consist only of such rows; their logits equal the earlier rows' and the lowest index wins a tie, so they can never be the greedy token. The sketch pass of the greedy path now skips them (2.7% of its 541 MB); step() still computes every logit, so step logits are unchanged (bit-identical to track A); generate == step greedy at every batch size 1..8 (tools/batch_all.py). The tiles are found at load by exact row comparison. Same binary, skip on/off (e0146): B=1 generate -14/-14/+4, B=8 -10/-18/-2 us. Expect http_step_us_b1 ~2035 (s0022 2040), B=8 ~5380 (5391).
- evaluator: valid: 2045 us/step at B=1 and 5320 at B=8 over HTTP (online (submit/poll)); in process 2034 us/step at B=1 (generate, MBU 0.933)

### #24 s0024: Track A rebuild of the final source (partner of track B s0023)

- status: **valid**, http_step_us_b1 = 2501 (B=8: 6493), at 11.91 agent-hours, parent s0020, category other
- hypothesis: track A: no format differs from the checkpoint. The current source built with TRACK=A: s0020 plus code that is inert or neutral for BF16 (the previous-step expert prefetch during the union wait, -1..-3 us on track A in e0129; the duplicate-tile skip and the batch>1 prefetches are track-B only). Its logits are bit-identical to the track-B build of this source (bitcheck). Expect http_step_us_b1 ~2506 (s0020), B=8 ~6525.
- evaluator: valid: 2501 us/step at B=1 and 6493 at B=8 over HTTP (online (submit/poll)); in process 2490 us/step at B=1 (generate, MBU 0.791)

### #25 s0025: Final source, track B (no functional change from s0023)

- status: **valid**, http_step_us_b1 = 2050 (B=8: 5336), at 12.93 agent-hours, parent s0023, category cleanup
- hypothesis: track B; formats as s0023 (12-bit weights with escapes patched into the mma operands, layer-0 tables made by the kernel, int8 head sketch on the greedy path skipping the 439 duplicate-row tiles). No functional change: experiment switches that are off by default (prefetch variants, profiling probes), a different allocation order of the small buffers, the stats() label. Logits bit-identical to track A (bitcheck e0157); verification e0158/e0159. Expect http_step_us_b1 ~2042 +- 10 (s0022 2040, s0023 2045), B=8 ~5320-5390.
- evaluator: valid: 2050 us/step at B=1 and 5336 at B=8 over HTTP (online (submit/poll)); in process 2027 us/step at B=1 (generate, MBU 0.936)

### #26 s0026: Batch-1: barrier-B poll with the last o_proj quantum, attention delay 30 in the share model

- status: **valid**, http_step_us_b1 = 2038 (B=8: 5334), at 14.57 agent-hours, parent s0025, category scheduling
- hypothesis: track B; formats as s0025 (12-bit weights with escapes patched into the mma operands, layer-0 tables made by the kernel, int8 head sketch on the greedy path). Two schedule settings of the batch-1 kernel measured together (both only move when ordinary blocks reach stage C; logits unchanged): the stage-B barrier's poll goes out with the last o_proj quantum (EARLY_B, alone -3..-5 us, e0172), and the share model's base attention delay is 30 instead of 26 at batch 1 (alone -5..-8, e0175). Together (e0177, 4 rounds): B=1 generate -12, long -19, step -12 us; B=8 unaffected (batch-1 kernel only; at batch 8 the delay change had cost +9). Track A unchanged (+4..+6 there, e0178). Expect http_step_us_b1 ~2032 (s0022 2040, s0025 2050; evaluator spread +-10), B=8 ~5330.
- evaluator: valid: 2038 us/step at B=1 and 5334 at B=8 over HTTP (online (submit/poll)); in process 2028 us/step at B=1 (generate, MBU 0.936)

