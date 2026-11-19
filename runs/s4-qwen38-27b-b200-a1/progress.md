# s4-qwen38-27b-b200-a1: Qwen3.8-27B on GB200

Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 50 submissions so far, 48 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 6406
- 1.10x the fastest general engine: 8657
- SGLang 0.5.23 (general engine): 9523
- vLLM 0.32.0 (general engine): 10110

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Megakernel v0: one launch per step, weights in MMA fragment order (unoptimized)

- status: **valid**, http_step_us_b1 = 1.699e+04 (B=8: 1.761e+04), at 1.99 agent-hours, category structure
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 activations and K/V, float32 recurrent state; RMSNorm's scalar is applied after the GEMM instead of before). First end-to-end submission to validate the gates and the HTTP path: in-process step time is 16.6 ms (per-op overheads not yet removed), so I expect valid gates (rel-L2 about 0.010 median, batch bitwise equal, generate and http tokens identical) and http_step_us_b1 about 16800, i.e. slower than both general engines.
- evaluator: valid: 16990 us/step at B=1 and 17606 at B=8 over HTTP (online (submit/poll)); in process 16837 us/step at B=1 (generate, MBU 0.382)

### #2 s0002: Kernel v3: no per-thread stack traffic, activations via L1, one stream loop per warp

- status: **valid**, http_step_us_b1 = 1.083e+04 (B=8: 1.406e+04), at 2.16 agent-hours, parent s0001, category memory
- hypothesis: track A (formats as the checkpoint; RMSNorm's scalar applied after the GEMM). Three measured causes of the 16.8 ms step are removed: (1) the op and context structs passed by reference lived in every thread's stack (5.2 us per op for an empty op, now 0.8); (2) the activation copy into shared memory per GEMM (7 us) is replaced by default (L1) loads from a buffer that no SM has read before in the launch, weights are read with ld.cg; (3) a warp streams its whole share in one un-unrolled loop instead of one call per tile piece, and cut tiles are combined through shared memory. In process the step is 10.67 ms at B=1 and 13.9 ms at B=8, so I expect http_step_us_b1 about 10850 and b8 about 14050: still slower than SGLang (9523 / 10231); the B=8 gap comes from the recurrent-state op (3 units per block) and is the next thing to fix.
- evaluator: valid: 10834 us/step at B=1 and 14059 at B=8 over HTTP (online (submit/poll)); in process 10786 us/step at B=1 (generate, MBU 0.596)

### #3 s0003: Exact multi-token passes: MTP-head drafts verified P positions at a time

- status: **error**, no score, at 2.49 agent-hours, parent s0002, category scheduling
- hypothesis: track B (no format differs from the checkpoint: BF16 weights and activations, float32 recurrent state; what changes is the decode loop). submit()/poll() draft P-1 tokens with the checkpoint's own mtp.* head (not part of the model's function) and run the full model on P positions in one pass (8 at B=1, 4 at B=8): the extra positions are extra MMA columns of the same weight stream, the recurrent state is committed afterwards by replaying the accepted rank-1 updates, and every position is computed with the arithmetic of a single step (multi-position logits are bit-identical to step() logits on 9 (B,P) settings; 1024-token decodes of the three timed seeds are identical to step-by-step greedy at B=1 and B=8). step() and generate() are unchanged (one token per pass, 10.2 ms). Measured in process on the client's seeds: B=1 7.7 tokens per pass on two of the three prompts (two-token loops) and 3.0 on the third (prose), 2158 / 2158 / 5401 us per token; B=8 2.2-2.4 tokens per pass, 8239-8644 us per step before the recurrent-op fix. I expect http_step_us_b1 about 2100 (the median repeat is a loop prompt; the prose repeat would be about 5400) and b8 about 7500, i.e. 4.5x and 1.35x SGLang; the http_tokens gate must stay 8/8 identical.
- evaluator: pod_lost: this job was running when pods [5, 6] died; it is not run a third time

### #4 s0004: Exact draft-and-verify: MTP head + context copy, up to 16 positions per pass

- status: **incorrect**, no score, at 2.68 agent-hours, parent s0003, category scheduling
- hypothesis: track B (no number format differs from the checkpoint: BF16 weights, activations and K/V, float32 recurrent state; the decode loop changes). submit()/poll() verify P positions per model pass, each computed with the arithmetic of a single step (bit-identical logits for P up to 16 measured), the delta-rule state committed afterwards by replaying the accepted rank-1 updates. Drafts: the checkpoint's own MTP head (mtp.*, at most 4 per pass), or, when the last 6 tokens occurred before and the continuation agrees with the MTP head's first draft, the continuation copied from the context, verified 16 at a time (B=1). step()/generate() unchanged (10.2 ms). In-process measurements on the client's seeds: B=1 14.4 tokens per pass and ~1350 us per token on the two looping prompts, 3.7 and ~4340 us on the prose prompt; B=8 7800-8100 us per step. The client's median of three repeats is a looping one, so I expect http_step_us_b1 ~1400 and b8 ~8000 (SGLang 9523 / 10231); the honest prose figure at B=1 is ~4400.
- evaluator: incorrect: failed numeric, batch
- gates that failed: batch, numeric

### #5 s0005: Draft-and-verify engine resubmitted with release/acquire grid synchronisation

- status: **valid**, http_step_us_b1 = 1145 (B=8: 7372), at 2.95 agent-hours, parent s0004, category correctness
- hypothesis: track B (formats as the checkpoint; the decode loop changes). Same engine as s0004, whose check stage failed for single sequences at a few positions while every reproduction of that flow on my side was bit-exact; the only identified defect is the missing device-scope ordering in the grid barrier and in the cut-tile counter, now release/acquire. If that was the cause, all gates pass (rel-L2 ~0.010, single == batched bitwise); expected http_step_us_b1 ~1400 (median repeat is a looping prompt; prose ~4500) and b8 ~8000 us. If single sequences fail again, the cause is elsewhere and data-dependent.
- evaluator: valid: 1145 us/step at B=1 and 7372 at B=8 over HTTP (online (submit/poll)); in process 10459 us/step at B=1 (generate, MBU 0.614)

### #6 s0006: Coalesced RMSNorm prologue and 32-column loop order (faster multi-column passes)

- status: **valid**, http_step_us_b1 = 1100 (B=8: 6502), at 3.48 agent-hours, parent s0005, category gemm
- hypothesis: track B (formats as the checkpoint; the decode loop changes). Two measured fixes for passes with many columns: the RMSNorm scalar r was recomputed in every block with one strided 4-byte load per lane and column (9.4 us per op at 16 columns, 18.6 at 32): now warp w reads tiles w, w+32, ... for all columns coalesced and warp c sums column c with the same butterfly, bit-identical r; and the 32-column stream loop goes k-step by k-step (in_proj 48 -> 44 us). B=1 single steps unchanged (10.39 ms). In process on the client's seeds B=8 went from 7119/7302 to 6272/6430 us per step, so I expect b8 ~6500 (s0005: 7372) and b1 unchanged ~1150.
- evaluator: valid: 1100 us/step at B=1 and 6502 at B=8 over HTTP (online (submit/poll)); in process 10470 us/step at B=1 (generate, MBU 0.614)

### #7 s0007: 32-position verify passes, L2 prefetch of the next matrix, barrier-free recurrence

- status: **valid**, http_step_us_b1 = 823.5 (B=8: 6460), at 3.89 agent-hours, parent s0006, category scheduling
- hypothesis: track B (formats as the checkpoint; the decode loop changes). Three changes, all bit-identical between single steps and multi-position passes: (1) a single sequence verifies up to 32 positions per pass while its drafts are copied from the context (was 16); (2) at every grid barrier each block asks L2 (cp.async.bulk.prefetch) for the start of its share of the next weight matrix (128 KB, 1 MB before small ops): single step 10.44 -> 10.23 ms; (3) the gated delta-rule recurrence keeps a column in 8 lanes of a warp, so a pass's positions need no block barrier (P=16: 37.7 -> 27 us per layer), compiled for 8 or 32 positions. In process on the timed seeds: B=1 827 / 826 / 5247 us per token (loop, loop, prose), B=8 6461 / 6480 / 6988. Expect http b1 ~870 (s0006: 1100), b8 ~6550 (unchanged within noise).
- evaluator: valid: 824 us/step at B=1 and 6460 at B=8 over HTTP (online (submit/poll)); in process 10271 us/step at B=1 (generate, MBU 0.626)

### #8 s0008: Per-sequence verify depth in a batch: copyable sequences get more positions

- status: **valid**, http_step_us_b1 = 843.2 (B=8: 5738), at 4.19 agent-hours, parent s0007, category scheduling
- hypothesis: track B (formats as the checkpoint; the decode loop changes). In a batch, the verify pass no longer gives every sequence the same number of positions: within 32 columns, a sequence whose drafts were copied from its context gets up to 8 positions, the others 2-4 (MTP drafts); the layout is a per-pass table of column starts used by every op, and the draft buffer is double-buffered because layouts change between passes. Every decode remains token-identical to step-by-step greedy (1024 tokens x 8 sequences on the three timed seed sets). Also: the head's argmax is folded into the head GEMM with a 64-bit atomicMax (18.4 -> 1.9 us per head). Per-sequence acceptance at B=8 went from 3.95 (loops) / 2.4 (prose) tokens per pass to 5.7-7.3 / 2.0-2.9; in process 5629 / 5667 / 5835 us per step against 6368 / 6386 / 6891. Expect b8 ~5800 (s0007: 6460) and b1 unchanged ~830.
- evaluator: valid: 843 us/step at B=1 and 5738 at B=8 over HTTP (online (submit/poll)); in process 10272 us/step at B=1 (generate, MBU 0.625)

### #9 s0009: Track-A engine: everything since s0002 that keeps the checkpoint's formats and data flow

- status: **valid**, http_step_us_b1 = 1.034e+04 (B=8: 1.203e+04), at 4.48 agent-hours, parent s0002, category structure
- hypothesis: track A (the checkpoint's BF16 weights, BF16 activations and K/V, float32 recurrent state; one token per model pass, no drafts). Same engine source as track B with the build switch engine/TRACK = A: submit()/poll() decode one token per pass. Changes since s0002 that apply here: barrier-free gated delta-rule recurrence with register-resident state, RMSNorm scalar computed coalesced and published without a block barrier, L2 prefetch of the next matrix at every barrier, argmax folded into the head GEMM, release/acquire grid barrier. In process a step is 10.1-10.2 ms at B=1 and 11.7-11.9 at B=8 (s0002: 10.79 / 13.95). Expect http b1 ~10300 and b8 ~11950 - still behind SGLang (9523 / 10231): the post-stream tail of every GEMM (~1.9 ms per step) is not solved.
- evaluator: valid: 10337 us/step at B=1 and 12028 at B=8 over HTTP (online (submit/poll)); in process 10268 us/step at B=1 (generate, MBU 0.626)

### #10 s0010: Attention per column group, recurrence split over 2 SMs per head, B=8 columns to loops

- status: **valid**, http_step_us_b1 = 621.6 (B=8: 4496), at 5.98 agent-hours, parent s0009, category scheduling
- hypothesis: track B: checkpoint BF16 weights and formats unchanged; exact draft-and-verify as s0007/s0008 (MTP-head and context-copy drafts, every verified position computed with single-step arithmetic). Kernel scheduling only, bit-identical per position: attention score/value-sum/out work units now include the query column group (160-192 -> ~4000 warps at 32 positions), the recurrent op batches its input loads, keeps the state in its own function and, for one sequence and >8 positions, runs each value head on 2 blocks (norm sums exchanged in the same order); chain launches skip when no sequence needs that draft; B=8 gives copyable sequences up to 16 columns, prose 1. In process the 32-position pass fell 23.6 -> 16.5 ms and the B=8 client figure 5.6-5.8 -> 4.45-4.62 ms; I expect http_step_us_b1 ~640 (from 824) and B=8 ~4600 (from 5738); gates unchanged (rel-L2 0.010, top-1 99%).
- evaluator: valid: 622 us/step at B=1 and 4496 at B=8 over HTTP (online (submit/poll)); in process 10103 us/step at B=1 (generate, MBU 0.636)

### #11 s0011: Track-A build of the s0010 kernel plus faster 32-column tails

- status: **valid**, http_step_us_b1 = 1.027e+04 (B=8: 1.18e+04), at 6.46 agent-hours, parent s0010, category scheduling
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Same kernel as s0010 plus finalize/epilogue load batching and per-kind epilogue instances; these mostly change multi-column passes, so at one column per sequence I expect the s0009 numbers within noise: http_step_us_b1 ~10300 (s0009 10337), B=8 ~11950 (s0009 12028, the recurrent op's state prefetch for 3 units per block may save ~50 us).
- evaluator: valid: 10267 us/step at B=1 and 11795 at B=8 over HTTP (online (submit/poll)); in process 10171 us/step at B=1 (generate, MBU 0.631)

### #12 s0012: Shorter GEMM tails at 32 columns: batched finalize loads, per-kind epilogues, free warp

- status: **valid**, http_step_us_b1 = 582.3 (B=8: 4145), at 6.78 agent-hours, parent s0011, category sync
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0010. Only the post-stream tail of every GEMM changes, bit-identically: cut-tile partials of all column groups requested at once, residual epilogue without per-column-group round trips or spill reloads (one instance per epilogue kind and use), float floor division with exact integer correction, and the block's outgoing cut tile finalized by a free warp instead of behind warp 31's own epilogue. In process the 32-position pass fell 16.1 -> 15.3 ms and B=8x4 15.4 -> 14.7 ms; test_spec gives 582 us (loops) and 4102-4270 us (B=8). I expect http_step_us_b1 ~585 (s0010 622) and B=8 ~4200 (s0010 4496).
- evaluator: valid: 582 us/step at B=1 and 4145 at B=8 over HTTP (online (submit/poll)); in process 10127 us/step at B=1 (generate, MBU 0.634)

### #13 s0013: Track-A build of the s0012 kernel

- status: **valid**, http_step_us_b1 = 1.018e+04 (B=8: 1.172e+04), at 6.88 agent-hours, parent s0012, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Same kernel as s0012; at one column per sequence only the free-warp finalize of the outgoing cut tile matters (-0.9 % B=1, -0.5 % B=8 per step in process), so I expect http_step_us_b1 ~10180 (s0011 10267) and B=8 ~11740 (s0011 11795).
- evaluator: valid: 10184 us/step at B=1 and 11724 at B=8 over HTTP (online (submit/poll)); in process 10127 us/step at B=1 (generate, MBU 0.634)

### #14 s0014: No replay after full acceptance; spill-free split recurrence; 4-token copy match

- status: **valid**, http_step_us_b1 = 512.4 (B=8: 3847), at 7.63 agent-hours, parent s0013, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0012. (1) each verify pass leaves its final recurrent state in a second buffer, taken over when every position was accepted instead of replaying 32 rank-1 updates per head and layer (same values: the replay repeats the same fused multiply-adds); (2) the split recurrence's idle half leaves at the block barrier, which removed ~700 bytes of spills from the per-position loop (16.7 vs 25.6 us per layer); (3) copied drafts after a 4-token match instead of 6 (one ramp pass less on looping text); (4) 32-column activations read from L2. In process the looping B=1 prompts went 582 -> 513 us per token and B=8 4102-4270 -> 3831-3992 us; I expect http_step_us_b1 ~515 (s0012 582) and B=8 ~3900 (s0012 4145).
- evaluator: valid: 512 us/step at B=1 and 3847 at B=8 over HTTP (online (submit/poll)); in process 10110 us/step at B=1 (generate, MBU 0.635)

### #15 s0015: Track-A build of the s0014 kernel

- status: **valid**, http_step_us_b1 = 1.017e+04 (B=8: 1.172e+04), at 7.74 agent-hours, parent s0014, category scheduling
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Same kernel as s0014; at one column per sequence the changes since s0013 (recurrent-op restructure, 32-column activation loads) are nearly neutral in process (B=1 -0.2 %, B=8 -0.8 %), so I expect http_step_us_b1 ~10170 (s0013 10184) and B=8 ~11650 (s0013 11724).
- evaluator: valid: 10175 us/step at B=1 and 11718 at B=8 over HTTP (online (submit/poll)); in process 10111 us/step at B=1 (generate, MBU 0.635)

### #16 s0016: Non-blocking GEMM prologues: cp.async residual rows, barrier-free RMSNorm scalar

- status: **valid**, http_step_us_b1 = 505.5 (B=8: 3762), at 9.02 agent-hours, parent s0015, category sync
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0014. Only GEMM prologues change, bit-identically: the residual rows are copied to shared memory with cp.async (no thread waits for its load before streaming) and, above 4 columns, the RMSNorm scalar no longer takes two block barriers (partials in shared memory, one warp finishes the reduction in warp_sum's order when the first epilogue needs it). Same-card A/B: 32-column passes -1.8 %; test_spec 506 us (loops) and 3744-3908 us (B=8). I expect http_step_us_b1 ~505 (s0014 512) and B=8 ~3760 (s0014 3847).
- evaluator: valid: 506 us/step at B=1 and 3762 at B=8 over HTTP (online (submit/poll)); in process 10109 us/step at B=1 (generate, MBU 0.635)

### #17 s0017: Track-A build of the s0016 kernel

- status: **valid**, http_step_us_b1 = 1.019e+04 (B=8: 1.147e+04), at 9.12 agent-hours, parent s0016, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Same kernel as s0016: at 8 columns (B=8) the RMSNorm prologue now avoids two block barriers (in process -1.8 % per B=8 step), at 1 column nothing changes; I expect http_step_us_b1 ~10175 (s0015 10175) and B=8 ~11520 (s0015 11718).
- evaluator: valid: 10195 us/step at B=1 and 11473 at B=8 over HTTP (online (submit/poll)); in process 10109 us/step at B=1 (generate, MBU 0.635)

### #18 s0018: Attention output folded into value sum; recurrent-op inputs 4 channels per thread

- status: **valid**, http_step_us_b1 = 487.9 (B=8: 3695), at 10.67 agent-hours, parent s0017, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0016. Two bit-identical kernel changes: the attention output (sum of value-sum chunks, gate) is done by the last chunk unit to arrive (one op and barrier less per attention layer), and passes of 4+ positions load the recurrent op's inputs 4 channels per thread with the q/k norm finished inside the warp (same butterfly order). Same-card A/B: 32-position pass -4.2 %, B=8 x 4 -2 %; test_spec 487 us (loops), 3672-3828 us (B=8). I expect http_step_us_b1 ~488 (s0016 506) and B=8 ~3700 (s0016 3762).
- evaluator: valid: 488 us/step at B=1 and 3695 at B=8 over HTTP (online (submit/poll)); in process 10103 us/step at B=1 (generate, MBU 0.636)

### #19 s0019: Track-A build of the s0018 kernel

- status: **valid**, http_step_us_b1 = 1.015e+04 (B=8: 1.144e+04), at 10.78 agent-hours, parent s0018, category scheduling
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Same kernel as s0018: at one position per sequence only the attention-output fold applies (-0.8 % B=1, -0.6 % B=8 per step in process; the recurrent-op input change is not used below 4 positions), so I expect http_step_us_b1 ~10120 (s0017 10195) and B=8 ~11420 (s0017 11473).
- evaluator: valid: 10150 us/step at B=1 and 11435 at B=8 over HTTP (online (submit/poll)); in process 10108 us/step at B=1 (generate, MBU 0.636)

### #20 s0020: First 8 verify passes one at a time: copied drafts take effect one pass earlier

- status: **valid**, http_step_us_b1 = 478.6 (B=8: 3690), at 11.21 agent-hours, parent s0019, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; same kernel and exact draft-and-verify as s0018. Host scheduling only: a pass's length is fixed when the previous pass is launched, so with two passes in flight a copyable continuation found by pass n first lengthens pass n+3; the first 8 passes of a cohort are now queued one at a time (n+2), costing ~0.2 ms of host latency per pass while it lasts. test_spec: looping prompts 487 -> 478 us (38 passes instead of 39), prose and B=8 unchanged. I expect http_step_us_b1 ~479 (s0018 488) and B=8 ~3690.
- evaluator: valid: 479 us/step at B=1 and 3690 at B=8 over HTTP (online (submit/poll)); in process 10106 us/step at B=1 (generate, MBU 0.636)

### #21 s0021: Verify passes skip the MTP head while copied drafts keep being accepted

- status: **valid**, http_step_us_b1 = 468.3 (B=8: 3626), at 11.66 agent-hours, parent s0020, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; same kernel arithmetic, exact draft-and-verify. Drafting policy only: when no sequence needs MTP drafts for the next pass (its copied drafts were all accepted and the copy drafter still matches, or it verifies a single position), the pass runs a program without the MTP head (0.33 ms of a 13.9 ms 32-position pass) and copies are taken without the MTP agreement check; any rejection brings the MTP head back. Every token is still the model's own greedy token. test_spec: looping B=1 478 -> 468 us, B=8 3604-3779 (from 3668-3820), prose unchanged. I expect http_step_us_b1 ~468 (s0020 479) and B=8 ~3630 (s0020 3690).
- evaluator: valid: 468 us/step at B=1 and 3626 at B=8 over HTTP (online (submit/poll)); in process 10090 us/step at B=1 (generate, MBU 0.635)

### #22 s0022: B=1 ramp: pass length chosen at launch from the previous pass's copy result

- status: **valid**, http_step_us_b1 = 457.6 (B=8: 3625), at 11.92 agent-hours, parent s0021, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; same kernel, exact draft-and-verify. Host scheduling only: while passes run one at a time (the first 8), every pass leaves room for 31 copied drafts and the next pass's length is chosen at its launch from the previous pass's result, instead of one pass earlier; the ramp on looping text shrinks from 5 to 4 short passes. test_spec: looping B=1 468 -> 457 us, prose and B=8 unchanged. I expect http_step_us_b1 ~458 (s0021 468), B=8 ~3625.
- evaluator: valid: 458 us/step at B=1 and 3625 at B=8 over HTTP (online (submit/poll)); in process 10108 us/step at B=1 (generate, MBU 0.636)

### #23 s0023: Track A: two recurrent units per block at one position (B=8 step -4 %)

- status: **valid**, http_step_us_b1 = 1.015e+04 (B=8: 1.099e+04), at 12.39 agent-hours, parent s0022, category scheduling
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state), one token per step, no speculation. Kernel change since s0019: at one position with more recurrent units than blocks (B >= 4) each half of a block runs its own (sequence, value head) unit, 384 units in 2 rounds instead of 3, every column's arithmetic unchanged (single == batched bitwise); plus the attention-output fold and cached positions. In process B=8 step 11.43 -> 10.97 ms, B=1 ~-1 %. I expect http_step_us_b1 ~10100 (s0019 10150) and B=8 ~10950 (s0019 11435).
- evaluator: valid: 10154 us/step at B=1 and 10988 at B=8 over HTTP (online (submit/poll)); in process 10078 us/step at B=1 (generate, MBU 0.637)

### #24 s0024: B=8 recurrent units balanced across blocks; replay deltas via shared memory

- status: **valid**, http_step_us_b1 = 457.2 (B=8: 3604), at 13.02 agent-hours, parent s0023, category scheduling
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0022. Small bit-identical kernel/scheduling changes since s0022: recurrent units of a B=8 verify pass placed longest-first on blocks by the host, replay deltas written from shared memory after the recurrence, sequence positions cached per launch, and (single steps only) two recurrent units per block. test_spec: B=1 loops 457-458 us (unchanged), B=8 3569-3733 us (s0022's run 3604-3779). I expect http_step_us_b1 ~458 (s0022 458) and B=8 ~3590 (s0022 3625).
- evaluator: valid: 457 us/step at B=1 and 3604 at B=8 over HTTP (online (submit/poll)); in process 10081 us/step at B=1 (generate, MBU 0.637)

### #25 s0025: Serving fixes for 2-7 sequences (token buffers, unit dispatch); same speed

- status: **valid**, http_step_us_b1 = 458 (B=8: 3604), at 15.46 agent-hours, parent s0024, category correctness
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0024. Correctness fixes for batch sizes the evaluator does not use: per-pass token buffers sized for 32 columns in per-sequence mode (B=2-7 delivered too few tokens), every block enters the recurrent op when the unit table is used (B<4 skipped units). B=1 and B=8 paths are unchanged, so I expect http_step_us_b1 ~457 and B=8 ~3600 (s0024 457 / 3604); online identity now holds at B=1,2,3,5,7,8.
- evaluator: valid: 458 us/step at B=1 and 3604 at B=8 over HTTP (online (submit/poll)); in process 10083 us/step at B=1 (generate, MBU 0.637)

### #26 s0026: Softmax with four warps per row

- status: **valid**, http_step_us_b1 = 456.8 (B=8: 3626), at 15.75 agent-hours, parent s0025, category attention
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0025. The attention softmax uses four warps per (head, position) row with partial max and sum combined in a fixed order (one batch of loads per pass); the order is the same at every batch and pass size, so verification stays exact. Same-card A/B -0.5 % per 32-position pass. I expect http_step_us_b1 ~455 (s0025 458) and B=8 ~3600; gates as before (rel-L2 ~0.010, top-1 ~99 %).
- evaluator: valid: 457 us/step at B=1 and 3626 at B=8 over HTTP (online (submit/poll)); in process 10104 us/step at B=1 (step, MBU 0.636)

### #27 s0027: Track-A build of the s0026 kernel

- status: **valid**, http_step_us_b1 = 1.014e+04 (B=8: 1.092e+04), at 15.85 agent-hours, parent s0026, category attention
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. Same kernel as s0026; since s0023 only the four-warp softmax affects single steps (-0.4 % B=1, -0.3 % B=8 in process). I expect http_step_us_b1 ~10115 (s0023 10154) and B=8 ~10955 (s0023 10988).
- evaluator: valid: 10139 us/step at B=1 and 10925 at B=8 over HTTP (online (submit/poll)); in process 10102 us/step at B=1 (step, MBU 0.636)

### #28 s0028: Prompt prefill in chunks (first token 0.68 s -> 0.09 s); same decode

- status: **valid**, http_step_us_b1 = 456.7 (B=8: 3624), at 16.27 agent-hours, parent s0027, category structure
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0026. The prompt is processed in chunks of up to 32 (B=1) / 4 (B=8) positions per pass instead of one (multi-position passes are bit-identical to single steps), so the time to the first token drops from ~0.68 s to ~0.09 s; the decode is unchanged and the client's decode window excludes it, so I expect http_step_us_b1 ~457 and B=8 ~3620 (s0026 457 / 3626) with http_tokens still 8/8 identical.
- evaluator: valid: 457 us/step at B=1 and 3624 at B=8 over HTTP (online (submit/poll)); in process 10108 us/step at B=1 (step, MBU 0.636)

### #29 s0029: Track-A build with chunked prompt prefill

- status: **valid**, http_step_us_b1 = 1.015e+04 (B=8: 1.092e+04), at 16.36 agent-hours, parent s0028, category structure
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per decode step, no speculation; the prompt is processed in chunks of up to 32 / 4 positions per pass as general engines do (bit-identical to one position at a time). Kernel as s0027; the decode is unchanged, so I expect http_step_us_b1 ~10140 and B=8 ~10925 (s0027 10139 / 10925) with the time to the first token ~0.09 / 0.22 s instead of ~0.67 s.
- evaluator: valid: 10146 us/step at B=1 and 10923 at B=8 over HTTP (online (submit/poll)); in process 10098 us/step at B=1 (generate, MBU 0.636)

### #30 s0030: Cut tiles handed over with sentinels (one memory trip instead of three)

- status: **valid**, http_step_us_b1 = 450.5 (B=8: 3573), at 17.03 agent-hours, parent s0029, category sync
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0028. A tile cut between two blocks is now handed over with a sentinel slot (the first block stores its partial, the last block polls the slot with relaxed gpu-scope loads, adds in block order and restores the sentinel) instead of store + acq_rel counter + load; same additions, bit-identical. Same-card A/B -1.4 % per 32-position pass, -1.9 % per B=8 step; test_spec loops 450 us, B=8 3522-3681. I expect http_step_us_b1 ~450 (s0028 457) and B=8 ~3570 (s0028 3624).
- evaluator: valid: 450 us/step at B=1 and 3573 at B=8 over HTTP (online (submit/poll)); in process 9948 us/step at B=1 (generate, MBU 0.646)

### #31 s0031: Track-A build of the s0030 kernel (sentinel hand-over of cut tiles)

- status: **valid**, http_step_us_b1 = 9981 (B=8: 1.071e+04), at 17.13 agent-hours, parent s0030, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per decode step, no speculation. Same kernel as s0030: cut tiles handed over through sentinel slots with relaxed gpu-scope accesses (one memory trip instead of three, same additions); in process -1.6 % per B=1 step and -1.9 % per B=8 step, so I expect http_step_us_b1 ~9985 (s0029 10146) and B=8 ~10715 (s0029 10923).
- evaluator: valid: 9981 us/step at B=1 and 10713 at B=8 over HTTP (online (submit/poll)); in process 9946 us/step at B=1 (generate, MBU 0.646)

### #32 s0032: Split recurrent op: norm sums exchanged through sentinel slots

- status: **valid**, http_step_us_b1 = 448.5 (B=8: 3571), at 17.37 agent-hours, parent s0031, category sync
- hypothesis: track B: checkpoint BF16 formats unchanged; exact draft-and-verify as s0030. The two blocks of a split head (one sequence, > 8 positions) exchange their 32-column norm sums through sentinel slots polled with relaxed gpu-scope loads instead of release/acquire flags plus loads (one memory trip instead of three); same sums in the same order. Same-card A/B -0.4 % per 32-position pass; test_spec loops 448 us. I expect http_step_us_b1 ~448 (s0030 450), B=8 unchanged ~3570.
- evaluator: valid: 449 us/step at B=1 and 3571 at B=8 over HTTP (online (submit/poll)); in process 9954 us/step at B=1 (step, MBU 0.646)

### #33 s0033: Track-A build of the s0032 kernel

- status: **valid**, http_step_us_b1 = 9968 (B=8: 1.073e+04), at 17.55 agent-hours, parent s0032, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per decode step, no speculation. Same source as s0032; the change since s0031 (norm-sum sentinels of the split recurrent op) is only used by passes of more than 8 positions, so I expect s0031's numbers within noise: http_step_us_b1 ~9980, B=8 ~10715.
- evaluator: valid: 9968 us/step at B=1 and 10728 at B=8 over HTTP (online (submit/poll)); in process 9938 us/step at B=1 (generate, MBU 0.646)

### #34 s0034: Track A: no far L2 prefetch before small ops (only 128 KB before each GEMM)

- status: **valid**, http_step_us_b1 = 9874 (B=8: 1.062e+04), at 18.20 agent-hours, parent s0033, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. At the barriers before a recurrent or attention op every block asked L2 for up to 1 MB of the next GEMM (152 MB in flight, more than L2 holds, competing with the recurrent state reads and delaying barrier releases); now only the barrier right before a GEMM prefetches its first 128 KB. Same-card sweep: B=1 step 9968 -> 9878 us, B=8 step 11334 -> 11157 us, bit-identical arithmetic. Expect HTTP ~9880 at B=1 (s0033: 9968) and ~10570 at B=8 (s0033: 10728).
- evaluator: valid: 9874 us/step at B=1 and 10622 at B=8 over HTTP (online (submit/poll)); in process 9829 us/step at B=1 (generate, MBU 0.653)

### #35 s0035: Track A: GEMM hand-over without 64-bit divisions after the stream

- status: **valid**, http_step_us_b1 = 9502 (B=8: 1.029e+04), at 18.81 agent-hours, parent s0034, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. Cycle stamps (tools/ho_stamps.py) showed the warps that hand over a cut tile spent 1.5-2.2 us after the block sync on integer work (64-bit floor divisions rebuilding the partition, a serial search) before the first memory access, and the receiver waited for the sender's late store. Now every warp writes its quad boundary to shared memory before streaming; the first warp of a tile is one ballot; a tile not longer than a block share spans two blocks without a division. Same partition, same additions: bitwise-identical logits (bits_cmp, test_multi, test_spec). Same card: B=1 step 9863 -> 9495 us, B=8 11165 -> 10810. Expect HTTP ~9490 at B=1 (s0034: 9874; SGLang 9523) and ~10290 at B=8 (s0034: 10622).
- evaluator: valid: 9502 us/step at B=1 and 10291 at B=8 over HTTP (online (submit/poll)); in process 9453 us/step at B=1 (generate, MBU 0.678)

### #36 s0036: Track C build of the s0035 kernel (division-free hand-over, no far prefetch)

- status: **valid**, http_step_us_b1 = 432.3 (B=8: 3437), at 18.89 agent-hours, parent s0035, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0032. Kernel changes since s0032, both bit-identical: no far L2 prefetch before small ops, and the cut-tile hand-over after the stream without 64-bit divisions (quad boundaries in shared memory, one ballot). Same card: B=1 x 32 pass 13192 -> 12700 us, B=8 x 4 13572 -> 13150; test_spec 432 us per token on the looping prompt (was 448). Expect HTTP ~435 at B=1 (s0032: 449) and ~3450 at B=8 (s0032: 3571).
- evaluator: valid: 432 us/step at B=1 and 3437 at B=8 over HTTP (online (submit/poll)); in process 9473 us/step at B=1 (generate, MBU 0.678)

### #37 s0037: Track A: attention units spread over blocks; recurrent op without a start load

- status: **valid**, http_step_us_b1 = 9380 (B=8: 1.02e+04), at 19.21 agent-hours, parent s0036, category scheduling
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. (1) The attention ops gave units to consecutive warps, so at B=1 the 152 key tiles sat on 5 SMs and the 32 prep units on one; units now go round-robin over blocks first (prep 133 -> 72 us, score 112 -> 69 us per step). (2) Every recurrent unit started with a dependent global load of its state-buffer index; it and the replay count are written only by a pass's last op and are now read once per launch into shared memory (B=8 recurrent op 1218 -> 1145 us). Bitwise identical (bits_cmp, test_multi, test_spec). Same card: B=1 step 9491 -> 9364 us, B=8 10805 -> 10647. Expect HTTP ~9375 at B=1 (s0035: 9502) and ~10140 at B=8 (s0035: 10291; SGLang 10231).
- evaluator: valid: 9380 us/step at B=1 and 10199 at B=8 over HTTP (online (submit/poll)); in process 9364 us/step at B=1 (generate, MBU 0.685)

### #38 s0038: Track A: GEMM finalize roles at each warp's stream end; leaner prologue

- status: **valid**, http_step_us_b1 = 9281 (B=8: 1.006e+04), at 19.92 agent-hours, parent s0037, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. After a GEMM's stream all 32 warps used to work out at once (divisions, tests) which tile each finalizes; each warp now decides at its own stream end while others still stream, and after the block sync reads one word. In the prologue only the threads that copy residual rows compute the block's tile range, and a warp's two quad bounds come from one pass of the division. Bitwise-identical logits (bits_cmp, test_multi). Same card: B=1 step 9371 -> 9255 us, B=8 10632 -> 10510. Expect HTTP ~9270 at B=1 (s0037: 9380) and ~10085 at B=8 (s0037: 10199).
- evaluator: valid: 9281 us/step at B=1 and 10065 at B=8 over HTTP (online (submit/poll)); in process 9238 us/step at B=1 (generate, MBU 0.696)

### #39 s0039: Track C build of the s0038 kernel

- status: **valid**, http_step_us_b1 = 425.9 (B=8: 3404), at 20.00 agent-hours, parent s0038, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0036. Kernel changes since s0036, all bit-identical: attention units round-robin over blocks, recurrent-op buffer index and replay count cached per launch, GEMM finalize roles at each warp's stream end, leaner GEMM prologue. test_spec: loops 428 us per token (s0036's kernel 432). Expect HTTP ~425 at B=1 (s0036: 432) and ~3380 at B=8 (s0036: 3437).
- evaluator: valid: 426 us/step at B=1 and 3404 at B=8 over HTTP (online (submit/poll)); in process 9236 us/step at B=1 (generate, MBU 0.696)

### #40 s0040: Track A: softmax in registers, early gates, split recurrence at B=1

- status: **valid**, http_step_us_b1 = 9258 (B=8: 1.007e+04), at 20.43 agent-hours, parent s0039, category attention
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. (1) softmax rows of at most 1024 keys keep their scores in registers for the max, sum and write passes (two L2 trips less); (2) the attention output fold loads its gates before the partial sums; (3) a single step of one sequence runs the recurrent op on 2 blocks per head (the split already used for long passes; 48 latency-bound units -> 96). All bit-identical (bits_cmp, test_multi, test_spec). Same card: B=1 step 9255 -> 9205 us, B=8 10500 -> 10457. Expect HTTP ~9230 at B=1 (s0038: 9281) and ~10025 at B=8 (s0038: 10065).
- evaluator: valid: 9258 us/step at B=1 and 10065 at B=8 over HTTP (online (submit/poll)); in process 9212 us/step at B=1 (generate, MBU 0.698)

### #41 s0041: Track C build of the s0040 kernel

- status: **valid**, http_step_us_b1 = 425.8 (B=8: 3393), at 20.62 agent-hours, parent s0040, category attention
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0039. Kernel changes since s0039, all bit-identical: softmax scores in registers up to 1024 keys, gates first in the attention output fold, split recurrent op for single steps of one sequence (ramp passes). test_spec: loops 424-425 us per token, B=8 3340-3490. Expect HTTP ~421 at B=1 (s0039: 426) and ~3380 at B=8 (s0039: 3404).
- evaluator: valid: 426 us/step at B=1 and 3393 at B=8 over HTTP (online (submit/poll)); in process 9213 us/step at B=1 (generate, MBU 0.697)

### #42 s0042: Track A: no local-memory reloads after op calls in the main loop

- status: **valid**, http_step_us_b1 = 9006 (B=8: 9900), at 21.48 agent-hours, parent s0041, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. The kernel's main loop kept the program pointer, op count, op index and thread-index values across each op's function call; the compiler saved them to local memory and reloaded them after the call, right before the grid barrier, where L1 no longer holds them after a weight stream (an L2 trip per op). They are now re-read from shared memory / special registers (SASS: local loads in the loop body 28 -> 8, none after a call). Bitwise-identical (bits_cmp vs HEAD, test_multi, test_spec, test_engine). Same card: B=1 step 9234 -> 8981 us, B=8 10506 -> 10243. Expect HTTP ~9005 at B=1 (s0040: 9258) and ~9815 at B=8 (s0040: 10065).
- evaluator: valid: 9006 us/step at B=1 and 9900 at B=8 over HTTP (online (submit/poll)); in process 8989 us/step at B=1 (generate, MBU 0.715)

### #43 s0043: Track C build of the s0042 kernel

- status: **valid**, http_step_us_b1 = 414.1 (B=8: 3275), at 21.63 agent-hours, parent s0042, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0041. Kernel change since s0041, bit-identical: the main loop no longer reloads its state from local memory after each op's call (B=1 x 32 pass -2.6 %). test_spec: loops 413-414 us per token (s0041's kernel 424). Expect HTTP ~415 at B=1 (s0041: 426) and ~3310 at B=8 (s0041: 3393).
- evaluator: valid: 414 us/step at B=1 and 3275 at B=8 over HTTP (online (submit/poll)); in process 8989 us/step at B=1 (generate, MBU 0.715)

### #44 s0044: Track A: debug timers compiled out of the default build

- status: **valid**, http_step_us_b1 = 8965 (B=8: 9833), at 22.65 agent-hours, parent s0043, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. Same kernel as s0042; the default build now compiles out the debug timers (per-op, per-block and per-warp stamp checks on the barrier and post-stream paths of every op). Bitwise-identical (bits_cmp vs HEAD, test_multi, test_spec, test_engine). Same card, CUDA-event timing: B=1 step 8996 -> 8958 us, B=8 10241 -> 10217. Expect HTTP ~8970 at B=1 (s0042: 9006) and ~9875 at B=8 (s0042: 9900).
- evaluator: valid: 8965 us/step at B=1 and 9833 at B=8 over HTTP (online (submit/poll)); in process 8936 us/step at B=1 (generate, MBU 0.719)

### #45 s0045: Track C build of the s0044 kernel (debug timers compiled out)

- status: **valid**, http_step_us_b1 = 414.8 (B=8: 3300), at 22.74 agent-hours, parent s0044, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0043. Only change: the default build compiles out the debug timers (bit-identical; B=1 step -0.4 %, B=8 -0.25 %). Expect HTTP ~412 at B=1 (s0043: 414) and ~3265 at B=8 (s0043: 3275).
- evaluator: valid: 415 us/step at B=1 and 3300 at B=8 over HTTP (online (submit/poll)); in process 8935 us/step at B=1 (generate, MBU 0.719)

### #46 s0046: Track A: remaining debug stamp checks compiled out

- status: **valid**, http_step_us_b1 = 8947 (B=8: 9819), at 22.95 agent-hours, parent s0045, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. Same kernel as s0044; the default build also compiles out the stamp checks inside the post-sync epilogue and the cut-tile hand-over. Bitwise-identical (bits_cmp vs HEAD, test_multi, test_spec, plain_check). CUDA-event timing on one card: B=1 step 8957 -> 8945 us, B=8 10220 -> 10201. Expect HTTP ~8955 at B=1 (s0044: 8965) and ~9815 at B=8 (s0044: 9833), within the evaluator's noise.
- evaluator: valid: 8947 us/step at B=1 and 9819 at B=8 over HTTP (online (submit/poll)); in process 8923 us/step at B=1 (generate, MBU 0.720)

### #47 s0047: Track C build of the s0046 kernel (final)

- status: **valid**, http_step_us_b1 = 416.5 (B=8: 3310), at 23.04 agent-hours, parent s0046, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0045. Only change since s0045: the remaining debug stamp checks compiled out (bit-identical, -0.1 to -0.2 % per pass). Expect HTTP ~413 at B=1 (s0043/s0045: 414/415) and ~3290 at B=8 (3275/3300).
- evaluator: valid: 417 us/step at B=1 and 3310 at B=8 over HTTP (online (submit/poll)); in process 8925 us/step at B=1 (generate, MBU 0.720)

### #48 s0048: Track A: GEMM debug-flag checks compiled out

- status: **valid**, http_step_us_b1 = 8938 (B=8: 9800), at 23.27 agent-hours, parent s0047, category sync
- hypothesis: track A: checkpoint formats and data flow (BF16 weights in MMA fragment order, BF16 activations, float32 recurrent state and softmax), one token per step, no speculation. Same kernel as s0046; the default build also compiles out the GEMM debug flags (skip-stream / stream-only modes of the timing tools) checked in every stream's prologue, tile end and post-stream path. Bitwise-identical (bits_cmp vs HEAD, test_multi, test_spec, plain_check). CUDA events: B=1 step 8945 -> 8923 us, B=8 10199 -> 10185. Expect HTTP ~8925 at B=1 (s0046: 8947) and ~9805 at B=8 (s0046: 9819).
- evaluator: valid: 8938 us/step at B=1 and 9800 at B=8 over HTTP (online (submit/poll)); in process 8911 us/step at B=1 (step, MBU 0.721)

### #49 s0049: Track C build of the s0048 kernel (final)

- status: **valid**, http_step_us_b1 = 414.6 (B=8: 3296), at 23.36 agent-hours, parent s0048, category sync
- hypothesis: track C: exact draft-and-verify (MTP head and context-copy drafts, verify passes bit-identical to stepping; every emitted token is the model's own greedy token) on checkpoint BF16 formats, as s0047. Only change since s0047: the GEMM debug-flag checks compiled out (bit-identical, -0.1 to -0.25 % per pass). Expect HTTP ~414 at B=1 and ~3290 at B=8 (s0043 414 / 3275, s0047 417 / 3310), within the evaluation noise.
- evaluator: valid: 415 us/step at B=1 and 3296 at B=8 over HTTP (online (submit/poll)); in process 8911 us/step at B=1 (generate, MBU 0.721)

### #50 s0050: Track B: lossless 12-bit weights (W12) on the s0048 track-A engine

- status: **valid**, http_step_us_b1 = 1.069e+04 (B=8: 1.127e+04), at 23.60 agent-hours, parent s0049, category memory
- hypothesis: track B: one token per sequence per model pass, the s0048 track-A engine and data flow, with the checkpoint's BF16 weights stored as an exact 12-bit code (15-exponent window per tensor, out-of-window weights patched exactly; decoded weights are bit-identical BF16, so the arithmetic is the track-A arithmetic up to the stream's partition). What changed since W12 was measured slower (hour ~5): the kernel around the stream (hand-over, barriers, local-memory reloads) is ~13 % faster, but the W12 decode loop is the same s0007-style loop with per-quad patch handling. Same card, CUDA events: B=1 step 8924 us (BF16) vs 10694 us (W12), B=8 10186 vs 11577 - I expect W12 to stay ~20 % slower: HTTP ~10720 at B=1 and ~11150 at B=8 (s0048 BF16: 8938 / 9800). Checks: reference rel-L2 0.0101 / 0.0296, top-1 99.4 %, single == batched bitwise, generate == step greedy, test_multi 10/10. Submitted as the run's first track-B measurement, as the operator asked, not as an improvement.
- evaluator: valid: 10687 us/step at B=1 and 11270 at B=8 over HTTP (online (submit/poll)); in process 10660 us/step at B=1 (step, MBU 0.603)

