# s12-q32-b200-a1: Qwen3-32B on GB200

Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 47 submissions so far, 47 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 8002
- 1.10x the fastest general engine: 10400
- sglang (general engine): 11440
- vllm (general engine): 12270

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Own persistent megakernel v1: split-K GEMM ops, tensor-core GQA attention

- status: **valid**, http_step_us_b1 = 1.149e+04 (B=8: 1.176e+04), at 0.52 agent-hours, category structure
- hypothesis: track A; formats as the checkpoint (BF16 weights re-laid out in MMA fragment order, BF16 activations, fp32 inside operators). New engine: one cooperative persistent kernel per pass/cohort, 5 grid-barrier ops per layer, weights streamed with ld.global.cs at ~7.6 TB/s inside ops, 192 KB L2 prefetch of the next op per block, device argmax + host-mapped token stream behind submit/poll. In process: 11.0 ms B=1, 11.2 ms B=8 at positions 0-127; expect http_step_us_b1 ~11250 (attention grows to pos 1088), B=8 ~11500; gates pass (own check vs HF bf16 median 0.015, bitwise batch-invariant).
- evaluator: valid: 11488 us/step at B=1 and 11763 at B=8 over HTTP (online (submit/poll)); in process 11113 us/step at B=1 (generate, MBU 0.714)

### #2 s0002: 512-thread blocks (128 regs), split attention, batch-class norm prep

- status: **valid**, http_step_us_b1 = 1.063e+04 (B=8: 1.148e+04), at 2.22 agent-hours, parent s0001, category structure
- hypothesis: track A; formats as the checkpoint (BF16 weights in MMA fragment order, BF16 activations, fp32 in operators). Changes since s0001: blocks of 16 warps x 128 registers with 8 weight loads in flight per thread (no spills; was 32 warps x 64 regs with spill-sensitive loops), attention spread over nblk/(8B) blocks per (seq, kv head) in fixed 64-position parts with a last-arriver combine (batch-invariant), norm prep loads only B columns. In process at pos 512 (tests/prof.py, card 1): B=1 11454 -> 10667 us, B=8 11712 -> 11508. Expect http_step_us_b1 ~10710 (s0001 11488), B=8 ~11560 (11763); gates as s0001 (bitwise batch-invariant).
- evaluator: valid: 10631 us/step at B=1 and 11480 at B=8 over HTTP (online (submit/poll)); in process 10566 us/step at B=1 (generate, MBU 0.749)

### #3 s0003: Attention waits on per-group q/k/v tile counters; o waits on unit counter

- status: **valid**, http_step_us_b1 = 1.059e+04 (B=8: 1.129e+04), at 3.10 agent-hours, parent s0002, category sync
- hypothesis: track A; formats as the checkpoint. Two grid barriers per layer replaced by dataflow counters: q/k/v epilogues count finished tiles per kv group (80 each) and attention blocks wait only for their group; attention units count themselves done and the o projection waits on that counter (it prefetches nothing extra). RMSNorm 1/rms from per-tile sums of squares written by the residual epilogues; norm prep spread evenly over threads. In process pos 512 (card 1): B=1 10667 -> 10604 us, B=8 11508 -> 11327. Expect http_step_us_b1 ~10570 (s0002 10631), B=8 ~11300 (11480); gates unchanged (bitwise batch-invariant, generate exact).
- evaluator: valid: 10586 us/step at B=1 and 11290 at B=8 over HTTP (online (submit/poll)); in process 10539 us/step at B=1 (generate, MBU 0.753)

### #4 s0004: Track B: exact 12-bit weight storage, bit-identical to track A

- status: **valid**, http_step_us_b1 = 9625 (B=8: 1.041e+04), at 3.45 agent-hours, parent s0003, category memory
- hypothesis: track B; formats differing from the checkpoint: every streamed matrix (all GEMMs and lm_head) in an exact 12-bit code (sign, 4-bit exponent offset in a per-row 16-exponent window, 7-bit mantissa; rows scaled by exact powers of two, factor applied in the epilogue; 3.33M out-of-window weights patched into the MMA operands from per-warp lists); embedding and activations as track A. 48.0 GB instead of 64.0 GB per step. Verified: all 31,983,534,080 weights decode exactly (MK_W12_VERIFY), logits bit-identical to track A of the same source (bitcheck 1152 positions, B=1 and B=8). In process pos 512: B=1 9642 us, B=8 10400 (track A same source 10506 / 11198). Expect http_step_us_b1 ~9625, B=8 ~10365; gates identical to the track-A build.
- evaluator: valid: 9625 us/step at B=1 and 10407 at B=8 over HTTP (online (submit/poll)); in process 9576 us/step at B=1 (generate, MBU 0.825)

### #5 s0005: Track A: work split in quads of 4 k-steps (same source as s0004)

- status: **valid**, http_step_us_b1 = 1.049e+04 (B=8: 1.118e+04), at 3.67 agent-hours, parent s0004, category gemm
- hypothesis: track A; formats as the checkpoint. The GEMM work is split over blocks and warps in quads of 4 k-steps instead of single k-steps (the partition track B needs for bit-identity; s0004 is this source's track-B build, bit-identical logits). Fewer, aligned split points: in process pos 512 B=1 10604 -> 10506 us, B=8 11327 -> 11198. Expect http_step_us_b1 ~10490 (s0003 10586), B=8 ~11160 (11290); numeric gates re-drawn (different summation grouping), expected within limits as s0004's (identical logits).
- evaluator: valid: 10489 us/step at B=1 and 11175 at B=8 over HTTP (online (submit/poll)); in process 10433 us/step at B=1 (generate, MBU 0.760)

### #6 s0006: Track A: RMSNorm applied per k-step in the stream at B>=3 (no norm prep)

- status: **valid**, http_step_us_b1 = 1.052e+04 (B=8: 1.099e+04), at 3.78 agent-hours, parent s0005, category gemm
- hypothesis: track A; formats as the checkpoint. At 3+ sequences the q/k/v, gate|up and lm_head GEMMs no longer normalise the whole residual per block before streaming (80 KB read + 2 block barriers, ~3.7 us per op at B=8): the residual is also kept in B-fragment layout by its producers, read per k-step like the o/down inputs, and bf16(w * bf16(x * r)) is formed right before the MMA (second product as mul.rn.bf16x2, the same correctly rounded result). Same arithmetic per element: B=1 (prep path) and B=8 (stream path) logits bitwise equal (tests/check.py). In process pos 512 B=8 11243 -> 11033 us; B=1 unchanged. Expect B=8 http ~10965 (s0005 11175), http_step_us_b1 ~10490.
- evaluator: valid: 10522 us/step at B=1 and 10994 at B=8 over HTTP (online (submit/poll)); in process 10467 us/step at B=1 (generate, MBU 0.755)

### #7 s0007: Cut tiles handed over with a sentinel slot instead of an atomic counter

- status: **valid**, http_step_us_b1 = 1.033e+04 (B=8: 1.083e+04), at 3.89 agent-hours, parent s0006, category sync
- hypothesis: track A; formats as the checkpoint. A 16-row tile split between two blocks was finished by whichever arrived second (atom.acq_rel + read of the other partial: two dependent round trips at every block's op end). Now the block holding the tile's start stores its partial into the tile's slot and moves on; the block holding the end polls the slot with strong loads until no component is the sentinel NaN, resets it and computes p0 + p1 as before (bit-identical logits). Ablation said the exchange cost ~245 us per step. In process pos 512: B=1 10546 -> 10383 us, B=8 11027 -> 10868. Determinism: 6 repetitions x B=8/1/3 x 300 steps identical. Expect http_step_us_b1 ~10360 (s0006 10522), B=8 ~10830 (10994).
- evaluator: valid: 10334 us/step at B=1 and 10833 at B=8 over HTTP (online (submit/poll)); in process 10287 us/step at B=1 (generate, MBU 0.768)

### #8 s0008: Track B build of s0007's source (12-bit storage + sentinel cut tiles)

- status: **valid**, http_step_us_b1 = 9467 (B=8: 1.028e+04), at 3.99 agent-hours, parent s0007, category sync
- hypothesis: track B; formats differing from the checkpoint: as s0004 (exact 12-bit storage of every streamed matrix, row power-of-two scaling, escapes patched into MMA operands; embedding, activations as track A). Source = s0007 (track A 10334 / 10833) built with engine/TRACK=B; since s0004 the shared skeleton gained producer sums of squares, sentinel cut-tile hand-over (-160 us in process) and per-track RMSNorm switch (track B keeps the norm prep). Logits bit-identical to the track-A build of this source (bitcheck before submission). Expect http_step_us_b1 ~9470 (s0004 9625), B=8 ~10260 (10407).
- evaluator: valid: 9467 us/step at B=1 and 10277 at B=8 over HTTP (online (submit/poll)); in process 9426 us/step at B=1 (generate, MBU 0.841)

### #9 s0009: B>=7 attention combine through DSMEM (clusters of 2); separate small/large-batch kernels

- status: **valid**, http_step_us_b1 = 1.032e+04 (B=8: 1.077e+04), at 4.46 agent-hours, parent s0008, category attention
- hypothesis: track A; formats as the checkpoint. At 7-8 sequences each (sequence, kv head) unit spans exactly 2 blocks; they now form a cluster: block 0 folds its parts' partial softmax states and hands (max, sum, accumulator) to block 1 through distributed shared memory, block 1 continues the fold (no global partials, no atomic; the fold is now sequential per part everywhere, so B=1 and B=8 stay bitwise equal). The 1-2-sequence path gets a kernel instance without this code and without the per-k-step norm (register allocation of its own). Same-job A/B against s0007's source (card 2): B=1 10358 -> 10337 us, B=8 10852 -> 10754. Expect http_step_us_b1 ~10310 (s0007 10334), B=8 ~10730 (10833).
- evaluator: valid: 10323 us/step at B=1 and 10768 at B=8 over HTTP (online (submit/poll)); in process 10269 us/step at B=1 (generate, MBU 0.771)

### #10 s0010: Track B build of s0009's source (12-bit storage, DSMEM combine, two kernels)

- status: **valid**, http_step_us_b1 = 9444 (B=8: 1.036e+04), at 5.23 agent-hours, parent s0009, category attention
- hypothesis: track B; formats differing from the checkpoint: as s0004/s0008 (exact 12-bit storage of every streamed matrix, row power-of-two scaling, escapes patched into the MMA operands; embedding and activations as track A). Source = s0009 (track A 10323 / 10768) built with engine/TRACK=B: since s0008 the shared skeleton gained the B>=7 DSMEM attention combine and separate small/large-batch kernels. Logits bit-identical to the track-A build of this source (bitcheck e0126 on this source: 1152 positions, B=1 and B=8). Expect http_step_us_b1 ~9450 (s0008 9467), B=8 ~10180 (10277).
- evaluator: valid: 9444 us/step at B=1 and 10356 at B=8 over HTTP (online (submit/poll)); in process 9406 us/step at B=1 (generate, MBU 0.842)

### #11 s0011: Track B: small-batch kernel instance at all batch sizes (B=8 regression of s0010)

- status: **valid**, http_step_us_b1 = 9453 (B=8: 1.027e+04), at 5.56 agent-hours, parent s0010, category structure
- hypothesis: track B; formats differing from the checkpoint: as s0004/s0008/s0010 (exact 12-bit storage, row power-of-two scaling, escapes patched into MMA operands). s0010's B=8 regressed (10356 vs s0008 10277): in the 12-bit build the large-batch kernel instance spills (172 B) and was slower at B=8 than the small instance (same job: 10325 vs 10257 us at pos 512). Track B now runs the small instance (global attention combine, norm prep) at every batch size; arithmetic unchanged: bitcheck vs this source's track A (which runs B=8 on the large instance) bit-identical at B=1 and B=8. Expect http_step_us_b1 ~9445, B=8 ~10240.
- evaluator: valid: 9453 us/step at B=1 and 10265 at B=8 over HTTP (online (submit/poll)); in process 9429 us/step at B=1 (generate, MBU 0.840)

### #12 s0012: Track B: escape patches through shared memory; last-warp prefetch issue

- status: **valid**, http_step_us_b1 = 9402 (B=8: 1.005e+04), at 6.31 agent-hours, parent s0011, category memory
- hypothesis: track B; formats differing from the checkpoint: as s0011 (exact 12-bit storage, row power-of-two scaling, escapes patched into the MMA operands; embedding/activations as track A). Two track-B-only changes: (1) a quad with out-of-window weights is decoded into a per-warp shared-memory buffer, each escape is one 16-bit store, then reloaded (was a 16-way register select chain per escape: ~60 instructions; layers 1-5 hold up to 234 escapes per warp share); (2) the post-stream L2 prefetch of the next GEMM is issued by the last warp instead of thread 0. Same job vs s0011's source at pos 512: B=1 ~9475 -> 9425 us, B=8 ~10241 -> 10091. Bitcheck vs this source's track A: bit-identical (B=1, B=8). Expect http_step_us_b1 ~9405 (s0011 9453), B=8 ~10120 (10265).
- evaluator: valid: 9402 us/step at B=1 and 10049 at B=8 over HTTP (online (submit/poll)); in process 9360 us/step at B=1 (generate, MBU 0.843)

### #13 s0013: Track A: three kernel instances (1-2, 3-6, 7-8 sequences) and per-warp argmax keys

- status: **valid**, http_step_us_b1 = 1.035e+04 (B=8: 1.077e+04), at 6.69 agent-hours, parent s0012, category structure
- hypothesis: track A; formats as the checkpoint. Since s0009: the 7-8-sequence path gets its own kernel instance without the global attention combine (every multi-part unit takes the DSMEM path there), the 3-6 path keeps the global combine; lm_head's argmax keys kept per warp in registers instead of shared 64-bit atomics. Arithmetic unchanged (binv B=2..8 vs B=1 bitwise to position 2039, stress 0 mismatches). Same job vs s0009-era source at pos 512: B=8 10777 -> 10762 us, B=1 unchanged. Expect http_step_us_b1 ~10325 (s0009 10323), B=8 ~10755 (10768): within noise of s0009 at B=1.
- evaluator: valid: 10351 us/step at B=1 and 10773 at B=8 over HTTP (online (submit/poll)); in process 10298 us/step at B=1 (generate, MBU 0.770)

### #14 s0014: Track A final source: cohort-tagged served tokens, 20 ms gather window

- status: **valid**, http_step_us_b1 = 1.035e+04 (B=8: 1.077e+04), at 7.30 agent-hours, parent s0013, category correctness
- hypothesis: track A; formats as the checkpoint. Kernel arithmetic and speed as s0013 (same-job A/B: B=1 10389 vs 10390, B=8 10757 vs 10762 us at pos 512); serving robustness only: tokens written to host-mapped memory carry a cohort tag (a late write of the previous cohort can no longer be taken for a token of the next one; the slot of a same-shape cohort's last token coincides), and a cohort starts 20 ms after the first request instead of 4 (insurance against staggered arrivals; costs only time to first token). Expect http_step_us_b1 ~10330 (s0009 10323, s0013 10351), B=8 ~10770.
- evaluator: valid: 10354 us/step at B=1 and 10766 at B=8 over HTTP (online (submit/poll)); in process 10300 us/step at B=1 (generate, MBU 0.769)

### #15 s0015: Track B final source: cohort-tagged served tokens, 20 ms gather window

- status: **valid**, http_step_us_b1 = 9402 (B=8: 1.007e+04), at 7.40 agent-hours, parent s0014, category correctness
- hypothesis: track B; formats differing from the checkpoint: as s0012 (exact 12-bit storage of every streamed matrix, row power-of-two scaling, escapes patched into the MMA operands through shared memory; embedding/activations as track A). Kernel as s0012 apart from cold-path changes (three kernel instances for track A, per-warp argmax keys, cohort tag on host-mapped tokens); same-job timing at pos 512 9422 / 10097 us (s0012's source 9425 / 10091). Serving: tokens tagged with the cohort, 20 ms gather window. Logits bit-identical to the track-A build (bitcheck). Expect http_step_us_b1 ~9405 (s0012 9402), B=8 ~10055 (10049).
- evaluator: valid: 9402 us/step at B=1 and 10071 at B=8 over HTTP (online (submit/poll)); in process 9359 us/step at B=1 (generate, MBU 0.846)

### #16 s0016: K/V of cached positions prefetched into L2 before the q/k/v wait (B<=2)

- status: **valid**, http_step_us_b1 = 1.03e+04 (B=8: 1.076e+04), at 8.57 agent-hours, parent s0014, category memory
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). At 1-2 sequences each attention block issues a bulk L2 prefetch of its parts' cached K/V before waiting for the q/k/v tiles, so the score and P x V rounds hit L2 instead of DRAM; same-job A/B vs s0014's source: B=1 -57 us at pos 576, B=8 unchanged (compiled out of the 3-8 instances). Expect ~10300 / ~10770 us over HTTP.
- evaluator: valid: 10299 us/step at B=1 and 10763 at B=8 over HTTP (online (submit/poll)); in process 10213 us/step at B=1 (generate, MBU 0.773)

### #17 s0017: B=8 attention in fewer rounds: one warp per part softmax, 3 K blocks per warp

- status: **valid**, http_step_us_b1 = 1.03e+04 (B=8: 1.074e+04), at 9.02 agent-hours, parent s0016, category attention
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The softmax statistics take one warp per 64-position part (4 lanes per head, 16 scores each) instead of one warp per (part, head) in rounds of 16, and at 7-8 sequences each warp scores 3 K blocks per round instead of 2: same-job B=8 -95 us at pos 576 (-125 at 1024), B=1 unchanged. Expect ~10300 / ~10670 us over HTTP.
- evaluator: valid: 10304 us/step at B=1 and 10742 at B=8 over HTTP (online (submit/poll)); in process 10224 us/step at B=1 (generate, MBU 0.772)

### #18 s0018: Attention combine folds two halves in parallel (cluster blocks, same split globally)

- status: **valid**, http_step_us_b1 = 1.034e+04 (B=8: 1.071e+04), at 10.32 agent-hours, parent s0017, category attention
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The combine folds parts [0, np/2) and [np/2, np) separately and merges; at 7-8 sequences both blocks of a cluster fold in parallel instead of block 1 after block 0 (B=1..8 stay bitwise equal). Workload A/B (tests/wl.sh) B=8 -25 us, B=1 unchanged: expect ~10305 / ~10720 over HTTP.
- evaluator: valid: 10344 us/step at B=1 and 10715 at B=8 over HTTP (online (submit/poll)); in process 10267 us/step at B=1 (generate, MBU 0.770)

### #19 s0019: Split grid barrier: first weight batch requested before the wait (1-2 sequences)

- status: **valid**, http_step_us_b1 = 1.023e+04 (B=8: 1.072e+04), at 10.56 agent-hours, parent s0018, category sync
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). In the 1-2 sequence kernel instance the grid barrier is split into arrive (end of an op) and wait (inside the next GEMM after each warp has requested its first 8 weight k-steps), and the o projection requests them before waiting for attention, so the first DRAM/L2 latency overlaps the barrier. Workload A/B: B=1 -118 us, B=8 unchanged (the 3-8 instances keep the old code: there it slowed every stream ~8%). Expect ~10225 / ~10715 over HTTP.
- evaluator: valid: 10230 us/step at B=1 and 10718 at B=8 over HTTP (online (submit/poll)); in process 10206 us/step at B=1 (generate, MBU 0.776)

### #20 s0020: Track B of s0019's source (attention rounds, two-half combine, split barrier)

- status: **valid**, http_step_us_b1 = 9301 (B=8: 9943), at 10.69 agent-hours, parent s0015, category sync
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V and softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The track-A changes since s0015 (K/V prefetch at B<=2, one-warp softmax, two-half combine) plus the split barrier with preloaded first quads for o and down at every batch size (track B runs the small instance everywhere): workload A/B B=1 -130, B=8 -170 vs the two-half-combine source. Expect ~9280 / ~9950 us over HTTP.
- evaluator: valid: 9301 us/step at B=1 and 9943 at B=8 over HTTP (online (submit/poll)); in process 9287 us/step at B=1 (generate, MBU 0.851)

### #21 s0021: Split grid barrier in every kernel instance (wait state kept in shared memory)

- status: **valid**, http_step_us_b1 = 1.028e+04 (B=8: 1.054e+04), at 11.33 agent-hours, parent s0019, category sync
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The split barrier with first weight batch requested before the wait now also in the 3-6 and 7-8 sequence instances; its prefetch range is kept in shared memory and the pending flag recomputed from the op list, so no state is carried in registers across ops (carrying it slowed every stream ~8% in those instances); the per-k-step norm GEMMs keep their original code. Workload A/B: B=8 -180 us, B=4 -130, B=1 +39. Expect ~10270 / ~10540 over HTTP (B=8 past 10629).
- evaluator: valid: 10284 us/step at B=1 and 10539 at B=8 over HTTP (online (submit/poll)); in process 10242 us/step at B=1 (generate, MBU 0.774)

### #22 s0022: Track B of s0021's source; 7-8 sequences on the cluster-combine kernel instance

- status: **valid**, http_step_us_b1 = 9336 (B=8: 9918), at 11.83 agent-hours, parent s0020, category attention
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). s0021's source (split barrier state in shared memory) plus track B at 7-8 sequences on the large instance (DSMEM two-half combine, one-warp softmax, 3 K blocks per round; no per-k-step norm), which no longer spills: workload B=8 9927 -> 9843. Expect ~9300 / ~9860 us over HTTP.
- evaluator: valid: 9336 us/step at B=1 and 9918 at B=8 over HTTP (online (submit/poll)); in process 9301 us/step at B=1 (generate, MBU 0.851)

### #23 s0023: First weight batch also requested across the RMSNorm prep (1-2 sequences)

- status: **valid**, http_step_us_b1 = 1.026e+04 (B=8: 1.054e+04), at 12.32 agent-hours, parent s0021, category sync
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). In the 1-2 sequence instance the norm-prep GEMMs (q/k/v, gate|up, lm_head) also request their first 8 weight k-steps per warp before the barrier wait and the prep, as o and down already do. Workload A/B: B=1 -30 us, B=8 unchanged. Expect ~10255 / ~10540 over HTTP.
- evaluator: valid: 10265 us/step at B=1 and 10535 at B=8 over HTTP (online (submit/poll)); in process 10226 us/step at B=1 (generate, MBU 0.775)

### #24 s0024: Track B: 3 quads in flight per warp in the norm GEMMs (inputs from smem just in time)

- status: **valid**, http_step_us_b1 = 9250 (B=8: 9780), at 13.29 agent-hours, parent s0022, category memory
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The 12-bit streams ran at 75-93% of their byte floor with one 3 KB batch per warp in flight; in the norm-prep GEMMs (q/k/v, gate|up, lm_head) the inputs are now read from shared memory right before each MMA, which frees the registers for 3 quads per batch (~50% more bytes in flight). Workload A/B: B=1 -105 us, B=8 -122. Expect ~9230 / ~9800 us over HTTP.
- evaluator: valid: 9250 us/step at B=1 and 9780 at B=8 over HTTP (online (submit/poll)); in process 9233 us/step at B=1 (generate, MBU 0.861)

### #25 s0025: Track B, B=1: o and down read their input column from smem, 3 quads in flight

- status: **valid**, http_step_us_b1 = 9197 (B=8: 9840), at 13.66 agent-hours, parent s0024, category memory
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). At one sequence the o and down projections copy their single input column (16 / 51 KB) into shared memory after the wait and read it just in time, so they too run 3 quads per batch in flight. Workload A/B: B=1 -52 us (B=8 -51 from the build, not this path). Expect ~9200 / ~9730 over HTTP.
- evaluator: valid: 9197 us/step at B=1 and 9840 at B=8 over HTTP (online (submit/poll)); in process 9142 us/step at B=1 (generate, MBU 0.863)

### #26 s0026: RMSNorm prep with packed bf16 arithmetic (same bits, half the instructions)

- status: **valid**, http_step_us_b1 = 9156 (B=8: 9714), at 14.16 agent-hours, parent s0025, category other
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The norm prep's normalize step (2.7 us of the 5 us prep at B=8) uses one bf16x2 convert and one bf16x2 multiply per pair instead of unpack/round/multiply/round in float (exact product, one rounding: identical bits; B=2..8 still bitwise equal to B=1). Workload A/B: B=8 -56 us, B=1 -30. Expect ~9170 / ~9785 over HTTP (B=8 HTTP noise ~100 us).
- evaluator: valid: 9156 us/step at B=1 and 9714 at B=8 over HTTP (online (submit/poll)); in process 9116 us/step at B=1 (generate, MBU 0.868)

### #27 s0027: Track A build of s0026's source (packed bf16 norm prep)

- status: **valid**, http_step_us_b1 = 1.025e+04 (B=8: 1.054e+04), at 14.45 agent-hours, parent s0023, category other
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The source in the tree after the track-B work; for track A it adds only the norm prep in packed bf16 arithmetic (same bits; the 12-bit loop changes are not in the track-A build). Workload A/B vs s0023's source: B=1 -9 us, B=8 equal. Expect ~10255 / ~10535 over HTTP: the final track-A engine measured.
- evaluator: valid: 10249 us/step at B=1 and 10537 at B=8 over HTTP (online (submit/poll)); in process 10215 us/step at B=1 (generate, MBU 0.774)

### #28 s0028: Attention combine split over all blocks of a unit (1-6 sequences)

- status: **valid**, http_step_us_b1 = 1.014e+04 (B=8: 1.053e+04), at 15.75 agent-hours, parent s0027, category attention
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). At 1-6 sequences the last block of a unit no longer folds all parts' partials alone (4.2 us at 10 parts: its reads are bandwidth-limited); the last to arrive releases a per-unit flag and every block of the unit folds 1024/nact of the outputs (same part order: bitwise unchanged), o waits for one signal per block. The 7-8 instance (cluster combine) is compiled unchanged. Workload A/B: B=1 -97 us, B=8 -8. Expect ~10150 / ~10530 over HTTP.
- evaluator: valid: 10141 us/step at B=1 and 10533 at B=8 over HTTP (online (submit/poll)); in process 10152 us/step at B=1 (generate, MBU 0.778)

### #29 s0029: Track B of s0028's source (attention combine split over a unit's blocks)

- status: **valid**, http_step_us_b1 = 9063 (B=8: 9703), at 15.84 agent-hours, parent s0026, category attention
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The global attention combine split over all blocks of the unit (s0028). Workload A/B vs s0026's source: B=1 -65 us, B=8 -44 (B=8 runs the cluster instance whose code is unchanged: allocation). Expect ~9090 / ~9670 over HTTP (B=8 HTTP noise ~100).
- evaluator: valid: 9063 us/step at B=1 and 9703 at B=8 over HTTP (online (submit/poll)); in process 9068 us/step at B=1 (generate, MBU 0.873)

### #30 s0030: Attention prep constants (norm weights, cos, sin) requested before the q/k/v wait

- status: **valid**, http_step_us_b1 = 1.013e+04 (B=8: 1.053e+04), at 16.36 agent-hours, parent s0028, category attention
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The attention prep's norm weights and RoPE tables do not depend on this step's q/k/v; requesting them before the tile-counter wait takes one L2 trip off the attention chain. Workload A/B: B=1 -20 us, B=8 neutral. Expect ~10120 / ~10530 over HTTP.
- evaluator: valid: 10131 us/step at B=1 and 10528 at B=8 over HTTP (online (submit/poll)); in process 10136 us/step at B=1 (generate, MBU 0.781)

### #31 s0031: Track B of the attention-prep source (s0030)

- status: **valid**, http_step_us_b1 = 9049 (B=8: 9683), at 16.45 agent-hours, parent s0029, category attention
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The attention prep's constant inputs requested before the q/k/v wait. Workload A/B: B=1 -15 us, B=8 neutral. Expect ~9050 / ~9700 over HTTP (B=8 HTTP noise ~100).
- evaluator: valid: 9049 us/step at B=1 and 9683 at B=8 over HTTP (online (submit/poll)); in process 9061 us/step at B=1 (generate, MBU 0.873)

### #32 s0032: Track B, 7-8 sequences: norm prep split over the cluster pair via DSMEM

- status: **valid**, http_step_us_b1 = 9035 (B=8: 9495), at 17.15 agent-hours, parent s0031, category memory
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). At 7-8 sequences (kernel launched in clusters of 2) each block of a pair reads and normalizes half of the residual (40 KB instead of 80 KB from L2, the prep's limit) and writes it into both blocks' shared memory over DSMEM, then a cluster barrier. Workload A/B: B=8 -150 us; B=1 unchanged (other instance). Expect ~9050 / ~9540 over HTTP (B=8 HTTP noise ~100).
- evaluator: valid: 9035 us/step at B=1 and 9495 at B=8 over HTTP (online (submit/poll)); in process 9063 us/step at B=1 (generate, MBU 0.875)

### #33 s0033: Track A of s0032's source; 7-8 instance's unused norm-prep path instantiated with the cluster prep

- status: **valid**, http_step_us_b1 = 1.012e+04 (B=8: 1.044e+04), at 17.47 agent-hours, parent s0030, category structure
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). s0032's source (track B's cluster-split norm prep) for track A, with the 7-8 instance's never-executed norm-prep GEMM instantiated with the cluster prep as well: the instance's register allocation spills 48 B instead of 96 and its streams run faster (an allocation effect, not a mechanism: the per-k-step norm path that runs at B=8 is unchanged). Same-job workload A/B: B=8 -85 us, B=1 unchanged. Expect ~10130 / ~10450 over HTTP.
- evaluator: valid: 10121 us/step at B=1 and 10440 at B=8 over HTTP (online (submit/poll)); in process 10137 us/step at B=1 (generate, MBU 0.782)

### #34 s0034: Track B: a quad's escapes patched by their lanes in parallel

- status: **valid**, http_step_us_b1 = 9016 (B=8: 9433), at 18.76 agent-hours, parent s0032, category memory
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). The escapes of a quad (one entry per lane in the warp's 32-entry window) are written into the patch buffer by their lanes at once instead of one by one by lane 0: layer 1 gate|up 88 -> 74 us, layers 2-4 gate|up / down -1..-5. Workload A/B: B=8 -48 us, B=1 about 0 (the other layers' ops moved +0.3 us with the build). Expect ~9035 / ~9450 over HTTP (B=8 HTTP noise ~100).
- evaluator: valid: 9016 us/step at B=1 and 9433 at B=8 over HTTP (online (submit/poll)); in process 9062 us/step at B=1 (generate, MBU 0.873)

### #35 s0035: Track B: at most 10 blocks per attention unit at B=1

- status: **valid**, http_step_us_b1 = 9025 (B=8: 9449), at 21.03 agent-hours, parent s0034, category attention
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). Runtime default only: an attention unit (sequence, kv head) spreads over at most 10 blocks instead of 19 at one sequence (packs the busy blocks; parts and the combine's split are unchanged, so results are identical). Workload A/B, two rounds: B=1 -27..-30 us; B=8 unaffected (2 blocks per unit). Expect ~8990 / ~9433 over HTTP.
- evaluator: valid: 9025 us/step at B=1 and 9449 at B=8 over HTTP (online (submit/poll)); in process 9003 us/step at B=1 (generate, MBU 0.879)

### #36 s0036: Track B: idle blocks prefetch 128 KB of o during attention (runtime default)

- status: **valid**, http_step_us_b1 = 8995 (B=8: 9439), at 21.57 agent-hours, parent s0035, category memory
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). Runtime default only: with at most 10 blocks per attention unit more blocks are idle during attention, and their L2 prefetch of o's first bytes (256 KB each) delays the attention's own loads; 128 KB each. Workload A/B, two rounds: B=1 -37 us, B=8 unchanged. Expect ~8990 / ~9440 over HTTP.
- evaluator: valid: 8995 us/step at B=1 and 9439 at B=8 over HTTP (online (submit/poll)); in process 8960 us/step at B=1 (generate, MBU 0.885)

### #37 s0037: Track A: next op's first-bytes prefetch issued by the last warp, not warp 0

- status: **valid**, http_step_us_b1 = 1.008e+04 (B=8: 1.04e+04), at 24.47 agent-hours, parent s0033, category sync
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). Ordered epilogue stamps show warp 0 is every GEMM block's critical warp (it finishes the block's first, cut tile) and reaches the hand-over 1.0 us after the stream ends, partly because thread 0 issues the next op's L2 prefetch first; issuing it from the last warp (as track B already does) moves that 0.5 us off the critical warp. Workload A/B, two rounds: B=1 -30 us, B=8 -32. Expect ~10090 / ~10410 over HTTP.
- evaluator: valid: 10083 us/step at B=1 and 10404 at B=8 over HTTP (online (submit/poll)); in process 10110 us/step at B=1 (generate, MBU 0.782)

### #38 s0038: GEMM block ranges and prefetch plans with 32-bit instead of 64-bit divisions

- status: **valid**, http_step_us_b1 = 8956 (B=8: 9390), at 24.78 agent-hours, parent s0036, category gemm
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). Every thread computed its block's range in each GEMM prologue, and the barrier's prefetch plan after each GEMM, with 64-bit divisions by the block count (an emulated sequence each), on the path between a barrier and the stream; all values fit 32 bits (the same partition, the same results). Workload A/B, two rounds: B=1 -60 us, B=8 -47. Expect ~8940 / ~9395 over HTTP.
- evaluator: valid: 8956 us/step at B=1 and 9390 at B=8 over HTTP (online (submit/poll)); in process 8915 us/step at B=1 (generate, MBU 0.885)

### #39 s0039: Track A of s0038's source (32-bit range divisions)

- status: **valid**, http_step_us_b1 = 1.006e+04 (B=8: 1.041e+04), at 25.02 agent-hours, parent s0037, category gemm
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). The GEMM block ranges and prefetch plans with 32-bit instead of 64-bit divisions (same values, same results; s0038 for track B). Workload A/B, two rounds: B=1 -25 us, B=8 unchanged. Expect ~10060 / ~10405 over HTTP.
- evaluator: valid: 10061 us/step at B=1 and 10407 at B=8 over HTTP (online (submit/poll)); in process 10074 us/step at B=1 (generate, MBU 0.785)

### #40 s0040: Attention combine arrival: release + poll a cumulative per-launch count (no atomic/flag)

- status: **valid**, http_step_us_b1 = 1.001e+04 (B=8: 1.041e+04), at 25.39 agent-hours, parent s0039, category attention
- hypothesis: track A (formats as the checkpoint: BF16 weights, BF16 K/V, float32 softmax). Ordered stamps: after P x V the split combine spends 1.8 us on arrival (the last arriver's atomic returns, then it releases a flag the others poll: two round trips for most blocks). Each block now releases one arrival on a per-launch-zeroed counter and polls it up to a closed-form cumulative target (one round trip); same arithmetic. Workload A/B: B=1 -38 us (B=8 uses the cluster combine: unchanged). Expect ~10025 / ~10405 over HTTP.
- evaluator: valid: 10012 us/step at B=1 and 10410 at B=8 over HTTP (online (submit/poll)); in process 10016 us/step at B=1 (generate, MBU 0.787)

### #41 s0041: Track B of s0040's source (counter-based combine arrival)

- status: **valid**, http_step_us_b1 = 8948 (B=8: 9404), at 25.54 agent-hours, parent s0038, category attention
- hypothesis: track B (every streamed matrix in the exact 12-bit per-row-window code, verified on all weights; activations, K/V, softmax as track A; bit-identical to track A of this source by tools/kit/bitcheck.py at B=1 and B=8). s0040's combine arrival (release + poll of a cumulative per-launch count instead of atomic + flag) in the track-B build; workload A/B B=1 -0..-15 us, B=8 unchanged (cluster path). Expect ~8945 / ~9390 over HTTP.
- evaluator: valid: 8948 us/step at B=1 and 9404 at B=8 over HTTP (online (submit/poll)); in process 8892 us/step at B=1 (generate, MBU 0.890)

### #42 s0042: Track A: norm weights staged into shared memory during the barrier wait (1-4 seq)

- status: **valid**, http_step_us_b1 = 9981 (B=8: 1.041e+04), at 28.26 agent-hours, parent s0040, category memory
- hypothesis: track A (BF16 weights as stored, formats as the checkpoint). Right after the barrier arrival before each norm-prep GEMM, the norm's 10 KB of weights are copied (cp.async) into an unused input column of shared memory, so the prep reads only the residual from L2 after the barrier (at B=1 half its bytes). Expect B=1 ~9980 (s0040 10012, -33 in the workload A/B), B=8 unchanged ~10410 (the 7-8 instance's SASS is identical).
- evaluator: valid: 9981 us/step at B=1 and 10408 at B=8 over HTTP (online (submit/poll)); in process 9998 us/step at B=1 (generate, MBU 0.793)

### #43 s0043: Track B of s0042's source (track-B SASS identical to s0041)

- status: **valid**, http_step_us_b1 = 8937 (B=8: 9383), at 28.36 agent-hours, parent s0041, category other
- hypothesis: track B (exact 12-bit weight storage + escapes, bit-identical to track A). s0042's source with engine/TRACK = B: the staging is compiled out of track B (GSTAGE 0), every kernel instance is SASS-identical to s0041, so expect s0041's figures within noise, ~8950 / ~9400; submitted so that engine/ is exactly a submitted version.
- evaluator: valid: 8937 us/step at B=1 and 9383 at B=8 over HTTP (online (submit/poll)); in process 8863 us/step at B=1 (generate, MBU 0.889)

### #44 s0044: Track A: per-k-step norm weights pre-ordered on the host, staged during barrier wait

- status: **valid**, http_step_us_b1 = 9974 (B=8: 1.028e+04), at 29.03 agent-hours, parent s0042, category memory
- hypothesis: track A (BF16 weights as stored; a fragment-order copy of each RMSNorm weight vector is an extra layout, values unchanged). At 3-8 sequences the per-k-step norm's prologue loaded its 10 KB of norm weights from L2 after the barrier and scattered them into fragment order; now the host keeps them in that order and they are copied (cp.async) into shared memory right after the barrier arrival. Expect B=8 ~10270 (s0042 10408, -136 in the workload A/B), B=1 unchanged ~9980.
- evaluator: valid: 9974 us/step at B=1 and 10276 at B=8 over HTTP (online (submit/poll)); in process 9988 us/step at B=1 (generate, MBU 0.793)

### #45 s0045: Track B of s0044's source (track-B SASS identical to s0041)

- status: **valid**, http_step_us_b1 = 8938 (B=8: 9373), at 29.13 agent-hours, parent s0043, category other
- hypothesis: track B (exact 12-bit weight storage + escapes, bit-identical to track A). s0044's source with TRACK = B: FSTG and GSTAGE are compiled out of track B (SASS identical to s0041 / s0043), the api fills an op word track B does not read. Expect s0041/s0043's figures within noise, ~8945 / ~9395.
- evaluator: valid: 8938 us/step at B=1 and 9373 at B=8 over HTTP (online (submit/poll)); in process 8897 us/step at B=1 (generate, MBU 0.889)

### #46 s0046: Track B: row factors copied with cp.async (no L2 stall of warps 0-1 at stream start)

- status: **valid**, http_step_us_b1 = 8911 (B=8: 9332), at 29.35 agent-hours, parent s0045, category gemm
- hypothesis: track B (exact 12-bit weight storage + escapes, bit-identical to track A). The GEMM prologue's row-factor load + shared store stalled warps 0-1 (warp 0 is the epilogue's critical warp) for an L2 round trip after every barrier; now cp.async, waited for at the stream end. Expect ~-50 at both sizes vs s0045 (8938 / 9373): ~8890 / ~9330.
- evaluator: valid: 8911 us/step at B=1 and 9332 at B=8 over HTTP (online (submit/poll)); in process 8882 us/step at B=1 (generate, MBU 0.890)

### #47 s0047: Track B: escape-list start and first window loaded independently before the wait

- status: **valid**, http_step_us_b1 = 8792 (B=8: 9238), at 29.61 agent-hours, parent s0046, category gemm
- hypothesis: track B (exact 12-bit weight storage + escapes, bit-identical to track A). Each warp's stream began with two dependent L2 round trips (escape list start, then its first 32-entry window) after the barrier; the host now stores every warp's first window next to the starts so both loads are independent and go out at the op start, before any wait. Expect ~-110 at both sizes vs s0046 (8911 / 9332): ~8800 / ~9230.
- evaluator: valid: 8792 us/step at B=1 and 9238 at B=8 over HTTP (online (submit/poll)); in process 8751 us/step at B=1 (generate, MBU 0.903)

