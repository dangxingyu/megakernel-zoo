# s9-qwen38-27b-b200-f1: Qwen3.8-27B on GB200

Score: `http_step_us_b1` (lower is better). Budget 8.0 agent-hours. 23 submissions so far, 23 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 6406
- target on track A: 1.10x the fastest general engine: 8720
- SGLang 0.5.23 (general engine): 9594
- vLLM 0.32.0 (general engine): 10233
- inherited engine, track A (run s5, s0016): 8886
- inherited engine, track B (run s5, s0017, exact 12-bit weights): 8117

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Inherited engine (s5 s0016) with the stats() track label fixed

- status: **valid**, http_step_us_b1 = 8885 (B=8: 9836), at 0.25 agent-hours, category cleanup
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; one token per pass). No kernel change: the inherited s0016 source with stats() reporting the project's track name (A; B for the W12 build; C for draft-and-verify). Baseline of this run on this pod: expect the operator's rematch figures, 8886 us at B=1 and 9843 us at B=8 over HTTP, within 0.5 %.
- evaluator: valid: 8885 us/step at B=1 and 9836 at B=8 over HTTP (online (submit/poll)); in process 8849 us/step at B=1 (generate, MBU 0.726)

### #2 s0002: Weight-stream loads use the cache-streaming operator (ld.global.cs)

- status: **valid**, http_step_us_b1 = 8701 (B=8: 9453), at 0.97 agent-hours, parent s0001, category memory
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0001: the 16-byte loads of the GEMM weight streams use ld.global.cs (cache streaming, likely accessed once) instead of ld.global.cg. Weight lines then stop displacing the step's other data (activations, state, descriptors) from L2. Logits bit-identical to s0001 (hash of 6 steps at B=1, 3, 8). Same-card events: B=1 8878 -> 8658 us, B=8 9785 -> 9452. Expect ~8675 us at B=1 and ~9480 at B=8 over HTTP.
- evaluator: valid: 8701 us/step at B=1 and 9453 at B=8 over HTTP (online (submit/poll)); in process 8664 us/step at B=1 (generate, MBU 0.741)

### #3 s0003: B=8 recurrent op in one pass per block (three units, inputs loaded once)

- status: **valid**, http_step_us_b1 = 8694 (B=8: 9222), at 1.39 agent-hours, parent s0002, category scheduling
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state stored every step; no format differs from the checkpoint's). One change vs s0002, for one-token passes of 7-8 sequences only: the gated-delta-rule op no longer runs its 384 units in two rounds of two per block (each round with its own input phase and four half-block barriers, ~3 us of latency per round and layer) but in one pass: all units' inputs loaded in one phase, a third unit per block shared by both halves (one column pass each), the state prefetch issued after the input loads. Same arithmetic per state column; logits bit-identical to s0002 at B=1..8 (hash of 20 steps each). It has its own kernel entry point so that the B=1 code is untouched. Same-card events: B=8 9376 -> 9244 us, B=1 8658 -> 8651. Expect ~8695 us at B=1 (unchanged) and ~9250 at B=8 over HTTP.
- evaluator: valid: 8694 us/step at B=1 and 9222 at B=8 over HTTP (online (submit/poll)); in process 8662 us/step at B=1 (generate, MBU 0.742)

### #4 s0004: Barrier-time prefetch re-tuned for cs loads: near 160 KB, far 64 KB

- status: **valid**, http_step_us_b1 = 8641 (B=8: 9145), at 1.73 agent-hours, parent s0003, category memory
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0003, host-side constants only: at the grid barrier before a GEMM each block asks L2 for the first 160 KB of its share of that matrix (was 128 KB), and for 64 KB at the barriers before the small ops that precede a GEMM (was 0). Both were tuned with ld.global.cg weight loads (where far > 0 lost); with cs loads prefetched lines survive longer. Same-card events, 4 x 4 sweep: B=1 8664 -> 8622 us, B=8 9221 -> 9140; logits bit-identical (hash at B=1..8). Expect ~8652 us at B=1 and ~9145 at B=8 over HTTP.
- evaluator: valid: 8641 us/step at B=1 and 9145 at B=8 over HTTP (online (submit/poll)); in process 8613 us/step at B=1 (generate, MBU 0.746)

### #5 s0005: Served decode: tokens written by the kernel into host-mapped memory

- status: **valid**, http_step_us_b1 = 8628 (B=8: 9134), at 1.93 agent-hours, parent s0004, category sync
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0004, in the served path (submit/poll) only: the kernel's last op writes each pass's chosen tokens straight into page-locked host memory mapped for the device (cudaHostAllocMapped), tagged per cohort, and poll() reads them there; the inherited path enqueued a device-to-host copy and an event between every two passes. Model passes, kernel and logits are unchanged (the tokens equal step-by-step greedy in tools/online_time.py). In-process serving, 512 tokens: 8639-8652 -> 8621-8622 us per step at B=1, 9142-9160 -> 9128-9153 at B=8. Expect ~8628 us at B=1 and ~9130 at B=8 over HTTP.
- evaluator: valid: 8628 us/step at B=1 and 9134 at B=8 over HTTP (online (submit/poll)); in process 8613 us/step at B=1 (generate, MBU 0.746)

### #6 s0006: Prefetch during small ops: consecutive pieces, idle blocks take their whole share

- status: **valid**, http_step_us_b1 = 8587 (B=8: 9114), at 2.18 agent-hours, parent s0005, category memory
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). Kernel source identical to s0005; one change in the host-made prefetch table (what each block asks L2 for at the grid barriers before the small ops that precede a GEMM): (a) the barriers before the recurrent op and the attention ops ask for consecutive 64 KB pieces of the block's share of the next matrix instead of the same first 64 KB again; (b) in one-token passes of 1-3 sequences, where the recurrent op leaves blocks without a unit (B=1: 56 of 152), the last 32 of those ask for their whole share of out_proj at the barrier before it. Tables are per batch size. Logits bit-identical (hash at B=1..8). Same-card events: B=1 8614 -> 8567 us, B=2 8852 -> 8795, B=8 9155 -> 9146. Expect ~8582 us at B=1 and ~9125 at B=8 over HTTP.
- evaluator: valid: 8587 us/step at B=1 and 9114 at B=8 over HTTP (online (submit/poll)); in process 8572 us/step at B=1 (step, MBU 0.750)

### #7 s0007: Track B build of the s0006 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7847 (B=8: 8360), at 2.37 agent-hours, parent s0006, category memory
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix stored as an exact 12-bit code with per-row 16-exponent windows and patched escapes, as s5's s0017; BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0006, built with engine/TRACK = 'A W12'; no other change. Verified on this source (tools/w12_full.sh): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps are bitwise equal to the BF16-storage build, submit/poll equals step-by-step greedy. The kernel changes of this run (cs weight loads, one-pass recurrent op at B=8, prefetch schedule, host-mapped tokens) carry over: in-process serving 7822 us per step at B=1 and 8316 at B=8 (s5's s0017: 8095 / 8862 over HTTP). Expect ~7835 us at B=1 and ~8330 at B=8 over HTTP.
- evaluator: valid: 7847 us/step at B=1 and 8360 at B=8 over HTTP (online (submit/poll)); in process 7797 us/step at B=1 (generate, MBU 0.823)

### #8 s0008: Attention layer as one op: a block per 256-key chunk, sentinel hand-overs

- status: **valid**, http_step_us_b1 = 8492 (B=8: 9077), at 3.30 agent-hours, parent s0006, category attention
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0006, for one-token passes: the four ops of a full-attention layer (prep, scores, softmax, value sum: 22-28 us per layer, mostly their four grid barriers) become one op. Block (sequence, kv head, 256-key chunk) computes the 6 query heads' fragments, the scores of its 16 key tiles and, after the unit's hub block has run the softmax reductions over all chunks' scores, its probability fragments and value sums; the hub adds the chunks' value sums in chunk order. Hand-overs between blocks use sentinel slots (relaxed gpu-scope store and poll on one location), the K/V append is read back through relaxed loads. Same MMA chains, same softmax partition and reduction order: logits bit-identical to the four ops over 1150 teacher-forced steps at B=1..8 (tools/att_check.py); when a unit has more chunks than blocks (B=8 beyond position 1023) the four ops run. Same-card events at context ~620: B=1 8587 -> 8534 us, B=8 9113 -> 9088. Expect ~8535 us at B=1 and ~9090 at B=8 over HTTP.
- evaluator: valid: 8492 us/step at B=1 and 9077 at B=8 over HTTP (online (submit/poll)); in process 8469 us/step at B=1 (generate, MBU 0.759)

### #9 s0009: Track B build of the s0008 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7706 (B=8: 8253), at 3.40 agent-hours, parent s0008, category attention
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0008, built with engine/TRACK = 'A W12'; no other change vs s0007 than s0008's one-op attention layer. Verified on this source (tools/w12_full.sh, tools/att_check.py): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps are bitwise equal to the BF16-storage build, the one-op attention equals the four ops over 1150 steps at B=1, 3, 8, submit/poll equals step-by-step greedy. In-process serving of 1024 tokens: 7700 us per step at B=1 and 8229 at B=8 (s0007: 7847 / 8360 over HTTP). Expect ~7715 us at B=1 and ~8245 at B=8 over HTTP.
- evaluator: valid: 7706 us/step at B=1 and 8253 at B=8 over HTTP (online (submit/poll)); in process 7690 us/step at B=1 (generate, MBU 0.835)

### #10 s0010: B=8: RMSNorm partials by the four prefetched warps, r finished before the stream ends

- status: **valid**, http_step_us_b1 = 8494 (B=8: 8982), at 3.96 agent-hours, parent s0008, category scheduling
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0008, in the kernel entry point of one-token passes of 7-8 sequences only: at 5-8 columns every warp loaded its 10 partial sums of squares of the RMSNorm scalar before starting its weight stream, in the 113 normalised GEMMs of a step. Now warps 0-3 of each block load all of them (two per thread; the same sums in the same order) - their shares of the matrix are the block's prefetched bytes, so they have 4-6 us of slack - and warp 0 finishes r at once instead of the first epilogue after the stream. Logits bit-identical to s0008 (hash of 20 steps at B=1..8). Same-card events: B=8 9087 -> 9001 us, B=7 9098 -> 9014, B=1 unchanged (its kernel is not touched). Expect ~8492 us at B=1 and ~8990 at B=8 over HTTP.
- evaluator: valid: 8494 us/step at B=1 and 8982 at B=8 over HTTP (online (submit/poll)); in process 8471 us/step at B=1 (step, MBU 0.759)

### #11 s0011: Track B build of the s0010 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7709 (B=8: 8184), at 4.05 agent-hours, parent s0010, category scheduling
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0010, built with engine/TRACK = 'A W12'; the only change vs s0009 is s0010's (RMSNorm partials by the four prefetched warps at B=7-8). Verified on this source (tools/w12_full.sh): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, submit/poll equals step-by-step greedy, the one-op attention equals the four ops at B=8. In-process serving of 1024 tokens: 8090 us per step at B=8 (s0009: 8253 over HTTP), B=1 unchanged. Expect ~7706 us at B=1 and ~8165 at B=8 over HTTP.
- evaluator: valid: 7709 us/step at B=1 and 8184 at B=8 over HTTP (online (submit/poll)); in process 7690 us/step at B=1 (generate, MBU 0.835)

### #12 s0012: B=8 recurrent op: the next state column's loads fly while the current one is computed

- status: **valid**, http_step_us_b1 = 8492 (B=8: 8956), at 4.31 agent-hours, parent s0010, category scheduling
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0010, in the recurrent op of one-token passes of 7-8 sequences (lmix_tri): a thread made its two or three state-column passes one after the other, each a 16-load round trip (~1.5 us), then its arithmetic, then its stores. Now the second column's loads are issued with the first's and the shared unit's column is loaded while the second is computed; k is read from shared memory at both uses instead of being held in 16 registers, which makes room for the second set of state registers. Same operations in the same order: logits bit-identical to s0010 (hash of 20 steps at B=1..8). Same-card events: B=8 9002 -> 8970 us, B=7 9012 -> 8962, B=1 unchanged (its kernel is not touched). Expect ~8494 us at B=1 and ~8952 at B=8 over HTTP.
- evaluator: valid: 8492 us/step at B=1 and 8956 at B=8 over HTTP (online (submit/poll)); in process 8470 us/step at B=1 (generate, MBU 0.759)

### #13 s0013: Track B build of the s0012 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7708 (B=8: 8166), at 4.43 agent-hours, parent s0012, category scheduling
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0012, built with engine/TRACK = 'A W12'; the only change vs s0011 is s0012's (pipelined state loads of the B=7-8 recurrent op). Verified on this source (tools/w12_full.sh): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, submit/poll equals step-by-step greedy. Expect ~7709 us at B=1 (unchanged) and ~8155 at B=8 over HTTP.
- evaluator: valid: 7708 us/step at B=1 and 8166 at B=8 over HTTP (online (submit/poll)); in process 7690 us/step at B=1 (generate, MBU 0.835)

### #14 s0014: B=8: conv-history rows prefetched during in_proj; far prefetch piece 80 KB

- status: **valid**, http_step_us_b1 = 8492 (B=8: 8948), at 4.84 agent-hours, parent s0012, category memory
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). Two small prefetch changes vs s0012, both only for one-token passes of 7-8 sequences (4+ for the second), measured separately on one card: (a) during in_proj one thread of block b asks L2 for the three conv-history rows of sequence b that the recurrent op right after it reads (60 KB, written 1-3 steps ago): B=8 8969 -> 8958 us; (b) the far prefetch piece at the barriers before small ops is 80 KB instead of 64 KB at B >= 4: 8958 -> 8950. Logits bit-identical to s0012 (hash of 20 steps at B=1..8); B=1 is not touched. Expect ~8492 us at B=1 and ~8940 at B=8 over HTTP (the B=8 change is about twice the evaluator's repeatability).
- evaluator: valid: 8492 us/step at B=1 and 8948 at B=8 over HTTP (online (submit/poll)); in process 8469 us/step at B=1 (generate, MBU 0.759)

### #15 s0015: Track B build of the s0014 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7708 (B=8: 8193), at 4.96 agent-hours, parent s0014, category memory
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0014, built with engine/TRACK = 'A W12'; the only change vs s0013 is s0014's conv-history prefetch during in_proj at B=7-8 (the 80 KB far piece is for BF16 storage only; the 12-bit engine keeps 64 KB). Verified on this source (tools/w12_full.sh): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, submit/poll equals step-by-step greedy. Expect ~7708 us at B=1 (unchanged) and ~8155 at B=8 over HTTP (the change is at the evaluator's repeatability).
- evaluator: valid: 7708 us/step at B=1 and 8193 at B=8 over HTTP (online (submit/poll)); in process 7699 us/step at B=1 (generate, MBU 0.835)

### #16 s0016: Attention blocks ask for their K chunk at the barrier before the op (B >= 4)

- status: **valid**, http_step_us_b1 = 8494 (B=8: 8905), at 5.24 agent-hours, parent s0014, category attention
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0014, in the host-made prefetch table only (kernel source unchanged except that the conv-history prefetch of s0014 is now compiled out of the 12-bit build): for one-token passes of 4 or more sequences, each block of the one-op attention layer asks L2 for its 131 KB chunk of the K cache at the grid barrier before the op, in the place of a weights piece. The phase timer showed the op's largest phase to be the scores waiting for K tiles requested only at the op's start (7 of 21 us per layer at B=8). Logits bit-identical to s0014 (hash of 20 steps at B=1..8). Same-card events: B=8 8949 -> 8891 us, B=5 8886 -> 8832; B=1 not changed (8530 -> 8526 when switched on there: left off). Expect ~8492 us at B=1 and ~8900 at B=8 over HTTP.
- evaluator: valid: 8494 us/step at B=1 and 8905 at B=8 over HTTP (online (submit/poll)); in process 8471 us/step at B=1 (step, MBU 0.759)

### #17 s0017: Track B build of the s0016 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7710 (B=8: 8131), at 5.33 agent-hours, parent s0016, category attention
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0016, built with engine/TRACK = 'A W12'. Two differences from s0015: the conv-history prefetch during in_proj is compiled out of the 12-bit build (it cost s0015 27 us at B=8 against s0013; timed afterwards on one card: 8160 us without, 8188 with), and s0016's K-chunk request at the barrier before the attention op at B >= 4 (12-bit build, same card: B=8 8168 -> 8113 us, B=1 unchanged). Verified on this source (tools/w12_full.sh, tools/att_check.py): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, one-op attention equal to the four ops over 1150 steps at B=8, submit/poll equals step-by-step greedy. In-process serving of 1024 tokens: 7701 / 8110 us per step. Expect ~7708 us at B=1 and ~8115 at B=8 over HTTP.
- evaluator: valid: 7710 us/step at B=1 and 8131 at B=8 over HTTP (online (submit/poll)); in process 7689 us/step at B=1 (generate, MBU 0.836)

### #18 s0018: Grid barrier: a block's prefetch requests go out after its counter increment

- status: **valid**, http_step_us_b1 = 8466 (B=8: 8899), at 6.07 agent-hours, parent s0016, category sync
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0016, in the grid barrier after every op: thread 32 of a block issued the block's L2 prefetch requests (160 KB before a GEMM, up to 414 KB at small batches) at the same moment as thread 0 issued the barrier's counter increment, so the increment of the last block to arrive could queue behind that block's own prefetch lines in the SM's memory queue and hold the release for all blocks (the final per-op profile showed qkv 2.2 us longer since its barrier carries the whole-share request). Thread 32 now waits for a shared-memory flag that thread 0 sets right after its increment. No arithmetic or data flow changes: logits bit-identical to s0016 (hash of 20 steps at B=1..8). Same-card events, two cards: B=1 8529 -> 8504 and 8537 -> 8510 us, B=8 8886 -> 8877 and 8897 -> 8885. Expect ~8467 us at B=1 and ~8895 at B=8 over HTTP.
- evaluator: valid: 8466 us/step at B=1 and 8899 at B=8 over HTTP (online (submit/poll)); in process 8444 us/step at B=1 (generate, MBU 0.761)

### #19 s0019: Track B build of the s0018 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7689 (B=8: 8116), at 6.18 agent-hours, parent s0018, category sync
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0018, built with engine/TRACK = 'A W12'; the only change vs s0017 is s0018's (a block's barrier-time prefetch requests are issued after its counter increment). Timed on the 12-bit build before submitting (same card, 3 alternations): B=1 7702 -> 7686 us, B=8 8114 -> 8103. Verified on this source (tools/w12_full.sh, tools/att_check.py): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, one-op attention equal to the four ops at B=1 and 8, submit/poll equals step-by-step greedy. Expect ~7694 us at B=1 and ~8120 at B=8 over HTTP.
- evaluator: valid: 7689 us/step at B=1 and 8116 at B=8 over HTTP (online (submit/poll)); in process 7678 us/step at B=1 (generate, MBU 0.836)

### #20 s0020: Grid barrier: one prefetch piece per poll, none after the release

- status: **valid**, http_step_us_b1 = 8455 (B=8: 8880), at 6.43 agent-hours, parent s0018, category sync
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). One change vs s0018, in the grid barrier: thread 32 issued all of the block's 64 KB prefetch pieces at once (160 KB before a GEMM, up to 414 KB at small batches), so thread 0's polls of the barrier counter queued behind all of them and a block that arrived late kept asking after the release. Now it issues one piece per poll of thread 0 (a counter in shared memory) and stops when the block has seen the release. No arithmetic or data flow changes: logits bit-identical to s0018 (hash of 20 steps at B=1..8). Same-card events on two cards: B=1 8505 -> 8463 and 8507 -> 8471 us, B=8 8875 -> 8874 and 8890 -> 8877. Expect ~8434 us at B=1 and ~8890 at B=8 over HTTP.
- evaluator: valid: 8455 us/step at B=1 and 8880 at B=8 over HTTP (online (submit/poll)); in process 8429 us/step at B=1 (generate, MBU 0.762)

### #21 s0021: Track B build of the s0020 kernel: exact 12-bit weight storage

- status: **valid**, http_step_us_b1 = 7665 (B=8: 8102), at 6.55 agent-hours, parent s0020, category sync
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). Same source as the track-A submission s0020, built with engine/TRACK = 'A W12'; the only change vs s0019 is s0020's (one barrier-time prefetch piece per poll, none after the release). Timed on the 12-bit build before submitting (same card, 3 alternations): B=1 7687 -> 7662 us, B=8 8102 -> 8090. Verified on this source (tools/w12_full.sh, tools/att_check.py): all 25,621,954,560 stored weights decode to the checkpoint's bits, all 257 GEMM outputs at B=1 and B=8 and the logits of 48 steps bitwise equal to the BF16-storage build, one-op attention equal to the four ops at B=1 and 8, submit/poll equals step-by-step greedy. s0020 gained 11 us at B=1 over HTTP where the same-card events said 36; expect ~7680 us at B=1 and ~8105 at B=8 over HTTP.
- evaluator: valid: 7665 us/step at B=1 and 8102 at B=8 over HTTP (online (submit/poll)); in process 7645 us/step at B=1 (generate, MBU 0.840)

### #22 s0022: Far prefetch piece 128 KB for the BF16 engine (host constant)

- status: **valid**, http_step_us_b1 = 8445 (B=8: 8878), at 7.09 agent-hours, parent s0020, category memory
- hypothesis: track A (checkpoint BF16 storage, BF16 activations and K/V, float32 recurrent state; no format differs from the checkpoint's). Kernel source identical to s0020; one host constant: the prefetch piece that blocks ask for at the barriers before small ops (recurrent op, attention op) is 128 KB instead of 64 KB (80 KB at B >= 4), for BF16 storage only. With s0020's barrier (one piece per poll, none after the release) a larger request no longer holds late blocks. Swept on the evaluator's card (events): 64/80 KB 8462 / 8858 us at B=1 / B=8, 96 KB 8453 / 8851, 128 KB 8448 / 8850, 160 KB 8454 / 8859; on another card of the pod 128 KB was worse, so this is a small, card-level effect. Logits bit-identical (hash of 20 steps at B=1..8). Expect ~8447 us at B=1 and ~8873 at B=8 over HTTP: at the evaluator's repeatability. The 12-bit build of this source keeps 64 KB (128 KB measured +2 us there) and is s0021 unchanged.
- evaluator: valid: 8445 us/step at B=1 and 8878 at B=8 over HTTP (online (submit/poll)); in process 8417 us/step at B=1 (step, MBU 0.764)

### #23 s0023: Track B build of the s0022 source (unchanged from s0021: repeat measurement)

- status: **valid**, http_step_us_b1 = 7668 (B=8: 8114), at 7.18 agent-hours, parent s0022, category memory
- hypothesis: track B (formats differing from the checkpoint's: every streamed weight matrix as an exact 12-bit code with per-row 16-exponent windows and patched escapes, BF16 twins of out_proj / o_proj for one-column passes; everything else as track A). The s0022 source built with engine/TRACK = 'A W12'. s0022's only change (far prefetch piece 128 KB) applies to BF16 storage; the 12-bit build keeps 64 KB, its kernel source is s0021's, so this build behaves as s0021 and the submission is a repeat measurement of it from the final source (bit-identity to the BF16 build was verified on this kernel for s0021: tools/w12_full.sh). Expect s0021's 7665 us at B=1 and 8102 at B=8 over HTTP within the evaluator's repeatability (about 10 us).
- evaluator: valid: 7668 us/step at B=1 and 8114 at B=8 over HTTP (online (submit/poll)); in process 7653 us/step at B=1 (generate, MBU 0.839)

