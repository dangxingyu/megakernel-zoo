# s15-mini-b200-a2: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 26 submissions so far, 25 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3355
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (PR #69028, built from source) (general engine): 3692

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: First persistent kernel: all phases on device, 6+ grid barriers per layer

- status: **valid**, http_step_us_b1 = 7709 (B=8: 1.938e+04), at 0.55 agent-hours, category structure
- hypothesis: track A (checkpoint formats; expert inputs BF16 carried exactly as f16 hi+lo, weights decoded exactly). First correct engine: one cooperative kernel per step, ~390 grid barriers. Expect valid gates (fork-like error ~0.03 median) and ~7.6 ms/step B=1, ~17.8 ms B=8 over HTTP (in-process w1-like 7.7 / 17.8 ms): slower than vLLM; this is the baseline to account the gap from.
- evaluator: valid: 7709 us/step at B=1 and 19377 at B=8 over HTTP (online (submit/poll)); in process 7604 us/step at B=1 (generate, MBU 0.090)

### #2 s0002: Gate F/G vectorized, parallel routing+G1, expert loads unrolled, attention on MMA

- status: **build_failed**, no score, at 0.71 agent-hours, parent s0001, category structure
- hypothesis: track A (formats unchanged). Removes the 2.5 ms scalar F/G gate phase, the serial thread-0 routing and block-0 G1, unrolls expert unit loads 4x, and moves attention QK/PV to mma with tagged chunk records. In process w1-like 4990 us (B=1) / 11491 (B=8) measured on card 1; expect HTTP ~5000 us B=1 and ~11.7 ms B=8 (from 7709 / 19377), numerics unchanged in distribution (attention summation order changed).
- evaluator: engine/build.sh failed (rc 1)

### #3 s0003: Tagged dataflow hand-offs instead of grid barriers; vectorized gate; MMA attention

- status: **valid**, http_step_us_b1 = 5329 (B=8: 1.373e+04), at 1.36 agent-hours, parent s0002, category sync
- hypothesis: track A (formats unchanged). Phases hand over through (value, tag) 64-bit relaxed stores polled by consumers; no grid barrier inside a step; F/G gate vectorized; routing warp-parallel; attention on mma with tagged chunk records. In process (card 1) w1-like 4985 us B=1, 12513 us B=8 (ctx 64-192). Expect HTTP ~5050 us B=1 (from 7709) and ~12.8 ms B=8 (from 19377); numerics unchanged in distribution.
- evaluator: valid: 5329 us/step at B=1 and 13731 at B=8 over HTTP (online (submit/poll)); in process 5088 us/step at B=1 (generate, MBU 0.134)

### #4 s0004: Block-owned GEMV strips (no cross-block split-K), butterfly reductions, leaner gates

- status: **valid**, http_step_us_b1 = 4043 (B=8: 1.022e+04), at 1.68 agent-hours, parent s0003, category structure
- hypothesis: track A (formats unchanged). Removes the fence+atomic+reload reduction of split-K pieces (each GEMV strip owned by one block, warps split K, smem sum in warp order; w2 block per output strip sums the experts in id order), butterfly block reductions, gate D blocks compute their own mixes, F/G loads hoisted. In process 4007 us w1-like at ctx 64-320 (from 4985 for s0003). Expect HTTP ~4300 us B=1 (s0003 5329) and ~9.8 ms B=8 (13731).
- evaluator: valid: 4043 us/step at B=1 and 10216 at B=8 over HTTP (online (submit/poll)); in process 3855 us/step at B=1 (generate, MBU 0.176)

### #5 s0005: B>=2 mixes published by designated blocks; w2 split expert-major + per-strip sum

- status: **valid**, http_step_us_b1 = 3887 (B=8: 7927), at 2.00 agent-hours, parent s0004, category structure
- hypothesis: track A (formats unchanged). For B>=2 the attention/MLP-input mixes and gate norms are computed once per (token, stream) by designated blocks and published as tagged rows (was: every block, tokens in series); w2 split into an expert-major part (each block stages <=2 experts' inputs) publishing per (token, expert rank) rows and a per-strip part summing them in id order. In process B=8 w1-like 7217 us (s0004 ~9.2 ms), B=1 3976 (unchanged). Expect HTTP B=8 ~7.5 ms (from 10216), B=1 ~4050 (4043).
- evaluator: valid: 3887 us/step at B=1 and 7927 at B=8 over HTTP (online (submit/poll)); in process 3816 us/step at B=1 (generate, MBU 0.178)

### #6 s0006: Device loop per serving cohort; o_proj weights preloaded; B>=2 qkv L2 prefetch

- status: **valid**, http_step_us_b1 = 3731 (B=8: 7670), at 2.18 agent-hours, parent s0005, category launch
- hypothesis: track A (formats unchanged). One cooperative launch runs a whole cohort (prompt + greedy decode, argmax fed back on device, tokens to host-mapped memory): no per-step host launch in the served path. o_proj weights loaded into registers before the attention-output poll (bit-identical): -157 us in process at B=1. For B>=2 each block prefetches its qkv strips into L2 while the mix waits. In process (card 1) B=1 3724 us, B=8 6744 us at ctx 64-576. Expect HTTP B=1 ~3730 (s0005 3887: also removes ~70 us of serving overhead), B=8 ~6.8 ms (7927).
- evaluator: valid: 3731 us/step at B=1 and 7670 at B=8 over HTTP (online (submit/poll)); in process 3653 us/step at B=1 (generate, MBU 0.184)

### #7 s0007: Gate F/G tasks placed on blocks without a D job

- status: **valid**, http_step_us_b1 = 3565 (B=8: 7668), at 2.71 agent-hours, parent s0006, category scheduling
- hypothesis: track A (formats unchanged). The gate's F/G tasks were spread over all blocks, including the D-job blocks which start them only after their own D job, so the norm waited on the D chain twice. Now (when one round fits, i.e. B=1) only the other blocks take F/G tasks; numerics unchanged (placement only). A/B in process B=1 3735 -> 3564 us, B=8 unchanged. Expect HTTP B=1 ~3560 (s0006 3731), B=8 ~7.6 ms (7670).
- evaluator: valid: 3565 us/step at B=1 and 7668 at B=8 over HTTP (online (submit/poll)); in process 3503 us/step at B=1 (generate, MBU 0.193)

### #8 s0008: Gate D split into x-half and message-half jobs

- status: **valid**, http_step_us_b1 = 3625 (B=8: 7731), at 2.83 agent-hours, parent s0007, category scheduling
- hypothesis: track A (formats unchanged). Each gate D strip is computed by two jobs: the x half (needs the token's mix) and the message half (needs only the previous pass), 36 units each, summed by F/G in a fixed order (B-independent). The message half leaves the critical path and the x half streams half the bytes. A/B in process B=1 3564 -> ~3500 us. Expect HTTP B=1 ~3500 (s0007 3565), B=8 ~7.6 ms.
- evaluator: valid: 3625 us/step at B=1 and 7731 at B=8 over HTTP (online (submit/poll)); in process 3542 us/step at B=1 (generate, MBU 0.189)

### #9 s0009: Re-evaluation of s0007 source (noise check)

- status: **valid**, http_step_us_b1 = 3562 (B=8: 7668), at 3.07 agent-hours, parent s0008, category other
- hypothesis: track A (formats unchanged). Identical source to s0007 (3565 us B=1), resubmitted to measure evaluation-to-evaluation spread: s0008 measured +60 us on card 0 while a same-job A/B on card 1 showed it 45 us faster. Expect 3565 +- 30 us.
- evaluator: valid: 3562 us/step at B=1 and 7668 at B=8 over HTTP (online (submit/poll)); in process 3509 us/step at B=1 (generate, MBU 0.193)

### #10 s0010: Attention records always 64 keys: bitwise batch invariance at every batch size

- status: **valid**, http_step_us_b1 = 3590 (B=8: 7864), at 3.38 agent-hours, parent s0009, category correctness
- hypothesis: track A (formats unchanged). Online-softmax records now always cover 64 keys (B>=3 items load 128 keys but compute two records), so a sequence's logits are bit-identical alone and in any batch (was: differed from position 64 on at B>=3, within tolerance). Same arithmetic at B=1. Expect B=1 unchanged (~3565 us, s0007), B=8 within +-2% of 7668.
- evaluator: valid: 3590 us/step at B=1 and 7864 at B=8 over HTTP (online (submit/poll)); in process 3519 us/step at B=1 (generate, MBU 0.191)

### #11 s0011: Attention: two records' PV on parallel warp groups (B>=3)

- status: **valid**, http_step_us_b1 = 3563 (B=8: 7694), at 3.53 agent-hours, parent s0010, category attention
- hypothesis: track A (formats unchanged). s0010 made attention records 64 keys at every batch size (bitwise batch invariance) but ran the two records of a B>=3 item one after the other; now warps 0-7 and 8-15 do them in parallel; same arithmetic per record. Card-1 A/B: B=8 -140 us, B=1 -25 us. Expect B=8 ~7.73 ms (s0010 7864), B=1 ~3570 (3590).
- evaluator: valid: 3563 us/step at B=1 and 7694 at B=8 over HTTP (online (submit/poll)); in process 3516 us/step at B=1 (generate, MBU 0.191)

### #12 s0012: Build with --use_fast_math (approximate exp/tanh/div in the vector code)

- status: **valid**, http_step_us_b1 = 3330 (B=8: 7361), at 3.92 agent-hours, parent s0011, category other
- hypothesis: track A (formats unchanged; arithmetic precision of the vector code only). The critical path is full of expf / tanhf / divisions (softmax, sigmoid, SwiGLU-tanh activation, gate tanh, norms); fast-math intrinsics shorten those chains. Same-job A/B on card 1: B=1 3607 -> 3379 us, B=8 6890 -> 6637; reference error unchanged within sampling (median 0.031 -> 0.027 at B=1); batch invariance bitwise. Expect HTTP B=1 ~3350 (s0011 3563), B=8 ~7.45 ms (7694); numeric gates pass with margin.
- evaluator: valid: 3330 us/step at B=1 and 7361 at B=8 over HTTP (online (submit/poll)); in process 3287 us/step at B=1 (generate, MBU 0.206)

### #13 s0013: Fix: attention merge staging overlapped the merger's own record beyond 3711 positions

- status: **valid**, http_step_us_b1 = 3393 (B=8: 7409), at 4.76 agent-hours, parent s0012, category correctness
- hypothesis: track A (formats unchanged). Correctness fix outside the evaluated range: from 59 attention records (position 3712) the merge staging in shared memory overwrote the merger's own record (same union), giving garbage, NaN router logits and an out-of-range expert id (illegal address). Own record moved out of the union; top-8 made NaN-safe; relative-bias index clamped; generate() validates ids. Positions < 3648 unchanged (bit-identical logits expected); expect 3330 +- 10 us B=1, ~7.36 ms B=8.
- evaluator: valid: 3393 us/step at B=1 and 7409 at B=8 over HTTP (online (submit/poll)); in process 3332 us/step at B=1 (generate, MBU 0.199)

### #14 s0014: Merge fix v2: own record staged first (smem layout as s0012)

- status: **valid**, http_step_us_b1 = 3344 (B=8: 7394), at 4.90 agent-hours, parent s0013, category correctness
- hypothesis: track A (formats unchanged). Same correctness fix as s0013 (positions >= 3712) but without moving arrays out of the shared-memory union: the merger copies its own record into the staging area before polling the others (its slots never reach the record's location). Also drops the per-step id check added to _check in s0013 (host syncs). Expect s0012's numbers: 3330 +- 10 us B=1, ~7.36 ms B=8, gpu_never_waits back.
- evaluator: valid: 3344 us/step at B=1 and 7394 at B=8 over HTTP (online (submit/poll)); in process 3303 us/step at B=1 (generate, MBU 0.203)

### #15 s0015: Router top-8 with redux.sync on order-preserving integer keys

- status: **valid**, http_step_us_b1 = 3305 (B=8: 7257), at 4.98 agent-hours, parent s0014, category other
- hypothesis: track A (formats unchanged). The top-8 of each token (every block, every application, on the critical path) used 8 rounds of 5 shuffle levels x 3 values; now per round one local max, a max-redux of the keys and a min-redux of the index among equal keys: same choice (bitcheck: bit-identical at B=1 and 8). Card-1 A/B: B=1 -40 us, B=8 -160 us. Expect B=1 ~3305 (s0014 3344), B=8 ~7.25 ms (7394).
- evaluator: valid: 3305 us/step at B=1 and 7257 at B=8 over HTTP (online (submit/poll)); in process 3255 us/step at B=1 (generate, MBU 0.206)

### #16 s0016: Expert GEMVs: per-group barriers, deeper load queues; no id-check host syncs

- status: **valid**, http_step_us_b1 = 3251 (B=8: 7089), at 6.30 agent-hours, parent s0015, category sync
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). w13/w2a 3-warp groups sync with named barriers instead of the block (rounds no longer wait for the slowest of 24 warps), FP4 loads 6 units deep (FP8 3), token-id check without host syncs; arithmetic unchanged (bitwise). Expect B=8 HTTP -150 us (~7.1 ms), B=1 -20 us (~3285).
- evaluator: valid: 3251 us/step at B=1 and 7089 at B=8 over HTTP (online (submit/poll)); in process 3195 us/step at B=1 (generate, MBU 0.216)

### #17 s0017: Experts with >4 tokens split into two union entries; NG=2 GEMV code removed

- status: **valid**, http_step_us_b1 = 3012 (B=8: 6231), at 7.20 agent-hours, parent s0016, category structure
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). Routing splits an expert routed by >4 tokens into two union entries (tokens 0-3 / 4-7, weights read twice, second mostly from L2), so w13/w2a only need one MMA n-group; removing the NG=2 instantiations drops the kernel's stack frame 600 -> 24 B. Bit-identical per token. Card 1: -230 us B=1, -700 us B=8; expect ~3030 / ~6.4 ms over HTTP.
- evaluator: valid: 3012 us/step at B=1 and 6231 at B=8 over HTTP (online (submit/poll)); in process 2960 us/step at B=1 (step, MBU 0.234)

### #18 s0018: Deeper expert loads: w13 12 units per warp in one batch, MXFP8 6 deep

- status: **valid**, http_step_us_b1 = 3000 (B=8: 6174), at 7.87 agent-hours, parent s0017, category other
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). With the stack frame at 24 B there is register room: w13 now issues all 12 NVFP4 units of a warp's share at once (was 2 x 6), MXFP8 pieces 6 deep (was 3); same unit order, bit-identical. Card 1: B=8 -65 us, B=1 -10; expect ~3000 / ~6.17 ms over HTTP.
- evaluator: valid: 3000 us/step at B=1 and 6174 at B=8 over HTTP (online (submit/poll)); in process 2938 us/step at B=1 (generate, MBU 0.235)

### #19 s0019: Gate down-projection GEMV: each warp's 3 units loaded in one batch

- status: **valid**, http_step_us_b1 = 2990 (B=8: 6180), at 8.34 agent-hours, parent s0018, category other
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). The message-gate D GEMV (72 units, 3 per warp) loads all 3 units at once instead of 2 + 1 (one HBM latency less per D job, 2 D phases per gated application); bit-identical. Card 1 (3 alternations): B=1 -23 us, B=8 +8; expect ~2980 B=1, B=8 unchanged (~6.18 ms).
- evaluator: valid: 2990 us/step at B=1 and 6180 at B=8 over HTTP (online (submit/poll)); in process 2948 us/step at B=1 (step, MBU 0.235)

### #20 s0020: Routing builds union token lists from per-expert token bit masks

- status: **valid**, http_step_us_b1 = 2964 (B=8: 6081), at 8.47 agent-hours, parent s0019, category other
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). The union tables scanned all B x 8 (token, rank) slots per expert serially (~128 dependent smem reads per lane at B=8); now each selection atomicOr's a token bit per expert and lists come from the mask bits (same ascending token order: identical tables). Card 1: B=1 -30 us, B=8 -60; expect ~2960 / ~6.12 ms.
- evaluator: valid: 2964 us/step at B=1 and 6081 at B=8 over HTTP (online (submit/poll)); in process 2930 us/step at B=1 (generate, MBU 0.236)

### #21 s0021: B=1: router weight strip prefetched into L1 during the MLP-input mix

- status: **valid**, http_step_us_b1 = 2948 (B=8: 6085), at 8.77 agent-hours, parent s0020, category other
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). At B=1 the 8 router blocks prefetch their 74 KB router strip into L1 (prefetch.global.L1) while the MLP-input mix waits for o_proj; the router GEMV then hits L1. Arithmetic unchanged. Card 1 (3 alternations): B=1 -22 us, B=8 unchanged; expect ~2945 / ~6.08 ms.
- evaluator: valid: 2948 us/step at B=1 and 6085 at B=8 over HTTP (online (submit/poll)); in process 2918 us/step at B=1 (generate, MBU 0.237)

### #22 s0022: Current key/value short conv: all loads issued before the first wait

- status: **valid**, http_step_us_b1 = 2888 (B=8: 6111), at 9.71 agent-hours, parent s0021, category attention
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). The attention item holding the current position computed its K/V short conv channel by channel, waiting on each raw value before loading the history ring: ~1.6 us longer than other items, and the merger waits for it. kv_conv4 issues raw values, taps and history of 4 channels before any wait; bit-identical. Card 1: B=1 -55 us, B=8 flat; expect ~2895 / ~6.08 ms.
- evaluator: valid: 2888 us/step at B=1 and 6111 at B=8 over HTTP (online (submit/poll)); in process 2875 us/step at B=1 (generate, MBU 0.240)

### #23 s0023: Norm weights, q/r scales and router bias loaded before the polls

- status: **valid**, http_step_us_b1 = 2825 (B=8: 5992), at 9.86 agent-hours, parent s0022, category sync
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). Polls are volatile asm with a memory clobber, so constant loads written after them (norm weights of the attention-input and MLP-input mixes, attention q/r scales, router bias) cost one extra L2 round trip each on the critical path; they are now issued before the polls. Bit-identical. Card 1: B=1 -65 us, B=8 -110; expect ~2825 / ~6.0 ms.
- evaluator: valid: 2825 us/step at B=1 and 5992 at B=8 over HTTP (online (submit/poll)); in process 2807 us/step at B=1 (generate, MBU 0.246)

### #24 s0024: Expert scale factors and o_proj conv history loaded before their polls

- status: **valid**, http_step_us_b1 = 2776 (B=8: 5860), at 10.10 agent-hours, parent s0023, category sync
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). The w13 / w2a epilogues read the expert's NVFP4 scale factor from global after the GEMV, and o_proj's epilogue its conv history and taps after the attention-output poll: one L2 round trip each on the critical path. Scale factors now go to smem at routing start (before its polls), the o_proj history to smem before its poll. Bit-identical. Card 1: B=1 -57 us, B=8 -150; expect ~2770 / ~5.85 ms.
- evaluator: valid: 2776 us/step at B=1 and 5860 at B=8 over HTTP (online (submit/poll)); in process 2755 us/step at B=1 (generate, MBU 0.251)

### #25 s0025: MXFP8 expert pieces 4 units deep (was 6)

- status: **valid**, http_step_us_b1 = 2755 (B=8: 5825), at 11.46 agent-hours, parent s0024, category other
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). Re-tuned after s0022-s0024 changed the critical path: MXFP8 (layers 20-23) GEMV pieces load 4 units per batch instead of 6; bit-identical. Card 1 (3 alternations): B=1 -9 us, B=8 -16; expect ~2767 / ~5.84 ms (within noise of s0024 at B=1).
- evaluator: valid: 2755 us/step at B=1 and 5825 at B=8 over HTTP (online (submit/poll)); in process 2730 us/step at B=1 (generate, MBU 0.253)

### #26 s0026: B>=3 attention: merging item holds one record when that adds no wave

- status: **valid**, http_step_us_b1 = 2761 (B=8: 5814), at 12.52 agent-hours, parent s0025, category attention
- hypothesis: track A (formats as the checkpoint: NVFP4/MXFP8 experts, BF16 rest; unquantized expert inputs). At B>=3 the last attention wave is the merging items (chunk 0, two 64-key records); with items {0},{1,2},{3,4},... the merger loads and computes one record, chosen only when the item count needs no extra wave. Records and merge unchanged: bit-identical. Card 1: B=8 -12..-28 us, B=1 unchanged; expect ~2755 / ~5.80 ms.
- evaluator: valid: 2761 us/step at B=1 and 5814 at B=8 over HTTP (online (submit/poll)); in process 2726 us/step at B=1 (generate, MBU 0.253)

