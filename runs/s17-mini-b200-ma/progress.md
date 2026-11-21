# s17-mini-b200-ma: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 5.0 agent-hours. 14 submissions so far, 14 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3370
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (inkling-2-mini image) (general engine): 3707

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Merge: a2 s0025 kernel at batch 1, a1 s0024 kernel at batch >= 2

- status: **valid**, http_step_us_b1 = 2749 (B=8: 5522), at 0.16 agent-hours, category structure
- hypothesis: track A (formats unchanged from the checkpoint; both kernels unquantized BF16 expert inputs). One engine loads both track-A engines of the night (each its own weight copy, 44 GB total) and serves a cohort with the a2 kernel (s15-mini-b200-a2 s0025) at B=1 and with the a1 kernel (s15-mini-b200-a1 s0024) at B>=2. Expect B=1 HTTP ~2755 (as a2 s0025, +-15) and B=8 ~5515 (as a1 s0024; same-job serve test: a1 5878 vs a2 6096 = -220 us).
- evaluator: valid: 2749 us/step at B=1 and 5522 at B=8 over HTTP (online (submit/poll)); in process 2727 us/step at B=1 (generate, MBU 0.253)

### #2 s0002: B>=2 kernel: weight ring stages of 9 items x 2 (port of a1 s0025)

- status: **valid**, http_step_us_b1 = 2752 (B=8: 5450), at 0.24 agent-hours, parent s0001, category gemm
- hypothesis: track A (formats unchanged). Port of s15-mini-b200-a1 s0025 into the B>=2 kernel (engine/a1): TMA weight-ring stages of 9 items x 2 instead of 12 x 2 (no partial stages for 18/36/72-item tasks). a1 measured B=8 5513 -> 5443 over HTTP. Expect B=8 ~5450 (-70), B=1 unchanged (~2750, other kernel).
- evaluator: valid: 2752 us/step at B=1 and 5450 at B=8 over HTTP (online (submit/poll)); in process 2719 us/step at B=1 (generate, MBU 0.254)

### #3 s0003: Batch-1 kernel compiled for B=1 only, with the mark sites present (KB1+MKPROF)

- status: **valid**, http_step_us_b1 = 2732 (B=8: 5471), at 1.13 agent-hours, parent s0002, category cleanup
- hypothesis: track A (formats unchanged). The a2 kernel now only serves B=1 (B>=2 runs on the a1 kernel), so it is compiled with the batch size a compile-time 1 (idea of s15-mini-b200-b2 s0020/s0021 K1) and with the mkprof mark sites present but inactive (no profile buffer): bitwise equal to s0002's batch-1 kernel (dump at 12 positions), card 1 one-engine-per-process: 2724 vs 2744 us (3 alternations), serve test 2737 vs ~2758. Expect B=1 HTTP ~2732 (-20), B=8 unchanged (~5450). KB1 alone was +375 us (a code-generation cliff), so the gain is a code-layout effect and may not transfer exactly.
- evaluator: valid: 2732 us/step at B=1 and 5471 at B=8 over HTTP (online (submit/poll)); in process 2671 us/step at B=1 (generate, MBU 0.257)

### #4 s0004: B>=2 kernel: gate phase 0 without mixer blocks, warp-per-message M phase

- status: **valid**, http_step_us_b1 = 2722 (B=8: 5355), at 1.73 agent-hours, parent s0003, category sync
- hypothesis: track A (formats unchanged; same arithmetic, reorganized). In the a1 B>=2 kernel (1) gate phase 0: every block mixes one sequence itself (19 blocks per sequence) and computes that sequence's z rows from its own smem copy: no mixer->flag hand-off and no xm round trip (bitwise equal to s0003); (2) phase M: one warp per (sequence, layer-gate) message, then the message-half GEMV with each D row loaded once for all 8 sequences (pass-2 messages rel 1e-7 vs the old path). Both keep the old paths compiled but never taken, which keeps the NVFP4/MXFP8 decode loops at 540/476 SASS instructions (without that they compile 8-21% longer and the change is +500 us). Card 1, one engine per process: B=8 5623 -> 5495 (generate 512), serve test 5668 vs ~5820. Expect B=8 HTTP ~5340 (-130), B=1 unchanged (~2732).
- evaluator: valid: 2722 us/step at B=1 and 5355 at B=8 over HTTP (online (submit/poll)); in process 2670 us/step at B=1 (generate, MBU 0.259)

### #5 s0005: Final-source check: s0004 engine unchanged (repeatability)

- status: **valid**, http_step_us_b1 = 2729 (B=8: 5357), at 2.28 agent-hours, parent s0004, category other
- hypothesis: track A (formats unchanged). No change: engine/ is byte-identical to s0004 (the study code is kept outside engine/). Purpose: a second evaluation of the final tree to measure the evaluator's repeatability on this engine. Expect B=1 2722 +- 15, B=8 5355 +- 30.
- evaluator: valid: 2729 us/step at B=1 and 5357 at B=8 over HTTP (online (submit/poll)); in process 2668 us/step at B=1 (generate, MBU 0.259)

### #6 s0006: B>=2 kernel: NVFP4 block scales built on the FMA pipe (IMAD), not the ALU pipe

- status: **valid**, http_step_us_b1 = 2720 (B=8: 5279), at 2.83 agent-hours, parent s0005, category gemm
- hypothesis: track A (formats unchanged; bitwise equal to s0004). ncu shows the B=8 NVFP4 decode loop throttled on the ALU pipe (788 M ALU vs 425 M FMA warp instructions per step; math-pipe throttle 19% of the loop's stalls). The E4M3 block scale (b << 20) + (120 << 23) was compiled to LEA (ALU); with the multiplier passed as an opaque register ptxas emits IMAD (FMA pipe): ALU work per item 32 -> 24 instructions. Card 1: B=8 -35..-70 us (5516 vs 5585 at pos 320-576). Expect B=8 HTTP ~5300 (-55), B=1 unchanged (~2725).
- evaluator: valid: 2720 us/step at B=1 and 5279 at B=8 over HTTP (online (submit/poll)); in process 2666 us/step at B=1 (generate, MBU 0.259)

### #7 s0007: B>=2 kernel: MXFP8 weights decoded to bf16 bits straight from the E4M3 bytes

- status: **valid**, http_step_us_b1 = 2729 (B=8: 5231), at 3.17 agent-hours, parent s0006, category gemm
- hypothesis: track A (formats unchanged; bitwise equal to s0006). The MXFP8 (layers 20-23) decode loop is ALU-bound: per weight pair F2FP + SHF + 2 LOP3. bf16 bits (byte << 4) + (byte & 0x80) x 240 equal the E4M3 value x 2^-120 for normal and subnormal codes; with the UE8M0 scale as 2^(e-7) the product is the same bit for bit (exhaustive GPU check over all byte pairs and scale exponents 100..130: 0 mismatches; logits bitwise equal). Per pair 2 ALU + 2 FMA-pipe instructions; loop 476 -> 417, ALU 350 -> 168 per 18 HMMA. Card 1: B=8 -67 us (generate), serve test -71. Expect B=8 HTTP ~5215 (-65), B=1 unchanged.
- evaluator: valid: 2729 us/step at B=1 and 5231 at B=8 over HTTP (online (submit/poll)); in process 2672 us/step at B=1 (generate, MBU 0.255)

### #8 s0008: Final-source check: s0007 engine unchanged (repeatability)

- status: **valid**, http_step_us_b1 = 2731 (B=8: 5229), at 3.68 agent-hours, parent s0007, category other
- hypothesis: track A (formats unchanged). No change: engine/ is s0007 byte for byte. Purpose: a second evaluation of the final tree (B=8 changed in s0006/s0007). Expect B=1 2725 +- 15, B=8 5231 +- 30.
- evaluator: valid: 2731 us/step at B=1 and 5229 at B=8 over HTTP (online (submit/poll)); in process 2675 us/step at B=1 (generate, MBU 0.258)

### #9 s0009: B>=2 kernel: NVFP4 block-scale pairs by one E4M3x2 conversion + f16->f32 on the FMA pipe

- status: **valid**, http_step_us_b1 = 2728 (B=8: 5141), at 4.04 agent-hours, parent s0008, category gemm
- hypothesis: track A (formats unchanged; bitwise equal to s0007). The NVFP4 decode loop is issue/ALU-bound. The two block scales of a k16 block (rows g, g+8) are adjacent bytes: one exact cvt.rn.f16x2.e4m3x2 of the word half gives both, f16->f32 is HADD2.F32 on the FMA pipe; this replaces 2 PRMT (ALU) + 2 IMAD per pair. Loop 541 -> 500 instructions per 40 HMMA (ALU 229 -> 191). Card 1: B=8 -100 us (generate and teacher-forced), serve -101. Expect B=8 HTTP ~5130 (-100), B=1 unchanged (~2725).
- evaluator: valid: 2728 us/step at B=1 and 5141 at B=8 over HTTP (online (submit/poll)); in process 2671 us/step at B=1 (generate, MBU 0.259)

### #10 s0010: B>=2 kernel: a1's PMAXC 2 and SCNT_REL 1 settings on the new base

- status: **valid**, http_step_us_b1 = 2726 (B=8: 5119), at 4.23 agent-hours, parent s0009, category scheduling
- hypothesis: track A (formats unchanged). Two of a1's own switches re-measured on the s0009 code: attention pieces of at most 2 chunks merged within one item at 128-255 keys (PMAXC 2) and the w13 completion counters published by red.release (SCNT_REL 1). Card 1, generate 1024: 5390 vs 5409/5425 us. Expect B=8 HTTP ~5115 (-25), B=1 unchanged.
- evaluator: valid: 2726 us/step at B=1 and 5119 at B=8 over HTTP (online (submit/poll)); in process 2666 us/step at B=1 (generate, MBU 0.259)

### #11 s0011: B>=2 kernel: NVFP4 block scales applied with packed fp32 FMAs (FFMA2)

- status: **valid**, http_step_us_b1 = 2727 (B=8: 5116), at 4.51 agent-hours, parent s0010, category gemm
- hypothesis: track A (formats unchanged; bitwise equal to s0010). Each row's two accumulators get their block scale in one fma.rn.ftz.f32x2 (same rounding as FFMA.FTZ); ptxas takes the repeated scale operand directly, so the NVFP4 decode loop goes from 500 to 427 instructions per 40 HMMA. Card 1: B=8 -30..-40 us (generate 1024 / 512). Expect B=8 HTTP ~5085 (-35: below vLLM's 5105), B=1 unchanged (~2725).
- evaluator: valid: 2727 us/step at B=1 and 5116 at B=8 over HTTP (online (submit/poll)); in process 2664 us/step at B=1 (generate, MBU 0.259)

### #12 s0012: B>=2 kernel: mixer flags published by red.release (a1's PUB_REL 1)

- status: **valid**, http_step_us_b1 = 2727 (B=8: 5118), at 4.59 agent-hours, parent s0011, category sync
- hypothesis: track A (formats unchanged; bitwise equal to s0011). Mixer blocks publish their rows with one release reduction after the block barrier instead of fence + atomic. Card 1: -4 us (generate 512), +2 (1024): within noise; card-1 A/Bs have stopped predicting the evaluator (s0011), so this is a direct test. Expect B=8 5116 +- 10, B=1 unchanged.
- evaluator: valid: 2727 us/step at B=1 and 5118 at B=8 over HTTP (online (submit/poll)); in process 2669 us/step at B=1 (generate, MBU 0.259)

### #13 s0013: B>=2 kernel: grid-barrier spin with 64 ns back-off (a1's BAR_SLEEP 64)

- status: **valid**, http_step_us_b1 = 2725 (B=8: 5117), at 4.66 agent-hours, parent s0012, category sync
- hypothesis: track A (formats unchanged; bitwise equal to s0011). Thread 0 of each block sleeps 64 ns between polls of the grid-barrier counter, reducing L2 polling traffic during the latency-bound phases. Card 1: -5 us (generate 512), +12 (1024). Direct evaluator test; expect B=8 5116 +- 15, B=1 unchanged.
- evaluator: valid: 2725 us/step at B=1 and 5117 at B=8 over HTTP (online (submit/poll)); in process 2673 us/step at B=1 (generate, MBU 0.258)

### #14 s0014: B>=2 kernel: packed fp32 FMAs for the NVFP4 scales, actually enabled (s0011 had it off)

- status: **valid**, http_step_us_b1 = 2728 (B=8: 5093), at 4.75 agent-hours, parent s0013, category gemm
- hypothesis: track A (formats unchanged; bitwise equal to s0011). s0011's FMA2 switch was never defined (its insertion anchor was gone), so s0011-s0013 ran the s0010 kernel. Now on: the NVFP4 decode loop 500 -> 427 instructions per 40 HMMA (FFMA2.FTZ, same rounding). Card 1: B=8 -30..-40 us (generate), serve -39. Expect B=8 HTTP ~5080 (below vLLM's 5105), B=1 unchanged (~2725).
- evaluator: valid: 2728 us/step at B=1 and 5093 at B=8 over HTTP (online (submit/poll)); in process 2675 us/step at B=1 (generate, MBU 0.258)

