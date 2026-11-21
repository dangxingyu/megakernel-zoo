# s15-mini-b200-b2: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 52 submissions so far, 50 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3355
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (PR #69028, built from source) (general engine): 3692

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Persistent kernel v1: phases + grid barriers, exact BF16-input NVFP4/MXFP8 experts

- status: **build_failed**, no score, at 0.60 agent-hours, category structure
- hypothesis: track A (checkpoint formats: BF16, NVFP4 codes+E4M3 scales+amax factor, MXFP8 codes+UE8M0; expert inputs BF16 unquantized). First own engine: one persistent kernel per step, ~490 grid barriers; in-process 6.4 ms/step at B=1, 9.7 ms at B=8 (contexts 64-128). Expect valid gates (own check: median rel-L2 0.016 vs fp32 ref, slot-reversal bitwise) and HTTP ~6.5 ms B=1 / ~10 ms B=8 (slower than vLLM; baseline for what follows).
- evaluator: engine/build.sh failed (rc 1)

### #2 s0002: Persistent kernel v1: phases + grid barriers, exact BF16-input NVFP4/MXFP8 experts

- status: **error**, no score, at 0.60 agent-hours, parent s0001, category structure
- hypothesis: track A (checkpoint formats: BF16, NVFP4 codes+E4M3 scales+amax factor, MXFP8 codes+UE8M0; expert inputs BF16 unquantized). First own engine: one persistent kernel per step, ~490 grid barriers; in-process 6.4 ms/step at B=1, 9.7 ms at B=8 (contexts 64-128). Expect valid gates (own check: median rel-L2 0.016 vs fp32 ref, slot-reversal bitwise) and HTTP ~6.5 ms B=1 / ~10 ms B=8 (slower than vLLM; baseline for what follows).
- evaluator: the engine failed during the correctness run (rc 1)
- gates that failed: run

### #3 s0003: Persistent kernel v1: phases + grid barriers, exact BF16-input NVFP4/MXFP8 experts

- status: **valid**, http_step_us_b1 = 6578 (B=8: 1.266e+04), at 0.71 agent-hours, parent s0002, category structure
- hypothesis: track A (checkpoint formats: BF16, NVFP4 codes+E4M3 scales+amax factor, MXFP8 codes+UE8M0; expert inputs BF16 unquantized). First own engine: one persistent kernel per step, ~490 grid barriers; in-process 6.4 ms/step at B=1, 9.7 ms at B=8 (contexts 64-128). Expect valid gates (own check: median rel-L2 0.016 vs fp32 ref, slot-reversal bitwise) and HTTP ~6.5 ms B=1 / ~10 ms B=8 (slower than vLLM; baseline for what follows).
- evaluator: valid: 6578 us/step at B=1 and 12658 at B=8 over HTTP (offline (step through the cohort adapter)); in process 6384 us/step at B=1 (step, MBU 0.108)

### #4 s0004: Track-A base: adopt s15-mini-b200-a1 s0001 engine unchanged (persistent kernel)

- status: **valid**, http_step_us_b1 = 5042 (B=8: 1.439e+04), at 1.85 agent-hours, parent s0003, category structure
- hypothesis: track A (checkpoint formats unchanged: BF16, NVFP4 codes+E4M3+amax factor, MXFP8+UE8M0; expert inputs BF16). Source of the parallel track-A run s15-mini-b200-a1 submission s0001 (valid there at 5047/14420 us), taken unchanged as the base for this run's track-B work (my own engine s0003 6578 us is slower). Expect valid and ~5050 us B=1 / ~14400 us B=8 over HTTP (in process here 4759 / 12447 at ctx 64-128).
- evaluator: valid: 5042 us/step at B=1 and 14392 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4870 us/step at B=1 (step, MBU 0.142)

### #5 s0005: W12 unembedding: exact 12-bit BF16 code (row window + escapes), rotate decode

- status: **valid**, http_step_us_b1 = 5062 (B=8: 1.433e+04), at 2.27 agent-hours, parent s0004, category memory
- hypothesis: track B (formats differing from the checkpoint: the BF16 unembedding is stored as an exact 12-bit code: per-row 16-binade exponent window, 34676 escapes kept exactly; everything else as track A s0004). Bit-identical to track A (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8: 100% identical). Head reads 0.69 instead of 0.92 GB; decode is prmt + rotate + lop3 per register with the exponent base folded into an exact power-of-two row scale. Expect HTTP -30 us at B=1 (5042 -> ~5012) and -60 us at B=8 (14392 -> ~14330), measured in process -31 / -62 us.
- evaluator: valid: 5062 us/step at B=1 and 14330 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4892 us/step at B=1 (step, MBU 0.141)

### #6 s0006: Track-A base v2: adopt s15-mini-b200-a2 s0006 engine unchanged (tagged dataflow)

- status: **valid**, http_step_us_b1 = 3723 (B=8: 7707), at 2.54 agent-hours, parent s0004, category structure
- hypothesis: track A (checkpoint formats unchanged; expert inputs BF16). Source of the parallel track-A run s15-mini-b200-a2 submission s0006 (valid there at 3731/7670 us: tagged dataflow, no grid barriers, device greedy loop behind submit/poll), taken unchanged as the base for this run's track-B work, replacing s0004 (5042). Expect valid and ~3730 us B=1 / ~7670 us B=8 over HTTP.
- evaluator: valid: 3723 us/step at B=1 and 7707 at B=8 over HTTP (online (submit/poll)); in process 3667 us/step at B=1 (generate, MBU 0.184)

### #7 s0007: MSGZ: messages and message half of the loop gates computed once per pass

- status: **valid**, http_step_us_b1 = 3665 (B=8: 7708), at 3.06 agent-hours, parent s0006, category structure
- hypothesis: track B (formats as track A; data flow differs only by an exact re-association): in loop passes 2-3 the messages (attention over depth of the previous pass) are fixed for the whole pass, so all 48 messages and z_m = D[:, 2304:] unit(m) of every gate are computed once at the pass start; each gate computes only D[:, :2304] unit(x) and adds z_m. Exact in exact arithmetic, not bit-identical (fp32 summation split; own check vs fp32 ref: median 0.0180 vs 0.0176 for A, same top-1). Expect -15..-30 us HTTP B=1 (base s0006 measured here 3731) and -0..-30 us B=8 (in-process 3-instance A/B: -14 step / -26 generate at B=1, -3 / -33 at B=8).
- evaluator: valid: 3665 us/step at B=1 and 7708 at B=8 over HTTP (online (submit/poll)); in process 3633 us/step at B=1 (generate, MBU 0.187)

### #8 s0008: W12U on the a2 base: exact 12-bit unembedding (bit-identical)

- status: **valid**, http_step_us_b1 = 3743 (B=8: 7678), at 3.15 agent-hours, parent s0006, category memory
- hypothesis: track B (formats: the BF16 unembedding stored as an exact 12-bit code, per-row exponent window + escapes; otherwise track A s0006). Bit-identical to track A (same logits / greedy tokens in own A/B; round trip exact on all weights). Head phase 177 -> 141 us (mkprof), end to end -15 us in a 3-instance A/B at B=1 (inside the instance noise), noise at B=8. Expect HTTP B=1 3723 -> ~3705, B=8 unchanged within noise.
- evaluator: valid: 3743 us/step at B=1 and 7678 at B=8 over HTTP (online (submit/poll)); in process 3681 us/step at B=1 (generate, MBU 0.182)

### #9 s0009: SKH: certified int8 sketch of the unembedding + exact rescoring (greedy path)

- status: **valid**, http_step_us_b1 = 3805 (B=8: 7654), at 3.48 agent-hours, parent s0006, category memory
- hypothesis: track B (formats: an additional int8 copy of the unembedding, per-row scale, with exact per-row error and norm bounds; BF16 unembedding kept for exact rescoring and for step() logits). On the greedy path (submit/poll, generate) pass 1 reads 0.46 GB of int8 with the input as two exact int8 digits (int32 sums exact), bounds every row's track-A fp32 logit, and only strips whose capped upper bound reaches the best capped lower bound are recomputed with the track-A arithmetic (same MMA chain): the argmax is identical to track A by construction; step() logits are untouched (bit-identical). Own checks: 6 x 1024 greedy tokens identical to A at B=1 and B=8; 1.2-1.6 exact strips per token. Head 158 -> ~108 us (mkprof). Expect HTTP B=1 3723 -> ~3670 (-55), B=8 7707 -> ~7650 (-57), from a 2-instance generate A/B.
- evaluator: valid: 3805 us/step at B=1 and 7654 at B=8 over HTTP (online (submit/poll)); in process 3754 us/step at B=1 (generate, MBU 0.177)

### #10 s0010: Track A: start-up placement tuning of the polled tagged buffers

- status: **valid**, http_step_us_b1 = 3802 (B=8: 7685), at 3.71 agent-hours, parent s0006, category memory
- hypothesis: track A (formats and data flow unchanged; layout only): the B=1 step depends on where the polled (value, tag) buffers land (same build: 3604-3788 us over six engine instances; moving xpt alone 75 us). At load the engine tries 8 placements (random page offsets), times 64 greedy steps at B=1 each and keeps the fastest. Results unchanged (layout only). Measured over 4 instances: spread 41 us instead of 184, typical ~3640 us. Expect HTTP B=1 ~3640 (s0006 measured 3723 on a slower instance), B=8 unchanged ~7700.
- evaluator: valid: 3802 us/step at B=1 and 7685 at B=8 over HTTP (online (submit/poll)); in process 3746 us/step at B=1 (generate, MBU 0.181)

### #11 s0011: Track B combined: MSGZ (per-pass message gates) + SKH (certified int8 head sketch)

- status: **valid**, http_step_us_b1 = 3618 (B=8: 7652), at 3.88 agent-hours, parent s0010, category structure
- hypothesis: track B (formats: an additional int8 copy of the unembedding with per-row error bounds; data flow: per-pass hoisting of the loop message gates, exact in exact arithmetic, not bit-identical to A; the sketch's argmax is identical to the same build's exact head: 3 x 512 greedy tokens identical to B MSGZ at B=1 and B=8). Base: track A with start-up placement tuning (s0010). Same-card 2-instance generate A/B vs A: -72 us at B=1 (-1.95%), -99 us at B=8 (-1.27%). Expect HTTP B=1 ~3730 and B=8 ~7600 on the evaluation card (card 0 ran ~80 us slower than card 1 for s0009/s0010).
- evaluator: valid: 3618 us/step at B=1 and 7652 at B=8 over HTTP (online (submit/poll)); in process 3601 us/step at B=1 (generate, MBU 0.185)

### #12 s0012: Track B combined v2: MSGZ v2 (tokens as MMA columns at the pass start) + SKH

- status: **valid**, http_step_us_b1 = 3612 (B=8: 7583), at 4.14 agent-hours, parent s0011, category structure
- hypothesis: track B (formats: + int8 sketch of the unembedding with per-row bounds; data flow: per-pass hoisting of the loop message gates, exact in exact arithmetic, not bit-identical to A; the sketch argmax is identical to the same build's exact head, 3 x 256 greedy tokens at B=1 and B=8). Change vs s0011: the pass-start message phase is split into message mixes and a z_m GEMV with all tokens as MMA columns (192 items instead of 192 x B): its span 88 -> 42 us at B=8. Same-card 2-instance generate A/B vs track A (s0010 build): -84 us at B=1 (-2.25%), -174 us at B=8 (-2.22%). Expect B=1 ~3600-3620 and B=8 ~7520 over HTTP (s0011: 3618 / 7652).
- evaluator: valid: 3612 us/step at B=1 and 7583 at B=8 over HTTP (online (submit/poll)); in process 3538 us/step at B=1 (generate, MBU 0.188)

### #13 s0013: L0T: layer-0 q/k/v/gate rows from a per-token-id table built at load (bit-identical)

- status: **valid**, http_step_us_b1 = 3616 (B=8: 7548), at 4.62 agent-hours, parent s0012, category memory
- hypothesis: track B; formats: adds a derived BF16 table [201024][8672] (3.5 GB) of layer 0's qkv rows per token id, built at load by the engine's own GEMV (mk_l0b); everything else as s0012. Layer 0's input is rmsnorm(embed[tok])*embed_norm, a function of the token id only, so application 0 copies its 17 KB row instead of reading the 40 MB qkv matrix; bit-identical (bitcheck: step logits bitwise equal at B=1 and B=8, generate tokens equal). Same-card A/B: -3 us B=1, -32 us B=8; expect B=8 7583 -> ~7550, B=1 unchanged within the +-80 us evaluation noise.
- evaluator: valid: 3616 us/step at B=1 and 7548 at B=8 over HTTP (online (submit/poll)); in process 3587 us/step at B=1 (generate, MBU 0.186)

### #14 s0014: Track A: gate F/G tasks on blocks without a D job (port of a2 s0007)

- status: **valid**, http_step_us_b1 = 3622 (B=8: 7711), at 4.83 agent-hours, parent s0010, category structure
- hypothesis: track A; formats: the checkpoint's (unchanged from s0010). Port of s15-mini-b200-a2 s0007: the elementwise gate F/G tasks of the gated applications run only on blocks that have no gate-D job (those blocks start F/G late, after their own D job); bit-identical to s0010 (bitcheck, B=1 and B=8). Same-card A/B -164 us B=1, +2 B=8: expect B=1 ~3640 (vs s0010 3802 on a slow card 0 / ~3700 typical), B=8 unchanged ~7690.
- evaluator: valid: 3622 us/step at B=1 and 7711 at B=8 over HTTP (online (submit/poll)); in process 3540 us/step at B=1 (generate, MBU 0.191)

### #15 s0015: Track B (MSGZ SKH L0T) on the new track-A base s0014 (gate F/G placement)

- status: **valid**, http_step_us_b1 = 3425 (B=8: 7463), at 4.91 agent-hours, parent s0013, category structure
- hypothesis: track B; formats: as s0013 (12-bit nothing; int8 head sketch + exact rescoring, layer-0 token table, per-pass message gates within the gates); the change is the track-A F/G placement of s0014 under the same track-B items. Same-card A/B -178 us B=1 (3570 -> 3392 in process), B=8 flat: expect B=1 ~3440 (vs s0013 3616), B=8 ~7550.
- evaluator: valid: 3425 us/step at B=1 and 7463 at B=8 over HTTP (online (submit/poll)); in process 3388 us/step at B=1 (generate, MBU 0.196)

### #16 s0016: Gate-D strip L2 prefetch at job start (track-A change under track B items)

- status: **valid**, http_step_us_b1 = 3418 (B=8: 7449), at 5.22 agent-hours, parent s0015, category memory
- hypothesis: track B; formats: as s0015 (int8 head sketch + exact rescoring, layer-0 token table, per-pass message gates within the gates); the change is track A: each gate-D job's block issues one cp.async.bulk.prefetch.L2 of its 72 KB D strip before it mixes x, so the strip GEMV after the mix reads L2 instead of HBM (sub-marks: the GEMV step is 3.5 us, ~1.8 of it waiting for the slowest warp's first-touch loads). Bit-identical (prefetch only). Same-card A/B -31.5 us B=1, -31 us B=8: expect ~3395 / ~7430 (vs 3425 / 7463).
- evaluator: valid: 3418 us/step at B=1 and 7449 at B=8 over HTTP (online (submit/poll)); in process 3363 us/step at B=1 (generate, MBU 0.197)

### #17 s0017: Track A: gate-D strip L2 prefetch at job start (GDL2)

- status: **valid**, http_step_us_b1 = 3609 (B=8: 7705), at 5.31 agent-hours, parent s0014, category memory
- hypothesis: track A; formats: the checkpoint's (as s0014). Each gate-D job's block issues one cp.async.bulk.prefetch.L2 of its D strip before it mixes x, so the strip GEMV reads L2 instead of HBM; bit-identical to s0014 (prefetch only). On track B it gave -31 us same-card (s0016 3418 vs s0015 3425 over HTTP); on track A the strip is 72 units (x and m halves), so expect ~-30..-40 us: ~3590 B=1, ~7680 B=8.
- evaluator: valid: 3609 us/step at B=1 and 7705 at B=8 over HTTP (online (submit/poll)); in process 3559 us/step at B=1 (generate, MBU 0.191)

### #18 s0018: Router top-8 with redux.sync on integer keys + merge-staging fix (port of a2 s0014/s0015)

- status: **valid**, http_step_us_b1 = 3380 (B=8: 7404), at 5.74 agent-hours, parent s0016, category structure
- hypothesis: track B; formats: as s0016 (int8 head sketch + exact rescoring, layer-0 token table, per-pass message gates within the gates); the change is track A: the router's top-8 selection uses order-preserving integer keys and redux.sync (max, then min index) instead of a 5-level shuffle tree per round, plus a NaN guard and the attention merge staging fix for more than 58 chunks (port of s15-mini-b200-a2 s0014/s0015). Same choice by construction; bit-identical logits and tokens (bitcheck, B=1 and B=8). Same-card A/B -35 us B=1, -125 us B=8: expect ~3385 / ~7330 (vs 3418 / 7449).
- evaluator: valid: 3380 us/step at B=1 and 7404 at B=8 over HTTP (online (submit/poll)); in process 3306 us/step at B=1 (generate, MBU 0.201)

### #19 s0019: Track A: router top-8 with redux.sync + merge-staging fix (a2 s0014/s0015 port)

- status: **valid**, http_step_us_b1 = 3561 (B=8: 7585), at 5.82 agent-hours, parent s0017, category structure
- hypothesis: track A; formats: the checkpoint's (as s0017). The router's top-8 selection with order-preserving integer keys and redux.sync, the NaN guard and the attention merge staging fix, as in s0018 but without the track-B items; bit-identical to s0017 (same choice by construction; bitcheck on the track-B build). s0018 moved -38 us B=1 / -45 us B=8 against s0016: expect ~3570 / ~7660 (vs 3609 / 7705).
- evaluator: valid: 3561 us/step at B=1 and 7585 at B=8 over HTTP (online (submit/poll)); in process 3485 us/step at B=1 (generate, MBU 0.192)

### #20 s0020: K1: a second kernel compiled for batch 1 only (same source), launched by batch

- status: **valid**, http_step_us_b1 = 3150 (B=8: 7444), at 6.55 agent-hours, parent s0018, category structure
- hypothesis: track B; formats: as s0018 (int8 head sketch + exact rescoring, layer-0 token table, per-pass message gates within the gates); the change is track A organisation: build.sh compiles mk.cu twice, once with -DKB1=1 where the batch is the constant 1 (every B >= 2 path and the two-column-group expert code are dead), and the host launches the batch-1 kernel for B = 1 and the generic kernel otherwise. Same arithmetic in the same order: bit-identical (bitcheck B=1/B=8). The 80-register generic kernel carries 524 B of spill loads, the batch-1 kernel 36 B. Same-card A/B: -206 us B=1, B=8 unchanged: expect ~3175 / ~7404 (vs 3380 / 7404).
- evaluator: valid: 3150 us/step at B=1 and 7444 at B=8 over HTTP (online (submit/poll)); in process 3102 us/step at B=1 (generate, MBU 0.214)

### #21 s0021: Track A: K1, a second kernel compiled for batch 1 only

- status: **valid**, http_step_us_b1 = 3114 (B=8: 7577), at 6.63 agent-hours, parent s0019, category structure
- hypothesis: track A; formats: the checkpoint's (as s0019). build.sh compiles mk.cu twice, once with -DKB1=1 (the batch a constant 1: B >= 2 paths and the two-column-group expert code dead), the host launches by batch; bit-identical to s0019 by construction. On track B the same change moved 3380 -> 3150 at B=1 (s0020): expect ~3330 / ~7585 (vs 3561 / 7585).
- evaluator: valid: 3114 us/step at B=1 and 7577 at B=8 over HTTP (online (submit/poll)); in process 3079 us/step at B=1 (generate, MBU 0.218)

### #22 s0022: Track A: KN, the generic kernel compiled for batch >= 2 only (batch 1 has its own kernel)

- status: **valid**, http_step_us_b1 = 3116 (B=8: 7659), at 6.88 agent-hours, parent s0021, category structure
- hypothesis: track A; formats: the checkpoint's (as s0021). The generic kernel is compiled with the batch known to be at least 2 (HDB = max(B, 2): the batch-1 paths are dead), batch 1 always runs the batch-1 kernel; bit-identical by construction (bitcheck on the track-B build). Spill loads of the generic kernel 524 -> 396 B; same-card A/B -72 us at B=8, B=1 untouched: expect ~3114 / ~7505 (vs 3114 / 7577).
- evaluator: valid: 3116 us/step at B=1 and 7659 at B=8 over HTTP (online (submit/poll)); in process 3087 us/step at B=1 (generate, MBU 0.217)

### #23 s0023: Track B items per kernel: MSGZ and L0T only in the B>=2 kernel, SKH in both

- status: **valid**, http_step_us_b1 = 3109 (B=8: 7401), at 7.07 agent-hours, parent s0020, category structure
- hypothesis: track B; formats: as s0020 (int8 head sketch + exact rescoring in both kernels; layer-0 token table and per-pass message gates now only in the generic kernel). Same-card A/B inside the batch-1 kernel against the track-A batch-1 kernel: MSGZ +126 us, L0T +79 us, SKH -8 us (with register headroom the plain gate path is faster than MSGZ's and L0T's control flow costs more than its 40 MB); in the generic kernel they still pay at B=8. TRACK tokens NAME:N / NAME:1 select an item per kernel. Expect B=1 ~3105 (track A's s0021 3114 minus the sketch), B=8 ~7444 (as s0020).
- evaluator: valid: 3109 us/step at B=1 and 7401 at B=8 over HTTP (online (submit/poll)); in process 3082 us/step at B=1 (generate, MBU 0.215)

### #24 s0024: Track A: batch-1 kernel keeps 6 NVFP4 units in flight per warp (FP4_UNR 6)

- status: **valid**, http_step_us_b1 = 3113 (B=8: 7595), at 7.43 agent-hours, parent s0021, category structure
- hypothesis: track A; formats: the checkpoint's (as s0021). Only the batch-1 kernel changes: each warp loads 6 units of expert codes and scales before it computes (was 4), so a w13 piece is 2 load round trips instead of 3 and a w2 piece 1 instead of 2; same unit order, bit-identical. In the generic kernel this spilled (612 B) and gave nothing; the batch-1 kernel has the registers (12 B of spill loads). Same-card A/B on the track-B build: -41 us at B=1: expect ~3075 / ~7577 (vs 3114 / 7577).
- evaluator: valid: 3113 us/step at B=1 and 7595 at B=8 over HTTP (online (submit/poll)); in process 3089 us/step at B=1 (generate, MBU 0.218)

### #25 s0025: Track A: batch-1 kernel re-polls the gate norm's 12 words together (BPG)

- status: **valid**, http_step_us_b1 = 3102 (B=8: 7571), at 7.51 agent-hours, parent s0021, category structure
- hypothesis: track A; formats: the checkpoint's (as s0021). Only the batch-1 kernel changes: in the norm stage after the gate's F/G step every thread waits for 12 tagged words; they are stale when the block arrives, and waiting word by word costs one relaxed-load round trip per word after the data is there. The 12 words are now reloaded together until all carry the tag (unconditional loads, registers only); bit-identical (bitcheck). In the generic kernel it lost 33 us (extra polling traffic), in the batch-1 kernel of the track-B build it gained 62 us on top of FP4_UNR 6: expect ~3055 / ~7577 (vs 3114 / 7577).
- evaluator: valid: 3102 us/step at B=1 and 7571 at B=8 over HTTP (online (submit/poll)); in process 3069 us/step at B=1 (generate, MBU 0.219)

### #26 s0026: Track B: KN, the generic kernel compiled for batch >= 2 only

- status: **valid**, http_step_us_b1 = 3118 (B=8: 7414), at 7.59 agent-hours, parent s0023, category structure
- hypothesis: track B; formats: as s0023 (sketch head in both kernels; per-pass message gates and layer-0 token table in the generic kernel). The generic kernel is compiled with the batch known to be at least 2 (batch-1 paths dead; spill loads 524 -> 396 B); bit-identical by construction (bitcheck). On this build's generic kernel KN measured -72 us (e0220) and -63 us (e0223: -248 vs -184 against track A) at B=8; on track A's it lost 48 us (s0022). Expect ~3109 / ~7335 (vs 3109 / 7401).
- evaluator: valid: 3118 us/step at B=1 and 7414 at B=8 over HTTP (online (submit/poll)); in process 3071 us/step at B=1 (generate, MBU 0.214)

### #27 s0027: EXQ: expert items taken by a block's warps from a queue (no rounds, no block syncs) at B>=2

- status: **valid**, http_step_us_b1 = 3121 (B=8: 7102), at 7.67 agent-hours, parent s0026, category structure
- hypothesis: track B; formats: as s0026; the change is track A organisation in the generic (B >= 2) kernel: w13 and w2 no longer run in rounds of 8 items x 3 warps with a block-wide reduction and sync per round (5-6 rounds per phase at B=8); every warp takes the next item of its block from a shared-memory counter, accumulates its units in the same three parts summed in the same order, runs the epilogue and takes the next. Bit-identical (bitcheck B=1 and B=8). Model: w13 41 -> ~27 us and w2 26 -> ~15 us of work per application. Same-card A/B -619 us at B=8 (7428 -> 6809); B=1 untouched (batch-1 kernel). Expect ~3118 / ~6800 (vs 3118 / 7414).
- evaluator: valid: 3121 us/step at B=1 and 7102 at B=8 over HTTP (online (submit/poll)); in process 3055 us/step at B=1 (generate, MBU 0.215)

### #28 s0028: Track A: EXQ (expert items from a per-block queue) + KN in the B>=2 kernel

- status: **valid**, http_step_us_b1 = 3092 (B=8: 7434), at 7.79 agent-hours, parent s0025, category structure
- hypothesis: track A; formats: the checkpoint's (as s0025). The generic kernel is compiled for B >= 2 (KN) and its w13 / w2 phases take their items from a per-block queue instead of rounds of 8 with block syncs (EXQ, as s0027); bit-identical to s0025 (bitcheck: logits bitwise equal at B=1 and B=8). Same-card A/B -597 us at B=8 (7697 -> 7100 in process); over HTTP s0027 kept about half of its in-process gain: expect ~3102 / ~7270 (vs 3102 / 7571).
- evaluator: valid: 3092 us/step at B=1 and 7434 at B=8 over HTTP (online (submit/poll)); in process 3053 us/step at B=1 (generate, MBU 0.220)

### #29 s0029: NG2NI: two-column-group expert rounds out of line in the B>=2 kernel

- status: **valid**, http_step_us_b1 = 3083 (B=8: 6725), at 7.87 agent-hours, parent s0027, category structure
- hypothesis: track B; formats: as s0027; track A organisation of the generic kernel: the round path for a block with an expert of more than 4 tokens (two MMA column groups; 1.7% of the union's experts at B=8, and unused when the queue path runs) becomes an out-of-line function, so its registers are not part of the common path's allocation: spill loads of the generic kernel 584 -> 32 B. Same arithmetic (the executed path at B=8 is unchanged). Same-card A/B -396 us at B=8 (6837 -> 6440 in process); over HTTP about half of in-process gains has shown: expect ~3120 / ~6900 (vs 3121 / 7102).
- evaluator: valid: 3083 us/step at B=1 and 6725 at B=8 over HTTP (online (submit/poll)); in process 3052 us/step at B=1 (generate, MBU 0.215)

### #30 s0030: Track A: NG2NI (two-column-group expert rounds out of line) in the B>=2 kernel

- status: **valid**, http_step_us_b1 = 3075 (B=8: 6805), at 7.94 agent-hours, parent s0028, category structure
- hypothesis: track A; formats: the checkpoint's (as s0028). As s0029 on track A: the two-group round path of w13 / w2 becomes an out-of-line function in the generic kernel (its registers leave the common path's allocation); same arithmetic. On track B it moved 7102 -> 6725 at B=8 over HTTP: expect ~3092 / ~7100 (vs 3092 / 7434).
- evaluator: valid: 3075 us/step at B=1 and 6805 at B=8 over HTTP (online (submit/poll)); in process 3069 us/step at B=1 (generate, MBU 0.219)

### #31 s0031: B>=2 kernel: 3 BF16 units in flight per warp in the GEMV pieces (BF_UNR 3)

- status: **valid**, http_step_us_b1 = 3112 (B=8: 6691), at 8.15 agent-hours, parent s0029, category structure
- hypothesis: track B; formats: as s0029; setting of the generic kernel only: piece_bf16 (qkv, router, gate D, unembedding strips) loads 3 units (12 x 16 B per lane) before it computes instead of 2; same unit order, bit-identical. Possible now that the generic kernel has register headroom (32 B of spill loads after NG2NI; the setting cost 38 us in the batch-1 kernel). Same-card A/B -62 us at B=8: expect ~3083 / ~6690 (vs 3083 / 6725).
- evaluator: valid: 3112 us/step at B=1 and 6691 at B=8 over HTTP (online (submit/poll)); in process 3063 us/step at B=1 (generate, MBU 0.215)

### #32 s0032: B>=2 kernel: gate-D strip staged into shared memory by TMA during the mix (GDPF)

- status: **valid**, http_step_us_b1 = 3107 (B=8: 6700), at 8.23 agent-hours, parent s0031, category structure
- hypothesis: track B; formats: as s0031; setting of the generic kernel only: each gate-D job copies its D strip into shared memory with one bulk copy (mbarrier) while the block mixes x, and the strip GEMV reads shared memory (same MMA order: bit-identical, bitcheck at 4.8 h). It lost 97 us in the old generic kernel (spills 648 B); with register headroom it measured -49 us at B=8 on the NG2NI base (not measured on top of BF_UNR 3): expect ~3090 / ~6650 (vs 3112 / 6691).
- evaluator: valid: 3107 us/step at B=1 and 6700 at B=8 over HTTP (online (submit/poll)); in process 3072 us/step at B=1 (generate, MBU 0.214)

### #33 s0033: B>=2 kernel: the mix's source words re-polled together (BPX bit 2, load_srcs)

- status: **valid**, http_step_us_b1 = 3104 (B=8: 6628), at 8.30 agent-hours, parent s0031, category structure
- hypothesis: track B; formats: as s0031; setting of the generic kernel only: load_srcs (the <= 6 source rows of every depth mix, 2 tagged words per source and thread) waits for all its words together: stale ones (the newest source, published by the previous application) are reloaded in one round trip instead of one per word; unconditional reloads, values taken only under the right tag: same values (bitcheck with the combination running). Same-card A/B at contexts 320-832: -136 us at B=8 (6783 -> 6646). Expect ~3100 / ~6590 (vs 3112 / 6691).
- evaluator: valid: 3104 us/step at B=1 and 6628 at B=8 over HTTP (online (submit/poll)); in process 3082 us/step at B=1 (generate, MBU 0.215)

### #34 s0034: Track A: B>=2 kernel with 3 BF16 units in flight per warp (BF_UNR 3)

- status: **valid**, http_step_us_b1 = 3090 (B=8: 6816), at 8.38 agent-hours, parent s0030, category structure
- hypothesis: track A; formats: the checkpoint's (as s0030). Setting of the generic kernel only, as s0031 on track B (6725 -> 6691 at B=8): piece_bf16 loads 3 units before it computes; same unit order, bit-identical. Expect ~3075 / ~6770 (vs 3075 / 6805).
- evaluator: valid: 3090 us/step at B=1 and 6816 at B=8 over HTTP (online (submit/poll)); in process 3054 us/step at B=1 (generate, MBU 0.219)

### #35 s0035: Track A: B>=2 kernel re-polls the mix's source words together (BPX bit 2)

- status: **valid**, http_step_us_b1 = 3099 (B=8: 6840), at 8.46 agent-hours, parent s0034, category structure
- hypothesis: track A; formats: the checkpoint's (as s0034). Setting of the generic kernel only, as s0033 on track B (6691 -> 6628 at B=8): load_srcs waits for all its source words together; same values. Expect ~3085 / ~6750 (vs 3090 / 6816).
- evaluator: valid: 3099 us/step at B=1 and 6840 at B=8 over HTTP (online (submit/poll)); in process 3072 us/step at B=1 (generate, MBU 0.219)

### #36 s0036: Track B batch-1 kernel = track A's (BPG) + the sketch head as an out-of-line phase

- status: **valid**, http_step_us_b1 = 3073 (B=8: 6631), at 8.66 agent-hours, parent s0033, category structure
- hypothesis: track B; formats: as s0033. Only the batch-1 kernel changes: it gets the batched gate-norm re-poll of the track-A base (BPG, s0025) and ph_final (sketch pass + exact rescoring) becomes an out-of-line function so that the head's registers stay out of the per-application allocation. Same arithmetic. Same-card A/B against the track-A batch-1 kernel: sketch head -19 us, out of line -30 us. Expect ~3050 / ~6628 (vs 3104 / 6628; track A s0030: 3075 / 6805).
- evaluator: valid: 3073 us/step at B=1 and 6631 at B=8 over HTTP (online (submit/poll)); in process 3053 us/step at B=1 (generate, MBU 0.217)

### #37 s0037: GLIN: attention gates' down-projection linearised over the raw mix sources (batch-1 kernel)

- status: **valid**, http_step_us_b1 = 3005 (B=8: 6635), at 9.17 agent-hours, parent s0036, category structure
- hypothesis: track B; formats: as s0036; data flow of the batch-1 kernel's attention gates: D [unit(x); unit(m)] is computed as (sum cx[s] D_x v_s) / rms(x) + (sum cm[s] D_m u_s) / rms(m) with the raw mix sources as MMA columns, in parallel with the depth mix instead of after it (exact in exact arithmetic: the mix is linear in its sources; the unit vectors are no longer rounded to BF16 before the projection). Within the gates, not bit-identical: against the reference on 3 x 160 own positions rel L2 median 0.033 / 0.058 / 0.024 (base 0.035 / 0.064 / 0.022), top-1 478 / 480 (477). Same-card A/B -61 us at B=1 (3041 -> 2980). Expect ~3010 / ~6630 (vs 3073 / 6631).
- evaluator: valid: 3005 us/step at B=1 and 6635 at B=8 over HTTP (online (submit/poll)); in process 2998 us/step at B=1 (generate, MBU 0.220)

### #38 s0038: Batch-1 kernel: the router's logit words re-polled together (BPX bit 3)

- status: **valid**, http_step_us_b1 = 2993 (B=8: 6635), at 9.43 agent-hours, parent s0037, category structure
- hypothesis: track B; formats: as s0037; setting of the batch-1 kernel only (track A organisation): the routing warp waits for its 4 words of router logits per lane together instead of one after another; same values. Same-card A/B -17 us and -18 us at B=1 in two sweeps. Expect ~2990 / ~6635 (vs 3005 / 6635).
- evaluator: valid: 2993 us/step at B=1 and 6635 at B=8 over HTTP (online (submit/poll)); in process 2965 us/step at B=1 (generate, MBU 0.223)

### #39 s0039: Track A: batch-1 kernel re-polls the router's logit words together (BPX bit 3)

- status: **valid**, http_step_us_b1 = 3073 (B=8: 6802), at 9.50 agent-hours, parent s0030, category structure
- hypothesis: track A; formats: the checkpoint's (as s0030). Setting of the batch-1 kernel only, as s0038 on track B (3005 -> 2993): the routing warp waits for its 4 logit words per lane together; same values, bit-identical. Expect ~3060 / ~6805 (vs 3075 / 6805).
- evaluator: valid: 3073 us/step at B=1 and 6802 at B=8 over HTTP (online (submit/poll)); in process 3057 us/step at B=1 (generate, MBU 0.221)

### #40 s0040: SMTRIM: shared memory of a block under the 132 KiB carve-out (4.9 KB of unused members dropped)

- status: **valid**, http_step_us_b1 = 2960 (B=8: 6610), at 9.94 agent-hours, parent s0038, category memory
- hypothesis: track B; formats: as s0038; track A organisation of both kernels: two unused members of the block's shared-memory struct are removed (138,256 -> 133,408 B). Shared memory and the L1 data cache are carved from one per-SM store in steps; padding experiments showed the footprint itself costs (+76 KB: +658 us at B=8, +226 us at B=1). Under 132 KiB the carve-out drops one step and L1 grows by 32 KiB. Bit-identical (bitcheck). Same-card A/B (3 instances): -26 us B=1, -47 us B=8. Expect ~2970 / ~6590 (vs 2993 / 6635).
- evaluator: valid: 2960 us/step at B=1 and 6610 at B=8 over HTTP (online (submit/poll)); in process 2938 us/step at B=1 (generate, MBU 0.225)

### #41 s0041: Track A: SMTRIM, shared memory of a block under the 132 KiB carve-out

- status: **valid**, http_step_us_b1 = 3030 (B=8: 6842), at 10.01 agent-hours, parent s0039, category memory
- hypothesis: track A; formats: the checkpoint's (as s0039). The two unused shared-memory members are dropped (138,256 -> 133,408 B per block): the carve-out falls one step and L1 grows by 32 KiB; bit-identical. On track B this moved 2993 -> 2960 at B=1 and 6635 -> 6610 at B=8 (s0040). Expect ~3045 / ~6770 (vs 3073 / 6802).
- evaluator: valid: 3030 us/step at B=1 and 6842 at B=8 over HTTP (online (submit/poll)); in process 3018 us/step at B=1 (generate, MBU 0.223)

### #42 s0042: SM1: batch-1 kernel's shared memory sized for one token (99 KB, under the 100 KiB carve-out)

- status: **valid**, http_step_us_b1 = 2944 (B=8: 6601), at 10.09 agent-hours, parent s0040, category memory
- hypothesis: track B; formats: as s0040; track A organisation of the batch-1 kernel: its shared-memory union holds one token's columns, 64-key attention items (the B <= 2 size) and 48 merge records: 133,408 -> 99,264 B, one more carve-out step, 32 KiB more L1. Bit-identical (bitcheck; limits check to position 2031). Contexts in the batch-1 kernel capped at 3072 (evaluator: 2048). Same-card -28 us at B=1: expect ~2935 / ~6610 (vs 2960 / 6610).
- evaluator: valid: 2944 us/step at B=1 and 6601 at B=8 over HTTP (online (submit/poll)); in process 2923 us/step at B=1 (generate, MBU 0.225)

### #43 s0043: Batch-1 kernel: current key/value short conv with all loads issued first (KVC4, port of a2 s0022)

- status: **valid**, http_step_us_b1 = 2932 (B=8: 6589), at 10.17 agent-hours, parent s0042, category structure
- hypothesis: track B; formats: as s0042; track A organisation of the batch-1 kernel only: the conv of the current key/value issues all loads of its 4 channels (tagged values, history ring, taps) before the first wait (port of s15-mini-b200-a2 s0022; same per-channel arithmetic, bit-identical by bitcheck). Same-card -32 us at B=1 (+158 at B=8, so the generic kernel keeps the old code). Expect ~2915 / ~6601 (vs 2944 / 6601).
- evaluator: valid: 2932 us/step at B=1 and 6589 at B=8 over HTTP (online (submit/poll)); in process 2911 us/step at B=1 (generate, MBU 0.226)

### #44 s0044: B>=2 kernel: union token lists from per-expert token bit masks (RTMASK, port of a2 s0020)

- status: **valid**, http_step_us_b1 = 2928 (B=8: 6498), at 10.24 agent-hours, parent s0043, category structure
- hypothesis: track B; formats: as s0043; track A organisation of the generic kernel only: routing builds each union expert's token list from a bit mask set while the top-8 is written (one atomicOr) instead of scanning B x 8 choices per expert; same ascending token order, bit-identical (bitcheck). Same-card -108 us at B=8 (neutral at B=1, kept off there). Expect ~2932 / ~6500 (vs 2932 / 6589).
- evaluator: valid: 2928 us/step at B=1 and 6498 at B=8 over HTTP (online (submit/poll)); in process 2884 us/step at B=1 (generate, MBU 0.227)

### #45 s0045: Track A: SM1 + KVC4 (batch-1 kernel) + RTMASK (B>=2 kernel)

- status: **valid**, http_step_us_b1 = 2994 (B=8: 6743), at 10.33 agent-hours, parent s0041, category structure
- hypothesis: track A; formats: the checkpoint's (as s0041). The three bit-identical organisation changes measured on track B: batch-1 shared memory sized for one token (SM1, -28 us B=1), current key/value conv with loads issued first in the batch-1 kernel (KVC4, -32 us B=1), union token lists from bit masks in the generic kernel (RTMASK, -108 us B=8). Expect ~2975 / ~6700 (vs 3030 / 6842).
- evaluator: valid: 2994 us/step at B=1 and 6743 at B=8 over HTTP (online (submit/poll)); in process 2961 us/step at B=1 (generate, MBU 0.227)

### #46 s0046: Final-source check: s0044 configuration on the final tree (only default-off options added)

- status: **valid**, http_step_us_b1 = 2934 (B=8: 6497), at 11.07 agent-hours, parent s0044, category structure
- hypothesis: track B; formats: as s0044. No intended change: since s0044 the source gained only options that are off by default (RTL1, GDU, removed W13IL) and a template parameter on piece_bf16 with the old value as default. Purpose: confirm that the final committed tree builds and validates as s0044 did. Expect 2928 +- 30 / 6498 +- 40.
- evaluator: valid: 2934 us/step at B=1 and 6497 at B=8 over HTTP (online (submit/poll)); in process 2886 us/step at B=1 (generate, MBU 0.229)

### #47 s0047: Batch-1 kernel: 6 NVFP4 expert units in flight per warp (FP4_UNR 6)

- status: **valid**, http_step_us_b1 = 2879 (B=8: 6509), at 11.77 agent-hours, parent s0044, category structure
- hypothesis: track B; formats: as s0044; setting of the batch-1 kernel only: each warp loads 6 units of expert codes and scales before computing (was 4); same unit order and MMA sequence, bit-identical. On the current base (after SM1, KVC4) same-card -43 us at B=1 (2 instances). Expect ~2890 / ~6498 (vs 2928 / 6498).
- evaluator: valid: 2879 us/step at B=1 and 6509 at B=8 over HTTP (online (submit/poll)); in process 2848 us/step at B=1 (generate, MBU 0.232)

### #48 s0048: Track A: batch-1 kernel keeps 6 NVFP4 units in flight per warp (FP4_UNR 6)

- status: **valid**, http_step_us_b1 = 2949 (B=8: 6730), at 11.86 agent-hours, parent s0045, category structure
- hypothesis: track A; formats: the checkpoint's (as s0045). Setting of the batch-1 kernel only, as s0047 on track B (2928 -> 2879): 6 units of expert codes loaded per batch instead of 4; bit-identical. Expect ~2950 / ~6743 (vs 2994 / 6743).
- evaluator: valid: 2949 us/step at B=1 and 6730 at B=8 over HTTP (online (submit/poll)); in process 2927 us/step at B=1 (generate, MBU 0.230)

### #49 s0049: Final-source check 2: s0047 configuration on the final tree

- status: **valid**, http_step_us_b1 = 2878 (B=8: 6509), at 12.11 agent-hours, parent s0047, category structure
- hypothesis: track B; formats: as s0047. No intended change: since s0047 only default-off options were added (FP8_UNR template constant at its old value, TUNE16 token). Purpose: confirm that the final committed tree builds and validates as s0047. Expect 2879 +- 30 / 6509 +- 40.
- evaluator: valid: 2878 us/step at B=1 and 6509 at B=8 over HTTP (online (submit/poll)); in process 2845 us/step at B=1 (generate, MBU 0.231)

### #50 s0050: Final-source check 3: s0047 configuration on the final tree (MSGZ4, FGR options off)

- status: **valid**, http_step_us_b1 = 2878 (B=8: 6509), at 12.50 agent-hours, parent s0049, category structure
- hypothesis: track B; formats: as s0047. No intended change since s0049: two options added with defaults that reproduce the old code (MSGZ4 0, FGR 1). Purpose: the final committed tree validates. Expect 2878 +- 30 / 6509 +- 40.
- evaluator: valid: 2878 us/step at B=1 and 6509 at B=8 over HTTP (online (submit/poll)); in process 2842 us/step at B=1 (generate, MBU 0.232)

### #51 s0051: Final-source check 4: final tree (gate_ks guard, no-op at the defaults)

- status: **valid**, http_step_us_b1 = 2879 (B=8: 6511), at 12.94 agent-hours, parent s0050, category structure
- hypothesis: track B; formats: as s0047. No intended change: gate_ks() gains a guard that is constant-false at the default DKS 1. Purpose: the final committed tree validates. Expect 2878 +- 30 / 6509 +- 40.
- evaluator: valid: 2879 us/step at B=1 and 6511 at B=8 over HTTP (online (submit/poll)); in process 2842 us/step at B=1 (generate, MBU 0.232)

### #52 s0052: B>=2 kernel: w2b's 8 expert contributions re-polled together too (BPX bits 2 + 7)

- status: **valid**, http_step_us_b1 = 2880 (B=8: 6486), at 13.53 agent-hours, parent s0051, category structure
- hypothesis: track B; formats: as s0047; setting of the generic kernel only: besides the mix sources, the 8 expert contributions of a row in w2b are waited for together (stale ones reloaded in one round trip); values taken only under the right tag, same arithmetic. Same-card A/B -26 / -10 us at B=8 (3 instances, two context ranges). Expect ~2879 / ~6490.
- evaluator: valid: 2880 us/step at B=1 and 6486 at B=8 over HTTP (online (submit/poll)); in process 2833 us/step at B=1 (generate, MBU 0.233)

