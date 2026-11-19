# s11-q27-b200-b1: Qwen3.8-27B on GB200

Score: `http_step_us_b1` (lower is better). Budget 30.0 agent-hours. 29 submissions so far, 29 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 6406
- target on track A: 1.10x the fastest general engine: 8720
- SGLang 0.5.23 (general engine): 9594
- vLLM 0.32.0 (general engine): 10233
- inherited engine, track A (s9 s0022): 8450
- inherited engine, track B (s9 s0023): 7710

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: W12 rows stored scaled by exact powers of two: one decode base, factor in epilogue

- status: **valid**, http_step_us_b1 = 7557 (B=8: 8003), at 0.69 agent-hours, category memory
- hypothesis: track B; formats differing from the checkpoint: every streamed matrix in exact 12-bit storage (as inherited), now with each row stored as W[r]*2^(107-lo_r) so the decode base is one constant and the epilogue multiplies the row sum by 2^(lo_r-107) (exact: powers of two, no subnormals); out_proj/o_proj BF16 twins at B=1 as inherited. Removes the per-tile row-base table loads (cp.async + wait per tile, cold loads at each op start): ablation with constant bases measured -144 us per B=1 pass. Same-card events: B=1 7675 -> 7567 us, B=8 8128 -> 8020. Expect http_step_us_b1 ~7560 (s0023: 7668), B=8 ~8005. Logits bit-identical to track A of the same source (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8: bit_identical 1.0); all 25,621,954,560 stored weights decode to the scaled checkpoint bits and times the factor equal the checkpoint value (MK_W12_VERIFY: 0 mismatches).
- evaluator: valid: 7557 us/step at B=1 and 8003 at B=8 over HTTP (online (submit/poll)); in process 7539 us/step at B=1 (generate, MBU 0.852)

### #2 s0002: Layer-0 in_proj output from a per-token table built by the kernel itself

- status: **valid**, http_step_us_b1 = 7542 (B=8: 7969), at 0.87 agent-hours, parent s0001, category structure
- hypothesis: track B; formats differing from the checkpoint: as s0001 (exact 12-bit rows scaled by powers of two, BF16 twins of out_proj/o_proj at B=1), plus a 248,320 x 16,480 bf16 table of layer 0's in_proj output (8.2 GB, made at load by this kernel's own in_proj op on every token id; all rows verified against one-column passes: 0 mismatches). Layer 0's input is the token embedding, so its in_proj (qkv|z|b|a, 126 MB of W12 weights) depends on the token only: the embed op copies the row and the in_proj op and its barrier go. Same-card events: B=1 7560 -> 7546, B=8 8006 -> 7979. Expect http ~7542 (s0001 7557) and B=8 ~7976 (8003). Logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8). Also: submit() rejects token ids outside the vocabulary and requests longer than the context.
- evaluator: valid: 7542 us/step at B=1 and 7969 at B=8 over HTTP (online (submit/poll)); in process 7524 us/step at B=1 (generate, MBU 0.854)

### #3 s0003: Greedy passes: certified int8 sketch of lm_head + exact rescoring of candidate tiles

- status: **valid**, http_step_us_b1 = 7525 (B=8: 7950), at 1.36 agent-hours, parent s0002, category gemm
- hypothesis: track B; formats differing from the checkpoint: as s0002, plus an int8 per-row copy of lm_head (1.27 GB) with certified bound constants and a BF16 copy for rescoring. Served/generate passes (no logits) bound every row's head output from the int8 sketch (x as two exact int8 digits, integer tensor-core sums, Cauchy-Schwarz bound + 2^-10 ||w|| ||x|| for the exact path's float32 accumulation) and recompute exactly only tiles whose upper bound reaches (1-2^-7-2^-20) x the best lower bound (proof: notes/head_sketch.md), with the head GEMM's own pieces, MMA chains and additions; step() keeps the exact head. Measured head on real text: sketch ~175 us + rescore ~20 us vs 274 exact. Same build, served path (online_time.py): B=1 7586 -> 7542, B=8 7958 -> 7927. Expect http ~7500 (s0002 7542), B=8 ~7940. Greedy tokens identical to the exact head's argmax at 9216/9216 teacher-forced positions (105 with exact bf16 ties) at B=1 and B=8, and with every tile forced to be a candidate; step() logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8).
- evaluator: valid: 7525 us/step at B=1 and 7950 at B=8 over HTTP (online (submit/poll)); in process 7533 us/step at B=1 (generate, MBU 0.850)

### #4 s0004: Head sketch in its own launch after the main kernel; x statistics in the sketch

- status: **valid**, http_step_us_b1 = 7495 (B=8: 7936), at 1.80 agent-hours, parent s0003, category launch
- hypothesis: track B; formats as s0003 (12-bit rows scaled by powers of two, layer-0 table, int8 lm_head sketch + BF16 copy for rescoring). The sketch/rescore ops compiled into the main kernel cost +28-59 us per pass through code generation (s0003's in-process step_us +34 us over s0002 with the exact head); now they run in a second kernel launched right after the main one (same stream and barrier counter), the main kernel's code is the exact-head engine's again, and the x statistics are reduced inside the sketch (no epilogue change). Same card, generate on real text: s0002 kernel 7540 / 7956 us, this build 7488 / 7938 (B=1 / B=8); served path 7484 / 7924. Expect http ~7485 (s0003 7525, s0002 7542), B=8 ~7935. Greedy tokens == exact head argmax at 9216/9216 teacher-forced positions (105 exact ties); step() logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8). Two launches per greedy pass (single_launch milestone will not hold).
- evaluator: valid: 7495 us/step at B=1 and 7936 at B=8 over HTTP (online (submit/poll)); in process 7473 us/step at B=1 (generate, MBU 0.855)

### #5 s0005: W12 stream: next escape position kept in a register, no shuffle per iteration

- status: **valid**, http_step_us_b1 = 7337 (B=8: 7763), at 2.50 agent-hours, parent s0004, category gemm
- hypothesis: track B; formats as s0004 (exact 12-bit rows scaled by powers of two, layer-0 table, int8 head sketch). Ablations on the RS build: no escape code at all -280 us per B=1 pass, lists empty but loaded -106, lists L2-hot -20: most of the escape cost was the per-iteration shuffle + test of the 2-quad fast path, not patches or the list latency. The next escape's quad is now read once and refreshed only after a slow-path quad. Same card, generate on real text vs the s0004 build: B=1 7479 -> 7351, B=8 7955 -> 7782. Expect http ~7370 (s0004 7495), B=8 ~7765 (7936). Logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8); greedy through the sketch == exact argmax 9216/9216; served == greedy; 20 x 128-step runs deterministic.
- evaluator: valid: 7337 us/step at B=1 and 7763 at B=8 over HTTP (online (submit/poll)); in process 7315 us/step at B=1 (generate, MBU 0.872)

### #6 s0006: W12 build: epilogue norm gains and row factors copied to shared memory at op start

- status: **valid**, http_step_us_b1 = 7327 (B=8: 7763), at 4.89 agent-hours, parent s0005, category memory
- hypothesis: track B; formats as s0005 (exact 12-bit rows scaled by powers of two, layer-0 table, int8 head sketch). The post-stream epilogue of every GEMM loaded its rows' norm gains and 12-bit row factors from global memory; the constants of the block's tiles now go to shared memory with cp.async at op start (8 warps), waited for with the residual rows at the stream end. Bound with constants -57 us; this form, same card, build.sh flags, 3 rounds: B=1 7394 -> 7381, B=8 7810 -> 7801. Expect http ~7325 (s0005 7337), B=8 ~7755 (7763). Logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8); greedy through the sketch == exact argmax 9216/9216; served == greedy.
- evaluator: valid: 7327 us/step at B=1 and 7763 at B=8 over HTTP (online (submit/poll)); in process 7308 us/step at B=1 (generate, MBU 0.872)

### #7 s0007: No BF16 twins of out_proj/o_proj: W12 at every batch size, BF16 loops not compiled

- status: **valid**, http_step_us_b1 = 7304 (B=8: 7741), at 5.00 agent-hours, parent s0006, category memory
- hypothesis: track B; formats: every streamed matrix in exact 12-bit storage (rows scaled by powers of two) at every batch size, layer-0 table, int8 head sketch; the BF16 copies of out_proj / o_proj that one-column passes used are gone (4 GB less memory). With the s0005 loop W12 is now faster than BF16 there too, and without the twins the BF16 stream call sites (-DW12_MIXED) are dropped from the build. Same card, build.sh flags, 3 rounds on top of s0006: B=1 7381 -> ~7366, B=8 unchanged (twins were B=1 only). Expect http ~7312 (s0006 7327), B=8 ~7763. Logits bit-identical to track A (bitcheck 1152 positions, B=1 and B=8); greedy through the sketch == exact argmax 9216/9216; served == greedy.
- evaluator: valid: 7304 us/step at B=1 and 7741 at B=8 over HTTP (online (submit/poll)); in process 7285 us/step at B=1 (generate, MBU 0.876)

### #8 s0008: Head sketch: per-block tile counts balanced by measured block speed at load

- status: **valid**, http_step_us_b1 = 7313 (B=8: 7738), at 5.30 agent-hours, parent s0007, category scheduling
- hypothesis: track B (formats as s0007: W12 weights, L0 table, int8 head sketch on greedy passes; logits bit-identical to track A). The sketch's blocks are systematically 5-10 us late on the same SMs every pass (lag correlation 0.93); at load the engine times each block's stream over 2x12 greedy passes and gives it tiles in proportion to its speed (94-106 tiles instead of 102). Same-job A/B: -5 us B=1, -4 us B=8; expect ~7299 / ~7737.
- evaluator: valid: 7313 us/step at B=1 and 7738 at B=8 over HTTP (online (submit/poll)); in process 7285 us/step at B=1 (generate, MBU 0.874)

### #9 s0009: Head sketch in 7 bits per weight (1.11 GB instead of 1.27 GB), still certified exact

- status: **valid**, http_step_us_b1 = 7291 (B=8: 7712), at 5.58 agent-hours, parent s0008, category memory
- hypothesis: track B (formats as s0008 except the greedy-pass head sketch: q = round(w/s), s = max|w|/63, stored as q+64 in 7 of every 8 bytes, unsigned x signed IMMA with the exact digit-sum correction; logits bit-identical to track A, greedy tokens exact: candidates 6.9 tiles mean instead of 2.3, all rescored with the exact head arithmetic). The sketch streams 0.159 GB less; same-job A/B (generate, real text, 3 rounds) -18..-21 us B=1, -19..-23 B=8: expect ~7293 / ~7718.
- evaluator: valid: 7291 us/step at B=1 and 7712 at B=8 over HTTP (online (submit/poll)); in process 7266 us/step at B=1 (generate, MBU 0.875)

### #10 s0010: Barrier L2 prefetch of the next GEMM's first piece per block: 144 KB -> 176 KB

- status: **valid**, http_step_us_b1 = 7273 (B=8: 7683), at 6.17 agent-hours, parent s0009, category memory
- hypothesis: track B (formats as s0009; logits bit-identical to track A: prefetch only). The barrier before each GEMM asks L2 for the first bytes of every block's share; re-swept on the current 12-bit build (env only, cards 1 and 3, 3 rounds each): 176 KB -14..-15 us B=1, -25..-26 B=8 against 144 KB; 160 KB similar at B=1, 192 KB worse. Expect ~7277 / ~7687.
- evaluator: valid: 7273 us/step at B=1 and 7683 at B=8 over HTTP (online (submit/poll)); in process 7250 us/step at B=1 (generate, MBU 0.877)

### #11 s0011: Head kernel: argmax op folded into the finish op (one grid barrier less)

- status: **valid**, http_step_us_b1 = 7275 (B=8: 7697), at 7.32 agent-hours, parent s0010, category sync
- hypothesis: track B (formats as s0010; logits bit-identical to track A, greedy tokens exact). In the greedy-pass head kernel each finishing block now unpacks and clears its own column's argmax key instead of block 0 doing it in a separate op behind a grid barrier. nsys head-kernel median 194.0 -> 192.2 us (B=1), 209 -> 207 (B=8): expect ~-2 us, inside the HTTP noise (~7271 / ~7681).
- evaluator: valid: 7275 us/step at B=1 and 7697 at B=8 over HTTP (online (submit/poll)); in process 7250 us/step at B=1 (generate, MBU 0.877)

### #12 s0012: Head kernel: exact rescoring chain relayed in 40 turns of 8 k-steps (was 64 of 5)

- status: **valid**, http_step_us_b1 = 7266 (B=8: 7682), at 7.50 agent-hours, parent s0011, category gemm
- hypothesis: track B (formats as s0011; logits bit-identical to track A, greedy tokens exact: same MMA chain, only the warp hand-overs change). The rescoring of candidate tiles replays the head GEMM's 320-MMA chain through the block's warps; fewer, longer turns save hand-overs. nsys head-kernel median 192.0-192.5 -> 188.6-189.1 us (B=1), 207.5-208 -> 204.5-205 (B=8): expect ~-3 us, inside the HTTP noise (~7271 / ~7692).
- evaluator: valid: 7266 us/step at B=1 and 7682 at B=8 over HTTP (online (submit/poll)); in process 7259 us/step at B=1 (generate, MBU 0.876)

### #13 s0013: Head sketch: power-of-two x scale, digits by exact multiply instead of division

- status: **valid**, http_step_us_b1 = 7263 (B=8: 7675), at 8.25 agent-hours, parent s0012, category gemm
- hypothesis: track B (formats as s0012; logits bit-identical to track A, greedy tokens exact: the bound holds with |x/sx - q| <= 1/2 exactly). The sketch's per-pass x digits used one IEEE division per x value (40 per thread at B=8); with sx rounded up to a power of two the quotient is an exact multiply. nsys head-kernel median -1.2 us (B=1), -2.7 us (B=8): expect ~-1 / -3 us, inside the HTTP noise (~7265 / ~7679).
- evaluator: valid: 7263 us/step at B=1 and 7675 at B=8 over HTTP (online (submit/poll)); in process 7248 us/step at B=1 (generate, MBU 0.877)

### #14 s0014: B<=3 attention barrier prefetch: 256 KB of o_proj + the K chunk (12-bit build)

- status: **valid**, http_step_us_b1 = 7261 (B=8: 7674), at 9.06 agent-hours, parent s0013, category memory
- hypothesis: track B (formats as s0013; logits bit-identical to track A: prefetch only). Before the one-op attention layer the barrier asked L2 for the whole o_proj share of every block at B<=3; at the scored workload's contexts (to 1088) asking for 256 KB of it plus the attention blocks' K chunk (as B>=4 already does) is faster: nsys main-kernel median over 1024 greedy steps 7084.5 -> 7081.3 us. Expect ~-3 us at B=1 (inside the HTTP noise, ~7260), B=8 unchanged.
- evaluator: valid: 7261 us/step at B=1 and 7674 at B=8 over HTTP (online (submit/poll)); in process 7246 us/step at B=1 (generate, MBU 0.878)

### #15 s0015: B<=3: attention barrier asks for 20 K chunks per unit (was 5), 12-bit build

- status: **valid**, http_step_us_b1 = 7258 (B=8: 7688), at 9.61 agent-hours, parent s0014, category memory
- hypothesis: track B (formats as s0014; logits bit-identical to track A: prefetch table only). At the barrier before the one-op attention layer, 20 instead of 5 blocks per attention unit ask L2 for a K chunk instead of an o_proj piece. nsys main-kernel median over 1024 greedy steps (contexts to 1088), three cards: -7.3 / -7.5 / -8.6 us; -1.5 us at contexts <= 364. Expect ~-5 us at B=1 over HTTP (~7256), B=8 unchanged (4 blocks per unit there).
- evaluator: valid: 7258 us/step at B=1 and 7688 at B=8 over HTTP (online (submit/poll)); in process 7240 us/step at B=1 (generate, MBU 0.877)

### #16 s0016: B<=3 prefetch table retune: no K past chunk 5, whole o_proj share, 40 idle blocks

- status: **valid**, http_step_us_b1 = 7249 (B=8: 7682), at 10.51 agent-hours, parent s0015, category memory
- hypothesis: track B (formats as s0015; logits bit-identical to track A: prefetch table only, 12-bit build at B<=3). One retune of the barrier prefetch table measured jointly and by part (notes/journal.md): blocks with K chunks past the 5th (beyond any scored context) ask for nothing instead of unused K, the other blocks ask for their whole o_proj share at the attention barrier, and 40 instead of 32 idle blocks prefetch out_proj before the recurrent op. nsys main-kernel median: -11.0 us at contexts to 1088 (two cards), -10.8 at <= 364. Expect ~-11 us at B=1 over HTTP (~7247), B=8 unchanged.
- evaluator: valid: 7249 us/step at B=1 and 7682 at B=8 over HTTP (online (submit/poll)); in process 7234 us/step at B=1 (generate, MBU 0.879)

### #17 s0017: Head sketch: 94 % of tiles in 6 bits, frequent candidates stay 7-bit (0.96 GB)

- status: **valid**, http_step_us_b1 = 7244 (B=8: 7684), at 13.27 agent-hours, parent s0016, category memory
- hypothesis: track B (formats as s0016 except the greedy-pass head sketch: tiles that were rarely candidates on own text (engine/sk6_tiles.bin, 93.9 %) stored in 6 bits per weight, the rest in 7; logits bit-identical to track A, greedy tokens exact: every row's bound uses its own quantization, a 6-bit tile that becomes a candidate is rescored exactly). The sketch streams ~0.15 GB less; more candidates cost part of it back. nsys same job: head kernel 189 -> 180 us, pass -7 us at B=1, ~-3 at B=8: expect ~7242 / ~7679.
- evaluator: valid: 7244 us/step at B=1 and 7684 at B=8 over HTTP (online (submit/poll)); in process 7225 us/step at B=1 (generate, MBU 0.880)

### #18 s0018: Head sketch at B>1: all-7-bit copy through the plain kernel instance (B=1 keeps 6/7)

- status: **valid**, http_step_us_b1 = 7239 (B=8: 7677), at 13.75 agent-hours, parent s0017, category memory
- hypothesis: track B (formats as s0017; logits bit-identical to track A, greedy tokens exact). At B=8 the mixed 6/7-bit sketch made rare passes with hundreds of candidates x 8 columns (head p99 316 us); passes of B>1 now use a second, all-7-bit copy of the sketch through a separately compiled 7-bit instance of the sketch kernel. nsys head mean at B=8 209.6 -> 206.2 us (p99 316 -> 213), B=1 unchanged: expect B=8 ~-4 us (~7680, inside its noise), B=1 ~7244.
- evaluator: valid: 7239 us/step at B=1 and 7677 at B=8 over HTTP (online (submit/poll)); in process 7220 us/step at B=1 (generate, MBU 0.876)

### #19 s0019: Mixed head sketch: per-block tile counts calibrated at load again (B=1)

- status: **valid**, http_step_us_b1 = 7245 (B=8: 7680), at 14.62 agent-hours, parent s0018, category scheduling
- hypothesis: track B (formats as s0018; logits bit-identical to track A, greedy tokens exact: the split of tiles over blocks does not change any sum). With 6- and 7-bit tiles mixed across blocks, the load-time calibration of tiles per block by measured block speed (off since s0017) pays again: nsys steady-state head mean at B=1 182.9 -> 180.4 us. Expect ~-2.5 us at B=1 (~7236, inside the noise), B=8 unchanged.
- evaluator: valid: 7245 us/step at B=1 and 7680 at B=8 over HTTP (online (submit/poll)); in process 7251 us/step at B=1 (generate, MBU 0.879)

### #20 s0020: B<=3 near prefetch per matrix: gate_up 160 KB, down 192 KB (others 176)

- status: **valid**, http_step_us_b1 = 7238 (B=8: 7679), at 15.17 agent-hours, parent s0019, category memory
- hypothesis: track B (formats as s0018; logits bit-identical to track A: prefetch table only). The barrier before each GEMM asks L2 for the first bytes of every block's share; per-matrix sizes measured at the scored contexts (nsys main-kernel median, two jobs x 3 runs): gate_up 160 KB and down 192 KB instead of 176 give -1.0 / -1.3 us. Expect ~-1 us at B=1, inside the HTTP noise (~7238), B=8 unchanged.
- evaluator: valid: 7238 us/step at B=1 and 7679 at B=8 over HTTP (online (submit/poll)); in process 7225 us/step at B=1 (generate, MBU 0.879)

### #21 s0021: Confirmation: s0020 configuration with the later experiments compiled out (no change)

- status: **valid**, http_step_us_b1 = 7242 (B=8: 7679), at 15.88 agent-hours, parent s0020, category cleanup
- hypothesis: track B (formats as s0020; logits bit-identical to track A). The workspace after s0020 only adds experiment code behind build flags that are off (PDL, SK_SCAN3, SK_LAST, SK_XS, SK_W0X, SK_PFW, SK76_SHARE) and tooling; the built library's kernels are those of s0020 (head and main kernel code unchanged). Expect the same numbers as s0020 within noise (~7238 / ~7679).
- evaluator: valid: 7242 us/step at B=1 and 7679 at B=8 over HTTP (online (submit/poll)); in process 7227 us/step at B=1 (generate, MBU 0.879)

### #22 s0022: B=1 head kernel: last rescoring block finishes the pass (two grid barriers less)

- status: **valid**, http_step_us_b1 = 7259 (B=8: 7708), at 16.67 agent-hours, parent s0021, category sync
- hypothesis: track B (formats as s0020; logits bit-identical to track A, greedy tokens exact). In one-sequence greedy passes the head kernel's finish op and both barriers around it are replaced by the last block to leave the rescoring (one atomic counter) unpacking the argmax key and publishing the token; B>1 passes keep the finish op. nsys (var_gen, 3 runs): B=1 head 182.4 -> 180.6 us, B=8 unchanged. Expect ~-2 us at B=1 (~7236, inside the noise).
- evaluator: valid: 7259 us/step at B=1 and 7708 at B=8 over HTTP (online (submit/poll)); in process 7245 us/step at B=1 (generate, MBU 0.877)

### #23 s0023: Final state: s0020 configuration (SK_LAST and other experiments compiled out)

- status: **valid**, http_step_us_b1 = 7261 (B=8: 7711), at 16.83 agent-hours, parent s0022, category cleanup
- hypothesis: track B (formats as s0020; logits bit-identical to track A, greedy tokens exact). The shipped engine after reverting SK_LAST (s0022): build flags of s0020, experiment code behind flags that are off, SK_LAST plumbing in api.py inactive without its flag. Expect s0020's numbers within the evaluation's noise (~7240 / ~7680).
- evaluator: valid: 7261 us/step at B=1 and 7711 at B=8 over HTTP (online (submit/poll)); in process 7227 us/step at B=1 (generate, MBU 0.878)

### #24 s0024: B=1 head sketch: frequent-candidate tiles in 8 bits instead of 7 (6 % of tiles)

- status: **valid**, http_step_us_b1 = 7252 (B=8: 7705), at 17.80 agent-hours, parent s0023, category memory
- hypothesis: track B (formats as s0020 except: in the B=1 mixed head sketch the 6.1 % of tiles outside the 6-bit set are stored as plain int8 instead of 7 bits; logits bit-identical to track A, greedy tokens exact). +9.5 MB of sketch, but these are the frequent candidates: census mean candidates 33 -> 11, p99 768 -> 202, so the rescoring shrinks. nsys same job: B=1 head median 182.3 -> 177.2 us, B=8 unchanged. Expect ~-4 us at B=1 relative to the current evaluation level (s0023 7261; ~7257), B=8 unchanged (~7711).
- evaluator: valid: 7252 us/step at B=1 and 7705 at B=8 over HTTP (online (submit/poll)); in process 7219 us/step at B=1 (generate, MBU 0.877)

### #25 s0025: Head sketch: the 6/8-bit mixed sketch for every batch size (no all-7-bit copy)

- status: **valid**, http_step_us_b1 = 7255 (B=8: 7707), at 18.17 agent-hours, parent s0024, category memory
- hypothesis: track B (formats as s0024 except that passes of B>1 now also use the mixed 6/8-bit head sketch instead of an all-7-bit copy; logits bit-identical to track A, greedy tokens exact). With the hot tiles in 8 bits (s0024) the mixed sketch's candidates no longer explode at 8 columns: nsys head mean at B=8 206.3 -> 201.5 us; B=1 unchanged. Expect B=8 ~-5 us against the current level (s0024 7705 -> ~7700), B=1 ~7252.
- evaluator: valid: 7255 us/step at B=1 and 7707 at B=8 over HTTP (online (submit/poll)); in process 7242 us/step at B=1 (generate, MBU 0.877)

### #26 s0026: Head sketch prologue: x read as aligned u32 pairs instead of u16 scalars

- status: **valid**, http_step_us_b1 = 7250 (B=8: 7692), at 18.97 agent-hours, parent s0025, category memory
- hypothesis: track B (formats as s0025: W12 weights, mixed 6/8-bit certified head sketch; logits bit-identical to track A). The sketch prologue's x loads (2 B scalars, 152 blocks reading the same x) become 4-byte pairs with the same values; head kernel measured -3 us at B=1 and -9.5 us at B=8 (3 nsys rounds), so expect ~7252 / ~7698 against s0025's 7255 / 7707 (within the evaluation's +-20 us drift).
- evaluator: valid: 7250 us/step at B=1 and 7692 at B=8 over HTTP (online (submit/poll)); in process 7235 us/step at B=1 (generate, MBU 0.877)

### #27 s0027: Recurrent op re-reads B and spec flag from shared memory (no spill reloads)

- status: **valid**, http_step_us_b1 = 7221 (B=8: 7698), at 21.96 agent-hours, parent s0026, category memory
- hypothesis: track B (formats as s0026: W12 weights, mixed 6/8-bit certified head sketch; logits bit-identical to track A). lmix_work kept B*96 and the spec flag live; ptxas spilled them at entry and reloaded them from local memory on the critical path of every recurrent layer (~0.7 us x 48). Re-reading them from shared memory where used: main kernel -31 us at B=1 (-38 at the scored contexts), +10 at B=8 (code generation). Expect ~7220 / ~7700 against s0026's 7250 / 7692.
- evaluator: valid: 7221 us/step at B=1 and 7698 at B=8 over HTTP (online (submit/poll)); in process 7202 us/step at B=1 (generate, MBU 0.881)

### #28 s0028: ptxas register-usage level 10 for the track-B build (same source)

- status: **valid**, http_step_us_b1 = 7175 (B=8: 7681), at 25.03 agent-hours, parent s0027, category other
- hypothesis: track B (formats as s0027: W12 weights, mixed 6/8-bit certified head sketch; logits bit-identical to track A). The same source compiled with ptxas --register-usage-level=10 (another register allocation of the whole kernel; arithmetic unchanged, bitcheck identical): main kernel -42 us at B=1 (3 rounds), -38..-51 at the scored contexts, -21..-23 at B=8 in same-card nsys A/B. Expect ~7180 / ~7678 against s0027's 7221 / 7698.
- evaluator: valid: 7175 us/step at B=1 and 7681 at B=8 over HTTP (online (submit/poll)); in process 7161 us/step at B=1 (generate, MBU 0.887)

### #29 s0029: ptxas -disable-optimizer-consts added to the track-B build (same source)

- status: **valid**, http_step_us_b1 = 7170 (B=8: 7652), at 25.98 agent-hours, parent s0028, category other
- hypothesis: track B (formats as s0028: W12 weights, mixed 6/8-bit certified head sketch; logits bit-identical to track A). The same source with ptxas -disable-optimizer-consts in addition to register-usage level 10 (code generation only; bitcheck identical): main kernel -7 us at B=1 (3 rounds), -4 at the scored contexts, -9..-15 at B=8 in same-card nsys A/B. Expect ~7170 / ~7670 against s0028's 7175 / 7681, i.e. within the evaluation's noise at B=1.
- evaluator: valid: 7170 us/step at B=1 and 7652 at B=8 over HTTP (online (submit/poll)); in process 7154 us/step at B=1 (generate, MBU 0.886)

