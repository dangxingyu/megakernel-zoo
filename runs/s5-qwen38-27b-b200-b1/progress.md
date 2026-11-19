# s5-qwen38-27b-b200-b1: Qwen3.8-27B on GB200

Score: `http_step_us_b1` (lower is better). Budget 6.0 agent-hours. 17 submissions so far, 14 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 6406
- 1.10x the fastest general engine: 8657
- SGLang 0.5.23 (general engine): 9523
- vLLM 0.32.0 (general engine): 10110
- the engine you inherited (track A, s0048 of run s4-qwen38-27b-b200-a1): 8938

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Inherited engine s4/s0048 re-measured as this run's baseline

- status: **valid**, http_step_us_b1 = 8940 (B=8: 9808), at 0.12 agent-hours, category other
- hypothesis: track A (checkpoint BF16 formats, unchanged). The inherited s0048 engine as-is, to measure the baseline in this run's pod; expect ~8940 us at B=1 and ~9800 us at B=8 over HTTP, within 1%.
- evaluator: valid: 8940 us/step at B=1 and 9808 at B=8 over HTTP (online (submit/poll)); in process 8913 us/step at B=1 (generate, MBU 0.721)

### #2 s0002: W12: exact 12-bit weights, per-row exponent windows, escapes as sparse correction

- status: **incorrect**, no score, at 0.57 agent-hours, parent s0001, category memory
- hypothesis: track B (weights stored as exact 12-bit codes [sign][exponent - lo_row + 1][mantissa] with one 15-exponent window per output row, out-of-window weights (1.5e-4) in per-warp lists added as w*x - v0*x at each tile piece's end; activations, K/V, state unchanged). 25% fewer weight bytes; the W12 stream runs at ~6.3 TB/s vs 7.3 for BF16 in this kernel, so expect -5.5% at B=1 (8940 -> ~8450 us) and -6% at B=8 (9808 -> ~9230), all gates passing at BF16-like error.
- evaluator: incorrect: failed numeric_long
- gates that failed: numeric_long

### #3 s0003: Exact W12: 12-bit weights with per-row exponent windows, bit-identical to BF16

- status: **incorrect**, no score, at 0.87 agent-hours, parent s0001, category memory
- hypothesis: track B (weights stored as exact 12-bit codes, one 15-exponent window per output row; the 1.5e-4 out-of-window weights patched into the MMA operand registers from fixed-stride per-warp lists, so every logit is bit-identical to s0001; activations, K/V, state unchanged). 25% fewer weight bytes but the W12 stream is bound by bytes in flight: same-card events give B=1 -4.0%, B=8 -6.8%; expect ~8590 us at B=1 and ~9150 at B=8 over HTTP, gates identical to s0001.
- evaluator: incorrect: failed http_tokens
- gates that failed: http_tokens

### #4 s0004: Exact W12 (bit-identical to BF16), now also for multi-column prefill passes

- status: **valid**, http_step_us_b1 = 8670 (B=8: 9193), at 1.12 agent-hours, parent s0001, category memory
- hypothesis: track B (weights stored as exact 12-bit codes, one 15-exponent window per output row; out-of-window weights (1.5e-4) patched into the MMA operand registers, so every logit equals s0001's bit for bit; activations, K/V, state unchanged). s0003 + the W12 loop for prefill passes of 2-4 column groups (s0003 failed http_tokens there). Same-card events B=1 -2.4 %, B=8 -6.5 %: expect ~8730 us at B=1 and ~9170 at B=8 over HTTP, gates as s0001.
- evaluator: valid: 8670 us/step at B=1 and 9193 at B=8 over HTTP (online (submit/poll)); in process 8631 us/step at B=1 (generate, MBU 0.745)

### #5 s0005: Exact W12, W12-only build: unused BF16 and old W12 stream calls compiled out

- status: **error**, no score, at 1.46 agent-hours, parent s0004, category cleanup
- hypothesis: track B (same storage as s0004: exact 12-bit codes with per-row exponent windows, escapes patched into the MMA operands; logits bit-identical to s0001). One change vs s0004: the W12 build no longer contains the call sites of the unused BF16 / old W12 stream loops; same-card events 8645 -> 8591 us at B=1 (-0.6 %), B=8 9522 -> 9508. Expect ~8620 us at B=1 and ~9180 at B=8 over HTTP.
- evaluator: pod_lost: this job was running when pods [1, 2] died; it is not run a third time

### #6 s0006: Exact W12, W12-only build (resubmission of s0005, lost to pod failures)

- status: **valid**, http_step_us_b1 = 8557 (B=8: 9136), at 1.54 agent-hours, parent s0004, category cleanup
- hypothesis: track B (storage as s0004: exact 12-bit codes, per-row exponent windows, escapes patched into the MMA operands; logits bit-identical to s0001). One change vs s0004: the W12 build omits the call sites of the unused BF16 / old W12 stream loops (the inactive -DW12_RING code is not compiled). Same-card events 8645 -> ~8580 us at B=1, 9522 -> ~9495 at B=8; expect ~8610 / ~9170 over HTTP.
- evaluator: valid: 8557 us/step at B=1 and 9136 at B=8 over HTTP (online (submit/poll)); in process 8548 us/step at B=1 (generate, MBU 0.751)

### #7 s0007: Track A: activation fragments loaded only for real columns (zeros for the rest)

- status: **valid**, http_step_us_b1 = 8892 (B=8: 9812), at 1.64 agent-hours, parent s0001, category gemm
- hypothesis: track A (checkpoint BF16 formats, no format change). The GEMM stream loaded the B-operand fragments of all 8 MMA columns; lanes of columns >= NC now take zeros instead of loading (bit-identical for the real columns). At B=1 this cuts activation loads 8x: same-card events 8929 -> 8871 us (-0.65 %), B=8 unchanged. Expect ~8880 us at B=1 and ~9800 at B=8 over HTTP (s0001: 8940 / 9808).
- evaluator: valid: 8892 us/step at B=1 and 9812 at B=8 over HTTP (online (submit/poll)); in process 8836 us/step at B=1 (generate, MBU 0.727)

### #8 s0008: Exact W12 + compact one-column activation layout at B=1 (8x smaller L1 footprint)

- status: **valid**, http_step_us_b1 = 8409 (B=8: 9131), at 2.00 agent-hours, parent s0006, category memory
- hypothesis: track B (storage as s0006: exact 12-bit weights with per-row windows, bit-identical logits; GEMM input activations are still the same bf16 values, only their buffer layout changes). One change vs s0006: at one column the GEMM input fragments hold column 0 only (32 B per k-step instead of 256), so the activations' L1 footprint at B=1 shrinks 8x; separate B=1 loop instance with a constant stride. Same-card events B=1 8570 -> 8428 us (-1.7 %), B=8 unchanged: expect ~8420 us at B=1 and ~9140 at B=8 over HTTP.
- evaluator: valid: 8409 us/step at B=1 and 9131 at B=8 over HTTP (online (submit/poll)); in process 8367 us/step at B=1 (generate, MBU 0.766)

### #9 s0009: Exact W12 with 16-exponent row windows: half the escapes, less patching

- status: **valid**, http_step_us_b1 = 8263 (B=8: 9009), at 2.41 agent-hours, parent s0008, category memory
- hypothesis: track B (exact 12-bit codes, now code = exponent - lo_row over 16 exponents since every out-of-window weight is patched in from the lists; logits bit-identical to s0001). Escapes 3.79 M -> 1.92 M; the patch path cost ~400 us per B=1 pass. Same-card events B=1 8430 -> 8268 (-1.9 %), B=8 9493 -> 9378 (-1.2 %): expect ~8250 us at B=1 and ~9030 at B=8 over HTTP.
- evaluator: valid: 8263 us/step at B=1 and 9009 at B=8 over HTTP (online (submit/poll)); in process 8226 us/step at B=1 (generate, MBU 0.781)

### #10 s0010: Exact W12: escape positions counted down, loop counters back in registers

- status: **valid**, http_step_us_b1 = 8227 (B=8: 8975), at 2.75 agent-hours, parent s0009, category memory
- hypothesis: track B (same storage content as s0009: exact 12-bit codes, 16-exponent row windows; logits bit-identical to s0001). One change: the escape lists store each escape's quad as the share's remaining-quad count, so the loop compares against its own counter and carries one counter less; the counters no longer live in local memory. Same-card events B=1 8268 -> 8224 (-0.5 %), B=8 9379 -> 9321 (-0.6 %): expect ~8220 / ~8960 over HTTP.
- evaluator: valid: 8227 us/step at B=1 and 8975 at B=8 over HTTP (online (submit/poll)); in process 8202 us/step at B=1 (generate, MBU 0.782)

### #11 s0011: Exact W12, byte-permute layout: cheaper decode (5 ops per register)

- status: **valid**, http_step_us_b1 = 8194 (B=8: 8938), at 3.99 agent-hours, parent s0010, category memory
- hypothesis: track B (same 12 bits per weight, windows and escape lists as s0010, words rearranged: [sign|mantissa bytes][exponent nibbles]; logits bit-identical to s0001). The W12 loop is sensitive to its instruction count (+54 instructions per quad cost +20 %); a byte permute with sign replication decodes two weights in 5 instructions instead of ~6-7. Same-card A/B: B=1 8225 -> 8176 (-0.6 %), B=8 9321 -> 9265: expect ~8180 / ~8920 over HTTP.
- evaluator: valid: 8194 us/step at B=1 and 8938 at B=8 over HTTP (online (submit/poll)); in process 8169 us/step at B=1 (generate, MBU 0.787)

### #12 s0012: Exact W12: activation loads issued before the per-quad escape test

- status: **valid**, http_step_us_b1 = 8169 (B=8: 8908), at 4.25 agent-hours, parent s0011, category memory
- hypothesis: track B (format as s0011; logits bit-identical to s0001). One change: for quads processed singly (and pairs holding an escape), the activation fragment loads are issued before the escape test instead of after it, shortening each such quad's dependency chain. Same-card A/B: B=1 8176 -> 8151 us, B=8 9261 -> 9233 (-0.3 %): expect ~8170 / ~8910 over HTTP.
- evaluator: valid: 8169 us/step at B=1 and 8908 at B=8 over HTTP (online (submit/poll)); in process 8141 us/step at B=1 (generate, MBU 0.789)

### #13 s0013: Exact W12: compact-layout test hoisted out of the activation producers' loops

- status: **valid**, http_step_us_b1 = 8162 (B=8: 8897), at 4.55 agent-hours, parent s0012, category cleanup
- hypothesis: track B (format as s0012; logits bit-identical to s0001). One change: lmix and the attention output write the next GEMM's input through a layout test that read the column count from shared memory per element; it is now read once per function. Same-card A/B: B=1 8153 -> 8142 us, B=8 9231 -> 9216 (-0.1 / -0.2 %): expect ~8160 / ~8895 over HTTP, within the evaluator's noise.
- evaluator: valid: 8162 us/step at B=1 and 8897 at B=8 over HTTP (online (submit/poll)); in process 8144 us/step at B=1 (step, MBU 0.789)

### #14 s0014: Exact W12 + BF16 out_proj/o_proj for one-column passes (storage chosen per op)

- status: **valid**, http_step_us_b1 = 8120 (B=8: 8914), at 4.96 agent-hours, parent s0013, category memory
- hypothesis: track B (s0013's exact 12-bit storage; out_proj and o_proj additionally kept in BF16 and used by B=1 passes, where their short W12 streams (6 quads per warp, ~1 escape per warp) are slower; logits bit-identical to s0001). Same-card A/B vs s0013: B=1 8148 -> 8104 us (-0.55 %), B=8 9215 -> 9245 (+0.3 %): expect ~8120 at B=1 and ~8925 at B=8 over HTTP.
- evaluator: valid: 8120 us/step at B=1 and 8914 at B=8 over HTTP (online (submit/poll)); in process 8098 us/step at B=1 (generate, MBU 0.793)

### #15 s0015: Exact W12: 96 KB per block of the next matrix prefetched into L2 during small ops

- status: **valid**, http_step_us_b1 = 8104 (B=8: 8860), at 5.23 agent-hours, parent s0014, category scheduling
- hypothesis: track B (format as s0014; logits bit-identical to s0001). One change: at each barrier before a non-GEMM op, every block asks L2 for the first 96 KB of its share of the next weight matrix (MK_PF_FAR, 0 before; BF16 engine unchanged). Same-card A/B vs s0014: B=1 8104 -> 8088 us (-0.2 %), B=8 9248 -> 9242: expect ~8105 / ~8910 over HTTP, close to the evaluator's noise.
- evaluator: valid: 8104 us/step at B=1 and 8860 at B=8 over HTTP (online (submit/poll)); in process 8111 us/step at B=1 (generate, MBU 0.791)

### #16 s0016: Track A: grid-barrier spin back-off 200 -> 25 ns

- status: **valid**, http_step_us_b1 = 8880 (B=8: 9827), at 5.46 agent-hours, parent s0007, category sync
- hypothesis: track A (checkpoint BF16 formats). The grid barrier's wait loop backs off 25 ns instead of 200 ns between polls. On the W12 engine it gave -0.15 % at B=1 (8092 -> 8078 us, 4 same-card alternations); s4 found barrier sleep variants noise on the BF16 engine. Expect s0007's 8892 / 9812 within +-0.3 %.
- evaluator: valid: 8880 us/step at B=1 and 9827 at B=8 over HTTP (online (submit/poll)); in process 8873 us/step at B=1 (generate, MBU 0.724)

### #17 s0017: Exact W12 with the 25 ns grid-barrier back-off

- status: **valid**, http_step_us_b1 = 8095 (B=8: 8862), at 5.55 agent-hours, parent s0015, category sync
- hypothesis: track B (format as s0015; logits bit-identical to s0001). One change vs s0015: the grid barrier backs off 25 ns instead of 200 ns (track A twin s0016: noise level). Same-card events on the W12 engine: B=1 8092 -> 8078 us (-0.15 %), B=8 9240 -> 9229: expect ~8090 / ~8850 over HTTP, at the evaluator's noise level.
- evaluator: valid: 8095 us/step at B=1 and 8862 at B=8 over HTTP (online (submit/poll)); in process 8119 us/step at B=1 (step, MBU 0.792)

