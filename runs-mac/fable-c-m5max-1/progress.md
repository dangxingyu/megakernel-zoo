# fable-c-m5max-1: Inkling-2-Mini on an Apple M5 Max (40-core GPU, 48 GB)

Score: `step_us_b1` (lower is better). Budget 6.0 agent-hours. Track C. 21 submissions, 20 valid.

Reference lines of this run:

- llamacpp-q4: 25130 (natural text at batch 1: 25030 us per token)
- llamacpp-q4-ds: 20556 (natural text at batch 1: 16323 us per token)
- llamacpp-q4-pr: 25016 (natural text at batch 1: 25082 us per token)
- mlxlm-native: 30359 (natural text at batch 1: 30486 us per token)
- omlx-oq4e: 16744 (natural text at batch 1: 16779 us per token)
- omlx-oq4e-ds: 13767 (natural text at batch 1: 8307 us per token)
- target: 1.10x the fastest general engine: 12515 (at batch 1)

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Own K-row target pass in MLX + Metal kernels, no speculation yet

- status: **valid**, step_us_b1 = 26852, at 0.28 agent-hours, category structure, quick evaluation
- hypothesis: mlx-lm removed from the timed path: the target as our own pass (BF16 GEMV, NVFP4/MXFP8 expert, attention and router Metal kernels; glue in lazy MLX ops), one token per step, no draft yet. Expect all gates to pass (error vs float reference about 0.4x of mlx-lm's bf16 path) and step_us about 27500 (mlx-lm 30359); this is a correctness checkpoint.
- evaluator: valid: 26852 us/step at B=1 (MBU 0.365)

### #2 s0002: DSpark draft + exact K-row verify in submit/poll (lazy MLX graphs)

- status: **valid**, step_us_b1 = 15646, at 0.53 agent-hours, category structure
- hypothesis: First speculative engine: the DSpark head as MLX ops + own GEMV kernels (5 ms per cycle), verify pass of K = 1 + kd rows (kd = longest draft prefix with cumulative draft probability >= 0.3), two host syncs per cycle. In-process served: random-token prompt 27 ms/token at 1.37 tokens/cycle, prose 13.9 ms/token at 3.26, code about 11. Expect http_step_us_b1 about 24000-27000 (no gain over one-token decoding on synthetic prompts yet), natural_b1 about 14000-16000, gates generate and http_tokens_b1 exact.
- evaluator: valid: 15646 us/step at B=1 over HTTP (online (submit/poll)); in process 26700 us/step at B=1 (step, MBU 0.367)

### #3 s0003: Compiled pass, shared expert reads (2 pairs per slot), n-gram copy proposals

- status: **valid**, step_us_b1 = 10545, at 0.77 agent-hours, category structure
- hypothesis: Three changes after s0002 (15646 synthetic at 2.90 tokens/event, 12704 natural at 4.21): (1) the K-row pass traced once per K with mx.compile (K=1 27.3 -> 20.7 ms, K=8 50.7 -> 47 ms in process); (2) expert kernels read each distinct expert once per slot of 2 (row, expert) pairs, bit-identical per row (K=8 on prose: 33 distinct of 64 picks, 26.5 -> 19.3 ms); (3) when the committed suffix (>= 5 tokens) occurred before, its continuation is proposed instead of the DSpark draft and 8 rows are verified (no draft cost in those cycles). In process: prose 13.9 -> 10.2 ms/token, a random-token prompt that loops 27 -> 5 ms/token. Expect natural_b1 about 9500-10500 and http_step_us_b1 about 11000-13000 if the synthetic continuations repeat as my random prompts do (tokens/event up from 2.9); step_us about 20500.
- evaluator: valid: 10545 us/step at B=1 over HTTP (online (submit/poll)); in process 18717 us/step at B=1 (generate, MBU 0.521)

### #4 s0004: Depth-mix kernel, half activations (lower error), 16-row passes for copies

- status: **valid**, step_us_b1 = 14178, at 0.97 agent-hours, category structure
- hypothesis: After s0003 (10545 synthetic at 3.04 tokens/event; natural 8335 at 5.08, 410 of 866 events at the 8-token cap; numeric gate at 1.1x / 1.6x of the BF16 reference with a 2x limit): (1) the depth attention of learned queries (stack + sums + softmax in MLX, 7.5 ms of glue per pass) as one Metal kernel per mix: pass K=4 29.2 -> 26.4 ms, K=8 about 47 -> 36.5; (2) activations enter the kernels as IEEE half instead of BF16 bits (same bytes, 11 significant bits instead of 8): median logit error vs a float32 run 12x lower in my test, expect gates numeric / kl well under the BF16 reference; (3) copy proposals verify 16 rows (15 copied tokens) while the previous copy cycle was accepted in full. In process (1024 tokens): prose 10.15 -> 8.55 ms/token, code 7.37 -> 5.55, looping random prompt 4.97 -> 3.55. Expect natural_b1 about 7000-7400, http_step_us_b1 about 8500-9500, step_us about 18500.
- evaluator: valid: 14178 us/step at B=1 over HTTP (online (submit/poll)); in process 16876 us/step at B=1 (step, MBU 0.581)

### #5 s0005: Fused glue kernels, two-tier K/V cache, 8-bit copies for the draft

- status: **valid**, step_us_b1 = 9758, at 1.24 agent-hours, category structure
- hypothesis: s0004: natural 6166 (1.35x) but synthetic 14178: the three synthetic repeats are different prompts (6227 / 14178 / 15767); the two that do not fall into a loop run at 1.9 tokens/event, one cycle = draft 5.1 + verify of ~3 rows 24 ms. This submission cuts the cost of such a cycle: (1) glue between the big kernels fused (depth mix with the RMS norms folded in, message gates, k/v convs + clips, short conv + residual + window, the attention output gate): pass K=1 18.4 -> 16.3, K=3 24 -> 21.9, K=8 36.2 -> 32.7 ms; (2) K/V rows committed into a small recent tier instead of slice updates of the whole cache (commit 1.1 -> 0.3 ms); (3) the draft's matrices (3 layers, unembed, fold) read from 8-bit per-row-scaled copies, draft only: 5.2 -> 3.7 ms per cycle with identical acceptance on my prompts. All rows still bit-identical to step(). In process: prose 8.55 -> 7.34 ms/token, code 5.55 -> 5.02. Expect natural_b1 about 5400-5700, the non-looping synthetic runs about 12.8-14.2 ms/token (so http_step_us_b1 13000 +- 1500 unless two of three loop), step_us about 16400.
- evaluator: valid: 9758 us/step at B=1 over HTTP (online (submit/poll)); in process 14948 us/step at B=1 (step, MBU 0.656)

### #6 s0006: Data point: DSpark drafts only (copy proposals off), compiled draft

- status: **valid**, step_us_b1 = 11145, at 1.47 agent-hours, category scheduling
- hypothesis: A control, not an attempt at the best number: the s0005 engine with the copy (n-gram) proposals switched off, so that every draft comes from the checkpoint's DSpark head (and the draft traced with mx.compile, 3.7 -> 3.6 ms). Target numerics are identical to s0005, so the same continuations are decoded; only the proposals differ. Expect natural_b1 about 9000-11000 (tokens/event about 3, no 16-token events) and http_step_us_b1 about 12500-14500 (tokens/event about 2), i.e. around the 1.10x line on synthetic and below oMLX+DSpark on natural: the difference to s0005 is what the copy proposals contribute on looping continuations.
- evaluator: valid: 11145 us/step at B=1 over HTTP (online (submit/poll)); in process 14916 us/step at B=1 (step, MBU 0.657)

### #7 s0007: Final source: s0005 engine + compiled draft (copy proposals on)

- status: **valid**, step_us_b1 = 9745, at 1.56 agent-hours, category other
- hypothesis: Validation of the final workspace: s0005 with the draft traced by mx.compile (static shapes: whole context array + position mask), draft + commit 3.7 -> 3.6 ms per DSpark cycle; target numerics and policy defaults identical to s0005, so the same tokens and the same gate values (numeric median 0.0240, p99 0.1013). Expect http_step_us_b1 9758 +- 3% and natural_b1 5731 +- 3%.
- evaluator: valid: 9745 us/step at B=1 over HTTP (online (submit/poll)); in process 14947 us/step at B=1 (step, MBU 0.656)

### #8 s0008: Exact argmax via 8-bit short list in verify passes; draft Markov steps in two stages

- status: **valid**, step_us_b1 = 9496, at 2.06 agent-hours, category gemm
- hypothesis: Two changes on the s0007 source, target logits untouched (tests/bitref.py bit-identical): (1) verify, prefill and generate passes need only each row's argmax: the 8-bit copy of the unembedding (already resident for the draft) gives approximate logits with a rigorous per-row error bound, and only the outputs that can still be the maximum (1-2 on text, ~100 on random input) are recomputed with the exact GEMV arithmetic; 0 mismatches against the full argmax on 1344 rows and on random inputs; pass K=1..8 about 0.6 ms faster (K=3 21.9 -> 21.3), K=16 unchanged. (2) the draft runs 3 Markov steps, looks at the cumulative probability and runs the other 4 only if it is still >= 0.3: draft 3.6 -> 3.0 ms per cycle on prose, +0.15 ms on code. Same proposals, same tokens. Expect http_step_us_b1 9745 -> about 9450 (-3%), natural_b1 5763 -> about 5700 (-1%), generate_step_us 14970 -> about 14300; gates generate and http_tokens_b1 still exact.
- evaluator: valid: 9496 us/step at B=1 over HTTP (online (submit/poll)); in process 14367 us/step at B=1 (generate, MBU 0.655)

### #9 s0009: Attention on 1024 virtual threads: short chains for small passes, same bits for all

- status: **valid**, step_us_b1 = 11851, at 2.72 agent-hours, category kernel
- hypothesis: (Second start of this submission: my shell killed the first at the beginning of its evaluation.) In-situ profile (tests/insitu.py): the attention kernel is latency-bound in small passes: 256 threads per (row, head) sum the values as 8 chains of up to 128 keys, and a one-row pass slows from 14.85 ms at position 150 to 16.95 at 1150. New layout: the arithmetic is defined on 1024 virtual threads (32 chains of <= 32 keys, sums per virtual simdgroup); passes of fewer than 8 rows run it with 1024 real threads, wider ones with 256 real threads each doing 4 virtual ones with the same lane, so rows stay bit-identical across pass widths (checked K = 1, 3, 8, 16; 1024 served tokens equal step-by-step on 3 prompts). In process (short-list passes): K=1 15.9 -> 15.0 ms at position 600 and 16.95 -> 15.3 at 1150, K=3 21.2 -> 20.7 / 22.4 -> 21.4, K=8 and K=16 unchanged. Expect step_us 14961 -> about 14200, generate 14367 -> about 13600; DSpark cycles about 3% cheaper, so natural_b1 5644 -> about 5550. The attention sums change in the last bits, so the synthetic continuations are re-rolled: http_step_us_b1 anywhere from 9000 (as s0008, when the prompts loop) to about 12000 (when two of three do not).
- evaluator: valid: 11851 us/step at B=1 over HTTP (online (submit/poll)); in process 14143 us/step at B=1 (generate, MBU 0.666)

### #10 s0010: Latency of the small kernels: qkv_post per simdgroup, gate projections in 4 lanes

- status: **incorrect**, no score, at 2.99 agent-hours, category kernel
- hypothesis: On the s0009 source. Small passes are latency-bound in the kernels that ran long dependent chains per thread: (1) qkv_post (k/v convs, clips, v norm) looped 128 dims sequentially per head; now one simdgroup per head with 4 dims per lane and simd_sum for the norms; (2) the forget/gain projections of the message gates summed 64 terms in one fma chain per output; now 4 accumulator lanes. Rows stay bit-identical across pass widths (K = 2, 3, 5, 8 vs 1; 1024 served tokens equal step-by-step on 3 prompts). In process at position 600 (short-list passes): K=1 15.0 -> 14.25 ms, K=3 20.7 -> 19.8, K=8 32.7 -> 32.1, K=16 unchanged. Expect step_us 14719 -> about 13900, generate 14143 -> about 13400. Numerics change in the last bits again, so the continuations are re-rolled once more: http_step_us_b1 and natural_b1 will land somewhere in the range the earlier builds span (9500-11900 and 5600-7300), about 4% below where this engine's predecessor would have landed on the same text.
- evaluator: incorrect: failed numeric

### #11 s0011: Test: NVFP4 expert weights rounded to BF16 after dequantization

- status: **valid**, step_us_b1 = 14843, at 3.05 agent-hours, category correctness, quick evaluation
- hypothesis: A question to the numeric gate, not a speed attempt. My engine agrees with a float32 run of mlx-lm's model to a median logit error of 0.004, yet the gate's rel_l2 against float64 stays at median 0.023 and p99 0.06-0.12 (s0010 failed it at 0.12095 > 0.12071). Hypothesis: the reference's expert weights are the NVFP4 products rounded to BF16 (what a loader that materializes BF16 tensors holds), not the exact code x scale x amax / denominator. The exact and the rounded weights give expert-block outputs that differ by 2.8e-3 relative (tests/w16.py), per application. This build rounds each dequantized NVFP4 weight to BF16 inside the expert kernels (two rounded scales per group of 16: bf16(scale) and bf16(1.5 scale); equal to a float64 product with mx.dequantize'd weights to 5e-6). If the hypothesis holds, numeric rel_l2 falls from median 0.023 to well under 0.01 and p99 under the BF16 reference's 0.060; if not, it stays or rises. Speed: about +1 ms per one-row pass expected (two LUT gathers per weight).
- evaluator: valid: 15470 us/step at B=1 (MBU 0.633)

### #12 s0012: Bit-identical latency fixes in the small kernels (same logits as s0008)

- status: **valid**, step_us_b1 = 9029, at 3.90 agent-hours, category kernel
- hypothesis: On the s0008 source, with the target's logits bit for bit unchanged (tests/bitref.py: K = 1, 3, 8, 16), so the evaluator decodes the same continuations and the gates see the same values: a clean A/B. Finding: a GPU thread that runs a long loop with a load per iteration costs ~0.3 us per iteration, and the small kernels (few threads) are bound by that latency, not by bytes or dispatch count. Fixes that keep every sum's order: qkv_post computes convs / scales in 32 lanes and only the norm is summed in order by one lane; the attention value chains load 4 keys per iteration before their 4 ordered fmas, 512 real threads for the per-key scores in passes under 4 rows; vector loads in the gate projections, slot plan, combine and threadgroup sums; depth-mix sums share one barrier; the short list recomputes flagged outputs per block of 32 instead of launching 32 threads per vocabulary row; the draft's Markov fold runs one thread per vocabulary row. In process, cool, position 600: K=1 15.9 -> 14.3 ms, K=2 19.1 -> 17.3, K=3 21.2 -> 19.6, K=8 32.55 -> 31.6, K=16 52.7 -> 51.85; draft stage 2.6 -> 2.45. Expect step_us 14961 -> about 13800, generate 14367 -> about 13000, http_step_us_b1 9496 -> about 8850 (-7%), natural_b1 5644 -> about 5380 (-5%), identical event histograms and numeric values (median 0.0240, p99 0.1013).
- evaluator: valid: 9029 us/step at B=1 over HTTP (online (submit/poll)); in process 13314 us/step at B=1 (generate, MBU 0.699)

### #13 s0013: Control: s0012 engine with DSpark drafts only (copy proposals off)

- status: **valid**, step_us_b1 = 9799, at 4.28 agent-hours, category scheduling
- hypothesis: A control, like s0006 was for s0005: the s0012 source with the copy (n-gram) proposals switched off, so every draft comes from the checkpoint's DSpark head. Same target bits as s0006, so the same continuations and the same cycles as s0006 (2.56 / 4.14 tokens per event); only the passes and the draft are faster. s0006 measured 11145 / 8467; s0012 is 4.9% / 4.7% faster than s0008 with copies, and DSpark cycles gain more than wide passes, so expect http_step_us_b1 about 10500 (1.31x oMLX+DSpark) and natural_b1 about 8000 (1.04x, i.e. above the fastest general engine with DSpark drafts alone, short of 1.10x).
- evaluator: valid: 9799 us/step at B=1 over HTTP (online (submit/poll)); in process 13218 us/step at B=1 (generate, MBU 0.701)

### #14 s0014: Down kernel: 4 weight rows per simdgroup in passes of 8+ rows (same logits)

- status: **valid**, step_us_b1 = 8982, at 4.97 agent-hours, category kernel
- hypothesis: On the s0012 source, logits bit-identical again (tests/bitref.py), so the same continuations and gate values. The experts' down kernel lets a simdgroup handle 4 weight rows instead of 2 in passes of at least 8 rows (same lane layout per row): in process K=16 51.9 -> 51.0 ms, K=8 -0.2 ms; passes under 8 rows are unchanged (it costs 0.17 ms at 3 rows, so it is not used there). 16-row passes are about 44% of the natural workload's time and 12% of the synthetic one's: expect natural_b1 5378 -> about 5340 (-0.7%) and http_step_us_b1 9029 -> about 9010 (-0.2%); both are inside the evaluator's repeat noise (s0005 vs s0007: 0.1% / 0.6%), so this submission is as much a second measurement of s0012 as a test of the change.
- evaluator: valid: 8982 us/step at B=1 over HTTP (online (submit/poll)); in process 13217 us/step at B=1 (generate, MBU 0.702)

### #15 s0015: Draft policy for the faster small passes: rows only at cumulative p >= 0.5, no minimum

- status: **valid**, step_us_b1 = 10070, at 5.21 agent-hours, category scheduling
- hypothesis: On the s0014 source; the target's logits and therefore the text are unchanged, only how many draft rows a DSpark cycle verifies. So far a cycle verified drafts while their cumulative probability stayed >= 0.3 and at least one. The draft is overconfident (cumulative 0.5-0.6 is accepted 22-45% of the time, tests/calib.py), and with the s0012-s0014 kernels a first extra row costs 3.0 ms and a second 2.3 against ~9 ms per token. Replaying 622 recorded cycles with today's pass costs: threshold 0.3 / minimum 1 gives 8.38 ms/token, threshold 0.5 / no minimum 8.11 (-3.2%); in process, alternating: prose 10.04 / 10.28 -> 9.86 / 9.88 ms/token, code 5.39 / 5.76 -> 5.43 / 5.53. The same threshold now also decides whether the last 4 Markov steps run. Copy cycles are untouched. Expect http_step_us_b1 8982 -> about 8750 (-2.5%), natural_b1 5333 -> about 5270 (-1%), fewer tokens per event (about 2.8 synthetic); gate values unchanged.
- evaluator: valid: 10070 us/step at B=1 over HTTP (online (submit/poll)); in process 13219 us/step at B=1 (generate, MBU 0.702)

### #16 s0016: Remeasurement of the s0014 source (was s0015 slow, or its policy?)

- status: **valid**, step_us_b1 = 8975, at 5.40 agent-hours, category other
- hypothesis: The s0014 source unchanged (threshold 0.3, at least one draft row). s0015 changed only the draft policy and came out 12% / 18% slower on every repeat of both workloads, against a replay and an in-process test that predicted 1-3% faster. If the evaluation's HTTP stage simply ran slower that time (heat, or anything else on the machine), this run reproduces s0014: about 8982 / 5333 with the same event histograms (610 / 161 one-token events). If it comes out near 10070 / 6297 as well, the machine is slower now and s0015 says nothing about the policy; only if this run is fast is the policy shown to be worse.
- evaluator: valid: 8975 us/step at B=1 over HTTP (online (submit/poll)); in process 13212 us/step at B=1 (generate, MBU 0.702)

### #17 s0017: Remeasurement of the s0015 source (policy 0.5 / no minimum row)

- status: **valid**, step_us_b1 = 8658, at 5.59 agent-hours, category other
- hypothesis: The s0015 source unchanged. s0015 measured 10070 / 6297; the s0014 source measured 8982 / 5333 before it and 8975 / 5378 after it (s0016), so the machine did not drift. Two readings remain: the policy really costs 12-18% over HTTP by a mechanism I do not see (then this run gives about 10070 / 6297 again), or that one evaluation was disturbed (then this run gives what the replay predicted, about 8750 / 5270, with s0015's event histograms: 701 / 205 one-token events). I put the odds near even.
- evaluator: valid: 8658 us/step at B=1 over HTTP (online (submit/poll)); in process 13201 us/step at B=1 (generate, MBU 0.702)

### #18 s0018: Control: final engine (s0017) with DSpark drafts only

- status: **valid**, step_us_b1 = 9910, at 5.72 agent-hours, category scheduling
- hypothesis: The s0017 source with the copy (n-gram) proposals off: every draft from the checkpoint's DSpark head, with the final draft policy (rows only while the cumulative probability is >= 0.5, no minimum row). The previous control, s0013 (s0012 engine, old policy), measured 9799 / 7437. The policy was worth -3.6% on synthetic text with copies on and nothing on natural text; without copies every cycle is a DSpark cycle, so expect http_step_us_b1 about 9400 (1.46x oMLX+DSpark) and natural_b1 about 7250 (1.15x), with fewer tokens per event than s0013's 2.56 / 4.14.
- evaluator: valid: 9910 us/step at B=1 over HTTP (online (submit/poll)); in process 13215 us/step at B=1 (generate, MBU 0.702)

### #19 s0019: Second measurement of the final source (s0017)

- status: **valid**, step_us_b1 = 9070, at 5.80 agent-hours, category other
- hypothesis: The s0017 source unchanged, measured once more with the last evaluation the board has room for: the same source has given 10070 / 6297 (s0015, disturbed) and 8658 / 5381 (s0017). The machine has been idle since s0018. Expect 8658 +- 1% and 5381 +- 1% with the same event histograms (701 / 205 one-token events, 89 / 192 of 16 tokens) and a p99 gap between chunks near 52 ms; a result near s0015's would mean the disturbance belongs to this source after all.
- evaluator: valid: 9070 us/step at B=1 over HTTP (online (submit/poll)); in process 13201 us/step at B=1 (generate, MBU 0.703)

### #20 s0020: Which half of the unstable policy? Threshold 0.5 with at least one draft row

- status: **valid**, step_us_b1 = 9927, at 5.88 agent-hours, category scheduling
- hypothesis: A diagnosis with the last evaluation, not a candidate for the final source. The policy of s0015 / s0017 / s0019 (threshold 0.5, no minimum row) measured 10070, 8658, 9070 on identical text while its parent s0014 (threshold 0.3, at least one row) reproduced to 0.1%. That policy changed two things; this source changes only the threshold (0.5, still at least one draft row), so no one-row pass is ever served. If one-row passes between wider ones are what makes the speed vary, this run has a p99 chunk gap near 52 ms and lands at or slightly under s0014 (about 8900 / 5330; the replay gives the threshold alone -1.8% on DSpark cycles). A p99 gap of 60-70 ms would put the instability on the threshold itself.
- evaluator: valid: 9927 us/step at B=1 over HTTP (online (submit/poll)); in process 13214 us/step at B=1 (generate, MBU 0.701)

### #21 s0021: Third measurement of the final source (s0014)

- status: **valid**, step_us_b1 = 9812, at 5.95 agent-hours, category other
- hypothesis: The s0014 source unchanged, as the last evaluation the board has room for, so that the run ends on a measurement of what is in the workspace. It measured 8982 / 5333 (s0014) and 8975 / 5378 (s0016) with a p99 chunk gap of 52.6 ms. The four runs with the 0.5 draft threshold in between gave gaps of 52-71 ms. Expect 8980 +- 1% and 5350 +- 1% with a 52-53 ms gap; a slow run here would mean the variation I blamed on the threshold belongs to the machine or the evaluation instead.
- evaluator: valid: 9812 us/step at B=1 over HTTP (online (submit/poll)); in process 13208 us/step at B=1 (generate, MBU 0.699)
