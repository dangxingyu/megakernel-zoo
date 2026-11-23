# Journal: fable-c-m5max-1 (Inkling-2-Mini, M5 Max, track C)

## 0:00-1:15 Setup and first measurements

- Read mlx-lm's `inkling2` model (the spec) and the B200 `draft.py` (the DSpark head).
- Microbenchmarks (tests/bench1.py, bench2.py), 48 dense BF16 matrices of 3072x2304 back to back:
  - `mx.matmul` bf16: 369 GB/s at 1 row, 299 GB/s at 8 rows.
  - own GEMV kernel (one simdgroup per output, lane reads 4 bf16 per iteration, float4 fma, simd_sum; rows of the
    pass as a compile-time template K; x as bf16 bits): 451 GB/s at K=1, 404 at K=4, 352 at K=8.
  - the same kernel with K as a run-time loop bound: 5x slower at K=8 (accumulators leave the registers). Template K.
  - bit equality: row r of a K-row call equals the 1-row call for every variant tried (explicit fma, safe math mode).
  - a lazy small MLX op costs about 3.5 us (build + eval); a slice update of a 2.4 MB cache about 4 us.
- MLX glue ops (rms_norm, sum over the last axis, softmax over axis 0, argmax) give each row the same bits whatever the
  number of rows (tests/det.py). So: heavy ops as own kernels, glue as MLX ops, K static per pass.
- Format check (tests/fmt.py): NVFP4 weight = E2M1(code) * E4M3(scale) * amax / denominator (2688); low nibble first.
  MXFP8 (layers 20-23): E4M3(code) * 2^(scale - 127). Both equal mx.dequantize.

## 1:15 First own pass (engine/mk_target.py, mk_kernels.py)

- K-row pass: every row bit-identical to the 1-row step for K = 2, 3, 5, 8 and after a rollback (8 rows, keep 3)
  (tests/check.py).
- Against a float32 run of mlx-lm's model (tests/ref32.py; weights cast to float32): mlx-lm bf16 median abs logit error
  0.139 (p99 0.65); ours with mlx-lm's rounding points 0.144; ours rounding only where a BF16 GEMV reads its input
  0.053 (p99 0.41). Kept the latter (MK_ROUND=0).
- Speed, lazy MLX graph: K=1 27.3 ms per pass (3.8 ms building the graph in Python, 23.5 ms eval); K=2 30.6, K=4 36,
  K=8 50.7 ms. mlx-lm: 30.4.

## s0001 (quick): own pass, no speculation
- Expected: all gates pass, step_us about 27500. Measured: valid, step_us 26852 (MBU 0.365).
- Conclusion: the pass computes the target (numeric, numeric_long to 1151, kl all ok). Next: the draft.

## DSpark draft (engine/mk_draft.py)
- Written from the B200 `draft.py` (TorchDraft) and checked against oMLX's `inkling2/dspark.py` for the semantics
  (features = hidden entering visits 5, 23, 41 + final-norm output; block = anchor + 7 masks, bidirectional; context
  K/V with RoPE at absolute positions; Markov head with the hidden_out x unembed fold).
- A worry that turned out to be nothing: on my uniform-random prompt the first draft was almost never accepted (the
  model falls into a period-2 loop the draft does not predict) and on a short prose prompt d1 matched 47%. The
  evaluator's numbers settle it: 2.90 tokens/event synthetic, 4.21 natural (B200 engine: 2.88 / 5.22 with n-gram).
- Draft cost: 5.1 ms per cycle (3 layers on 8 rows + unembed for 7 rows + 7 Markov steps), lazy MLX ops + own GEMV.

## s0002: draft-and-verify in submit/poll (lazy graphs)
- Expected: synthetic 24000-27000, natural 14000-16000. Measured: valid, 15646 synthetic (2.90 tokens/event),
  12704 natural (4.21 tokens/event; 215 of 974 events at the 8-token cap); gates generate / http_tokens_b1 exact.
- Conclusion: my random prompt underestimates acceptance on the evaluator's synthetic prompts. Cycle = about 45 ms.

## s0003: mx.compile of the pass, shared expert reads, n-gram copy proposals
- Expected: synthetic 11000-13000, natural 9500-10500, step_us 20500.
- Measured: valid, 10545 synthetic (3.04 tokens/event; 1.306x oMLX+DSpark: target met), natural 8335 (5.08
  tokens/event; 0.997x oMLX+DSpark: just short), step_us 18812, generate 18717.
- What each part did in process: compile K=1 27.3 -> 20.7 ms (the Python graph build and ~1/3 of the op overhead);
  expert slots: on prose an 8-row pass picks 33 distinct experts of 64 (min 15, max 41); slots of 2 pairs 26.5 -> 19.3
  ms per pass; slots of 4 or 8 are slower (compute-bound: 38.9 / 94 ms), the old per-pair kernel at K=8 was already at
  ~520 GB/s. n-gram copies (suffix >= 5 tokens seen before -> its continuation, 8 rows): loops are common in 1024-token
  continuations.
- Numeric gate margin was thin: rel_l2 median 0.0270 (limit 0.0494), p99 0.0956 (limit 0.1207): rounding only GEMV
  inputs to BF16 is no better than the BF16 reference. Fixed next by reading activations as IEEE half.

## Glue profile (tests/glue.py, K=4, heavy kernels stubbed)
- real pass 29.2 ms; glue only 9.8 ms: depth mixes 7.5 ms (!), convs 2.1, message gates 2.6, rest 4.5 (overlapping).
- depth mix as one kernel per mix (576 threads per row): glue 6.8 ms, pass K=4 26.4, K=1 18.4, K=8 36.6 ms.
- MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER: no gain at any value tried (10..2000).

## s0004: depth-mix kernel, half activations, 16-row passes for copies
- Expected: natural 7000-7400, synthetic 8500-9500, step_us 18500.
- Measured: valid. natural 6166 (7.59 tokens/event; 195 events of 16 tokens) = 1.347x oMLX+DSpark. Synthetic 14178
  (2.63 tokens/event): WORSE than s0003's 10545. step_us 16876. numeric median 0.0223 (s0003 0.0270), p99 0.1017.
- Why synthetic moved the wrong way: the headline is the median of three repeats and the repeats are different prompts:
  s0003 10049 / 10545 / 15902, s0004 6227 / 14178 / 15767. A run that falls into a loop decodes at 6-10 ms per token
  (copy proposals, 16 rows), a run that does not at 14-16 (1.9 tokens/event). The half-precision activations changed
  the numerics, so the continuations differ from s0003's (two of three looped there, one of three here). The synthetic
  headline is therefore decided by the non-looping case: one cycle = draft 5.1 ms + verify of ~3 rows ~24 ms for
  1.9 tokens. Plain stepping would be 16.9 ms/token, so DSpark gives only ~1.15x there. To be safe the cycle must
  come down to <= 23.7 ms.
- The numeric gate's median improved less than my float32 comparison promised (12x): the harness's float64 reference
  differs from mlx-lm-in-float32 by more than the BF16 rounding of activations explains; not pursued.

## After s0004: where a low-acceptance cycle goes (K = 2..3 rows)
- bytes: dense 2.5 GB + unembed 0.93 GB per pass, experts ~1.6 GB per row (14.7 distinct of 16 picks at K=2, 18 of 24
  at K=3: little sharing), draft 2.1 GB. Expert kernels are at 380-450 GB/s for every K (tests/e3.py), dense GEMVs
  ~450-560.
- glue: with every heavy kernel stubbed the K=2 pass still takes ~5 ms. A trivial kernel dispatch costs 2.5 us
  (independent) to 4.3 us (chained) (tests/gk.py); the pass had ~2000 dispatches.
- Done: mix2 (depth mix + input RMS norm + output layernorm + half bits in one kernel), gates (pre/post kernels for the
  message gates, 3 at once), qkv_post (k/v convs, clips, v norm, windows), sconv2 (conv + residual + window),
  attention with the output gate folded in. Pass K=1 18.4 -> 16.3, K=2 -> 19.5, K=3 -> 21.9, K=8 36.2 -> 32.7,
  K=16 57 -> 52 ms. About 1050 dispatches left.
- MLX slice updates copy the whole array (128 MB K + 128 MB V per cycle if stacked; same bytes as 96 separate ones):
  K/V commits now go to a 128-row recent tier, flushed to the big cache every ~96 positions. Commit 1.1 -> 0.3 ms.
- 8-bit per-row-scaled copies of the draft's matrices (layers, unembed, fold): draft 5.2 -> 3.7 ms per cycle,
  tokens/cycle unchanged on prose (1.733) and code (4.974).

## s0005: fused glue kernels, two-tier K/V cache, 8-bit draft copies
- Expected: natural 5400-5700; non-looping synthetic runs 12.8-14.2 ms/token, headline 13000 +- 1500; step_us 16400.
- Measured: valid. Synthetic 9758 (repeats 9192 / 9758 / 11082; 2.90 tokens/event) = 1.411x oMLX+DSpark; natural
  5731 (7.21 tokens/event) = 1.449x; step_us 14948 (MBU 0.656), generate 14966. Both workloads above 1.10x in one
  submission. numeric median 0.0240 / p99 0.1013 (limit 0.0494 / 0.1207).
- The synthetic headline came out better than my estimate for non-looping runs: all three repeats had copy cycles
  (89 events of 16 tokens over the four runs). The estimate for a run with none stays ~13 ms/token.

## After s0005: things tried that did not pay (all off in the final source, or not kept)
- qkv_post with one simdgroup per head (4 dims per lane) instead of a sequential loop per head: not faster (the
  dispatch costs 11-14 us either way; MLX dispatch overhead, not the loop). Not kept.
- The message gates as ONE kernel per (row, gate) (down_proj as 64 simd_sum reductions inside a 576-thread group):
  correct (rows bit-identical across K) but slower: K=2 pass 20.7 vs 19.6 ms. Kept behind MK_GATE2=1, off.
- K policy (tests/calib.py: 622 cycles on 5 prompts with all 7 drafts verified): the draft's cumulative probability
  is overconfident (0.5-0.6 -> 22-45% accepted, 0.9-1.0 -> ~90%); the replay prefers theta 0.5 and no minimum draft
  row (9.17 vs 9.32 ms/token), and one plain step after an unsure draft (9.03). Measured for real (tests/nat.py):
  8.64 (theta 0.3, min 1) / 8.57 / 8.61 ms/token: no difference outside noise. Defaults left at theta 0.3, min 1.
- n-gram minimum match 3 or 4 instead of 5: tokens/cycle within 1%, more rejected 8-row passes. Kept 5.
- mx.compile of the draft with static shapes (whole context array + mask): draft+commit 3.72 -> 3.57 ms per cycle. Kept.
- A caution for anyone comparing my error numbers: the median |logit error| against the float32 run moves between
  0.002 and 0.025 with any change of summation order (qkv simd, fused gates): one router near-tie early in the 24-step
  test sequence shifts every later position. The evaluator's rel_l2 p99 was 0.096 / 0.102 / 0.101 for three
  numerically different engines.
- Timing comparisons across back-to-back test runs are confounded by heat: the same token stream (same tokens/cycle)
  measured 4.50 and then 5.08 ms/token two runs later.

## What a cycle costs now (K rows; in process, context ~600)
- pass: K=1 16.3, K=2 19.5, K=3 21.9, K=8 32.7, K=16 52 ms. Bytes at K=2: dense 2.5 GB + unembed 0.93 + experts 3.2
  (14.7 distinct of 16) = 6.8 GB -> ~15 ms at 450 GB/s; the other ~4.5 ms are ~1050 MLX dispatches at ~4 us.
- draft (8-bit copies, compiled): ~3.5 ms incl. the commit of the previous cycle; skipped in copy cycles.
- Not done, in order of expected gain: (1) the pass as one Metal command buffer from Objective-C++ (own buffers, no
  per-dispatch MLX overhead): ~2-3 ms per pass; (2) exact argmax through an 8-bit shortlist of the unembedding with a
  rigorous error bound and exact rescoring: ~0.9 ms per pass; (3) 32-row passes for copy cycles: ~10% on looping text.

## s0006: control, DSpark drafts only (copy proposals off)
- Expected: natural 9000-11000 (about 3 tokens/event), synthetic 12500-14500 (about 2 tokens/event).
- Measured: valid. Synthetic 11145 (repeats 11108 / 11145 / 12636; 2.56 tokens/event) = 1.235x oMLX+DSpark; natural
  8467 (4.14 tokens/event; 175 of 989 events at the 8-token cap) = 0.981x. Gate values identical to s0005 (same target
  numerics), no 16-token events (the evaluator ran its own copy of the source).
- Conclusion: with nothing but the checkpoint's DSpark head the engine is above the 1.10x line on synthetic text and
  2% short of oMLX+DSpark on natural text; the copy proposals are worth 1.14x on synthetic and 1.48x on natural
  (s0005: 9758 / 5731), all of it on continuations that repeat. Better than I predicted on both: the DSpark draft
  does predict much of the repetition (100 + 175 events of 8 tokens).
- An untested DSpark-only idea for the 8-token cap: a second draft block anchored on the 7th draft (positions
  c+7.., context still < c) when the first block is confident, verified in a 15-row pass. Estimated ~4% on natural.

## s0007: final source (s0005 + compiled draft)
- Expected: 9758 +- 3% / 5731 +- 3%, same tokens and gate values. Measured: valid, 9745 synthetic (1.413x), 5763
  natural (1.441x), identical event histograms and numeric values. The result reproduces to 0.5%.

## Dead end after s0007: a chained second DSpark block (notes/chained_block_experiment.diff)
- Idea: when all 7 drafts are confident, run the draft again anchored on the 7th draft (block at positions c+7..,
  context still the positions < c, no target features for the drafted positions) and verify 12 or 15 rows.
- Measured (tests/nat.py, 640 tokens, copy proposals off): tokens/cycle hardly move (gsm 5.81 -> 5.97, code 6.14 ->
  6.52) while rows/cycle rise (6.6 -> 8.1, 7.3 -> 9.3); 8.52 -> 8.97 ms/token overall. Without the target's features at
  the drafted positions the second block is rarely right. Not kept; the engine is the s0007 source.

## Dead end after s0007: merging dispatches (notes/merged_dispatches_experiment.diff)
- The three q/k/v GEMVs in one grid, the three gate down-projections and the three forget/gain projections in one
  dispatch each, router + slot plan in one threadgroup, expert combine + post-MLP conv in one kernel: ~290 of ~1050
  dispatches removed, logits bit-identical (tests/bitref.py: K = 1, 3, 8, 16).
- Measured: nothing. K = 1 / 2 / 3 / 8 / 16: 16.6 / 19.6 / 21.9 / 33.2 / 52.8 ms before and 16.6 / 19.6 / 21.9 / 33.1 /
  53.1 after (two alternating runs each).
- So the 2.5-4 us per dispatch of the isolated microbenchmark (tests/gk.py) is not what a dispatch costs inside a
  pass, and my earlier "glue only" figure (~5 ms with the heavy kernels stubbed) was mostly the stubs' own MLX ops.
  The earlier fusions paid because they removed real work (stack copies, sum reductions, whole-cache slice updates),
  not because they removed dispatches. What is left of a K=2 pass (19.5 ms): dense GEMVs 6.1 ms (3.4 GB, ~560 GB/s),
  experts 7.1 ms (3.2 GB, ~450 GB/s), attention 1.3, the small kernels ~3.5 (gates ~1.0, router chain ~1.0, qkv 0.5,
  conv 0.5, mixes 0.6), by their isolated timings. Not kept; the engine stays the s0007 source.
- With that the estimate for "the pass from Objective-C++ in one command buffer" (notes above: 2-3 ms per pass) loses
  its basis: unmeasured.

## Stop (1.56 board-hours used, 7 submissions)
- Final engine = s0007 source. Why stop with board time left: the last four in-process experiments returned nothing,
  the passes run at ~450-560 GB/s on their bytes, and what remains (8-bit shortlist for an exact argmax ~4%, 32-row
  copy passes ~3-5% on natural text) risks the exact-token gates for a few percent. Not verified by me and worth a
  look by whoever continues: why the evaluator's rel_l2 against float64 stays at median 0.024 / p99 0.10 (BF16
  reference 0.025 / 0.06) although activations are now read with 11 significant bits; my guess (unchecked) is that the
  float64 reference and mlx-lm's interpretation differ in something systematic, e.g. where expert weights or
  activations are rounded.

## Resumed (board 1.68 h). In-situ profile first (tests/insitu.py: a kernel class dispatched twice, delta = its cost)
- K=3 pass: dense GEMVs 4.7 ms, unembed 1.7, experts 8.7 (446 GB/s on 3.9 GB), attention 1.2, qkv_post 1.1, the rest
  together ~1.4. A dispatch with 9 inputs and 6 outputs costs ~11 us before it does anything (qkv_post: 13 us).
- Expert kernel variants, none better: inputs of a group hoisted out of the weight-row loop (no change, bit-identical),
  threadgroups of 64 / 1024 (no change), 2 or 4 groups per lane iteration (K=8 33.4 -> 36.9 / 40.2 ms: worse).
- Heat: after several minutes of back-to-back tests the same passes ran 25% slower (K=8 40.5 vs 33.1 ms), no warning
  from pmset; 7 idle minutes restore them. Earlier "thermal noise" remarks in this journal were this.

## s0008: exact argmax through an 8-bit short list; Markov steps in two stages
- Expected: synthetic about 9450, natural about 5700, generate about 14300.
- Measured: valid. 9496 synthetic (1.450x), 5644 natural (1.472x), generate 14367 (step 14961 unchanged: step()
  still returns full logits). Same event histograms and gate values as s0007.
- Short list (kernels: argmax_short): approx = 8-bit copy's product; |approx - exact| <= scale[n] / 2 * ||x||_1
  (+2% and 0.02 slop); outputs whose upper bound reaches the best lower bound of any row are recomputed by the exact
  GEMV code, the others are -inf. Candidates per row: median 1, max 18 on text; ~100 on random-token input; the whole
  vocabulary for a tiny input (still correct). 0 argmax mismatches on 1344 rows (tests/short2.py).

## Dead end: 32-row copy passes
- Built (plan with up to 192 slots, proposals of 31 tokens after a fully accepted 16-row pass). The K=32 pass takes
  287 ms (K=16: 52.7): the GEMV with 32 float4 accumulators per lane falls off a cliff. Splitting the dense GEMVs into
  two 16-row halves would leave only the experts' sharing as a gain (<= 3% on natural text). Dropped; source = s0008.

## Dead ends on the 16-row pass (tests/nghist.py, tests/loop16.py)
- Copy cycles are healthy: 389 16-row cycles on 8 prompts x 1024 tokens, 98% accepted in full; 59 8-row copy cycles
  average 4.9 tokens. 16-row passes carry ~75% of the tokens on looping continuations, so their 52.7 ms matter most.
- In such a pass the experts take 28-33 ms: 24-49 distinct experts of 128 picks, 71-80 slots of 2 pairs (17 GB,
  ~520 GB/s). Slots of 4: 57 slots but 35-41 ms (compute-bound: 228 row-pairs computed instead of 158); slots of 8: 61.
- Slots interleaved per output row, hoping the slots of one expert would share the read through a cache: 32.3 -> 38.7
  ms. No sharing, worse streams. Not kept.

## More dead ends on the pass (all measured in one cool session, alternating)
- K = 1 with the per-pair expert kernels (no slot plan): bit-identical, 16.6 vs 16.6 ms. No gain.
- 4 dummy outputs on each of 96 conv dispatches: +0.15-0.2 ms per pass, i.e. ~0.5 us per output buffer in situ.
  Neither dispatches nor outputs explain the small kernels' cost.
- Empty pair positions of a slot skipped with a uniform branch: slots of 2 get slower (31.8 -> 33.3 ms on a prose
  loop), slots of 4 faster (35.1 -> 32.3) but not better than slots of 2.

## Attention is latency-bound in small passes (found with tests/passt.py at different positions)
- One-row pass: 14.85 ms at position 150, 15.9 at 600, 16.95 at 1150: 2 ms of attention growth, although its bytes are
  nothing (K/V of 1024 keys = 4 MB per application). 256 threads per (row, head) summed the values as 8 chains of up
  to 128 dependent fmas.
- With 1024 threads per group (32 chains): K=1 16.95 -> 15.4 at position 1150, K=3 22.4 -> 21.4; K=16 +0.5.
- Kept both: the arithmetic is defined on 1024 virtual threads; a real thread runs 1 (passes of < 8 rows) or 4 (wider)
  virtual ones with the same lane, so simd_sum sees the same lanes and the bits do not depend on the real threadgroup
  size (rows of K = 1, 3, 16 passes bit-identical, tests/bitref.py; K=16 52.8 ms as before).
- Process mistake: I put a 400 s cool-down in front of `mk submit` inside one background job; the job's 10-minute
  limit killed the evaluation at its start (s0009, no result). The stale submissions/s0009 directory then blocked the
  next `mk submit`; renamed to s0009-killed-no-result. I also edited engine/ during that wait, which would have gone
  into the submission; stashed in time. From now on: cool down first, submit as its own job, edit only after src.tgz
  exists.

## s0009: attention on 1024 virtual threads
- Expected: step 14200, generate 13600, natural about 5550, synthetic anywhere from 9000 to 12000 (re-rolled).
- Measured: valid. step 14719, generate 14143 (MBU 0.666), numeric median 0.0190 / p99 0.0645 (was 0.0240 / 0.1013).
  Synthetic 11851 (1.16x; repeats 11597 / 11851 / 13600; 2.65 tokens/event), natural 7267 (1.14x; repeats 5610 /
  7267 / 9627; 5.98 tokens/event, 134 events of 16 tokens instead of 192, 94 of 8, 22 of 10).
- The engine is faster per pass and scores worse: the last-bit change of the attention sums gives other
  continuations, and they repeat less cleanly. Over the numerically different builds of this run the same workload
  has given: synthetic 10545 (s0003), 14178 (s0004), 9496-9758 (s0005-s0008), 11851 (s0009); natural 8335, 6166,
  5644-5763, 7267. The spread between builds (±15-25%) is larger than anything I have gained since s0005. A reader
  should take "1.2x to 1.45x on synthetic, 1.1x to 1.47x on natural, depending on which continuations the build's
  rounding produces" as the result, not the best line of the board.
- Next: make the engine less dependent on clean loops. In s0009's natural runs many copy passes are accepted in
  part (22 events of 10 tokens, 94 of 8): repeats with a varying token (numbered lists and the like). After such a
  miss the suffix no longer matches 5 tokens and the engine falls back to DSpark until 5 new tokens line up.

## s0010: qkv_post per simdgroup + 4-lane gate projections: INCORRECT (numeric p99)
- Expected: step about 13900, generate about 13400; headline somewhere in the earlier range.
- Measured: step 13935, generate 13334 as predicted, but gate numeric FAILED: rel_l2 p99 0.12095 against the limit
  0.12071 (median 0.0229, fine). No HTTP stage.
- The p99 of this gate is as chaotic as the continuations: 0.0956 (s0003), 0.1017 (s0004), 0.1013 (s0005-s0008),
  0.0645 (s0009), 0.1210 (s0010), for engines that all agree with a float32 run of mlx-lm's model to a median logit
  error of 0.004. I wrote down the thin margin after s0003 and went on re-rolling it; this is the bill.
- So the open question from the stop note is now the first thing to settle: why is my tail ~1.7x the BF16
  reference's (0.060) when my median is below its median (0.0247)? Hypothesis: the float64 reference uses expert
  weights dequantized to BF16 (as a PyTorch loader would materialize them), while I (and mlx-lm) multiply the exact
  products code x scale x amax / denominator. Test: round the dequantized NVFP4 weights to BF16 inside the expert
  kernels and look at the numeric gate of a --quick submission.

## s0011 (quick): are the reference's expert weights BF16-rounded? No.
- Expected if the hypothesis held: numeric rel_l2 median well under 0.01, p99 under 0.060.
- Measured: valid, median 0.0272, p99 0.0678 (step 15470: the two LUT gathers cost 1.5 ms). The median did not
  move, so the reference does not use BF16-rounded expert weights. Hypothesis rejected; MK_W16 not kept
  (tests/w16.py shows the emulation itself is exact to 5e-6, and that the two weight definitions differ by 2.8e-3
  relative per expert block).
- Read the Hugging Face inkling2 / inkling2_loop modeling code against my pass (norms, clips, convs, relative bias,
  sinks, router, experts, message gates, loop structure): no semantic difference found.

## What the numeric error actually is (per-step rel_l2 against the float32 run, tests of 24 steps)
- mlx-lm BF16: median 0.018, max 0.071, flat over the steps. Mine (half activations): 0.000-0.003 for the first
  steps, then 0.01-0.04: it grows with the position. The elementwise "median |logit error| 0.004" I quoted earlier
  hid that. The error is not a systematic offset: tiny rounding differences flip a near-tie somewhere (router top-8),
  and every later position sees the flipped history. Half activations flip less often than BF16 but the flips are as
  large, so by position ~100 the median is BF16-like (the gate's 0.023) and the p99 is a matter of which positions
  flipped: 0.064 to 0.121 over six builds. A build with a different summation order anywhere is a new draw; about one
  in six of mine landed above the limit (s0010).
- Float32 activations everywhere (notes/act32_experiment.diff; kernels read float4 instead of half4, K/V caches
  float32): per-step error median 0.0017, max 0.012 (3-4x lower), rows still bit-identical across K. But the passes
  get much slower: K=1 14.25 -> 14.9 ms, K=3 19.75 -> 21.65, K=8 31.9 -> 36.3, K=16 52.4 -> 64.7 (the K input rows of
  every GEMV and expert slot are read as 16 B instead of 8 B per 4 elements). Not kept: it buys gate margin with
  10-23% of the speed.

## Where this leaves the run
- Best and final source: s0008 (9496 synthetic, 5644 natural, valid, reproduced by s0005 / s0007 with the same
  continuations). Fastest valid passes: s0009 (step 14719, generate 14143) with 11851 / 7267 on its own
  continuations. s0010 (step 13935) is 0.2% over the numeric p99 limit.
- Two lotteries sit on top of the engineering, and both are decided by last-bit numerics: which continuations the
  three synthetic / natural prompts get (loops or not: 9.5-14.2 k and 5.6-8.3 k us/token over the builds of this
  run), and the p99 of the numeric gate. I did not use the remaining submissions to draw more tickets.

## Stop (3.12 board-hours used, 11 submissions)
- What is left splits in two. Changes that keep the target's bits (so s0008's continuations and gate values): policy
  and draft details worth <= 1-2% each, below the run-to-run heat noise. Changes that alter the bits (attention
  layout, qkv per simdgroup, 4-lane gate sums: together -10% on a one-row pass, -7% at three rows): each is a new draw
  of the continuations and of the numeric p99, and would move the headline by luck far more than by its own gain.
  Spending the remaining evaluations on such draws until one beats s0008 would be selecting on noise.
- If someone continues: the fast-kernel source is tag s0010-incorrect-numeric (step 13935); a valid variant of it
  needs either a different draw or a wider numeric margin (float32 activations give the margin and cost 10-23%:
  notes/act32_experiment.diff). The honest way to rank engines on this workload would be more prompts per workload
  (the three repeats are three prompts), which is the evaluator's side.

## Resumed again (board 3.15 h): the small kernels are bound by the latency of loops with a load per iteration
- I was wrong to file the remaining small-pass time under "last-bit lottery". The gains of s0009 / s0010 did not need
  a new summation order; they needed shorter dependent loops, and those can keep every sum's order.
- Evidence: 1000 chained trivial dispatches cost 2.9 us each (1.8 encode + wait), so dispatch count is ~1 ms per
  pass at most; but qkv_post (128-iteration loops with a conv per iteration), the attention value chains (up to 128
  keys, a load each) and the gate projections (64 loads per output) each took 20-40 us per dispatch: about 0.3 us
  per loop iteration with a memory load. In kernels with few threads that latency is the dispatch's duration.
- Fixes, all bit-identical to s0008 (tests/bitref.py, K = 1, 3, 8, 16):
  - qkv_post: 32 lanes compute convs / scales (4 dims each) into threadgroup memory, one lane sums the norm in dim
    order from float4 loads, the lanes write. K=1 15.9 -> 15.2 ms, K=2 19.1 -> 18.2 (with 512 real attention threads
    for the per-key scores in passes under 4 rows, which alone is 0.1-0.15 ms).
  - attention value chains: 4 keys per iteration, loads first, then the 4 fmas in key order; gate projections with
    float4 loads and scalar-order fmas: K=1 -> 14.5, K=2 -> 17.5, K=3 21.2 -> 19.8; at position 1150 K=1 16.95 -> 14.85.
  - vector loads in the slot plan, combine and threadgroup sums, loads-first in the per-key score: no further change.
  - depth-mix sums behind one barrier; the short list's exact recompute per block of 32 outputs (6.4 M threads ->
    200 k; exact on tests/short2.py): K=1 -> 14.3, K=3 -> 19.6.
  - draft: the Markov fold (51 MB, one input row) with one thread per vocabulary row: draft stage 1 2.6 -> 2.45 ms,
    stages 1 + 2 3.57 -> 3.15.
- Totals, cool, position 600: K=1 15.9 -> 14.3 ms, K=2 19.1 -> 17.3, K=3 21.2 -> 19.6, K=8 32.55 -> 31.6,
  K=16 52.7 -> 51.85. Served prose (same 252 cycles as s0008): 7.4 -> 6.8 ms/token.

## s0012: bit-identical latency fixes
- Expected: synthetic about 8850 (-7%), natural about 5380 (-5%), step about 13800, generate about 13000, identical
  event histograms and numeric values.
- Measured: valid. 9029 synthetic (1.525x; -4.9%), 5378 natural (1.545x; -4.7%), step 14013, generate 13314 (MBU
  0.699). Event histograms and numeric values identical to s0008: the A/B is clean. New best.

## After s0012: what did not add to it
- GEMV with 3 weight loads per loop iteration (loads first, same fma order, bit-identical): K=1..3 unchanged, K=8
  -0.35 ms, K=16 +2.5 ms. The big GEMVs are bound by bytes, not by loop latency. Not kept.
- Starting the next cycle's first GPU step (draft, or the verify pass of a copy proposal) at the end of the current
  cycle, before the tokens go back to the server (notes/prelaunch_experiment.diff): exact, but no gain even with a
  simulated 200 us of host work per token, because mx.async_eval does not return until most of the GPU work is done
  (11.9 of 14 ms for a one-row pass): the launch just moves the wait. A worker thread could overlap them, but MLX
  streams are thread-local ("There is no Stream(gpu, 1) in current thread"): every call would have to move to that
  thread. The server's per-token cost is small anyway (tpot p50 12 us), so not pursued.
- In-situ profile of a 16-row pass (worst case, 16 distinct tokens): experts 32.7 of 53.5 ms, dense 6.6, attention
  3.6, unembed 2.5, the gate projections 2.3.

## s0013: control, s0012 engine with DSpark drafts only
- Expected: synthetic about 10500, natural about 8000.
- Measured: valid. 9799 synthetic (1.405x oMLX+DSpark), 7437 natural (1.117x); same events as s0006 (2.56 / 4.14
  tokens per event), same gate values. s0006, the same control on the s0005 engine, was 11145 / 8467: -12%, more than
  the 5% the engine with copies gained, because DSpark cycles are small passes and those gained most.
- So with nothing but the checkpoint's DSpark head the engine is now above the 1.10x line on both workloads
  (on these continuations); the copy proposals add 1.09x on synthetic and 1.38x on natural on top (s0012: 9029 / 5378).

## Dead end: load-first loops in the GEMVs of small matrices and in the gate projections
- Router and gate down-projection GEMVs with 3 weight loads per iteration, gate forget/gain projections with 4
  iterations of loads first (all bit-identical): K=1 / 3 unchanged (14.3 / 19.7 ms), K=8 -0.35, K=16 +0.3. Not kept.

## State at 4.4 board-hours
- Engine = s0012 source (tag s0012-valid): 9029 / 5378 with copies, 9799 / 7437 with DSpark drafts only (s0013).
- What moved the result in this last stretch was not a new idea but a measurement I had read wrongly twice: the
  time of the small kernels is the latency of their loops with loads, which can be cut without touching a sum's
  order. Left on the table, by size: 16-row passes are 60% expert reads at ~520 GB/s on slots of 2 pairs (a kernel
  that reads each expert once for all its rows at that rate would save ~8 ms of 52); the one-row pass is at 14.3 ms
  against ~10.5 ms for its bytes at the rates the big kernels reach.

## Last measurements (not submitted; engine stays the s0012 source)
- Down kernel with 4 weight rows per simdgroup in passes of >= 8 rows (bit-identical, tests/bitref.py): K=16 51.9 ->
  51.0 ms, K=8 -0.2, and +0.17 ms at K=3 if used there. Worth ~0.7% on natural text, which is the size of the
  evaluator's own repeat noise (s0005 vs s0007: 0.1% / 0.6%), so I did not spend an evaluation on it. One line in
  mk_target.py (experts2(..., RDN if K < 8 else 4)) for whoever continues.
- Other expert layouts at 16 rows on a prose loop (M=2): RU/RD (2,2) 32.5 ms, (1,1) 36.2, (1,2) 33.5, (2,1) 35.5,
  (2,4) 31.4, (4,2) 49.7, (3,4) 41.9, (2,8) 31.4, (2,6) 32.5.
- Passes of 20 / 24 rows: 82 / 100 ms (16 rows: 50.5): the cliff of the 32-row attempt starts right above 16 rows.
- 512 real attention threads in wide passes: K=16 +1 ms. Kept at 256 for >= 4 rows.
- tests/insitu2.py (a kernel class computed but taken off the dependency chain): mix2 1.3 ms, conv 1.1, gates 1.0,
  experts 1.0, attention 0.7, router 0.4 + 0.5 of a 16.2 ms one-row pass. I do not trust this method to say what a
  fused kernel would save (the merged-dispatch experiment saved nothing); recorded as is.

## Stop (4.7 board-hours, 13 submissions)
- Final engine = s0012 source: 9029 / 5378 (1.52x / 1.54x oMLX+DSpark); DSpark drafts only (s0013): 9799 / 7437
  (1.41x / 1.12x). The last bit-identical changes I could find are each below the evaluator's repeat noise.

## s0014: down kernel with 4 weight rows per simdgroup in wide passes
- Expected: natural about 5340, synthetic about 9010, same events and gate values.
- Measured: valid. 8982 synthetic (1.533x), 5333 natural (1.558x), step 13968, generate 13217; identical events and
  numeric values. As predicted (and as small as predicted). Final source = s0014 (tag s0014-valid).
- Left unspent: about one board-hour. What I still know how to do keeps the logits and is worth under 1% each
  (the draft's context append as a compiled function, 3 weight loads per iteration in the small GEMVs of 8-row
  passes); the larger items (expert reads shared across more than 2 rows at bandwidth, the one-row pass's 3.5 ms above
  its bytes) need a kernel design I do not have.
- The draft's context append as a compiled function: 12.57 / 12.54 vs 12.44 / 12.57 ms/token on prose (DSpark only,
  alternating runs): nothing. Not kept. Engine = s0014 source.

## s0015: draft policy threshold 0.5, no minimum row
- Expected: synthetic about 8750 (-2.5%), natural about 5270 (-1%), fewer tokens per event, same gates.
- Measured: valid, same gate values, but 10070 synthetic (+12%) and 6297 natural (+18%). Tokens per event 2.77 / 6.85
  (was 2.90 / 7.21): 701 instead of 610 one-token events on synthetic, 205 instead of 161 on natural; the 16-token
  events are the same 89 / 192.
- The replay (-3.2%) and the in-process alternation (prose -2 to -4%) said the opposite. 44 more one-token cycles
  cannot cost 18% of the natural runs, and all three repeats of both workloads are 10-17% slower, the looping one
  included; step_us (measured early in the evaluation) is unchanged. So either the HTTP stage ran throttled this
  time, or the policy has a cost I do not see. Next: resubmit the s0014 source unchanged to tell the two apart.

## s0016: the s0014 source again
- Expected: about 8982 / 5333 if s0015's slowness was the machine; about 10070 / 6297 if the machine is slower now.
- Measured: valid, 8975 / 5378, the s0014 histograms. So the machine is as before, and s0015 was slower either
  because of its policy or because of something during that one evaluation. The per-run excess of s0015 (about 1 s)
  divided by its extra one-token cycles (11-23 per run) would be 50-85 ms per such cycle, which no pass costs; I have
  no mechanism for the policy being that expensive. One more measurement of the s0015 source decides it.

## s0017: the s0015 source again
- Expected: about 10070 / 6297 if the policy is costly, about 8750 / 5270 if s0015 was disturbed; odds near even.
- Measured: valid, 8658 synthetic (1.590x), 5381 natural (1.544x), s0015's event histograms. So s0015 was a
  disturbed evaluation (p99 gap between chunks 70 ms there, 52 ms in every other run), and the policy (draft rows
  only while the cumulative probability is >= 0.5, no minimum row) is worth -3.6% on synthetic text (8982 / 8975 ->
  8658) and nothing on natural text (5333 / 5378 -> 5381), where 16-row copy passes carry most tokens.
- Lesson for the record: one evaluation of seventeen was 12-18% off with nothing in its result marking it except
  tpot_p99; had I not remeasured, I would have thrown away a 3.6% gain and written down a mechanism that does not
  exist. My guess is heat from the in-process tests I ran before the cool-down (5.5 minutes that time).
- Final source = s0017 (tag s0017-valid).

## s0018: control, final source with DSpark drafts only
- Expected: synthetic about 9400, natural about 7250 (the new policy helping every cycle).
- Measured: valid, 9910 synthetic (1.39x), 7665 natural (1.08x); 2.45 / 3.93 tokens per event (s0013: 2.56 / 4.14,
  9799 / 7437). Wrong direction: without copies DSpark has to carry the repeating stretches itself, and the 0.5
  threshold cuts drafts there that would have been accepted (8-token events 100 -> 89 and 175 -> 163). With copies on,
  those stretches go to the copy proposals and the threshold only trims the hard text (s0014 -> s0017: -3.6% synthetic).
- So the two policies are each right for one configuration. The final source keeps copies on with threshold 0.5.

## End of the run (5.73 of 6 board-hours, 18 submissions)
- Final source = s0017 (= s0015): 8658 synthetic (1.59x oMLX+DSpark), 5381 natural (1.54x). DSpark drafts only:
  1.39-1.41x synthetic, 1.08-1.12x natural (s0018, s0013).

## s0019: the s0015 / s0017 source a third time
- Expected: 8658 +- 1% / 5381 +- 1%, p99 chunk gap near 52 ms.
- Measured: valid, 9070 / 5653, p99 gap 62.5 / 59.9 ms. In between s0017 and s0015.
- So I was wrong about s0015 being "a disturbed evaluation": the three runs of this source are 10070 / 6297, 8658 /
  5381, 9070 / 5653 (p99 gap 71, 52, 62 ms), while the s0014 source gave 8982 / 5333 and 8975 / 5378 (52.6 ms both).
  The 0.5 / no-minimum policy makes the served speed vary between evaluations; its mean (9266 / 5777) is worse than
  s0014. It is the only configuration that serves one-row passes between wider ones; I did not find the mechanism
  (the in-process tests did not show it) and there is no board time left to look.
- Final source: back to s0014 (tag s0014-valid), the one that reproduces. The board's best line (s0017, 8658) is the
  lucky run of a source I do not ship. The s0018 control used that policy too and inherits the doubt.

## End of the run (5.80 of 6 board-hours, 19 submissions)
- Final source = s0014: 8982 / 5333 and 8975 / 5378 in two evaluations (1.53x / 1.55x oMLX+DSpark). DSpark drafts
  only: s0013, 9799 / 7437 (1.41x / 1.12x).

## s0020: threshold 0.5 with at least one draft row (diagnosis, last evaluation)
- Expected: p99 chunk gap near 52 ms and about 8900 / 5330 if one-row passes were what made s0015 / s0017 / s0019
  vary; a gap of 60-70 ms if the threshold itself does.
- Measured: valid, 9927 / 6203, p99 gap 68.9 / 66.6 ms, no one-row pass served (635 / 175 one-token events are
  two-row passes whose draft was rejected). So one-row passes are not the cause: the threshold of 0.5 alone gives the
  slow behaviour. Four runs with threshold 0.5: p99 gap 71, 52, 62, 69 ms; two runs with 0.3: 52.6, 52.6.
- The p99 gap is the duration of the slowest 1% of cycles, i.e. of 16-row copy passes, which the threshold does not
  touch directly. What the threshold changes around them: fewer rows in the DSpark cycles and fewer runs of the draft's
  second Markov stage. I have no measurement that says why that slows the wide passes (a GPU power state that follows
  the lighter duty cycle is a guess, nothing more). In-process tests did not show it. Unresolved; whoever continues
  should look at per-cycle times by pass width over HTTP before touching the draft policy again.
- Final source unchanged: s0014 (workspace engine/, tags s0014-valid and final).

## s0021: the final source (s0014) a third time: slow. My reading of s0015-s0020 was wrong again
- Expected: 8980 +- 1% / 5350 +- 1%, p99 chunk gap 52-53 ms.
- Measured: valid, 9812 / 5856, p99 gap 65.4 / 63.2 ms, the s0014 event histograms. The source that "reproduces to
  0.1%" is 9% slower here.
- So the slow, varying runs are not a property of the 0.5 draft threshold. Ordering the evaluations by what preceded
  them: s0014, s0016, s0017 each started after 6-7 idle minutes: p99 gap 52-53 ms. s0015 started 5.5 minutes after a
  series of in-process tests: 71. s0019 after a shorter idle: 62. s0020 and s0021 were started within a minute or two
  of the previous evaluation's end: 69 and 65. Heat (or whatever recovers in ~7 idle minutes) explains all seven
  better than the policy does; I had the cool-down rule in my notes and dropped it for the last three submissions
  to save board time.
- What that leaves open: the 0.5 / no-minimum policy has exactly one cool measurement (s0017: 8658 / 5381) against
  two cool ones of s0014 (8982 / 5333, 8975 / 5378): possibly a real -3.6% on synthetic text, not established. The
  s0018 control and the s0020 diagnosis ran warm and say nothing reliable.
- Final source stays s0014: it is the one with two cool, agreeing measurements. Numbers to quote: 8982 / 5333
  (1.53x / 1.56x) when the machine starts cool; 9812 / 5856 (1.40x / 1.42x) when it does not.

## End of the run (5.95 of 6 board-hours, 21 submissions)
