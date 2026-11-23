# Journal: fable-c-m5max-2 (Inkling-2-Mini, M5 Max, track C; continues fable-c-m5max-1, whose records are in run1/)

## 0:00 Where the time goes, from run 1's numbers (before measuring anything)

- Events of the baseline (4 runs of 1024 tokens per workload): synthetic 1414 events, 610 of them one token (the first
  DSpark draft rejected), 89 of 16 tokens; natural 568 events, 192 of 16 tokens (75% of the tokens), 161 of one.
- Time: natural = 192 wide passes x 51 ms (45%) + 376 other cycles at ~32 ms for 1024 tokens (11.7 ms/token);
  synthetic = 12% wide passes + 88% other cycles at ~24 ms, 12.1 ms/token. Plain stepping is 13.2 ms/token, so outside
  loops speculation buys only ~1.1x: that regime is where most of the time is on both workloads.
- A cycle there: draft 3.2 ms + verify of 2-4 rows (17.3 / 19.6 / 22 ms) + 2-3 host syncs, for ~2 tokens.

## s0001: the inherited engine, unchanged
- Expected 8980 / 5330. Measured: valid, 9394 synthetic (repeats 8877 .. 10574), 5637 natural (4346 .. 5894), the
  s0014 event histograms and gate values (numeric p99 0.1013), step 13991, generate 13222, w1_like 15305 (the token
  read on the host every step), p99 chunk gap 61.9 / 59.7 ms. So today's machine state is 4.5% slower than run 1's
  cool runs (52.6 ms gaps) although nothing had run for an hour: the reference for this run's A/Bs is 9394 / 5637.
- step with a host read per token costs 15305 - 13991 = 1.3 ms more than without (part of it attention growth with
  the longer context): a host sync is not free, and a DSpark cycle has 2-3 of them.

## Cheap measurements that decide (all in process)

### NVFP4 decode in the expert kernels (tests/e4.py; hashes of the outputs equal for all three)
| rows | table per code (was) | arithmetic (select + shifts) | table per code byte (float2 x 256) |
|---|---|---|---|
| 1 | 4.22 ms | 5.0 | 4.15 |
| 3 | 8.9 | 14.1 | 8.9 |
| 8 | 18.1 | 34.0 | 18.0 |
| 16 | 32.6-35.3 | 62.8 | 30.4 |
The expert kernels are compute-bound from 3 rows up (arithmetic decode costs +60-90%), so "decode instead of read"
is the wrong direction for them; halving the table lookups (one per code byte) is 7% at 16 rows. Kept (MK_NVDEC=2).

### Lossless 12-bit code for the dense BF16 matrices (tests/w12.py)
- Census, per-row window of 16 exponents (lo = max exponent - 15): 8.8e-5 of the weights fall below the window, no
  zeros; 19% of the rows have at least one such weight.
- Format: 6 bytes per 4 weights (4 bytes sign | mantissa, 4 exponent nibbles), one lo per row; a row with a weight
  outside its window stays plain BF16 in a side array (so no escape handling in the loop, and no change to any sum:
  each weight is decoded to its stored 16 bits before the same fma).
- q|gate|rel|sink and o of all 48 applications (2.13 GB as BF16): bit-identical outputs; K=1 4.44 -> 3.24 ms, K=3
  4.03 -> 3.23, K=8 4.09 -> 3.49, K=16 7.16 -> 6.87. The decode keeps up with the memory: 1.72 GB in 3.24 ms.
- Decision: build it (about -1 ms per small pass, every sum's order kept, so a clean A/B).

### Host syncs (the chained cycle) and the draft's alternatives
- Built: the next cycle's draft traced behind the verify pass (Draft._cyc_pure): the accepted prefix is counted on the
  GPU, the anchor is the target's token at the last accepted row, the context append covers every row of the pass
  (rows past the accepted ones sit at positions the draft masks until a later pass rewrites them). One host sync per
  DSpark cycle instead of 2-3. Served tokens equal the step-by-step tokens (tests/serve.py, 3 prompts x 1024).
- Measured: prose 12.70 -> 12.59 ms/token, code 5.89 -> 5.88. A host sync costs ~0.2 ms here, not the ~1 ms I read
  into w1_like - step. Kept (it is the structure the next ideas need), worth ~1%.
- tests/trace.py: at every position of 11 greedy continuations (1024 tokens), the DSpark chain with the top-3 of each
  step and the chain forced through the second-best first token. First draft right 60-82% on prose, 98% on code /
  GSM; the second-best first token is right in 1-5% of the anchors (12% on the hardest prose), the third in 0-5%.
  Replay with a sibling row for the second-best token at any threshold: 0.0-0.3% (tests/replay.py).
- Decision: no token tree. The alternatives of this draft head are not worth a row (a row costs ~2.5 ms, a token is
  worth ~10): the tree's kernel changes (ancestor tables in the attention and the two conv kernels) would buy nothing.
- My random-token prompts are not the evaluator's synthetic workload: they collapse into loops (13 tokens / cycle
  with copies) while the evaluator's synthetic runs give 2.9. The free-run tokens in my own result decode to fluent
  prose ("In the midst of this tumultuous period, the city of Gotham ..."), so the regime that decides the headline
  is prose at ~2 tokens per cycle. tests/texts2.json adds five prose prompts for that.

## s0002: 12-bit dense matrices, byte-table NVFP4 decode, chained draft
- Expected: step 13990 -> ~13000, synthetic 9394 -> ~8950, natural 5637 -> ~5400, same events and gate values.
- Measured: valid. step 13090, generate 12376 (MBU 0.749), synthetic 9031 (-3.9%), natural 5427 (-3.7%); event
  histograms and numeric values identical to s0001 (p99 0.1013): a clean A/B. It started only ~1.5 minutes after my
  tests (I misread the clock), so it ran warm; a cool rerun may be a little better.

### Where a pass's time is, once more (tests/cpu.py, tests/spin.py + sample)
- Every weight loop cut to one iteration (MK_WHATIF_SHORT=1: same dispatches, buffers and graph, wrong results):
  K=1 7.0 ms, K=3 7.5, K=8 10.9 per pass, against 12.2 / 15.8 / 25.1 for the real passes. So a one-row pass is
  ~5 ms of weight streaming on top of ~7 ms of everything else.
- Process CPU time is 11.5-12.7 ms per pass for every K (user 8 + sys 4). `sample` on a K=1 loop: the main thread
  (MLX evaluates on the caller's thread) spends about half its time encoding (eval_impl -> gpu::eval ->
  CustomKernel::eval_gpu, AGX dispatchThreads, buffer cache, hash sets of the barrier tracking) and half waiting for
  the GPU (MLX keeps at most 10 command buffers in flight). So ~6.4 ms of CPU per pass go into encoding ~1050
  dispatches (~6 us each), and the 7.0 ms of the what-if pass is that CPU path, not the GPU. In the real pass the
  CPU still waits half the time, so the GPU is the bound there; what the what-if cannot tell me is how much of the
  real pass's ~4 ms above its bytes is GPU-side latency of small kernels and how much is the GPU waiting for the
  encoder in the stretches of small dispatches. Only a pass encoded outside MLX (own command buffer) would tell.

### Coarser draft copies (the draft only proposes): no
- 4-bit unembedding copy for the exact-argmax short list: the rigorous bound keeps the whole vocabulary (median
  200058 candidates; 6 bits: median 82-266, mean 850-1900; 8 bits: median 1). tests/short4.py. Not usable.
- Draft matrices at 4 / 5 bits (values rounded, read by the 8-bit kernel): unembedding at 4 bits loses 6-8% of the
  accepted tokens; at 5 bits (unembedding + layers) the five texts of texts.json were unchanged, so I built a packed
  5-bit kernel (8 weights in 5 bytes) - and on five more prose prompts it loses acceptance (story 1.74 -> 1.55
  tokens / cycle, science 1.70 -> 1.53) and is slower overall in the in-process A/B (9.08 vs 8.97 ms/token). Off.

### Draft policy (tests/ab.py: every configuration decodes each text in ABBA order inside one process; all return the
same tokens, so the difference is the policy; ten texts x 512 tokens, ms/token)
| policy | ms/token |
|---|---|
| static, cumulative p >= 0.3, at least one draft row (run 1's) | 9.25 |
| static, cumulative p >= 0.5, no minimum (run 1's unconfirmed one) | 8.90 / 8.95 |
| static 0.6, no minimum | 9.00 |
| calibrated online (P(accept | depth, bucket of p) counted per request), prefix probability >= 0.25 / 0.30 / 0.35 | 9.01 / 8.97-9.00 / 9.10 |
| calibrated, threshold = row ms x the request's recent tokens per ms (clipped 0.2 .. 0.6) | 8.98 |
- The optimum is flat: everything that stops verifying rows at ~25-35% acceptance is within 1%, and 3% better than
  run 1's rule. So run 1's suspicion was right and its -3.6% was real; its evaluations were confounded by heat.
- Replay on the degenerate random-token continuations with copies off: the calibrated rule 8.2 against 9.9 ms/token
  (the static rule verifies 7 rows on text where draft 3 is accepted 18% of the time at p >= 0.97).
- Chosen: calibrated with the rate threshold (MK_POLICY=cala): equal on prose, and it does not depend on the draft's
  probabilities meaning the same thing on every text.

## s0003: draft depth from the online calibration and the request's tokens per ms
- Expected: synthetic 9031 -> ~8700, natural 5427 -> ~5350, tokens per event ~2.6 / 6.9, 16-token events unchanged.
- Measured: valid, 8550 synthetic (-5.3%; repeats 7932 .. 9480), 5425 natural (flat); 2.72 / 6.72 tokens per event,
  one-token events 610 -> 705 and 161 -> 213, 16-token events 90 / 192. Started after 7 idle minutes; the p99 chunk gap
  (59.9 / 57.0 ms) is the same as s0002's, which started warm: the state of the HTTP stage is set by the two and a
  half minutes of evaluation before it, not by what ran before the evaluation.
- Natural text does not move: half of its time is 16-row copy passes, which the policy does not touch.

## s0004: the same engine again (noise), with the system's memory watched
- Measured: valid, 8528 / 5378 (s0003: 8550 / 5425), same histograms: the headline repeats to 0.3%, natural to 0.9%.
- Memory (tests/memmon.sh): the evaluator runs its stages in separate processes one after the other (14.7 GB, 11.6 GB,
  then the server at 19.9 GB resident); free memory goes to ~0.1 GB while an engine is up (25.5 GB free before), but
  pageouts and swapouts do not move. So the HTTP stage is not under memory pressure; my suspicion that two engine
  copies were resident was wrong.

## s0005: control, DSpark drafts only (copy proposals off)
- Expected: synthetic ~9200, natural ~7000.
- Measured: valid, 9118 synthetic (1.51x oMLX + DSpark), 7120 natural (1.17x); 2.35 / 3.80 tokens per event, no event
  above 8 tokens. Run 1's control on its final engine: 9799 / 7437.
- So of the final engine's 8528-8550 / 5378-5425, the copy proposals account for 6% on synthetic and 24% on natural.

### Things measured after s0003 that did not lead anywhere (all in process)
- Short copy matches (tests/replay3.py): a suffix match of 1-4 tokens whose continuation the DSpark chain does not
  share is right about its first token 3-14% of the time; when the chain agrees, the copy adds nothing beyond the
  chain (1.0-1.5 tokens against DSpark's 2.1). A match of >= 5 tokens with full DSpark agreement is right for 14.8 of
  15 tokens, with DSpark disagreeing on the first token 3.5 of 15. Going to 16 rows at once on agreement would not
  pay either: an 8-row and a 16-row pass cost the same per token (3.1 ms). No hybrid proposal built.
- A draft back-off after a cycle in which no draft row was worth verifying: the next cycle still accepts 0.52 drafts
  on average (prose), more than the 0.24 that pays for a 2.4 ms draft. Not built.
- Expert slot layouts at 1-3 rows (tests/e3.py): the current ones are the best; one pair per slot at 2 rows is 7.19
  against 7.38 ms (2.6%, within what I can resolve). A per-slot branch that runs the one-pair loop for single-pair
  slots (bit-identical, MK via experts2(BR=True)): 7.24 -> 7.04 at 2 rows, nothing from 3 rows up. Not wired in.
- The relative-bias table of the attention stored [distance][16] (one cache line per key instead of 16): bit-identical,
  no change at any K (MK_PROJT=1, off).
- 480 extra dependent trivial dispatches per pass (MK_WHATIF_PAD=10): +0.58 ms, i.e. 1.2 us per dependent stage; the
  pass has ~860 such stages, so fusing even a third of them is ~0.3 ms. Not pursued (run 1's merge of 290 mostly
  independent dispatches measured nothing, consistently).
- MLX's command-buffer thresholds (MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER): MLX commits at 50 MB of bound
  buffers, which with 128 MB K/V arrays and 590 MB expert arrays means a commit every 1-3 dispatches. Free-running
  passes gain 2-3% with ops 200 / no MB limit (K=3 15.9 -> 15.45 ms, sys time 3.7 -> 1.3 ms per pass); the served path
  loses with every setting tried (10.55-10.64 ms/token at the defaults, 10.65-10.68 at ops 50 / no MB limit, 11.0 at
  ops 200): after each host sync the GPU waits for the first commit. Left at the defaults.
- The attention dispatch has no large fixed GPU cost: at one key it is ~3 us above a trivial kernel with the same 13
  buffers (tests/attb2.py); the 18-33 us per dispatch of chained microbenchmarks is mostly MLX binding 13 buffers on
  the CPU. In a served cycle the CPU encodes ~6.4 ms per pass while the GPU needs 12-16, so that cost is hidden
  except in the first ~0.5 ms after a sync (graph replay 0.6 ms + first commit).

### More that did not pay (after s0005)
- Host time between a cycle's sync and the next launch is 1.2 ms (graph replay 0.63, the async_eval of the committed
  state ~0.45; n-gram search and calibration are 0.03). Leaving the committed state lazy (evaluated as an input of the
  next pass, MK_LAZY_STATE=1): 8.525 against 8.518 ms/token over eight texts in ABBA order. Nothing: that host time is
  not GPU idle time.
- In-situ GPU time of the small kernels (tests/whatif.py: a class replaced by trivial dispatches with the same
  dependencies, free-running passes, which are GPU-bound): message gates 1.8 ms of a 12.7 ms one-row pass (1.55 of
  15.8 at three rows), attention 1.07 ms at position 600 (value sums 0.45, scores 0.26), exact-argmax short list 0.5,
  depth mixes 0.2, qkv_post and the convs ~0. Experts in situ 4.3 ms and dense 3.9 ms at one row, as their
  microbenchmarks. The attributions are not additive (the gates' four parts sum to 0.66 ms of the 1.8).
- Gates with fewer stages, both bit-identical (tests/bitref.py), neither kept: the first stage folded into the depth
  mix (mix2g): 12.69 against 12.70 ms; also the forget / gain projections folded into the last stage (gate_fgpost):
  13.91 ms, +1.2 ms, the same loss as run 1's single fused kernel. A threadgroup of 576 threads does its per-thread
  loops on as many GPU cores as there are groups (1-3 per call at one row), while the separate projection kernel
  spreads 4608 threads over small groups. MK_GATE3 / MK_GATE_FGPOST, off.
- Attention value sums with 8 / 16 / 32 keys per iteration (loads first, same fma order, bit-identical): 12.69 /
  12.79 / 13.15 against 12.68 ms. Not latency of the loads. MK_ATT_NV stays 4.
- A what-if that returned early from two of three heads made the pass 3.6 ms faster: invalid (unwritten outputs
  propagate), removed. My what-ifs with garbage values are only trustworthy at one row, where the expert count does
  not depend on the values.

## s0006 / s0007: single-pair branch in the expert slots of 2-3 row passes
- The change: a slot of two pairs that holds one runs the one-pair loop (uniform branch per simdgroup; hashes of the
  outputs equal). In process 2 rows 16.35 -> 16.2 ms, 3 rows 18.67 -> 18.55.
- s0006 (started a minute after a full-length test): step 14597, synthetic 8248, natural 5863, p99 gaps 52 / 64 ms.
  A warm start does show, in the in-process stages and here in the natural stage; I was wrong after s0003 to conclude
  that the start state does not matter.
- s0007 (same source, 7 idle minutes): valid, step 13101, generate 12364, synthetic 8249, natural 5283, p99 gaps
  52.5 / 52.4 ms, the s0003 histograms. Against s0003 / s0004 (8550 / 8528, 5425 / 5378, gaps 57-60 ms) that is -3.4%
  and -2%, more than this change can give (0.7% in process): the 16-row cycles themselves were 12% shorter in this
  evaluation, so the machine was in a faster state than an hour earlier. The evaluations of this run are comparable
  to about 4%; the size of each step is what the in-process A/Bs say.

## s0008 / s0009: the inherited engine and the final source, paired in one machine state
- s0008 = the s0001 source again after 7 idle minutes: valid, 8985 synthetic / 5331 natural, step 13946, p99 gaps
  52.4 / 53.0 ms: run 1's cool numbers (8982 / 5333) to 0.1%. So s0001 (9394 / 5637, gaps 60-62 ms) was measured in a
  slower state of the machine, although it too started idle; I do not know what changed between 02:56 and 05:10.
- s0009 = the final source 7 idle minutes later: valid, 8241 / 5276 (s0007: 8249 / 5283), gaps 52.5 ms.
- The run's result on equal terms: **synthetic 8985 -> 8241-8249 (-8.2%), natural 5331 -> 5276-5283 (-1.0%)**.
  The natural workload gains much less than the in-process figures suggested (-4% for the kernels in s0002 against
  s0001, both in the slow state): its headline is the median of three prompts, 75% of whose tokens come from 16-row
  copy passes, where the 12-bit matrices give 0.3 of 49 ms and the byte-table decode 2 ms; and the two states of the
  machine differ most on exactly those compute-bound passes.
- A guess I tested and dropped: the rate that sets the draft threshold counted copy cycles too, which could push
  the threshold to its cap after a loop. Counting DSpark cycles only (MK_RATE_DSPARK=1): 5.807 against 5.798
  ms/token on six 1024-token texts with loops (the static rule: 5.891). No difference; off.

## s0010: the DSpark-only control on the final source, in the machine state of s0007-s0009
- Expected ~8750 / ~6800. Measured: valid, 9021 synthetic (1.53x oMLX + DSpark), 7044 natural (1.18x), 2.35 / 3.75
  tokens per event, no event above 8 tokens (s0005, slower state: 9118 / 7120). Only 1% from the machine's state:
  what differs between the two states is the 16-row passes, and this control has none.
- With copies the final source is 8241-8249 / 5276-5283: the copy proposals are worth 9% on synthetic and 25% on
  natural text.

## End of the engineering (2.9 of 6 board-hours, 10 submissions)
- Final source = engine/ (s0007 / s0009, tag `final`). What is left on my list is below what an evaluation resolves:
  each remaining idea was measured in process first and gave 0-1% or a loss (journal above). The regime that decides
  the headline, prose at ~1.75 tokens per 19.5 ms cycle, is limited by how often the DSpark head's first draft is
  right (60-70%); loops are limited by the GPU's arithmetic at ~3.1 ms per token.

## Resumed (3.1 board-hours): the expert kernels' layout, again
- tests/loop16.py on a 16-row copy pass in a loop (GSM text): 46.7 distinct experts of 128 picks per application; slots
  of 2 pairs: 79 slots, i.e. 17 GB read in 31.3 ms = 540 GB/s. So the wide expert pass is bound by reading each
  expert 1.7 times, and slots of 4 (57 slots) were slower only because of what they compute: padded pairs, and 16
  accumulators per lane with two gate/up row pairs per simdgroup.
- A branch on the slot's pair count (uniform over the simdgroup; a slot runs the n-pair loop for the n pairs it
  holds) and ONE gate/up row pair per simdgroup. All layouts give equal output hashes. Experts of a pass, ms:
  | rows | slots of 2, 2 row pairs (was) | slots of 4, 1 row pair, branch |
  |---|---|---|
  | 2 | 7.10 | 6.95 |
  | 3 | 8.83 | 8.35 |
  | 5 | 12.8 | 11.9 |
  | 8 | 17.9 | 16.4 |
  | 16 (loop) | 31.3 | 28.5 (down kernel 4 rows per simdgroup; 29.1 with 2) |
  Other points at 16 rows: slots of 3: 29.3, of 5-6: 29.6, of 8: 32.2; slots of 4 with 2 row pairs: 34.2 (40.6 without
  the branch). Whole passes on distinct tokens: 3 rows 18.58 -> 18.1, 8 rows 30.1 -> 28.9, 16 rows 50.3 -> 47.4.
  Served in process (alternating): looping text 3.04 -> 2.84 ms/token, code 4.96 -> 4.72.

## s0011: expert slots of 4 pairs, pair-count branch, one gate/up row pair per simdgroup
- Expected: synthetic 8245 -> ~8000, natural 5280 -> ~5000, same histograms.
- Measured: valid, 8105 synthetic (-1.7%), 5124 natural (-3.0%), p99 gaps 53.4 / 50.0 ms, numeric values unchanged.
  Right direction, about half of what the in-process figures promised on synthetic text.

### After s0011, not kept
- Gate/up kernel with one weight row per simdgroup and a separate activation kernel (experts3), to allow slots of
  6-8 pairs: 30.7-33.3 ms against 28.3-30.2 for slots of 4 with a row pair (same hashes). More x loads per weight
  than it saves in reads.
- The 16-row dense GEMV as two neighbouring simdgroups of 8 rows per output (bit-identical): 9.8 ms against 6.9.
  The second read of the weight row is not served from a cache. MK_GEMV_SPLIT=1.
- The same with two sweeps over the weight row inside one simdgroup (8 accumulators each, the second sweep from the
  core's cache): 8.4 ms against 6.9; four sweeps 12.1. So the 16-row dense GEMV is not short of registers; it is
  bound by its 16 activation loads and fmas per 4 weights.
- Expert kernels by format (tests/e5.py): one row NVFP4 441-456 GB/s for every layout, MXFP8 340 -> 380 GB/s with one
  row pair per simdgroup (0.08 ms per pass: not taken); 16 rows with each expert read once: 351-361 GB/s, i.e.
  ~25 ms is the arithmetic floor of a 16-row loop pass's experts, and the engine is at 28.5.

## s0012: the DSpark-only control on the final source (s0011)
- Expected ~8850 / ~6800. Measured: valid, 8808 synthetic (1.56x oMLX + DSpark), 7047 -> 6847 natural (1.21x), 2.34 /
  3.77 tokens per event (s0010, the source before the 4-pair slots: 9021 / 7044). With copies (s0011): 8105 / 5124.
- MXFP8 applications alone under the new layouts (tests/e5.py): all within 0.1 ms of each other per pass; no separate
  layout for them.

## s0013: the final source (s0011) again
- Measured: valid, 8096 / 5142 (s0011: 8105 / 5124), p99 gaps 52.8 / 49.8 ms, same histograms. Reproduces to 0.4%.

### Attention: its time is the latency of one (row, head) group, not its reads
- A what-if that keeps every output finite: two of three heads skip their scores and value sums (MK_WHATIF_ATT=16).
  One-row pass 12.71 against 12.71 ms, while skipping them for all heads gives 11.98. So the attention dispatch takes
  as long as its slowest threadgroup and the GPU has cores to spare for the others: sharing the K/V reads among the
  three heads of a kv-head (which I had estimated at 0.2-0.4 ms) would gain nothing. Not built.
- Together with the gate fusion that got slower when work moved into a 576-thread group, this suggests that a
  threadgroup's work runs on one core, so the small kernels with large threadgroups (attention 512, mixes and gates
  576) are bound by per-group work, not by the pass's width. Splitting them into several dispatches of small groups
  (scores per key, sums per simdgroup, a combine stage) would have to keep every simd_sum's lanes; each added
  dependent stage costs 1-4 us. Unmeasured, and the largest thing I know of that is left in a pass (attention ~1.1
  ms and gates ~1.8 ms of a 12.7 ms one-row pass).

### Policy parameters with the final kernels (tests/ab.py, ABBA in one process)
- Threshold floor 0.15 / 0.20 / 0.25 and row cost 2.0 / 2.5 / 3.0 ms: 8.79-8.85 ms/token, the defaults (0.2, 2.5) best.
- Markov steps before the first look at the probabilities: 3 (8.277), 2 (8.300), 4 (8.320), 1 (8.365), all 7 in the
  chained graph (8.452). Stays 3.
- Copy proposals from a longer match: 8 tokens 5.975-5.982 against 6.030-6.032 ms/token for 5 (seven 1024-token
  texts), flat to 12, worse at 3-4.
- A DSpark veto on a copy whose first token the draft in hand contradicts: -0.7% on those texts, and on the looping
  random prompt 108 cycles instead of 74 (+20%): where the draft cannot predict the loop it vetoes the copy at every
  entry. Off (MK_NG_AGREE).

## s0014 / s0015: copy proposals from 8 matching tokens: not kept
- s0014: 9252 / 5788 with 16-row cycles of 63-65 ms: the machine's slow state, after 25 minutes of in-process A/Bs
  and 7 idle minutes. So that state is what long test sessions leave behind, and 7 idle minutes do not clear it;
  s0001-s0005 were measured in it for the same reason.
- s0015 (same source, 15 idle minutes): valid, 8165 synthetic / 5100 natural, cycles 52.9 / 49.8 ms; 16-token events
  87 / 190 instead of 90 / 192. Against s0011 / s0013 (8105 / 8096, 5124 / 5142): +0.8% and -0.6%. A wash; the
  in-process -0.9% does not carry over to the evaluator's prompts. Reverted to 5: the final source is s0011's.

## s0016 / s0017: the workspace as it is left
- s0016 (s0011's engine, later switches off, but the 12-bit GEMV still in its rewritten sweep-loop form): valid,
  8203 / 5165, step 13163, cycles 54.5 / 50.6 ms, the s0011 histograms.
- s0017 (the GEMV back to the s0011 text; the two forms time the same in process: 13.53-13.67 ms one row, 18.02-18.25
  three rows): valid, 8333 / 5300, cycles 57.3 / 54.3 ms. Slower with the kernel that measured 8105 / 8096: the
  machine again, drifting after an hour of evaluations and tests with 12-minute pauses. So s0014-s0016's +0.6% in
  step time was the machine too, not the kernel.
- The final source has now been measured four times: 8105, 8096, 8203, 8333 synthetic and 5124, 5142, 5165, 5300
  natural, with 16-row cycles of 53.4, 52.8, 54.5, 57.3 ms: the number follows the cycle time. Quote the first two
  (the state in which the inherited engine measured 8985 / 5331) and keep the spread next to them.

## End of the run (5.5 of 6 board-hours, 17 submissions)
- Final source = engine/ (tag `final`), s0011's engine. What moved the number in this run: lossless 12-bit dense
  matrices, the draft depth from an online calibration, expert slots of 4 pairs with a pair-count branch, and small
  items (byte-table NVFP4 decode, chained draft). What the measurements ruled out is in notes/summary.md.

## s0018: the final source after 16 idle minutes (last evaluation)
- Measured: valid, 8040 synthetic (1.71x oMLX + DSpark), 5143 natural (1.62x), step 13139, 16-row cycles 49.7 / 49.5
  ms, the s0011 histograms. The best line of the run and the coolest machine: five evaluations of the final source
  now read 8040, 8096, 8105, 8203, 8333 with cycles of 49.7, 52.8, 53.4, 54.5, 57.3 ms.
- End of the run at 5.85 of 6 board-hours, 18 submissions. Final source = engine/ (tag `final`).

## s0019: the final source a few minutes after s0018
- Expected 8200-8350 / 5200-5300 with longer 16-row cycles. Measured: valid, 8061 / 5134, cycles 49.8 / 49.5 ms: as
  s0018. Wrong prediction: one evaluation after a long idle does not push the machine into its slower state; that
  takes the long in-process sessions (25 minutes of A/Bs before s0014) or a run of evaluations with short pauses.
- Six evaluations of the final source: 8040, 8061, 8096, 8105, 8203, 8333 synthetic; 5124-5300 natural.
- End of the run: 5.92 of 6 board-hours, 19 submissions.
