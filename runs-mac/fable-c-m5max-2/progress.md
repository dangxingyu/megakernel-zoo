# fable-c-m5max-2: Inkling-2-Mini on an Apple M5 Max (40-core GPU, 48 GB)

Score: `http_step_us_b1` (lower is better). Budget 6.0 agent-hours. Track C. 19 submissions, 19 valid.

Continues runs-mac/fable-c-m5max-1 (seed engine: its s0014).

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

### #1 s0001: Baseline: the earlier run's final engine (s0014 source) unchanged

- status: **valid**, http_step_us_b1 = 9394, at 0.13 agent-hours, category other
- hypothesis: The seed engine as inherited, started on an idle machine: expect about 8980 synthetic / 5330 natural (2.90 / 7.21 tokens per event), step about 13970, as the earlier run's cool measurements. This is the reference for every A/B of this run.
- evaluator: valid: 9394 us/step at B=1 over HTTP (online (submit/poll)); in process 13222 us/step at B=1 (generate, MBU 0.700)

### #2 s0002: Lossless 12-bit dense matrices, byte-table NVFP4 decode, draft chained behind the verify pass

- status: **valid**, http_step_us_b1 = 9031, at 0.50 agent-hours, category gemm
- hypothesis: Three changes that keep every logit bit (tests/bitref.py identical for K = 1, 3, 8, 16), so the continuations and gate values of s0001: (1) q|gate|rel|sink, k, v, o of every layer read from a lossless 12-bit re-encoding (6 bytes per 4 weights, per-row exponent window, rows with an outlier kept as BF16): the two largest matrices 4.4 -> 3.2 ms per one-row pass, 4.0 -> 3.2 at three rows; (2) NVFP4 codes decoded through a table per code byte: experts of a 16-row pass 32.6 -> 30.4 ms; (3) the next cycle's DSpark draft runs behind the verify pass in one graph (acceptance counted on the GPU): one host sync per DSpark cycle instead of 2-3, worth about 1% in process. Expect step 13990 -> about 13000, synthetic 9394 -> about 8950 (-5%), natural 5637 -> about 5400 (-4%), same event histograms.
- evaluator: valid: 9031 us/step at B=1 over HTTP (online (submit/poll)); in process 12376 us/step at B=1 (generate, MBU 0.749)

### #3 s0003: Draft depth from an online calibration and the request's tokens per ms

- status: **valid**, http_step_us_b1 = 8550, at 1.08 agent-hours, category scheduling
- hypothesis: Policy only (target logits untouched, same continuations): the rows of a DSpark cycle are chosen while the calibrated probability that the whole prefix is accepted (P(accept | depth, bucket of the draft's probability), counted per request from a prior) stays above row-ms x the request's recent tokens per ms (clipped 0.2-0.6); no minimum row. In process, ten texts in ABBA order, it is 3% faster than the static rule (cumulative p >= 0.3, at least one row): 8.98 vs 9.25 ms/token, with fewer tokens per event. Started after 7 idle minutes (s0002 started warm). Expect synthetic 9031 -> about 8700, natural 5427 -> about 5350, tokens per event down to about 2.6 / 6.9, 16-token events unchanged (89 / 192).
- evaluator: valid: 8550 us/step at B=1 over HTTP (online (submit/poll)); in process 12376 us/step at B=1 (generate, MBU 0.749)

### #4 s0004: The s0003 engine again, with the system's memory watched during the evaluation

- status: **valid**, http_step_us_b1 = 8528, at 1.46 agent-hours, category other
- hypothesis: No functional change against s0003 (switches added since are off: transposed relative-bias table, single-pair branch in the expert slots, padding dispatches). A repeat for the noise of the headline, and a look from outside (vm_stat, ps every 3 s) at whether the HTTP stage runs under memory pressure: the Mac had 25.5 GB free before the run and 12 of 13 GB of swap in use, and the engine holds about 20 GB. Expect 8550 +- 3% / 5425 +- 3%, the s0003 histograms.
- evaluator: valid: 8528 us/step at B=1 over HTTP (online (submit/poll)); in process 12426 us/step at B=1 (generate, MBU 0.746)

### #5 s0005: Control: the s0003 / s0004 engine with DSpark drafts only (copy proposals off)

- status: **valid**, http_step_us_b1 = 9118, at 1.61 agent-hours, category scheduling
- hypothesis: A control, not a candidate: copy (n-gram) proposals switched off, so every draft comes from the checkpoint's DSpark head; target logits and continuations as s0003. The earlier run's control on its final engine was 9799 synthetic / 7437 natural (2.56 / 4.14 tokens per event). With the 12-bit matrices, the chained draft and the calibrated depth (which stops verifying deep rows on repeating text where the static rule kept 7), expect synthetic about 9200 and natural about 7000.
- evaluator: valid: 9118 us/step at B=1 over HTTP (online (submit/poll)); in process 12375 us/step at B=1 (generate, MBU 0.749)

### #6 s0006: Single-pair branch in the expert slots of 2-3 row passes (same bits)

- status: **valid**, http_step_us_b1 = 8248, at 2.22 agent-hours, category gemm
- hypothesis: One bit-identical kernel change on the s0004 engine: in passes of 2-3 rows most expert slots hold one (row, expert) pair, and the slot kernel computed its empty second pair; a branch that is uniform over the simdgroup now runs the one-pair loop for those slots. In process the 2-row pass is 16.35 -> 16.2 ms and the 3-row pass 18.67 -> 18.55 (tests/passt.py), logits bit-identical (tests/bitref.py). Expect synthetic 8528-8550 -> about 8480 (-0.7%), natural unchanged within its 1% repeat noise, the s0003 histograms.
- evaluator: valid: 8248 us/step at B=1 over HTTP (online (submit/poll)); in process 13610 us/step at B=1 (generate, MBU 0.671)

### #7 s0007: The s0006 source again, started after 7 idle minutes

- status: **valid**, http_step_us_b1 = 8249, at 2.40 agent-hours, category other
- hypothesis: s0006 started a minute after a full-length test of mine and its in-process stages ran 11% slow (step 14597 against 13085), natural 5863 with a p99 chunk gap of 64 ms, synthetic 8248 with 52 ms: a warm, mixed measurement. Same source, cool start: expect step about 13090, synthetic about 8480, natural about 5400, the s0003 histograms.
- evaluator: valid: 8249 us/step at B=1 over HTTP (online (submit/poll)); in process 12364 us/step at B=1 (generate, MBU 0.748)

### #8 s0008: The inherited engine (s0001 source) again, in the machine state of s0007

- status: **valid**, http_step_us_b1 = 8985, at 2.60 agent-hours, category other
- hypothesis: A second baseline: s0001 measured 9394 / 5637 with 16-row cycles of 60-62 ms; s0007 ran with 52.5 ms cycles, so the two are not in the same machine state. The s0001 source again after 7 idle minutes, to be followed by the final source: if the machine is still in the faster state, expect about 8980 / 5330 (run 1's cool numbers) and a p99 chunk gap near 52.6 ms.
- evaluator: valid: 8985 us/step at B=1 over HTTP (online (submit/poll)); in process 13216 us/step at B=1 (generate, MBU 0.703)

### #9 s0009: The final source (s0007) again, right after the baseline's re-measurement

- status: **valid**, http_step_us_b1 = 8241, at 2.78 agent-hours, category other
- hypothesis: Pairing with s0008 (the inherited engine in the current machine state: 8985 / 5331, 16-row cycles 52.4-53 ms): the final source after the same 7 idle minutes. Expect 8249 +- 1% synthetic and 5283 +- 2% natural, i.e. -8% and about -1% against s0008.
- evaluator: valid: 8241 us/step at B=1 over HTTP (online (submit/poll)); in process 12449 us/step at B=1 (generate, MBU 0.748)

### #10 s0010: Control again: the final source with DSpark drafts only

- status: **valid**, http_step_us_b1 = 9021, at 3.05 agent-hours, category scheduling
- hypothesis: s0005, the DSpark-only control (9118 / 7120), ran in the machine's slower state. The same control on the final source (copy proposals off, nothing else) in the state of s0007-s0009, after 7 idle minutes: expect about 8750 synthetic and 6800 natural (-4% as the other evaluations moved), 2.35 / 3.80 tokens per event.
- evaluator: valid: 9021 us/step at B=1 over HTTP (online (submit/poll)); in process 12367 us/step at B=1 (generate, MBU 0.749)

### #11 s0011: Expert slots of 4 pairs, pair-count branch, one gate/up row pair per simdgroup

- status: **valid**, http_step_us_b1 = 8105, at 3.42 agent-hours, category gemm
- hypothesis: Bit-identical kernel layout change on the s0009 source (tests/bitref.py; equal output hashes for every layout): from 2 rows up the expert kernels use slots of up to 4 (row, expert) pairs with a uniform branch on the slot's pair count and one gate/up row pair per simdgroup. Run 1 found slots of 4 slower than slots of 2; with the branch (no padded pairs) and half the accumulators they are faster: experts of a pass 8.83 -> 8.35 ms at 3 rows, 17.9 -> 16.4 at 8, 31.3 -> 28.5 at 16 (a loop); whole passes -2.6% at 3 rows, -5.6% at 16; served in process -6.8% on a looping text, -4.8% on code. Expect synthetic 8245 -> about 8000 (-3%), natural 5280 -> about 5000 (-5%), same histograms, started after 7 idle minutes.
- evaluator: valid: 8105 us/step at B=1 over HTTP (online (submit/poll)); in process 12412 us/step at B=1 (generate, MBU 0.748)

### #12 s0012: Control: the final source (s0011) with DSpark drafts only

- status: **valid**, http_step_us_b1 = 8808, at 3.68 agent-hours, category scheduling
- hypothesis: The DSpark-only control on the final source: s0011 with copy proposals off and nothing else changed. s0010 (the source before the 4-pair expert slots) was 9021 / 7044; the new layout is worth 2.6% on 3-row passes and 4% on 5-8 row passes in process, so expect about 8850 synthetic and 6800 natural, 2.35 / 3.75 tokens per event.
- evaluator: valid: 8808 us/step at B=1 over HTTP (online (submit/poll)); in process 12351 us/step at B=1 (generate, MBU 0.749)

### #13 s0013: The final source (s0011) again

- status: **valid**, http_step_us_b1 = 8096, at 3.87 agent-hours, category other
- hypothesis: Repeat of s0011 (8105 / 5124) after 7 idle minutes, for its reproducibility: the previous final source repeated to 0.1% (8249 / 8241, 5283 / 5276). Expect 8105 +- 1% and 5124 +- 2%, the same histograms.
- evaluator: valid: 8096 us/step at B=1 over HTTP (online (submit/poll)); in process 12451 us/step at B=1 (generate, MBU 0.743)

### #14 s0014: Copy proposals from 8 matching tokens instead of 5

- status: **valid**, http_step_us_b1 = 9252, at 4.65 agent-hours, category scheduling
- hypothesis: Proposal policy only (same target bits): a copy is proposed when the last 8 committed tokens occurred before, not 5. Short matches produce 8-row passes that fail early; in process, seven 1024-token texts in ABBA order: 5.975-5.982 against 6.030-6.032 ms/token (-0.9%), flat from 8 to 12, and unchanged on a pure loop. Expect synthetic 8100 -> about 8050 and natural 5130 -> about 5080, fewer events of 2-7 tokens from failed copies, 16-token events within one or two of 90 / 192; this is at the edge of what an evaluation resolves (0.4%).
- evaluator: valid: 9252 us/step at B=1 over HTTP (online (submit/poll)); in process 12494 us/step at B=1 (generate, MBU 0.744)

### #15 s0015: The s0014 source again after 15 idle minutes

- status: **valid**, http_step_us_b1 = 8165, at 4.96 agent-hours, category other
- hypothesis: s0014 (copies from 8 matching tokens) ran with 16-row cycles of 63-65 ms: the machine's slow state again, after 25 minutes of in-process A/Bs and only 7 idle minutes; its 9252 / 5788 says nothing about the change. Same source after 15 idle minutes: if the machine is back (cycles near 52 ms), expect about 8050-8150 synthetic and 5080-5180 natural; tokens per event 2.67 / 6.67.
- evaluator: valid: 8165 us/step at B=1 over HTTP (online (submit/poll)); in process 12441 us/step at B=1 (generate, MBU 0.744)

### #16 s0016: Final source: the s0011 engine (copies from 5 tokens), last check

- status: **valid**, http_step_us_b1 = 8203, at 5.23 agent-hours, category other
- hypothesis: The workspace as it is left: s0011's engine with later switches off (8-token match reverted, veto off, split GEMV off; logits bit-identical, tests/bitref.py). After 12 idle minutes expect s0011 / s0013 again: 8100 +- 1% synthetic, 5130 +- 1.5% natural, 2.71 / 6.69 tokens per event, 16-row cycles near 50-53 ms.
- evaluator: valid: 8203 us/step at B=1 over HTTP (online (submit/poll)); in process 12507 us/step at B=1 (generate, MBU 0.744)

### #17 s0017: Final workspace: 12-bit GEMV in its s0011 form

- status: **valid**, http_step_us_b1 = 8333, at 5.51 agent-hours, category other
- hypothesis: s0014-s0016 carried a rewritten 12-bit GEMV (a sweep loop that runs once) and their step time was 0.6% above s0011 / s0013 (13163-13172 against 13080-13101); in process the two forms time the same. The kernel is back to the s0011 text, everything else as s0016. After 12 idle minutes expect step about 13090, 8100 +- 1% synthetic, 5130 +- 1.5% natural.
- evaluator: valid: 8333 us/step at B=1 over HTTP (online (submit/poll)); in process 12398 us/step at B=1 (generate, MBU 0.747)

### #18 s0018: Final source once more, after 16 idle minutes

- status: **valid**, http_step_us_b1 = 8040, at 5.85 agent-hours, category other
- hypothesis: No change against s0017 (8333 / 5300 with 16-row cycles of 57 ms, the machine drifting slower over an hour of evaluations). After the longest idle of the run, if the cycles are back near 52-53 ms expect s0011's 8100 +- 1% and 5130 +- 1.5%; if they are not, the number will sit with s0016 / s0017 and say so.
- evaluator: valid: 8040 us/step at B=1 over HTTP (online (submit/poll)); in process 12392 us/step at B=1 (generate, MBU 0.746)

### #19 s0019: Final source, started a few minutes after the previous evaluation

- status: **valid**, http_step_us_b1 = 8061, at 5.92 agent-hours, category other
- hypothesis: No change against s0018 (8040 / 5143 after 16 idle minutes, 16-row cycles 49.6 ms). Started only a few minutes after s0018 ended: a data point for how much a short pause costs. Expect 16-row cycles of 53-57 ms and about 8200-8350 synthetic, 5200-5300 natural, the same histograms.
- evaluator: valid: 8061 us/step at B=1 over HTTP (online (submit/poll)); in process 12385 us/step at B=1 (generate, MBU 0.745)
