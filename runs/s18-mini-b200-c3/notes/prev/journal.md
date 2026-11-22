# Journal (run s16-mini-b200-c1, track C)

## s0001 track C v1 — 2877 us b1 (natural 1595), b8 7055 (natural 7303); tokens/event 2.79 / 4.93 (nat)
Expected ~3200 / ~1900 / ~7100. Measured better at B=1 (acceptance over 1024 tokens higher than my 256-token runs).
Mechanism: verify pass = mk_step<true> (rows = consecutive positions of one sequence; in-pass k/v computed in the
attention item from the tagged qkv outputs, conv lags of in-pass rows from the producing rows: shuffles in o_proj,
smem in w2b; ring of 16 conv slots), bit-identical to step() (tests/c/vcheck.py 0 of 1200 positions; the evaluator's
generate/http_tokens gates 8/8). Draft = torch ops in a CUDA graph (2.15 ms/cycle), verify 8 rows ~7.2 ms (ctx 600).
Lesson on the way: one extra 512 B of smem pushed the step kernel over the 132 KB carve-out (+200 us); the general
per-row code in the step kernel cost +160 us until the kernel became a template (V = verify) with V=false code = old.

## after s0001 (not yet submitted)
- Shared attention records for verify passes (global layers / positions < 1023): bit-identical (vcheck); attention
  phase 27.7 -> 20.6 us per application after batching the conv-lag polls and merging q/r tasks; 8-row pass 6518 -> 6264.
- w2a one warp per item (3 chains in block_strips order): bit-identical, w2a work 16.4 -> 14.7 us, pass 6264 -> 6031.
- w13 one warp per item: SLOWER (work 19.4 -> 25.5 us): 6 serial batches of 6 units per warp, each ~4 us (load
  latency under load ~2.5 us + ~1.3 us of decode/mma per batch with the loads idle). Kept off (W13_ONEWARP 0).
- Expert union of 8 consecutive positions (random-token context): 21.3 per application (13..43); MoE time
  ~10.6 + 1.8 us per union expert (w13 + w2), i.e. ~2.5 TB/s marginal.
- 64 KB more dynamic smem for the verify kernel costs ~50 us per pass (6036 -> 6086): smem staging is affordable there.
- n-gram drafts (engine/ngram.py, suffix >= 3 tokens -> copy the continuation, else DSpark): offline at the same
  anchors 2.74 -> 3.09 (evaluator), 4.54 -> 4.96 (natural); served (torch draft): evaluator tokens/event +4..7%,
  natural prompt 0 5.25 -> 6.87, prompt 1 6.13 -> 5.95, prompt 2 3.53 -> 3.70.
- Fused CUDA draft (fork, engine/draftk.cu): 534 us per cycle vs 2150; served natural 1879 -> 1536, eval 3204 -> 2617.

## s0002 track C v2 — 2158 us b1 (natural 1221), b8 7045; tokens/event 2.96 / 5.35
Expected ~2200 / ~1180 / 7050, 2.95 / 5.3: confirmed within 4%. Fused draft (-1.6 ms/cycle), n-gram drafts (+6% tokens per
cycle on b1, +8.5% natural), verify -0.45 ms. Cycle now ~6.6 ms at ctx 600, of which the 8-row verify pass ~6.0 ms (90%).
vLLM + DSpark: 1544 / 871 (1.40x / 1.40x ahead).

## s0003 dynamic draft length — 1764 us b1 (natural 1194), b8 7043; tokens/event 2.89 / 5.28
Expected ~1750 / ~1120: b1 confirmed, natural gained only 2% (1221 -> 1194) vs 8% in my in-process 4-prompt test.
Mechanism: kd = longest prefix with prod p_i >= 0.4 (fused draft computes p_i = exp(zmax - lse) per Markov step,
deterministic); verify reads K = kd + 1 from vpos[1]; n-gram proposals keep 8 rows. theta 0.25 / 0.4 / 0.55 flat.
w2a per-block work (K=8, -DMKPROF, tests/c/w2stat.py): grows with tokens per expert: 12.3 us (<= 3 tokens), 15.3 (4),
17-22 (5-8, NG=2), MXFP8 23-40; maxima ~2x the means (waits).

## s0004 K=2 redundant mixes + w2a work balance + n-gram kernel — 1705 us b1 (natural 1143), b8 7048
Expected 1705 / 1145: confirmed. In-process served A/Bs (tests/c/serve_c.py, 4 prompts x 1024) predict the HTTP
figures well when scaled by the s0003 ratio (HTTP / in-process 0.80 eval-like, 0.95 natural).
- w13 through a per-warp cp.async smem ring (7 units deep, 60 KB extra smem): bit-identical but much SLOWER
  (K=2 4132 -> 4698, K=8 5916 -> 7182 us). Off (W13_ASYNC 0). The register-batch loop with hoisted address math wins.
- w13 rolling double buffer of 4 (3) unit batches in registers (verify kernel): bit-identical, SLOWER +430 (+320) us
  at K=2. Off (W13_ROLL 0). Third MoE streaming attempt that lost to the plain 6-unit batches at 80 registers.
- W2_STEP=1 (one-warp w2a + balance in the step kernel at B >= 2): B=8 6180 -> 5989 us, B=1 +35 us (code generation);
  a track-A change, not submitted.

## s0005 draft early exit + draft GEMVs + n-gram min match 5 — 1697 us b1 (natural 1129), b8 7044
Expected ~1665 / ~1135: natural confirmed, b1 only -0.5% (predicted -2.3%). My eval-like set (ids.npy prompts) is not
the HTTP stage's prompt set: the n-gram threshold gain (-1.8% there) did not carry over. Tokens/event 2.88 / 5.25.
- kv_conv4 (the 4 channels' loads of a k/v conv issued before any wait): verify kernel SLOWER (+150 us at K=2, +125
  at K=8), step kernel unchanged: the four sequential polls were not on the critical path. Off (KV4 0).

## s0006 (track A) B>=2 step kernel instance with one-warp, work-balanced w2a — 3227 us b1, 6356 b8
Bit-identical to s15 s0016 (bitcheck 1152 positions B=1 and B=8). In process (768 greedy steps) B=8 6595 -> 5961 us.
Predicted b1 ~3255 unchanged, b8 7089 -> ~6450: confirmed (b8 -10.3%, b1 -0.7% = noise). The best track-A engine is now
this one (engine/TRACK = A builds it); the track-C engine uses the same step kernel for its B=8 cohorts.
- Shared attention: the in-pass positions' raw k/v kept in smem by the k/v tasks (the cache / ring writes no longer
  re-poll them): K=4 -58 us, K=8 -146 us (5918 -> 5772), K=2/3 unchanged; bit-identical (vcheck 1100 mixed, 700 k8).
- Not done (estimated < 1%): merging in the chunk-0 record item for K <= 2 (needs 2 rows' merge staging in smem);
  shared records for local layers past position 1023 (only ~6% of the evaluator's tokens).
- bench/tma.cu (148 x 768 threads, 1 GB stream, steady state): the engine's NVFP4 register-load pattern (batches of 6
  units) 6.90 TB/s; with the real decode + mma per unit 6.68 (NG=1) / 5.92 TB/s (NG=2); cp.async.bulk per warp into
  smem 7.4 TB/s. So neither the load path nor the decode limits the MoE phases (~4 TB/s effective, 63 MB in ~20 us at
  K=8): the loss is phase-level (ramp after routing, the epilogue warp's idle loads, partly filled last rounds).
- w13 rounds (sub-marks 90..99, verify K=8 / K=4): a full round of 8 items (166 KB per SM) takes 6.3-6.5 us (~3.8 TB/s);
  partial last rounds 4.3-5.8 us. The benchmark's steady state would be ~3.7 us per round: ~3.2 us per 6-unit batch in
  the engine vs ~1.8 us in the benchmark. Not resolved (candidates: access locality across the 8 concurrent items,
  the epilogue warp delaying its group's next round). Stopped here.
- w13 epilogue split over the group's 3 warps (bit-identical): SLOWER (K=8 +45, K=4 +30 us). Reverted.
- Draft (fork): programmatic dependent launch across the draft's kernels: -25 us per cycle (~400 us with one Markov
  step); fusing attention post-processing / merges was slower (+8 / +36 us), kept off.

## s0007 raw k/v in smem (shared attention) + draft PDL + B>=2 step kernel — 1673 us b1 (natural 1106), b8 6352
Expected 1673 / 1108 / 6356 / natural_b8 ~6580: all confirmed within 0.2%. In-process served A/B (4 + 4 prompts x 1024)
x the per-workload HTTP ratio predicts the evaluator to ~0.2% now.
K=2 vs K=1 per application now: plain 77.0 vs 53.7 (mixes +3.5 each for the second row computed in every block,
attention +3.8, w13 +6, w2 +4.6); gated 94.4 vs 77.3.

## s0008 FP8 draft copies + n-gram inside the draft + designated final rows — 1619 us b1 (natural 1067), b8 6351
Expected ~1636 / ~1064: b1 better (-3.2%), natural confirmed. The draft's FP8 E4M3 per-row-scaled copies of the base
unembedding and fold (draft only; the target is untouched) cost -0.3% tokens/cycle and save ~60 us/cycle; n-gram
cycles now skip the DSpark block / Markov kernels (121 us instead of ~400 us of draft). vLLM+DSpark: 1544 / 871.
- Merge inside the chunk-0 record item for 2-row passes (bit-identical): SLOWER (K=2 +97 us; K=3..8 +20..50 from code
  generation). Reverted. Sweep with the s0008 engine: theta 0.3 / 0.5 and n-gram minimum 4 all within 0.3% of the
  defaults (0.4, 5).

## s0009 (track A) raw k/v for the conv ring from smem in the step kernels — 3206 us b1, 6359 b8
Expected ~3200 / ~6340: b1 confirmed (-0.7% vs s0006 3227), b8 unchanged within noise. Bit-identical to s0016.
Best track-A engine = this source with engine/TRACK = A. FIN_DES 2 vs 3: no difference.
- Gate D items (verify kernel): the loop-message mix (sources ready since the previous pass) before the x mix that
  waits for the current partials: bit-identical; K=2 -13, K=3 -40, K=4 -7, K=8 -18 us.
- Draft (fork): FP8 copies of the draft layers' weights: -10 us per DSpark cycle only (layer GEMVs latency-bound),
  tokens/cycle unchanged; left off (MK_DK_FP8=all enables).
- nsys of a served natural request (s0008 engine): cycle 4396 us = verify 4008 + draft span 111 (PDL overlaps its
  210 us of kernels) + ~8 us idle before the next verify: no host gaps left; the verify pass is 91% of the cycle.
- Verify K=1 vs step B=1 (crit.py): attention +3 us (general per-row path), w2 +4 us (one warp per item with only ~8
  items per block). Using 3 warps per item below 12 / 16 / 24 items per block: SLOWER at every K >= 2 (+100..170 us).
  Kept one warp per item always (W2_1W_MIN 0); K=1 passes do not occur in cycles (kd >= 1).

## s0010 gate D message-first mixes (verify) — 1617 us b1 (natural 1064), b8 6339; -0.15% (expected -0.5%)
- __noinline__ shared attention: K=2..4 +30..60 us, K=8 -65 us: mixed, off. vcheck 4000 positions: 0 differ.
- K=1 passes when p1 < 0.3: ~6% of cycles, ~0.35% estimated gain: not done.
- Started (fork, proto16/): 16-row verify passes for n-gram proposals (B-operand rows in 8-row halves per phase).
- Offline (cycles.json, same anchors): n-gram anchors with an 8-token match give 7.23 / 7.36 tokens per cycle with 8 rows
  and 13.47 / 13.75 with 16 rows (evaluator / natural). A 16-row pass of ~8.5 ms would cut those cycles from ~0.80 to
  ~0.63 ms/token.
- 16-row verify passes (fork, proto16/, timing proxy: every smem-bound phase run twice per application, results
  unchanged): K=8 5740 -> 8660 us with all doubled, 7767 with only MoE doubled. Estimated V(16): ~9.6-9.8 ms with
  8-row halves (n-gram cycles 0.80 -> ~0.73 ms/token, ~-6% overall), ~8.6 ms if MoE takes 16 tokens per expert in one
  pass (~-12% overall; needs a new smem layout: 148 KB expert inputs, router rows from global, MAXU 128, RING 32,
  qkv / gate norms in halves). Not built: estimated 4-6 h with real bug risk vs ~5 h left. Recorded for a later run.

## s0011 poll only the needed qkv input streams (verify) — 1598 us b1, b8 6343
Expected ~1595 / ~1050: b1 confirmed. In the step kernel (B=8) the same change is within noise (-0.6%): off.

## s0012 hybrid mixes for 2-row verify passes — 1588 us b1, b8 6366
Expected ~1585: confirmed. Row 0 mixed in every block, row 1 by the last block and published: one mix + one hop.
- Hybrid mixes for 3 / 4 rows (HYB_MAX 3 / 4): bit-identical, much slower (K=3 4250 -> 4610 us). Kept 2 rows only.

## s0013 designated rows on the last blocks for 3-4 row passes — 1587 us b1, b8 6376 (within noise, as predicted)
- 16-row verify passes built in proto16/ (fork): bit-identical (vcheck random K 1..16, 1100 positions; K=16 1020
  positions) but V(16) = 10.14 ms (V(12) 8.70): 8-token n-gram cycles would go 0.79 -> ~0.76 ms/token only, and the
  extra code made K<=8 slower (K=2 +290 us). Dead end with MoE in 8-row halves; one-pass MoE for 16 tokens needs 148 KB
  of expert-input smem (does not fit; a BF16 path would not be bit-identical to step()). Not integrated.
- Soak (tests/c/soak.py): 40 back-to-back requests (36 track-C B=1, 4 cohorts of 8), mixed prompt sources / lengths /
  n: 0 differing from step-by-step greedy. FP8_UNR 2 / 4, FP4_UNR 8: no change. Gate-norm hybrid at 3 rows: +90 us.
- n-gram majority vote over all occurrences of the matched suffix (offline): 6.80 / 7.13 vs most recent 6.79 / 7.17:
  no gain. Final HEAD check: vcheck 1100 mixed 0 differ; served eval-like / natural streams identical.
- K distribution (tests/c/khist.py, final engine, 4 + 4 prompts x 1024): evaluator-like K=2 34% (1.29 tokens/cycle),
  3 21% (1.78), 4 14% (2.12), 5 7%, 6 4%, 7 2%, 8 19% (6.11); natural K=2 18% (1.23), 3 8%, 4 4%, 5-7 7%, 8 63% (7.25).
  Per token: K=2 cycles ~3.1 ms (still better than K=1 at 3.5), K=8 n-gram/confident cycles ~0.8-0.9 ms.
  V(K) vs context: K=1 3385 / 3499 / 3545, K=2 3880 / 4028 / 4098, K=8 5664 / 5669 / 5843 us at ctx 100 / 600 / 1000.
  The evaluator-prompt headline is dominated by short cycles whose first draft is accepted ~29% of the time.
- HEAD step path vs s15 s0016 (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8): bit-identical, batch-invariant.
- Short n-gram matches (2-4) when DSpark's p1 < 0.8 (offline): +0.15 tokens on ~6.5% of evaluator cycles: < 0.5%, not done.
- Certified INT8 argmax in the verify pass's final phase (INT8 per-row copy, rigorous bound on |exact - approx|,
  exact recompute of the 1-3 candidate strips per row): tokens exact (vcheck tok mode 0 of 1100 mismatches) but no
  gain: the INT8 pass is int8 -> bf16 conversion bound (K=8 -67 us without the recompute) and the second pass + grid
  barrier take most of it back (net -4 us at K=2, -27 at K=8). Off (MK_UN8=0 default; the copy is not built).

## s0014 final source (s0013 + off-by-default switches) — 1585 us b1: valid, as expected (noise vs s0013 1587)
- Soak 100 requests (90 track-C B=1 + 10 cohorts of 8) on the final source: 0 differing from step-by-step greedy.
- SPEC_DEPTH 2 / 3 / 4: 1948 / 1947 / 1949 us/token (no host-side effect).
- Spec path at the 4096 context: P=64 + 3992 new tokens and P=3000 + 1000: identical to greedy.

## s0015 (track A) BF_UNR 3 — 3169 us b1 (from 3206), bit-identical to s0016
- Verify kernel: tagged row polls (poll_rows / mlp_poll) 2 instead of 8 deep: K=2 4005 -> 3944, K=3 4188 -> 4105,
  K=8 5630 -> 5488 us (16 deep: much slower; the polls of all blocks for the same published rows contend). Step kernels
  keep 8 (no change measured there).

## s0016 BF_UNR 3 + verify row polls 2 deep — 1559 us b1 (natural 1031) (expected ~1550 / ~1020)
## s0017 merge polls 1 deep + w2a staging 2 deep (verify) — 1541 us b1: below vLLM+DSpark's 1544 for the first time
Lesson: in the dataflow kernel, many blocks polling the same freshly published data with deep batches of outstanding
loads slows the hand-off; 1-2 polls in flight per thread were better everywhere in the verify kernel (row publishes,
attention records, w2a inputs; not the o_proj inputs).
- More poll-depth knobs (verify kernel): router logits one at a time -3 us (ctx 600) / -12 us (ctx 1100), kept;
  mix sources one source at a time +10..25 us, q polls one at a time +5..15, per-row merger polls 1 deep +45 us at
  ctx 1100: off. Per-row attention past position 1023 makes K=8 passes ~360 us slower than the context growth alone.

## s0018 (track A) step kernels: row polls + w2a staging 2 deep — 3141 us b1, 6300 b8 (expected ~3125 / ~6300)
## s0019 track C with s0018's step kernel + sequential router-logit polls — 1540 us b1, b8 6310
## s0020 no L2 prefetch in verify passes — 1531 us b1 (expected ~1525)
Prefetch stopped paying once the polls were shallower: re-check earlier decisions after a change of the timing balance.
- Re-checks after the poll changes (verify kernel): VDES_MIN 4, HYB_MAX 3, SHARED_MIN 3, W2_1W_MIN 12, GATE_MFIRST 0,
  GRP_EXP false, FP4_UNR 4: all slower or equal; step-kernel prefetch off: no effect at B=8; theta 0.3 / 0.4 / 0.5:
  1918 / 1891 / 1919 (eval-like), 1092 / 1075 / 1154 (natural): 0.4 stays.
- Final build: soak 100 requests 0 differing, robust.py 10/10 identical, vcheck 2600 positions 0 differing.
- w2b expert-contribution polls in batches of 4 / 2 (verify): within noise; 8 kept.
## s0021 3-4 row passes poll published rows 1 deep — 1531 us b1 (noise-level; expected ~1527). LS_SEQ_S 1: +35 us (off).
- Final step path vs s15 s0016 at B=1, 2, 3, 5, 8 (600 positions): bit-identical, batch-invariant.
- w13 contiguous item ranges per block (verify; same-expert strips adjacent in memory): K=8 +85 us, others neutral: off. Access locality is not the w13 limiter.
- Gate F/G with one task per (controller, channel) looping the rows (verify): +800 us (too few parallel tasks): off.
- Gate F/G with 2 rows per task: the first version had a divergent early exit around a full-mask shuffle and hung the
  test job (card 1 blocked until its timeout); fixed (warp-uniform loop with a per-row guard), still off by default.
- Gate F/G 2 rows per task (fixed): +600 us: off. The per-(row, controller, channel) task layout is best.
## s0022 final committed source — 1528 us b1, b8 6302 (valid; noise vs s0021)
- Last knobs: o_proj input polls 1 deep (+10..115 us), w2b contributions 1 deep (+50..90 us): defaults kept.
- Final source (s0022): soak 60 requests 0 differing, robust.py 10/10, vcheck 1100 positions 0 differing.
- Last batch: DREV_MAX 3, MG_UNR 2, PR1_MAX 5: equal or slower; defaults are the final settings.
- W2_BPC8 2 (+40 us at K=2), HYB2 off (+145 us at K=2), VDES_MIN 2 (+15): defaults kept.
