# Journal, run s18-mini-b200-c3 (track C, direction: acceptance length)

## 0.0-1.3 h: baseline and acceptance simulator
- s0001 = inherited s16 s0022 unchanged: 1527 us/token (synthetic b1), natural_b1 1017, b8 6297, natural_b8 6500;
  tokens/event 2.88 / 5.22 (b1 synthetic / natural). Reproduces s0022 (1528 / 1020) within 3 us.
- Simulator (tests/c3/collect.py, sim.py, sim2.py, sim16.py; analysis tests/c3/ana/*.py, run locally on the npz):
  the engine's own greedy text (step + generate) for 12 synthetic prompts (client prompts.make, seeds 0, 10000,
  20000, 30000, 1..8: the HTTP B=1 prompts are seeds 10000/20000/30000) and 12 natural prompts (natural_prompts.py
  64/48, indices 0..11: HTTP B=1 uses 1..3); the target features from the engine's verify passes (verify argmax ==
  greedy on all 24 sequences); the reference DSpark (harness, bf16) per anchor; acceptance walked cycle by cycle.
  Check against the engine: on the headline prompts the simulator gives 1505 us/token synthetic (engine 1527),
  4.91 tokens/cycle natural (engine 5.22), 3.15 synthetic (engine 2.88): close enough to rank policies.
- Verify cost V(K) (tests/c/tv.py, ctx 600 random tokens, card 1): 3469, 3896, 3967, 4190, 4669, 5052, 5280, 5444 us.
  In served natural text (nsys, tests/c3/cycprof.py) K=8 passes take 5.8-6.3 ms (context to 1088), K~3 3.8-4.0 ms;
  verify end -> next verify start 230-330 us (draft span).

## Findings on acceptance (all on the 24 sequences above; per-set walk, n-gram rule of the engine)
1. Static trees: the best static tree of <= 7 draft nodes (by frequency of the truth rank prefixes) IS the chain.
2. Dynamic trees (best-first by the draft's own path probability, real Markov steps, top-4 children):
   tokens/cycle at 8 rows: chain 2.76 -> tree 3.03 (synthetic), 4.38 -> 4.60 (natural); with n-gram copies
   2.87 -> 3.14 / 4.59 -> 4.79. But with the measured V(K) the cost-optimal policies are close:
   best chain rule (prod p >= 0.4, as the engine) 1686 / 1139 us/token; best tree rule (size by expected tokens per
   time, path probs ^2) 1644 / 1134: -2.5% synthetic, -0.4% natural.
3. Draft-length rules: calibrated expected-tokens-per-time rule 1680 / 1141 vs 0.4 threshold 1686 / 1139 (no gain);
   the oracle length would be 1538 / 1069: the draft's probabilities cannot tell which cycles will fail.
   Calibration: draft p is overconfident in the middle (p 0.75 -> 0.56 observed), right above 0.99.
4. The 8-row cap binds on natural text: 36% (12 prompts) / 44% (headline prompts) of cycles accept all 7 drafts.
   n-gram copies of 15 tokens in 16-row passes: natural 4.45 -> 5.32 tokens/cycle, synthetic 2.76 -> 3.01;
   us/token -12% / -7.5% if V(16) = 6.5 ms, -7.5% / -5% at 7.5 ms, break-even ~9 ms. s16's proto16 (16 rows as two
   8-row halves) measured V(16) = 10.1 ms.
5. DSpark with a 16-row block (trained with 8): depths 1-7 get worse (synthetic conditional 0.73..0.79 -> 0.73..0.74
   at depth 7), depths 8-15 conditional 0.67, 0.61, 0.57, 0.50 (natural): a 16-row pass would serve n-gram copies
   mostly.

## s0002 (draft-length rule p^2.5 / expected tokens per verify time)
Expected -0.2..-0.6% synthetic, -0.7% natural. Measured 1535 (+0.5%, s0001 1527) / natural 1013 (-0.4%); tokens/event
2.88 / 5.20. In-process A/B on the same prompts showed 1579 vs 1577 for the three synthetic headline prompts: the
synthetic +8 us is evaluation noise; the rule is neutral. Kept (natural slightly better).

## 16-row verify passes (s0003)
- A second library engine/mk16.cu (s16 proto16 at s0013 + gating + one-pass MoE) runs only 9..16-row passes; both
  kernels are queued every cycle, Hdr.flags 0x100 / 0x200 make the one whose row count (vpos[1], on the device) does
  not match return at once (~5 us). Shared with the main library: weights, K/V cache, history rings (RING 32 in both;
  step() bit-identical to s0001: bitcheck 1152 positions, B=1 and 8).
- vcheck16 (tests/c3/vcheck16.py): 16-row and 8-row passes mixed vs step(): 0 of 1020 positions differ, every build.
- V(16) at ctx 600 (tests/c3/tv16.py): proto16 as is 10.6 ms; + big smem (213 KB) 11.26; + one-pass MoE with a
  serial union build 13.7 (!); parallel union 10.02; router input aliased onto x16 rows 12..15 (smem 176 KB) and
  per-expert row masks for the union: 8.76 ms; V(12) 7.67. One-pass qkv (two column groups per weight fragment, no
  smem growth): no gain (8786 vs 8772), off (QKV1 0).
- Draft side (draftk.cu VR = 16): acceptance over 16 rows, emit 16, context K/V of 16 rows (second GEMV job set that
  returns unless st[7]), n-gram copies of 15 tokens when the anchor c <= 1008 (every row's window starts at 0).
- Served A/B (in-process, 12 prompts): synthetic 1773 -> 1729 (-2.5%), natural 1278 -> 1246 (-2.5%); HTTP headline
  prompts synthetic -3.5%, natural -6.3% (mean of the three); tokens identical to step-by-step greedy on 10 prompts.
- s0003: 1454 us (-5.3%, predicted ~1480) / natural 990 (-2.3%, predicted ~950); tokens/event 3.13 / 6.76.
  The HTTP natural_b1 is the MEDIAN of three per-prompt rates (step_us_min 879 = nat1, median 990 = nat3, max 1319 =
  nat2): my prediction used the mean of the three; nat3 alone moved 1021 -> 996.

## s0004-s0006
- s0004 (16-row final unembedding in one pass: V(16) 8.77 -> 8.61 ms): 1447 / 982, exactly as predicted (1445 / 982).
- s0005 (16-row n-gram passes only after a cycle that accepted >= 7 drafts): 1430 / 959 (predicted 1443 / 958);
  tokens/event 3.11 / 6.64. Simulated rules by match length were within +-1.4% and inconsistent across prompt sets;
  the history rule was consistent (12 natural prompts -1.9%, headline prompts =).
- s0006 (FP8 copies of the draft layers; DK_KDMIN 0 = a 1-row pass when no draft pays): 1416 / 960 (predicted
  1420 / 957); tokens/event 3.07 / 6.61.

## Measured and dropped (16-row library, all bit-identical, ctx 600, K=16)
- one-pass qkv (both column groups per weight fragment, xs16[2 streams][16 rows], no smem growth): 8786 vs 8772.
- one router poll + one 16-column router GEMV (ROUT1): 8635 vs 8603 (the per-half polls pipeline behind the GEMV).
- two-row attention merge items (MRG2: 80 record + 64 merge items, one round of the grid): 8852 vs 8612.
- poll depths PR16_UNR 2/4, MG16_UNR 1/2, W2_BPC16 1/3/6, BF_UNR 3, no L2 prefetch: all within +-30 us; kept
  W2S_UNR_V 2 (-28 us).
## Measured and dropped (draft)
- double-buffered FP8 GEMV loads (FP8-only instance, 120 registers, no spills): 357 vs 356 us per draft.
- looping blocks for the base unembedding GEMV (X staged once): 357 vs 366 (noise).
- strips per block (SB_MID 1/2, SB_LO 700): all slower (372..461 vs 357).
- the draft graph with every kernel returning after its PDL handshake: 42 us. A persistent fused draft kernel
  could save at most that minus its own barriers: not built.
- FP4 base logits (simulated): -0.7% accepted on synthetic text for ~-45 us: no.
## Measured and dropped (acceptance)
- DSpark chain + n-gram extension to 16 rows when all 7 drafts are confident (simulated): +0.3..0.6% (worse).
- 16-row passes past anchor 1008 (needs the per-row attention path for 16 rows; simulated): <= 0.8%.
- 32-row passes for copy regimes (simulated copy lengths: 15-cap 12.2 / 9.6 tokens, 31-cap 23.2 / 16.9; at
  V(32) ~ 1.7 V(16)): ~ -1.5% overall: no.
- Jacobi-style reuse (the target's argmax at the row after a rejected draft predicts the corrected continuation):
  1.4-7.5% hit rate (tests/c3/jacobi.py): useless.
- two records per attention record item in the 16-row pass (q / r of the 48 queries once per item, REC2): 8980 vs
  8592 us (the merges wait longer for records computed in turn); bit-identical; off.
- per-request online calibration of the draft-length rule (scale p^2.5 by the observed / predicted acceptance of
  recent cycles; simulated, decays 0.8-0.97): within +-0.3%.
- copying-regime knobs served A/B (12-14 prompts): MK_NG16_ACC 5/6/7, MK_NGRAM_MIN 4/5/6: all within +-0.5%.
- soak (tests/c3/soak16.py, 50 requests: synthetic / natural / random / repeated prompts, n up to 1100, B=8 cohorts
  in between): 0 differing from step-by-step greedy.

## s0007 (final committed source = s0006 + off-by-default switches)
1419 / 962 (s0006 1416 / 960), tokens/event 3.07 / 6.61: reproducible within 3 us.

## s0008 (1-row cycles on the B=1 step kernel; length rule with V(1) = 3100) — reverted
Expected 1410 / 957; measured 1426 / 963, b8 6297 (+18: the step-kernel code change), tokens/event 3.03 / 6.57.
The in-process A/B used the 12-prompt MEAN (synthetic -0.8%), but the HTTP headline is the median prompt (syn10000),
which had moved 1430 -> 1439 in that same A/B: more 1-row cycles lose tokens on that prompt. The step kernel does a
1-row cycle in 3202 us vs 3426 on the verify kernel, draft inputs identical, step() bit-identical (bitcheck).
Reverted to s0007's engine. Lesson: judge policy changes on the three HTTP prompts' median, not the mean.

## After s0008 (all measured, all kept off or unchanged)
- draft-length constants on the HTTP-median prompts and 12-prompt means (tests/c3/sweep_dk.sh): DK_GAM 2 / 3,
  DK_D 300 / 500, DK_KDMIN 1: all within +-0.5% of the defaults (2.5, 400, 0).
- draft attention splits NSPLIT 16 / 24 / 48 / 64: 32 (default) = 24 = 16 (360-367 us), 48 / 64 slower.
- trees on the two HTTP-median prompts (simulated): synthetic seed 10000 -0.1%, natural 3 -0.5%.
- 16-row pass: row-batched F/G gates (mk.cu FG_ROWS ported as FG16, 4 / 8 / 16 rows per task): bit-identical,
  8996 / 9378 / 10236 vs 8590 us: fewer serial tasks lose more than the saved weight reads. Off.
- main-kernel refinements after s0013 missing in mk16 (LS_SEQ, QSEQ, FG_ROWS off by default; poll depths): already
  covered by the PR16 / MG16 / W2S tests.
- concurrency (tests/c3/concur.py: mixed prompt lengths at once, arrivals during a request, a B=8 cohort behind a B=1
  request): all streams identical.
- 16-row pass: router halves on blocks 0-7 / 8-15 in parallel (RTR2): 8574 -> 8543 us, bit-identical: kept.
- 16-row pass: two of a row's four gate-D strips per job after one mix (GD2, 96 jobs instead of 192): bit-identical,
  8620 vs 8548 us, and V(12) +240 us although it does not activate there (code generation): off.
- soak: 200 requests (180 speculative, 84750 tokens, B=8 cohorts in between): 0 differing.
- s0009 (router halves on separate blocks, final source then): 1418 / 961-ish (see board); predicted 1419 +- 10.
- 16-row pass: two queries per warp task in the shared-record attention prologue (Q2; lane l carries the 4-element
  sums of the original lanes l and l + 16, whose level-16 shuffle add is exactly s_l + s_(l+16)): bit-identical,
  8520 vs 8545 us over 4 alternating rounds: kept.
- 16-row w2a: three warps per item for blocks with > 24 / 36 items (W2_1W_MAX16): no gain against its own build,
  and merely adding the (no-op by default) condition made the whole 16-row pass +145 us (code generation): reverted.
  Several earlier "slower" verdicts (QKV1, MRG2, GD2, REC2) may be partly this effect; the s17 run's advice (check
  the compiled hot loops, keep replaced paths compiled) applies. All verify / step instances are at the 80-register
  cap of 768 threads with 500-900 B of spills; NTH is fixed by bit-identity (the warp split of every reduction).

## The s17 track-A engine against this step() (tools/kit/bitcheck.py, 1152 positions)
- s17 s0014 at B=1 (its a2 tagged-dataflow kernel, ~2.74 ms per step): BIT-IDENTICAL to this engine's step().
- s17 at B=8 (its a1 kernel, 5.09 ms per step): not bit-identical (rel. L2 median 0.023 vs ours; not batch-invariant
  against its own B=1 either), so it cannot serve track-C B=8 cohorts (the http_tokens gate is strict).
- Consequence for a merge: verify passes built on a2's design can keep step()'s function bit for bit; a2 keeps its
  own K/V and ring layout, so mixing it with these verify kernels inside one sequence would need a shared layout.

## s0011 (short n-gram copies when DSpark's first draft is unconfident)
Idea from tests/c3/ana/a25.py: on cycles without a long match, a 4-token earlier occurrence + DSpark p1 < 0.7 ->
verify the 3 copied tokens (K=4). Served A/B (3 rounds): HTTP-median synthetic 1430 -> 1419, natural 966 -> 960.
s0011: 1408.6 / 954.1 (predicted ~1408 / ~954); tokens/event 3.09 / 6.62. New best.
Neighbours (2 rounds): match >= 3 tokens 1416 (vs 1420), P 0.8 / 0.9, K 3 / 5: noise.
- s0012 (short copies from 3-token matches): 1404.4 / 954.6 (predicted ~1404 / ~954); tokens/event 3.10 / 6.65.
- agreement-aware copy rules (copy agrees with DSpark's d1; pod walk tests/c3/agree_walk*.py): worse (they replace
  confident DSpark chains); L >= 2 or length-dependent thresholds: within +-0.5%.
- plain n-gram copies from 7-token matches instead of 5 (MK_NGRAM_MIN; 5-6-token matches now go to DSpark or the
  short-copy rule): served 3 rounds, HTTP-median natural 960 -> 950, synthetic 1416 -> 1417; 8: worse.

## 16-row passes past position 1008 (s0014)
- mk16 per-row attention path takes 16 rows of one sequence (in-pass k/v arrays MAXR, k/v warp tasks loop over warps
  6..23): vcheck16 to position 1700 with every pass 9..16 rows: 0 differing; mixed to 1300: 0; V(16) unchanged
  (8466 vs 8467 at ctx 600, 8751 vs 8753 at 990). V(16) at ctx 1040 9669 (V(8) 6321).
- LIM16 1008 -> unlimited, plus st[26] = the request's last anchor for a 16-row pass (P + n - 16): served per prompt:
  synthetic 10000 / 30000 / 2 / 3 -11..-13 us, natural 1 / 2 / 6 -12..-19, natural 3 +6, natural 4 +15; means
  synthetic 1678 -> 1671, natural 1195 -> 1191; HTTP-median synthetic 1396 -> 1384, natural 941 -> 947.
- robust16 at max_context 4096: 18 cases identical; served identity 6/6.
- s0014 (16-row passes at every position): 1383.6 / 946.8 (predicted ~1385 / ~946); natural min 860, max 1258.
- copying-regime threshold past position 1008 (st[27], MK_NG16_ACC_HI): 15 (16-row there only right after a fully
  accepted 16-row pass): HTTP-median synthetic 1383-1384 (kept), natural 942-943 (vs 947); 11: 943.
