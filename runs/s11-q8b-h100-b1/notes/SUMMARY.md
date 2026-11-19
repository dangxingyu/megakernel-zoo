# Run s11-q8b-h100-b1: summary (updated after s0022)

## Results (evaluator, http_step_us_b1 / B=8)
- Track B: s0001 (inherited) 4162 / 4457 -> **s0020 3813 / 4253**, s0022 3813 / **4231** (-8.4% / -5.1%),
  1.58x / 1.62x SGLang (6031 / 6839).
  (s0018 3825, s0019 3843: the last three are within HTTP noise of each other; in process s0019/s0020 are ~15 us
  faster than s0018.)
- Track A: s10 s0004 5365 / 5622 -> **s0016 5134 / 5462** (-4.3% / -2.8%), 1.17x / 1.25x SGLang.
- 18 submissions, 17 valid (s0015 invalid, see Incidents). Every speed change in both tracks keeps the logits
  bit-identical to the inherited engine of its track (bitcheck, 1152 positions, B=1 and B=8): the numeric gates are
  those of the inherited engine (track B numeric_long median 0.0074, max 0.498).

## What paid (in-process, same-card A/B, then confirmed by the evaluator)
1. L2 staging in idle HBM windows (both tracks; reads only, bit-identical):
   - attention window, idle blocks bulk-prefetch 9 x 16 KB of later gate+up tiles (B<=2): B -64 (s0003), A -43 (s0002).
   - each matvec phase end, 3-4 x 16 KB of the next phase's middle tile: B -79 (s0005), A -117 (s0004); with 3-8
     sequences B=8 -87 (s0006), A -102 (s0010).
   The microbenchmark behind it: L2-hot streaming alone is only ~3.4 TB/s on this card, but L2 hits ADD to an HBM
   stream (151 MB with 25 MB pre-staged: 55 -> 48 us), so idle HBM time can be banked into L2 for the next phase.
2. Online submit/poll contract: one device-loop launch per cohort, tokens tagged in host-mapped memory: -44 (s0008),
   A -67 (s0009). (subagent)
3. Certified int8 head sketch + exact rescoring on the greedy/online paths: -57 (s0011). (subagent)
4. Layer-0 q/k/v table built at load by the kernel's own phase: -16 / B=8 -20 (s0007).
5. Exception MMAs (weights outside the 12-bit window) read x from shared memory instead of L2 before each tile's
   block barrier: -41 (s0017; s0018 limits it to the greedy kernel, where it does not cause spills).
6. Re-tuning after staging: O2PF 0 -33 (s0014); track A: 1.5 staged tiles + SG 4: -28 / -24 (s0016); tail staging
   aimed at gate+up tile 2 (-17 in process, s0019); attention-window wait 8 / polite 7 (-11, s0020).
7. Sketch pass 2 rescoring spread over all blocks via a global candidate list (3+ sequences): B=8 -12 (s0021, subagent).
8. 3+ sequences: attention-window staging deferred until the attention blocks' score pass is done: B=8 -10, B=4 -30
   (s0022).

## Incidents
- s0015 (track A) invalid: NPF_ATT 0 exposed a latent bug (a stash-count reset inside that prefetch's #if); fixed,
  resubmitted as s0016. The constant had been timed but not bitchecked before submitting.
- s0017 made k_step/k_generate spill (step() +44); fixed in s0018. Rule since: bitcheck + spill report of all three
  kernels + both paths timed before any submission.

## Verification of the final source (after s0018)
bitcheck vs inherited, 1152 positions: track A B=1/8 bit-identical; track B B=1/2/8 bit-identical; batch invariance
1.0. Online tests (served == generate == step+argmax, 1024 / 8192 tokens, cohorts, validation), edge tests (token ids
outside the vocabulary, context and batch limits), sketch certification (sketch argmax == exact at 10,880 positions
incl. all-tiles mode; free runs), stress 8 min (2753 segments, 109,324 steps, bitwise repeatable): all pass.

## Limits (final source)
Every batch size 1..8 bitwise equal to B=1; contexts to 9216 positions vs float32: median 0.005-0.009, max 0.129
(HF bf16 0.012-0.021 / 0.408), top-1 1.000, bitwise batch invariance through the split path; online path with
3000- and 2000-token prompts identical to generate(); out-of-range token ids, context and batch limits raise.

## Measured dead ends
- Entropy coding of the exponent (census: 10.59 bits/weight entropy vs 12 stored): the decode needs ~30-35 more
  ALU ops per MMA, past the measured knee of the stream (+24 free, +32 -4%, +40 -16%).
- L2 staging of a whole phase (L2-hot rate ~3.4 TB/s); evict_last hint on staging (+25 us); staging the attention
  blocks' o tiles or gate+up tiles; staging first instead of middle tiles; more than ~9-12 pieces in the window.
- Split attention below 1136 positions (extra grid barrier: +70..110 us where active). After staging, a perfect
  attention speed-up is worth only 12-25 us at B=1 (what-if, attention capped at 16 / 256 positions); ~125-150 at B=8.
- TMA bulk copies or bulk L2 prefetch ahead of the stream: no higher HBM rate (2.8-3.05 TB/s).
- int4..int7 head sketches (bounds too loose: int7 already 43 candidate tiles at B=1); split attention in two chunks
  with a fixed-order merge (correct, B=8 +45: latency-bound body, lost o stash); staging down tiles in the window,
  staging before the o stash (+61), gate+up tiles moved to idle blocks, end-of-head prefetch retarget.

## Final source
engine/ is byte-identical to the s0022 submission (TRACK = B). Its track-A build keeps s0016's device behaviour
(track-specific defaults for the s0019/s0020 constants). Final suite on this source (e0225-e0227): bitcheck both
tracks identical, online / edge / sketch tests pass, stress 6 min PASS. Track A: put A in engine/TRACK (or MK_TRACK=A); that
build is the s0016 engine plus code that only the track-B greedy kernel uses (bitcheck vs the inherited track A:
identical). Experiment switches that measured negative are in git history (commit "experiment options ...").

## What I would try next
1. B=8 attention window (~16 us/layer, HBM half idle): every scheduling variant (split in two chunks, K/V staging,
   load pipelining, prefetch) measured neutral or worse; the body is latency-bound per warp and more attention
   blocks cost the o stash. A different mapping (e.g. score MMAs with K as the A operand for 2 heads per block, or
   the 68 idle blocks computing V^T p for chunks merged inside the o projection's input load) would be the next study.
2. B=1 per-layer excess (~4 us/layer): the o phase still waits ~4 us for staging spill-over at short context;
   a staging amount that adapts to the measured window length (e.g. from the attention blocks' stamps) rather than a
   constant might help; constant amounts were tuned out.
3. Step boundary on the greedy path (sketch pass 2, argmax, embedding, layer-0 attention start): ~10-20 us left.

## Method notes for the next run
- L2 hits add bandwidth on top of the HBM stream on H100: banking idle-HBM windows into L2 for the IMMEDIATELY next
  phase pays; staging further ahead, with evict_last hints, or competing with latency-critical loads does not.
- Bitcheck and the ptxas spill report of all three kernels (k_step, k_generate, k_gensk) before timing anything;
  time both the generate and the step() paths; confirm small gains with a second, longer A/B (build noise ~10-15 us).
