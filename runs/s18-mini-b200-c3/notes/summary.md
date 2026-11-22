# Run s18-mini-b200-c3 (track C, Inkling-2-Mini, 1x B200; direction: acceptance length) — summary

## Result
Best: **s0006** (re-evaluated as s0007 from the final committed source: 1419 / 962) = engine/ at commit "draft: FP8 copies of the draft layers ..." (later commits change notes, tests
and off-by-default switches only):
- http_step_us_b1 **1416** us/token (start: s0001 = inherited s0022, 1527; vLLM + DSpark 1544)
- natural_b1 **960** (1017; vLLM + DSpark 871), b8 6279 / natural_b8 6491 (no speculation at B >= 2, unchanged)
- tokens per event (= tokens per verification step): synthetic 2.88 -> **3.07**, natural 5.22 -> **6.61**
  (vLLM + DSpark 3.48 / 5.21 on its own text)
- every gate passes in every submission; generate / http_tokens strict (track C): 8/8 identical.

| sub | change | b1 | natural b1 | tokens/event b1 / nat |
|---|---|---|---|---|
| s0001 | inherited s0022 unchanged | 1527 | 1017 | 2.88 / 5.22 |
| s0002 | draft length by expected tokens per verify time (p^2.5) | 1535 | 1013 | 2.88 / 5.20 |
| s0003 | 16-row verify passes for 15-token n-gram copies (second library) | 1454 | 990 | 3.13 / 6.76 |
| s0004 | 16-row final unembedding in one pass | 1447 | 982 | 3.13 / 6.76 |
| s0005 | 16-row passes only in a copying regime | 1430 | 959 | 3.11 / 6.64 |
| s0006 | FP8 draft-layer copies, 1-row passes allowed | 1416 | 960 | 3.07 / 6.61 |

## What I found (details: journal.md, gap.md, trackc.md)
1. With the inherited verifier (V(K) 3.47 .. 5.44 ms for K = 1..8), the 8-row cap was the binding limit of
   acceptance: 36-44% of natural cycles accepted all 7 drafts. Trees or multi-candidate drafting within 8 rows are
   worth 1-2.5% at most (simulated on the engine's own greedy text with the reference draft head; the best static
   tree of <= 7 nodes is the chain), because each extra row costs verify time.
2. A 16-row verify pass for n-gram copies (bit-identical to step(), built on s16's proto16 with a one-pass 16-row MoE
   and one-pass final unembedding: 10.6 -> 8.57 ms) gave the run's main gain: -5.3% synthetic, -2.3% natural in
   one step; a copying-regime rule (16 rows only after a cycle that accepted >= 7) another -2.3% natural.
3. The HTTP natural_b1 figure is the median of three per-prompt rates (natural prompts 1, 2, 3 -> 883 / 1293 / 960):
   it moves with natural prompt 3 alone.
4. Measured dead ends: one-pass qkv / router / two-row merges in the 16-row pass, poll depths, double-buffered or
   looping draft GEMVs, a fused persistent draft (the empty 46-kernel graph is 42 us), FP4 or vocabulary-restricted
   draft logits, Jacobi reuse of rejected rows (1.4-7.5% hits), 16-row DSpark blocks, DSpark + n-gram extensions,
   32-row passes and 16-row passes past position 1008 (<= 1.5%).

## What would move the numbers next
1. The verifier floor: V(1) = 3.47 ms is 80% of a synthetic cycle (step() is 3.1 ms; the s17 track-A step 2.74 ms):
   another run's direction. A cheaper V(16) - V(8) raises the value of 16-row passes (simulated -12% natural at
   V(16) = 6.5 ms vs -6% now).
2. B = 8 speculation (tokens per event 1.0): needs multi-sequence verify passes of > 8 rows (rows = (sequence,
   position)); with n-gram drafts only, est. -5..-9% natural_b8 and ~0 synthetic b8 for a large kernel change.
3. Trees: only once V(K) is much flatter.
