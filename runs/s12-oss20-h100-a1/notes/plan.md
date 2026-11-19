# Plan (rewritten 2026-11-18, ~15.0 agent-h, after three submissions under 0.5%)

State: track B s0042 1634.0 / 3405 (best B=1), s0043 1634.0 / 3396, s0044 1634.3 / 3392 (current engine: 6-bit sketch
for B>1, 5-bit group-scaled sketch + DPK + SKPF0 in the B=1 build). 2.05x / 1.46x vLLM. Track A s0036 1876.5 / 3394
(s0041 1877.6). 44 submissions, all valid. Last full verification: e0452 (+ e0441 for SK5).

Since the last plan (14.1 h): SK5 (5-bit codes with u8 group scales, certified; 381 MB instead of 434) with DPK (hi/lo
digits in one integer mma at B=1): B=1 -5; B=8 kept on the 6-bit sketch (SK5's heavier candidate tail cost ~20 us at
B=8 on the evaluator's workloads). Closed: I64 (exact int64 sketch accumulation, neutral), SKPFF (earlier sketch
prefetch, +2..+4), static per-block rebalancing (slow blocks change between runs: e0453), RSPL (router partials over
three warps: +4.5 / +38).

Measured facts that bound what is left:
- Pass 1 of the SK5 head is ~10 us over its load-only time (bench sk5bw 146 us); ALU count is not the reason (I64).
- Served-like prompts (word salad + "continue", tests/skstat2.py): ~50 candidate tiles per step, tail steps cost
  ~1-1.5 us/step on average.
- Every change that adds code to the kernel risks a +5..+40 us code-generation penalty; only measured A/B decides.

Next (low expected value; one idea per submission; same-session A/B, 3 rounds, gen B=1 as the reference because serve
timings on some pods are erratic):
1. Small head items in the B=1 build: speculative clist[bid] load with the candidate count (one L2 round trip), P2CH=3
   in the B=1 build re-test.
2. B=8 for the record: per-warp parallel candidate recompute in the general build (PB8-style, with TMA).
3. Keep the notes current; final verification of the committed state near the end (bcheck, alone-vs-batch, soak,
   limits, roundtrip, sk5_check) and a last track-A buildability check.
Rewrite at ~19 agent-h or after 3 more submissions under 0.5%.
