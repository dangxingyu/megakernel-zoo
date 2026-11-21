# Plan v5 (final; see notes/summary.md). Best track A: s0025 = 2755 us B=1 (1.340x vLLM); engine/ = s0026 (2761 us B=1, 5814 us B=8)

State: one persistent kernel (152 x 768, 80 regs, stack ~24-32 B), tagged dataflow, bitwise batch-invariant at every
B, gpu_never_waits. Since v4: s0022 (current K/V conv: all loads before the first wait), s0023 (norm weights, q/r
scales, router bias before the polls), s0024 (expert scale factors / o_proj conv history before their polls).

The most productive lesson of the run (s0022-s0024, -170 us B=1, -250 us B=8): wait_tag / pollf are volatile asm
with a memory clobber, so every ordinary load written after a poll is issued after it and costs a full L2 round trip
(~1 us while HBM streams) on the critical path. Audit done for every poll site; remaining loads after polls are
weight streams, where any prefetch (L2 or L1) measured worse.

Remaining structure (B=1, ~47 us per plain application): ~10 hand-offs (~1.5 us each), 2 block reductions per mix,
the attention merge (~5 us), streaming phases at their per-SM in-flight limit. B=8: attention waves (~31 us/app),
w13 at bandwidth, w2a stragglers (latency noise), designated-block mixes (2 hand-offs each).
Remaining time: no restructures; small bit-identical changes only, each A/B'd 3 times; notes kept current.
