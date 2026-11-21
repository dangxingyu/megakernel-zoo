# Plan (rewritten at 3.7 agent-hours; track A s0009 4942 / 10527 us, track B s0008 4882 / 10813 us)

## Three largest losses (gap.md; ncu source-level stall samples e0153 / e0154)
1. Grid-barrier waits: ~32% of warp samples at B=1 (after attention 8.6%: only 32-136 blocks have attention units;
   after W2 6.4%; after O-proj 5.4%), ~18% at B=8.
   a) per-head dependency counters so attention starts as soon as its q/k/v strips are written (s6 measured -4.3%)
      - cost: counters + polling in every unit, ~2 h; expected gain ~2 us/application.
   b) give idle blocks work during attention: L2 prefetch of the O-proj weights by blocks without units (measured
      variants of prefetch hurt the latency-bound phases: low expectation).
2. NVFP4 expert GEMVs at ~4 TB/s (B=8: ~84 us/application): bound by per-warp dependency chains (decode -> hmul ->
   mma chain), NOT by bytes in flight (cp.async ring: same 4.1 TB/s, e0155) nor by the cvt rate (fake decode: same).
   a) two accumulators: +15% standalone, 0 in the engine; b) f16-pre-converted scales (track B storage change):
   removes ~3 of ~16 instructions per sub-tile; c) tcgen05 kind::f16 with smem decode (large effort; s7a got -1.9%).
3. Vector work scaling with B (prologue/epilogue chunk tasks: 4.5 rounds at B=8): ~27 us/application over B=1.
   a) process two chunk tasks per thread with loads issued together; b) distribute instead of replicate.

Decision: the remaining large items are multi-hour with low measured upside; the register-allocation lottery makes
small changes +-3% noise unless A/B'd. Next: robustness (B=2..7, long contexts, limits) because audits check it,
then 1a if time allows, then final notes. Track B record: sketch head (-0.8% B=1), layer-0 table (-0.1%, not
submitted), loop-gate linearization bounded at <= 5% before its own costs (not built).
