# Findings of this run (s17-mini-b200-mb)

## Card / harness
- This pod's card 0 runs the a2 kernel's B=1 ~150 us slower than the earlier runs' card 0 (a2 s0025: 2900 here vs
  2755 in a2's run; b2 s0047 2949 vs 2879); B=8 the same. Card 1 reproduces the earlier numbers (2744 us). Compare
  submissions within this run only.
- engine/api.py parsed TRACK at import: an A/B harness that builds two engines in one process measured both with
  the first TRACK. Fixed (TRACK re-read in Engine.__init__). This had made a "K1" A/B look like -8 us; properly it
  is +384 us at B=1 (the batch-1-only compile of the a2 kernel falls off a code-generation cliff: fewer spills,
  slower; run s17-mini-b200-ma found the same, +375, and -20 with the inactive mkprof mark sites compiled in).
- The a2 kernel + SKH does not launch when built with -DMKPROF (cooperative launch fails): profile builds of the
  SKH configuration need a smaller footprint; production builds are unaffected.

## Attention at B >= 3 (a2 kernel, ATT3, bit-identical)
- B=8 attention per application: 10.8 us at ctx 200, 31.8 at ctx 1000 (128-key block items in 3.4 sequential
  waves; each item a ~6 us chain of block syncs, load wave 3.3 us). K/V bytes at ctx 1000: 32.8 MB = 5.5 us at
  6 TB/s.
- ATT3 (three independent 8-warp teams per block, one 64-key record each, unpadded XOR-swizzled tiles in the same
  shared memory, next record's K/V tile loaded during softmax / PV, block merge): 31.8 -> 25.6 us at ctx 1000,
  -298 us per step at ctx ~960, -81 at 64-320; HTTP B=8 5860 -> 5701 (s0004). Per record 6.6 us in 5 steps of
  ~1.2 us; 3 rounds at ctx 1000. Inlined it is free elsewhere; out of line the call's stack frame (72 -> 256 B)
  cost every other phase 1-2 us (net -22 us).
- Superseded by the a1 kernel at B>=2 (5919 vs 6355 at ctx ~960 on the same card).

## w2 at B=8 (a2 kernel)
- w2a moves ~430 KB per SM at ~15 GB/s per SM (w13: 37 GB/s): rounds of 8 items x 10 KB, 3.2 us each; 2 warps x 9
  units per item instead of 3 x 6 (more bytes in flight): B=8 -14 (noise), B=1 +36. Not the limit.

## Track-B items on this base (measured)
- SKH (certified int8 head sketch + exact rescoring, b2's idea, ported with a fast-math-safe certificate): -49 us at
  B=1 on card 1, -21 over HTTP (s0006); greedy tokens identical; ~1-1.6 exact strips per token.
- GLIN (linearised attention gate): +86..+110 us at B=1 on the a2 s0025 base (D leaves the critical path, F/G and
  the gate norm get slower); within the gates, not bit-identical. Not taken.
- MSGZ / L0T: -50 / -41 us at B=8 on the a2 kernel; the a1 kernel already computes the messages per pass, and L0T on
  a1 is +28..+33 us (a1 prefetches application 0's qkv weights while it picks the token).
- Dual engine: B=1 (a2) and B=8 (a1) differ by rel. L2 ~0.05 on random tokens (different arithmetic); the
  evaluator's alone-vs-batch gate passes (tolerance), bitwise equality across batch sizes is lost.

## Code-generation modes of the batch-1 (a2) kernel
The same source compiled with the batch a constant 1 (KB1) runs either ~2660-2720 us or ~3130-3165 us at B=1 on
card 1, nothing in between: KB1 alone 3136, KB1 + the inactive mkprof mark sites 2676 (s0013 -75 us over HTTP), and
on top of that FP4_UNR 8, BF_UNR 3 or FP8_UNR 6 each fall back to ~3140-3165 (e0112). ptxas reports no spills in
either mode, so it is not spilling. The final engine sits in the fast mode. Any change to this kernel must be timed
against it, because a small source change can flip the mode.
SASS size does not explain it (e0114): mk_step 24,976 instructions in the fast build vs 25,056-25,088 in slow ones
(FP4_UNR 8, BF_UNR 3); SK_UNR 8 (fast-ish) 24,936. It is a placement / scheduling effect of the code, not its size.
Unresolved; a later run could diff the hot loops' SASS of the two modes (cuobjdump -sass) and their i-cache sets.
