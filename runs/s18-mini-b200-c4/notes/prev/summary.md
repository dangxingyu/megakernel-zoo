# Summary of the merge run s17-mini-b200-ma (track A throughout)

## Best submission
**s0014: 2728 us/token at B=1, 5093 us/token at B=8 over HTTP** (valid, all gates; engine/ is byte-identical to it).
- B=1: 1.36x vLLM's image (3707), 1.59x SGLang (4329). The 1.10x claim holds at B=1.
- B=8: 1.002x vLLM's image (5105), 0.997x the vLLM source build (5077), 1.17x SGLang. The 1.10x target at B=8
  (4641) is not reached; the parity milestone is not passed (it counts the 5077 us vLLM source build).
- Repeatability: identical sources re-evaluated to within 7 us (B=1) / 2 us (B=8) (s0005, s0008).
- Board's B=1 "best" is s0006 (2720): the same B=1 kernel as s0014; the 8 us are evaluator noise.

## How the engine is built (and where each idea came from)
| part | source |
|---|---|
| B=1: engine/a2 tagged-dataflow kernel | s15-mini-b200-a2 s0025 |
| B=1 compiled with B constant (KB1) + inactive mark sites (s0003, -20 us) | idea of s15-mini-b200-b2 s0020/s0021 ("K1") |
| B>=2: engine/a1 persistent kernel, ring stages 9 x 2 | s15-mini-b200-a1 s0024 / s0025 |
| wrapper choosing the kernel per cohort (s0001) | this run |
| GZ0 + MSGW: gate stages without mixer hand-off, warp-per-message M (s0004, -116) | this run (MSGW = the a1 form of b2's MSGZ idea) |
| SCIM 4 -> 8, FP8D, FMA2: decode work moved to the FMA pipe / fewer instructions (s0006, s0007, s0009, s0014: -262 at B=8) | this run |
| PMAXC 2 + SCNT_REL 1 (s0010, -22) | a1's own switches, re-measured |

## What worked
1. Merging by batch size: a2's kernel is the best B=1 design, a1's the best B=8 design; one engine runs both.
2. Measuring code generation, not only time: a1's B=8 kernel's decode loops compile to different lengths when
   unrelated code changes (540 vs 583 instructions per 40 HMMA). Keeping a replaced path compiled behind a
   never-true runtime condition restores the allocation; tests/sasssearch.sh / sassall.sh check a variant in 20 s.
   This made GZ0/MSGW pay (they had measured +200..+500 us before the fix).
3. ncu per-pipe counts: the B=8 decode loop was ALU-pipe bound. Bit-identical moves of work to the FMA pipe and
   fewer instructions: IMAD instead of LEA for the scales (an opaque multiplier: ptxas turns known powers of two
   back into LEA), E4M3 scale pairs by one cvt + HADD2.F32, MXFP8 bytes -> bf16 bits directly (exhaustive GPU
   check), packed fma.f32x2. NVFP4 loop 540 -> 427, MXFP8 476 -> 417 instructions; B=8 5355 -> 5093 us.
4. Checks: bitcheck (s0007 = s0004 at B=1/3/8), exhaustive FP8D check, limits / cohort tests, reference check.

## What did not work (all measured; journal.md has the numbers)
- B=8: deeper / smaller rings (+200..+300), early ring refill (+324, spills), out-of-line decode loop (+470),
  multiply-high byte extraction (+55..+420), SCALE_BF16, two-level grid barrier (+300), release barrier arrival,
  one-round norms (+28..+112), gated norms on mixer blocks (+55..+800), warp-per-sequence mixes (+880), splitting
  leftover w13 tasks (+35..+110), early w2 issue, D-row prefetch, one-round attention merge, GZ0 on one stage
  only, MIXT 0 / RKQ / RI / BAR_SLEEP / PUB_REL (neutral or worse).
- B=1: KB1 alone (+375, a code-generation cliff), batched re-polling (+27), depth knobs, build flags (noise).
- A mistake to avoid: a switch inserted by a scripted edit at a vanished anchor was never defined, so s0011-s0013
  silently ran the old kernel. Check the compiled loop (or a #error guard) after every switch added by script.

## What I would try next
1. B=8 kernel organisation: 35-45% of the non-MoE warp time is block barriers behind grid-barrier / flag waits.
   A dataflow (tagged hand-off, a2-style) B=8 kernel, or a2's kernel at B=8 with a1's attention and w2 design,
   removes barriers rather than tuning them.
2. B=8 expert decode is now issue-bound at ~43 instructions per NVFP4 item with 2 warps per scheduler: more warps
   per SM for the expert phases (a kernel shape with fewer registers there), or the tensor-core path from shared
   memory (tcgen05) with decoded weights staged in smem.
3. B=1: the step is ~10 hand-offs per application (barrier 36%, poll/memory waits 32% of stalls in ncu); fewer
   hand-offs per layer (fused gate stages, cheaper attention merge) rather than per-phase speed.
4. Always: SASS check of the hot loops before timing, one engine per process, and direct evaluator tests when
   card-1 differences are below ~20 us.
