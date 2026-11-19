# Summary

## Run s10-q8b-h100-f1: final summary (written at the end of the 3 agent-hours)

**Best submissions** (4 submitted, 4 valid; headline `http_step_us_b1`, us per token over HTTP at one sequence):
- Board best: **s0002, track B, 4136.1** (4461.4 per step at 8 sequences) = 1.458x / 1.533x SGLang (6031 / 6839).
- Track A (the like-for-like number): **s0004, 5364.9** (5621.9 at 8 sequences) = 1.124x / 1.216x SGLang.
  The inherited track A failed today's `numeric_long`; s0004 passes it with median 0.0157 (limit 0.0320) and
  largest error 0.640 (limit 1.068).
- `engine/` is byte-identical to the s0004 submission (mk.cu, api.py, build.sh, TRACK = A). To build the track-B
  result from it: put `B` in `engine/TRACK` (or `MK_TRACK=B`); that build's device code is identical to s0002's
  (SASS compared, 46018 lines), so it reproduces s0002.

**What worked**
1. Measuring before changing: HF BF16 with `attn_implementation="eager"` against its default attention showed the
   engine's whole excess (1.21x) without touching the engine; a per-layer replay showed the inherited engine
   equal to HF eager and the fixed engine equal to HF default at every layer.
2. The fix: attention scores stay float32 (one macro, `SCORE_RB 0`, at the two score sites). No speed cost.
3. Jitter builds (score scale moved by 1-3 float32 ulp) as independent draws of the rounding noise: they separate
   what is a property of the data flow (the median) from what is a draw (the largest error).
4. Keeping every later change bitwise identical to s0001 (`tests/cmp_exact.py`): the numeric gates of s0003 and
   s0004 equal s0001's to the last digit.
5. One scheduling gain on track A: idle blocks pull K of all attention units into L2 before any V (1-2
   sequences): -9 us per token at one sequence (0.17%).

**What did not work** (all measured, numbers in notes/journal.md)
- More V loads in flight below 1136 positions: no effect (V is already in L2).
- Attention scheduling knobs on the 64..1088 workload (ATT_PF, ATT_WAIT 0 / 24 / 32 / 64, TOUCH 0,
  ATT_WAIT_MAX_B 8, ATT_POLITE 9): equal or worse than the inherited settings.
- The K-first touch order on track B: slightly slower there; with 8 sequences: inside the spread.
- The inherited TRACE build did not compile (one missing argument, fixed); its "layer total" line is still wrong.

**What I would try next**
1. Attention at one sequence with more SMs per unit below 1136 positions, without a grid barrier per layer and
   with ONE summation order for all batch sizes (about 100 us per token at mean context 576 is at stake; one SM
   per unit reads L2 at 165 GB/s). It changes the arithmetic order, so re-measure the tail with
   `tests/jit_long.sh` and `tests/set2.sh` before submitting: the largest error is re-drawn.
2. A resident kernel that publishes each token to the host while it goes on (online contract): the per-token
   launch + host read is 22-23 us and the torch argmax 15-18 us at one sequence; a device argmax alone gains 0.3%.
3. A context-dependent ATT_WAIT (about 6 us at one sequence) - only if a workload-independent rule is found.
4. Track A at one sequence: the block-wide normalising copy at each norm phase (~0.5 us x 73), still open from
   the earlier run.
5. For the evaluator's owner: the "largest error <= 2x HF's largest" limit is passed by an HF-equivalent engine
   with a real but not wide margin (14 draws over two sets: 0.46-1.85x, 4 above 1.5x, none above 2x).

## Run s10-q8b-h100-f1: details
Result: both tracks valid on today's evaluator, speed as inherited.
- **Track A: s0004** (final source; s0001 = the fix, s0003 = same device code as s0001): 5365 us per token over
  HTTP at B=1, 5622 per step at B=8 (s0001: 5376 / 5634; in process 5231 / 5342) = 1.124x / 1.216x SGLang.
  All four submissions of the run are valid. numeric_long median 0.0157 (HF BF16 0.0160, limit 0.0320),
  p99 0.259 (HF 0.224), largest 0.640 (HF 0.534, limit 1.068), top-1 99.80%. The inherited track A (s0066 of the
  earlier run) failed this test with 0.0192 / 0.328 / 1.183.
- **Track B: s0002**: 4136 / 4461 us over HTTP (in process 4040 / 4179); numeric_long 0.0074 / 0.153 / 0.498.
  Bit-identical to the inherited track B (1300 positions, B=1 and B=8, 1.78e9 logits). Differs from track A by the
  12-bit weight code, float32 intermediates and the fused RMSNorm; not bit-identical to track A.
- **Cause of the failure**: the inherited track A rounded every attention score to BF16 twice,
  bf16(bf16(q.k) * scale) - the "eager" formulation of the model code. The fused attention kernels (torch SDPA,
  which Hugging Face runs by default; flash-attention; FlashInfer) keep the scores in float32. Evidence, all on
  my own 8 x 1153-token set against HF float32 (tests/ref_long.py, long_err.py, score_round.py, tail_dir.py):
  (1) HF BF16 eager is 1.21x HF BF16 default (0.0144 / 0.0119), the engine's excess exactly; (2) the inherited
  engine equals HF eager layer by layer to three digits, the engine with float32 scores equals HF default layer by
  layer (0.99-1.02x), one rounding is halfway; (3) inside one attention operator the score rounding costs 0.008
  (once) / 0.011 (twice) of the output against 0.001 for BF16 probabilities and 0.0017 for the output rounding;
  (4) it is flat in position - not a long-position effect; (5) the over-limit positions were deterministic
  products of the eager rounding (HF eager and the inherited engine agree to 1% at such a position).
- **The fix**: one macro (SCORE_RB 0) at the two score sites. No cost in speed.
- **The margin**: the median is a property of the data flow (0.98-1.03x HF in seven independent draws of the
  rounding noise). The largest error is a draw from the same heavy tail HF BF16 draws from: at a few unstable
  positions every implementation (HF, both tracks) has a large error along one direction with a random amplitude
  (on my set 0.31-2.05 over twelve implementations at one position, HF 1.11). The engine is deterministic, so
  0.640 is reproduced exactly; any change of arithmetic order re-draws it. Keep speed changes bitwise identical
  (tests/cmp_exact.py) or re-measure with tests/jit_long.sh. Over two independent sets and 14 draws the engine's
  largest error was 0.46-1.85x HF's largest (4 of 14 above 1.5x, none above 2x): the 2x limit on the largest
  error is passed by an HF-equivalent engine with a real but not wide margin; the evaluator's 1.20x is typical.
- **Speed**: -9 us per token at B=1 (0.17%, s0004): the idle blocks pull K of all attention units into L2 before
  any V (track A, 1-2 sequences; reads only, logits bitwise identical to s0001). Measured and rejected: more V
  loads in flight at mid context, six attention scheduling knobs on the 64..1088 workload (details in
  notes/journal.md). The HTTP headline is the in-process step + 100 us of attention growth to 1088 positions +
  46 us of host round trip per token; the growth at B=1 is one SM per unit reading L2 at 165 GB/s, not misses.
  The host round trip, measured (tests/host_trip.py): torch argmax kernel 15-18 us, launch + host read per token
  22-23 us; a device argmax behind the online contract would gain the first part only (0.3%).
- New tools: tests/ref_long.py, long_err.py, abl_long.sh, jit_long.sh, score_round.py, tail_dir.py, cmp_exact.py,
  presub.sh, final_chk.sh, set2.sh, host_trip.py, long_stats.py, att_ctx.sh, abctx2.sh; tests/orig/ holds the inherited and the s0001 kernel source.

# Summary of the earlier run (s2-q8b-a1) that built the engine
## Best submissions
- Track B (lossless, model-specific): **s0063 - 4020.0 us per step at B=1** (generate; step 4025.1),
  4180.3 us at B=8. The committed `engine/` holds this kernel (re-submitted as s0065: 4047.3 / 4174.4, same
  kernel plus a host-side diagnostic export; the difference is the ~20-30 us spread between evaluations).
- Track A (same formats and data flow as vLLM / SGLang; build with `engine/TRACK=A` or `MK_TRACK=A`):
  s0053 5229.7 us at B=1, s0064 / s0066 5370.2 / 5367.5 at B=8 (s0066 = the committed source).
- Like-for-like workload (prompt 64 + 1024 greedy tokens, token on the host every step, `step()` + argmax +
  host read): track B 4163 / 4452 us per token (B=1 / B=8) = 1.46x / 1.54x SGLang 0.5.18 + torch.compile
  (6066 / 6836); track A 5384 / 5680 = 1.13x / 1.20x. Prompt 8192 + 1024: track B 4777 / 7905, track A
  5989 / 9177 (no baseline posted).
- Correctness: every submission valid (66 / 66); bitwise equality alone vs in a batch of 8 (also across the
  split-attention path at long context); generate == step + argmax; float64 / float32 checks to 9216 positions;
  randomised stress tests bitwise repeatable.

## Design in one paragraph
One persistent kernel per step() / generate() call: 132 blocks x 24 warps (23 loader warps stream weights with
a software pipeline across the per-tile block barrier, one coordinator warp reduces partials and runs the
epilogues), a call-free interpreter of phase descriptors in shared memory, 5 grid barriers per layer (atomic
counter + poll). Weights pre-permuted into MMA fragment order; track B stores them in a lossless 12-bit code
with exact power-of-two row/column scales and exception MMAs. Attention: one block per (sequence, kv head) with
the K/V history pulled into L2 by idle blocks; from 1136 positions with 1-2 sequences a split over all SMs with
a canonical summation order shared by both paths.

## What worked (largest first)
1. Fewer bytes: the 12-bit lossless weight code (-15%, track B only).
2. Keeping HBM streaming across every boundary: block team with a pipelined tile loop, last-use loads, stash of
   the next phase's first tile into shared memory during attention and at phase ends (adaptive, with a GO flag),
   prefetch of the next tile during barrier polls.
3. Not spilling: thread / block / grid indices read through inline asm, descriptors in shared memory.
4. Shared-memory size under the L1 threshold; cheap decode layout of the 12-bit stream.
5. Removing waits that were not dependencies (late in the run): no block barrier at phase entry with 3+
   sequences, per-warp input slices with 1-2 sequences, attention-free blocks not waiting after q/k/v, no fence
   behind the K/V append, matvec tiles on 128 SMs so every block has the same tile count.
6. Track A: RMSNorm applied by the loaders (no norm phase, 73 grid barriers less per step at B>=3), normalised
   slices kept in shared memory.
7. Long context: split attention for 1-2 sequences, V pulled into L2 during the softmax, softmax passes cut.

## What did not work
Dependency-started phases and run-ahead loaders; dynamic tile claiming and calibrated tile counts; flag /
stamp-based barriers instead of the counter (three variants, all slower); almost every L2 prefetch issued more
than a few hundred ns ahead; cp.async stash; early K or V loads before the prep or score pass; tile-major or
padded K/V layouts; bf16 split scores (unexplained bitwise mismatch); spreading attention blocks across the GPU;
fewer loads in flight in the streaming loop; per-warp normalisation in track A (spills). Details and numbers for
each are in notes/journal.md.

## Open problems / what I would try next
- Prompt 8192 at B=8: about one process in six runs ~8% slower with the same library (physical placement of the
  K/V cache, not visible from virtual addresses). A rescore of that case may need a repeat. Next: allocate K/V
  in chunks with a deliberate per-unit interleave and measure the slow-mode rate over many processes.
- The remaining gap to the byte bound (track B B=1: ~4020 vs ~3640 us) is mostly grid-barrier latency (~2 us per
  phase, 5 phases per layer) and the attention + o chain (~5 us over the o bytes per layer). Next: overlap the
  attention chain with the o phase per head (fewer cross-SM round trips), and a chunk-merge attention
  formulation shared by all batch sizes so that one sequence can use many SMs at all context lengths (flat step
  time in context, as another run found) without giving up bitwise batch invariance.
- Track A at B=1: the block-wide normalising copy at each norm phase (~0.5 us x 73) - a version without spills.
