# Summary of the run (Qwen3-8B decode megakernel, one H100)

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
