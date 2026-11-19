# Summary: run s11-north-h100-b1 (track B for North-Mini-Code-1.0 on one H100)

Details and every measurement with its job id: notes/journal.md. Where the time goes: notes/gap.md. Checkpoint
findings: notes/checkpoint.md. Plan and losses: notes/plan.md.

## Result (us per step over HTTP, the evaluator's client; B=1 is the headline)

| | B=1 | B=8 |
|---|---|---|
| vLLM 0.32.0 | 5049 | 9277 |
| Cohere's megakernel | 3616 | 7176 |
| inherited track A (s0022 of s4-north-h100-a1) | 2540 | 6900 |
| inherited track B (s0021) - not bit-identical to its track A | 2150 | 6050 |
| this run, track A (s0024, final source) | 2501 | 6493 |
| this run, track B (s0026, final source) | **2038** | 5334 |
| this run, track B, best B=8 (s0023) | 2045 | **5320** |

Track B's logits are bit-identical to the track-A build of the same source at every submission since s0001
(tools/kit/bitcheck.py, 1152 positions, B=1 and B=8; both builds batch-invariant), and all 30,484,201,472 weights
widen to exactly the operand the checkpoint defines through the kernel's own decode (mk_w12_widen, 0 mismatches).
B=1 and B=8 differences of < 10 us between submissions are within the evaluator's spread.

## What changed (each measured alone, same-card A/B with tools/abm.sh; then evaluated)
1. Bit-identical track B (s0001): the 12-bit widening adds an exponent bias of 32 (a PRMT constant byte), so every
   escape is a normal bf16 in the scaled operand space and is XORed into the mma operand before the product (one
   uniform header-bit test per quantum; the ~22% of quanta with an escape take one compact patch loop). The layer-0
   tables are made by the kernel's own layer 0 (a DUMP instantiation, one launch for all 262,144 ids, 2.4 s).
   Escape fix-ups after the sum are gone: B=1 2150 -> 2123, B=8 6050 -> 5769.
   (First version: patch code per unrolled pair made the kernel 3x larger; instruction-cache misses cost 600 us.)
2. Batch-1 kernel bulk-prefetches (cp.async.bulk.prefetch.L2) its next-layer stage-A tiles right after signalling
   stage C (s0003, track B only: the BF16 build loses 51 us with it): 2123 -> 2108.
3. 8-column operand strides padded by 4 words (s0004 A / s0005 B): the x operand of all 8 batch columns sat in one
   shared-memory / L1 bank (8-way conflict; the 12-bit loop's per-SM ceiling 32 -> 42 GB/s): B=8 5769 -> 5646,
   track A B=8 6900 -> 6823.
4. Attention runs two position chunks at a time on warps 0-7 / 8-15 and the gate|up share model counts the paired
   passes (s0006 A / s0007 B): B=8 5646 -> 5612, track A 6823 -> 6745.
5. Share model's attention cost per K tile 12 at batch > 1 (s0008 A / s0009 B): B=8 5612 -> 5587 (track A: 0).
6. Head sketch: 8 loads in flight in the batch > 1 kernels (s0010 A / s0011 B): B=8 5587 -> 5533.
7. The batch > 1 kernels launch without the dead 16 KB shared-memory area (L1 28 -> 60 KB; the batch-1 kernel keeps
   it: below ~200 KB the SMs stream unevenly) (s0012 A / s0013 B): B=8 5533 -> 5437, track A 6688 -> 6569.
8. Share table cached across steps (s0014 / s0015), RMSNorm x loads bypass L1 (s0016 / s0017): B=8 -> 5392.
9. "Small prefetch while waiting at a barrier" (track B, batch-1 kernel unless noted): the half tile + first whole
   stage-A tile after the stage-C signal (s0018: 2101 -> 2076), the first two experts' down tiles after o_proj before
   barrier B (s0019: 2051; also track A s0020: 2517 -> 2506), and - using how this model routes (a token keeps on
   average 4.6 of the previous token's 8 experts per layer) - one gate|up tile of the previous step's union during
   the union wait (s0021: 2044). The first two also in the batch > 1 kernels (s0022: B=8 in process -38..-44 us).
   The same prefetches issued while a block works (norm, stage A, before o_proj, at release), in larger amounts,
   by class (early blocks take more), or by the idle free blocks at the stage-C tail all lost (+13..+200 us).
11. Batch-1 schedule settings measured together (s0026): the stage-B barrier poll goes out with the last o_proj
   quantum and the share model's base attention delay is 30 at batch 1 (12-bit build; neutral on track A): -12..-19 us
   in the A/B, 2038 over HTTP.
10. The greedy head sketch skips the 439 tiles made only of rows that copy an earlier row (s0023; checkpoint fact,
   notes/checkpoint.md section 3): same binary on/off -8 us B=1, -10 us B=8 on average.
Also: contract guards in api.py (token ids, context, batch size); checks tools/batch_all.py (row 0 bit-identical at
every batch size 1..8, generate == step greedy), tools/long_ctx.py (to position 2047, refusals), tools/soak.py (12 x
512-token served runs identical at B=1 and B=8), tools/serve_all.py (cohorts of 1..8 requests stream the alone-greedy tokens), tools/kit/bitcheck.py (track B == track A), tools/verify_w12.py
(all 30,484,201,472 weights through the kernel's own decode).

## Checkpoint facts this run measured (notes/checkpoint.md)
- Escapes need exponent codes -23..+22 around the row window: a constant bias of 32 makes all of them normal bf16
  operands (the basis of bit-identity).
- The final hidden state's energy is spread over the dimensions (top 1024 of 2048 hold 51-66%): no partial-dimension
  head sketch.
- 7,129 head rows copy an earlier row; 439 tiles are made only of them, all at the vocabulary's end.
- A token shares 4.77 of its 8 experts with the previous token per layer (3.6 in layer 1, 5.1 in layer 48).

## What I would try next
- The norm's x loads take 1.2 us (B=1) / 1.9 us (B=8) per layer in the engine but 0.35 / 0.5 us in isolation;
  refuted causes: bulk prefetch, TLB, hot lines, the norm-weight load, the leader, L1 size. Finding it is worth up to
  ~45 us at B=1 and ~70 at B=8.
- B=1 stage B ends with only the ordinary blocks streaming o_proj (their o_proj takes 5.3 us after gate|up while
  the other classes are done ~2.5 us earlier); the share-model constants are at a local optimum, so this needs a
  different split of o_proj (e.g. K-split across classes with a fixed-order combine).
- At B=8, a stage-C spread of ~2 us between blocks with equal work: speed-aware row-tile shares for the batch > 1
  kernels, which now run with less shared memory and therefore on unevenly fast SMs.

## Measured and not kept (details in the journal)
Bulk prefetch at barrier release (+70 us B=1), after the norm's loads (+78), of the down tiles after gate|up
(+90 B=1 / +440 B=8), of attention K/V (+46..+84 B=8), in the batch-8 kernel (+19..+66); deeper loads per quantum
(QDP=4, +17); software pipelining across quanta (0 in the bench); butterfly norm reduction (+16 B=8); union polled by
an idle warp (UPOLL, 0 / +24); 64 attention blocks at B=8 (+160); 4 blocks per attention task at B=1 (-20 short,
+39 long); share-model constants GS_DO1 / GS_A0 / GS_DO (local optimum); TLB and hot-line explanations of the slow
norm x loads (both refuted by microbenchmarks); a head sketch over a fixed subset of dimensions (the final hidden
state's energy is spread: top 1024 of 2048 dims hold 51-66%).

## Process lessons (for whoever continues)
- Resolution: two identical builds in tools/abm.sh differ by up to ~10 us (B=1) / ~20 us (B=8, long contexts);
  separately compiled variants also move by up to ~15 us through code generation alone (even step(), which a
  greedy-path change cannot affect). When a change can be switched at run time, A/B the same binary (MK_HSKIP).
  The evaluator's B=1 figure varies by +-5..10 us between submissions of logically identical batch-1 code.
- Ablations with wrong results (ABL_NOSTREAM, OABL, AABL, ESC_ABL) bound a gain before building it; they told
  apart the escape-patch costs (execution vs. code size) and showed which o_proj / stage-A work is critical.
- Microbenchmarks can mislead in exactly the relevant way: bench/xread.cu first measured stale L1 hits (plain loads,
  values never checked), bench/tlb.cu first streamed one contiguous slice per SM instead of the engine's many pages.
- Kernel code size matters: patch code replicated into every unrolled loop cost 600 us (instruction fetch), the
  compact version nothing.
- /tmp in the pod is shared between jobs on different cards (abm.sh results file collided once); pods were lost
  ~15 times, and a job that dies with two pods is not run again.
