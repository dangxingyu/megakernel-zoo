# Plan (rewritten at board hour ~17)

Status: track A s0033 10121 / 10440 (1.130x / 1.120x SGLang); track B s0036 8995 / 9439 (1.272x / 1.239x) and s0034
9016 / 9433 (best B=8, 1.240x). Tree = s0036 exactly (engine/TRACK = B); MK_TRACK=A builds s0033's track-A library.

## What moved the numbers since hour 12 (journal for each)
- Track B memory-level parallelism: 3 quads per warp in flight where the inputs can come from shared memory just in
  time (norm GEMMs; o / down at B=1 with the input column staged): streams from 75-93% to 95-97% of the floor.
- Packed bf16 arithmetic in the norm prep (same bits).
- Attention combine at 1-6 sequences split over all blocks of the unit (the single combiner's reads were
  bandwidth-limited at ~25-35 GB/s per SM under the weight prefetch traffic).
Repeated lesson: the 7-8 sequence instance's code generation moves by 30-40 us (B=8) with any code change compiled
into it; changes meant for other instances are compiled out of it (template flags).

## Remaining losses (B=1 track A, per layer): norm prep ~2 us x 2 (latency under load), attention chain ~9 us
(prep 2.5, scores 0.9, P x V 1.8, combine ~3, signal 1.4), per GEMM op tail + epilogue + barrier ~4.5 us x 3.
Track B B=8: norm prep ~4 us x 2, o / down streams at 76% / 87% (inputs from L2, 2 quads in flight).

## Next
Only small, separately measured items; at the end: final verification, notes, engine/TRACK = A in the tree.

## Revision at board hour ~23.6 (the run was extended to 36 h; 12.4 h left)
Order: (1) measure the ~2.5 us from a block's stream end to its epilogue end in every GEMM op (up to ~640 us/step on
both tracks) with timer reads ordered against bar.sync (memory clobber), split into reduction / hand-over wait /
writes; fix the dominant part. (2) Track B's escape-heavy layers 1-5 (~50 us/step at both B). (3) The paired-tile
stream rewrite for track B's o/down at B=8 (2-4% at B=8) only if time remains. engine/ stays = s0036 until a
submission beats it; every candidate gets the full check suite before submission.

## Status at board hour ~26.5
Done since the extension: ordered stamps located the GEMM epilogue's critical warp (warp 0, cut tile) and the
attention chain's phases; fixed: prefetch from the last warp (s0037), 32-bit range divisions (s0038/s0039),
counter-based combine arrival (s0040/s0041). Measured negative: per-warp cut-tile hand-over, narrowed reduction scan
(also in 32-bit), staged fold, per-op prefetch before norm GEMMs, V prefetch at B=8, o without its preload.
engine/ = s0041 (TRACK = B). Remaining time: small measured items only; no rewrite of the stream partition.
