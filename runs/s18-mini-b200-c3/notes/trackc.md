# Track C engine, run s18-mini-b200-c3 (direction: acceptance length)

Inherited: s16-mini-b200-c1 s0022 (notes/prev/trackc.md: the 8-row verify kernel, the fused DSpark draft, n-gram
copies, submit/poll serving). This file describes what this run added; numbers in notes/journal.md and gap.md.

## 16-row verify passes (engine/mk16.cu -> build/libmk16.so)
- A second kernel library, used only for verify passes of 9..16 rows: s16's proto16 (rows bit-identical to step(),
  written at s0013) plus this run's changes. step() and the 8-row verify kernel (engine/mk.cu) are untouched except
  RING 32 (history ring slots; step() bit-identical to s0001 by tools/kit/bitcheck.py, B=1 and 8, 1152 positions).
- Shared with the main library (api.py builds a second Ctx, Ctx16, with the same pointers): all weights, the K/V cache,
  the four history rings. Own: every tagged scratch buffer (16-row sized), pbt16 / fout16 / next_tok16, barrier.
- Both verify kernels are queued every cycle (the host does not know the next pass's row count; the draft writes it on
  the device): Hdr.flags 0x200 makes the 8-row kernel return unless vpos[1] <= 8, 0x100 makes the 16-row one return
  unless vpos[1] > 8 (block-uniform, before any barrier or tagged store; ~5 us per cycle).
- Valid only while every row's attention window starts at 0 (positions <= 1023): the draft proposes 16 rows only at
  anchors c <= LIM16 = 1008.
- Changes to proto16 that paid (V(16) at ctx 600: 10.6 -> 8.57 ms): MoE in one pass over 16 rows (router per 8-row
  half; one routing / w13 / w2a; an expert chosen by more than 8 rows takes two union entries of <= 8 tokens, so
  every item path is the 8-row one; entries built from per-expert row masks); expert inputs of 16 rows in smem with
  the router input aliased onto x16 rows 12..15 (converted in place after the router, rows in order, every element by
  the same thread): smem 213 -> 176 KB; the final unembedding in one pass (two column groups per weight fragment:
  piece_bf16_2); w2a staging 2 loads deep.
- Tested and off (switches kept, measured neutral or slower): QKV1, ROUT1, MRG2, PR16_UNR, MG16_UNR, W2_BPC16.
- Checks: tests/c3/vcheck16.py (16- and 8-row passes mixed vs step(): every position's logits bit for bit, 0 of 1020
  in every build), tests/c3/robust16.py (submit/poll vs step-by-step greedy, 14 edge cases incl. 16-row passes across
  the 1008 limit, prompts of 990-1003 tokens, edge ids, the too-long fallback: all identical).

## Draft side (engine/draftk.cu, draft.py)
- Verify rows VR = 16 next to the block's NB = 8: acceptance over up to 16 rows (target tokens from next_tok16 when
  the pass had > 8 rows: st[7]), emit of 16, the draft's context K/V for 16 rows (a second GEMV job set that returns
  unless st[7]), the n-gram history of 16 tokens.
- n-gram copies: 15 drafts (a 16-row pass) when the match is >= 5 tokens, c <= 1008 and the cycle just verified
  accepted >= 7 drafts (MK_NG16_ACC: a copying regime), else 7 drafts in 8 rows.
- DSpark draft length: kd maximizes (1 + sum_k cum_p_k^2.5) / (V(kd + 1) + 400 us) over the measured V table
  (DK_RULE 1), kd >= 0 (a 1-row pass when no draft pays), Markov early exit at cum p < 0.3.
- FP8 per-row-scaled copies of the draft layers (as already for the base unembedding and the fold): draft only.

## Acceptance (HTTP, tokens per event = tokens per verification step)
synthetic b1 2.88 -> 3.07, natural b1 5.22 -> 6.61 (vLLM + DSpark: 3.48 / 5.21 on its own text).

## What did not pay (simulated with tests/c3/sim*.py + ana/*.py, or measured) and why
- Trees within 8 rows: the best static tree of <= 7 nodes is the chain; dynamic trees and chain + leaves gain
  1.2-2.5% (synthetic) over the best chain rule because the verify cost grows with rows; a depth-1 alternative alone
  0.5-0.9%. A 16-row DSpark block degrades depths 1-7.
- Vocabulary-restricted or FP4 draft logits lose acceptance as fast as they save time. Jacobi reuse of the rows
  after a rejection: 1.4-7.5% hits. DSpark + n-gram extension, 32-row passes, 16 rows past 1008: <= 1.5%.
- Draft kernel fusion: the 46-kernel graph costs 42 us empty; double-buffered / looping GEMVs and other strip
  splits: no gain.

## For a merge with a cheaper verifier
The 16-row library is the s0013 verify code + the changes above; a faster 8-row verifier (e.g. the s17 step
improvements) does not carry over to it automatically. The gain of 16-row passes grows as V(16) - V(8) falls
(simulated: -12% natural at V(16) = 6.5 ms vs -6% at 8.6). Trees become worth building only if V(K) gets flat.

## Scenarios for a merge with a cheaper verifier (tests/c3/ana/a21.py; simulated on the 24 collected sequences)
V8(K) = floor + slope x (measured V8(K) - V8(1)), V(16) = V8(8) + slope x 3.17 ms; us/token synthetic / natural:
| verifier | chain, 8 rows | + 16-row copies | + dynamic trees | 16-row gain | tree gain |
|---|---|---|---|---|---|
| today (3.47 ms floor) | 1672 / 1136 | 1619 / 1096 | 1601 / 1094 | -3.1 / -3.5% | -1.1 / -0.2% |
| floor 2.9 ms (s17-like step) | 1468 / 1008 | 1430 / 986 | 1415 / 984 | -2.6 / -2.3% | -1.0 / -0.2% |
| floor 2.9, row costs / 2 | 1320 / 863 | 1263 / 805 | 1241 / 801 | -4.3 / -6.7% | -1.8 / -0.5% |
| floor 2.9, row costs / 4 | 1238 / 783 | 1173 / 708 | 1141 / 699 | -5.3 / -9.6% | -2.7 / -1.2% |
Take-away: cheaper rows (V(K) - V(1)) make the 16-row copies worth 2-3x more; trees stay <= 3% in every case.

## Multi-sequence passes (built, bit-identical, not used)
mk16.cu takes an optional row table (Hdr.mt: rows, then per row sequence, position, earlier rows of its sequence):
rows of up to 8 sequences at diverging positions, per-row attention path (tests/c3/vmulti.py: 471 rows in 40 passes,
0 differing from each sequence's B=1 step()). Cost at ctx 600 (tests/c3/tvmulti.py): 8 rows 6396 us (step B=8:
5889), 10 rows 7825, 12: 8350-8480, 15: 9125, 16: 9609. B=8 speculation with n-gram drafts (ana/a20.py) loses
(natural 5889 -> 6460+ us per token per stream): sequences in copying stretches finish early, the cohort then runs
with fewer live sequences, the slowest stream sets the decode window; DSpark at B=8 would be 8 draft passes.
