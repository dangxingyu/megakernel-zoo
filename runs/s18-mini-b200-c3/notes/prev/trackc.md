# Track C engine: exact draft-and-verify for Inkling-2-Mini on one B200 (run s16-mini-b200-c1)

Final summary: notes/summary.md (best submission s0022, 1528 us/token b1).

## Design
- **Verifier** = the track-A step function. `step()` is unchanged (bitcheck against s15 s0016: bit-identical at B=1/8).
- **Verify pass** = `mk_step<true>` (engine/mk.cu): rows b < K (K = vpos[1], from the device) are one sequence at
  positions vpos[0] + b. Every per-row computation is the step kernel's own arithmetic, so each row's logits are bit
  for bit the logits `step()` computes at that position (tests/c/vcheck.py: 0 of 1100 / 2600 positions differ, random
  K 1..8, through the 1024 window). What differs from a batch of sequences:
  - K/V of earlier rows of the pass: computed inside the attention item from the producing rows' tagged qkv outputs
    (conv lags of in-pass rows from those rows; older lags from a 16-slot ring per position), not read from the cache.
  - short-convolution lags of the attention output (o_proj epilogue: shuffles between lanes) and of the MoE output
    (w2b: shared memory) come from the earlier rows of the pass.
  - shared attention records when every row's window starts at 0 (global layers, or positions < 1023): one item per
    (kv head, 64-key record) computes that record for all rows (same per-record arithmetic), merge items per (row, head).
  - mixes of 2-row passes redundant in every block (no designated-block hop); w2a one warp per item as three chains in
    block_strips order, ranges balanced by work; in-pass raw k/v kept in smem.
- **Draft** = the bundled DSpark head (mtp.safetensors) as fused CUDA kernels (engine/draftk.cu, libdk.so, compiled
  without fast math; a torch reference TorchDraft with the same function, MK_DRAFTK=0). Per cycle: acceptance (a
  leading drafts equal to the target's tokens, a < K), the draft's context K/V of the verify rows, the 8-row block
  through 3 layers, the Markov recurrence; per Markov step p_i = exp(zmax - lse) (fixed-order sums); the draft
  length kd = longest prefix with prod p_i >= 0.4 (steps beyond it skipped), K = kd + 1.
- Since s0008 the draft's base unembedding and fold are read from FP8 E4M3 per-row-scaled copies (draft only; the
  target never sees them) and the n-gram rule runs inside the draft cycle (k_ngram), skipping the DSpark block and
  Markov kernels when it is used.
- **n-gram copies** (engine/ngram.py rule, `mk_ngram_k` in mk.cu): if the committed suffix ending at the anchor
  matches an earlier occurrence over >= 5 tokens (longest, then most recent, up to 8), its continuation replaces the
  drafts and K = 8.
- Further verify-pass details since s0010: gate D items mix the loop messages first; after designated publishes each
  block polls only the qkv input streams its strips read; 2-row passes mix row 0 in every block and row 1 on the last
  block (published); the final-norm rows of passes >= 3 rows come from designated blocks.
- **Serving** (`submit()` / `poll()`): a B=1 cohort runs draft-and-verify (prompt prefilled by verify passes of 8
  rows, bit-identical); 3 cycles queued on the stream, each cycle's committed tokens copied to pinned memory and handed
  over per cycle (one poll event = one cycle). Cohorts of 2-8 run the step kernel's device loop (no speculation).
  Requests too long for the spec path's margin fall back to the device loop.
- Exactness: every committed token is the target's own argmax at a position whose logits are bit-identical to
  `step()`'s; the evaluator's generate / http_tokens gates (no tie allowance in track C) pass 8/8 in every submission.

## Results (HTTP, us per token per stream)
| sub | change | b1 | natural b1 | b8 | tokens/event b1 / nat |
|---|---|---|---|---|---|
| s0001 | torch draft + 8-row verify | 2877 | 1595 | 7055 | 2.79 / 4.93 |
| s0002 | fused draft, n-gram, shared attention | 2158 | 1221 | 7045 | 2.96 / 5.35 |
| s0003 | dynamic draft length | 1764 | 1194 | 7043 | 2.89 / 5.28 |
| s0004 | K=2 mixes, w2a balance, n-gram kernel | 1705 | 1143 | 7048 | 2.89 / 5.28 |
| s0005 | draft early exit, GEMVs, n-gram >= 5 | 1697 | 1129 | 7044 | 2.88 / 5.25 |
| s0006 | track A: B>=2 step kernel w2a | 3227 (A) | - | 6356 | - |
| s0007 | raw k/v in smem, draft PDL, s0006 step kernel | 1673 | 1106 | 6352 | 2.88 / 5.25 |
| s0008 | FP8 draft copies, n-gram inside the draft, designated final rows | 1619 | 1067 | 6351 | 2.88 / 5.22 |
| s0009 | track A: raw k/v for the conv ring from smem | 3206 (A) | - | 6359 | - |
| s0010 | gate D: loop-message mix first (verify) | 1617 | 1064 | 6339 | 2.88 / 5.22 |
| s0011 | poll only the qkv input streams a block uses (verify) | 1598 | 1050 | 6343 | 2.88 / 5.22 |
| s0012 | hybrid mixes for 2-row verify passes | 1588 | 1050 | 6366 | 2.88 / 5.22 |
| s0013 | designated rows on the last blocks (3-4 rows) | 1587 | ~1050 | 6376 | 2.88 / 5.22 |
| s0014 | s0013 + off-by-default switches | 1585 | 1048 | 6376 | 2.88 / 5.22 |
| s0015 | track A: bf16 GEMV 3 units deep | 3169 (A) | - | 6353 | - |
| s0016 | BF_UNR 3 + verify row polls 2 deep | 1559 | 1031 | 6353 | 2.88 / 5.22 |
| s0017 | merge polls 1 deep, w2a staging 2 deep (verify) | 1541 | 1026 | 6350 | 2.88 / 5.22 |
| s0018 | track A: step-kernel polls 2 deep | 3141 (A) | - | 6300 | - |
| s0019 | s0018 step kernel + sequential router polls | 1540 | 1021 | 6310 | 2.88 / 5.22 |
| s0020 | no L2 prefetch in verify passes | 1531 | 1019 | 6302 | 2.88 / 5.22 |
| s0021 | 3-4 row passes poll published rows 1 deep | 1531 | 1020 | 6314 | 2.88 / 5.22 |
| s0022 | final committed source (s0021 + off-by-default switches) | 1528 | 1020 | 6302 | 2.88 / 5.22 |
vLLM + DSpark: 1544 / 871 / 2711 (tokens per step 3.48 / 5.21 at B=1).

## Findings worth keeping
- Verify cost V(K) at ctx 600: 3530, 4130, 4375, 4540, ~4950, ~5350, ~5700, 5770 us (K = 1..8). The first extra row
  costs ~600 us; rows 3-4 ~200 us each; MoE union growth dominates beyond.
- 8 consecutive positions of one random-token sequence: 21 union experts per application (13..43).
- Draft probabilities are overconfident (an expected-tokens-per-time rule needs p^2 to tie the plain threshold).
- n-gram copies on top of DSpark: +6% tokens per cycle on evaluator prompts, +8.5% natural (s0002).
- Dead ends (all bit-identical, all slower): one warp per w13 item, cp.async smem ring for w13, rolling register
  double buffer for w13, FP4 batch depth 4/12, kv_conv with all loads before waits, L2 prefetch at K=2.
- One extra 512 B of smem pushed the kernel over the 132 KB carve-out step (+200 us); code in the step kernel for the
  verify path cost +160 us until the verify kernel became its own template instance.

## For a next run
- 16-row verify passes for n-gram proposals were built (proto16/, bit-identical) but cost 10.1 ms: with MoE in 8-row
  halves they do not pay. They would need the MoE to take 16 tokens per expert in one pass, i.e. a new expert-input
  layout that fits shared memory (148 KB at 16 rows today). See proto16/README.md.
- Best track-A engine: this source with engine/TRACK = A (s0018: 3141 / 6300).
- Late lesson: shallow tagged polls (1-2 loads in flight per thread) beat deep batches wherever many blocks poll the
  same freshly published data; that flipped the L2 prefetch from a gain to a loss. Re-check switches after such shifts.
