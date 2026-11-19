# Track B: certified int8 sketch of the head (HSK), greedy paths only

Built in engine/mk.cu (`head_sketch<BIG>`, -DHSK=1 = TRACK B) and engine/api.py (`head_sketch()`). step() (mode 0)
keeps the exact BF16 head; generate() and the served path (modes 1, 2) choose the token through the sketch.

## Format (load time, api.head_sketch, float64 on the GPU)
- per row i of lm_head (150144 x 2048): s_i = float32(max|w_i| / 127), q_i = round(w_i / s_i) in [-127, 127] (int8),
  e_i = w_i - s_i q_i (exact in float64), stored in the A-fragment order of mma.m16n8k32.s8: 307 MB.
- en_i,b = ||e_i,b||_2 over 32 blocks of 64 columns, rounded UP to BF16, as mma.m16n8k16 A tiles: 9.6 MB.
- wn_i >= ||w_i|| + ||e_i|| >= ||s_i q_i|| (float32, rounded up): with s_i, 1.2 MB.
The BF16 head stays as it is (step() and pass 2 read it).

## Per greedy step (kernel)
1. x = the head's BF16 operand (the same fragments the exact head multiplies). Per sequence: k = 14 - E (max|x| < 2^E),
   X = round(x 2^k) = 256 hi + lo (hi in [-64, 64], lo in [-128, 127], both int8); r = x - X 2^-k (exact; non-zero
   only for |x_j| < max/256); ||x_b|| per block (rounded up to BF16), ||x||, ||r||.
2. Pass 1: approx_i = s_i 2^-k (256 S_hi + S_lo), S = q_i . digits in int32 (exact). Batch 1-2: one mma per tile
   (hi digits in B columns 0..B-1, lo digits in 4..4+B-1); batch > 2: two. Bound:
   bound_i = (1 + 2^-8) sum_b en_i,b ||x_b|| + wn_i (||r|| (1 + 2^-10) + 2^-12 ||x|| (1 + 2^-10)) (1 + 2^-8).
   Terms: |e_i . x| <= sum_b ||e_i,b|| ||x_b|| (Cauchy-Schwarz per block); |s_i q_i . r| <= ||s_i q_i|| ||r||;
   2^-12 ||w|| ||x|| covers the float32 accumulation of the exact head (128 chained mma steps) and the float
   conversions of the approximation (measured exact-head error: <= 2^-15.4 ||w|| ||x||, tools/head_study.py).
3. Exchange (one extra grid hand-over): L_b = max over all rows of approx - bound.
4. Pass 2: row groups (16 rows) with any upper bound >= L_b are recomputed with the exact head's arithmetic (two
   warps per group: the chains over even and odd tiles, as c4[0], c4[1] of the exact head; then (c0 + c1) + (0 + 0)).
   The token = best exact value over the recomputed rows, lowest index on a tie. A row not recomputed has
   exact <= approx + bound < L_b <= exact of the best lower-bound row: it can be neither the maximum nor tie it.

## Where it is used (final form)
- Only in the batch 1-2 kernel instance (the batch > 2 instance of the track-B build has the exact head and is the same
  code as track A's: at batch 8 the sketch gained nothing: two mma per tile, ~71 candidate row groups, pass 2 7.8 us).
- The track-B build keeps the 16-mma form of the attention scores in its batch 1-2 instance (KQT_SMALL = 0): the same
  bits as track A's 8-mma form (KQT), and the better register allocation for this build (B=1 in process: B 1464-1472
  vs A 1471-1477 us on the same cards).

## Verification
- tools/hsk_verify.py: max |exact - approx| / bound = 0.1285 over 1600 head inputs (teacher-forced real text and
  random ids, batch 8) x 150144 rows: the bound is >= 7.8x the largest error seen.
- tools/kit/bitcheck.py A vs B: step() logits bit-identical at B=1 and B=8 (1152 positions, every 8th), batch
  invariant (e0038).
- tools/gen_cmp.py: generate() B=1 6 x 1024 tokens, B=8 3 cohorts x 8 x 1024, served cohort 8 x 1024 (full streams
  since the fix of journal 22): identical to track A (e0104, e0124).
- tools/hsk_bsizes.py: greedy tokens at every batch size 1..8 (384 each), a long context to ~1900 positions at B=2,
  and inputs at the end of the vocabulary: identical to the exact head of the same binary (e0121, e0125).

## Cost (tools/hsk_stats.py, -DHSK_STATS, block 0, scored workload)
- candidates on generated text: 9.6 row groups per step at B=1, 71 at B=8 (of 9384); on random ids up to all.
- B=1: prep 4-6 us, pass 1 49-52 us (317 MB; bench_hsk.cu: 45.6 us in isolation vs 84.9 for the BF16 loop), publish
  + wait 3.3, exchange 1, pass 2 1.4: 63 us against 66-79 us for the exact head. B=8: 81 vs 80 (no gain: two mmas per
  tile and ~8 us of pass 2).
- What it cost to get there: the call perturbs the register allocation of both kernel instances (the batch-1
  instance went from 176/20/16 to 192/40/100 bytes and +60-90 us in early versions). Holding the first batch across
  the preparation: 884 B of spills, +150 us. Passing a pointer to a caller variable: address-taken. The final form
  (few arguments, out-of-line, L2 bulk prefetch of the first 16 KB per warp) drew 176/16/8 for the batch-1 kernel.
