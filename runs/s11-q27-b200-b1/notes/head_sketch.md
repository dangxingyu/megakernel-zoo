# Certified int8 sketch of lm_head (track B, build flag -DHEAD_SK; mk.cu sketch_work / rescore_work)

Used only by greedy passes that return no logits (served decode through submit/poll, generate()); step() keeps the
exact head. The chosen token is the exact head's argmax (largest bf16 logit, smallest index on ties), proven below.

## What the exact head computes
For column c: z_i^c = the float32 result of the head GEMM's MMA chain(s) for row i (pieces of the tile's 80 quads as
the warps' shares cut it, combined as finalize / the sentinel hand-over do), logit_i = bf16(fl(z_i^c * r)) with r > 0
the RMSNorm scalar, chosen = argmax over i of logit_i, ties to the smallest i (amax_key).

## The sketch (load time, api.py _sketch_build)
- s_i = max|W_i| / 127 (float32), q_i = round(W_i / s_i) in [-127, 127] (int8), e_i = W_i - s_i q_i (exact in float64).
- c1_i >= ||e_i|| + 2^-10 ||W_i||, c2_i >= s_i ||q_i||, both computed in float64 and rounded up to float32 with a
  relative slack of 2^-20.

## Per pass (sketch_work)
- x (the head's bf16 input) per column: sx = max|x| / 16256, q_k = rint(x_k / sx), q_k = 128 d1_k + d0_k with
  |d1| <= 127, |d0| <= 64 (int8). x^ = sx q. ||x||, ||x - x^|| in float64, rounded up.
- A1_i = sum_k q_ik d1_k, A0_i = sum_k q_ik d0_k: integer tensor-core sums (m16n8k32 s8, int32, |A| < 2^27: exact).
- zh_i = fl(fl(fl(128 A1 + A0) * s_i) * sx): three roundings, |zh_i - s_i sx (128 A1 + A0)| <= 2^-22.9 |zh_i|.
- |z_i(real) - s_i q_i . x^| <= |e_i . x| + |s_i q_i . (x - x^)| <= ||e_i|| ||x|| + s_i ||q_i|| ||x - x^||
  (Cauchy-Schwarz, exact real arithmetic).
- |z_i^c - z_i(real)| <= 2^-10 ||W_i|| ||x||: the float32 accumulation of the exact path (320 MMA k-steps + 1 combine;
  each step's error is at most a few ulp of the running sum of |terms| <= ||W_i|| ||x||: ~321 * 2^-22 = 7.7e-5 under a
  2-ulp-per-step model; 2^-10 = 9.8e-4 is a 13x margin. This is the one non-proved assumption (the MMA's internal
  rounding is not documented); the margin is what s4n's audit asked for.
- B_i = c1_i ||x|| + c2_i ||x - x^|| + 2^-20 |zh_i|, times (1 + 2^-20), all with round-up float ops; zu_i = zh_i + B_i
  (round up), zl_i = zh_i - B_i (round down). So zl_i <= z_i^c <= zu_i.
- Per tile and column: max zu over its 16 rows (tmax); per column: L = max zl over all rows (atomicMax on an
  order-preserving key).

## Rescoring (rescore_work)
- thr = L (1 - 2^-7 - 2^-20) (rounded down) if L > 0, else -inf (every tile: the exact head, slow but correct).
- Claim: if zu_i < thr then logit_i < logit_j for the row j with zl_j = L. Proof: z_i^c <= zu_i < z_j^c (1 - 2^-7 - 2^-20).
  If z_i^c <= 0 < z_j^c the bf16 signs settle it. Otherwise fl rounding (2^-24) and bf16 rounding (2^-8 relative,
  round to nearest) give logit_i <= z_i r (1 + 2^-24)(1 + 2^-8) < z_j r (1 - 2^-7 - 2^-20)(1 + 2^-24)(1 + 2^-8)
  < z_j r (1 - 2^-24)(1 - 2^-8) <= logit_j. So every row that can be the argmax, or tie with it, has zu >= thr and
  lies in a candidate tile.
- Candidate tiles are recomputed with the exact head's arithmetic: the same pieces (the head GEMM's partition of the
  80 quads of the tile over its warp shares), each an MMA chain from 0 over the same k-steps in the same order, with the
  same B operands (the head's x buffer and layout), combined as the head combines them ((0 + p1) + p2 inside a block,
  (0 + (0 + p1)) + (0 + p2) across blocks, p alone for a whole tile), times r, bf16, argmax with the smallest index.
  The weights come from a BF16 copy of lm_head: in the W12_RS engine every exact-path piece is the BF16 piece times
  2^(L0 - lo_r) (power of two, no subnormals) and is multiplied back before r: the same float32 values.
- The chain is relayed through the 32 warps of a block (5 k-steps each, two rounds), so the 320 dependent MMAs are not
  waiting for loads.

## Checks
- tools/sk_check.py: generate() (sketch) against the min-index argmax of step() logits (exact head), teacher-forced on
  the generated sequence, B=1 and B=8.
- tools/head_census.py (offline, float64): int8 per-row sketch, 4608 real positions: candidates per position mean 1.9
  rows, p99 11, max 70 (45 tiles); the true argmax was inside the candidate set at every position.

## 7-bit variant (SK7, s0009)
- q = round(w / s) with s = max|w_i| / 63, |q| <= 63; the bound is the same formula (c1 = ||w - s q|| + 2^-10 ||w||,
  c2 = s ||q||, both from the 7-bit q): nothing in the proof depends on the number of levels.
- Stored as u = q + 64 in [1, 127]: per lane and unit of 8 k-steps the 32 fragment words A become 28 words S,
  S[7m + j] = (A[8m + j] & 0x7F7F7F7F) | (bit j of each byte of A[8m + 7]) << 7; the kernel undoes this exactly
  (checked on all 1,271,398,400 values: tools/sk7_test.sh).
- The tensor cores compute sum_k u_k d_k (u8 x s8, int32, exact: |sum| <= 127 * 127 * 5120 < 2^31); the kernel
  subtracts 64 * sum_k d_k (the column's digit sums, exact integers, one per digit) once per tile, which gives
  sum_k q_k d_k exactly, i.e. the same integers the int8 path computes for the 7-bit q.
- Candidates (head_census, 3072 positions): 6.9 tiles mean, p99 78, max 821 (int8: 2.3 / 16 / 67).

## Power-of-two x scale (SK_P2X)
- sx = the smallest power of two >= max|x| / 16256 (computed with an upward-rounded product, then rounded up to a
  power of two), so |x| / sx <= 16256 and x / sx = x * 2^-E is exact: q = rint(x / sx) has |x / sx - q| <= 1/2 with no
  division rounding, |q| <= 16256 (two int8 digits, d1 = rint(q / 128) in [-127, 127], d0 = q - 128 d1 in [-64, 64]).
- ||x - x^|| <= sqrt(5120) sx / 2: the kernel keeps the old, larger constant (sx * 35.795 * 0.50098, rounded up).
  sx may be up to 2x the old scale, so this term can double; it is ~1e-4 of ||x|| against ~0.5 % for the 7-bit
  weight term (candidates unchanged in practice).

## Mixed 6 / 7-bit tiles (SK76)
- Each 16-row tile is stored in 7 bits (u = q + 64, |q| <= 63, s = max|w| / 63) or 6 bits (u = q + 32, |q| <= 31,
  s = max|w| / 31); the bound constants of a row come from its own q and s, so the proof is unchanged; the kernel
  subtracts the tile's own offset (64 or 32) times the digit sums. Which tiles are 6-bit only changes how many tiles
  become candidates (rescored exactly), never which token is chosen.
- The set (engine/sk6_tiles.bin): tiles that were candidates under 6 bits at <= 80 of 6144 own greedy positions
  (tools/mixed_census.py, prompts seeds 60000+): 93.9 % of tiles. Storage: the 7-bit tiles first, then the 6-bit
  ones; op.p11 maps storage slot -> tile for the sketch and the rescoring scan.

## Hot tiles in 8 bits (SK768)
- In the B=1 mixed sketch, the tiles outside the 6-bit set are stored as the original int8 sketch (s = max|w| / 127,
  signed digits, no offset); the proof is the int8 one. Fewer of them become candidates (census: mean 33 -> 11).
