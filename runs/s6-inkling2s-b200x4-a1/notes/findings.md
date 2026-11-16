# Findings about the checkpoint and the node (each with the measurement behind it)

## F1. BF16 dense weights fit an exact 12-bit code (track B)  [measured, all weights]
tools/census_bf16.py, exec e0020: every value of the 337 dense BF16 matrices of the decoder (per layer wq, gate, wk,
wv, wo, wr, router; and the unembedding): 5,710,970,880 values.
- zeros: 1; subnormals: 0; exponent entropy 2.57 bits.
- values outside the best 16-exponent window chosen per row: 419,063 (7.3e-5); per 16-row strip: 596,073 (1.0e-4);
  per tensor: 668,233 (1.2e-4).
- As packed by engine/w12.py for my layout (row-sliced per card, plus the per-card zero pad rows which get window
  base 0 so that their zeros are exact): 432,825 escapes in 5,984,649,216 values, 12.0009 bits per value.
Allows: 25% fewer bytes for the 11.5 GB of dense weights read every step (~2.9 GB per step at B=1, ~720 MB per card).
Does not allow: anything below 12 bits with a fixed-width code (entropy 2.57 bits of exponent + 1 sign + 7 mantissa).

## F2. 12-bit decode speed on one B200  [measured, tools/bench/w12.cu, e0023]
GEMV streaming of 150 M values: BF16 3.13 G values/s (6.26 TB/s), 12-bit 3.99 G values/s (5.98 TB/s of 12-bit bytes):
+27% values per second. Decode per bf16x2 register: PRMT with sign-replicate (sign to bit 15/31) + LOP3 mask +
SHF/LOP3 nibble extract + IMAD (offset*128 + base) + LOP3 = ~5 instructions.

## F3. Engine-visible properties of the node
- sm_100a: no warp-level block-scaled FP4 MMA; cvt.rn.f16x2.e2m1x2 with free byte selects; NVFP4 GEMV via
  cvt + HMUL2 (E4M3 block scale, exact in f16) + mma.m16n8k16.f16 streams at 5.9 TB/s (BF16 6.2 TB/s) per card.
- 1024-thread blocks get 64 registers; with 222 KB of shared memory per SM the L1 has ~34 KB: any register spill or
  dynamically indexed local array costs an L2 round trip per access (seen as 5-10x slow phases).

## F4. Expert reuse between consecutive decode tokens (B=1, greedy on text)
- Measured (tools/t_overlap.py, e0120): 597 steps of greedy decoding from 3 text prompts (stdlib sources), per layer
  how many of token t+1's 8 experts were among token t's 8: mean 2.81 (35%); among the union of the last 4 tokens:
  4.25 (53%). Per layer 0.6 (layer 0) .. 4.7 (layers 12, 36, 42); early layers reuse least.
- Allows: an L2 prefetch of the previous token's experts would make ~1/3 of the B=1 expert reads L2 hits.
  Does not allow: a faster w13 at B=1 by itself, because the phase ends with its slowest warps and 2/3 of the warps
  still read from HBM (not pursued).
- At B=8 the union of the 8 sequences' experts on text is 47.5 per layer on average (35..61; random token ids
  route far more alike and gave a 15% too optimistic B=8 time).
