# Inkling-2-Mini checkpoint census (text decoder, model.llm.*)

Exact counts over every weight (integer ops on the GPU, every element, no sampling), card 1, 2026-11-20.
Scripts: `tools/census/ckpt_census.py` (everything below; ~10 s on a B200, 19.5 GB read once),
`tools/census/padding_probe.py` (padding-row statistics, spec/evaluator vocabulary), `tools/census/inventory.py` (header).
Raw output: `../execs/e0006/out/census.json` (exec e0006), padding probe exec e0007, header exec e0004.
GPU time used in total: under 1 minute. Nothing was done on a subset.

## Headline

| finding | number | allows (lossless) | does not allow |
|---|---|---|---|
| Padding rows 200058..201023 of embed and unembed are **not zero** | 966 rows x 2304 = 2,225,664 nonzero values in each (all of them) | spec (transformers#51050 `lm_head = Linear(hidden, unpadded_vocab_size)`) and the evaluator's reference (`inkling2_ref.py`: `unembed[:200058]`) never read them: drop them, compute 200058 logits; -4.45 MB per step of unembed read | treating them as zeros or claiming they are unused padding of zeros |
| No zeros, subnormals, Inf/NaN anywhere in the 2-D BF16 matrices | 0 of 1,608,221,696 | a 12-bit code needs no zero special case | - |
| 12-bit code (sign + 4-bit exponent offset in a per-row 16-binade window + 7 mantissa) | 120,996 escapes of 1,608,221,696 (7.5e-5) over all BF16 matrices; unembed 34,676, embed 26,887 | 25% fewer BF16 bytes (unembed 0.93 -> 0.69 GB) with an exact escape list | 11-bit (8 binades): 1.31% escapes - too many |
| No duplicate rows, no zero rows in any BF16 matrix | 0 / 0 (unembed included; 128-bit row hash, bit-exact verification) | - | row dedup |
| embed and unembed are different tensors | bit-equal at only 17,222 of 463M positions | - | tying them |
| NVFP4 E4M3 block scales: positive, normal, 36-50 distinct per tensor, bytes 0x4c..0x7e overall | 0 zero, 0 subnormal, 0 negative, 0 NaN of 1,274,019,840 scales; every expert's max scale is exactly 448 (0x7e) | fixed 6-bit scale code (byte - 0x40) on all 40 tensors: 4.5 -> 4.375 bits/weight (-2.8% of NVFP4 bytes, ~44 MB/step at B=1) | 5 bits (up to 50 distinct values) without a per-tensor table and escapes |
| E2M1 codes are near-incompressible | entropy 3.968 bits of 4 (pooled); byte (pair) entropy 7.93-7.94 of 8 | - | any fixed-width code smaller than 4 bits |
| Every NVFP4 16-block contains a code of magnitude 6.0 (code 7 or 15) | 1,274,019,840 of 1,274,019,840 blocks; no block is all zero | (confirms scale = block amax/6 quantization; the largest abs dequantized weight of an expert = amax exactly) | a storage saving at fixed width |
| NVFP4 denominators | 2688.0 (= 6 x 448) in all 40 tensors | a single constant | - |
| MXFP8 UE8M0 scales | 5-6 distinct values per tensor, 113..118 (2^-14..2^-9) | fixed 3-bit scale code (value - 113): 8.25 -> 8.09 bits/weight (~5 MB/step, negligible) | treating the scale as per row: 0 rows constant in w13, 186-223 of 294,912 in w2 |
| MXFP8 E4M3 weights | byte entropy 6.52-6.55 bits, 254 distinct bytes (all but NaN 0x7f/0xff), 0 NaN, ~47k / ~27k subnormals, ~6.2k / ~2.9k zeros | entropy coding only (variable length) | a 7-bit window code: 1.08-1.22% escapes |
| F32 tensors (attn_res queries, cross-pass queries, gate.bias, amax) | no zeros, no NaN, no duplicate rows, not BF16-representable (0 of 5120 amax, 0 of every query vector; 2 of 55,296 cross-pass values) | - | storing them in BF16 (not lossless) |

## 1. Padding rows of the vocabulary (rows 200058..201023, 966 rows)

| tensor | nonzero values in padding | nonzero rows | max abs | rows identical | row RMS padding (min..max) | row RMS real rows (min..max) |
|---|---|---|---|---|---|---|
| embed | 2,225,664 (all) | 966 | 5.28 | no | 0.99980..1.00019 | 0.99963..1.00026 |
| unembed | 2,225,664 (all) | 966 | 0.1045 | no | 0.020833..0.020836 | 0.02081..0.02085 |

- The padding rows look like random initialisation: Gaussian (kurtosis 2.995 / 2.997; real rows 3.05 / 3.19), mean |cos| between
  padding rows 0.017, max 0.10, no -0 values.
- Every row of embed has RMS 1.000 and every row of unembed RMS 1/48 = 0.020833 (= 1/sqrt(2304)) within BF16 rounding: the rows are
  normalised. (Not a storage lever by itself: the BF16 values are still needed bit for bit.)
- No zero rows in the real vocabulary of either tensor.
- The spec's `lm_head` is `nn.Linear(hidden, unpadded_vocab_size=200058)` and the reference slices `unembed[:200058]`: the padding rows
  are outside the function. An engine must not compute logits from them if it returns 201024 columns (they would be nonzero garbage);
  returning 200058 columns is what the reference has (check `meta["vocab"]` of the evaluator before relying on that).

## 2. BF16 matrices (all 2-D BF16 tensors of model.llm)

Zero treatment: **there are no zeros (and no -0, no subnormals) in any BF16 matrix**, so the choice only matters by construction.
Three variants of the 12-bit code (window chosen per row to minimise escapes, exact optimum per row):
- A: offset o -> exponent field base+o (16 binades), every code is a normal number; a zero would be an escape (there are none).
- B: code (offset 0, mantissa 0) means +-0; the value 2^(base-127) (exact power of two at the window base) becomes an escape.
- C: offset 0 means BF16 exponent field 0 (zero/subnormal exactly), offsets 1..15 = 15 binades.
A is best here (B costs +705 escapes, C doubles them). "esc B, 1 window/tensor" = a single base for the whole tensor (no per-row base).
Exponent entropy = order-0 entropy of the 8-bit exponent field (per tensor, weighted by size; "pooled" = one table for the class;
"in-row" = mean of per-row tables). "exp span" = max over rows of (max - min nonzero exponent + 1).

| class | tensors | weights | zeros | subnormal | H(exp) tensor, wtd mean | H(exp) pooled | H(exp) in-row mean | H(mant) | esc A (zero=escape) | esc B (code0=zero) | esc B share | esc C (15 binades+field0) | esc B, 1 window/tensor | esc 11-bit B (8 binades) | exp span max (binades) | rows span<=16 | dup rows | zero rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| embed | 1 | 463159296 | 0 | 0 | 2.5501 | 2.5501 | 2.5453 | 6.9721 | 26887 | 27058 | 5.84e-05 | 53378 | 45417 | 5881633 (1.27%) | 31 | 175844/201024 | 0 | 0 |
| unembed | 1 | 463159296 | 0 | 0 | 2.5455 | 2.5455 | 2.5405 | 6.9715 | 34676 | 34866 | 7.53e-05 | 69006 | 41245 | 5697163 (1.23%) | 36 | 169286/201024 | 0 | 0 |
| wq | 24 | 169869312 | 0 | 0 | 2.5626 | 2.5628 | 2.5571 | 6.9716 | 13408 | 13485 | 7.94e-05 | 26913 | 18508 | 2296541 (1.35%) | 35 | 61497/73728 | 0 | 0 |
| wk | 24 | 56623104 | 0 | 0 | 2.5711 | 2.5715 | 2.5647 | 6.9716 | 4866 | 4903 | 8.66e-05 | 9719 | 7506 | 812450 (1.43%) | 27 | 20182/24576 | 0 | 0 |
| wv | 24 | 56623104 | 0 | 0 | 2.5967 | 2.5969 | 2.5900 | 6.9717 | 4911 | 4929 | 8.70e-05 | 9721 | 7372 | 858508 (1.52%) | 33 | 20158/24576 | 0 | 0 |
| gate_attn | 24 | 169869312 | 0 | 0 | 2.5673 | 2.5673 | 2.5619 | 6.9717 | 13366 | 13438 | 7.91e-05 | 26660 | 17567 | 2323613 (1.37%) | 29 | 61483/73728 | 0 | 0 |
| wo | 24 | 169869312 | 0 | 0 | 2.5875 | 2.5878 | 2.5809 | 6.9718 | 16133 | 16231 | 9.55e-05 | 31858 | 22008 | 2294385 (1.35%) | 31 | 41405/55296 | 0 | 0 |
| wr | 24 | 21233664 | 0 | 0 | 2.5887 | 2.5896 | 2.5772 | 6.9716 | 3300 | 3315 | 1.56e-04 | 6355 | 5595 | 338010 (1.59%) | 27 | 6499/9216 | 0 | 0 |
| sink | 24 | 1327104 | 0 | 0 | 2.5410 | 2.5440 | 2.5317 | 6.9697 | 110 | 112 | 8.44e-05 | 241 | 177 | 17661 (1.33%) | 24 | 486/576 | 0 | 0 |
| router | 24 | 7077888 | 0 | 0 | 2.5713 | 2.5727 | 2.5648 | 6.9713 | 637 | 641 | 9.06e-05 | 1276 | 911 | 102238 (1.44%) | 27 | 2504/3072 | 0 | 0 |
| ctrl_down | 48 | 14155776 | 0 | 0 | 2.6160 | 2.6175 | 2.6131 | 6.9716 | 1740 | 1751 | 1.24e-04 | 3412 | 2013 | 227655 (1.61%) | 30 | 1733/3072 | 0 | 0 |
| ctrl_forget | 48 | 7077888 | 0 | 0 | 2.5899 | 2.6699 | 2.4283 | 6.9713 | 306 | 307 | 4.34e-05 | 641 | 1058 | 81055 (1.15%) | 24 | 110286/110592 | 0 | 0 |
| ctrl_gain | 48 | 7077888 | 0 | 0 | 2.6417 | 2.7644 | 2.4313 | 6.9714 | 319 | 324 | 4.58e-05 | 679 | 1212 | 81937 (1.16%) | 24 | 110273/110592 | 0 | 0 |
| q_l2_clip.scale | 24 | 73728 | 0 | 0 | 1.5076 | 1.8051 | 0.8403 | 6.9187 | 0 | 0 | 0.00e+00 | 0 | 1 | 10 (0.01%) | 13 | 576/576 | 0 | 0 |
| rel_l2_clip.scale | 24 | 9216 | 0 | 0 | 2.2733 | 2.4713 | 1.7134 | 6.7109 | 0 | 0 | 0.00e+00 | 0 | 0 | 17 (0.18%) | 13 | 576/576 | 0 | 0 |
| rel_logits_proj.proj | 24 | 393216 | 0 | 0 | 2.6902 | 2.7467 | 2.0921 | 6.9633 | 24 | 24 | 6.10e-05 | 39 | 45 | 3822 (0.97%) | 20 | 362/384 | 0 | 0 |
| k_sconv.static_w | 4 | 16384 | 0 | 0 | 2.9178 | 3.0300 | 2.6238 | 6.9473 | 3 | 3 | 1.83e-04 | 7 | 5 | 445 (2.72%) | 20 | 13/16 | 0 | 0 |
| k_sconv.weight_proj | 4 | 294912 | 0 | 0 | 2.7122 | 2.7276 | 2.6123 | 6.9711 | 133 | 136 | 4.61e-04 | 236 | 181 | 5582 (1.89%) | 25 | 57/128 | 0 | 0 |
| v_sconv.static_w | 4 | 16384 | 0 | 0 | 2.6552 | 2.8119 | 2.4255 | 6.9282 | 1 | 1 | 6.10e-05 | 2 | 1 | 273 (1.67%) | 22 | 15/16 | 0 | 0 |
| v_sconv.weight_proj | 4 | 294912 | 0 | 0 | 2.8018 | 2.8166 | 2.6533 | 6.9704 | 176 | 177 | 6.00e-04 | 285 | 214 | 7045 (2.39%) | 24 | 48/128 | 0 | 0 |
| **all 2-D BF16** | 426 | 1608221696 | 0 | 0 | | | | | 120996 | 121701 | 7.57e-05 | 240428 | 171036 | 21030043 (1.31%) | | | 0 | 0 |

Rows that have at least one escape (variant B): embed 25,324 of 201,024; unembed 31,896 of 201,024; wq 12,292 of 73,728;
gate_attn 12,309 of 73,728; wo 13,965 of 55,296; wk 4,425; wv 4,433 of 24,576; wr 2,727 of 9,216; router 569 of 3,072;
sink 92 of 576; controllers down 1,345 of 3,072, forget 307, gain 324 of 110,592. Escapes are rare but spread over rows: an escape
path per row (or a per-tile escape list) is needed, not a per-tensor exception.
Best per-row window base (exponent field): embed 113..115, unembed 107..111 (tensor window 114 / 108).

Per layer: H(exp) of the tensor table / esc B count (12-bit, code 0 = zero, per-row window). Controllers: 4 tensors (attn,k,v,mlp) summed, H weighted.

| layer | wq | wk | wv | gate_attn | wo | wr | sink | router | ctrl_down | ctrl_forget | ctrl_gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.550 / 499 | 2.556 / 185 | 2.567 / 176 | 2.553 / 529 | 2.581 / 692 | 2.557 / 68 | 2.599 / 9 | 2.576 / 19 | - | - | - |
| 1 | 2.553 / 509 | 2.559 / 183 | 2.588 / 185 | 2.567 / 579 | 2.593 / 679 | 2.584 / 106 | 2.608 / 12 | 2.576 / 23 | - | - | - |
| 2 | 2.555 / 593 | 2.562 / 163 | 2.596 / 187 | 2.577 / 629 | 2.594 / 713 | 2.583 / 110 | 2.615 / 11 | 2.594 / 25 | - | - | - |
| 3 | 2.552 / 508 | 2.557 / 208 | 2.584 / 197 | 2.570 / 565 | 2.584 / 675 | 2.581 / 109 | 2.581 / 8 | 2.584 / 27 | - | - | - |
| 4 | 2.553 / 582 | 2.558 / 167 | 2.604 / 187 | 2.570 / 541 | 2.595 / 724 | 2.585 / 107 | 2.600 / 12 | 2.571 / 19 | - | - | - |
| 5 | 2.565 / 534 | 2.576 / 196 | 2.587 / 221 | 2.575 / 565 | 2.600 / 717 | 2.571 / 109 | 2.546 / 4 | 2.594 / 25 | - | - | - |
| 6 | 2.561 / 499 | 2.572 / 188 | 2.575 / 212 | 2.564 / 600 | 2.569 / 650 | 2.582 / 129 | 2.559 / 9 | 2.555 / 14 | 2.676 / 182 | 2.611 / 27 | 2.636 / 30 |
| 7 | 2.555 / 556 | 2.566 / 207 | 2.592 / 227 | 2.564 / 580 | 2.589 / 710 | 2.580 / 132 | 2.558 / 3 | 2.557 / 20 | 2.598 / 117 | 2.598 / 32 | 2.642 / 29 |
| 8 | 2.554 / 540 | 2.558 / 175 | 2.590 / 206 | 2.564 / 537 | 2.584 / 699 | 2.587 / 167 | 2.546 / 3 | 2.556 / 17 | 2.588 / 116 | 2.592 / 17 | 2.648 / 26 |
| 9 | 2.553 / 567 | 2.561 / 191 | 2.596 / 233 | 2.565 / 509 | 2.573 / 661 | 2.598 / 168 | 2.529 / 2 | 2.555 / 26 | 2.588 / 136 | 2.590 / 26 | 2.639 / 29 |
| 10 | 2.562 / 576 | 2.567 / 168 | 2.600 / 206 | 2.571 / 603 | 2.586 / 670 | 2.601 / 179 | 2.554 / 5 | 2.564 / 24 | 2.607 / 160 | 2.579 / 27 | 2.615 / 23 |
| 11 | 2.569 / 572 | 2.581 / 206 | 2.604 / 226 | 2.568 / 535 | 2.587 / 686 | 2.588 / 142 | 2.511 / 1 | 2.567 / 24 | 2.614 / 153 | 2.614 / 24 | 2.663 / 31 |
| 12 | 2.563 / 544 | 2.573 / 182 | 2.585 / 197 | 2.564 / 539 | 2.565 / 632 | 2.617 / 181 | 2.565 / 3 | 2.557 / 22 | 2.619 / 154 | 2.576 / 27 | 2.617 / 20 |
| 13 | 2.556 / 546 | 2.564 / 211 | 2.588 / 174 | 2.560 / 522 | 2.571 / 655 | 2.593 / 178 | 2.506 / 2 | 2.557 / 23 | 2.603 / 149 | 2.550 / 25 | 2.655 / 26 |
| 14 | 2.561 / 536 | 2.571 / 189 | 2.593 / 196 | 2.567 / 535 | 2.573 / 600 | 2.596 / 165 | 2.522 / 4 | 2.569 / 29 | 2.609 / 160 | 2.560 / 18 | 2.640 / 23 |
| 15 | 2.557 / 582 | 2.564 / 214 | 2.602 / 206 | 2.566 / 537 | 2.584 / 695 | 2.594 / 170 | 2.511 / 5 | 2.575 / 33 | 2.624 / 146 | 2.579 / 35 | 2.632 / 28 |
| 16 | 2.567 / 576 | 2.567 / 182 | 2.610 / 218 | 2.577 / 571 | 2.610 / 753 | 2.595 / 153 | 2.549 / 4 | 2.591 / 50 | 2.630 / 135 | 2.632 / 21 | 2.651 / 31 |
| 17 | 2.577 / 590 | 2.586 / 218 | 2.599 / 241 | 2.573 / 591 | 2.603 / 689 | 2.574 / 109 | 2.460 / 1 | 2.582 / 30 | 2.635 / 143 | 2.599 / 28 | 2.661 / 28 |
| 18 | 2.567 / 612 | 2.581 / 248 | 2.615 / 211 | 2.566 / 525 | 2.582 / 611 | 2.601 / 145 | 2.508 / 2 | 2.597 / 44 | - | - | - |
| 19 | 2.564 / 564 | 2.574 / 218 | 2.631 / 230 | 2.564 / 554 | 2.582 / 650 | 2.600 / 151 | 2.502 / 1 | 2.600 / 59 | - | - | - |
| 20 | 2.564 / 578 | 2.576 / 249 | 2.597 / 190 | 2.565 / 613 | 2.582 / 672 | 2.597 / 136 | 2.506 / 3 | 2.557 / 12 | - | - | - |
| 21 | 2.568 / 627 | 2.580 / 228 | 2.597 / 191 | 2.565 / 549 | 2.597 / 691 | 2.598 / 164 | 2.526 / 1 | 2.558 / 25 | - | - | - |
| 22 | 2.570 / 568 | 2.585 / 229 | 2.614 / 191 | 2.567 / 575 | 2.583 / 620 | 2.586 / 133 | 2.532 / 2 | 2.560 / 32 | - | - | - |
| 23 | 2.607 / 627 | 2.615 / 298 | 2.607 / 221 | 2.575 / 555 | 2.633 / 687 | 2.580 / 104 | 2.491 / 5 | 2.556 / 19 | - | - | - |

What it allows: a 12-bit BF16 format (4-bit offset, per-row or even per-tensor base) is exact with 7.5e-5 escapes (1.1e-4 per tensor
window); for the B=1 step this is ~25% of the BF16 weight bytes (attention ~2.6 GB/step, unembedding 0.93 GB, controllers),
roughly 0.9 GB of the 5.54 GB step. Verify the packer bit for bit on all weights before claiming it.
What it does not allow: 11-bit (8 binades) has 1.31% escapes overall; the mantissa is full entropy (6.97 of 7 bits); exponent
entropy is ~2.55-2.6 bits, so an entropy code could reach ~10.6 bits but not at fixed width; no zero rows/columns, no duplicate rows,
no identical tensors to share.

## 3. NVFP4 experts (layers 0-19, w13 [128,2304,1152] U8 and w2 [128,2304,576] U8)

- Blocks: w13 42,467,328 per layer, w2 21,233,664; total 1,274,019,840 blocks = 20,384,317,440 codes.
- **All-zero 16-blocks: 0** (neither all code 0 nor all +-0) in every tensor.
- **Per-block max |code| is 6.0 (code 7/15) in every one of the 1,274,019,840 blocks.**
- E2M1 code histogram, pooled over all 40 tensors (code 0..15 = +0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, ... -6):
  703,647,091 / 1,381,153,139 / 1,306,536,436 / 1,192,551,510 / 1,538,767,057 / 1,540,927,481 / 1,386,206,747 / 1,143,017,501 /
  703,609,090 / 1,381,072,248 / 1,306,393,556 / 1,192,489,942 / 1,538,713,884 / 1,540,691,982 / 1,385,957,241 / 1,142,582,535.
  Entropy 3.9683 bits (per tensor 3.9672..3.9727); magnitude entropy 2.968 bits, sign exactly 1 bit. -0 (code 8) is as frequent as
  +0 (3.45% each): the sign of zero is kept. In exact arithmetic +-0 are the same value (a track-B packer may merge them), but that frees
  one code of 16 and no bit.
- Scales (E4M3, one per 16 inputs): 0 zero, 0 subnormal, 0 negative, 0 NaN in all 40 tensors. Every expert of every tensor has its
  largest scale exactly 0x7e = 448 (128 of 128 experts, all 40 tensors). Smallest scale overall 0x4c = 6.0 (layers 15 w13, 17 w2).
  Exponent-field histogram of all scales (fields 9..15; values 4..7.5, 8..15, 16..30, 32..60, 64..120, 128..240, 256..448):
  38 / 54,255 / 11,772,887 / 185,832,455 / 572,109,613 / 490,099,358 / 14,151,234 (fields 0..8: 0).
  Scale byte entropy 3.81..4.50 bits per tensor.
- amax (F32 [128] per tensor): no zeros; not BF16-representable (0 of 5,120). Denominator 2688.0 in all 40 tensors (= 6 x 448), so
  the per-expert factor is amax/2688 and the largest |dequantized weight| of each expert equals its amax.

| layer | w13 scale distinct | w13 scale byte range | w13 scale H | w13 amax min..max | w2 scale distinct | w2 scale byte range | w2 scale H | w2 amax min..max | E2M1 H w13 / w2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 39 | 0x58..0x7e (16..448) | 3.902 | 0.1053..0.3311 | 39 | 0x56..0x7e (14..448) | 3.996 | 0.1522..0.4441 | 3.9680 / 3.9672 |
| 1 | 43 | 0x52..0x7e (10..448) | 4.222 | 0.1064..0.4135 | 37 | 0x5a..0x7e (20..448) | 4.000 | 0.1466..0.4345 | 3.9683 / 3.9675 |
| 2 | 43 | 0x54..0x7e (12..448) | 4.328 | 0.1054..0.4509 | 38 | 0x59..0x7e (18..448) | 4.109 | 0.1442..0.4687 | 3.9689 / 3.9675 |
| 3 | 48 | 0x4f..0x7e (7.5..448) | 4.425 | 0.1083..0.6996 | 38 | 0x59..0x7e (18..448) | 3.982 | 0.1454..0.4124 | 3.9689 / 3.9675 |
| 4 | 41 | 0x56..0x7e (14..448) | 4.107 | 0.1063..0.4206 | 41 | 0x56..0x7e (14..448) | 4.224 | 0.1495..0.5871 | 3.9686 / 3.9675 |
| 5 | 46 | 0x51..0x7e (9..448) | 4.495 | 0.1099..0.5308 | 45 | 0x52..0x7e (10..448) | 4.298 | 0.1504..0.7355 | 3.9684 / 3.9679 |
| 6 | 41 | 0x56..0x7e (14..448) | 4.074 | 0.1039..0.4018 | 40 | 0x55..0x7e (13..448) | 4.145 | 0.1438..0.4775 | 3.9676 / 3.9675 |
| 7 | 36 | 0x5b..0x7e (22..448) | 3.814 | 0.1044..0.2440 | 40 | 0x57..0x7e (15..448) | 4.145 | 0.1443..0.4623 | 3.9680 / 3.9674 |
| 8 | 37 | 0x5a..0x7e (20..448) | 3.876 | 0.1050..0.2855 | 40 | 0x57..0x7e (15..448) | 4.015 | 0.1455..0.5297 | 3.9683 / 3.9675 |
| 9 | 38 | 0x59..0x7e (18..448) | 3.938 | 0.1054..0.2918 | 41 | 0x54..0x7e (12..448) | 4.226 | 0.1458..0.5977 | 3.9682 / 3.9677 |
| 10 | 45 | 0x52..0x7e (10..448) | 4.393 | 0.1029..0.4876 | 43 | 0x54..0x7e (12..448) | 4.313 | 0.1552..0.6793 | 3.9682 / 3.9683 |
| 11 | 46 | 0x51..0x7e (9..448) | 4.431 | 0.1076..0.5700 | 46 | 0x51..0x7e (9..448) | 4.218 | 0.1746..0.7591 | 3.9681 / 3.9694 |
| 12 | 45 | 0x51..0x7e (9..448) | 4.423 | 0.1077..0.5605 | 40 | 0x56..0x7e (14..448) | 4.150 | 0.1510..0.4947 | 3.9676 / 3.9683 |
| 13 | 45 | 0x52..0x7e (10..448) | 4.392 | 0.1058..0.4572 | 42 | 0x55..0x7e (13..448) | 4.212 | 0.1583..0.5612 | 3.9677 / 3.9684 |
| 14 | 48 | 0x4f..0x7e (7.5..448) | 4.500 | 0.1055..0.6745 | 43 | 0x54..0x7e (12..448) | 4.199 | 0.1613..0.5680 | 3.9679 / 3.9691 |
| 15 | 47 | 0x4c..0x7e (6..448) | 4.462 | 0.1118..0.6327 | 44 | 0x53..0x7e (11..448) | 4.274 | 0.1702..0.6633 | 3.9680 / 3.9700 |
| 16 | 47 | 0x50..0x7e (8..448) | 4.295 | 0.1234..0.7156 | 45 | 0x52..0x7e (10..448) | 4.295 | 0.1543..0.7151 | 3.9686 / 3.9716 |
| 17 | 50 | 0x4d..0x7e (6.5..448) | 4.287 | 0.1274..0.7359 | 50 | 0x4c..0x7e (6..448) | 4.327 | 0.2038..0.8521 | 3.9685 / 3.9727 |
| 18 | 46 | 0x51..0x7e (9..448) | 4.477 | 0.1039..0.6119 | 44 | 0x53..0x7e (11..448) | 4.450 | 0.1493..0.7524 | 3.9676 / 3.9682 |
| 19 | 41 | 0x56..0x7e (14..448) | 4.151 | 0.1014..0.4002 | 44 | 0x53..0x7e (11..448) | 4.341 | 0.1526..0.6432 | 3.9674 / 3.9681 |

What it allows: a fixed 6-bit scale code (byte - 0x40: observed 0x4c..0x7e -> 12..62) is lossless for every NVFP4 scale of the
checkpoint (-2 bits per 16 weights, 2.8% of NVFP4 bytes; at B=1 ~44 MB of the ~1.58 GB of NVFP4 expert bytes per step); a per-tensor
table of <= 50 entries also fits 6 bits. The denominator is a constant.
What it does not allow: compressing the E2M1 codes (3.968 bits of 4); skipping blocks (none is zero); a 5-bit scale code without a
table and escapes; storing amax in BF16.

## 4. MXFP8 experts (layers 20-23, w13 [128,2304,2304] / w2 [128,2304,1152] F8_E4M3, UE8M0 scale per 32 inputs)

| tensor | distinct UE8M0 scales (value: count) | rows with >1 distinct scale (of 294,912) | per-row range (binades: rows) | E4M3 byte entropy | exponent-field entropy | subnormals | zeros (+0 / -0) | NaN |
|---|---|---|---|---|---|---|---|---|
| L20.w13 | 5: 113 (2^-14): 46107, 114 (2^-13): 17080897, 115 (2^-12): 4105612, 116 (2^-11): 1028, 117 (2^-10): 20 | 294912 | 1: 251476, 2: 43261, 3: 161, 4: 14 | 6.5196 | 2.5890 | 46933 | 6243 (3188 / 3055) | 0 |
| L20.w2 | 6: 113 (2^-14): 24, 114 (2^-13): 1727970, 115 (2^-12): 8826746, 116 (2^-11): 61221, 117 (2^-10): 814, 118 (2^-9): 57 | 294726 | 0: 186, 1: 242290, 2: 51755, 3: 622, 4: 58, 5: 1 | 6.5321 | 2.5939 | 26955 | 2715 (1382 / 1333) | 0 |
| L21.w13 | 5: 113 (2^-14): 45654, 114 (2^-13): 17069870, 115 (2^-12): 4116848, 116 (2^-11): 1257, 117 (2^-10): 35 | 294912 | 1: 251800, 2: 42855, 3: 240, 4: 17 | 6.5198 | 2.5893 | 46460 | 6299 (3160 / 3139) | 0 |
| L21.w2 | 6: 113 (2^-14): 31, 114 (2^-13): 1720555, 115 (2^-12): 8824333, 116 (2^-11): 71022, 117 (2^-10): 858, 118 (2^-9): 33 | 294720 | 0: 192, 1: 235894, 2: 58078, 3: 713, 4: 35 | 6.5341 | 2.5958 | 26967 | 2856 (1412 / 1444) | 0 |
| L22.w13 | 6: 113 (2^-14): 46381, 114 (2^-13): 17045310, 115 (2^-12): 4138864, 116 (2^-11): 2976, 117 (2^-10): 130, 118 (2^-9): 3 | 294912 | 1: 250346, 2: 44035, 3: 468, 4: 61, 5: 2 | 6.5205 | 2.5899 | 46852 | 6303 (3119 / 3184) | 0 |
| L22.w2 | 6: 113 (2^-14): 38, 114 (2^-13): 1711123, 115 (2^-12): 8819875, 116 (2^-11): 84642, 117 (2^-10): 1108, 118 (2^-9): 46 | 294702 | 0: 210, 1: 227810, 2: 65985, 3: 858, 4: 48, 5: 1 | 6.5369 | 2.5984 | 26821 | 2867 (1386 / 1481) | 0 |
| L23.w13 | 6: 113 (2^-14): 46495, 114 (2^-13): 17015428, 115 (2^-12): 4167109, 116 (2^-11): 4451, 117 (2^-10): 180, 118 (2^-9): 1 | 294912 | 1: 249668, 2: 44519, 3: 642, 4: 82, 5: 1 | 6.5217 | 2.5909 | 46849 | 6238 (3063 / 3175) | 0 |
| L23.w2 | 6: 113 (2^-14): 83, 114 (2^-13): 1673301, 115 (2^-12): 8834329, 116 (2^-11): 107326, 117 (2^-10): 1660, 118 (2^-9): 133 | 294689 | 0: 223, 1: 211629, 2: 81903, 3: 996, 4: 145, 5: 16 | 6.5451 | 2.6062 | 27333 | 3144 (1550 / 1594) | 0 |

- The scale is **not constant along a row**: every w13 row and all but 186-223 w2 rows have 2+ distinct scales; also no expert and no
  column (expert, k-block) has a constant scale (128/128 experts, 9216/9216 resp. 4608/4608 columns vary).
- Every 32-block's largest magnitude byte is 0x77 (= 240, the only field-14 value that occurs as a block max) or in 0x78..0x7e
  (256..448): w13 ~1.77M of 21.2M blocks at 0x77, the rest field 15. No all-zero 32-block.
- 254 distinct weight bytes (every byte except NaN 0x7f / 0xff). Mantissa entropy 2.98 of 3 bits.
- A 7-bit code (sign + exponent fields 8..15 + 3 mantissa) would have 1.08% (w13) / 1.19-1.22% (w2) escapes.

What it allows: a fixed 3-bit scale code (value - 113, 6 values) is exact for all 8 tensors (-5/32 bit per weight; ~5 MB/step at B=1,
negligible). Weights: only variable-length entropy coding (~6.53 of 8 bits), not a fixed width.
What it does not allow: a per-row (or per-32x32-tile) scale; any fixed width below 8 bits for the weights without >1% escapes.

## 5. Norm weights (*_norm.weight, BF16 [2304])

No norm weight has a zero. Several have **negative entries** (count per layer below; embed_norm 2), and some are tiny
(min |w| 9.58e-5 in layers.7.mlp_norm, 4.08e-4 in layers.6.mlp_norm, 8.39e-4 in layers.23.v_input_norm).
embed_norm: max 11.3125, min |w| 0.1377, 2 negative, mean 0.756. Final norm: max 19.625, min 0.0282, mean 5.51, 0 negative.

| layer | attn_norm max / min |w| | mlp_norm max / min | q_input_norm | k_input_norm | v_input_norm | negatives (all norms of the layer) |
|---|---|---|---|---|---|---|
| 0 | 6.875 / 0.122 | 2.01562 / 0.00403 | - | - | - | 2 |
| 1 | 5.53125 / 0.334 | 2.90625 / 0.00443 | - | - | - | 2 |
| 2 | 7.15625 / 0.285 | 3.84375 / 0.00244 | - | - | - | 3 |
| 3 | 7.34375 / 0.25 | 3.96875 / 0.00232 | - | - | - | 1 |
| 4 | 5.8125 / 0.135 | 3.875 / 0.002 | - | - | - | 2 |
| 5 | - | 5.125 / 0.017 | 7.15625 / 0.171 | 6.0625 / 0.0272 | 4.25 / 0.0198 | 1 |
| 6 | 4.46875 / 0.062 | 2.15625 / 0.000408 | - | - | - | 5 |
| 7 | 4.1875 / 0.142 | 2.14062 / 9.58e-05 | - | - | - | 4 |
| 8 | 3.46875 / 0.189 | 1.89062 / 0.000984 | - | - | - | 1 |
| 9 | 3.20312 / 0.167 | 1.85938 / 0.00142 | - | - | - | 1 |
| 10 | 3.625 / 0.227 | 2.1875 / 0.0129 | - | - | - | 0 |
| 11 | - | 2.40625 / 0.0131 | 3.21875 / 0.264 | 4.28125 / 0.0195 | 5 / 0.00114 | 0 |
| 12 | 2.98438 / 0.14 | 2.25 / 0.0119 | - | - | - | 0 |
| 13 | 4.34375 / 0.168 | 2.42188 / 0.0248 | - | - | - | 0 |
| 14 | 4.25 / 0.114 | 2.65625 / 0.0378 | - | - | - | 0 |
| 15 | 3.9375 / 0.185 | 3.1875 / 0.0311 | - | - | - | 0 |
| 16 | 3.67188 / 0.19 | 4.78125 / 0.0605 | - | - | - | 0 |
| 17 | - | 4 / 0.0618 | 3.90625 / 0.196 | 5.75 / 0.0249 | 3.29688 / 0.00129 | 3 |
| 18 | 3.09375 / 0.138 | 2.73438 / 0.0542 | - | - | - | 0 |
| 19 | 2.25 / 0.216 | 2.625 / 0.0562 | - | - | - | 0 |
| 20 | 3.1875 / 0.102 | 1.67188 / 0.063 | - | - | - | 0 |
| 21 | 4.34375 / 0.0859 | 1.73438 / 0.108 | - | - | - | 0 |
| 22 | 5.5 / 0.117 | 1.6875 / 0.143 | - | - | - | 0 |
| 23 | - | 1.89844 / 0.208 | 4.5 / 0.00757 | 7.1875 / 0.0117 | 13.5 / 0.000839 | 11 |

What it allows: nothing for storage (4.6 KB each). Folding a norm gain into the next BF16 matrix is NOT lossless (the product
must be rounded to BF16; tiny and negative gains make it worse); keep the gain as a separate multiply.

## 6. F32 tensors

- attn_res.attn_queries.0..23 ([2304]; [3,2304] at indices 4, 10, 16, 22, the global layers' 3-query attention inputs), attn_res.mlp_queries.0..23,
  attn_res.output_query: no zeros, no NaN, all values distinct
  within each vector (attn_queries.22 has one repeated value), no duplicate rows in the [3,2304] ones, L2 norm 8.7..51, max |q| 1.6..13.5.
  0 values BF16-representable: they must stay FP32.
- middle_cycle_cross_pass_queries / _kv_queries [24, 2304]: no zeros, no zero rows, no duplicate rows, row L2 17..68,
  max |q| 12.7 / 12.4; 2 resp. 0 of 55,296 values BF16-exact. The two tensors are different.
- gate.bias [128] x 24: no zeros, max |b| 0.065 (layer 0) .. 0.977 (layer 22), not BF16-representable.
- No two F32 tensors are identical except the 40 NVFP4 `outer_scale_denominator` scalars (all 2688.0).
- Nothing notable for storage (all F32 tensors together are < 1 MB).
