# Weights study (operator notes 21:30 / 21:35): what holds exactly in Inkling-2-Mini's weights

All counts are exact (every weight, integer counts on the GPU; tests/expert_census.py, e0207, full per-layer JSON in
execs/e0207/out/expert_census.jsonl). BF16 matrices: tools/kit/weights_census.py (e0102, /tmp/study1.txt in the
journal). Router statistics: tests/routing_study.py (e0102), tests/routing_study2.py (below).

## 1. NVFP4 experts (layers 0-19; 20.4e9 weights: w13 [128][2304][2304], w2 [128][2304][1152] per layer)
Stored: 4-bit E2M1 code per weight, one E4M3 scale per 16 weights along K (0.5 bit/weight), one FP32 factor per
expert (amax / 2688). 4.5 bits/weight; the engine reads 576 B per 16 x 64 unit (512 codes + 64 scales).

| property | measured | consequence |
|---|---|---|
| code entropy (order 0) | 3.968 bits of 4 (w13 and w2, every layer 3.967-3.973) | no entropy to take; codes are near uniform |
| entropy of code pairs (bytes) | 3.967-3.972 bits/weight | no pair structure either |
| -0 code (8) | 3.45% of weights, as often as +0 (3.45%) | the sign of zero is unused information: 0.069 bit/weight, not extractable at a fixed rate |
| block maximum | **every one of the 1.274e9 blocks contains a +-6 (code 7): 100.000%** | exact: scale = block max / 6; no zero block, no saturated block, no constant block |
| zero rows / columns / experts | 0 / 0 / 0 | nothing to skip |
| duplicate experts / rows / scale rows | 0 of 2560 experts, 0 of 5.9e6 rows (w13), 0 (w2) | nothing to share |
| distinct 64-bit blocks | 100.00% distinct | no dictionary |
| numerical rank (6 experts x w13, w2) | 99% of the energy needs 1693-1740 of 2304 (w13), 1045-1063 of 1152 (w2) | not low rank (and low rank would not be exact) |
| block scales: distinct E4M3 values | 36-50 per tensor, 51 over all | 8-bit field carries 3.8-4.5 bits (order 0), **3.30-3.47 bits within a row** |
| block scales: exponent window per row | all scales of a row within 4 consecutive exponents except 0-6e-5 of them (2 exponents: 1.7-4.1% outside) | **exact 5-bit scale = E4M3 byte minus a per-row base (2-bit exponent offset + 3-bit mantissa), escapes <= 6e-5** |
| scale equal to its neighbour | printed per layer in the JSON (no run structure) | no run-length code |
| expert factor amax / 2688 | 128 distinct per tensor, none a power of two, none exact in BF16 | stays FP32 (128 values) |

What it is worth: 5-bit scales take 576 -> 552 B per unit, -4.2% of the NVFP4 expert bytes: 1.58 GB -> 1.51 GB per
step at B=1 (-66 MB = 1.2% of the 5.54 GB step). The entropy floor of the format (3.97 + 3.4 / 16) is 4.18
bits/weight against 4.5 stored: -7% is all there is, exactly, in the NVFP4 tensors.

## 2. MXFP8 experts (layers 20-23; 4.08e9 weights)
Stored: E4M3 code per weight, one UE8M0 scale per 32 weights. 8.25 bits/weight.

| property | measured | consequence |
|---|---|---|
| code entropy | 6.52-6.55 bits of 8 (exponent 2.59, mantissa 2.98, sign 1.0); 254 of 256 codes used, no NaN; -0 occurs (1333-3184 per tensor) | 18% entropy headroom, but only through a variable-length code |
| block max exponent | always 14 or 15 (of 0..15) | the power-of-two scale pins the top of every block |
| exponent offset below the block max | geometric: 13.3 / 30.0 / 25.6 / 15.2 / 7.9 / 4.0 / 2.0 / 1.0 % ... | fixed-rate windows: 7-bit code (8 exponents) **1.0-1.2% escapes**, 6-bit 16-18%, 5-bit 57-62% |
| scales | 5-6 distinct values per tensor, entropy 0.70-0.74 bits of 8; range within a row <= 2 for 99.94% of rows | exact 2-bit scale per block: 8.25 -> 8.06 bits/weight (-2.3%) |
| zero blocks / rows / columns, duplicates | 0 | |

A 7-bit code would save 12.5% of 4 of 48 applications' expert bytes (66 MB -> 58 MB each) at ~370 escapes per
16-row strip (the BF16 12-bit code has <= 64): not a fixed-rate format a kernel can patch cheaply. Not pursued.

## 3. FP32 groups
154 tensors, 247848 values (1 MB): 42 values exact in BF16, 66 exact in FP16. They stay FP32; irrelevant for bytes.

## 4. BF16 matrices (earlier: e0102)
12-bit exact code (sign + 4-bit exponent offset in a per-row window + 7 mantissa bits): 7.6e-5 escapes over 2.24e9
weights; 11-bit 1.3% escapes. No zero rows, no identical tensors. Value entropy 10.47 bits.
Engine results with it: unembedding (W12U) bit-identical, +-0 us end to end; qkv (W12Q) bit-identical, +102 us B=1 /
+176 us B=8 (the decode keeps fewer bytes in flight than plain BF16 loads; the kernel is at its register limit).

Scale details (NVFP4, all 40 tensors): only codes 76..126 occur (51 consecutive byte values, exponents 9..15; one
tensor's range is at most 51 wide, layer 0 w13 uses 88..126; sign never set, no zero scale); mantissa entropy 2.96-2.98 of 3 bits, exponent entropy 1.06-1.64 bits; a scale equals its
K neighbour in 9.4-10.9% of the cases and its row neighbour in 9.6-11.0% (no runs); 10-30 distinct values per row
(mean 14.6), 26-36 per expert. Fixed-rate options, both exact:
 - 6 bits = byte - 76, one global base, **zero escapes**: 576 -> 560 B per unit (-2.8%);
 - 5 bits with one global base (91): 9.0e-4 of the blocks outside;
 - 5 bits = byte - per-row base: 9815 escapes in 1.274e9 blocks (7.7e-6): 576 -> 552 B (-4.2%), needs an escape path.

## 5. Router statistics (tests/routing_study2.py, e0208: reference model, 8 different natural-text sequences x 512)
| question | measured |
|---|---|
| popularity per application (4096 tokens) | top-1 expert 5.7% of the selections (2.1-12.5%), top-8 28% (15-74%), top-32 63%; effective number of experts 2^H = 77 of 128 (19-105); 3.6 experts never used on average. Only layer 17 in passes 1-2 is skewed (top-8 = 74% / 65%, effective 19 / 25) |
| hot experts resident in L2 | the best 18 experts of the whole model (84.7 MB, the card's maximum persisting L2) cover 6.1% of the selections = 5.7% of a step's expert bytes; 129 MB (all of L2): 8.0%; 500 MB: 21%. And pinning 72 MB of L2 (L2P experiment, small weights) cost +9 us B=1 / +115 us B=8 because the streaming phases lose that L2. **Not worth it.** |
| the three passes re-reading layers 6-17 | a loop application shares 57% of its 8 experts with the previous pass of the layer at B=1 (68% of the union at B=8), but 430 MB (B=1) / 2205 MB (B=8) of expert bytes are read between the two uses; L2 is 129 MB, all shared memory 34.5 MB. XPF (prefetching the previous pass's experts) measured +246 us B=1 / +1339 us B=8. **Cannot be served from a cache on this card.** |
| previous token | 62% of an application's B=8 union was used at the previous token (B=1: 3.3 of 8): 13.7 GB are read in between |
| batch 8: distinct experts per application | **union of the 8 sequences: mean 43.3 experts (p10 37, p50 44, p90 49; 64 = all apart, 8 = all alike)**; 68.5% of the union's experts serve 1 token, 21.6% 2, 6.3% 3, 1.9% 4, 1.66% more than 4 |
| batch 8: expert bytes per step | **9.97 GB** (B=1: 1.84 GB; all apart 14.7 GB): 208 MB per application; the whole B=8 step reads ~13.7 GB: 1.7 ms at 8 TB/s |

Does the engine load each distinct expert once per step for all its tokens? Yes (a2 base, checked in the source):
routing() builds the union (sm.uexp / uncnt / utok); ph_w13's items are (union expert, 16-row strip) with the
expert's tokens as MMA columns (one column group up to 4 tokens, two above); ph_w2a's items are (union expert,
output strip), the tokens' activations staged once per block. The evaluator's B=8 client sends a different prompt
per stream ($MK_HARNESS/eval/client.py), so the union above is the realistic case.
Where the B=8 time goes (mkprof e0183): w13 + w2 spans 89 us per application for 208 MB = 2.3 TB/s (latency-bound
rounds of 8 items per block, 4 units in flight per warp), i.e. ~4.3 ms of the 7.4 ms step. At 5 TB/s the step
would be ~5.2 ms. The batch-8 gap is the expert *streaming rate*, not redundant bytes.

## 6. The architecture's properties (21:35 note) and what the engine does with each
| property | status | measured |
|---|---|---|
| three passes with shared weights, rank-64 message gate | MSGZ (track B): the messages and the message half of all 48 gate down-projections of a pass computed once at the pass start (the 2 message sources are the same for all 12 layers of a pass) | -47 us B=1 / -153 us B=8 (within gates: FP32 sum split) |
| shared weights across passes as cached bytes | not possible (430 MB between uses), see above | XPF +246 / +1339 |
| layer 0 sees only the token embedding | L0T (track B): its q/k/v/gate rows from a table per token id built at load | -3 / -32 us, bit-identical |
| vocabulary padded to 201024, 200058 real rows | the head reads 12504 strips = 200064 rows and returns the 200058 real logits (base engine) | 0.4% of the head |
| final softcap 30 (monotone) | SKH (track B): certified int8 sketch + exact rescoring; candidates compared on the capped values; tanh only for candidates on the greedy path | -45 / -32 us, argmax identical |
| sliding window 1024, d_rel 16, log scaling | window bounds K/V per local layer at 1024 positions; tau = 1 exactly below 8192 positions (multiplication is the identity); the 16 x 1024 relative table is read per 64-key record | nothing to skip found |
| attention residuals in blocks of 6 + rank-64 gate | **GLIN (track B, batch-1 kernel)**: the mix is linear in its <= 5 published sources, so the gate's down-projection is applied to the raw sources as MMA columns (plus the 2 message sources) in parallel with the mix and combined with the mix coefficients afterwards; the unit vector is no longer rounded to BF16 first | **-61..-69 us at B=1** (s0037: 3073 -> 3005), within the gates (reference rel L2 median 0.033 / 0.058 / 0.024 vs 0.035 / 0.064 / 0.022); +276 us in the B >= 2 kernel (MSGZ is better there); the MLP-input gate (8 columns) -40 vs -69: off |
| the same linearity for the attention input | QLIN: W_qkv (w . x) r = sum_s (cf_s r) W_qkv (w . v_s), q/k/v strips on 5 source columns while one block mixes | numerically fine (0.0325 / 0.0495 / 0.0271), +58 us at B=1: the strips were waiting for the newest source, not for the mix. Off |
| 128 experts, top 8, softcaps, route_scale 8 | no exactly-zero activations (smooth gate), no skip | - |
| per-pass K/V and conv caches | inherent state | - |

## 7b. Head sketch precision (offline, sketch4_study.py)
int4 / int5 / int6 per-row sketches leave median 198,924 / 2,513 / 92 of 200,058 rows unexcluded; an int6 first
level with int8 on the surviving strips would read 346 + 2.6 MB per token instead of 461 MB (-25%, ~19 us at B=1).
Not built.

## 8. What the weights and routing study led to in the engine (summary at 10 h)
Nothing in the stored weights is redundant enough to pay for a decode in this kernel (S6, W12Q, W12U measured).
The model-specific wins all came from the architecture's algebra and from where work can be hoisted:
MSGZ (per-pass message half of the gates, B >= 2 kernel), SKH (softcap-aware certified sketch head), L0T (layer 0
depends on the token id only, B >= 2 kernel), GLIN (linearised gate, batch-1 kernel). Track B against track A at
the end (s0047 / s0049 vs s0048, same kernel organisation): -70 us at B=1 (2879 vs 2949) and -221 us at B=8
(6509 vs 6730).

## Conclusion
The weights are dense and close to maximum entropy in their stored formats: the exact savings available are
-2.8% (6-bit NVFP4 scales, no escapes) or -4.2% (5-bit, rare escapes) of the NVFP4 expert bytes, -2.3% (2-bit
MXFP8 scales), and the 12-bit BF16 code (-25% of the BF16 matrices, measured slower in this latency-bound kernel).
There are no zero, duplicate, saturated or low-rank experts, no hot expert set, and no reuse between passes that
fits a cache. Batch 8 reads 10 GB of distinct expert bytes per step, each once; its gap to vLLM is the rate at
which the expert rounds stream them (2.3 TB/s).

## 7. Engine measurements of the exact formats found here
| format | bytes | exactness check | time (same-card A/B) |
|---|---|---|---|
| S6: 6-bit NVFP4 block scales (byte - 76) | -2.8% of NVFP4 expert bytes (576 -> 560 B per unit) | round trip on all 1.274e9 scales at load; logits bitwise equal | **+149 us B=1, +469 us B=8** (e0214) |
| W12Q: 12-bit BF16 qkv strips | -25% of qkv bytes | round trip on all qkv weights; logits bitwise equal | +102 us B=1, +176 us B=8 (e0200) |
| W12U: 12-bit BF16 unembedding | -25% of 921 MB | round trip; bitwise equal | +20 / -29 us (s0008, noise) |
Why they do not pay here: in this engine no weight phase runs at the card's bandwidth (qkv 5, experts 3, o_proj
1.8 TB/s against ~6.5 measured for a pure stream); the time of a phase is (bytes in flight per warp, limited by
its registers) x (load latency), plus hand-offs. A format that needs more instructions or more registers per byte
lengthens the chain by more than the bytes it removes. The exact formats become worth their decode only once the
expert rounds stream at bandwidth.

## 9. Near-duplicate experts (tests/expert_similarity.py, e0272; dequantized w13 / w2, all 128 x 128 pairs)
| layer | w13 max cos (pair) | w2 max cos | closest pair: same code bytes | cos to the layer's mean expert (max / median) |
|---|---|---|---|---|
| 0 | 0.0020 | 0.0023 | 0.43% | 0.093 / 0.091 |
| 6 | 0.0024 | 0.0024 | 0.44% | 0.096 / 0.091 |
| 12 | 0.0027 | 0.0024 | 0.43% | 0.096 / 0.092 |
| 17 | 0.0122 | 0.0023 | 0.43% | 0.112 / 0.099 |
| 19 | 0.0054 | 0.0025 | 0.43% | 0.106 / 0.097 |
| 21 (MXFP8) | 0.0035 | 0.0024 | 1.35% | 0.100 / 0.095 |
The experts are as good as orthogonal (median pair cosine ~0.0005); the closest pair shares code bytes at chance
(1/256 = 0.39% for FP4 bytes); the cosine to the mean expert is ~1/sqrt(128) = 0.088, i.e. no shared component
(a "mean expert + residual" factoring would save nothing). No near duplicates to exploit, exactly or otherwise.
