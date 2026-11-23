# Where the BF16 path's ~0.025 error comes from (Inkling-2-Mini, M5 Max, 2026-11-23)

The reference is `eval/inkling2_ref.py` mode high in float32 on the CPU
(`~/mk-ref/inkling-2-mini-m5max-fp32`). The question was why every BF16 engine sits about 0.025 (median rel-L2 of the
logits) from it: the repository's BF16 path (mode fork), our track-C engine (0.024), and mlx-lm (0.026). mlx-lm even
stayed at that level after `model.set_dtype(mx.float32)`.

All numbers are on the reference's first sequence: 128 check positions plus every 8th position up to 1,152, 256 in
all.

## 1. Most of the error is expert-routing flips

| configuration (against high, float32) | rel-L2 median | p99 | top-1 | tokens with a route flip, per layer |
|---|---|---|---|---|
| fork (the BF16 path) | 0.0272 | 0.318 | 0.969 | 39% |
| fork with high's expert choices forced | **0.0045** | 0.027 | 0.996 | 0 |

- **What a flip is.** Inkling-2-Mini picks 8 experts per token. The 8th and 9th scores are often within BF16 noise of
  each other, so in 39% of (token, layer) pairs at least one choice differs from float32.
- **Forcing the routes removes most of the error.** With float32's routes forced, the error falls 6x: 0.027 to 0.0045
  median, and 0.32 to 0.027 at p99.
- **The rest is continuous rounding, about 0.0045.**
- **Why the error grows with position.** It is larger at long positions (median 0.033, against 0.025 for the check
  positions). A flipped choice changes the K/V history that every later position attends to.

## 2. The continuous 0.0045, by family

Each row rounds one family as the engines do, keeps everything else in float32, and forces high's routes
(`scripts/mac_error_attribution.py`, results in `attribution.json`).

| only this family in BF16 | median | p99 |
|---|---|---|
| attention (projections, k/v convs, clips, attention, output proj) | 0.0024 | 0.029 |
| attention residuals (depth mix and partial sums) | 0.0023 | 0.006 |
| MoE (router input and experts) | 0.0023 | 0.010 |
| RMS norms before attention and MoE | 0.0014 | 0.007 |
| short convolutions after attention and MoE | 0.0013 | 0.005 |
| loop message gates | 0.0005 | 0.002 |
| unembedding | 0 | 0 |
| all families together | 0.0043 | 0.020 |

The shares add in quadrature: the square root of the sum of squares is 0.0045, against 0.0043 measured with all
families together. No single family dominates.

**Consequence.** The numeric gate allows 2x the BF16 path's error (median 0.049, p99 0.121). That limit, and the 0.10
p99 our engine lands at, are set by routing near-ties, not by any one rounding point. This is why one build in six
can land on either side of the p99 limit: the flips are effectively random. It also explains why float32
activations cut the error only 3-4x and not more: the routes still flip, just less often.

## 3. mlx-lm's "float32" is not float32 on this GPU

**MLX 0.32.4 on the M5 Max: float32 `mx.matmul` with more than one row has relative error 7.7e-4 against float64,
about TF32 precision.** A single row (GEMV) is accurate to 1.5e-7, and MLX's CPU to 8e-7. `mx.gather_qmm` was accurate
in a 16-row test.

| check | result |
|---|---|
| mlx-lm after `set_dtype(float32)`, whole sequence | logits 0.030 median, the same as its BF16 (0.026); layer 0 already 0.0017 off |
| the same with mlx-lm's own Metal kernels off | unchanged: the kernels are not the cause |
| NVFP4 dequantization (mlx-lm's sanitize vs the reference) | equal to 4e-8 |
| mlx-lm in float32 fed one token at a time (every product a GEMV) | logits 3e-4 median, top-1 1.000, no route differences in the first 18 layer applications |

So mlx-lm and `eval/inkling2_ref.py` compute the same function. The residual 3e-4 is MLX's half-precision
depth-attention kernel and the like. The reference itself is fine.

**Consequence for engines.** On this machine, an MLX float32 GEMM over K rows does not give a one-row step's bits.
This is added to the run-2 prompt.

Scripts: `scripts/mac_error_attribution.py`, `scripts/mac_mlxlm_vs_reference.py`, `scripts/mac_mlxlm_layer_diff.py`
(`--stepwise`, `--no-kernels`). Logs are in this directory.
