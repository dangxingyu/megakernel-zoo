# What the Qwen3-32B checkpoint holds (exact measurements)

## Census (tools/kit/weights_census.py, all 450 matrices, 32,761,446,400 weights, exec e0085, hour ~3)
- No zeros, no subnormals, no zero rows or columns, no identical tensors.
- Entropy per value 10.54 bits (exponent 2.62, mantissa 6.97, sign 1.00); exponent entropy within a row 2.57.
- Low mantissa bits zero: 1 bit 49.98%, 2 bits 24.98% (uniform): no coarser grid.
- Fixed-width code [sign][exponent - window start of the row][7 mantissa bits]:
  12 bits (16-exponent window per row): 1.03e-4 of weights outside (12.0033 bits/weight with 32-bit escapes);
  11 bits (8-exponent window): 1.66% outside; 10 bits: 15.3%.
  Per group (12-bit escape share): down_proj 1.13e-4, gate_proj 1.44e-4, up_proj 0.74e-4, o_proj 0.76e-4,
  q_proj 0.84e-4, k_proj 0.81e-4, v_proj 0.77e-4, lm_head 0.83e-4.
- Allows: an exact 12-bit weight stream (track B) with ~3.4M escapes patched exactly; 48 GB instead of 64 GB per step.
- Does not allow: an 11-bit code without a large escape cost (1.7%), any structural sparsity or sharing.

## Exact 12-bit storage built and verified (hour ~4.5)
- engine/w12.py: per-row 16-exponent window, rows scaled by 2^(128 - lo_r) (decoded exponent 0x80 | e'), layout with
  three words per lane and k-step (a0-a2 in place, a3 in their free bits), escapes patched from per-warp lists.
- All 31,983,534,080 streamed weights: kernel-arithmetic decode == scaled checkpoint bits, and scaled value x row
  factor == checkpoint value in float64: 0 mismatches (MK_W12_VERIFY=1). 3,331,127 escapes. 47.98 GB vs 63.97 GB.
- lo_r ranges 106..111 over the matrices sampled (scale factors 2^17..2^22, all products far from fp32 limits).
- Logits bit-identical to track A (tools/kit/bitcheck.py, 1152 positions, B=1 and B=8, every 4th position) once
  both tracks split the work in the same quads of 4 k-steps (before: 19 of 81,920 q/k/v outputs differed by one
  bf16 ulp from a different split-K summation grouping, not from the storage).
- Escapes concentrate in layers 1-5 (gate_proj: up to 234 in one warp's share, down_proj up to 79); those layers'
  gate|up ops take up to 110 us vs 63 elsewhere.
- Re-verified on the final tree (board hour ~22.5, exec e0408, MK_W12_VERIFY=1): 31,983,534,080 weights decoded with the kernel's arithmetic == scaled checkpoint bits, and x row factor == checkpoint value in float64: 0 mismatches, 3,331,127 escapes.
- Escape distribution over the kernel's units (tests/esc_tiles.py, exec e0437): layer 1 gate|up 298,518 escapes, 55%
  of (16-row tile, 64-k quad) units hold at least one, every tile has escapes, the worst warp range is 79% escape
  quads (down: 54%, 79%); layer 2 47% / 59%, layer 3 35% / 47%, layer 4 20% / 28%, layer 5 18% / 18%, layer 10 7% / 7%.
  Allows: nothing tile-local (no escape-dense tiles to store apart). Plain BF16 for layers 1-5's MLP would cost
  ~1 GB/step (~130 us), more than the escapes' ~50 us.
