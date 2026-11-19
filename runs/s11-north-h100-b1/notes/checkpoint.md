# What this run measured in the checkpoint (CohereLabs/North-Mini-Code-1.0). Earlier findings: notes/prev/checkpoint.md

Each finding: what was measured, on what, the number, what it allows and what it does not.

## 1. Escapes of the 12-bit format in the operand space (study/esc_census.py, e0006, all 30,484,201,463 non-zero weights)
- With the inherited per-row 15-value exponent window (and per-neuron down-column shifts), the 3,710,152 escapes need
  an exponent code of -23 .. +22 relative to the window base (head -17..22, qkv -15..17, o -16..17, router -12..17,
  expert gate|up -22..19, expert down -23..22, dense layer 0 -11..21). Plus 9 exact zeros.
- Per 16-row tile: at most 29 escapes (o_proj), per (tile, 8-k-step quantum) at most 16 (head), per (tile, lane) at
  most 6, per (tile, quantum, lane) at most 4, per (tile, k-step pair, lane) at most 3.
- Allows: a constant exponent bias of 32 in the widening (multiple of 16, so a PRMT constant byte 0x10 supplies it)
  makes every escape a normal bf16 operand V = W 2^(32 - r - c): escapes can be XORed into the mma operand and every
  product is the checkpoint's weight times a power of two -> results bit-identical to BF16 storage (verified: all
  30,484,201,472 weights through the kernel's own decode, e0013/e0026; logits bitcheck e0012/e0025).
- About 22% of 8-k-step quanta (2048 weights) hold an escape (1.2e-4 x 2048 = 0.25 on average).

## 2. The final hidden state's energy is spread over the dimensions (study/head_energy.py, e0074)
- Final normed hidden states of 8 real sequences x 304 decode positions: the 64 / 256 / 1024 / 1536 dimensions with the
  most mean energy hold a median 14% / 29% / 66% / 85% (minimum 1.4% / 10% / 51% / 76%) of a token's energy. The
  largest single dimensions (1366, 1899, 752, ...) are not "massive activations": the top 8 hold 5%.
- Does not allow: a head sketch over a fixed subset of dimensions with a Cauchy-Schwarz bound for the rest (the rest
  keeps >= 49% of ||x|| even with 1536 dimensions). The int8 sketch over all dimensions stays.

## 3. Duplicate head rows sit in whole tiles at the end of the vocabulary (study/dup_tiles.py, e0143)
- The tied head has 7,129 rows that are bit-for-bit copies of an EARLIER row (grouped by a hash, then compared on
  all 2048 values). 439 of the 16,384 16-row tiles consist only of such rows: tiles 15939-16383, in six runs
  (15939-15943, 15945-15951, 15953-16135, 16137-16143, 16146-16374, 16376-16383).
- Allows (greedy path only): such a row's logit equals the earlier row's and the lowest index wins a tie, so it is
  never the greedy token; the sketch pass skips these tiles (2.7% of its 541 MB). step() keeps every logit.

## 4. Routing persistence from the kernel's own routing tables (study/route_persist.py, e0154)
- Greedy B=1 decode of 8 prompts (2136 tokens after the prompt): a token shares on average 4.77 of its 8 experts
  per layer with the previous token (layer 1: 3.64, layers 1-4: 3.6-4.3, layers 45-48: 4.6-5.1). The earlier run
  measured 4.62 on HF BF16 routing.
- Allows: prefetching (not computing) the previous token's experts' gate|up tiles while a block waits for this
  token's union (UPF, s0021: -6..-10 us at B=1, one tile per block; two tiles per block +19). Does not allow skipping
  or reusing any computation: the routing is only known after the router.
