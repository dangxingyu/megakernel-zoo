# What was measured in the checkpoint (CohereLabs/North-Mini-Code-1.0, 30,484,201,472 BF16 weights)

Each finding: what was measured, on what, the number, what it allows and what it does not.

## 1. The weights are statistically plain (harness census e0002, study/census2.py e0006; exact counts, all weights)
- 9 exact zeros in 30.5e9 weights, no denormals, no zero row or column, no two identical tensors.
- Order-0 entropy 10.48 bits per weight (exponent 2.56, mantissa 6.97, sign 1.00); within a row the exponent
  entropy is 2.54: no row-level structure beyond scale. Low mantissa bits are uniform (no coarser grid).
- Experts of a layer are unrelated: cosine between two experts' gate/up/down matrices 0.000-0.009 (|max| 0.12 once),
  bit-identical positions 0.09% (chance). gate vs up of the same expert: cosine < 0.01. Layer 10 vs 11 attention
  matrices: |cos| < 0.012.
- Allows: nothing beyond a fixed-width exponent code (section 3). Does not allow: sharing, delta coding, pruning.

## 2. The head (= embedding table, 262144 x 2048)
- 255,015 distinct rows: 7,129 rows are copies of another row (groups of 2712, 1431, 556, 513, 512, ...; ids from
  ~255,000 up and a few others such as 232801, 544): their logits are equal by construction. A head that computes
  each distinct row once would read 2.7% less of the table (0.44% of a batch-1 step). Not built yet.
- Row norms: 0.39 (the copies) to 2.17, median 1.58: no Cauchy-Schwarz pruning of the argmax.
- Spectrum: top singular value 201.5, then 43.4, 42.8, 39.2 ...; energy in the top 256 / 512 / 1024 directions
  30% / 45% / 68%: too flat for a low-rank certified head.

## 3. RMSNorm weights and expert neurons that are switched off (study/census2.py, study/dead_neurons.py e0105)
- Layers 1 and 2 have norm weights from 4.4e-5 to 0.31 (12.8 binades); layer 24: 0.059-0.79; the final norm
  0.010-1.81; two negative norm weights in layer 1 and in the final norm.
- 1,210 of the 4,718,592 expert neurons (14 experts, layers 1, 3, 4, 6, 7, 10, 48) have a down-projection column
  2^20 to 2^62 smaller than the largest column of their expert (1,192 of them more than 2^20). In layers 1-7 and 48
  their gate and up rows are equally small (about 2^-45 of the expert's median row); in layer 10 one expert has 559
  of its 768 neurons like this with normal gate and up rows. Layers 11-47 have none.
- Allows: nothing for speed (they still have to be evaluated: the product is tiny, not zero). Matters for storage:
  a fixed exponent window per row cannot hold a down row that crosses such columns (up to 5,936 escapes in one
  16-row tile); with one power-of-two shift per neuron moved into the x operand the worst tile has 17.

## 4. A 12-bit tile format fits this checkpoint with 1.2e-4 escapes (study/w12_census*.py, tools/verify_w12.py e0111)
- W[i][j] = 2^(r_i + c_j) V[i][j]: r_i one byte per row, c_j one shift per expert neuron (down matrices only),
  V with sign, 7 mantissa bits and an exponent code 1..15 (0 = zero).
- Exact on all weights: 3,710,152 of 30,484,201,463 non-zero weights are outside their row's window (1.217e-4) and
  are kept with their true value in the tile header (at most 56 per tile of 16 x 2048, 24 per 16 x 768; the
  largest count found is 29). The arena is 45,988,003,840 bytes = 12.07 bits per weight including headers.
- tools/verify_w12.py decodes every tile back from its bytes and compares all 16 bits of every weight: equal.
- One window for the whole model would escape 4.75e-4, one per tensor 2.2e-4 .. 3.7e-4 (and 0.69% of the router),
  one per 16-row tile 1.9e-4 .. 3.5e-4 but with tiles of up to 21,550 escapes (section 3).
- Shifting the columns of everything that reads the normed input (one shift per residual dimension and layer)
  makes it worse (gate|up 1.25e-4 -> 1.38e-4, q 1.65e-4 -> 2.30e-4): not used.

## 5. How the routers behave (study/routing_stats.py, e0136; Hugging Face BF16 choices on our own 8 sequences x 128 positions)
- A token shares 4.62 of its 8 experts with the previous token of the same sequence in the same layer (3.17 in the
  first mixture layer, 4.35 in the middle, 5.16 in the last; at least 4 shared in 75% of the cases). With the token
  2 / 4 / 16 back: 4.18 / 3.88 / 3.45.
- Usage is concentrated: the 8 most used experts of a layer take 45.9% of the choices, the 32 most used 82.8%;
  839 of the 6144 experts were never chosen in these 1024 tokens.
- Eight sequences at the same position need 27.8 experts per layer together (8 to 48; 34.5 in the first mixture
  layer, 19.9 in the last), not 64: the batch-8 step reads 43% of the worst case.
- Allows: starting gate|up for the previous token's experts before the router of the layer has spoken (58% of that
  work is needed). Does not allow: skipping anything; the choice is only known after the router.
- Caveat: the prompts are the harness's random-common-word documents and the model's continuation of them.

## 6. Layer 0 is a function of the token (architecture, config.json + modeling_cohere2_moe.py)
- The block is parallel (attention and MLP both read norm(x)) and layer 0's x is the embedding row, so for layer 0
  the RMSNorm, q|k|v before RoPE and the whole (dense, 3072-wide) MLP output depend on the token id only.
- Built (engine/api.py _layer0_tables, track B): two tables made at load with the kernel's rounding points,
  q|k|v [262144 x 5120] and MLP output [262144 x 2048] BF16, 3.76 GB, 3 s. Per step layer 0 then needs only its
  attention, o_proj and the residual add: 21 MB (q|k|v) + 37.7 MB (dense MLP) of the 6.61 GB are not read (0.9%).
- Measured in process, same binary, tables on / off (e0147): batch 8 6605 against 6666 us (-61 us); batch 1 2586
  against 2573 us (+13 us, inside the run-to-run spread of 1%): at batch 1 layer 0 is then bounded by the attention
  chain (~17 us), not by its bytes.
- Does not extend to layer 1: its input already contains layer 0's attention output.

## 7. The head's argmax is decided by very few rows (certified int8 sketch, track B; e0171-e0178, s0010)
- The tied head (262144 x 2048) quantised per row to int8 with s_i = max|w_i|/127: ||w_i - s_i q_i|| / ||w_i|| is
  0.8% at the median, 5% at worst (numpy check on synthetic rows; the real rows give bounds of the same size).
- With the input as two exact int8 digits and Cauchy-Schwarz on the stored error (plus 2^-10 ||w|| ||x|| for the
  float32 rounding of the exact path), the bound per logit is ~3.8 at the median of the final hidden states, 75x the
  actual sketch error. Still, on our sequences only 1.2-1.5 tiles of 16 rows (of 16384) per sequence and step have
  an upper bound reaching the best lower bound; they are computed exactly and the argmax equals the exact one in
  4608 of 4608 greedy steps (B=1 and 8, prompts of 16 and 64). The rest of the head is read as 541 MB instead of
  805 MB.
- Allows: reading 2/3 of the head on the greedy path. Does not allow: a 4-bit sketch (bounds ~20x wider, most tiles
  would survive) or a low-rank one (section 2: flat spectrum).

## 8. Exponent windows (study/window_census.py, e0274)
- Share of weights outside the best per-row window of 2^k exponent values, layers 1/10/24/47 and the head:
  15 values 1.2-1.7e-4, 7 values 2.4-2.9%, 3 values 25-27%, alike in every tensor class (q, k, v, o, gate|up,
  shifted down, head). The weights are Gaussian-like in the exponent: 12 bits (sign, 4-bit code, 7 mantissa bits)
  is the narrowest fixed-width format with a short escape list; 11 bits would escape 2.6%.

## 9. The model repeats its context on this workload (study/copy_draft.py, e0379)
On random-common-word documents the greedy continuation is largely a copy: for 82% of positions the last 1-3 tokens
occurred before, and the token that followed them then is the greedy token 78.5% of the time (4-token drafts:
2.9 tokens accepted on average). Allows: self-speculative decoding with copied drafts, verified exactly. Caveat:
measured on our own 8 prompts, not the harness's.
With 1000 new tokens (e0381): drafts at 89% of positions, first token right 84.6%, 4-token drafts 3.2 accepted.
Cost side: w consecutive positions need 8 / 11.4 / 14.0 / 18.2 / 24.3 experts per layer for w = 1 / 2 / 3 / 5 / 9.
Longer drafts (e0382): 8 / 16 / 32-token copy drafts accepted for 6.2 / 12.2 / 24.0 tokens on average: the greedy
continuation settles into loops.
