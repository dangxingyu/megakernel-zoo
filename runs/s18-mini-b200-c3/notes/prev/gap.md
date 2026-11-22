# Gap accounting, track C (B=1 unless stated) — updated after s0010

Headline = cycle time / tokens per cycle. s0010: evaluator prompts 1617 us/token at 2.88 tokens/cycle (cycle ~4.65 ms);
natural 1064 us at 5.22 (~5.55 ms). vLLM + DSpark: 1544 (3.48 tok/step, ~5.4 ms cycles) / 871 (5.21, ~4.5 ms).

## Cycle (nsys, served natural request, s0008 engine): 4396 us median
  verify pass 4008 us (91%) + draft span 111 us (PDL; 210 us of kernels; n-gram cycles skip DSpark) + ~8 us idle.

## Verify V(K), ctx 600 (tests/c/tv.py, s0010): 3497 (K=1), 4110, 4320, 4520, ..., 5740 (K=8) us
Per application (tests/crit.py, -DMKPROF):   K=1 54 / 77 (plain / gated) us; K=2 77 / 94; K=3 79 / 101; K=8 ~120 / ~135.
  K=3 plain vs step B=1: attn_in +3, attn +5.5, mlp_in +3.5, w13 +8.5 (2 rounds from 12 union experts), w2 +5.6.

## Named losses
1. One-row floor ~3.5 ms per pass (70-80% of an evaluator-prompt cycle): ~10 dataflow hand-offs per application,
   gates 2 x ~10 us per gated application. Amortizing it over more rows is the remaining structural lever
   (16-row passes for n-gram proposals: fork prototype in proto16/).
2. Rows 2..K: MoE union (+1.8 us per extra expert per application), w13 rounds (8 items per block per round),
   attention records (+4..12 us per application). Nine bit-identical restructurings measured; kept the ones that won.
3. Tokens per cycle on evaluator prompts: 2.88 vs vLLM's 3.48 (different greedy text: vLLM's target quantizes the
   expert inputs). n-gram copies beat DSpark at every match length >= 5 (offline, same anchors).
4. Draft: ~0.12 ms (n-gram cycles) .. ~0.40 ms (DSpark cycles); base unembedding (FP8 copy) ~75 us, 3 layers ~180 us
   (latency-bound GEMVs: FP8 layer weights saved only 10 us).
