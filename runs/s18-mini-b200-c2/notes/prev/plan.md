# Plan (track C), rewritten at 4.2 agent-hours

State: s0008 1619 us b1 / natural 1067 / b8 6351 (vLLM+DSpark 1544 / 871 / 2711). Track A best: s0009 3206 / 6359.
Cycle (evaluator prompts): verify V(K~3) ~4.37 ms + draft 0.12 (n-gram cycles) .. 0.40 ms (DSpark cycles).
Tokens per cycle 2.88 (evaluator; vLLM 3.48 on its own, different greedy text) / 5.22 (natural; vLLM 5.21).

## Three largest losses
1. One-row floor of a verify pass, ~3.5 ms (70-80% of a cycle).
   a) Amortize it over more rows: a 16-row pass would cut n-gram (L=8, 85% fully accepted) cycles from ~0.78 to
      ~0.65 ms/token (est. -8..-12% overall) — but only if MoE and the GEMVs take 16 B-operand rows at once; the
      expert-input smem (f16 hi/lo, 74 KB at 8 rows) and the 3-stream qkv inputs (111 KB) do not fit at 16 rows, and
      doing the GEMVs in 8-row halves doubles MoE bytes (no gain). Not feasible in the remaining time.
   b) Fewer hand-offs per application: each would need step()'s arithmetic kept; previous run's territory; no.
2. Marginal rows: MoE union + two w13 rounds from K=2; 9 bit-identical restructurings measured, all slower except
   one-warp w2a, work balance, raw k/v in smem, K=2 redundant mixes, designated final rows.
3. Draft: now small (FP8 copies, n-gram fold); remaining: FP8 for the draft layers (fork, measuring).

Next: FP8 draft layers if acceptance holds; then validation (robust.py, vcheck long, bitcheck), notes. Remaining
verify-kernel ideas are < 0.5% each (attention for positions >= 1023, designated final for K=2: measured 0).

## Update at 4.8 h
The 16-row pass was measured by proxy (proto16/README.md): worth ~-6% with 8-row halves, ~-12% with one MoE pass of 16
tokens per expert; the latter is the best remaining lever for a future run (design in proto16/README.md and the
journal). Not started here: the rewrite does not fit the remaining time safely.
