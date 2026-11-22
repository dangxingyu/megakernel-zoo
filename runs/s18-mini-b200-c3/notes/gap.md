# Gap accounting, track C (B=1), after s0005 (1430 us/token synthetic, natural 959; vLLM + DSpark 1544 / 871)

Time per token = sum over cycles (verify pass V(K) + draft) / tokens. Measured on s0005's own HTTP runs (engine
tokens-per-event histograms in result.json) and in process (tests/c3/serve1.py, tv16.py, dtime.py, dseq.py).

## Cycle mix (HTTP, s0005)
- synthetic b1: 1319 events for 4096 tokens; 571 events of 1 token, 341 of 2; 98 16-row passes carry 38% of tokens;
  est. time share: <=8-row cycles 87%, 16-row 13%. <=8-row cycles average 2.08 tokens.
- natural b1: 617 events; 165 16-row passes carry 63% of tokens; time share 16-row 41%, <=8-row 59%.

## Costs (ctx 600 unless stated; card 1)
- V(K) 8-row kernel: 3469, 3896, 3967, 4190, 4669, 5052, 5280, 5444 us (K = 1..8); ctx 990: V(8) ~5750.
- V(16) 16-row library: 8570 (ctx 600), ~8880 (ctx 990); V(12) 7470. Per application (16 rows, plain):
  attn_in 4.6, qkv 18.0, attn 28.5, oproj 12.6, mlp_in 4.4, router 10.5, routing 4.8, w13 34.8, w2 48.5, w2b 2.2;
  8-row kernel: qkv 7.7, attn 15.6, oproj 7.1, router+routing 7.3, w13 25.9, w2 29.1 (MoE grows with the expert union).
- Draft (DSpark cycle): 369 us with PDL (392 without); per kernel (dseq.py): 3 draft layers 3 x 67 = 201 us, base
  unembedding (FP8, 461 MB) 96 us, Markov step 1 17 us + empty early-exit steps ~19 us, start/ctx K/V 26 us.
  n-gram cycle after a 16-row pass: ~145 us span (context K/V of 16 rows + empty kernels).
- Gated second verify launch (both libraries queued each cycle): +4..6 us per cycle.

## Named losses and what is left of each (in my direction)
1. The 8-row verify floor V(1) 3.47 ms (step() at B=1 is 3.1 ms): 80% of a synthetic cycle. Not this run's lever.
2. Low acceptance on synthetic text (generic prose, depth-1 acceptance 0.73): trees / leaves <= 2.5% (simulated,
   needs tree verification), draft-length rules within 1%, a 1-row option -0.3..-0.6%, vocabulary restriction and FP4
   base logits lose acceptance as fast as they save time.
3. 16-row pass cost 8.57 ms: made one-pass where it paid (MoE, final unembedding); qkv, router, merges one-pass were
   no faster; poll depths neutral. Remaining structure (expert union, 48 queries of attention) is inherent.
4. Draft 369 us per DSpark cycle (~9% of a synthetic cycle): persistent fused draft est. -100 us; FP8 draft layers
   -0.5% (measured), Markov loop fusion est. -30 us.
5. 16-row passes only for anchors <= 1008: lifting it is worth <= 0.8% (simulated).
