# Run s16-mini-b200-c1 (track C, Inkling-2-Mini, 1x B200) — final summary

## Best submission
**s0022** (= `engine/` as committed; identical file for file to the submitted src.tgz):
- http_step_us_b1 **1528 us/token** on the evaluator's prompts (vLLM + DSpark 1544; start of run, track A s0016: 3251)
- natural_b1 **1020** (vLLM + DSpark 871), b8 **6302** / natural_b8 6508 (vLLM + DSpark 2711 / 1851; no speculation at B=8)
- tokens per event 2.88 (evaluator prompts) / 5.22 (natural)
- every gate passes; generate / http_tokens strict (track C): 8/8 identical in every submission.
Best track A: **s0018** 3141 / 6300 (this source with engine/TRACK = A), bit-identical to s15 s0016.

## What the engine does
- Verify pass `mk_step<true>`: up to 8 consecutive positions of one sequence per launch, each row bit-identical to
  step() (vcheck: 0 differing of 4000 positions; per-row arithmetic is step()'s, in-pass K/V and conv lags from the
  producing rows, shared attention records for rows whose window starts at 0).
- Draft: the bundled DSpark head as fused CUDA kernels (FP8 copies of the base unembedding / fold for the draft only,
  programmatic dependent launch, Markov early exit) + n-gram copy drafts (suffix match >= 5 tokens, K = 8 rows,
  DSpark skipped); draft length K = 1 + longest prefix with prod p_i >= 0.4 (dynamic).
- submit()/poll(): B=1 cohorts speculate (3 cycles queued on the stream, one event per cycle); B >= 2 cohorts use the
  step kernel's device loop; requests too long for the spec margin fall back.

## What worked (headline effect)
- Fused draft 2.15 -> 0.5 ms/cycle, n-gram copies (s0002: 2877 -> 2158); dynamic draft length (-> 1764).
- Verify-kernel latency items: shared attention records, one-warp w2a + work balance, raw k/v in smem, 2-row hybrid
  mixes, gate message-mix first, polling only needed streams (-> 1587).
- Shallow tagged polls (1-2 loads in flight per thread where many blocks poll the same fresh data) + no L2 prefetch,
  bf16 GEMV 3 units deep (-> 1528). Same polls in the step kernels gave track A 3206 -> 3141.

## What did not work (all measured; details in journal.md)
- MoE streaming restructurings (one warp per w13 item, cp.async smem ring, rolling register buffer, contiguous item
  ranges, FP4/FP8 batch depths): all slower; bench/tma.cu shows no streaming limit — losses are phase-level.
- 16-row verify passes (proto16/, bit-identical) cost 10.1 ms: no gain with MoE in 8-row halves.
- Certified INT8 argmax for the final unembedding: exact but no gain (int8 decode bound).
- Expected-tokens-per-time draft length, n-gram majority vote, n-gram min 3/4, θ 0.3/0.5, 3-4 row hybrids,
  gate F/G per channel, in-item merges, kv_conv batching, deeper / per-source polls: equal or slower.
- B=8 speculation: not possible with 8 rows per pass.

## What I would try next
1. A 16-row verify pass whose MoE takes 16 tokens per expert in one pass (new expert-input smem layout that fits;
   proto16/README.md): ~-12% estimated, mostly on natural text (8-token n-gram matches accept all 7 drafts ~85%).
2. The one-row floor (~3.5 ms per pass, 70-80% of an evaluator-prompt cycle): fewer dataflow hand-offs per layer
   application while keeping step()'s arithmetic.
3. B=8 speculation (16-64 rows per pass) once (1) exists.
