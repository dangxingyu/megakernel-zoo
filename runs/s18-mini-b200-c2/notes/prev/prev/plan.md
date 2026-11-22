# Plan (rewritten at 3.6 agent-hours; best track A s0011 = 3563 us B=1, 7694 us B=8, bitwise batch-invariant)

Step (B=1, in process ~3.52 ms at the HTTP workload): 24 plain applications x ~60 us + 24 gated x ~83 us + 137 us
unembedding. Card-1 A/Bs are indicative only (s0008: -45 us on card 1, +60 us on card 0); confirm by submission.

Three largest losses:
1. Hand-off latency on the critical path: ~8 tagged hand-offs per application (w2b->mix, qkv->attention, records->
   merge, merge->o_proj, o_proj->MLP mix, router->routing, w13->w2a, w2a->w2b) at ~1.5-2 us each while weights stream
   (~0.9 ms/step). Mechanisms: (a) fewer hand-offs by computing consumer work on the producer's block (e.g. the new
   partial in every block from the w2 rows: removes w2b but adds ~25 MB of L2 reads per application); (b) faster
   detection: pipelined re-polls (failed twice: code bloat / spills at 30+ poll sites). Next: (a) only where the data
   is small.
2. Streaming at ~5.4 TB/s (~0.94 ms of bytes). Mechanisms: (a) fewer bytes (W12, exact, built and bit-identical:
   no gain at B=1 - decode latency); (b) higher rate (deeper loads in flight: no change; L2 prefetch: B>=2 only).
3. In-block chains (block reductions, attention smem/sync chain, routing top-8): ~0.7 ms. Mechanisms: (a) merging
   reductions (Gram matrix: no gain measured); (b) placing dependent work off the producer's block (paid twice:
   F/G off the D blocks -160 us). Next: look for remaining same-block dependent work.
B=8 (7.7 ms vs 4.6 target): attention waves (2-4 per application), expert rounds; parity with vLLM (5.08) not in
reach without batched expert GEMMs; keep correctness (bitwise invariance) and take B=8 gains only when cheap.
