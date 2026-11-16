# Summary of the run (Qwen3-8B serving engine on one H100, Mooncake tool & agent trace)

## Best submission
- **Track B (default build, `engine/` as committed = s0039): slo_share_r2 = 0.952** (180 of 189 requests within
  TTFT <= 5 s and mean time between tokens <= 0.1 s at 1.57 req/s). TTFT p50 0.23 / p90 3.61 / p99 7.76 s, time
  between tokens p50 23.9 / p99 73 ms, longest gap 88 ms, offline step 5.41 / 5.55 ms (B=1 / B=8).
  Engines on the same requests: vLLM 0.64, SGLang 0.67 (TTFT p50 1.6-1.8 s, p90 4.9-5.2 s).
  The highest share on the board, s0032 (0.958), had time between tokens p50 25.3 ms, over the operator's 25 ms
  line; s0039 meets every line except the one below.
- **Track A (`engine/TRACK` = A, same source): s0038, 0.947** (179 of 189), tbt p50 23.7 ms, TTFT p50 0.25 / p90
  4.02 s, offline 5.53 / 5.84 ms (lines 5.62 / 5.97).
- Over 300 s of arrivals (the operator's measurement): 1.61 req/s 0.977, TTFT p99 6.9 s (engines ~6 s: still
  short by ~0.9 s); 2.39 req/s 0.962, TTFT p99 7.2 s (engines 9.5-10.1 s). Other rates: 0.86 req/s 0.951,
  3.07 req/s 0.889 (past capacity).
- Correctness: every submission after the first few valid; logits bitwise independent of batching, chunking,
  prefix-cache reuse and kernel variant; tests/run_all.sh passes on both tracks.

## Design in one paragraph
Passes of up to 2048 token columns (decode rows and prompt chunks together) over the weights, with canonical
arithmetic. The whole serving loop is one CUDA graph on the device: WHILE { k_sched (deadline scheduler,
Moore-Hodgson order, prefix cache, pass size from a cost model with a measured correction); SWITCH (light decode
passes / 18 buckets of pass sizes with per-shape GEMM kernels); k_commit }. GEMMs on wgmma + bulk copies +
mbarriers with producer/consumer warpgroups; attention on wgmma with K/V staged by the copy engine and 4096-
position super-segments folded in order; consecutive kernels joined by programmatic dependent launch edges.

## What worked (largest first)
1. Moving the GEMM and attention to Hopper wgmma / bulk copies (s0005-s0007), then the attention rewrite with q as
   ready fragments and the running-maximum softmax (2.35 ns per query-key pair).
2. Scheduling: deadline order (Moore-Hodgson, shortest first in a burst), a 25 ms normal pass target (time between
   tokens p50 under 25 ms), larger passes only while a prompt is heading past a guard (tail guard, 7 s since s0039)
   or a near miss; the device-side scheduler (no host round trip per pass).
3. Launch work: one graph for the serving loop, PDL edges with early weight stages, light decode branches without
   empty kernels, per-shape kernels per bucket: offline step 5.67 -> 5.41 ms.
4. Small, measured latency removals: staged residual in the add GEMMs' epilogue (cp.async, one round trip instead
   of 16; s0037, more tbt margin), whole-line stores in track A's norm kernel (-6% prompt pass), track A's
   one-column steps normalizing inside q/k/v and gate/up (B=1 under its line).

## What did not work
- Attention variants: three warpgroups, ping-pong, async scores, q in registers, polynomial exp: no faster.
- Multicast (bulk and tensor map), L2 persistence for the residual stream, L2 prefetch of the residual: no gain.
- Track A norms folded into the add GEMM (last block or grid barrier) or applied per input stage: slower.
- Faster k_prep (packed V stores): bit-equal and 2% faster per pass, but same-session replays lost the 180th
  request (passes got smaller under the old cost model), and a fixed version pushed tbt p50 to the 25 ms line.
- Scheduler knob sweeps (normal target 23-24.5 ms, near-miss target, margin, prefill share, boost, tail target
  100 ms): never 181 under the tbt line; the share sits on a knife edge, and every claim needs a same-session A/B
  against the exact previous build (tests/ab_prev.sh).
- The other runs' biggest win (lossless 12-bit weights) speeds weight-streaming decode passes, not the
  compute-bound prompt passes that decide this metric.

## What I would try next
- Recalibrate the scheduler's cost model for the current kernels, so that kernel speed-ups (e.g. the packed V
  stores, commit 9bd1106) turn into prompt throughput instead of smaller passes.
- More long-context prefill throughput (the misses are the last 26-33k-token prompts of 141-246k-token bursts;
  ~10-15% would save another request and close the TTFT p99 gap at 1.61 req/s): epilogue traffic of the add GEMMs
  (~1.7 ms of a 1024-column pass), fusing k_prep into the q/k/v epilogue for large passes, fewer kernel launches.
- Track A in the serving loop: a resident normalized input for decode passes of more than one column.
