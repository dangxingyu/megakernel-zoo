# Run summary: Inkling-2-Mini decode engine on one B200 (track A)

## Result
- Best B=1: **s0025, 2755 us/step over HTTP** (1.34x vLLM 3692, 1.57x SGLang 4329; target 3356 met).
- engine/ = **s0026** (submitted, valid): 2761 us B=1 (within eval noise of s0025; the change only touches B>=3
  code), **5814 us B=8** (best B=8; vLLM 5077, target 4615: not met, 0.87x vLLM). s0026 is s0025 plus a B>=3
  attention item layout; results are bit-identical. Kept as the final engine because it equals s0025 at B=1 and
  is better at B=8.
- Track A throughout: weights as stored (NVFP4 / MXFP8 experts with their scales, BF16 elsewhere), unquantized
  expert inputs. All gates pass, including the hidden set. Batch invariance is bitwise at every B. single_launch,
  device_loop and gpu_never_waits all pass. HTTP equals in-process time (no serving overhead).
- Progression (B=1 / B=8 HTTP, us): s0001 7709/19377, s0006 3731/7670, s0012 3330/7361, s0015 3305/7257,
  s0016 3251/7089, s0017 3012/6231, s0020 2964/6081, s0023 2825/5992, s0024 2776/5860, s0025 2755/5825,
  s0026 2761/5814.

## Design
One persistent cooperative kernel per serving cohort: 152 blocks x 768 threads, 80 registers, device loop with
argmax feedback. Blocks hand off through tagged 64-bit (value, tag) words (relaxed b64 stores / polls), with no
grid barriers. GEMVs run on tensor cores (m16n8k16) over weights pre-permuted into fragment order. Each 3-warp group
of the w13 / w2a GEMVs syncs on its own named barrier. Routing builds a union of distinct experts, and experts with
more than 4 tokens are split into two entries. Attention uses 64-key records, so the merge is identical at every B.

## What worked (largest first)
1. **Register pressure is global** (s0017, -240 / -860 us). At 80 registers, unused template instantiations (the
   two-n-group expert GEMV) spilled kernel-loop state in every phase. Splitting popular experts made that code
   dead; deleting it took the stack frame from 600 to 24 B.
2. **Loads behind polls** (s0022-s0024, -190 / -250 us). wait_tag / pollf are volatile asm with a memory clobber,
   so an ordinary load written after a poll waits for it, costing about 1 us (an L2 round trip under load) on the
   critical path. Fixed: the current-key conv history, norm weights, q/r scales, router bias, expert scale factors,
   and o_proj conv history now load before their polls.
3. Tagged dataflow instead of grid barriers, block-owned GEMV strips, the device loop (s0003-s0007).
4. Per-group named barriers in the expert GEMVs (s0016) and deeper load batches (FP4 12 / 6, FP8 4, gate D 3).
5. --use_fast_math (s0012), redux-based top-8 (s0015), per-expert token bitmasks in routing (s0020), and a router
   strip L1 prefetch at B=1 (s0021).

## What did not work (all measured; details in journal.md)
- **Any prefetching during a latency chain:** L2 bulk prefetch of qkv / w2 / K-V, L1 prefetch of gate D / qkv / w2a.
  Under load the extra traffic raises hand-off latency (0.2 -> 1.4 us). It helps only where a long wait precedes a
  stream: qkv / router at B>=2, the router strip at B=1.
- **Register preloads and in-phase register prefetches:** w2a round 0, next attention item's K/V. They spill.
- **More shared memory:** each 30 KB above the 132 KB carve-out costs about 50 us (in-flight loads are staged in
  L1). This ruled out 256-key attention items and double-buffered 64-key items.
- **Split gate down-projection** (three tries), W12 12-bit storage (track B), NTH=576, combined stream polls,
  w2a software pipeline, and w2a interleaved into w13 (a convoy: +3.4 ms at B=8).
- **Message precompute** (5 variants; tests/premsg*_mk.cu.txt). In passes 2-3, each loop layer's messages depend
  only on the previous pass's partials after layers 11 and 17. So the message mix and the message half of gate D
  can be computed one application early. It is correct and batch-invariant, and B=8 is -50..-100 us faster, but
  B=1 is +60..+165 us slower. The precompute alone costs ~18 us; the rest comes from consuming its results. That
  cost is fixed (it appears even when only one gate type uses the results), and extra spills explain only part of
  it. Not kept.
- The cross-pass weight reuse suggested by the operator: L2 cannot hold weights across the ~12 applications
  between passes.

## Where the time goes (B=1, ~2.73 ms in process)
24 plain applications x ~47 us plus 24 gated applications x ~68 us, plus ~137 us unembedding. The step is
hand-off bound: about 10 tagged hand-offs per application at ~1.5 us each under load, two block reductions per
mix, and the attention merge (~5 us). Streaming phases are at their per-SM in-flight limit, and byte time is
about 1 ms of the 2.7 ms. At B=8 (~5.8 ms): attention runs 3-4 waves of 128-key items (~31 us per application),
w13 is at bandwidth, w2a stragglers come from latency noise, and the designated-block mixes cost two hand-offs
each.

## What I would try next
1. **Message precompute at B=1:** find why consuming the precomputed values slows the B=1 gate chain. Compare SASS
   for the gate-D / F-G phases with and without the path. The B=8 gain is waiting if this is solved without
   breaking bitwise invariance.
2. **Fewer hand-offs per application:** fold w2b into the next application's input mix, or merge router and
   routing.
3. **B=8 attention:** a smem budget that allows 3-4 records per item, e.g. by shrinking the xs / x16 union
   members at B>=3, would cut the waves from 4 to 2.
4. **Re-tune with a register budget in mind:** keep one code path per phase. Each extra path costs 20-50 us
   through allocation, which swamps small wins.
