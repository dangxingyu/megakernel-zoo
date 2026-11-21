# Findings about the checkpoint (each with the measurement behind it)

## Expert choices repeat across the loop's passes (tests/t_routes.py, e0108; README text, 400 tokens, B=1, own engine)
- Same token, same loop layer: pass 2 shares on average 4.2-5.8 of its 8 experts with pass 1 (layers 6-16; layer 17
  only 3.4); pass 3 with pass 2: 4.4-5.7 (layer 17: 2.2); pass 3 with pass 1: 3.4-5.0.
- Consecutive tokens, same application: 1.7-5.6 of 8 (mean ~3.6).
- Allows: predicting experts for prefetch or speculation. Measured use: L2 prefetch of the predicted experts' w13
  rows during attention (EXPF): +174 us at B=1, +577 at B=8 (the extra 23 MB / layer of HBM traffic slows the
  latency-bound attention / o_proj / mix phases more than the w13 cold start gains). Not used.

## Scales
- NVFP4 (layers 0-19): block scales are positive normal E4M3 everywhere (no zero, subnormal or NaN scale; checked on
  all weights at load); per-expert denominators all 2688. This allows the exact 2-instruction E4M3 -> fp32 scale
  decode (byte << 20) + (120 << 23) used after the MMA.
- MXFP8 (layers 20-23): UE8M0 scales in [113, 118], 5-6 distinct values per tensor; the E4M3 weights contain
  subnormals (~47K per w13 tensor), so the decode goes through cvt.f16x2.e4m3x2 (exact) and a 2^112 rebias.

## Findings of the second half (B=8 focus), 2026-11-21
- Natural text routes 8 sequences to a union of 22-53 experts per application (mean 37 at pos 300, README text);
  random tokens route to far fewer: B=8 benchmarks must use natural tokens (tests/nat.py).
- Expert GEMV decode (mma.sync m16n8k16, E2M1 / E4M3 decode in registers, fp32 scale per k16 block): ~58 essential
  instructions per 576-B NVFP4 item; the ring's per-stage overhead (mbarrier wait, two bulk-copy issues, proxy
  fence) was ~100 instructions; 12-item stages x 2 and no fence: 54 -> 71 items/us/SM from L2, ~5 TB/s at full grid
  from HBM. Bytes in flight beyond that do not help; cp.async.bulk.prefetch.L2 hurt in every form tried (5 forms).
- A 2-stage ring is a refill round trip when a warp has a single 36-item unit (B=1 w13): three stages in flight
  (idle ring + xs buffer) -44 us per step.
- B=8 MoE structure: w13 and w2 both as contiguous (expert slot, rowtile) units per block (w2 needs at most 2 slots'
  activation rows), outputs y to global, combine after a barrier in slot order; per-slot completion counters in
  place of the E->F barrier. Splitting units for warp balance lost every time (3 forms).
- Local memory (stack arrays) lives in L2 here: pointer arrays passed to helpers cost a dependent round trip before
  the loads they describe. Moving them to shared memory / closed forms: stack 1280 -> 208 B, -250 / -290 us.
- The kernel is register-bound (255) and large (554 KB): every added path moved allocation (often +60 us at B=1 even
  when not taken). Instantiating it per batch size (template on B: B=1, B=8, generic) was worth -70 / -255 us and
  made B>1-only changes measurable in isolation. Out-of-line helpers do not shield the caller: their call sites still
  move its allocation (watchdog printf out of line +420, a noinline z-row helper +1000).
- --use_fast_math: the fp32 vector code on the critical path is instruction-latency bound (-121 / -239 us); the
  evaluator's numeric gates did not move (median 0.026, p99 0.11; long 0.019 / 0.23).
- mixnorm (block attention residual + RMSNorm) is ~6.4 us in the engine vs ~3-4 warm: 41% of its stall samples are
  instruction-cache misses (75 KB of predicated unrolled code); a 1-query/1-stream instantiation helps the B=8 kernel.
- Measurement: two instances of the same library differ by up to ~56 us at B=8 (memory placement); B=1 within ~2 us.
- Operator suggestions, status: (1) each distinct expert's weights are loaded once per application for all tokens
  routed to it (union of the batch's choices; MMA columns = tokens) - done since s0001; (2) attention and the loop
  passes are batched over the 8 sequences (one item per (sequence, kv head, piece), one grid round) - done;
  (3) reuse of expert weights across the loop passes: not feasible here - between pass p and p + 1 of a loop layer
  the 11 other loop layers stream >= 110 MB each at B=8 (~150 MB with their experts) through a 126 MB L2, and an
  expert (3 MB NVFP4) does not fit in shared memory; the message gate's message half (D[:, H:] . unit(msg)) is
  computed once per pass for all four gates (phase M), the only part that is pass-invariant.
- Refill latency vs decode rate (s0025-s0027): in the B=1 kernel the weight streams are refill-latency bound: loading
  a full ring stage into registers and refilling the slot (TMA, after fence.proxy.async) before the stage's MMAs was
  worth -70 us lone-process / -78 HTTP (3428 -> 3350, 1.102x vLLM), and 6-item stages beat 9 there (-36). In the B=8
  kernel the same streams are decode bound: the early refill (any weight kind), smaller stages, and 3-stage rings all
  lost (+5 .. +975); 9-item stages (no partial stages for the 18/36/72-item tasks) beat 12 (-70 HTTP).
- Placement: a build whose B=8 kernel SASS differed only in the order of a few zero-initialisations ran +30 us at B=8
  lone-process, and padding kernels did not move it: lone-process A/B at B=8 has a ~30 us floor tied to the binary.
  The evaluator's B=8 HTTP repeats spread +-65 us within one submission.
- Attention sub-chunk size at B=8: 128 keys better while pieces are one 128-key chunk (ctx < ~512: -56), 256 better
  for 2-chunk pieces (ctx ~960: +45); on the score's workload the two are within ~30 us in process.
