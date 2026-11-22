# Journal

## s0001 first persistent kernel (track A)
Expected: valid, ~7.6 ms B=1 / ~17.8 ms B=8. Measured: valid, 7709 / 19377 us over HTTP (in process generate 7604).
Gates: numeric median 0.0246 (HF BF16 0.0272, limit 0.054), p99 0.088; numeric_long median 0.0196; top-1 0.970
(HF 0.962); KL median 6.7e-4 (HF 9.4e-4); invariance bitwise; hidden pass. The engine is closer to the float64
reference than HF BF16 (expert weights exact instead of BF16-rounded).
Conclusion: correctness baseline. The profile (mkprof) shows latency chains, not bytes: ~85 us per layer
application vs ~13 us of bytes; F/G gate phase alone was 2.5 ms (scalar strided loads; fixed after s0001).

## s0002 build failed: the kernel included ../tools/kit/mkprof/mkprof.cuh, outside engine/ (only engine/ ships).

## s0003 tagged dataflow (no grid barriers) — 5329 / 13731 us (expected ~5050 / 12.8 ms)
Relaxed (value,tag) 64-bit polls. An earlier .cg-polling variant measured 3.7 ms but was WRONG at B>1 (stale values
with current tags, like Inkling-2-Small e0064-67): its speed came from accepting stale data. In process ctx 64-192 vs
the eval's 64-1088: attention growth explains most of the miss.

## s0004 block-owned GEMV strips + butterfly reductions + leaner gates — 4043 / 10216 (expected ~4300 / 9.8 ms)
Cross-block split-K reductions (fence + atomic + reload = 3-4 loaded round trips, ~1.4 us each while HBM streams)
removed. Microbenchmark (bench/lat.cu): dependent L2 round trip 0.2 us quiet, 1.4 us under streaming.

## s0005 B>=2 designated mix blocks; w2 split — 3887 / 7927 (expected ~4050 / 7.5 ms)
B=8 mixes were 40 us each (every block, tokens in series); now ~10 us. w2 at B=8 was bound by 144 blocks each
polling all experts' inputs (85 MB of coherent L2 reads per layer); now expert-major + per-strip sum.

## s0006 device loop + o_proj weights preloaded + B>=2 qkv L2 prefetch — 3731 / 7670 (expected ~3730 / 6.8 ms)
B=1 as predicted. Preloading o_proj weights into registers before the attention-output poll: -157 us (A/B).
L2 prefetch: helps B=8, hurts B=1 (+80 us) -> B>=2 only. Not paid: hist staging, Gram single-reduction mix,
attention chunk 128 at B=1, shallower streaming queues, poll back-off.

## s0007 F/G tasks off the D blocks — 3565 / 7668 (expected ~3560): confirmed.
## s0008 gate D split x/message halves — 3625 (expected ~3500): REFUTED on card 0.
Card-1 A/B (3 alternations, same job) had it 45 us faster. s0009 = s0007 source re-evaluated: 3562 (repeatable to
3 us). Conclusion: card 1 and card 0 differ for this change; card-1 A/Bs are indicative only. Reverted.

## W12 track B (not submitted): exact 12-bit code for qkv / o_proj / unembed, bitcheck bit-identical to A at B=1 and
B=8 over 600 positions; 93,282 escapes of 1.129 G weights (8.3e-5). Timing: B=1 3645 vs 3625 us (A) - the decode
adds latency and the phases are not bandwidth-limited enough; B=8 40 ms (pathological, not investigated). Shelved.

## Batch invariance at B>=3 was not bitwise from position 64 (attention chunks 128 vs 64 keys); fixed by 64-key
records everywhere (tests/invar2.py, bitcheck 600 positions: bit-identical).

## s0010 64-key attention records at every B (bitwise batch invariance) — 3590 / 7864 (expected ~3565 / ~7.7 ms)
Evaluator batch gate: bitwise_equal false (max 2.06) for s0007 -> true. Cost from running the two records of a
B>=3 item one after the other. ## s0011 their PV on parallel warp groups — 3563 / 7694: cost recovered.
Not paid (card-1 A/B or eval): deferring the output-gate rows past attention (+350 us), pipelined re-polling (+1.3 ms,
code bloat), merger blocks on the o_proj-free blocks (0), merge staging batch 12 (0).

## s0012 --use_fast_math — 3330 / 7361 (expected ~3350 / ~7.45 ms): confirmed; 1.109x vLLM at B=1.
The vector code on the critical path (softmax exp, sigmoid, the tanh soft-caps of the MLP activation, gate tanh,
divisions in norms / clips) was instruction-latency bound: accurate expf / tanhf / IEEE division are long chains.
Gates: numeric median 0.0246 (limit 0.054), p99 0.131 (0.254), top-1 0.974 (HF 0.962), KL median 8.2e-4; batch
bitwise. FTZ checked harmless: MXFP8 UE8M0 scales 113..118, no zero NVFP4 scales (tests/mxscales.py).

## s0013 merge-staging overlap fix (positions >= 3712 crashed) — 3393 (expected 3330): REFUTED (+63 us) — moving
arrays out of the smem union changed allocation, and an id check added to _check put 2 more host syncs per step().
## s0014 fix v2 (own record staged first; layout as s0012) — 3344 / 7394: fix kept, cost mostly gone.
## s0015 top-8 by redux.sync (bit-identical) — 3305 / 7257 (expected ~3305 / ~7.25 ms): confirmed. 1.117x vLLM B=1.

## NTH=576 (18 warps, 96 regs) — dead end
Predicted fewer spills / cheaper block reductions. Measured (card 1, same job): B=1 3621 vs 3344 us, B=8 7091 vs 6494
(generate). Still spills (448 B) at 96 regs; fewer warps in flight hurt the streaming GEMVs more than spills cost. Reverted.

## B=8 experiments after s0015 (none kept)
- Smem size matters a lot (L1 = 256 KB - carve-out): +30 KB pad -> +50 us B=1, +70 KB -> +260 us. Current 133.8 KB is just
  under the 132 KB carve-out. Hence attention items of 256 keys (CH=256, 4 records; bitwise invariant) lost: +300 B=1,
  +490 B=8 from smem alone.
- Ctx by value (__grid_constant__, constant bank): +60 B=1 / +115 B=8 (dynamic-index constant loads slower than L1).
- Spill map (tests/spills2.sh): spills are at phase call sites (kernel-loop state saved around w2a / attn_in), not in loops.
- Distinct experts at B=8 (random prompts + greedy): 32.7 per application (147 MB). Expert phases ~64 us = 2.3 TB/s:
  bound by bytes in flight per SM (~48 KB in registers at FP4_UNR 4, ~2 us latency under load).
- L2 bulk prefetch at B=8, every form: whole w2a range at w13 start +227 us, K/V of attention items during qkv +34,
  rolling one round ahead in w13/w2a +250. Dead end.
- Token ids: host min/max check removed for device tensors (2 syncs per step()/generate()); kernel clamps ids.
## s0016 per-group named barriers in w13/w2a + FP4 6 / FP8 3 deep loads + no id-check syncs — 3251 / 7089
(expected ~3285 / ~7.1 ms): confirmed, B=1 better than predicted. Mechanism: a w2a round at B=8 cost ~4.5 us for
~74 KB per SM = the slowest of 24 warps; groups now advance independently. Bitwise invariance kept.
- w2a weights of round 0 preloaded into registers before the staging poll (like o_proj): stack frame 600 -> 1136+ B
  even with 1 preloaded unit; B=1 +250 us. Dead end at 80 registers.
- Smem sized for B=1 (L1 124 -> ~156 KB): ~-10 us only; L1 above 124 KB does not pay, below it costs (in-flight
  loads are staged in L1: with a 70 KB pad, qkv/oproj/w13/w2 each +1-1.7 us per application).
