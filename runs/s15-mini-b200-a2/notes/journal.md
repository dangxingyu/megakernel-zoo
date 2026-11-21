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
## s0017 experts with >4 tokens split into two union entries; NG=2 code removed — 3012 / 6231 (expected ~3030 / 6.4 ms)
Confirmed and better at B=8. Profiles with a greedy context (tests/prof.py now generates its context) showed experts
with 6-8 of 8 tokens; their blocks took the NG=2 path at ~30 us vs ~14 in w2a. Splitting them in routing made the
NG=2 instantiations dead; deleting them cut the kernel's stack frame 600 -> 24 B and every phase got faster
(B=1 -240 us: spills of kernel-loop state at phase boundaries were costing ~5 us per application).
Lesson: at 80 registers the cost of code is global; templates instantiated "just in case" spill everything.
- After s0017: BF_UNR 3 +25 us; FP8_UNR 6 B=8 -45 / B=1 +-10 (parked); next item's K/V in registers during attention
  compute: stack 24 -> 144 B, +170 / +260 us (K only: +50 / +160). Registers held across a phase spill everything.
- Attention K/V buffers swapping roles (next item's V cp.async into the free K buffer after scores): B=8 +27..+50 us.
- Merge M/S step (3 warps, <=17 records) measures ~2 us; not instruction cache (a second copy right after is
  equally slow), SASS is ~100 ordinary instructions; unexplained. Per-thread M/S instead: +45 / +95 us.
- Code size: mk_step is 469 KB of SASS (29k instructions); routing 3.1k, attention 3.8k, attn_in 3.2k, gate_d 2x2.7k.
  gate_d __noinline__ (one copy): no change.
## s0018 w13 12 units per warp in one batch, MXFP8 6 deep — 3000 / 6174 (expected ~3000 / ~6.17 ms): confirmed.
- Unembed 3-4 units deep: no change (at bandwidth). Split gate D (4 K pieces, s0008 design) retried: with weight
  preload stack 152 B, +320 us; without preload +150 us. Dead end.
## s0019 gate D GEMV 3 units per warp in one batch — 2990 / 6180 (expected ~2980 / 6.18): confirmed (-10 us).
- Message mix moved before the input mix in gate D (its sources are older): B=1 +5, B=8 +50. Not on the critical path.
## s0020 routing token lists from per-expert token bit masks — 2964 / 6081 (expected ~2960 / 6.12): confirmed.
- Router strip L1 prefetch at B=1 (RTPF): -22 us (3 alternations), kept. Gate D L1 prefetch (GDPF 1/3) and qkv first
  units L1 prefetch (QPF 2) at B=1: +30..+40 us (off). w2a cross-item software pipeline (tests/w2stream.cu) retried
  after s0017: B=8 -13, B=1 +10: not kept; w2a is not bound by its round latency.
## s0021 B=1 router strip L1 prefetch — 2948 / 6085 (expected ~2945 / 6.08): confirmed.
- The "2 us merge max/sum step" was a profiler artifact: SUBMARK (thread 0: global store + globaltimer) stalls warp 0;
  the same step on warps 8-10 measures 0.6 us, and in a normal build the warp choice makes no difference. Sub-mark
  intervals in which warp 0 works right after a mark are inflated; crit.py (phase marks) is not affected in the same way.
- B=8 L2 prefetch of the gate D strips / F-G weights during the mixes: +12 us. Disabling the existing B>=2 qkv/router
  L2 prefetch: +30 us (it still pays).
- Attention 64-key items double-buffered in the two halves of kk/vv (cp.async groups) at every B: B=1 -47, B=8 +75
  (+220 at long context); restricted to B<=2 (runtime switch, both paths compiled): B=1 +50, B=8 +50. Reverted.
- Removing routing's leading barrier (masks cleared at the end instead): +18 us B=1 (allocation). ptxas -O2 / -dscm=cg:
  no change. qkv L2 prefetch in gated applications at B=1 (the gate chain is ~17 us): +75 us. Prefetch traffic during
  any B=1 latency chain costs more than it saves.
- w2a round-robin items (balanced like w13) would need B operands from L2 (no per-block staging): ~37 KB of tagged
  loads per 10 KB item, ~210 MB of L2 traffic per application at B=8. Not tried; the staging is why w2a is
  contiguous per block.
- Split gate D retried a third time with the F/G placement fixed (it assumed 12 D jobs, the split has 48): profile
  gate D 7.9 -> 5.0 us but F/G 2.1 -> 4.5 and the norm phase 6.9 -> 11.7; normal build B=1 +90, B=8 +230. Closed.
- W12 (track B) re-timed after the spill fix: B=1 +110 us, B=8 39 ms (pathological). Closed.
## s0022 current K/V conv issues all loads before the first wait — 2888 / 6111 (expected ~2895 / 6.08): confirmed.
The item holding the current position was ~1.6 us longer than the others (profile): kv_conv waited on each raw value
before loading the history ring, 4 channels in sequence. Lesson: wait_tag / pollf are volatile asm with a memory
clobber, so an ordinary load written after a wait is issued after it: one L2 round trip (~1 us under load) each time.
## s0023 norm weights / q-r scales / router bias before the polls — 2825 / 5992 (expected ~2825 / ~6.0): confirmed.
- Next: expert scale factors f13/f2 in smem (loaded at routing start; utok shrunk to 4 tokens per entry pays for it),
  o_proj epilogue conv history + taps staged before the attention-output poll: B=1 -57, B=8 -150 (card 1).
  mlp_mix polls the attention partial with the older sources, final mix loads queries/norm weights first: ~0.
## s0024 expert scale factors / o_proj conv history before their polls — 2776 / 5860 (expected ~2770 / 5.85): confirmed.
- w2a round-0 weights L1-prefetched before the staging poll: +17 us. Not kept.
- After s0024 (none kept): merger gate/sink waited at the output instead of at merge start: 0. Relative-bias rows
  through registers, stored after the polls (first version broke B>=3 invariance: 16 x 128 entries need 3 per
  thread; fixed): 0. Gate D / qkv L1 prefetch re-tested: 0 / inconclusive. Expert scale-factor smem stores after
  the routing polls: 0. One combined poll of the 3 published streams at B>=2 (12- or 8-deep): B=8 +137..+225 us.
- QPF=1 (qkv first unit per warp L1-prefetched at B=1), 4 alternations: identical. Attention-input mix at B=1:
  source wait 1.2 us, depth mix 1.55 (warp-0 marks, inflated), norm 0.66. A Gram-matrix fusion of the norm into the
  mix reduction would only apply to the attention input (the MLP input needs a block max for its f16 scale): ~12 us,
  not attempted.
## Message precompute (not kept; source kept in tests/premsg_mk.cu.txt + tests/premsg_api.diff)
Spec fact: for passes 2-3 every loop layer's messages come from the previous pass' partials after layers 11 and 17
(msg = [P11(p-1), P17(p-1)]), final when the pass starts. Built: during application a's MLP-gate D phase, the 4 B
highest blocks compute application a+1's messages and the message half (units 36-71) of its 4 D strips; gate D of
loop layers >= 1 then does only the input mix + x-half GEMV + the precomputed half; F/G reads the messages per
application. Correct and batch-invariant. Measured: precompute alone +15 us at B=1; full path B=1 +165, B=8 -62.
Profile: gate D 7.8 -> 6.0 us, F/G's own wait shorter, yet ~5 us more per gated application elsewhere (profile
grid barriers hide where). Unexplained; same signature as the split-D attempts (faster D, slower rest).
- 3-stream poll with one word index per slot (P3B=3, 2 round trips instead of 3 at B>=2): B=8 -7, B=1 +20. Not kept.
- Load-depth re-sweep on the s0024 code (2-3 alternations): W13_UNR 6 worse, D_UNR 2 worse (+40), QKV_UNR 3 worse
  B=8; FP8_UNR 2/3/4 all better than 6: 4 best (B=1 -9, B=8 -16).
## s0025 MXFP8 pieces 4 deep — 2755 / 5825 (expected ~2767 / 5.84): confirmed (best).
- FP4_UNR 3 / 4 (w2a's NVFP4 depth; w13 has its own 12): worse (+10 / +35).
- Message precompute, second round (source: tests/premsg2_mk.cu.txt + premsg2_api.diff). Profile builds do NOT use
  grid barriers (USE_BAR is never defined), so the earlier profile was real: gate D -2 us, F/G +3, norms +2..5.
  B=1: precompute on blocks 144-151 during attention (idle there: <= 144 items, no o_proj strip) instead of the
  MLP-gate phase: +165 -> +105; 8 smaller jobs: +130; messages / D halves in the parity buffers instead of the
  per-application ones (safe: application an - 2 is complete everywhere when an's precompute runs): +90 / +118.
  B=8 -45..-62 throughout. Not kept: the B=1 cost is not explained (allocation noise of +-30 on top of ~+90).
- ... and with the x-half D GEMV on 12 warps x 3 units (one batch each): B=1 ~2900 (still +135). Closed.
- w2a's own MXFP8 depth (3 or 6, w13 keeps 4): no difference. qkv GEMV with per-group named barriers: no difference.
Final: s0025 (2755 / 5825) is the best track-A engine and what engine/ holds.
- Message precompute, third round (tests/premsg3_mk.cu.txt with tests/premsg2_api.diff): the B=1 cost tracks
  register pressure (mk_step spill loads 168 B with it vs 116 B without). Holding the precomputed halves in smem
  instead of registers across the mix: B=1 +125 -> +63, B=8 -40; precompute call moved out of ph_attn: B=1 +72,
  B=8 -57..-100. A D-job end sleep (contention test) changed nothing. Still B=1 > +60: not kept. If revisited:
  find the remaining live state the pre path adds (both gate-D paths are compiled into one function).
- Fourth round (tests/premsg4_mk.cu.txt): the precompute path as the only gate-D path (first loop layer's
  precompute beside its own D jobs): B=1 +60..80, B=8 -58; spills 84/136 B vs 76/116 B. Closed for this run.
- Precompute alone in the attention placement (results unused): B=1 ~+18 us; so ~+50 comes from the gate-D / F/G
  consumers on the precomputed path. Unresolved; closed.
- B>=3 attention: merging item = record 0 alone (items {0},{1,2},{3,4},...) when that needs no extra wave
  (ATTALT): bit-identical, B=8 -12..-28 us, B=1 unchanged.
## s0026 B>=3 merging item of one record when no wave is added — 2761 / 5814 (expected ~2755 / 5.80): B=8 -11,
B=1 within noise (+6). Bit-identical to s0025; engine/ = s0026. Partial by construction: at context ~1000 the last
wave still holds 56 two-record items (chunk 1) whose groups' mergers wait for them. A full balance (the last wave
all single-record items: 48 singles + 64 mergers at nrc = 16) would save ~1.5 us per application at long contexts,
needs a non-uniform item -> (group, records) map; not done.
- Full attention wave balance at B>=3 (ATTBAL: first W-1 waves two-record items, last wave singles + mergers):
  bit-identical, B=8 -7..-9 us, B=1 +15..20 (allocation). Not kept.
- Precompute, fifth look: precomputed path only for the MLP gate, or only for the attention gates: both B=1 ~+60
  (as for both): a fixed cost of having the path, not per use. The path as a __noinline__ function: B=1 +100,
  B=8 -70. Closed.
- w13 items strip-major instead of expert-major (bit-identical): B=8 +30 us. Expert-major order (entries complete in
  order, w2a staging of early entries can start) is worth keeping.
- w2a interleaved into w13 (a block runs its w2a after the w13 round that completes its entries; staging moved to
  the dead xr rows; deadlock-free): bit-identical but B=8 8.6 ms (+3.4 ms): a convoy. Blocks that run w2a early
  delay their remaining w13 rounds, which hold other blocks' entries, so every round slips by a w2a duration.
  Any block's later w13 items belong to entries someone else waits for; role-split blocks would starve both
  phases of memory parallelism. Dead end.
- Data point (not submitted): message precompute enabled only at B>=2: B=8 -50..-65 us (context 64-564), -35
  (to 1064); B=1 +5..18; batch invariance no longer bitwise (every position differs slightly). Not worth ~1% at B=8.
