# Plan (run s18-mini-b200-c4, direction: batch 1, fused), written at ~0.9 agent-h

State: s0002 = 1457 us/token b1 (natural 954), b8 5075 (no speculation at B>=2). vLLM+DSpark 1544 / 871.

## Three largest losses (gap.md)
1. MoE at K rows: w13 + w2 ~52 us/app at K=8 vs ~20 floor; ~32 us/app at K=4 vs ~12.
   (a) serve w13 rounds >= 1 from smem filled by bulk TMA at phase start (free smem: router input + expert-input rows
       >= K + extra dynamic smem): upper bound -73..-394 us per pass (K=2..8), ~-5% weighted. Cost: TMA/mbarrier code.
   (b) w2 the same (its staging area has free space): w2 compute-only 12 vs 7 loads-only -> maybe -3%.
   (c) B-operand layout with LDS.64/128 (fewer shared loads per unit): <= -10% of decode issue.
2. Small-K fixed overhead: V(2) - step ~965 us; mixes (attn_in, mlp_in +3..4 us/app), attention (+4), o_proj (+2).
   (a) batched redundant mixes of all K rows in every block (no hop) vs designated publish/poll (c1 tuned: hybrid at 2).
   (b) attention: the step's chunk path for K<=2 with in-pass rows; shared records only from K>=3.
3. Draft ~390 us per DSpark cycle (eval: ~10% of a cycle); floor ~190.
   (a) fused persistent draft (one launch, grid barriers): ~-100..-150 us/cycle, -3% eval.
   (b) cheaper base unembedding (FP4 copy or better GEMV): -40..-60 us/cycle.

## Next: 1(a) behind a switch (W13_TMA), bitcheck with xcheck, A/B with tv3 on natural text.

## Rewrite at ~2.6 agent-h (state: s0005 1374 / natural 872; s0006 pending)
What paid: layout / selection changes that add no code (W2S_RS bank padding -3.7%, W2V=0 -1.6%, load depths),
and W2X2 (a rewrite of a path that replaces code). What did not: every mechanism that adds code to the verify kernel
(TMA overflow, KV4, KVSPLIT, mark sites: +70..+230 us even at K=1) — ptxas spills rise (240 -> 300-370 B stack).
Three largest losses now (K=8 natural, per application vs step): w13 +20 us, w2 ~+8 us, attention +7, mixes +10.
1. w13 (4 rounds at K=8; the 12-unit batch is only ~6 loads deep after ptxas scheduling).
   (a) two items per group per round with 6-unit half-chains: same number of load latencies -> no gain expected.
   (b) smem staging: no free smem without L1 loss (measured). Leave.
2. Attention: the record item holding the pass's positions (k/v conv, 5.2 vs 3 us). KV4/KVSPLIT cost more in code
   generation than they save. (a) move those k/v convs into the qkv epilogue (no extra code in attention?)
   (b) leave.
3. Draft (eval ~7.6% of a cycle, natural ~4%): (a) skip the ~40 early-exiting DSpark kernels on n-gram cycles with a
   graph conditional / device graph launch (~-1.2% natural); (b) persistent draft kernel (PDL already overlaps weight
   loads; ~-1..-2% eval at large effort). NOT inside the verify kernel: added code there slows every pass.
Next: 3(a) prototype; then look for more layout-only items (smem conflicts, poll depths) with the profile.

## Rewrite at ~3.6 agent-h (state: s0008 1318 / natural 834 = 1.17x / 1.044x vLLM+DSpark; b8 5074 unchanged)
Since the last plan: W2X2 (-1.5%), DSpark kernels behind a graph IF node (-0.7% natural), one specialized verify
kernel per row count (-2.4%). The measured lever of this kernel family is register allocation: specializing removed
40% of the spills and gained more than any mechanism added to the code.
Three largest losses now (vs one step, K=8 natural per application): w13 +20 us, attention +7, mixes +10; and per
cycle: the seven empty verify launches (~55 us), the draft (eval ~350 us per cycle).
1. Re-test the code-adding mechanisms inside specialized kernels (KVSPLIT, KV4_SH, W13X2, certified IMMA argmax
   with ph_final inline): their losses were spills in the generic kernel; per-K allocation may change that.
2. The seven empty launches: (a) graph SWITCH node set by a one-thread kernel reading vpos[1] (needs device-side
   pass tags and cooperative kernels in conditional bodies); (b) fewer libraries covering two K each (2-body inline):
   3 empty launches instead of 7, if 2-body kernels keep the per-K speed.
3. Draft: persistent draft kernel (~-100 us per DSpark cycle at best); not inside the verify kernel.
Next: 1 (running), then 2(a) if 1 gives nothing.
