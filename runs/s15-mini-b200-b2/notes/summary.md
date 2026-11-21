# Run summary: s15-mini-b200-b2 (Inkling-2-Mini decode on one B200, track B)

## Best submission
**s0052**, track B: **2880 us/step at B=1, 6486 us/step at B=8** over HTTP (in process 2833 us at B=1). Valid,
every gate passed, milestones build / correct / device_loop / single_launch / gpu_never_waits.
`engine/` is byte-identical to s0052's source; `engine/TRACK` =
`B MSGZ:N SKH L0T:N K1 KN EXQ:N NG2NI:N BF_UNR=3:N BPX=4:N BPG:1 NI_FINAL=__noinline__:1 GLIN:1 BPX=8:1 KVC4=0:N
RTMASK=0:1 FP4_UNR=6:1 BPX=132:N` (tokens explained in engine/build.sh; NAME:1 = batch-1 kernel only, :N = the B >= 2
kernel only).
Best track A (same tree, TRACK `A K1 BPG:1 BPX=8:1 KN EXQ:N NG2NI:N KVC4=0:N RTMASK=0:1 FP4_UNR=6:1`): s0048,
2949 / 6730.
Against vLLM (3692 / 5077): 1.28x at B=1 (the 1.10x target, 3356, is reached), 0.78x at B=8 (target 4615 not reached).
Track B against track A: -69 us at B=1, -244 us at B=8.

## Progress
| sid | B=1 | B=8 | step |
|---|---|---|---|
| s0003 | 6578 | 12658 | own engine |
| s0006 | 3723 | 7707 | adopted the parallel track-A run's engine (a2 s0006) |
| s0012 | 3612 | 7583 | MSGZ + SKH (track B) |
| s0018 | 3380 | 7404 | ports of a2 track-A changes, one L2 prefetch |
| s0020 | 3150 | 7444 | K1: batch-1 kernel compiled from the same source |
| s0029 | 3083 | 6725 | EXQ expert work queue + NG2NI (B >= 2 kernel) |
| s0037 | 3005 | 6635 | GLIN linearised loop gate (track B) |
| s0042 | 2944 | 6601 | shared memory under the carve-out steps |
| s0047 | 2879 | 6509 | a2 ports per kernel, 6 FP4 units in flight (batch-1) |
| s0052 | 2880 | 6486 | final |

## What worked
- Track B (model-specific, measured against track A):
  - MSGZ: the message gate's message half computed once per loop pass (the rank-64 gate of layers 6-17), B >= 2.
  - SKH: certified int8 sketch of the unembedding with softcap-aware candidate comparison and exact rescoring.
  - L0T: layer 0's q/k/v rows from a per-token-id table (layer 0 sees only the embedding), B >= 2.
  - GLIN: the gate's down-projection applied to the raw sources of the depth mix in parallel with the mix (the
    block attention residual is a linear mix): -69 us at B=1, within the gates.
- Track A organisation (bit-identical):
  - K1 / KN: the same source compiled for batch 1 and for B >= 2; the batch-1 kernel lost almost all spills.
  - EXQ: expert items from a per-block queue instead of synchronised rounds; NG2NI: the rare >4-token path out of line.
  - SMTRIM / SM1: shared memory per block under 132 KiB / 100 KiB (more L1); the footprint itself was a cost.
  - Batched re-polls of tagged hand-offs where few words are stale; ports of a2's exact changes (router top-8 with
    redux.sync, gate F/G placement, kv conv loads first, routing bit masks).

## What did not work (measured; details in notes/journal.md)
- Exact weight formats (weights study, notes/weights_study.md): 6-bit NVFP4 scales (+149 / +469 us), 12-bit BF16
  qkv (+102 / +176), 12-bit unembedding (neutral). The weights are near maximum entropy; no zero, duplicate,
  near-duplicate (max pair cosine <= 0.012) or low-rank experts; hot-expert residency covers ~6%; pass reuse needs
  430 MB - 2.2 GB of cache.
- Bulk L2 / L1 prefetches of weights or K/V (9 variants), TMA / cp.async staging in hot loops, persisting L2.
- Attention: 64- or 256-key items, two items per block, double-buffered K/V (the needed shared memory costs more).
- Gate K-split, linearised gates at B >= 2 or for the MLP gate, linearised q/k/v, deeper expert load batches in the
  B >= 2 kernel, poll back-off, interleaved w13 order, 4-controller message items.

## What I would try next
1. B=8 expert / attention pipeline (w13 + w2 ~62 us and attention ~25 us of a ~167 us application): the a1 run's
   design (TMA weight rings, w2 fixed at compile time) reaches 5608 us at B=8.
2. Attention at B >= 2 with a second K/V buffer that fits inside the present 133 KB (the masked gain of
   double buffering was ~-100..-300 us).
3. Two-level head sketch (int6 first level, int8 on the surviving strips): -25% of the head's bytes, ~19 us at B=1
   (offline study done).
4. The batch-1 MLP gate's D stage (11.8 us per gated application, the largest single item at B=1).
