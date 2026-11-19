# Plan (rewritten 2026-11-17, agent clock ~7 h; after s0008)

Best: track A s0009 1422.2 us/token HTTP (B=8 3427), in process 1418-1421 (s0008 + the layout fix). Track B on this source is ~10 us slower
than track A in process (the HSK=1 binary cannot use KQT without a bad allocation, and its allocation costs about
what the sketch saves): no track-B submission unless a build beats track A by > 5 us in process.

## State of the batch-1 step (MK_CLK, position 700, one layer ~27 us)
A 3.3 | wait for the pairs 1.5 | attention + C 8.9 (attention itself ~6.2: norms 1.0, scores 2.2, weights 1.35,
V mix 1.05, out 0.4) | C->D 0.5 | D + router 1.9 | routing 2.5 | E 5.0 | F 3.3.
HBM-idle windows are now used (q/k/v at the end of C, shared expert at the start of A). What is left is latency:
grid-wide hand-overs (6 per layer), the attention's dependent chains, two rounds in E.

## What was measured since s0008 and lost (journal 53-58)
Output projection prefetch (three placements), uneven E shares, v deferred, scores-stage rearrangements (SCO, ILP2),
K/V per-line L2 prefetch (0), batch-8 placements, routing local array (WRTS +80: allocation), every existing
bit-identical switch (local optimum). The batch-1 kernel's allocation decides most single edits by +-20..80 us.

## Resolved
- The HTTP - in-process gap (25 us since s0007) was the layout of the kernel library's small buffers (parity of a
  2 MB-granular address bit), decided by what the process allocated before the Engine; a 512-byte allocation first
  gives the good layout in the evaluation's engine process: s0009 1422.2 (journal 62-68).

## Open
- Ideas left that are not dead by measurement but need a large rewrite: a softmax in the transposed (K x q^T)
  layout (track A, new summation order: needs both kernel instances changed for batch invariance); E in one round
  at batch 1 (needs 24 tile registers per warp).

## Handoff (agent clock ~9.8 h)
engine/ = s0027 (engine/TRACK = B: the certified int8 head sketch on s0026's kernel; bit-identical to track A): 1352.6 us/token at B=1, 3333 at B=8 over HTTP. TRACK=A builds s0026 (1393.1 / 3339: the best track-A figure).
Leads that are measured but not taken, for whoever continues:
- Batch 8: 4 accumulators in phase D (NACC_D=4, tools/naccd.patch) was -50..-70 us at B=8 on tools/ids.npy rows,
  but submitted as s0011 it gave no batch-8 gain on the evaluator's workloads (HTTP 3443 vs 3427) and B=1 +12.
  Batch-8 tuning needs a benchmark with prompts like the client's (natural text, one per stream).
- Layout: the batch-1 step has a ~30 us good/bad state set by the relative placement of the library's small buffers
  (journal 62-73); the 512-byte pad gives the good state in the evaluation's process today; a robust calibration
  needs a long timing per candidate (the short one does not predict the scored run).
- Everything else tried in this run is closed with numbers in journal.md.

- Cache policy (journal 111-116): the evict-first hint on the routed experts' and the head's tiles paid -69 at B=8;
  the same hint on the output projection (+56 at B=1), router tiles (0), evict-last on the K/V cache (+135 / +190)
  and pinning part of the head in L2 (+3..+50) did not. Prefetches that lost before still lose with it.
- Idle windows (journal 120-125): the last layer's end-of-C slot now prefetches the head (s0015, -4); new kernel code
  at that site grows the batch-1 frame (176 -> 192 B) and loses, so reuse existing statements via pointers.
- Track B is ahead again (journal 128-131): re-check A vs B whenever the batch-1 kernel changes (tools/gabt.sh); the
  KQT_SMALL choice for the B build depends on the allocation it draws. HPF3B (B=8 -3, both tracks) is in the source, off.
- Dense layers (journal 141-150): 51 -> ~40 us each (s0018 preloads, s0020 epilogue). Left: gate/up 7.8 us, warp 0's
  write loop ~1 us per row group (cause not found, journal 149), down 4.5 us; the six row groups on six warps cut the
  epilogue to 3 us but cost the batch-1 allocation (+70).
- Layer 2 (first MoE layer of a step) costs ~6.5 us more than a warm MoE layer (instruction fetch, journal 138-139).

## Handoff (agent clock ~15.8 h; s0022)
engine/ = s0022: TRACK=B (head sketch) on s0021's kernel, 1371.9 us/token B=1 HTTP, 3353 B=8; TRACK=A = s0021
(1398.2 / 3361). Every change since s0017 went in as track A first, then the same source as track B.
What is left, by size (journal 141-159):
- Dense layers ~40 us each vs ~26 for a MoE layer: warp 0's epilogue ~5 us (~1 us per row group for a few shared
  loads and stores: unexplained; two warps cut it to 2.7 us on the timeline but the batch-1 allocation then cost +38),
  gate/up 7.8 us (6.4 TB/s), down 4.5 us. Any edit here is an allocation draw: A/B end to end, not by timeline.
- Head sketch pass 1 at 6.4 TB/s (exact head 8.0 TB/s): not loads in flight, not cache policy, not the row-group
  layout, not the per-row-group small loads (journal 132-140, 159). ~8 us if found.
- Layer 2 (first MoE layer of the step) +5-6 us: instruction fetch of the MoE code after the dense layers and the
  head; blocks 112-127 always late (placement; see prev/journal on ROLE_MAP).
- Head + step boundary ~10 us besides the sketch; the second hand-over after the head is not on the critical path
  (journal 135-136).
Tools: tools/bt_gen.py (MK_CLK stamps in the generate path; "starts" = where a layer begins in the step; MK_CLK_BLK,
MK_CLK_EDUR), tools/gabt.sh (A/B with a track per side), tools/bcmp_tracks.sh (greedy tokens A vs B), res.sh (frames).

## Handoff (agent clock ~17 h; s0025)
engine/ = s0025: TRACK=B on s0024's kernel, 1360.9 us/token B=1 HTTP, 3332 B=8; TRACK=A = s0024 (1399.5 / 3355).
Since s0022: the sketch's pass 2 through shared memory (s0023, -4) and block 0's token publish by atomic exchange
before the host's progress word (s0024/s0025, -7 in track B). What paid in this stretch were stamps per block and
per layer type that found single late blocks and slow layer types (tools/bt_gen.py, MK_CLK_P2, HSTAMP in
tools/allg.patch-style study edits); what lost was nearly every edit inside the batch-1 kernel that changed its
allocation (+13..+115), whatever the timeline said.
Left (journal 160-189): pass 2 in the blocks with candidates (~3.6 us per candidate, blocks often hold 2-3 of ~9.6 per
step; speculation and prefetch variants lost, journal 180-189), block 0's
gather (~2 us: one load trip + settle), layer 2's cold code (+5-6), the dense layers (~40 us each), pass 1's rate.
- L1 size matters (journal 202-203): a larger shared carveout costs +20; fitting the batch-1 kernel into 100 KB of
  shared memory (relayout of the attention staging and the sketch's buffers) might give the reverse.
