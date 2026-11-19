# Summary of run s11-flint-b200-b1 (track B on the inherited s0077 engine; Flint, one B200)

## Final state (end of run)
**Best: s0027, track B, 1352.6 us/token at batch 1 over HTTP (3333 us/step at batch 8)** = `engine/` (TRACK=B).
Bit-identical to track A of the same source (tools/kit/bitcheck.py 1.0 at B=1/B=8; greedy and served tokens
identical at batch 1-8). Best track A: s0026, 1393.1 (the same source with TRACK=A). Start of the run: the inherited
engine re-measured at 1487.7 (s0001); 1422.2 after the layout fix (s0009).

What worked (largest first):
- The serving process's buffer layout: a 512-byte allocation before the engine's own (-30; journal 62-73).
- The dense layers 0-1 (~51 us each vs ~27 per MoE layer): tiles preloaded before the barriers, the down preload
  after warp 0's writes with plain stores + one fence, the gate/up loop unrolled by 16 (~-28; journal 141-156).
- Idle-warp bulk L2 prefetch into windows where HBM is idle for every block (next q/k/v at the end of C, the shared
  expert at the start of A; -40), the last layer's slot reused for the head through an extra layer pointer (-4).
- The head sketch (track B, certified int8 copy of lm_head) once its build drew the same allocation as track A, and
  its pass 2 staged through shared memory (one late block held the grid; ~-20 together; journal 128-131, 160-162).
- Evict-first L2 policy on the streamed expert and head tiles (batch 8 -69, batch 1 -6).
- Fences off the critical path: the chosen token published by atomic exchange before the host's progress word (-7).
- A never-true runtime clause removed so the batch-1 kernel drops a dead branch (-8: its register allocation).

What did not: any prefetch that overlaps a streaming phase (K/V, experts, output projection, in-stream bulk prefetch:
+7..+38); attention split across SMs; uneven expert shares; atomicMax token gather, per-block gather, speculative and
paired pass-2 variants; inlining or out-lining helpers; nearly every edit inside the batch-1 kernel that moves its
register allocation (+13..+125 whatever the timeline said); compiler options, switch combinations, scheduling
barriers, L1/shared carveout preferences (flat or worse). Details with numbers: journal.md 1-211.

What I would try next (plan.md has the sizes): fit the batch-1 kernel into <= 100 KB of shared memory (a one-step
larger carveout costs +20, so more L1 may pay ~that much); overlap pass 2's candidates (blocks hold 2-3 of ~9.6 per
step, ~3.6 us each) with a second staging buffer; the first MoE layer's +5-6 us of cold code; the dense layers'
remaining ~40 us each (warp 0's write loop); look for more dead runtime branches in the batch-1 kernel (the cheapest
lever found late). Method: stamp per block and per layer type (tools/bt_gen.py), ncu stall sampling per line
(tools/ncu_gen.py + ncu_lines.py), and judge every edit by an end-to-end A/B in the scored layout (tools/gabt.sh).

## Results (sealed evaluations, HTTP, us per token at batch 1 / us per step at batch 8)

| | submission | batch 1 | batch 8 |
|---|---|---|---|
| inherited s0077 (operator's rematch) | - | 1480 | 3400 |
| track A, inherited arithmetic re-measured on this pod | s0001 | 1487.7 | 3426 |
| track B: certified int8 head sketch (on the s0001 kernel) | s0002 | 1472.5 | 3438 |
| track A: attention scores as K x q^T (bit-identical) | s0003 | 1476.7 | 3417 |
| track B on s0003 (sketch; 16-mma scores form in the B build) | s0004 | **1471.4** | 3437 |
| track B, sketch in the batch 1-2 kernel only | s0005 | 1471.8 | 3428 |
| track B final (s0005 kernel + submit() input check) | s0006 | 1472.0 | 3456 (noise: same batch-8 kernel as s0003-s0005, 3417-3437) |
| track A: idle-warp L2 prefetch (next q/k/v in the routing, shared expert in the attention wait) | s0007 | 1463.0 | 3422 |
| track A: the same prefetches moved to windows idle for every block (end of C, start of A) | s0008 | 1452.0 | 3426 |
| track A: a 512-byte device allocation before the engine's own (buffer layout of the HTTP engine process) | s0009 | 1422.2 | 3427 |
| the same source re-evaluated (reproducibility of the layout fix) | s0010 | **1418.5** | 3427 |
| trade-off: 4 accumulators in phase D (B=8 gain on random-token prompts did not carry to the client's prompts) | s0011 | 1430.2 | 3443 |
| K2B on (two K batches in flight for the batch-8 attention pairs on main blocks; bit-identical) | s0012 | 1419.0 | 3425 |
| L2 evict-first hint on the expert and head tile loads (bit-identical) | s0013 | 1417.4 | **3356** |
| the same hint without L1::no_allocate (bit-identical) | s0014 | 1411.8 | 3373 |
| the last layer's idle q/k/v prefetch slot fetches the head's first 20 MB (bit-identical); best track A | s0015 | 1407.8 | 3368 |
| **track B** on s0015: the int8 head sketch with the 8-mma scores form (bit-identical to track A) | s0016 | 1400.4 | 3362 |
| track A: the dense layers' MLP tiles preloaded before their barriers (bit-identical; a loss in the A build's allocation) | s0017 | 1417.6 | 3379 |
| **track B** on s0017 (bit-identical to track A) | s0018 | 1379.6 | 3367 |
| track A: the dense layers' epilogue (down preload after the bump; plain stores + one fence; bit-identical) | s0019 | 1401.8 | 3367 |
| **track B** on s0019 (bit-identical to track A) | s0020 | 1374.4 | **3350** |
| track A: the dense layers' gate/up loop unrolled by 16 (bit-identical) | s0021 | **1398.2** | 3361 |
| **track B** on s0021 (bit-identical to track A) | s0022 | 1371.9 | 3353 |
| track B: the sketch's pass 2 staged through shared memory (one load round per candidate) | s0023 | 1368.1 | 3352 |
| track A: the chosen tokens published by atomic exchange, the bump before the host's progress word | s0024 | 1399.5 | 3355 |
| **track B** on s0024 (bit-identical to track A) | s0025 | 1360.9 | **3332** |
| track A: a never-true clause removed (the batch-1 kernel drops a dead branch) | s0026 | **1393.1** | 3339 |
| **track B** on s0026 (bit-identical to track A) = `engine/` | s0027 | **1352.6** | 3333 |

Track B is bit-identical to track A of the same source (tools/kit/bitcheck.py: logits at B=1 and B=8, 1152
positions); greedy and served tokens identical (generate, serve, every batch size 1-8, long context, vocabulary tail).

## What was built
1. **Certified int8 sketch of lm_head** (track B; notes/headsketch.md). An extra int8 copy of the head (per-row
   scale, per-row error norms over 32 column blocks) is read on the greedy paths; integer IMMA against two exact
   digits of x; a rigorous bound (Cauchy-Schwarz per block + rounding terms; >= 7.8x the largest error seen) selects
   the rows to recompute with the exact head's own mma chains. step() keeps the exact head. Head phase 79 -> 63 us,
   about -15 us per token at batch 1 (s0002: 1487.7 -> 1472.5).
2. **Scores as K x q^T** (track A, bit-identical): the tensor core gives identical results for the transposed
   product (0 of 1.25e9 chained scores differ), so a key group costs 8 mma instead of 16 (s0003: -11 us at batch 1).

3. **L2 prefetch from an idle warp into windows where HBM is idle for every block** (track A, bit-identical: cache
   hints only). Warp 12 of each main block bulk-prefetches its slice of the next layer's q/k/v at the end of phase C
   and of the layer's shared expert at the start of phase A (batch 1-2 kernel). s0007 (first placement): -14 HTTP /
   -29 in process; s0008 (placement by the cycle-counter timeline): -11 more. Every other payload or window lost
   (K/V three ways, predicted / previous-token experts, dense MLP, output projection, own E row groups, batch 8):
   journal 39-53. Track B on this source is not faster than track A (the HSK=1 binary's allocation costs what the
   sketch saves), so the best engine is track A; track B stays buildable (engine/TRACK = B).

4. **Buffer layout of the serving process** (track A, layout only). The evaluation's HTTP engine process allocates
   nothing on the device before the Engine; the engine's small tensors then land where the batch-1 step is ~30 us
   slower than in any test process that had made one small allocation first (journal 62). Found with a two-process
   copy of the serving loop (tools/sock_long.py) that reproduced the HTTP figure to 0.5 us. A 512-byte allocation at
   the start of Engine.__init__: HTTP 1452 -> 1422, now equal to the in-process figure.

5. **Cache policy on the streams read once** (track A, bit-identical: hints only; s0013-s0014). The routed experts'
   and the exact head's tiles are loaded with an L2 evict-first policy (createpolicy + ld.global.L2::cache_hint,
   without L1::no_allocate): B=8 -69, B=1 -6. The same hint on every other stream was neutral or lost (journal
   111-120).

6. **The last layer's idle prefetch slot fetches the head** (track A, s0015). A layer entry past the last whose q/k/v
   pointer is the head: the existing end-of-C prefetch statement fetches the head's first 20 MB in the last layer
   (-4). Any new kernel code at that site grew the batch-1 frame 176 -> 192 B and lost (journal 122-125).

7. **Track B ahead again** (s0016). With the 8-mma scores form the track-B build draws the same 176 B frame as
   track A, and the sketch's ~16 us of head time shows: ~6 us ahead of track A (journal 128-131). Every submission
   since is a track-A kernel change submitted first, then the same source as track B.

8. **The dense layers** (layers 0 and 1; s0017-s0022). They took ~51 us each against ~27 for a MoE layer (timed per
   layer type with tools/bt_gen.py): their MLP (75 MB) loaded tile by tile, nothing preloaded before the barriers,
   and a slow single-warp epilogue. 16 gate/up and 16 down tiles per warp preloaded before the barriers (s0017/s0018:
   -20 in track B), the down preload moved behind warp 0's writes, which became plain stores with one fence (s0019/
   s0020: -5), the rest of the gate/up loop unrolled by 16 (s0021/s0022: -3). Dense layers now ~40 us each.

9. **The sketch's pass 2** (track B only, s0023). Stamped per block, the head's end waited ~8 us for one block: the one
   holding the step's candidate row group, recomputed on two warps in four load rounds (~9 us). All 16 warps now load
   the candidate's tiles into shared memory in one round and the same two chains run from there: 3.6 us, -4.

## What was measured and dropped (journal items)
- 12-bit weight code for the layers: the decode at the mma costs 3-5x what the bytes save (probes, both batch sizes).
- Layer-0 per-token tables: nothing at batch 1 (layer 0's phase A is not on the critical path).
- Attention split over two SMs at batch 1, two designs. The second has batch-invariant two-half arithmetic
  (bit-identical between one SM and two). It removes 70% of the per-position slope but adds ~2 us per layer of
  merge and signalling: net +100 us.
- K/V and expert prefetch to L2 by bulk prefetch: the prefetch traffic costs more than it saves (+62; +86..+212).
- Register-allocation draws: compiler flags, barriers, call-site variants, separate kernel instances; the batch-1
  kernel's allocation moves the step by +-10..50 us and dominated every small change (journal 19-21).

## Where the time is now (s0022, journal 151)
Per MoE layer ~26 us (A 3.2 | wait for the pairs 1.4 | attention + C 8.9 | D + router 1.9 | routing 2.3 | E 5.0 |
F 3.0); layer 2 (the first MoE layer of a step) ~5-6 us more (instruction fetch; blocks 112-127 late in E, journal
138-139); the dense layers ~40 us each (gate/up 7.8, warp 0's epilogue ~5, down 4.5); head + step boundary ~73 us
(track B sketch ~63: prep 5, pass 1 50 at 6.4 TB/s, publish 3, exchange 1.3, pass 2 1.5).

## Where the time was (notes/gap.md, s0009)
Batch 1, s0009: ~1420 us per token over the scored workload (HTTP = in process now): ~690 us of bytes at the
practical rate, ~500 us of per-layer skeleton (six grid-wide hand-overs, routing, the attention's fixed stages),
~125 us of context-dependent attention, ~80 us head (exact BF16 head; track B's sketch does not pay on this source).
The step grows from ~1320 us at position ~200 to ~1495 at ~960 (0.17 us per position of context).

## Method notes for whoever continues
- Measure the served path the way the scored process runs it: tools/sock_long.py (two processes) or
  tools/gaps_long.py (numpy prompts, nothing allocated on the device before the Engine). A test that allocates
  first can sit in another buffer layout (journal 62-70: +-30 us).
- /tmp is shared by the pod's cards: per-card names in any A/B script (journal 49).
- The batch-1 kernel's register allocation moves the step by 20-80 us on most edits; the source is at a local
  optimum of every existing bit-identical switch (journal 58, 61).

## Tools added (second half)
tools/ab2.sh (calibrated alternating A/B of two builds), absrc.sh (a source file against engine/), abserve.sh,
abenv.sh, sweep.sh (SW_B batch size), clkcmp.sh (MK_CLK timelines), res.sh / spill_lines.sh (stack and local-memory
lines per source line), bench_l2pf.cu (cross-SM L2 prefetch), route_rep.py, sock_long.py / gaps_long.py /
rows_long.py (served path as the scored process runs it), frag_long.py, nvml_long.py; patches of closed experiments:
mk_qpfx.cu.bak, vdef.patch, carve.patch.

## Tools added
tools/head_study.py, hsk_verify.py, hsk_stats.py, hsk_bsizes.py, gen_cmp.py (served cohort fixed, journal 22),
lcmp.py, dbgcmp.py, tf_time.py, mkprobe.py + probe_ab.sh, abgt.sh, bench_i8.cu, bench_hsk.cu, bench_mma.cu,
bench_mmaT.cu, w12_census.py. Engine switches: TRACK A/B/F (F = the previous run's float32 stream), MK_HSK_OFF,
MK_LIB, HSK_STATS.
