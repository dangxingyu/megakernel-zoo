# Where the step goes (B200, card 1-3 of the pod, inherited engine s0077, track A)

All numbers measured in this run (2026-11-16). gen_long = tools/gen_long.py (prompt 64 + 1024 greedy tokens by
generate(), the scored workload in process); tf = tools/tf_time.py (teacher-forced step(), host launch per step).

## Batch 1: 1467 us per token (gen_long, e0001/e0004; HTTP headline of s0077: 1480)

| item | us | from |
|---|---|---|
| floor: 5.05 GB at 7.3 TB/s (practical rate, notes/prev findings 10) | 690 | prev run bench10-12 |
| skeleton with every weight tile read from a 512 KB L2-resident window (WSKEL=3) | 1142 | e0003 |
| ... of which: layer weights from L2 instead of HBM (WSKEL=1 vs base) | 266 (layers) | e0003 |
| ... head from L2 instead of HBM (WSKEL=2 vs base) | 56 (+ ~30 of L2 reads still in the skeleton) | e0003 |
| head standalone loop (bench_i8.cu, 615 MB, same loop shape) | 109 (5.6 TB/s) | e0020 |
| layer at pos 200 / 900 (MK_CLK, B=1): A 3.2, wait for pairs 2.5, attention+C 5.1/8.0, C->D 1.1, D+router 2.0, routing 2.3, E 5.0, F 3.2 | 24.8 / 27.4 | e0002 |

Reading: the weight stream above an L2-speed stream costs ~320 us of 1467; the rest is the skeleton (per-layer
chain of hand-overs, attention, routing: ~1150 us). Lossless byte reduction (track B) can only act on the first part.

## What fewer bytes buy, measured with probes (tools/mkprobe.py, results wrong on purpose; tf B=1 pos 64-320 / 700-956)

| probe | B=1 | B=8 (forced routing, WHATIF=1) |
|---|---|---|
| base | 1378 / 1536 | 3392 / 3579 |
| all weight batches load 3/4 of their tiles (bytes of a 12-bit code, no decode) | -87 / -78 | (confounded) |
| expert batches only at 3/4 | -35 / -35 | -221 / -173 |
| expert mmas with a 15-op decode in front (full bytes) | +127 / +157 | — |
| expert batches at 3/4 + decode | +122 / +133 | +233 / +418 |

=> a 12-bit code with the decode at the mma costs 3-5x what its bytes save, at both batch sizes (the expert phases
are short: 1.5 batches per warp at B=1, nothing to overlap the decode with). W12 for the layers: dead end here.

## Update after s0003 (track A, KQT) and the head sketch (2026-11-16, agent clock ~2.3 h)

Batch 1, scored workload, in process (gen_long, same card): track A 1471-1477 us; track B 1463-1472.
HTTP: s0003 (A) 1476.7, s0002 (B, older kernel) 1472.5.

| item | us per token | measurement |
|---|---|---|
| floor: bytes at the practical rate (5.05 GB at 7.3 TB/s) | 690 | prev run bench |
| head, exact (track A), block 0 from norm to publish | 66-79 | hsk_stats (-DHSK_STATS) |
| head, int8 sketch (track B): prep 4-6, pass 1 49, publish 3.3, exchange 1, pass 2 1.4 | 63 | hsk_stats |
| context-dependent attention (0.25 us per position; mean position 576) | ~125 | tf_time, HALFK (-95 for half the keys) |
| layer at position 900 (B=1): A 3.2, wait for the pairs 2.5-2.8, attention + C 7.4-8.0, C->D 1.1, D + router 2.0, routing 2.3, E 5.0, F 3.2-3.3 | 27.4 | MK_CLK |
| of which the attention itself (q/k/v norms 0.8, scores 2.1 (KQT) / 2.4, weights 1.3-1.4, V mix 1.1, out 0.4) | 6.1 | MK_CLK attention blocks |
| weight bytes above an L2-speed stream | ~320 | WSKEL |
| allocation draw of the batch-1 kernel (same code paths, different builds) | +-10..50 | many builds in journal 19-21 |

Named losses that remain, largest first: per-layer skeleton (hand-overs ~6 x 1 us, routing 2.3, router 2.0 per MoE
layer: ~500 us), context-dependent attention (~125 us), weight bytes above L2 speed (~320 us; W12 measured dead,
head done), head sketch overheads (~10 us).

## Upper bounds measured with probes (results wrong on purpose; gen_long B=1, same card, base ~1475)

| what is removed or made free | B=1 us | source |
|---|---|---|
| all layer weights from an L2-resident window | -266 | WSKEL=1 (e0003) |
| expert tiles (E, F) from an L2-resident window | -109 | mkprobe WSKEL=64 (e0128) |
| half of every pair's keys | -95 | HALFK=2 (e0042) |
| routing computation and the wait for the router logits | -37 (worse allocation) | RTP (e0133) |
| the shared expert's down projection | -14 | SDP (e0132) |
| layer 0's phase A (attention reads stale q/k/v) | +4 | L0P (e0069) |
| 3/4 of the bytes of every weight batch (W12 without decode) | -87 | mkprobe WSKEL=4 (e0016) |

## Update after s0008 (2026-11-17, agent clock ~8 h)

Batch 1 in process 1417-1427 (gen_long / the evaluator's w1_like_generate), HTTP 1452. Idle-warp L2 prefetch into
windows where HBM is idle for every block paid -40 us in process in total (s0007 + s0008); the routing stragglers
(blocks 112-127, delayed by prefetch traffic in the routing window) are gone. Probes on the attention (journal 55):
the key groups beyond the staged 28 cost ~45 us per token at the scored positions, 15 of it the trips to memory and
the rest their dependent mma chains; L2 prefetch of them changes nothing (the trip from L2 is as long as the chain).
