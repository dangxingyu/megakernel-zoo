# Where a step's time goes (Qwen3-32B, one B200 = GB200 card, 152 SMs, 129 MB L2)

Step time = floor + named losses. Each line names its measurement.

## Floor (hour 0.7, card 1, bench/bw.cu, 32 GB buffer, per-block contiguous regions, scaled to 64.0 GB)
| loader | rate | 64 GB pass |
|---|---|---|
| ld.global.cs v4, 1024 thr, 4 loads/thread in flight, + mma | 7616 GB/s | 8403 us |
| same, 8 in flight | 7562 | 8463 |
| same + 132 grid barriers (one per ~245 MB) | 7616 | 8403 (barriers free in this form) |
| 768 thr x 8 | 7627 | 8391 |
| TMA bulk ring 8 x 16 KB, 512 thr | 7555 | 8471 |
| TMA ring + 107 grid barriers (consumers only, producer runs ahead) | 6315-7019 | 9118-10135 (worse!) |

Practical floor: 64.0 GB at 7.6 TB/s = **8.4 ms** (B=1), B=8 + 0.35 GB K/V.
Targets: 10400 us (B=1), 10629 (B=8). Budget above floor ~2.0 ms per step = ~31 us per layer.

## s0001 engine (hour ~1.5), B=1, context 512, tests/prof.py (block 0 barrier stamps), card 1: step 11454 us
Byte floor per layer at 7.6 TB/s: qkv 105 MB 13.8 us, o 84 MB 11.0, gate|up 524 MB 69.0, down 262 MB 34.5 = 128.3 us.
| op | measured us | floor | loss | what (tests/prof2.py phase stamps, B=8 pos 0) |
|---|---|---|---|---|
| qkv | 22.4 | 13.8 | 8.6 | norm prep 3.4 (B=1) / 4.4 (B=8), epilogue 2.4, barrier+skew ~1, stream ~= floor |
| attention | 13.7 (6.6 at pos 0, 19.7 at 1024) | 0 | 13.7 | one block per (seq, kv head): K/V of 512 positions by 8 SMs |
| o | 17.1 | 11.0 | 6.1 | epilogue 2.7 (cut tile + x read), barrier |
| gate/up | 81.0 | 69.0 | 12.0 | prep 3.5, stream max-min 5 us imbalance, epilogue 2.9 |
| down | 41.3 | 34.5 | 6.8 | epilogue 2.7 |
| per layer | 175.5 | 128.3 | 47.2 | x 64 = 3.0 ms |
| lm_head | 218 | 205 | 13 | prep 3.4, epilogue 5 |
Total: 8.42 floor + 3.02 layers + 0.013 head + ~0.005 embed = 11.45 ms.
HTTP adds ~60 us over in-process w1_like (s0001: 11488 vs 11426).
Prefetch sweep (L2 prefetch of the next op's first bytes per block): 32 KB 11319, 128-256 KB 11006-11020, 512 KB 11273
(pos 0); paced (first 64-128 KB after the stream, rest one 64 KB piece per barrier poll): 11422-11502 vs 11454 at pos 512
(within noise).

## Hour ~4: skeleton (MK_DBG=16: weight loads replaced by constants, no prefetch), pos 512, tests/prof.py
| op | B=1 skeleton | B=1 real | B=8 skeleton |
|---|---|---|---|
| qkv | 11.8 | 22.1 | 13.2 |
| attention | 11.3 | 12.0 | 15.3 |
| o | 10.2 | 17.4 | 10.8 |
| gate/up | 22.8 | 80.9 | 24.4 |
| down | 16.3 | 41.8 | 16.8 |
| lm_head | 49.9 | 219 | 51.3 |
| step | 4687 | 11376 | 5205 |
Fixed cost per GEMM op (prep + epilogue + barrier) ~7-8 us is not overlapped by the weight stream.
Ablations (B=1 pos 512): cut-tile exchange skipped (MK_DBG=4, wrong results) 11376 -> 11159 (-0.9 us per GEMM op);
residual read in the RESID epilogue skipped (MK_DBG=8): no change.
Attention alone (tests/attn_only.py, split version): B=1 8.2 / 11.0 / 12.2 us at pos 64 / 512 / 1024; B=8 9.9 / 15.1 / 20.7.

## Hour ~2.8 (board clock), 512-thread engine (s0002 + producer ssq + attention->o counter), pos 512, tests/prof.py
B=1 10674 us: qkv 19.5, attn+o 26.4, gate/up 77.6, down 39.8 per layer; lm_head 215. Loss per layer 35 us.
B=8 11396 us: qkv 22.1, attn+o 31.5, gate/up 80.6, down 40.4; lm_head 218. Loss per layer 46 us.
- Norm prep ablation (MK_DBG=32, wrong results): B=1 -0.15 ms, B=8 -0.47 ms per step (B=8 qkv 22.1 -> 18.3 us);
  of the B=8 prep, compute is 1.2 us (MK_DBG=64) and 2.5 us is reading 80 KB of residual per block + syncs.
- Grid barrier alone (bench/gbar.cu, idle card, 152 x 512): 1.17-1.25 us; arrivals over K counters (bench/gbar2.cu)
  1.27-1.47 us: latency, not atomic serialization.
- L2 hit latency (bench/lat.cu, pointer chase, unloaded): 300 cycles.
- Prefetch into L2 during attention (idle blocks 590 KB, waiting blocks 128-590 KB): attention+o 26.4 -> 27.6-29.8 us
  (worse): queued DRAM traffic delays attention's K/V loads.
- K/V first round preloaded before the prep, 4 P x V tasks per round: no gain (spills).

## Hour ~5 (board 4.8): per-op tail and epilogue (tests/tail.py, block stream-end stamps after the partials are
stored), track A, pos 512. Columns: start skew | median stream end | last - median | spread | epilogue after last.
B=1: qkv 0.17|14.48|1.11|1.85|1.80  o 10.41|21.00|0.93|1.69|1.53  gu 0.17|69.05|2.56|4.27|2.27
     down 0.20|33.16|1.85|3.00|1.56  lm 0.25|203.50|3.75|6.50|6.00
B=8: qkv 0.17|14.96|0.96|1.64|1.82  o 11.98|23.28|0.91|1.70|1.92  gu 0.15|70.58|2.55|4.27|2.48
     down 0.27|33.64|1.91|3.08|1.88  lm 0.25|207.25|4.25|8.75|6.25
Reading: the median block streams at the byte floor (gu 69.05 vs 69.0 us); per op the loss is the slowest block's
tail (1-2.6 us) + epilogue after it (1.5-2.3 us, lm_head 6) + barrier (1.2): ~13.6 us/layer = ~0.87 ms/step;
the o projection's blocks start up to 10-12 us apart (attention).

## Final accounting (board hour ~8), in process at pos 512
Track A B=1 10387 us = floor 8420 (64.0 GB at 7.6 TB/s) + per op: tails 1-2.6 us + epilogues 1.5-2.3 us +
3 barriers x 1.2 us per layer + norm prep 1.4 us x 3 + attention ~10 us/layer + lm_head tail/epilogue ~10 us.
Track A B=8 10775 us: as B=1 plus attention +3 us/layer (K/V of 8 sequences, more parts per block), on-the-fly
norm +1.5 us per gate|up, lm_head +4 us.
Track B B=1 9425 / B=8 10092 us: floor 6.31 ms (47.98 GB); medians stream at 93-97% of 7.6 TB/s; escape-heavy
layers 1-5 add ~100 us; B=8 pays the norm prep (2.3-3.7 us per norm GEMM) since the per-k-step norm does not fit
next to the 12-bit decode.

## Hour ~12: B=8 per-block timeline after the attention and barrier changes (tests/prof4.py, prof5.py, pos 640)
q/k/v op: barrier wait ends ~4.3 us after the first block arrives, stream end 17.5 (median) / 18.5 (max), epilogue
20.3; attention: prep 22.5, scores 24.9, softmax + P x V 29.6, cluster fold 30.5, sync 31.25, output 31.5 (max);
o projection, gate|up, down: stream within ~0.3 us of the byte floor after the wait. B=8 - B=1 per layer (barrier
mode): attention +2.0, o +1.3, gate|up +1.2, down +0.5; lm_head +4.2.
Workload means (tests/wl.sh, positions 128/384/640/896): track A B=1 10262, B=8 10487; track B B=1 9301, B=8 9836.

## Final accounting (board hour ~17.5), tree = s0033's source
HTTP: track A 10121 / 10440 us (B=1 / B=8), track B 9035 / 9495 (s0032; same library as the tree's track-B build).
Byte floors at 7.6 TB/s: A 8.42 ms, B 6.31 ms. Overheads over the floor: A B=1 ~1.70 ms, B=8 ~2.0 ms; B B=1 ~2.7 ms,
B=8 ~3.2 ms. Where they are (per layer, tests/prof4-6.py):
- per GEMM op: slowest block's tail 1-2.6 us + epilogue (reduction, cut-tile hand-over) 1.5-2 us + barrier 1.2 us;
- norm prep (B<=2 both tracks, track B at all B): ~2 us after the barrier at B=1 (L2 latency under the weight
  prefetch traffic), track B B=8 ~2.5 us with the cluster split;
- attention chain at B=1 ~8.5 us (prep 2.5, scores 0.9, P x V 1.8, combine ~2, signal 1.4); at B=8 ~11 us;
- track B streams: norm GEMMs 95-97% of the 12-bit floor, o / down at B=8 76% / 87% (inputs from L2, 2 quads in
  flight: 3 quads need the inputs' registers; L1 just-in-time reads did not hold).

## L2-to-SM read rate (bench/l2bw.cu, 152 blocks x 512 threads, each re-reading its own L2-resident buffer)
| buffer per SM | 16 B loads in flight per thread | no DRAM traffic | half the warps streaming DRAM |
|---|---|---|---|
| 80 KB | 1 | 42.8 GB/s per SM | 15.8 |
| 80 KB | 2 | 64.7 | 20.2 |
| 80 KB | 8 | 66.6 | 42.0 |
| 40 KB | 8 | | 33.7 |
| 10 KB | 8 | | 17.7 |
~66 GB/s per SM (~10 TB/s over 152 SMs: only ~1.3x the DRAM stream) is the ceiling; under DRAM traffic a block's L2
reads get ~20-40 GB/s. This is why an 80 KB norm prep takes ~2 us, why one block combining all partials was slow,
and why track B's o / down at B=8 stay at 76-87% of their floor: 12-bit weights through L2 (~43 GB/s per SM) plus
their inputs re-read per tile (~28 GB/s per SM) exceed the per-SM L2 rate. Reducing those reads needs one warp to use
each input fragment for two tiles (a stream-loop rewrite).

## Final tree, per op (tests/prof.py, MK_ATT_FLAG=0 so attention and o are separated by a barrier; pos 640; us)
Byte floors at 7.6 TB/s per layer: track A q/k/v 13.8, o 11.0, gate|up 69.0, down 34.5; track B 10.4, 8.3, 51.7, 25.9.
| | step | q/k/v | attention | o | gate|up | down | lm_head | embed |
|---|---|---|---|---|---|---|---|---|
| A, B=1 | 10356 | 17.7 | 10.4 | 15.9 | 76.5 | 37.9 | 213.9 | 6.5 |
| A, B=8 | 10703 | 18.8 | 12.9 | 16.2 | 78.0 | 37.9 | 215.4 | 6.1 |
| B, B=1 | 9276 | 17.0 | 10.9 | 17.1 | 62.5 (max 74.7) | 34.6 (max 41.0) | 172.4 | 6.9 |
| B, B=8 | 9753 | 18.8 | 13.6 | 17.2 | 64.4 (max 74.2) | 35.6 (max 42.2) | 173.5 | 5.9 |
(In barrier mode the step is ~250 us longer than with the dataflow waits the engine uses.) Over the floor per layer:
track A B=1 ~33 us (of which attention 10.4), track B B=1 ~45 us; the maxima of track B's gate|up / down are the
escape-heavy layers 1-5.
