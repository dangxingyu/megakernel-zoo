# Step time = floor + named losses (updated after every profile)

## Current (2026-11-18, ~15 agent-h): track B s0042-s0044 1634 / ~3395 (HTTP); in-process gen B=1 ~1628
B=1 profile of the B=1 build (e0443, ctx 500, us per layer, span / wait): gate_up 17.3 / (after-gu wait 4.4), down
stream 10.9 / 3.6, qkv 9.7, o_proj stream 6.7, attention 6.0, qkv epilogue 4.8 / 2.1, o_proj epilogue 4.4 / 1.7, E entry
(routing + norm2) 3.0, norm1 2.4, top-k 1.0, union 0.6, staging 0.7. Head (SK5): pass 1 ~155-160 (load floor 146, bench
sk5bw), threshold 6.4, list 5, recompute ~8 per tile round, argmax / barrier.
Waits (slowest block minus typical) ~13 us/layer: per-step random variance (e0453), not static.

## Current (2026-11-17, ~9 agent-h): track B s0028 1692 / 3283; track A s0029 1924 / 3444 (HTTP)
B=1 per layer ~61 us (in-process 1672 - head ~200, / 24). Measured spans (e0284, e0307, e0316, e0320):
| item | us/layer | nature (evidence) |
|---|---|---|
| gate_up | ~17 | HBM-bound after a ~1 us task start + ~2-3 us cold first data (trace e0316; per-SM demand 36 GB/s > HBM share 23 GB/s) |
| down | ~12.5 (2.5 wait) | per-warp math ~1.1 us/stage + slowest-of-4-units combine (trace e0309, DPAIR e0322); L2-resident weights only -2 (SKEL e0307) |
| qkv | norm 2.7 + stream 8.2 + epilogue 1.5 (+3 wait) | wait = service variance (e0291) |
| attention | 3.7 sliding / 4.8-9.1 full (ctx 300-1000) | per-warp key loop of dependent L2 round trips; 4 split-key variants lost |
| o_proj | load 0.9 + stream 6.8 + epilogue 1.9 (+1.8 wait) | stream not HBM-bound after ring prefetch (OPF e0293) |
| routing + norm2 | ~3 | norm2 long pole |
| barriers | 5 x ~1.05 | release-add counter is the fastest variant (bench/gbar2.cu) |
Head (track B): pass 1 ~150 (434 MB, byte-proportional, e0262) + list 6 + exact recompute 26.

## Older accounting (kept for the record)


Floor: B=1 3.72 GB at 3.05 TB/s (bench/bw.cu, e0027) = 1220 us (layers 2.56 GB = 35 us/layer, head 1.16 GB = 380 us).
Track B head bytes: 0.58 GB sketch -> floor 1030 us.
B=8 on real text: union 15.4 experts/layer mean (12-20; tests/union.py e0033) -> 2.44 + 4.9 GB = 2.4 ms.

## Current: s0004 track A 2254 / 3850; s0005 track B 2061 / 3671 (HTTP; HTTP ~ in-process w1_like_generate + 6 us)
B=1 per layer (prof e0079/e0100, gen ctx 64, random prompt), span us (work / wait):
| phase | span | work | wait | bytes | loss/layer | cause / evidence |
|---|---|---|---|---|---|---|
| A norm1 | 3.8 | 3.6 | 0 | 0 | ~2.5 | one warp, two dependent L2 passes |
| A qkv | 12.2 | 8.3 | 3.5 | 9.7 | ~2.5 | per-warp stages (3 vs 2.4 avg rowtiles); balancing at block level did not pay |
| B attn (8 blocks) | 5.1 | - | 4.5 | ~0 | ~4 | 8 blocks; grows with context (+85 us/step at HTTP contexts, s0004 w1_like vs step) |
| C o_proj | 8.9 | 5.5 | 2.8 | 7.7 | ~1.5 | 1 vs 2 rowtiles |
| D norm2 + router (45 blocks) | 5.4 | 3.6 | 1.4 | 0.06 | ~4 | norm 3.6 us |
| E top4 + union | 2.9 | | | | ~2.5 | |
| E gate_up + after | 18 + 7 | 15 + 1.4 | 3.4 | 11.7 | ~13 | decode issue-bound (~86 instr/item, ALU pipe); NOCVT -345 us/step |
| F down | 16 | 11.4 | 3.9 | 5.8 | ~10 | same, 5-6 units per block |
| head (A) | 424 | | | 380 | ~44 | track B: pass 1 205 + exchange + pass 2 ~25 |
Sum of losses ~42 us/layer = ~1000 us/step; skeleton (SKEL build, weights from L2) 1659 us at B=1.
B=8: decode ~2 ms of 3.8 (NOCVT 2298 vs 4278 in e0041).
