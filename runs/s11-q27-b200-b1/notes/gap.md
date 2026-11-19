# Where a step's time goes (track B, B=1 unless stated; in-process, context ~600)

Step time = floor + named losses. Every number names its measurement. Updated after every profile.

## Inherited engine (s9 s0022 source, TRACK "A W12"), hour 0.5, card 1
- events (tools/tv.py, NO_STAMPS build): B=1 7678 us, B=8 8140 us; track A same card 8456 / 8868; logits hash equal
  (f74e8d8fae06) for both tracks.
- stamped build per op (tools/b8_split.py): pass 7747 us. GEMMs ~6730 us, lm_head 277, recurrent op 9.5 x 48 = 457,
  attention op 17.8 x 16 = 285.

| item | us | measurement |
|---|---|---|
| floor: 39.5 GB of weights (W12 + BF16 twins of out_proj/o_proj at B=1) at 7.27 TB/s | 5430 | s4q C1 plain-load rate; bench/tma12.cu mode 0 (BF16 + MMA, 36 quads/warp) 6.99 TB/s incl. a barrier per 51 us |
| skeleton (weight streams off, flag 8) | 1787 | prof_b.sh RUNS=1:8: GEMM ops 4.1-5.1 us each (x241 ~1.1 ms), recurrent op 8.4 x 48 = 404, attention 14.6 x 16 = 234 |
| of which overlaps with streaming in the real step | ? | step 7747 = 5430 + 1787 + 530: the two do not add exactly |
| W12 stream excess over BF16-equivalent per op (~3.7 us per W12 op: max warp stream end vs bytes/7.3 TB/s; BF16 ops 0-0.7 us) | ~550 | tools/wtail.py both builds: down 21.9 vs 18.3, in_proj 21.1 vs 17.3, gate_up 40.5 vs 36.6, lm_head 270.8 vs 261 |
| of which escape patching | 135 | MK_W12_DROP_ESC=1 (wrong results): 7747 -> 7612 |
| lm_head (W12, 1.905 GB) | 277 | b8_split; int8 sketch would read 1.27 GB (tools/head_census.py) |
| after the slowest warp's stream end, per GEMM op (finalize, epilogue, barrier) | 3.1-3.8 per op, ~0.85 ms | tools/wtail.py |

## Negative results
- W12_PFL (barrier thread 32 asks L2 for the next W12 GEMM's escape lists + row bases before its weight pieces):
  B=1 7675 -> 7808, B=8 8140 -> 8270 (tools/abB.sh, 3 rounds, card 1). The hypothesis "cold list/ctab loads hold
  each warp's first decode" is not supported by this form; the extra dependent prog loads on thread 32 may be what costs.

## After s0001-s0003 (hour ~3.5 of wall time, agent-hour ~1.6)
- W12_RS removed the row-base loads (-106 us B=1); L0 table removed layer 0's in_proj op (-15 us).
- lm_head on greedy passes: exact W12 head 272 us -> sketch ~175 + rescore ~20 (in its own launch since s0004:
  compiled into the main kernel its op cases cost +28-59 us per pass in code generation).
- Measured upper bound, not yet realised: epilogue per-row constants as constants (EPI_DBG_CONST) -57 us B=1 / -87 B=8;
  the smem-prefetch implementation (EPI_PF) gave +33 / -18 (code generation again).
- Main-kernel code generation is a loss source of its own: +11 to +59 us for code that never runs. Any change to the
  main kernel is now A/B'd against the previous submission's kernel source.

## s0013 build (hour ~9.5 of agent time), B=1, nsys (tools/main_nsys.sh, generate after a 64-token prompt)
| item | us | measurement |
|---|---|---|
| main kernel per greedy pass | 7062-7063 | median of 387 launches, two runs |
| head kernel (7-bit sketch 1.113 GB + exact rescoring) | 188 | median of 324 launches |
| sum (in process) | 7250 | HTTP 7263 (contexts 64-1088; GPU idle 0.025 %, one launch pair per step) |
| head: floor of its bytes at 7.3 TB/s | 152 | 1.113 GB |
| head: prologue (x stats, digits) | 4.4 | SK_PROF B=1 (B=8: 14.8) |
| head: block / warp imbalance of the stream | ~14 | block end max vs warp median (SK_PROF) |
| head: rescore op (summaries 1.8, scan to 6.9, chain ~8) | ~15 | SK_PROF, after KPT=8 |
| main: escape mechanism (check + slow path) | 178 | W12E_DBG_NOESC bound (wrong results) |
| main: per-GEMM tails (imbalance, hand-over, barrier) | ~1000 | s9 profile; fixed by the static partition (F12) |

## s0020 build (agent-hour ~15.7), B=1, contexts to 1088 (nsys, generate 1024 steps after a 64-token prompt)
| item | us | measurement |
|---|---|---|
| pass period (main + head + gaps) | ~7248 | nsys start-to-start; HTTP 7238 (score), in process 7225 |
| main kernel | ~7065 | median of 1000 passes; 7062 at contexts <= 364 |
| head kernel (mixed 6 / 7-bit sketch, 0.96 GB + rescoring) | ~181 | median 179-182, mean ~183 (rare passes with thousands of candidates) |
| head: sketch op (last block) | ~157 | SK_PROF hk_prof.py: bytes at 7.2 TB/s ~133, prologue ~5, imbalance ~19 |
| head: rescoring (summaries, scan, chain) | ~17 | scan median 4.6 / mean 7.5 us; chain of 320 MMAs in 40 relay turns ~8-9 |
| gaps between kernels | 0.26 x 2 | PDL does not shorten them (and costs main-kernel code generation) |
| B=8: main / head | ~7500 / ~206 | head through the all-7-bit copy (s0018) |
| main kernel after s0027 (LM_VOLC) | B=1 ~7005 (card 1) / ~7028 at NSTEP=1024 | -31 / -38 against s0026 (main_nsys, 3 rounds); HTTP 7221 |
| head kernel after s0024-s0026 (6/8-bit sketch, u32 x loads) | B=1 ~175.5 / B=8 ~187 | head_nsys medians; prologue B=1 ~4.9, B=8 ~10.5 (SK_PROF); s0026 HTTP 7250 / 7692 |
| final head, stamped (SK_PROF, B=1, real text): entry -> sketch start | 1.8 | hk_prof.py, medians over 256 passes (stamped build ~+3.5 us overall) |
| final head: sketch prologue (x 2.8, digits 2.0) | 4.9 | sk_prof.py; B=8 10.8 (x 4.9, digits 5.9) |
| final head: stream, warp median end / block max median / last block | 139.3 / 146.9 / 153.6 | bytes ~0.97 GB at ~7.2 TB/s; warp 0 ends at ~92 (first bytes in L2) |
| final head: sketch op end (last block) -> barrier -> rescore end -> finish end | 157.7 -> 158.7 -> 175.6 -> 179.2 | rescoring ~17 (candidate blocks 78, jobs 177), finish ~2.3 |

## Per op at the end of s11 (stamped build of the final flags, step() at B=1; tools/ops_ctx.sh)
| op | us per op | bytes at 7.3 TB/s | over the bytes | count | us per pass |
|---|---|---|---|---|---|
| gate_up | 42.0 | 36.6 (267 MB) | 5.4 | 63 | 2657 |
| down | 24.0 | 18.3 (134 MB) | 5.7 | 64 | 1537 |
| in_proj (linear layers) | 22.5 | 17.3 (127 MB) | 5.2 | 47 | 1058 |
| qkv (attention layers) | 20.6 | 15.0 (110 MB) | 5.6 | 16 | 330 |
| out_proj / o_proj | 12.5 / 12.8 | 6.5 (47 MB) | ~6 | 47 / 16 | 791 |
| recurrent op | 9.3 | - | - | 47 (+1 with layer 0's) | 438 |
| attention (one op), context 100 / 600 / 1000 | 14.9 / 17.0 / 17.5 | - | - | 16 | 239-280 |
Every 12-bit GEMM carries ~5.5 us over its bytes (the stream's ramp, its tail, the hand-over, the barrier): ~1.3 ms
per pass, the largest remaining loss, held in place by the static partition and bit-identity (F10, F12).

## Per op with the s0027 flags (tools/ops_ctx.sh, stamped build, hour ~23.6)
- us per op at contexts 100 / 600 / 1000: gate_up 41.9 (A 42.4), down 23.9-24.2, in_proj 22.5, out_proj 12.5 / o_proj 13.0,
  qkv 20.5, recurrent op 9.3-9.4, attention 15.2 / 16.7 / 17.2: the same as the end-of-s11 table above.
- Caution: the stamped build does not show LM_VOLC's -31 us (unstamped main_nsys and the evaluation do): the stamps
  change the kernel's code generation (F22), so stamped per-op numbers cannot resolve effects of this size; the tool's
  pass total is also broken (prints a negative number) with these flags.
