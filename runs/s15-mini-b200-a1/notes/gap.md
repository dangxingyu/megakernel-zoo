# Step account (B=1 unless noted). step = floor + named losses, us per step

## s0001 (v1): HTTP 5047 us B=1, 14420 B=8; in process 4776 us (ctx 64-192) / 10623 B=8
- Floor (bytes): 5.54 GB per step -> 692 us at 8 TB/s nominal, ~760 us at 7.3 TB/s (playbook B200 plain BF16 stream).
- Skeleton (weights from a 512 KB window, -DSKEL, e0010): 4488 us B=1, 8542 B=8 (ctx 64-128) -> weight streaming
  adds only ~290 us at B=1: the step is latency, not bytes.
- Phase spans per step (mkprof e0010, B=1, pos 200; span = first block in -> barrier release):
  | phase | per step | span us | us/step | work (median block) | note |
  | B attention | 48 | 16.4 | 788 | 0.65 (imb 23x) | chunk/merge chains of dependent L2 loads |
  | F w2 | 48 | 14.7 | 706 | 11.7 | 12 MB -> 1 TB/s: MMA accumulation chain + decode, 8 warps |
  | E w13 | 48 | 14.5 | 697 | 11.5 | 24 MB -> 2 TB/s: same |
  | A0 q/k/v gate z | 24 | 16.1 | 386 | 6.6 (imb 2.3) | materialise ux/um + 4608-dots |
  | A qkv GEMV | 48 | 8.0 | 384 | 5.4 | 39.7 MB in 5.4 us = 7.3 TB/s: at the bound |
  | D0 mlp gate z | 24 | 14.9 | 357 | 5.8 | |
  | A norm (mix+norm) | 48 | 7.0 | 334 | 6.1 | serial L2 loads, 2-3 block syncs per row |
  | E top-8 + union | 48 | 5.8 | 280 | 5.3 | |
  | C o_proj | 48 | 5.3 | 253 | 3.1 | |
  | D mix+norm | 48 | 4.8 | 229 | 4.1 | |
  | D1, A1 gate x' | 24+24 | 7.1, 6.9 | 336 | | |
  | D router | 48 | 3.2 | 153 | | 8 blocks |
  | head | 1 | 140.5 | 141 | 132 | 0.92 GB in 140 us = 6.6 TB/s |
  | M messages | 2 | 5.0 | 10 | | |

## after the first optimisation round (in process 4109-4204 us B=1, ctx 64-128; e0036 profile, B=1 pos 200)
Per layer application ~85 us. Phase spans (us per occurrence): attention 13.3, E w13 stream 10.3 (+top8 6.2,
pf_down 3.7, epi 1.7), A qkv GEMV 8.9 (+mix 4.6), C o_proj 7.6, D mix 3.2 + router 3.5, F w2 4.7 + stage 2.5 +
accum 1.2 + conv 1.6; loop passes 2-3: A0 5.9, A1 5.6, D0 11.0, D1 7.3; head 140 us/step.
Named losses found by probes (e0037-e0040):
- one dependent L2 load = 0.23 us (fine);
- first-touch HBM loads of small weights ~1 us each, ~10 per layer (~10 us/layer): L2 prefetch added (untested)
- top-8: 8 dependent reduction rounds 1.7 us -> rank count (untested)

## s0012 (HTTP 3634 B=1, 6045 B=8; in process at ctx 64-320: ~3590 / ~6000) -- mkprof spans at pos 300, natural text
Floors: B=1 5.54 GB -> ~1010 us at the ~5.5 TB/s this kernel's rings stream (micro: 5 TB/s NVFP4 at full grid);
B=8 with the measured union (mean 37 experts per application, 22-53): ~9.4 GB -> ~1.7 ms.
B=1 per application (us, span): attention 12.3 | w13 ~8.7 (24 MB: latency-bound, one unit per warp; deep path) |
qkv 8.6 (40 MB: at the stream rate) | o_proj 6.4 (14 MB: ramp + 2 stages) | w2 4.5 | routing (router 3.6 + top-8 3.6
+ E tail 3.4) | F stage 2.8 + accum 1.3 + fin 1.6 | mixes: A 2.3, D 1.8 | loop gates: A0 1.0, A1 3.7, D0 3.5, D1 5.5.
  Named losses: ~8 grid barriers per application x ~1.2 us; attention item fixed costs (q/r loads up to 2.9 us,
  partial write + count + last-arriver merge ~3 us); routing chain ~10 us of mostly latency; head 140 us (0.92 GB).
B=8 per application: w13 33.7 (110 MB = 3.3 TB/s; warp imbalance ceil(4.4) = 5 units: ~12%) | w2 20.4 | F stage 7.4
+ E tail 5.2 (both mostly waiting on the slowest warps' w13) | attention 11.9 | qkv 9.7 | A mix 9.2 | D mix 6.1 |
o_proj 6.6 | top-8 5.6 | router 3.4 | loop gates (24 applications) A0 16.4, A1 7.3, D0 18.5, D1 9.9.
  Named losses: mixnorm ~6.4 us per mix on 8 mixer blocks (41% of its stalls are instruction-cache misses: 75 KB of
  unrolled code) + ~1.5 us flag hand-off, ~97 mixes per step; w13 warp imbalance ~4 us per application; MoE near the
  ring's streaming rate otherwise.
Things that do not move it (measured, see journal): bytes in flight (ring 10x3 / 16x2), any L2 bulk prefetch, warp
balance by splitting units, release/acq_rel instead of fences (except the attention count: long ctx only), L1 size,
added code paths (each costs B=1 ~60 us by layout/allocation even when unused).
