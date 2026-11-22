# Gap accounting (merge run s17-mini-b200-ma), updated after s0004

Engine (s0004): engine/a2 kernel (s15-mini-b200-a2 s0025, compiled KB1 + inactive mark sites) at B=1;
engine/a1 kernel (s15-mini-b200-a1 s0025 + GZ0/GZK + MSGW/MSGK) at B>=2.
HTTP 2722 us (B=1, 1.36x vLLM's 3707) / 5355 us (B=8, 0.95x vLLM's 5105). Targets 3370 / 4641.

## B=8 (a1 kernel): step ~5.35 ms HTTP; in process 5495 us (generate 512 from natural prompts, card 1)
Floor: 48 apps x ~37 union experts = ~9.4 GB -> 1.5-1.7 ms at 5.5-6.5 TB/s.
Spans per application, profile build (marks inflate ~15%), natural text, position 576 (e0060):
| phase | us/app | us/step | what bounds it (measurement) |
|---|---|---|---|
| E w13 stream | 32.7 | 1570 | ~125 MB/app = 3.8 TB/s; decode loop 540 SASS instr / 40 HMMA (~54 per 576-B item, = the essential count); ~43 items/us/SM; 2 warps/SMSP |
| F w2 stream | 18.9 | 908 | ~62 MB/app |
| F stage + E pf_down (w13 tail, act TMA) | ~11 | ~530 | hand-off w13 -> w2 |
| B attention | 12.6 | 603 | one grid round, pieces of 128-key chunks; K/V ~19 MB/app |
| A qkv GEMV | 9.6 | 460 | 40 MB/app = 4.2 TB/s |
| A norm (mixer mix + rows; gated: norm of xg) | 9.5 | 456 | mixnorm ~6.4 us latency + hand-off |
| gated extra (A0, A1, D0, D1, M) | ~45 per gated app | ~1080 | ablation before GZ0/MSGW: 58 us per gated app; now -128 us/step |
| C o_proj, D mix, router, top-8, fin | ~22 | ~1050 | |
| head | 142 | 142 | 0.92 GB at 6.5 TB/s |

Largest named losses: (1) MoE ~3000 us/step, ~1300 above its byte floor: the decode loops are at their essential
instruction count but run at ~30% issue (2 warps/SMSP, latency-bound); every variant that changes them (deeper
rings RS 3-4: +280; early slot refill EISS: +324, spills; SCALE_BF16: +35; out-of-line loop: +470) loses.
(2) gated applications ~1080 us/step (latency chains: mix -> z -> x' -> norm, 3 grid barriers).
(3) attention ~600 us/step.

## B=1 (a2 kernel): 2722 us HTTP; in process 2670 (generate, ctx 64-320), 2714 (w1-like generate, 64-1088)
Critical path (crit.py, KB1 profile build, pos 576): plain application 43.9 us (attn_in 4.3, qkv 6.6, attn 8.9,
oproj 3.5, mlp_in 3.5, router 0.8, routing 3.1, w13 6.4, w2 5.5, w2b 1.5); gated 66.0 us (+ gate_d 6.8, gate_fg 2.4,
gate_norm 8.1, mgate_d 8.5, mgate_fg 1.7, mgate_norm 3.3); head ~137 us. Floor 0.69 ms (nominal) / ~1.0 ms at the
measured streaming rate: the step is ~1.7 ms of hand-off and latency chains.

## Code generation (the main finding of this run; journal.md)
Both kernels' speed follows ptxas' global register allocation more than the work changed: a1's NVFP4 decode loop
compiles to 540 or 583 instructions per 40 HMMA depending on unrelated code; a2 with KB1 is 2743 or 3120 us
depending on whether inactive mark sites exist. Keeping an old path compiled behind a never-true runtime condition
restores the good allocation (GZK, MSGK); tests/sasssearch.sh checks a variant's decode loops in 20 s.

## Update after s0007 (HTTP 2729 / 5231 us; B=8 0.976x vLLM, target 4641 still -590 us away)
- s0006 SCIM 4 (-76) and s0007 FP8D (-48) came from ncu's pipe counts: the expert decode loops were ALU-pipe bound
  (ALU 788 M vs FMA 425 M warp instructions per step). After them the NVFP4 loop is issue-bound at ~54
  instructions per item with its ALU part at the minimum of this decode scheme (16 F2FP + 8 PRMT).
- Stall picture at B=8 (ncu, s0006 build): decode loops ~18% of warp samples; block barriers behind grid-barrier
  and flag waits 35-45% of the qkv / attention / o_proj+D regions; out-of-line norm_f32_rows_w 4.7% (long
  scoreboard), mixnorm<1,1> 4.3% (i-cache 32%), mixnorm<3,3> 1.9% (i-cache 74%).
- Barrier variants (two-level +300, release arrival +7) and one-round norms (+28..+112) do not pay.
