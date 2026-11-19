# Where the step goes (B200; updated 2026-11-18 ~29 agent-h; per-layer numbers from the engine of s0047, head of s0056)

Floor: 5.01 GB per B=1 step at 7.3-7.4 TB/s (measured stream rates) = ~690 us (track B reads 4.45 GB: ~610 us).
HTTP = decode window only (client.py: last stream's first token to first stream's last token), positions 64..1088.
Best: track B B=1 s0051 1686.6, B=8 s0056 2991 (engine = s0056); track A B=1 s0057 1759.7 (engine), B=8 s0050 3061.
Targets (1.10x SGLang): B=1 1997 met; B=8 3113 met in both tracks (B 3017, A 3061).

## The scored B=8 workload is not the bench's real text
The client's prompts are random common words + "Continue the text above..." (one per stream). Their continuations
route much more alike than the bench's real text: in process B=8 3142 us (tools/bench.py --synth 1) vs 3644 (real
text) vs HTTP 3125. Profile it with tools/prof_run.py --synth 1.

## Per layer (mkprof)
B=1 (pos 600): A 10.0, B ~6, C 7.0, D 3.8, E 10.7, F 9.1 + head ~110 (prep 7, sketch 82, tail ~8+).
B=8 synthetic (pos 150..1050): A 10.9-11.1, B 6.2-12.2 (grows with position: ATT_BLK up to 512, split units above),
C 7.7, D 4.8-5.1, E 29-34 + F 20-24 (E+F together ~6.8 TB/s: HBM rate), combine 2.1;
head ~118 = prep 9 + sketch stream 87 (6.95 TB/s) + barrier wait ~5 + filter 1.8 + row copy 1 + rescoring <= 8.5
(one warp per candidate tile: its mma/load chain, not bytes) + reduction.
Bound by: HBM in E/F at B=8; per-warp ring stage chains and per-SM bytes of the heaviest blocks in A (3-rowtile
blocks) and C (2-rowtile blocks); grid barriers (~1.2 us floor, 6 per layer at B>=3); attention item chains (~2 us
per load stage; partly HBM latency: the K/V cache falls out of L2 between steps -- KVPF, an L2 prefetch of each
block's K/V chunks in phase A, gives B=1 -14, B=8 -21 us in track B).

## Measured dead ends (details in journal.md)
L2 prefetch of weights in any form (also predicted experts on the HTTP-like B=8 workload) and of head
candidates (the K/V cache prefetch, KVPF, does help in track B); data-polling barriers; cluster two-level barriers; flat qkv/o split with last-arriver; o_proj K-quarter
split (OSPLIT); 8-row o_proj tiles; ring depth (also from HBM: RS 3 = RS 4 = 42 GB/s/SM); register copy of fp4
stages; 3 accumulators; split fp4 mma chains; predicated B loads; E/F halves; staged norm via TMA; rolled-loop norm;
router over 64 blocks / in halves; int4-6 head sketches; attention/o_proj overlap by counters (BCOV); partner split
of o_proj rowtiles (CSPLIT); down units as K halves (FHALF); early down prefetch (EPF); attention on cluster pairs
(ATT_CL2); 2 chunks per warp; wide combine by the last arriver (FCOMB); V by cp.async in the wide instance; router
blocks' norm in quarters around a split barrier (DQ); global candidate queue for the head rescoring (HSK_RQ);
pooled parallel rescoring in the wide instance (FASTRS_W); sketch prep with restrict pointers / unrolling.

## After s0042 (~16.5-19.5 agent-h)
Gains came only from re-sweeping switches whose values flipped with the register allocation (ETAIL, HSK_XG, FASTRS_W:
-40 us at B=8). Structural attempts lost to allocation even when the profile showed the intended effect: QX (phase A
-0.9 us/layer in the profiling build; +18..+48 us in the normal build), HSK_WARM (rescoring -1.5 us, sketch +0.8),
PFDLY, EBPF, ATT_BLKW_F 2, V_EARLY, layout constants, compiler options (incl. --register-usage-level).
B=8 head now (pos 600): prep 9, sketch 87, wait 4, filter 1.6, norm rebuild <= 9 (cold), pooled rescoring <= 7.

## Final per layer (mkprof, synthetic prompts, pos 600, engine of s0047, profiling build)
B=1: A 10.0, B 6.5, C 7.2, D 3.9, E 10.4, F 9.7 (= 47.7) + head 106 (prep 3, sketch 83, tail ~13 incl. barriers).
B=8: A 11.1, B 7.6, C 7.6, D 4.5, E 31.5, F 21.6, combine 2.2 (= 86.1) + head 124 (prep 9, sketch 87, tail ~24).
B=8 also varies by +-5 % with the prompt set (findings.md).
Track A (engine of s0049, same conditions): B=1 A 10.6, B 6.4, C 7.2, D 3.8, E 10.8, F 9.2 + head 173 (exact BF16
head: median block 155 us, slowest 170: per-SM bandwidth spread -- the largest single item left in track A);
B=8 A 11.0, B 7.7, C 7.6, D 4.6, E 31.6, F 20.8, combine 2.1 + head 170 (median 159, slowest 166).

## After s0056 (~29 agent-h, profiling build, synthetic prompts, pos 600)
B=8 head 124 -> 112-113 us: prep 8.4, sketch stream ~87.5 (work) / 89.4 (span), post-sketch wait ~5, rescoring
max ~4.6-4.9 (was up to ~14: tiles now split in 4 K segments over the warps), the rest ~8 (reduction, barrier).
B=1 head unchanged (106-107: prep 3.1, sketch 83.7 / 86.5, wait ~5, rescoring ~5.5, after ~10 incl. the barrier).
B=1 attention phase per layer by context: 4.47 us (pos 100), 5.91 (500), 7.00 (1000; > 8 chunks per KV head go
through global partials): ~55-60 us of the step's growth over the HTTP decode window.
Track A exact head with HDYN (s0057): B=1 168.7 -> 164.8, B=8 170.3 -> 169.3 (aggregate HBM rate: ~159 us floor).
