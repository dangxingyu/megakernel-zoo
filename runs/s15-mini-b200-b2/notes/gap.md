# Step account (state at ~10.3 agent-hours)

Best: track B s0052 (engine/TRACK): 2880 us (B=1) / 6486 us (B=8) over HTTP; track A s0048: 2949 / 6730.
vLLM 3692 / 5077; target 3356 / 4615. B=1: 1.28x vLLM (target 1.10x reached). B=8: 0.78x (not reached).
Floor: 5.54 GB per step at B=1 (692 us at 8 TB/s); B=8 13.7 GB measured (1.7 ms).

## B=1 (batch-1 kernel; cpath e0240 before SMTRIM / SM1 / KVC4): ~82 us per gated application, ~47 ungated
Gate chain of a gated application with GLIN: gate_d +6.7, gate_fg +1.8, gate_norm +5.6 (attention gates);
mgate_d +12.0, mgate_fg +1.5, mgate_norm +3.6 (MLP gate) = 31 us x 24 applications = 0.75 ms.
Weight phases: qkv 7.2, oproj 5.9, w13 6.7, w2 6.1 (~26 us per application, ~15 at bandwidth).
Hand-off phases: attn 6.7, router + routing 5.4, x17 2.6, mixes ~5.

## B=8 (generic kernel; mkprof e0232 before RTMASK / SMTRIM): spans per application
w2 41 / work 16, attn 35 / 21, w13 30 / 27, attn_in 28 / 9, x17 26 / 13, oproj 21 / 18, mlp_in 21 / 6,
gates ~30 per stage. Expert streams now at 4-4.7 TB/s (EXQ + NG2NI); the rest is hand-off latency and attention.
The parallel track-A run a1 reaches 5608 us at B=8 with TMA weight rings and a w2 path fixed at compile time: the
structural gap at B=8 is the expert / attention pipeline design, beyond what this run can rebuild.

## Measured dead ends (do not retry on this engine)
bulk L2 prefetch of weights or K/V (9 variants), TMA / cp.async staging in hot loops, compressed weight formats
(S6, W12Q, W12U), K-split of gate jobs, 64- or 256-key attention items, two attention items per block (needs
214 KB of shared memory: the footprint alone costs +658 us), persisting L2, poll back-off, deeper expert batches
in the generic kernel, linearised gates in the B >= 2 kernel or for the MLP gate, linearised q/k/v (QLIN).

## Final profile (e0270, final build, context 600; critical-path increments of a gated application, us)
| stage | B=1 | B=8 |
|---|---|---|
| gate_d / gate_fg / gate_norm | 7.2 / 3.6 / 2.8 | 6.9 / 6.4 / 8.2 |
| qkv | 6.9 | 5.6 |
| attn | 5.6 | 25.3 |
| oproj | 6.4 | 7.4 |
| mgate_d / mgate_fg / mgate_norm | 11.8 / 1.5 / 4.1 | 7.9 / 1.0 / 7.9 |
| router + routing | 4.6 | 5.4 |
| w13 | 6.1 | 34.8 |
| w2 + x17 | 6.4 | 27.1 |
| application total | ~77 | ~167 |
Head (final): 77.6 us at B=1 (int8 sketch pass at ~6 TB/s), 120.6 us at B=8. At B=8, w13 + w2 (62 us) and attention
(25 us) are half of every application; at B=1, the MLP gate's D stage (11.8 us) is the single largest item.
