# Step account (run s17-mini-b200-mb, merge track B)

## Current state (2.6 agent-hours): best track B s0006 = 2867 / 5480 us; track A s0005 = 2888 / 5464.
Engine: dual (engine/api.py dispatches by batch size): B=1 -> engine/a2 (s15-mini-b200-a2 s0025 kernel + SKH head
sketch, track B), B>=2 -> engine/a1 (s15-mini-b200-a1 s0025 kernel, track A). HTTP - in-process (w1-like generate,
same workload): B=1 22 us, B=8 ~-110 (the device loop serves). Trace: 2 launches per step, GPU idle 0.13%.
B=1 per application now (card 1, e0052, a2 kernel at ctx 500): plain 46.6 us (attn_in 4.8 | qkv 6.6 | attn 10.0 |
oproj 3.5 | mlp_in 3.5 | router 1.3 | routing 3.0 | w13 6.5 | w2 5.8 | w2b 1.6), gated 66.0 (gate chain ~29.8);
head 137 -> ~100 us with SKH (agent profile). 24 x 46.6 + 24 x 66 + 100 = 2.80 ms (card 1 generate 2.69).
Named B=1 losses: gate chains 24 x 29.8 = 0.72 ms; ~10 hand-offs per application; attention merge ~4-5 us x 48.
B=8 (a1 kernel): see s15-mini-b200-a1 notes/gap.md; its attention is one grid round (key-range pieces), the
experts stream through TMA weight rings; gap to the 1.10x target (4641) is 0.84 ms.

## Earlier account of the a2 engine (before the dual split)

Card note: this pod's card 0 (evaluation) runs B=1 slower than the earlier runs' cards: a2 s0025 2900 us here vs 2755
in a2's run, b2 s0047 2949 vs 2879; B=8 unchanged (5860 / 6511). Card 1 (experiments) reproduces a2's in-process
2744 us (generate, e0001). Compare submissions in this run with each other, not with the other runs' numbers.

Floor: 5.54 GB per step at B=1 -> 692 us at 8 TB/s nominal, ~1.0 ms at the ~5.4-5.5 TB/s these kernels stream
(a2 / a1 measurements). B=8: ~10 GB of distinct expert bytes (union ~43 experts per application, b2 weights_study
section 5) + 3.7 GB fixed -> ~1.7 ms at 8 TB/s.

## B=1 (in process ~2.74 ms on card 1): from a2's crit.py account (a2 notes/gap.md, engine s0020-s0024)
24 plain applications x ~47 us + 24 gated x ~68 us + head ~137 us + device-loop step overhead.
Plain application: attention-input mix 4.7 | qkv 6.6 (40 MB, at bandwidth) | attention 10.5 (merger ~7.8 on path) |
o_proj 4.5 | MLP-input mix 3.8 | router 1.0 | routing 4.0 | w13 7.0 | w2 5.7 | w2b 1.5.
Gated extra ~21 us: gate D (input mix + message mix + norm + 72-unit GEMV) ~7.5, F/G 2.0, norm 6.5 (attention gates);
MLP gate D 9.4, F/G 1.9, norm 4.7 (some since cut by s0022-s0024).
Losses named: (1) gate chains of 24 gated applications ~0.5 ms; (2) ~10 hand-offs per application x ~1.5 us
= ~0.7 ms; (3) attention merge ~5 us x 48 = 0.24 ms; (4) weight phases at their per-SM in-flight limit.

## B=8 (in process ~5.78 ms generate on card 1, e0001)
Per application ~128 (plain) / ~150 (gated) us: attention 32 | w13 25 | w2 28 | o_proj 11 | mixes 8-10 each |
routing ~5. a1's different design reaches 5513 us HTTP (TMA weight rings, w2 fixed at compile time).

## Track B items available (from b2) and their b2 gains
GLIN (B=1 gate linearised) -61..-69 us; SKH (head sketch) -8..-45; MSGZ (B>=2) -153 at B=8; L0T (B>=2) -32 at B=8.
