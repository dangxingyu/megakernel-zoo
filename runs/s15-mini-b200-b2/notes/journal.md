# Journal (s15-mini-b200-b2, track B run)

## s0001-s0003 own engine v1 (0.7 h): valid 6578 / 12658 us HTTP
Own persistent kernel (engine_own/). Lessons in notes/findings.md (i-cache 128 KB, local-memory spills, serialized
load loops, compiler reordering of loads after dependent MMAs). Optimisation stalled at ~6.2 ms in process.

## s0004 track-A base = s15-mini-b200-a1 s0001 unchanged: valid 5042 / 14392 us HTTP (in process 4870)
Decision (1.9 h): adopt the parallel track-A run's validated engine as this run's track-A base (playbook s7b did the
same with its sibling); my track-B changes then apply to the engine that will be merged. Expected ~5050: measured 5042.

## XPF (cross-pass expert prefetch into L2), not submitted
Hypothesis: loop passes share ~5 of 8 experts of a layer with the previous pass (measured below); prefetching the
previous pass's experts (cp.async.bulk.prefetch.L2, 4.5 MB per expert) at the start of the application should make
w13/w2 reads L2 hits, -1..2%. Measured (tests/ab.py, same process, alternating, e0103): bit-identical logits, but
+354 us (+7.4%) at B=1 ctx 64, +212 (+4.2%) at ctx 600, +9% at B=8; previous-token variant (XPF2) +10.5%.
Conclusion: the extra fill traffic queues in front of the latency-critical loads of phases A-D (the TMA rings already
keep the memory system busy); L2 prefetch of speculative bytes loses on this engine. Dead end.

## W12 on the a1 base, per-layer matrices (W12 = qkv/o/router): negative, not submitted
Bit-identical (round trip exact on all 1.14e9 streamed BF16 weights, 94k escapes), but +516..+655 us at B=1: the a1
kernel is 21.8K instructions (2.7x the i-cache); every added code path / sync on the per-application path costs more
than the bytes save (staging alone +68 us, e0116). The unembedding (one long stream) is the only matrix where it pays.

## s0005 W12U on the a1 base: valid 5062 / 14330 (expected ~5012 / ~14330): B=8 as predicted, B=1 +20 instead of -30
Same-process A/B had said -31 us; later found that engine instances differ by up to 100 us (placement): the B=1
miss is within that noise. Bit-identical per bitcheck (1152 positions, B=1 and B=8).

## s0006 track-A base v2 = s15-mini-b200-a2 s0006: valid 3723 / 7707 (expected ~3730 / ~7670)
Re-based all track-B work on the faster track-A engine (tagged dataflow, device loop behind submit/poll).
W12U ported (piece_w12: 512 B escape header per strip, 3 x uint4 per lane per unit, row power-of-two scale):
bit-identical (same greedy tokens, logits equal), head phase 177 -> 141 us (mkprof e0126), end to end -14..-17 us at
B=1 (inside the instance noise), B=8 noise.

## MSGZ (s0007 submitted)
Upper-bound study (gate_d without message mix and D_m half, m := x) said -187 us B=1 / -270 B=8, but it changed
values (routing). Real implementation: ph_msgz at the pass start (2 x 14.9 us spans), gate_d/mgate_d spans -2.7/-3.5,
gate_fg/mgate_fg -2.5/-3; 3-instance A/B: -14 (step) / -26 (generate) us at B=1, -3 / -33 at B=8.

## s0007 MSGZ: valid 3665 / 7708 (expected -15..-30 / -0..-30 vs 3723 / 7707): B=1 -58 (better than predicted), B=8 +1
Within gates (fp32 summation split); new best at B=1 (below vLLM's 3692).

## s0008 W12U on a2 base: valid 3743 / 7678 (expected ~3705 / unchanged): B=1 +20, B=8 -29: within the instance noise
(bit-identical, head phase -36 us; the end-to-end gain is below the +-50 us noise of one evaluation).

## SKH certified int8 head sketch (s0009 submitted)
Offline study first (tests/sketch_study.py, real final hidden states, e0140): per-row int8 + Cauchy-Schwarz bound
e_i |x^| + n_i |x - x^| leaves median 1, p90 5, max 36 candidate rows per token (|x| ~ 176). Kernel bugs found on
the way: block_sum reuses sm.red and sm.bcast (sigma/|x| overwritten -> L = 3e6 at B=8; digits could clamp): moved
to own smem slots; the softcap saturates for big logits (ties broken by row index): candidates now compare the capped
values 30 tanh(u/30) >= 30 tanh(L/30) - 4 ulp (exact in the saturated regime too). The exact pass first ran each
candidate strip on one warp (31 us); staged through smem by the whole block in one load round: 11 us, same MMA chain
as track A (bit-identical keys). Pass-1 load depth 6 -> 12: 99 -> 89 us. Head 158 -> 89 + 9 (barrier) + 11 us.
6 x 1024 greedy tokens identical to track A (B=1, B=8); 2-instance generate A/B -54.5 us B=1, -57 us B=8.

## s0009 SKH alone: valid 3805 / 7654 (expected ~3670 / ~7650): B=8 as predicted, B=1 +82 vs s0006
## s0010 track A + start-up placement tuning: valid 3802 / 7685 (expected ~3640 / ~7700)
The evaluator's own tuning log for s0010 read 3704-3842 us (card 1 reads 3607-3677 for the same build): card 0 ran
~100 us slower during s0009/s0010. Single evaluations carry +-80 us at B=1 between submissions; the same-card
interleaved A/Bs are the evidence for deltas (SKH -45 / -32, MSGZ -3.5 / -32 with tuning on both sides).
## s0011 combined track B (MSGZ + SKH on the tuned base): valid 3618 / 7652, NEW BEST (in-process A/B -72 / -99 us)
Greedy tokens of the combined build identical to its own non-sketch build (B MSGZ) over 3 x 512 tokens, B=1 and B=8.
Its tuning log on card 0: 3587-3660 (card 0 fast this time).

## XPF on the a2 base: +246 us B=1, +1339 us B=8 (B=8: the predicted unions are ~150 MB per application, more than
L2): dead end on both engines.

## L0T layer-0 token tables (track B, ~4.3 h)
Layer 0's attention input is h0 = rmsnorm(embed[tok]) * embed_norm: its 8672 q/k/v/gate rows depend on the token id
alone. Table [201024][8672] bf16 (3.5 GB) built at load by the engine's own arithmetic (25128 launches, 0.4 s); at run
time application 0 copies the rows instead of the 40 MB GEMV. Not output caching: a weight-derived precomputation
over every token id, input-independent (like folding the embedding into the first projection). Bitcheck (tests/
l0t_check.py): step logits bitwise equal over 2 x 160 positions at B=1 and B=8, random ids, generate tokens equal.
First version: +130 us B=1 / +77 us B=8 (A/B). The same code compiled in but never run (no table): +118 us B=1.
**Code presence alone costs ~2.5 us per application**: +1248 SASS instructions (duplicate h0 code in attn_in, a
build-mode store in the qkv epilogue). v2: build mode in its own kernel (mk_l0b), the copy out of line
(__noinline__), the h0 code shared: +224 instructions in mk_step.
v3: one out-of-line call l0t_app0 (h0 + copy) in mk_step, phase code untouched (spill loads 556 -> 536 B; v2 had
600 B: the +49 us of v2 tracks the spill/regalloc change, not the instruction count). ncu: no-instruction stalls are
0.84 of 31 cycles per issue in both builds (i-cache is not the cost). A/B v3 (k=3): -3 us B=1, -32 us B=8.
## s0013 L0T: valid 3616 / 7548 (expected B=8 ~7550, B=1 unchanged): as predicted. Bit-identical (bitcheck).

## FGPLACE (track A, port of s15-mini-b200-a2 s0007): gate F/G tasks only on blocks without a D job
Read the parallel track-A run's later submissions (a2 s0007-s0012; s0012 = --use_fast_math, an approximation: not
taken). s0007's F/G placement is a 10-line change on the same code: bit-identical (bitcheck A vs A), same-card A/B
-164 us B=1 on track A, -178 us B=1 on track B (3570 -> 3392 in process), B=8 flat.
## s0014 track A + FGPLACE: valid 3622 / 7711 (expected ~3640 / ~7690): new track-A base (s0010 was 3802 / 7685)
## s0015 track B (MSGZ SKH L0T) + FGPLACE: valid 3425 / 7463, NEW BEST (expected ~3440 / ~7550)
## GDPF (track A, gate-D strip TMA-prefetched into smem during the mix): bit-identical, +97 us B=1 / +292 us B=8.
Sub-marks: the strip GEMV went 3.5 -> 2.5 us (the rest is the 24-warp reduction and syncs), but the step got slower
(third time a prefetch of bytes that are read anyway anyhow loses on this engine: XPF, W12 staging, GDPF). Off (GDPF 0).
## DKS (gate-D K-split over 3 blocks at B <= 2, partials summed in F/G): +138 us B=1 (gate_norm +6 us). Off.
Sub-marks inside the gate-D GEMV step (3.5 us): warp 0's MMAs 1.3, waiting for the slowest warp 1.8, epilogue 0.5:
first-touch HBM latency of the strip.
## GDL2 (one cp.async.bulk.prefetch.L2 of the job's D strip at gate-D job start): A/B -31.5 us B=1, -31 B=8
## s0016 track B + GDL2: valid 3418 / 7449 (expected ~3395 / ~7430); in process 3363 (s0015: 3388). NEW BEST.
## RTL2 (router strips into L2 at mlp_in, B=1): +3 us, neutral. Off.
## ATL2 (K/V chunk of the block's attention items into L2): before qkv +42 us, at the application start +42 us. Off.
The K/V bytes compete with the bandwidth-bound qkv stream (or evict it); L2 prefetch pays only for bytes needed in
a latency-bound window (GDL2).
## s0017 track A + GDL2: valid 3609 / 7705 (expected ~3590 / ~7680): new track-A base (s0014: 3622 / 7711)
## L2P (track A: small latency-critical weights - router, gate D/F-G, norms, queries - in one 74 MB arena under an
L2 access-policy window, persisting, 72.5 MB set-aside of the 84.7 MB max): +9 us B=1, +115 us B=8. Off (TRACK token).
With RTL2 neutral too: the first touch of these weights is not on the critical path (F/G issues its row loads first).
## W12Q (track B: exact 12-bit qkv strips, piece_w12r over unit ranges, partial rescaled by the row power of two):
bit-identical (bitcheck, all 24 layers' qkv round trip asserted, <= 64 escapes per strip), but +102 us B=1 / +176 B=8.
cpath: the qkv phase itself +7.9 -> +10.4 us: the decode path keeps fewer bytes in flight (9 x 16 B per lane per
3-unit batch) than the BF16 path, and the 80-register budget leaves no room for a deeper batch (spill loads 644 B).
## RTREDUX (track A, port of a2 s0014/s0015: router top-8 with redux.sync on integer keys, NaN guard, attention merge
staging fix for > 58 chunks): bit-identical (bitcheck), A/B -35 us B=1, -125 us B=8 (e0203). Parked unsubmitted:
operator notes (21:30 / 21:35) ask for a study of the NVFP4 / MXFP8 weights, scales and router statistics
before the next engine change.

## Weights and routing study (operator notes 21:30 / 21:35; 5.6-5.9 h): notes/weights_study.md
Exact census of all 24.5e9 expert weights (tests/expert_census.py, e0207) and router statistics on 8 x 512 tokens
of natural text (tests/routing_study2.py, e0208). Headlines: FP4 code entropy 3.968 of 4 bits; every block of 16
contains a +-6; no zero / duplicate / saturated blocks, rows or experts; NVFP4 block scales use only 51 consecutive
E4M3 codes (76..126) -> exact 6-bit scale without escapes (-2.8% expert bytes) or 5-bit with a per-row base
(7.7e-6 escapes, -4.2%); MXFP8 codes 6.52 bits of entropy but 1% escapes for a 7-bit window code; MXFP8 scales 0.7
bits (2-bit code, -2.3%). Routing: effective 77 of 128 experts per application, best 18 experts (max persisting
L2) cover 6% of the selections; B=8 union 43.3 experts per application = 9.97 GB of expert bytes per step, each
read once by the engine; pass reuse needs 430 MB (B=1) / 2.2 GB (B=8) of cache. B=8 gap = expert streaming rate.
Next: (1) submit RTREDUX; (2) track A: L2 prefetch of certain expert bytes (W2PF before w13, W13PF next round);
(3) track B: 6-bit NVFP4 scales (S6) as the weight-property format, measured whether or not it pays.
## s0018 track B + RTREDUX: valid 3380 / 7404, NEW BEST (expected ~3385 / ~7330; B=8 gained 45 of the predicted 119)
## Expert streaming at B=8 (the main gap; 2.3 TB/s over 10 GB per step)
W13PF=1 (next round's w13 items prefetched into L2 at the start of a round): +314 us at B=8 (e0209). Large L2
prefetches issued together with a round's demand loads queue in front of them (same lesson as XPF / ATL2).
Model: throughput = bytes in flight / latency; in flight is limited by registers (FP4_UNR 4 units = 24 of 80
registers per thread hold weight bytes). Variants under test: FP4_UNR 6 / 3, prefetches issued only after a warp's
last loads of a round (W13PF=2, W2PFL), qkv strips prefetched at B=1 too (QPF_MINB=1).
## s0019 track A + RTREDUX: valid 3561 / 7585 (expected ~3570 / ~7660): new track-A base (s0017: 3609 / 7705)
Track B (s0018, 3380 / 7404) against it: -181 us B=1 (-5.1%), -181 us B=8 (-2.4%).
Results (e0210, 2 instances each, track-B build, generate):
| variant | B=1 | B=8 | spill loads |
|---|---|---|---|
| FP4_UNR 6 (6 units in flight per warp) | +3 | -11 | 612 B (base 524) |
| FP4_UNR 3 | +43 | +214 | 480 B |
| W2PFL (w2 range prefetched after the last w13 loads) | +90 | +284 | 564 B |
| W13PF=2 (next round prefetched after a warp's last loads) | - | +150 | 524 B |
| QPF_MINB=1 (qkv strips prefetched during attn_in at B=1 too) | +135 | - | |
Every bulk L2 prefetch of tens of KB per block loses, also in the "idle" window: only GDL2 (one 36 KB strip per
gate-D block, during a 4-6 us mix) pays. In flight 3 -> 4 units helps (0.9 us per application), 4 -> 6 does not
(with 88 B more spill loads): the register file is the limit.
## S6 (track B: exact 6-bit NVFP4 block scales, byte - 76, no escapes; unit 576 -> 560 B, -2.8% of the NVFP4 bytes)
The weight property from the census turned into fewer HBM bytes: scales packed 4 pairs x 12 bits per lane group,
decoded through a 4096-entry table of f16 pairs in shared memory (the same cvt instruction fills the table, so the
A fragments are bit for bit those of the stored format). Round trip asserted on every scale of the 40 NVFP4
tensors at load; bitcheck PASS (logits bitwise equal, B=1 and B=8, random ids). Timing (e0214, 3 instances):
**+149 us B=1, +469 us B=8** (+3.1 us per application at B=1). The expert phases are not bandwidth-bound (3 TB/s),
so 2.8% fewer bytes buy nothing, and the decode (two dependent shared-memory lookups per unit, two scalar loads
instead of one 8-byte load) lengthens every unit's chain. Prediction was <= 0; kept behind TRACK token S6, off.
## SCS (track A: NVFP4 scales of a warp's piece via cp.async into shared memory, 6 code units in flight):
bit-identical, +150 us B=1 / +266 us B=8; spill loads 524 -> 824 B (UNR 6) / 560 B (UNR 4). Off. The expert inner
loop is at a local optimum of the 80-register kernel; six variants (W13PF x2, W2PF, FP4_UNR, S6, SCS) all lose.
## Batched re-polling of tagged hand-offs (track A), two forms, both negative
Hypothesis: a consumer that arrives before its producer waits word by word (12 words per thread in the gate norm
stage), one relaxed-load round trip (~0.45 us) per word after the data is there; reloading all stale words together
would take one round trip. BPOLL (10 sites, conditional reloads): bit-identical, +101 us B=1 / +183 us B=8, spill
loads 524 -> 1636 B (conditionally assigned arrays go to local memory). BPG (gate norm only, unconditional reload of
the 12 words, no spill change): +33 us B=1, -13 us B=8. Reading: while a block waits, serial waiting keeps one poll
in flight per thread; the batched form keeps 12 (152 x 768 threads): the extra relaxed loads slow the producers
more than the saved round trips. Off (BPG 0).

## K1 (track A): the same source compiled twice, a batch-1-only kernel next to the generic one (6.4 h)
Every structural experiment of the last two hours failed on the 80-register kernel's spills, also the ones whose
latency model was right (FP4_UNR). The generic kernel carries every B >= 2 path and the two-column-group expert
code; with -DKB1=1 (HDB = 1: `hd.B` is a constant) the compiler removes them: **spill loads 524 B -> 36 B**,
bit-identical (bitcheck), **-206 us at B=1** (3338 -> 3132 in process, 3 instances), B=8 unchanged (the generic
kernel is launched for B != 1). With FP4_UNR 6 in the batch-1 kernel only (12 B of spills): -236 us.
The decoupling also means each regime can now be changed without perturbing the other.
s0020 submitted: track B + K1 (expected ~3175 / ~7404).
## s0020 track B + K1: valid 3150 / 7444, NEW BEST (expected ~3175 / ~7404). B=1 passes the 1.10x target (3356):
1.172x vLLM (3692). B=8 unchanged within noise (generic kernel).
K8 (a kernel with the batch fixed to 8): spill loads 884 B (loops over the batch unroll), +158 us at B=8. Off.
## s0021 track A + K1: valid 3114 / 7577 (expected ~3330 / ~7585): B=1 far better than predicted, and BELOW the
track-B build with K1 (s0020: 3150). K1 gave track A -447 us but track B -230 us: in the batch-1 kernel the
track-B items no longer pay at B=1 (before K1 they did: 3380 vs 3561, part of that was relief of the generic
kernel's register pressure rather than the saved work). To measure per item on the same card ("A K1" vs each).
## KN and the batch-1 kernel's own settings (e0220)
KN (the generic kernel compiled with the batch known to be >= 2; batch 1 always goes to the batch-1 kernel): spill
loads 524 -> 396 B, bit-identical, **-72 us at B=8** (7498 -> 7426, 3 instances).
Batch-1 kernel only (-D for that compile; vs K1 as submitted, 2 instances, B=1):
| setting | us | spill loads |
|---|---|---|
| FP4_UNR 6 | -41 | 12 B |
| FP4_UNR 6 + BPG (batched gate-norm re-poll) | -103 | 12 B |
| FP4_UNR 6 + GDPF (gate-D strip via TMA into smem) | -68 | 36 B |
| FP4_UNR 6 + BF_UNR 3 | -3 | 44 B |
| FP4_UNR 6 + DKS 3 | +36 | 12 B |
| FP4_UNR 6 + QPF_MINB 1 | +68 | 12 B |
| FP4_UNR 12 | +366 | 32 B |
Three ideas that were negative in the generic kernel (more units in flight, batched re-polling, TMA strip staging)
are positive once the kernel has register headroom: their models were right, the spills were the confound.
## s0022 track A + K1 + KN: valid 3116 / 7659 (expected ~3114 / ~7505): B=8 +82 vs s0021 instead of -72.
The -72 us was measured on the track-B build; on track A the B >= 2 compile moves the other way (to be confirmed
same-card). Every (track, kernel) pair needs its own A/B: the generic kernel's time follows its register
allocation more than its work.
## Track-B items inside the batch-1 kernel, each against "A K1" (e0221, 3 instances, B=1 generate)
| build | us vs A K1 | b1 kernel spill loads |
|---|---|---|
| A K1 | 3086 (base) | 8 B |
| B MSGZ SKH L0T K1 | +50 | 36 B |
| B MSGZ K1 | +126 | 24 B |
| B SKH K1 | -8 | 28 B |
| B L0T K1 | +79 | 16 B |
Before K1 the same items measured -47 (MSGZ), -45 (SKH), -3 (L0T) against the generic track-A kernel. With register
headroom the plain gate path got faster than MSGZ's, and L0T's control flow (a goto into the application loop)
costs more than the 40 MB it saves. To profile (cpath) before deciding the composition per kernel.
## Per-kernel items (TRACK tokens NAME:1 / NAME:N) and s0023
cpath of a gated application in the batch-1 kernel, A K1 vs B MSGZ K1 (e0222): MSGZ's gate-D stage is 3.6 us
shorter as designed, but the norm stage after F/G ends 4-8 us later (gate_norm span 10.2 -> 14.6 us, mgate_norm
7.2 -> 8.0), so the chain is level again by the end of qkv and behind after the MLP gate. Suspect: the messages
(mg2t) and z_m (mzt) are written once at the pass start and are HBM-cold when a later layer's F/G and D epilogue
read them with relaxed loads; the plain path publishes m in the same application (L2-hot). MSGZ_PF (a small L2
prefetch of the layer's slice at the gate's start) prepared to test this.
## s0023 "B MSGZ:N SKH L0T:N K1": valid 3109 / 7401, NEW BEST on both (expected ~3105 / ~7444)
Track A base is s0021 (3114 / 7577): track B is -5 us at B=1 (the sketch) and -176 us at B=8 (2.3%).
## s0024 "A K1 FP4_UNR=6:1": valid 3113 / 7595 (expected ~3075): no gain on track A (s0021: 3114); the -41 us was
measured with the track-B items inside the batch-1 kernel. Settings have to be measured on the exact build.
## Per-kernel sweeps on the track-A base "A K1" (e0223, 2 instances each)
Generic kernel at B=8 (tokens with :N):
| generic kernel | us vs A K1 | spill loads |
|---|---|---|
| KN | +48 | 464 B |
| NI_ALL (every phase out of line) | +2 | 100 B |
| MSGZ SKH L0T | -184 | 524 B |
| MSGZ SKH L0T + KN | **-248** | 396 B |
| MSGZ SKH L0T + NI_ALL | -80 | 316 B |
| MSGZ SKH L0T + FP4_UNR 6 | -203 | 612 B |
Out-of-line phases remove most spills (524 -> 100 B) without any gain: at B=8 the spills are not the limit, the
round structure of the expert phases and the hand-offs are. KN helps the track-B generic kernel, hurts track A's.
Batch-1 kernel at B=1 (tokens with :1): FP4_UNR 6 + BPG -12; + GDPF +20; on top of FP4_UNR 6 + BPG: batched polls
at the other sites +45..+80 (load_vec_f, load_srcs, attention q, o_proj, w2 staging, w2b) except routing -18 and
load_vec_bf +2; SKH +30; NI_ALL +291. The batch-1 kernel is near a local optimum at ~3050-3090 us in process.
Next for B=8: EXQ, the expert items of a block taken by its warps from a counter (no rounds of 8 with block
syncs; model: w13 41 -> ~27 us, w2 26 -> ~15 us per application).
## s0025 "A K1 BPG:1": valid 3102 / 7571 (expected ~3055): -12 us vs s0021, as the same-card sweep said (-12 with
FP4_UNR 6), not the -62 of the track-B build. Track-A base now s0025.

## EXQ (track A organisation, generic kernel): the expert items of a block taken by its warps from a counter (7.6 h)
The round structure (8 items x 3 warps, block-wide reduce + sync per round, 5-6 rounds per phase at B=8) replaced
by a queue: every warp takes the next item of its block (atomicAdd on a shared-memory counter), runs its 36 (w13) /
18 (w2) units in the same three parts and sums them in the same order as the three-warp split, runs the epilogue,
takes the next item. No block syncs inside the phase; warps never wait for each other. Rounds containing an expert
with more than 4 tokens go through an out-of-line two-group function. Bit-identical (bitcheck: logits bitwise
equal at B=1 and B=8). **B=8: 7428 -> 6809 us (-619 us, -8.3%)**, 2 instances (e0224). Used when a block has more
than 8 items (never at B=1, where 3 warps per item is the latency-optimal split).
## s0026 "B MSGZ:N SKH L0T:N K1 KN": valid 3118 / 7414 (expected ~3109 / ~7335): KN is within the evaluation noise
(s0023: 3109 / 7401). Kept (the EXQ measurements were made on top of it).
## s0027 "B MSGZ:N SKH L0T:N K1 KN EXQ:N": valid 3121 / 7102 (expected ~3118 / ~6800): B=8 -312 us over HTTP
(in process -619 us at contexts 64-576; the HTTP run goes to 1088 positions where attention weighs more). NEW BEST
at B=8.
## EXQ follow-ups (e0225, B=8, 2 instances)
| build | us | generic spill loads |
|---|---|---|
| track B + EXQ, + FP4_UNR 6 (generic) | -26 | 672 B |
| + FP4_UNR 8 | +123 | 792 B |
| **+ NG2NI** (two-column-group rounds out of line) | **-396** (6837 -> 6440) | **32 B** (was 584) |
| track A: A K1 BPG:1 + EXQ | -568 | 684 B |
| track A: + EXQ + KN | -597 (7697 -> 7100), bitcheck PASS | 564 B |
The inline two-group expert code (needed by 1.7% of the union's experts) was the generic kernel's main spill source.
## s0028 "A K1 BPG:1 KN EXQ:N": valid 3092 / 7434 (expected ~3102 / ~7270): track-A base. B=8 -137 us over HTTP
against -597 in process: at the HTTP run's contexts (to 1088) attention and the hand-offs weigh more than at 64-576.
## s0029 "B MSGZ:N SKH L0T:N K1 KN EXQ:N NG2NI:N": valid 3083 / 6725, NEW BEST on both (expected ~3120 / ~6900)
B=8: 7414 (s0026) -> 7102 (EXQ) -> 6725 (NG2NI) in an hour, all bit-identical organisation of the B >= 2 kernel.
## Generic kernel with EXQ + NG2NI (e0226, B=8): NG2NI bitcheck PASS
FP4_UNR 6 -27 us; BPG -33; GDPF -49; BF_UNR 3 -62; FP4_UNR 6 + BPG + GDPF -67 (1% each, not additive).
Track A + NG2NI: -596 us in process (7129 -> 6533).
mkprof at B=8, context 600 (spans / work per application, us): w2 35.8 / 16.3, w13 31.0 / 27.5, attn 26.8 / 15.1,
attn_in 23.1 / 7.5, x17 21.6 / 11.2, mlp_in 20.8 / 6.8, oproj 20.2 / 15.5, gate_fg 30.3 / 6.6, gate_d 24.3 / 7.8,
mgate_fg 23.8 / 12.1, mgate_d 22.5, gate_norm 13.4, mgate_norm 10.8, routing 8.9, qkv 7.3.
The expert phases now stream at 4-4.7 TB/s (129 MB in 27.5 us, 65 MB in 16.3 us): the register-limited rate.
What is left at B=8 is hand-off latency: spans 2-4x the work in attn_in, mlp_in, x17, the gate stages.
## s0030 "A K1 BPG:1 KN EXQ:N NG2NI:N": valid 3075 / 6805 (expected ~3092 / ~7100): track-A base.
Track B (s0029: 3083 / 6725) against it: +8 us at B=1 (noise; the batch-1 kernels differ only by the sketch
head), -80 us at B=8 (1.2%).
## s0031 "... NG2NI:N BF_UNR=3:N": valid 3112 / 6691 (expected ~3083 / ~6690): B=8 -34 us as predicted; B=1 +29 with
an unchanged batch-1 kernel: evaluation-to-evaluation noise (start-up placement).
## Generic kernel poll sites and settings at B=8, contexts 320-832 (e0227; base EXQ + NG2NI = 6780 us)
| generic kernel | us |
|---|---|
| BPX bit 2 (mix sources, load_srcs: 2 words per source) | **-136** |
| BPX bit 6 (w2 input staging) | -40 |
| BPX bit 7 (w2b: 8 expert contributions) | -37 |
| BPX bit 5 (o_proj input) | -14 |
| BPX bit 1 (load_vec_f) / bit 3 (routing) / BPG | -8 / +1 / -4 |
| BPX bit 9 (poll_rows: 16 words) | **+372** |
| BF_UNR 3 | -45 |
| GDPF | -44 |
Batched re-polling pays where few words per thread are stale (the newest source of a mix) and hurts where
thousands of threads would keep 16 polls in flight each.
## s0032 "... BF_UNR=3:N GDPF:N": valid 3107 / 6700 (expected ~6650): no gain over s0031 (6691); GDPF:N dropped.
## s0033 "... BF_UNR=3:N BPX=4:N": valid 3104 / 6628 (expected ~3100 / ~6590): best B=8 (s0031: 6691).
## Combination sweep (e0228, B=8, contexts 320-832; base EXQ + NG2NI + BF_UNR 3 + GDPF)
+ BPX 4: +41 us (on the base without GDPF it was -136: the settings interact through the register allocation);
BPX 196 / 228: +43 / +50; ACS 256 (256-key attention items, B > 2): +394 us at contexts 576-1088 and +573 with
BPX 228: larger attention items are slower than twice as many small ones. Dead. Bitcheck of BF_UNR 3 + GDPF +
BPX 228 against the plain build: PASS.
## s0034 "A ... NG2NI:N BF_UNR=3:N": valid 3090 / 6816 (expected ~6770): neutral on track A (s0030: 6805).
## s0035 "A ... BF_UNR=3:N BPX=4:N": valid 3099 / 6840 (expected ~6750): no gain on track A; base stays s0030
(3075 / 6805). The two settings pay only in the track-B generic kernel (6725 -> 6628).
## Sweep e0229
Generic kernel, B=8, contexts 320-832 (base 6690-6720): ACS 64 (64-key attention items for B > 2) +411 us;
POLL_NS 100 / 300 (sleep between polls) +461 / +715. Both attention item sizes around 128 lose: a block's items
run one after another with ~4 us of fixed cost each -> run two items per block side by side (next).
Batch-1 kernel, B=1 (base "A K1 BPG:1 KN EXQ:N NG2NI:N" ~3070): SKH -19; SKH + ph_final out of line -30;
L0T +33 (inlined +8); MSGZ +19; MSGZ + MSGZ_PF +37 (the cold-message theory is wrong); POLL_NS 100 +399.
Track B's batch-1 kernel becomes track A's plus the sketch head (out of line).
## s0036 "B MSGZ:N SKH L0T:N K1 KN EXQ:N NG2NI:N BF_UNR=3:N BPX=4:N BPG:1 NI_FINAL=__noinline__:1": valid 3073 / 6631
(expected ~3050 / ~6628). NEW BEST (B=1 by 2 us over track A's s0030, B=8 -174 us vs track A's 6805).
## ATT2 (generic kernel): two attention items side by side in a block
At B=8 a block runs 2-4 attention items one after another (N = 64 x chunks > 152 blocks), each using at most 12 of
its 24 warps in every stage. ph_attn2: the two halves of a block (384 threads) run one item each in lockstep
through the same syncs, each with its own buffers (At at2[2], +93 KB of shared memory); a half without an item only
passes the syncs; the merge stage runs when either half merges. Same arithmetic per item.
ATT2 result (e0231): bit-identical (bitcheck 2 x 300 positions at B=8), but **+506 us** at contexts 320-832 and
+349 us at 576-1088. An attention item's stages use every thread of the block for their loads (cp.async K/V, bias
table, zeroing) even where only 3-12 warps compute: with half the threads an item takes more than twice as long.
Together with ACS 64 (+411) and ACS 256 (+394): the 128-key whole-block item is the optimum of this design. Off.
mkprof at B=8, context 1000 (e0232; spans / work per application): w2 41.0 / 16.4, attn 34.9 / 21.1, w13 30.4 /
27.3, attn_in 27.9 / 8.8, x17 26.0 / 13.2, oproj 21.1 / 18.0, mlp_in 20.7 / 6.0, gate_fg 36.3, gate_d 29.9.
One attention item: load wave 3.3 us (K/V 64 KB by cp.async, bias table, q / r polls), bias 0.3, scores + softmax
1.5, PV 0.9, merge 4.0 (chunk-0 items only); 3.4 items per block at this context.
ATL2=3 (K/V of a block's next attention item into L2 during the current one, B=8): +287 us at contexts 576-1088,
+228 us at 320-832 (e0233). Off. cp.async.bulk.prefetch.L2 has now lost in eight places and won in one (GDL2).
## Limits check of the two-kernel builds (tests/limits_check.py, e0234)
"A K1 BPG:1 KN EXQ:N NG2NI:N" against the plain single-kernel "A": logits bitwise equal at B = 2, 3, 5, 7 (96
positions each: the generic kernel's queue and round paths); 2000 greedy tokens to position 2031 identical at B=1
and B=8, logits there bitwise equal. The track-B build against the single-kernel "B MSGZ SKH L0T": the same, except
that the B=1 logits differ in the last bits (its batch-1 kernel has no MSGZ; tokens identical). PASS.
## GLIN (track B, batch-1 kernel): the gate's down-projection linearised over the raw mix sources (9.1 h)
x = sum_s cx[s] v_s over the <= 5 published sources of the depth mix and m = sum_s cm[s] u_s over the 2 message
sources, so D [unit(x); unit(m)] = (sum cx[s] D_x v_s) / rms(x) + (sum cm[s] D_m u_s) / rms(m) exactly. The 12
gate-D blocks no longer wait for the mix: they poll the raw sources into shared memory as MMA columns
(0..4 = [v_s | 0], 5..6 = [0 | u_s]) and publish Z = D x columns (64 x 7 values); 3 other blocks compute x, m and
the coefficients in parallel and combine. Uses two architecture facts: the block attention residual (a softmax mix
of at most 6 earlier states) and the rank-64 gate. Not bit-identical (the unit vectors are no longer rounded to
BF16 before D; FP32 combination): "within the gates". Attention gates only (e0235, 2 instances): **-61 us at B=1**
(3041 -> 2980). Against the reference (tests/check_ref.py, 3 x 160 positions, B=1): rel L2 median 0.0332 / 0.0579 /
0.0236 (base 0.0347 / 0.0644 / 0.0218), max 0.33 / 0.43 / 0.19 (0.33 / 0.44 / 0.18), top-1 478 / 480 (477 / 480).
## s0037 "... GLIN:1": valid 3005 / 6635, NEW BEST at B=1 (expected ~3010 / ~6630). Every gate passes.
Track B against track A (s0030: 3075 / 6805): -70 us at B=1 (2.3%), -170 us at B=8 (2.5%).
GLIN=2 (the MLP-input gate linearised too: 8 columns = block partials, pin, attention output, 2 message sources):
-40 us against the base where attention-only GLIN gives -69 (e0237): the MLP gate's mix waits for the attention
output anyway, and the batch-1 kernel's spill loads go 48 -> 124 B. Reference check as good (0.035 / 0.063).
GLIN stays on the attention gates.
GLIN in the generic kernel instead of MSGZ (B=8, contexts 320-832, e0238): +276 us (attention gates), +255 us
(both gates); reference check at B=8 as good as the base (rel L2 median 0.0202 vs 0.0195). At B=8 the hoisted
message half (MSGZ, tokens as MMA columns once per pass) beats per-token source columns. Composition stays
GLIN:1 (batch-1 kernel) + MSGZ:N (generic kernel).
Around GLIN in the batch-1 kernel (e0239): message mix before the x mix (GLIN_MF) +58 us (spill loads 48 -> 76 B);
with FP4_UNR 6 +3; BPX bit 3 (the router's 4 logits words re-polled together) -17 us (-18 in the earlier sweep).
## s0038 "... GLIN:1 BPX=8:1": valid 2993 / 6635, NEW BEST at B=1 (expected ~2990).
MLP-gate GLIN again (e0241): GLIN=2 with the message mix first +24 us against GLIN:1; with BPX 8 -6; gates out of
line +95. Dropped for good.
cpath with GLIN (application 19, batch-1 kernel, e0240): gate_d +6.7 us (was 9.7), gate_fg +1.8, gate_norm +5.6,
qkv +7.2, attn +6.7, oproj +5.9, mgate_d +12.0, mgate_fg +1.5, mgate_norm +3.6, router + routing 5.4, w13 +6.7,
w2 +6.1, x17 +2.6: 81.7 us (84.7 before).
## s0039 "A K1 BPG:1 BPX=8:1 KN EXQ:N NG2NI:N": valid 3073 / 6802 (expected ~3060): level with s0030 (3075 / 6805).
Track-A reference build. Track B s0038 against it: -80 us at B=1 (2.6%), -167 us at B=8 (2.5%).
## QLIN (track B, batch-1 kernel): q/k/v projection of the ungated applications on the raw mix sources
W_qkv (w . x) r = sum_s (cf[s] r) W_qkv (w . v_s): every block stores the columns w . v_s (BF16) and starts its
strips at once, the last block (no strips) computes the mixes and publishes cf r, the epilogue combines 5 columns.
Reference check as good as the base (rel L2 median 0.0325 / 0.0495 / 0.0271, top-1 477 / 480), but **+58 us** at
B=1 (e0242): the mix was not what the qkv strips were waiting for (both wait for the newest source; the mix itself
is ~2 us against a 7 us GEMV that now carries 5 columns and a combining epilogue). Off.
## ATP (generic kernel: the next attention item's K / V copied by cp.async into a second buffer during the current
item): bit-identical (bitcheck 2 x 300 positions), **+494 us** at contexts 576-1088, +588 us at 200-712 (e0243).
Nearly the same penalty as ATT2 (+506 / +349). Both grow the shared memory of a block from 138 KB to 214 KB
(two attention buffers); shared memory and the L1 data cache share one per-SM store, so the footprint itself is
suspected. Test: the base with 76 KB of unused padding (SMPAD).
## Shared-memory footprint is a cost by itself (e0244): padding only, same code
| unused shared memory added to a block (138 KB now) | B=8 | B=1 |
|---|---|---|
| +38 KB | +77 us | +9 us |
| +76 KB | **+658 us** | **+226 us** |
Shared memory and the L1 data cache are carved from the same per-SM store, in steps: at 214 KB the cache is nearly
gone. ATT2 (+506) and ATP (+494 / +588) both needed 214 KB: their penalties are this effect, not their logic.
Any design that needs a second attention buffer has to fit under ~164 KB.
## SMTRIM: two unused shared-memory members dropped (4.9 KB): 138,256 -> 133,408 B, under the 132 KiB carve-out
Bit-identical (bitcheck). 3 instances: **-26 us at B=1, -47 us at B=8** (e0246): the smaller carve-out leaves 32 KiB
more L1 data cache per SM. (Evaluator's max_context is 2048: serve_engine default, kernel_eval max(2048, hctx).)
## s0040 SMTRIM (track B): valid 2960 / 6610, NEW BEST on both (expected ~2970 / ~6590).
The masked effects, for the record: with the footprint penalty of 214 KB (+658 us at B=8) taken out, ATP was worth
about -70..-160 us and ATT2 -150..-300 us at B=8, but neither fits under 164 KiB with 128-key items (ATP needs 185
KB), so they stay off. For a later run: an attention item layout that leaves room for a second K/V buffer.
## s0041 track A + SMTRIM: valid 3030 / 6842 (expected ~3045 / ~6770): B=1 -43 vs s0039, B=8 +40 (noise range).
## SM1: the batch-1 kernel's shared memory sized for one token (64-key items, 48 merge records, one column)
133,408 -> 99,264 B (under the 100 KiB carve-out step); the batch-1 kernel then serves contexts to 3072 (the
engine caps maxctx; the evaluator uses 2048). Bit-identical (bitcheck), limits check PASS (2000 tokens to 2031),
-28 us at B=1 (3 instances, e0248).
## s0042 SM1 (track B): valid 2944 / 6601, NEW BEST (expected ~2935 / ~6610).
## KVC4: port of a2 s0022 (current-key/value short conv: all loads of 4 channels issued before the first wait; same
per-channel arithmetic). Testing.
KVC4 (e0249): bit-identical, -32 us B=1, +158 us B=8: batch-1 kernel only (KVC4=0:N).
## RTMASK: port of a2 s0020 (union token lists from per-expert token bit masks; same ascending order): bit-identical
(bitcheck), -108 us at B=8 (generic kernel), +4 at B=1 (e0250): generic kernel only.
## s0043 KVC4:1: valid 2932 / 6589, NEW BEST (expected ~2915).
## s0044 RTMASK:N: valid 2928 / 6498, NEW BEST (expected ~2932 / ~6500).
FP4_UNR in the generic kernel on the RTMASK base (e0251, B=8): 12 +182 us, 6 +128, 9 +88: 4 stays.
Batch-1 kernel (e0252): RTL1 (port of a2 s0021, router strip prefetched into L1 during the MLP-input mix) +30 us;
BF_UNR 3 +102 us. Off.
## s0045 track A "A K1 BPG:1 BPX=8:1 KN EXQ:N NG2NI:N KVC4=0:N RTMASK=0:1" (+SMTRIM, SM1 defaults): valid 2994 / 6743
(expected ~2975 / ~6700). Track-A base. Track B s0044 (2928 / 6498) against it: -66 us B=1 (2.2%), -245 us B=8 (3.6%).
GDU=3 (linearised gate's D strip in 3-unit batches, batch-1 kernel, a2 s0019 idea): +77 us (e0253). Off.
Generic-kernel batched polls on the RTMASK base (e0254, B=8): BPX 68 / 132 / 196 / 20: +0 / -24 / -13 / -16 us: noise level, not taken.
ATP with 64-key items for B > 2 (fits the union, smem 133.5 KB): +202 us at contexts 576-1088, +57 at 200-712, and not bit-identical at B=8 (64-key records change the merge). Dead (e0255).
L0T in the batch-1 kernel on the current base (e0256): +21 us (out of line), +92 (inlined). Stays :N.
## Two-level head sketch, offline (tests/sketch4_study.py, 384 real final states): rows a level-1 bound cannot exclude
int4: median 198,924 of 200,058 rows (useless); int5: median 2,513 rows / 1,502 strips, max 133,576 rows; int6: median
92 rows / 70 strips (p90 391, max 2,542). int6 + int8 on surviving strips: 346 MB + 2.6 MB median (94 MB max) per
token against 461 MB for the int8 sketch: -25% of the head's bytes = ~19 us at B=1 at the head's 6 TB/s. Not built:
a 6-bit unpack path into the int8 MMA fragments plus a second pass, for 0.6% of the step, with 3 h left.
## W13IL (w13 queue interleaved across experts): the exec hung (e0259, no output): blocks with <= 8 items take the
round path, which kept the old item -> (expert, strip) mapping, so some items ran twice and others never and w2
waited forever. A correct version must change the mapping in both paths. Removed; card 1 blocked until the exec's
50-minute timeout.
## s0046 final-source check: valid 2934 / 6497 (s0044: 2928 / 6498): the committed tree reproduces the best build.
W13IL v2 (one item mapping for the queue and the round paths): bit-identical (bitcheck), limits PASS, +27 us at
contexts 200-712 and +46 us at 576-1088 (B=8, e0260). The w2 wait is not the experts' completion order. Off.

# Summary of the run (11.6 h)
| stage | B=1 HTTP | B=8 HTTP | what |
|---|---|---|---|
| own engine v1 (s0003) | 6578 | 12658 | |
| a2 s0006 base (s0006) | 3723 | 7707 | adopted track-A engine |
| + MSGZ + SKH (s0012) | 3612 | 7583 | track B: per-pass message gates, certified int8 head sketch |
| + FGPLACE, GDL2, RTREDUX (s0018) | 3380 | 7404 | track A ports / one prefetch |
| + K1 batch-1 kernel (s0020) | 3150 | 7444 | same source compiled for B=1: spills 524 -> 36 B |
| + per-kernel items (s0023) | 3109 | 7401 | MSGZ / L0T only where they pay |
| + EXQ, NG2NI (s0029) | 3083 | 6725 | expert work queue, rare path out of line |
| + GLIN (s0037) | 3005 | 6635 | track B: linearised attention gate (batch-1 kernel) |
| + SMTRIM, SM1 (s0042) | 2944 | 6601 | shared memory under carve-out steps |
| + KVC4, RTMASK (s0044) | **2928** | **6498** | a2 ports, per kernel |
Track A final (s0045): 2994 / 6743. vLLM: 3692 / 5077. B=1 1.26x vLLM; B=8 0.78x.
The weights study (notes/weights_study.md) found no exact redundancy worth its decode in this kernel; the track-B
wins come from the model's algebra (per-pass messages, softcap-aware sketch, linear depth mixes, token-only layer 0).
## Sweep e0261 on the s0044 base: FP8_UNR (MXFP8 units in flight, layers 20-23) 3 / 6 in the generic kernel +218 /
+293 us at B=8; in the batch-1 kernel -3 / +67. FP4_UNR 6 in the batch-1 kernel: **-43 us** (it was +3 before SM1 /
KVC4 changed the kernel's registers).
## s0047 FP4_UNR=6:1: valid 2879 / 6509, NEW BEST at B=1 (expected ~2890).
Batch-1 kernel on the FP4_UNR 6 base (e0262): BPX 12 +53, BPX 24 -3, BPX 10 +6, FP8_UNR 3 +20; GDPF does not fit
the 99 KB layout (static assert). Bitcheck of FP4_UNR=6:1 against s0044's build: PASS.
## s0048 track A + FP4_UNR=6:1: valid 2949 / 6730 (expected ~2950): final track-A base.
Final: track B s0047 2879 / 6509 against track A s0048 2949 / 6730: -70 us at B=1 (2.4%), -221 us at B=8 (3.3%).
Sweep e0263: generic MSGZ_PF -6 (noise), NI_ATTN out of line +239, NI_GATEFG out of line +170; batch-1 FP4_UNR 8 /
9: +121 / +138. Nothing taken.
TUNE16 (16 start-up placement trials instead of 8): -12 / +4 us at B=1 (e0265, 2 x 2 instances): noise, not taken.
## s0049 final-source check 2: valid 2878 / 6509 (s0047: 2879 / 6509): the committed tree reproduces the best build.
Sweep e0266 on the final base: GLIN=2 +29, GLIN=2 + GLIN_MMF +20, GLIN_MF +13 us at B=1. Nothing taken.
## Final build against the reference (tests/check_ref.py, own texts, e0267)
B=1, 3 x 160 positions: rel L2 median 0.0332 / 0.0579 / 0.0236, max 0.33 / 0.43 / 0.19, top-1 478 / 480.
B=8, 64 positions x 8: rel L2 median 0.0232, max 0.44, top-1 505 / 512. (The evaluator's gates on its hidden
sequences passed for every submission: s0049 valid.)
MSGZ4 (four controllers' messages per (layer, token) item at the pass start, generic kernel): bit-identical (bitcheck), but +160 us at B=8 (e0268). Off.
FGR (F/G tasks kept off D-job blocks for up to 2 / 3 rounds, generic kernel): -11 (noise) / +203 us at B=8, FGR 2 +26 at B=2 (e0269). Off (FGR 1 = previous rule).
## s0050 final-source check 3: valid 2878 / 6509: the final committed tree (engine/TRACK as is) is the best build.
DKS=2/3 in the batch-1 kernel (e0271): hangs. With GLIN the attention gates' D values come from ph_gate_lin (gvt),
but ph_gate_fg(c0 = 0) still derives the K-split from gate_ks(B, 3) and polls gvp partials nobody writes.
DKS must stay 1 wherever GLIN is on (a guard for a later run: gate_ks() = 1 when GLIN && !MSGZ). The exec runs into
its 50-minute limit; the final tree (DKS 1) is unaffected.
## s0051 final-source check 4: valid 2879 / 6511: the final committed tree validates (final state).
## Expert similarity (e0272): experts near-orthogonal (max pair cosine 0.002-0.012), no shared component; notes/weights_study.md section 9.
BPX=132:N recheck (e0273, 3 instances, B=8): -26 us (contexts 200-712), -10 us (576-1088). Submitted as the last change.
## s0052 '... BPX=132:N': valid 2880 / 6486 (expected ~2879 / ~6490): best B=8. FINAL STATE (engine/TRACK).
BPX=196:N on the final build (e0274, 3 instances, B=8): +12 us. Not taken; final state stays s0052.
