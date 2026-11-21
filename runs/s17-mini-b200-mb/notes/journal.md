# Journal (run s17-mini-b200-mb, merge track B)

## Setup (0.0-0.4 h)
Read the ledgers: a2 s0025 (track A) 2755 / 5825 beats b2 s0047 (track B) 2879 / 6509 at both batch sizes; b2's
base is a2 s0006 + ports up to a2 s0015/s0020/s0022, missing a2 s0016-s0025 (expert per-group barriers, split of
>4-token experts with the NG=2 code deleted: -240 us B=1 / -860 B=8, loads before polls s0022-s0024) and
--use_fast_math (a2 s0012, -230 us B=1; b2 declined it as "an approximation"). Track B over track A in b2 was only
-70 / -221 us. Decision: base = a2 s0025, port b2's track-B items onto it one at a time (fewer and smaller ports
than bringing ten a2 changes into b2). fast math: a2 used it on track A and the gates accept it; it is a rounding
choice of the implementation, not a change of the function; kept as part of the base.
## s0001 a2 s0025 unchanged: valid 2900 / 5860 (expected 2755 / 5825): B=1 +145 us. In process 2856 (a2: 2730).
## s0002 b2 s0047 unchanged: valid 2949 / 6511 (b2: 2879 / 6509): B=1 +70. Card 0 of this pod is slower at B=1;
B=8 the same. Card 1 runs a2's code at 2744 us (e0001). The a2 base is ahead of b2 by 49 / 651 us here.
## K1 (port of b2 s0020): batch-1 kernel compiled separately (-DKB1, HDB macro): e0001 A/B -8 us B=1, -16 B=8
(a2's kernel already spills little: 116 -> 80 B spill loads). Kept as infrastructure for per-kernel items.
Three ports delegated to subagents in variants/{skh,msgz,glin} (each its own copy, card 1 shared).
## B=8 account on card 1 (e0002, crit.py, A K1): attention 10.8 us per application at ctx 200, 31.8 at ctx 1000
(+1.0 ms per step); w13 + w2 ~50 us flat. Attention items (128 keys, sub=2): load wave 3.3 us, bias 0.6, scores 1.4,
softmax + PV 1.0, merger 3.8 (subprof); ctx 1000 = 512 items over 152 blocks = 3.4 sequential waves. K/V bytes at
B=8 ctx 1000: 32.8 MB per application = 5.5 us at 6 TB/s: the phase is a latency chain, not bytes.
## ATT3 (new, track A): B>=3 attention by 3 independent 8-warp teams per block, one 64-key record each (named
barriers, unpadded XOR-swizzled K/V tiles so that 3 teams fit the union: 108.9 KB <= xs 111.0 KB), then a block-level
merge per (token, kv head) from arec (record 0 included). Record arithmetic copied from ph_attn (bit-identity
expected). v1 noinline: attention 31.8 -> 25.9 us/app but the call costs every other phase 1-2 us (stack 72 -> 256 B):
-22 us at ctx ~960 (e0021). Inlined (NI_ATT3=__forceinline__:N): -296 us at ctx ~960 (e0030; possibly v2 code).
v2 pipelined: next record's K tile issued after the scores, its V tile after the PV (cp.async groups).
## s0003 "A K1 ATT3:N": valid 3190 / 5701 (expected ~2900 / ~5680). B=8 -158 us as predicted (bit-identical,
in process generate 5483 -> 5429). B=1 +290: not ATT3 (the batch-1 kernel has no ATT3 code) but placement: the
same "A K1" build ran 2737 us (my e0001) and 3166 us (an agent's e0019) at B=1 on card 1. B=1 time is bimodal with
the buffers' placement. Port of b2 s0010's start-up placement tuning (8 random placements of the polled buffers,
timed at B=1 and B=8, keep the fastest) next.
## K1 re-measured: api.py parsed TRACK at import, so ab2 ran both instances with the first TRACK (e0001 was A vs A).
Fixed (TRACK re-read per Engine). Properly: K1 +384 us at B=1 (2752 vs 3136, e0045): the batch-1 compile is slower
on this base. Dropped. ATT3 in the single kernel: B=1 +14 (noise), B=8 -81 at ctx 64-320 (e0045).
Placement tuning (b2 s0010 port): no effect on card 1 (+17 / +1 us, 3 instances each, e0049): left off (NOTUNE).
## s0004 "A ATT3 NOTUNE": valid 2911 / 5701 (expected ~2900 / ~5700): confirmed. New track-A base.
ATT3 sub-marks (e0060, B=8 ctx 1000): per record 6.6 us (polls + K wait 1.9, bias + scores 1.2, softmax + V wait
1.3, PV 1.2, next V issue 0.9), merge 4.9; 1024 records on 456 teams = 3 rounds -> 25.6 us per application. Latency
bound at ~1/3 of the bandwidth; more teams do not fit the shared memory. Left as is.
## w2a at B=8 (e0069, ctx 500): staging 1.55 us, round 1 3.9, later rounds 3.2 (8 items x 10 KB per round per SM =
26 GB/s per SM); w2 crit span 26.8 us vs w13 23.0 for twice the bytes. W2WG=2 W2_UNR=9 (2 warps x 9 units per item,
12 items per round): B=8 -14 (noise), B=1 +36 (e0065): bytes in flight are not the limit. Default stays 3.
## Agents' ports (variants/, ~1.7 h each)
- SKH (b2 s0009 idea, subagent; variants/skh): certificate redone for the fast-math tanh: eps = max |head_cap(a) -
  30 tanh(a/30)| over all 2^32 float inputs, measured at load with the same inline function (2.4057e-4); a strip is
  skipped only if G(u) + 2 eps + 1e-6 < G(L) (in double); sigma a power of two (exact digits under fast math); fp32
  chain term 6e-4 n |x|. Bitcheck identical (exact path), greedy tokens identical (generate + serving, B=1..8,
  contexts to 2048), 1-1.6 exact strips per token. A/B on the single-kernel base: -31 B=1 / -52 B=8.
- GLIN (b2 s0037, subagent; variants/glin): correct (gate vectors closer to the exact function than track A's),
  not bit-identical at B=1; +86..+110 us at B=1 on this base (D leaves the path, F/G and the norm get slower: the same
  signature as a2's message-precompute attempts). Not taken.
- MSGZ / L0T on the a2 base (variants/msgz): MSGZ -50, L0T -41 at B=8 alone, -92..-101 together in a separate B>=2
  entry; moot since B>=2 is served by a1 (which already computes messages per pass; L0T on a1: +28..+33, no gain).
## Dual engine (structure from s17-mini-b200-ma s0002): a2 lineage at B=1, a1 s0025 at B>=2.
Card 1, same job (e0082): B=8 a1 5919 vs a2+ATT3 6355 at ctx ~960, 5927 vs 6003 at ~560. ATT3 becomes moot.
## s0005 dual engine, track A: valid 2888 / 5464 (expected ~2900 / ~5450). Track-A base.
## s0006 + SKH on the a2 side (track B): valid 2867 / 5480 (expected ~2845 / ~5465): B=1 -21 (card 1 said -49).
## Limits of the dual engine (e0091): B=1,2,3,5,8 to context 2048 (generate), finite, ids valid; errors beyond 2048,
for id 201024 and on a batch change.
## HTTP vs in process (s0006): B=1 2867 vs w1-like generate 2845 (22 us); the serving path is not a loss.
## MKPROF + SKH: the a2 kernel built with -DMKPROF and SKH fails the cooperative launch (profile builds only). The
ma run's "KB1 with the inactive mark sites" (-20 us at B=1) therefore does not combine with SKH as is: not taken.
## s0007: a1 kernel from s17-mini-b200-ma s0004 (gate phase 0 without mixer blocks, warp-per-message M phase):
valid 2862 / 5364 (expected ~2870 / ~5380; card 1 -115 at B=8). NEW BEST.
## s0008 track A of the current tree ("A NOTUNE": dual engine, ma s0004 a1 kernel, no SKH): valid 2892 / 5349
(expected ~2885 / ~5365). Best track-A engine. Track B (s0007) against it: -30 us at B=1 (SKH), +15 at B=8 (noise:
the B=8 engines are identical).
## s0009 start-up placement tuning on (B=1 objective, 9 trials): valid 2838 / 5352 (expected 2750 if card 0's
deficit were placement, else ~2862): -24 us. Card 0's trials (stats): 2768-2802 us (34 us spread, best taken);
card 1's: 2698-2709. So card 0 is ~80 us slower than card 1 for any placement (the card), and placement is worth
~20 us there. NEW BEST.
## a1 placement tuning (TUNE8 token, mk_set_params, 20 polled buffers re-placed, 9 trials at B=8; e0103): trials
5055-5068 us (13 us spread), B=8 generate 5725 vs 5718, tokens identical (B=8, B=1, 512 steps). Off. Same job: the
B=1 engine ran 2802 vs 2721 us in two processes with the same build and tuning on: process-to-process variance
(allocation history of the weights, not the polled buffers) of +-40 us at B=1 on card 1.
## SKH:8 (sketch head in the a1 engine, subagent, variants/a1skh): exact (bitcheck B=2/8, tokens identical B=2..8
to ctx 2048, serving), -16 us at B=8 (instance spread +-20), -31 at B=2; head 141 -> 117 us but other phases slower
(call-site spills). Not merged.
s0010 final-source check: valid 2838 / 5347 (s0009 2838 / 5352): reproduced.
s0011 TUNE8: valid 2837 / 5377 (expected ~5330 or unchanged): +30 at B=8 (noise); card 0 a1 trials 5048-5060 (11 us spread). Off; TRACK back to B SKH.
## s0012 + SKH:8 (sketch head in the a1 engine too): valid 2839 / 5335 (expected ~2838 / ~5335): B=8 -12..-17 vs
s0009/s0010. Kept: final configuration "B SKH SKH:8".
## MKPROF + SKH crash found: sk_bound had an unguarded profile write (thread 0 wrote through a null c.prof in MKPROF
builds; compute-sanitizer e0107). Fixed (SUBMARK). Production builds never compiled it.
## KB1 + MKPROF on the B=1 engine (idea from s17-mini-b200-ma s0003: batch a compile-time 1 with the inactive mark
sites; KB1 alone is the +384 us cliff): card 1, 3 instances each, -40 us, tokens identical (e0108).
## s0013 "B SKH SKH:8 KB1 MKPROF": valid 2764 / 5366 (expected ~2805 / ~5335): B=1 -75 (layout effect, larger on
card 0), B=8 +31 (noise: engine unchanged). NEW BEST.
## MKPROF:8 (the same layout trick on the a1 kernel): +315 us at B=8 (e0109, 2 processes each). Not taken.
## s0014 track A of the final tree ("A KB1 MKPROF"): valid 2799 / 5369 (expected ~2795 / ~5350). Best track A.
Track B s0013 against it: -35 us at B=1 (SKH), -3 at B=8 (SKH:8, noise).
## s0015 a1 kernel + s17-mini-b200-ma s0006/s0007 decode changes (NVFP4 block scales on the FMA pipe, MXFP8 decoded
straight to bf16 bits; bitwise equal per ma's checks; tokens identical here): valid 2767 / 5265 (expected ~2765 /
~5240; card 1 -134). NEW BEST at B=8.
## Final-tree verification (e0111): bitcheck "A KB1 MKPROF" vs "B SKH SKH:8 KB1 MKPROF", 1152 positions, B=1 and
B=8: bit_identical 1.0 at both; greedy tokens (generate 512, B=8 and B=1) identical between the tracks; limits
(B=1,2,3,5,8 to 2048, ids, batch change) pass. B=1 vs B=8 not bitwise (two engines), as expected.
HTTP vs in process (s0015): B=1 2767 vs w1-like generate 2752; B=8 5265 vs 5359 (the serving path is not a loss).
## Load-depth sweep on the final B=1 kernel (KB1 + MKPROF + SKH; e0112, 2 instances each, base 2658): FP4_UNR 4 +54,
SK_UNR 8 +22, and FP4_UNR 8 / BF_UNR 3 / FP8_UNR 6 all ~3140-3165 (+475..+505): the same slow mode as the plain KB1
compile (3136). The batch-1 kernel has two code-generation modes ~480 us apart; the final build sits in the fast one
and small source changes can flip it (check any future change of this kernel against this).
## s0016 faster SKH digit stage in the a1 head (subagent; same digit values, tokens identical): valid 2767 / 5240
(expected ~2767 / 5230-5260): B=8 -25. Final.
## s0017 track A of the s0016 tree: valid 2800 / 5221: SKH:8 does not pay at B=8 in this layout (s0016 5240).
## ma s0009 (NVFP4 block-scale pairs by one cvt.rn.f16x2.e4m3x2): card 1 -142 at B=8 with SKH:8 off (e0117; SKH:8 on
was +10 on top), tokens identical. s0018 "B SKH KB1 MKPROF": valid 2757 / 5167 (expected ~2767 / ~5100): B=8 -54
vs s0017. FINAL.
## s0019 track A of the final tree ("A KB1 MKPROF"): valid 2801 / 5170 (expected ~2795 / ~5167). Best track A.
Track B s0018 against it: -44 us at B=1 (SKH), -3 at B=8 (same engine; noise).
## PMAXC 2 + SCNT_REL 1 in the a1 engine (s17-mini-b200-ma s0010 settings; attention pieces of <= 2 chunks: merge
rounding changes, greedy tokens at B=8 differ at near-ties; track A): card 1 -49 at B=8 (e0118).
s0020: valid 2770 / 5128 (expected ~2760 / ~5135): B=8 -39; B=1 +13 is noise (the B=1 engine is unchanged from
s0018). FINAL configuration: "B SKH KB1 MKPROF PMAXC=2:8 SCNT_REL=1:8".
## s0021 track A of the final tree ("A KB1 MKPROF PMAXC=2:8 SCNT_REL=1:8"): valid 2806 / 5155 (expected ~2800 /
~5130). Best track A of the final tree. Track B s0020 against it: -36 us at B=1 (SKH), -27 at B=8 (same B>=2 engine:
evaluation noise, about +-25 us at B=8 across evaluations of identical engines in this run).
