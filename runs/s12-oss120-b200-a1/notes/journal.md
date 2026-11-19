# Journal

## s0001 v1 megakernel (grid barriers, register loads)
Expected: gates pass, HTTP ~3600 B=1. Measured: valid, 3613 / 8775 (B=1/B=8). numeric median 0.0135 (limit 0.0303),
long median 0.049 (limit 0.096), batch bitwise equal. Concluded: correctness baseline holds; own fp32 reference
(tools/ref_check.py) tracks the evaluator's verdicts.

## s0002/s0003 (failed: build header outside engine/; then crash past ~400 positions at B=1)
The attention's P array (8 x 2064 bf16) had grown past the red region after RED_SLOTS shrank to 50 and overwrote the
task tables / mbarriers. Not caught because my tests stopped at position 160. tools/presubmit.sh now runs B=1 and
B=8 to position 1152 before every submission.

## s0004 TMA rings + prefetch
Expected HTTP ~2500 B=1, ~7600 B=8. Measured 2523 / 6735 (in process generate 2360). B=8 better than predicted
(real text routes narrower than my random-token bench). Concluded: the in-process generate predicts HTTP B=1 within
~160 us here (HTTP - generate = 163 us; s0001: 113 us).
HTTP figure includes the prompt: 63 prompt steps (no head) + 1024 decode steps over 1024 tokens: 2360 x 1087/1024
= 2505 ~ 2523 measured. So HTTP ~ generate x 1.06 (+ ~20 us). Multi-token prompt passes would remove it but the
contract says one token per sequence per model pass: not done.

## s0005 flat down phase at B<=2
Expected HTTP ~2330 B=1, B=8 unchanged. Measured 2276 / 6738 (generate 2108 in process). Bits identical to the B=8
path (ref_check --B 3, flat B=1 vs non-flat batch). Concluded: per-block imbalance (blocks with 2x the bytes) is a
first-order loss here, unlike in s2 (dense, many tiles per block); same likely for o_proj (180 rowtiles) and qkv
(320 rowtiles over 152 blocks).

Barrier cost on its own (bench/barrier.cu, e0066): 1.21-1.33 us per grid barrier with all 152 blocks arriving
together (atomic add + acquire poll; red.release no better; a gather by block 0 is 3.6-52 us). 6 per layer = 216 per
step = ~270 us of pure barrier latency.

L2 prefetch (cp.async.bulk.prefetch.L2) of each warp's whole task list when the list is built (e0067-e0071):
generate 2178 (off) -> 2323 (experts only) / 2317 (dense phases except head) / 2389 (both) / 2432 (incl. head).
Worse in every form; likely the prefetches occupy the TMA unit ahead of the ring's own copies. Off (L2PF=0).

Flat split of qkv / o_proj over all blocks with the last-arriving block finishing a split rowtile (threadfence +
atomic counter + reread of the other part), e0078: generate 2179 -> 2452 us (worse). The finishing chain (fence,
atomic, reread, per segment, serially) costs more than the imbalance it removes. Stashed (git stash).

## Accuracy yardstick (tools/ref_check.py --T 300 --B 8, 8 texts repeated, own fp32 reference)
HF BF16 path (tools/hf_check.py 300 8, e0100): median 0.0878, max 0.972, top-1 0.9171.
Engine with scaled cvt (e0101): median 0.0881, top-1 0.9212; with fp32 post-scaling (e0102): 0.0841 / 0.9179.
The texts are hard (repetition, near-ties) but the engine tracks HF; use this pair as the reference point.

## Scaled e2m1 conversion (cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2)
Exists on sm_100a and is exact for all codes x scales 100..150 (bench/cvt3.cu, exhaustive). One instruction per
two dequantised BF16 weights incl. the E8M0 scale -> the fp32 post-scaling (8 FFMA + zero-init per chunk) is gone and
the mma accumulates the dequantised weights exactly as the BF16-weight data flow does. generate 2145-2186 -> 2125.
Without any fp4 compute (CNOP1 probe) the step was 1891: fp4 compute was ~230 us of the step.

## Grid barriers removed (NOBAR probe, results wrong): generate 2125 -> 1413 us.
Barriers + waiting for the slowest block cost ~710 us/step (~20 us/layer); the per-block work is ~39 us/layer.

## s0006 scaled cvt + norm weights by bulk copy
Expected HTTP ~2215 B=1. Measured 2161 / 6872 (generate 2010). B=1 now under SGLang (2197). B=8 is 2x off SGLang.
Profile at position 900 (e0105): attention span 13 us/layer (vs 4.6 at 64): 8 blocks at B=1, each reading all K/V of
its head; over the HTTP positions 64..1088 attention averages ~9 us/layer -> next: split keys over warps.

## s0007 split attention (64-key chunks, warp each, last warp merges)
Expected HTTP B=1 ~2075, B=8 ~6350. Measured 2079 / 6640. A block-per-8-chunks variant with smem merge was slower
on 64+1024 (2112 vs 2067); keeping sliding layers on the old per-unit block path (ATT_HYB) changed B=1 by <6 us and
cost B=8 200-400 us: all layers split. top-1 vs fp64 0.9785 (limit 0.9714; s0004 had 0.9868): routing draw.

## s0008 B>=3 expert phases without rounds + even down units + combine step
Expected B=8 ~4500. Measured 4646 (B=1 2099, +20: noise / allocation). In-register segmented reduction keeps the bits
of B=1 (bitwise batch check still true).

## after s0008: batched norm loads (B=8 4930 -> 4477 in process), multi-slot down rounds (-> 4423), interleaved
MXFP4 records (one bulk copy per stage instead of codes + scales: B=1 2050 -> 2029, B=8 4423 -> 4318). consume
harness: with fp4 compute removed (CNOP1) fp4 rings still streamed only 5.5 TB/s vs 6.85 for BF16 -> the second,
small bulk copy per stage was the limit, not the decode.

## s0009 interleaved MXFP4 records
Expected HTTP B=1 ~2045, B=8 ~4070. Measured 2071 / 4325.
Critical-path view per layer (tools/timeline.py on mkprof, pos 300), B=1: A 10.6 (work median 7.5, max 9.4),
B 6.4 (attention 5.3), C 7.6 (median 3.75, max 6.6: R=2 blocks), D 5.0 (router 4.0), E 15.1 (median 12.3),
F 8.1; barrier release after the last arrival 0.85-1.1 us each. B=8: A 12.9, B 8.6, C 11.2, D 8.6, E 43.3, F 25.6,
combine 2.1.

## s0010 router K split (RQ=4 over 32 blocks)
Expected HTTP B=1 ~2045. Measured 2035 / 4279. In-process sweep RQ 1/2/4/8: 2037/2013/2006/2026.
Ring top-up right after barriers (issue the 3rd stage before the norm instead of in the consume loop): +40 us at
B=1 (critical small loads queue behind weight stages). Off.

## s0011 top-4 by integer max on packed keys, union in warp 0
Expected HTTP B=1 ~2000. Measured 1997 (= the 1.10x line exactly) / 4144.

## s0012 E->F barrier replaced by per-slot counters
Expected HTTP B=1 ~1960, B=8 ~4030. Measured 1959 / 4017. (A per-unit signal with a fence inside the consume loop
was +66 us at B=8: the fence stalls the warp's stream; one signal per block and slot after the block's units.)

Norm sources by bulk copy into xs + in-place normalisation (NO_XSTAGE A/B, e0151): B=1 1961 -> 2061, B=8 4187 -> 4442
(worse). The o_proj input copy with batched __ldcg loads: B=1 +29, B=8 -23 vs s0012 (register picture changed: 246
regs). Both reverted.

## s0013 two kernel instances (B<=2 / B>=3), warp-per-column norm in the wide one
Expected HTTP B=1 ~1950, B=8 ~3950. Measured 1951 / 3953.

o_proj as 8-row tiles (weights as the mma B operand, activations as A; 360 octets -> 2-3 per block instead of 1-2
rowtiles), e0166/e0167: B=1 1926 -> 1934, B=8 4111 -> 4100; C's slowest block unchanged (6.6 -> 6.9 us), median
3.8 -> 4.9. The 28 two-rowtile blocks were not what bounded C. Reverted (patch in variants/kind3_o8.patch).

Consume harness corrected (each pass reads new memory; before, passes re-read data that partly stayed in L2; and an
earlier "fp4 compute costs nothing" (e0128) was measured with a CNOP1 switch that no longer existed in the source):
48 tasks x 45 items per block, 8 warps, ring 3 x 9: BF16 6.4-6.9 TB/s, fp4 5.8 TB/s; fp4 without the decode/mma
6.2 TB/s; the rest of the gap is the 6% larger records and the extra B/scale loads. TMA copy size and source
alignment alone do not matter (bench/tma_align.cu: 6.9-7.3 TB/s for 4608-4992 B copies, any 32 B alignment).
Fp4 consume with codes copied to registers before handing the stage back (CPYREG): B=1 +30, B=8 +207 us. Ring of
4 x 7 items: +180 us (segments of 9 no longer match stages). __trap() on a task-table overflow check: +40 us
(allocation). All reverted.
Next layer's q/k/v rows pulled into L2 by the non-router blocks during D (QKV_L2PF, B>=3): B=8 4111 -> 4163 (worse).
L2 prefetch has now lost in every place tried; the expert stream evicts it or it competes for the TMA unit.

## s0014 no B-column selects (+ batched o_proj input copy)
Expected HTTP B=1 ~1925, B=8 ~3810. Measured 1927 / 3824.
SASS of a full fp4 stage (9 chunks): 274 instructions = 144 F2FP + 36 PRMT + 36 HMMA + 27 LDS.128 + 9 LDS + 22
other. F2FP (e2m1 -> bf16x2, scaled or not) runs at ~44.6/clk/SM (bench/f2fp.cu), i.e. ~1/3 of full rate.

## s0015 two accumulators per stage
Expected HTTP B=1 ~1905, B=8 ~3600. Measured 1906 / 3756. top-1 vs fp64 0.981 (margin restored; was 0.9746).
B=8 missed by 156: tools/bench.py used random token ids, which route to ~27 experts per layer; real text routes
narrower, so the expert phases weigh less in the HTTP figure than in my bench. Bench switched to real text.
Attention merge with chunk statistics loaded at once and partials in batches of 4 chunks: same-card 64+1024
B=1 1900 -> 1888, B=8 4220 -> 4245 (kept: B=1 is the headline; B=8 within its run-to-run spread).
Probe: C->D barrier replaced by every thread polling the xm words for a sentinel (POLL_XM, results not guaranteed):
B=1 1870 -> 2134 (+264). 152 x 256 threads spinning on data swamp L2; data polling is not a cheap barrier here.
Dropped the sentinel-handover plan.

o_proj heavy blocks moved to 0..27 (router blocks keep one rowtile): 1857 -> 1862 (no gain); 3 prefetch stages for
the two-rowtile blocks: no gain. Reverted.
Sub-phase marks at B=1 pos 600 (e0213): A = combine 1.75 + norm 1.14 + consume 3.26 + epilogue 1.58 (+ wait 2.5);
C = copy 0.6 + consume 2.53 (imbalance 2.0) + epilogue 0.75 (+ wait 3.5); E = rlog+top-4 2.1 + stream 7.0 +
epilogue/prefetch 2.15; F 5.6 (+ wait 2.7); D 2.1 (+ 1.7). Epilogues wait on bias / cos / residual loads issued
only after the reduction: next, load those at phase start into shared memory.
Epilogue inputs (bias, rotary values, residual) loaded at phase start and kept in shared memory (EPRE): no gain at
B=1 (+7 us); and the combined A+C version produced wrong results (each half alone was correct; not debugged,
dropped). Lesson: a bitwise batch check (B=1 vs B=3) does not catch an error that is the same in both: the B=3
--noref check passed while the reference comparison showed median error 1.2. Always run the reference check.
Attention item timeline (warp-0 marks, e0216, B=8 pos 700): q/K/V loads + scores 3.05 us, exp + V mix + partial
store + fence 2.1, atomic 0.34; B=1: 1.5 / 1.7 / 0.5. Changes: (1) partial hand-over as one atom.add.acq_rel by
lane 0 after __syncwarp instead of __threadfence + atomicAdd: B=1 1857 -> 1844, B=8 4207 -> 4187 (kept);
(2) E->F slot signals as red.release.gpu after the block barrier instead of per-thread fences: neutral (kept);
(3) K/V cache in mma fragment order (2 x 16 B loads per key group instead of 8 x 4 B; V loaded with K before the
scores instead of after the softmax): testing.

## s0016 K/V fragment order + acq_rel partial hand-over
Expected HTTP B=1 ~1860, B=8 ~3720. Measured 1840 / 3708. (V loaded up front with K was worse: +48 us at B=1,
register pressure; V loaded after the scores with the new layout is the kept form.)
Input validation: ids outside the vocabulary are read as 0 by the kernel and flagged in host-mapped memory; the next
step/generate/reset raises ValueError (tools/edge_test.py: bad id, batch 9, submit checks, context full).
E/F halves at B>=3 (each half's units spread over all blocks; gate_up and down both do half 0 first so down units
never wait) with per-unit release signals: B=8 4214 -> 4491 (halves) / 4332 (per-unit signals only); B=1 +16.
More rounds and slot spans per block cost more than the wait they hide. Reverted.

## s0017 expert bias rows prefetched at unit start
Expected HTTP B=8 ~3500, B=1 ~1840. Measured 3545 / 1858 (B=1 in process 1834, unchanged: HTTP spread ~15-20 us).
B=8 timeline (pos 700): A 10.1, B 9.8, C 9.5, D 5.5, E 31.1, F 23.5 (was 26.6), combine 2.3. E is now ~92% of its
byte time (U~22: 1.27 MB per block), F ~80%.

## After s0017 (~5.2-6 agent-h): down-phase loads, generic-pointer serialization, register ceiling
B=8 profile (e0233, pos 700) with new F sub-marks: F's activation-row staging (np rows x 360 uint4 per block, one
load then one smem store per iteration) had imbalance 4.2x (blocks with many routed rows). Loads now issued 8 per
thread before their stores (load_act_rows): F 23.8 -> 20.8 us/layer; same-card A/B (abs.sh vs HEAD): B=8 step
3794 -> 3756, gen 3912 -> 3900; B=1 gen 1843 -> 1834. Bitwise B=8 batch check and ref check OK. Kept.
Sub-phase marks B=1 vs B=8 (e0238/e0239): A norm (norm_warp, B=8) 2.9 us vs 1.8 (flat); C copy 2.0 vs 0.6; C consume
3.4 vs 2.7; D 3.6 vs 2.2. Microbenchmark of norm_warp alone (bench/normb.cu, normb2.cu): 1.7 us even at B=1 / one
block; 2250 of its ~3650 cycles are the store loop: smem weight loads and fragment-order stores go through generic
pointers, so every iteration's load waits for the previous iteration's stores (possible aliasing). Loading the
weight words first: alone 2.13 -> 1.74 us at B=8, but in the kernel B=8 got worse (+25 us/step: registers).
Same idea in norm_to_smem (B<=2): spills, +140 us at B=1. combine_flat with loads first: spills (+165 us) or, as a
noinline function, +80 us. All three kept as off-by-default switches (NORM_WREG, NORM_BATCH, COMB_BATCH).
Lesson: both instances sit at the 255-register ceiling; any added live state in a phase costs more than the
latency it hides. Restructuring must lower register use elsewhere first (or move work into noinline helpers
whose call sites have little live state -- the noinline combine still lost 80 us, reason unknown).

## Expert prefetch into L2 from a predictor (XPF, ~6.3-7 agent-h): negative
Predictor (findings: 85.6% recall at top-4) computed in phase A by the 32 router blocks (router rows x ln2/ln1, BF16),
predicted top-4 union in phase B by warp 7 of idle blocks, cp.async.bulk.prefetch.L2 of the experts' gate_up (+down
at B<=2, cap 96 MB). In-kernel predictor recall at layer 35: 0.921 (tools/study/pchk.py). Results (same-card A/B):
XPF_NB=16: B=1 +350 us, B=8 +1500 us (16 SMs' TMA units busy with 3 MB of prefetch each -> stragglers);
XPF_NB=152: E stream 6.9 -> 7.2 us (no gain); per-lane prefetch.global.L2 (XPF=2): +880 us (64-bit index math per
line makes warp 7 a straggler). ncu DRAM bytes (64+16 steps, B=1): 80.10 GB -> 81.92 GB, i.e. only +113 MB/step:
the prefetched lines were hit, they just did not make E faster.
Why (bench/l2pf.cu, bench/consume.cu with -DREUSE -DTMA_NOHINT): a plain 3-stage TMA ring streams 233 KB/SM in
7.4 us cold, 2.2 us L2-warm, ~4 us after a prefetch (prefetcher identity irrelevant); the engine's own runner with
fp4 consume: 32 tasks x 9 chunks 5.7 us cold vs 5.2 us L2-hot; 10 x 45: 9.1 vs 8.6. Short expert streams are bound by
the per-warp pipeline (3 stages in flight, ~1.2 us per stage turn incl. consume), not by where the bytes come from.
Also: the engine's ring copies carry an EVICT_FIRST hint; evict-first lines are not retained in L2 (warm pass with
the hint: 5.4 us vs 2.2 without). XPF stays in the source, off.

## Track B head sketch (HSK, flat instance) (~7-8 agent-h)
Offline (tools/study/hsk_bound.py, own fp32 reference, 4 texts x 64 positions, BF16 head input): bound holds with
max |err| / bound 0.11; candidates per position median 2 rows (max 14), 16-row tiles median 2 (max 12) of 12568.
Kernel: int8 sketch records (90 m16n8k32 chunks + 3 BF16 tiles of error-block norms + row constants per rowtile,
605 MB), head input as two int8 digits per column (hi in B columns 0..3, lo in 4..7), per-block running best lower
bound (smem atomicMax) and candidate list, one grid barrier for the global bound, candidate rowtiles recomputed by
the exact head's own tasks (one warp per rowtile: same bits). tools/hsk_check.py: generate() == step()-greedy for
200 tokens at B=1 and B=2; ~2.0 candidate rowtiles per head at B=1, 3.1 at B=2; no list overflow.
Head (mkprof, B=1 pos 600): 176.6 -> 118.4 us: sketch stream 82 us (605 MB, 7.3 TB/s), prep 8.8 (combine + norm +
digits; the digit prep was 16 us with one warp per column, now all warps), exact rescoring of a candidate tile
~9 us on one warp (even with the tile prefetched into L2 before the barrier).
Same-card A/B (HSK=0 vs 1): gen/serve B=1 1842 -> 1807, B=2 2350 -> 2331; step() +25 us (B=1) although step never
runs the sketch: the layer phases of the flat kernel got slower (F +0.36 us/layer) from code layout / allocation;
inline or noinline makes no difference (noinline: +25, inline: +19).

## s0018 track B: HSK head sketch (flat instance) + batched down-phase loads
Expected HTTP B=1 ~1795, B=8 ~3530. Measured 1799 / 3500 (in process generate 1775). Gates: batch bitwise equal,
generate 8/8 and http_tokens 8/8 identical to step-greedy, numeric median 0.0149 (limit 0.0303).
Wide sketch (HSK_WIDE): correct at B=3/8 after two fixes (lane B offset t*8 for KIND 3; candidate list in the free
digit area, cap 2896: 15.6 candidate tiles per head at B=8, no overflow), head 180 -> 137 us, but the call raised
the wide instance's spills (stack 112 -> 240 B) and its layers lost ~50 us/step: off by default.
Wide prep: 19 us in the kernel with fully unrolled straight-line code vs 4 us warm in a microbenchmark; compact
rolled loops over a shared-memory copy of the rows: 9.6 us. Conversion-pipe ops (186 F2I + 184 FRND) replaced by
the magic-add rounding (v 2^k + 1.5 2^23) and integer digit split.
Compact norm_warp (rows staged into xs with batched loads, normed in place by rolled loops, same bits): B=8
+270 us/step (gen 3921 -> 4189), B=1 unchanged. The rolled loops pay a shared-memory round trip per iteration;
the instruction-fetch explanation for the slow unrolled HSK prep does not generalise (NORM_C kept, off).

## C-phase imbalance and the norms (~8-9 agent-h)
Per-block C work (tools/blockwork.py, B=1 pos 600): blocks with two o_proj rowtiles 6.37 us, one rowtile 4.05: C is
bound by its 28 two-rowtile blocks (+2.3 us/layer). OSPLIT (one rowtile per block, the 28 extra rowtiles in K
quarters on blocks 0..111, partials summed ((q0+q1)+q2)+q3 at D's start by every block into xm): correct (bitwise
batch check, ref median 0.026) but B=1 +70, B=8 +1000 us/step: 152 blocks storing the same 448 xm rows contend in L2
and the loop's loads serialize behind its stores. A working form has to patch each block's own norm input instead
(~1-1.5 us of loads at B=8 vs ~1.8 us gained): not pursued. OSPLIT off.
norm_warp weight loads one iteration ahead (NORM_PIPE1): warm microbenchmark 2.14 -> 1.75 us at B=8, in the kernel
unchanged. Inlined wide sketch (HSK_WIDE=1, flat via a noinline wrapper): B=8 serve -15 us, B=1 unchanged.
OSPLIT v2 (each block patches its own D-norm input from the quarter partials; blocks 0..27 write their split rows
of xm, one writer per row): correct, net neutral (B=1 +3, B=8 -1): mkprof C 7.85 -> 6.94, D 4.04 -> 4.71 and A
10.10 -> 11.12 (A's code unchanged: allocation). Off.
V loads by cp.async into xs with the K loads (V_ASYNC): B=1 +41, B=8 +55. Off.
Barrier forms (bench/cbar.cu): flat 1.26 us; two-level with hardware cluster barriers (C = 2/4/8): 1.59 / 1.51 /
1.79 us (the cluster sync costs ~0.3 us each way). Barrier floor here ~1.2 us; replacing grid barriers by counters
only pays where the dependency is partial (none of A->B, C->D, D->E is, given where the slow blocks sit).
Router over 64 blocks (RQ=8): B=1 +27, B=8 +8. int4/5/6-bit head sketches (offline, QMAX 7/15/31): candidates
median 142k / 677 / 26 rows: the certified bound is too loose below int8.

## Bulk copies for the wide instance's staging (~9.5-10 agent-h)
Consume harness (bench/consume.cu, NOCONSUME / REUSE / TMA_NOHINT switches): from L2 the runner streams 67-90 GB/s/SM
without per-item work, 38 GB/s/SM with the fp4 decode (cap ~5.9 TB/s, just under HBM); from HBM both ~41-43. The cap
is not the conversions (NOCVT 41), not the mma chain (two 2-mma chains per item: no change; ACC3/ACC1 worse), not
shared-memory bandwidth (B loads of absent columns predicated off: slower). Not understood; left.
s0019 (track B, wide sketch inlined): B=1 1803, B=8 3459 (expected ~3470). B=8 HTTP repeats spread 3387-3568 with
card 1 idle as well: the spread is the B=8 HTTP measurement's own.
ATMA (wide only): attention output written in fragment order, phase C stages each row with one bulk copy instead of
16 loads + scattered smem stores per thread: B=8 -46 us, B=1 unchanged (applied to both instances first: B=1 +10,
from code in the flat instance). FTMA (wide only): gate_up writes act rows in fragment order, the down phase stages
them by bulk copies: B=8 -31 us (B=1 +8 until the index change was also made per-instance). Rule: every
instance-specific change behind `if constexpr (WIDE)`; dead code in the other instance still moves its allocation.
NORM_TMA (wide A / D norms stage their input rows by bulk copy): B=8 +160 us. At a phase start the SM's bulk-copy
queue holds the prefetched weight stages, and the latency-critical rows wait behind them (C's attention-output copy
is fine: its weights were prefetched a whole phase earlier). Off.

## s0020 track B: wide-instance staging by bulk copies (ATMA + FTMA), wide sketch off
Expected HTTP B=1 ~1800, B=8 ~3415. Measured 1798.4 / 3413.3 (B=8 repeats 3314-3504). Milestone `parity` passed
(SGLang 1.22x at B=1, 1.003x at B=8). With the sketch present in the wide instance ATMA+FTMA gave only -35 us at
B=8 (vs -77 without it) and the inlined wide sketch had turned into +14 us: HSK_WIDE off again.

## s0021 track B: the B<=2 instance stages attention output and act rows by bulk copies too
Expected HTTP B=1 ~1776, B=8 ~3413. Measured 1774.0 / 3403 (in process 1755). Per-instance switches ATMA_FLAT /
FTMA_FLAT (B=1 -22, B=2 -32 us in process); batch bitwise check B=8 (wide, fragment-order staging) vs B=1 (flat) holds.
Head-sketch first task pulled into L2 during the head prep (HSK_L2PF 76 / 36 items per warp): B=1 +6 / +1 us. Off.

## s0022 track A (TRACK=A: same engine without the head sketch)
Expected HTTP B=1 ~1830, B=8 ~3403. Measured 1830.9 / 3391: parity on track A too (best track-A B=1 so far: s0016
1840; B=8: s0017 3545). The B>=3 path is identical in both tracks now (the sketch is off there).

## Overlap attention / o_proj (BCOV) and K-split re-tune (~9-9.5 agent-h)
BCOV: no grid barrier between B and C; the warp finishing a (sequence, kv head) output adds 1 to adone[kvh]
(release); each C task's warp waits (lane 0 acquire-polls) for the kv head its 256-column K segment covers and copies
those columns into xs itself. Correct (bitwise batch check, ref check, long contexts) but B=1 +65, B=8 +68 us/step.
Probably 1216 warps polling 8 counter words. Off (BCOV 0); a one-poller-per-block form is untested.
Prefetch depth 3 at the barriers (PF_STAGES=3): B=1 +28. Off.
K-split counts re-tuned on the current engine (track B build): NSEG_O 16 -> 8 (one 32-kstep task per warp for a
one-rowtile block instead of two 16-kstep ones): B=1 serve 1757 -> 1738, B=8 3851 -> 3766 (!). NSEG_QKV 10 -> 5:
B=1 -1, B=8 -30. (Earlier tunings were made on much older engines; the task-boundary cost changed with them.)

## s0023 track B: o_proj K split 16 -> 4
Expected HTTP B=1 ~1745, B=8 ~3320. Measured 1738.6 / 3323 (in process 1727). NSEG_O sweep (track B build, serve):
16: 1757 / 3851; 8: 1738 / 3769; 4: 1726 / 3765; 2: 1795 / 3874. NSEG_QKV 5 / 4 / 2 with NSEG_O 4 or 8: no gain.
Track A gains the same (B=1 -19, B=8 -86 in process). Own ref check median 0.0397 -> 0.0439 (3 x 160, noise level;
the summation order of the o_proj partials changed, identically in both instances: bitwise batch check holds).

## s0024 track B: ring tasks on whole stages (OALIGN o_proj segments 63/63/63/67; router 5 tasks of 9 ksteps)
Expected HTTP B=1 ~1734, B=8 ~3280. Measured 1739.2 / 3219 (B=8 better than predicted). Same-card serve A/B: OALIGN
B=8 -39 (B=1 0); RTASK 8 -> 5: -5 / -5; RTASK 4: B=1 +16; NSEG_GU 3: +30; NSEG_DN 3: +10; NSEG_QKV 20 (RED_SLOTS 60):
+6 / +13. Pattern: tasks that end in a partial ring stage cost a stage turn each; whole-stage tasks pay off.
CSPLIT (two-rowtile o_proj blocks hand segments 2, 3 of their second rowtile to a one-rowtile partner that computes
them first and publishes the partials with a release counter; the owner sums in the usual order): bit-identical to
CSPLIT 0 at B=1 and B=8, but +33 us at both. With NSEG_O 4 every warp holds at most one o_proj task, so a
two-rowtile block runs 8 warps instead of 4 and takes about as long: per-warp stage count, not per-SM bytes, sets the
phase time here. Off. (Same reasoning: the 3-rowtile qkv blocks are at 8 stages per warp vs 6; a split would gain at
most one stage per warp.)
FHALF (B>=3 down units as two K-half tasks 27 + 18 chunks, second half into ypart1, summed in the combine step; the
flat epilogue uses the same order ((s0+s1)+s2+bias)+(s3+s4)): bit-identical across instances, B=8 +163 us. Twice the
tasks (each with its bias loads and task-start costs) cost more than the per-warp balance gains. Off.
HSK_WIDE re-measured after the stage re-cuts: still +32 us at B=8. V_ASYNC re-measured: B=1 -10, B=8 +12; V_EARLY:
B=8 +42. Trying V_ASYNC in the flat instance only (V_ASYNC_FLAT).
V_ASYNC_FLAT (V by cp.async with K, flat instance only): B=1 -10, B=8 0. On. PF_STAGES=1: B=1 +54. Sketch tail
records by plain loads at task start (HSK_TAILREG, 10 whole stages per task): tokens identical, B=1 +10. Off.

## s0025 track B: B<=2 attention loads V by cp.async with K
Expected HTTP B=1 ~1729, B=8 ~3219. Measured 1736.5 / 3237 (B=8 repeats 3215-3476; in process generate 1717 vs
s0024's 1721: the bench's -10 did not fully carry).

## s0026 track A (TRACK=A: s0025's engine without the head sketch)
Expected HTTP B=1 ~1792, B=8 ~3230. Measured 1810.2 / 3228 (in process 1786). Best track A so far on both.
The head sketch is worth ~74 us at B=1 over HTTP (s0025 1736.5 vs s0026 1810.2).
Warning found at ~10.8 agent-h: the ATT_HYB switch (sliding layers without chunk split) no longer has a code path:
`if (split_att) {...}` has no else since the non-split attention was removed, so ATT_HYB=1 skipped attention in the
sliding layers and "measured" B=1 -66 / B=8 -485 us. It is now a compile error. (No submission used it.) Lesson:
check that an old switch still has its code path before trusting a large gain; run the reference check on it.
Staged wide norms re-tested separately: A only (NORM_TMA_A) +112, D only (NORM_TMA_D) +142 us at B=8: the staged form
is slower in itself, not because of the bulk-copy queue. Off.
ncu source-level stall sampling (B=1 generate, e0368): 47% of warp samples at block barriers (idle warps while
one thread polls the grid barrier), 14% fixed-latency waits, 13% global loads, compute negligible.
HSK_FASTRS (a candidate head rowtile rescored by warp 0 with all 20 stages in flight in the idle ring memory of all
warps, own mbarriers, consume_tasks<0>'s per-stage arithmetic): tokens identical, B=1 -3.5 us. On.
FTAIL (B>=3 down phase: the n mod 8 remainder units of a round as one-stage segments summed in shared memory in the
same order): bit-identical, B=8 -5 us. On.

## s0027 track B: HSK_FASTRS + FTAIL
Expected HTTP B=1 ~1733, B=8 ~3215. Measured 1730.2 / 3200 (in process 1706). Best at both batch sizes.
Probe: 4-stage ring with a smaller xs in the B<=2 instance (RS=4, XS_BYTES=26496, B=1 only): +5 us. Ring depth is
not what limits the per-warp stage chains in the kernel either.
ETAIL (B>=3 gate_up: the n mod 8 remainder units as one-stage segments, summed in shared memory in the same order;
compiled into the wide instance only -- as plain code the flat instance lost 18 us at B=1): bit-identical, B=8 -17 us.
ATT_BLK (attention units with at most 8 chunks -- sliding layers, full layers below position 512 -- handled by one
block per (sequence, kv head), one chunk per warp, partials in shared memory, merged by warp 0 after a block barrier
with the same arithmetic as the global merge): same bits (reference medians unchanged to the last digit, bitwise
batch check). Against the committed source (track A build): B=1 -14, B=8 -46 us; track B build B=8 ~-25, B=1 0.
ATT_BLKW=2 (B>=3: up to 2 chunks per warp, partials in xs): B=8 +15. One chunk per warp only.

## s0028 track B: ATT_BLK + ETAIL
Expected HTTP B=1 ~1727, B=8 ~3165. Measured 1731.1 / 3174 (B=8 repeats 3150-3413; in process B=1 1695). Best B=8.
A two-block-per-unit form for 9-16 chunks would bring back one global hand-off (atomic + merge from L2): estimated
gain < 1 us per full layer, not built.
ATT_CL2 (cooperative launch in clusters of 2 -- 76 co-resident, clusters of 4 only 36 -- and attention units of 9-16
chunks on a block pair, the partner's partials read through distributed shared memory by rank 0's warp 0, same
single-level merge): bitwise equal to ATT_CL2=0 at B=1 and B=8 up to position 1100 (tools/cmp_long.py), but
64+1024 serve B=1 1724 -> 1785, B=8 3696 -> 3727. Off.

## s0029 track A (s0028's engine with TRACK=A)
Expected HTTP B=1 ~1805, B=8 ~3175. Measured 1799.5 / 3174. Best track A at both batch sizes.
Head at B=1 after the sketch (mkprof marks 36/37): candidate filtering 1.4 us, rescoring on the candidate block up to
~8 us (the ~2 candidate tiles usually sit in one block and were rescored one after the other), reduction 1.5 us.
FASTRS now rescores up to 3 candidate tiles in parallel (one warp each, the 25 pooled stage slots shared out and
refilled as consumed): B=1 -5 us; HSK_PF 0 (no L2 prefetch of local candidates): -3.5 us. Sketch stream: the same
blocks are consistently 2-3 us slow or fast (corr 0.88): per-SM bandwidth; a tile (~7 us per warp) is too coarse
for dynamic balancing of a ~5 us spread.

## s0030 track B: parallel candidate rescoring, no L2 prefetch of candidates
Expected HTTP B=1 ~1722, B=8 ~3174. Measured 1719.0 / 3177 (in process 1683). Best B=1.
STEPARG (the per-step token pick as a warp-parallel argmax over the 152 per-block candidates instead of a serial scan
by one thread per sequence): same tokens, B=1 ~-1.5, B=8 -7 us. On.
Out-of-line calls in the wide instance (~11 agent-h): the exact head's consume as a noinline call costs B=8 +11 us
(HEAD_OL); the wide head sketch through a noinline wrapper (HSK_WIDE_OL): B=8 serve -16 us (head 175 -> ~135, call
and allocation ~+25), step() +33 (step never runs the sketch). On (HSK_WIDE=1, HSK_WIDE_OL=1).

## s0031 track B: wide head sketch out of line + warp-parallel token pick
Expected HTTP B=8 ~3150, B=1 ~1715. Measured 3144 (repeats 3120-3387) / 1715.2 (in process 1680). Best at both;
B=8 is 31 us from the 1.10x line (3113).
Wide sketch tail (mkprof B=8): filter + norm rebuild up to ~11 us and rescoring up to ~14 us on candidate blocks.
Pooled parallel rescoring in the wide instance (HSK_FASTRS_W, up to 3 or 8 tiles): slowest block -3 us but serve
+3..+5 us; off. Global lower bound read by one warp per column (was one warp looping over columns): B=8 -3.3, B=1
-1.7 us; on. Prep loops unrolled by 4 (PREP_UNROLL): no change.

## s0032 track B: parallel global-bound read
Expected HTTP B=8 ~3141, B=1 ~1713. Measured 3139 (repeats 3116-3382) / 1719.5 (noise vs s0031's 1715). Best B=8.
(The pod was lost during this evaluation; it re-ran on the new pod, epoch 3.)
Router K split in halves (RQ=2, 16 router blocks): B=1 +14, B=8 +4 us. RQ stays 4. Wide-instance local-memory
traffic (SASS): the rows[] pointer arrays of the norm calls and a few once-per-phase spills, not in hot loops.
Found at ~11.3 agent-h: the TRACK=A build had been broken since the HEAD_OL study switch (a call to an HSK-only
function inside `if constexpr`, which still needs its names declared). Fixed with a preprocessor guard; presubmit.sh
now compiles both tracks first. (Track A was last submitted as s0029, before the break.)

## s0033 track A: s0032's engine with TRACK=A (valid)
Expected HTTP B=1 ~1797, B=8 ~3168. Measured 1796 / 3164 (in process generate 1773). Best track A now s0033
(s0029: 1799.5 / 3174). The track-A build break (HEAD_OL guard) is fixed and presubmit compiles both tracks.
B=8 3164 in track A vs 3139 in track B: the wide sketch (HSK_WIDE, since s0030) is the difference.

## After s0033 (~11.5 agent-h): wide combine by the last arriver (FCOMB): negative
Fresh B=8 profile (pos 600, e0411): A 10.8, B 8.6, C 7.7, D 4.9, E 28.3, F 19.3 (last end 18.3), combine 2.1, head 128.
Bench (bench/consume.cu, NOCONSUME, from HBM): RS 3 and RS 4 stream the same 42 GB/s/SM -- ring depth is not the
B=8 expert-phase limit either.
FCOMB (B >= 3: per-rowtile counter of finished down units, atom.acq_rel; the block completing a rowtile does its 16
columns of the combine with the combine step's arithmetic; the combine step and its grid barrier removed): expected
B=8 -40 us (one barrier + 0.75 us of combine work per layer). Measured +76 us (serve 3638 -> 3714, same card,
3 rounds). Profile: F last end 18.3 -> 20.9 us, i.e. the atomic round trip + the combine's L2 loads at every block's
tail cost as much as the barrier and the step they replaced (combine total 2.1 -> 0), and the wide instance's stack
grew (192 -> 208 B). Off (FCOMB 0, code kept behind the switch).
Wide head tail with sub-marks (B=8 pos 600, e0420): after the sketch's barrier, bound read + filter max 1.7 us, the
norm rebuild on candidate blocks (norm_warp, inside the out-of-line head: cold instruction cache) max 8.4 us (median
0.4: only candidate blocks), rescoring max ~14 us.
HSK_XG (blocks 0..B-1 publish the prep's normed rows -- norm_warp's bits -- in fragment order before the barrier;
candidate blocks copy them into xs instead of re-norming): rebuild max 8.4 -> 1.2 us; tokens identical (hsk_check
B=3/8, same candidate counts). Same card 3 rounds: B=8 serve 3638 -> 3622.5 (-15.5), B=1 unchanged. On.
Single-tile rescoring (B=8, marks inside the loop): first stage lands ~3 us after issue, then ~0.26-0.41 us per
9-item stage (54-90 cycles per item; the mma chain alone needs ~10: bench/mmalat.cu 20 cycles latency, 10 per mma
with 2 chains): one candidate tile costs one warp ~8.5 us. An L2 prefetch of the final candidates one barrier ahead
changed nothing (not HBM latency). FASTRS_W (pooled slots, 3 or 8 tiles) re-measured after HSK_XG: no change.
HSK_RQ (candidate tiles through a global queue + one more grid barrier, tile j rescored by block j: no block holds
more than one tile): tokens identical, serve B=8 3621 vs 3621. Off (code kept). The rescoring tail is bound by the
per-tile consume chain of one warp; splitting a tile over warps would change the exact head's summation order.
Measurement note: two A/B jobs at once on cards 2 and 3 slow both by ~20 us at B=8 (shared power or clocks); A/B
inside one job stays valid, but compare absolute numbers only between solo runs.

## s0034 track B: HSK_XG (wide rescoring copies the prep's normed rows)
Expected HTTP B=8 ~3124, B=1 ~1717. Measured 3125 / 1722 (in process generate 1679). Best B=8; 12 us from 3113.

## After s0034 (~12.5-14 agent-h): an HTTP-like workload, and B=8 negatives
The HTTP client's prompts are random common words + "Continue the text above with a very long and detailed passage"
(a different one per stream). tools/bench.py --synth 1 builds prompts the same way: B=8 in process 3142-3144 us
(generate == serve, all streams 1024 tokens, 8 distinct texts) vs 3644 on the bench's real text, and HTTP s0034 3125.
So the scored B=8 workload routes far more alike than real text; prof_run.py --synth 1 profiles it. Synthetic B=8
per layer (pos 150..1050): A 10.9-11.1, B 6.2-12.2, C 7.7, D 4.8-5.1, E 29-34 + F 20-24 (together ~6.8 TB/s: at HBM
rate), combine 2.1, head ~118 (prep 9, sketch 87, tail ~19).
s0030-s0034 HTTP B=8: one of the three repeats is always ~240 us slower (3365-3410); median token gap (tpot p50)
3107 us vs window mean 3125.
Negatives (all same card, synthetic serve B=8 unless noted; bits checked where relevant):
- V_ASYNC in the wide instance (ATT_BLK partials moved to red so that V can take xs -- the old wide V_ASYNC=1 path
  overwrote the partials: its "2110 us" was a wrong-results artifact): bitwise equal, +7 us (real-text gen +16). Off.
- KVPF (phase A pulls each block's attention K/V chunks into L2): bitwise equal, 0 at B=1 and B=8. The ~2 us load
  stages of an attention item are not HBM latency.
- DQ (B >= 3 router blocks norm only their K quarter before the router, the other three quarters after arriving at
  the D -> E barrier): bitwise equal, +117 us (a second norm_warp call site in the wide instance). Off.
- Wide sketch prep (8 us at B=8, PREP_T timestamps: staging 1.0, sum 1.0, norm+max 2.1-2.3, digits 3.1-3.3, block
  norms 0.8): __restrict__ pointers (PREP_RESTRICT) and unroll 2/4: no change.
- Candidate rescoring: fragments of a stage loaded before its mma chain (HSK_RSPRE): no change (8.5 vs 8.6 us).
- XPF re-tested on the synthetic B=8 workload (E+F at HBM rate there, A-D leave HBM mostly idle, so prefetching
  predicted experts' gate_up into L2 during B-D looked worth it): all 152 blocks issuing, cap 96 / 48 / 32 / 0 MB per
  layer: serve B=8 3144 -> 3609 / 3421 / 3384 / 3318 (B=1 +167 at 96). The predictor and its phase-A tasks alone cost
  +174 us; the prefetch itself adds +66 even at 32 MB. Off.
- HSK_RSPRE back to 0 (it also changes the B <= 2 rescoring loop, unmeasured at B=1).
- Wide knobs re-checked on the synthetic B=8 bench: ETAIL 0 +4, PF_STAGES 3 +10 (one round each), FTAIL 0: -10.3 us
  serve (3 rounds: 3144 -> 3134), real-text gen -12; bitwise equal to FTAIL 1 up to position 1100 (B=1, B=8),
  generate == stepwise. FTAIL now 0 (it was -5 us at s0027; the allocation has moved since).

## s0035 track B: FTAIL 0
Expected HTTP B=8 ~3115, B=1 ~1720. Measured 3115 / 1722 (in process generate 1682). Best B=8; 2 us from 3113.
The synthetic bench predicted the HTTP change exactly (-10).
Sweep on top (one round each; bit-changing variants judged on bench.py --fixed: step() teacher-forced over a saved
synthetic continuation, same content for every build): HSK_WIDE_OL 0 +2; NSEG_QKV 5 B=1 -16 / B=8 +144; RTASK 3:
serve -62 but fixed-content -3 (B=1 +14): its router sums change the bits, so the free-running text and its routing
changed -- a content artifact, rejected; STEPARG 0 -7 (serve), -11 (fixed); ATT_BLKW 2 (B >= 3 units of up to 16
chunks on one block, 2 chunks per warp, bitwise equal): -10.7, -11.3, -11.0, -10.4 (serve), -9 (fixed), B=1 0;
ATT_BLKW 2 + STEPARG 0: -8 (they do not add). ATT_BLKW 2 next.

## s0036 track B: ATT_BLKW 2 (B >= 3 units of up to 16 chunks on one block)
Expected HTTP B=8 ~3104, B=1 ~1720. Measured 3105 / 1722 (in process 1682). Milestone h0_step: 1.10x the fastest
general engine at both batch sizes (SGLang 2197 / 3424: 1.276x at B=1, 1.103x at B=8).

## s0037 track A: s0036's engine with TRACK=A
Expected HTTP B=1 ~1797, B=8 ~3131. Measured 1799 / 3135 (in process 1764). Best track A at B=8 (s0033 3164); B=1
within noise of s0033 (1796). Track A is 1.22x SGLang at B=1 and 1.092x at B=8 (3113 needs -22: its exact head is
~171 us at B=8 vs the sketch's ~118; the head's last-tile imbalance is ~6 us of it).
Sweep after s0037 (synthetic serve B=8, one round unless noted): FASTRS_W 1 -1.7; ATT_BLKW 3 +40; PREP_UNROLL 2 -6;
NORM_PIPE1 (norm_warp: each weight word loaded one iteration ahead; journal ~6 h: warm 2.14 -> 1.75 us alone, 0 in
the kernel then): -23.5 us (3 rounds: 3122.5 -> 3099), real-text gen -25, B=1 0, bitwise equal to position 1100.
NORM_PIPE1 now 1. On top of it: PREP_UNROLL 2 +0.8, STEPARG 0 +14, FASTRS_W 1 +0.8.

## s0038 track B: NORM_PIPE1
Expected HTTP B=8 ~3082, B=1 ~1720. Measured 3083 / 1722 (in process 1682). Best B=8: 1.111x SGLang.

## Track A tuning on the synthetic bench (~15 agent-h; track A build, B=8 serve, one round unless noted)
NORM_PIPE1 1 in the track A build: +25 (track B: -23.5): its default now follows the track (HSK). On the track A
base: STEPARG 0 -14.7 (B=1 +3), PF_STAGES 3 -13.8 (B=1 +37), ATT_BLKW 1 -10, ETAIL 0 +23, FTAIL 1 +27; combinations
STEPARG 0 + PF 3 -8.6, STEPARG 0 + ATT_BLKW 1 -4, all three -1. The prefetch depth is now a template parameter
(PF_STAGES for B <= 2, PF_STAGES_W for B >= 3, passed by the kernel's call sites): PF_STAGES_W 3 alone: track A
-18.8, -19 (B=1 unaffected), track B +6.4; with STEPARG 0 -9.5, with ATT_BLKW 1 -9.6, PF_STAGES_W 1 +51; HEAD_OL
(now compiled in track A too) -9.4, with PF_STAGES_W 3 -9.5; on top of PF_STAGES_W 3: NORM_PIPE1 1 +21, ETAIL 0 +43.
Track A default: PF_STAGES_W 3 (track B: 2). Track B build checked unchanged (same bits, 3099 vs 3099 us).

## s0039 track A: PF_STAGES_W 3 in the track A build
Expected HTTP B=8 ~3116, B=1 ~1799. Measured 3117 / 1798 (in process 1763). Best track A at B=8 (1.098x SGLang);
4 us above the 1.10x line (3113).
Track A head balance (~15.5 agent-h). Per-block exact-head time at B=8 (synthetic, pos 600): median ~161 us, max
167-171, block means 158.7-167.3, step-to-step correlation 0.3-0.76 (tools/study/headcorr.py). HBAL (head rowtile
bounds re-balanced every step from the blocks' measured times, smoothed and clamped to +-15 %; whole rowtiles, so
the same bits; generate == stepwise, ref check unchanged): track A B=8 +56 us, B=1 +10 (and with HBAL compiled out
the added Misc/Params fields alone moved B=1 by +17). Reverted. Other track A switches on top of s0039: PF_LATE +10,
L2PF 2 +65 (B=1 +91). Track A stays at s0039 (3117, 4 us above the line).
B=1 switches on the synthetic bench (~16 agent-h, one round each, B=1 serve): V_ASYNC_FLAT 0 -6.2 (3 rounds; B=8
+0..4: wide code unchanged; bitwise equal), EPF 1 +47, HSK_TAILREG 1 +7, HSK_PF 2 +6.5, ATMA_FLAT 0 +20.
COMB_PIPE (combine_flat: each thread's next item loaded before the current item's stores; no spills reported): +20,
with V_ASYNC_FLAT 0 +38. Removed again.
bench/tma1w.cu: one block alone receives the 20 stages (92 KB) of a head rowtile in 1.35 us cold, first stage 0.76 us,
whether lane 0, 20 lanes or 4 warps issue the copies. In the kernel the rescoring warp sees its first stage after ~3 us
and then ~0.26-0.41 us per stage with the data present (54-90 cycles per item): the consume itself, unexplained.

## s0040 track B: V_ASYNC_FLAT 0
Expected HTTP B=1 ~1716, B=8 ~3085. Measured 1719 / 3083 (in process generate 1681, s0038 1682). B=1 -3 vs s0038
(best B=1 stays s0031 1715); B=8 unchanged.
NSEG_QKV_F (the B <= 2 instance's qkv split alone, 5 instead of 10): B=1 -14.2 us (3 rounds; step() -13.7), B=8
unchanged -- but the two instances then sum qkv in different orders, and greedy continuations of 8 real-text prompts
alone vs in a batch diverge after 1..956 tokens (mean matching prefix 0.20, 0 of 8 identical; current build: 8 of 8
identical over 1024 tokens, tools/study/alone_vs_batch.py). The http_tokens gate compares the server's batched tokens
with the engine's own step-by-step greedy tokens: not worth the risk. Reverted.

## Correction (~16.3 agent-h): KVPF was never measured before
The KVPF result above ("bitwise equal, 0 at B=1 and B=8") was a stale-library artifact: KVPF=1 did not compile
(ATT_BLK / ATT_BLKW were defined further down the file than phase A's use), and tools/abs.sh kept the previous
libmk_sb.so after a failed compile (so cmp_long also compared a build with itself). abs.sh now deletes both
libraries first; the two macros moved up. Measured properly (synthetic serve, 3 rounds; fixed-content step() 1 round):
track B: B=1 -14 (1714 -> 1700), B=8 -21 (3102 -> 3081); step() B=1 -22, B=8 -23; bitwise equal to position 700.
Track A: +17 at B=8 (DQ in track A: +58; FCOMB +106; BCOV +313). KVPF now follows the track (on in B, off in A).
So the K/V cache does fall out of L2 between steps and its first loads in phase B wait on HBM -- the ~2 us load
stages of an attention item are partly HBM latency after all.

## s0041 track B: KVPF
Expected HTTP B=1 ~1705, B=8 ~3062. Measured 1702.9 / 3066 (in process 1660). New best at both batch sizes:
1.290x / 1.117x SGLang.
SPF (phase A also pulls this block's qkv / o_proj bias slices, the router bias and the attention sinks into L2):
track B B=1 -5.7 us (4 rounds; fixed-content step() -3.5), B=8 0, bitwise equal; track A B=1 -1.7, B=8 +13. Follows
the track (on in B). Track A combinations with KVPF (PF_STAGES_W 2 / STEPARG 0 / ATT_BLKW 1 / HEAD_OL 1): +10..+26.

## s0042 track B: SPF
Expected HTTP B=1 ~1697, B=8 ~3066. Measured 1700.3 / 3064 (in process 1653). Best at both: 1.292x / 1.117x SGLang.

## s0043 track A: V_ASYNC_FLAT 0 (and same-bits code moves) in the track A build
Expected HTTP B=1 ~1788, B=8 ~3114. Measured 1787 / 3114 (in process 1751). Best track A at both; B=8 is 1.0995x
SGLang (the line is 3112.7).
Track A after s0043: KVPF as an out-of-line call (KVPF_OL) B=1 -8.3 / B=8 +24; SPF of the router bias + sinks only
B=1 -2 / B=8 +10; both B=1 -17.5 / B=8 +27. The track A wide instance loses with any of them. Track A stays at s0043.
KVPF_PLACE 1 (the K/V prefetch issued one phase earlier, with the cross-barrier prefetch of the layer's qkv weights):
track A B=1 -10 / B=8 +40, track B B=1 -1 / B=8 +20. Stays at phase A's start. L2PF in track B (whole task lists
into L2 at prefetch time): bit 0 (fp4) B=1 +102 / B=8 +547, bit 1 (bf16 <= 64 items) +81 / +66, both +173 / +660.
Weight-stream L2 prefetch remains a dead end; the K/V cache and the small parameters were the exceptions (latency
of a few scattered lines right after a barrier, not stream bandwidth).
Head-candidate rescoring in isolation (bench/rescore1.cu: one warp, one 92 KB tile, 20 bulk copies at once, the same
item<0> arithmetic): 3.4 us cold or L2-warm, consume-bound (~37 cycles per item); in the kernel ~8.5 us. Suspected
cold instruction cache (the head's code runs once per step; the step's 4.4 GB stream evicts its lines): device-side
function pointers are code-segment offsets (0x1a0, bench/codeaddr.cu), so code cannot be L2-prefetched; a smaller
rescoring loop body (HSK_RSROLL: items in a rolled loop over pairs, same order, tokens identical) B=1 +1.5, wide
(with FASTRS_W) +3.8. Not confirmed; off.
HSK_L2PFW (wide sketch: each warp's first 48 KB task pulled into L2 at the head start, while the 9 us prep leaves
HBM idle; ~58 MB in all): B=8 +22 us. Off. Track A on fixed content: RTASK 3 +38, RQ 2 +15.

## Closing state (~16.5 agent-h)
Track B s0042: HTTP 1700 / 3064 us (1.292x / 1.117x SGLang). Track A s0043: 1787 / 3114 (1.229x / 1.0995x).
The engine source at HEAD builds both (engine/TRACK selects; presubmit compiles both and runs the checks on the
selected one). Ideas left, none with a clear expected gain: a prep for the wide sketch that stages outside the
ring (so its stream can start during the prep), a faster head rescoring (3.4 us in isolation vs ~8.5 in the kernel,
cause unknown), predicted-expert prefetch into the ring (not L2) for E at B=1 (needs a cheaper predictor than XPF's).

## ~16.5-17 agent-h (after the restart of the clock)
Rescoring twice in the kernel (RSTWICE study, B=1): first pass 6.7-8.2 us, second pass over the same tile 5.1-5.4 us
(isolated: 3.4). HSK_WARM (the pooled rescoring as one out-of-line function, warmed by a 2-stage dummy call between
the post-sketch barrier's arrive and wait): rescoring max 5.4-5.6 us, but the sketch stream +0.8 us (allocation of
the out-of-line head): head total 106.5 -> 106.1, serve B=1 +0.8. Off. PFDLY (blocks without candidates wait 2 / 4 us
before the next step's cross-barrier prefetch, to keep HBM free for the candidates' tiles): +12 / +12 at B=1, +13 at
B=8 (the same for both delays: the code's presence). Off.
Re-sweep of same-bits switches after KVPF/SPF (track B, synthetic serve, one round): FTAIL 1 +19 (B=8), ETAIL 0
-5.6 (3 rounds -5.6..-5.8, real-text gen -11, B=1 0, bitwise equal), ATT_BLKW 1 +48, NORM_PIPE1 0 +23, STEPARG 0
+6 / +12, PF_STAGES_W 3 +16, FASTRS_W 1 +8, HSK_WIDE_OL 0 +3.5, V_ASYNC_FLAT 1 B=1 +31, PREP_UNROLL 2 0, HSK_FASTRS
0 B=1 +1, HEAD_OL 1 +8. ETAIL now 0. (The pod was replaced during this, epoch 4: absolute numbers moved by ~5 us.)

## s0044 track B: ETAIL 0
Expected HTTP B=8 ~3058, B=1 ~1700. Measured 3054 / 1706.5 (in process 1657, s0042 1653; the B <= 2 code is
unchanged, the pod was replaced in between). Best B=8: 1.121x SGLang. Best B=1 stays s0042 (1700).
Track A re-sweep on the s0043 base (B=8 synthetic serve): ETAIL 0 +40, FTAIL 1 +20, ATT_BLKW 1 +12, STEPARG 0 +31, PF_STAGES_W 2 +12, HEAD_OL 1 +12, NORM_PIPE1 1 +32. Track A stays at s0043.
Layout / attention after s0044 (track B, one round): TTN 12 +14 / TTN 24 +14 (B=8), XS_BYTES +256 0; ATT_BLKW_F 2
(B <= 2 attention units of 9-16 chunks on one block, partials in xs, which V_ASYNC_FLAT 0 leaves free; bitwise equal
to position 1100): B=1 +22. Off.
Compiler options on the track B build (variant only, one round): -Xptxas -O2 0 / -1 (B=1), --allow-expensive-optimizations +2.4, -dlcm=cg B=1 +15, --extra-device-vectorization 0.
Track A layout / compiler options (B=8): TTN 12 +37, TTN 24 +37, XS_BYTES +256 0, +512 -4 (noise), -Xptxas -O2 +5.6, expensive optimizations -0.7, extra vectorization 0. Track A stays at s0043.
HEAD vs s0044 source (track B): same bits, serve within noise. V_EARLY (V loads into registers right after K, both instances): track B B=1 +25, B=8 +152; track A +24 / +150.
Same-bits re-sweep after s0044 (track B, B=8 synthetic serve): PREP_UNROLL 2 -5, HSK_WIDE_OL 0 -14 (B=1 serve +12
outlier, gen 0), XS_BYTES +256 / +512 0, HSK_FASTRS 0 0, HSK_XG 0 -24 (3 rounds: 3073 -> 3049, real-text gen -27,
B=1 0). Combinations: XG 0 + WIDE_OL 0 +23, XG 0 + PREP_UNROLL 2 -24.7. HSK_XG now 0 (it was -15 at s0034: the
out-of-line wide head's allocation has moved since).

## s0045 track B: HSK_XG 0
Expected HTTP B=8 ~3030, B=1 ~1703. Measured 3026 / 1692.6 (in process 1646). Best at both: 1.298x / 1.132x SGLang.
(The B <= 2 code did not change since s0042 (1700): B=1 HTTP varies by several us between evaluations.)
Head-only switches after s0045 (B=8 synthetic serve): FASTRS_W 1 -10 (3 rounds), FASTRS_W 1 + WMAX 3 -9, HSK_RQ 1
-9.5 (2 rounds), both -9.7, PREP_RESTRICT 0, HSK_RSPRE 0, HSK_L2PFW +7, PREP_UNROLL 2 / 4 +6 / +4; B=1 unchanged.
HSK_FASTRS_W now 1.

## s0046 track B: HSK_FASTRS_W 1
Expected HTTP B=8 ~3016, B=1 ~1695. Measured 3017 / 1692.5 (in process 1647). Best at both: 1.298x / 1.135x SGLang.
Head switches after s0046 (one round): FASTRS_WMAX 3 +4, 5 0, HSK_RQ 1 0, HSK_XG 1 +47, PREP_UNROLL 2 +10, HSK_RSPRE 1 0, HSK_RSROLL 1 +24, HSK_WIDE_OL 0 +45 (B=8); B=1 0.
FASTRS_FMAX 1 / 2 / 8 (B <= 2 parallel candidate tiles, default 3): within +-1.5 us at B=1. Stays 3.
EBPF (B <= 2: E's start pulls the block's gate_up / down bias rows into L2): B=1 serve 0, step +7; bitwise equal. Off.
QX (qkv: every block owns 2 rowtiles, the 16 extra V rowtiles of kv heads 4..7 as 160 segment tasks on blocks
0..39 with partials in global; the attention warp of a kv-head >= 4 unit's last chunk sums V(pos) itself with the
epilogue's arithmetic): bitwise equal to position 1100, generate == stepwise. Profile (pos 600): phase A -0.92 us
(B=1) / -0.94 (B=8) per layer, B unchanged -- but in the normal build B=1 -2..-4, B=8 +30 (wide stack 160 -> 192 B,
16 B spills); per instance: B <= 2 only -2 (step -3.5), B >= 3 only +18. Also: the QX rewrite of the qkv task list
cost +16 us at B=8 even with QX off (code not compiled out); reverted to s0046's source.
QXW (QX in the B >= 3 instance only, strictly behind #if so that the default build is s0046's code: checked equal,
3039 vs 3039): alone +34, with ETAIL 1 +22, NORM_PIPE1 0 +46, ATT_BLKW 1 +46, FTAIL 1 +39, STEPARG 0 +21,
PF_STAGES_W 3 +34, KVPF 0 +48 (B=8). Abandoned: the phase-A gain the profile shows (-0.9 us/layer) never survives
the wide instance's register allocation. Source back to the committed state.
ptxas --register-usage-level 0 / 3 / 7 / 10 (variant only): B=1 +19 / +18 / +13 / +16, B=8 +61 / +62 / +49 / +56. Default kept.
HSK_XG 2 (candidate blocks fetch the published rows by bulk copies, almost no registers): serve B=8 0, gen +4..+11; tokens identical. Off.
Pair sweep after s0046 (all 28 pairs of FTAIL 1, NORM_PIPE1 0, ATT_BLKW 1, STEPARG 0, PF_STAGES_W 3, HSK_WIDE_OL 0,
SPF 0, ETAIL 1; B=8 synthetic serve, one round): every pair worse, +12.6 (PF_STAGES_W 3 + SPF 0) to +90 (ATT_BLKW 1 +
HSK_WIDE_OL 0). The s0046 configuration is a local optimum of the same-bits switches.
QXO (QX with V(pos) summed by an out-of-line call qx_vpos; 1: both instances, 2: B >= 3, 3: B <= 2; the B >= 3 qkv
task list chosen by if constexpr so that instance keeps the original code): bitwise equal to position 1100, tokens
identical. QXO 1: B=1 -5.8, B=8 +27; QXO 2: B=8 +24; QXO 3: B=1 -4.5 (5 rounds: -3 .. -6.6), B=8 +3.3. QXO = 3 in
track B (B=1 is the headline), 0 in track A.

## s0047 track B: QXO 3
Expected HTTP B=1 ~1688, B=8 ~3020. Measured 1690.7 / 3021 (in process 1643). Best B=1 (1.299x SGLang); best B=8
stays s0046 (3017).
B <= 2 switches after s0047 (B=1 synthetic serve): EBPF 1 +8, HSK_FASTRS 0 +6, ATMA_FLAT 0 +24, FTMA_FLAT 0 +22, EPF 1 +31, HSK_TAILREG 1 +5.5, HSK_PF 2 +18.5, ATT_BLKW_F 2 +25.5.
Track A with QXO 1 / 2 / 3: B=8 +56 / +40 / +6, B=1 +4 / 0 / +4. Track A stays at s0043.
OSPLIT 1 re-test (fixed content): B=1 +53, B=8 +104.
Pair sweep of B=1-relevant switches after s0047 (15 pairs of EBPF 1, HSK_FASTRS 0, HSK_TAILREG 1, FASTRS_FMAX 2, STEPARG 0, SPF 0; B=1 serve): all worse or noise (best: FMAX 2 + FASTRS 0 serve -2.2 / gen +6.7).
Old structural switches re-tested after s0047 (track B, B=8 / B=1 serve): FCOMB 1 +72 / 0, CSPLIT 1 +144 / +54, BCOV 1 +325 / +17, NORM_TMA 1 +262 / 0, KVPF_PLACE 1 +29 / +3, KVPF_OL 1 +2 / +3.
XS_BYTES +128 / +384 / +640 / +768 / +1024 (shared-memory layout offsets): within +-1 us in both tracks (B=1, B=8).
B=8 serve by prompt seed (synthetic): 1000 3043, 2000 3135, 3000 3054, 4000 3099, 5000 3209, 6000 3137 (findings).
Robustness on the slowest prompt set (seed 5000, B=8 serve 3209): ETAIL 1 +20, FTAIL 1 +61, HSK_XG 1 +39,
FASTRS_W 0 +12 -- the choices made on seed 1000 hold.

## Closing state (~21.3 agent-h)
Track B: best B=1 s0047 1690.7 us (1.299x SGLang), best B=8 s0046 3017 us (1.135x); s0047 is the current engine
(1690.7 / 3021). Track A: s0043 1787 / 3114 (1.229x / 1.0995x). HEAD builds both (engine/TRACK = B by default;
presubmit compiles both and passes the checks in each). Since s0042 every gain came from re-sweeping same-bits
switches after allocation-moving changes, plus one structural change that survived only in the B <= 2 instance
(QXO 3); pair sweeps, layout offsets, compiler options and the old structural switches show the current
configuration to be a local optimum.
SPF_ROPE (SPF also prefetches the position's RoPE cos/sin row): B=1 +8, B=8 +14. Off.
QXO 1 (both instances) + ETAIL 1 / FTAIL 1 / NORM_PIPE1 0 / STEPARG 0 / FASTRS_W 0: B=8 +23 / +31 / +54 / +46 / +21 / +17 (QXO 1 alone +23).
sketch_verify (engine of s0047): 0 mismatches in 22528 generated tokens (B=1, 2, 8; findings).

## ~21.5-22 agent-h: track A head tail (HTAIL)
HTAIL (track A exact head: each block's n mod 8 last rowtiles as 4 whole-stage K segments over the warps, summed in
order by one warp, then the usual logits / argmax step; the same in both instances, so alone == batch; tokens:
generate == stepwise at B=1/2/8): alone B=1 -7, B=8 +18 (2 rounds). With PF_STAGES_W 2 +22, STEPARG 0 +36,
ATT_BLKW 1 +23, ETAIL 0 +61, NORM_PIPE1 1 -2.4 (3 rounds: B=8 serve -2.7, gen +2.4; B=1 -6.5). Track A defaults now
HTAIL 1 + NORM_PIPE1 1.

## s0048 track A: HTAIL + NORM_PIPE1
Expected HTTP B=8 ~3111, B=1 ~1781. Measured 3101 / 1774.6 (in process 1740). Best track A at both: 1.238x / 1.104x
SGLang -- track A now also clears the 1.10x line at B=8 (3112.7).
Track A re-sweep on the s0048 base (one round, B=8 / B=1): STEPARG 0 +11 / +9, ATT_BLKW 1 -8 / 0, FTAIL 1 +28, ETAIL 0
+28, PF_STAGES_W 2 +7, KVPF 1 -27 / +13, SPF 1 -5 / -3, QXO 3 -2 / +18. KVPF split per instance (KVPF: B >= 3,
KVPF_F: B <= 2, by if constexpr; track B's build checked unchanged, bitwise equal): track A with KVPF in the B >= 3
instance only: B=8 -32.7 (2 rounds), B=1 0; + ATT_BLKW 1 -5, + SPF 1 -21.5 (worse than alone). Defaults: KVPF 1,
KVPF_F = HSK.

## s0049 track A: KVPF in the B >= 3 instance
Expected HTTP B=8 ~3068, B=1 ~1775. Measured 3074 / 1774.3 (in process 1740). Best track A at both: 1.238x / 1.114x.
Track A re-sweep on the s0049 base (B=8 / B=1): ATT_BLKW 1 +29, SPF 1 +6, STEPARG 0 +30 / +22, PF_STAGES_W 2 +2.6, ETAIL 0 +35, FTAIL 1 +33, QXO 3 +5.6 / +21, KVPF_F 1 +6.5 / +12.6. Track A stays at s0049. (Pod replaced during this sweep, epoch 6.)

## Closing state (~22.3 agent-h)
Track B: best B=1 s0047 1690.7 (1.299x), best B=8 s0046 3017 (1.135x). Track A: s0049 1774.3 / 3074 (1.238x / 1.114x).
HEAD builds both (engine/TRACK = B); presubmit passes in both tracks on the epoch-6 pod. Lesson of the last hours:
in track A too, switches that lost earlier (NORM_PIPE1, KVPF) won after another change moved the allocation, so
every larger change should be followed by a re-sweep of the switches that touch the same instance.
Idea not built (time): the head sketch's tail rowtiles split over warps like HTAIL -- the int32 sketch sums are order-free, but a split needs integer partials and the tile's norm/constant records parked in shared memory, which the head fully uses (ring, digits, candidate list).
Final track B single-switch check on the epoch-6 pod (B=8 / B=1): ETAIL 1 +22, NORM_PIPE1 0 +31, KVPF_F 0 0 / +14.5, QXO 0 -1.4 / -1.4 (its -4.5 at B=1 was measured on the epoch-5 pod; s0047's HTTP moved B=1 by -1.8: pod-dependent, within noise), STEPARG 0 +10 / +15, FASTRS_W 0 +10. Configuration kept.
SPF per instance (study, reverted): track B SPF off in B >= 3 only: B=8 +12; off in B <= 2 only: B=1 +13. Track A SPF on in B <= 2 only: 0; in B >= 3 only: B=8 +4.6.
HTS (HTAIL segments per tail rowtile) 2 / 5 / 10 vs 4, track A: B=8 -7 (3 rounds: -7.3, gen -9) / +7 / +3.5, B=1 0 / +13 / +13. HTS now 2.

## s0050 track A: HTS 2
Expected HTTP B=8 ~3067, B=1 ~1774. Measured 3061 / 1775.8 (in process 1738). Best track A B=8 (1.119x SGLang); best
track A B=1 stays s0049 (1774.3).
HTR (also split one more full round of head rowtiles; tail reduction as a loop over tiles per warp): the loop rewrite alone (HTR 0) cost track A +12 / +13 us (B=8 / B=1, allocation); HTR 1 on top +0..+3. Reverted.
Track A re-sweep on the s0050 base (B=8 serve, one round): ATT_BLKW 1 +28, SPF 1 +22, STEPARG 0 +15, PF_STAGES_W 2 +11,
NORM_PIPE1 0 +23, ETAIL 0 +29. Track A stays at s0050.

## Final state (~23.4 agent-h)
Track B: best B=1 s0047 1690.7 us (1.299x SGLang), best B=8 s0046 3017 us (1.135x).
Track A: best B=1 s0049 1774.3 us (1.238x), best B=8 s0050 3061 us (1.119x).
HEAD = s0047's track B engine and s0050's track A engine (engine/TRACK selects; default B).
sketch_verify on the final track B source (epoch-6 pod): B=8 seed 3000, B=1 seed 5000, B=2 seed 7000: 0 mismatches in 11264 tokens.
HEAD vs submitted sources: TRACK=A == s0050 (identical files); TRACK=B vs s0047: 69 lines of track-A-only code differ, logits bitwise equal to 1100 (B=1, B=8), serve within 1 us.
Track A (s0050 engine) B=8 serve by prompt seed 1000..6000: 3080, 3174, 3086, 3139, 3248, 3176 (findings).
QXO 0 vs 3 on the epoch-6 pod (3 rounds): QXO 0 -1.3 at B=1 and B=8. QXO's B=1 gain (-4.5 on the epoch-5 pod) is pod-dependent; HTTP s0046 (no QXO) 1692.5 vs s0047 (QXO 3) 1690.7 -- within noise either way. Kept as submitted.

## s0051 track B: the final engine/ as committed
Expected HTTP B=1 ~1691, B=8 ~3020 (s0047's kernel). Measured 1686.6 / 3029 (in process 1639). Best B=1 headline;
the -4 us vs s0047 is evaluation-to-evaluation variation (same kernel, bitwise equal).

## Extension (36 agent-h total; from ~23.7 h)
ATT_OL (phase B as a __noinline__ function attn_phase<WIDE>, called once per layer; attention switches moved to the
config section): bitwise equal; ptxas: track A wide 239 registers, no spills; track B wide 208 B stack, 40/96 B spills.
Serve: track B B=1 +40, B=8 +107; track A +24 / +63. An out-of-line call inside the layer loop costs far more
than any allocation it frees (every live register of the kernel is saved around it, 36 times per step): the
"phases as out-of-line functions" plan from the closing notes is not viable as such. Off (code kept behind ATT_OL).
Expert reuse between consecutive steps at the same layer (EXPDUMP study build, tools/study/exp_reuse.py, synthetic
prompts, 290 steps): B=1 0.28 of a step's 4 experts were chosen at the step before (per layer 0.04-0.49); B=8 0.56
of the union (19.1 experts). Speculating E's experts from the previous step is useless at B=1; at B=8 it would be an
L2 prefetch of expert weights, which lost before (XPF). Not pursued.
STAIL (wide sketch's tail rowtiles in 2 K halves, int32 partials in the free end of xs; tokens and candidate counts identical): B=8 +14.5; with FASTRS_W 0 +4, HSK_XG 1 +46, PREP_UNROLL 2 +8, HSK_RQ 1 +15. Off.
Per-phase cross-barrier prefetch depth (PF_DN_F/W, PF_Q_F/W, PF_O_F/W; defaults = the instance's depth, same code):
depth 3 for down B <= 2 +9 (B=1), down B >= 3 +19 (B=8), qkv B <= 2 +9, qkv B >= 3 +6, o_proj B <= 2 +9, o_proj
B >= 3 +17; down depth 1 at B <= 2 +14. Depth 2 everywhere stays.
Multi-seed check of the B=8 choices (track B, seeds 2000 / 6000, one round each): reverting ETAIL +21 / +21, FTAIL +52 / +52, ATT_BLKW +22 / +20, NORM_PIPE1 +31 / +30, FASTRS_W +11 / +11 -- the same signs and sizes as on seed 1000: the tuning did not overfit one prompt set.
HSK_EARLYRB (wide head: rebuild the normed input between the post-sketch barrier's arrive and wait; tokens identical): all blocks +18 (B=8); only blocks with more than 12 / 40 blocks still to arrive +12 / +15. Reverted: the in-kernel cost of the rebuild code outweighs the tail it hides.
Instruction cache or allocation? Code size: 18.2K (B >= 3) / 16.3K (B <= 2) SASS instructions per kernel instance
(~290 / 260 KB). ncu warp-stall reasons on one generate() launch (64 steps, cycles per issued instruction):
B=1: barrier 5.35, wait 1.74, long_scoreboard 1.53, selected 1.00, short_scoreboard 0.77, branch_resolving 0.50,
no_instruction 0.44 (3.7 %), math_pipe_throttle 0.23, not_selected 0.21 (total ~12). B=8: wait 1.52, barrier 1.97,
selected 1, long_scoreboard 0.59, short_scoreboard 0.44, branch 0.41, math 0.40, not_selected 0.34, no_instruction
0.32. Instruction fetch is a small share: the "any code moves performance" effect is not mainly the i-cache.
Grid barrier forms (bench/barrier.cu, all 152 blocks arriving together): fence + atomicAdd + acquire poll (the
engine's) 1.345 us; red.release + acquire poll 1.206; + __nanosleep(10..40) between polls 1.155; per-block release
flags 3.5-56 (much worse). In the engine (GB_MODE, bitwise equal to position 1100): red.release alone B=1 +2 / B=8 +5;
red.release + nanosleep: 20 ns -7.1 / -5, 40 ns -7.8 / -5.5 (3 rounds: -7.4 / -1 avg), 80 ns -7.4 / -3, 160 ns
-2.9 / +2.4. Track A with 20 / 40 ns: B=1 -10.8 / -11..-14, B=8 +4.8 / +4.2. The win is the back-off: 152 threads
polling one L2 line slow the arrival and each other. Default now GB_MODE 2, GB_NS 40 (both tracks).

## s0052 track B: GB_MODE 2 (red.release arrival + 40 ns back-off poll)
Expected HTTP B=1 ~1680, B=8 ~3025. Measured 1689.8 / 3019 (in process 1647; s0051 1639 for the same kernel minus
this change). The effect depends on the card (same-card A/B, GB 2 vs GB 0): card 2 B=1 -7.4 / B=8 -1..-5.5, card 3
-6 / -9, card 1 +1 / -14; the evaluation card shows B=1 +3 (inside its +-4 us noise) and B=8 3019 vs 3029.
Per-card baselines also differ by ~10 us (card 1 B=1 1697 vs card 2 1686). Kept (B=8 gains everywhere).

## s0053 track A: GB_MODE 2
Expected HTTP B=1 ~1765, B=8 ~3065. Measured 1770.9 / 3075 (in process 1730, s0050 1738). Best track A B=1
(s0049 1774.3); B=8 +14 vs s0050 (3061), which stays track A's best B=8.
WS_NS (back-off in the down phase's wait for the gate_up slot counters) 20 / 40 ns: within +-1 us. ARRV (layer-end
barrier: arrive, issue the next layer's prefetch, then wait; bitwise equal): B=1 +3.3, B=8 -2.2. Both off.
Re-sweep after GB_MODE 2 (track B, B=1 / B=8 serve, one round): ETAIL 1 0/+3, FTAIL 1 0/+28, ATT_BLKW 1 0/+24, NORM_PIPE1 0 0/+17, STEPARG 0 +12/-1, PF_STAGES_W 3 0/+5, FASTRS_W 0 0/+5, HSK_XG 1 0/+48, KVPF 0 0/+40, SPF 0 +18/+12, QXO 0 +4/-3, HSK_WIDE_OL 0 0/+54. Configuration kept.
DCNT (the D -> E grid barrier replaced by the 32 router blocks' completion counter, back-off poll; bitwise equal): B=1 +7, B=8 -0.7. Off.
-maxrregcount 232 / 240 / 248 (variant only, track B): B=1 0, B=8 +13.5 each.

### GB_NS 30 / 60 vs 40 (card 2, track B, synth, same-card A/B, 2 rounds each)
Predicted: within +-2 us (40 already sits between 20 and 80, which were neutral to slightly worse).
Measured: ns30 B=1 1679.3/1678.9 vs 1679.7/1679.4, B=8 3035.8/3035.6 vs 3041.7/3035.4; ns60 B=1 1679.2/1679.1 vs
1679.3/1679.0, B=8 3035.1/3035.2 vs 3035.7/3035.4. All within noise: the back-off length is flat from 30 to 60 ns.
Keep 40.

## ~26-27 agent-h: K-split candidate rescoring (HSK_RSK)
Candidate head tiles were rescored by one warp each (20 stages, consume-bound: ~8.5 us per tile). A tile split over
warps changes the exact head's sum, so the exact head itself now sums each rowtile as G K segments of 20/G stages
(each from zero, added in order: consume_tasks<0, SEGL>); rescoring units (tile, segment) go one per warp, the
partials meet in a free pooled-ring slot and one warp per tile adds them in the same order (tokens = the exact head's
argmax, hsk_check equal at B=1/2/3/8, 300 tokens).
- Both instances, G=4: profiling build B=1 head 105-107 -> 102 us (rescoring max 15.7 -> 4.5), but the normal build
  B=1 serve +8..+19 (2 runs), B=8 -14..-19. G=2: B=1 +8..+19, B=8 -7.
- Wide only (flat rescoring one warp per tile, flat exact head segmented as well): B=1 +3.6, B=8 -13.
- Wide only, flat instance untouched (HSK_RSK_F 0: B<=2 exact head and rescoring as before): 3 rounds card 2,
  B=1 1679.4 vs 1679.4, B=8 serve 3030.3 vs 3042.8 (-12.6), gen -13; G=2 B=8 -6.5. Default now HSK_RSK 4 (track B),
  HSK_RSK_F 0.
  Cost: the head logits of a sequence alone (flat) and batched (wide) are no longer bitwise equal (max |d| 5.5e-5
  in presubmit's batch check); greedy alone vs batch still identical for 8 x 1024 tokens (alone_vs_batch.py).

## s0054 track B: HSK_RSK 4 in the wide instance
Predicted HTTP B=8 3019 -> ~3007, B=1 ~1688 +-4. Measured 1689.9 / 3007 (best B=8 in track B; was s0046 3017).
All gates passed; batch gate no longer bitwise (max |dlogit| 5.5e-5, head only), http_tokens 8/8, generate 8/8.

## Re-sweep after s0054 (track B, synthetic serve, one round; cards 2+3 at once)
B=8 serve vs base: ETAIL 1 +20, FTAIL 1 +35, ATT_BLKW 1 +34, NORM_PIPE1 0 +35, STEPARG 0 +7 (B=1 +12), PF_STAGES_W 3
-4.7 (3 rounds: +2.6, dropped), HSK_XG 1 +59, HSK_XG 2 -7.6 (3 rounds: -8.3 serve, -10 gen, B=1 0), HSK_WIDE_OL 0 +73,
KVPF 0 +42, SPF 0 +10 (B=1 +5). HSK_XG now 2 (the published normed rows by one bulk copy per row; same bits).

## s0055 track B: HSK_XG 2
Predicted HTTP B=8 3007 -> ~2999, B=1 ~1689. Measured 1689.9 / 3000 (best B=8 in track B; 1.141x SGLang).
Flat K-split (HSK_RSK_F 1) on the s0055 base, one round each, B=1 serve vs base: alone +7.8 (card 2) / FASTRS_FMAX 2
(no code change in that path) +9.3 (card 3); with QXO 0 +1.1, HSK_PF 2 +1.1, KVPF_F 0 +46, ATT_BLKW_F 2 +24,
V_ASYNC_FLAT 1 +34, HSK_L2PF 8 +8.4. The flat instance stays as it is (its head saves ~3.5 us in the profile, the
allocation costs ~8).

## Wide sketch prep fused (HSK_PREP2)
The B >= 3 prep (one warp per sequence, every block redundantly) ran 5 passes over the row: staging 1.0, sum 1.0,
norm+max 2.1, digits 3.1, block norms 0.8 us. HSK_PREP2 takes the digit scale from an upper bound of the normed
maximum (mx ri (1 + 2^-6), mx = max |w x| computed exactly with the sum of squares), so the normed values and the
digits come out of one pass (normed bits unchanged; the scale is at worst one power of two coarser: sketch bounds
stay certified; candidate rowtiles 4321 -> 4323 in 300 B=8 heads, tokens identical B=3/8).
- First form with the row in registers and fully unrolled loops: prep 8.6 -> 19.6 us, B=8 serve +40 (cold
  instruction cache: this code runs once per step; the rolled loops of the old prep matter).
- Rolled loops, smem staging kept: prep 8.6 -> 8.4 us in the profile, serve B=8 3014.6 -> 3008.2 (3 rounds, -6.3),
  gen -4.3, B=1 0. More than the prep's own change: layout / allocation of the wide head.

## s0056 track B: HSK_PREP2
Predicted HTTP B=8 3000 -> ~2994, B=1 ~1690. Measured 1689.8 / 2991 (best B=8 in track B, 1.145x SGLang).

## ~27.5-28 agent-h: track A dynamic head tail (HDYN)
HDYN (track A exact head): each block streams 76 rowtiles of its own (9-10 per warp); the other 1016 are claimed one
at a time from a global counter (ccnt[29/30], alternating per step, the next one reset at each step start) by warps
whose list ran out (consume_tasks' Claim functor appends to the warp's task table). Replaces HTAIL (its K-split tail
sums: logits differ from HTAIL's by <= 2.1e-5; alone == batch stays bitwise).
- Same card vs s0053's engine: S=72 B=1 +3, B=8 -16.8 (3 rounds); S=64 +2 / -15.8; S=76 -0.4 / -13.5 and +2 / -14;
  S=80 +3..+8 / -14. Profile (pos 600): head B=1 168.7 -> 164.8 (slowest block 165.7 -> 162.2, median 156 -> 158),
  B=8 170.3 -> 169.3: the head's blocks together already stream at ~7.3 TB/s, so balancing gains only ~4 us there;
  the rest of the B=8 effect is allocation.
- Re-sweep on HDYN S=76 (one round): KVPF_F 1 B=1 -14.6 / B=8 -20 (was +13 at B=1 in track A before), QXO 3 -8.5 /
  -19.6, SPF 1 -4 / +4, ETAIL 0 +2 / +2, FTAIL 1 / ATT_BLKW 1 / NORM_PIPE1 0 / STEPARG 0 / PF_STAGES_W 2 / KVPF 0 worse.
  3 rounds: HDYN+KVPF_F B=1 -11.2, B=8 -19.9; HDYN+KVPF_F+QXO 3 B=1 -16.6, B=8 -22.
Track A defaults now HDYN (S=76), KVPF_F 1, QXO 3 (track B's build unchanged: it had those values already).
Presubmit (TRACK=A): batch vs single bitwise 480/480, alone vs batch greedy identical 8 x 1024.

## s0057 track A: HDYN + KVPF_F 1 + QXO 3
Predicted HTTP B=1 1771 -> ~1755, B=8 3075 -> ~3053. Measured 1759.7 (best track A B=1, 1.249x SGLang) / 3067
(-8 vs s0053, the same-card -22 only partly; s0050's 3061 stays the track A B=8 best).

## ~28-28.5 agent-h: track B dynamic sketch tail (SDYN)
The track A result suggested the same for the head sketch (B=1: blocks wait a median 4.8 us at the post-sketch
barrier). SDYN: each block streams SDYN_S sketch rowtiles of its own; the rest are claimed from a global counter
(ccnt[29/30]) by warps whose list ran out; a block records its claimed tiles in hsi[50..63] (at most 14) and maps
candidate rows to local indices SDYN_S + slot for its 96-bit candidate mask. Tokens are the exact argmax whatever
block sketches a tile (bounds per tile); hsk_check B=1/2/3/8 400 tokens equal, sketch_verify B=8 seeds 1, 2: 0 of
8192 mismatched.
- Same card (2 + 3 at once): S=76 B=1 -5.0, B=8 -6.7; S=72 -12.6 / -6.4. Card 2 alone (2 rounds): S=72 -6.0 / -7.3,
  S=76 -5.5 / -6.4, S=68 -4.2 / "-130": its pool (2232) exceeded the 152 x 14 record slots, so tiles went
  unsketched -- a compile-time check now forbids pools larger than the slots. Default SDYN_S 72 (pool 1624).

## s0058 track B: SDYN (failed on card 0) and what it really measured
Predicted HTTP B=1 1690 -> ~1684, B=8 2991 -> ~2984. Measured 1692.1 / 3010 (in process on card 0: B=1 gen +1, B=8
w1-gen +14). Cards 1 and 3 (SDYN 0 vs 1 on the s0058 source): B=1 -17.7 / -12.9, B=8 -5.6 / -7.1.
Cause found: HDYN (s0057) had restructured consume_tasks for every task list (top-up before the empty-list return),
which also changed track B's kernel; s0058 carried it, and the cards 1-3 A/Bs compared SDYN on top of it. With
the old form restored for lists without claims (CT_ORIG, = HSK: track B's build SASS-identical to s0056's again,
track A's to s0057's), cards 2 / 3 (2 rounds): SDYN 1 B=1 +7.3 / +9.9, B=8 +38 / +46; the restructure alone
(CT_ORIG 0) +8 / +16, +25..+53. SDYN only won back allocation the restructure had cost. SDYN off (code kept).
Lesson: after touching code both tracks share, compare both tracks' SASS with their last submissions.
Track A with CT_ORIG 1 (HDYN kept, 3 rounds card 2): B=1 -1, B=8 +5.6: track A keeps the s0057 form (CT_ORIG = HSK).

## Re-sweep on s0056's kernel (track B, one round each, cards 2+3)
B=8 serve vs base: ETAIL 1 +23, FTAIL 1 +44, ATT_BLKW 1 +37, NORM_PIPE1 0 +29, PF_STAGES_W 3 +17, HSK_XG 0 +15,
HSK_WIDE_OL 0 +40, KVPF 0 +44, SPF 0 +36 (B=1 +6), FASTRS_WMAX 4 0 (unused since HSK_RSK). Nothing to take.

## Striped grid-barrier arrival (bench/barrier.cu modes 12-21)
Block b adds to counter b % S (own 128 B line), the waiters check all S: warp-parallel polls S = 2..32: 1.99-2.10 us
per barrier; thread 0 with acquire loads S = 2 / 4: 2.1-3.0 / 2.6-3.8; relaxed loads + one fence S = 2 / 4 / 8:
2.16 / 2.18 / 2.25 -- all worse than the single counter (1.155 us with the 40 ns back-off). Same-address reds are
not the bottleneck; polling more lines is. Not taken to the engine.

## NSEG re-check on s0056's kernel (fixed content, 2 rounds, B=1 / B=8 step())
NSEG_QKV 5 +9.3 / +2.3, 4 +41 / +38; NSEG_GU 10 +213 / +3, 2 +58 / +23; NSEG_DN 10 +83 / +3, 2 +81 / +5. The
current values (10 / 5 / 5) stay.

## Flat (B <= 2) switches on s0056's kernel (track B, one round)
B=1 serve vs base: ATT_BLKW_F 2 +19 (B=8 -7), V_ASYNC_FLAT 1 +15, QXO 0 +3.7, KVPF_F 0 +17.5, STEPARG 0 +15.6
(B=8 +26), HSK_PF 2 +23, FASTRS_FMAX 2 +0.2, PF_STAGES 3 +18, GB_MODE 0 +5 (B=8 +24). Nothing to take.
Track A HDYN 0 (HTAIL back) on the s0057 defaults (3 rounds): B=1 +1.5..+3, B=8 +18.5: HDYN stays.

## s0056's engine vs s0047's on three prompt sets (card 2, in process, one round each)
Serve B=8 (synthetic seeds 1000 / 2000 / 5000): 3040 -> 3008, 3132 -> 3101, 3205 -> 3176 (-30..-32); real-text gen
B=8 3527 -> 3499; B=1 serve 1686.7 -> 1679.2 (the barrier back-off on this card). The extension's B=8 gains are not
specific to the tuning prompt set.

## ~30 agent-h: the B <= 2 norms (NORM2_STG, NORM_G8; track B)
- NORM_NOSUM study (no sum of squares, ri = 1, wrong results): fixed-content B=1 1750 -> 1704 (-46) in the normal
  build, but +4.8 in the profiling build (norm1 span unchanged): an allocation artifact, not an upper bound.
- NORM2_STG (phase D's norm input rows staged into shared memory by all threads before norm_to_smem, which then reads
  smem instead of L2 twice; bitwise equal to position 1100): profile D 3.80 -> 3.57 us/layer, but A/C/E/F +0.07..0.19
  each; serve B=1 +9 (3 rounds), B=8 +3.4. Off (code not kept).
- NORM_G8 (sums of squares in 8 groups of the 23 lane-strided iterations, reduced per group, groups added in
  order; B <= 2 spreads the groups over the 8 warps, B >= 3 and the wide prep do the 8 groups inside one warp: the
  same bits in every norm; tokens exact, hsk_check B=1/3/8 equal): profiling build B=1 generate -11.7 (norm1 span
  1.74 -> 1.51, D 3.81 -> 3.62), normal build fixed-content B=1 +5.7, B=8 +44 (8 butterflies per row in norm_warp).
  Off (code kept, NORM_G8 0).
The profiling build and the normal build allocate differently; a structural gain seen in the former has, again and
again in this run, not survived in the latter.
- MARK_SB (normal-build phase marks as scheduling boundaries; same bits): 1 (memory clobber) B=1 0 / B=8 +-1; 2 (empty
  volatile asm in a thread-0 branch) B=1 +9, B=8 +12..+20; 2 + NORM_G8 2 B=1 +17. Off.
- NORM_G8 2 (per-lane group partials added in order, one butterfly: cheap in norm_warp): profiling build -12 at B=1,
  normal build fixed-content B=1 +17..+38, B=8 +7. Off.
- HSK_RSK_F 1 with the B <= 2 exact head left alone (timing study, RSK_F_STUDY): B=1 +4 (3 rounds; with the
  exact head's segmented order +8): the B <= 2 sketch function's own code change costs about what its rescoring
  saves.
- Head cross-barrier prefetch depth (PF_H_F / PF_H_W, default = the instance's depth): 3 stages B <= 2 0 (B=1),
  B >= 3 +7 (B=8). Unchanged.
- NORM_G8 on several cards (B=1 serve, 3 rounds, B=1-only build of the bench): G8 1 card 3 (pod epoch 7) -10.5,
  card 2 +2.4 / +5.4, card 1 (epoch 8) -7.3; G8 2 card 2 +25. The B >= 3 cost of G8 1 is not its 8 butterflies:
  a transpose-reduce (at xor 16 / 8 / 4 a lane keeps half of its groups and adds the partner's; bitwise equal to the
  8 butterflies to position 1100) still leaves B=8 +51 on fixed content. With alone == batch required, the B <= 2
  order cannot differ from the B >= 3 one: NORM_G8 stays off.
(The pod was replaced during this, epoch 8: cards 1-3 are new hardware.)

## ~30-30.5 agent-h: per-SM speed and the B=8 consume (studies, no engine change)
bench/smbw.cu + tools/study/blockspeed.py: per-SM stream speed differs by ~10 % stably; the engine's per-block
phase times do not follow it (findings.md). ROUTE_FIX / NOCONSUME / NOCVT / FP4_NOB (study switches, off): at B=8
~15 us per layer of consume is not hidden by the ring (findings.md).

## ~30.5-31 agent-h: track A, SPF in the B <= 2 instance only
Track A re-sweep on the s0057 defaults (one round): SPF 1 B=1 -18 / B=8 +31, PF_H_F 3 -4 / 0, PF_H_W 2 0, HDYN_S 72
/ 80 0, ATT_BLKW_F 2 +12, STEPARG 0 +12 / +76, GB_NS 20 -1 / 0. SPF split per instance (SPF_F / SPF_W; a prefetch:
same bits; both track builds stay SASS-identical to s0056 / s0057 with the old values): SPF_F 1 (3 rounds) B=1
-17.9, B=8 +4.7; with PF_H_F 3 -17.6 / +3.3 (no extra). SPF_F now 1 in both tracks (track B had it already).
Presubmit (TRACK=A): batch vs single bitwise 480/480, alone vs batch identical 8 x 1024.

## s0059 track A: SPF_F 1
Predicted HTTP B=1 1760 -> ~1742, B=8 3067 -> ~3071. Measured 1738.5 (best track A B=1, 1.264x SGLang) / 3076.

## Track A re-sweep on s0059 (one round), ETAIL 0
B=1 / B=8 serve vs base: V_ASYNC_FLAT 1 +38, ATT_BLKW_F 2 +29, KVPF_F 0 +30 / -8, PF_STAGES 3 +31, QXO 1 0 / +62,
FTMA_FLAT 0 +28, ATMA_FLAT 0 +27, EBPF 1 +22, PF_STAGES_W 2 0 / +13, ETAIL 0 0 / -10.2. ETAIL 0, 3 rounds on two
cards: B=8 -12 / -9.6, B=1 0 (same bits). ETAIL now 0 in both tracks (track B had 0 already).
