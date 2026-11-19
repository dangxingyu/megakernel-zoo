# Journal, run s11-q8b-h100-b1 (track B on the inherited s10 engine)

## s0001: inherited track B re-measured (hour ~0.5)
Expected ~4140-4160 / 4460-4470. Measured 4162.3 / 4457 (in process step 4019 at mean ctx 192, w1-like step 4152,
w1-like generate 4114). Matches the operator's rematch. Reference for this run.

## Census of the scaled weights (e0001, tests/census2.py), hour ~0.8
Exact counts over all 7.57e9 weights after the engine's own fit_shifts (row/column power-of-two scales):
d = row-max exponent - exponent: 0: 1.7%, 1: 17%, 2: 29%, 3: 23.5%, 4: 14%, 5: 7.5%, 6: 3.9%, 7: 2%, 8: 1%, halving on.
H(d) = 2.65 bits, H(d | row) 2.57, H(mantissa | d) = 6.93, H(sign) = 1.000. Order-0 total 10.59 bits/weight;
Huffman(d) + 8 = 10.70. Fixed 3-bit window (11 bits): 1.5% escapes; 2-bit: 15%; 4-bit (today): 7e-5.
So the only compressible part is the exponent field (4 bits stored, 2.65 carried): at most ~11% of the bytes.

## ALU headroom of the 12-bit stream (e0002, tests/vl/bwalu.cu), hour ~1.2
Engine-shaped stream (132 x 24 warps, mask-and-OR decode, MMA), + N dependent ALU ops per MMA:
+0: 3126-3132 GB/s; +8: 3122; +16: 3119; +24: 3109; +32: 3015; +40: 2640; +48: 2370; +64: 1883.
A variable-length exponent code at ~10.9 bits/weight needs ~10% more MMAs per second and, by my count for a
pair-LUT prefix decoder (peek, LDS, 64-bit shift, length bookkeeping, refill, merge), ~30-35 extra ALU ops per
MMA: past the knee. Conclusion: entropy coding beyond the 12-bit code is ALU-bound on H100 with SIMT decoding;
not built. (Dead end, measured.)

## Profile of the step (e0006, TIMING build, tests/timing.py + time_layers.py)
B=1 pos 192: per layer 103.2 us (slope), embedding + lm_head 318 us. Phase sums: qkv 10.98, attention window
(idle blocks: stash 1.09 + wait 7.16) 8.25, o 7.09, gate+up 50.64, down 25.49. Bytes at 3.12 TB/s: qkv 12.1,
o 8.07, gu 48.4, down 24.2 (92.8 per layer). The matvec phases are within ~1.4 us of their bytes; the loss is the
attention window (~4.2 us of idle HBM beyond the o stash) and the o phase after it (~2.9 us).
Attention body at pos 192 / 900: 5.48 / 9.12 us (prep 1.0, scores 1.45 / 3.14, softmax .84 / 1.25, V 1.0 / 2.6,
store+fence ~1.1). B=8 pos 576: body 13.2 us.

## What-if: attention capped at 16 positions (e0010, wrong results on purpose)
B=1 mean over 64..1088: normal 4094-4123, capped 4032 (difference 0-26 us below 192 positions, 117-150 at
960-1088). B=8: 4403-4413 vs 4213. So a perfect attention speed-up is worth ~62-92 us at B=1, ~195 at B=8; a
split with ~1 us merge per layer would keep maybe half at B=1.

## L2 as a staging area (e0008, e0009, e0011: tests/vl/l2stage*.cu, l2bw.cu)
- L2-hot streaming is only 3.3-3.5 TB/s on this card for the engine's access pattern (cold 2.85-2.97 TB/s):
  staging a whole phase into L2 cannot make it much faster.
- But L2 hits ADD to the HBM stream: a 151 MB stream (gate+up size) with 25 MB staged in L2 beforehand (bulk
  prefetch or plain loads, evict_last) took 48.3-49.3 us instead of 55-56 us cold; 37 MB staged: 46 us. ~0.27 us
  saved per MB staged (78% of the staged bytes' HBM time). Interleaved staging slightly better than front-loaded.
- Mechanism for the engine: the attention window leaves HBM idle for ~4 us (pos 192) to ~7.5 us (pos 900) per
  layer after the o stash; the idle blocks can stage later gate+up tiles into L2 there.

## L2 staging in the engine (e0012, e0015, e0016, e0017, e0022, e0023, e0025), hours 3.3-4.6
Idle blocks of the attention phase (1-2 sequences), after their o stash, bulk-prefetch (cp.async.bulk.prefetch.L2,
16 KB pieces) later gate+up tiles of their own block (tile ng/2, ...). Same-card A/B, generate 64..1088, B=1:
- track B, evict_last hint: base 4122.6, 1 tile 4065-4067, 2 tiles 4076-4079, 3 tiles 4112, 2 + o rest 4124.
- track B, no hint (3 rounds): base 4120.5, 1 tile 4051, 2 tiles 4039, 9 pieces (1.5 tiles) 4033, 4 pieces 4051.
  The hint costs ~25 us (evict_last lines linger?).
- B=8 (window longer, but HBM busier): +17..+46 us -> off with 3+ sequences (STG_MAX_B 2).
- track A (3 rounds): base 5330, 1 tile (128 KB, no hint) 5284, 4 pieces 5284, 12 pieces 5293.
Prediction from the microbenchmark (0.27 us per MB staged, ~12 MB/layer at short context) was ~110 us; measured
87 (B) / 46 (A). More staging is worse: the staging traffic delays the attention blocks' L2 hits (V pass, stores).
Also at the end of o / gate+up / down (loader_tail): 16 KB pieces of the next phase's middle tile (STG_TAIL),
first look (2 rounds, on top of 1-tile attention staging 4057-4066): 2 pieces 4019, 4 pieces 3997-4016, 6: 4022-4040.

## s0002 (track A, attention-window staging): valid, http 5362.7 / B=8 5684
Expected http ~5330 (in-process A/B -46). In process (w1-like, 64..1088): step 5364.9 -> 5322.3, generate
5328.8 -> 5282.3 vs s10 s0004: the predicted -43..-47 exactly. Over HTTP only -2 (5365 -> 5363): the HTTP path
added +40 us over the in-process step this time (s0001: +10, s10 s0004: +0; B=8 +57 vs +9). HTTP noise of
evaluation to evaluation is ~+-25 us; read the in-process w1-like numbers for the kernel effect.

## s0003 (track B, attention-window staging, 9 x 16 KB): valid, http 4095.2 / B=8 4475.5
Expected w1-like step ~4065, http 4075-4100. Measured w1 step 4087.2 (-64 vs s0001), generate 4045.9 (-68), http
4095 (-67). A bit less than the same-card A/B (-87; separate builds, +-15 us). Numerics unchanged (bit-identical).

## s0004 (track A, tail staging 4 x 16 KB): valid, http 5227.2 / B=8 5662.7
Expected w1 step ~5200, http 5200-5240. Measured w1 step 5205.8 (-117 vs s0002), http 5227 (-136). Confirmed.

## s0005 (track B, tail staging 3 x 16 KB): valid, http 4015.3 / B=8 4477
Expected w1 step 4010-4030, http 4020-4050. Measured w1 step 4008.5 (-79 vs s0003), generate 3963.7, http 4015.
Confirmed. Run total on track B: 4162 -> 4015 (-3.5%); track A 5365 -> 5227 (-2.6%).

## Layer-0 table (track B, var/l0): built and verified, not yet submitted
Table [151936][6144] float32 (3.7 GB) built at load by the kernel's own embedding + q/k/v phase, 8 ids per launch
(11 s). Bitcheck vs the engine without the table: bit-identical (1152 positions, B=1 and B=8). Timing: two separate
builds A/B (e0044) showed nothing at B=1 (build noise); the same library with MK_L0TAB on/off (tests/l0_onoff.py):
B=1 -1..-16 us (mean ~-10), B=8 -14..-15 us. Predicted -12..-14.

## Tail staging at B=8 (e0050): 3 pieces 4421 -> 4329 (-92), 6 pieces -38. Separate limit STG_TAIL_MAX_B.

## STG_OA (idle blocks stage the attention blocks' cold o tiles at window start), e0060: no effect (3953 vs 3953, +-15). Off.

## s0006 (track B, tail staging with 3-8 sequences, 3 pieces): valid, http 3998.2 / B=8 4415.0
Expected B=8 w1 step ~4370 (s0005 4460), B=1 unchanged. Measured B=8 w1 step 4373.5 (-86.5), http 4415 (-62);
B=1 w1 step 4008.3 (unchanged), http 3998 (-17, HTTP noise). Confirmed.

## s0007 (track B, layer-0 q/k/v table): valid, http 4007.8 / B=8 4369.2
Expected w1 step ~3998 (B=1) / ~4355 (B=8). Measured 3992.3 (-16 vs s0006) / 4353.2 (-20). Confirmed in process;
HTTP B=1 +10 (noise), B=8 -46. GPU memory 20.0 GB (table 3.7 GB).

## Subagent: online contract + int8 head sketch (var/online, built on s0001)
Online (submit/poll, one k_generate launch per cohort, tagged tokens in host-mapped memory, no device sync in poll):
-40.4 us/token B=1, -43.6 B=8 vs the adapter loop (same build, 3 rounds). Sketch (greedy paths only, certified
int8 bound + exact rescoring with the head's own code): generate -50 (B=1), -47 (B=8); online+sketch vs adapter
-94 / -90. Token identity: 10,880 teacher-forced positions (all-tiles-rescored mode too), free runs 1024 / 8192.
Sketch code perturbed register allocation of the hot loop (+134..+355 us) until a k_gensk-only variant (SK_LITE=2).
Porting the online part onto the current engine now (gen_tag replaces the unused Params.trace_stop).

## s0008 (track B, online submit/poll contract): valid, http 3952.2 / B=8 4316.9 (path: online)
Expected 3955-3965 / 4310-4320. Measured 3952 / 4317: confirmed. HTTP now = w1-like generate (3945.9) + 6 us;
trace: 1 launch per step, host syncs 0.008 per step, GPU idle 0.09%. http_tokens 8/8 identical.
Run total track B: 4162 -> 3952 (-5.0%).

## Staging amount sweeps (e0069, e0070)
B=1 attention-window pieces per idle block, 3 rounds, per context bin: 6: 3953, 9: 3932, 12: 3931, 18: 3965 mean;
9 and 12 equal in every bin (no context dependence worth a rule). B=8 attention-window staging 3 pieces: -30 below
192 positions, +37 at 960-1088 (net ~-7): not used.

## o-phase barrier spread at pos 192 (e0076, tests/timing_blocks.py: per-block barrier waits)
The o phase ends with a 6.6 us spread; the last blocks are ordinary idle blocks (66, 67, 14, 78, ...), not the
attention blocks: their second o tile competes with gate+up staging still in flight. (clock64 is per SM: absolute
stamps cannot be compared across blocks; use the own-barrier waits.)

## Split attention from fewer positions (e0078, LONG_NT 24/40/56 vs 72): worse wherever it switches on (+70..+110
us per step in those bins at B=1). The extra grid barrier costs more than the parallel scores gain. Dead end.

## s0009 (track A, online contract): valid, http 5159.9 / B=8 5587.9. Expected 5170-5180. Confirmed (-67 vs s0004).

## s0010 (track A, tail staging 3+ sequences): valid, http 5162.3 / B=8 5486.3. Expected B=8 ~5495: confirmed (-102).
## s0011 (track B, int8 head sketch on the greedy/online paths): valid, http 3894.7 / B=8 4248.2
Expected ~3900 / ~4265. Confirmed (-57 / -69). Track B run total: 4162 -> 3895 (-6.4%), B=8 4457 -> 4248.
## HBM request mechanisms (e0080, tests/vl/bwtma.cu, 152 MB): loads 2.82-2.97 TB/s, TMA g2s 3.0-3.02, loads + bulk
L2 prefetch of the next tile 2.72-2.79 (worse). No mechanism lifts the practical rate meaningfully. Not pursued.

## STG_GUA (idle blocks also stage the attention blocks' gate+up pieces; those were the last blocks of gate+up):
e0088, 3 rounds: B=1 3871 vs 3872, B=2 3960 vs 3970: no effect. Off.
## Rounds in one process (e0093): 3886 us/token in all 8 rounds, with or without 3 s pauses; SM clock 1830 MHz,
620-660 W. No round-to-round drift in the engine.
## s0012 (correctness: token ids outside the vocabulary): valid, http 3869.0 / B=8 4263.3. In process unchanged
(w1 generate 3893.1 vs 3892.6); the -26 at http is noise. Edge tests (tests/test_edge2.py) all raise ValueError.

## Subagent: 4-bit head sketch study (var/study4/head4.py, e0097): dead end. Certified bound ~18x int8's: int4
per-row: every tile a candidate; best int4 (per-32 bf16 scales, block bound) 6436 of 9496 tiles at B=1: net +124
us/token. int8 with block-norm bound: half the candidates, +39 MB: net ~0. Sketch trimming (var/sketch3): x
preparation with 16-byte loads kept (B=8 ~-15, B=1 ~0, one-instance A/B); L2 prefetch of likely pass-2 tiles: no;
pass 2 costs ~19 us at B=8 (cold stream of ~36 candidate tiles serialised per owning block), ~0 at B=1.
Measurement note (subagent): two engine instances of identical code differ by up to ~10 us at B=1.
## STG_TAIL_FIRST (stage the next phase's first tile instead of the middle one), e0094: worse or equal. Off.

## s0013 (sketch x preparation, 16-byte loads): valid, http 3881.5 / B=8 4246.6. B=8 w1 generate 4249.6 -> 4238.7
(-11), B=1 unchanged (noise): as predicted.
## int5/6/7 head sketches (e0098, var/study4/head6.py): B=1 greedy candidate tiles: int8 5.9, int7 43 (p99 637),
int6 1091 (block bound 149), int5 6921. Bytes saved (int7 -78 MB ~ -25 us) are eaten by rescoring serialised per
block. int8 stays.

## Stress (tests/stress.py 6 min, e0099): PASS, 2083 segments / 81,838 steps, random B, positions to 9216, step and
generate mixed, every segment bitwise repeatable (track B with table, sketch, staging, online code).
## Staging the down projection's tiles in the attention window (STG_DN, e0105): +40..+60 us in every variant (also
instead of gate+up). Staged lines only pay for the immediately following phase. Off.

## Two-chunk split attention with fixed-order merge by the second arriver (subagent, var/att2, off): dead end
Correct, batch invariant (bitwise), numerics fine (vs HF fp32 at 128..1152, B=8: median 0.0054 vs root 0.0052).
Timing (same process, 3 rounds): T0=256: B=8 +43 / +47 us, B=1 neutral (+6 / -0.7). Profile B=8 pos 576: attention
body 13.70 -> 12.35 only (scores 5.7 -> 3.6, V pass unchanged ~4.1: latency-bound per warp), wait after body
1.04 -> 2.19 (merge), and with 128 attention blocks the idle blocks' o stash is gone (o first tile 2.86 -> 4.81).
Per layer 110.4 -> 112.8. Larger K/V touch (TOUCH_TILES 384/512) also worse at B=8.

## Knob sweep on the staged engine (e0116-e0119, 3 rounds, B=1 / B=8 deltas vs default, us)
WPF 4: 0/-4, 16: +25/-2; WPF_DEPTH 2: +3/-7, 8: -20/-12 (noisy); NPF 4: -15/0 (noisy); NPF_ST 0: +21/-1, 8: +11/+1;
ATT_WAIT 0: +1/-1, 32: +25/-2; ATT_POLITE 0: +29/-7; O2PF 0: -25 (-26/-30/-21) / +5, 12: +30/+3; SG 1: +21/+21,
4: +20/-5; NPF_ATT 6: 0/+19; GB 2: +65/+4; GBW 1: +9/+2, 3: +18/+261 (spills); NPF_STW 4: +6/-1, 16: +4/+19.
CAP_D 2 violates the smem layout assert. Confirming O2PF 0 (+ WPF_DEPTH 8, NPF 4) with 4 rounds (e0120).
Confirmation (e0120, 4 rounds): O2PF 0: B=1 -26 (-21/-50/-11/-23), B=8 -0.5; + WPF_DEPTH 8 / + NPF 4: no further gain.
CAP_H 2/8: +31/+26; CAP_HW 2/8: B=8 +16/+15 (e0122): defaults right. Second batch (e0123/e0124): WPF_SMALL_B 1/8
-17, NPF_EMB 0 -15, ECB 3 -18 at B=1 (noisy rounds), LM_NEXT 0, KBATCH 0, TOUCH_FL 2, ECB 1, FENCE_BAR 0: ~0.
## s0014 (O2PF 0): valid, http 3865.3 / B=8 4254.6. w1 generate 3898.6 -> 3865.2 (-33; predicted ~-26). Bit-identical.

## Track A knob sweep (e0128-e0132, 3 rounds, very stable +-2): NP 12 (STG_GU 2) B=1 -22; SG 4: -7 / B=8 -22;
NPF_ATT 0: -4 / B=8 -19; NPF_ST 8 -6; ATT_POLITE 0 +40; CAP_H 2/8 +16/+19; others ~0.
Combination (e0134): NP12 + SG4 + NPF_ATT0: B=1 -32, B=8 -63 (rounds -32/-28/-37, -66/-57/-67). Adopted for track A.
Track B: NPF_ATT 0 ~0, SG 3 +8 / +29 (e0135). GU_SHIFT (gate+up tiles of the attention blocks moved to the 4
blocks without gate+up tiles, e0133): no effect. Off.

## s0015 (track A re-tune: NP12 + SG4 + NPF_ATT0): INCORRECT (numeric, tokens, batch, numeric_long failed)
Cause (bisect by bitcheck, e0137-e0140): NPF_ATT 0. The attention blocks' "nothing stashed" reset of the cold o
tile's stash count sat inside the `#if NPF_ATT > 0` block of the inherited code; with 0 the o tile loop used a
stale count and stale shared-memory entries. SG 4 and NP 12 are bit-identical. My mistake: the bitcheck for the
track-A constants ran in parallel with the submission instead of before it; the sweep's "-19 us" for NPF_ATT 0 was
partly the engine skipping loads. Fixed the reset placement (harmless at the default 3); track A back to NPF_ATT 3.
Rule from now on: bitcheck every constant change BEFORE timing it or submitting it.
## s0016 (track A, NP12 + SG4, fix of s0015): valid, http 5133.8 / B=8 5462.2. Expected ~5135 / ~5440. Confirmed
(-28 / -24 vs s0010). NPF_ATT 0 with the fixed reset (e0144): -4 / -5, not adopted.
## STG_EARLY (attention-window staging requested before the o stash, e0147): +61 (4 rounds +52..+69). Off.
## LM_NEXT_O (end-of-head prefetch at layer 0's o tiles with the layer-0 table, e0148): B=1 -7, B=8 +3: noise. Off.

## Exceptions: cost and fix (EXC_SMEM)
What-if NOEXC (exception MMAs skipped, wrong results, e0149): B=1 -38 us, B=8 ~0. Cause: the coordinator runs the
exception MMAs before each tile's block barrier and, with 1-2 sequences (XSLICE), gathered each exception's x value
from the compact global copy: one L2 round trip under the full stream while all loader warps wait at the barrier.
Fix: from a phase's second tile on, read x from the loaders' shared-memory slices (same layout [kstep][seq][16]);
the first tile still reads L2 (slices may not all be stored yet). Bitcheck B=1/2/8 bit-identical (e0150); online and
edge tests pass (e0152). Same card, 4 rounds (e0151): B=1 -40 (-44/-32/-35/-48), B=2 -58, B=8 -8.
## s0017 (EXC_SMEM): valid, http 3837.4 / B=8 4240.1; w1 generate 3865 -> 3824 (-41, predicted -40). But the evaluator's
step() path got slower (w1 step 3989 -> 4033): same-card step() A/B (tests/time_ctx_step.py) +69 at B=1. ptxas:
the new path made k_step and k_generate spill (32 B stack, 48 B spill stores; k_gensk clean). A 32-bit shared
address instead of the pointer did not help. Now enabled only in k_gensk (template SK): all three kernels 0 spills,
step() A/B +2.6 (noise), generate -35.5, bitcheck identical (e0158). My miss: I timed only the generate path and did
not check spills of all kernels before submitting.
## s0018 (EXC_SMEM only in k_gensk): valid, http 3825.2 / B=8 4257.6. w1 step 3989.9 (back), w1 generate 3841.8.
Run total track B: 4162 -> 3825 (-8.1%). What-ifs after it (e0159): NOEXC -1 (exceptions now free), NOTAB -14/+28.
## STG_KV (each block bulk-prefetches its share of the layer's K/V history at the end of q/k/v, e0166/e0167):
B=8 +32, B=4 +15, B=1 +5..+8 (also with TOUCH 0: worse). Off.
## Note: mk_set_stop_layer (debug) is not supported by generate()/k_gensk (barrier counts use all 36 layers): a test
of mine hung card 6 until its job timeout (e0168). Test removed; not an engine path.
## Per-transition tail amounts (STG_TAIL_O / STG_TAIL_G, e0171; refactor: Desc.stgN): after o 6: +38, 9: +49; after
gate+up 6: +15; both 6: +39. Uniform 3 stays (defaults unchanged).

## Load scheduling in the one-block attention body (subagent, var/att3, all off): dead end
One-instance runtime A/B (noise +-1): K L2 prefetch per lane +7 / B=8 +26; bulk 4 KB +7 / +13; VSB 4 +4.5 / 0;
V prefetch before softmax +23 / +10; at score start +20 / +44; bulk 512 B +25 / -8 then +12 (sign flips between
instances); register pipeline of K tiles: spills (stack 136-216) in all three kernels, not timed. Bitchecks of all
variants identical. B=1 always worse (K/V already in L2 there through the idle blocks' touch).

## ptxas flags (e0185-e0188): --allow-expensive-optimizations first looked -17 (3 rounds), the 4-round confirmation
gave +8 at B=1, step() path 0, track A 0: build noise. Bit-identical (proper bitcheck with a separate build.sh,
e0188). -Xptxas -O2: mixed. Not adopted.

## Final state (run clock ~5.5 h): engine/ = s0018 source exactly (TRACK B). Best: track B s0018 3825 / 4258,
track A s0016 5134 / 5462 (build TRACK=A from the same source). Further ideas are within noise or measured dead.

## After the time check (run clock 5.7 h): more B=8 / B=1 scheduling
- ATT_WAIT for B=8 (+ small staging) e0189: +18..+29. XS_ALL 1 (x fragments kept in smem, B>=3) e0191: +9 / +15
  (bit-identical). Off.
- Tail staging after o aimed at gate+up tile 2 instead of the middle one (STG_TAIL_GU_TI 2): 4 rounds tile 1 -7,
  2 -17, 3 +21; 6-round confirmation B=1 -20 (5 of 6 rounds -20..-26), B=8 -8.5. Bitcheck both tracks identical,
  no spills (e0194). Hypothesis: the tiles right after the stashed first tile are starved (one group per warp in
  flight while the first tile is multiplied from smem).
- s0019 (that change): valid, http 3842.7 (+18: HTTP noise) / B=8 4257.4; in process w1 generate 3841.8 -> 3825.0
  (-17), w1 step 3989.9 -> 3952.6 (-37). Confirmed in process.
- Attention-window staging aimed at earlier tiles (STG_GU_A/S, e0195): tiles 1,2 +83; 3,5 +34; 3,6 +40; 4,5 +23.
  The spread default (4, 8) stays.
- Track A with STG_TAIL_GU_TI (e0196, 3 rounds): 2: +8.5 / -4; 1: +2.5 / -34. Track A keeps the middle tile (0).
## Skeleton (e0199, -DSKEL): 1874 us B=1, 2357 B=8 (real 3814 / 4225). See gap.md.
## STG_O (stage the idle blocks' cold 2nd o tile in the window, no hint, e0203): +18 (with 9 gate+up pieces),
+39 (with 3). Off. Pods lost repeatedly (epochs 7-13); several sweeps dropped after two losses.
## STG_TAIL_D 4/6 (more of q/k/v's second tile after down, e0204): 0 / +21. Off.
## Fine re-tune batch (e0205-e0207, 3 rounds, B=1 / B=8): ATT_POLITE 3 +3/-9, 7 -20 (-19/-21/-21) / -14; ATT_WAIT 8 -20
(-21/-16/-22) / -6, 24 -4/-11; CAP_H 3/5 +19/+16; STG_NP 8/10 +4/-5; WPF 6/10 +1/+26; WPF_DEPTH 3/6 +7/0.
Confirming ATT_POLITE 7 and ATT_WAIT 8 (e0208).
5-round confirmation (e0208): ATT_POLITE 7 -10, ATT_WAIT 8 -10, both -11 (rounds -21/-5/-4/-5/-21) at B=1, B=8 -5.
Bitcheck both tracks identical, no spills (e0210). Adopted for track B only (track A keeps 16 / 5).
## s0020 (track B: ATT_WAIT 8, ATT_POLITE 7): valid, http 3812.6 (best) / B=8 4252.6; w1 generate 3833.0 (s0019 3825.0):
the -11 change is inside evaluation noise; board best by draw as much as by change.
## e0211: ATT_POLITE 9/12 +32/+33, ATT_WAIT 4 +6, 4+9 +43: defaults 8/7 stay.
## NPF_ST 2 / 0, 5 rounds (e0214): -16 / +2 at B=1, but rounds bimodal (0 or ~-40): not robust. Off. Checking for a slow/fast instance mode.

## Instance-level step-time modes (placement)
- Same build, 10 processes (e0215): 3811-3813 (6) vs 3833-3836 (4) at B=1, 64..1088.
- 8 engine instances in one process (e0219): 3761 (3) vs 3811 (5) at 64..576: the mode is set per instance.
- Not the barrier counter's page (16 places, e0216: 3760-3765, ~4 us effect), not the weight buffer (copies, e0217:
  2-5 us), not the int8 sketch (copies, e0218: ~1 us), not K/V, activations or small scratch (re-allocated in place,
  e0221: no change). Copies of the per-tile table (ttab) or the layer-0 table flip it by ~15-20 us (e0222).
- Load-time calibration (MK_PLACE=n: ttab + n copies, the fastest of a 96-step timing kept; opt-in, default off):
  6 instances without 3772-3788 (mean 3779.6), with 3770-3777 (mean 3774.5) (e0223). The large 50 us modes did not
  recur on that pod. Marginal; left off. Diagnostic APIs (mk_set_sync_page, mk_set_weights, mk_set_tabs,
  mk_realloc) kept for a future study.
## SK_STG_TI (sketch path: stage the 2nd/3rd sketch tile at the last down's end, e0224): +9 / +14 at B=1. Reverted (engine/ = s0020).
## NB_MAT 0 (all 132 blocks for the matvec phases, e0228): B=1 +29; NB_MAT_W 0: B=8 +27. 128 stays.

## Sketch pass-2 balancing (subagent, var/sk4, merged): candidate tiles go to a global list (per-step parity slots,
atomic counter, one extra grid barrier), block b rescores entries b, b+132, ... with the exact head's code; the
per-block best + global (value, lowest row) reduction is unchanged. On from 3 sequences (the extra barrier costs
~2.6 us at B=1). One-instance A/B: B=8 -14.0, B=6 -10, B=4 -6, B=1 path unchanged; compiled builds: B=8 -12.6..-18.9.
Identical results (sketch == exact at 10,880 positions with balancing forced on at every B, online, bitcheck).
## s0021 (sketch pass-2 balancing, 3+ sequences): valid, http 3821.8 / B=8 4241.2. B=8 -12 (predicted -13); B=1 noise
(w1 generate 3818.8 vs 3833.0).
## Track A (e0237): STG_TAIL_GU_TI 1 +5 / -4 (earlier -34 at B=8 was noise); ATT_WAIT 8 + ATT_POLITE 7: +27 / +9. Track A unchanged.
## ATT_STASH_CAP 6/7 (stash less of the o tile in the window, e0238): +65 / +21. Full stash stays.
## STG_GU_A/S 5,9 / 6,10 / 4,7 (e0239): -2 / -3 / +3: flat. Default stays.
## Mode determinism (e0240, var/diag): alternating two copies of ttab, then of the layer-0 table, inside one instance:
no change (+-1 us) in 3 processes (process levels 3771 / 3770 / 3780). The flips seen in e0222 were not caused by
these two addresses; the per-instance mode (10-50 us) remains unexplained. Closed.
## Block -> SM placement (e0241, var/diag: %smid per block): fixed mapping, block b on SM (b + 124) mod 132, so the 8
attention blocks always sit on SMs 124-131 (adjacent). No modes in that run (3778-3780 over 8 instances).
Role permutation (logical block = hw block x BPERM mod 132, bit-identical relabelling, e0242): BPERM 17 / 5 / 65:
B=1 -2 / -8 / -6, B=8 +10 / +14 / +15. Not adopted (engine/ = s0021).

## B=8 attention load issue (subagent microbenchmark var/att4/mb/att_mb.cu, pos 576 shape): bandwidth-bound, not
latency-bound, once the idle blocks' o-stash/prefetch traffic runs beside it: K pass 7.43 us (registers, today) vs
7.27-7.35 (TMA into smem, what fits) vs 7.45 (L2 bulk prefetch); V 7.51 vs 7.45 (TMA) vs 7.70 (VSB 4). Alone: K 4.09
vs 3.57 only if all 37 tiles (148 KB) were staged, which does not fit. No engine change. Remaining B=8 levers need
fewer bytes in the window (e.g. an exact compact K/V code: ~1.5% at B=8, ~0 at B=1) - not pursued.
## K/V cache census (tests/kv_census.py, e0247; real text, 1024 positions, HF bf16): per 128-value vector, 16-binade
window anchored at the max exponent: K 3.1e-4 of values outside (+2.7e-4 exact zeros), 3.2% of vectors with an escape
(39% of 16-position tiles); V 6.6e-5, 0.8% of vectors. 8-binade window: K 7.5%, V 1.6% outside. A 12-bit K/V code with
tile-level fallback would save ~19% of K/V bytes: ~5 us at B=1, ~40 at B=8, for a rewrite of append, score and V
paths. Not built.

## 12-bit K cache (subagent, var/kv12, off): dead end
What-if ceilings (wrong results): K bytes -18 at B=8 (all from position 512 on), K+V -26; B=1 noise. Built for B>=3:
per-position power-of-two scale (undone exactly on the score), tile-level bf16 fallback for escapes, metadata behind
the K cache. Bit-identical logits (B=1/2/8 to 1200 positions, sketch / online / edge tests). But B=8 +7.9 (RMW
append) / +13.8 (OR append): the score pass gains 0.85 us at pos 576 while the attention prep (critical path in every
layer) grows ~1 us. Also fails tests/stress.py's rewind scenario (re-appending other values at old positions trusts
stale tile flags) - today's bf16 append is immune. Not merged.

## 11.5-bit pair exponent code (subagent, var/w115): stopped at gate 2
Gate 1 (exact census, all weights, engine windows): 7-bit joint pair code, 127 pairs + escape: escapes 0.19-0.21% of
pairs (pair-code entropy 5.30 bits), exception records ~157 per tile (max 1504), +1.04% bytes: net byte saving 3.1%.
Gate 2 (engine-shaped stream microbenchmark var/w115/mb/bw115.cu): 11.5-bit with LUT decode 2077-2080 Gweights/s vs
12-bit 2084-2088 (-0.3%); layout ceiling without decode +1.7% (23 words per lane cannot use 16-byte loads: 3049 vs
3127 GB/s); decode costs ~2%. Best case after records ~+1%: below the gate. Not built.
## TOK_NOFENCE (online tokens without the system fence, e0271, 3 rounds online path): B=1 -1, B=8 -3: noise. Not adopted.

## Limits, final source (s0021): tests/test_batches.py PASS (every B 1..8 row 0 bitwise = B=1, generate/step mixes,
resets); bitcheck B=1..8 batch invariance 1.0 at 1300 positions (e0272). tests/test_ctx.py 9216 1 2 8 (e0273): vs HF
float32 median 0.005-0.009 / max 0.129 (HF bf16 0.012-0.021 / 0.408), top-1 1.000, finite, row 0 bitwise equal alone
vs B=2 / B=8 through the split path. Online path with long prompts (tests/online/test_long_online.py, e0274):
served == generate for 200/200 tokens after a 3000-token prompt and 450/450 for a cohort of three 2000-token prompts.

## Deferred attention-window staging for 3+ sequences (STG8_NP): idle blocks stash at once as before, then poll the
attention blocks' score-pass stamps (now also written at B>=3) and only then stage 16 KB gate+up pieces, so the
staging fills the softmax / V-pass part of the window instead of competing with the K reads (the window is
bandwidth-bound only during the score pass: subagent att4). e0275: B=4 -30 (6/12/18 pieces alike), B=8 6 pieces -9,
12/18 +9. e0276 (4 rounds): 3 pieces B=8 -11 (-12/-12/-10/-12), 6 -9; B=1 path unchanged (+-4). Adopted for track B
(3 pieces). Bitcheck B=1/4/8 identical, online tests green, no spills (e0278).
## s0022 (STG8_NP 3, track B): valid, http 3813.4 / B=8 4231.5 (-10 vs s0021, predicted -11); B=1 w1 generate 3811.2.
Track A (e0277): B=4 -32..-39, B=8 -4 / +1: not applied to track A (its evaluated numbers are B=1 and B=8).
## NPF_DEFER (3+ sequences: the idle blocks' L2 prefetch of their 2nd o tile waits for the score-pass stamps, e0279):
B=8 -9.4 (-10/-10/-9), B=4 0; dropping the prefetch: B=4 +15. Adopted (track B). Bitcheck B=1/4/8 identical, online
green, no spills (e0280).
## s0023 (NPF_DEFER, track B): valid, http 3815.0 / B=8 4220.6 (-11, predicted -9).
## e0281 on s0023: STG8_NP 6/9 +7/+12 at B=8; ATT_WAIT_MAX_B 8: B=8 +45, B=4 -12. Defaults stay.
## B=8 profile on s0023 (e0282): 108.9 us per layer at pos 576 (113.9 in e0051); attention body 13.76 of which the V
pass is now the longest part (5.85 us; V cold from HBM: the idle blocks' touch is off beyond 256 tiles x sequences).
TOUCH_VONLY (beyond the touch limit, the idle blocks still pull V only into L2, e0283): B=8 -14.5 (-15/-14/-14), B=4
-1. Adopted (track B). Bitcheck B=1/4/8 identical, online green, no spills (e0284).
## s0024 (TOUCH_VONLY, track B): valid, http 3816.7 / B=8 4211.3 (-9, predicted -14).
## TOUCH_KL 64/128/256 on s0024 (e0285): B=8 +4 / +17 / +55. Off.
## TOUCH_VONLY on track A (e0286/e0287): B=8 -36 (4 rounds -53/-18/-28/-42), B=1 ~0; NPF_DEFER on track A +31 (stays
off there). TOUCH_VONLY now on for both tracks. Track A bitcheck identical, no spills (e0288).
