# Journal (run s18-mini-b200-c4, track C, direction: batch 1, fused)

## Orientation (0-1 h)
- c1 (s16) verify kernel = a2 s0016 lineage + mk_step<true>; s17's B=1 kernel = a2 s0025 + KB1/mkprof sites.
- tests/c/xcheck.py: c1 s0022's verify passes vs the s17 a2 step at B=1: 0 of 600 positions differ (bit-identical).
  So c1's engine can be the speculative path of the s17 engine without any kernel change.
- Card 1 (ctx 600): a2 step 2789 us; c1 verify K=1..8: 3401 3821 3940 4197 4652 5037 5314 5523 us.

## s0001 track-C baseline (s17 for step / B>=2 + c1 s0022 engine/c for B=1 requests)
Expected b1 ~1530, natural ~1020, b8 ~5090. Measured 1522 / 1016 / 5075 (natural_b8 5469); tokens/event 2.88 / 5.22.
Confirmed. All gates pass.

## merged verify kernel (engine/c/mk.cu = git merge-file c1 s0022 / base a2 s0016 / a2 s0025+s17; 22 conflicts by hand)
- the a2 s0024 o_proj conv-history preload rewritten for c1's 16-slot ring (only lags older than the pass's rows);
  c1's kv_conv kept (a2's kv_conv4 assumes the 3-slot ring); a2's union split (<= 4 tokens per entry) taken.
- xcheck 1200 positions (mixed K, crossing the 1024 window): 0 differ from the a2 step.
- tv2 ctx 600, same job alternating: c1 K1..8 3409 3850 3928 4155 4639 5017 5265 5415; merged 3326 3726 3795 4017 4479
  4824 5044 5115 (K=8 -5.5%).

## s0002 merged verify kernel — 1457 us b1 (natural 954), b8 5075
Expected ~1455 / ~975: b1 confirmed, natural better (-6%).

## after s0002: MoE investigations (all bit-identical where exact; see gap.md)
- Routing statistics on real text (ustat.py): K=8 30-34 union entries (26-32 distinct experts); random tokens 21.
  Real text costs MORE at K=8 than random tokens (V8 5698 vs 5115) — benchmark on real text (tv3.py) from here on.
- Bulk L2 prefetch of the block's expert items after routing: +200..+430 us. L1 prefetch of the next w13 round:
  +40..+150; allocating loads: +900 at K=8. W2_BPC 3/6, W2V=0, KV4 (broken: at.raw unset), SHARED_MIN 1,
  VDES_MIN 2, HYB2 0, MG_UNR 2: equal or slower.
- Timing-only builds (DIAG_MOE): no loads -> K8 -790 us, no decode/MMA -> -1050 us; w13 rounds are compute+latency
  per round (58 SASS instr per 576-B unit: 20 F2FP, 16 HMUL2, 8 LDS, 4 HMMA).
- W13_TMA (bulk copies of round>=1 items into free smem, mbarriers): bit-identical but slower (+230 us at K=1 with
  83 KB extra dynamic smem: L1 shrink; without extra smem +70..+100 at K=2-4: overflow does not fit, copies compete).
- Cycle statistics (cycstat.py): eval half the cycles K<=3 with ~1.5 tokens.

## s0003 w2 staged inputs padded (W2S_RS = IE + 8) — 1403.5 us b1 (natural 890), b8 5075
Found with per-block w2 times by token count (w2blk.py): 14.2 / 15.9 / 18.8 / 22.1 us for 1-4 tokens per expert;
the staged rows' stride (576 words) put all 8 B-operand columns on one bank. ncu: shared-load conflicts 111 M -> 16 M
per K=8 pass. Verify K=8 -8%, K=4 -5%. Expected ~1410 / ~900: confirmed.

## s0004 W2V=0 (step's 3-warp w2 items) — 1380.6 us b1 (natural 876.9), b8 5075
Expected ~1380 / ~875: confirmed. Once the bank conflicts were gone the one-warp items lost at every K.

## after s0004 (all measured on natural text, tv3; none kept unless said)
- MK_DK_FP8=all (FP8 copies of the draft's layer weights): in-process natural -0.4%, eval -0.8%: kept (default).
- INT8 certified argmax (MK_UN8=1): -12 us: the final phase (160 us) is not bound by unembedding bytes.
- KV4 (kv_conv4 with raw values fixed; bit-identical): +80..+200 us; KV4_SH (shared attention only): +100..+200
  even at K=1 (not using it): ptxas spills rise (stack 240 -> 368 B). Code generation, as s17 found.
- Record items holding the pass's positions: 5.2 us of load tasks vs 2.8-3.3 for the others (subint.py, tag 59).
- W13_STAG (half the w13 groups start 1-3 us later): equal or slower. -DMKPROF build: +120 us. W13_UNR 6 equal,
  8/9 +100. SASS: of a 12-unit chain's 24 weight loads only 6 are issued before the first HMMA (80 registers):
  the batch is not really 12 deep, and no build variant changes that.
- nsys natural: verify 4981 us median, draft span 126 us median on n-gram cycles (40 kernels exit early), mean 208.

## s0005 FP8 draft layers + FP8_UNR 3 + RT_SEQ 0 — 1374.1 us b1 (natural 872.3); expected ~1360 / ~868
b1 under-delivered (-0.5% vs -1.5%): eval tokens/event 2.878 -> 2.868 with the FP8 draft layers.
## s0006 W2X2 (two w2 items per group per round) — 1353.2 (natural 855.4); expected ~1355 / ~855: confirmed.
## s0007 DSpark kernels in a CUDA-graph IF node skipped on n-gram cycles — 1350.3 (natural 848.8); expected ~849.
## after s0007 (measured, not kept)
- W13X2 (two w13 items per round, 6+6 batches): +10..30 us (same number of load latencies).
- FG_ROWS 2/4/8: +400..+1100 us. VDES_MIN 5/9, HYB_MAX 3: slower. ptxas -O2 / expensive-opts / -dlcm=cg: noise.
- Theta 0.3 / 0.4 / 0.55: flat within 0.6% (in-process, 4 prompts each, two rounds).
- Positions > 1023 (local layers on the per-row attention path): K8 +11%, but only ~6% of eval tokens.
- Certified IMMA argmax (UN8=2, tests/c/imma/imma_test.cu for the fragment layout; xcheck token mode 2300 positions
  0 mismatches; 1-57 strips recomputed per pass): final phase 158 -> 108 us, but only with ph_final __noinline__,
  which costs 30-40 us elsewhere; inline it costs ~140 us of spills. Net ~0: off (code kept behind UN8=2).

## Row-count specialization (the code-generation finding, used)
- Compile-time K (KBV experiment: whole verify kernel for exactly K rows): K=2 3601 -> 3400, K=8 4807 -> 4688 us.
- One kernel with a switch to 8 __noinline__ bodies: bit-identical but 1.3 KB spills per body (call ABI): much slower.
  8 inlined bodies in one kernel: mixed (K1/K7/K8 -110..-130, K2 +40) and a 4m40s build.
- 8 libraries (KVONLY=k, built in parallel, 35 s): spills 376 B (generic 600 B); all eight launched per cycle (the
  draft picks K on the device), seven return at entry (~55 us per cycle in total). tv3: -53..-191 us at every K;
  served natural 930 -> 910, eval 1544 -> 1509. Submitted as s0008.
- Next possible: a CUDA graph SWITCH node (draft sets K) to drop the seven empty launches (~-55 us/cycle); needs
  device-side tags (graph parameters are fixed) and the cycle reordered as [draft -> verify].

## s0008 specialized verify kernels per K — 1318.3 (natural 834.4); expected ~1320 / ~830: confirmed.
## s0009 per-K compile switches (kflags.txt) — 1310.5 (natural 830.7); expected ~1308 / ~828: confirmed.
Sweeps (abk.sh, 8 libraries per variant, deltas vs plain): BF_UNR=2 -19..-31 at K>=2, W13X2 K1 -53 / K5 -28, OP_UNR_V=2
K3 -23; code-generation cliffs per K: FP8_UNR=4 K4 +959, BF_UNR=4 K1 +441, GATE_MFIRST=0 K5 +1286, VDES_MIN=4 K3 +630.
KVSPLIT / KV4_SH also lose inside specialized kernels. D_UNR, QKV_UNR, POLL_NEEDED, W2B_UNR_V: nothing.
## s0010 verify through a graph SWITCH node (device tags) — 1312.4 (natural 830.7); expected ~1304 / ~828: NOT
confirmed (in-process -0.4%, HTTP flat). Off by default (MK_GVERIFY=1 enables); the empty launches cost less in a
queued stream (~20-30 us per cycle) than isolated.

## after s0010 (measured; none kept)
- Certified IMMA argmax inside the specialized kernels: inline +140..+250 us; ph_final out of line -4..-26 at K>=2,
  +108 at K=1. Not worth the INT8 copy and the certification risk.
- Per-K perturbations (abk.sh with kflags as base): MKPROF mark sites +100..+190, ptxas -O2 noise, W13_UNR 9
  +50..+160, FP4_UNR 4/8 noise or worse, D_UNR / QKV_UNR / POLL_NEEDED / GATE_MFIRST / W2B_UNR_V: nothing.
- Draft: FP8 GEMV prefetch one unit ahead (G_PF8): eval +2.7% (register pressure at 2 blocks/SM); G_SBMAX=2 breaks
  the drafter (2303 us/token), SM_MINB=3 / SB_MID=2 nothing. nsys (eval prompt 1): DSpark span 440 us incl. ~28 us
  of empty verify launches; layers 59 us each, base unembedding GEMV 115 us (~4 TB/s), Markov ~50 us; n-gram cycles
  41 us. The IF node costs ~9 us on DSpark cycles (net loss ~0.2% on eval, net gain on natural).
- Markov top-M fold (idea, not built): the step's exp-sum over the whole vocabulary sets the draft length; net
  ~30 us per DSpark cycle at best.
- Attention merge: needs the max over all records first (no record-by-record finalisation).
## Parallel runs (read-only look at 05:xx): c2 (B=8 speculation) b8 6048 so far; c3 (acceptance length) 1416 /
natural 959 with 16-row n-gram verify passes (its main lever, -4.6% natural) built on c1's older kernel; c3 also
found the fused persistent draft a dead end (empty 46-kernel graph 42 us). The two directions combine: c3's 16-row
passes gain more the flatter V(K) is (its simulation: -12% natural at V(16) = 6.5 ms).
- Draft GEMV with a warp's units loaded up front (G_ALLLOAD, 2 or 4 strips per block; SB_MID capped at G_SBMAX):
  eval 1508 -> 1523..1535, natural 908 -> 914..919. Off. The draft GEMV designs are exhausted for this run.
- X2_SHAREB first version deadlocked (per-lane condition around mma.sync; fixed to __all_sync); the job held card 1
  until its 3600 s timeout. Lesson: per-lane predicates must never guard warp-collective MMAs.
- Robustness (tests/c/robust.py, soak.py on the s0009 engine): P=1/7/13, n=1/2, vocabulary-edge ids, repeat loop,
  long 1944-token generation, P=500 / P=1000 (positions past 1024), too-long-for-spec fallback: all identical to
  step-by-step greedy; soak 20 requests (B=1 and B=8 cohorts mixed): 0 differing.
- c3's 16-row library at its V(16) = 8.57 ms would give ~0.63 ms/token on n-gram cycles vs 0.65 for this engine's
  8-row cycles: no gain without porting this run's kernel work into it (a second large merge). Not done.
- X2_SHAREB (fixed, warp-uniform) with W13X2, contiguous or not: +180..+460 us per pass. Off.
- Last per-K sweep (QSEQ, LS_SEQ, PF_MIN=2, DREV_MAX=8, FIN_DES=9): nothing beyond noise except DREV_MAX=8 at K=5
  (-45 us; ~3 us per cycle weighted): not taken. Caution for A/B jobs: the first tv3 run of a job can be cold (here the
  base's nat0 run read K3 4751 vs ~3690 in every other build); compare per-prompt rows, not only the first one.
## s0011 final source re-evaluated — 1317.5 (natural 834.1) vs s0009 1310.5 / 830.7: equivalent code, evaluator noise.

## 16-row verify passes (from run s18-mini-b200-c3), integrated at ~5.3 agent-h
- c3's 16-row library (engine/c/mk16.cu: c1 s0013 lineage + one-pass 16-row MoE, its own scratch buffers, shared
  weights / caches / rings) had the w2 bank-conflict bug of s0003: padding its staged rows: V(16) on real text
  9517 / 9494 -> 8447 / 7859 us (nat0 / nat2, ctx 600); V(8) of c3's 8-row kernel 5958 / 5857 (this run's 4936 / 4701).
- Integrated: mk16 (+ padding), c3's draftk.cu / draft.py (16-row acceptance, n-gram 15-token copies in a copying
  regime, its draft-length rule with this run's verify-cost table, short copies) + this run's IF node re-applied,
  c3's api.py changes merged into c/eng.py, RING 32 for every library, 8-row kernels return when vpos[1] > 8.
- xcheck (mixed 1..16-row passes, 1000 positions): 0 differ. Served identical.
- In-process vs the s0011 engine, natural prompts 1-3 (the evaluator's natural_b1 = median of these three) and eval
  prompts, two rounds: natural median 836 -> 814 (mean 950 -> 902), eval mean 1507 -> 1496. Submitted as s0012.
- A full 3-way merge of this run's kernel into mk16 (38 conflicts) was not done: c3 re-indexed tagged buffers from
  MAXB to MAXR rows, and code this run added (no base lines) would keep 8-row strides silently. Instead the measured
  winners are ported by hand, each bit-checked: KB16 (rows a compile-time 16) first.
- Robustness with 16-row passes (robust.py, soak 20): all identical (the 1944-token case now takes the fallback: margin 16 rows x 5).
## s0012 16-row n-gram passes (c3's library + padding, c3's draft rules, IF node, own cost table) — 1260.0 us b1
(natural 810.2), b8 5081; tokens/event 3.04 / 6.69. Expected ~1302 / ~810: natural confirmed, b1 better than
predicted (my eval set of 3 prompts under-represents the copying regime of the evaluator's word-list prompts).
## s0013 16-row library: W2V16=0 + W13_UNR16=12 — 1253.0 (natural 801.2); expected ~1253 / ~797: confirmed.
- 16-row policy sweep (MK_NG16_ACC 5/6/7, MK_NG16_MIN 5/7): flat. mk16 knobs: BF_UNR 3 -1% (taken), PR16_UNR,
  MG16_UNR, PF_MIN, FP8_UNR: noise.
- Profile of the 16-row pass (crit16, plain app): qkv 17.6 (8-row kernel 7.8), o_proj 13.0 (6.8), router 9.6 (1.6),
  attention 23.5 (17), w13 38.8 (25.6), w2 25.8 (14.8); gated: gate_d 15.6 (6.7), gate_fg 12.2 (5.9). The dense
  GEMVs run in two 8-row halves (weights twice, two latencies): ~1.5 ms of the 8.1 ms pass.
