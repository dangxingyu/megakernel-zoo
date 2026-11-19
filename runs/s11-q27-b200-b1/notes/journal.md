# Journal of run s11-q27-b200-b1 (track B on the s9 s0022 source)

## hour 0-1: accounting
- Baseline, same card (card 1/2, tools/tv.py): track B 7678 / 8140 us (B=1 / B=8), track A 8456 / 8868; logits hash
  equal. Stamped per-op profile: notes/gap.md.
- Track B realises only ~half of its 25 % byte saving: 814 us faster than track A for 11.7 GB less (1.6 ms at 7.3 TB/s).
- bench/tma12.cu (new): isolated W12 2-quad loop streams 7.05 TB/s of real bytes at 36 quads/warp (BF16 6.99): the
  W12 format itself is not slower per byte in isolation. Per-warp TMA ring (cp.async.bulk into smem): 2 stages ~ ld
  loop, 3+ stages collapse to 3.3 TB/s (TMA queue limit?). Not pursued.
- tools/wtail.py (new, per-warp stream end): W12 ops end their stream ~3.7 us later than bytes/7.3 TB/s, BF16 ops
  0-0.7 us. Constant per op whatever its size: a start/fixed cost, not a rate.
- Ablations (stamped, B=1 pass 7755): no escape list/test 7601 (-154), constant row bases 7611 (-144), both 7329 (-426).
- Negative: barrier-time L2 prefetch of the next op's lists + row bases (W12_PFL): +133 us.
- tools/w12v2_census.py: per-column power-of-two shifts on top of per-row windows cut escapes only 6 % (1,922,136 ->
  1,801,273; lm_head -30 %). Escapes are the Gaussian tail of tiny weights. Not worth a format.
- tools/head_census.py: certified lm_head sketch (Cauchy-Schwarz bounds, real x from greedy text, 4608 positions):
  int8 per row 1.9 rows mean / p99 11 / max 70 candidates (45 tiles); int6/group128 9.2 / 120 / 1399; int5 and int4
  useless. int8 (1.27 GB vs W12 1.905 GB) is the format.

## hour 1-1.5: W12_RS (submission s0001)
- Rows stored as W[r] 2^(107 - lo_r): codes unchanged, decode base constant, epilogue multiplies by 2^(lo_r - 107).
- Expected (from the constant-base ablation): -144 us at most. Measured same card: B=1 7675 -> 7567, B=8 8128 -> 8020.
- Verified: MK_W12_VERIFY 0 bit mismatches against the scaled bits, 0 value mismatches (decoded x factor == checkpoint)
  over all 25.6e9 weights; bitcheck A vs A W12 1152 positions B=1, B=8: bit-identical, batch-invariant; served ==
  greedy (online_time.py) B=1 and B=8.
- Trap found: torch.pow(2.0, -9) in float64 on CUDA is 0.0019531249999999998, not 2^-9; my own check flagged 97,280
  "mismatches" until it used torch.ldexp. The factor table also uses ldexp now.
- s0001 (track B, W12_RS): predicted ~7560 / ~8005; VALID 7557 / 8003 (s0023 inherited: 7668 / 8114): -1.45 % / -1.37 %.
- Negative: a leaner escape slow path for quads with one escape (3 independent shuffles, predicated patch, no
  per-k-step shuffle loop, -DW12_ESC1): B=1 7590 -> 7645, B=8 8046 -> 8067 (3 rounds, card 1). As s5 K6 said: every
  change to the patch path changes how ptxas schedules the shared hot loop; removed from the source.
- Negative: W12 out_proj / o_proj at B=1 instead of the BF16 twins (MK_W12_DUAL=0) with RS: +15 us. Twins stay.

## hour 1.5-2.5: layer-0 table (s0002), lm_head sketch (in progress)
- L0_TABLE: layer 0's in_proj output for all 248,320 token ids (8.2 GB) made at load by the kernel's own in_proj op in
  32-column passes; every row checked against a one-column pass (MK_L0T_VERIFY): 0 mismatched rows. Same card:
  B=1 7560 -> 7546, B=8 8006 -> 7979 (predicted -23 at B=1 from the op's 24 us: the embed op's 32 KB copy per column
  costs part). bitcheck identical; served == greedy.
- s0002 (track B, L0_TABLE): predicted ~7542 / ~7976; VALID 7542 / 7969.
- Head sketch first build (-DHEAD_SK): greedy tokens through the sketch == exact argmax at 2304 / 2304 teacher-forced
  positions (31 with exact bf16 ties), but SKETCH 208 us (B=1) / 233 (B=8) and RESCORE 36 / 89 us against the exact
  W12 head's 272 / 277: no gain at B=1, +73 us at B=8. The rescore's candidate scan loaded tmax serially.
- Sketch v1 -> v2 (all timed on real text: generate()/served path on harness prompts; random tokens give flat logits
  and 10x more candidate tiles - tv.py is the wrong timer for this change):
  - rescore candidate scan: serial tmax loads 36 -> batched; then a per-block summary of upper bounds so only
    candidate blocks' tiles are scanned (every block reading all 15,520 tiles was ~38 MB of L2 traffic).
  - relay hand-overs: smem flag spin ~0.27 us per turn; named barriers impossible (15 ids, waiting warps would join an
    earlier hand-over); mbarrier per turn: job 17 -> ~13 us.
  - next quad's loads issued before the current quad's MMAs: sketch 208 -> 288 us (register pressure). Reverted.
  - producer-side x statistics (the last down epilogue publishes max|x| and sum x^2 per column; ||x - x^|| in closed
    form): sketch prologue 7.2 -> 4.1 us (B=1), 20.5 -> 12.0 (B=8).
  - Bug caught by a debug run: with the fallback threshold (-inf) padding tiles past the vocabulary compared equal and
    became jobs (illegal address). Fixed; the all-candidates path is exact (432/432, -DSK_DBG_ALLCAND).
  - Phase stamps (B=1, real text): sketch stream ends median warp 169 us, slowest block 191 (6.6 TB/s); rescore:
    summaries 4.1, scan 6.1, first job end ~19 us; 8 candidate blocks, 9 candidate tiles.
- s0003 submitted (sketch v2): served path same build B=1 7586 -> 7542, B=8 7958 -> 7927.
- s0003 (sketch v2): predicted ~7500 / ~7940; VALID 7525 / 7950. Miss of +25 us at B=1. The evaluator's in-process
  step_us (exact head, step()) went 7526 -> 7559 (+34 us) against s0002: the build itself got slower on every pass,
  and the sketch saved ~51 us on top. My A/B had toggled the sketch inside one binary (MK_HEAD_SK=0/1), which cannot
  see a code-generation cost of the code that HEAD_SK compiles in. Rule from now on: every A/B includes the previous
  submission's kernel (bench/variants/mk_sNNNN.cu).
- Negative: sketch stream two quads per iteration (SK_PAIR): 208 B of spills, B=1 +35 us.
- Upper bound measured: epilogue per-row constants (norm gains, RS factors) replaced by constants (EPI_DBG_CONST,
  wrong results): B=1 7534 -> 7478 (-57), B=8 7970 -> 7883 (-87). Prefetching them during the stream is a target.

## hour ~4: head kernel, epilogue constants
- s0004 (track B, sketch ops in their own launch; x statistics inside the sketch): predicted ~7485 / ~7935;
  VALID 7495 / 7936. In-process generate 7473. Two launches per greedy pass (single_launch milestone gone).
- Bisect of the HEAD_SK code-generation cost in the main kernel (exact path, tv.py): +59 us with everything; +41 without
  the epilogue statistics block; the two new switch cases alone +28; removing the ARGMAX change made it worse (+55).
  Code that never runs moves the main kernel by tens of us: hence the separate head kernel.
- Epilogue per-row constants (gains, RS factors) prefetched into shared memory: at op start by 8 warps (EPI_PF)
  B=1 +33 / B=8 -18; by warp 2 during the previous barrier (EPI_BPF) +70 / +35. Bound with constants -57 / -87.
  The barrier is the most code-sensitive place of the kernel. EPI_BPF removed; EPI_PF left behind its flag (off).
- Negatives (each same-card A/B against the current source): nvcc options sm_100a / ptxas -O2 / extra vectorization:
  identical timings. Decode-only main kernel (no NG=2/4 GEMM instances): +68 us B=1 (as s9 found: pruning code
  worsens the allocation). Whole out_proj prefetched by the idle blocks during the recurrent op (host table only):
  -6 to -9 us (out_proj at 6 quads per warp is not DRAM-bound). Norm weight hoisted before the partner poll in the
  recurrent op: +4. Head kernel shared memory 64 / 128 KB equal, 200 KB +62 us (L1 too small for the sketch stream).
- head_census.py with the engine's exact rule (3072 positions): int8/row 2.3 candidate tiles mean (p99 16, max 67);
  int7/g128 1.13 GB 3.3 (p99 31); int6/g64 0.99 GB 13.2 (p99 212, max 2965); int5 useless.
- Escape handling re-measured on the RS build: no list / no test (W12E_DBG_NOESC) B=1 7541 -> 7261 (-280), B=8
  7988 -> 7660 (-328); lists zeroed but loaded (MK_W12_DROP_ESC) -106 / -133. With the row-base loads gone, each
  warp's first escape-list load (a cold DRAM load issued behind the barrier's weight prefetch) is what the first
  iteration waits on: ~174 us per B=1 pass.
- Which part of the escape cost? Lists L2-hot (every block reads block 0's windows, W12E_DBG_HOTLIST): only -20 us;
  hot and empty: -121; no escape code: -280. So ~160 us was the escape *test* in the hot 2-quad loop (a shuffle of
  the list register per iteration, the branch on it), not the patches or the list's latency.
- W12E_NQREG: the next escape's quad read once into a register and refreshed only after a slow-path quad.
  B=1 7540 -> 7399 (-141), B=8 7998 -> 7820 (-178) same card; vs the s0004 build on real text 7479 -> 7351 / 7955 ->
  7782. Bit-identical (bitcheck), served == greedy, sketch exact, 20 x 128-step determinism.
- s0005 (track B, NQREG): predicted ~7370 / ~7765; VALID 7337 / 7763. In-process generate 7315.
- Negative: the list window in shared memory (W12E_SMLIST: broadcast LDS instead of shuffles, list registers dead in
  the fast loop): +120 us B=1. Removed.

## hour ~5 (agent-hour ~3): main-kernel variants after s0005 (each 3 rounds, same card, against the s0005 source)
- Escape slow path out of line (__noinline__, W12E_SLOWCALL): +334 us B=1 (the call saves/restores through local
  memory). One-escape slow path (W12_ESC1) again: +43. Lists emptied (DROP_ESC) on the NQREG build: -114 B=1 /
  -150 B=8: the remaining escape cost is the patch path itself.
- Epilogue constants loaded by each warp at its own stream end into shared memory (EPI_SE): +66 / +80. Third form
  to lose; the -57 us bound is not reachable by adding code to the stream or the barrier. Removed.
- x loads as explicit ld.global (XLD_GLOBAL): -6 / -4 (kept as an option, not submitted alone).
- Two k-steps of x per 16-byte load (X_C2, with the BF16 twin loop and the rescore taught the layout): bit-identical
  but +144 / +14.
- Pattern of this run: the main-kernel changes that paid removed work from a hot path (RS: per-tile loads; NQREG: a
  shuffle per iteration; L0 table: an op); every change that added code - even code that runs once per op or never -
  lost 30-330 us through code generation.
- W12_OR (window base 112 so the decode is an OR of disjoint fields, one instruction fewer per register; format
  exact): +140 us B=1. Several variants of this hour land at ~7540 = the pre-NQREG level (X_C2 7544, W12_OR 7540,
  SMLIST 7530): the main kernel seems to have two code-generation "modes" ~140 us apart, and NQREG's gain may partly
  be landing in the good one. SASS around the W12 loads is nearly identical between a 7400 and a 7540 build (register
  move forms only); the cause is not visible in the hot loop. Not pursued further.
- nsys on generate (real text, B=1): head kernel 217 us per pass (sketch ~197 + rescore ~15 + argmax/finish).
- Rescore in the owner block (SK_OWN: tile maxima kept in shared memory from the sketch, no global scan; one warp
  runs the exact chain from a TMA ring of 7 x 16 KB): exact (2304/2304), rescore op 15.5 -> ~9.5 us in the phase
  profile, but generate on real text B=1 7336 -> 7340, B=8 7768 -> 7763: no gain. The chain is bound by the MMA
  dependency (1212 cycles per 32 MMAs = 38 cycles each, the data waits only 91 cycles); splitting the loop at the
  piece boundary or plain shared loads made it slower (1572-1706). Left behind -DSK_OWN (off).

## hour ~6.5 (agent-hour ~4.3): where a B=1 pass spends its warps (ncu source counters, one pass, s0005 build)
- Warp-stall samples by source line (tools/ncu_src3.sh, ncu_lines2.py): ~55 % in the W12 fast loop waiting for its
  weights (first decode instruction after the loads), 10.5 % at the main loop's head (warps parked after the grid
  barrier's block sync), 4.3 % in thread 32's barrier prefetch loop, 4.5 % at the block sync after the stream
  (in-block stragglers), 3.2 % BF16 twin loop, the rest spread (lmix, attention, epilogues).
- A fast (s0005, 7400) and a slow (W12_OR, 7540) build: total samples +2 %; the slow build's samples move from the
  load-use instruction into the decode instructions (PRMT / LOP3 ~2x): its loop schedule is worse, the instruction
  count is the same (the OR constant became a LOP3 in place of a VIADD).
- Loop-order variants of the 2-quad fast path: the 8 x loads first (W12V1) +68 us; next decode before the MMA
  (W12V3) identical code. The hot loop is at a local optimum for source-level reordering.
- Exact zeros in GEMM inputs (tools/zero_census.py, 40 real steps): 0.05-0.08 % of elements, no all-zero k-step:
  nothing a stream could skip.
- nsys (real text, B=1): main kernel 7104 us, head kernel 216 us, gap between them 0.26 us (PDL would gain nothing).
- step() / generate() now reject batches larger than max_batch (host-side check; submit() already validates ids and
  lengths).

## hour ~7.5 (agent-hour ~5): a measurement trap, re-measurements, s0006
- The archived s0005 source read 7483 us against 7393 for the current build in one job; source and Python were
  ruled out (identical SASS; the same .so gives identical times through either api copy). Found on the way:
  -Xptxas -v (build.sh) changes the SASS, but not the timing (7336 / 7336). A later job read 7337 / 7337 for the same
  pair: the level of a whole job can be off by ~50-90 us. Only alternating comparisons inside one job count
  (findings F1). The A/B tools now build exactly as build.sh (with -v).
- Re-measured with build.sh flags (3 rounds, B=1 / B=8 vs base 7394 / 7810): W12_OR +140, X_C2 +144, W12V1 +70 (the
  verdicts hold); EPI_PF -13 / -8 (it had been +33 in the earlier A/B without HEAD_SK / -v); XLD_GLOBAL -6 alone, but
  EPI_PF + XLD_GLOBAL only -7; EPI_PF + no BF16 twins -23 / -3.
- Unsplit recurrent op at B=1 (flag 65536): +71 us. Twins replaced by W12 at B=1 (MK_W12_DUAL=0): -10 / -11 alone.
- s0006 submitted (track B, EPI_PF).
- s0006 (track B, EPI_PF): predicted ~7325 / ~7755; VALID 7327 / 7763.
- s0007 (track B, no BF16 twins, no BF16 stream call sites): predicted ~7312 / ~7763; VALID 7304 / 7741 (-23 / -22).
- On the s0007 base (build.sh flags, 3 rounds, a noisy job: base 7379-7400): one-escape slow path (ESC1) and
  explicit global x loads (XLD_GLOBAL) within the noise; own-block rescoring (SK_OWN, real text) +3 / +8. Dropped.

## agent-hour ~5.2: sketch tile balance (s0008), 7-bit sketch census
- Per-block sketch stream end (SK_PROF, 24 passes): spread max - median 11.5 us per pass, per-block lag correlation
  between passes 0-11 and 12-23 0.93: the same blocks are slow every pass. Tile ranges per block now come from a
  table (op.p7), each block writes its stream time (op.p8); at load 2 x 12 greedy passes, tiles proportional to
  measured speed (94-106 tiles per block instead of 102; the integer sums make any split exact).
  Same job alternating (generate, real text): balance off 7310 / 7740, 7309 / 7726; on 7304 / 7731, 7304 / 7727:
  -5 / -4 us (predicted -10). Sketch exact (1728 positions, 18 ties), bitcheck identical (1152 x B=1, B=8),
  served == offline. s0008 submitted.
- head_census (3072 positions, engine's exact bound): int8 per row 2.3 tiles mean (p99 16); int7 per row 6.9
  (p99 78, max 821), 1.113 GB; int6 per row 264 (p99 4667), per 1024 53, per 256 26 (p99 417): int6 is out,
  int7 per row is the candidate (-0.159 GB, ~-22 us if the stream rate holds).
- s0008 (track B, sketch tile table balanced by measured block speed): predicted ~7299 / ~7737; VALID 7313 / 7738
  (s0007 7304 / 7741): no gain beyond noise; in-process generate 7285 as s0007. Third submission in a row < 0.5 %:
  plan rewrite. Gates: numeric_long max rel-L2 0.457 against a limit of 0.592 (the tail that s5's rounding change
  re-rolled to 0.713): every change keeps the logits bit-identical to track A.
- No escape code at all (W12E_DBG_NOESC, wrong results; abW 3 rounds, card 3): B=1 7365 -> 7187, B=8 7805 -> 7631:
  the escape mechanism costs ~178 us per pass (check + list ~64, slow path ~114).
- Escape slow path, bit-identical variants (abW, build.sh flags, same job, logits hash unchanged):
  W12E_P1 (a pair with one escape: entry, value and next entry in one shuffle round, then a copy of the plain
  8-k-step body with the patch at the escape's k-step): +375 / +455 us (first form spilled 8 B; without spills
  still +375). The outer loop body grew from 32 to 40 HMMA (the body duplicated); the loss is far larger than any
  instruction count: code size / placement of the hot loop (instruction cache) is the likely mechanism.
  W12E_P2 (the same through the plain body, a warp-uniform k-step test + owner-lane patch per k-step): +1640 us.
  The 12-bit fast loop does not tolerate either; escape work stops here (left behind -DW12E_P1 / -DW12E_P2, off).
- 7-bit sketch: SK7 (7 loads, then 4 groups) spills 152 B in sketch_work, SK7S (loads 4-6 after two groups) 84 B
  (the 8-bit sketch: 72 B). generate, real text, 3 rounds alternating: s8 7291-7295 / 7719-7723, SK7 7275-7279 /
  7706-7717, SK7S 7274 / 7700-7702: SK7S -18..-21 / -19..-23 us.
- s0009 submitted (track B, 7-bit sketch SK7 + SK7S).
- Sketch phase profile (SK_PROF, SK7S build, B=1): prologue 4.6 us; stream end per warp median 154, block max median
  162, max 168 (warp 0 ends at ~115: the barrier's 144 KB L2 prefetch covers ~63 % of its share); rescore op:
  summaries 2.0 us, scan to 6.9, first job end ~18.7 (30 jobs on 22 blocks, max 1 per block).
- L2 persisting window on the first 128 MB of the sketch for the head launches (host-side launch attribute only,
  MK_L2W = carve-out MB; 2 rounds, same job): 0: 7272 / 7698; 32 MB: 7275 / 7710; 64 MB: 7365 / 8035; 96 MB:
  launch fails (carve-out above the device maximum). The persisting lines cost the main kernel more than the sketch
  gains. Dropped (left behind MK_L2W, off).
- s0009 (track B, 7-bit sketch): predicted ~7293 / ~7718; VALID 7291 / 7712 (new best; in process 7266).
- In-sketch rescoring at B=1 (SK_SPEC: each block rescores its own candidates after its sketch against the best lower
  bound known then; more than SPEC_MAX candidates or no positive bound: left to the rescore op): exact (4608/4608),
  but B=1 +13..+15 us (3 rounds; SPEC_MAX 1 or 2 alike; 64 KB with a 2-slot ring the same; the head kernel's shared
  memory size alone 64 vs 132 KB: no effect). Dropped (left behind -DSK_SPEC, off).
- Prefetch piece sweep (env only, card 3, 2 rounds): near 112 KB +45, 144 KB (current) 7273-7289 / 7692-7698,
  176 KB 7256-7257 / 7654-7672, 208 KB 7275 / 7678; far 32 / 96 KB = 64 KB. Finer sweep on two cards running.
- Prefetch near 176 KB re-checked with sub_check (sketch exact 4608, bitcheck identical, served == offline); s0010
  submitted. Other prefetch knobs on top (card 3, 2 rounds; the job's level moved +30 between rounds): idle blocks
  16 / 48, attention far piece 256 KB, far 48 / 80 KB: all within noise of the base. Left as they are.
- Dynamic units in the sketch (SK_DYN: warps take 8-k-step units from a block counter, integer sums added into a
  ring of 8 tile slots in shared memory, the last unit's warp finalizes): exact (4608/4608), spills 84 -> 8 B, but
  B=1 +33..+45 us, B=8 +7..+33 (3 rounds). The per-unit overhead (counter, 8 shared atomics per lane, two block
  fences, warp syncs) costs more than the 8-14 us spread inside a block. Dropped (left behind -DSK_DYN, off).
- s0010 (track B, prefetch near 176 KB): predicted ~7277 / ~7687; VALID 7273 / 7683 (new best; in process 7250).
- Escape census by row order (tools/esc_perm_census.py, all 256 matrices, exact masks): quads holding an escape
  7.31 % in the natural row order, 2.28 % with rows sorted by escape count (pairs 14.07 % -> 3.84 %): the escapes sit
  in few rows. A row permutation would cut the slow path ~3.5x (est. -80 us), but every tile is cut between warps
  at tile-specific k positions (shares are shorter than a tile), so a moved row gets other split points: the
  logits would change rounding (gate risk of numeric_long, cf. s5). Not done; noted in the plan.
- BAR_SLEEP_NS sweep (3 rounds, card 1): 25 = 50 = 100 ns (7270-7273 at B=1), 0: +38 us. Kept 25.
- Norm-only exclusion of head tiles (|z_i| <= ||w_i|| ||x||; tools/norm_census.py, 768 real positions): 0 tiles
  excluded at any position (the top logit needs cos 0.19; row norms 0.34-1.69, ||x|| ~880). Dead end.
- Build options (numerics unchanged, logits hash identical, 3 rounds, card 1): -arch=sm_100a 7269-7271, ptxas
  --allow-expensive-optimizations 7271-7273, base 7270-7271 at B=1; -Xptxas -dlcm=cg +207 us. Nothing there.
- Rescore scan: it reads the candidate blocks' tile maxima in rounds of 8 blocks (one L2 round trip each; 22-24
  candidate blocks = 3 rounds). SK_SCAN1: all in one round (prefix sums of the blocks' tile counts, up to 4 tiles
  per thread, loads issued together). Testing.
- Per-process level noise: in some jobs a build's generate() time flips between two levels ~25 us apart from one
  process to the next (base 7258 / 7284 / 7285 in one job); the sketch tile balance is not the cause (bal0 = bal1,
  5 alternations, 7256-7259, another job). Small head changes are therefore measured as the head kernel's own GPU
  time: tools/head_nsys.sh (nsys, generate 300 steps after a 64-token prompt, median over 324 launches).
- Head kernel median, B=1 / B=8 (two runs each): base 194.0-194.3 / 208.6-209.6; SK_SCAN1 198.5-199.1 / 213.3-213.9
  (worse; its build also moves sketch_work's spills 84 -> 96 B); SK_OWN 198.4-198.7 / 213.7-214.2 (avg 220: outliers);
  SK_FOLD (argmax op folded into the finish op: each finishing block unpacks its own column; one grid barrier less)
  192.2 / 206.7-207.8: -2 us, exact (4608/4608), served == offline.
- L2 bulk prefetch of the warp's unit 2 / 4 ahead in the sketch (SK_PF2): head kernel 193-195 -> 232-234 / 256-261 us
  at B=1 (B=8 +42 / +77): the sketch stream is at bandwidth; extra requests only compete. Dropped.
- s0011 (track B, argmax folded into the head kernel's finish op): predicted ~-2 us, inside the noise (~7271 /
  ~7681); VALID 7275 / 7697 (s0010 7273 / 7683; in process 7250 both): no visible change, as predicted.
- Rescore relay with 8 k-steps per turn (SK_KPT=8: 40 turns instead of 64; 28 B spill): head kernel median
  192.0-192.5 -> 188.6-189.1 us (B=1), 207.4-208.0 -> 204.5-205.0 (B=8), exact 4608/4608. KPT=10 (32 turns): 136 B
  spill, 195.8-196.0 (worse).
- s0012 (track B, rescoring relay 40 turns of 8 k-steps): predicted ~-3 us inside the noise (~7271 / ~7692); VALID
  7266 / 7682 (new best on the board; in process 7259 vs 7250 for s0011: the difference is noise either way).
- Gate study on 64 own sequences (tools/own_ref64.py, 8 groups of 8, positions 128-1144 every 8th, own fp32 run):
  medians HF BF16 0.0161, engine 0.0159, perturbed 0.0159 / 0.0163; per-group maxima 0.74-0.95 for all four (engine /
  HF ratio 1.0-1.15, never near 2): these sequences are harder than the evaluator's (its HF max is 0.296), so they
  do not reproduce the regime where the evaluator's draws were 0.457 and 0.713. No basis for taking the gamble.
- HF emulation with the residual stream in float32 (own 64 sequences): median rel-L2 0.0161 -> 0.0087, but the
  per-group maxima stay 0.66-0.93 (HF BF16 0.74-0.91): the tail positions are not made by the residual's rounding.
  A more accurate engine would not buy margin on the long-context max. Rounding changes stay out for this run.
- Per-matrix near prefetch (MK_PF_NEAR_MAP, env only; 3 rounds per card): against 176 KB everywhere, gate_up 128 /
  256 KB +6..+9 us B=1, down 128 / 256 KB +14..+25, in_proj 128 / 256 KB +2..+20, out_proj / o_proj 128 / 256 KB
  0..+29 (noisy job): the uniform 176 KB is the optimum for every kind.
- Near prefetch of the sketch alone (MK_PF_NEAR_MAP=15520x20:..., 2 cards x 3 rounds): 0 / 64 / 176 / 512 KB within
  +-3 us of each other (stable card: 7257 / 7255 / 7253 / 7253). The sketch's first piece does not matter.
- Sketch prologue phases (SK_PROF, B=1): x loaded + reduced 2.30 us, digits stored 4.35; B=8: 5.63 / 14.85.
- Recurrent op -> out_proj dataflow (out_proj warps waiting on the ~3-4 heads their k-range needs instead of the grid
  barrier): not built. The warps that depend on the last heads stay on the critical path and a flag write + poll costs
  about the barrier's release; the gain would be the difference (~0.3-0.6 us x 48), and s4's per-tile dataflow lost.
- Sketch digits with a power-of-two x scale (SK_P2X: sx = smallest power of two >= max|x| / 16256, so x / sx is an
  exact multiply and the 40 IEEE divisions per thread at B=8 go away; the bound holds with |x/sx - q| <= 1/2 exactly,
  the x-quantization term at most 2x larger, negligible next to the 7-bit weight term). Testing.
  SK_P2X: exact (4608/4608), head kernel 188.8-189.3 -> 187.7-187.9 us (B=1), 204.2-204.4 -> 201.6-201.7 (B=8).
- s0013 (track B, power-of-two x scale in the sketch): predicted ~-1 / -3 us inside the noise (~7265 / ~7679);
  VALID 7263 / 7675 (new best on the board; in process 7248).
- SK_SCAN2 (16 candidate blocks per scan round, 2 tiles per thread): exact, head kernel +0.7..+1.2 us. Dropped: the
  rescore's scan phase is not its round trips.
- SK_XPF (next entry's x requested before this entry's digits): sketch_work spills 96 -> 196 B, head kernel +2..+4 us
  (B=1), +5.5 (B=8). Dropped: the sketch kernel is register-bound.
- Main-kernel time measured precisely: tools/main_nsys.sh (median of the decode passes' main-kernel launches): two runs
  of the s0013 build 7062.4 / 7063.0 us at B=1.
- Env knobs re-checked by nsys (main + head median, 2 runs each; card 1 runs ~16 us above card 2 in absolute level):
  near 160 KB = 176 KB, 192 KB +8; far 32 KB +4, 96 KB = 64 KB; idle-block prefetch 16 +9, 48 +0.5; attention far piece
  256 KB -0.3; MK_PF_CUM=0 +3.5. Every current default is at its optimum within ~0.5 us.
- Attention redesign (flash-decoding over all 152 blocks, ~7.5 us instead of ~17-19 per layer on paper): not taken -
  any other chunking of the keys changes the softmax / PV summation order (rounding, F12).
- Robustness: sketch path exact at B = 2, 3, 5, 7 too (3264 positions, 27 exact ties; tools/sk_check.py BS=...);
  submit/poll == offline greedy at B = 1 and 8 (tools/test_online.py).
- W12E_NOINL (the escape quad's k-step loop out of line, struct return): bit-identical, 12 B spill, main kernel
  7062 -> 7281 us (B=1), 7467 -> 7724 (B=8). The hot loop's code may not change in any way (F10). Dropped.
- MK_PF_LMIX_ALL=1 / MK_PF_ATT_K=1 at B=1 (nsys, context <= 364): within +-1.5 us of the defaults.
- Attention barrier prefetch at long contexts (nsys, 1024 greedy steps after a 64-token prompt, B=1, 3 runs each): the
  barrier before the one-op attention layer asked L2 for the blocks' whole o_proj share at B <= 3: 7084.1-7084.7 us;
  256 KB 7081.6-7083.2; 128 KB 7092; 512 KB 7083.5-7084.8; 256 KB + the attention blocks' K chunk (as at B >= 4)
  7080.8-7081.5. Made the 12-bit build's default at B <= 3 (track A unchanged). Confirmation runs: one level-shifted
  process on each side (7096.7 new, 7106.9 old); medians over all runs 7081.5 vs 7084.7.
- s0014 (track B, attention barrier prefetch at B <= 3: 256 KB + K chunk): predicted ~-3 us inside the noise (~7260);
  VALID 7261 / 7674 (new best; in process 7246).
- K chunks per attention unit asked at that barrier (MK_PF_ATT_CH, long contexts, 3 runs): 3: 7080.3-7081.5,
  5 (default): 7080.7-7081.3, 8: 7080.8-7081.9, 12: 7078.2-7079.3.
- MK_PF_ATT_CH at long contexts (card 3, 3 runs each): 16: 7074.9-7076.2, 20: 7073.0-7074.7, 24: 7076.2-7077.2,
  28: 7080.6-7081.6, 38: 7094-7095 (5: 7080.7-7081.3). Chunks are 256 keys, so chunks >= 5 hold no keys at these
  contexts; the blocks that ask for them skip their o_proj piece at that barrier. Not the mechanism alone: a 16 KB
  o_proj piece for every block is +18 us (with 5 or 20 chunks), and adding the whole o_proj share before o_proj does
  not help either. Empirical; checking on other cards.
- MK_PF_ATT_CH=20 vs 5 on other cards (3 runs each, long contexts): card 1 7085.7-7087.1 vs 7093.0-7094.9; card 2
  7073.4-7074.2 (+ one level-shifted 7088.1) vs 7081.8-7083.9; short contexts (card 3) 7059.2-7059.9 vs 7060.9-7062.6.
  Made the 12-bit build's default at B <= 3.
- s0015 (track B, 20 K chunks per attention unit at B <= 3): predicted ~-5 us (~7256); VALID 7258 / 7688 (new best;
  in process 7240; B=8 path unchanged by construction, 7688 vs 7674 is noise).
- Long-context re-sweep on top of 20 K chunks (3 runs each): card 1: near 160 KB = 176, 192 +9..+45 (level shifts);
  far 48 / 96 KB = 64. Card 3: attention o_proj piece 320 KB (= the whole share) 7070.8-7071.7 vs 256 KB 7073.3-7074.1,
  192 KB +9; idle-block prefetch 40 blocks 7071.4-7072.8, 24 blocks +3.5. The attention-barrier knobs interact.
- Grid of the attention-barrier knobs (whole o_proj share, idle blocks 40, chunks 16 / 24; cards 2 and 3, 3 runs):
  both jobs ran with level shifts of up to ~20 us between processes (card 3 base 7092-7102, an hour earlier 7073);
  the differences (+-2-3 us) are inside them. Stopped tuning; defaults stay (20 chunks, 256 KB, idle 32).
- Served path vs generate (nsys, B=1, prompt 64 + 1024 tokens): per pass 7270.4 vs 7263.6 us (main 7077.2 vs 7073.8,
  head 188.45 vs 187.52, gaps 0.26 us both): ~4-7 us for the served passes' host-mapped token write and process level.
  Host-mapped output is still the better choice (MK_HOSTMAP=0: +10 us). HTTP (7258) ~= the served path in process.
- Mechanism of the 20 K chunks (stamped build, per-op times, step() at position 900, B=1, 2 runs): the attention op
  17.8-17.9 us with 20 chunks vs 18.2 with 5 (x 16 layers ~ -6 us); o_proj and every other op unchanged. At the
  barrier before the attention op, blocks of a unit beyond its working chunks used to ask for 256 KB of o_proj each
  (5 chunks: 132 blocks, ~33 MB in flight while the latency-bound attention op runs); with 20, 60 of them ask for a
  128 KB K chunk past the context instead (~26 MB in total). Less traffic during the attention op is what helps;
  all-K (38 chunks) leaves too much of o_proj cold (+14), and 16 KB of o_proj per block likewise (+18).
- Tooling bug found: jobs on different cards share the pod's /tmp; env_nsys / main_nsys / head_nsys / var_gen used
  fixed /tmp names, so parallel jobs with equal variant names could read each other's profiles (the "noisy" parallel
  grids at hour ~10 were affected; the sequential single-card results and the HTTP results are not). Names are
  per card now ($MK_CARD).
- Attention-barrier traffic, clean single-card runs (nsys, long contexts, 3 runs each, card 3): s0015 defaults
  7073.8-7074.9 us; no prefetch for K chunks >= 5 (MK_PF_ATT_KMAX=5) 7069.4-7070.6; + the whole o_proj share for the
  other blocks 7064.7-7065.5; + 40 idle-block prefetches before the recurrent op 7063.0-7063.8 (each alone: whole
  share 7070.8-7071.5, idle 40 7072.0-7072.4). Confirmed: card 2 long 7062.2-7062.9 vs 7073.2-7074.0; card 1 short
  7059.1-7059.9 vs 7069.6-7071.8. Made the 12-bit build's default at B <= 3.
- s0016 (track B, B<=3 prefetch table retune: no K past chunk 5, whole o_proj share, 40 idle blocks): predicted ~-11
  us (~7247); VALID 7249 / 7682 (new best; in process 7234).
- Around the s0016 defaults (card 3, long contexts, 3 runs each; base 7061.8-7063.0): idle 48 = base, 56 +2.5; K
  chunks 12 / 16 / 24 / 28: +10 / +3.5 / +3.5 / +10; KMAX 4 = 5; near 160 KB = 176; far 48 KB = 64, 96 KB +4.5;
  idle bytes 256 KB +5. The defaults are a local optimum.
- Near piece per kind around 176 KB (card 3, long contexts, 3 runs; base 7062.2-7062.9): down 160 +6, 192
  7060.9-7062.6; gate_up 160 7061.0-7061.5, 192 +4.5; in_proj 160 = base, 192 +4.5.
- gate_up 160 + down 192 KB near pieces together (one job at a level +32 us): -1.3 us; 144/192 -0.5; 160/208 +2.
  Too small to submit; left at 176 KB.
- B=8 prefetch sweep (card 2, long contexts, 3 runs): the main-kernel median scatters 7491-7537 us within one
  variant (B=8 passes vary much more than B=1); attention far piece whole / 256 KB, far 32 / 96 KB, no K chunk: no
  difference outside that scatter. B=8 left as it is.
- Next pass's weights asked from L2 during the head's rescoring (SK_NEXTPF, head kernel only; nsys, long contexts,
  3 runs): layer 0's out_proj (47 MB): main unchanged, head +2 (out_proj is already prefetched by the idle blocks);
  each block's first 512 KB / 1 MB of layer 0's gate_up share: main -1.6 / -2.5 us, head +3.3 / +6. Net loss: the
  rescoring's latency chain suffers more from the traffic than gate_up gains. Dropped (left behind -DSK_NEXTPF, off).
- BAR_SLEEP_NS 12 / 25 / 40 (nsys, long contexts, 3 runs): 7062.8-7063.6 / 7062.7-7063.2 / 7062.6-7063.3: no
  difference. Kept 25.

## Close (agent-hour ~12)
- Final engine = s0016 (track B, 7249 / 7682 HTTP; logits bit-identical to track A; sketch exact). Re-verified after
  the last (inactive) code additions: sub_check (sketch 4608/4608 exact, bitcheck identical at B=1 and B=8,
  served == offline). Track A builds from the same source (engine/TRACK = A).

## agent-hour ~12-13: mixed 6 / 7-bit head sketch (SK76)
- tools/mixed_census.py (own prompts, 4608-6144 positions, cross-validated halves, the engine's bound rule): tiles
  stored in 6 bits unless they were candidates at > K positions under 6 bits. Half-A K = 3 / 10 / 20 / 30: 52 / 74 /
  85 / 91 % of tiles in 6 bits, 1.030 / 0.995 / 0.977 / 0.968 GB (7-bit: 1.113), candidates on half B mean 8.5 /
  14.1 / 21.8 / 28.9 (7-bit 7.3), extra rescoring rounds ~0.01-0.13 per pass. Weight-only selections (smallest row
  norms / max |w|) are much worse (p99 400-600 candidates): the set is data-derived (engine/sk6_tiles.bin, 1940 B).
- SK76 (7-bit slots first, then 6-bit ones, slot -> tile map; u = q + 32 for 6 bits, 6 x 16 B per lane and unit;
  byte-balanced tile table; calibration off): exact (sketch_check 4608/4608 at B = 1, 8, 3). nsys (long contexts, 2-3
  runs): head kernel 188.0 -> 182.3-182.7 us at B=1, unchanged at B=8 (201.5-202.8). (Main-kernel medians differ by
  ~2 us only because the build without calibration has 24 fewer short-context passes in the median.)
  SK_PROF: the sketch stream ends ~14 us earlier, but candidate blocks 22 -> 49, jobs 30 -> 86, and the rescore op's
  scan grows 6.9 -> 15.6 us (first job end 15.5 -> 25 us). SCAN1 / SCAN2 on top: head 184.5-186.8 (worse).
- SK76 tuning (nsys head kernel, long contexts): the slot table must balance tiles, not bytes (6-bit slots weighted 7
  like 7-bit ones: K40 set 182.3 -> 180.4 us; weights 6.5 / 7.5 worse): a unit costs a warp about the same either way.
  6-bit share: 54 % (K6) 185-187, 74 % (K20) 186, 85 % (K40) 180.4-181.0, 94 % (K80) 178.8-179.7, 98 % (K160)
  179.8-180.2 (all-7-bit 188). Chosen: K80 (93.9 % of tiles in 6 bits). Exact: 8192/8192 positions (79 ties) at
  B = 1, 8, 2, 5; served == offline.
- Same job, equal pass sets (calibration off both): B=1 sum 7246.1-7246.8 vs 7252.8-7254.2 (head 179.0-180.6 vs
  188.9-189.4; the main kernel +2 us: its last barrier's prefetch of the sketch's first piece - without it main is
  equal but the head +3; the main kernel's SASS is identical); B=8 7693.7-7700.4 vs 7690.3-7704.1.
- s0017 (track B, SK76: 93.9 % of head-sketch tiles in 6 bits): predicted ~7242 / ~7679; VALID 7244 / 7684 (new best
  at B=1; in process 7225). B=8 did not gain: head kernel at B=8 median 198.8 but mean 210.4 (p99 316, max 2045 us)
  vs 7-bit mean 203.5: passes with hundreds of candidates (x 8 columns) make the rescoring scan and chains long
  (SK_PROF: a pass with 393 jobs on 113 candidate blocks, scan 30 us, 3 jobs per block). B=1: mean 181.1, median 179.6.
- Dual sketch (MK_SK76_B1ONLY=1, default): one-sequence passes use the mixed 6 / 7-bit sketch, passes of B > 1 an
  all-7-bit copy (second set of tensors, +3.6 GB) through a plain instance of the kernel (sketch_work<false>: 96 B of
  spills as before SK76; the mixed instance 136 B). First form chose by program name (the "_b1" programs no longer
  exist since s0007: everything used the 7-bit copy, and that through the SK76 code path: B=8 mean 211). Fixed: chosen
  by B at launch ("_sk7" programs). nsys, same job: B=8 head mean 206.0-206.4 (p99 212-213) vs s0017 209.4-209.8 (p99
  316); B=1 unchanged (181.4-182.5 vs 181.4-182.4). Exact (4608/4608).
- s0018 (track B, B>1 passes on an all-7-bit sketch copy): predicted B=8 ~-4 (~7680), B=1 ~7244; VALID 7239 / 7677
  (new best at both; in process 7220).
- SCAN1 / SCAN2 re-tested on the dual build (nsys, 3 runs): SCAN2 = current (182.6-183.7 vs 182.1-182.7 us), SCAN1
  +2. Left off.
- A 5-bit tier for the coldest tiles (TIER5 census, 6144 positions): no tile stays below 20 5-bit candidate positions
  (0.1 %). Dead end.
- Head phases (SK_PROF hk_prof.py, B=1, 256 passes; time from head-kernel entry): sketch op end (last block) 157.4 us,
  after its barrier 158.5, rescore end 175.9 (mean 183.2: tail passes), finish 179.2. The sketch op carries ~24 us over
  its bytes (0.96 GB at 7.2 TB/s ~ 133 us): prologue ~5, the rest stream imbalance. Rescore ~17 us. Per pass (512
  passes): candidate blocks median 2 / mean 14.8 / p90 48 / max 151; jobs median 3 / mean 61 / p90 84 / max 3066;
  scan end median 4.6, mean 7.5, p90 15 us.
- Warp 0 taking SK_W0X more units (its start is L2-prefetched): 16 = current, 32 +14 us, 48 +33 (B=1 head). Dropped.
- Sketch near piece on the mixed layout (nsys, 2 runs): 0 / 64 / 176 (default) / 352 KB: head 186.0 / 184.6 / 182.1 /
  181.7 us, main equal within 1.5: the default stays.
- L2 prefetch of each sketch warp's first 1 / 2 / 3 units at the head kernel's start, during the prologue (SK_PFW):
  head +4 us at B=1 (B=8 = ); every prefetch issued from inside the head kernel has lost (F14). Dropped.
- Load-time tile calibration with the mixed sketch (MK_SK_BALANCE=1; its tile table is the B=1 sketch's): steady-state
  head mean (last 1000 passes, 3 runs) 180.0-180.6 vs 182.8-183.1 us without (median 179.0-179.6 vs 181.7-182.2). On the
  uniform 7-bit sketch it gained nothing (s0008); on the mixed one the blocks differ in their 6 / 7-bit mix. On again.
- s0019 (track B, load-time calibration back on for the mixed sketch): predicted ~-2.5 us (~7236); VALID 7245 / 7680
  (s0018 7239 / 7677) and in process 7251 vs 7220: the evaluator's process got a worse tile table than my three nsys
  runs (calibration timings are per process). Reverted (off with SK76). Lesson: a load-time measurement can land
  badly in the one process that is scored.
- 6-bit set with calibration (3 runs): K40 / K80 / K160 head mean 181.1-181.5 / 180.1-181.3 / 180.4-180.6 (K40 has the
  smaller tail, p99 190 vs 203-222): equal on the mean; K80 stays.
- SK_LAST (the rescoring's last block finishes the pass via an atomic counter; no barrier after the rescoring, none
  after the finish): exact, served == offline, head 182.4-182.75 vs 182.2-182.7 us: no gain (the kernel end waits for
  the slowest block anyway). Left off.
- SK_XS (B > 1: the x entries of the sketch prologue staged in shared memory instead of loaded twice): exact; head at
  B=8 205.6-205.7 vs 205.7 us: no gain. Left off.
- SK_SCAN3 (one warp per candidate block, two syncs whatever the count; same job order): exact; head mean B=1 183.6
  vs 183.0, B=8 equal. Whatever the scan phase spends, it is not its rounds or syncs. Left off.
- Near pieces gate_up 160 KB + down 192 KB (others 176): main kernel -1.3 / -1.0 us in two jobs (3 runs each).
- s0020 (track B, B<=3 near pieces gate_up 160 KB / down 192 KB): predicted ~-1 us (~7238); VALID 7238 / 7679 (new
  best; in process 7225).
- Programmatic dependent launch (-DPDL + MK_PDL=1): the first test was INVALID - the helper template sat inside
  extern "C" and the build failed, so the job ran the card's previous library (both sides identical). Found by a
  from-scratch build check (tools/build_check.sh); fixed; re-measured below. Nothing broken was submitted (s0020 was
  built and checked before the PDL code). Lesson: the build step's error must stop a test script.
- B=8 re-check of the B=1 prefetch wins (2 cards x 3 runs, long contexts): per-matrix near map +3..+6 us, whole
  o_proj share at the attention barrier +2..+5: the B=8 defaults stay.
- PDL re-measured on a real build (3 runs): B=1 pass period 7275.2-7275.5 with MK_PDL=1 vs 7258.6-7258.9 without, and
  the PDL-compiled binary without MK_PDL ~10 us above the normal build (two instructions at the main kernel's start
  changed its code generation, F10). B=8 +4. Dropped (left behind -DPDL, off).
- SK76_SHARE (one staging array for the 6- and 7-bit branches of the mixed sketch instance): spills unchanged (136 B),
  head equal. Left off.
- s0021 (cleanup confirmation of the s0020 configuration with later experiments compiled out): expected s0020's
  numbers; VALID 7242 / 7679 (s0020 7238 / 7679): same within noise; the shipped tree evaluates as intended.
- Final robustness (tools/robust.sh, from-scratch build): sketch path exact at 21504 positions (B = 1, 2, 4, 6, 8; 211
  exact ties); submit / poll == step-by-step greedy over 1024 tokens at B = 1, 3, 8.
- Near pieces of out_proj / o_proj (key 320x96) and qkv (896x80) at 160 / 192 KB (3 runs each): all within +-1 us of
  176 (qkv 192 +1.5). The prefetch table is converged.
- The cheapest rounding-changing escape-locality variant re-examined before deciding on a gate gamble
  (tools/gu_perm_census.py, every 4th layer): one permutation of the intermediate index per layer (gate_j and up_j
  move together, as the gate_up tiles interleave 8 gate and 8 up rows; down's columns follow) clusters almost
  nothing - escape quads 6.93 % -> 6.65 % - because escape-heavy gate rows and escape-heavy up rows are different j's;
  the busiest warp share gets worse (10 -> 18 slow quads). Independent row orders (2.3 %) would need epilogue changes.
  Not worth the gate risk; not done.
- CORRECTION: the timing halves of xs_test / last_test / scan3_test were invalid - their measurement step rebuilt
  engine/ from build.sh without the variant flag, so "xs", "last" and "scan3" timed the default build (their
  exactness checks did use the variants). Re-measured below with var_gen builds (each variant compiled separately).
- Re-measured with var_gen builds (2 runs, long contexts; head median / pass): current B=1 181.8-182.4 / 7247.7-7249.4,
  B=8 205.4-205.9; SK_RING (B=1 rescoring chain from a TMA ring, one warp) B=1 185.3-185.5 (pass equal), B=8 216
  (worse); SK_XS B=1 ~185, B=8 equal; SCAN3 B=1 183.6-184.6, B=8 equal; SK_LAST B=1 179.7-180.6 / 7245.5-7246.1
  (-2), B=8 210 (+4.6: thread 0 of the last block finishes 8 sequences one after the other). -> SK_LAST for the
  B=1 programs only.
- SK_LAST for the one-sequence programs only (B>1 passes keep the finish op): var_gen, 3 runs: B=1 head 180.2-180.9 vs
  181.8-183.2 us (pass -1.4..-2.4), B=8 equal; exact 7168 / 7168 (B = 1, 8, 3, 2), submit / poll == greedy over 1024
  tokens at B = 1, 3, 8.
- s0022 (track B, SK_LAST for one-sequence passes): predicted ~-2 us (~7236); VALID 7259 / 7708, in process 7245
  (s0020 7238 / 7679 / 7225; its confirmation s0021 7242 / 7679 / 7227). B=8 +29 although its path is unchanged
  (only the head kernel's code changed; var_gen measured B=8 equal) - most likely an evaluation-level shift, but not
  provable either way. Not re-submitted to re-draw the measurement; SK_LAST taken out of build.sh so that the final
  engine is the configuration evaluated twice at ~7240 (s0020 / s0021).
- s0023 (final state = s0020 configuration, experiments compiled out): expected ~7240 / ~7680; VALID 7261 / 7711, in
  process 7227. The same configuration measured 7238 / 7679 (s0020) and 7242 / 7679 (s0021) earlier: the evaluation's
  level moved by ~+20 / +30 us between s0021 and s0022/s0023. Against this later level SK_LAST (s0022, 7259 / 7708)
  was -2 / -3 over HTTP, as measured in process, but +18 in the evaluator's in-process figure: noise dominates; the
  conservative state is kept.
- Near prefetch spread over the block's 32 warp starts (PF_WARPS + MK_PF_WARPS=1; 32 pieces of 5.5 KB issued at
  once, after thread 0's barrier increment): main kernel 7154.5-7154.7 us vs 7072.4-7072.6 for the same build without
  it and 7066.4-7066.9 for the normal build (the barrier change alone +6, code generation). Concentrated first-bytes
  prefetch is far better than per-warp starts: the warps' cold starts are not what the GEMM tail is made of. Dropped
  (left behind -DPF_WARPS, off).
- V of the attention units' first 5 chunks asked at the attention barrier by 16 of the otherwise idle blocks per unit
  (MK_PF_ATT_V=1; one contiguous 40 KB range per dimension group): main kernel +1.2..+1.5 us (3 runs; with 21 chunks
  the same). The op's own V requests at its start are early enough. Off.
- Head kernel shared memory 40 KB instead of 64 KB (more L1): equal at B=1 and B=8 (3 runs). Left at 64 KB.
- Final exactness run (fresh build, sketch path vs exact head, teacher forced on the generated text, 1024-token
  greedy runs): 0 mismatches at 18432 positions (B = 1 and 8, two prompt sets; 153 exact ties).
- SK768: in the B=1 mixed sketch the 6.1 % hot tiles (the frequent candidates) in plain int8 instead of 7 bits (+9.5 MB).
  Census (24 x 192 own positions): candidates mean 33.2 -> 11.3, p90 25 -> 11, p99 768 -> 202. var_gen, 3 runs: B=1
  head median 181.8-182.8 -> 177.2-177.3 us (pass -4), B=8 unchanged (its path is the all-7-bit copy); exact
  6144/6144; mixed-instance spills 136 -> 108 B.
- s0024 (track B, SK768): predicted ~-4 us vs the current evaluation level (s0023 7261 -> ~7257); VALID 7252 / 7705,
  in process 7219 (s0023: 7261 / 7711 / 7227): -9 / -6 / -8 at the same level. Kept.
- 6-bit set with 8-bit hot tiles (3 runs): K40 / K80 / K160 head mean 180.8-181.1 / 177.7-179.5 / 177.4-178.2 us
  (median 180.5-180.8 / 177.2-178.7 / 176.0-176.7): K160 within ~0.5 us of K80 on the mean; K80 stays.
- With SK768 the mixed sketch also wins at B=8 (MK_SK76_B1ONLY=0, 3 runs): head mean 201.2-201.7 vs 206.3-206.4 us
  for the all-7-bit copy (p99 256 vs 213; max ~350). The dual copy is dropped with SK768 (one sketch for all B).
- s0025 (track B, the mixed 6/8-bit sketch for every batch size): predicted B=8 ~-5 (~7700), B=1 ~7252; VALID
  7255 / 7707, in process 7242 (s0024 7252 / 7705 / 7219): within the evaluation's noise (the in-process figure moved
  +23 us on an unchanged B=1 path). Kept on the same-job nsys evidence (B=8 head mean -5 us in 3 runs) and because it
  drops the 3.6 GB all-7-bit copy.
- Tile-table weight of a 6-bit slot with 8-bit hot slots (3 runs): 7 / 6.5 / 6: head mean 179.0-179.3 / 177.4-177.8 /
  177.7-178.4 us, main kernel +1 for 6.5 / 6, pass total equal within 0.5 us. Left at 7.
- SK_DSR (hour 18.5; sketch prologue digit sums accumulated per thread and reduced over the warp before one shared
  atomic per column and warp, instead of 2 atomics per entry on 2-16 shared addresses). Expected -0.5..-1 us at B=1
  if the same-address atomics serialized. Exact (sk_check 0 / 4608). head_nsys 3 rounds: B=1 head median
  178.4-179.3 (base) vs 179.7-181.5 (DSR), B=8 195.9-196.8 vs 196.1-196.7; SK_PROF prologue phases equal (B=1
  x loaded 2.8-3.1, digits 4.6-5.1 us; B=8 6.1-6.7 / 13.1-13.6). The atomics were not the prologue's cost. Off
  (left behind -DSK_DSR). Profile detail: warp 0's stream ends at ~95 us vs ~140 for warps 1-31 (its first bytes are
  in L2 from the main kernel's last barrier); the stream as a whole runs at ~7.2 TB/s, so this slack does not convert
  (SK_W0X measured the same earlier).
- SK_XU (prologue x loops unrolled, loads issued together): predicted B=8 head -2..-3 us; measured WORSE: B=8 prologue
  13.3 -> 20 us, head 196-198 -> 202.7-203.7 (3 rounds), B=1 +2. The prologue is not latency-bound per entry. Off.
- SK_XV (sk_xload reads x as 4 aligned u32 pairs (k, k+1) instead of 8 u16 scalars; both input layouts keep the pair
  adjacent and 4-byte aligned, so the values are the same). head_nsys 3 rounds: B=1 head median 178.4-178.7 ->
  175.4-175.5 us, B=8 196.0-196.4 -> 186.6-186.9 (more than the prologue's own -2.5 us at B=8 in SK_PROF: the
  function's code generation changes as well). Exact: sk_check 0 / 2304.
- s0026 (track B, SK_XV): predicted ~7252 / ~7698 (head -3 / -9.5 us); VALID 7250 / 7692, in process 7235
  (s0025 7255 / 7707 / 7242): -5 / -15 / -7, consistent with the prediction at the evaluation's current level. Kept.
  Checks before it: sk_check 0 / 4608 and 0 / 6912, bitcheck bit-identical (B=1, 8), served == offline (B=1, 8),
  build_check of both tracks and all experiment flags.
- SK_ST31 (column stats of the sketch prologue by warp 31, idle at B=1, instead of threads 0-7): predicted up to -1 us
  at B=1; head_nsys 3 rounds equal (B=1 175.4-175.7 vs 175.3-175.9, B=8 equal); SK_PROF: thread 0's digits -0.5 us
  but the block's sync time unchanged (other warps finish their digits at the same time). Exact (0 / 4608). Code
  removed again so that engine/ equals the evaluated s0026 source byte for byte.
- SK_XS on top of SK_XV (x staged in shared memory for the B>1 digit pass): worse, head B=1 175.4-175.8 -> 178.6-178.8,
  B=8 186.9-187.0 -> 190.6-191.0 (3 rounds). With 4-byte loads the second L2 read is cheaper than the staging. Off.
- SK_Z4 (cut-tile slots zeroed with 16-byte stores, 2 per thread instead of 8): predicted <= -0.3 us; head B=1
  175.55-175.65 -> 175.39-175.44, B=8 186.5-187.2 -> 186.3-186.9 (3 rounds), exact (0 / 2304). Below anything the
  evaluation resolves; not submitted, code not kept (engine/ stays the s0026 source).
- SK_W0X re-measured on the final build (warp 0 ends its stream at ~92 us vs ~140 for the others): W0X=12 head B=1
  174.8-175.4 vs 175.5-175.7 (-0.3), B=8 -0.3; W0X=24 +10.5 / +11 us (warp 0 becomes the block's last warp). The slack
  is warp 0's L2-resident start, which only covers ~12 units; nothing resolvable. Not kept.
- (hour ~20.5, budget extended) Escape regrouping inside cut-structure classes (bit-identical; engine/rowperm.py,
  tools/cls_perm_census.py, tools/rp_time.sh; findings F19): census per-op max slow quads 1118 -> 632 (best fit), but
  timing with outputs not mapped back: -14.5 us B=1, -18.5 B=8, only -5 us for the kinds whose epilogue has no per-tile
  reduction (in_proj, qkv). Predicted (old estimate) -80 us: wrong; sorting by escape count would even have made the
  slowest warp worse. Not built (main-kernel epilogue change for -5 us). MK_ROWPERM=fit stays as an experiment switch
  in api.py (off by default; the engine's function is unchanged).
- Code placement sweep (tools/pad_sweep.sh, -DPADN=0..7, 16; findings F20): 0..+5.6 us against each card's base, no
  better placement. Predicted: a lever of 5-30 us if F10 were placement; it is not. Reverted.
- Recurrent op at B=1 (LMIX_PROF, block 0, per layer): entry 0.4, a 1.0 us stretch of cold code, inputs 2.2,
  recurrence 2.4, norm + output 1.5 us. ncu stall sampling: no-instruction stalls at op entries (F21).
- IC_WARM (warm the recurrent op's entry at the barrier before it): predicted up to -48 us; measured +149 / +156 us
  (B=1 / B=8; B=8 never runs the warm path): main-loop code generation. Reverted.
- MAIN_REGS (existing flag): spills 48 -> 56 / 60 B, +44 / +91 us. GP_NOINL (GEMM start-up in a noinline function):
  spills 48 -> 16 / 12 B, predicted a gain from fewer spill reloads per op; measured +27.5 / +18 (NSTEP=1024) / +9..+41
  (B=8). Reverted; spill bytes are not the mechanism (F21).
- Build-option draws with unchanged floating-point semantics (tools/flag_draws.sh; nsys main-kernel medians against
  s0026 on the same card): --extra-device-vectorization, -Xptxas -O2, --allow-expensive-optimizations=false,
  -lineinfo, -dopt on: all within +-1.3 us. -Xcicc -O2 (NVVM): B=1 -4.6 us (card 1, 3 rounds), -2..-7 (card 2), but
  B=8 +1..+5: mixed and below the evaluation's resolution. Not kept. (-rdc=true did not build as a shared library with
  var_gen's one-step command.)
- B=8 prefetch table at the scored contexts (env_nsys, NSTEP=1024, 2 rounds per card, each against its card's base):
  per-matrix near map of B<=3 -0.3 / -4.2, whole o_proj share at the attention barrier -1.1 / -6.0, near 160 KB
  +15 / +7, near 192 KB +5 / -1, far 96 KB +2 / -4, far 32 KB +7 / -3 us: all within noise. Converged; unchanged.
- Final tree re-verified on the replaced pod (epoch 5): sketch 0 / 4608, bitcheck identical (B=1, 8), served == offline
  (B=1, 8), build_check of both tracks and all experiment flags.
- Schedule-draw study with k executed fillers per op at the top of the main loop (DRAW_K, `nanosleep.u32 0`, scratch
  copy of the source): +14, +37, +51, +80, +114, +148, +175, +205, +242 us for k = 1..9, i.e. linear, ~+27 us per
  filler per pass (~70 ns per nanosleep(0) per op and warp: it yields the warp). The fillers' own cost swamps any
  schedule effect, so this does not measure the draw distribution; the never-executed filler sweep (F20, +-5 us)
  remains the only clean one. Not pursued further.
- Extended exactness of the final engine's sketch path (tools/sk_check.py, new SEED option; greedy through the sketch
  vs the exact head's argmax with min-index ties, teacher forced on the generated text, 768 steps, B = 1, 2, 4, 8):
  0 mismatches in 23040 positions for each of four prompt sets (seeds 20000, 40100, 40200, 40300): 92160 positions,
  939 of them with exact ties at the top.
- Callee-only edits are safe (hour ~21.8): a never-taken block inside lmix_work leaves the top-level loop's SASS
  identical and changes 28 of the hot loop's 2342 lines (fillers 470, MAIN_REGS 1108); timing B=1 -0.4 us (3 rounds),
  B=8 +3. So the ops' own functions can be worked on.
- LM_VOLC (recurrent op lmix_work re-reads B and the spec flag from shared memory where used; they were kept live,
  spilled at entry and reloaded from local memory before the unit was chosen: the ~1 us cold stretch of F21). lmix_work
  <8,true,false> spill loads 104 -> 52 B. main_nsys: B=1 7036.2 -> 7005.5 (3 rounds, card 1), NSTEP=1024 7066 -> 7028
  (card 2), B=8 +10 (7484.5 -> 7494.8, 3 rounds, card 3; the B=8 path uses lmix_dual/tri: code generation). Checks:
  sketch 0 / 4608, bitcheck identical (B=1, 8), served == offline (served B=1 7175 us/step), build_check.
- s0027 (track B, LM_VOLC): predicted ~7220 / ~7700; VALID 7221 / 7698, in process 7202 (s0026 7250 / 7692 / 7235):
  -29 / +6 / -33, as measured in process. New best on the board (s0020 7238 / 7679).
- ATT_VOLC (the one-op attention's base pointers and max_ctx re-read from shared memory where used, as LM_VOLC):
  spill loads 172 -> 144 B, but B=1 +42 us (3 rounds), NSTEP=1024 +30, B=8 mixed: those values are used inside the
  op's loops, so re-reading them per use costs more than the spills. Reverted. Rule: re-read only values that are used
  once, early, on the critical path (the LM_VOLC case).
- LM_VOLC2 (after the lm_core call, u / part / hv / P / c0 / na / keep / NG derived again from the loop index and
  shared memory instead of kept across the call): lmix_work<8,true,false> spills 56 / 52 -> 28 / 12 B, but B=1 +61 us
  (3 rounds), NSTEP=1024 +65, B=8 -5. Reverted. Even inside a callee the spill count does not predict the time; LM_VOLC
  removed two specific reloads that sat before the unit was chosen.
- ATT_PFV (only the attention op's entry K / V prefetch computes its addresses from fresh shared reads): B=1 +12 us
  (3 rounds), NSTEP=1024 +16. Reverted. The attention op does not take edits; lm_core's reloads follow a block barrier
  or come at the state store (not the LM_VOLC pattern). Spill hunting stopped at one win (LM_VOLC) and three losses.
- Recurrent op norm exchange through clusters (bench/cluster_probe.cu: 2-block clusters, 76 resident = all 152 blocks;
  cooperative + cluster launch accepted). All exact (sketch 0 / 4608, bitcheck identical, served == offline):
  LM_CLUSTER (sums read over DSMEM after a cluster barrier) +41 us B=1 / +24 NSTEP=1024; the cluster launch alone
  (LM_CLUSTER_LAUNCH) +6; so the barrier version costs +35 (its release waits for lm_core's 32 KB state stores);
  LM_CLUSTER2 (relaxed remote store of the two sums into the partner's shared memory, local sentinel poll; slots set in
  the embedding op) +6 / +0: the faster exchange pays back only the launch's cost. Reverted.
- MK_PF_STATE (host table only: at the barrier before each recurrent op the 96 busy blocks ask L2 for their 32 KB of
  state, buffer 0, instead of an out_proj piece): B=1 +1.5 us (3 rounds), NSTEP=1024 +2, 16 KB +3.5. The op's own
  state prefetch at its start is early enough. Not kept.
- Final engine (s0027) robustness, fresh builds: tools/robust.sh sketch 0 / 21504 (B = 1, 2, 4, 6, 8, 512 steps),
  served == greedy over 1024 tokens at B = 1, 3, 8; sk_check SEED=50000 N=1024 0 / 18432; tools/stress.py 20 runs x 128
  steps at B=1 and B=8 bitwise identical (no races).
- Variants of the recurrent op's entry around LM_VOLC (scratch copy, 1 round each against s0027 on the same card):
  only the spec flag re-read +37 us, only B +24, LM_VOLC + NG re-read +4, + the column-layout flag +4, the state
  prefetch issued inside the unit loop +46. LM_VOLC as submitted is the best of its neighbourhood.
- -Xcicc -O2 on the s0027 build: B=1 +0.7 us (3 rounds), NSTEP=1024 +4, B=8 -5: neutral. Not kept.
- LT_VOLC (the B=8 recurrent op lmix_tri re-reads flags and NG from shared memory, as LM_VOLC): B=8 +3.5 us (3
  rounds), NSTEP=1024 +9, B=1 -1. Reverted: the LM_VOLC effect does not carry over to the B=8 form.
- Callee re-read draws on the per-GEMM functions (scratch copy; acceptance rule fixed beforehand: B=1 <= -15 us in 3
  rounds, B=8 <= +5, exact): epilogue_k NC re-read +35 us, use_r +1; finalize KQ / nwarps (DR3) -11 in a 1-round screen
  on a card whose base read high, then on the stable card +5 (3 rounds), NSTEP=1024 +6, B=8 -10. None accepted.
- lm_core re-read draws (same rule): spec re-read +12 us, keep derived again +41, both +48. None accepted. Tally of
  function-local edits in this extension: one kept (LM_VOLC, -29 us over HTTP), fifteen measured losers (+1..+61);
  the search is stopped.
- Serving path measured on a job card (F26): HTTP 7193.8 / 7697.3 us (B=1 / 8) = in-process generate() 7194 / 7694.
- Column power-of-two scaling (F27): exact (bitcheck identical), escapes -3.7 %, main kernel ~-2..-5 us at B=1. Not kept;
  patch in tools/colscale_api.patch.
- Schedule-draw study around s0027 (nine never-taken-block variants, fixed acceptance rule): all +4.8..+33 us; none
  confirmed or submitted. Findings F24 (end).
- SK_XT (B > 1 head prologue: the 80 KB input copied into shared memory by one bulk copy + mbarrier, both passes read
  it there; head shared memory 132 KB): exact (sketch 0 / 7680, B = 1, 2, 4, 8), but head B=1 +3.6 us, B=8 +6.3 (3
  rounds): the larger shared-memory allocation costs the stream's L1 more than the prologue gains. Reverted.
- ptxas options (semantics-preserving) screened against s0027: --register-usage-level 0 / 3 +160 us, 5 (default) 0,
  7 -41.5, 9 -13, 10 -42; -split-compile=4 +14, -no-bb-merge +12, -ret-end +6. Confirmed: level 10 B=1 7005.7 ->
  6963.6 (3 rounds, card 1), NSTEP=1024 -38 / -51 (card 2), B=8 -23 / -21 (card 3); level 7 the same at B=1 but B=8
  -23 / +8. Level 10 in build.sh for track B (MK_PTXAS_EXTRA overrides). Checks: sketch 0 / 4608 and 0 / 21504, bitcheck
  identical (B=1, 8), served == offline (served B=1 7131 us vs 7175), build_check both tracks + 10 flags, stress 20 x 128
  bitwise identical at B=1 and 8.
- s0028 (track B, ptxas --register-usage-level=10): predicted ~7180 / ~7678; VALID 7175 / 7681, in process 7161
  (s0027 7221 / 7698 / 7202): -46 / -17 / -41. New best.
- Around s0028 (1 round each against it): level 8 the same, level 6 -0.7, level 10 without LM_VOLC +55, + -Xcicc -O2
  +18, + allow-expensive-optimizations +15, + -no-bb-merge -3 (skewed base). s0028 is the best sampled configuration.
- Earlier flags re-screened under level 10 (1 round against s0028): MAIN_REGS +56 us; SK_LAST pass -1.3; SK_SCAN3 head
  +1.8; SK_W0X=12 head -1.2 / pass -3. None above the acceptance threshold; s0028 stays.
- Remaining ptxas options with level 10 (1 round against s0028): -O2 -0.4 us, -disable-optimizer-consts -4, -Ofc=min
  +1484. None kept; the option space is covered.
- The nine never-taken-block draws again, under level 10 against s0028 (tools/draw_n.sh with F_EXTRA): all slower,
  +3.3..+32 us at B=1. s0028 is the best of its sampled neighbourhood as s0027 was of its own.
- Pod replaced (epoch 6): s0028 re-verified from scratch (sketch 0 / 4608, bitcheck identical at B=1 and 8, served ==
  offline, build_check both tracks + 10 flags); engine/ byte-identical to the evaluated s0028 source.
- Small candidates confirmed with 3 rounds against s0028: -disable-optimizer-consts B=1 -7 us (card 1), NSTEP=1024 -4,
  B=8 -9..-15 (card 3 B=1 +4 / -7 / -5); SK_W0X=12 pass -1.5; column scaling (F27) under level 10 -0.4 / -1.6 / +19.
  -disable-optimizer-consts added to the track-B ptxas options; checks: sketch 0 / 4608 and 0 / 21504, bitcheck identical,
  served == offline, build_check, stress identical.
- s0029 (track B, + ptxas -disable-optimizer-consts): predicted ~7170 / ~7670; VALID 7170 / 7652, in process 7154
  (s0028 7175 / 7681 / 7161): -5 / -29 / -7. New best.
- Register-usage levels 7 / 8 / 9 with -disable-optimizer-consts against s0029 (1 round): +0.8 / +1.0 / +0.7 us: converged.
- Against s0029 (1 round): -maxrregcount=60 +109 us, 56 +84, --extra-device-vectorization -5 (card with a skewed base).
  None kept; the build-option space around s0029 is covered.
## Close of the extension (agent-hour ~23.4)
- Final engine = s0027 (7221 / 7698, new best on the board); engine/ byte-identical to its evaluated source; robustness
  checks above. Findings F19-F25 added; plan.md and summary.md current.
## Close (agent-hour ~18.3; extended to s0026 at ~19.3)
- Final engine = s0025 (track B): 7255 / 7707 at the evaluation's late level (best on the board s0020 7238 / 7679 at
  the earlier level, F17). Verified: from-scratch builds of both tracks and every experiment flag; sketch exact (4608
  + 9216 positions over B = 1, 2, 3, 5, 8); bitcheck bit-identical to track A; served == greedy at B = 1, 3, 8.
- Extension (hours 18.4-19.5): SK_DSR, SK_XU (worse), SK_XV (kept: s0026 7250 / 7692), SK_ST31. Final engine = s0026;
  pre-submission checks above; build_check of both tracks and all experiment flags passed with SK_XV on.
