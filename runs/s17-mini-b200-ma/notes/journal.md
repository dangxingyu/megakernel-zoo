# Journal (run s17-mini-b200-ma, merge run, track A)

Base: engine/a2 = s15-mini-b200-a2 s0025 (B=1 2755 / B=8 5825); engine/a1 = s15-mini-b200-a1 kernel (B>=2).

## s0001 hybrid: a2 s0025 kernel at B=1, a1 s0024 kernel at B>=2 — 2749 / 5522 (expected ~2755 / ~5515): confirmed
Idea source: a1 s0024 (s15-mini-b200-a1, B=8 5513) + a2 s0025 (s15-mini-b200-a2, B=1 2755). engine/api.py is a
wrapper: both engines load (44 GB), cohort engine chosen by batch size; batch gate not bitwise (max 2.53, passes),
invariance bitwise, http_tokens 8/8. Same-job serve test before: a1 kernel 5878 vs a2 kernel 6096 at B=8.

## s0002 a1 s0025 port (ring stages 9 x 2 in the B>=2 kernel) — 2752 / 5450 (expected ~2750 / ~5450): confirmed.

## B=8 accounting by ablation (real builds, t8tf.py, pos 320-576, natural text; base 5718 us)
Skipping (results wrong, timing only): all loop gates (SKG) 4325 (-1393 = 58 us per gated application); gate phase 0
(A0/D0, SKG1) -1300; phase 1 (SKG2) -900; phase M (SKG3) -280; head (SKH) -142. Most other SK* switches hang.
a2's kernel spends ~22 us extra per gated application at B=8 (crit path) vs a1's ~58: the gate chain is the largest
named B=8 loss.
## MSGW (not kept): warp-per-message phase M at B>1 (one warp per (seq, layer-gate) message, then the message-half
GEMV with each D row loaded once for all 8 sequences, unit rows staged in smem). Correct (msg rel 2e-3, zm 1e-3 vs
the old path after one step; t_ref8 B=8 n=40 median 0.0151 vs 0.0124 base, top-1 1.0). M itself 33.5 us (profile).
But the B=8 kernel: inline v0[9]/v1[9] -> 744 B stack, +450 us; noinline helper -> 160 B stack, +480 us (w13 stream
32.0 -> 38.2 us/app in the profile); inline with 3-chunk groups -> 48 B stack, +160..+320 us. Code kept behind
MSGW=0. Lesson (as a1's): the a1 B=8 kernel cannot take added code in any phase without its allocation moving.

## Where the B=8 time of added code goes (e0026-e0031): the decode loops' code generation
GZ0 (B>1 gate phase 0: each block mixes one sequence itself, z rows from smem, no mixer -> flag hand-off) is
BITWISE EQUAL to the base and +510 us. Profile: w13 stream 32.0 -> 39.0 us/app, w2 18.5 -> 21.1 (same as MSGW).
SASS of the B=8 kernel (tests/sass_hm.py): the NVFP4 decode loop (40 HMMA) is 540 instructions in the base and 583
with GZ0 (+8%); the MXFP8 loop 476 -> 578 (+21%); no spills within 40 instructions of any HMMA (tests/sass_spills.sh).
So the expert phases, which are issue-bound at 8 warps/SM, follow the register pressure of unrelated code.
- Code padding before phase E (PADN 4..33, never executed): base 5617..5676, GZ0 unchanged: not placement.
- consume_tasks out of line (CTNI): +470 us alone (the loop reaches its captured state through generic loads).
Conclusion: in this kernel a change elsewhere is only worth trying if the decode loop's SASS stays at 540 / 476.
## KB1 in the batch-1 (a2) kernel: a code-generation cliff, and a way around it
KB1 (hd.B a compile-time 1; idea of b2 s0020/s0021 "K1"): bitwise equal, spills 116 -> 80 B, but 3120 vs 2743 us.
In MKPROF-compiled builds (marks present, no profile buffer: inactive) KB1 is 2724 vs 2744 (plain) / 2799 (marks, no
KB1). Compiler memory barriers at the mark sites (MARK_BAR) do not reproduce it. Not understood; measured 3 times.
## s0003 KB1 + mark sites in the batch-1 kernel — 2732 / 5471 (expected ~2732 / ~5450): B=1 confirmed (-20).
B=8 +21 is noise: the B=8 kernel's SASS is identical to s0002's (md5 of the cuobjdump of mk_kernel_t<8>).
## Sweep of a1 switches after RI=9 (e0025, B=8 generate 512, 2 rounds): PMAXC 2 -20 (a1: +12..+20 at long
context, not taken), BAR_SLEEP 64 -7, SCNT_REL -5, PUB_REL +3, PFE_DN 1 +38, RED_SLOTS 48 +38.
## Engine load order (MK_ORDER=81: a1 engine allocated first): B=8 -17..-60, B=1 +5..+8 (serve test, 2 rounds). Not taken.
## Why GZ0 slows the expert phases (SASS of the NVFP4 loop, tests/sass_loop.py)
Base: distinct registers per item, scales and MMAs of consecutive items interleaved. GZ0 build: every block scale goes
through one register (PRMT R5 -> LEA R5 -> 2 FFMA -> next PRMT R5), MMA results reuse R144, and 20 filler
instructions (@!UPT UIADD3 URZ) appear: the loop is register-starved, a global allocation choice of ptxas. Searched
(SASS only, 20 s per batch): 6 code variants of GZ0 (z rows unrolled / rolled / 3-chunk / out of line), without its
mix call or xm stores, -maxrregcount 224..248: the loops stay at 576-583 (base 540 / 476). Not controllable here.
## B=1 kernel knobs re-tuned under KB1+MKPROF (e0044): FP4_UNR 4/8 +35/+45, W13_UNR 9 +30, BF_UNR 3 +445 (cliff),
D_UNR 2 +25, FP8_UNR 3 +8, QPF 1 +8: the current settings stay. Build flags (e0037): -dlcm=cg -10, -dlcm=ca -6,
ptxas -O2 -4, extra vectorisation -3, -Xcicc -O2 +420 (cliff). B=8 kernel flags (e0038): none helps; MKPROF +285.
## SCALE_BF16 in the B=8 kernel (a1's unused path): as many HMUL2 as the FFMAs it replaces, +35 us, and 4.6e-2
median deviation from the base (unvalidated path). Not used.
## Limits and cohort switching (tests/limits.py, e0050): max_context 100000 -> effective 4096 (a1 refused > 4096:
fixed in the wrapper); batch change without reset, host ids 201024 / -1, request beyond 4096, step 4097: ValueError;
serve cohorts B=8 -> B=1 -> B=3 all complete; generate at B=2 works (a1 generic kernel).
B=1 (a2 kernel) vs B=8 slot 0 (a1 kernel): step-0 logits rel 4.0e-3; greedy diverges at token 9 of 40 at a near-tie
(B=1 top-2 margin 0.032, B=8 0.114; tests/tie.py); teacher-forced on the same tokens they agree at the other 39.
PMAXC 2 (B=8): -30 us per step at positions 64-256 (two-chunk pieces not merged), noise at 320-576: ~-6 us on the
scored workload; not submitted.

## The way around the decode-loop code generation: keep the old path compiled, never taken (GZK / MSGK)
With the original mixer path of gate phase 0 kept in the binary behind a runtime condition that is never true
(p.prof_cap == -77777), GZ0's build has the base's decode loops again (540 / 476 SASS instructions per 40 / 18 HMMA)
and runs 5545 vs 5623 us (bitwise equal). The same for MSGW needed one more variant: without the first-row D preload
of the message-half GEMV (MSGV bit 2) the loops stay at 540 / 476 (with it 567-604): 5495 vs 5540.
Method: compile variants, count the decode loops in SASS (tests/sasssearch.sh, 20 s for 8 variants), time only the
variants whose loops match the base. MSGW correctness: pass-2 messages vs the old path rel 1e-7, zm 5e-6
(tests/msgstop.py, kernel stopped after the first M phase); logits vs base 2.2e-2 median: the bf16 cascade.
## s0004 GZ0+GZK, MSGW+MSGK+MSGV 4 in the B>=2 kernel — 2722 / 5355 (expected ~2732 / ~5340): confirmed.
B=8 -116 vs s0003 (-95 vs s0002). Reference check (t_ref8, B=8, 40 positions): median 0.0151, top-1 1.0.
## Not kept after s0004 (B=8 kernel, one engine per process, generate 512)
- NFW: norm_f32_rows_w (gated MLP / q-k-v input norms) in one round of loads per row: bitwise equal; noinline
  versions starve the decode loops (501/549), inlined ones keep them (473/540) but measure +21 / +100 us.
- a1 s0026 + s0027 ported by patch (per-kernel ring stage plumbing, early slot refill): EISS 3 (early refill in the
  B=8 kernel too) +324 us (a1 measured +354; 220 B spill loads); the patched source without it +25 us at B=8.
- NFW re-checked with all HMMA clusters (tests/sassall.sh): identical to the base, so its +21..+100 us is real.
- Bound on the w13 -> w2 hand-off: w2 without waiting for its slots' w13 units (NOWAIT13, results wrong):
  5499 / 5526 vs base 5502 / 5554: no measurable gain; the "F stage" span is block imbalance that the barrier after
  w2 absorbs anyway. Not pursued.
- Ablations re-run on s0004 with the SASS check (pos 320-576, base 5595): SKG1 4052, SKG2 4648 (loops intact), SKG3
  5398..5451, SKG 4327 (loops starved). These overstate: skipping gate work feeds stale inputs to the router, which
  changes the expert union. The profile (A1 5.75, D1 ~5 + norm per occurrence) is the better estimate (~240 us/step).
- B=1 build points around KB1+MKPROF (3 rounds): SUBMARK compiled out -15 (one +57 outlier), -dlcm=cg -O2 -10,
  ptxas -O2 0: within the timing noise of tests/t1.py (one ~+60 outlier per ~6 runs). Not taken.
- qkv at B=8: its ring stages are already issued at the end of the previous application's MoE (PFE_QKV).
- GZV 64 (GZ0's D rows bulk-copied into smem before the mix): bitwise equal, decode loops intact, +6..+28 us. The
  D-row loads were not on the critical path. Not kept (source in tests/a1_mk_study_variants2.cu.txt).
## s0005 = s0004 source re-evaluated: 2729 / 5357 (s0004: 2722 / 5355): repeatable to 7 / 2 us.
- ATM2 (B=8 attention merge: partial statistics and partial outputs in one round of loads; bitwise equal): +19 us.
  Around s0004 every small change measured so far costs ~20 us through the allocation even with intact HMMA loops.
- W2E (B=8: each warp issues its first w2 ring stages right after its own w13 tasks instead of after the block's
  w13 tail and two block barriers): bitwise equal, decode loops intact, +8..+10 us. Not on the critical path
  (source in tests/a1_mk_study_variants3.cu.txt).
## ncu attribution of the B=8 step (findings.md): issue 24.7%, the decode loops are ~1/5 of the warp samples; the
rest is spread over block barriers behind grid-barrier/flag spins, mbarrier waits and small phases. Consistent with
every local change around s0004 measuring within +-25 us: no single remaining loss is large.
- BPG in the B=1 kernel (b2's idea: the gated-norm stage re-polls its 9 stale words together instead of one by one):
  bitwise equal, +27 us (3 rounds). The KB1 profile's longer gate_norm span (5.2 -> 8.1 us) is the producers'
  (gate F/G) completion moving into it, not the polling. Not kept.
- a1 s0031 (MIXT 0: one mixnorm instantiation in the B>1 kernels; a1: B=8 serve -32) on our kernel: bitwise equal,
  5535 / 5535 vs 5496 (+40; one base outlier 5549). Not taken. a1 s0032 (router GEMV in one K part at B=8: a1's
  HTTP 5469 vs 5435) not tried. a2 had nothing new for B=1 after s0025.

## End of run (06:40, ~2.5 agent-h used): best track A = s0004 (2722 / 5355; s0005 same source 2729 / 5357).

## The NVFP4 decode loop is ALU-pipe bound when it runs (ncu, e0097/e0098)
Stall reasons inside the NVFP4 loop: selected 33%, not selected 20%, math-pipe throttle 19%, wait 12%, dispatch 9%.
Pipe counts over a B=8 step: ALU 788 M warp instructions, FMA 425 M (lite 228 + heavy 215), LSU 125 M, tensor 64 M.
Per NVFP4 item the ALU pipe carries 16 F2FP + 8 PRMT + 8 LEA; the FMA pipes 16 FFMA.
SCIM: build the block scale (b << 20) + (120 << 23) with IMAD (FMA pipe). ptxas turns any multiply by a known power
of two back into LEA (also from a __shared__ variable stored once), so the multiplier is read opaquely once per
consume_tasks call (volatile mov from a __device__ variable) and passed to item(). Loop: 541 instructions,
IMAD 78 instead of LEA 77. Bitwise equal. B=8: 5428 / 5461 vs 5496 / 5499 (generate 512), 5516 vs 5585 (pos
320-576, twice); serve test 5587 (vs ~5668 before).
## s0006 SCIM 4 (NVFP4 block scales by IMAD on the FMA pipe) — 2720 / 5279 (expected ~2725 / ~5300): confirmed.
B=8 -76 vs s0004. ncu after it (e0106-e0108): w2 loop math-throttle 17% -> 12% of its stalls, loop samples -9%;
the loops are now issue-bound (selected 35%, not selected 17%, dispatch 12%, i-cache 11%).
- SCIM 5 / 6 (byte extraction by multiply + multiply-high instead of PRMT): IMAD.HI is slow here: +55 / +420 us.
  The ALU work per NVFP4 item is now 16 conversions + 8 byte permutes; no cheaper exact form found (IMAD only
  clears garbage above a byte, not below it; PRMT/LDS-LUT alternatives need ALU address work).
- Outside the decode loops the largest stall is `barrier` (35-45% of the qkv / attention / o_proj+D regions):
  warps at block barriers while one thread waits on a grid barrier or flag = inter-block imbalance and hand-offs.
## FP8D: MXFP8 decode straight from the E4M3 bytes (layers 20-23, B>=2 kernel)
bf16 bits = (byte << 4) + (byte & 0x80) x 240 (the carry moves the sign from bit 11 to 15) is the E4M3 value x 2^-120
for normal and subnormal codes (bf16 subnormals); the UE8M0 scale becomes 2^(e-7). Per weight pair: PRMT + LOP3
(ALU) + 2 IMAD (FMA pipe) instead of F2FP + SHF + 2 LOP3 (ALU). Loop 476 -> 417 instructions per 18 HMMA,
ALU 350 -> 168. Exhaustive GPU check (tests/fp8d/fp8d_check.cu): all 65536 byte pairs x scale exponents 100..130,
new vs old product bit for bit: 0 mismatches (NaN codes excluded). Logits bitwise equal (200 positions x 8).
B=8: 5363 vs 5430 (generate 512, twice).
## s0007 FP8D — 2729 / 5231 (expected ~2725 / ~5215): confirmed (B=8 -48 vs s0006; B=1 kernel unchanged).
(The first build of it had FP8D undefined = 0: an insertion anchor had changed; caught by the SASS loop length.)
- ncu after s0006: out-of-line norm_f32_rows_w (gated rows' norms) holds 4.7% of the B=8 samples, 64% long
  scoreboard; mixnorm_t<1,1> 4.3% (32% i-cache, 20% barrier); mixnorm_t<3,3> 1.9% (74% i-cache).
- NFW retried on the s0006+FP8D base: inlined, decode loops intact: +28 / +112 us. The norm's long-scoreboard time is
  not removed by issuing its loads in one round. Not kept.
- HBAR (two-level grid barrier: 8 groups of 19 blocks, group-last arrives at a top counter): bitwise equal, +300 us:
  the returning group atomic adds a round trip; 152 non-returning REDs on one counter pipeline well at L2.
- BARR (grid barrier arrival as red.release instead of fence.sc + atomic): +7 us (noise). Not kept.
- Sweep on the s0007 base (e0123, generate 512, base 5362): MIXT 0 +37, RI 12 +40, PMAXC 2 +6, RKQ 3 +35, BAR_SLEEP 64 +8.
- tools/kit/bitcheck.py, s0004 tree vs s0007 tree (512 random-token positions, every 4th, B=1 and B=8): bit_identical
  1.0 at both batch sizes (SCIM 4 and FP8D change no bit). Batch invariance within each build (B=8 row 0 vs B=1):
  not bitwise (two different kernels): rel L2 median 0.026, max 0.10, top-1 0.992 on random tokens; the evaluator's
  batch gate (vs the reference, with tolerance) passes in every submission.
- W13H (B=8: the last round's leftover w13 units split into K halves on otherwise idle warps, halves summed after
  the block barrier): decode loops intact, numerics fine (t_ref8 median 0.0137, top-1 1.0), but +35..+110 us: the
  combine + epilogue after the barrier lands on the critical path. Same verdict as a1's split attempts. Not kept
  (source in tests/a1_mk_study_variants4.cu.txt).
- Final checks on s0007: bitcheck vs s0004 at B=3 (generic kernel, which also carries SCIM 4 / FP8D): bit_identical
  1.0 over 256 positions; tests/limits.py passes (cohorts 8 -> 1 -> 3, limits).
- Considered and not built: an IMAD-based E2M1 -> bf16 construction (the two nibbles of a byte need different shifts,
  so it costs >= 2 ALU instructions per pair, more than F2FP's 1).
- B=1 (a2) kernel pipe counts (ncu, e0129): ALU 235 M, FMA 217 M, LSU 104 M, tensor 19 M warp instructions per step;
  stalls barrier 36%, long scoreboard 32%, math 3%: not pipe-bound, so SCIM/FP8D-style moves do not apply there.
## s0008 = s0007 source re-evaluated: 2731 / 5229 (s0007: 2729 / 5231): repeatable to 2 us at both batch sizes.
- B=1 -Xptxas -dlcm=cg re-measured (4 rounds, generate 1024): 2735 vs 2737 mean: noise, not taken.
- GZSEL (GZ0 only for the attention gates / only for the MLP gate): +28 / +93 us; both together (s0004) stays best.
- MIXW (plain applications' attention-input norm by a1's unused warp-per-sequence mixnorm_w in every block, no
  mixer hand-off): +880 us. The warp-serial three-pass mix is far slower than 8 mixer blocks + hand-off.
- GNM (gated rows' norms by the mixer blocks + publish, as in plain applications): +55 (MLP norm) / +165 (attention norms) / +800 (both: loops starved). Not kept.
## SCIM 8: NVFP4 block-scale pairs by one exact E4M3x2 -> f16x2 conversion, f16 -> f32 by HADD2.F32 (FMA pipe)
The two scales of a k16 block (rows g, g+8) are adjacent bytes of the scale word: one cvt.rn.f16x2.e4m3x2 of the
word's low or high half (selected with mov.b32 {l,h}: no shift) gives both, exactly; f16 -> f32 is HADD2.F32 on the FMA
pipe. Replaces 2 PRMT (ALU) + 2 IMAD per pair. NVFP4 loop 541 -> 500 SASS instructions per 40 HMMA, ALU 229 -> 191.
(SCIM 7 with a shift for the high half: 544, ALU +20 SHF.) Bitwise equal. B=8: 5261 vs 5363 (generate 512),
5359 vs 5459 (pos 320-576), serve 5415 vs 5516.
- Sweep on s0009 (e0138/e0139): PMAXC 2 -23, SCNT_REL -8, BAR_SLEEP 64 -5, PUB_REL -4 (generate 512); PMAXC 2 + SCNT_REL at generate 1024: 5390 vs 5409/5425. Taken (a1's own switches).
## s0009 SCIM 8 — 2728 / 5141 (expected ~2725 / ~5130): confirmed (B=8 -90).
## s0010 PMAXC 2 + SCNT_REL 1 — 2726 / 5119 (expected ~5115): confirmed (B=8 -22). vLLM 5105: 14 us short of parity.
- After s0010 (generate 1024, 3 rounds, base 5390): BAR_SLEEP 64 +12, PUB_REL +2, ATM2 +20, FP8S (MXFP8 broadcast
  scales by IMAD) neutral, GNM (gated norms by mixer blocks) +55..+800. Not kept.
- Ring-shape sweep on s0010 (e0143): RS 3 x RI 6 +200, RI 12 +50, TPW 12 0. Not kept.
## FMA2: the NVFP4 block scales applied by packed fma.rn.ftz.f32x2 (two accumulators of one row per instruction)
ptxas emits FFMA2.FTZ taking the repeated scale directly (no extra moves): NVFP4 loop 500 -> 427 SASS instructions
per 40 HMMA (145 FFMA -> 72 FFMA2). Same per-element rounding as FFMA.FTZ: bitwise equal. B=8: 5197 vs 5236
(generate 512), 5358 vs 5390 (generate 1024). (a1 had measured packed FMA slower when the loop was ALU-bound.)
## s0011 FMA2 — 2727 / 5116 (expected ~5085): REFUTED as a gain (-3 over HTTP; evaluator in-process generate 4961 vs
4960 for s0010). Card 1 (README prompts, one engine per process) had -30..-40 twice. The loop is 427 instead of 500
instructions, so the NVFP4 decode loop is no longer what bounds those phases on the evaluator's workload; kept
(bitwise equal, not slower). B=8 is 11 us above vLLM's 5105.
## Direct evaluator tests of two switches (card-1 A/Bs within noise): s0012 PUB_REL 1: 2727 / 5118; s0013 BAR_SLEEP 64:
2725 / 5117 (s0011: 2727 / 5116). Both neutral; reverted. engine/ = s0011 configuration (only comments differ).
## Correction: s0011 did NOT contain FMA2. The `#define FMA2 0` insertion anchor (#ifndef W13H) no longer existed and
the later sed matched nothing, so FMA2 was undefined (= 0) in s0011-s0013: those three are the s0010 kernel. Hence
the "refuted" s0011 prediction. Fixed (define next to FP8D), checked by the SASS loop length (427) and bitwise equality.
## s0014 FMA2 (actually on) — 2728 / 5093 (expected ~5080): confirmed (B=8 -23 vs s0010/s0011).
Speedups: vLLM image 1.359x (B=1) / 1.002x (B=8); vLLM no-prefix 1.004x; the vLLM source build and -O3 screen
(5077 / 5078) 0.997x; SGLang 1.59x / 1.17x. The parity milestone is not passed: it counts every general engine,
including the vLLM source build at 5077 us (16 us faster than s0014 at B=8).
