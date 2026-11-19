# Journal (run s11-north-h100-b1)

Earlier runs' journal: notes/prev/journal.md. Measurements: eNNNN = mk exec job ids of this run.

## 0. Baseline and accounting (hour 0-0.5)
- dev reference: tools/make_ref.py (e0001, 8 sequences, HF bf16 vs float32 ref median 0.068).
- inherited engine, in process (e0004 B, e0005 A): see notes/gap.md and notes/plan.md.
- floor microbenchmark bench/floor.cu (e0007), skeleton build (e0008), per-stage trace (e0009): notes/gap.md.

## 1. Bit-identical track B (hours 0.5-2)
Escape census (study/esc_census.py, e0006, all weights): in the scaled space of the row window an escape needs an
exponent code -23..+22; max 6 escapes per (tile, lane), 4 per (tile, quantum, lane), 29 per tile.
Change: the widening adds a constant exponent bias of 32 (PRMT fills the high byte with 0x10 instead of 0x00), so every
escape is a normal bf16 V = W 2^(32 - r - c); escapes (and the 9 exact zeros) sit as code 0 in the tile (widen to
0x1000) and are XORed into the operand register before the mma. Row sums are undone by 2^(r - 128). Layer-0 tables are
made by the kernel itself (a DUMP instantiation runs layer 0 only for 8 token ids per launch).
- First version (patch code per k-step pair inside the unrolled decode loop): bit-identical to track A (e0012:
  bitcheck 1152 positions, B=1 and B=8, tables off: bit_identical 1.0 everywhere; e0013: all 30,484,201,472 weights
  widen to the exact operand through the kernel's own decode, mk_w12_widen) but B=1 generate 2126 -> 2705 us.
- Ablations (e0015-e0019, B=1 generate): full 2725; header loads but never patching 2050; no escape code 2052;
  patch code compiled but never taken 2039; preparation run, patches never applied 2040. So the cost is executing
  the patch code, not testing for it. SASS (e0021): the patch loop was replicated per unrolled pair at every inlined
  call site: 90k instead of 28k SASS lines per kernel instantiation -> instruction-cache misses whenever a patch runs.
- Fix: per quantum, one uniform test of a header bit; quanta without escapes run the inherited loop unchanged;
  the others widen all 4 pairs, patch in ONE compact loop (esc_patch), then the same mma sequence. The full head
  runs 16 quanta on one accumulator (same order of additions as one 128-k-step chain).
- Result of the fix (e0023-e0026): tables off B=1 generate 2118 / step 2200, B=8 step 5801; tables on (kernel-made)
  2113 / 2195, B=8 step 5778 (inherited: 2126 / 2209 / 5887). Bitcheck with kernel-made tables: bit-identical at B=1
  and B=8 (e0025); mk_w12_widen on all weights: 0 mismatches (e0026). Dev gates equal to track A's to the last digit.
  Load time 32 -> 52 s (32768 DUMP launches for the tables).
- s0001 submitted (track B bit-identical).

## 2. Skeleton accounting (e0008, inherited build)
| build (B=1, in process) | step | generate |
|---|---|---|
| full | 2209 | 2126 |
| no weight streaming | 1044 | 1238 |
| no streaming, no attention | 704 | 894 |
| no attention | 2164 | 2088 |
Attention is ~340 us of skeleton at B=1 but only ~40 us (80 at long context) on the critical path. The 704 us of
skeleton without attention is almost fully additive to the 1492 us stream (2088 = 1492 + 704 - 108).
=> the largest B=1 loss: HBM idle during the ~3 grid-wide exchanges per layer (x, union, h).

## 3. TMA bulk L2 prefetch on H100 (bench/tma_pf.cu, e0028)
- cp.async.bulk.prefetch.L2 stages data: a region read by 132 x 512 threads with 2 x 16 B in flight per thread after a
  bulk prefetch (any wait 2-20 us): 8.4 MB 2.56 us (cold 3.33), 16.8 MB 4.51 (6.02), 33.6 MB 8.42 (11.26), 50 MB 12.6
  (16.4). Chunk size 1-64 KB: no difference. (A read of a region just read is NOT faster than cold: 3.01 vs 3.33 us at
  8 MB - unexplained; the bulk-prefetched read is.)
- The interference half of the bench was flawed (background polled a flag every iteration); not used.
- Engine experiment BPF=1: after signalling stage C a block bulk-prefetches its next-layer stage-A share (123 KB).

### s0001 (track B, bit-identical): valid, http 2123 us B=1 / 5769 us B=8 (predicted ~2140 / 5950-6050; inherited 2150 / 6050)
In process: generate 2109 at B=1 (w1_like 2124). B=8 gained 281 us, far more than predicted from the teacher-forced
step (-109). Likely cause (not measured separately): the inherited after-sum escape fix read, in stage C at B > 1, the
x value of every escape straight from global memory (hx0/hx1, dependent loads per escape, 8 columns), and at B=8 the
served text routes differently from the teacher-forced one. Numeric gates: median 0.0808 single / 0.0704 batched,
numeric_long median 0.114 (limit 0.225), http_tokens 8/8 identical.

## 4. Measurements after s0001 (hours 1-2.7)
- Current trace B=1 (e0038): layer 40.8 us. Skeleton (e0039): no streaming 876 us/step, no streaming + no attention
  510; no attention alone 2112 vs 2110 full: attention is off the B=1 critical path now. Stream and skeleton add.
- Leader profile (e0040, us per layer, B=1): leader polls router rows 4.4, top-8 1.5, union + details 1.2; blocks wait
  for the union 0.33 only: stage A's qkv stream is as long as the routing chain. B=8 union 33.8 on that text.
- Skeleton trace (e0043): 12.1 us/layer; the leader's routing chain is critical at every stage there.
- Per-class trace (e0044): ordinary blocks stream gate|up + o_proj at ~17 GB/s per SM (2.3 TB/s total) while the
  32 attention blocks are busy (B=1: until +18 us of the layer; B=8: until +45, +67 at ctx 830).
- Bulk prefetch variants (abm.sh, same card): BPF=1 (stage A share after signalling C) B=1 -14..-24, B=8 +27..+70 ->
  kept for the batch-1 kernel only; BPF=2 (down tiles after gate|up) +90 B=1 / +440 B=8 (steals bandwidth from blocks
  still in gate|up); ATT_BPF (K/V of the task at task start) +6 / +46..+84. QDP=4 +17 B=1. NATT8=64 +160 B=8.
  Butterfly RMSNorm reduction +16 B=8 (+9 B=1 by code generation alone).
- Track A with BPF=1: +51 us at B=1 (e0054): BPF is on for the 12-bit build only.
- NORM_PROF (e0052): the x loads of the norm take 1.35 us (B=1) / 2.0 us (B=8) per layer; in isolation 0.45 / 0.6
  (bench/xread.cu, e0053; replicas of x do not help). Cause not found yet.
- bench/stream_bench.cu with N active SMs (e0047, e0060, e0061): plain reads reach 2.85 TB/s with 64 SMs, the 12-bit
  loop only 2.0 (32 GB/s per SM) regardless of loads in flight or software pipelining across quanta. Cause: the x
  operand's column stride (1024 words) put all 8 batch columns in one shared-memory bank. Padding to 1028: 64 SMs
  2.6 TB/s, 100 SMs 2.76, 132 SMs +2-4%.

### s0002 (track A rebuild): valid 2540 / 6901 (predicted 2540 / 6900). Numeric gates identical to s0001's digits.
### s0003 (track B, BPF=1 batch-1 kernel): valid 2108 / 5769 (predicted ~2105 / ~5770).
### s0004 (track A, padded operand strides): valid 2535 / 6823 (predicted ~2537 / ~6810).
### s0005 (track B, padded operand strides): valid 2099 / 5646 (predicted ~2108 / ~5620).

## 5. Hours 3.8-5: prefetch timing, head energy, paired attention chunks
- BPF in the batch-8 kernel (e0069): +19..+66 us. BPF issued at barrier release instead of after the signal (e0070):
  +70 B=1. BPF issued right after the norm's x loads (e0073): +78 B=1 / +37 B=8. Any bulk prefetch issued while the
  block itself is working (norm, stage A) costs more than it saves; only the barrier wait can absorb it.
- EARLY_U=2 (union load just before the stage-A signal): -4 us B=1 (noise). PADZ (zero operand in other banks): 0.
- XLD_CG (norm x loads bypass L1, NORM_PROF e0068): x loads 101 -> 89 us/step at B=8, 67 -> 75 at B=1. Not taken.
- Ordinary block profile (PROFBLK=50, e0071): see gap.md.
- Head energy (study/head_energy.py, e0074): no fixed subset of dimensions holds the final state's energy
  (notes/checkpoint.md section 2): no partial-dimension head sketch.
- Paired attention chunks (ATT_PAIR, e0075-e0078): an attention block runs chunk pairs on warps 0-7 / 8-15; the
  per-chunk arithmetic is now defined for 8 warps at every batch size (bit-identical A vs B, batch-invariant, e0077).
  Alone: B=8 short +14, long -98 (the share model still assumed 8 sequential passes); with the share model counting
  paired passes (ATT_PM): B=8 -62 / -61 / step -63, B=1 -5. Four blocks per task at B=1 (ATTB1=4): B=1 -20 short but
  +39 long: rejected.
### s0006 (track A, paired attention chunks): valid 2524 / 6745 (predicted ~2535 / ~6760).
### s0007 (track B, paired attention chunks): valid 2103 / 5612 (predicted ~2095 / ~5590). B=1 +4 against s0005 is
within the evaluator's spread (same-card A/B: -5); in process 2104 vs 2085 (s0005) on card 0.
- GS_A2 (per-K-tile attention cost in the share model) sweep (e0079): 8: B=8 long -61, short +4; 12: long -100, short
  +14. Now a separate batch>1 constant GS_A2M (e0080).
- GS_A2M sweep (e0080, 3 rounds): 12: B=8 long -97, short +19; 16: -84 / +23; 20: -62 / +27. Kept for later (a
  constant).
- NWPF (next layer's norm weights bulk-prefetched at stage C start, NORM_PROF e0081): norm x loads 68 -> 69 us/step:
  the slow x loads are not behind the norm-weight miss. Off.
- Leader publish time (trace event 11, e0082): the union's exchange returns at +6.8 us (B=1) / +11.0 (B=8); ordinary
  blocks signal stage A at +7.2 / +11.7 but see the union only at +8.2 / +12.7: their early load (issued with the last
  stage-A quantum) misses and the poll after the signal is one more round trip. UPOLL: warp 15 (idle while warps 0-2
  finish the stage-A tiles) polls the union until it is published or the finishing warps are done.
- UPOLL (e0083): B=1 +2, B=8 +24 (UPOLL=2). Trace of the UPOLL build (e0085): ordinary blocks still see the union
  ~0.95 us after their stage-A signal: the union (exchange returns +6.8) lands just after the poller's last load;
  the poller exits when the finishing warps are done. At most ~0.5 us/layer on this path. Off.
- Share-model sweeps (e0086-e0088, 3 rounds each): GS_DO1 26 +24 / 14 -2; GS_A0 18 +39 / 34 +2 (B=8 long -45, short
  +16); GS_DO 26 +23 / 14 +3: the B=1 schedule is at a local optimum of its constants.
- Norm x-load time in ordinary blocks (PROFBLK=50/120, e0089): the same as the leader's (65 / 98 us per step): not a
  leader effect.
- TLB hypothesis (bench/tlb.cu, e0092): reading x after streaming 0-256 MB of fresh pages takes 0.32-0.38 us (B=1) /
  0.48-0.51 us (B=8) whether x shares the barrier counter's page or not: no TLB effect. The engine's 1.35 / 2.0 us
  remain unexplained; at B=1 the bulk prefetches issued right before the barrier are the next suspect (e0093).
### s0008 (track A, GS_A2M=12): valid 2534 / 6753 (predicted ~2524 / ~6710): no B=8 gain on track A (s0006 6745).
### s0009 (track B, GS_A2M=12): valid 2109 / 5587 (predicted ~2100 / ~5575): B=8 -25 against s0007. B=1 (constant
not used there) 2099 / 2103 / 2109 over s0005 / s0007 / s0009: the evaluator's spread at B=1 is about +-5..10 us.
- Contract guards (api.py): submit() refuses token ids outside the vocabulary, empty prompts, n < 1 and requests
  longer than the context with ValueError (before: an out-of-range id read past the embedding table; an over-long
  request tripped an assert inside poll()); step()/generate() refuse batches outside 1..max_batch and positions past
  the context (before: K/V written past the sequence's slice). Token values in step() are not checked on the host
  (that would need a device sync per step).
- Pods lost repeatedly (epochs 7-12, clock stopped); jobs that died with two pods were not rerun (e0091, e0093,
  e0094, e0096) and were requeued.
- tools/batch_all.py (e0095, NORM_PART build): row 0 bit-identical to batch 1 at every batch size 2..8, generate ==
  step greedy at every batch size: PASS.
- NORM_PART (norm sums of squares from stage-C partials, e0098/e0100): +106 us B=1, +191 B=8 on track B, +111 /
  +224 on track A. Rejected (the named barrier + extra exchange sit on the stage-C tail, the most sensitive span).
- SKQ (head sketch loads in flight): 8 in every kernel (e0097): B=1 +4..+8, B=8 -27..-55. Per instantiation
  (SKQB=8 for batch > 1, e0098/e0100): B=8 -28..-53 (track B), -48..-58 (track A, which has no sketch: code
  generation of the batch > 1 kernel), B=1 unchanged. Kept; submitted as s0010 (A) / s0011 (B).
### s0010 (track A, SKQB=8): valid 2526 / 6688 (predicted ~2530 / ~6700).
### s0011 (track B, SKQB=8): valid 2099 / 5533 (predicted ~2105 / ~5550). Best B=8 so far.
- Norm x-load probe without BPF (e0101, ordinary block): x loads 62 -> 65 us/step: the bulk prefetch is not the
  cause either. SKQ for the batch-1 kernel (e0102): 2 -> +13, 6 -> +9: 4 stays.
- XVKEEP=0 (drop the dead 16 KB escape-value area: shared memory under the next carve-out step, L1 28 -> ~60 KB):
  e0103 / e0104.
- XVKEEP=0 (e0103): B=8 -98..-108, B=1 +17 (below ~200 KB of shared memory the SMs stream unevenly: prior run's
  finding). XVSPLIT (dead area at the end, allocated only by the batch-1 kernel's launch): track B (e0105) B=8
  -110..-116, B=1 +-2; track A (e0106) B=8 -91..-135, B=1 +4..+7. Bitcheck identical (e0107); serve tokens equal and
  batch_all PASS (e0108).
### s0012 (track A, XVSPLIT): valid 2526 / 6569 (predicted ~2526 / ~6590).
### s0013 (track B, XVSPLIT): valid 2107 / 5437 (predicted ~2100 / ~5430). Best B=8.
- Next probe: TB=9 / QB=144 (slot area 98 -> 72 KB: batch > 1 kernels one more carve-out step, L1 ~92 KB; the
  batch-1 kernel padded to its old size) (e0109).
- TB=9 / QB=144 (e0109): B=8 +21..+48 (a third gate|up batch costs more than the larger L1 gains). Rejected.
- NORM_PROF with a stamp after the norm weights' load (e0110): that load has returned (0.16 us/layer); x alone
  takes 1.16 / 1.9 us per layer at B=1 / B=8. Cause still unknown (not BPF, not TLB, not hot lines, not the weight
  load, not the leader, not L1 size).
- GSCACHE (share table recomputed only when ntc changes, e0111): B=1 generate -7, long -7; step 0; B=8 +-5.
- tools/long_ctx.py (e0112): positions 0..2047 at B=1 and B=8, row 0 bit-identical, finite; position 2048 refused;
  submit() refuses an out-of-range id and an over-long request.
### s0014 (track A, GSCACHE): valid 2517 / 6567 (predicted ~2519 / ~6569).
### s0015 (track B, GSCACHE): valid 2104 / 5408 (predicted ~2097 / ~5437); in process 2081 (lowest so far).
- Re-tests with the larger batch>1 L1 (e0113-e0115): XLD_CG B=8 -27..-36, B=1 -4/+1 (kept: s0016/s0017); QDP=4
  +8 / +40; ATT_BPF +26 / +21..+115.
- B1PAD (batch-1 kernel with 227 KB of shared memory): e0116.
### s0016 (track A, XLD_CG): valid 2517 / 6526 (predicted ~2517 / ~6535).
### s0017 (track B, XLD_CG): valid 2101 / 5392 (predicted ~2100 / ~5375).
- B1PAD (227 KB for the batch-1 kernel, e0116): +3 / -2: nothing.
- Trace e0117: see gap.md. The C->A hop at B=1 is 1.0 us with the full prefetch and 0.5 without.
- Prefetch amount (e0118, e0119, track B): router half tiles only +13; half tile + first whole tile (BPF=65) -24 /
  -28 / -22 against all three; no half tile +23; half of the first tile +5. Track A with BPF=65 (e0120): +8..+15.
### s0018 (track B, BPF=65): valid 2076 / 5384 (predicted ~2078). Best B=1.
- DPF (batch-1: after o_proj, the down tiles of the first 1/2/4 experts to L2): e0121.
- DPF (batch-1: after o_proj, before barrier B, the down tiles of the first k experts to L2; e0121): k=1 -17,
  k=2 -23 / -19 / -26, k=4 +11 (B=1 generate / long / step).
### s0019 (track B, DPF=2): valid 2051 / 5385 (predicted ~2055). Best B=1.
- Verification of the s0018 source (e0122-e0124): all 30,484,201,472 weights widen exactly (0 mismatches),
  bitcheck A vs B identical at B=1 and B=8, batch_all PASS (1..8), long_ctx OK.
- UPF (track-B behaviour: routing persistence): during the union wait thread 32 prefetches this block's share of the
  PREVIOUS step's union's gate|up tiles (1/2, 1 or 2 tiles): e0126. DPF on track A: e0125.
- DPF on track A (e0125): DPF=1 -7 / -11 / -4, DPF=2 -9 / -15 / -8 at B=1: a kernel change for both tracks.
- UPF (e0126): 1/2 tile -7, one tile -9 / -6 / -10, two tiles +19 at B=1.
### s0020 (track A, DPF=2): valid 2506 / 6525 (predicted ~2507 / ~6526).
### s0021 (track B, UPF=2 = one tile): valid 2044 / 5392 (predicted ~2042). Best B=1.
- HPF (next step's first weights to L2 after the head, batch-1 greedy path): e0127 (track B), e0128 (track A).
- HPF (e0127): 0 (+1..+3) for every variant: the step transition does not profit. Off.
- UPF on track A (e0129): -1..-3 (neutral). Two identical track-A builds in e0128 differed by up to 12 us (B=1
  long) and 20 us (B=8 long): the A/B resolution is about +-10 us.
- Batch > 1 versions of the barrier-wait prefetches (BPF=69, DPFB, UPFB): e0130. SPF (free blocks prefetch the
  second stage-A tiles once all but SPF blocks have signalled stage C): e0131.
- e0130: batch>1 prefetches: BPF=69 (stage-A half + first tile, also batch>1) B=8 -22 / -15 / -26; DPFB -27 / +7 /
  -30; UPFB +12 / +20 / +7. e0131: SPF +146..+204 at B=1 (the free blocks' prefetch at the stage-C tail lands on the
  next stage A of every block). Rejected.
- Trace e0134 (s0021 source): B=1 layer 39.6 us (40.75 at s0005), stage C quanta 6.8 (7.6). Critical: the ordinary
  blocks' o_proj (5.3 us after gate|up, +23.3 -> +28.6) while most other blocks are done.
- OQ2 (o_proj's two quanta with all loads issued first, e0136): correct (bitcheck e0137) but +34 B=1 / +45 B=8: the
  o_proj tail is not a per-warp round-trip problem. Rejected.
- e0132: BPF=69 (+batch>1) B=8 -22 / -19 / -26; + DPFB -46 / -15 / -48. Track A (e0133): DPFB -12 / +16 / -15
  (neutral); its "BPF=68" variant was a no-op (bit 0 is the enable bit) and serves as a noise estimate (+5..+8).
### s0022 (track B, BPF=69 + DPFB): valid 2040 / 5391 (predicted ~2044 / ~5350). In process B=8: generate -38,
step -44, w1_like (contexts to 1088) -9 against s0021: the gain is at short contexts; the HTTP figure follows the
long-context workload.
- e0138: EARLY_U=2 +16, UPOLL=1 +10 at B=1 on the current build. The union path is closed.
- HSKIP (study/dup_tiles.py, e0143/e0144): 439 head tiles of copies of earlier rows are skipped by the sketch pass
  (generate == step greedy at every batch size, e0144). A/B e0145.
- e0139: UPF=1, DPF=1/3 with all prefetches in place: within +-3 (B=1): amounts stay (UPF 2, DPF 2).
- Verification of the s0022 source (e0140-e0142): all weights widen exactly, bitcheck A vs B identical (B=1, B=8),
  long_ctx OK, served tokens equal step-by-step greedy (serve B=1 2030, B=8 5007 us/token).
- HSKIP A/B across builds (e0145): B=1 +9 / +1 / +12 (step +12, which HSKIP cannot affect: code generation), B=8
  -22 / -27 / -17. Same binary with MK_HSKIP=0/1 (e0146, 3 rounds): B=1 generate -14 / -14 / +4, B=8 -10 / -18 / -2.
### s0023 (track B, HSKIP): valid 2045 / 5320 (predicted ~2035 / ~5380). Best B=8 (B=1 within the spread of s0022).
- Class-adaptive prefetch amounts (DPFA: attention blocks and the leader prefetch all their down tiles; BPFA:
  attention, leader and free blocks prefetch their whole stage-A share): e0147. Re-checks: BPF=5 (whole share in
  every kernel) e0148; GS_A2M 16 / 8 e0149.
- e0148: BPF=5 (whole stage-A share, every kernel) +30 B=1 / +52 B=8. e0150: DPFA (attention blocks and leader
  prefetch all their down tiles) +62 / +75. e0151: BPFA (early classes prefetch the whole stage-A share) +14 / +20.
  Only the small fixed amounts pay. Pods lost twice more (epochs 13-16); e0147, e0149 died with them.
### s0024 (track A, final source): valid 2501 / 6493 (predicted ~2506 / ~6525).
- Determinism soak (tools/soak.py, e0152/e0153): 12 (track B) / 8 (track A) repeated served runs of 512 tokens at
  B=1 and B=8: 0 differ.
- Routing persistence measured from the kernel's routing tables (e0154): 4.77 of 8 experts shared with the previous
  token (notes/checkpoint.md section 4).
- e0155/e0156: SKQB 12 / 16 and GS_A2M 16 / 8 all equal or worse: current constants stay.
- Final verification of the final source (e0157-e0159): bitcheck A vs B identical (B=1, B=8, both batch-invariant),
  all weights widen exactly, batch_all PASS, long_ctx OK.
- The norm's slow x loads, last probes: the same x loaded a second time right after the first (NORM_PROF2, e0161):
  first 1.18 / 1.39 us per layer, second 0.33 / 0.71 (B=1 / B=8): a first-touch effect. Hand-over buffers colocated
  in the counters' 2 MB pages (MK_HOT, e0162/e0163): no effect on the step or on the first load (not address
  translation). bench/xread.cu was flawed (plain loads hit stale L1 lines; values never checked): with L2-only loads
  the first read after the barrier takes 0.47 / 0.60 us whether x was written by atomicExch, st + fence or st.cg +
  fence (e0164). x bulk-prefetched by each block right after its own signal (XPF, e0165): no change. Unexplained.
- Final source = s0023's behaviour (MK_HOT off by default, all probes off); serve check e0166: tokens equal.
### s0025 (track B, final source): valid 2050 / 5336 (predicted ~2042 / 5320-5390): within the spread of s0022/s0023.
- OSPLIT (batch-1: half of the o_proj tile moved into stage C, running sum carried so the additions keep their order;
  bitcheck identical e0168): +15 / +8 / +17 track B (e0167), +52 track A (e0169). Rejected: o_proj placement is at a
  local optimum like the share constants.

## Closing (agent-hour ~13.2)
Final engine = s0025 (track B; echo A > engine/TRACK builds track A = s0024's behaviour). All experiment switches in
mk.cu default to the evaluated configuration; their measurements are above. What I would do next: notes/summary.md.

## Continued (agent-hour 13.2 ->)
- NORM_PROF3 (e0170): 64 dependent ALU instructions right at the norm's entry cost 0.04 us/layer: instruction fetch
  is not why the first x load is slow. bench/tlb.cu with L2-only reads (e0171): 0-64 MB streamed before or after x
  is written leaves the first read at 0.33-0.38 / 0.50-0.53 us: no eviction or queueing effect. Still unexplained.
- EARLY_B (stage-B barrier poll issued with the last o_proj quantum, batch-1 kernel): e0172 (B), e0173 (A).
- EARLY_B (e0172 B / e0173 A): B=1 -3..-5 on both tracks (below the evaluator's resolution).
- OABL ablation (e0174, results wrong): ordinary blocks skipping the last 1/4 of their o_proj tile -19 us at B=1,
  1/2 -54..-65. The helpers' spare time (attention blocks ~2.7 us, free blocks ~10 us per layer) covers about 1/4:
  a K-split of o_proj would gain ~10 us after its exchange: not built.
- Share-model re-sweeps (e0175/e0176): GS_A0 22 +16 / 30 -8 (B=1), GS_DL 8 / 24 flat.
- EARLY_B + GS_A0=30 together (e0177, 4 rounds): track B B=1 -12 / -19 / -12, B=8 +9; track A (e0178) +4..+6.
  Made the 12-bit build's batch-1 defaults (GS_A01 = 30 at batch 1 only).
### s0026 (track B, EARLY_B + GS_A01=30): valid 2038 / 5334 (predicted ~2032 / ~5330). Best B=1 (by 2 us).
- e0180/e0181: GS_A01 34 / 38 +2..+17, GS_DO1 16 / 24 +3 / +20 at B=1: GS_A01 30 and GS_DO1 20 stay.
- AABL ablation (e0182, results wrong): ordinary blocks skipping their stage-A half tile -20..-26 us at B=1, most of
  it the ~2 MB per layer of bytes no longer read: moving the halves elsewhere would gain little. Not built.
- tools/serve_all.py (e0183): served cohorts of 1, 2, 3, 5, 7, 8 requests (prompts 17-64, n 40-64) stream exactly
  the tokens each prompt gets alone step by step: PASS. Bitcheck of the s0026 source (e0184): identical.
- e0185/e0186: GS_A2 (batch 1) 3 / 8 / 12 and GS_A1 96: all within +-7 us. Schedule constants are flat.
- Final soak (e0187-e0190): track B 2 x 40, track A 25 served runs of 512 tokens at B=1 and B=8: 0 differ;
  serve_all PASS three times.
- Not pursued (estimated below the evaluator's resolution): token hand-over at the step start (~12 us/step in all:
  consumers could read the layer-0 table rows themselves, ~3 us), free blocks taking one down unit per row tile at
  B=8 (~0.5 us/layer), 16-warp execution of the single batch-1 chunk (only helps beyond 1024 positions).
- ATTB1=4 on the current schedule (e0191/e0192): B=1 short -14..-15, step -16..-17, long +73..+75: loses on the
  HTTP workload (contexts to 1088). A context-switched variant (4 below ~400 positions) would gain ~3-5 us there.
