# Journal (s12-oss20-h100-a1)

## 0.0-0.6 h: port
- Base: the sibling run's gpt-oss-120b B200 engine (s12-oss120-b200-a1 s0005, same layer design: hidden 2880,
  64/8 heads, expert intermediate 2880). Ported: NL 24, NE 32, NB 132, sm_90a.
- e2m1 decode on sm_90 (no cvt.e2m1x2): PRMT + SHF + LOP3 per register -> bf16 bits sign<<15|mag<<6 =
  code * 2^-126 (subnormal for e=0). bench/subn.cu: mma.bf16 accepts subnormal A, HMUL2 keeps subnormals.
  MoE inputs stored * 2^63 (exact), group sums * 2^(S-64) in fp32 (exact); router undoes 2^63 (exact).
- Bug hunt (~1 h): with RS=3 or 4 TMA ring stages the per-warp task table got overwritten with weight bytes
  during the first consume (heisenbug: printf/debug builds hid it; createpolicy vs hard-coded L2 policy word,
  .shared::cta vs ::cluster, volatile smem loads: no change). RS=2 runs clean. Cause not found yet.
- First timings (tests/timing.py, card 3): step B=1 2270 us, generate B=1 2335 (ctx 64-1088), serve B=1 2333;
  step B=8 4659, generate/serve B=8 5890-5902.

## s0001 (0.7 h): port. Expected HTTP 2345 / 5900; measured 2314 / 5745 (valid; numeric median 0.0110).
## s0002 (1.4 h): whole-unit MoE warp tasks + last-arriver down combine + 12 warps + cheaper decode
- Expected B=1 ~2255, B=8 ~4200; measured 2275 / 4284 (valid, top-1 vs fp64 0.981). Both beat 1.10x vLLM.
- Ablations (e0037, e0041, same card): NOCVT (no e2m1 decode) gen B=1 2186 -> 1841, B=8 4278 -> 2298;
  SKEL (weights from L2) 1659 / 2129. The decode is the B=8 bottleneck (~2 ms of 4.3).
- SASS (e0044): 85.7 instructions per 64-column item (36 LOP3, 23 IMAD, 9 SHF, 8 FFMA, 4 LDS, 4 HMMA): the
  decode needs two masks per register (one immediate per LOP3), so ~4 ops per bf16x2 register; the ALU pipe
  (LOP3/SHF at half rate) is the limiter. No bit layout lets a register decode with 1-2 ops (tests: no rotation
  or shift tiling of the target bits exists).
- step() timing with a repeated token understates B=8 (fewer experts): use generate/serve timing.
- Norm as one warp per sequence, two 12-chunk passes of loads in flight, __noinline__ (A/B e0054, same card):
  B=1 2218 -> 2213, B=8 4336 -> 4053 (inline variants cost B=1 +40..130 us through register pressure).
- RS>=3 ring bug, isolated in bench/ring.cu (standalone): consumer holds the next task's descriptor when the ring
  has 3+ stages; -G and -Xptxas -O0 pass, -O1/-O3 fail; a volatile re-read of the descriptor fixes the standalone.
  The engine still fails at RS 3/4 with volatile TT and PState accesses: parked, RS=2 kept.

## s0003 (2.0 h): faster norms + down combine with per-sequence slot lists
- Expected ~2270 / ~3850; measured 2275 / 3853 (valid).
- DSPLIT=2 (down units split in K halves): B=1 no gain (2224 vs 2217), B=8 +320: the MoE phases at B=1 are
  bound by decode instruction issue per SM (~86 instr/item), not by the number of warps streaming.
- Router on 45 blocks (4-kstep slices), every block sums the 45 partials in order and does top-4 (e0081):
  B=1 -12 us, B=8 -52 us (expected -70 at B=1).
- Dead end: qkv/o_proj as (rowtile, segment) units balanced over blocks with a last-arriver finish of rowtiles
  split between two blocks (e0083-e0086): bit-identical, but B=1 +220..+410 us (fences under load, serialized
  epilogue passes, codegen with generic lambdas). Reverted.
- Dead end: cp.async.bulk.prefetch.L2 of the block's o_proj rows during attention (e0087): B=1 +75 us
  (the prefetch occupies the SM's queue when o_proj starts; HBM-idle windows are too short per SM).

## s0004 (2.6 h): router on 45 blocks. Expected ~2263 / ~3800; measured 2254 / 3850 (valid).
## s0005 (3.0 h): TRACK B certified int8 head sketch (greedy path), candidate tiles spread over all blocks.
- Study (tests/sketch_study.py, real text): int8 rows leave median 1 / p90 7 / max 40 candidate rows of 201088;
  int6 median 52, int4 useless. Candidates per step in generation (skstat): 0.7-17 tiles, up to 88 in one block
  -> pass 2 needs the global list (one more grid barrier; pass-2 tail 60-120 us -> ~20 us).
- Soak: 54/54 sequences x 256-512 tokens identical to exact step greedy. Pass 1 205 us (2.8 TB/s).
- Expected ~2050 / ~3680; measured 2061 / 3671 (valid). Best overall.
- Dead end again: qkv/o split rowtiles with tagged prefix exchange (no fences): bit-identical, B=1 +165 us
  (+108 even without waiting). Per-warp stage count (RS=2 pipelines) seems to set phase time, not block balance.
- RS>=3 bug solved: `volatile int ct` in consume_tasks (bench/ring.cu: V2 passes at -O3; engine RS=3 at
  -Xptxas -O0 matched RS=2 logits). Ring shapes at equal smem (e0104): (RS,RI) (2,9) 2219, (3,6) 2235,
  (4,4) 2310, (6,3) 2352 us B=1: fewer, larger stages win; more bytes in flight per warp not testable without
  shrinking the activation buffers (attention scores alias them).
- Dead end: attention reworks (e0106-e0120). Profile at ctx 1000: attention 13.8 us/layer (5.1 at ctx 64) on
  8 blocks. (a) split-KV with 64-key chunks on all blocks: B=1 -10, B=8 +645 (per-item round trips);
  (b) unrolled K/V loads (4 groups in flight): +95 (no change out of line either);
  (c) balanced split-KV (16 chunks at B=1, smem-staged combine): ctx-1000 phase 10.6 us but ctx-64 9.1 us; net 0;
  (d) + chunks >= 128 keys + bulk K/V prefetch into smem before the barrier: ctx-1000 9.8, ctx-64 5.9; net +60.
  Attention after the barrier is a chain of 3-5 dependent global round trips; reverted to the 8-block version.
- Limits (tests/limits.py, e0138): token ids outside [0, vocab) are replaced by 0 in the kernel and raise
  ValueError at the latest one call later (mapped flag, no sync); batch 0/9, context end for step/generate/
  submit, empty prompt, zero tokens all raise; serving at exactly the context edge works.
- Attention 16-byte K/V layout (s0006, track B): expected ~2025 / ~3630; measured 2037 / 3641 (valid).
- Attention score unroll AU=2/4 with the new layout: -4 us (noise). Norm over all warps: B=1 -15 us,
  B=8 +170 (teacher forced) -> not kept.
- Deeper rings with room made by small activation buffers (B=1 prototype, SCS=512, e0141, teacher forced):
  RS=2 RI=9 2187-2192, RS=3 RI=9 2239-2245, RS=4 RI=7 2253 us: more bytes in flight per warp is slower.
  The MoE phases are issue-bound, not latency-of-memory-bound. Direction closed.
- Dead end: evict_last bulk L2 prefetch of the next layer's qkv during attention/router windows (e0142):
  B=1 +145 us (teacher forced). L2 staging closed (3 attempts, all slower).
- ncu + standalone phase benchmarks (see findings.md): decode is not the B=1 limit; the HBM burst ramp of the
  routing-dependent MoE phases is. Ideas that only add bytes in flight do not help.
- Plain ld.global warming of the block's o_proj rows during attention (e0150): neutral. With SKEL showing o_proj
  7.3 us from L2 vs 8.8 real, the dense phases are skeleton-bound, not HBM-bound.

## s0007 (track A, KV layout): expected ~2220 / ~3810; measured 2237 / 3819 (valid).
- DSPLIT=2 again (teacher forced, e0152): B=1 0, B=8 +90. Down at B=1 is not short of warps.
- Down counter: one atom.acq_rel by lane 0 after __syncwarp instead of __threadfence in every lane + atomicAdd
  (e0153): B=1 -3, B=8 -18. Grid barrier: red.release.gpu add instead of __threadfence + atomicAdd (e0155,
  with the counter change): B=1 -19, B=8 -37. Logits unchanged.
- Greedy token choice at step start: one warp per sequence with all 132 candidate loads in flight and a
  shuffle reduction, instead of one thread scanning 132 (value, index) pairs (e0157, with the above):
  serve B=1 2206 -> 2163, B=8 3827 -> 3789.

## s0008 (track B): release atomics + warp argmax: expected ~1995 / ~3600; measured 2026 / 3599.
## s0009 (track B): routing block in registers: expected ~1940 / ~3450; measured 1992 / 3457.
- (Same-card teacher-forced gains overstate HTTP gains at B=1 by ~2x for these sync changes.)
- Epilogue preloads into registers before the stream: +18 us; gate_up bias staging in smem (+1 block sync):
  +25 us. Each extra block-wide sync + small loop costs ~1 us per layer.
- Attention flash-decoding inside the block (online softmax per warp over its 16-key blocks, P from score
  registers into the PV mma, one combine with the sink): 2 block syncs instead of 5. Teacher forced ctx 64-1064
  (e0165): B=1 2154 -> 2127, B=8 3450 -> 3422. Software-pipelined K/V loads in it: +62 us (spills).
## s0010 (track B, flash attention in block): expected ~1970 / ~3430; measured 1935 / 3411 (valid).
- Down unit/row tables built inside the routing warp (one block sync less) + staging pair table (e0168):
  B=1 -17, B=8 -63 (teacher forced). Committed.
- Epilogue inputs by cp.async into smem before the stream (qkv biases/cos/sin, o biases/residual): +30 us.
  Third failed epilogue-preload variant: the epilogue's first global loads are not worth moving.
- Norm with the raw row stashed in smem (one L2 pass): +370 us (e0177), reverted.
- Routing tail split over warps 0-2 (union/weights | per-sequence lists | down tables), each deriving the
  union from the selection masks (e0178): B=1 -12, B=8 -17.

## s0011 (track A catch-up): expected ~2125 / ~3590; measured 2072 / 3509 (valid).
- NSEG_QKV 4 / NSEG_O 6 (segments of exactly 5 ring stages): B=1 -6 (e0181); 5/8: +33.
- Sketch pass-1 loads in flight SKQ 9 -> 15: -10 us (e0182, track B). 6: +6, 18: same as 15.
- Attention with two 16-key blocks' loads in flight per iteration: +57 (e0183). Round-robin sketch tiles
  (no candidate list, one grid barrier less): +7 (e0184). Both reverted.

## s0012 (track B): routing tail / segments / SKQ: expected ~1910; measured 1896 / (B=8 see board) (valid).
- Expert prediction: the router applied to the layer input (normed with ln1, weights Wr*ln2/ln1) predicts
  3.5 of 4 experts per layer (tests/predict_route.py, real text; previous token: 1.65).
- Speculative gate_up prefetch at B=1 (bench/mk_spec_attempt.cu): predictor partials as extra phase-A tasks
  on the router blocks, predicted top-4 at phase-C start, each warp's first 2 ring stages of its unit for the
  predicted experts issued at the end of phase C, kept if right else drained (router moved to warp 11).
  Correct (B=1 with speculation bit-identical to B=8 without) but B=1 +180 us, B=8 +80 (e0190): the
  machinery (prediction round trip on the critical path, router on one warp, traffic during phase D) costs
  more than the hidden ramp. First version deadlocked: a warp re-consumed a stale task state.
- Sketch width study (tests/sketch_study.py, real text): candidate rows int8 median 1/p90 7/max 40; int7 5/28/165;
  int6 52/285/2245; int5 2230/25k/88k.
- 6-bit certified sketch (4 codes per 3 bytes in fragment order, unpacked to int8 = 4q in registers; 434 MB
  instead of 579): soak 27/27 identical; candidate tiles per step 43-51 at B=1, 275-315 at B=8 (skstat);
  gen B=1 1854 -> 1834, B=8 3531 -> 3518 (e0198). Expected -45; pass 2 grew.
- L2 warming of the heavy qkv blocks' third rowtile by light blocks (plain loads, e0200): +80 us.
## s0013 (track B, 6-bit sketch): expected ~1876 / ~3375; measured 1863 / 3333 (valid). 1.80x vLLM at B=1.
- qkv pairing (bench/mk_qpair_attempt.cu): each 3-rowtile block hands K-segments 2-3 of its third rowtile to a
  paired 2-rowtile block (both 2.5 rowtiles), half-sum published as tagged words, every rowtile summed
  (s0+s1)+(s2+s3). Correct, batch-invariant; B=1 +100 us (e0201). Third failed qkv balancing: closed.
- Down tail ablation (no counter/combine, results wrong, e0205): B=1 -40 us. Flat combine for B<=2 at the next
  norm by every block (bench/mk_flatc_attempt.cu): bit-identical to the last-arriver path at every B (bcheck), but
  B=1 +38..44 us, B=8 +50 (e0208). Reverted.
- SKQ sweep (e0213/e0214): track B gen B=1 9: 1850, 15: 1833; track A tf neutral. SKQ must divide 90 (12 read past the tile: guard added).
- Layer-0 per-token q|k|v table (track B, built by the kernel itself in a mode that runs layer 0 phase A only;
  bench/mk_l0tab_attempt.cu): bit-identical to track A (bitcheck 1.0 at B 1, 8; soak 18/18) but gen B=1 +25 us,
  tf +17 (e0217) although it removes a norm and a 29.5 MB stream: added code costs the kernel's hot loops more.
- Hypothesis (before measuring): track B router fold. logit_e = rinv * sum_k (ln2_k Wr_ek) x_k + b_e is the HF router
  in exact arithmetic; products ln2_k*Wr_ek are exact in fp32. Each block computes its o_proj rows' partial sums
  (32 experts + sum of squares) in phase C's epilogue; phase E sums the 132 partials (warps NW-1-b) while warps < B
  run norm2. Removes phase D (one grid barrier, the 45-block router stream, norm2 then routing in series).
  Expect gen B=1 -50 us (D span 5.4 + routing 2.9 -> ~max(norm 3.6, routing ~3) + C epilogue +1), B=8 -60.
- Router fold measured (e0223, card 2, 2 rounds): gen B=1 1869 -> 1814 (-55, predicted -50), tf_b1 2053 -> 2003;
  B=8 gen 3434 -> 3434, tf 3429 -> 3439 (predicted -60: wrong; at B=8 each routing warp sums 132 x 32 partials
  (17 KB/sequence/block, 3x the old 45-slice traffic) and warps 4-7 route then norm in series). Preloading the
  folded weights before the epilogue (RFPRE): spills, +50. bcheck B=2..8 bitwise vs alone, long ctx ok (e0226);
  soak 27/27 identical, limits ok (e0227).
- Hypothesis (before measuring): norms from sum-of-squares partials (NSS, track B, on RFOLD): the down combine
  writes per-(rowtile, sequence) sums of squares of the x rows it stores, phase C those of xm; norm1 (layers > 0),
  norm2 and the final norm then read the row once instead of twice. Expect -1 us per norm: gen B=1 -40..-50.
## s0015 (track B, router fold): expected ~1815 / ~3330; measured 1813 / 3328 (valid; numeric median 0.0103).
- NSS single-pass norms (e0229): serve B=1 1814 -> 1803 (predicted -40..-50: wrong, -12); tf_b8 neutral; gen/serve
  B=8 +270 on random prompts (different trajectory -> wider expert unions; tf is the fair B=8 number).
- Hypothesis (before measuring): NSSW = NSS norms spread over all 384 threads (2 chunks each, one round of loads,
  no cross-warp reduction needed). norm1 span 3.35 -> ~1.5 us: gen B=1 -40; E_entry unchanged (routing bound).
- NSSW measured (e0234): gen B=1 +25, tf B=8 +350. Fifth failed norm variant: spreading the norm over all warps
  loses even without a cross-warp reduction. Removed.
- Accuracy on the evaluator's real text: s0015 numeric median 0.0103 (s0013 0.0115), long median 0.0168 (0.0169),
  top-1 0.975 (0.982; limit 0.948). Random-token check.py T=1000 (e0231-3): A / RFOLD / RFOLD+NSS B=8 median
  0.0828 / 0.0862 / 0.0866, top-1 0.896 / 0.893 / 0.885 (random tokens: near-ties everywhere; noisy).
- Head profile, track B gen B=1 ctx 300 (e0236): pass 1 169 us, threshold/list 5.6, exact recompute of candidate
  tiles 26 (work 12.5, imbalance 6.4). The recompute's tile load was a load->store loop (15 dependent round trips).
  Hypothesis: all 15 loads per thread in flight (P2B): pass 2b 26 -> ~16, gen B=1 -10, B=8 -20 (2-3 tiles/block).
  Measured (e0237): gen B=1 1817 -> 1816, serve +6, B=8 +10. Wrong: the loads were not the cost. Reverted.
- Hypothesis (before measuring): sketch pass 1 is latency-bound per warp (45 loads, wait, math; 0.84 IPC/SM,
  2.57 TB/s). Double-buffering batches across tiles (SKPF) overlaps a warp's loads with its math: pass 1 169 ->
  ~150 us, gen B=1 -15..-20.
  Measured (e0238): register double-buffering spills (256-272 B stack): gen B=1 1815 -> 2383 (SKQ 15), 1941 (9),
  1991 (10). Reverted. A version without register cost would stream the sketch through the TMA rings.
- Hypothesis (before measuring): SKCA = sketch pass 1 via cp.async into 3 per-warp smem buffers of 5 k-steps (2
  batches ahead, across tiles; no registers held). Same expectation as SKPF: pass 1 169 -> ~150, gen B=1 -15..-20.
  Measured (e0239): gen B=1 1816 -> 1814, B=8 +3: no gain; pass 1 (~158 us of the 169 span, ~2.75 TB/s) is not
  per-warp latency-bound. Reverted. Sketch pass 1 closed unless the byte count changes.
- Sketch with exact outlier dims (tests/sketch_study2.py, e0235): the top hidden dim holds 20% of |x|^2 (top 16:
  29%, top 256: 50%). Candidates (mean rows): int6 130 -> 59 (16 exact dims) -> 27 (64); int5 7476 -> 1937 -> 904.
  int5 + 16 exact dims would be 366 MB instead of 434 (-23 us of pass 1) but pass 2 would need per-row recompute
  (~2000 scattered rows mean, 23k max). Not pursued yet.
- Hypothesis (before measuring): attention loads are one dependent HBM round trip per 16-key block per warp (~5-6
  at ctx 1000). Bulk L2 prefetch of the block's K/V window by the attention blocks at phase-A start (KVPF) makes
  them L2 hits: gen (ctx 64..1088) B=1 -20..-25 us, B=8 similar or more.
  Measured (e0240): gen B=1 1812 -> 1795, serve 1813 -> 1798 (-14); B=8 +18..+21 (64 blocks prefetching compete
  with the weight stream). Kept for B <= 4 only.
  Interleaved 3 rounds (e0242, card 6): serve B=1 1814 -> 1801, gen 1817 -> 1798; B=8 +12 whether the prefetch runs
  at B=8 or not (gated at B<=4 or ungated give the same B=8 time): a code-layout cost, not the traffic.
  (e0241 on card 5 showed no B=1 gain once: single-run outlier, not reproduced.)
## s0016 (track B, K/V L2 prefetch): expected ~1800 / ~3340; measured 1802 / 3343 (valid).
- NSS on s0016 (e0243, 3 rounds interleaved): serve B=1 1805 -> 1779, gen 1798 -> 1775, tf_b1 2010 -> 1989;
  tf_b8 neutral; gen/serve B=8 +255 on timing.py's random prompts (different greedy trajectory). Checks clean.
- Hypothesis (before measuring): AFLAG = no grid barrier between phase A and attention; blocks count finished
  q/k/v rowtiles per kv head (release add), attention blocks wait for their 40. Attention starts when ~17 producer
  blocks are done instead of 132, one barrier less per layer: gen B=1 -25..-50.
- Hypothesis (before measuring): DPAIR = each down unit's K split over a warp pair of the same block (partials
  added in smem, half 0 + half 1). At B=1 only 5-6 of 12 warps per block had down work; pairs request ~85% of the
  down bytes before the E->F barrier instead of ~40%: down 13.3 -> ~10 us/layer, gen B=1 -50..-70; B=8 ~0.
## s0017 (track B, NSS norms): expected ~1780; measured 1777 / 3353 (valid; median 0.0106, top-1 0.975).
  AFLAG / DPAIR measured (e0244, 2 rounds, gen B=1 / tf B=8): both off 1785 / 3444; DPAIR only 1785 / 3545;
  AFLAG only 1787 / 3461; both 1780 / 3560. Both predictions wrong: the A->B grid barrier is not what delays
  attention (its producers finish together), and splitting down units over warp pairs buys nothing at B=1 (the
  down phase is not limited by requests in flight) while costing +100 at B=8. Reverted (bench/mk_aflag_dpair_attempt.cu).
- Norm weights loaded with the row in the NSS norm (NSSWP, e0246): gen B=1 +30, tf B=8 +50. Reverted.
- Grid barrier with per-block flags polled by one warp (bench/gbar2.cu, e0247): 2.34-2.55 us vs 1.05 for the release-add counter. Closed.
- Pass-2b mma chain split in 4 accumulators (timing probe, e0248): gen +4, pass 2b span 26 -> 23. The chain is not the cost. Closed.
- Sub-phase profile (e0249, B=1 ctx 300, us/layer): A = norm1 3.4 + qkv stream 8.0 + epilogue 3.2 (+1.9 wait);
  C = attn load 0.9 + stream 6.8 + epilogue 1.8 (+1.8 wait). Epilogue loads start after the block barrier.
- Hypothesis (before measuring): EPL = each thread's epilogue inputs (qkv bias + cos/sin, o bias + residual) loaded
  after its warp's stream and before the block barrier (registers held only across the barrier, unlike the three
  failed pre-stream preloads): A epilogue -0.8, C -0.5 us/layer: gen B=1 -25..-30, B=8 -10.
  Measured (e0250/e0251, 3 rounds, gen B=1 / tf B=1 / tf B=8): none 1774 / 1982 / 3443; qkv part only 1764 / 1968 /
  3441; o_proj part only 1767 / 1971 / 3443; both 1762 / 1958 / 3455. Prediction too high (-25..-30 gen; got -10).
  Kept the qkv part (EPLA) only.
## s0018 (track B, EPLA): expected ~1768; measured 1766 / 3354 (valid).
- Hypothesis (before measuring): GBPF = prefetch.global.L1 of each gate_up unit's bias line when its task is
  built (after routing): the epilogue's bias load becomes an L1 hit; gen B=1 -10..-15 (one round trip per layer).
  Measured (e0252): gen B=1 -1.5, tf noise. The gate_up bias load is not on the critical path. Reverted.
- Hypothesis (before measuring): CPF = L1 prefetch of the o_proj epilogue's bias and residual lines at phase-C start (no registers, unlike EPLC): gen B=1 -5, B=8 0.
  CPF measured (e0253): gen B=1 +5. Reverted.
- Down-phase marks (e0255, B=1): warp 0's unit stream ends ~10 us after phase start, ypart + acq_rel atomic ~1 us,
  combine 1-2 us. MoE phases look HBM-burst-bound (gate_up 35 MB ~2.1 TB/s, down 17.6 MB ~1.7 TB/s), and the
  down burst starts only after gate_up's.
- Hypothesis (before measuring): DNPF = bulk L2 prefetch of each block's down units (known after routing) at
  gate_up start, by warp 11: one 52 MB burst instead of two; gate_up a little slower, down from L2:
  gen B=1 -40..-70; B=8 unclear (union 15 experts: 70 MB of down weights > L2 share).
  Measured (e0256): gen B=1 1763 -> 1875 (+112), B=8 tf +315. Reverted. Fifth failed L2 prefetch of weights
  (o_proj during attention, qkv warming, next-layer qkv evict_last, down units here); only small prefetches (the
  attention window's K/V, <= 2 MB per layer) have paid.
- Norm ablation (e0257, timing only): without norm-weight loads norm1 3.42 -> 2.86 us; without partial loads unchanged.
- Hypothesis (before measuring): NSSW1 = all-warps NSS norm for norm1 and the final norm only (NSSW lost +25 with norm2 queued behind routing): norm1 3.4 -> ~2: gen B=1 -30.
  Measured (e0258): gen B=1 +122, B=8 +320. norm_ss compiles with spills (24 B) and its call sites add saves; closed for good.
## s0019 (track A catch-up): expected ~2055 / ~3510; measured 2063 / 3520 (valid).
- Hypothesis (before measuring): ACP = attention K/V staged per warp by cp.async into shared memory: the first key block's copy runs with the q load and block sync, the next block's under the current block's math (no extra registers): attention -1 us/layer at ctx ~576: gen B=1 -20, B=8 -20.
  ACP measured (e0261): gen B=1 1760 -> 1799 (+38), tf +5. Reverted.
- Sketch pass-1 byte scaling probe (e0262): reading half of each tile: head span 167 -> 94 us (~21 fixed + ~146 proportional to bytes).
- int5 sketch + exact high-energy dims (tests/sketch_study3.py, e0263, T=128, n=512): 16/32/48/64 exact dims:
  366/371/375/380 MB; candidate rows mean 2290/1974/1517/1333 (p99 18k/16k/13k/12k); B=1 pass-2 rounds 1.8/1.6/
  1.4/1.35; B=8 5.5-9 rounds. Net B=1 ~ -15 us, B=8 +40..+70: not pursued.
- Decode cost now (e0264, timing-only NOCVT): tf B=1 1979 -> 1651 (-330 us), B=8 3447 -> 1970 (-43%).
- bench/dec.cu (e0265-e0269): the engine's item (64 columns: 4 code words, 1 scale word, 2 B loads, 4 HMMA, 8 FFMA)
  compiles to ~98 SASS instructions, ~75 of them decode. With 9-item unrolled stages one warp per scheduler takes 133
  cycles/item, two take 244 (near 1 IPC per scheduler): the MoE item loop is issue-bound. Decode by masked multiply
  (nibble pair x (2^12+2^6) or x (2^8+2^2), no carries, one mask) is bit-identical and 7-8% faster per item.
- Hypothesis (before measuring): MULDEC in the engine (both tracks, same bits): MoE phases ~7% faster:
  gen B=1 -20..-30, B=8 -100..-150; logits bit-identical to before.
  Measured (e0270, 3 rounds): gen B=1 1769 -> 1761, tf B=1 1979 -> 1968, tf B=8 3440 -> 3410, gen B=8 3704 -> 3680. Same results (check identical). Prediction too high for B=1, too high for B=8.
## s0020 (track B, MULDEC): expected ~1758 / ~3325; measured 1752 / 3331 (valid).
- Hypothesis (before measuring): AKV1 = each attention warp's first K/V block loaded into registers before the q exchange (block sync): one L2 round trip less per layer on the attention path: gen B=1 -15.
  AKV1 measured (e0276): gen B=1 +60, B=8 +65. Reverted (attention: any extra live state loses).
- Hypothesis (before measuring): SC63 = MXFP4 scale bytes stored as S+63 (track B, lossless re-encoding; max S+63 asserted <= 255): one add less per item: gen B=1 -3, B=8 -8.
  SC63 measured (e0277): gen B=1 +1, B=8 -4: noise. Reverted.
- Attention by context (e0278, B=1): span 4.6 us/layer at ctx 64, 7.6 at ctx 1000 (full layers ~10.6).
- Routing probe (e0279): reading a third of the router partials changes E_entry by 0.17 us: norm2 (~3 us) is the long pole there.
- Broadcast read probe (bench/bcast.cu, e0280): one warp per block reading the same freshly written 5.8 KB row after a grid barrier: 1840 cycles (~1.0 us); distinct rows 2900. No L2 hot-spot penalty.
- NORM_INLINE (e0281): 272 B stack / 676 B spill loads, gen B=1 +120. Register pressure binds the whole kernel.
- Hypothesis (before measuring): NSS4 = each NSS norm row split over 4 warps (6 chunks each, partials summed per warp in the same order: same bits), fewer live registers per warp and 4x parallel compute: norm1/norm2 3.2 -> ~1.8 us: gen B=1 -50..-60. Caveat: norm2 now shares warps 0-3 with nothing at B=1 (routing on warp 11), but at B>=3 the routing warps (NW-B..) overlap norm tasks.
  Measured (e0282): gen B=1 1758 -> 1729, tf B=1 1963 -> 1923; tf B=8 3421 -> 3489 (+70: norm tasks on the routing warps). Restricted to B <= 2.
  Gated (e0283, 3 rounds): gen B=1 1763 -> 1739, tf B=1 1964 -> 1916, B=8 tf -5; bcheck/soak clean.
## s0021 (track B, NSS4): expected ~1728; measured 1734 / 3322 (valid).
- Hypothesis (before measuring): RFPF = L1 prefetch of the block's folded router rows (4 KB, HBM-cold) at phase-C start; the epilogue's rf loads become L1 hits: C epilogue -0.7 us/layer: gen B=1 -15.
  RFPF measured (e0285, 3 rounds): gen B=1 1739 -> 1729, tf B=1 -8, B=8 -10. Kept.
- Hypothesis (before measuring): DBPF = L2 prefetch of the down combine's bias rows (all routed slots, the block's down rowtiles) at gate_up start: the combine's HBM round trip becomes L2: gen B=1 -10..-15.
  DBPF measured (e0286): gen B=1 1727 -> 1722 (consistent), tf B=1 +3, B=8 +8. Kept for B <= 2.
## s0022 (track B, RFPF + DBPF): expected ~1720; measured 1721 / 3331 (valid).
- Hypothesis (before measuring): SVPF = bulk L2 prefetch of the next layer's norm weights, biases, sinks and router bias (~55 KB) at down-phase start by 5 blocks: norm/epilogue first reads become L2 hits: gen B=1 -10..-20.
  SVPF measured (e0287, 3 rounds): gen B=1 1725 -> 1712, gen B=8 -15, tf mixed (B=1 -3). Kept.
## s0023 (track B, SVPF): expected ~1709; measured 1713 / 3315 (valid).
- Hypothesis (before measuring): HDPF = L1 prefetch of each sketch tile's 256-B bound header at tile start (it was a dependent cold load after the tile's math, ~8 per warp): pass 1 -3..-8 us.
  HDPF measured (e0288): gen B=1 -1.3, B=8 -3: noise. Reverted.
- Hypothesis (before measuring): KVPFE = the K/V window prefetch of layer l+1 issued at layer l's down-phase start instead of layer l+1's phase-A start (more time to land): gen B=1 -5..-10 (long contexts).
  KVPFE measured (e0289): -1.3, noise. Reverted.
## s0024 (track A catch-up): expected ~2035 / ~3490; measured 2035 / 3480 (valid).
- qkv imbalance probe (e0291, timing only): capping blocks at 2 qkv rowtiles (-17.5% bytes): qkv stream 8.28 -> 7.10,
  A-phase barrier wait 3.44 -> 3.53. The wait is service variance, not the 3-vs-2 rowtile split; 8-row
  granularity (weights as the B operand) would gain <= ~0.6 us/layer. Not pursued.
- Hypothesis (before measuring): NSS (+NSS4) in track A (sum of squares accumulated where the rows are produced; same fp32 formats and function, only the summation order and place differ): track A gen B=1 -40..-50 (as in track B: -21 then -24).
  Measured (e0292, 3 rounds): track A gen B=1 2036 -> 1954, tf B=1 2035 -> 1949, tf B=8 +12. NSS now in both builds (FLAGS).
## s0025 (track A, NSS): expected ~1955; measured 1952 / 3487 (valid).
- Hypothesis (before measuring): OPF = during attention, each warp L2-prefetches the o_proj bytes of its tasks beyond its two prefetched ring stages (~13 MB/layer, HBM otherwise idle): C stream 6.8 -> ~5 us: gen B=1 -30..-45; B=8 smaller.
  OPF measured (e0293, 3 rounds): gen B=1 +3.6, B=8 +8. Reverted. Sixth failed weight prefetch: phase C's stream is not HBM-bound after the ring prefetch.
- Hypothesis (before measuring): UNION1 = at B=1 the routing warp builds the slot/unit tables itself (the union is the selection) before the norm2 block sync; the union step and one block sync disappear: gen B=1 -15..-20; B>1 unchanged.
- UNION1 measured (e0294): gen B=1 +31, B=8 +24 although B=8 never takes the new path: same registers/spills
  (e0295), so a scheduling/allocation perturbation of the one big function. Reverted.
- Code-placement probe (e0296): 1-16 never-executed instructions before the step loop: gen B=1 1711.6-1712.4
  for every shift. Placement does not matter; allocation/scheduling of the whole kernel does.
- SASS (e0297): every consume loop does STL/LDL of ct ('volatile int ct', the RS>=3 ptxas workaround) per stage.
- Hypothesis (before measuring): plain ct at RS=2 (bench/ring.cu showed the bug only at RS>=3): one local load less per stage: gen B=1 -5..-15; results must stay bit-identical.
  Measured (e0298, 3 rounds): gen B=1 1712.6 -> 1705.3, tf B=1 -10, B=8 -8; results identical (check, bcheck, soak 18/18). Kept.
- Hypothesis (before measuring): NSS norms take a base pointer instead of the rows_in[8] array (which was stored to the stack before each noinline call and re-loaded by the callee before its row loads): gen B=1 -3..-6.
  Base-pointer norms measured (e0300): -0.2, neutral. Reverted. (Absolute times drift ~10 us between exec sessions on one card; only interleaved same-session A/B counts.)
## s0026 (track B, plain ct): expected ~1706; measured 1704 / 3308 (valid).
- Hypothesis (before measuring): AS2 = full-attention layers split their keys over 2 blocks per (sequence, kv head); each block writes unnormalised (m, l, acc) and phase C merges both halves while loading the attention output (no extra barrier). Full layers ~9 -> ~5.5 us at ctx 1000, ~4.8 -> ~4 at ctx 300: gen B=1 (ctx 64..1088) -20, B=8 -20.
  Measured (e0304): correct (bcheck, both tracks) but gen B=1 +20, tf B=1 +35, tf B=8 +280 (bench/mk_as2_attempt.cu).
  Fourth split-key attention variant lost; the merge in phase C's load (scalar fp32 loads + 3 expf per element on
  every block) costs more than the shorter key loop saves. Attention closed.
- Last-arriver analysis (tests/lastarr.py, e0301-e0303): kv head 0's attention block is last at the A->B->C barrier
  in 2/3 of layers, but the per-head spread is only 0.1-0.3 us (sliding 3.7 us, full 4.8 at ctx 300, 9.1 at ctx 1000).
- Routing re-probe after NSS4 (e0305): reading a third of the partials: E entry 2.87 -> 2.56 us. Routing split would gain <= 0.3 us/layer: skipped.
- Hypothesis (before measuring): NSS4W = in the 4-warp norm, the 6 norm-weight float4 per lane are loaded with the row (24 registers): one round trip less per norm: gen B=1 -15..-25.
  Measured (e0306, 3 rounds): gen B=1 1705.8 -> 1701.3, gen B=8 -5, tf mixed. Kept (prediction -15..-25 too high: the weight loads were mostly overlapped already).
## s0027 (track B, NSS4W): expected ~1700; measured 1699 / 3305 (valid).
- SKEL probe (e0307, weights from a 512 KB L2 window): gate_up 17.3 -> 12.7 us, down 12.5 -> 10.6 (work 9.3 -> 8.5), qkv 8.2 -> 4.6, o 6.6 -> 4.4, head 168 -> 102. Down at B=1 is not HBM-bound.
- Down-phase trace (DBGT build, tests/trace_down.py, e0309, B=1 layer 5, warp 0): stage data waits 0.2-0.5 us,
  math 1.0-1.3 us per 9-item stage (~120 ns/item; compute-latency-bound), first stage ~2.0 us (cold code?), result
  write + atomic + combine ~1.1 us; total ~9.5 us.
- Hypothesis (before measuring): UNR1 = MXFP4 item loop unrolled 3 (or 1) instead of 9 (smaller code; the first stage of each phase looked i-cache cold): if i-cache matters, gen B=1 -10..-30; if ILP matters, worse.
  UNR1 measured (e0312): unroll 3: gen B=1 -2, B=8 -13, tf B=1 +11 (mixed); unroll 1: +12 / +33. Reverted; i-cache is not decisive.
- gate_up trace (DBGT_GU, e0316, B=1 layer 5, warp 0): consume starts ~1.1 us after phase start (task build + stage issue), first data 1.5-3.5 us later, stage math 2.3 us first then ~1.5 (170 ns/item), data waits ~2 us total, swiglu epilogue ~1 us; ~14 us.
- fp8 item probe (bench/dec.cu V3, e0317): e2m1 -> e4m3 decode (7 ops/word) + 4 mma.m16n8k32.e4m3 + per-column B
  scales: 186 cycles/item (1 warp/scheduler) vs 128 for MULDEC bf16; 498 vs 331 with 12 warps. fp8 mma.sync is
  slower here; closed (it would also have needed an exact hi/lo split of the activations with fallbacks).
- Hypothesis (before measuring): FUSE = MXFP4 items stored as 544 B (codes then scales; layout only, both tracks), one bulk copy per ring stage instead of two: at MoE phase start 12 warps issue half as many copies; gen B=1 -10..-25 if copy issue limits the ramp, else ~0. Results must be bit-identical.
  Measured (e0318, 3 rounds): gen B=1 1701.5 -> 1691.8, tf B=1 -6, gen B=8 -26, tf B=8 -20; check statistics identical on both tracks. Kept.
## s0028 (track B, FUSE): expected ~1690 / ~3285; measured 1692 / 3283 (valid).
## s0029 (track A catch-up): expected ~1925 / ~3450; measured 1924 / 3444 (valid).
- TMA issue probe (bench/tmaissue.cu, e0321): issuing 1-2 bulk copies costs ~300-400 cycles per warp; landing scales with bytes (per-SM ingest ~60 GB/s from L2). gate_up's start is the HBM delivering ~14 MB of first stages (~4.7 us at 3 TB/s); steady-state gate_up consumes ~36 GB/s/SM > ~23 GB/s/SM HBM share: HBM-bound. Down at B=1 is math-bound per warp (5-6 warps/SM).
- Hypothesis (before measuring): DPAIR again (B <= 2 only; fused layout): down at B=1 is math-bound per warp with 5-6 busy warps; pairs halve per-warp items: down 9.5 -> ~6.5 us/layer on the trace, gen B=1 -30..-60 unless codegen perturbation eats it; B=8 unchanged (whole-K).
  Measured (e0322): trace: warp 0's math ends at 5.0 us (was 7-8) but the pair exchange + result write take 3.3 us;
  gen B=1 -2, tf B=1 +2, B=8 tf +8. The down phase's end is set by each rowtile's slowest of 4 units (different
  blocks) and its combine, not by one warp's math. Reverted for good.
- NOHINT probe (e0323): bulk copies without the L2 evict_first policy: gen B=1 +117, B=8 +137. The policy protects the small/hot data from the weight streams.
- Hypothesis (before measuring): SKEF = the sketch head's 434 MB of pass-1 loads with the L2 evict_first policy (like the weight streams): the step's end no longer flushes L2; next-step early layers keep their small data: gen B=1 -5..-15.
  Measured (e0324, 3 rounds): gen B=1 1691.8 -> 1688.3, B=8 ~0. Kept.
- Hypothesis (before measuring): PFOE = pf_o issued by each warp right after its qkv stream (ring drained) instead of after the epilogue: phase A shortened by pf_o's issue time: gen B=1 -5..-10.
  Measured (e0325, 3 rounds): gen B=1 1688.3 -> 1678.9, tf B=1 -2..-8, gen B=8 -16, tf B=8 -12. Kept.
## s0030 (track B, PFOE + SKEF): expected ~1680; measured 1680 / 3274 (valid).
- Verification (e0326, e0328): all checks pass on s0030 (track B) and the current track A build; MXFP4 storage roundtrip bit-for-bit on all 48 expert tensors.
- Hypothesis (before measuring): EFLAG = per-slot gate_up counters (one release add per block and slot) instead of the grid barrier before the down phase; each block waits only for the ~33 blocks producing its slots' activations: E->F wait 3.8 -> ~2.5 us plus one barrier less: gen B=1 -20..-35 (attention's analogue AFLAG gave 0, so maybe 0).
  Measured (e0330): correct; gen B=1 -0.7, gen B=8 -12, tf mixed. Reverted (bench/mk_eflag_attempt.cu). Barrier waits are not shortened by per-slot dependencies: every slot has a slow producer.
- Parameter re-sweep (e0331/e0332): NSEG_O 5 neutral, 8 +16; NSEG_QKV 3 +13, 5 +36 (tf); SKQ 10 +1, 18 +5. gen B=8 swings (-250 for NSEG_QKV 5) are trajectory effects (tf: +36). Current values stay.
- norm2 trace (DBGT_NORM, e0334, B=1 layer 5): call site -> norm entry 0.4-1.0 us, entry -> loads done 0.8-1.4, stores 0.25, then the block sync waits for the routing warp (E entry ~2.7 us). The noinline call itself costs up to 1 us.
  (Correction: the first stamp of each DBGT trace includes a global load of the trace flag (~0.5 us); 'start' offsets are probe artefacts, stage durations are not.)
- Hypothesis (before measuring): RSPL = the routing warp's 3 dependent load batches split over 3 warps (B <= 3; same (h0+h1)+h2 order at every B), top-4 after the norm2 barrier: E entry 2.9 -> ~2.3 us: gen B=1 -10..-15.
  RSPL measured (e0339): gen B=1 +4.4, tf mixed; reverted (bench/mk_rspl_attempt.cu).
- Hypothesis (before measuring): XPF (B=1) = during attention, 32 idle blocks compute the predicted router logits (Wr*ln2/ln1 on the normed attention input, 3.5/4 right), 92 others pull the first two ring stages of the predicted experts' 1440 gate_up units (14 MB) into L2 while HBM is idle. gate_up's cold start (2-3.5 us) and part of its HBM-bound stream move into the idle window: gen B=1 -25..-60; logits unchanged.
  Measured (e0340-e0342): first version +45 (the prediction dot product loaded serially: attention phase 5.6 -> 8.8
  us; gate_up work 14.5 -> 12.6 but imbalance 1.40, wait 3.5 -> 6.4: blocks of mispredicted experts set the end).
  With the loads in flight: top-4 +29, top-5 +40, top-6 +60 (gen B=1). Seventh failed weight prefetch: even
  in the HBM-idle attention window, the prefetch traffic costs more than the earlier first stages save.
  Reverted (bench/mk_xpf_attempt.cu, bench/api_xpf_attempt.py).
- Hypothesis (before measuring): DPS = the head's digit preparation (two passes over 2880 values by one warp per sequence) split over all 12 warps (partials in fixed order; margins cover the order): -1.5..-2.5 us per step at B=1, more at B=8.
  DPS measured (e0344): gen B=1 +3, B=8 +13. Reverted.
- Head census / norm-skip study (e0345, e0346): nothing to skip (0.5% of rows).

## State at ~10 agent-h (summary)
- Best: track B s0030 1680 / 3274 us (HTTP B=1 / B=8; 1.99x / 1.52x vLLM). Track A s0029 1924 / 3444 (1.74x / 1.44x).
- Session gains (s0013 1863 -> s0030 1680, -183 us): router fold (-49), single-pass norms and their 4-warp split
  (-21, -24, -4.5), cold-load prefetches without registers (K/V window -13, router rows/down biases -14, small
  vectors -12), qkv epilogue loads before the barrier (-10), masked-multiply decode (-9), plain loop counter (-7),
  fused MXFP4 items (-9.6), earlier o_proj prefetch (-9.4), evict_first sketch loads (-3.5).
- Closed (measured worse or zero): split-key attention x4, attention register/smem staging x3, L2 weight prefetch x7
  (incl. predicted experts in the attention window), counters instead of barriers (A->B, E->F), warp-pair down
  units x2, routing split, union in the routing warp, norm over all warps x3, fp8 decode, int5 head sketch (study),
  row skipping (study), deeper rings, parameter re-sweeps.
- ncu source hotspots (e0348/e0349): 45% of stall samples on branch instructions (mbarrier/barrier spins), BSSY 8.6%, decode LOP3 8.8%; pass-2b tile STS 2.4%, sketch PRMT 2.2%. Nothing new.
- o_proj imbalance probe (e0350, timing only): capping blocks at 1 o_proj rowtile (-27% bytes): C stream 6.56 -> 5.34, C epilogue span 4.39 -> 3.21 (wait 1.70 -> 1.35). The 2-rowtile blocks set phase C's end.
- Hypothesis (before measuring): OOCT = o_proj as 360 octets (8 rows; weights as the n8 operand, activations as the m16 operand from xs), 2-3 octets per block (max 24 rows instead of 32), K in 4 segments: phase C critical path -1.0..-1.5 us/layer: gen B=1 -25..-35, B=8 -10..-30.
  OOCT measured (e0351): correct (bitwise batch, both tracks) but gen B=1 +4.6, tf B=1 +5, tf B=8 +11..+22 (gen B=8 -225 is a trajectory change). Balancing rows does not buy what the byte-removing probe suggested. Reverted (bench/mk_ooct_attempt.cu, bench/api_ooct_attempt.py).
- Per-block consistency (tests/blockprof.py, e0352, B=1, 144 occurrences): gate_up slowest blocks 98-112 (15.9 vs median 13.8 us, mean rank ~16/132: consistent); qkv slowest 115-131 (rank 13-25); down and o_proj weakly consistent.
- Card dependence (e0353-e0356): block->SM map stable per card, different between cards; slowest gate_up blocks 98-112 on card 4 but 0-14 on card 5, and on card 4 qkv's slow blocks (115-131) differ from gate_up's. Hypothesis: address -> HBM-channel affinity of each block's contiguous data range (physical placement varies per process/card).
- Hypothesis (before measuring): GURR = gate_up units round-robin over blocks instead of contiguous ranges: per-block HBM channel spread evens out, gate_up imbalance/wait (3.8 us) shrinks: gen B=1 -20..-40 if the theory holds, else ~0.
  GURR measured (e0357, 3 rounds): gen B=1 1683.1 -> 1683.0, tf B=1 mixed; tf B=8 3356 -> 3337, gen B=8 -31. Kept (B=8).
## s0031 (track B, GURR): expected ~1680 / ~3245; measured 1680.3 / 3250 (valid).
- Hypothesis (before measuring): QRR = qkv rowtiles round-robin over blocks (b, b+132, b+264): per-block reads spread; qkv's consistently slow blocks even out: gen B=1 -5..-15, B=8 similar.
- QRR (qkv rowtiles round-robin) measured (e0358, 3 rounds after two pod losses): gen B=1 -1, tf B=1 +5, B=8 neutral. Reverted. Block-level slowness in qkv is not data placement either.
- Final verification of the s0031 engine (e0359, after three pod losses): track B and A: B=1 vs B=8 bitwise, B=2..8
  rows bitwise vs alone, contexts to 2036 finite with generate == step, serve == generate; track B soak 27/27 x 400
  tokens; limits (11 checks OK); expert storage roundtrip bit-for-bit on all 10,152,345,600 bytes.
- Re-test of old switches on the current code (hypothesis: DSPLIT=2 lost at B=8 earlier; with the per-warp math-bound down phase at B=1 it may now pay: gen B=1 -10..-20, B=8 worse).
- Multi-counter barrier (bench/gbar2.cu, e0360): 1 counter 1.05 us, 2 counters 1.00, 4 1.01, 8 1.03 per barrier; contention is not the cost (round-trip latency is). <= 6 us/step available.
  Re-test (e0361, 3 rounds): DSPLIT=2 gen B=1 -2.3, tf ~0, B=8 +138; RI=8 +15. Unchanged verdicts.
- Hypothesis (before measuring): GBATOM = grid barrier with atom.add.acq_rel (bench V3 0.97 vs 1.05 us) and no polling by the last arriver: ~123 barriers/step: gen B=1 -5..-12.
  Measured (e0362, 3 rounds): gen B=1 1683.2 -> 1679.2, tf B=1 -3, B=8 +6..+9. Kept (B=1 is the score).
## s0032 (track B, GBATOM): expected ~1676 / ~3257; measured 1678 / 3255 (valid). New best B=1.
## s0033 (track A catch-up): expected ~1920 / ~3425; measured 1896 / 3393 (valid; PFOE and SKEF-era shared changes also in).
- Hypothesis (before measuring): P2S = at B <= 2 pass 1 also writes each tile's max upper bound per column; after the threshold barrier every block scans all 12568 tiles and derives the same ordered candidate list (rank % NB == bid is its share): no list atomics, no second grid barrier: gen B=1 -2..-4; B=8 unchanged path.
  P2S measured (e0363/e0365): correct (soak 27/27) but gen B=1 1680 -> 1688 (+8), tf mixed. Reverted (bench/mk_p2s_attempt.cu).
- Staggered barrier polls (bench/gbar3.cu, e0366, engine-form barrier with uneven block work): 1 poll 1.41 us/iteration, 2 polls 1.72-1.89, 3 polls 2.08-2.32. Acquire loads of one thread do not overlap; closed.
- Hypothesis (before measuring): NSS4ALL = 4-warp norm also at B > 2 for norm1 and the final norm (no routing warps beside them): B=8 norm1 4.0 -> ~2.5 us/layer: tf B=8 -20..-35; B=1 code path unchanged (codegen aside).
  NSS4ALL measured (e0367): tf B=8 3347 -> 3418 (+72: 32 tasks over 12 warps = 3 sequential round trips), gen B=1 +6. Reverted.
- L2 policy variants for the weight streams (e0368): evict_first 1.0 (current) 1678; evict_first 0.5: +69; evict_unchanged: +118; evict_first+evict_unchanged: ~0. Current policy stays.
- PFLAST (evict_last on the K/V and small-vector bulk prefetches, e0369): gen B=1 +0.5, tf B=8 +15. Reverted.
- Sketch access-pattern bandwidth (bench/skbw.cu, e0370, 434 MB standalone): 4-B pattern 160 us (2.73 TB/s), 16-B loads 160, scattered tiles 165. In-engine pass 1 (~150-158 us) is at the pattern's limit for 384 threads/SM. Closed.
- Cluster feasibility (bench/clus.cu, e0371): clusters of 2 keep all 132 CTAs (66 clusters) resident and cooperative launch with clusters works; clusters of 4 only 120 CTAs.
- Hypothesis (before measuring): CLUS = the same kernel launched as clusters of 2: neutral (precondition for DSMEM pairing).
  CLUS measured (e0372): gen B=1 +2.5, tf B=1 +5, B=8 ~+5.
- Hypothesis (before measuring): OPAIR (with CLUS) = o_proj rowtiles balanced inside clusters of 2 (9 tasks per CTA instead of 12/6), the partner's 3 segment partials stored into the owner's red slots via shared::cluster, one cluster barrier; sums in the same order: logits bit-identical. Phase C -1..-1.5 us/layer: gen B=1 -20..-30 vs CLUS (+2.5 vs base).
  OPAIR measured (e0373, 3 rounds): gen B=1 base 1677.6, CLUS 1679.9, CLUS+OPAIR 1684.8; tf B=1 likewise worse.
  (Not bit-identical either: router partial sums group different rows per block.) Third failed o_proj balancing
  (OOCT, OPAIR; the byte-removing probe overstated it). Reverted (bench/mk_opair_attempt.cu).
- Hypothesis (before measuring): PB8 = when candidate tiles outnumber blocks (B=8: ~300 on real text, more on random prompts), each warp recomputes one tile from global memory (12 k-steps in flight) instead of one tile per block at a time: B=8 -15..-40 (random prompts more); B=1 path unchanged (codegen aside).
  PB8 measured (e0374, 3 rounds): serve B=8 -8, serve B=1 +1.7 (gen +1.2). Reverted (B=1 is the score; bench/mk_pb8_attempt.cu).
- Hypothesis (before measuring): BFIX=1 = the kernel compiled with B a constant 1 (dead B>1 paths, lower register pressure): B=1 gen -10..-40 if the compiler profits; to be used as a second library chosen per cohort.
  BFIX=1 measured (e0375, 3 rounds, B=1 only): gen 1677.4 -> 1675.2, serve 1679.1 -> 1676.8, tf -2.8.
- B=1 library (build.sh: libmk_b1.so / libmk_1.so with -DBFIX=1; api routes B=1 step/generate/cohorts to it). Hypothesis: same bits (bcheck B=1 alone vs batches), gen B=1 -2.
  Measured (e0376): bcheck/soak/track A all bitwise; gen B=1 with the B=1 library vs without: +1.3, +1.7, -1.5 (noise).
  B=1-only variants (e0377, 3 rounds, B=1 library): EPLC ~0; DSPLIT=2 gen 1677 -> 1665, serve 1680 -> 1668 (-12); both ~-10.
  DSPLIT=2 in the B=1 build only (e0378): greedy alone vs batch 2/8 identical, mean prefix 0.37; tf logits alone vs batch max |d| 10.75 (routing flips cascade). Rejected: numerics must match across batch sizes.
  DSPLIT=2 in both track-B builds (e0379, 3 rounds): serve B=1 1677.2 -> 1666.5, gen -11, tf B=1 -8; serve B=8 3602 -> 3701 (+99), tf B=8 +135. Taken for track B (score is B=1; B=8 stays at ~1.48x vLLM).
  Verification (e0380): bitwise B=2..8 vs alone (B=1 on the B=1 build), alone vs batch greedy 8/8 identical over 600 tokens, soak 27/27, limits 11 OK, track A bcheck ok.
## s0034 (track B, B=1 build + DSPLIT=2): expected ~1668 / ~3355; measured 1667.6 / 3406 (valid; B=8 +151). New best B=1.
- B=1-build sweep (e0381, 3 rounds, gen/serve B=1): base 1663.4/1665.3; GURR=0 1649.0/1650.1 (-14.5); SKQ=10 -1; SKQ=18 +0.5; RI=8 +1.5; EPLC +3.
  GURR=0 in the B=1 builds (e0382): all exactness checks pass; serve B=1 ~1650.
## s0035 (track B, GURR=0 in the B=1 build): expected ~1652; measured 1652.6 / 3405 (valid). New best B=1.
- Switches off in the B=1 build (e0383, 3 rounds, gen B=1; base 1648.3): KVPF +32, SVPF +18, PFOE +15, EPLA +6, NSS4W +6, RFPF +5.5, GBATOM +3.5, DBPF -0.8. All kept.
- Hypothesis (before measuring): UNION1 in the B=1 build only (no B>1 code to perturb there): gen B=1 -10..-20.
  UNION1 in the B=1 build (e0384): gen B=1 +9.5. Reverted for good.
- B=1-build compiler/switch sweep (e0385): ptxas -O2 +0.4, allow-expensive +0.4, DBPF=0 -1.1 (noise-level).
  AKV1 in the B=1 build (e0386): +60 again. Closed.
- Track A with its B=1 build (e0388): serve B=1 1901 -> 1878 (-23), bitwise checks pass.
## s0036 (track A, B=1 build): expected ~1873; measured 1876.5 / 3394 (valid).
- B=1 build (e0389, 3 rounds): RFPRE +63 (spills); SKQ=10 + DBPF=0: gen 1650.4 -> 1645.2 (-5). Adopted in the B=1 builds.
## s0037 (track B, SKQ=10 + DBPF off in the B=1 build): expected ~1648; measured 1651.1 / 3408 (valid). New best B=1.
- Segment counts in the B=1 build (e0391, tf B=1, base 1852): NSEG_O 5 +2.5, 8 +25; NSEG_QKV 3 +32, 5 +50. Current values stay.
- Hypothesis (before measuring): DSPLIT=3 (down units in K thirds, combine (y0+y1)+y2): B=1 build tf -5..-10 more than DSPLIT=2; B=8 worse.
  DSPLIT=3 measured (e0392, tf): B=1 build 1852 -> 1891 (+39), B=8 general 3573 (vs ~3480 with DSPLIT=2). Reverted.
- P2S in the B=1 build (e0393): gen B=1 +23. Reverted for good.
- Hypothesis (before measuring): a B=8 build (BFIX=8) for full-batch cohorts: tf/gen B=8 -20..-60 (constant folding), bitwise unchanged.
  B=8 build (e0394, 3 rounds): tf/gen/serve B=8 identical to the general build (3525/3747/3748). API now routes per batch size (1 and 8) to optional builds; only the B=1 builds are built.
  B=8 build without GBATOM (e0395): tf B=8 +9, serve +4. No B=8 build.
  Register use (e0396): general and B=1 builds both 168 regs, 8-12 B spills; the B=1 build's gain is different code generation plus B=1-tuned switches.
- GORD (rowtile-major gate_up order) in the B=1 build (e0397): +6. Reverted.
- QRR in the B=1 build (e0398): +14. Reverted.
- Verification of the committed state (e0399, builds libmk/libmk_1/libmk_b/libmk_b1): track B check, bcheck B=2..8 bitwise, long ctx, soak 27/27 x 400, alone-vs-batch greedy 8/8 identical (600 tokens), limits 11 OK; track A bcheck and alone-vs-batch 8/8; expert storage roundtrip bit-for-bit.
- Hypothesis (before measuring): VSPLIT = the general track-B build keeps whole-K down units but accumulates the two K halves separately and adds them (same bits as DSPLIT=2, which the B=1 build keeps): bitwise across batch sizes, B=8 back to ~DSPLIT=1 speed (-100..-150), B=1 unchanged.
  VSPLIT measured (e0400/e0401): bitwise exact (bcheck, alone-vs-batch 8/8) but serve B=8 3750 vs DSPLIT=2 3700 vs DSPLIT=1 3602, per-item or per-stage boundary check alike. Reverted (bench/mk_vsplit_attempt.cu).
- DORD (down units half-major) in the B=1 build (e0402, 3 rounds): gen -1.9, serve -3.2. Adopted in the B=1 builds.
- KVPFE in the B=1 build (e0403): neutral. Reverted.
- s0038 (track B, DORD in B=1 builds): expected 1651 -> ~1648; measured 1646.6 / 3405 (in-process gen 1627.1). Confirmed (slightly better
  than predicted). New best. Track A build unchanged (DORD only in B1EXTRA, which also applies to libmk_1.so -> track A B=1 too).
- Hypothesis (before measuring): more assignment-only orders in the B=1 build (DUNIT macro): DORD=2 (h, rt, slot:
  a block's units cover all 4 slots of ~3 rowtiles of one K half, so a rowtile's 8 units sit in 2 blocks; all slots staged)
  and DORD=3 (rt, slot, h: a rowtile's 8 units mostly in one block, combine local); GORD=1 (gate_up rowtile-major over
  slots). Expect each within +-3 us of DORD=1 (e0405, e0406); DORD=3 best guess -2 (combine waits on siblings, not far blocks).
- DORD=2 / DORD=3 (e0405, 3 rounds): gen +30 / +43, serve +31 / +43 vs DORD=1 (alone-vs-batch 8/8). GORD=1 (e0406): +5. All lose.
  Reading: the orders that stage all 4 slots' activation rows lose ~1.3-1.8 us/layer; the phase F staging loop
  (ld.cg then st.shared per iteration) runs its loads one round trip at a time.
- Hypothesis (before measuring): STGU = phase F staging with all of a thread's loads issued before its stores
  (assignment-free, same bits). B=1 blocks stage 1-2 rows (<= 2 loads/thread): gen -0..-10 (boundary blocks lose a
  round trip); B=8 general build (up to ~16 rows, ~15 loads/thread) STGU=4: -20..-60 us.
- STGU (e0407 B=1 build, e0408 general build B=8, 3 rounds): B=1 STGU=2 +20, =4 +33; B=8 STGU=2 +13, =4 +27. Wrong
  prediction: batching the staging loads costs (code/registers in the one big kernel) more than any round trips it
  saves; the round-trip reading of DORD=2/3 is therefore not supported either (cause of their loss unknown).
- Hypothesis (before measuring): STGC = phase C attention-output staging with STGC loads in flight per thread (B=1: 3 -> 1
  round trip; B=8: 22 -> 6). After STGU's loss I expect a loss again (+5..+20) unless the round trips are real.
- STGC (e0409 B=1 build STGC=3, e0410 general B=8 STGC=4): +15 / +38. Loss as predicted. Reverted STGC and the
  DUNIT/GORD refactor (mk.cu back to s0038).
- Head profile, B=1 build gen at ctx 500/900 (e0411/e0412, random prompt, up to 3 candidate tiles per block): pass 1 166 us;
  threshold 6.7; per candidate tile: tile load (92 KB, load->store loop) 6-9 us, 180-step mma chain ~5 us (work 3.9).
- Hypothesis (before measuring): P2TMA = the candidate tile by one 92 KB cp.async.bulk on a new mbarrier (no registers, all
  bytes in flight; P2B's register version gave nothing in e0237): tile load 6-9 -> ~1.5 us: gen B=1 -4..-8 (one tile per
  block on real text), B=8 -10..-20 (2-3 tiles per block).
  P2TMA measured (e0413 B=1 build, e0414 general B=8, 3 rounds): gen B=1 1643.3 -> 1635.0 (-8.3), serve 1644.9 -> 1636.0 (-8.9);
  B=8 gen -6.3, serve -7.9. Alone-vs-batch 8/8, soak 18/18. Adopted in the track-B builds (libmk_b, libmk_b1).
## s0039 (track B, P2TMA): expected ~1638 / ~3398; measured 1639.2 / 3396 (in-process gen 1618). Confirmed. New best.
- Sketch stream benches (e0416 bench/sktma.cu, e0417 bench/skbw.cu, 434 MB): per-warp TMA rings 154.5-155.8 us without and
  158.9-160.5 with the evict_first hint (4-B loads 161): not worth a rewrite. Tiles interleaved over all 1584 warps
  (tile = n*1584 + block*12 + warp): 157.6-158.2 vs 161.4-162.0 (-3.5).
- Hypothesis (before measuring): SKGI = that interleave in the engine (assignment only; same bits): gen B=1 -2..-4, B=8 same.
  SKGI measured (e0418 B=1 build, e0419 general B=8; soak 18/18): gen B=1 +11.6, serve +11.0; B=8 +3. Wrong. Reverted.
- Head profile with P2TMA (e0420): per tile the bulk copy takes ~3.5 us (one SM, ~26 GB/s), then the chain ~3.9 us work.
- Hypothesis (before measuring): P2CH = the tile's copy in P2CH chunks with one mbarrier each; warp 0's chain waits per chunk
  (same k order, same bits) so the chain overlaps the copy: per tile -2..-3 us: gen B=1 -2..-3, B=8 -5..-8 (2-3 tiles/block).
  P2CH measured (e0421 B=1 build, e0422 general B=8; soak 18/18): B=1 P2CH=6 +1.5, P2CH=3 -0.5 (noise); B=8 P2CH=6 gen -10.0,
  serve -9.0. Half right: B=1 has no gain (one tile per block; the chain is not on the critical path there). P2CH=6 in the
  general track-B build only.
## s0040 (track B, P2CH=6 general build): expected B=8 ~3388, B=1 ~1639; measured 1639.1 / 3386. Confirmed.
## s0041 (track A catch-up, DORD + DBPF=0 in the track-A B=1 build): expected ~1870; measured 1877.6 / 3394 (s0036 1876.5). Wrong: no gain for track A.
- Hypothesis (before measuring): QW = the head's x quantization (two 90-iteration passes by one warp per sequence while 11
  warps wait at B=1) spread over all 12 warps (240 columns each, amax and sums through shared memory; digits identical,
  bound sums in another order under their 1.001 margin): gen B=1 -2..-3, B=8 ~0.
  QW measured (e0425/e0426, soak 18/18): B=1 gen -0.9, serve -1.6 (noise level); B=8 +6.5. Reverted.
- Head sketch format study (tests/tier_study.py e0427, tests/tier_study2.py e0428; real text T=256, 2048 positions, the
  engine's bound model): two tiers by tile fail (under int5/int4/int3 row scales every tile is a candidate at some position).
  Group scales (one per row and 32 columns) shrink the bound: candidate tiles mean / p99: int6 row (435 MB) 163 / 1042;
  int6 g32 (452 MB) 15.7 / 59; int5 g32 (380 MB) 201 / 1465; int5 g16 (398 MB) 85 / 556; int4 g32 (308 MB) 7624 (useless).
- SK5 = int5 codes + u8 group scales on a float32 row base (api.pack_sketch5; per k32-step pair 5 code words + 1 scale
  word per lane; decode to 8q int8 bytes; per-step integer dots scaled in float). tests/sk5_check.py (e0429): codes and
  scales equal an independent quantization; |exact - sketch| / bound <= 0.099 over 1024 positions x 201088 rows;
  candidate tiles mean 167, p50 58, p99 2212. Soak 18/18, alone-vs-batch 8/8 (e0435). (One crash, e0431, not reproduced
  under compute-sanitizer or after a clean rebuild: probably a half-built library; watch.)
- Hypothesis (before measuring): SK5 in both track-B builds: pass 1 reads 381 instead of 434 MB (-12%; pass 1 is byte-
  proportional at ~2.97 TB/s, e0262): -17 us; more decode work and a few more candidates: gen B=1 -10..-15, B=8 -5..-15.
  SK5 measured (e0436 B=1 build, e0437 general B=8, 3 rounds): gen B=1 1636.3 -> 1634.0 (-2.3; serve too noisy on this pod),
  B=8 gen -9.6, serve -6 (one baseline outlier). Head span (e0438 prof, ctx 500): 165.6 -> 160.3 only; SKQP 3/9/15:
  161.5/162.8/170.0 (e0439). Wrong: with ~40 instructions per k-step pass 1 is no longer byte-proportional.
- Hypothesis (before measuring): DPK (B=1 build) = hi and lo x digits in columns 0 and 1 of one sketch mma (B=1 uses one
  of 8 columns): one IMMA and half the epilogue per k-step, same integers: SK5 pass 1 -5..-8; int6 path -0..-4.
  DPK measured (e0440, 3 rounds, gen B=1; serve unreliable on this pod): b1 1636.0, int6+DPK 1636.7, SK5 1634.3, SK5+DPK
  1629.5 (-6.5 vs b1). Soak 18/18 for both DPK variants. Adopted: SK5 in both track-B builds, DPK in the B=1 track-B build.
## s0042 (track B, SK5 + DPK): expected ~1633 / ~3376; measured 1634.0 / 3405 (valid; new best B=1). B=8 wrong: eval session drift ~+7 (step_us), B=8 gen +34, w1_like_generate +24: on the evaluator's workloads the heavier candidate tail of SK5 costs B=8 recompute rounds. Next: int6 sketch for the general build, SK5 only for the B=1 build.
- Two sketches (e0442): general track-B build back on the 6-bit sketch, B=1 build on SK5+DPK; bcheck, alone-vs-batch 8/8, soak 27/27 x 400, limits pass.
## s0043 (track B, 6-bit sketch for B>1, SK5 for B=1): expected ~1634 / ~3386; measured 1634.0 / 3396 (valid). B=8 recovered within session noise.
- Hypothesis (before measuring): I64 = SK5+DPK sketch accumulated exactly in int64 (IMAD.WIDE per output) instead of I2F+FFMA: fewer ALU ops per k-step: gen B=1 -1..-4. (Load-only bench e0448: int6 pattern 160.6 us, SK5 146, SK5 with 16-B loads 157: SK5 pass 1 in the engine runs ~10 us over its load floor.)
  I64 measured (e0449, 3 rounds, soak 18/18): gen B=1 1630.1 -> 1630.0: neutral; ALU count is not what holds SK5 pass 1. Reverted.
- Hypothesis (before measuring): SKPF0 = L2 bulk prefetch of each warp's first 1-2 sketch load batches (3.4-6.7 KB) at the end of the last layer (pass 1 starts after the barrier, norm and quantization): gen B=1 -1..-3.
  SKPF0 measured (e0450, e0451, 3 rounds each, gen B=1): 1 batch -2.2 / -1.7; 2 and 4 batches the same as 1; issued at the last layer's down-phase start instead (SKPFF) +2 / +4. Adopted SKPF0=1 in the B=1 track-B build.
## s0044 (track B, SKPF0 in the B=1 build): expected ~1632; measured 1634.3 / 3392 (valid). No gain (third submission under 0.5%: plan rewrite due).
- Rebalancing probe (e0453, B=1 build prof, 6 steps x 24 layers, gate_up per-block medians, ranges rotated by 33 blocks or not, 2 runs each): the slowest blocks differ between runs of the same build (slowest-10 mean ranks 34-48 of 132); only the 10-unit blocks are reliably fastest (12.3 vs ~13.8 us). Per-step variance is random: static per-block rebalancing cannot help. Closed.
- Hypothesis (before measuring): RSPL = the RFOLD router partials (132 x 32 per sequence, 33 float4 per lane) summed as
  three group sums (G0 + G1) + G2 in every build (an association change, same in all builds); at B = 1 two idle warps load
  groups 0 and 1 while the routing warp loads group 2 (one L2 round trip instead of three; named barrier). E_entry is
  max(routing ~3 us, norm2 ~2.4 us): gen B=1 -5..-15; B=8 ~0.
  RSPL measured (e0455, 3 rounds; first try e0454 hit a misaligned float4 in Misc): exact (bcheck bitwise, alone-vs-batch 8/8, soak 18/18) but gen B=1 +4.5, B=8 +38. Reverted.
- Hypothesis (before measuring): CLSPEC = the block's first candidate-list entry loaded with the candidate count (one dependent L2 round trip less on the recompute path): gen B=1 -0.5..-1, B=8 -0.5..-1 (noise level; tried because it adds almost no code).
  CLSPEC (e0456, 3 rounds): gen B=1 +0.6, B=8 -0.3: neutral. P2CH=3 in the B=1 build: -0.3: neutral. Reverted / not adopted.
- B=1 build switch re-sweep after SK5 (e0457/e0458, 3 rounds, gen B=1; base 1628.7 / 1627.7): DORD=0 +3.5, GURR=1 +3.2, DBPF=1 +2.6, EPLC=1 +4.4, RI=8 +2.8, SVPF=0 +16.3, RFPF=0 +6.4, SKQP=3 -0.8 (noise). Current choices hold.
- Granularity probe (e0459, timing only): gate_up at exactly 10 units per block (-8% bytes): span 17.3 -> 16.7, wait 4.3 -> 3.6. The phase-end wait is random per-block service variance even with equal work. Closed.
- Hypothesis (before measuring): SKDB = SK5 pass 1 with double-buffered load batches (SKQP=3: 36 load registers; batch k+1 in flight during batch k's math; same order, same bits): pass 1 ~158 -> ~150: gen B=1 -4..-8 if it does not spill.
  SKDB measured (e0462, 3 rounds, gen B=1): SKQP=3 double-buffered 1628.7 -> 1624.9 (-3.8; 168 regs, 80 B stack as before), SKQP=5 double-buffered spills (+24). Soak 18/18. (e0460/e0461 were build errors: a splice inside the DPK conditional, then a name clash.) Adopted SKDB=1, SKQP=3 in the B=1 track-B build.
## s0045 (track B, SKDB in the B=1 build): expected ~1630; measured 1628.5 / 3396 (valid). New best.
- SKDB variants (e0464, 3 rounds; base SKQP=3 + SKPF0=1 1624.3): SKQP=1 +2.7, SKPF0=2 +1.6, SKPF0=0 +7.5 (the first-batch prefetch matters more with double buffering).
- Hypothesis (before measuring): SKPFN = at each sketch tile start, L2 prefetch of the warp's next tile's first 1-2 batches (each of a warp's ~8 tiles starts its pipeline cold): gen B=1 -2..-5.
  SKPFN (e0465, 3 rounds): 1 batch -1.2 (noise), 2 +1.5, 4 +4. Not adopted.
- Hypothesis (before measuring): ATDB = attention_flash (noinline, own register allocation) with double-buffered K/V loads per warp (next 16-key block in flight during this block's math; same order, same bits): full layers at long context -1..-3 us: gen B=1 (contexts 64..1088) -10..-25, B=8 -10..-30. (The e0183 two-blocks-in-flight attempt, +57, predates the noinline split.)
  ATDB measured (e0466, 3 rounds; bitwise, soak 18/18): gen B=1 +60, B=8 +157 (stack 80 -> 144 B: spills even in the noinline function). Closed for good.
- Hypothesis (before measuring): QW in the B=1 build only (re-added; digits as DPK): gen B=1 -1..-3 (was -0.9 gen / -1.6 serve before SKDB).
  QW in the B=1 build (e0468, 3 rounds; soak 18/18, alone-vs-batch 8/8): -0.7, noise. Not adopted.
- Hypothesis (before measuring): HDPF5 = L1 prefetch of each SK5 tile's header at tile start (8 lanes x 32 B; the header is read after the tile's math): gen B=1 -1..-4 (with SKDB the dependent header load is more exposed).
  HDPF5 (e0469, 3 rounds): +0.2, neutral. Reverted.
- Hypothesis (before measuring): SKDB6 = the general build's int6 pass 1 double-buffered (batches of 5 or 9 k-steps; B=1 build unaffected): gen B=8 -3..-8.
  SKDB6 (e0470, general build, 3 rounds, gen B=8; soak 18/18): 3 k-steps 3677.0 -> 3672.1 (-5), 5: +2, 9: +4. B=8-only and within evaluation noise: not adopted (noted for a B=8 bundle).
- ptxas default-caching flags in the B=1 build (e0471, 3 rounds): -dlcm=cg +11, -dlcm=ca +0.8 (default); extensible-whole-program variant failed to load. Nothing.
- Hypothesis (before measuring): SKLN = the first SKLN sketch tiles (400/800/1200 = 12/24/37 MB) loaded with evict_last so they stay in L2 across steps (everything else streams evict_first): pass 1 reads that share from L2: gen B=1 -3..-10 if L2 keeps them, + if other data is displaced.
  SKLN (e0472, 3 rounds): 400/800/1200 tiles +7/+14/+15. Loses: L2 residency of sketch tiles displaces more useful data. Reverted.
- Hypothesis (before measuring): TOPR = router top-4 by rank counting (31 independent shuffles + 4 ballots instead of 4 dependent 5-step shuffle argmax rounds; same set, order and tie rule): E_topk -0.3 us/layer: gen B=1 -3..-8.
  TOPR measured (e0473, 3 rounds; bcheck bitwise, alone-vs-batch 8/8): gen B=1 1625.0 -> 1621.3 (-3.8), B=8 +10.8. Adopted in the B=1 track-B build only (identical selection, so builds may differ).
## s0046 (track B, TOPR in the B=1 build): expected ~1625; measured 1625.0 / 3396 (valid). New best.
- Hypothesis (before measuring): UN1 (B=1 build) = the routing warp writes the slot tables right after top-4 (4 lanes; same tables as the union code) so the union step and one block barrier disappear: E_union 0.64 us/layer: gen B=1 -5..-12 (if the routing warp's extra ~0.1 us does not extend E_entry).
  UN1 measured (e0475, 3 rounds; bitwise, 8/8, soak 18/18): gen B=1 +4.6. Wrong: the routing warp is the long pole of phase E's start; work added to it costs more than the union step saves. Reverted.
- Hypothesis (before measuring): RSS = the routing warp's 5 sum-of-squares loads issued with its first partial batch instead of after the third (one dependent L2 round trip less on phase E's long pole): gen B=1 -5..-15; RBN=2 (two load batches) another round trip less if no spills.
  RSS / RBN (e0476, 3 rounds; alone-vs-batch 8/8): RSS +1.8, RSS+RBN=2 +0.8, RBN=2 -0.2: neutral; the routing loads are not what holds phase E's start. Reverted.
- Verification of the committed state s0046 (e0478, tests/final_verify.sh): track B and A numerics vs ref32, bitwise B=8 vs alone, long contexts, alone-vs-batch 8/8 (both), soak 27/27 x 400, limits, SK5 bound check (0.099), expert storage roundtrip bit-for-bit (10.15 GB). All pass.
- Hypothesis (before measuring): UN2 (B=1 build) = gate_up's first stages issued right after the norm/routing barrier (slot k's expert = k-th set bit of the routing mask, same as the union's ascending order), the union tables built while they are in flight, then one barrier and the gate_up consume: the union step (~1.1 us/layer span) leaves the critical path: gen B=1 -5..-15.
  UN2 measured (e0479, 3 rounds; bitwise, 8/8, soak 18/18): gen B=1 +7.8. Reverted.
- TOPR in the track-A B=1 build (e0480, 3 rounds; alone-vs-batch 8/8): gen B=1 1876.0 -> 1872.4 (-3.6). Adopted. SKDB6=3 re-test (e0481, B=8): ~0 (one baseline outlier); not adopted.
## s0047 (track A, TOPR in the track-A B=1 build): expected ~1873; measured 1871.3 / 3394 (valid). Best track A (s0036 1876.5).
- Hypothesis (before measuring): UNB = union loops bounded by B instead of MAXB (no arithmetic; B=1 build): gen B=1 -1..-3.
  UNB (e0483, 3 rounds): +3.0. Reverted.
- Hypothesis (before measuring): SKL2D = with SKDB, lane 0 bulk-prefetches the next two batches into L2 every two batches (no registers; more bytes in flight per warp): pass 1 closer to its load floor: gen B=1 -2..-5.
  SKL2D (e0484, 3 rounds): +5.5. Reverted.
- Hypothesis (before measuring): GBREL = grid-barrier polls as ld.relaxed with one fence.acq_rel after the count is reached (instead of ld.acquire per poll): ~0.05-0.1 us per barrier x ~123: gen B=1 -3..-10, B=8 similar.
  GBREL measured (e0485, 3 rounds): +36.5 (fence.acq_rel.gpu after relaxed polls costs ~0.3 us per barrier more than acquire polls). Reverted; the acquire-poll barrier stays.
- Hypothesis (before measuring): HPW = the served path's per-step __threadfence_system + host progress store moved from warp 0 (which runs norm1 next) to warp 11 of block 0 (idle in phase A: block 0 has 2 qkv rowtiles = 8 tasks): a system-scope fence no longer delays block 0 every step: serve B=1 -1..-5 (gen unaffected: mode 1).
  HPW (e0486, 4 rounds, gen+serve): serve 1621.7 -> 1621.2, neutral (the per-step system fence is not on the critical path; serve ~ gen + 0.6 on this pod). Reverted.
- Hypothesis (before measuring): RBH = the router bias loaded at the start of the routing warp (it was read after the partial-sum reductions): gen B=1 -0..-3.
  RBH measured (e0487, 3 rounds; alone-vs-batch 8/8): gen B=1 1620.3 -> 1617.0 (-3.2). Adopted in the B=1 track-B build.
- Hypothesis (before measuring): GBPF = each gate_up unit's 64-B bias line prefetched into L1 when its task is built (the epilogue reads it after the 13-us stream as a dependent, possibly HBM-cold load): gen B=1 -2..-10.
## s0048 (track B, RBH): build_failed — the RBH patch had been applied twice (test patch kept, adoption script re-applied it); the pre-submit verification e0488 passed only because a missing libmk_b1.so silently fell back to the general build. Fixed the duplicate; tests/final_verify.sh now fails on a build error or a missing library.
- Verification after the RBH fix (e0490, tests/final_verify.sh: build ok, 4 libraries): all pass (both tracks, storage roundtrip, SK5 bound).
  GBPF measured (e0489, 3 rounds; alone-vs-batch 8/8): gen B=1 1616.5 -> 1612.5 (-4.1).
## s0049 (track B, RBH in the B=1 build, resubmitted): expected ~1622; measured 1617.2 / 3399 (valid). New best (larger gain than predicted).
- Hypothesis (before measuring): ATSK = attention's 8 sink logits prefetched into L1 at attention start (read by the final combine after the key loop; L2-warm via SVPF): gen B=1 -1..-7.
- Hypothesis (before measuring): EMBPF = before the step's last barrier every block with candidates bulk-prefetches its local best token's embedding row (5.7 KB) into L2; the next step's embedding read (a cold HBM load on every block's critical path) then hits L2: gen B=1 -0.5..-1.5.
  ATSK measured (e0492, 3 rounds): -0.6, noise. Not adopted (code off). Verification of the GBPF state (e0491, final_verify): build ok, all pass.
- Hypothesis (before measuring): DNBPF = each down unit's combine bias line prefetched into L2 when its task is built (the combine reads it as part of its single round trip; HBM-cold otherwise): gen B=1 -0..-3.
## s0050 (track B, GBPF in the B=1 build): expected ~1613; measured 1612.6 / 3388 (valid). New best.
  EMBPF (e0493, 3 rounds): +0.4, noise (off). DNBPF: -1.6 (1610.9 -> 1609.4), small; re-test with 5 rounds.
- Hypothesis (before measuring): RPF = the next position's cos/sin rows (2 x 128 B) prefetched into L2 during the last layer's down phase (layer 0's qkv epilogue reads them cold): gen B=1 -0..-1.
  DNBPF 5-round re-test (e0494): +0.4, noise. Not adopted (code off).
- Hypothesis (before measuring): QBPF / OBPF = L1 prefetch of the block's qkv / o_proj bias rows when its tasks are built (the epilogues read them with plain loads after the stream; GBPF's pattern): gen B=1 -1..-4 each.
