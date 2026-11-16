# Journal of run s5-qwen38-27b-b200-b1 (inherited: s4 run's s0048; its notes are in *_s4.md)

## hour 0-0.2: baseline
- s0001: the inherited engine unchanged (track A): 8940 / 9808 us over HTTP (s4's figure 8938 / 9800). Eval 334 s.

## hour 0.2-1.9: W12 that pays (track B)
- Bench (bench/w12b.cu, isolation, 152x1024 persistent): the 12-bit decode loop streams 19-24 % more weights per us
  than BF16 (17 quads/warp: 20.6 vs 25.3 us; 36: 41.5 vs 51.5). W12 bytes without any decode: only 2 % faster than
  with decode -> the decode's integer work is not the limit; bytes in flight are (3 x 16 B per thread vs 4).
  L2 prefetch of quad q+2/q+4 (bulk or per line): no gain or worse. 3 quads per iteration: +6 % at 36 quads/warp.
- New engine loop gemm_stream_w12c (NG=1 only): the tuned BF16 loop with 3 loads + decode, no per-quad patch test;
  out-of-window weights carry code 0 (decodes to v0 = 2^(lo-1)) and are corrected at each tile piece's end by
  w*x - v0*x (two exact products) from a per-warp list. Exact in exact arithmetic; only the fp32 rounding order of
  ~1.5e-4 of the terms differs from BF16 storage.
- First try: 11.1 ms (slower than BF16's 8.9). Two causes, both found by measurement:
  1. a dynamic index into the accumulator array (a + (row>>3)*2) put the accumulators in local memory: STL.128 in
     every iteration of the hot loop (SASS). Fixed with constant indices.
  2. escapes cluster: with one window per tensor, a few rows of down_proj / out_proj / o_proj hold thousands of
     above-window weights (layer 0 down_proj: one row 12,359), so some warps had 300-900 serial corrections while
     the median had 20-60 (findings K1). Per-row windows: max per warp 10-20 (lm_head 175), escapes 4.73 M -> 3.79 M.
     Corrections made lane-parallel (each lane loads one entry's x values, then shuffles in list order).
- Check of each GEMM on identical inputs against BF16 storage (tools/w12_ops.py): outputs bit-identical except
  1-3 bf16 values per op (one rounding step). Bit-for-bit storage check (tools/w12_verify.sh, MK_W12_VERIFY=1):
  25,621,954,560 weights of 257 matrices, 0 mismatches. Logits vs BF16 storage over 48 steps: rel-L2 1.2e-2,
  top-1 98 % (the BF16 trajectory noise: any change of rounding gives this; per-op outputs are identical).
- Same card, CUDA events: B=1 8905 -> 8419 us (-5.5 %), B=8 10196 -> 9555 (-6.3 %).
- Where the rest of the 25 % went (tools/prof_kinds.py, stream-only): BF16 streams at ~7.3 TB/s in the engine, the
  W12 loop at ~6.3 TB/s of real bytes. gate_up stream 48.9 -> 42.2 us (-14 %), in_proj -11 %, down -7 %, out_proj
  0 % (6 quads per warp: fixed costs). Two quads per iteration spills at 64 registers (44 B) and is slower.
  Per-row table loads, list load, corrections together: ~0.5-1 us per op.
- s0002 (track B, W12 with sparse correction): predicted ~8450 / ~9230; INCORRECT: failed numeric_long, max rel-L2
  0.713 > 0.592 (median 0.0096 as BF16, top-1 99.1 % vs 98.9 % for BF16; s0001 max 0.457, p99 0.098 vs 0.114).
  Not a bug: on identical inputs all 257 GEMMs at B=8 agree with BF16 storage except 450 single-ulp roundings in
  total (tools/w12_ops.py ALL=1). The max over ~8000 positions is a tail statistic that any change of rounding
  re-rolls. Decision: no re-rolls; make W12 bit-identical to BF16 storage instead (escapes patched into the MMA
  operand registers, gemm_stream_w12e), so the gates see exactly the BF16 engine's logits.

## hour 1.9-2.8: exact (bit-identical) W12
- gemm_stream_w12e: escapes patched into the MMA operand registers (the tensor cores see the checkpoint's bf16 bits),
  so every result is bit-identical to BF16 storage. Lists at a fixed stride per warp (no header load), entries sorted
  by (quad, k-step, lane); one shuffle + compare per quad (pair) finds the ~15 % of quads with escapes.
  First version: no gain at B=1 (8962 vs 8936): the escape test sat before the quad loads, so every stream start
  waited a full round trip for the list; and a list refill every 16 escapes stalled. Loads issued before the test +
  a 32-entry ring refilled a half ahead: 8724. Activation loads of the patch path hoisted (down/out_proj read x from
  L2): 8603 us at B=1 (-4.0 %), 9518 at B=8 (-6.8 %); 257 GEMMs bitwise equal at B=1 and B=8, logits bitwise equal.
- s0003 (exact W12): INCORRECT, http_tokens (lengths_ok false): the HTTP path prefills 64-token prompts in passes
  of up to 32 columns (NG = 4), which still ran the old W12 loops (they do not know the per-row format). My tests
  covered step() only. Fixed: gemm_stream_w12e for NG = 2, 4 (k-step by k-step), shared-memory area per NG;
  tools/w12_full.sh now also runs submit/poll against step-by-step greedy (tools/test_online.py): equal at B=1, B=8.
- Side result (track A, not yet submitted): at B=1 the streams load activation fragments for all 8 MMA columns;
  lanes of columns >= NC can take zeros instead (bit-identical, tools/x_ab.sh). BF16: ~-0.7 % per pass; in the W12
  loop the same change costs +1.4 % (codegen), so it is opt-in (-DX_REAL_ONLY) and off in W12.
- s0004 (track B, exact W12 incl. prefill passes): predicted ~8730 / ~9170; VALID, measured 8670 / 9193 us over HTTP
  (s0001: 8940 / 9808): -3.0 % / -6.3 %; gates identical to s0001 (logits bit-identical by construction).
  B=8 is below the 1.10x line (9301); B=1 is 13 us above it (8657).
- s0005 (W12-only build): lost twice to "pod ended: signal" during its evaluation (not run a third time). The same
  build then ran a 1024-token online decode at B=1 and B=8 + timing on a card without incident: infrastructure.
- Ring A/B (tools/ring_ab.sh): 96 KB more shared memory alone costs +3.5 % per B=1 pass (8579 -> 8879 us), all of it
  in the GEMMs (gate_up +188 us, in_proj +57, down +41; tools/op_times_p.py with MK_SMEM_MIN); the cp.async ring
  with that memory: 8588. The ring gains ~300 us of stream and the smaller L1 takes them back. Left off (-DW12_RING).
- s0006 (= s0005 resubmitted): predicted ~8610 / ~9170; VALID 8557 / 9136 us over HTTP. Both below the 1.10x line
  (8657 / 9301); milestone h0_step.

## hour ~2.0 (board clock): the activations' L1 footprint at B=1
- Bench (bench/run_pad.sh): unused dynamic shared memory up to 150 KB does not slow the W12 / BF16 stream in
  isolation (200 KB: -12 %). So the engine's +3.5 % with 96 KB more is lost L1 *reuse*: at B=1 every warp reads the
  8-column activation fragments (80 KB for K = 5120), each block re-reading the range ~7 times.
- Real-column predicated loads in the W12 loop: +1-1.5 % again (codegen). Compact layout at NC = 1 (X_COMPACT: column
  0's fragment values only, 32 B per k-step; every lane reads them, no predicate; producers write via xg_idx): with a
  runtime stride +3.4 % (B=8 +5 %: the hot loop's address arithmetic); with the stride a template constant (separate
  B=1 instantiation gemm_stream_w12e<1, 4>): B=1 8570 -> 8428 us (-1.7 %), B=8 unchanged, logits bitwise equal,
  online == offline.
- s0007 (track A, real-column activation loads in the BF16 loop): predicted ~8880 / ~9800; VALID 8892 / 9812.
- s0008 (compact one-column activations): predicted ~8420 / ~9140; VALID 8409 / 9131.
- cp.async ring again, now with the compact layout (the L1 reason gone): B=1 8427 -> 8410 (-0.2 %, noise level),
  B=8 +0.3 %, and its online B=1 decode gave wrong tokens (offline logits were bitwise right; not debugged). Removed.
  Conclusion: at this point the W12 stream is not limited by bytes in flight that a 2-stage ring can add.

## board hour ~2.4: where the remaining W12 time goes; 16-exponent windows
- Spill fix attempt (escape-list pointer recomputed instead of kept): 8418 vs 8427, noise; reverted.
- Far L2 prefetch of the next matrix during small ops (MK_PF_FAR): 64 KB/block -0.2 % (noise), 256-384 KB +0.4-0.8 %.
- Cost of the escape patching, measured by emptying the lists (MK_W12_DROP_ESC=1, wrong results): B=1 8429 -> 8020 us,
  B=8 9492 -> 9180: ~400 us per pass, though escapes are 1.5e-4 of the weights (~15 % of quads hold one and leave
  the fast path).
- With every escape patched in from the lists, code 0 needs no escape meaning: a 16-exponent window per row
  (code = exponent - lo). Escapes 3.79 M -> 1.92 M (the most common escape exponent is the one just below the
  15-window). B=1 8430 -> 8268 us (-1.9 %), B=8 9493 -> 9378 (-1.2 %); bit-for-bit storage check 0 mismatches;
  logits bitwise equal to BF16 storage; online == offline.
- s0009 (16-exponent windows): predicted ~8250 / ~9030; VALID 8263 / 9009.
- Patch cost after s0009 (lists emptied): 8268 -> 8024 us, still ~245 us for ~7.5 % of quads (~0.6 us per patched
  quad per warp: leaving the fast path costs the warp's memory-level parallelism more than the patch instructions).
  Variant "peek the quad's entries once, patch k-steps with predicated selects": bit-identical but 8325 vs 8266 (+0.7 %,
  codegen). Reverted.
- The B=1 W12 loop kept its counters cnt and left in local memory (2 LDL + 2 STL per iteration; SASS, tools/sass_ctx.py).
  Escape positions now count the quads left in the share (the loop counter itself), so the loop carries cnt and
  tend (cnt at the tile end) instead of cnt, left and qrel; list end = a zero entry. Spill 36/44 -> 8/16 bytes;
  B=1 8268 -> 8224 us (-0.5 %), B=8 9379 -> 9321 (-0.6 %); storage check 0 mismatches, all bitwise checks pass.
- s0010 (countdown escape positions): predicted ~8220 / ~8960; VALID 8227 / 8975.
- Remaining spill of the B=1 loop: tend (one L1 reload per iteration). Freeing the escape-list pointer (recomputed in
  the refill instead): 8222 -> 8370 us (+1.8 %), reverted. Code generation of this loop is the limit of micro-tuning.
- Three quads per iteration for long shares (W12E_TRIPLE): spills, +21 % per pass. Reverted.

## board hour ~3.0: what the patching costs, and why it resists
- Timing-only variants of the B=1 loop (wrong results; same-card events, B=1):
  s0010 8225 | list walked, patch instruction removed 8273 | quad's entries consumed by a minimal loop, then the plain
  decode (no patch code at all) 7920 | lists emptied 7987.
  So neither the patch instructions nor the list walk cost much: the per-k-step slow path does, by being executed
  and by being present in the loop function.
- Descriptor design (walk once, <= 2 descriptors per lane, predicated patch; packer stores quads beyond that raw:
  only 2 raw quads in the whole model): bit-identical, 0 spills, but 8727 (walk on every quad), 8341 (walk only for
  quads with entries), 8380 (branch-free patch), 9060 (escape quads out of line: the call spills the loop's state).
  Without the descriptor patch (wrong results) the same code runs 7986: using the walked descriptors in the MMA
  operands is what costs, ~0.9 us per escape quad per warp, far more than its instructions; a code-generation effect
  I could not get around. Reverted to s0010.
- Open: ~300 us per B=1 pass (3.7 %) are tied up in exact escape handling. A packer that removes escapes altogether
  (no fixed-width code does, K4) or a kernel structure with registers to spare (fewer, fatter warps) would release it.
- mk profile ncu on the W12 engine was running when pods 3 and 4 died (not retried; kernel replay of a persistent grid-barrier kernel may hang). Weight bytes per step are exact from storage: 38.45 GB (W12, incl. per-row bases) vs 51.24 GB.
- Last loop variant: two-quad loads issued whenever 2+ quads remain, the tile-end test after the loads (so the spilled
  tend is not needed before them): 8247 vs 8224 us, reverted. engine/ = s0010's source (diff -r against its src.tgz).

## board hour ~3.5-3.7: per-lane escape lists (reverted)
- SASS of the s0010 escape path: every k-step of a quad with escapes runs SHFL.IDX + BRA.DIV checks + BSSY/BSYNC
  regions before its HMMA (tools/sass_fn_dump.py). Design without any shuffle: per-lane lists (each lane keeps its
  next two entries in 2 registers, a quad is tested with one vote, patches predicated on the lane's own register,
  lanes refill from their own list; quads with 3+ escapes in one lane stored raw: 2 in the model). Bit-for-bit
  storage check 0 mismatches, 257 GEMMs and logits bitwise equal, online == offline, 0 spills.
- But slower: 8510 us at B=1 (s0010 8225); with the lists emptied 7903. Timing-only variants: without the refill
  load 8328, without the second-descriptor patch 8501, without both and without the raw branch 8343. So with no
  memory access and no shuffle left in it, the escape path still costs ~1.2 us per pair with an escape (~2.5x a plain
  pair), the size of a memory round trip. Not found without a profiler (Nsight Compute on this kernel coincided with
  pod losses). Reverted to s0010.
- Per-lane lists, branch-free: every quad applies the lane's descriptor with selects (no vote, no branch; quads with
  2+ escapes in one lane raw: 3363): bit-identical, but 9885 us (+20 %). ~54 extra integer instructions per quad
  cost ~1.6 ms: in the engine the W12 loop is sensitive to its instruction count (the isolated bench was not).
  Reverted. Next: a cheaper decode.
- Byte-permute W12 layout (W12_FMT=prmt, -DW12_PRMT): per lane and k-step [sign|mantissa bytes 0-3][bytes 4-7]
  [8 exponent nibbles]; decode per register = prmt with sign replication, mask, shift+mask, 3-input add (5 instead of
  ~6-7). Same 12 bits per weight, same windows and escape lists. Bit-for-bit storage check 0 mismatches; 257 GEMMs
  at B=8 and B=1 and logits bitwise equal; online == offline. Same-card alternating A/B: pack 8225 / 9321 us,
  prmt 8176 / 9265 (-0.6 % / -0.6 %).
- s0011 (byte-permute layout): predicted ~8180 / ~8920; VALID 8194 / 8938.
- Explicit ld.global for activation loads (instead of generic LD): 8178 vs 8177, no change; reverted.
- Activation loads of a single quad issued before its escape test (w12e_quad1): same-card alternating A/B vs s0011
  8176 -> 8151 us at B=1, 9261 -> 9233 at B=8 (-0.3 %); all checks pass. Two k-steps of the compact activation
  layout per 16-byte load (-DX_C2, half the activation load instructions): 8240 vs 8155, slower; left off.
- s0012: predicted ~8170 / ~8910; VALID 8169 / 8908.
- Exponent nibbles placed so that register 0 needs no shift and register 3 a rotation (-8 instructions per pair): 8157 vs 8152, no gain; reverted.
- The pair's 8 activation loads also before the escape test (X_PAIRPRE): 8249 vs 8157, worse (8 more live registers through the test); reverted. engine/ = s0012.
- The compact-layout test of the activation producers (lmix, attention output) hoisted out of their write loops:
  same-card alternating A/B vs s0012: 8153 -> 8142 us at B=1, 9231 -> 9216 at B=8 (-0.1 / -0.2 %); all checks pass.
- s0013: predicted ~8160 / ~8895; VALID 8162 / 8897.
- The same hoist in the GEMM epilogues and embed: 8160 vs 8142, worse; reverted. engine/ = s0013.
- Mixed storage per batch size: the BF16 stream call sites now cost only ~0.1 % in the W12 build (8148 vs 8142).
  out_proj / o_proj (6 quads per warp, ~1 escape per warp) stream faster in BF16 at one column but slower at B=8
  (all out_proj/o_proj in BF16: B=1 -0.5 %, B=8 +0.7 %). So both copies are kept (+4 GB) and one-column passes run a
  program whose out_proj / o_proj rows use the BF16 copies (api: progs["full_b1"]; the BF16 loop got the compact
  one-column layout, gemm_stream_bf<1, 4>). First try crashed: the twin rows were not wired to the activation
  buffers (fixed by copying XF / OUT). Same-card A/B vs s0013 (whole engine dirs, tools/ab_dir.sh): B=1 8148 -> 8104
  (-0.55 %), B=8 9215 -> 9245 (+0.3 %, the extra call sites); logits bitwise; online == offline.
- s0014: predicted ~8120 / ~8925; VALID 8120 / 8914.
- Prefetch budgets on the s0014 engine (same card, alternated): near 64/196/256 KB worse than the default 128 KB;
  far (next matrix into L2 during small ops) 32 KB 8095, 64 KB 8091, 96 KB 8083, 128 KB 8089, 192 KB 8117 vs 0:
  8102 us. Default far = 96 KB for the W12 engine (BF16 stays 0): whole-engine A/B vs s0014 8104 -> 8088 (-0.2 %),
  B=8 9248 -> 9242.
- s0015 (far prefetch 96 KB, W12): predicted ~8105 / ~8910; VALID 8104 / 8860 (pod lost once during its evaluation; rerun).
- Grid-barrier spin back-off (BAR_SLEEP_NS) on the s0015 engine, 4 alternations: 400 ns 8115, 200 ns 8092, 100 ns
  8086, 50 ns 8082, 25 ns 8078, 0 ns 8081 us (B=8: 9240 -> 9229 at 25 ns). A kernel knob: submitted on track A first.
- s0016 (track A, barrier back-off 25 ns): VALID 8880 / 9827 vs s0007 8892 / 9812: noise level on BF16.
- s0017 (W12 + 25 ns barrier back-off): predicted ~8090 / ~8850; VALID 8095 / 8862 (pod lost once during it; rerun).
- Far-prefetch re-check with the 25 ns back-off (s0017 engine): 64 KB 8097, 96 KB 8090, 128 KB 8096 us: 96 KB stays. End of run: engine/ = s0017.
