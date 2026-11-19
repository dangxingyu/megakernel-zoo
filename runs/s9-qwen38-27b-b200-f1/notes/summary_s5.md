# Run s5-qwen38-27b-b200-b1: exact 12-bit weights for Qwen3.8-27B on one B200

Inherited: s4's s0048 (track A), re-measured here as s0001: 8940 us per token at B=1, 9808 us per step at B=8 over
HTTP. SGLang 0.5.23: 9523 / 10231; the project's 1.10x line: 8657 / 9301.

## Result

| submission | track | what | B=1 (us) | B=8 (us) |
|---|---|---|---|---|
| s0001 | A | inherited engine | 8940 | 9808 |
| s0007 | A | activation fragments loaded only for real columns (BF16 loop) | 8892 | 9812 |
| s0004 | B | exact W12: 12-bit codes, per-row 15-exponent windows, escapes patched into MMA operands | 8670 | 9193 |
| s0006 | B | + W12-only build (no unused stream call sites) | 8557 | 9136 |
| s0008 | B | + compact one-column activation layout at B=1 | 8409 | 9131 |
| s0009 | B | + 16-exponent row windows (half the escapes) | 8263 | 9009 |
| s0010 | B | + escape positions counted down (loop counters back in registers) | 8227 | 8975 |
| s0011 | B | + byte-permute word layout: 5-instruction decode per register | 8194 | 8938 |
| s0012 | B | + activation loads issued before the per-quad escape test | 8169 | 8908 |
| s0013 | B | + layout test hoisted out of the activation producers' loops | 8162 | 8897 |
| s0014 | B | + BF16 twins of out_proj / o_proj used by one-column passes | 8120 | 8914 |
| s0015 | B | + 96 KB per block of the next matrix prefetched into L2 during small ops | 8104 | **8860** |
| s0016 | A | grid-barrier back-off 200 -> 25 ns (BF16 engine; noise level) | 8880 | 9827 |
| s0017 | B | + the 25 ns barrier back-off | **8095** | 8862 |

Not valid: s0002 (W12 with a sparse fp32 correction: numeric_long max 0.713 > 0.592, see below), s0003 (exact W12 without
the multi-column prefill loop: http_tokens), s0005 (lost to machine failures; resubmitted as s0006).

Best: s0017, 8095 / 8862 us: 1.176x / 1.154x SGLang, 1.104x / 1.107x the inherited engine (s0015: 8104 / 8860). Every logit of every track-B submission is
bit-identical to the BF16 engine's (s0001): the gates see exactly the same numbers; only the speed differs.
engine/TRACK = "A W12" builds the track-B engine; "A" builds the track-A engine (BF16 storage) from the same source.

## The model-specific mechanism (notes/findings.md K1-K7; the inherited run's notes are in *_s4.md)

- The checkpoint's weights fit a 12-bit code [sign][exponent - lo_row (4 bits)][7 mantissa bits] (stored as
  [sign|mantissa] bytes and exponent nibbles, decoded with a byte permute) with one
  16-exponent window per output row; 1,922,136 of 25,621,954,560 streamed weights (7.5e-5) fall outside and are
  patched into the MMA operand registers from per-warp lists. 12.01 bits per weight; verified bit for bit on all
  257 streamed matrices (MK_W12_VERIFY=1); all GEMM outputs and logits bitwise equal to BF16 storage.
- Why per row: a few output rows of down_proj / out_proj / o_proj (residual-stream channels with large weights) hold
  thousands of above-window weights each under one window per tensor (layer 0 down_proj: one row 12,359); per-row
  windows remove these clusters, which otherwise serialize the slowest warp of an op.
- Why patching, not a sparse fp32 correction (s0002): the correction changes the rounding of 1-3 outputs per op and
  the gates' maximum over ~8000 positions moved from 0.457 to 0.713 (limit 0.592); bit-identity removes that risk.
- Why 16 exponents: with every escape patched in, code 0 needs no escape meaning; the escapes halve (the most common
  escape exponent is the one just below a 15-window).

## What limits it (measured)

- The 12-bit stream is bound by bytes in flight per warp, not by its decode (bench/w12b.cu: decode-free 12-bit loads
  only 2 % faster). In the engine (64 registers per thread) the W12 loop streams ~6.3-6.6 TB/s of real bytes against
  ~7.3 for BF16; the 25 % byte saving became 10-14 % per long stream, less for short ones.
- The W12 loop is sensitive to its instruction count in the engine (+54 integer instructions per quad: +20 %; a
  decode one instruction cheaper per register: -0.6 %), although the isolated bench was not.
- Exact escape handling costs ~300 us per B=1 pass (findings K6): 7.5 % of quads hold an escape; the patch's
  dependency into the MMA operands, not its instructions, costs, and no restructuring tried here removed it.
- A 2-stage cp.async ring helps in isolation (-6 % per stream) but its 96 KB of shared memory cost the GEMMs their
  L1 reuse of activations (+3.5 % per pass); with the compact activation layout it is neutral.
- Code generation of the stream loop decides several percent: extra call sites, predicated loads, a runtime stride or
  one register more each moved a pass by 0.5-1.8 % (journal).

## What did not work (all measured on the same card; details in journal.md)

- W12 with the escapes as a sparse fp32 correction (s0002): exact in exact arithmetic, but the changed rounding moved
  the gates' maximum over ~8000 positions to 0.713 (limit 0.592). Replaced by bit-identical patching.
- One exponent window per tensor: escapes cluster in a few large-weight residual rows and serialize warps (K1).
- Deeper W12 streams: two quads per iteration spilled before cold state moved to shared memory; three quads per
  iteration +21 %; rolling register prefetch and L2 prefetch of later quads no gain; a 2-stage cp.async ring gains
  ~300 us of stream but costs the same in L1 reuse (neutral even with the compact activation layout).
- Cheaper escape handling: peeked descriptors, branch-free patches, the escape quad out of line, per-lane lists
  (+3.5 %), per-lane lists without any branch (+20 %): all bit-identical, all slower than the shuffle walk (K6).
- Activation loads for real columns only inside the W12 loop (+1.4 %), a runtime stride for the compact layout
  (+3.4 %), two k-steps per 16-byte activation load (+1 %), the pair's activation loads before its escape test,
  rotated exponent nibbles, explicit global loads: slower or noise.
- Re-tuning knobs: near-prefetch budget, residual-row preload, shared-memory size: the defaults stay.
- Nsight Compute on this persistent kernel coincided with machine losses; not used again.

## Next

- The escape cost (K6, ~300 us at B=1): the escape-free decode loop runs ~3.7 % faster than the exact one. A
  structure with registers to spare (fewer, fatter warps, or a stream split into producer/consumer warps) could
  absorb the patches; within this kernel, code generation decided every attempt.
- The fixed cost per GEMM op (prologue + barrier ~3.5 us, hand-over tail ~2 us, 257 ops): ~1.4 ms of a ~8.1 ms pass.
- The B=8 recurrent op (1.1 ms per pass): its units need a second partial round at B >= 7.
- out_proj / o_proj at B=8 and other per-op storage choices: measure each op in both formats at each batch size.

## Building and checking (all from engine/ and tools/)

- engine/TRACK "A W12" (default here, = s0017): build.sh compiles with -DBAR_SLEEP_NS=25 -DNO_STAMPS -DW12_ONLY
  -DX_COMPACT -DW12_MIXED -DW12_PRMT; api.py packs every streamed matrix into the exact 12-bit format at load (+~2 s),
  keeps BF16 twins of out_proj / o_proj for B=1 passes and prefetches 96 KB per block of the next matrix during small
  ops. engine/TRACK "A": the track-A BF16 engine (= s0016).
- tools/w12_full.sh: per-GEMM bitwise check against BF16 storage at B=8 and B=1 (both engines in one process), the
  bit-for-bit storage check of all 257 matrices (MK_W12_VERIFY=1), logits A/B, submit/poll vs step greedy.
- tools/ab_src.sh REF.cu: same-card alternating A/B of the W12 engine against another mk.cu (code generation moves
  single changes by 0.5-2 %: never compare across runs or cards).
- tools/w12_census.py, tools/esc_census.py, tools/dup_census.py: the exact checkpoint counts of findings K1, K4, K5.
