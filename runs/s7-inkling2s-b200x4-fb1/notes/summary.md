# Run s7-inkling2s-b200x4-fb1 (track B on the s6 engine): summary

**State after the extension (15 agent-hours; sections "Extension" below). The 12-hour summary follows unchanged from
"Inherited" on: it describes my own kernel (s0002-s0028), which the extension replaced by the sibling run's.**

## Extension (2026-11-15, 11.72 -> 15.00 h): track B on the sibling run's kernel

The operator extended the run by 3 agent-hours with three tasks in this order; all three are done. Then the
other dense matrices were measured in the 12-bit code on the new kernel: none pays.

### In short

- **Final track B: s0036 = `engine/`, 3291 us per token at B=1 and 5378 us per step at B=8 over HTTP = 1.431x / 1.228x
  SGLang** (4711 / 6604 us; with the operator's later 4.73 / 6.57 ms: 1.437x / 1.222x). TRACK = `B W12U L0T`; TRACK =
  `A` builds s0032's kernel. The same kernel was evaluated three times: s0031 3290.9 / 5375.7, s0033 3290.0 / 5377.5
  (the lowest figure on the board), s0036 3291.2 / 5378.3; they differ only on the host side (s0033: token-id check
  in `submit()`; s0036: the 12-bit round trip asserted at every load).
  s0035 is the same engine as one binary for both tracks: 3295 / 5383 (1.430x / 1.227x), 5 us slower by the build.
- **Track A baseline in this run: s0029 = the sibling run fa1's final source (its s0020), byte for byte: 3308 / 5394 us**
  (1.424x / 1.224x). s0032 is the same kernel from my final source with TRACK = A (context guard, dormant table code):
  3308 / 5395; s0034 (one binary for both tracks) 3307 / 5394.5. Three track-A builds within 1.3 us at B=1 and 1.7 us
  at B=8; one track-B build three times (s0031, s0033, s0036): within 1.2 / 2.6 us.
- **What the checkpoint is worth on this kernel: 12 us per step, 0.37 % at B=1 and 0.22 % at B=8**: s0035 against
  s0034, one kernel binary with and without the two changes, the evaluator's HTTP figures (3307.1 -> 3294.7, 5394.5 ->
  5382.7); phase stamps and in-process A/Bs say the same. (s0033 against its track-A twin s0032 shows 18 / 18 us, but
  those are two builds, and builds of this engine differ by several us for reasons outside the change: G11, G12.)
  Two items, both lossless and bit-identical to track A (the logits of every step compared, the tables for every
  token id, every packed value): the 12-bit unembedding and the layer-0 token tables.
- **A request or step beyond the context limit now fails loudly** (ValueError before anything is launched) in the
  track-B source, as the audit asked.

### Best submission, what worked, what did not, what next (extension)

- **Best submission: s0036 = `engine/` (track B), 3291.2 us per token at B=1 and 5378.3 us per step at B=8 over HTTP.**
  The lowest figure on the board is s0033, 3290.0 / 5377.5: the same kernel source (`ink.cu`, `build.sh` identical),
  1.2 us apart, which is the evaluator's repeatability; s0036 adds the load-time round-trip check and is the one kept.
- **Worked:** the sibling's kernel as baseline (s0029); the exact 12-bit unembedding and the layer-0 token tables
  (without the gate rows) on it, bit-identical to track A on 6,244 steps and for every token id, worth 12 us per step
  in a one-binary pair (s0035 against s0034); the context and token-id guards at no cost; one binary for both tracks
  by compiling the unembedding GEMV per kernel (s0034 / s0035); phase stamps and same-library switches as the way to
  measure.
- **Did not work:** the 12-bit code for router, o_proj and qkv (all slower; the qkv "gain" I first saw was a
  difference between builds); a table for the rest of layer 0 (nothing end to end); the first one-binary forms
  (s0030: B=1 8-10 us slower; `scratch/ni/`: B=8 60 us slower); comparing separately built variants at all.
- **Next:** the list at the end of this section; first the register lottery of the step function (up to 10 us at
  B=1 and 110 us at B=8 between builds of the same layer code).

### Submissions of the extension (one pod, epoch 5)

| submission | track | what | B=1 (us) | B=8 (us) |
|---|---|---|---|---|
| s0029 | A | run fa1's s0020 source, unchanged (`diff -r` empty): the baseline | 3308.4 | 5394 |
| s0030 | **B** | 12-bit unembedding + layer-0 tables on that kernel, both chosen at load in one binary; context guard | 3300.7 | 5376 |
| s0031 | **B** | the same with the unembedding decoder chosen at build time (the build with both decoders ran 8-10 us slower at B=1) | 3290.9 | 5376 |
| s0032 | A | source of s0031 with TRACK = A (guard + dormant table code) | 3307.9 | 5395 |
| s0033 | **B** | s0031 + `submit()` refuses token ids outside [0, 201024) (host side only; `ink.cu` identical) | **3290.0** | **5378** |
| s0034 | A | one binary for both tracks: unembedding decoder chosen at load, compiled per kernel (`scratch/mix/`) | 3307.1 | 5394.5 |
| s0035 | **B** | the same binary as s0034 with TRACK = `B W12U L0T`: **the clean pair, -12.3 / -11.8 us** | 3294.7 | 5382.7 |
| s0036 | **B** | s0033 + the 12-bit round trip asserted at every load (`ink.cu` identical): **`engine/`** | 3291.2 | 5378.3 |

### What was done

1. **Baseline.** `engine/` = the unpacked `s0020.tgz` of run fa1 (kept in `scratch/fa20/engine/`), submitted as s0029.
2. **Port of the two track-B changes** (journal, findings G9). The 12-bit unembedding went over as it was. The layer-0
   tables had to change: this kernel computes the 96 output-gate strips of the qkv GEMV beside the attention, in the
   blocks that hold no attention item, with a split that depends on batch and position, so the gate rows of a token
   are not summed in one order even in track A. They are left out of the table (they were in s0028); the table holds
   the normed embedding and the q, r, sink, k, v rows (141 strips, one order everywhere). `ink_l0t_ok()` states the
   condition in the source and the loader checks it.
3. **Verification before each track-B submission** (tools/x_verify.sh, x_bits.py, x_cmp.py, x_l0t.py; all on the pod):
   - logits: sha1 of the FP32 logits of every step, 64 random prompt tokens then greedy, for the unchanged s0020
     source, my source built as A, and built as B: equal on every step (B=1 512, B=8 1152 to position 1151, B=2 256,
     B=4 256, B=5 160; from e0112 on also B=3 160, 2,496 steps, the form in which s0034 / s0035 were checked);
     generate() and the online path return the same tokens; the final engine once more (e0122) at B=1 for 1300
     steps (across the 1024-position window), B=8 for 2048 steps (to the last position of the context), B=6 and
     B=7 for 200 steps: every batch size 1-8 covered, 6,244 steps in all;
   - 12-bit code: round trip of every packed matrix asserted at load (unembedding: 926,318,592 values, 55,788 escapes);
     since s0036 at every load, not only in the verification runs;
   - tables: layer 0 alone for all 201,024 ids on all four cards, the table as installed (B=8, position 0, tensor-core
     kernel) against B = 1, 2, 3 (plain kernel) and 4, 5, 8 (tensor-core kernel) at positions 0-2047: 0 of 1.85e9
     projection values and 0 of 3.7e9 normed-embedding values differ in every comparison; the tables' sha1 is the same
     from the A binary and the B binary.
4. **Context guard** (findings G10): `ctx_limit = min(max_context, 2176)`; `step()`, `generate()`, `submit()` raise
   before launching, `_launch()` refuses any position outside the limit. Tested at max_context 96 (B=8) and at the
   kernel's own limit (2176 positions at B=1, the first run of this kernel beyond position 1151).
5. **What else the checkpoint offers on this kernel** (findings G11): the 12-bit code for each other dense matrix,
   one at a time. Router +75 / +91 us, o_proj +25 / +16 us (slower, as on my kernel). qkv first looked like -10 us at
   B=1 and +22 at B=8 in separately built variants; I built a per-batch-size choice (both forms kept, a third kernel
   instantiation for a single sequence), verified it bit for bit, and then found with a switch inside one library and
   with stamps that the 12-bit qkv is **8 us slower at B=1** too (its GEMV 6.28 -> 7.16 us per layer). The first
   "gain" was a difference between builds. Not kept (`scratch/w12q_one/`). And the rest of layer 0 that depends on
   the token id alone (the 1/rms of its normed embedding, one more team reduction): -1.1 us in its phase, nothing end
   to end (`scratch/piv/`). Nothing beyond the two items pays.

### What the two items are worth here

In process on one pod, separate builds of one source (greedy 1024 tokens, min of 3, two rounds):

| build | B=1 | B=8 |
|---|---|---|
| s0020 source | 3304.6, 3304.0 | 5453.8, 5454.6 |
| my source, TRACK = A | 3304.0, 3303.8 | 5460.0, 5456.2 |
| B W12U (12-bit unembedding) | 3298.2, 3298.1 | 5442.0, 5445.6 |
| B L0T (layer-0 tables) | 3299.8, 3298.9 | 5448.9, 5451.0 |
| B W12U L0T (= s0031) | 3288.7, 3289.7 | 5436.3, 5437.4 |

Over HTTP, these two builds of one source: s0031 against s0032 -17.0 us (0.51 %) at B=1, -19 us (0.35 %) at B=8.

Phase stamps (card 0, means over 96 steps; findings G12), us at B=1 / B=8: layer 0's attention input 6.66 -> 4.54 /
6.46 -> 5.06; its main qkv GEMV 6.41 -> 1.88 / 6.43 -> 1.85; the unembedding GEMV 69.67 -> 64.78 / 70.96 -> 65.26:
**-11.5 / -11.7 us**. One binary with both items switched at load (s0030's source, in process): **-11 / -14 us**.
One binary with the unembedding GEMV out of line (`scratch/ni/`, e0118): 12-bit unembedding -4.6 / -5.9 us, tables
-5.7 / -8.0, both **-11.2 / -15.0 us**: the items add up.
Builds of this engine differ by up to +-10 us at B=1 and +-7 us at B=8 for reasons outside the change under test
(G11), so the stamps and the one-binary figures are the ones to quote, and the last pair of submissions put that
measurement into the evaluator: **s0035 against s0034, one binary: -12.3 us at B=1, -11.8 us at B=8 over HTTP**
(in that library in process: -11.9 / -11.6).

On my own kernel the same two items were worth 22-26 us (0.55 %) at B=1, measured on one binary six times. Here the
gate rows are no longer tabulated, layer 0's phases were already twice as fast, and the unembedding GEMV gains 5
instead of 10 us: about half the absolute gain on a 17 % faster kernel. The conclusion of the 12-hour run stands on
the faster kernel: this checkpoint has no redundancy a lossless engine could skip (census, findings G1), and exact
smaller storage helps only where a GEMV is bound by the bytes it streams (here: the unembedding, and only by 7 %).

### Where the time is now (findings G13)

Per layer at B=1 (context ~540, card 0): attention input 10.7, qkv 6.3, attention 10.8, o_proj 3.0, post-attention
10.6, router 3.0, routing 3.8, w13 8.0, quant2 5.4, w2 5.0 = 66.8 us; unembedding 64.9 us once per step. At B=8:
112 us per layer (NVFP4 layers 106-112 with their w13 on tcgen05, MXFP8 layers 134 with a 50 us w13). No per-layer
dense GEMV is bound by its bytes; the unembedding, 2 % of the step, is the only place where exact smaller storage
pays, and layer 0 the only place where anything depends on the token id alone.

### Lessons of the extension

- **Builds of this engine cannot be compared at the 10 us level.** Each build repeats to 0.3 us, and two builds
  whose difference is outside the code being run differ by up to 10 us at B=1 and 7 us at B=8. This produced one
  wrong conclusion that I acted on (a 12-bit qkv "gain" of 9.5 us in two builds, a loss of 8 us in a same-library
  A/B) and one wrong mechanism in a hypothesis (s0031: the two-decoder build is 8-10 us slower, but its unembedding
  GEMV differs by 0.8 us; the rest is the layers' code in that build). What works: a switch inside one library, as in
  the 12-hour run, or phase stamps of the changed phases.
- The two-decoder build (s0030) measured slower than the one-decoder builds, so s0031-s0033 are two builds of one
  source, tied together by bitwise logits and by the tables' sha1. One binary for both tracks came back at the end
  (s0034 / s0035) by compiling the unembedding GEMV per kernel in the form that left that kernel's layers fast.
- Before tabulating anything, check that track A computes it in one summation order. Here that needed reading the
  GEMV schedule (gate split), not just the model.
- Taking one change back at a time on one pod (tools/x_var.sh) showed which change moved the build's speed (not
  the loader order, not the layer-0 code); only the stamps showed where the time went.

### What I would try next on this kernel

1. **Take the register lottery out of the step function.** Each kernel is one function of ~21,000 instructions; a
   change of 56 instructions in the last phase re-assigns the registers of all 48 layers (disassembly, G11), and the
   layers then run up to 10 us (B=1) or 60-110 us (B=8, `scratch/ni/`, `scratch/piv/`) faster or slower for no reason
   in the source.
   The phases as separately compiled functions with fixed interfaces (the sibling started this for attention,
   routing and quantization) would make a build's speed a property of its source and probably find a few us for
   both tracks: s0033's two-build form is 5 us faster than the one-binary form of s0035 for no reason in the
   source, and s0020's assignment is merely the best one seen, not a chosen one.
2. **Track B, what is left:** almost nothing. The 1/rms of the normed embedding from a per-token table (removes the
   second team reduction of layer 0) was measured at the very end in one library: -1.1 us in its phase, nothing end
   to end (`scratch/piv/`, not kept). Untried: the 12-bit unembedding loop with two units in flight like the BF16
   loop (the unembedding is 65 us of the step; a few us at most). The gate rows of layer 0 could only be tabulated if track A summed them in
   one order; they are computed by idle blocks, so nothing would be gained.
   Also small, track A, for callers that synchronise every step (not the scored HTTP path): the four cards' kernels
   of such a step start 30 us apart because they are launched by four Python calls; one launch call for all four
   would give that back (findings G13; in the device loop the kernels start within 2.4 us).
3. **Input validation in `step()` / `generate()`:** `submit()` now refuses token ids outside [0, 201024) (s0033); the
   in-process calls still take any id, because their tokens are on the card and a check would cost a synchronisation.
4. **Track A:** the sibling run's own list (its summary, "What I would try next": overlapping the fixed phases with
   the expert GEMVs at B=8, the tensor-core path's fixed costs, the grid barriers).
5. **In the evaluator:** a track-B submission is only compared with track A by the agent. A bitwise comparison of the
   two tracks' logits on the hidden sequences (the gate numbers of s0029 and s0031 are equal in every printed digit,
   which suggests it but does not prove it) and a timing of both tracks from one library would make the track-B
   figure the evaluator's own.

---------------------------------------------------------------------------------------------------------------

## The 12-hour run on my own kernel (s0001-s0028)

Inherited: run s6's s0003 (track A). Unmodified it is **not valid** in this run's evaluator (s0001: http_tokens, a
sequence's logits depended on its batch slot); with the two explicit FMAs it is s0002: **5317 us per token at B=1, 9340 us
per step at B=8** over HTTP. SGLang: 4711 / 6604; the 1.10x line: 4283 / 6004.

## In short

- **Best submission: s0028 (track B), 3974 us per token at B=1 and 6433 us per step at B=8 over HTTP = 1.185x / 1.027x
  SGLang.** Its track-A twin (same source, engine/TRACK = "A") is s0027: 3997 / 6455 = 1.179x / 1.023x. The baseline of
  this run (inherited engine made valid) was s0002: 5317 / 9340. B=1 is past the 1.10x line (4283), B=8 is at parity
  (6604) and 6.7 % short of its 1.10x line (6004). 28 submissions, 27 valid.
- **Worked, model-specific (track B, lossless, logits bit-identical):** the unembedding in an exact 12-bit code and
  load-time token tables for layer 0: together 22-26 us at B=1 (0.55 %) and about 20 us at B=8 (0.3 %), in six same-pod pairs.
- **Worked, kernel (track A):** hardware E2M1 conversion instead of a compare chain in the input quantizers (-7 % at B=1);
  GEMV schedules without global hand-offs and with a block's two strips at once; prefetch, fence and cross-card store
  costs taken off the critical paths; compact w2 fragments, pipelined MXFP8 loads and larger NVFP4 groups at B>=2;
  tensor-core attention scores; 64-position attention items at B=1.
- **Did not work:** any smaller storage for the experts (the checkpoint has no redundancy); the 12-bit code for the
  per-layer dense matrices; pre-scaled residual queries; and a list of kernel variants that were bit-identical but slower
  (below, "Not valid / not kept").
- **Next:** see the last section.

## Result

| submission | track | what | B=1 (us) | B=8 (us) |
|---|---|---|---|---|
| s0002 | A | inherited engine + slot-order fix (the baseline) | 5317 | 9340 |
| s0003 | A | run s6's later kernel (tagged-data exchange, ...) | 4819 | 8895 |
| s0004 | A | parallel mailbox polls; B>=2 w2 prefetch issued at once; B=1 token phases as functions | 4794 | 8486 |
| s0005 | A | B>=2: w2 input as compact per-pair fragments in shared memory | 4797 | 8031 |
| s0006 | A | w13 / unembedding on block-contiguous strips, shared-memory hand-off; 4 FP4 units in flight at B>=2 | 4682 | 7614 |
| s0007 | A | loader only (buffers before the unembedding; unembedding storage chosen at load) | 4688 | 7606 |
| s0008 | **B** | unembedding in the exact 12-bit code (same binary as s0007) | 4679 | 7596 |
| s0009 | A | prefetch by idle warps; one cross-card store per destination; batched re-polls at B>=2 | 4568 | 7263 |
| s0010 | A | NVFP4 input quantization with the hardware E2M1 conversion; w2 inputs 16 per thread | 4246 | 7117 |
| s0011 | A | B>=2 routing per token block, conv helpers; top-8 with a shorter chain | 4239 | 7035 |
| s0012 | A | kernel prepared for the token tables, switch off (reference of the next two) | 4235 | 7044 |
| s0013 | **B** | layer-0 token tables (same binary as s0012) | 4221.5 | 7037 |
| s0014 | **B** | 12-bit unembedding + layer-0 tables (same binary; evaluated on the next pod) | 4195 | 7004 |
| s0015 | A | B>=2: MXFP8 loads pipelined (next unit's loads before the current unit's MMAs); pod 2 | 4215 | 6899 |
| s0016 | **B** | 12-bit unembedding + layer-0 tables on the s0015 binary; pod 2 | 4188 | 6884 |
| s0017 | A | B=1: attention scores on the tensor cores, value sum 8 dimensions per thread; pod 2 | 4174 | 6901 |
| s0018 | **B** | 12-bit unembedding + layer-0 tables on the s0017 binary; pod 2 | 4151 | 6882 |
| s0019 | A | B>=2: NVFP4 units loaded per group 6 (w13) / 5 (w2) instead of 4; pod 2 | 4176 | 6803 |
| s0020 | A | prefetch policy re-measured per kernel, reader-side fences removed, tensor-core scores at B>=2; pod 2 | 4074 | 6540 |
| s0021 | **B** | 12-bit unembedding + layer-0 tables on the s0020 source; pod 2 | 4051 | 6513 |
| s0022 | A | block-per-strip GEMVs: a block's two strips at once on the two halves of its warps; pod 2 | 4014 | 6425 |
| s0023 | **B** | 12-bit unembedding + layer-0 tables on the s0022 source; pod 2 | 3991 | 6406 |
| s0024 | A | B=1: attention items of 64 positions (chunk size a template parameter; 128-position twin beyond 2048); pod 4 | 3997 | 6459 |
| s0025 | **B** | 12-bit unembedding + layer-0 tables on the s0024 source; pod 4 | 3975 | 6436 |
| s0026 | **B** | s0025 with the MK_NO_RED switch ignored (api.py only, same kernel binary); pod 4 | 3975 | 6436 |
| s0027 | A | B>=2 kernel builds its pair map for every batch size (it can run a single sequence again); pod 4 | **3997** | **6455** |
| s0028 | **B** | 12-bit unembedding + layer-0 tables on the s0027 source; pod 4 | **3974** | **6433** |

Final track A: **s0027, 3997 / 6455 us = 1.179x / 1.023x SGLang** (inherited: 0.886x / 0.707x). B=1 is above the 1.10x line
(4283); B=8 is at parity (6604) but not at 1.10x (6004). Milestones: build, correct, device_loop, gpu_never_waits, parity.
Final track B: **s0028, 3974 / 6433 us = 1.185x / 1.027x**: the same source with engine/TRACK = "B W12U L0T".
Repeatability of the evaluator on one pod with one kernel binary: s0025 3974.72 / 6436.0, s0026 3975.02 / 6436.4 us:
the track-B difference of 22 us at B=1 is about 70 times the B=1 repeatability.
(The lowest B=8 figures on the board are s0022 / s0023, 6425 / 6406, on the previous pod: the B>=2 kernel of s0024 is
bit-identical to theirs, the 0.5 % is the node.)
The pod was replaced after s0013 and again before s0024; the nodes differ by ~0.5 % on the same binary (s0012 against
s0015 at B=1: 4235 / 4215 with an unchanged B=1 kernel). Track-B differences are therefore quoted only within one pod
and one binary.
engine/TRACK: "A" = track A; "B W12U L0T" = both model-specific changes (each can be given alone). One binary for all.

## What the checkpoint allows (track B) and what it is worth

Census of every weight (findings G1): no zero / duplicate / dead / unreachable structure anywhere; NVFP4 codes use 3.97 of 4
bits, MXFP8 codes 6.53 of 8 (no fixed-width code without ~1 % escapes), BF16 dense values fit 12 bits (F1). So nothing can
be skipped and only the dense BF16 share can be stored smaller. Two lossless model-specific changes exist and were measured,
each against the same binary with the switch off, logits bitwise equal:

| change | op level | B=1 over HTTP | B=8 over HTTP |
|---|---|---|---|
| unembedding in the exact 12-bit code (926,318,592 values, 55,788 escapes, verified bit for bit) | its GEMV 77 -> 67 us | -9.2 us (-0.20 %), s0008 vs s0007 | -10 us (-0.13 %) |
| layer-0 token tables (normed embedding + layer-0 projections per token id, built at load in 3.9 s) | layer 0: two norms and a 35 MB GEMV -> two row reads | -13.3 us (-0.31 %), s0013 vs s0012 | -7 us (-0.10 %) |
| both | | -26.4 us (s0016 vs s0015), -23.2 (s0018 vs s0017), -22.2 (s0021 vs s0020), -22.7 (s0023 vs s0022), -21.8 (s0025 vs s0024), -22.7 us (**-0.57 %**, s0028 vs s0027); in process -24 us | -15, -19, -27, -19, -23, -22 us (**-0.2 to -0.4 %**); in process -19 us |

On the s0022 kernel in process (one binary, same greedy text; exec e0090; us per step at B=1 / B=8):
track A 4013 / 6023; 12-bit unembedding 4002 / 6014 (-11 / -9); layer-0 tables 4002 / 6012 (-11 / -11);
both 3990 / 6002 (-23 / -21): the two add up, 0.57 % and 0.35 %.

The evaluator's gate numbers of s0020 and s0021 (relative L2 against the float64 reference, per position statistics) are
equal in every printed digit: the track-B engine returns the track-A engine's logits bit for bit on the hidden sequences too.

Measured and rejected: attention-residual queries stored pre-scaled by 4608^-1/2 (bit-identical, but +0.9 % at B=1:
findings G8); the 12-bit code for the per-layer dense matrices (qkv 8.8 -> 10.6 us, o_proj 6.1 -> 7.3, router 2.3 ->
3.4: they are not bandwidth-bound, findings G2).

Why so little (findings G5): a B=1 step of ~4240 us streams all its weights in ~700 us. The rest is per-token vector work
bound by the instruction rate of one SM, the exchanges between the cards and the fixed cost per GEMV: nothing a lossless
change of storage can reach. The model-specific share of this engine's 1.185x at B=1 is 0.55 % (0.3-0.4 % at B=8).

## What moved the engine (track A; all found with stamps on the score's workload, tools/t_prof.py)

1. **The E2M1 rounding as a compare chain, three times per value** (s0010, -7 % at B=1): the hardware conversion
   `cvt.rn.satfinite.e2m1x2.f32` rounds identically (tools/ptx/t5.cu); 16 values per thread remove the shuffles.
2. **GEMV schedule** (s0006, -2.4 % / -5 %): whole strips per block and a shared-memory hand-off instead of global slots.
3. **Serial bulk prefetches**: `cp.async.bulk.prefetch.L2` holds its warp ~0.4 us; the w2 prefetch over the union cost 20 us
   per layer at B=8 (s0004), the K/V + QKV prefetches 5 us in front of the B=1 attention-input phase (s0009).
4. **Dense 8-column activation fragments for w2 at B>=2** (s0005, -5 % at B=8): 1 KB of activations per 576 B of weights.
5. Cross-card store instructions cost per instruction, not per lane (s0009); stale mailbox words re-read together at B>=2.
6. MXFP8 stream bound by loads in flight: next unit's loads issued before the current unit is computed (s0015, B=8 -2 %).
7. Attention scores on the tensor cores at B=1 (s0017, -0.9 %).

8. Scheduling choices re-measured after the bottlenecks moved (s0020, -2.4 % / -3.9 %): K/V prefetch (serial bulk
   prefetches in front of a grid barrier at B>=2), w2 prefetch, reader-side fences, tensor-core scores at B>=2.
9. NVFP4 group size of the B>=2 streams 6 / 5 instead of 4 (s0019, B=8 -1.4 %).
10. Block-per-strip GEMVs (qkv, o_proj, router, w2): a block's two strips at once on 12 warps each instead of one after
    the other on 24: one synchronisation / reduction / epilogue round per GEMV less (s0022, -1.5 % / -1 % on a fixed stream).

11. B=1: attention items of 64 instead of 128 positions (s0024, -0.5 %): twice as many blocks busy, each with half the
    instruction count; the chunk size is a template parameter and a 128-position twin kernel serves contexts beyond 2048.

A caution for every B=8 number (findings G7): a change that is not bit-identical changes the greedy text and with it the
number of distinct experts per layer (44-50); such builds are compared on a fixed token stream (tools/t_replay.py).

## What limits it now (measured on the s0020 kernel, context 576; s0022 takes ~1 us per layer off qkv / o_proj / w2)

- B=1, 81.7 us per layer: attention input 14.4, qkv 9.3, attention 9.1, o_proj 6.2, post-attention 15.7, router 2.6,
  top-8 4.4, w13 11.4, quant2 4.2, w2 4.5; final 94 (unembedding 75.5, 66 with the 12-bit code).
  B=8, 135.9 us per layer: 14.4, 11.0, 12.9, 3.3, 15.6, 3.7, 5.4, w13 38.5, 7.2, w2 24.0; final 99.
- The token phases run at ~0.6 warp-instructions per cycle on the token's SM (18 warps), a single warp's dependent chain at
  ~16 cycles per instruction; the other cards' data is there within ~1 us of the slowest producer block. Offloading the
  element-wise part to idle blocks helps only as far as they can start earlier (B>=2: -1 %).
- B=8: the experts are 46 % of the step (~51 distinct experts per layer: 152 + 76 MB per card and layer at 4-5 TB/s with
  24 warps per SM); the 8 MXFP8 layers cost about 1.4x an NVFP4 layer. To reach 1.10x at B=8 (6004) the step would have
  to lose another 6.5 %: the stream rate (more loads in flight without spills: more registers per thread, i.e. fewer, fatter
  warps for the GEMVs only) and the five grid barriers per layer are the candidates.
- The step functions are fragile: 6 more live registers, 4.6 KB more shared memory or a different allocation order of the
  runtime buffers move unrelated phases by 1-6 % (frame / spill bytes: tools/ptxas_stats.sh). Outlined phases help the
  B=1 kernel and hurt the B>=2 kernel; every knob has to be measured per kernel and re-measured after other changes.

## Not valid / not kept

- s0001: the inherited engine unmodified (http_tokens).
- Query-dot cache for finished residual blocks, single polling block, conv helpers at B=1, 16-byte paired cross-card stores,
  block-contiguous schedule for w2 / qkv / o_proj / router, unrolled fragment staging, FP8 two units in flight, the
  load-ahead form for NVFP4, 1024 / 896 / 832 threads per block, gpu-scope local channels, attention with the 8-dimension
  value sum at B>=2, conv helpers called by the helper threads only: bit-identical where applicable, slower or equal
  (journal). The variants after s0020 (qkv prefetch off at B>=2: B=8 -0.3 %, B=1 +0.2 %) are in scratch/.

## A defect found and fixed late (s0027)

From s0005 to s0026 the B>=2 kernel built the pair map of its compact w2 fragments only for B > 1 and hung when given a
single sequence (reachable through the debugging switch MK_NO_RED=1; found when a test of mine held the pod for 40
minutes). s0027 builds the map whenever that kernel runs: logits on the normal paths bitwise unchanged, and a single
sequence runs through the B>=2 kernel again (4390 us per token). The evaluated B=1 path always used the B=1 kernels.

## What I would try next

1. **B=8 to 1.10x (another 6.7 %).** The expert streams are 46 % of a B=8 step and run at 4-5 TB/s with 24 warps per SM.
   More loads in flight per warp needs more registers than the 80 a 768-thread block leaves: the load-ahead form made the
   NVFP4 loop spill and the whole step 15-30 % slower. A GEMV-only kernel shape with fewer, fatter warps (or the expert
   GEMVs in their own launch with their own thread count) is the direct test; the MXFP8 layers showed the effect (-19 %
   on their w13 with more warps, -2 % of the step with one unit loaded ahead).
2. **The five grid barriers per layer at B>=2** (about 6 us of 128): replace the ones in front of fragment staging by
   per-token tags, which needs the remaining plain loads of cross-block data (c.w13sc, the k / v fragment groups,
   c.ofrag, c.qkvo) turned into L2 loads first.
3. **Token phases on more than one SM per token.** They are bound by instruction rate (0.6 warp-instructions per cycle
   on the token's SM), and the element-wise parts could be split over idle blocks; the conv helpers show both that it
   works at B>=2 (-1 %) and that the cross-block hand-over eats most of it. The reductions (mix dots, norms) are the
   hard part; a Gram-matrix form of the residual mix would remove one of them at the price of other rounding.
4. **Make the step functions robust.** Every experiment here was at the mercy of the register allocation of two huge
   inlined functions (6 more live registers moved unrelated phases by 1-6 %). Phases as own functions helped the B=1
   kernel and hurt the B>=2 one; finding out why (which GEMV loop loses its registers) would make every later change
   cheaper to evaluate.
5. **Track B has little left.** The census shows nothing to skip and no smaller exact code for the experts. What remains
   is small: a pipelined 12-bit unembedding loop (about 5 us per step) and tables for more of layer 0 only if its K/V
   convolution history were tabulated too (not possible per token id). A different direction, outside "lossless
   storage", would be needed for more: for example an exact certified argmax for greedy decoding instead of the full
   unembedding (77 us per step), which this run did not attempt.
6. **Measure B=8 on a fixed token stream in the evaluator's terms.** The B=8 figure moves by several percent with the
   greedy text (findings G7); a teacher-forced B=8 timing next to the free-running one would make submissions with
   different rounding comparable.

## Tools

tools/t_prof.py (the score's workload in process, phase stamps, union sizes), abv.sh / abv2.sh (build variants, ptxas frame
and spill bytes), ab_env.sh (load-time variants of one binary), cmp2.sh / t_cmp.py (bitwise logits of two builds),
t_xcard.py / t_arrive.py / t_arrive8.py / t_rc.py (cross-card timelines, arrival probes, stamps inside the poll+conv),
census_all.py (every weight of the checkpoint), ptx/t5.cu (hardware E2M1 conversion against the chain), t_attn0.py /
t_cmp_long.py (tensor-core attention against the scalar path), rehearse.py.
