# Findings about the checkpoint and the node (each with the measurement behind it)

## F1. BF16 dense weights fit an exact 12-bit code (track B)  [measured, all weights]
tools/census_bf16.py, exec e0020: every value of the 337 dense BF16 matrices of the decoder (per layer wq, gate, wk,
wv, wo, wr, router; and the unembedding): 5,710,970,880 values.
- zeros: 1; subnormals: 0; exponent entropy 2.57 bits.
- values outside the best 16-exponent window chosen per row: 419,063 (7.3e-5); per 16-row strip: 596,073 (1.0e-4);
  per tensor: 668,233 (1.2e-4).
- As packed by engine/w12.py for my layout (row-sliced per card, plus the per-card zero pad rows which get window
  base 0 so that their zeros are exact): 432,825 escapes in 5,984,649,216 values, 12.0009 bits per value.
Allows: 25% fewer bytes for the 11.5 GB of dense weights read every step (~2.9 GB per step at B=1, ~720 MB per card).
Does not allow: anything below 12 bits with a fixed-width code (entropy 2.57 bits of exponent + 1 sign + 7 mantissa).

## F2. 12-bit decode speed on one B200  [measured, tools/bench/w12.cu, e0023]
GEMV streaming of 150 M values: BF16 3.13 G values/s (6.26 TB/s), 12-bit 3.99 G values/s (5.98 TB/s of 12-bit bytes):
+27% values per second. Decode per bf16x2 register: PRMT with sign-replicate (sign to bit 15/31) + LOP3 mask +
SHF/LOP3 nibble extract + IMAD (offset*128 + base) + LOP3 = ~5 instructions.

## F3. Engine-visible properties of the node
- sm_100a: no warp-level block-scaled FP4 MMA; cvt.rn.f16x2.e2m1x2 with free byte selects; NVFP4 GEMV via
  cvt + HMUL2 (E4M3 block scale, exact in f16) + mma.m16n8k16.f16 streams at 5.9 TB/s (BF16 6.2 TB/s) per card.
- 1024-thread blocks get 64 registers; with 222 KB of shared memory per SM the L1 has ~34 KB: any register spill or
  dynamically indexed local array costs an L2 round trip per access (seen as 5-10x slow phases).

# Run s7-fb1 (track B): findings on this run's node

## G1. The checkpoint holds no redundancy a lossless engine could skip  [measured, all weights, exec e0002, tools/census_all.py, 61 s]
Every tensor of the text decoder, integer counts on the GPU:
- Experts, NVFP4 layers 0-39 (w13 [384,4608,4608], w2 [384,4608,2304] codes; one E4M3 scale per 16): zero codes 6.9 % (+0 and -0
  about equally: 3.4 % each), **no** all-zero block, row or column in any expert, no zero or negative block scale; 0 dead
  intermediate neurons of 884,736 per layer (a neuron would be dead if its gate row, its up row or its w2 column were zero);
  1,769,472 of 1,769,472 rows distinct in every layer, 18,432 of 18,432 experts distinct in the model (64-bit row hashes).
  Order-0 entropy of the 4-bit code 3.968 bits; of the code byte 7.93 bits; of the E4M3 block scale 3.8-4.4 bits (41-53
  distinct codes per layer, all in 74..126). amax: 384 distinct values per layer, denominator 2688 everywhere.
- Experts, MXFP8 layers 40-47: E4M3 code entropy 6.53 bits (254 codes used; exponent field 12-14 holds 73 %, 15: 8 %, below 8:
  1.2 %); UE8M0 block scale entropy 0.68-0.77 bits: 79-83 % of the blocks have scale 114 (2^-13), 16-21 % 113 or 115, 7-9
  distinct values. No zero rows/columns, all rows distinct.
- Router bias: range -0.94 .. +0.15, spread within a layer 0.16 .. 0.99 (< 1, the width of the sigmoid): all 384 experts of
  every layer can be chosen (bias + 1 > the 8th largest bias) and none is always chosen.
- Dense BF16 (wq, gate, wk, wv, wo, wr, router, sink, dynamic-tap projections, embed, unembed): one zero value in 6.6e9, no zero
  row or column, no duplicate row (also none among the 200,058 real unembedding / embedding rows), no two identical tensors,
  embedding and unembedding not tied; mantissa entropy 6.972 of 7 bits, low mantissa bits uniform (no coarser grid).
- Small tensors (norm gains, short-conv taps, clip scales, relative-bias projection [16,1024], attention-residual queries):
  no zeros, no identical tensors, no zero tap (the oldest taps are small, mean |w| 0.04-0.12, but not zero), relative-bias
  projection non-zero at every distance.
Allows: nothing to skip (no pruned neuron, no shared or duplicate expert, no unreachable expert, no zero tap or bias column).
Storage: BF16 dense -> 12 bits (F1, 25 % of 11.5 GB); MXFP8 -> ~6.55 bits by entropy but no fixed-width code without ~1 %
escapes (7 bits: exponents 8..15 cover 98.8 %; per-block windows do not help because a block of 32 near-Gaussian weights
usually holds a small one); NVFP4 codes are incompressible (3.97 of 4 bits), their scales 4 of 8 bits (0.25 bit per weight).
Does not allow: fewer expert bytes per step. The experts are 97 % of the checkpoint and 41-85 % of a step's bytes.

## G2. What the 12-bit dense code is worth in this engine: +16 % on the unembedding, a loss on every other GEMV  [exec e0005]
Same kernel (s0003), BF16 dense against W12 dense (engine/TRACK "A W12"), the score's workload (tools/t_prof.py), logits
bitwise identical (tools/t_cmp.py, 6 steps at B=1 and B=8). The W12 build of run s6 was 10x slower only because its L2
prefetches pointed at the (null) BF16 arrays: a prefetch of an unmapped range faults; fixed (PF_QKV / PF_WO / PF_WR).
| us per op | BF16 B=1 | W12 B=1 | BF16 B=8 | W12 B=8 | bytes per card |
|---|---|---|---|---|---|
| qkv (q, gate, r, sink, k, v) | 8.8 | 10.6 | 11.3 | 11.8 | 35 MB -> 26 MB |
| o_proj | 6.1 | 7.3 | 3.8 | 4.8 | 14 MB -> 10.6 MB |
| router | 2.3 | 3.4 | 3.8 | 4.9 | 3.5 MB -> 2.7 MB |
| unembedding | 79.4 | 66.8 | 83.5 | 74.5 | 463 MB -> 347 MB |
| whole step | 4818 | 4950 | 8948 | 8995 | |
Allows: W12 for the unembedding only (-12.6 us per step at B=1 = 0.26 %, -9 us at B=8 = 0.1 %).
Does not allow: a gain on the per-layer dense GEMVs. They are not bandwidth-bound here (35 MB at 6 TB/s would be 5.8 us; the
op takes 8.8): the 25 % fewer bytes are worth less than the decode (5 integer instructions per register, a header load per
unit and the escape walk). The dense weights are 59 % of a B=1 step's bytes but their streams are 17 us of a 97 us layer.

## G3. Track B, item 1: the unembedding in the exact 12-bit code  [s0008 against s0007, same binary]
926,318,592 BF16 values, 55,788 escapes (6.0e-5), verified bit for bit at load (MK_W12_VERIFY) and by bitwise equal
logits. The unembedding GEMV: 77 -> 67 us per step at both batch sizes. Over HTTP: 4687.7 -> 4678.5 us at B=1 (-0.20 %),
7606 -> 7596 at B=8 (-0.13 %, inside that figure's noise).
Does not allow more: the unembedding is the only dense GEMV of this model that is bandwidth-bound (463 MB per card at once);
the per-layer dense GEMVs (35 + 14 + 3.5 MB per card and layer) lose with the 12-bit code (G2).

## G4. Track B, item 2: what layer 0 computes from the token id alone  [exec e0053, one binary]
The first finished block is the RMS-normed embedding, and layer 0's attention input is its norm with layer 0's gain: the
normed embedding (4608 BF16 per id) and the card's q / gate / r / sink / k / v projection rows of layer 0 (3792 of 3840 BF16
per id and card, 1.54 GB per card) depend on the token id only. Tabulated at load with the engine's own kernel (25,128
one-layer launches of 8 ids, 3.9 s): a step then reads two table rows instead of two norms and a 35 MB GEMV. Logits bitwise
equal (tools/t_cmp.py, B=1 and B=8). In process: B=1 4236 -> 4222 us (-0.33 %), B=8 7095 -> 7088 (-0.1 %).
With the 12-bit unembedding as well: 4212 / 7076 (-0.57 % / -0.27 %).
Does not allow more: from layer 0's attention on, everything depends on the context (K/V of earlier positions, the short
convolutions' histories), and the attention-residual mixes of later layers depend on all earlier outputs.

## G5. Why track B is worth 0.6 % here and was worth 10 % for Qwen3.8-27B (run s5)
- 97 % of this checkpoint's bytes are experts in 4-bit / 8-bit codes with no redundancy (G1); the dense BF16 share of a step
  is 11.5 of 19.3 GB at B=1 but in 145 small GEMVs (3.5-35 MB per card) that are bound by latency, not by bandwidth.
- A B=1 step of 4240 us streams its weights in about 700 us; the rest is the per-token vector phases (instruction count on
  one SM), the exchanges between the cards and the fixed cost per GEMV. A lossless storage change can only touch the 700 us.
- What did move the engine was not model-specific: see the journal (track A, 5317 -> 4239 us at B=1, 9340 -> 7035 at B=8).

## G6. Facts about the card found in this run (each with its measurement)
- `cp.async.bulk.prefetch.L2` holds the issuing warp ~0.4 us per instruction, whatever the number of lanes: 51 serial ones
  cost 20 us (e0006), 4 + 7 in front of a phase 5 us (e0030). Issue them from warps that are idle, or in one instruction
  per range, and never in front of a barrier other blocks wait at.
- A cross-card store instruction (`st.relaxed.sys` to peer memory) costs per instruction, not per lane or byte: halving the
  number of stores changed nothing, halving the number of store instructions gave 0.9 % (e0027, e0028).
- A system-scope load of a mailbox word takes 40-210 ns (e0032); the data of the other cards is present within ~1 us of the
  slowest producer block (e0033, e0056): what looked like waiting was instruction time of the consumer.
- One SM runs per-token vector code (18 warps of 576 threads) at ~0.6 warp-instructions per cycle; one warp alone runs a
  dependent chain at ~16 cycles per instruction (stamps inside the poll+conv, e0035; top-8).
- `cvt.rn.satfinite.e2m1x2.f32` + `cvt.rn.f16x2.e2m1x2` round to E2M1 exactly like the specification's nearest-even rule,
  signed zeros included (3096 values around every tie, e0038); a NaN input gives 6 instead of 0.
- The placement of the runtime buffers (allocation order) moves the B=1 step by up to 1.5 % (e0020); the same binary ran
  the layer loop 25 us slower after the 12-bit packer had run before the buffers were allocated (e0018).
- The second machine's node runs the same binary 0.45 % faster than the first (s0012 / s0015, in process 4236 / 4217).

## G7. The B=8 step time depends on the text, and the text on the rounding  [execs e0082-e0084]
A change that alters only the summation order (not bit-identical) flips near-tie choices, the greedy continuation of the
same prompts becomes another text, and with it the number of distinct experts per layer: 44.1 against 49.9 at one probe
step, and the in-process B=8 figure on the greedy workload went from 6473 to 6022 us for a kernel change that is worth 1.0 %
on a fixed token stream (6497 -> 6432, tools/t_replay.py: the saved tokens teacher-forced through step()).
Allows: comparing bit-identical builds on the greedy workload (same path); comparing others only on a fixed stream.
Does not allow: reading a B=8 difference of a few percent between two submissions with different rounding as a kernel
effect (the evaluator's B=8 figure is measured on each engine's own greedy text). B=1 always has 8 experts per layer.
In this run the non-bit-identical kernel steps were s0006 (GEMV schedule), s0017 (B=1 attention), s0020 (tensor-core
scores at B>=2: 0.35 % on the fixed stream) and s0022 (paired strips); all others are bit-identical to their predecessor.

## G8. Track B, item 3 (measured, not kept): attention-residual queries stored pre-scaled by 4608^-1/2  [exec e0091]
The mixes multiply every query element by 4608^-1/2 per use; the products can be formed once at load (the same fp32
multiply: logits bitwise equal). As a second copy next to the unscaled queries (the B>=2 kernel keeps the unscaled ones),
selected at load in the B=1 kernel: B=1 4017 -> 4053 us (+0.9 %), B=8 6018 -> 6036 (+0.3 %, although that kernel does not
use them). The saved 8 multiplies per query and thread are outweighed by where the 2 x 48 extra small tensors land and what
that does to the placement of everything allocated after them (the same sensitivity as in G6). Not used; the variant is in
scratch/ink_qpre_variant.cu, scratch/api_qpre_variant.py.

## G9. Track B on the sibling's kernel (run fa1, s0020): what carries over, what does not  [execs e0105-e0109, s0029-s0031]

The two items of G3/G4 were moved onto the faster kernel of run fa1 (plain kernel at batch 1-3, tcgen05 kernel from
batch 4). Measured on one machine, in process, greedy 1024 tokens after a 64-token prompt (text identical in all builds,
the logits are bit-identical): s0020 source 3304 / 5454 us; 12-bit unembedding alone 3298 / 5444; layer-0 tables alone
3299 / 5450; both 3289 / 5437: -15 us at B=1, -17 us at B=8 between these separately built libraries. The figure from
one binary, by the evaluator, is **-12.3 us (0.37 %) at B=1 and -11.8 us (0.22 %) at B=8** (s0035 against s0034; G12
for all measurements of it, G11 for why separate builds overstate it here). On my kernel the same two items gave
-22..26 us (0.55 %) at B=1. Differences that matter:

- **The output-gate rows of layer 0 cannot be tabulated bit-identically on this kernel.** Its qkv GEMV is split: the 96
  gate strips are computed beside the attention by the blocks that hold no attention item, 1 to 4 strips per block
  depending on how many blocks are free, i.e. on batch and position (B=8: one strip per block up to ~96 positions, then
  2-4 with the 24 warps divided among them). The FP32 summation order of the same token's gate rows therefore changes
  along a sequence in track A itself. A table holds one order; it would be equal in exact arithmetic but not bit for
  bit, and nothing would be gained either: those blocks are idle. The table here holds what is summed in one order
  everywhere (q, r, sink, k, v: 141 strips, one per block, 72 units over 24 warps) plus the normed embedding.
  `ink_l0t_ok()` in the source states the condition (team phases, gate split always taken in a local layer: needs
  nblk - 8 x 2 x 8 >= 24, true with 152 blocks) and the loader refuses the tables if it does not hold.
  Verified for every token id, all cards, 8 configurations (B = 1, 3 plain kernel; 4, 5, 8 tensor-core kernel;
  positions 0-2047): 0 of 5.56e9 values differ; the tables' sha1 is the same from the A and the B binary.
  Logits of the final engine (both tracks) against the unchanged s0020 source: bit-identical on 6,244 steps, every
  batch size 1-8, B=1 to position 1299 and B=8 to position 2047 (e0109, e0112, e0122).
- **The build that carries both unembedding decoders (s0030) is 8-10 us per step slower at B=1 than the build with
  one (s0031), in either track; the cost is not in the unembedding.** Found by taking one change back at a time
  (e0108: 3312 with both decoders against 3304 with one, three builds each way) and seen over HTTP (s0030 3300.7,
  s0031 3290.9). Stamps (e0115) put it elsewhere: the unembedding GEMV takes 70.4 us (two decoders, BF16 chosen)
  against 69.6 (one), and 64.7 against 64.8 with the 12-bit code, while the 48 layers take 8-13 us more in the
  two-decoder build. It is an effect of the build on unrelated code (G11), not the price of a branch in the GEMV. On
  my kernel the same load-time choice cost nothing (s0007/s0008). The storage is a build choice here (`-DW12U` from
  the track word) because that build measured faster, with the consequence that the two tracks are two binaries.
  The same-binary A/B of the two items therefore comes from s0030's source (e0107): **-11 us at B=1 (3312 -> 3301),
  -14 us at B=8 (5448 -> 5434)**.
- The loader order (unembedding before or after the runtime buffers), which mattered on my kernel (G3), makes no
  difference here (3311.9 against 3311.9 / 3312.8).
- The layer-0 code in the kernel (mode word, table copy, `qkv_skip`) costs nothing with the tables off (3312.1 with it
  compiled out against 3311.9 / 3312.5 with it).

## G10. The context limit was not enforced by either kernel  [audit `audits/ink-final`, open doubt 1; e0106, e0108]

Positions >= max_context write K/V rows into the next (sequence, head, layer) slice of the cache; positions >= 2176
need an 18th chunk record in a global layer and overwrite the next head's records (MAXCHUNK = 17 x 128). Neither the
sibling's kernel nor mine checked. In the track-B source (s0030, s0031) `Engine.ctx_limit = min(max_context,
ink_max_positions())`; `step()`, `generate()` and `submit()` raise ValueError before anything is launched and
`_launch()` refuses any position outside [0, limit) whatever the caller. Tested at max_context 96 (B=8) and at the
kernel's own limit with max_context 4096 (B=1: 2176 steps run, logits finite; the 2177th is refused): a request that
ends exactly on the last position is served and equals generate(). Token ids: `submit()` refuses ids outside
[0, 201024) since s0033 (they index the embedding, and with track B the tables); `step()` and `generate()` do not
check, their tokens are on the card and a check would cost a synchronisation per call.

## G11. The 12-bit code for the other dense matrices on the sibling's kernel: none pays; and how I first got it wrong  [execs e0111-e0114]

One matrix at a time on top of the 12-bit unembedding. Logits bitwise equal to the s0020 source in every variant, the
round trip of every packed matrix asserted (qkv: 3,359,637,504 values in 192 matrices, 250,853 escapes).

**Result, from a same-library A/B and from phase stamps (e0114):** one library that holds both forms of qkv and both
kernels; a single sequence run through the kernel that reads the 12-bit form, or (MK_NO_ONE=1) through the BF16 one:
**3299.6 / 3300.0 us per step with the 12-bit form against 3291.8 / 3292.3 with BF16: +8 us.** By stamps the qkv
GEMV takes 7.16 instead of 6.28 us per layer. Router (+75 us at B=1, +91 at B=8) and o_proj (+25 / +16) lose by more
than any placement effect. So on this kernel, as on mine (G2), the exact 12-bit code pays for the unembedding only:
the per-layer dense GEMVs are bound by their decode instructions and hand-offs, not by bytes.

**How the first two measurements misled me.** Comparing separately built variants in process (us per step, two rounds
each, repeatable to 0.3 us within a build):

| build (all with the 12-bit unembedding) | B=1 | B=3 | B=8 |
|---|---|---|---|
| e0111: BF16 qkv, no tables | 3297.2, 3298.5 | | 5441.5, 5443.6 |
| e0111: 12-bit qkv in every kernel, no tables | 3288.1, 3288.3 | | 5463.9, 5465.2 |
| e0112: tables, BF16 qkv (= s0031) | 3289.5, 3288.7 | 3893.3, 3893.2 | 5436.9, 5437.1 |
| e0112: tables, 12-bit qkv in the plain kernel (batch 1-3), BF16 in the tensor-core kernel | 3279.9, 3279.8 | 3924.3, 3922.9 | 5430.9, 5432.0 |
| e0113: tables, 12-bit qkv only in a third kernel for one sequence | 3300.6 | 3885.8 | 5444.2 |
| e0114: that library, one sequence through the BF16 kernel | 3291.8, 3292.3 | | |

Two builds said "-9.5 us at B=1", which I took for a property of the 12-bit qkv and built a per-batch-size choice on.
The third build, with the same kernel code for one sequence and the same allocations, said +11. Across these builds
the tensor-core kernel at B=8 (identical code and data in the last three) moved between 5431 and 5444, and the plain
BF16 kernel at B=3 between 3886 and 3893. **A build-to-build comparison on this engine carries about +-10 us at B=1 and
+-7 us at B=8 that has nothing to do with the change under test** (which kernels share the library, what the loader
allocated before the runtime buffers), although each build repeats to 0.3 us. That is the size of every track-B effect
here. What separates them: a switch inside one library (as in the 12-hour run, G3/G4), or phase stamps.

**Where the build effect comes from (e0117, `tools/x_sass.sh`: `cuobjdump -sass` of the libraries, one kernel
function compared instruction by instruction, addresses and immediates masked).** Each kernel is one function of
about 21,000 (plain) / 22,300 (tensor-core) instructions to the compiler:

| pair of builds | instructions | lines that differ: opcode / registers masked / full |
|---|---|---|
| s0020 source against my source as A (mode word, `qkv_skip`, layer-0 branch; same speed) | 20,992 / 21,064 | 5,568 / 11,494 / 14,926 |
| my source as A against as `B W12U` (only the unembedding decoder differs) | 21,064 / 21,120 | 2,608 / 5,430 / 20,404 |
| my source as A against s0030's source (second decoder in the unembedding; layers 8-13 us slower) | 21,064 / 21,320 | 3,062 / 5,688 / 22,372 |

Changing 56 instructions' worth of decoder in the last phase of the step gives a function in which nearly every
instruction uses other registers and a quarter of the lines differ even with registers masked: the register
assignment and part of the instruction order of all 48 layers are redone. The speed of the layers then differs by
whatever that reassignment happens to cost (spills are equal in all of them: 16 bytes of stores). So the effect is
code generation, not the placement of code or data, and a change is only measured cleanly when the rest of the
kernel's machine code does not change with it: a switch read at run time inside one library, or the changed phase
compiled out of line.

Used in the end (e0119, s0034 / s0035, `scratch/mix/`): entry functions are compiled separately, so each kernel can
keep the form of the unembedding GEMV under which its layers came out fast: out of line per decoder in the plain
kernel (B=1 3304 us, as s0020), both decoders inline in the tensor-core kernel (B=8 5450, as s0020). That gives one
binary for both tracks at track A's speed: s0034 3307.1 / 5394.5 over HTTP against s0029's 3308.4 / 5393.7.

First attempt (e0118, `scratch/ni/`): the unembedding GEMV as an out-of-line function per decoder, chosen at load.
It gives one binary for both tracks and a clean, additive A/B (G12). It also shows the other side of the lottery:
in that library the tensor-core kernel runs B=8 at 5515 us against 5455 in every other build of the same layer
code (+1.1 %), while the plain kernel is unchanged (3304). An out-of-line call in the last phase changed the
register assignment of the 48 layers of that kernel for the worse. A later scratch build (`scratch/piv/`, e0125: ten
more lines in layer 0's branch of the attention-input phase) ran B=8 at 5550 us. So there are about 110 us (2 %) at
B=8 between the best and the worst assignment seen in one evening; the sibling's s0020 sits at the good end, and
nobody chose that.

## G12. Op-level worth of the two track-B items on the sibling's kernel, by phase stamps  [exec e0114, tools/x_stamp.py]

Card 0, block 0, means over 96 greedy steps at context 64-160, us:

| phase | track A, B=1 | track B, B=1 | track A, B=8 | track B, B=8 |
|---|---|---|---|---|
| layer 0, attention input (embedding norm, attention norm) | 6.66 | 4.54 | 6.46 | 5.06 |
| layer 0, main qkv GEMV (with its barrier) | 6.41 | 1.88 | 6.43 | 1.85 |
| unembedding GEMV | 69.67 | 64.78 | 70.96 | 65.26 |
| sum of the three | 82.7 | 71.2 (**-11.5**) | 83.9 | 72.2 (**-11.7**) |

So the tables are worth 6.0-6.6 us per step (the main qkv GEMV of layer 0 is gone except for its barrier, and one of
the two team reductions of the attention input) and the 12-bit unembedding 4.9-5.7 us (7-8 % of its GEMV; on my
kernel 13 %: there it ran 77 -> 67 us). Repeated in e0115 on two more builds: unembedding 69.6 / 70.4 -> 64.7 / 64.8,
layer 0 13.0-13.2 -> 6.4-6.7 us.

Three measurements of the same thing:

| how | B=1 | B=8 |
|---|---|---|
| phase stamps, sum of the changed phases (this table) | -11.5 us | -11.7 us |
| one binary, switch at load (s0030's source, e0107, in process) | -11 us | -14 us |
| one binary, unembedding GEMV out of line (`scratch/ni/`, e0118, in process): 12-bit unembedding / tables / both | -4.6 / -5.7 / -11.2 us | -5.9 / -8.0 / -15.0 us |
| **one binary, by the evaluator over HTTP (s0035 against s0034; `scratch/mix/`)**; in process in that library (e0119) | **-12.3 us**; -11.9 us | **-11.8 us**; -11.6 us |
| two builds of one source, in process (e0109) / over HTTP (s0031 against s0029, s0032) | -15 / -17 us | -17 / -19 us |

What is left of layer 0 was measured too (e0125, `scratch/piv/`, one library, switch at load, logits bit-identical):
the 1/rms of the normed embedding from a per-token table removes layer 0's second team reduction: its attention
input 5.5 -> 4.4 us by stamps, and nothing measurable end to end (3295.2 / 3294.6 -> 3294.3 / 3294.6 us at B=1, 5550 ->
5550 at B=8). Not kept: the two items are what this checkpoint offers on this kernel.

The first four agree, and in the one-binary measurements the two items add up; the last includes whatever else
differs between two builds (G11: up to +-10 us). **What the checkpoint gives on this kernel is 12 us per step: 0.37 %
at B=1 and 0.22 % at B=8 (s0035 against s0034, one binary, the evaluator's HTTP figures).** The submitted pair
shows 0.5 % / 0.35 %, part of which is the build.

## G13. Where the time is in the final engine (track B, s0036), by phase  [exec e0123, tools/x_phase.py]

Card 0, block 0, mean us per layer over 64 greedy steps at context 512-576, steps issued one at a time with a host
synchronisation. That mode has one artefact, measured in e0124 (`tools/x_skew.py`): the four cards' kernels of a
synchronised step start 29-30 us apart (four launches from Python, one after the other), and card 0 waits for the
others in its first cross-card phase, layer 0's post-attention (42 us instead of 10.6). In the device loop
(`generate()`, and the online path with its launches queued ahead) the kernels start within 2.4 us of each other and
layer 0's post-attention is 10.6 us like every other layer's. So read layer 0's post-attention below as ~10.6 us for
the scored paths; a caller that synchronises every step (the evaluator's w1-like loop) pays the 30 us, which one
launch call for all four cards would remove.

| B=1 | attn_in | qkv | attention | o_proj | post_attn | router | routing | w13 | quant2 | w2 | layer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| layer 0 (tables) | 4.6 | 2.0 | 13.5 | 3.5 | 40.9 | 3.0 | 3.7 | 8.7 | 6.2 | 6.3 | 92.6 |
| NVFP4 local (32) | 9.8 | 6.4 | 10.6 | 3.0 | 10.0 | 3.0 | 3.8 | 7.6 | 5.8 | 4.8 | 65.1 |
| NVFP4 global (7) | 15.6 | 6.4 | 11.1 | 3.0 | 10.0 | 3.0 | 3.9 | 7.6 | 6.2 | 4.9 | 72.0 |
| MXFP8 (8) | 11.8 | 6.3 | 10.7 | 3.0 | 9.5 | 3.0 | 3.9 | 9.6 | 2.7 | 5.8 | 66.5 |

Final phases at B=1: mix 9.8, unembedding 64.9 (12-bit; 69.7 in BF16), argmax exchange 3.0.

| B=8 | attn_in | qkv | attention | o_proj | post_attn | router | routing (+ NVFP4 w13 on tcgen05) | w13 | quant2 | w2 | layer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| layer 0 (tables) | 5.4 | 2.2 | 13.5 | 3.9 | 39.8 | 3.0 | 32.8 | 3.2 | 6.4 | 22.0 | 132.4 |
| NVFP4 local | 11.5 | 6.5 | 12.2 | 3.2 | 11.5 | 3.0 | 29.8 | 2.9 | 7.0 | 18.6 | 106.3 |
| NVFP4 global | 17.1 | 6.5 | 12.2 | 3.3 | 11.3 | 3.0 | 30.0 | 3.0 | 7.0 | 18.8 | 112.2 |
| MXFP8 | 13.7 | 6.4 | 12.1 | 3.3 | 10.7 | 3.0 | 4.3 | 49.9 | 5.0 | 25.7 | 134.4 |

Final phases at B=8: mix 11.2, unembedding 65.4, argmax exchange 15.3.

Reading for track B: at B=1 the per-layer dense GEMVs are 6.4 + 3.0 + 3.0 us of 66 and none of them is bound by
bytes (G11); the token phases (attention input, post-attention, routing, quant2: ~30 us per layer) and the attention
(10.7) hold no weights to store differently; the expert GEMVs (12.6 us) read codes that use 3.97 of 4 and 6.53 of 8
bits (G1). The unembedding is 2 % of the step and the only GEMV where bytes matter. That is why this checkpoint is
worth 12 us and not more, on this kernel as on mine. Reading for track A: at B=8 the eight MXFP8 layers cost 134 us
against 106-112 (their w13 on the fragment path 49.9 us against ~26 us for an NVFP4 layer's w13 on tcgen05), and
the global layers' attention input 5-6 us more than the local ones' (three queries instead of one).
