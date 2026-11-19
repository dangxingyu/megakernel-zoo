# s7-inkling2s-b200x4-fb1: Inkling-2-Small on GB200

Score: `http_step_us_b1` (lower is better). Budget 15.0 agent-hours. 36 submissions so far, 35 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 603
- 1.10x the fastest general engine: 4283
- SGLang PR #50186 image lmsysorg/sglang:dev-inkling-2-small (arm64 sha256:198af079c599) (general engine): 4711
- inherited engine: run s6, submission s0003 (track A): 5324

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Inherited engine unchanged (s6 s0003): baseline on this run's node

- status: **incorrect**, no score, at 0.10 agent-hours, category other
- hypothesis: track A: checkpoint formats unchanged. The inherited engine resubmitted unmodified to get this run's own baseline; expect http_step_us_b1 about 5324 and B=8 about 9310 as in run s6 (within 2%).
- evaluator: incorrect: failed http_tokens
- gates that failed: http_tokens

### #2 s0002: Inherited engine + explicit FMAs in expert scaling (slot-order invariant)

- status: **valid**, http_step_us_b1 = 5317 (B=8: 9340), at 0.24 agent-hours, parent s0001, category correctness
- hypothesis: track A: checkpoint formats unchanged. s0001 (the inherited engine unmodified) failed http_tokens: a sequence's logits depended on its batch slot (FMA contraction differing between MMA columns in the w2/MXFP8 scaling, known from run s6). Explicit __fmaf_rn in the two places; t_perm shows logits bitwise equal under a row permutation and the shuffled online run reproduces the greedy tokens 8/8. Expect valid, speed unchanged: about 5317 us at B=1 and 9300 us at B=8.
- evaluator: valid: 5317 us/step at B=1 and 9340 at B=8 over HTTP (online (submit/poll)); in process 5303 us/step at B=1 (generate, MBU 0.114)

### #3 s0003: Kernel of run s6's s0005: tagged-data cross-card exchange, router per strip, attention fixes

- status: **valid**, http_step_us_b1 = 4819 (B=8: 8895), at 0.38 agent-hours, parent s0002, category sync
- hypothesis: track A: checkpoint formats unchanged. The later kernel of run s6 (its s0004+s0005, measured there at 4778 / 8845 us) on top of s0002: every cross-card value travels with its tag in one 8-byte store and is polled by its consumer (no system fence, grid barrier or counter for the three exchanges per layer), router by block-per-strip, attention K/V row on 8 warps, batched mix reductions. Rehearsal here: online 4795 us at B=1 and 8967 us at B=8, tokens identical to step greedy, logits bitwise invariant under a row permutation. Expect http_step_us_b1 about 4800 (-10% vs s0002) and B=8 about 8950 (-4%).
- evaluator: valid: 4819 us/step at B=1 and 8895 at B=8 over HTTP (online (submit/poll)); in process 4806 us/step at B=1 (generate, MBU 0.126)

### #4 s0004: Parallel mailbox polls; B=8 w2 prefetch issued at once; B=1 token phases as own functions

- status: **valid**, http_step_us_b1 = 4794 (B=8: 8486), at 1.15 agent-hours, parent s0003, category scheduling
- hypothesis: track A: checkpoint formats unchanged. Three latency fixes found with phase timers on the score's workload: (1) the serial L2 prefetch of the union experts' w2 slabs cost about 0.4 us per expert on one warp (20 us per layer at ~51 experts at B=8): at B>=2 thread u now issues expert u's prefetch (w13 68.7 -> 51.2 us, w2 27.6 -> 34.1); (2) the 16 cross-card mailbox words of a thread are polled 8 at a time with the history loads issued first; (3) the B=1 kernel's per-token phases are separate functions (spill loads 788 -> 52 bytes in the step function). In-process greedy workload: 4795 / 8517 us against 4818 / 8948 for s0003; expect http_step_us_b1 about 4795 (-0.5%) and B=8 about 8470 (-4.8%).
- evaluator: valid: 4794 us/step at B=1 and 8486 at B=8 over HTTP (online (submit/poll)); in process 4768 us/step at B=1 (generate, MBU 0.127)

### #5 s0005: B>=2: w2 input as compact per-pair fragments staged in shared memory

- status: **valid**, http_step_us_b1 = 4797 (B=8: 8031), at 1.33 agent-hours, parent s0004, category memory
- hypothesis: track A: checkpoint formats unchanged. At B>=2 the w2 GEMV read dense 8-column activation fragments from global memory (1 KB per 64-k unit against 576 B of NVFP4 weights, although an expert has ~1.25 routed tokens of 8). Now one compact one-column fragment per routed (token, expert) pair (64 pairs, 74 KB), quantized by one block per pair and staged in shared memory; lanes of a non-routed column feed zeros. Logits bitwise equal to the dense build. In process at B=8: 8510 -> 8069 us (w2 34.0 -> 27.8 us per layer); expect B=8 about 8040 over HTTP (-5%) and B=1 unchanged (4794).
- evaluator: valid: 4797 us/step at B=1 and 8031 at B=8 over HTTP (online (submit/poll)); in process 4769 us/step at B=1 (generate, MBU 0.127)

### #6 s0006: w13 and unembedding on block-contiguous strips with shared-memory hand-off; 4 FP4 units in flight at B>=2

- status: **valid**, http_step_us_b1 = 4682 (B=8: 7614), at 1.73 agent-hours, parent s0005, category gemm
- hypothesis: track A: checkpoint formats unchanged. The expert GEMV w13 split the flat (strip, unit) space evenly over all warps of the grid; nearly every strip was cut between warps and summed through global slots, a fence and an atomic (hand-off + epilogue 6 us at B=8). Now block b owns whole strips, its 24 warps split the block's units and a cut strip is summed in warp order through shared-memory slots. Same for the unembedding. NVFP4 loads 4 units ahead per warp at B>=2 (2 at B=1). In process: B=1 4796 -> 4684 us, B=8 7975 -> 7653 us; expect http_step_us_b1 about 4690 (-2.2%) and B=8 about 7620 (-5%).
- evaluator: valid: 4682 us/step at B=1 and 7614 at B=8 over HTTP (online (submit/poll)); in process 4664 us/step at B=1 (generate, MBU 0.129)

### #7 s0007: Loader only: runtime buffers allocated before the unembedding; unembedding storage chosen at load

- status: **valid**, http_step_us_b1 = 4688 (B=8: 7606), at 2.43 agent-hours, parent s0006, category cleanup
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A: BF16 unembedding). Preparation for a clean track-B comparison: the kernel now holds both unembedding loops (BF16 and the exact 12-bit code) and picks one from the context at load, and the per-step buffers are allocated before the unembedding is packed, so that the BF16 and 12-bit engines are the same binary with the same buffer placement (with the buffers allocated after the unembedding, the same binary ran the B=1 layer loop 25 us slower when the 12-bit packer had run). No kernel change intended: in process 4691 / 7655 us against 4684 / 7653 for s0006; expect http_step_us_b1 about 4690 and B=8 about 7615.
- evaluator: valid: 4688 us/step at B=1 and 7606 at B=8 over HTTP (online (submit/poll)); in process 4670 us/step at B=1 (generate, MBU 0.129)

### #8 s0008: Track B: unembedding stored in an exact 12-bit code (bit-identical values, 25% fewer bytes)

- status: **valid**, http_step_us_b1 = 4679 (B=8: 7596), at 2.58 agent-hours, parent s0007, category memory
- hypothesis: track B: the unembedding (926,318,592 BF16 values, 1.85 GB) is stored as sign + 7 mantissa bits + a 4-bit exponent offset in a per-row window of 16 exponents; the 55,788 values outside their row's window are escapes patched into the MMA operands; every other tensor in the checkpoint's format. Same binary as s0007 (engine/TRACK selects the storage at load); the roundtrip is verified bit for bit on all values at load (MK_W12_VERIFY) and the logits are bitwise equal to the BF16 engine's. The unembedding GEMV drops from 77 to 67 us per step (in process 4691 -> 4681 at B=1, 7655 -> 7646 at B=8): expect http_step_us_b1 about 10 us below s0007 (-0.2%) and B=8 about 10 us lower (inside its noise). The 12-bit code for the per-layer dense matrices was measured and is a loss (they are not bandwidth-bound), so it is not used.
- evaluator: valid: 4679 us/step at B=1 and 7596 at B=8 over HTTP (online (submit/poll)); in process 4660 us/step at B=1 (generate, MBU 0.130)

### #9 s0009: Prefetch issued by idle warps; one cross-card store per destination per strip; batched re-polls at B>=2

- status: **valid**, http_step_us_b1 = 4568 (B=8: 7263), at 3.57 agent-hours, parent s0008, category sync
- hypothesis: track A: checkpoint formats unchanged (BF16 unembedding). Three exchange/latency fixes from stamps inside the token phases: (1) a bulk L2 prefetch instruction holds its warp about 0.4 us, and 11 of them ran in front of the B=1 attention-input phase: now issued by warps 21-23, which the token phases leave idle; (2) a cross-card store instruction costs per instruction, not per lane: the GEMV epilogues send a strip with one 16-byte store per destination card and column parity (rows g, g+1 from even lanes, g+7, g+8 from odd lanes) instead of two 8-byte ones; (3) at B>=2 all stale mailbox words of a thread are re-read in the same round; poll back-off 16 ns. Logits bitwise equal to s0007's build. In process 4570 / 7308 us against 4691 / 7655; expect http_step_us_b1 about 4565 (-2.6%) and B=8 about 7260 (-4.5%).
- evaluator: valid: 4568 us/step at B=1 and 7263 at B=8 over HTTP (online (submit/poll)); in process 4548 us/step at B=1 (generate, MBU 0.133)

### #10 s0010: NVFP4 input quantization with the hardware E2M1 conversion; w2 inputs one 16-block per thread

- status: **valid**, http_step_us_b1 = 4246 (B=8: 7117), at 3.81 agent-hours, parent s0009, category other
- hypothesis: track A: checkpoint formats unchanged (BF16 unembedding); the quantization of every expert GEMV's input is computed as defined, with other instructions. The token phases are bound by instruction count on one SM, and the E2M1 rounding was a 7-way compare chain evaluated three times per value (two candidate block scales, then the chosen one again). cvt.rn.satfinite.e2m1x2.f32 rounds exactly like that chain, signed zeros included (tools/ptx/t5.cu over every tie; only a NaN input differs): two values per conversion, the code times the block scale formed exactly in f16, both candidates kept so that nothing is recomputed; the w2 inputs take a whole block of 16 per thread (no shuffles; the error sum keeps the shuffle tree's order). Logits bitwise equal to the previous build. In process: B=1 4570 -> 4247 us (quant2 7.7 -> 2.9 us per layer, w13-input quantization 3.6 -> 1.4), B=8 7309 -> 7164; expect http_step_us_b1 about 4245 (-7%), which is below the 1.10x line of 4283, and B=8 about 7120 (-2%).
- evaluator: valid: 4246 us/step at B=1 and 7117 at B=8 over HTTP (online (submit/poll)); in process 4229 us/step at B=1 (generate, MBU 0.143)

### #11 s0011: B>=2: routing per token block, conv work on idle blocks; top-8 with a shorter dependency chain

- status: **valid**, http_step_us_b1 = 4239 (B=8: 7035), at 4.76 agent-hours, parent s0010, category scheduling
- hypothesis: track A: checkpoint formats unchanged (BF16 unembedding). Three more instruction-count fixes in the per-token phases, logits bitwise equal to s0010's build: (1) at B>=2 token block t alone computes token t's top-8 and publishes ids and weights with tags (before: every block computed all B rows, 8 warps sharing each SM); (2) at B>=2 the all-reduce of the cards' partials and the short conv (element-wise, ~230 instructions per thread) run on 4 otherwise idle blocks per token, which publish the exact BF16 result with tags; the token block reads 4 words per thread; (3) top-8: scores in registers, lane-local best by a tree, value-only shuffles and a ballot. In process: B=8 7165 -> 7083 us, B=1 4247 -> 4241; expect http_step_us_b1 about 4240 and B=8 about 7040 (-1.1%).
- evaluator: valid: 4239 us/step at B=1 and 7035 at B=8 over HTTP (online (submit/poll)); in process 4219 us/step at B=1 (generate, MBU 0.143)

### #12 s0012: Kernel prepared for load-time token tables (switch off): reference for the track-B submissions

- status: **valid**, http_step_us_b1 = 4235 (B=8: 7044), at 5.06 agent-hours, parent s0011, category cleanup
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). Same kernel as s0011 plus the hooks the layer-0 token tables need (a mode word in the step header, read only in layer 0's attention-input phase); with the switch off nothing else differs. This is the binary the next track-B submissions use, so that their difference to this one is the tables / the 12-bit unembedding alone. In process 4236 / 7095 us against 4241 / 7083 for s0011's build: expect http_step_us_b1 about 4238 and B=8 about 7045 (both within noise of s0011).
- evaluator: valid: 4235 us/step at B=1 and 7044 at B=8 over HTTP (online (submit/poll)); in process 4216 us/step at B=1 (generate, MBU 0.143)

### #13 s0013: Track B: load-time token tables for layer 0 (normed embedding, layer-0 projections)

- status: **valid**, http_step_us_b1 = 4222 (B=8: 7037), at 5.14 agent-hours, parent s0012, category other
- hypothesis: track B: every weight in the checkpoint's format (BF16 unembedding); in addition two tables computed at load from the checkpoint by the engine's own kernel: per token id the RMS-normed embedding (4608 BF16) and each card's layer-0 q / gate / r / sink / k / v projection rows (3840 BF16), which depend on the token id only (the first attention-residual block is the embedding; layer 0 has no earlier block to mix). A step reads two table rows instead of two RMS norms and layer 0's 35 MB projection GEMV. Same binary as s0012 (engine/TRACK selects); logits bitwise equal to it (B=1 and B=8). In process 4236 -> 4222 us at B=1 and 7095 -> 7088 at B=8: expect http_step_us_b1 about 14 us below s0012 (-0.3%), B=8 about 7 us lower (inside its noise).
- evaluator: valid: 4222 us/step at B=1 and 7037 at B=8 over HTTP (online (submit/poll)); in process 4202 us/step at B=1 (generate, MBU 0.144)

### #14 s0014: Track B: 12-bit unembedding and layer-0 token tables together

- status: **valid**, http_step_us_b1 = 4195 (B=8: 7004), at 5.56 agent-hours, parent s0013, category other
- hypothesis: track B: the unembedding in the exact 12-bit code (s0008: bit-identical values, 25% fewer bytes) and the layer-0 token tables (s0013) together, every other tensor in the checkpoint's format. Same binary as s0012 (track A, 4235 / 7044) and s0013 (tables alone, 4221.5 / 7037); logits bitwise equal to both. The two changes touch different ops (the unembedding GEMV; layer 0's attention input and projections), so their gains should add: in process 4236 -> 4212 us at B=1 and 7095 -> 7076 at B=8; expect http_step_us_b1 about 4211 (-0.55% against s0012) and B=8 about 7025 (-0.27%, inside its noise).
- evaluator: valid: 4195 us/step at B=1 and 7004 at B=8 over HTTP (online (submit/poll)); in process 4191 us/step at B=1 (step, MBU 0.144)

### #15 s0015: B>=2: MXFP8 expert GEMVs load the next unit before computing the current one

- status: **valid**, http_step_us_b1 = 4215 (B=8: 6899), at 5.89 agent-hours, parent s0014, category gemm
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A, BF16 unembedding, no tables). The MXFP8 expert stream (layers 40-47) had one 64-k unit (1 KB) in flight per warp and is bound by loads in flight (more warps per block made only this stream faster): the B>=2 kernel now issues the next unit's three loads before the current unit's two MMAs. Logits bitwise equal; the B=1 kernel's code is unchanged. Measured on the replacement pod (about 0.45% faster than the one s0012 ran on): in process B=8 7060 -> 6938 us (-1.7%), B=1 4216 unchanged. Expect http_step_us_b1 about 4215 on this pod (this is also the track-A reference for s0014 on the same pod) and B=8 about 6900.
- evaluator: valid: 4215 us/step at B=1 and 6899 at B=8 over HTTP (online (submit/poll)); in process 4205 us/step at B=1 (generate, MBU 0.144)

### #16 s0016: Track B on the s0015 kernel: 12-bit unembedding and layer-0 token tables

- status: **valid**, http_step_us_b1 = 4188 (B=8: 6884), at 6.13 agent-hours, parent s0015, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables (as s0014), every other tensor in the checkpoint's format; same binary as s0015 (track A, 4214.8 / 6899 on this pod), engine/TRACK selects. Logits bitwise equal to the track-A engine. Expect the same differences as measured before on one binary and one pod (s0014 against s0015 at B=1: -19 us): http_step_us_b1 about 4195 (-0.47%) and B=8 about 6880 (-0.27%, inside its noise).
- evaluator: valid: 4188 us/step at B=1 and 6884 at B=8 over HTTP (online (submit/poll)); in process 4182 us/step at B=1 (generate, MBU 0.144)

### #17 s0017: B=1: attention scores on the tensor cores; value sum with 8 dimensions per thread

- status: **valid**, http_step_us_b1 = 4174 (B=8: 6901), at 6.62 agent-hours, parent s0016, category attention
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A); the attention keeps its formats (q, K, V in BF16, scores, softmax and the value sum in FP32), the scores are accumulated by m16n8k16 BF16 MMAs (A = the 6 local q heads, B = 8 key rows per warp, 8 k-steps; one more MMA for the relative bias, rounded to BF16 as before) instead of 128 scalar FMAs per (head, position) thread, and the value sum takes 8 dimensions per thread with one 16-byte load per position. Only in the B=1 kernel (in the B>=2 kernel the same code costs 3%: its step function's frame grows). The summation order changes, so the B=1 logits differ from s0015's in rounding (layer-0 attention outputs: rel L2 1e-5 median; logits after 48 layers: 0.07 median, as between any two valid paths of this model); B=8 is bitwise unchanged. In process B=1 4214 -> 4177 us; expect http_step_us_b1 about 4178 (-0.9% against s0015) and B=8 about 6900, gates as before.
- evaluator: valid: 4174 us/step at B=1 and 6901 at B=8 over HTTP (online (submit/poll)); in process 4166 us/step at B=1 (generate, MBU 0.145)

### #18 s0018: Track B on the s0017 kernel: 12-bit unembedding and layer-0 token tables

- status: **valid**, http_step_us_b1 = 4151 (B=8: 6882), at 6.76 agent-hours, parent s0017, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the kernel of s0017 (track A, 4174.3 / 6901 on this pod), engine/TRACK selects the storage at load. Logits bitwise equal to the track-A engine of the same build. Expect the differences measured twice before (s0016 against s0015: -26 us at B=1, -15 us at B=8): http_step_us_b1 about 4148 (-0.6%) and B=8 about 6886.
- evaluator: valid: 4151 us/step at B=1 and 6882 at B=8 over HTTP (online (submit/poll)); in process 4142 us/step at B=1 (generate, MBU 0.146)

### #19 s0019: B>=2: NVFP4 units loaded per group: 6 for w13, 5 for w2 (was 4)

- status: **valid**, http_step_us_b1 = 4176 (B=8: 6803), at 7.30 agent-hours, parent s0018, category gemm
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). The NVFP4 expert streams of the B>=2 kernel load a group of units, then compute it; a sweep of the group size on the score's workload gives 6 units for w13 and 5 for w2 (in process at B=8: both 4: 6938 us; 5/5: 6857; 6/5: 6841; larger or smaller is worse; the load-ahead form used for MXFP8 makes this loop spill and is 15% slower). Logits bitwise equal; the B=1 kernel is unchanged. Expect B=8 about 6805 over HTTP (-1.4% against s0017's 6901) and http_step_us_b1 about 4174 (unchanged).
- evaluator: valid: 4176 us/step at B=1 and 6803 at B=8 over HTTP (online (submit/poll)); in process 4166 us/step at B=1 (generate, MBU 0.145)

### #20 s0020: Prefetch policy re-measured per kernel; reader-side fences removed; tensor-core scores at B>=2

- status: **valid**, http_step_us_b1 = 4074 (B=8: 6540), at 8.05 agent-hours, parent s0019, category scheduling
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). A sweep of the kernel's scheduling choices on the score's workload after the earlier changes moved the bottlenecks; each kept only if faster in process: (1) K/V row prefetch: at B>=2 one thread per (token, head) range instead of 32 serial bulk prefetches on one warp of every non-token block in front of the grid barrier (6639 -> 6473 us), none at B=1 (4106 -> 4072); (2) w2 prefetch: none at B>=2 (6840 -> 6757), all at once at B=1 (4142 -> 4099); (3) three fences on the reader side of loads that bypass L1 anyway (quant2 in both kernels, attention combine at B>=2) removed (-19 / -38 us); (4) attention scores by tensor-core MMAs also in the B>=2 kernel (inline, scalar value sum: -0.8%; layer-0 attention outputs agree with the scalar path to 5e-6 rel L2); (5) 8 conv helper blocks per token (-0.5%); NVFP4 group of 3 units at B=1 (-0.4%). In process: 4176 -> 4074 us at B=1 and 6840 -> 6473 at B=8; expect http_step_us_b1 about 4075 (-2.4%) and B=8 about 6440 (-5.3%), which would put B=8 below SGLang's 6604 (parity at both batch sizes).
- evaluator: valid: 4074 us/step at B=1 and 6540 at B=8 over HTTP (online (submit/poll)); in process 4068 us/step at B=1 (generate, MBU 0.148)

### #21 s0021: Track B on the s0020 kernel: 12-bit unembedding and layer-0 token tables

- status: **valid**, http_step_us_b1 = 4051 (B=8: 6513), at 8.41 agent-hours, parent s0020, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the source of s0020 (track A, 4073.7 / 6540 on this pod) with engine/TRACK = B W12U L0T. Logits bitwise equal to the track-A engine of the same build. Expect the differences measured three times before (-19 to -26 us at B=1, -15 to -19 us at B=8): http_step_us_b1 about 4050 (-0.6%) and B=8 about 6522; parity at both batch sizes kept.
- evaluator: valid: 4051 us/step at B=1 and 6513 at B=8 over HTTP (online (submit/poll)); in process 4045 us/step at B=1 (generate, MBU 0.149)

### #22 s0022: Block-per-strip GEMVs run a block's two strips at once on the two halves of its warps

- status: **valid**, http_step_us_b1 = 4014 (B=8: 6425), at 8.98 agent-hours, parent s0021, category gemm
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). In qkv, o_proj, router and w2 a block owns strips b and b + 152 and ran them one after the other on all 24 warps, each with its own synchronisation, reduction and epilogue round; now the two strips run concurrently on 12 warps each (one round). The pieces of a strip are other unit ranges, so results differ from s0020 in rounding (deterministic, slot-invariant; layer-0 attention outputs equal to rounding). Measured on a fixed token stream (teacher-forced, tools/t_replay.py), because builds with different rounding take different greedy paths and the number of distinct experts per layer (44-50 at B=8), hence the B=8 step time, depends on the text: B=1 4089 -> 4024 us (-1.6%), B=8 6497 -> 6432 (-1.0%). Expect http_step_us_b1 about 4015 (-1.4% against s0020); B=8 about 6475 if the greedy text keeps about the same expert unions (it may move by a few percent either way).
- evaluator: valid: 4014 us/step at B=1 and 6425 at B=8 over HTTP (online (submit/poll)); in process 4005 us/step at B=1 (generate, MBU 0.151)

### #23 s0023: Track B on the s0022 kernel: 12-bit unembedding and layer-0 token tables

- status: **valid**, http_step_us_b1 = 3991 (B=8: 6406), at 9.06 agent-hours, parent s0022, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the source of s0022 (track A, 4013.9 / 6425 on this pod) with engine/TRACK = B W12U L0T. Logits bitwise equal to the track-A engine of the same build, so the greedy text and the expert unions are the same as s0022's. Expect the differences measured four times before (-19 to -26 us at B=1, -15 to -27 us at B=8): http_step_us_b1 about 3991 (-0.57%) and B=8 about 6405.
- evaluator: valid: 3991 us/step at B=1 and 6406 at B=8 over HTTP (online (submit/poll)); in process 3981 us/step at B=1 (generate, MBU 0.152)

### #24 s0024: B=1: attention items of 64 positions (twice as many blocks busy, each with half the work)

- status: **valid**, http_step_us_b1 = 3997 (B=8: 6459), at 11.08 agent-hours, parent s0023, category attention
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). At B=1 an attention item (one KV head, 128 positions) ran on one block while most blocks idled, and its time is instruction count on that SM: now 64 positions per item, the chunk size a template parameter of the kernel; a second instantiation with 128-position items serves single sequences beyond 2048 positions (the combine has one lane per chunk; tested across the switch to 2170). The chunk partials are other ranges, so B=1 results differ from s0022 in rounding (layer-0 attention outputs equal to rounding, 88% of steps bitwise); the B>=2 kernel is unchanged (bitwise). In process B=1 4015 -> 3994 us (attention 9.2 -> 8.7 us per layer, o_proj +0.7 for the larger combine), B=8 6023 unchanged: expect http_step_us_b1 about 3995 (-0.5% against s0022) and B=8 about 6425.
- evaluator: valid: 3997 us/step at B=1 and 6459 at B=8 over HTTP (online (submit/poll)); in process 3957 us/step at B=1 (generate, MBU 0.153)

### #25 s0025: Track B on the s0024 kernel: 12-bit unembedding and layer-0 token tables

- status: **valid**, http_step_us_b1 = 3975 (B=8: 6436), at 11.16 agent-hours, parent s0024, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the source of s0024 (track A, 3996.5 / 6459 on this pod, epoch 4) with engine/TRACK = B W12U L0T. Logits bitwise equal to the track-A engine of the same build. Expect the differences measured five times before on one pod (-19 to -26 us at B=1, -15 to -27 us at B=8): http_step_us_b1 about 3974 (-0.57%) and B=8 about 6440.
- evaluator: valid: 3975 us/step at B=1 and 6436 at B=8 over HTTP (online (submit/poll)); in process 3932 us/step at B=1 (generate, MBU 0.153)

### #26 s0026: Track B, final: s0025 with the MK_NO_RED debugging switch made harmless (host side only)

- status: **valid**, http_step_us_b1 = 3975 (B=8: 6436), at 11.31 agent-hours, parent s0025, category cleanup
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the kernel source of s0024 / s0025 unchanged (same binary). The only difference to s0025 is in api.py: the inherited debugging switch MK_NO_RED, which would launch the B>=2 kernel for a single sequence and hang (that kernel builds its pair map only for B > 1), is now ignored with a message. Not on the evaluated path: expect the numbers of s0025 within the evaluator's repeatability on this pod, http_step_us_b1 about 3975 and B=8 about 6436.
- evaluator: valid: 3975 us/step at B=1 and 6436 at B=8 over HTTP (online (submit/poll)); in process 3932 us/step at B=1 (generate, MBU 0.153)

### #27 s0027: B>=2 kernel builds its pair map for every batch size (a single sequence no longer hangs it)

- status: **valid**, http_step_us_b1 = 3997 (B=8: 6455), at 11.53 agent-hours, parent s0026, category correctness
- hypothesis: track A: checkpoint formats unchanged (engine/TRACK = A). Correctness fix off the evaluated path: since s0005 the B>=2 kernel built the (token, expert) pair map of its compact w2 fragments only for B > 1 and waited forever when given a single sequence (debugging switch MK_NO_RED); the map is now built whenever that kernel runs, and the switch works again (rehearsal: B=1 through the B>=2 kernel, 4390 us per token, tokens equal to its step-by-step greedy). Logits on the normal paths bitwise equal to s0024 / s0026; in process 3997 / 6053 us against 4001 / 6053. Expect the numbers of s0024 on this pod: http_step_us_b1 about 3997 and B=8 about 6459.
- evaluator: valid: 3997 us/step at B=1 and 6455 at B=8 over HTTP (online (submit/poll)); in process 3957 us/step at B=1 (generate, MBU 0.153)

### #28 s0028: Track B, final: 12-bit unembedding and layer-0 token tables on the s0027 kernel

- status: **valid**, http_step_us_b1 = 3974 (B=8: 6433), at 11.62 agent-hours, parent s0027, category other
- hypothesis: track B: the unembedding in the exact 12-bit code and the layer-0 token tables, every other tensor in the checkpoint's format; the source of s0027 (track A, 3996.7 / 6455 on this pod) with engine/TRACK = B W12U L0T. Logits bitwise equal to the track-A engine of the same build. Expect the difference measured five times before (-22 us at B=1, about -20 us at B=8): http_step_us_b1 about 3975 and B=8 about 6435.
- evaluator: valid: 3974 us/step at B=1 and 6433 at B=8 over HTTP (online (submit/poll)); in process 3932 us/step at B=1 (generate, MBU 0.153)

### #29 s0029: Sibling run fa1's final kernel (its s0020 source, unchanged) as this run's track-A baseline

- status: **valid**, http_step_us_b1 = 3308 (B=8: 5394), at 11.84 agent-hours, parent s0028, category other
- hypothesis: track A: checkpoint formats unchanged. This is the final source of run s7-inkling2s-b200x4-fa1 (its s0020: plain kernel at batch 1-3, from batch 4 a second kernel that runs the NVFP4 w13 GEMV through tcgen05 block-scaled MMAs), taken byte for byte (diff -r against the unpacked s0020.tgz is empty), TRACK = A. Submitted as the operator asked so that this run's evaluation has it as the baseline for the track-B changes that follow. Expected: what it measured in its own run, about 3300 us/step at B=1 and 5390 at B=8 over HTTP (my previous kernel: 3997 / 6455).
- evaluator: valid: 3308 us/step at B=1 and 5394 at B=8 over HTTP (online (submit/poll)); in process 3252 us/step at B=1 (generate, MBU 0.185)

### #30 s0030: Track B on the sibling's final kernel: 12-bit unembedding + layer-0 token tables; context limit enforced

- status: **valid**, http_step_us_b1 = 3301 (B=8: 5376), at 12.46 agent-hours, parent s0029, category other
- hypothesis: track B: on the kernel of s0029 (run fa1's s0020 source), two lossless changes that depend on this checkpoint, both made at load time (engine/TRACK = 'B W12U L0T'; 'A' gives the track-A engine from the same binary). Formats that differ from the checkpoint: (1) the unembedding (201024 x 4608 BF16) is stored in an exact 12-bit code (per-row exponent window, sign + 7 mantissa bits, 4-bit exponent offset, 55,788 escapes) and decoded to the same BF16 bits in the GEMV; round trip asserted on all 926,318,592 values. (2) What layer 0 computes from the token id alone is tabulated per id by the engine's own kernel at load (5 s): the RMS-normed embedding and each card's layer-0 q, r, sink, k, v projection rows; a step copies two rows and layer 0's main qkv GEMV has no strips left. The output-gate rows of layer 0 are NOT tabulated, unlike in s0028: in this kernel they are computed beside the attention with a split that depends on batch and position, so a table could not be bit-identical. Verified before submitting: logits bit-identical to the unchanged s0020 source and to this source at TRACK=A on 2,336 steps (512 at B=1, 1152 at B=8 to position 1151, 256 at B=2, 256 at B=4, 160 at B=5), generate() and online tokens equal; table rows equal for all 201,024 ids on all four cards in 8 configurations (B=1,3 plain kernel, B=4,5,8 tensor-core kernel, positions 0 to 2047): 0 of 5.56e9 values differ. Also in this source, as the operator asked after the audit: a request or a step beyond the context limit (min(max_context, 2176 positions), where the kernels would overwrite other sequences' K/V) now raises an error in step(), generate() and submit() before anything is launched (tools/x_guard.py). Prediction from two in-process rounds on this pod (generate, 1024 tokens): against the s0020 source -3 us at B=1 (3304 -> 3301; the binary with both storages is 8 us slower at TRACK=A, the two changes give -11) and -18 us at B=8 (5452 -> 5434); over HTTP about 3305 / 5376 against s0029's 3308 / 5394. The checkpoint is worth 0.1-0.3 % on this kernel.
- evaluator: valid: 3301 us/step at B=1 and 5376 at B=8 over HTTP (online (submit/poll)); in process 3254 us/step at B=1 (generate, MBU 0.185)

### #31 s0031: Track B: unembedding decoder chosen at build time (s0030's binary carried both)

- status: **valid**, http_step_us_b1 = 3291 (B=8: 5376), at 12.81 agent-hours, parent s0030, category other
- hypothesis: track B: same two lossless, checkpoint-specific changes as s0030 on the kernel of s0029 (formats that differ from the checkpoint: the unembedding in the exact 12-bit code, round trip asserted on all 926,318,592 values; layer 0's normed embedding and q/r/sink/k/v projection rows tabulated per token id by the engine's own kernel at load), and the same context guard. One change against s0030: its binary compiled both unembedding decoders into one GEMV and chose at load; taking one change back at a time (exec e0108) showed that this alone made the BF16 path 8 us per step slower at B=1 than the s0020 source, so s0030's gain was measured against a slowed twin and partly eaten. Now the track word W12U sets -DW12U in build.sh and only one decoder is compiled (the loader checks that library and TRACK agree); TRACK=A builds a kernel as fast as s0020's (3304.0 / 3303.8 against 3304.6 / 3304.0 us in process). Verified again before submitting (exec e0109): logits bit-identical between the unchanged s0020 source, this source built as A and built as B W12U L0T on 2,336 steps (B=1 512, B=8 1152, B=2 256, B=4 256, B=5 160), generate() and online tokens equal; table rows for all 201,024 ids equal in 7 further configurations (0 of 5.56e9 values differ) and the tables' sha1 equal between the A and B binaries. Prediction from two in-process rounds on this pod (generate 1024 tokens; s0020 source 3304 / 5454): W12U alone -6 / -10 us, L0T alone -5 / -4, both -15 us at B=1 (3289) and -17 at B=8 (5437); over HTTP about 3293 / 5376 against s0029's 3308 / 5394 (s0030: 3301 / 5376).
- evaluator: valid: 3291 us/step at B=1 and 5376 at B=8 over HTTP (online (submit/poll)); in process 3243 us/step at B=1 (generate, MBU 0.186)

### #32 s0032: Track-A twin of s0031: the s0020 kernel with the context guard (same source, TRACK = A)

- status: **valid**, http_step_us_b1 = 3308 (B=8: 5395), at 13.02 agent-hours, parent s0029, category other
- hypothesis: track A: checkpoint formats unchanged. This is the source of s0031 with engine/TRACK = A: the kernel of s0029 (run fa1's s0020) plus what the track-B source added outside the two checkpoint-specific changes, i.e. the context guard in api.py (a request or step beyond min(max_context, 2176) positions raises an error before anything is launched; one integer comparison per launch) and the dormant layer-0 mode code in the kernel. Built as A it compiles only the BF16 unembedding decoder and loads no table. Logits bit-identical to the unchanged s0020 source on 2,336 steps at B=1, 2, 4, 5, 8 (exec e0109). Submitted so that the evaluation has both tracks from one source and shows that the guard is free. Prediction: equal to s0029 within the evaluator's repeatability (in process on this pod 3304.0 / 3303.8 us at B=1 against 3304.6 / 3304.0 for the s0020 source; B=8 5460 / 5456 against 5454 / 5455): about 3308 / 5396 over HTTP.
- evaluator: valid: 3308 us/step at B=1 and 5395 at B=8 over HTTP (online (submit/poll)); in process 3258 us/step at B=1 (generate, MBU 0.185)

### #33 s0033: Track B: s0031 plus a token-id range check in submit() (same kernel source)

- status: **valid**, http_step_us_b1 = 3290 (B=8: 5378), at 13.73 agent-hours, parent s0031, category other
- hypothesis: track B: the engine of s0031 unchanged in kernel, formats and tables (formats that differ from the checkpoint: the unembedding in the exact 12-bit code; layer 0's normed embedding and q/r/sink/k/v projection rows tabulated per token id at load; both verified bit for bit as stated for s0031, ink.cu is byte-identical to s0031's). One host-side addition in api.py, next to the context guard the audit asked for: submit() refuses a request that holds a token id outside [0, 201024) with a ValueError; such an id would index past the embedding and past the layer-0 tables (the audit lists it under input validation). Cost: min/max of the prompt list once per request. Tested with tools/x_guard.py (ids 201024 and -1 refused, 201023 served, all context-limit cases as before) and the rehearsal (online = greedy 100 % at B=1 and B=8). Prediction: equal to s0031 within the evaluator's repeatability on this pod (0.5 us at B=1, ~1-2 us at B=8): 3291 / 5376 us over HTTP; this is also the repeat measurement of the track-B build.
- evaluator: valid: 3290 us/step at B=1 and 5378 at B=8 over HTTP (online (submit/poll)); in process 3241 us/step at B=1 (generate, MBU 0.186)

### #34 s0034: Track A from a binary that serves both tracks (unembedding decoder chosen at load, compiled per kernel)

- status: **valid**, http_step_us_b1 = 3307 (B=8: 5394), at 14.16 agent-hours, parent s0032, category other
- hypothesis: track A: checkpoint formats unchanged. Same kernel as s0029/s0032 in the layers; the difference to s0032 is how the unembedding GEMV is compiled, so that one library serves both tracks again (build.sh no longer reads the track word W12U): in the plain kernel (batch 1-3) the GEMV is an out-of-line function per decoder, in the tensor-core kernel (batch >= 4) both decoders are inline and the loader's pointer chooses. Each form was measured per kernel: inline in the plain kernel the build ran 8 us slower at B=1 (s0030), out of line in the tensor-core kernel 60 us slower at B=8 (exec e0118); the cause is the compiler re-assigning the registers of the whole 21,000-instruction step function when this phase changes (disassembly, e0117), not the decoder. As A the loader fills BF16 fragments and builds no table. Logits bit-identical to the unchanged s0020 source on 2,496 steps (B=1 512, B=8 1152, B=2 256, B=3 160, B=4 256, B=5 160; exec e0119). Submitted first, on its own, as the reference of the track-B submission that follows from the same binary. Prediction (in process on this pod 3303.9 / 3304.8 us at B=1 and 5448.5 / 5452.3 at B=8; s0020 source 3304 / 5455): over HTTP equal to s0029 and s0032 within 2 us: about 3308 / 5392.
- evaluator: valid: 3307 us/step at B=1 and 5394 at B=8 over HTTP (online (submit/poll)); in process 3262 us/step at B=1 (generate, MBU 0.185)

### #35 s0035: Track B from the same binary as s0034: 12-bit unembedding + layer-0 token tables

- status: **valid**, http_step_us_b1 = 3295 (B=8: 5383), at 14.24 agent-hours, parent s0034, category other
- hypothesis: track B: the source and the kernel binary of s0034 (track A), with engine/TRACK = 'B W12U L0T' as the only difference: both checkpoint-specific changes are made by the loader. Formats that differ from the checkpoint: the unembedding (201024 x 4608 BF16) in the exact 12-bit code (round trip asserted on all 926,318,592 values); layer 0's RMS-normed embedding and each card's layer-0 q, r, sink, k, v projection rows tabulated per token id by the engine's own kernel at load (the output-gate rows are not tabulated: this kernel sums them in an order that depends on batch and position). Context limit and token-id checks as in s0033. Verified before submitting (exec e0119): logits bit-identical to the unchanged s0020 source and to this binary as track A on 2,496 steps (B=1 512, B=8 1152, B=2 256, B=3 160, B=4 256, B=5 160); table rows for all 201,024 ids equal in 4 further configurations and their sha1 equal to that of every earlier build. This is the measurement the run asks for, the change against the same kernel without it, now in the evaluator: prediction from two in-process rounds of this library, -11.9 us at B=1 (3304.4 -> 3292.4) and -11.6 us at B=8 (5450.4 -> 5438.8), which is also what the phase stamps give (unembedding GEMV -4.9 / -5.7 us, layer 0 -6.6 / -6.0 us); over HTTP about 12 us below s0034 at both batch sizes, about 3296 / 5380. s0033 (two builds of one source) showed 17 / 19 us against its track-A twin; the 5-7 us beyond this were the build.
- evaluator: valid: 3295 us/step at B=1 and 5383 at B=8 over HTTP (online (submit/poll)); in process 3250 us/step at B=1 (generate, MBU 0.186)

### #36 s0036: Track B: s0033 with the 12-bit round trip asserted at every load (same kernel source)

- status: **valid**, http_step_us_b1 = 3291 (B=8: 5378), at 14.36 agent-hours, parent s0033, category other
- hypothesis: track B: the engine of s0033 unchanged in kernel, formats and tables (formats that differ from the checkpoint: the unembedding in the exact 12-bit code; layer 0's normed embedding and q/r/sink/k/v projection rows tabulated per token id at load; ink.cu and build.sh byte-identical to s0033's). One loader change: the assertion that unpacking the 12-bit unembedding gives back the checkpoint's BF16 bits for all 926,318,592 values, so far run only with MK_W12_VERIFY=1 (in my verification runs and in the audit), now runs at every load, so the scored configuration itself carries the check (load time unchanged at 21 s in the rehearsal; MK_W12_NOVERIFY skips it). Nothing in the timed path changes. Rehearsed (online = greedy 100 % at B=1 and B=8). Prediction: equal to s0033 and s0031 within the evaluator's repeatability for one build (0.9 us at B=1, 1.8 us at B=8 between those two): about 3290 / 5377 us over HTTP.
- evaluator: valid: 3291 us/step at B=1 and 5378 at B=8 over HTTP (online (submit/poll)); in process 3242 us/step at B=1 (generate, MBU 0.186)

