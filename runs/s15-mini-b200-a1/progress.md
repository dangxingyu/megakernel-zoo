# s15-mini-b200-a1: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 23.0 agent-hours. 32 submissions so far, 32 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3355
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (PR #69028, built from source) (general engine): 3692

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: v1 persistent megakernel (152x256, TMA weight rings, exact W4A16 / W8A16 experts)

- status: **valid**, http_step_us_b1 = 5047 (B=8: 1.442e+04), at 0.54 agent-hours, category structure
- hypothesis: track A: checkpoint formats unchanged (NVFP4/MXFP8 experts decoded exactly to BF16 in registers, inputs not quantized; BF16 elsewhere), only the layout differs. First own engine: one cooperative kernel per step, ~7 grid barriers per layer application (+4 in loop passes 2-3), host-driven step() path. Own test vs fp32 reference: median rel. L2 0.009-0.011, top-1 100% (40 positions, B=1 and 8). Expect all gates to pass and in-process ~4.8 ms (B=1) / ~10.6 ms (B=8); over HTTP via step()+argmax about 4.9 / 10.8 ms, i.e. slower than vLLM; this is the correctness baseline.
- evaluator: valid: 5047 us/step at B=1 and 14420 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4873 us/step at B=1 (step, MBU 0.142)

### #2 s0002: device greedy loop behind submit/poll; faster GEMV decode, attention, mixes, top-8

- status: **valid**, http_step_us_b1 = 4315 (B=8: 1.332e+04), at 1.85 agent-hours, parent s0001, category launch
- hypothesis: track A: checkpoint formats unchanged (NVFP4/MXFP8 experts decoded exactly to BF16, inputs not quantized; BF16 elsewhere). Since s0001: NVFP4 scale applied in fp32 after the MMA, bulk-copied staging, K/V chunks by bulk copy, one-round mixing, rank top-8, L2 prefetch of next weights/small params, and the server now drives a device-side greedy loop (submit/poll, tokens through host-mapped memory) instead of step()+argmax. In process 4.24 ms/step at B=1 (serve 4249 us at ctx 16-528). Expect HTTP B=1 about 4.35 ms (ctx to 1088 adds attention) vs 5047, B=8 about 14 ms; gates unchanged within noise (serve tokens equal step-greedy bitwise in own test).
- evaluator: valid: 4315 us/step at B=1 and 13320 at B=8 over HTTP (online (submit/poll)); in process 4231 us/step at B=1 (generate, MBU 0.163)

### #3 s0003: tensor-core attention on fragment-ordered K/V; mixer blocks for B>=2; qkv early prefetch

- status: **valid**, http_step_us_b1 = 4280 (B=8: 1.157e+04), at 2.61 agent-hours, parent s0002, category attention
- hypothesis: track A: checkpoint formats unchanged (NVFP4/MXFP8 experts decoded exactly to BF16, inputs not quantized; BF16 elsewhere; the K/V cache is stored BF16 in mma fragment order). Since s0002: attention scores and P.V on mma.sync from 16-key fragment-ordered K/V tiles (P as bf16 hi+lo), 128-key chunks, last-arriving chunk merges; for B>=2 the mixes are computed once per sequence by a mixer block and published (others bulk-copy the rows) instead of per-warp in every block; qkv ring prefetched before its barrier; MXFP8 accumulators with a static index. In process -45 us (B=1) and -1550 us (B=8) at ctx 300. Expect HTTP B=1 ~4210 (s0002 4315), B=8 ~11800 (s0002 13320); gates unchanged within noise (README text 320 positions: median vs fp32 ref 0.0388 vs s0002 0.0377).
- evaluator: valid: 4280 us/step at B=1 and 11567 at B=8 over HTTP (online (submit/poll)); in process 4164 us/step at B=1 (generate, MBU 0.166)

### #4 s0004: B>=2 gates via mixer blocks; no large L2 weight prefetches; parallel w2 rounds

- status: **valid**, http_step_us_b1 = 4103 (B=8: 9491), at 2.93 agent-hours, parent s0003, category memory
- hypothesis: track A: checkpoint formats unchanged (NVFP4/MXFP8 experts decoded exactly to BF16, inputs not quantized; BF16 elsewhere). Since s0003: (1) the L2 prefetch of the next qkv / o_proj / router weights (54 MB per application, issued during attention) is removed: the extra HBM traffic slowed the latency-bound attention, o_proj and mix phases (same-job A/B: -125 us B=1, -344 us B=8); small per-layer parameters are still prefetched; (2) at B>=2 the loop gates take their mixes from mixer blocks (-1250 us in process at B=8); (3) the w2 rounds are planned in parallel. Expect HTTP B=1 ~4170 (s0003 4280) and B=8 ~9900 (s0003 11567); gates unchanged (logits of the B=1 path are bitwise those of s0003's kernel except routing-free changes).
- evaluator: valid: 4103 us/step at B=1 and 9491 at B=8 over HTTP (online (submit/poll)); in process 4006 us/step at B=1 (generate, MBU 0.173)

### #5 s0005: B=8: 256-key attention chunks at long context, MMA relative bias, faster B>=2 gates/top-8

- status: **valid**, http_step_us_b1 = 4083 (B=8: 8531), at 3.97 agent-hours, parent s0004, category attention
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). Since s0004: attention items of 256 keys when 128-key items would need >1 grid round (B=8 ctx>=256: rounds 4->2 at ctx 1000), relative bias by one MMA per key tile from bulk-copied bias rows, B>=2 gate z rows one warp per (row, sequence), radix top-8 for B>2. Expect B=8 HTTP 9491 -> ~8100 us (in-process -900 at ctx 64-192, -800 more at ctx 900); B=1 unchanged ~4100 +-50 (compile-layout noise).
- evaluator: valid: 4083 us/step at B=1 and 8531 at B=8 over HTTP (online (submit/poll)); in process 3992 us/step at B=1 (generate, MBU 0.173)

### #6 s0006: B=8 MoE: no w2 L2 prefetch, pipelined w2 rounds, 12-item ring stages, w13 without rounds

- status: **valid**, http_step_us_b1 = 3961 (B=8: 7291), at 4.68 agent-hours, parent s0005, category memory
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). The B=8 MoE was decode/overhead bound: the w2 L2 prefetch delayed w13 and was evicted (-430 us B=8, -100 B=1), w2 rounds drained the weight ring (pipelined: -196), ring stages of 6 items had ~100 instructions of overhead each (12-item stages, no proxy fence: +27% decode rate, -390). Expect B=8 HTTP 8531 -> ~7500 us, B=1 4083 -> ~3950.
- evaluator: valid: 3961 us/step at B=1 and 7291 at B=8 over HTTP (online (submit/poll)); in process 3873 us/step at B=1 (generate, MBU 0.179)

### #7 s0007: B=8: w2 as (slot, rowtile) units like w13; per-slot w13 counters replace the E->F barrier

- status: **valid**, http_step_us_b1 = 4009 (B=8: 6974), at 5.26 agent-hours, parent s0006, category memory
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). At B=8 the w2 phase ran 2-3 rounds because each block owned one output rowtile and needed all ~64 pair activations; now blocks take contiguous (slot, rowtile) units (<=16 activation rows), write bf16 y, and a combine after a barrier sums each token's experts in slot order; the E->F grid barrier is replaced by per-slot completion counters. In process B=8 -290 us (6878 -> 6586); expect HTTP B=8 7291 -> ~7000, B=1 unchanged ~3960.
- evaluator: valid: 4009 us/step at B=1 and 6974 at B=8 over HTTP (online (submit/poll)); in process 3927 us/step at B=1 (generate, MBU 0.176)

### #8 s0008: Attention in one grid round at B=8: key-range pieces, 256-key sub-chunks, online softmax

- status: **valid**, http_step_us_b1 = 4065 (B=8: 6851), at 5.57 agent-hours, parent s0007, category attention
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). At B=8 the 256-key attention items took 2-3 grid rounds beyond ctx 512 (+1700 us from ctx 64 to 1088); items are now (sequence, kv head, piece) with npc = min(chunks, 152/64) pieces processed as 256-key sub-chunks with an online softmax and separate K/V barriers so the next sub-chunk's K loads during softmax/PV. In process B=8 -300..-590 us beyond ctx 512 (mean over ctx 64-1088 ~-190), B=1 unchanged within 30 us. Expect HTTP B=8 6974 -> ~6800, B=1 ~4000.
- evaluator: valid: 4065 us/step at B=1 and 6851 at B=8 over HTTP (online (submit/poll)); in process 3977 us/step at B=1 (generate, MBU 0.174)

### #9 s0009: mixnorm argument arrays in shared memory instead of stack arrays (local memory)

- status: **valid**, http_step_us_b1 = 3977 (B=8: 6662), at 5.71 agent-hours, parent s0008, category memory
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). mixnorm (~100 calls per step, on the critical path at both batch sizes) spent 2.9 of its 6.8 us in the load phase: its source codes and query/weight/row pointer arrays were stack arrays (local memory, which misses to L2 here: a dependent round trip before every source load). Now block-uniform arrays in shared memory, call signatures unchanged; stack frame 1280 -> 880 B. In process B=1 -62 us, B=8 -137 (bitwise identical). Expect HTTP B=1 4065 -> ~4000, B=8 6851 -> ~6720.
- evaluator: valid: 3977 us/step at B=1 and 6662 at B=8 over HTTP (online (submit/poll)); in process 3883 us/step at B=1 (generate, MBU 0.178)

### #10 s0010: No stack pointer arrays in mix_wait_rows / norm paths / w2 combine; coefficient arrays in registers

- status: **valid**, http_step_us_b1 = 3846 (B=8: 6411), at 5.83 agent-hours, parent s0009, category memory
- hypothesis: track A (all formats as the checkpoint stores them; expert inputs unquantized BF16). Continuing s0009: the remaining stack arrays on critical paths (mix_wait_rows' 8-24 source/destination pointers read by thread 0 before the row copies, mixnorm's runtime-indexed coefficient arrays, the loop-gate norm source arrays, the w2 combine's expert lists) are gone: stack frame 880 -> 208 B. In process B=1 -140 us, B=8 -150 (bitwise identical). Expect HTTP B=1 3977 -> ~3850, B=8 6662 -> ~6520.
- evaluator: valid: 3846 us/step at B=1 and 6411 at B=8 over HTTP (online (submit/poll)); in process 3780 us/step at B=1 (generate, MBU 0.183)

### #11 s0011: Fast math in the fp32 vector code (approx exp/tanh/rsqrt/div, FTZ); o_proj stream after attention

- status: **valid**, http_step_us_b1 = 3679 (B=8: 6147), at 6.55 agent-hours, parent s0010, category other
- hypothesis: track A (weights, activations and K/V in the checkpoint's / specification's formats; expert inputs unquantized BF16; the fp32 vector math (softmax exp, sigmoid, tanh soft-caps, rsqrt/divisions of the norms and clips) uses the fast approximate instructions, FTZ). The critical path's vector code is instruction-latency bound (accurate expf/tanhf/IEEE division are long chains); in process B=1 -121 us, B=8 -239; vs fp32 reference README 320 positions median 0.0347 (precise 0.0379). Expect HTTP B=1 3846 -> ~3720, B=8 6411 -> ~6170; numeric gates median ~0.025 / p99 ~0.13 as before.
- evaluator: valid: 3679 us/step at B=1 and 6147 at B=8 over HTTP (online (submit/poll)); in process 3628 us/step at B=1 (generate, MBU 0.190)

### #12 s0012: B=1 deep w13 stream (3 stages in flight), acq_rel attention counter, redux top-8

- status: **valid**, http_step_us_b1 = 3634 (B=8: 6045), at 6.84 agent-hours, parent s0011, category memory
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fp32 vector math with fast approximate instructions as s0011). At B=1 each warp streams one 36-item w13 unit through a 2-stage ring (one refill round trip); now all three 12-item stages are issued at once (the idle ring plus a buffer in xs, dedicated barriers): B=1 -44 us in process, bitwise identical. Attention partial counts by one acq_rel atomic (long ctx B=1 -18, B=8 -76); top-8 by redux.sync (B=8 -25). Expect HTTP B=1 3679 -> ~3630, B=8 6147 -> ~6090.
- evaluator: valid: 3634 us/step at B=1 and 6045 at B=8 over HTTP (online (submit/poll)); in process 3589 us/step at B=1 (generate, MBU 0.193)

### #13 s0013: Kernel specialised by batch size (B=1, B=8 templates + generic): other paths fold away

- status: **valid**, http_step_us_b1 = 3577 (B=8: 5737), at 7.25 agent-hours, parent s0012, category structure
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fp32 vector math with fast approximate instructions as s0011). Every code path added to the 554 KB kernel cost B=1 ~60 us even when not taken (layout / allocation at 255 registers); instantiating the kernel with B as a compile-time constant for B=1 and B=8 (generic for 2-7) folds the other batch sizes' paths away: B=1 kernel without spills (16 B stack). In process B=1 -70 us, B=8 -255 (bitwise identical). Expect HTTP B=1 3634 -> ~3565, B=8 6045 -> ~5790.
- evaluator: valid: 3577 us/step at B=1 and 5737 at B=8 over HTTP (online (submit/poll)); in process 3536 us/step at B=1 (generate, MBU 0.195)

### #14 s0014: B>1 kernels: mixnorm specialised for 1 query/1 stream; gate z loads before stores

- status: **valid**, http_step_us_b1 = 3573 (B=8: 5674), at 7.51 agent-hours, parent s0013, category other
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). With the kernel specialised by batch size, two B>1-only changes pay in the B=8 kernel: mixnorm's 1-query/1-stream instantiation (a third of the code; 41% of its stalls were instruction-cache misses) for local layers and MLP mixes, and the 6 z loads per thread of gate phase 1 issued before their stores. In process B=8 -54 and -73 us, B=1 unchanged, bitwise identical. Expect HTTP B=8 5737 -> ~5620, B=1 ~3580.
- evaluator: valid: 3573 us/step at B=1 and 5674 at B=8 over HTTP (online (submit/poll)); in process 3538 us/step at B=1 (generate, MBU 0.195)

### #15 s0015: o_proj short-conv inputs loaded before the GEMV; B=8 w2 path fixed at compile time

- status: **valid**, http_step_us_b1 = 3573 (B=8: 5645), at 7.70 agent-hours, parent s0014, category cleanup
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). Two small changes: the attention short conv's taps and history are loaded before the o_proj GEMV (one L2 round trip off the critical path), and the B=8 kernel takes the expert-parallel w2 path unconditionally (dead path removed). In process B=8 about -45 us (within the ~50 us run-to-run spread at B=8), B=1 unchanged. Expect HTTP ~3575 / ~5630.
- evaluator: valid: 3573 us/step at B=1 and 5645 at B=8 over HTTP (online (submit/poll)); in process 3539 us/step at B=1 (generate, MBU 0.195)

### #16 s0016: B=1 kernel: 128-key attention sub-chunks; bias rows at a fixed smem offset

- status: **valid**, http_step_us_b1 = 3553 (B=8: 5671), at 8.10 agent-hours, parent s0015, category attention
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). The B=1 kernel's attention pieces are single 128-key chunks, so its sub-chunk size can be 128 (KPW 1, shorter softmax/PV loops) with bitwise identical results: in process B=1 -31 us at ctx 64, -9 at ctx 900; B=8 kernel unchanged (bitwise). Also moves the bias rows to a fixed +64 KB smem offset (the old offset collided with V-in-xs when CH2 = 128; no effect at CH2 = 256). Expect HTTP B=1 3573 -> ~3555, B=8 ~5645.
- evaluator: valid: 3553 us/step at B=1 and 5671 at B=8 over HTTP (online (submit/poll)); in process 3510 us/step at B=1 (generate, MBU 0.197)

### #17 s0017: B=1 kernel without its dead paths (expert-parallel w2, generic w13, F round loop)

- status: **valid**, http_step_us_b1 = 3515 (B=8: 5645), at 8.27 agent-hours, parent s0016, category cleanup
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). In the B=1 instantiation three paths are never taken (8 routed pairs: one F round, no expert-parallel w2; at most 8 w13 units per block: always the deep stream); deciding them at compile time removes their code (layout/allocation: -9, -49, -10 us in process, each bitwise identical). Expect HTTP B=1 3553 -> ~3490, B=8 unchanged ~5650.
- evaluator: valid: 3515 us/step at B=1 and 5645 at B=8 over HTTP (online (submit/poll)); in process 3463 us/step at B=1 (generate, MBU 0.198)

### #18 s0018: B=8 kernel without the round planning at the end of E (final state)

- status: **valid**, http_step_us_b1 = 3515 (B=8: 5608), at 8.39 agent-hours, parent s0017, category cleanup
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). The B=8 kernel no longer plans w2 rounds at the end of E (it always takes the expert-parallel w2 path): one block sync and the planning loop less per application; in process -18 us at B=8 (within the B=8 run-to-run spread), B=1 kernel unchanged. Expect HTTP B=1 ~3515, B=8 ~5630.
- evaluator: valid: 3515 us/step at B=1 and 5608 at B=8 over HTTP (online (submit/poll)); in process 3462 us/step at B=1 (generate, MBU 0.199)

### #19 s0019: Expert-choice recording (diagnostics) off the critical path

- status: **valid**, http_step_us_b1 = 3490 (B=8: 5606), at 8.55 agent-hours, parent s0018, category cleanup
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). Block 0 recorded every application's chosen experts (a diagnostics table) before starting its w13 units, delaying the block the E phase waits for; now behind a debug switch. In process B=1 -14 us (bitwise identical), B=8 within noise. Expect HTTP B=1 3515 -> ~3500, B=8 ~5610.
- evaluator: valid: 3490 us/step at B=1 and 5606 at B=8 over HTTP (online (submit/poll)); in process 3438 us/step at B=1 (generate, MBU 0.201)

### #20 s0020: Router stages at the end of o_proj; 32 reduction slots

- status: **valid**, http_step_us_b1 = 3489 (B=8: 5593), at 9.43 agent-hours, parent s0019, category cleanup
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). Two small changes measured with one engine per process (reproducible to +-1 us): the router's first weight stages issued at the end of o_proj (PFE_R) and 32 instead of 48 reduction slots (8 KB more L1). In process B=1 -2 us, B=8 -8..-12. Expect HTTP B=1 ~3488, B=8 ~5595.
- evaluator: valid: 3489 us/step at B=1 and 5593 at B=8 over HTTP (online (submit/poll)); in process 3438 us/step at B=1 (generate, MBU 0.201)

### #21 s0021: Weights in one arena in order of use (64 KB alignment, 16 KB skew per tensor)

- status: **valid**, http_step_us_b1 = 3506 (B=8: 5585), at 9.86 agent-hours, parent s0020, category memory
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). Weight placement alone moved B=8 by up to 80 us between engine instances; copying all weights into one allocation in the kernel's order of use with a per-tensor skew makes their offsets deterministic and measured B=1 -50 us (one engine per process, reproducible to +-1 us), B=8 equal. Expect HTTP B=1 3489 -> ~3440, B=8 ~5595.
- evaluator: valid: 3506 us/step at B=1 and 5585 at B=8 over HTTP (online (submit/poll)); in process 3461 us/step at B=1 (generate, MBU 0.200)

### #22 s0022: Mixer blocks on the blocks without o_proj / combine rows (144-151)

- status: **valid**, http_step_us_b1 = 3486 (B=8: 5563), at 10.10 agent-hours, parent s0020, category scheduling
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). At B>1 the 8 mixer blocks (one sequence's attention-residual mix each, ~97 mixes per step at B=8) were blocks 0-7, which also own o_proj / combine rowtiles; blocks 144-151 have none, so their mixes start with less pending work. One engine per process: B=8 -19 us at ctx 64, -9 at ctx 900; B=1 unchanged. Expect HTTP B=8 5593 -> ~5580, B=1 ~3490.
- evaluator: valid: 3486 us/step at B=1 and 5563 at B=8 over HTTP (online (submit/poll)); in process 3436 us/step at B=1 (generate, MBU 0.201)

### #23 s0023: Router GEMV split in K over 16 (B>1) / 24 (B=1) blocks

- status: **valid**, http_step_us_b1 = 3450 (B=8: 5565), at 10.26 agent-hours, parent s0022, category gemm
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). The router GEMV (128 x 2304) ran on 8 blocks, 18 ksteps per warp (two ring stages on the critical path between the MLP mix and routing); split in K over 16 blocks at B>1 and 24 at B=1, partial logits summed when routing reads them. One engine per process: B=1 -34 us, B=8 -17 (ctx 64) / -11 (ctx 900). Expect HTTP B=1 3486 -> ~3455, B=8 5563 -> ~5550.
- evaluator: valid: 3450 us/step at B=1 and 5565 at B=8 over HTTP (online (submit/poll)); in process 3398 us/step at B=1 (generate, MBU 0.203)

### #24 s0024: Gate phase 1: F/G chunk order rotated by row (shared-memory banks)

- status: **valid**, http_step_us_b1 = 3442 (B=8: 5513), at 10.50 agent-hours, parent s0023, category other
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). In the loop gates' phase 1 every thread read its own F/G row (rows 128 B apart) in the same chunk order, so lanes hit the same banks; rotating the chunk order by the row spreads them (summation order changes, deterministic). One engine per process: B=8 -29 us (ctx 64), -6 (ctx 900), B=1 -7. Expect HTTP B=8 5565 -> ~5545, B=1 ~3445.
- evaluator: valid: 3442 us/step at B=1 and 5513 at B=8 over HTTP (online (submit/poll)); in process 3390 us/step at B=1 (generate, MBU 0.204)

### #25 s0025: Weight ring stages of 9 items x 2 (no partial stages)

- status: **valid**, http_step_us_b1 = 3453 (B=8: 5443), at 10.68 agent-hours, parent s0024, category memory
- hypothesis: track A (formats as stored; expert inputs unquantized BF16; fast fp32 vector math as s0011). With 12-item ring stages every 18-item NVFP4 w2 unit ran as 12 + a slow 6-item partial stage (also the router, qkv halves); 9-item stages divide 18, 36 and 72, and the smaller ring leaves 27 KB more L1. One engine per process: B=8 -84 us (ctx 64) / -70 (ctx 900), B=1 -18, bitwise identical. Expect HTTP B=8 5513 -> ~5440, B=1 3442 -> ~3425.
- evaluator: valid: 3453 us/step at B=1 and 5443 at B=8 over HTTP (online (submit/poll)); in process 3401 us/step at B=1 (generate, MBU 0.203)

### #26 s0026: Ring stage size per kernel: 6 items x 2 in the B=1 kernel, 9 x 2 elsewhere

- status: **valid**, http_step_us_b1 = 3428 (B=8: 5460), at 10.75 agent-hours, parent s0025, category memory
- hypothesis: track A (formats as the checkpoint). B=1's w13/w2 tasks are short and latency-bound, so smaller stages (6 items) start decoding sooner; B=8 keeps 9-item stages. Lone-process: B=1 -36 us, B=8 unchanged; expect HTTP B=1 ~3420, B=8 ~5440.
- evaluator: valid: 3428 us/step at B=1 and 5460 at B=8 over HTTP (online (submit/poll)); in process 3372 us/step at B=1 (generate, MBU 0.205)

### #27 s0027: B=1 kernel: ring slot refilled before the stage's MMAs (stage words in registers)

- status: **valid**, http_step_us_b1 = 3350 (B=8: 5455), at 11.01 agent-hours, parent s0026, category memory
- hypothesis: track A (formats as the checkpoint). In the B=1 kernel each full ring stage's weight and scale words are loaded to registers first, then the slot is refilled by TMA before the MMAs, so the next load overlaps this stage's compute (B=8 kernel unchanged: it would spill). Lone-process B=1 -70; expect HTTP B=1 ~3360, B=8 unchanged ~5450.
- evaluator: valid: 3350 us/step at B=1 and 5455 at B=8 over HTTP (online (submit/poll)); in process 3297 us/step at B=1 (generate, MBU 0.210)

### #28 s0028: 128-key attention sub-chunks in the B=8 kernel too (as in B=1)

- status: **valid**, http_step_us_b1 = 3385 (B=8: 5466), at 11.64 agent-hours, parent s0027, category attention
- hypothesis: track A (formats as the checkpoint). Attention sub-chunks of 128 keys instead of 256 in the B>1 kernels: one-chunk pieces (ctx < ~512 at B=8) stop paying for a half-empty 256-key softmax/PV, long pieces take two sub-chunks. One engine on the serve workload: B=8 5491 -> 5461; expect HTTP B=8 ~5425, B=1 unchanged ~3350.
- evaluator: valid: 3385 us/step at B=1 and 5466 at B=8 over HTTP (online (submit/poll)); in process 3318 us/step at B=1 (generate, MBU 0.208)

### #29 s0029: B=1 kernel: ring stages of 9 items again (with the early slot refill)

- status: **valid**, http_step_us_b1 = 3328 (B=8: 5457), at 12.16 agent-hours, parent s0028, category memory
- hypothesis: track A (formats as the checkpoint). With the early slot refill the B=1 streams no longer wait on refills, so larger 9-item stages (fewer per-stage costs, no partial stages for 18/36/72-item tasks) beat the 6-item stages chosen without it. One engine on the serve workload: B=1 3377 -> 3330; expect HTTP B=1 ~3310-3340, B=8 unchanged ~5460.
- evaluator: valid: 3328 us/step at B=1 and 5457 at B=8 over HTTP (online (submit/poll)); in process 3273 us/step at B=1 (generate, MBU 0.211)

### #30 s0030: B=1 kernel: early ring refill only on the expert streams (not the BF16 ones)

- status: **valid**, http_step_us_b1 = 3307 (B=8: 5495), at 12.29 agent-hours, parent s0029, category memory
- hypothesis: track A (formats as the checkpoint). The early slot refill helps the short, latency-bound expert streams (NVFP4/MXFP8 w13/w2) but costs on the long BF16 streams (qkv, o_proj, router), which already run at the streaming rate. One engine on the serve workload: B=1 3330 -> 3304; expect HTTP B=1 ~3305, B=8 unchanged ~5460.
- evaluator: valid: 3307 us/step at B=1 and 5495 at B=8 over HTTP (online (submit/poll)); in process 3252 us/step at B=1 (generate, MBU 0.212)

### #31 s0031: B>1 kernels: one mixnorm instantiation (drop the 1-query/1-stream copy)

- status: **valid**, http_step_us_b1 = 3313 (B=8: 5435), at 12.50 agent-hours, parent s0030, category structure
- hypothesis: track A (formats as the checkpoint). The <1,1> mixnorm copy added in s0014 now costs more in code size / allocation than its shorter code saves: removing it measured B=8 -32 us on the serve workload (one engine, twice), generic kernel B=2 -350. Expect HTTP B=8 ~5430 (within the +-65 repeat spread), B=1 unchanged ~3307.
- evaluator: valid: 3313 us/step at B=1 and 5435 at B=8 over HTTP (online (submit/poll)); in process 3264 us/step at B=1 (generate, MBU 0.212)

### #32 s0032: B=8 kernel: router GEMV in one K part (B=1 kernel keeps 3)

- status: **valid**, http_step_us_b1 = 3314 (B=8: 5469), at 12.78 agent-hours, parent s0031, category gemm
- hypothesis: track A (formats as the checkpoint). At B=8 the router's 8 x 2 K-split blocks plus the partial sums cost more than one K part per block: serve workload (one engine, twice) B=8 -14 us, B=1 unchanged. Expect HTTP B=8 ~5420 (inside the +-65 repeat spread), B=1 ~3310.
- evaluator: valid: 3314 us/step at B=1 and 5469 at B=8 over HTTP (online (submit/poll)); in process 3264 us/step at B=1 (generate, MBU 0.212)

