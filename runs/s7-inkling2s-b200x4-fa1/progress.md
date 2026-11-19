# s7-inkling2s-b200x4-fa1: Inkling-2-Small on GB200

Score: `http_step_us_b1` (lower is better). Budget 12.0 agent-hours. 20 submissions so far, 19 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 603
- 1.10x the fastest general engine: 4283
- SGLang PR #50186 image lmsysorg/sglang:dev-inkling-2-small (arm64 sha256:198af079c599) (general engine): 4711
- inherited engine: run s6, submission s0003 (track A): 5324

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Baseline: run s6's s0005 source unchanged (LL exchange, slot-invariant FMA)

- status: **valid**, http_step_us_b1 = 4807 (B=8: 8861), at 0.13 agent-hours, category other
- hypothesis: track A: checkpoint formats unchanged (BF16 dense, NVFP4/MXFP8 experts as stored, inputs quantized as defined). This is run s6's submission s0005 source taken from the open records instead of the inherited s0003; no change of mine. In-process on this node: 4791 us (B=1) and 7681/8157 us (B=8, step/generate) at context 64-128. Expect the same as s6 measured: http_step_us_b1 ~4780 and B=8 ~8850, all gates passing.
- evaluator: valid: 4807 us/step at B=1 and 8861 at B=8 over HTTP (online (submit/poll)); in process 4772 us/step at B=1 (generate, MBU 0.127)

### #2 s0002: Team vector phases: a token's mix/norm/quantize on 6 blocks, 1 element per thread

- status: **valid**, http_step_us_b1 = 4163 (B=8: 7125), at 0.97 agent-hours, parent s0001, category structure
- hypothesis: track A: checkpoint formats unchanged. The per-token vector phases (all-reduce, short conv, residual mix, norms, NVFP4/MXFP8 input quantization) were bound by the instruction path of a thread (8 elements per thread, ~1800 instructions, ~5-9 ns each; measured with in-kernel timers), not by memory. Now one element per thread on 6 whole blocks per token, reductions exchanged between the blocks as tagged words, hardware E2M1/E4M3 encodes (verified equal to the model's rounding on 2^22-2^24 values); B=1 uses the same kernel. In-process rehearsal (online, context to 1088): 4164 us at B=1 and 7169 us at B=8. Expect http_step_us_b1 ~4200 (-13% vs s0001) and B=8 ~7200 (-19%), gates as before (own float64 check: median 0.051, top-1 100%).
- evaluator: valid: 4163 us/step at B=1 and 7125 at B=8 over HTTP (online (submit/poll)); in process 4162 us/step at B=1 (generate, MBU 0.145)

### #3 s0003: GEMV phases: two strips per block at once, w2 quant per routed pair, no staging for qkv/router

- status: **valid**, http_step_us_b1 = 3984 (B=8: 6537), at 1.84 agent-hours, parent s0002, category scheduling
- hypothesis: track A: checkpoint formats unchanged. Scheduling of the GEMV phases only: (1) a block's two strips run concurrently on half the warps each (one barrier and reduction round instead of two), (2) the w2 input quantization runs one routed (token, expert) pair per block in a single pass with hardware E2M1/E4M3 encodes, (3) the w13 epilogue computes two activation values per lane on all lanes instead of four on half, (4) no zero-select for unused batch columns, FP4 unroll 4 then 2, (5) the qkv and router GEMVs read their B fragment straight from L2 instead of staging it in shared memory. Rehearsal online (context to 1088): 3981 us at B=1, 6662 us at B=8. Expect http_step_us_b1 ~4000 (-4% vs s0002) and B=8 ~6650 (-7%); gates unchanged (own float64 check median 0.053, top-1 100%).
- evaluator: valid: 3984 us/step at B=1 and 6537 at B=8 over HTTP (online (submit/poll)); in process 3983 us/step at B=1 (generate, MBU 0.151)

### #4 s0004: Attention scores as BF16 MMAs; tag-checked hand-offs without fences in the w13 path

- status: **valid**, http_step_us_b1 = 3891 (B=8: 6335), at 2.77 agent-hours, parent s0003, category sync
- hypothesis: track A: checkpoint formats unchanged (the relative-bias table is stored distance-major, same BF16 values). (1) Attention: QK^T and the relative bias as BF16 MMAs with FP32 accumulation instead of a 128-step scalar dot per (head, position); the softmax pieces one (head, position) per thread. (2) The strip hand-off of the flat GEMV and the w13 -> w2-quantization hand-off carry a tag in every word and are read with relaxed loads until the tags are current, so the w13 epilogue has no fence, counter or atomic maximum (a fence costs ~0.5 us; removing them without tags made B=8 irreproducible, measured). Rehearsal online (context to 1088): 3888 us at B=1, 6422 us at B=8, deterministic over 120 steps, bitwise equal to the fenced build. Expect http_step_us_b1 ~3890 (-2.4% vs s0003) and B=8 ~6300 (-3.6%).
- evaluator: valid: 3891 us/step at B=1 and 6335 at B=8 over HTTP (online (submit/poll)); in process 3872 us/step at B=1 (generate, MBU 0.156)

### #5 s0005: Shared memory per block 222 -> 121 KB: the SM's L1 gets the difference

- status: **valid**, http_step_us_b1 = 3646 (B=8: 5870), at 3.10 agent-hours, parent s0004, category memory
- hypothesis: track A: checkpoint formats unchanged. Shared memory and L1 share the SM's 256 KB: the kernel asked for 222 KB (a 4608-element token state per block and a 74 KB fragment staging area), leaving ~34 KB of L1. Measured in tools/bench/gemv2.cu: the FP4 GEMV runs 15% slower with 222 KB of shared memory than with 150 KB or 73 KB. Now the token state is 768 elements per team block and the GEMV B fragments (w13, unembedding, as qkv/router/w2 already) are read from global memory through L1; only the attention's K/V chunk rows stay staged (without staging the attention was 5 us slower). 121 KB of shared memory. Results bitwise identical to s0004's build (40 steps, B=1 and B=8), deterministic over 120 steps. Rehearsal online (context to 1088): 3642 us at B=1 (s0004: 3888), 5946 us at B=8 (6422). Expect http_step_us_b1 ~3650 (-6% vs s0004) and B=8 ~5870 (-7%), i.e. B=8 just past 1.10x SGLang (6004).
- evaluator: valid: 3646 us/step at B=1 and 5870 at B=8 over HTTP (online (submit/poll)); in process 3626 us/step at B=1 (generate, MBU 0.166)

### #6 s0006: Expert GEMV unroll 5 (NVFP4) / 4 (MXFP8); activation function moved to the w2 quantization block

- status: **valid**, http_step_us_b1 = 3661 (B=8: 5742), at 3.51 agent-hours, parent s0005, category gemm
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0005's build (30 steps, B=1 and B=8). With the larger L1 of s0005 a deeper load group pays at B=8: fixed-token step 5161 -> 5072 us (two repeats each, +-2 us), B=1 3633 -> 3652. The w13 epilogue now hands over (gate, up) words and the w2 quantization block evaluates the activation function one value per thread (B=1 3652 -> 3633, B=8 unchanged). Rehearsal online (context to 1088): 3657 us at B=1 (s0005: 3642), 5814 us at B=8 (5946). Expect http_step_us_b1 ~3660 (+0.4% vs s0005) and B=8 ~5740 (-2.2%).
- evaluator: valid: 3661 us/step at B=1 and 5742 at B=8 over HTTP (online (submit/poll)); in process 3641 us/step at B=1 (generate, MBU 0.166)

### #7 s0007: w13 GEMV for few experts: whole strips per block, four at a time (B=1)

- status: **valid**, http_step_us_b1 = 3440 (B=8: 5773), at 3.83 agent-hours, parent s0006, category scheduling
- hypothesis: track A: checkpoint formats unchanged. At B=1 the flat schedule split each of the 576 w13 strips over ~6 warps of different blocks; every piece went through the tagged hand-off (two relaxed loads per piece for the last warp) before the epilogue. With at most 16 distinct experts in a layer a block now takes its 3-4 strips whole, 6-8 warps each, one barrier, the reduction in shared memory: w13 12.1 -> 8.3 us per layer in process, step 3640 -> 3431 us at B=1; B=8 (40-55 experts, flat schedule as before) unchanged. Rehearsal online (context to 1088): 3438 us at B=1 (s0006: 3657), 5823 at B=8 (5814). Expect http_step_us_b1 ~3445 (-5.9% vs s0006) and B=8 ~5745 (unchanged).
- evaluator: valid: 3440 us/step at B=1 and 5773 at B=8 over HTTP (online (submit/poll)); in process 3425 us/step at B=1 (generate, MBU 0.176)

### #8 s0008: Attention: 32 or 64 positions per chunk block while blocks are free

- status: **valid**, http_step_us_b1 = 3401 (B=8: 5758), at 4.01 agent-hours, parent s0007, category attention
- hypothesis: track A: checkpoint formats unchanged. The V sum of the attention is a per-thread loop over the positions of the block's chunk (128 positions = 64 per thread); when the grid has enough blocks for every (token, KV head, chunk) the chunk is 32 or 64 positions (B=1: always in this workload; B=8: up to context ~600), which shortens that loop; the last chunk block still combines up to 17 chunks. Fixed-token step: B=1 3433 -> 3378 us at context 64, 3468 -> 3448 at 960; B=8 5084 -> 5019 at context 64, 5076 -> 5036 at 400. Deterministic over 1100 steps (through the chunk-size changes). Rehearsal online (context to 1088): 3400 us at B=1 (s0007: 3438), 5715 at B=8 (5823). Expect http_step_us_b1 ~3405 (-1% vs s0007) and B=8 ~5700 (-1%).
- evaluator: valid: 3401 us/step at B=1 and 5758 at B=8 over HTTP (online (submit/poll)); in process 3362 us/step at B=1 (generate, MBU 0.179)

### #9 s0009: Output-gate projection computed beside the attention phase by idle blocks

- status: **valid**, http_step_us_b1 = 3310 (B=8: 5716), at 4.20 agent-hours, parent s0008, category scheduling
- hypothesis: track A: checkpoint formats unchanged. The output gate (96 of ~240 strips of the q/gate/r/sink/k/v GEMV, 40% of its bytes) is only needed when the attention chunks are combined. When at least 48 blocks are free during the attention phase (B=1 always; B=8 up to context ~400) they run the gate strips there and publish sigmoid(gate) as tagged words that the combining block reads with relaxed loads; the GEMV before the attention shrinks to one strip per block. Fixed-token step: B=1 3374 -> 3287 us (context 64), 3439 -> 3356 (context 960); B=8 5037 -> 4957 at context 64, unchanged at long context. Deterministic over 1100 steps, slot-permutation invariant, own float64 check median 0.050. Rehearsal online: 3304 us at B=1 (s0008: 3400). Expect http_step_us_b1 ~3310 (-2.7% vs s0008); B=8 ~5740 +-1% (its generated text, and so its expert unions, change with the summation order).
- evaluator: valid: 3310 us/step at B=1 and 5716 at B=8 over HTTP (online (submit/poll)); in process 3266 us/step at B=1 (generate, MBU 0.185)

### #10 s0010: Gate split with 24 free blocks (B=8 up to context 1024); own strip piece from registers

- status: **valid**, http_step_us_b1 = 3322 (B=8: 5660), at 4.56 agent-hours, parent s0009, category scheduling
- hypothesis: track A: checkpoint formats unchanged. (1) The chunk size of the attention and the gate split are chosen together: the smallest chunk (32/64/128 positions) that leaves at least 24 blocks free for the gate strips (4 strips per free block, 6 warps each); at B=8 this keeps the gate split up to context 1024 (s0009: up to ~400). Fixed-token step at B=8: context 500: 5023 us (unchanged), context 960: 5073 -> 4993. (2) The warp that finishes a strip of the flat GEMV adds its own piece from registers instead of re-reading it with two relaxed loads (bitwise identical; B=8 4952 -> 4930). B=1 path unchanged. Deterministic over 1100 steps, slot-permutation invariant. Expect B=8 ~5670 (-1% vs s0009, inside its +-0.5% trial spread only partly) and http_step_us_b1 ~3310 (unchanged).
- evaluator: valid: 3322 us/step at B=1 and 5660 at B=8 over HTTP (online (submit/poll)); in process 3272 us/step at B=1 (generate, MBU 0.184)

### #11 s0011: Flat GEMV strip hand-off without a counter: the first-piece warp finishes the strip

- status: **valid**, http_step_us_b1 = 3314 (B=8: 5599), at 4.89 agent-hours, parent s0010, category sync
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0010's build (30 steps, B=1 and B=8). A strip of the flat GEMV (w13 at B=8, unembedding) is split over 2 warps; until now every piece bumped an atomic counter (an L2 round trip the warp waits for) and the last to arrive summed the pieces. The piece order makes that unnecessary: the warp holding the strip's first piece computes it last in its own range, the other warp computes its part first, so the first-piece warp reads the other tagged slot and runs the epilogue; no counter, no atomics. Fixed-token step at B=8: 4944 -> 4893 us (two repeats each), B=1 3288 -> 3284. Deterministic over 1100 steps. Rehearsal online: 3311 us at B=1, 5755 at B=8 (s0010: 3319 / 5815). Expect B=8 ~5600 (-1% vs s0010) and http_step_us_b1 ~3315 (unchanged within its 0.4% spread).
- evaluator: valid: 3314 us/step at B=1 and 5599 at B=8 over HTTP (online (submit/poll)); in process 3267 us/step at B=1 (generate, MBU 0.185)

### #12 s0012: Attention: the block of the last chunk combines; no chunk counter, no fence on its path

- status: **valid**, http_step_us_b1 = 3299 (B=8: 5558), at 5.09 agent-hours, parent s0011, category sync
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0011's build. The chunk blocks of a (token, KV head) used to bump an atomic counter after a fence, and the last to arrive combined the chunks after a second fence. Now the block of the last chunk (it also computes the new K/V row, so it is the last one anyway) combines: the others publish a tag after their fence, the combining block keeps its own chunk in shared memory, polls the tags and reads the others' records through L2. One fence and one atomic round trip less on the attention's critical path. Fixed-token step: B=1 3284 -> 3271 us, B=8 4893 -> 4859 (context 64). Deterministic over 1150 steps with three seeds at B=8/4/1. Rehearsal online: 3295 us at B=1, 5706 at B=8 (s0011: 3311 / 5755). Expect http_step_us_b1 ~3300 (-0.4% vs s0011) and B=8 ~5560 (-0.7%), both about the size of the trial spread.
- evaluator: valid: 3299 us/step at B=1 and 5558 at B=8 over HTTP (online (submit/poll)); in process 3250 us/step at B=1 (generate, MBU 0.186)

### #13 s0013: Attention, routing and w2-quantization phases compiled out of line (register cap)

- status: **valid**, http_step_us_b1 = 3296 (B=8: 5484), at 5.78 agent-hours, parent s0012, category other
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0012's build (30 steps, B=1 and B=8). The persistent kernel is one 19 000-instruction function at the register cap (REG:80 at 768 threads). Compiled out of line (__noinline__), the bodies of the attention, routing and w2-quantization phases no longer take registers from the GEMV loops: fixed-token step B=8 4858 -> 4770 us, B=1 3270 -> 3261 (STACK stays 96; with the team phases out of line as well it spills and B=1 gets slower; each combination was measured, they are not additive). Rehearsal online on the new machine: 3290 us at B=1, 5590 at B=8 (s0012: 3295 / 5706). Expect B=8 ~5460 (-1.8% vs s0012) and http_step_us_b1 ~3290 (-0.3%, inside its spread).
- evaluator: valid: 3296 us/step at B=1 and 5484 at B=8 over HTTP (online (submit/poll)); in process 3243 us/step at B=1 (generate, MBU 0.186)

### #14 s0014: NVFP4 w13 at batch >= 4: tcgen05 block-scaled MMAs from a second layout

- status: **error**, no score, at 7.79 agent-hours, parent s0013, category gemm
- hypothesis: track A: checkpoint formats unchanged (NVFP4 codes and block scales as stored: the second layout is verified bit for bit on all 40 NVFP4 layers; the w13 input quantized as the model defines, emitted as E2M1 codes and E4M3 block-scale bytes instead of their product). From batch 4 on a second kernel runs the NVFP4 w13 GEMV through the 5th-generation tensor core: a second copy of the w13 bytes laid out as its 128x64 operands (+46 GB per card) is streamed by cp.async.bulk into shared memory and multiplied by block-scaled MMAs (kind::mxf4nvf4), four streams per block. The m16n8k16 path is bound by the SM's instruction rate (0.69 us per expert per layer), this one by memory bandwidth (0.59 measured in process). Batches 1-3 run the unchanged kernel. Expect B=8 5484 -> ~5150 us (-6%), http_step_us_b1 unchanged ~3296.
- evaluator: the engine failed during the timed run (rc 3)

### #15 s0015: s0014 again: tcgen05 w13 at batch >= 4, race fix and a stall report

- status: **valid**, http_step_us_b1 = 3300 (B=8: 5552), at 8.65 agent-hours, parent s0014, category gemm
- hypothesis: track A: checkpoint formats unchanged; same change as s0014 (NVFP4 w13 at batch >= 4 through tcgen05 block-scaled MMAs from a second, bit-for-bit verified layout of the same bytes; batches 1-3 on the unchanged kernel). s0014 passed every gate and then deadlocked once in the timing stage after ~5000 batch-8 steps; 250 000 local steps of the same call patterns (un-synchronised prompt steps, generate, batch 1 then 8) on the submitted source did not stall, so the cause is not proven. Changed: the four read-out warps leave their loop together before the barrier phase counts move (a race found by reading), the token tensors of queued steps stay alive, and the engine prints where block 0 of every card is if a launch does not end within 45 s. Expect, if it does not stall, B=8 ~5200 us (-5% against s0013's 5484) and http_step_us_b1 ~3296; if it stalls, the report in the runner's output locates the wait.
- evaluator: valid: 3300 us/step at B=1 and 5552 at B=8 over HTTP (online (submit/poll)); in process 3250 us/step at B=1 (generate, MBU 0.186)

### #16 s0016: Tensor-core kernel: its calls moved out of the kernel body (routing phase)

- status: **valid**, http_step_us_b1 = 3300 (B=8: 5413), at 9.02 agent-hours, parent s0015, category scheduling
- hypothesis: track A: checkpoint formats unchanged (as s0015). In s0015 the tensor-core kernel gained nothing at batch 8: its NVFP4 w13 GEMV is faster per distinct expert (0.59 against 0.69 us), but the fragment-path loops of the same kernel (the MXFP8 layers) ran 10-15% slower per expert than in the plain kernel, because the calls to the out-of-line tensor-core functions (set-up, release, the GEMV) sat in the kernel body and cost those loops registers. The calls are now made from inside the routing phase function; MXFP8 w2 is back at 0.57 us per expert, w13 at 1.09 (plain kernel 1.06). Fair A/B (tools/t_replay.py: one generated text replayed teacher-forced through both kernels in one process): 5467 -> 5393 us per step at batch 8 (-1.4%). Expect the in-process step_us at B=8 5528 -> ~5450 and W1-like 5760 -> ~5680; over HTTP ~5450 +- 40 at B=8 (trial spread ~0.7%) and http_step_us_b1 unchanged ~3296.
- evaluator: valid: 3300 us/step at B=1 and 5413 at B=8 over HTTP (online (submit/poll)); in process 3254 us/step at B=1 (step, MBU 0.186)

### #17 s0017: Progress pointer out of the launch header: plain kernel back at STACK:96

- status: **valid**, http_step_us_b1 = 3298 (B=8: 5420), at 10.44 agent-hours, parent s0016, category other
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0016 (tools/t_same.py, 30 steps at B=1 and B=8). The pointer to the tensor-core kernel's progress words travelled in the per-launch header, a kernel parameter passed by value: 8 more bytes there showed as STACK:104 instead of 96 in the plain (batch 1) kernel, and its in-process step was 6.6 us above s0013's in both s0015 and s0016. The pointer now sits in Ctx. Measured on fixed tokens at B=1: 3263-3273 -> 3259.5 us per step; the batch-8 replay is unchanged (5398-5403 against 5393-5397 us). Expect http_step_us_b1 3300 -> ~3296 (step_us 3254 -> ~3248) and B=8 unchanged at ~5413 +- 40.
- evaluator: valid: 3298 us/step at B=1 and 5420 at B=8 over HTTP (online (submit/poll)); in process 3245 us/step at B=1 (generate, MBU 0.186)

### #18 s0018: Tensor-core w13: sub-items of 12 units instead of 9

- status: **valid**, http_step_us_b1 = 3298 (B=8: 5440), at 10.91 agent-hours, parent s0017, category scheduling
- hypothesis: track A: checkpoint formats unchanged. A block of the tensor-core kernel cuts its range of w13 units into sub-items dealt to its four streams and read out one by one; with sub-items of at most 12 units instead of 9 there are a quarter fewer read-outs while the streams stay balanced (24 units: unbalanced again). Replay of one generated text at batch 8 (tools/t_replay.py): 9 units 5399 us per step, 12: 5385 (-0.27%), 24: 5424, 36: 5512. The summation order of w13 changes (gates: t_gate 0.0525 / 0.0958, t_gate8 0.0507 / 0.0927; slot independence and determinism checked; soaks of 20 000 steps with 1, 2, 3 and 8 distinct sequences clean). Expect the in-process step_us at B=8 5440 -> ~5425 (-0.3%), HTTP B=8 ~5405 +- 40, http_step_us_b1 unchanged ~3298 (plain kernel untouched).
- evaluator: valid: 3298 us/step at B=1 and 5440 at B=8 over HTTP (online (submit/poll)); in process 3246 us/step at B=1 (generate, MBU 0.186)

### #19 s0019: Tensor-core w13: its scale-factor stores spread over three warp groups

- status: **valid**, http_step_us_b1 = 3307 (B=8: 5390), at 11.30 agent-hours, parent s0018, category scheduling
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0018 (tools/t_same.py, 30 steps at B=1 and B=8). The block scales of the w13 input are stored into tensor memory beside the routing phase: nine stores of 32 columns by warps 20-23. The routing phase waited for them (+0.6 us per layer against the plain kernel). Three groups of four idle warps (12-23) now issue three stores each. Replay of one generated text at batch 8 (tools/t_replay.py): 5383 -> 5345 us per step (-0.7%). Soaks with 1, 2 and 8 distinct sequences (20 000 steps each) clean. Expect the in-process step_us at B=8 5434 -> ~5395, HTTP B=8 ~5400 +- 40, http_step_us_b1 unchanged ~3298.
- evaluator: valid: 3307 us/step at B=1 and 5390 at B=8 over HTTP (online (submit/poll)); in process 3253 us/step at B=1 (generate, MBU 0.186)

### #20 s0020: Tensor-core kernel: w13 quantization emit inlined in its team phase

- status: **valid**, http_step_us_b1 = 3307 (B=8: 5378), at 11.52 agent-hours, parent s0019, category other
- hypothesis: track A: checkpoint formats unchanged; results bitwise identical to s0019 (tools/t_same.py, 20 steps at B=1 and B=8). The function that emits the w13 input of the tensor-core kernel as E2M1 codes and E4M3 scale bytes was out of line from the time both paths shared one kernel; it is only instantiated in the tensor-core kernel now, and inlined there its post-attention team phase is shorter (the kernel's stack figure goes from 192 to 176). Replay of one generated text at batch 8 (tools/t_replay.py): 5346 -> 5334 us per step (-0.22%). Soaks with 1, 2 and 8 distinct sequences (20 000 steps each) clean. Expect the in-process step_us at B=8 5392 -> ~5380, HTTP B=8 ~5380 +- 40, http_step_us_b1 unchanged (~3300 on this machine).
- evaluator: valid: 3307 us/step at B=1 and 5378 at B=8 over HTTP (online (submit/poll)); in process 3253 us/step at B=1 (generate, MBU 0.185)

