# Journal (run s4-flint-b200-a2)

## Card measurements (tools/bench1.cu, bench2.cu; card = one B200 of the machine)
- Streaming read, 16 B per thread per load, one block per SM: 152 blocks x 1024 threads 7.5 TB/s; 128 blocks x 1024 threads
  7.6 TB/s; 96 x 1024 7.4 TB/s; 152 x 512 6.7 TB/s; 152 x 256 4.4 TB/s. So 128 "main" blocks are enough to saturate
  the memory and 24 SMs are free for other work. The 8000 GB/s used for MBU is close to real (7.5 measured).
- Grid barrier in a persistent kernel (one atomic counter, thread 0 of each block spins): 0.83 us with nothing else.
- A phase of 21 MB spread over all blocks + barrier: 4.0 us (pure DRAM time 2.8 us). Loads issued BEFORE the barrier
  overlap with it: 8.4 MB phases 2.22 -> 1.54 us, 16.8 MB 3.35 -> 2.71 us (bench2 MK_PRE).
- Re-reading data another SM read just before (L2 hit) is only ~1.3x faster than DRAM: L2 is not a useful staging buffer.
- Global stores are NOT visible to other blocks after the barrier without __threadfence (v2 without fence: wrong
  hidden states, generate != step). Fence costs ~0.6 us per barrier when all 1024 threads call it.

## Checkpoint (harness census e0003; tools/hf_study.py e0006; tools/hf_semantics.py e0011)
- 39.79 B BF16 weights, no zeros, no denormals, no identical tensors, no zero rows/cols; exponent entropy 2.56 bits,
  mantissa 6.97 bits, 12-bit code (sign + 4-bit exponent window per row + 7 mantissa bits) has 6.3e-5 escapes
  -> 12.002 bits/weight; an 11-bit code 1.3% escapes (11.42 bits with 32-bit escapes). Looks like plain trained
  Gaussian-ish weights; nothing structural to exploit in the values themselves.
- Routing on 8 own sequences x 1152 tokens (HF BF16): expert usage entropy 6.7-7.6 of 8 bits per layer, top-6 experts
  take 8-27% of the picks, 12-24% of a token's experts repeat from the previous token (2% in layer 2), no correlation
  between consecutive layers (2.3% = chance). Over a batch of 8: 38.4 unique experts per layer on average (of 48).
  Router logits |max| ~11 (sigmoid far from saturation in f32).
- HF BF16 semantics (measured, element-exact): linear = f32 accumulate, one rounding; RMSNorm f32 -> BF16;
  SwiGLU = bf16(silu(g)) * u rounded; residual adds in BF16; experts: per-expert BF16 outputs times f32 routing
  weight, SUMMED IN F32 over the 6 experts, one rounding (the grouped/batched implementation, not the index_add loop).
- HF load takes 16 s (page cache hot); my loader 2.1 s.

## s0001 (track A, first megakernel) valid, 2679 us B=1, 7425 us B=8
Expected: valid, ~2700/7200. Measured: 2679/7425. Gates: rel-L2 median 0.060 (limit 0.121), max 0.98 (limit 1.98),
top-1 93.6% (HF 92.9%, limit 91.9%), batch bitwise equal, generate == step. Milestones build, correct, device_loop,
single_launch, gpu_never_waits. Profile (us per step at B=1, by switching parts off): attention 665, routing
(top-6 in every block, O(n^2) rank loops) 563, QKV phase 330, down phase 362, barriers+fence ~520, head 95.
Conclusion: an SM executes only ~1 warp-instruction per ns, so scalar bookkeeping code, not memory, is the bottleneck.

## Cost model of one SM (tools/bench3.cu, in-kernel clock profiler tools/prof.py on the v2 kernel)
- Instruction throughput is high: 32 warps do a shared-memory load+add in 19.5 ns per round (0.6 ns per warp-iteration),
  a dependent tensor-core mma chain costs 1.5 ns per warp-mma, __syncthreads 38 ns with 32 warps.
- __threadfence after a global store: 450-640 ns (it waits for the store to reach L2); without a pending store 156 ns.
- THE fact that shapes the engine: a warp has one global load outstanding and blocks on it for 0.2 us (L2) to 0.3 us
  (DRAM under load). Evidence: unrolling loads never raised the streaming rate (bench1), and in the v2 kernel every
  segment's duration is (number of serial loads per warp) x 0.3 us: head phase 294 serial tile loads per warp = 92 us
  (615 MB at 6.7 TB/s, already 90% of the card's rate); expert gate/up 16 loads = 6.4 us; "preload 10 tiles" = 3 us;
  8 separate 8-byte loads of activation fragments = 1.6 us; attention 13.6 us (about 40 serial loads + 6 syncs).
  Memory-level parallelism = number of warps doing loads. Registers hold what was loaded (64 regs x 1024 threads).
- So per layer: 13.7 us of weight loads are unavoidable (DRAM-bound at 7.5 TB/s with >= 3000 warps loading), and
  everything else (activation loads, fences, counter hops, attention) must overlap with them or be removed.
- v2 timeline per MoE layer (us): A norm 2.4, A work 4.4, attention 13.6 (main blocks idle 15.1), C 3.3, D 3.1,
  E route 3.9 + gate/up 6.4, F 7.6, hops 1.5-2.0 each. Total 56 us x 49 = 2.7 ms.

## Operator change (2026-11-10 19:00): the score is the HTTP figure
The headline is now decode tokens/s seen by the harness's HTTP client (prompt 64, 1024 new tokens, B=1 and B=8),
through engine.submit()/poll() when present. So the scored path is a device loop over a whole cohort with the
tokens streamed out while the kernel runs; logits are not needed on that path (argmax only).

## v3 (first try at "one load per consumer", K/V staging, counters): correct, but 3259 us at B=1 (slower than v2)
Expected ~1.1 ms. Measured 3.26 ms; every segment 3-5x slower than the instruction count predicted. Two causes found
with microbenchmarks (tools/bench6.cu) and the profile:
- A warp load costs by the number of cache LINES its 32 lanes touch, not by instruction: 32 lanes on 32 different lines
  = about 1 us for the block (bench6 mode 3), a contiguous 512-byte warp load 0.15-0.3 us. v3's activation layouts were
  "one load per thread" but scattered (q/k/v partials: 32 lines per warp; residual stream: 16; fragment copies: 16).
- With 1024 threads a block has 64 registers per thread; staging 10 tiles (40 registers) made the compiler spill
  (1392 bytes of spill stores); every spill reload is a local-memory access (about 20 ns each under L1 pressure).
- Stores do not block a warp (16 stores to distinct lines: +50 ns); one __threadfence after them: +400 ns flat.
Decision: v4 = activation buffers contiguous per consumer warp, weights and V tiles staged in SHARED memory.

## W12 on this card (tools/bench5.cu): worth it only where the memory is the limit
Streaming GEMV, 128 blocks x 32 warps: BF16 tiles 7.39 TB/s; 12-bit codes (3 loads + real decode -> 4 mma) 6.57 TB/s
of codes = 8.76 TB/s of BF16-equivalent (+18.5%); with 152 blocks +27%. The decode (about 60 instructions per
thread and tile) fits in the shadow of the blocking loads.

## v4 (activation buffers contiguous, tiles and V staged in shared memory): correct, 2593 us at B=1
Finer profile and microbenchmarks (bench4 fixed, bench6, bench7) taught four more things:
- Shared memory is not a free staging buffer: 16-byte smem loads/stores run at 60-90 GB/s per SM, the same order as
  the DRAM stream of one SM (49-59 GB/s). "Load tile -> store to smem" costs 0.5 us per tile and warp, and the mma
  loop that reads 160 KB of staged tiles back takes 2.1 us. Staging through smem is a net loss.
- A single warp executes about one instruction every 2 ns; leader code of 500 instructions costs 1 us.
- Stores to lines the SM has not touched recently block like loads: 10 store instructions x 8 lines = 2.7 us.
- No TLB effect: 16.8 MB phases cost 3.39 us whether they come from one 60 GB allocation or 1500 separate ones, in
  sequence or at random (bench7). 8 serial loads per warp at full load = 0.31 us each = 6.7 TB/s.
- Fences are required: without them the outputs differ (digest test, tools/digest.py); they cost 110 us per step.
- Hand-over cost (bench4): store + same-warp atomic counter + poll = 1.25 us; with a fence 1.9 us; all 1024 threads
  fencing 2.2 us; polling per-block version words instead of one counter 3-8 us.
Consequence for the plan: weights go DRAM -> registers -> mma directly; only small things (B fragments, partial
sums) use shared memory; leaders must be short; every global store pattern must be one or two contiguous lines.

## v5 (tiles loaded ahead into 8 registers per thread, lean routing, compact stores): correct, 2694 us (no gain)
The compiler spilled again (572 bytes of spill stores) and the leader segments got slower, not faster (A leader 5.5 us
for 5 store instructions). Reading: a block's stack frames (176 bytes x 1024 threads) do not stay in L1 next to the
weight stream, so a warp-wide spill reload touches 32 different lines and costs up to 1 us (same number as
bench6 mode 3). With 1024 threads x 64 registers, tiles cannot be staged in registers without spills, and shared
memory is too slow to stage them. Decision: 512 threads per block (16 warps x 128 registers): 16 tiles (64 registers)
can be held across the wait; the streaming rate of 16 warps per SM was measured at 6.5 TB/s (bench2), -13% on the
phases that are purely DRAM-bound.

## v6 (512 threads per block, 16 tiles staged in registers, no spills): correct, 2053 us; v6b 1951 us
- v6: 128 registers used, 12 bytes of spills. 2053 us at B=1 (v4 2593), 7222 at B=8.
- v6b: (1) the leader warp's stores + fence + counter bump were delayed about 2 us per hand-over by the other 15
  warps' preload loads queued ahead of them in the SM's memory interface: one __syncthreads after the leader's signal,
  before anyone issues preload traffic: A leader 2.6 -> 1.3 us, F x_update 1.55 -> 0.34 us; (2) expert gate/up and
  head loads in batches of 8 into registers before the mma: E 9.2 -> 6.8 us. Result 1951 us (in-process).
- Remaining per layer (us): E 6.8, attention chain 5.0, routing 2.95, D 2.35, C 2.06, four fences of 1.2 each,
  A (norm 1.2, mma 1.3, leader 1.3), preloads 1.5-2.2 each.

## Why the fence is needed, and how to do without (tools/bench8.cu)
Every block writes its own 128-byte slice of ONE shared buffer and bumps a counter; then every block reads the whole
buffer (so its lines are hot in every reader's L1). 3.1e10 reads per configuration:
- default loads, no fence: 2.8% stale reads, whatever the store's cache operator (default, .cg, .wt);
- any configuration with __threadfence in the writer: 0 stale, 1.97 us per hand-over;
- readers using ld.global.cg (bypass L1) or volatile loads, no fence: 0 stale, 1.54 us per hand-over.
So L1 caches are not coherent across SMs: the stale copy is in the READER's L1, and membar.gl apparently invalidates
it globally (which is why it costs 1.2 us in the kernel, where 144 blocks hold the line). The engine now reads every
buffer another block wrote in the same launch with ld.global.cg and writes without fences (to be verified bit for
bit against the fenced build with tools/digest.py).
- Machine replaced twice so far by the infrastructure ("machine ended: signal", nothing of mine in flight the second time).
  All in-kernel waits now have a watchdog (about 1 s) and every index derived from data is clamped, so a bug
  cannot leave a kernel spinning or produce a wild pointer.

## s0003 (track A, v6c): INCORRECT - tokens gate 91.6% < 91.9%
Expected: valid, http ~2050 us. Measured: every numeric gate passes (rel-L2 median 0.068-0.070, limit 0.121; batch
bitwise equal; generate == step), in-process 1995 us/step and 2012 us/token in the device loop, but top-1 agreement
with the float32 reference is 91.6% (HF BF16 92.9%, limit 91.9%), so no HTTP stage and no score. s0001 (other
summation trees, per-expert BF16 accumulation) had 93.6%. On my own 8 sequences: HF 90.2%, s0001's arithmetic 91.0%,
this one 89.9%: the same order. Conclusion: for an arithmetic that keeps HF's BF16 rounding points, this gate is
decided by which near-tied expert sets flip (48% of the (layer, token) pairs differ between BF16 and float32), i.e.
by summation order; 92.5 +- 1%.
Decisions: (1) track B gets a precision mode that is lossless in exact arithmetic and removes the lottery: residual
stream kept in float32 (HF rounds it to BF16 98 times per token) and the router input fed as hi + lo BF16 parts so
the top-6 is taken on float32-accurate logits; (2) track A stays the BF16 arithmetic behind the build switch; I will
use my own reference as a proxy to pick a summation variant before spending a submission on it.

## s0004 (track B precision mode on the s0003 kernel): VALID, http 2194 us (B=1), 7294 us (B=8)
Expected: tokens gate ~93.5%, rel-L2 median ~0.06, http ~2190 / 7250. Measured: see result.json (tokens gate passes,
rel-L2 median 0.055-0.056, http_tokens 8 of 8 identical, HTTP 2194 / 7294). The HTTP figure equals the device loop
over the same contexts (w1_like_generate 2201 us): the submit/poll path adds nothing; the cost over the 256-step
figure (2031 us) is the growing context (64 -> 1088). One warning: rel-L2 maximum of the batched check 1.76 against a
limit of 1.98 (one position where the top tokens differ); more precision would lower it.
Speed is still 0.90x vLLM at B=1 and 0.52x at B=8.

## Dead end: "virtual warps" (512- and 1024-thread kernels with bit-identical sums) - tools/mk_vwarp.cu.bak
Idea: define every sum on 32 virtual warps so that a 1024-thread kernel (twice the load parallelism, for batch 8)
and the 512-thread kernel (registers for staging, for batch 1) are bit-identical. Both built and gave the same digest.
Measured: 512-thread version 2281 us at B=1 (the plain 512-thread kernel: 2010) and 7909 at B=8; 1024-thread version
2290 at B=1 and 6889 at B=8 (plain 512-thread kernel: 6764). So the 1024-thread kernel does not help batch 8 (its
loads take 0.43 us each instead of 0.21: the expert phase runs at about 4 TB/s with either), and the virtual-warp
bookkeeping costs 270 us at batch 1. Reverted to the s0004 kernel. Batch 8 is not limited by the number of warps.

## Batch-8 profile (s0004 kernel, tools/prof.py --B 8; us per step, 8385 in the profiling build)
expert gate/up 2170 (46 us per layer: 160 MB at 3.5 TB/s, 7 rounds), down phase 1900 (6 rounds per layer; each round
copies the hidden fragments of all 8 sequences, 2 us, although a unit has 1.26 member sequences on average, and loads
16 tiles per warp directly, 3.2 us), routing 1417 (the 8 sequences one after the other, 2.5 us each, in every block),
attention chain 886 (32 pairs on 24 blocks, most of them unstaged), norms 470 (4 column pairs, reloads).
Plan for batch 8: routing in parallel over the sequences; copy only member fragments; aux blocks join the expert
phase; stage attention on all aux blocks. Even then about 5 ms; the 3.45 ms target needs the expert phases near the
card's streaming rate.

## s0005 (track B): VALID, http 1968 us (B=1) = 1.001x vLLM, 6273 us (B=8) = 0.61x
Expected ~1965 / ~6250. Measured 1968 / 6273; in process 1810 us/step, 1834 us/token (ctx 64-320).
What moved since s0004 (2194 / 7294), measured locally step by step (B=1 ctx 64-320 / HTTP-like workload):
- routing finals in parallel + attention mix by (dim chunk, parity) warps (partials 8 KB instead of 64 KB of
  shared memory), compact q operand, leader warps bump the counter: 2010 -> 1936 (B=8: 6764 -> 6187);
- split mma chains (4 accumulators), 5 leader warps in the q/k/v phase, 3 in the expert phase: 1936 -> 1884.
  Trying to hold all 20 q/k/v tiles in registers instead of 16: 2037 (208 bytes of spills): reverted;
- K of key groups 16..35 and 8 more V levels staged in shared memory: HTTP-like 2088 -> 2022;
- head streamed by all 152 SMs instead of 128: 1884 -> 1809, HTTP-like 2022 -> 1964.
A bug that cost an hour: a "//" comment inserted in the middle of a line commented out three pointer
assignments; the build was fine and the watchdog turned the resulting deadlock into slow garbage instead of a hang.

## s0006 (track A build of the s0005 kernel, 2 accumulators per chain): VALID, http 1901 us (B=1) = 1.037x vLLM, 6081 (B=8)
Expected: tokens gate ~93.5% (pass), http ~1940 / ~6200. Measured: see result.json; http 1901 / 6081, in process
1757 us/step. This is the apples-to-apples number: BF16 weights, BF16 activations and residual stream with the
rounding points of the HF BF16 path. How the summation order was chosen: on my own 8 sequences (HF BF16 top-1
agreement with float32 90.2%) the pure-BF16 kernel gives 88.6% / 91.3% / 89.5% with 1 / 2 / 4 accumulators per mma
chain; the middle one was submitted. The spread (3 points from the summation order alone) is the size of the gate's
margin, so for BF16-faithful arithmetic this gate measures luck in the near-tied expert choices, not correctness.
Track A is faster than track B by 3% (no hi+lo router pass, one load instead of two for the residual stream).
The track switch: engine/TRACK (A or B); engine/NACC_A holds track A's accumulator count.

## The shared-memory reservation decides how fast global loads are (tools/bench12.cu) - the biggest card finding so far
Trigger: two restructurings that the load model predicted to help did nothing (batch-8 member-only fragment copies:
5833 -> 5869 us; expert gate/up as 16-tile batches with pipelined loads: B=1 1806 -> 1817, B=8 5869 -> 5710), while
microbenchmarks of the same load pattern were 2-3x faster than the engine's phases. bench9/10/11 ruled out placement
(blocks reading one small region vs far-apart regions: same), page jumps (cycling over up to 128 regions 2-8 MB apart:
same, so no TLB effect), and the load operator (.cg/.cs/.nc/.lu 15% slower than plain at batch 8, same at 16).
What differs is the reservation: bench12, 16 warps x batch of 16 tile loads + mma, one block per SM:
  reservation      <=128 KB   132-160 KB   164-192 KB   196-224 KB
  16 blocks (latency-bound)  0.76 us      0.83         1.13         2.27 us per batch
  128 blocks (DRAM-bound)    7.70 TB/s    7.70         7.58         5.95 TB/s
The L1 data cache and shared memory share one 256 KB pool per SM; with 224 KB reserved (the engine since v6: K/V
staging in shared memory) 28 KB of L1 are left and every batch of loads takes 3x longer. Staging tiles in shared
memory to save load latency cost more load latency than it saved.
Other facts from the same benchmarks:
- A warp CAN have several loads in flight (the earlier "one outstanding load per warp" was wrong; it was the starved
  L1): unloaded, a batch of 8 tile loads takes 0.42 us and of 16 0.76 us; a single load-then-use takes 0.7 us in
  bench9. 128 blocks x 16 warps saturate the memory (7.4-7.7 TB/s) with batches of 8 or 16.
- L2 is 129 MiB (cudaDeviceProp: 135528448). Re-reading a region of up to 96 MB: 17-22 TB/s (2.5-3x the DRAM rate);
  128 MB and more: DRAM rate. prefetch.global.L1/L2 behave like blocking loads (no gain).
Engine change: NV2 = 0 (no V tiles in shared memory), NK2 20 -> 12, reservation 224 -> 128 KB.
Local (track B, card 2): B=1 step 1817 -> 1634 us, generate over the 64+1024 workload 1955 -> 1842; B=8 step
5710 -> 4767, 64+1024 workload 6200 -> 5278.

## s0007 (track B): shared-memory reservation 224 -> 128 KB - valid, HTTP 1782 us at B=1 (1.106x vLLM), 5245 at B=8
Expected http 1855 / 5300. Measured 1782 / 5245 (in process: step 1634, generate 1658, 64+1024 workload 1846 at B=1).
The B=1 HTTP figure is below my prediction because the client's decode rate does not include the prompt phase the
way my local 64+1024 number does. First submission at 1.10x of vLLM at batch 1; batch 8 is at 0.72x.
All gates pass with the same margins as s0005 (rel-L2 median 0.053, max 1.48 of 1.98, bitwise batch equality).

## s0008 (track A): the s0007 kernel in pure BF16 arithmetic - valid, HTTP 1685 us at B=1 (1.17x vLLM), 4796 at B=8 (0.79x)
Expected about 1720 / 5100 and a tokens gate near 93%. Measured 1685 / 4796 (see result.json for the gate values).
The expert gate/up sums changed their tree in this kernel (8 pieces of 16 tiles per row group instead of 5 of 25.6),
so the track-A summation variant was chosen again on my own reference (top-1 agreement over 8 x 128 positions:
1 accumulator 90.5%, 2 accumulators 90.8%, 4 accumulators 90.2%; Hugging Face BF16 90.2%): 2 accumulators stays.
(build.sh fix: MK_NACC now overrides the NACC_<track> file; before, the file won and my first comparison of the three
variants compared the same binary three times.)
Track B's tokens gate moved too with the new tree: s0005 95.65% -> s0007 93.2% (limit 91.9%): the precision mode
does not remove the near-tie lottery, it only shifts it.

## After s0008: batch-8 work, a rare nondeterminism, and a run of machine losses
- Profiler made usable again (it had become the biggest cost in the build it measures: 2317 us against 1634): events
  only in the layers with l % 6 == 2, fine-grained ids only with MK_PROF_FINE=1, clock read with ld.global.cg.
  Sliding MoE layer at batch 1 (track B, us): A 4.0, attention chain 6.4, C 2.2, C->D 2.0, D 2.3, router wait 0.7,
  routing 2.7, expert gate/up 5.6, E->F 2.6, F 3.0, F->A about 2: about 33.5. At batch 8: A 7.3, attention chain
  19.2, C 2.9, C->D 2.3, D 5.3, D->E 2.6, routing 7.9, gate/up 25.8, E->F 3.4, F 24.0: about 101.
- Attention at batch 7-8: 32 pairs on 24 aux blocks meant 8 blocks ran two pairs in sequence (14.6 us against 4.0 for
  one). The 8 extra pairs now run on main blocks 0..7 while they wait (after their own A signal and the A counter).
  Local, track B: batch 8 step 4767 -> 4338 us, 64+1024 workload 5278 -> 4525 us. Bit-identical.
- Down phase (F) at batch 8, three attempts, none kept yet:
  (1) fragment copy of round r+1 issued after the tile requests of round r+1: no gain (4356). Reason found with the
      profiler: a warp's loads return in the order issued, so the small copy loads wait for the 16 tile loads.
  (2) B operands loaded straight from the global buffer into registers before the tiles (no shared-memory copy):
      correct after fixing a slot race, but the compiler spills (116-388 bytes against 64) and batch 1 gets slower
      (1764-1943 us against 1642); batch 8 4345-4382.
  (3) the same as a non-inlined function: the call makes the caller save its registers; 408 bytes of spills, batch 1
      1840, batch 8 4755.
  Kept: the kernel with (1) (tools/mk_v8_fcopy.cu.bak); (2) is in tools/mk_v9_freg.cu.bak.
- Rare nondeterminism: tools/digest.py (about 1500 steps) gave a different digest in 1 of 40 runs with the s0008 kernel,
  1 of 40 with the kernel after the attention change, 3 of 66 with attempt (2). So it is older than today's changes and
  the submitted kernels have it. Not explained by the L1 rule (every cross-block read is ld.global.cg). Candidate:
  phases A and E store results from several warps (A: 5 leaders, E: 2) while thread 0 alone bumps the counter after a
  __syncthreads; a partial-line store (2 or 8 bytes into a line that is not in L1) may reach L2 after another warp's
  atomic. Change prepared (not yet measured, the machine is down): every warp that stores bumps the counter itself after
  its stores (KA = 5 bumps per block in A, KE = 2 in E), as the attention blocks already do.
  tools/flake.py repeats four workloads in one process and counts bitwise mismatches (which path flakes, how often).
- Machine: epochs 4 to 7 lived 3.5 to 9 minutes each (lost with jobs running and with nothing running; "machine ended:
  signal"), epochs 8 and 9 never came up. The clock is paused during the outages. tools/ids.npy now holds my token
  ids so that digest/profile/flake do not need the 2.5-minute Hugging Face reference run after every replacement.

## Offline work during the long machine outage (nothing below is measured yet; variants kept as tools/mk_x|z1..z4.cu.bak)
- X  (fix): the routing union kept its "first occurrence" flags at smi[480..543]; smi has 512 ints, so for batches over 4
  the flags of sequences 4..7 lay on un[0..31] and a warp could read a unit another warp had just written as a flag
  -> wrong unit count/index in that block only -> blocks disagree. This is the best explanation of the rare
  nondeterminism (it needs batch > 4; the digest has batch-8 parts). Present in s0005..s0008.
- Z1: router top-6 for batches over 2 on all sequences at once (a sequence on 2 warps, 4 experts per thread) instead
  of 2 sequences per pass; union through a byte table [expert][sequence] and per-warp ballots instead of two
  64-step scans. The algorithm was checked against brute force in numpy (300 random cases, a third with many ties).
  Expect routing at batch 8 7.9 -> about 3 us per layer.
- Z2: down phase, batches over 2: the round's tiles are requested as 8 + 8 with the fragment loads in between (they
  wait in registers for the "all warps past the mma" barrier, then go to shared memory). Reason: loads return in
  issue order per warp; issued after 16 tiles the copy waited for all of them. Expect F 24 -> about 16 us per layer.
- Z3: norm: all sequence pairs requested at once and kept in registers over both passes (was 2*npair-1 sequential
  round trips: 7 at batch 8). Expect A norm 4.7 -> about 1.5 us and D norm likewise at batch 8.
- Z4: attention: K tiles of non-staged key groups and V tiles beyond the 8 staged ones were loaded one per round trip;
  now 8 per round trip. Expect most of the long-context penalty (+196 us at batch 1 on the 64+1024 workload) and
  of the cost of the 8 unstaged pairs at batch 8 to go.
- Ideas with a measurement prepared: tools/bench13.cu (can L2 be filled without registers: prefetch.global or dead
  loads?), tools/spec_study.py (are a layer's experts predictable from the stream before its attention?).

## State when the loop service stopped (2026-11-11 01:35: "4 machines in a row ended before they were warm; the loop service stops trying")
The machine was lost at 3.78 agent-hours (epoch 7, 2026-11-11 00:02) and epochs 8-11 never became warm; the loop service then
stopped and waits for the operator. The clock is stopped. Nothing after s0008 has been evaluated.
- engine/ is again exactly the kernel of s0007 (track B) / s0008 (track A): the last sealed-valid source. engine/TRACK
  says which one is built (B during development; A + engine/NACC_A = 2 is s0008).
- Known defect of that kernel: the routing scratch overlap described above (rare wrong result at batch > 4, seen as
  1 differing digest in 40 runs). Fix prepared but not run on a GPU: tools/mk_x.cu.bak (minimal), or any z variant.
- Untested work, in the order it should be tried when a machine is back (each is the previous plus one change; all
  type-check with tools/hostcheck/check.sh; the routing and fragment-copy index logic were emulated in Python):
    tools/mk_x.cu.bak   s0008 + attention pairs 24..31 on main blocks + pipelined F copy (both measured, bit-identical)
                        + the routing scratch fix
    tools/mk_z1.cu.bak  + router top-6 on all sequences at once, table-based union
    tools/mk_z2.cu.bak  + fragment loads between the two halves of the tile batch (down phase, batch > 2)
    tools/mk_z3.cu.bak  + norm with all sequence pairs requested at once
    tools/mk_z4.cu.bak  + batched K/V tile loads in attention, MK_POLL_CG switch
  First commands: tools/flake.py 60 on mk_x (is the nondeterminism gone?), then digest/check/time per variant,
  tools/bench13.cu, tools/spec_study.py.

## Back on a machine (epoch 15, 2026-11-11 10:14): the nondeterminism is found and fixed; what the offline variants gave
Measured (track A unless said; step = tools/test_engine.py --time, us per step at B=1 / B=8):
- tools/spec_study.py (HF BF16, 8 x 384 tokens): the real 6 experts of a layer are among the top-k of the router applied
  to the stream BEFORE attention with recall 82.6% (k=6), 90.2% (8), 93.5% (10), 95.3% (12), 97.1% (16); early layers
  76%@6, late ones 90-95%@6. Batch 8, union: k=6 covers 85% of the units used, 84% of what is fetched is used.
- tools/bench13.cu: prefetch.global.L1/L2 gain 0.3 us on a 2.2 us read, dead loads gain nothing: there is no
  register-free way to fill L2. tools/bench14.cu: a grid-wide hand-over costs 0.78 us when thread 0 polls with a plain
  volatile load and 0.105 us with ld.global.cg (the plain poll sits on an L1 copy); in the engine MK_POLL_CG changes
  nothing (1621.4 vs 1621.5): its waits end when the preloaded tiles arrive, not when the poll returns.
- Variants: x (scratch fix) 1642/4362 track B, bit-identical; z1 (parallel top-6 + table union) bit-identical,
  B=8 4362 -> 4141 (track B), with 2 accumulators 1622/4081; z2 (fragment loads between tile halves) and z3 (norm
  with all pairs in registers) and z4 (batched K/V loads) all overflow the 128 registers (144, 836, 1152 bytes of
  spills) and are slower (z2 1773/4405, z3 2254/5991, z4 2322/6700): not kept. The register budget, not the load
  schedule, is what limits the main blocks now.
- The nondeterminism: tools/flake.py (60 repetitions in one process) never shows it; between PROCESSES 3 of 36
  batch-1 generations of 600 tokens differ (tools/flake_gen1.py prints prefix hashes), with the s0008 kernel and
  with x and z1 alike, at any position. It is not the routing scratch overlap (that needs batch > 4) and not L1
  staleness of the K/V caches (reading them with ld.global.cg made it worse: 11 of 36).
  Cause: a reader can see the counter move before the data is in L2. Evidence, each over 36-120 processes:
    plain stores, no fence (s0005..s0008)          9 of 120 differ     1583 / 4030
    st.global.cg stores, no fence                   9 of 120            1585 / 4028
    __threadfence in every warp that stores         0 of 120            1765 / 4250   (+11%)
    fences only at the cross-warp sites (A, E)      3 of 40             1621 / 4118
    fences only at the same-warp sites              1 of 40             1690 / 4236
    atomicExch writes, no fence                     1 of 102            1606 / 4100
    atomicExch writes + every writing warp bumps    0 of 108            1606 / 4120   (+1.5%)
  So neither "same warp" nor "L1-bypass store" orders a plain store before a later atomic; atomics of one warp are
  ordered among themselves. Early tile requests by the non-leader warps did not win the fence cost back (1765).
  New protocol (default; -DPLAIN_STORE gives stores + fences): every value another block reads is written with
  atomicExch (32/64 bit; the SwiGLU hidden units are paired into whole 32-bit words first), and every warp that
  writes bumps the counter itself (phase A counts 5 per block, expert gate/up 2).
  Outputs are bit-identical to s0008 (track A) on all four workloads of tools/flake1.py.
- Track B now also uses 2 accumulators per mma chain (engine/NACC_B): faster (1663 -> 1622) and 93.2% top-1 on my
  reference (4 accumulators: 91.1%, Hugging Face BF16: 90.2%).

## s0009 (track B) and s0010 (track A): race-free hand-overs + batch-8 changes - both valid
s0009: expected http 1750 / 4100, tokens ~94%. Measured 1772 / 4158 (1.112x / 0.913x vLLM), tokens 93.2%, in process
1633 / 4175. s0010: expected 1710 / 4250 and the gates of s0008. Measured 1780 / 4235 (1.107x / 0.896x), in process
1608 / (see result.json). Batch 8 gained 21% (track B) and 12% (track A) on HTTP; batch 1 on track A LOST 5.6%
(s0008: 1685), more than the 1.5% the atomic writes cost in my local comparison: to be located (next entry).
s0008 stays the best headline number, but it has the race; s0010 is the best race-free track-A engine.

## Why s0010 lost batch-1 speed, and the two-instance kernel
tools/serve_gaps.py (serve path, 64 + 1024, per-token gaps): no stalls; the s0009/s0010 kernel was uniformly slower
at batch 1 than s0008 (mean gap 1779 vs 1692 us, track A). Located on one card (step us B=1 / B=8, serve mean):
  s0008 kernel                                        1532 /  -    1692 / 4871
  s0010 kernel (at2)                                  1604 / 4120  1779 / 4348
  + the down-phase loop of s0008 again (y1)           1585 / 3963  1753 / 4197   the "pipelined copy" loop was a loss
  + two kernel instances, B<=2 and B>2 (y2)           1555 / 3952  1728 / 4168   each compiled without the other's paths
  y2 with plain stores, no fence (unsafe, reference)  1558 / 3877  1727 / 4066   -> the atomic writes cost 0 at B=1, 2.4% at B=8
  y2 with ld.global.cg counter polls                  1553 / 3965  1729 / 4167   -> no effect
  y2 + K/V tiles 8 per round trip into the staging registers (y3)  1561 / 3956  1738 / 4193   -> not kept
All bit-identical to s0008's outputs (tools/flake1.py). Reading: with all 128 registers in use, code for the other
batch class costs spills in the paths that run; a compile-time split removes that coupling.
Long context (fine profile at position 1000, sliding layer, B=1): the attention chain is 10.8 us against 5.2 at
position 150: scores 2.8 (0.8), weights 1.9 (0.8), V mix 2.5 (0.6). Batching the K/V loads only moves the scores
(2.8 -> 2.1): the rest is computation and shared-memory traffic of the softmax weights, not load latency.

## s0011 (track B) and s0012 (track A): two kernel instances - both valid
s0011: expected http 1725 / 3990; measured 1745 / 4152 (1.130x / 0.914x vLLM); batch 8 did not move on track B
(s0009: 4158) although it did on track A locally. s0012: expected 1730 / 4070; measured 1734 / 4079 (1.137x / 0.931x),
tokens 93.07% as s0008/s0010. s0012 is the best race-free engine at both batch sizes; s0008 (1685, with the race)
stays the board's best headline.

## Speculative expert prefetch at batch 1 (tools/mk_s1.cu.bak): built, correct, slower - not kept
Built on the two-instance kernel: the 16 router blocks run the router once more on norm2(layer input) (hi operand
only), aux 4 turns the 256 speculative logits into a top-6 list, and aux 4..23 load the predicted experts' gate/up
tiles (160 per warp, batches of 8 into T[8..15], XOR-folded so that the loads are real); router blocks stop when C
is signalled and add 7 batches after their real pass. Outputs bit-identical (tools/flake1.py).
Measured (track A, card 2): step 1555 -> 1675 us, serve mean 1728 -> 1930 us at batch 1.
Profile, per layer: expert gate/up -0.3 us only; router wait +1.1 (the router blocks answer C late and their real pass
is slower), A +0.5, attention +0.7 (memory contention with the prefetch traffic).
Why it cannot pay at batch 1 with this structure: the 6 predicted experts are 25 MB; the 4 idle SMs and the 16 router
SMs move about 3 TB/s between them when nothing else loads, but the router SMs are on the critical path (C -> router
-> routing) and the idle window overlaps the main blocks' own preloads. The main blocks have the bandwidth and 7 us of
idle time (attention) but no registers: their 16 tile registers hold the C tiles, and the list is not known before
about 6.5 us into the layer. So the prediction is good (83-95% recall) and the mechanism works, but there is no place
to put 25 MB per layer of extra loads without delaying something that is on the critical path.
Debugging note: an inline-asm `ld.global.cg` inside a polling loop was hoisted out of the loop by the compiler (the
block spun on a stale value until the watchdog fired, 6.4 s); polls must be volatile reads or carry a loop-variant
operand.

## Batch-8 instance, register-neutral changes (track A, card 2: step B=8 / serve mean B=8; all bit-identical)
  two-instance kernel (s0012)                                   3952 / 4168
  + down phase: tiles as 8, fragment copy, other 8 (b1)         3898 / 4117
  + hidden units written after the last slot by all warps (b2)  3881 / 4109
  + norm: later sequence pairs kept in shared memory (b4)       3854 / 4074   <- kept
  two barriers per down round instead of four (b3)              3940 / 4177   worse
  C tiles requested before a main block's own attention (b5)    3893 / 4134   worse
  K/V tiles 8 per round trip into the staging registers (b6)    3861 / 4102   no gain
Also tried and without effect: ld.global.cg counter polls (the first test of this had been a no-op: the switch was
not in that source), ptxas -O2 / --allow-expensive-optimizations / --disable-optimizer-constants.
A trap found on the way: the norm scratch reuses the upper half of the fragment buffer, and columns of sequences that
are not members of a unit used to be "multiplied by a zero weight"; with float32 bits lying there (track B) that gave
NaN. The leaders now select zero for non-members.
Reading: at batch 8 the expert phases run at 5.4 TB/s on average (gate/up 6.3, down 4.4); what is left outside them is
about 33 us per layer of norms, attention, routing and hand-overs. No single register-neutral change is worth more
than 1%.

## s0013 (track B) and s0014 (track A): batch > 2 instance, register-neutral changes
s0013: expected http B=8 4060; measured 1746 / 4101 (s0011: 1745 / 4152). s0014: see the board (expected 1734 / 3990).

## More batch-8 negatives, and W12 measured in the engine's regime
- Register-operand down phase inside the batch > 2 instance only (b7): 156-180 bytes of spills there, B=8 4120 us
  against 3854. The operands in registers (8 tiles + 8 operands = 48 registers, fewer than the 64 of 16 tiles) are
  not the problem by count; the allocation is. Dropped.
- tools/bench15.cu, W12 with one 512-thread block per SM and 128 KB reserved (16 tiles per warp as 12 loads + the real
  decode with per-row exponent bases): memory-bound stream 9.04 TB/s of BF16-equivalent weights against 7.70 (+17%);
  but the decode is 0.8 us of computation per 16 tiles and warp (16 blocks: 0.93 -> 1.72 us per batch), and with 0.85 us
  of other work between request and use the BF16 batch is 2.2 us and the W12 batch 2.7 us. In the engine the expert
  slots are 2.5-2.7 us for 2.2 us of memory time, so W12 would gain at most 5-10% in the gate/up phase and lose in
  the down phase. Not built.
Batch 8 stands at 0.95x of vLLM on track A (s0014: 3993 us) and 0.93x on track B (s0013: 4101 us).

## Where a layer's time goes at batch 1: per-block timestamps (tools/bt.py), and what follows from them
Profiling build with MK_BTONLY (no per-event samples, only one clock read per block and phase boundary), layer 8,
track A, medians over the 128 main blocks, us from the layer's start:
  A start 0.2 | A signalled 3.7 | C start 10.8 | C signalled 12.8 | D start 14.6 | router seen 17.2 | top-6 done 19.8 |
  gate/up slot 0 done 23.0 | slot 1 done 26.0 | E signalled 26.8-27.5 | F start 28.6-29.5 | F signalled 32.7-33.5
- The barrier itself is cheap: from the LAST block's signal to the FIRST block's start of the next phase 0.6-1.1 us.
- What looks like hand-over cost is the spread of the blocks inside a memory-bound burst: E signals spread over 1.3-1.9 us,
  F and A over about 1 us; the last block is served last, every other block waits for it. Which blocks are late is
  random from step to step (correlation of a block's mean lag between phases below 0.35).
- Atomic writes against plain stores: the same timeline within 0.3 us (so the atomics are not a tail).
- Expert gate/up at batch 1 is two bursts one after the other (16 warps x 16 tiles, then 8 warps x 16 tiles: 3.3 + 1.7 us)
  plus two leader epilogues; 6.9 us in this build against 3.3 us of pure memory time for its 25 MB.
Consequences tried:
- Counter polls with ld.global.cg: no gain in the engine (and 20 us lost to a different register allocation).
- Head with 16-tile batches into the staging registers (h1): step 1552 -> 1538 us, serve mean 1727 -> 1719. Kept.
- Look-ahead of the NEXT layer's q/k/v tiles by the idle SMs (p1, p2: no prediction needed): serve mean 1719 -> 1713,
  i.e. nothing: those tiles are requested while a block waits for the slowest block anyway.
- Speculative expert fetch by the MAIN blocks while they wait for the attention (s2..s5): the router blocks run the
  router on norm2(layer input), aux 4 publishes the top 6, every main block loads the 384 gate/up tiles it would read
  (24 per warp, 4 per round trip into scratch registers). In the engine 85.6% of the real experts are in the list
  (48252 of 56400). For a block whose tiles were fetched the first gate/up burst takes 1.7 us instead of 3.3 and the
  phase 3.9-5.2 us instead of 6.9. But the step is not faster (1568 us against 1544; serve mean 1707 against 1719):
  the phases become uneven, the blocks that are done early request their NEXT phase's tiles while they wait, those
  requests are ahead of the late blocks' current requests in the memory's queue, and a set of blocks (always the
  same: 73-75, 88, 94-99, 116-118, 122-127 in these runs) ends 8-15 us behind in every phase and every barrier waits
  for it. Making the fetch opportunistic (skip or stop when the attention is done) and starting it only after every
  block has signalled A did not break that cycle. Left in tools/mk_s5.cu.bak.
  What would be needed: the down-phase tiles of the predicted experts fetched too (another 17 MB in the same window)
  and next-phase requests issued only after the barrier, so that no block can be overtaken; not attempted.
Two traps: (1) a value loaded only to fill L2 must be used by every lane, or the compiler keeps the load for one lane
(a quarter of the tile); (2) an inline-asm load in a polling loop needs a loop-variant operand or it is hoisted.

## s0015 (track B) and s0016 (track A): head in 16-tile batches
s0015: expected http B=1 1738; measured 1733 / 4100 (s0013: 1746 / 4101). s0016: expected 1725 / 3993; see the board.

## Speculative expert fetch: closed, with an oracle measurement
More variants (s6: down tiles fetched too and down-phase requests after the barrier; s7: C tiles requested before the A
signal; s8: fetch as two full batches into the staging registers before the C tiles, the last quarter of the blocks to
signal A skips it): step 1632, 1633, 1587 us against 1544 for the kernel without speculation.
Two controls decided it:
- s8 with the fetch loads switched off (-DSPEC_NOFETCH; speculative router pass, list and the reordered C request still
  there): the same group of late blocks appears (E signals spread over 11 us) and the "typical" block's gate/up phase is
  3.9 us against 6.9 in the baseline. So neither the late group nor the faster typical block came from the fetch: once
  the blocks are split into an early and a late group, each group's burst is smaller and served faster, but every
  barrier waits for the late group.
- Oracle (tools/mk_oracle.cu.bak, -DORACLE_FETCH): every block first reads all 384 gate/up tiles of the REAL units
  (3.8 us), then runs the phase. With every tile in L2 the phase takes 6.1 us instead of 6.9-7.7: the first burst
  drops from 3.3 to 1.9 us, the rest (second slot, two leader epilogues, signal) does not move.
So at batch 1 even a perfect prediction is worth about 1 us of a 31 us layer. The prediction itself is good (85.6% in
the engine) and belongs in the findings; the mechanism is not built into the engine.

## More measurements before the next step
- tools/bench16.cu, publishing results to other SMs (one writer warp, 16 reader blocks, per round over the bare counter
  round of 0.377 us): plain stores +0.00 (but not ordered before the counter), plain + __threadfence +0.36 us,
  atomicExch 64-bit x 2 per lane +0.01..0.04 us, atomicExch 32-bit x 4 per lane +0.20..0.28 us. The engine's epilogues
  issue at most 2-3 atomic instructions per warp, so the atomic protocol costs almost nothing per hand-over.
- Rolling tile requests in the expert gate/up phase (e1: the next piece's first 8 tiles are requested as soon as the
  first 8 are used): bit-identical, B=1 1544 -> 1541 us, B=8 3854 -> 3863 us: nothing. Not kept.
- tools/early_route_study.py (HF BF16, 4 x 256 tokens, 9 layers = 9216 layer-tokens): the top-6 set of
  z = R' x + (R' W_o) a  (R' = R diag(w2), x = layer input, a = attention output before o_proj) equals the top-6 of the
  model's own formula evaluated in float64 from the same x and a in 9216 of 9216 cases; evaluated in float32 from
  3 BF16 limbs of R' and M it still equals it in 9216 of 9216; it equals what the BF16 model picked in 98.84%
  (the rest are near-ties of this one layer's rounding).

## Early routing, stage A (track B, tools/mk_er1.cu.bak): built, slower at batch 1, parked behind a switch
Router blocks compute z0 = R' x at the layer's start (x as 3 BF16 limbs, R' as 2) and z = z0 + M a right after the
attention (M = R' W_o as 4 BF16 limbs, built at load time in float64); the main blocks scale z by the norm's 1/rms.
Numerically fine (local gates as before, generate identical 8 of 8), step 1635 us at batch 1 against 1577 for the
s0015 kernel, 3929 against 3961 at batch 8. Why slower: the second pass has to read 8 MB of M per layer on 16 SMs
(about 1 TB/s for 16 of 128 SMs), 6 MB of it after the attention has ended: later than the classic router (1 MB after
C). Two more reasons not to pursue it now: (1) M is a product, so 4 limbs are an approximation of 3e-11 relative
(exact M needs ~8 limbs and exact integer arithmetic at load time): not "identical in exact arithmetic" as track B
demands; (2) stage B (selection off the main blocks) would be needed before it pays at all.
State: the code is in engine/mk.cu under -DEARLY_ROUTE (build: MK_EARLY=1 or a file engine/EARLY_ROUTE, track B only),
off by default, kept working with the new hand-over protocol (flake 0 of 79).

## The atomic hand-over was not strict; the sentinel protocol (s0017/s0018)
How it was found: the early-routing build differed between runs in 6% of 600-token generations at batch 1
(tools/flake.py; the s0015 kernel: 1 of 357 repetitions, so the hole was there all the time, 25x rarer).
tools/zdump.py (per step and layer: hash of the layer input, router data, the attention output as a router block read
it; 400 steps in each of 14 processes per card) showed: router pass 1 identical, the ATTENTION OUTPUT differs, e.g.
one head of 32 wrong in all 128 dims with a 15x larger norm: its softmax saw a stale q. In another case the expert MLP
differed with identical routing. So hand-over data in general was read stale, rarely.
Tests on the early-routing build (79 repetitions each, 5-6 mismatches expected):
  fences added to the atomic writes at every hand-over: 0;  only phase A: 10;  A + attention: 4;  F: 16;  C + F: 9;
  router: 13;  the writing warp waits for the results of its exchanges before it bumps ("settle"): 0 of 237.
Conclusion: an atomicExch whose result nobody uses does not hold the warp up, and the atomicAdd that follows can
reach L2 first. (tools/bench17.cu does not reproduce it in isolation: 0 stale of 4.4e9 words even with plain stores;
it needs the engine's traffic.) Cost of settling at the end of every phase: batch 1 1643 -> 1765 us (+7%), batch 8
+5%: one more trip through the memory queue on the path every other block waits on (bench17: +0.3 us per hand-over
idle, and the hand-over as a whole goes from 1.8 to 4.1 us when every warp has 16 tile loads in flight: the memory
queue is one FIFO, as the North run of this project found on the H100).
The sentinel protocol (what the engine does now, for x, q/k/v, the attention output, the hidden units and the router
logits):
  1. At the START of its phase, when every reader of the old content is known to be done, the writer exchanges a
     sentinel (0x7fc17fc1: a NaN with a payload in float32 and in BF16, never a result of the model's arithmetic) into
     the words it is going to write, and reads the exchanges' results back (free: its first loads are in flight).
  2. Right then it bumps the counter: the signal only says "the sentinels are in place".
  3. At the end of the phase it exchanges the data in. No wait, no second signal.
  4. A reader waits for the counter (long there), loads, and loads a word again while it holds the sentinel.
Why it is strict: a reader that has seen the counter can find in a word only the sentinel or the new data, never the
old content; "all readers of the old content are done" was checked hand-over by hand-over (journal of the session:
(a)-(j)); the router blocks got one more wait for that (all main blocks have begun the previous down phase).
Measured locally (step us, batch 1 / batch 8; all variants 0 mismatches):
  racy s0015/s0016 kernel            track B 1577 / 3961    track A 1538 / 3854
  settle at the end of every phase   track A 1660 / 4055
  sentinel, signal at the phase end  track B 1722 / 4300    track A 1684 / 4154
  sentinel, signal at the start (x, q/k/v, attention output)   track B 1644 / 4229   track A 1613 / 4095
  + hidden units and router logits the same way                track B 1595 / 4106   track A 1568 / 4036
So strictness costs 1-2% at batch 1 and 4-5% at batch 8 (the sentinel doubles the number of atomic writes; the
hidden units at batch 8 are 12k 32-bit exchanges per layer). 0 mismatches in 79 x 3 + 149 repetitions.

### Results and the second round (s0017 - s0019)
s0017 (track B, sentinel + signal at the phase start, exchange result read back at once): expected http 1755 / 4250,
measured 1786 / 4298 (s0015: 1733 / 4100). Worse than expected at batch 1, most at long contexts.
Decomposition on one machine (track B, step us batch 1 / batch 8, and the 64+1024 workload through generate):
  racy s0015 kernel                          1564 / 3955     1747 / 4075
  s0017                                      1596 / 4105     1797 / 4261
  s0017 without reading the results back     1550 / 4042     1735 / 4185   (not strict; shows what the read-back costs)
  signal after the data + sentinel           1750 / 4258     1953 / 4456   (the three trips through the queue again)
So the protocol itself is faster than the racy one at batch 1; the cost was the wait for the sentinel exchange's
result at the start of each phase. Tried:
- results kept in registers across the wait that precedes the phase ("issue in slack, settle later"): 1621 / 4382,
  slower: the carried values spill (192-byte frame, 148 bytes of spill stores) - the register budget again.
- exchange issued a phase ahead, result dropped, confirmed by a read-back LOAD issued with the phase's first loads:
  1577 / 4052; plus tile requests before the expert phase's sentinel exchanges: 1562 / 4041 (1744 / 4219 on the
  workload). This is s0018. It needs x_mid in its own buffer (P->xm): with one buffer the sentinel for the next
  writer could not be written before the previous readers were done, i.e. not ahead.
- a pause between two re-loads of a sentinel word (to thin out the polling): 1912 and 3223 us with 12 and 48 pause
  loads: latency of the poll is what counts, not its traffic. Dropped.
s0018 (track B): expected http 1735 / 4255, measured 1733 / 4268. Strict, and as fast as s0015 at batch 1.
s0019 (track A build): expected 1757 / 4200, measured 1783 / 4179 (s0016: 1724 / 3989). Track A lost more at batch 1
than track B (local 1539 -> 1572 against 1564 -> 1562); not understood yet (same source, different register picture).
Local determinism: 0 mismatches in 297 repetitions of tools/flake.py (97-step, 600-token and batch-8 runs each).

## A sequence was not bitwise the same alone and in a batch of 7 or 8 (s0017-s0019, track B): FMA contraction
Found while testing a new down phase: tools/bcheck.py (every batch size 2..8 against the single-sequence runs, every
sequence, bitwise) passes for batches up to 6 and fails for 7 and 8, sequences 6 and 7 only, from step 5-7 on, with
small differences (logits 0.08-0.16). My own check (tools/test_engine.py) compared only sequences 0 and 1 alone
against the batch; the evaluator's batch gate reported bitwise_equal = true for the three submissions (the sequence it
checks was not one of the two). The s0015 kernel passes bcheck, the s0018 kernel built for track A passes.
Localized with debug dumps (tools/bdbg.py, layer inputs and, with -DDBG_MID_LAYER / -DDBG_ATT_LAYER, the
post-attention stream and values inside the attention): raw q/k/v identical, scores of the first key group
identical, the V mix of ONE head of the pair (sequence 6, K/V head 2) differs = its softmax weights differ by one
BF16 rounding step. That pair runs on a main block at batch 7-8 (24 attention blocks, 28-32 pairs).
Cause: attention() is inlined twice (attention blocks; main blocks) and the compiler contracts a*b + c*d into an FMA
differently in the two copies. On track B the rotary step v*cos + v'*sin has no BF16 rounding between the products
and the sum (track A rounds each product), so the two copies differ by an ulp, and now and then a weight rounds the
other way. It had been equal by luck until the code around the attention changed (s0017).
Not the cause (tested): hand-over (the q rows a pair reads are identical); L1 staleness of the K/V cache when a
block changes SM between launches (loads past L1: same failure, and they cost 14 us at long contexts: not kept).
Fix: every multiply-add of attention() is written with __fmaf_rn / __fmul_rn / __fadd_rn (five expression classes).
Which fusion each class had on the attention blocks was determined with tools/digest1.py (batch-1 outputs only):
rotary and sums of squares = left product fused (the other two choices change the digest), the others are not
visible in the digest; all pinned to "left product fused". Digest unchanged on both tracks (so batch-1 results and
everything the evaluator measured before are the same), bcheck: every batch size bitwise equal over 128 steps.
Lesson for the findings: "same summation tree" is not enough for bitwise batch invariance when the same source is
compiled in two places; the multiply-adds have to be explicit.

## A new down phase for batch > 2: built, correct, not faster (tools/mk_downstream.cu.bak)
Profile of the batch-8 step (tools/prof.py --B 8, per layer): expert gate/up 25 us (154 MB), down phase 22.6 us for
81 MB, wait for the attention 11-12 us, norms 5.7 + 4.8, routing 5.3-6.9, C 4.4. The down phase ran in 5 rounds of
8 items (copy of the hidden units, 16 tiles per warp, three barriers per round).
Built: (1) one compact copy of the hidden units of every (item, member sequence) into shared memory (64 slots of
1 KB; tables from the routing), (2) every warp streams half-items (16 tiles) without barriers, partial sums to
shared memory, (3) item sums and routing weights in parallel, then warp 0 adds each sequence's six units in
ascending order (bitwise the batch-1 order; verified with bcheck).
Measured: the stream itself takes 15.3 us (5.3 TB/s: the same rate as the gate/up phase and the head; with 8 or 16
tiles per request alike), copy 2.5 us, sums 1-2 us: the phase 22.6 -> about 20 us, but the step did not get faster:
4203 us against 4041 (track B), 4251 against 3925 (track A); the tables and the missing tile preload cost in the
expert phase what the down phase gains. Not kept. Side results: storing the down matrices block-major (a block's
16 KB slices of all experts next to each other) changes nothing (no TLB effect, as tools/bench9 said);
a single warp adding 40 items one after the other takes 6.6 us (a sequential loop in one warp is slow);
4 loads in flight per thread in the copy made the whole kernel spill (240-byte frame) and every phase 15% slower.
s0020 (track B, attention multiply-adds explicit): expected http 1733 / 4268, measured 1733 / 4233; batch gate bitwise_equal true (as before: the gate's sequence was never one of the two affected).
s0021 (track B, router top-6 by ranks, batch 1-2): expected http 1717 / 4233, measured 1715 / 4231. Local A/B (control = -DROUTE_RANK=0): step 1549.9 -> 1533.1, workload 1734 -> 1720.
s0022 (track B, first slot's expert epilogues by warps 8 and 9 at batch 1-2): expected http 1700 / 4231, measured 1701 / 4224. Local A/B (control = -DE_LEADERS_FIXED): step 1535.6 -> 1519.9, workload 1719 -> 1705.
s0023 (track B, shared-expert epilogue moved, K/V cache stores deferred): expected http 1690 / 4205, measured 1699 / 4221 (in process 1508 as locally; over HTTP at long contexts only 2 us of the 14 remain).
s0024 (track A build of the s0023 kernel): expected http 1735 / 4170, measured 1732 / 4111 (s0019: 1783 / 4179).
s0025 (track B, K/V cache tiles 8 per request in the batch 1-2 kernel): expected http 1662 / 4221, measured 1663 / 4222: the board's best, and the first strict engine below the racy s0008 (1685). Local A/B on one card: workload 1691 -> 1656 (track B), 1716 -> 1662 (track A); in the batch > 2 kernel the same change cost 27 us per step at batch 8 (track B) and 15 (track A): not applied there. Two batches in flight (second one in the V half of the staging registers) and a look-ahead for K were slower at every context (1507 -> 1547): not kept.
s0026 (track A build of the s0025 kernel): expected http 1680 / 4111, measured 1679 / 4116.

## Long contexts, and two things that are not the limit (after s0026)
- First batch of extra V levels requested right after the scores instead of at the mix (KV_VEARLY): workload
  1655 -> 1637 us (track B), 1661 -> 1657 (track A); bit-identical. Kept (s0027).
- Batched K/V loads in the batch > 2 kernel, separately for the pairs on main blocks (-DKV_BATCH_MAIN=1: step
  4070 -> 4054 at short contexts, workload 4194 -> 4212) and for its attention blocks (-DKV_BATCH_AUXBIG=1:
  4073, 4194): nothing either way. The 11 us the batch-8 main blocks wait for the attention are not load latency.
- tools/bench18.cu: is a stream slower when its bytes sit in a few scattered 4 MB units (the experts of a step) than
  when they are contiguous, or would a striped storage help? Same slots and barriers as the engine's expert phase,
  128 blocks, 80 GB buffer, new random units every iteration:
      6 units (25 MB):   contiguous 6.41 TB/s, 6 units of 4 MB 6.42, striped by row group 6.44, striped by tile 5.96
      38 units (159 MB): contiguous 7.30 TB/s, 38 units 7.32, striped by row group 7.45, striped by tile 7.11
  No effect of where the experts lie (and no reason to store them differently). It also says the expert phase's
  stream is not what is slow: 38 units take 21.8 us here against 25-27 us for the engine's whole phase at batch 8
  (routing tables, sentinels, 16 warps writing the hidden units at the end), and 6 units 3.9 us against about
  5 us at batch 1.
- The head on the 128 main blocks only (-DHEAD_MAIN) costs 200 us per step in the normal build: not a clean
  measurement of "does a 129th block add throughput" (the attention blocks then stage during the head), not pursued.
s0027 (track B, first extra V batch requested after the scores): expected http 1647 / 4222, measured 1646 / 4227 (board best).
s0028 (track A build of the s0027 kernel): expected http 1673 / 4116, measured 1669 / 4118.
- Pairs on main blocks (batch 7-8) with their first 16 key groups staged in the block's registers before the attention (-DMATTN_STAGE=1): step 4068 -> 4137 us, with batched loads 4253: slower (the q loads of the attention then wait behind the 16 tile requests of the same warp). Off.

## Could the experts be known (or guessed) earlier? Certification study (tools/cert_route_study.py)
z = R' x + M a (M = R' W_o) decides the top-6 set (journal above). With M cut to ONE BF16 limb (2 MB per layer: fits
the registers of the 16 router SMs) the top-6 of z equals the exact one in 99.83% of 11520 layer-tokens (HF BF16
activations, 9 layers); with two limbs in 100%. Certified beyond doubt (weakest of the six minus its error bound
above the strongest other plus its bound, plus a float32 budget of 4e-6 of the absolute sums): 1 limb 79.1% with the
exact bound and 55.6% with a bound an engine could compute from the same tiles; 2 limbs 99.70% / 99.62%.
A rigorous certificate would need worst-case float32 bounds of both ways of computing (the classic one accumulates
x + W_o a first: its error scales with |R'| |W_o| |a|, far larger than |M| |a|), i.e. more matrices and passes: not
attempted. What the numbers do allow: SPECULATION. The predicted list is right 99.8% of the time, so the main blocks
can request the expert tiles from it and confirm against the real router while the tiles are on their way; a wrong
guess costs one batch of tiles. No change of the function, valid on both tracks.

## Speculative routing from an early linear predictor: built up to the measurement, does not pay (tools/mk_spec.cu.bak)
Idea (from the certification study): the router blocks compute z = R' x + M a with one BF16 limb of M = R' W_o right
after the attention (R' x during phase A), attention block 0 ("lister") turns z into the unit list exactly as a main
block does from the real logits, and the main blocks request their first expert tiles from that list while the real
router is still running; the real router decides (valid on both tracks). Built as a switch (kernel
tools/mk_spec.cu.bak with MK_SPEC=1; engine/api.py and build.sh keep the packing and the flag).
Measured in the engine (tools/spec_agree.py, 200 steps = 9400 layers; entries of the predicted list that differ from
the real one, a wrong layer has 1-6 of them at batch 1 and up to 48 at batch 8):
    track B: batch 1  66 entries (under 0.7% of the layers), batch 8 2224
    track A: batch 1 349 entries (1-2% of the layers: the real router sees x_mid rounded to BF16), batch 8 11083
So the prediction is as good in the engine as in the study. What kills it is time on the router SMs (track B, batch 1,
step us, baseline 1500; -DSPEC_NOWAIT = the main blocks do not use or wait for the list):
    router tiles no longer preloaded in the staging registers (they hold R' and M now)     1659
    + pass 1 (R' x during phase A)                                                         1643   (batch 8: 4662, base 4071)
    + pass 2 (M a after the attention, publish z)                                          1798
    + lister                                                                               1899
Reading: at batch 1 the 16 router SMs ARE the critical path between the attention and the expert phase (the real
logits arrive when the main blocks have just finished D); their 1 MB of router tiles sitting in registers is worth
159 us per step, and everything the predictor makes them do before the real router delays the real router. And the
list would arrive too late anyway: predictor 1.5 us after the attention + lister 3 us is as long as the main blocks'
C + D (4.7 us), so at most the routing itself (1.8 us) could be saved, on a timeline with no slack.
Two more negatives from this build: (1) making the blocks that wait for the attention output wait for a hint counter
with one thread instead of re-loading sentinel words with all 512 (-DATTN_HINT): slower, 1502 -> 1529 us: the
re-loading is not what slows anything, the extra trip through the queue is; (2) the predictor passes waiting for
complete inputs instead of re-loading (the "quiet" variant): no better.
The default build of that source was bit-identical but 25 us slower at batch 8 (code that is compiled in but never
runs still changes the register picture of the batch > 2 kernel), so engine/mk.cu is the s0027 source again.
s0029 (track B, batch > 2: 12 routing candidates merged by ranks): expected http 1646 / 4190, measured 1646 / 4188. Local A/B: step 4074 -> 4042 (track B), 3883 -> 3851 (track A). End-of-phase-C ordering (-DC_ORDER=1: x_mid exchanges before the D preload; =2: D preload before the epilogue): +6 / +11 us on track B, +6 / -14 on track A: noise of the register picture, left at 0.
s0030 (track A build of the s0029 kernel): expected http 1669 / 4095, measured 1671 / 4056.
s0031 (track B, batch > 2: router tiles preloaded on the router blocks, their K/V staging after the router duty): expected http 1646 / 4150, measured 1646 / 4123. Local A/B: step 4042 -> 4010 (track B), 3851 -> 3819 (track A). After it the router is no longer what the batch-8 main blocks wait for (profile: router duty 6.5 us after C, main blocks need the logits 8.1 us after C).
s0032 (track A build of the s0031 kernel): expected http 1671 / 4035, measured 1671 / 4085 (s0030: 4056; s0028: 4118): the batch-8 HTTP figure moves by about 30 us between evaluations of neighbouring kernels.
- The two dense layers load their tiles inside loops (48 and 24 per warp, one per iteration). Requesting them in
  batches of 16 / 8 into the staging registers (bit-identical): batch 1 1500.8 -> 1497.4 us (track B), 1516.5 -> 1511.1
  (track A); batch 8 -15 / +7: within the noise, the unrolled loops were not serial. Not kept.
- Robustness of the s0031 kernel: tools/digest_long.py (1100 teacher-forced steps, batch 1 and 8) identical in 5 processes on each track; tools/flake.py 0 mismatches in 179 repetitions on each track.
- Batch > 2 down phase once more, inside the existing rounds: the next round's hidden units copied right after this
  round's mma and before its tiles are requested, all 16 tiles in one request (-DFCOPY_EARLY, bit-identical): step
  4014 -> 4147 us (track B), 3819 -> 3978 (track A). Slower: the copy's loads sit in the memory queue behind the other
  blocks' tiles and hold the tile request back. The order "8 tiles, folds, copy behind them, 8 more" stays.
  (A note on reading tools/prof.py: the fine profile adds about 0.3 us per marker; the old down phase has 28 markers
  per layer at batch 8 and reads 34 us there against 22.6 in the coarse profile. Compare phases only within one mode.)
- Pairs on main blocks (batch 7-8), third attempt at their tail: the block's 16 output-projection tiles requested
  inside the attention, right behind its q/k/v loads (-DMATTN_CPRE=1), so that they are there when the attention ends
  (the profile shows 3.4 us between the end of such a block's attention and its phase C): step 3991 -> 4071 us
  (track B), 3819 -> 3879 (track A). Slower: this path loads its K and V tiles one by one where it uses them, and
  each of those loads now waits behind the 16 tiles. With batching (KV_BATCH_MAIN) the registers are taken. Off.
  Kept from this edit: the attention issues its k/v-row loads together with the q-row loads instead of after the q
  rows were used (bit-identical; batch 8 4008 -> 3991 us on track B, batch 1 1516 -> 1507 on track A, else unchanged).
s0033 (track B, attention issues the k/v-row loads with the q-row loads): expected http 1646 / 4105, measured 1652 / 4120: no change beyond the noise of the evaluation (s0031: 1646 / 4123).
s0034 (track A build of the s0033 kernel): expected http 1663 / 4085, measured 1671 / 4097 (s0032: 1671 / 4085): no change. Both tracks are at the same source (tools/mk_s33.cu.bak); the noise floor of the HTTP figures (about 6 us at batch 1, 30 at batch 8) is reached for changes of this size.

## Robustness of the contract paths (after s0034)
- tools/api_stress.py, both tracks: engines (1, 1152), (8, 1152), (4, 2048), (8, 4096); batches 1, 2, 3, 4, 8;
  generate() and submit()/poll() against step-by-step greedy (12-token prompt, 40 tokens); two cohorts of different
  shapes back to back and a request that does not fit: 38 checks, all ok.
- Found by reading, not by a failing test: the attention keeps 128 key groups at most (the weights area in shared
  memory, 8 groups per warp), i.e. 2048 positions, and Engine() accepted any max_context: a full-attention layer
  beyond position 2048 would have written past that area. engine/api.py now caps max_context at 2048 (a longer
  request is refused like any request that does not fit). tools/long_ctx.py: generate() equals step-by-step greedy
  to position 2040 at batch 1 and 2, logits finite. The evaluator's and the baselines' workloads end at 1152.
- The kernel's watchdog was invisible to the host: every wait gives up after a few seconds of spinning and sets a flag,
  the launch then runs through with whatever is there, and the host cleared the flag before the next launch without
  ever reading it. It has never fired outside of bugs during development, but a silent wrong answer is the worst way
  to fail. Now: the flag is not cleared by a launch; poll() reads it with the progress word (same copy) and raises;
  reset() and stats() read it (they wait for the device anyway; no read inside a decode step, so the "GPU never
  waits" property is untouched); on an event the host takes the counters' values from the device, so the engine works
  again after reset(). Test with an injected unreachable wait (MK_EXTRA=-DWD_TEST, tools/wd_test.py): the launch
  returns after 0.8 s, reset() raises, poll() raises, afterwards generate() equals step-by-step again.
s0035 (track B, watchdog reported to the host, context cap): expected http 1652 / 4120 unchanged, measured 1650 / 4117; milestones unchanged (gpu_never_waits kept).
s0036 (track A build of the s0035 engine): expected http 1671 / 4097 unchanged, measured 1672 / 4099.
- Whole sliding window staged (K groups 28..33 in the partial-sum and norm-weight areas of the attention blocks, the
  17th V level in the upper part of the weights area; -DWSTAGE, bit-identical, tools/long_ctx.py ok): batch 1 step
  1502 -> 1517 us (track B), 1510 -> 1526 (track A); 64+1024 workload 1644 -> 1645 and 1652 -> 1683. No gain where it
  was meant to help and a loss at short contexts. Not kept: what the batched requests left (one trip for 5 key groups
  and one tile) is not what the long-context attention spends its time on; that is the softmax and the mix over twice
  as many keys.

## The token gather of generate/serve (s0037/s0038)
generate() was 21-26 us per token slower than step() at batch 1 although it does not write the logits. Reading the
code after the head: block 0 collects the 152 blocks' (best value, index) pairs with one thread per sequence in a
loop of 151 iterations x 2 loads, one trip to memory each: 302 serial trips per token. That is on the path the HTTP
score measures (submit/poll), and no profile of mine had covered it (tools/prof.py runs step()).
- All threads of block 0, up to 3 pairs each in one trip, lanes and warps combined (largest value, lowest index on
  a tie: order-free, so the same token), inlined: correct, but the step got 118 us slower at batch 1 and 340 at batch
  8, in the step path too, which never runs this code: the register picture.
- The same as a function of its own (not inlined): batch 1 fine, batch 8 100 us slower on track A (192-byte frame).
- The function in the batch 1-2 kernel only, the old loop in the batch > 2 kernel: batch 8 unchanged; batch 1 (same
  card, us per token): track B generate 1528 -> 1487, workload 1644 -> 1606, serve 1530 -> 1486 (step 1505 -> 1486);
  track A generate 1526 -> 1477, workload 1645 -> 1593, serve 1526 -> 1479 (step 1499 -> 1472).
Digests, bcheck, api_stress, long_ctx unchanged/ok, flake 0 of 69 on both tracks.
(The machine was replaced twice during this test: 20:50 - 23:25 without a machine, clock stopped.)
s0037 (track B, token gather in one trip, batch 1-2 kernel): expected http 1612 / 4117, measured 1606 / 4116 (board best; 1.227x vLLM at batch 1).
s0038 (track A build of the s0037 kernel): expected http 1622 / 4099, measured 1606 / 4094 (the board's best; both tracks at 1606 at batch 1 = 1.227x vLLM).

## Token gather of the batch > 2 kernel (s0039 / s0040)

The batch > 2 kernel had kept the serial gather (one thread per sequence, 151 x 2 loads in a row) because the
function of s0037 slowed that instance (track A +100 us). Third form: 16 threads per sequence, each walks every 16th
block (10 pairs in a row), results through shared memory, block 0's first B threads combine the 16; written inline,
no shuffles, so that nothing new is live across it (switch GATHER_BIG, default 1). Spill lines of both instances
unchanged. Same card, against -DGATHER_BIG=0:
  track B batch 8: generate 3954 -> 3926, 64+1024 workload 4082 -> 4061, serve 3956 -> 3928; batch 1 unchanged
  track A batch 8: generate 3889 -> 3867, workload 4074 -> 4052, serve 3895 -> 3865; batch 1 unchanged
digest_long equal to the control on both tracks, bcheck bitwise at every batch size, flake 0 of 39, api_stress ok.
s0039 (track B): expected http 1606 / about 4090, measured 1606 / 4088.
s0040 (track A): expected http 1606 / about 4070, measured 1608 / 4028 (better than predicted by more than the
local difference; evaluations of one kernel differ by about 30 us at batch 8).
Conclusion: the serial gather was worth 20-30 us per step at batch 8 (10 dependent-free loads in a row cost about as
much as the trip of the barrier; the 151 in a row cost 25 us more). What remains after the head at batch 8 is the
head itself.

## Why the step grows with the position (s0041 / s0042)

tools/pos_time.py (generate in chunks of 128 tokens): batch 1 1467 us per token at positions 0-128, 1717 at
1024-1152; batch 8 3958 -> 4332 (track B; track A 3838 -> 4239). The scored workload runs from position 64 to 1088,
so this is 140 us per token at batch 1 and about 250 at batch 8.
Profile (coarse, sliding layers): batch 1 attention block 9.1 -> 18.5 us per layer between position 150 and 1000;
batch 8: the only interval that grows is the wait of the main blocks for the attention (10.4 -> 15.5 us), and it is
the 8 pairs on main blocks that are late (the attention blocks are done 10 us earlier). Fine profile of those pairs
(position 150 -> 1000): scores 1.10 -> 2.48, weights stage 1.04 -> 2.49, V mix 2.47 -> 2.48 (!), stage 2 1.52 -> 2.87.
1. Hypothesis: the scores of a thread's 2nd.. key groups live on the stack (ex[7][4], 112 bytes of local memory per
   thread, evicted from L1 by the weight stream). Moved groups 16-47 to shared memory (EX_SM): same digests, no gain
   (batch 1 +14 us at long contexts, batch 8 equal). Not memory. Not kept (tools/mk_exsm_rcp.cu.bak).
2. Hypothesis: the divisions e / sum (4 per key group and thread). Timing-only test with w = e * (1/sum) (different
   bits): 64+1024 workload track B 1603 -> 1564 / 4061 -> 4044, track A 1590 -> 1533 / 4047 -> 3913. It is the
   division.
3. Exact form (divr: q = e*r, q + fma(-q, sum, e)*r): digests identical on both tracks, workload track B 1604 -> 1578
   / 4064 -> 4012, track A 1591 -> 1541 / 4048 -> 3969. tools/bench19.cu: 0 mismatches against the division in 6e10
   quotients down to e = 2^-100 (mismatches only where quotients are subnormal); 38.5 -> 12.2 ns per quotient and
   warp. A guard for e < 2^-100 (real division there) keeps even those bits but the branch made the batch > 2 kernel
   of track A spill (68/92 bytes instead of 56/40; workload 4021 against 3971), decided on the results or on the
   inputs: off (SOFTMAX_GUARD).
bcheck bitwise at every batch size, flake 0 of 39, api_stress and long_ctx ok on both tracks.
s0041 (track B): expected http 1578 / 4040, measured 1560 / 4055.
s0042 (track A): expected http 1558 / 3950, measured 1548 / 3928 (1.273x vLLM at batch 1, 0.966x at batch 8).
Conclusion: on an SM that runs 16 warps the instruction count of the attention is on the critical path; the K/V
traffic is not the only thing that grows with the context.

## The 8 pairs on main blocks at batch 7-8 (s0043 / s0044)

After s0042 the batch-8 profile still shows the main blocks waiting 10-15 us per layer for the 8 pairs that run on
main blocks 0..7. Switches tried, each against its control on the same card (64+1024 workload, generate, batch 8):
- MATTN_CPOST (output-projection tiles requested behind the attention's last stores, in front of fence and barrier):
  track B 4015 -> 4224, track A 3978 -> 4193. Removed.
- MATTN_SPRE (sentinel of the post-attention stream before the attention): 4019 -> 4016 / 3972 -> 3984. Removed.
- MATTN_EPOLL (phase A's counter asked by warp 5 while the leaders store; the barrier after the leaders passes it
  on; thread 0's wait_ctr in front of the attention dropped): 4013 -> 3999 / 3974 -> 3968, time test 3908 -> 3887 /
  3838 -> 3832. Kept.
- MATTN_KPRE=1 (first K group requested with q/k/v): 4002 -> 4020 (noise); =2 (and 8 V levels): 3997 -> 4094 /
  3963 -> 4070, although positions below 256 gain 45 us (pos_time: 4020 -> 3975). Removed.
- CD_EPOLL (the C -> D counter asked by warp 1 in front of its tile requests): 4001 -> 4007 / 3966 -> 3971, batch 1
  equal. Removed.
bcheck, digest_long (equal to s0041/s0042), flake 0 of 59, api_stress, wd_test ok on the kept kernel.
s0043 (track B, MATTN_EPOLL): expected http 1560 / about 4040, measured 1558 / 4039.
s0044 (track A): expected http 1548 / about 3920, measured 1546 / 3920.

## The norm at batch > 2 (s0045 / s0046), and three ideas that gave nothing

- D_SM (batch 1-2: phase D's 8 tiles per warp staged in the 64 KB of shared memory that are free at these batch
  sizes, during the wait for the attention; D then reads them from shared memory): same digests; track B 1473.3 ->
  1473.4, track A 1442.1 -> 1451.6 (workload 1541 -> 1554). D's tile request was never on the critical path (it
  overlaps the hand-over of the stream) and tiles from shared memory are slower operands than registers. Removed.
- Norm loads in one trip (NORM_BATCH; four versions, tools/mk_normbatch_v1..v5.cu.bak):
  v1/v2 arrays for the pairs in flight: 500-830 bytes of stack, batch 8 5300-5600 us. v3 with an inline retry loop
  instead of four ldh calls: 11756 bytes of spills, 8500-14000 us. v4 landing in T[8..15] (free during phase D), all
  pairs retried together: no spills, but 4079 / 4014 us against 3906 / 3783. v5 retry on the first pair only, then the
  others together: 4082 / 4078. Fine profile (track A, position 150, block 0): D start -> norm done 4.41 -> 4.01,
  norm done -> mma done 0.45 -> 0.97. The four trips were not what the norm waits for. Not kept.
- On the way: v1 with NB=1 gave another digest at batch 8 on track B (deterministic; bcheck: sequences 2 and 6 differ
  from the same sequence alone): the compiler fused the sum of squares differently in the new copy. Findings 13e (c).
  ss4 with the 12 possible forms: track A gives the reference digests with every form (exact squares); track B with
  none (each inlined copy has its own form today). The pinned form is not submitted (it changes track B's bits).
- NORM_WS (the per-sequence scale once per warp): digests identical, bcheck 7 of 7, flake 0 of 49, api_stress and
  long_ctx ok. Track B: step 3905 -> 3880, generate 3885 -> 3862, workload 4003 -> 3970 at batch 8. Track A: 3782 ->
  3780, 3966 -> 3964.
s0045 (track B): expected http 1558 / about 4010, measured 1559 / 4021.
s0046 (track A): expected http 1546 / 3920 unchanged, measured 1546 / 3912.

## Attention at long contexts after s0046: four more organisations, two benchmarks, nothing kept

Fine profile of an attention block (batch 1, track A, position 150 -> 1000, us): stage 2 2.07 -> 2.40, scores 1.19 ->
3.04, exponentials 0.65 -> 1.18, weights 0.74 -> 1.04, V mix 0.47 -> 1.45; the rest equal. All four experiments give
the reference digests (they change who computes, not what):
- KV_SM2 (tools/mk_kvsm2.cu.bak): batch 1 per 128 positions from 0: track B 1450 1463 1501 1532 1576 1588 1611 1622
  1643 against 1448 1459 1494 1519 1591 1599 1622 1630 1654; workload 1562 against 1570. Track A 1541 against 1538.
- ATT_ACC (tools/mk_attacc.cu.bak): track B batch 1 1439 .. 1700 against 1448 .. 1655, workload 1589 against 1571;
  batch 8 workload 4152 against 3972. Track A worse everywhere (spill lines 64/88 and 48/32).
- KSWAP (tools/mk_kswap.cu.bak): digests identical (the mma gives the same sums with the operands exchanged), batch 1
  step 1543 against 1471 (track B), 1517 against 1441 (track A); batch 8 +5 to +35.
- FRAG_NB (tools/mk_fragnb.cu.bak): batch 8 step 3923 against 3880 (track B), 3795 against 3780 (track A).
Benchmarks written to understand why (findings 13f): tools/bench20.cu (primitives of a stage: about 4 ns per
instruction and warp with 16 warps; its mma variants with loop-invariant operands were hoisted by the compiler and
say nothing) and tools/bench21.cu (bank conflicts: 21 / 251 / 502 ns for 1 / 16 / 32 lanes per bank).
Also not kept, from the same hours (see findings 13d, 13e): MATTN_CPOST, MATTN_SPRE, MATTN_KPRE, CD_EPOLL, D_SM,
NORM_BATCH (five versions), the pinned sum of squares.

## A split attention pair, and what its control build gave away (s0047 - s0050)

MATTN_SPLIT (tools/mk_split_v2.cu.bak, mk_split_v4.cu.bak): at batch 7-8 the main block of a pair stops after the
softmax and exchanges the weights into a global buffer (SETTLE, one counter per pair on a cache line of its own); the
attention block a (done with its own pair 10 us earlier) loads the pair's V tiles into its 16 staging registers,
waits for the counter, mixes, writes the output words and stores V. Right at the first try: digests identical on both
tracks, bcheck bitwise at batch 6-8. Slow: inlined, 4149 / 4488 us per step (track B / A) against 3888 / 3801: the
profile shows phases A and C of every main block 2-4 us longer (the allocation of the whole batch > 2 kernel is
worse: spill line 68/56 instead of 56/40 on track A); as a function of its own, 4693 / 4835 (448 bytes of stack
around the call). The idea needs main and attention blocks in two kernel functions with a register allocation each;
not possible in what was left of the run. Not kept.

Its control build (MATTN_SPLIT=0, the same kernel code as s0045 in every executed path) was 12 us faster at batch 1
and 40 us slower at batch 8. Followed up with padding in front of each buffer in turn (host code only):
- only the counter array's address matters (PAD4: +4 KB: batch 1 1442 -> 1428, batch 8 3780 -> 3790; +64 KB: 1450 /
  3775; +1 MB: 1425 / 3789);
- counters moved one by one (offsets compiled in): the abort and progress words alone at +4 KB: batch 1 1425 (-17);
  the main-phase counter alone: batch 8 between 3760 (+16 KB) and 3807 (+20 KB); attention counter +4 KB: batch 8
  -9; router counter +4 KB: batch 1 -6;
- LDH_AB (abort word on every 16th attempt of a sentinel read): track A batch 1 1442 -> 1420, workload 1540 -> 1514;
  track B 1473 -> 1458, 1571 -> 1557; batch 8 track B 3878 -> 3856 (workload 3978 -> 3947), track A 3780 -> 3795
  (3969 -> 3974). All checks incl. tools/wd_test.py on the WD_TEST build.
  s0047 (track B): expected http 1546 / 3995, measured 1566 / 3979 (in process 1471 -> 1458, 64+1024 host-driven 1621
  -> 1610: only the HTTP batch-1 figure went the other way). tools/serve_long.py (submit/poll, 64+1024, poll every 0 /
  0.2 / 2 ms) shows the gain locally at every cadence: 1571 -> 1556.
  s0048 (track A): expected http about 1522 / 3912, measured 1527 / 3914.
- With LDH_AB the counter page still sets a level: slab of 4 MB, 16 pages from offset 0xf8a00: batch 1 1421 on pages
  1 2 4 7 8 11 13 14 and 1435 on the others; single-bit sweep: bits 12, 13, 14, 15, 17, 18, 19, 20 flip the level,
  bit 16 and 7-11 do not; batch 8 between 3778 and 3819. A slab of 256 MB with the same low 28 bits: batch 1 same
  level, batch 8 3886 instead of 3848 (track B), 3813 instead of 3779 (track A): physical placement.
- Start-up timing (api._place_counters): calibration figures of a run, track A: batch 1 1400 / 1417 by page, batch 8
  3189 (page 8) against 3205-3250; the same pages chosen in repeated runs. In process: track A 1421 / 3748 against
  1422 / 3779 without it, workload batch 8 3929-3940 against 3957, serve path 3938 against 3974; track B 1460 / 3823
  against 1460 / 3846, workload 3912 against 3951, serve path 3915 against 3957. All checks pass (api_stress with
  its batch-size changes, wd_test on the WD_TEST build).
  s0049 (track B): expected http 1553-1566 / about 3945, measured 1560 / 3936.
  s0050 (track A): expected http 1515-1527 / about 3880, measured 1527 / 3870 (board best; 1.291x vLLM at batch 1,
  0.981x at batch 8).

## Placement, continued (s0051 - s0054)

Page offsets from the environment for each hand-over buffer (64 KB of slack each, kernel untouched), track A, with
the counters placed: router logits batch 1 1418 (pages +0, +3, +5) against 1424 (+1, +2, +8); post-attention stream
1416-1418 (+1, +2, +8) against 1420-1421; attention output 1418, one page 1410; layer input +-2; batch 8 +-5 for all.
Extended start-up timing (mk_set_off): the batch 1-2 instance places counters (16 pages, 12 steps per timing) and
router logits, post-attention stream, attention output (16 pages each, 24 steps); the batch > 2 instance its counters.
- with 64 counter pages at batch 8 (s0051/s0052): local, same card and build: track A batch 1 1418 -> 1410, batch 8
  3772 -> 3751, workload 3954 -> 3934; track B 1460 / 3837 -> 3821, workload 3930 -> 3910.
  s0051 (track B): expected http 1560 / about 3920, measured 1555 / 3961 (in process batch 8 3942 -> 3934; HTTP
  repeats 3918-4000). s0052 (track A): expected about 1520 / 3850, measured 1516 / 3898 (repeats 3874-3900 against
  3840-3871 in s0050).
- back to the 16 counter pages and the allocations of s0049 for batch > 2, the batch 1-2 buffers as copies allocated
  last (s0053/s0054): the batch-8 timing picks page 8 again, as in s0049/s0050. Local: track A batch 1 1401, serve path
  1501 / 3924; track B 1452, serve path 1555 / 3907. bcheck, digests, flake 0 of 19, api_stress ok.
  s0053 (track B): expected http about 1550 / 3936, measured 1557 / 3950.
  s0054 (track A): expected http about 1508 / 3870-3890, measured 1516 / 3878 (board best: 1.300x vLLM at batch 1,
  0.979x at batch 8).
What the scored figure does and does not follow: the in-process batch-1 step fell 1442 -> 1402 over s0046..s0054, the
HTTP figure 1546 -> 1516; the last 9 us in process (s0052 -> s0054) did not show over HTTP.

## Last experiments, none kept (after s0054)

- Six placed buffers for the batch 1-2 kernel (tools/mk_place6.cu.bak: hidden units, q/k/v and layer input as well;
  each shows about 5 us between pages when moved alone, tools/offsweep.sh on tools/mk_offs.cu.bak: 1408-1414).
  Against the three of s0054, same card: track A batch 1 1401 -> 1409, serve path 1501 -> 1518; track B 1452 -> 1448,
  1555 -> 1550. Not a gain that holds: the new allocations move everything else's candidates, and at this size
  (5-8 us) the result is the draw of the candidate set, not the search. The engine stays at s0053/s0054.
- Two kernels side by side in generate and serve (tools/mk_two_kernels.cu.bak): the 128 main blocks and the 24
  attention/router blocks as two kernel functions (template parameter ROLE), launched on two streams, so that each
  role has a register allocation of its own (what the split attention pair of the previous section needs). Correct:
  generate == step-by-step, served tokens identical, at batch 1 and 8 on both tracks. Slow: generate 1676 against
  1402 us per token at batch 1 and 4128 against 3797 at batch 8 (track A; track B 1594 / 1455 and 4090 / 3793),
  although the main-only kernel of batch 1-2 compiles with 28/4 bytes of spills instead of 36/16. Two kernels that
  are resident together do not run like one: every hand-over between them is slower (about 5 us per layer). Not
  understood further; not kept.

## One kernel with a function per role, and its control (s0055 / s0056)

ROLE_SPLIT (tools/mk_rolesplit.cu.bak): the kernel's body as an inlined function with a role parameter; the kernel
runs the main role itself and calls a non-inlined function with the attention/router role on blocks 128..151 (one
launch, an allocation per role). 320 and 224 bytes of stack; batch 1 1669 us on both tracks; on track B the batch-8
digest changes (the norm's sum of squares fused otherwise). Not kept.
Its control (the body as an inlined function, one role parameter = both roles: the code of the engine): track B
batch 1 1385 instead of 1453, spill lines 64/88 and 36/16 instead of 68/92 and 68/84; track A 1434 instead of 1402,
spill lines 64/88 and 68/84 instead of 56/40 and 36/16. Digests identical in all four.
BODY_WRAP (wrapper on track B only): track B 1381 / 3805 in process, workload 1480 / 3907, serve path 1481 / 3916;
bcheck 7 of 7, digest_long equal to s0053, flake 0 of 39, api_stress and long_ctx ok. Track A build: spill lines and
timings of s0054 (1402 / 3747).
s0055 (track B): expected http about 1490 / 3950, measured 1473 / 3963 (the board's best at batch 1: 1.338x vLLM).
s0056 (track A, build unchanged): expected 1516 / 3878, measured 1516 / 3874.
Final verification of the s0053/s0054 engine before this (both tracks): bcheck 7 of 7, digest_long identical in three
processes, flake 0 of 149 at batch 1 and at batch 8, api_stress, long_ctx to 2040, wd_test on the WD_TEST build.

## Closing: allocation draws that were not better, final verification, the engine as it stands

Tried after s0056, each against the committed build on the same card, none kept:
- a never-taken guard (`B > MAXB + k`, k stores, tools/mk_draw.cu.bak) at the kernel's start or at the start of
  phases A, C, D, E, F, for one instance at a time: the batch 1-2 instances keep their allocation (36/16); the batch
  > 2 instance of track A goes from 56/40 to 64/88 or 68/92 and from 3743 to 3918-3936 us at every site and size.
  The guard is not a neutral draw (it tells the compiler that B may exceed 8).
- both role bodies inlined into the one kernel (tools/mk_rolesplit2.cu.bak): batch 1 1556 us on track A, 1623 on
  track B (whose digests change as well); batch 8 3743 / 3797 (-0 / -10).
So three allocations of the batch-1 kernel were seen in all (36/16: 1381-1402; 68/84: 1434-1453; the role forms:
1556-1669), and the engine has the first on both tracks.

Final verification of the committed engine (s0055 track B, s0056 track A; the track-A build is the one verified
before s0055): bcheck 7 of 7 batch sizes bitwise, digest_long identical across processes (track A 151d4654 /
ccbb90c7, track B 8496fb98 / a143694c: the digests of s0039, i.e. no change of the function since the softmax
rewrite and before), flake 0 mismatches in 149 repetitions at batch 1 and at batch 8, api_stress all ok, long_ctx
to position 2040, wd_test on the WD_TEST build (hang seen in 0.9 s, engine usable afterwards).
In process, last measurement: track A 1402 / 3743 us (step), serve path 64+1024 1501 / 3924; track B 1382 / 3807,
serve path 1480 / 3920.
Position dependence of the final engine (generate, us per token per 128 positions from 0):
  track A batch 1: 1391 1405 1460 1489 1554 1563 1587 1593 1617     batch 8: 3774 3829 3892 3921 3962 3978 4005 4006 4044
  track B batch 1: 1364 1374 1418 1445 1506 1516 1538 1546 1570     batch 8: 3829 3925 3995 4045 4090 4115 4128 4148 4173
(at the start of this session's last third: track B batch 1 1467 .. 1717, batch 8 3958 .. 4332.)
Board: track A s0056 1516 / 3874 (1.300x / 0.980x vLLM), track B s0055 1473 / 3963 (1.338x / 0.958x).

## Last measurements (the engine is s0055 / s0056; nothing below changed it)

- XF32 (tools/mk_xf32.cu.bak): track A with its BF16-valued residual stream kept in 32-bit words (track B's container;
  no value changes: digests of track A). Batch 1 1441 us in both kernel forms (spill lines 56/40 and 64/80 for the
  batch 1-2 instance: not the 36/16 draw), batch 8 3761 (+18: twice the bytes). Not kept. Track B's 20 us at batch 1
  are not simply the container.
- Abort-check interval once the counters are placed (track A): LDH_AB=1 batch 8 3740 against 3744 with 16, serve path
  equal; batch 1 1435 against 1402. One interval for both kernels is right.
- Profile of the final engine (track A, profiling build without the start-up placement, sliding layers, us per layer;
  position 150 / 1000). Batch 1: input norm 1.8 / 2.0, q/k/v 2.5 / 2.5, wait for the attention 5.1 / 6.3, output
  projection (from the attention's early signal) 5.2 / 8.4, to D 1.4 / 1.4, D 2.4 / 2.5, to E 0.5 / 0.5, routing
  2.5 / 2.5, expert gate/up 6.6 / 6.5, to F 1.7 / 2.8, down 4.6 / 5.1; attention block 6.9 / 10.8 and 1.7 / 6.9 for
  staging the next layer; router block 3.3 / 3.5. Batch 8: gate/up 24.9 / 25.7, down 22.7 / 23.1, wait for the
  attention 9.3 / 12.8, input norm 4.8 / 5.0, D 4.8 / 4.8, output projection 4.3 / 4.3, routing 3.6 / 3.6.
- Fine profile, batch 1: the barrier after the down phase's fragment copy is 0.26 us at position 150 and 1.39 at
  1000 (the block waits for the last block's hidden units), and tools/bt.py says who is last: the 16 main blocks
  112..127 are behind the other 112 in every phase, by 3.8 us at the A signal and 3.5 at the E signal at position
  150 and by 5.1 and 4.7 at position 1000 (mean lag by groups of 16 blocks at the A signal: 0.5 0.7 0.9 0.7 1.0 0.9
  0.5 3.9; correlation of a block's lag between phases 0.93-0.99). Earlier in the run (journal, "Where a layer's
  time goes at batch 1") the late blocks were random from step to step. Not explained: the start-up placement does
  not change it (measured with the placement forced in the profiling build); a pause of 200 or 1000 ns between the
  polls of the attention/router blocks in their long waits changes neither the lag nor the step (1402 / 3744). It may
  belong to the profiling build (its clock block is attention block 7, block 135), or to the SMs those blocks run
  on; if it is real in the engine, every barrier of every layer waits 3-5 us for these 16 blocks. This is the first
  thing to look at next.

## Which physical blocks carry which role (after the open item of the previous section)

ROLE_MAP=k (tools/mk_rolemap.cu.bak): the 24 attention/router roles on physical blocks k..k+23, the 128 main roles on
the others in order; everything else unchanged (start-up placement included), track A, in process:
  k = 128 (the engine)   batch 1 1402   batch 8 3744
  k = 112                        1405           4176
  k = 104                        1447           3962
  k = 64                         1453           4058
  k = 1                          1467           4498
So the block index is a place in the machine, and the light roles belong on the last 24: any other choice tried
costs up to 5% at batch 1 and 6-20% at batch 8. Together with the 16 main blocks that are always last (112..127,
i.e. the last main blocks before the light ones) this reads as: blocks are served in an order tied to their index
when the memory is busy (the earlier note "the last block is served last, every other block waits for it" is this,
and it is not random any more now that the phases are regular). The engine has had the light roles on the last 24
blocks from the start, by luck. Not done: finding out what the order is (a cluster of SMs? a slice of the L2?) and
whether main blocks of the last places should carry less than 1/128 of a phase's tiles (the expert phase's split is
arithmetic and could be weighted; the phases with fixed rows could not).

Two follow-ups to the place of the roles, neither kept (track A, in process, digests unchanged where the function
could change):
- a smaller share of the expert phase for the last 16 main blocks (tools/mk_esplit.cu.bak; shares last : others):
  equal 1405 / 3754 (the control with the same code); 5:8 1419 / 3768; 2:3 1424 / 3764; 3:4 1418 / 3758; 6:5 1416 /
  3786. Equal shares are best: the late blocks are late because they are served last, not because the phase would
  end earlier without them.
- the roles rotated within the last 24 blocks (tools/mk_auxrot.cu.bak; router blocks first, attention blocks last and
  two other rotations): batch 1 1414-1415 (with another allocation of the batch 1-2 kernel, so not comparable to 1402),
  batch 8 3903-4492 against 3744.
The engine's assignment (main 0..127, attention 128.., router 136..151, pairs of sequences 6-7 on main blocks 0..7) is
the best of everything tried.

## tools/bench22.cu: the order of service and the rate by number of streaming blocks

All blocks of a grid (512 threads, one block per SM) read 16 MB each from a buffer far larger than the L2, 16 loads in
flight per warp, and take a rank when done (mean over 12 launches). The card reports 152 SMs.
- Total rate: 16 blocks 2.35 TB/s (147 GB/s per SM), 64 blocks 6.44 TB/s, 128 blocks 7.14 TB/s, 152 blocks 7.36 TB/s.
  So the "58 GB/s per SM" used in these notes is the share of one of 128 streaming SMs, not a property of an SM: an
  SM that streams while few others do gets up to 2.5 times that (the attention blocks during the main blocks' wait).
- Order: the blocks fall into classes by index that repeat irregularly in pairs and fours: about 12 blocks finish
  among the first 17 (4-7, 20-21, 26-27, 44-45, 50-51), about 60 among the last 50 (16-19, 24-25, 32-37, 42-43, 48-49,
  52-55, 60-61, 68-69, 72-75, 80-81, 88-89, 92-95, 100-101, 108-109, 112-115, 120-121, 128-129, 132-135, 140-141,
  148-149), the rest in between. By index modulo 8: 89 89 67 69 86 82 61 61. This is not the engine's late group
  (112..127 as a whole): what makes those 16 late in the profiling build is still not known.

Last test of the roles' places (tools/mk_swap4.cu.bak, batch 1-2 kernel only, track A, same spill lines as the
engine): the four attention blocks on physical blocks 4..7 (the "early" class of bench22; the main blocks from there
on 128..131): 1442 us; on 44..47: 1478; on 120..123: 1433; within the last 24, on 132..135, 136..139, 144..147 or
148..151: 1406-1408; the engine (128..131): 1402. The engine's assignment is the best of all that were tried.

The assembler's optimisation level (-Xptxas -O2, -O4) is not another draw: same spill lines, timings and digests
on both tracks.

Two kernels side by side once more, with the main kernel launched first (tools/mk_two_kernels2.cu.bak; in the
first attempt the attention/router kernel was launched first and so got the first places, the worst assignment of
the table above): generate at batch 8 3900 us (track A) and 3920 (track B) instead of 4128 and 4090, against 3794
and 3786 for the one kernel; at batch 1 1641 and 1555 against 1402 and 1385. The order of the launches was a large
part of the loss at batch 8, not at batch 1. Still not usable.

And the split attention pair inside that two-kernel form, for batch > 2 in generate and serve only
(tools/mk_two_kernels_split.cu.bak; tokens identical): track A generate at batch 8 3888 with the split, 3917 without,
3794 for the one kernel; served 64+1024 workload 4158 / 4046 / 3924. Track B 3971 / 3924 / 3786 and 4248 / 4052 /
3907. The split is worth 30 us at short contexts where each role has its own allocation, loses over 100 at long
ones (the V half has 16 levels in registers and fetches the rest one by one), and the pair of kernels is 100 us
behind the one kernel to begin with. Closed for this run.

One accumulator per mma chain instead of two (MK_NACC=1; other bits, timing only): batch 1 1418 against 1403 (track
A), 1399 against 1380 (track B). Not a lead.
Last check of the final build: tools/flake.py with 300 repetitions on each track: 0 mismatches in 299 at batch 1
(600 steps per launch) and 0 in 299 at batch 8 (300 steps); digest_long as before.

From the other runs' records (progress/, opened at hour 23.3): s4-qwen38-27b-b200-a1, finding C11: a weak load
(ld.global.cg) in a spin missed another SM's later store in its sentinel hand-over; strong loads fixed it. This
engine's reads were the same kind of load (its writers are atomic exchanges; no stale read seen). LDX_RELAXED:
ld.relaxed.gpu for every hand-over read: same spill lines, timings (track A 1401.5 / 3743.5 against 1402.6 / 3742.9,
track B 1382 / 3805.5 against 1387 / 3804.6) and digests; bcheck 7 of 7, flake 0 of 24, api_stress, serve ok.
s0057 (track B): expected http 1473 / 3963 unchanged, measured 1475 / 3950.
s0058 (track A): expected 1516 / 3874 unchanged, measured 1520 / 3877.

The hardened build once more after its submissions: tools/flake.py 100 repetitions per track (0 mismatches at batch 1
and at batch 8 in 99 repetitions, on track A and on track B) and long_ctx to position 2040 on both.

End of the run. The engine is engine/ as committed (s0057 track B: 1475 / 3950 us over HTTP; s0058 track A: 1520 /
3877; the board's best figures are those of the same kernel before the hardening, s0055 1473 and s0056 1516),
verified after the last change; the open items are in notes/findings.md (13d: the 8 attention pairs without
an SM at batch 7-8, which a split over two blocks would solve if the kernel had registers for it, 13h; 13i: the
register allocation as a draw; 13j: the service order of the blocks).


# Extension (2026-11-15, 12 agent-hours): batch 8 on track A

What the operator measured after the run: same-card rematch vLLM 1.99 / 3.81 ms, s0054 1.51 / 3.88 ms (batch 1 /
batch 8); 2, 4, 8 streams 1.94 / 2.68 / 3.88 ms per step. Target: 3.47 ms per step at batch 8 on track A (1.10x)
without giving up batch 1. Audit (audits/flint-final/REPORT.md): s0054 is track A item by item; my "track B" (float32
stream and unrounded intermediates) is not bit-identical to its own track-A build (0 of 14,336 positions), and track
B now means bit-identical to the track-A build of the same source.

Decisions:
1. Only track A from here on. The float32-stream build is not submitted as track B any more (and not at all: it is
   not what this project calls B, and it is slower at batch 8). `engine/TRACK` = A. If a change is found that is
   bit-identical to track A but model-specific in its formats, it is verified position by position against the
   same source with the change off before it is labelled B.
2. First submission: the two defects of the audit (a served request that does not fit the context raised nothing;
   step() and generate() did not check the batch size). No speed involved.
3. Batch 8 needs 410 us per step, 8.4 us per layer of 76. Where the layer goes (final profile, journal above):
   expert gate/up 25 us for about 160 MB (6.4 TB/s, near the 7.1 the card gives 128 streaming blocks), down phase
   22.7 us for 81 MB (3.6 TB/s: five rounds, each a fragment copy and 16 tiles per warp), wait for the attention
   9.3-12.8 us with the memory idle, norms 4.8 + 4.8, output projection 4.3, routing 3.6. The two places with more
   than 5 us of slack are the down phase (11 us over its memory time) and the wait for the 8 pairs on main blocks.
   Order of work: (a) per-block timeline at batch 8 (tools/bt.py) to see which blocks every barrier waits for;
   (b) the down phase as one copy and one stream, this time with the per-tile work cut to one shared-memory load
   (the earlier attempt streamed at 5.3 TB/s and paid for its tables in the expert phase) and with a control build
   of the same shape for the register allocation; (c) the wait for the attention.

## s0059 (track A): the audit's two defects

Expected: nothing in speed (host code only); HTTP 1520 +-6 / 3877 +-30. Measured: valid, 1522 / 3898 over HTTP,
1403 in process at batch 1. Both figures inside the repeat spread of s0058 (3877): no cost, as expected.
`submit()` now raises for an empty prompt, n < 1 or a request that does not fit `max_context`; `step()` and
`generate()` raise for a batch outside 1..max_batch (tools/api_stress.py tests both).

## Batch-8 timeline, and the attention-first experiment (AFIRST): negative so far

tools/bt.py --B 8 (profiling build, layer 8, position 150, us from the first block's A start; median over the 128
main blocks / the 8 blocks that also carry an attention pair, blocks 0..7):
A done 7.6 / 9.0; C start 12.1 / 17.8; C done 18.1 / 22.6; D start 21.9 / 23.8; E start 27.4 / 29.7; E done
57.7 / 59.8; F start 59.2 / 61.6; F done 81.9 / 84.3. Every barrier after the attention waits for blocks 0..7: they
run their pair after their own phase A, with nothing staged, then load their output-projection tiles, issue their
sentinels and ask for the attention counter (3.4 us after the pair's last store, fine profile).

AFIRST (engine/mk.cu, switch, off): blocks 0..7 do no phase A at batch 7-8; they stage K/V like an attention block and
run the pair when q/k/v arrive; their 40 half row groups of q/k/v are computed by 40 other main blocks (warps 8..15
of blocks 8, 11, 14, ...; the same chains in the same order: bcheck and both long digests unchanged on the first
build). Measured in process, same card: 3746 -> 4747 us per step. Not the mechanism: the timeline showed every
phase of every block slower. Three separate things were found on the way:

1. The code generation of the fragment getter decided most of it (next section).
2. With that fixed, AFIRST is still +310 us (3689 -> 4002): D starts 2.2 us earlier relative to C (the intended
   effect: the pair blocks trail by 2.6 us instead of 4.6), but phase A is 2.8 us longer on every block (fine
   profile on a plain block: norm 5.2 -> 6.9 us, the 20 tiles 1.1 -> 1.9), attention to C +1.8, E +0.9, F +2.6.
   Tests: hosted code compiled in but never run +170 us (3693 -> 3862; a call, stack frame 176 -> 320 bytes; inlined
   it spills 216 bytes and is slower still); hosts running next to the unchanged owners: norm 5.7; pair blocks
   without any staging: norm 7.2 (so the staging traffic is not it). What makes phases A and F slower when blocks
   0..7 stop streaming phase A is not found; it has the look of finding 13j (any change of which block streams what
   costs up to 20% at batch 8).
3. A profile detail that explains "A norm 5 us": with the early signals a block starts the next layer when its own
   down phase is done, and its norm then waits (sentinel) for the residual rows of the slowest block. The time from
   the first block's A start therefore contains the spread of the down phase.

## The fragment getter decided the "register draw" of the batch > 2 kernel (tools/sass.sh, sass_seg.py)

Disassembly of the two builds (cuobjdump -sass; tools/sass_seg.py lists the runs of HMMA instructions with the
instructions in and between them). `xget` was `lane < Ls ? xf[chunk * Ls + lane] : 0`: a conditional load. In the
builds that were fast the compiler had predicated the load (about 5 instructions per tile); in others, including every
AFIRST build, it left a branch around each load and recomputed the shared-memory address inside it: 20-23
instructions per tile (a 16-tile run of the expert phase: 87 -> 346 instructions, of the down phase 66 -> 287).
At 4 ns per instruction and warp that is a microsecond per round. Which form a phase got changed with unrelated
code elsewhere in the kernel: this is a large part of what findings 13i calls the draw (it was never only spills:
the build with fewer spill bytes was the slower one here).
Now (XGET_BF): every lane loads; a lane without a column reads a zero word placed in front of the buffer, with
stride 0 (pointer and stride are selects, the load is unconditional). 16-tile runs: 33-36 instructions (2 per tile)
in every build looked at. Batch > 2 kernel only: the batch 1-2 kernel had the predicated form and got slower with
this one (1403 -> 1451 us at batch 1, its spills 36/16 -> 68/84 bytes), so it keeps the conditional (bit 0 of
XGET_BF).
Measured on one card: in process 3746 -> 3689 us per step at batch 8 (down phase in the profile 22.7 -> 18.1 us),
batch 1 1403 -> 1404; over HTTP (tools/serve_long.py, alternated twice) 3925, 3924 -> 3861, 3865 at batch 8 and
1512 -> 1512 at batch 1. Same bits: bcheck (every batch size against batch 1), digest_long and digest1 unchanged,
flake 0 of 49, api_stress ok, local gates as before.
Expected for the submission (track A): HTTP batch 8 3898 -> about 3835 (+-30), batch 1 1522 +-6.

## s0060 (track A): fragment getter without a conditional load

Expected: HTTP batch 8 3898 -> about 3835, batch 1 1522 +-6. Measured: valid, 3842 at batch 8 (-56), 1531 at batch 1
(+9; in process 1404, as before; the batch 1-2 kernel is the same code, spill line 36/16 unchanged, so this is the
server-side spread of batch 1, a little over the +-6 seen so far). Confirmed for batch 8.

## A load waits for the exchanges its warp issued before it; the hand-over poll moves off warp 0

Three observations in the batch-8 profile pointed the same way: (1) a main block with an attention pair needed 1.9 us
between requesting its output-projection tiles and seeing the attention counter, and the time stayed when the tiles
were requested elsewhere; (2) issuing that block's two sentinel exchanges in front of its attention instead of
behind it left "C start" where it was: the attention itself ended 1.9 us later; (3) "C signalled -> D start" is
1.2-1.5 us on every block. An exchange whose result nobody reads does not hold the warp up, but the warp's next
load returns behind it, about 2 us later. The thread that polls a hand-over counter was thread 0, and warp 0 is the
warp that issues the exchanges and bumps at the end of most phases (residual update, sentinels, counter).
Change: `wait_ctr` polls from thread WAIT_TID = 192 (warp 6: never a leader). In process, one card: batch 1
1403 -> 1389 us (threads 256, 480: 1392, 1389), batch 8 3690 -> 3682. HTTP path (tools/serve_long.py, same card):
batch 1 1511.8 -> 1493.8, batch 8 3860.8 -> 3852.5. Bits unchanged (bcheck 7 of 7, digest_long, digest1, flake 0 of 49,
api_stress, local gates, watchdog test).
Expected for the submission (track A): batch 1 1531 -> about 1513 (+-6), batch 8 3842 -> about 3834 (+-30).

## s0061 (track A): hand-over counters polled from warp 6

Expected: batch 1 1531 -> about 1513 (+-6), batch 8 3842 -> about 3834 (+-30). Measured: valid, 1514 / 3829 over HTTP,
1390 in process at batch 1. Both as predicted. Best track-A figures so far: 1514 / 3829.

## The scored run is a long-context run; K/V tile batches pinned and used in the batch > 2 kernel

tools/pos_time.py (us per token by 128-position chunk, batch 8, s0061 build): 3737 3790 3835 3866 3906 3925 3953 3962
3998. The HTTP figure is the mean over positions 64..1088, so 85% of it is beyond position 256. Two consequences:

1. HOSTC (a main block with an attention pair has its output projection computed by another main block, block 8 + o;
   with HSTG the owner stages its pair's first K/V tiles after its phase A): bit-exact, flake 0 of 99 on all four
   tests, and in process at positions under 330 it gains 25 us (3682 -> 3657) and 2 us per layer in the profile.
   By position: 3677 3718 3857 3906 3992 4010 4050 4061 4097: -60 and -72 in the first two chunks, +20 to +100
   beyond; over HTTP 3857 -> 3926. Reason (timeline at position 700): the attention takes 14 us there instead of 8,
   on every pair block; the host's second pass (its own rows: 16 more tiles per warp, a trip to memory after its
   first pass) can only start when the last attention output is in, and ends 2.2 us later, where the owner's own
   output projection ended 1.5 us later. Off. (A variant in which the hosts also issue and confirm the owner's
   sentinels removed the owner's lag in the profile, F signalled max 80.3 -> 77.9, but was unstable in the normal
   build, 3677-3719 between launches, and had lost the causal order of one sentinel: tools/mk_hostc_v5.cu.bak.)
   Three smaller experiments at the same place, all negative: MATTN_TAIL (sentinels in front of the attention: the
   attention ends 1.9 us later, see the entry on the poll thread), MATTN_CPRE=2, FCB (the copy of the attention
   output with 4 or 8 loads in flight: +77 / +460 us, its registers went to the stack).
2. The attention itself. Fine profile of an attention block at position 700 (sliding layer, 33 key groups): scores
   2.4 us, exponentials 1.3, weights 1.4, V mix 3.1; at position 150: 0.7, 0.5, 0.7, 0.7. The mix loads 8 or 9 V
   tiles one behind the other past the 8 staged ones. The batch 1-2 kernel requests them 8 at a time (KV_BATCH);
   in the batch > 2 kernel that had measured "nothing" twice. In the profiling build it does work (mix 3.06 -> 1.47
   us, the other blocks' C signalled 23.4 -> 21.3), in the normal build it did not: the batch was a row of plain C
   loads and the compiler had moved them down to their uses. With the batch loads as volatile asm (KV_ASM; the
   K groups staged in shared memory keep plain loads):
   batch 8, by position: 3703 3731 3802 3811 3879 3896 3924 3923 3955 (-30 to -55 in every chunk), with the batches on
   the attention blocks and on the main blocks' pairs (KV_BATCH_AUXBIG, KV_BATCH_MAIN; on the attention blocks alone
   it is +15: the pairs on the main blocks then trail);
   batch 1, by position: 1373 1384 1434 1464 1529 1537 1561 1568 1592 against 1380 1392 1458 1485 1550 1559 1584
   1591 1617: the same sinking had happened there (-7 to -25).
   HTTP path, same card: batch 8 3857.8 -> 3807.2; batch 1 1493.8 (s0061 build) -> 1484.0.
   Bits unchanged: bcheck 7 of 7, digest_long, digest1, flake 0 of 59 x 4, api_stress, local gates, long_ctx, watchdog.
   Also tried here: every tile load of the kernel as volatile asm (TLOAD_ASM): +4 to +10 us at both batch sizes, off.
Expected for the submission (track A): batch 8 3829 -> about 3780 (+-30), batch 1 1514 -> about 1504 (+-6).

Other measurements of this stretch (negative or not used):
- A warp's memory operations complete in the order issued. Requesting the down phase's first tiles before the expert
  phase's hidden units are written (FPRE_EARLY): +26 us at batch 8 (the exchanges then wait for the tiles, and every
  block's down phase waits for them). The next layer's phase-A tiles before the layer-end sentinels (APRE_EARLY):
  nothing. Residual rows written with a plain store instead of an exchange (XPLAIN): -4 us, inside the noise.
- The streaming rate does not depend on the shape of a round (tools/bench23.cu, 128 blocks): rounds of 16 loads then
  16 mma 7.25-7.6 TB/s, rolling reload 7.33-7.55, half rounds 7.36-7.6, with a barrier and 60 or 240 instructions
  per round the same; two rounds in flight (32 registers) 5.7-6.6; scattered 8 KB runs 7.45-7.56. The expert phase
  is at this rate (10.35 slots x 2.27 us + the first slot's latency + the final stores = 27.5 us against 26.7
  measured), the down phase 0.4 us per round above it.
- F_PIPE (the next round's fragments and all 16 tiles of the down phase requested together): +69 us; the fragment
  requests in front of the tiles delay the tiles by their own code (0.4 us per round).
- Speculative prefetch of expert weights (not built). tools/route_overlap.py: of a step's units (32.1 distinct
  experts per layer at batch 8 on tools/ids.npy), 36% were units one position earlier, 50% within the last 2
  positions (52 units), 63% within the last 4 (81 units); a sequence repeats 23% of its own 6. tools/bench24.cu:
  data that is in the L2 streams at about 4-5 times the memory's rate, but only about 64 MB of the 129 MiB survive,
  and 48 MB of other traffic evict them. A layer's routed weights are 205 MB at 32 units; the window in which the
  memory is idle before the routing is known is about 25 us. Upper bound of the gain as estimated: 16-23 MB of
  useful hits per layer, 2-3 us of 80.

## s0062 (track A): attention K/V batches pinned and used in the batch > 2 kernel

Expected: batch 8 3829 -> about 3780 (+-30), batch 1 1514 -> about 1504 (+-6). Measured: valid, 3784 at batch 8 (as
predicted) and 1494 at batch 1 (-20: twice the local HTTP difference, which was measured against a build whose
batch 1-2 kernel had already changed by the rewritten load expression). Best track-A figures so far: 1494 / 3784.

## Where the batch-8 step goes: timing studies in the normal build (WHATIF), and the copy of the attention output

The profiling build misled twice in this extension (its markers are loads, and a load waits for what its warp
issued before it): HOSTC and a weighted split of the expert phase both looked right in its timeline and did nothing
on the scored workload (tools/gen_long.py: prompt 64 from tools/ids.npy, 1024 greedy tokens, in process; HOSTC +10 us,
the split at 11:12, 7:8, 5:6, 3:4 for the pair blocks 3801-3809 against 3805). So the question "what would the step
be without X" is now asked of the normal build: `-DWHATIF=bits` removes one piece of work (the results are wrong;
bit 0 forces a fixed routing with 36 units, so that the garbage does not change what is streamed). Batch 8, us per
step on the scored workload, against forced routing alone (3698; the spill figures differ between these builds, so
each number carries the +-50 of a different register allocation):
- the down phase copies fragments in its first round only: 3487 (-211)
- the norms load the first pair of sequences only (one load instead of four, one behind the other): 3637 (-61)
- the attention sees no cached keys: 3440 (-258: all of the context-dependent attention)
- no routing computation in the main blocks: 3421 (-277)
- the copy of the attention output loads one sequence instead of eight: 3596 (-102)
- all of these: 2830.
The copy of the attention output: `frag_copy` loads, tests for the sentinel and stores one sequence after the other,
and a block that is waiting for the attention sits in the wait of the first one: when the last output arrives, up to
seven loads are still to come, each a trip to memory (0.25 us). FCB = 4: four loads issued together and kept in
registers; a word that holds the sentinel is waited for alone, as before, and the later ones that held it are
requested again together afterwards. (The first version of this, earlier today, passed arrays to a wait function
and paid for it in stack traffic: +77 us. Widths 2 and 8: +18 and +40.) Scored workload in process 3804.6 -> 3762.8;
HTTP path 3807.0 -> 3763.5; batch 1 unchanged (1483.7). Bits unchanged (bcheck 7 of 7, digests, flake 0 of 59 x 4,
api_stress, gates, long_ctx, watchdog).
Expected for the submission (track A): batch 8 3784 -> about 3741 (+-30), batch 1 1494 +-6.

## s0063 (track A): copy of the attention output with 4 loads in flight

Expected: batch 8 3784 -> about 3741 (+-30), batch 1 1494 +-6. Measured: valid, 3735 / 1497. As predicted.
Best track-A figures so far: 1494 (s0062) / 3735 (s0063); vLLM's trial figures are 1971 / 3796.

## The down phase with one copy of its fragments (F_ONE)

Two more attempts on the norms first (timing study: 61 us per step for their sequential loads): four loads in flight
(NORM_B=1) +33 us, two in flight (NORM_B=2) +7. Not kept: the norm runs with all 16 staging registers full.

The down phase (timing study: 211 us per step for its copies after the first round). A round was: 8 tiles requested,
then (two barriers later) the fragments of the round's 8 items copied with strong loads behind them, then the other
8 tiles: the stream of tiles stopped for the copy in every round. A first attempt to request copy and tiles together
(F_PIPE) had been slower (+69, on the scored workload +169).
F_ONE: all fragments of the phase are copied once, before the first round, into a compact layout that fills the 64 KB
fragment buffer exactly: slot b * 8 + r holds sequence b's hidden units of its r-th expert (r = 6, 7: the halves of
the shared expert), 32 chunks x 4 lanes. Every sequence is a member of exactly 6 units and of the shared expert, so
there are 8 B slots, 64 at batch 8, whatever the routing. The routing's union stage leaves two byte tables: the rank
of a unit within each member sequence, and the unit of each (sequence, rank). The copy: thread = (sequence, chunk,
half), eight loads, four in flight (as FCB). A round is then tiles only: 16 requested as soon as the previous
round's mma are done, and the B operand of a lane is its sequence's slot if the sequence is a member of the item,
else the zero word with stride 0 (no branch, as in xget; the non-member columns used to hold stale fragments that
were masked after the mma, now they are zero: the same sums). One barrier less per round.
First build bit-exact. Scored workload in process 3762.7 -> 3621.4; HTTP path 3765.0 -> 3618.6; batch 1 unchanged
(1483.3). Bits unchanged: bcheck 7 of 7, digest_long, digest1, flake 0 of 99 x 4, api_stress, gates, long_ctx, watchdog.
Expected for the submission (track A): batch 8 3735 -> about 3590 (+-30), batch 1 1497 +-6.

## s0064 (track A): down phase with one fragment copy up front

Expected: batch 8 3735 -> about 3590 (+-30), batch 1 1497 +-6. Measured: valid, 3577 / 1496. As predicted.
Best track-A figures so far: 1494 (s0062) / 3577 (s0064): 1.32x and 1.06x vLLM's trial figures (1971 / 3796).

## After s0064: what the timing studies leave, and seven more attempts (none kept)

Timing studies on the s0064 kernel (batch 8, scored workload, against forced routing alone = 3452; most of these
builds share one spill line, 88/80): no routing computation -206 (candidate extraction -110, merge -44, union -53);
attention without cached keys -232 (scores -66, exponentials and weights -37, V mix -96); the down phase without its
fragment copy -121; no copy of the attention output -89; no sentinels at the start of the expert phase -81; norms
with one load instead of four -45. (Forced routing itself is 166 us under the real run: it has 36 units where the
scored text has about 41.)

Attempts, all with unchanged bits where they are complete engines:
- E_SETTLE_LATE (wait for the expert phase's sentinel exchanges behind its first slot): +3.
- HS_PLAIN (those sentinels as plain stores, no wait; a probe, not safe as it is): -25. HST_PLAIN (the hidden units
  themselves as plain stores): +25, with XPLAIN +21. So a plain store is not seen sooner than an exchange.
- E_CTR (no sentinels for the expert phase's hidden units; every warp bumps the counter after its own writes and the
  down phase waits for that): +18. The counter arrives later than the first values did.
- HS_EARLY (the sentinels issued for every unit slot and sequence by the main blocks while they wait for the
  attention, confirmed before their C signal; nothing left to do at the start of the expert phase): +7. It is six
  times the exchanges (768 per block and layer), at the time the attention blocks use the memory.
- KV_DB (V mix with two batches of tiles in flight, the first tile of the second batch requested with the first
  batch): batch 8 -5, batch 1 +38.
- VPAD (V mix in whole batches of 8 tiles without a test per tile; key groups past the context get a zero weight
  and whatever tile the zero-initialised cache holds): batch 8 +10, batch 1 +64 (the batch 1-2 kernel got another
  allocation, 192/40/40 against 176/20/16).
- Knobs on the s0064 build: poll thread 256, 480, 320, 96 within +-9 us of 192; staged K/V tiles pinned: -3; residual
  rows by plain store: -10 at batch 8, -4 at batch 1 (not taken, see the entry on the poll thread).
Reading of this: what the studies show for the copies and the sentinels is mostly their one unavoidable trip to
memory (a copy by counter and acquire with weak loads behind it would still need the counter's trip and the batch's),
the routing's 200 us are six rounds of a warp maximum plus six barriers that every arrangement tried so far needs,
and the attention's 230 us are instructions per key group, not trips. The engine is at 3577 over HTTP with the
streaming at the card's rate; the remaining 107 us to 1.10x are not in any single place that these tools can see.

## s0065 (track A, cleanup): the s0064 engine with the later experiments behind switches

Expected: no change, batch 1 1496 +-6, batch 8 3577 +-30. Measured: valid, 1492 / 3571. `engine/` as committed is
this submission's source. Final verification of it: bcheck 7 of 7, digest_long and digest1 as since s0058, flake
0 of 149 on each of the four workloads, api_stress, local gates, long_ctx to position 2040, watchdog test.

## The pointer array of the one-copy code cost four spilled values (FO_V)

The down phase's copy (F_ONE) kept the four source addresses of a batch in a small array of pointers for the
sentinel path. Three forms of the same function: as submitted (spill line of the batch > 2 kernel 176/52/88 bytes);
the addresses recomputed where they are used (160/24/24); recomputed and without the joint re-request of the later
words (176/48/84). Scored workload in process: 3615.7 / 3576.2 / 3645.5. HTTP path: 3619.2 -> 3579.0, batch 1
unchanged (1483.1). An allocation effect, found by trying the obvious "fewer live values" rewrite of new code; the
spill line moved with it this time. Bits unchanged (bcheck 7 of 7, digests, flake 0 of 79 x 4, api_stress, gates,
long_ctx, watchdog).
Expected for the submission (track A): batch 8 3571 -> about 3531 (+-30), batch 1 1492 +-6.

## s0066 (track A): down-phase copy without its pointer array

Expected: batch 8 3571 -> about 3531 (+-30), batch 1 1492 +-6. Measured: valid, 3527 / 1493. As predicted.
Best track-A figures so far: 1492 (s0065) / 3527 (s0066): 1.32x and 1.076x vLLM's trial figures.

## Two more of the same kind (stack and registers), and HOSTC again

- EXS (the attention's scores of a thread's 2nd and 3rd key group in shared memory instead of in a local array that
  is on the stack): same bits, batch 8 +14, batch 1 +30. Off.
- RTS: the routing's merge of a sequence's 12 candidates kept the six best in two local arrays and read "its own" by
  lane index, which put both arrays on the stack (14 stack accesses per thread). The lane's candidate is now read
  from shared memory, where the rank merge has left it. Stack frame of the batch > 2 kernel 160 -> 144 bytes.
  Scored workload in process 3589.2 -> 3577.5 (two alternations); HTTP path 3591.5 -> 3575.5; batch 1 unchanged.
  Bits unchanged (bcheck, digests, flake 0 of 59 x 4, api_stress, gates, long_ctx, watchdog). (The machine was replaced
  before these runs: this card is about 14 us slower at batch 8 than the last.)
  Expected for the submission (track A): batch 8 3527 -> about 3512 (+-30), batch 1 1493 +-6.
- Earlier negatives measured again on the current kernel (scored workload in process, base 3576.4, same spill line
  144/44/44 unless noted): HOSTC 3513.0 (-63); expert-phase split 7:8 3601.8; two V batches in flight 3607 (52/96);
  norm with two loads in flight 3573.5; residual rows by plain store 3577. HOSTC was +10 before the down phase was
  rebuilt: with that phase no longer the place where every block waits, the late blocks with an attention pair
  matter. Next submission after RTS.

## s0067 (track A): routing merge without lane-indexed stack arrays

Expected: batch 8 3527 -> about 3512 (+-30), batch 1 1493 +-6. Measured: valid, 3499 / 1493 (batch 8 13 us better
than predicted, inside the spread). Best track-A figures so far: 1492 (s0065) / 3499 (s0067): 1.32x and 1.085x.

## HOSTC on the rebuilt kernel: the output projection of the 8 blocks with an attention pair, hosted

The mechanism is the one built and set aside earlier in this extension (see "The scored run is a long-context run"):
at batch 7-8 a main block with an attention pair (the owner, blocks 0..3 or 0..7) does no output projection; block
8 + o (its host) preloads the owner's 256 tiles during the wait for the attention, computes the owner's 16 rows
first (the same 16 warps x 16 tiles and fragments: the same sums), writes them into the owner's rows of the
post-attention stream, then requests its own tiles and does its own rows. Hand-over: the owner issues its sentinel
after its attention and waits for the exchange itself to return (a read-back could meet the host's value); the host
waits until it sees the sentinel before it stores; the owner reads its rows back into its shared-memory copy and
only then puts the next sentinel into its rows of the layer-output stream (the host has read the layer-input rows
there before it wrote). Without the owner staging K/V (HSTG off).
It had measured -60 us at short contexts and +10 on the scored workload while the down phase was the place where
every block waited; on the kernel of s0067: scored workload in process 3576.4 -> 3513.0 (same spill line), HTTP
path 3583.2 -> 3514.4, batch 1 unchanged. Bits unchanged: bcheck 7 of 7 (batch 7 and 8 use it), digest_long,
digest1, flake 0 of 199 on each of the four workloads, api_stress, gates, long_ctx, watchdog.
Expected for the submission (track A): batch 8 3499 -> about 3432 (+-30), batch 1 1493 +-6.

## s0068 (track A): hosted output projection for the pair blocks (HOSTC)

Expected: batch 8 3499 -> about 3432 (+-30), batch 1 1493 +-6. Measured: valid, 3460 / 1493: -39 at batch 8 where
the local A/B said -69 (inside the +-30, at its edge). The extension's first goal, 3470 us per step at batch 8 on
track A without giving up batch 1, is met by 10 us, which is less than the spread of one evaluation: it needs
margin. Best track-A figures: 1492 (s0065) / 3460 (s0068): 1.32x and 1.097x vLLM's trial figures (1971 / 3796),
1.33x and 1.10x the operator's same-card rematch (1990 / 3810).

## Every switch again on the HOSTC kernel; a counter hand-over that is not strict (E_CTR)

Scored workload in process, batch 8, base 3512 (spill line 144/44/44 unless noted): owners staging K/V (HSTG) 3528;
poll thread 256 / 480: 3512.5 / 3514.4; sentinels for every slot during the attention (HS_EARLY) 3565.5 (52/112);
norm with two loads in flight 3555.7; down-phase tiles before the hidden-unit exchanges (FPRE_EARLY) 3557.5; phase-A
tiles before the layer-end sentinels 3508.5; residual rows by plain store 3513.4; staged K/V pinned 3514.5; late
sentinel wait 3515.7; copy of the attention output 8 wide 3610 (52/104); and E_CTR 3494.6 (-17).
E_CTR: the expert phase writes its hidden units without sentinels, every warp bumps the counter after its own writes
(exchanges, then the bump from lane 0 of the same warp), and the down phase's copy waits for the counter and then
reads with strong loads. It was +18 before HOSTC and is -17 after; HTTP path 3515.6 -> 3498.0. Digests and bcheck
unchanged. But tools/flake.py with 200 repetitions: batch 8 step-by-step 3 mismatches in 199, batch 8 generate 1 in
199 (batch 1: none). So "the atomics of one warp reach the L2 in order, so a reader that sees the counter sees the
data" does not hold here with 32 lanes exchanging and lane 0 bumping: a reader gets the counter before some
exchange has landed, about once in 100 runs of 97-300 steps. Off, and marked as not strict in the source. (The
sentinel words exist for exactly this; the engine's remaining counters are all early signals, "the sentinels are in
place", never "the data is there".) This is also the answer to item 0c of "Where to continue" in the findings: a
hand-over by counter needs more than the order of one warp's atomics.

## Which blocks host (HOSTC_B)

Any 8 main blocks without a pair can be the hosts. Scored workload in process, us per step, by the first host
block: 8: 3512.5; 16: 3509; 24: 3502.6 (three runs: 3502.3-3503.3); 32: 3508; 40: 3509; 56: 3510; 88: 3516; 104:
3508; 120: 3533 (the last main blocks are the ones served last, findings 13j: the worst place for more work).
Hosts are now blocks 24..31. HTTP path 3514.9 -> 3505.2, batch 1 unchanged. Bits unchanged (bcheck 7 of 7, digests,
flake 0 of 199 x 4, api_stress, gates, long_ctx, watchdog).
Expected for the submission (track A): batch 8 3460 -> about 3450 (+-30), batch 1 1493 +-6.

## s0069 (track A): hosts on blocks 24..31

Expected: batch 8 3460 -> about 3450 (+-30), batch 1 1493 +-6. Measured: valid, 3448 / 1493. As predicted.
Best track-A figures: 1492 (s0065) / 3448 (s0069): 1.32x and 1.101x vLLM's trial figures (1971 / 3796), 1.33x and
1.105x the same-card rematch (1990 / 3810). The goal of the extension (3470) is met by 22 us.

## The copy of the attention output waits for all its pending words together (FCB_W)

In the 4-wide copy a word that held the sentinel was waited for alone (one load per attempt), and the later words of
the batch were requested again when it had arrived: one more trip after the last output is in. Now every pending
word of the batch is requested again in each attempt. Scored workload in process 3502.3 -> 3489.7 (two alternations,
same spill line); HTTP path 3508.4 -> 3494.8; batch 1 unchanged. The same change in the down phase's copy (FO_V=3):
+36 us (its wait is long, for the slowest block's expert phase, and four loads per attempt instead of one is the
wrong trade there). Bits unchanged (bcheck 7 of 7, digests, flake 0 of 199 x 4, api_stress, gates, long_ctx, watchdog).
Expected for the submission (track A): batch 8 3448 -> about 3435 (+-30), batch 1 1493 +-6.

## s0070 (track A): attention-output copy with joint re-requests

Expected: batch 8 3448 -> about 3435 (+-30), batch 1 1493 +-6. Measured: valid, 3426 / 1493. As predicted.
Best track-A figures: 1492 (s0065) / 3426 (s0070): 1.32x and 1.108x vLLM's trial figures (1971 / 3796), 1.33x and
1.112x the same-card rematch (1990 / 3810). The goal of the extension (3470) is met by 44 us.

## After s0070: the routing-merge rewrite on the batch 1-2 kernel (RTS_S), not kept

The same rewrite as RTS for the batch 1-2 kernel (the lane's candidate read from shared memory instead of from a
lane-indexed local array): frame 176 -> 160 bytes, but the spill line goes from 20/16 to 52/88 bytes and batch 1 on
the scored workload from 1479 to 1515. Not kept; `engine/mk.cu` is the source of s0070 again (the experiment is in
tools/mk_rts_s.cu.bak). The batch 1-2 kernel's allocation (176/20/16) is a good one and every change to code it
shares has to be measured at batch 1: three of today's experiments that left batch 8 unchanged cost batch 1 30-64 us
(KV_DB, VPAD, EXS).

## Final engine (s0070 source): repeat runs at other batch sizes and at long context

tools/flake_b.py (new): step-by-step (97 launches) and generate (300 steps) at batch 7, 5 and 3, 60 repetitions each,
and generate at batch 8 from a 64-token prompt for 700 steps, 15 repetitions: 0 mismatches everywhere. (Batch 7 uses
the hosted output projection with 4 owners and 4 hosts; tools/flake.py covers batch 1 and 8.)

## The norm's loads once more (NORM_B=3): the seventh no, with a reason

Four loads out together, pairs 1..3 parked in the scratch that the second pass reads anyway (so that only one pair
stays in registers): same spill line as the base (144/44/44), same bits, and +27 us (3490.5 -> 3517). With the
allocation ruled out, what is left is the order of events: the norm starts while the slowest blocks are still
writing their rows; loaded one after the other, the later pairs are requested later and find their values; loaded
together they all find the sentinel, and each then costs a wait call and a second load. A batch pays where the data
is already there (the copy of the attention output after the wait, the down phase's fragments), not where it is
still arriving. Source: tools/mk_normb3.cu.bak; `engine/mk.cu` is the source of s0070.
(An eighth form, the remaining pairs requested together only after a thread has waited for its row group: +16 us,
and my first version of it left one pair out of the scratch, which tools/bcheck.py caught at one batch size while
the batch-8 digest happened to pass. Dropped; not in the source.)

## Expert-phase shares for owners and hosts on the final kernel: equal is still best

With HOSTC the owners are free early and the hosts late, so: owners more (9:8, 5:4), hosts less (7 of 8, 3 of 4),
both, owners less (7:8). Scored workload in process against 3491-3494: 3495.6, 3536.6, 3512.5, 3527.3, 3513.0,
3510.7. Not in the source.

## Limits of the timing studies, measured on the final kernel

- Batch 8, final kernel (3491 on this card): forced routing alone gives 3555 (3517 with HOSTC off), above the real
  run, where on the s0064 kernel it was 166 us below it. The study builds have other spill lines (208/92/84 against
  144/44/44), and that is now worth more than the 5 units the forced routing saves. Their differences (no routing
  computation -223, attention without cached keys -328, no down-phase copy -259, no copy of the attention output
  -145, no expert-phase sentinels -150) are therefore upper bounds of a slow build, not a budget of the engine; the
  figures of findings 15g (s0064 kernel, one spill line for most builds) are the ones to quote.
- Batch 1: the same switches (with a forced routing of six units) take the batch 1-2 kernel from 1481 to 2005 us by
  the allocation alone (208/44/36 against 176/20/16). No timing study is possible there with this method: the batch
  1-2 kernel is 35% slower in a bad allocation. The body-wrapper form (BODY_WRAP=1) on track A: 3520 / 1518 against
  3489 / 1481.

## The host-driven path at batch 8 (the evaluator's `w1_like`): 1.07x, and why

s0070's result block: `speedup` (HTTP, the headline path: one launch per cohort, argmax and feedback on the device)
1.320 / 1.108; `speedup_w1_like` (prompt 64, 1024 greedy tokens through step(), the chosen token read on the host at
every step) 1.332 / 1.072: 1527 us at batch 1 and 3596 at batch 8, against 1493 and 3426 over HTTP. The milestone
`h0_step` is passed, `h0_w1_like` is not at batch 8.
tools/step_cost.py (positions 64..319, us per step): batch 1 generate 1381, step + argmax on the device 1418, the
same with the token read on the host and sent back 1447; batch 8: 3358, 3406, 3438. So a launch per step and the
argmax cost about 40-50 us here, the host round trip another 30, and the evaluator's loop adds more at batch 8 than
this one does (+170 against +79; it checks the logits of the timed steps). Not reachable from the engine's side: a
kernel that stays alive across host-driven steps cannot be used, because it holds every register of every SM while
it waits, and the host's own argmax kernel between two steps would never be scheduled.

## Last knobs on the final kernel (nothing)

Joint wait for q and k/v in the attention's first loads (batch > 2 only; the batch 1-2 kernel keeps its spill line
with the constant-false path compiled in): 3490 against 3490. First V batch not requested early (KV_VEARLY=0): +12 at
batch 8, +23 at batch 1. Abort word looked at every 8th / 32nd / 4th retry instead of every 16th: 0 / 0 / +5.
Body-wrapper form on track A: +31 / +37. None in the source.

## A timeline without memory operations: %clock64 (tools/bt_clk.py, build with MK_EXTRA=-DMK_CLK)

The profiling build's markers are loads; the other B200 run (finding C12) used the cycle counter instead. Added as a
study build of the NORMAL kernel (same roles, no clock block): thread 0 of every main block stamps the phase
boundaries of layer 8 from %clock64 into 16 free words of shared memory and writes them out when the layer is over;
the scale comes from the block's own first A start of the step to the end of its head against the wall clock (0.52
us per 1000 ticks). The counter has different bases on different SMs (two groups of blocks 6.7e8 ticks apart), so
only intervals inside a block are used. The default build is unchanged by the code (#ifdef only: same spill lines,
3489.9 on the scored workload).
Batch 8, teacher-forced text, final kernel, us from each block's own A start (median over the 128 main blocks):
                         position 150     position 700
  A done                     6.84             6.81
  pairs begun               11.12            11.16
  C done                    16.36            19.87      (blocks with a pair 17.90 / 21.15, their hosts 18.22 / 21.72)
  D start                   17.39            20.98      (hosts 18.66 / 22.14)
  E start (router seen)     22.44            25.88      (hosts 23.05 / 26.55)
  routing done              26.18            29.58
  E done                    52.05            54.47
  F start                   52.44            54.87
  F done                    67.99            70.17
Phase lengths at position 700: A 6.8; from A done to "every pair has begun" 4.4; attention and output projection
8.7 (5.2 at position 150); 1.1; norm, shared gate/up and router 4.9; routing 3.7; expert phase 24.9; 0.4; down phase
15.3. The two streamed phases are 40 of the 70 us (their memory time at 7.4 TB/s is about 35); the 30 us in front
of them are six hand-overs through memory (residual rows, q/k/v, attention output, post-attention rows, router
logits, hidden units), each about 2-2.5 us from the writer's exchange to the reader's value, plus the attention's
and the routing's instructions. The hosts' own rows are 1.9 us behind the others' and the norm and the router wait
for them: the one place left where a known amount is lost, and every way of removing it that I could think of
(the owner computing the host's rows, the owner computing its own with tiles requested during its attention) puts
the same trip to memory behind the last attention output.

## The attention's stages by the cycle counter, and two last experiments on them

tools/bt_clk.py also stamps the attention blocks (columns 128.. of the table). Steps between the stages, median of
the 24 attention blocks, us (sliding layer 8 at position 150 / at position 700 / full-attention layer 9 at 700):
norms, rotation and fragments 1.48 / 1.46 / 1.15; scores 0.68 / 2.50 / 2.70; exponentials and weights 0.80 / 1.26 /
1.68; V mix 0.45 / 1.32 / 1.73; output and K/V stores 0.37 / 0.42 / 0.41: 3.8 / 7.0 / 7.7 us from "q/k/v loaded".
(The first interval of these blocks, from their wait for phase A to "q/k/v loaded", reads 20 us on router blocks and
60 us on the others, which cannot be the wait itself; I did not find out what the stamp in front of it measures and
do not use it.)
- RSTG: the 16 router blocks with a pair stage key groups 16..27 in shared memory like the other attention blocks
  (their staging is put off until the router duty is done, so the fragment buffer is free): same bits, 3489.2 against
  3491.4: nothing, because the pairs on main blocks, which stage nothing, are the last ones anyway.
- K2B: those pairs with two batches of K tiles in flight (both halves of their free staging registers): same bits,
  +10 us with another spill line (160/52/60).
Both are in the source behind switches, off.

## s0071 (track A, cleanup): the s0070 engine with the cycle-counter study build

Expected: no change, 1493 +-6 / 3426 +-30. Measured: valid, 1493 / 3426. `engine/` as committed is this source.

## Inline sentinel waits instead of the call (LDI): nothing

Finding C14 of the Qwen3.8-27B run: a non-inlined device function saves and restores the caller's registers through
local memory. The sentinel wait `ldh` is such a call and is taken every layer at several places. As an inline loop
(batch > 2 kernel only; tools/mk_ldi.cu.bak): in the norms +4 us; in the attention's first loads 0; in the routing's
logit load 0; in the down phase's copy +68 (spill line 144/52/108). Not in the source: the call is not what these
waits cost here (they wait for data that has not arrived; the earlier gain from the inline wait in the copy of the
attention output came from asking for all pending words at once).

## Final engine (s0071 source): long repeat run and the step by position

tools/flake.py 400: 0 mismatches in 399 repetitions of each of the four workloads (step-by-step and generate, batch
1 and 8). tools/pos_time.py, batch 8, us per step by 128-position chunk: 3349 3386 3442 3448 3518 3528 3557 3559 3582
(the kernel of s0061, at the start of this day's work on long contexts: 3737 3790 3835 3866 3906 3925 3953 3962
3998): 390 to 416 us less at every position.

## Inside the phases by the cycle counter, and the sentinels issued by idle blocks (HS_AUX): no

Four more stamps per run in the study build (not kept in the source), position 700, median block, us from its A
start: A norm done 4.64, A mma done 5.80 (A done 6.7); D start 21.0, D norm done 25.3, D mma done 25.5; routing done
30.2, first expert tiles requested and sentinels settled 34.0, expert phase done 55.3; down phase start 55.7, its
fragment copy done 58.5, its rounds done 70.5 (5.8 rounds in 12.0 us: the memory's rate), done 71.8.
So the places with time that is not streaming, instructions or a single hand-over are three: the norm of phase A
(4.6 us) and of phase D (4.3 us), each waiting for the rows of the last block to finish before it and then loading
its pairs of sequences one after the other; the 3.8 us between the routing and the first expert mma (2.3 of it the
first tiles' trip, the rest the sentinel exchanges and their wait); and the 2.7 us of the down phase's copy (the
hidden units of the block that finishes its expert phase last).
HS_AUX, a last attempt at the second: the sentinels of the expert phase's hidden units issued for every slot, row
group and sequence by attention blocks 0..7 once every main block has begun its expert phase (they are idle then
and the memory is only streaming), confirmed, counted in a counter of its own that a main block waits for before
it writes its hidden units; nothing left between the routing and the first tiles. Same bits (bcheck, digest). +71 us
(3491 -> 3562; spill line 144/52/112). With HS_EARLY (+7 when the main blocks issue them during the attention) this
says the six-fold number of exchanges is what costs, wherever it is put. Source: tools/mk_hsaux.cu.bak.
(The eighth norm variant again with its store fixed, bcheck 7 of 7 and the base's spill line: +15 us. The norm does
not want its loads together, before or after its wait.)

## Batch 1 by the cycle counter, and the last experiment (K2X)

tools/bt_clk.py --B 1 --pos 700 (the study build's batch 1-2 kernel has another spill line, 192/32/32, so its layer
is about 5% longer than the engine's; the proportions are what counts). A layer of 28.5-29 us, median block: A 3.2
(norm 1.7); every pair begun 5.6; attention and output projection done 14.0-14.4; D start 15.2-15.5; router seen
17.2-17.5; routing done 19.7-20.2; first expert tiles requested and sentinels settled 22.5; expert phase done 24.7-
25.3; down phase: copy done 26.3, rounds done 28.0, done 28.5-29.0. So at batch 1 and a long context the attention
and output projection are 8.5 of 29 us, the routing 2.5, the wait between routing and first expert tiles 2.8, and
the streaming of the routed weights only about 4.5. The attention blocks' stages there (4 blocks): norms 0.7, scores
2.4, exponentials and weights 1.3, V mix 1.1, stores 0.4.
K2X: key groups 28..32 staged in the unused upper 20 KB of the weight buffer (layers with at most 48 key groups), so
that a full sliding window takes no K tile from global memory in the scores: same bits (bcheck, three digests),
batch 1 1485.2 / 1482.6 against 1485.5 / 1484.4, batch 8 unchanged. The 2.4 us of the scores stage are not the trip
for those tiles. Source with this and the extra stamp sites: tools/mk_k2x_clkx.cu.bak (`-DMK_CLK -DMK_CLK_X=1|2`,
and the same value in the environment for tools/bt_clk.py). `engine/mk.cu` is the source of s0071.
Inside the scores stage (stamps after each round of warp 0, same study source, position 700): the first round
(K tiles in registers) is done 0.7 us after the barrier, the second (K tiles from shared memory) 1.0 us later, the
stage 2.3-2.6 us after the barrier, with or without K2X and at batch 1 and 8 alike. A round is 16 mma in two chains
of 8 that each depend on the one before, plus the masks: about 70-80 instructions, which the 4 ns per instruction
of findings 13f would put at 0.3 us. The rounds are bound by the latency of the dependent mma chains, not by the
tiles' trips (which is why K2X, K2B and RSTG changed nothing), and shorter chains change the order of the sum
(tried as ATT_ACC in the first 24 hours: negative, and not the same bits).

### s0072: the routing's union with two barriers fewer (UNI2)

What a barrier costs was not known. The routing (router seen -> routing done: 3.7 us of the 70 per layer at batch 8,
findings 15l) had five in a row: after the candidates, after the merge, after zeroing the byte table of the union,
after writing it, after the ballots. The table lived where the candidates are (sm/smi[0..255]), so it could only be
zeroed after the merge had read them. UNI2=1 (batch > 2): the table is the first 2 KB of the partial-sum area (free
between phases D and E; the batch 1-2 kernel ranks its scores there already), every thread zeroes one word of it in
stage 1 (and the unit weights), and the six lanes of a sequence's merge warp write their bytes themselves. Two barriers
and one pass less. UNI2=2 also drops the barrier after the ballots: every warp counts the experts in use of the warps
before it by its own ballots (up to 7 more).
Measured on the scored workload in process (tools/abg.sh = build + gen_long.py, card 0, base before and after):
base 3491.0 3495.2 3490.9 3490.7 / 3489.9 3490.5 3491.1 3495.4; UNI2=1 3480.2 3480.5 3480.8 3480.7; UNI2=2 3481.5
3481.4 3481.5 3481.5; batch 1 1481 / 1480 / 1479 (its kernel has the same spill line 176/20/16; not touched).
So a barrier of a 16-warp block on this path is about 0.1 us (two of them: 10 us per step over 47 layers), and trading
the third for seven ballots is worth nothing. UNI2=1 taken.
Checks: bcheck 2..8 bitwise; digest_long 151d4654... / ccbb90c7..., digest1 ffef022d... (the bits of s0058);
flake.py 200: 0 mismatches x4; flake_b: 0; api_stress, long_ctx, wd_test, test_engine --check --long as before;
serve_long on card 3: 1481 / 3472-3475.
Expected from the sealed evaluation: batch 8 3426 -> about 3416, batch 1 1493.
Result s0072: valid, 1491 us per token at batch 1 and 3412 us per step at batch 8 over HTTP (s0071: 1493 / 3426).
Expected -10 at batch 8, measured -14 (two evaluations differ by 2-3 us on their own). The prediction held.

RT8 (after s0072; switch, off): batch > 2, a sequence's 256 router scores on ONE warp (8 per thread; six rounds give
the top 6 in order, no merge stage, no barrier in front of it; the table zeroed before phase D's barrier). Same bits
(bcheck, digest_long). gen_long batch 8: 3494.2 3495.0 3494.9 3499.7 / 3494.2 3494.3 3494.7 3494.7 against
3482.2 3481.8 3481.7 3481.7 / 3484.7 3480.8 3481.2 3480.9: +13 us. Eight sigmoids and rounds over eight values on
one warp cost 0.27 us per layer more than four on two warps plus the merge and its barrier: the candidate stage is
bound by the instructions of ONE warp (the other eight warps idle do not make it faster), so the direction is more
warps per sequence, not fewer, and at batch 8 there are no more.

### s0073: the routing's candidate stage by rank counts instead of shuffle rounds (RSORT=2)

Stamps inside the routing (study build, thread 0 of every main block, batch 8, position 700, layer 8; us since the
block's A start, median): router seen 25.92, logits loaded 26.56, candidates done 28.29, barrier passed 28.30,
merge done and barrier passed 28.87, routing done 29.38. So of the routing's 3.46 us: the load 0.64, the candidate
stage 1.73 (four sigmoids, six rounds of warp maxima; every warp of the block does it), the merge 0.57, the union 0.51.
RSORT=1: each thread sorts its four scores once (six adjacent exchanges on "strictly larger": stable), a round is then
the warp maximum of the first scores and a shift in the winner, which stores its own score and expert (no local
maximum, no slot shuffle, about 50 instructions less of 340). Same bits; gen_long batch 8 3485.8 3480.1 3480.0 3479.9 /
3478.5 3479.0 3479.7 3479.3 against 3481.1 3480.9 3481.4 3486.3 / 3482.8 3482.4 3483.0 3483.2: -2.5 us, a fifth of
what the instruction count said. The rounds are bound by their chain (five shuffles, a ballot, the kill: 30 shuffles
in a row per stage), as the batch 1-2 kernel's were before ROUTE_RANK (s0021).
RSORT=2: no rounds. The thread's best score goes to shared memory, the thread counts its rank R among the 32 of its
warp (8 loads; ties: the lower lane first). The thread's j-th score has at least R + j scores in front of it (the R
better threads' best ones, which also win ties because their experts are lower, and its own j better ones), so only
the first 6 - R scores of the six threads with R < 6 can be among the warp's six best: 4 + 4 + 4 + 3 + 2 + 1 = 18
candidates at fixed places (all 18 are written every time). Lanes 0..17 rank one candidate each among the 18 (largest
first, lower expert on a tie) and the six best go where the rounds put them, in the same order. Scratch: the
partial-sum area behind the union's byte table.
gen_long batch 8: 3466.9 3467.6 3467.7 3467.3 / 3474.0 3468.8 3468.6 3469.1 against 3481.7 3481.3 3482.0 3482.2 /
3481.3 3480.9 3486.5 3482.1: -14 us; batch 1 1481 (not touched; spill lines 144/44/44 and 176/20/16 as before).
Checks on the default build: bcheck 2..8 bitwise, digest_long 151d4654... / ccbb90c7..., digest1 ffef022d..., flake.py
200: 0 mismatches x4, flake_b: 0, api_stress, long_ctx, wd_test, test_engine --check --long as before; serve_long on
card 3: 1481 / 3467 (s0072's build there: 3472-3475).
Expected from the sealed evaluation: batch 8 3412 -> about 3398; batch 1 1491.
Result s0073: valid, 1490 us per token at batch 1 and 3405 us per step at batch 8 over HTTP (s0072: 1491 / 3412;
in process generate 3403, step 3408). Expected -14 at batch 8, measured -7: half. The local A/B was -14 twice on card 0
and -6 on card 3 (serve_long 3467 against 3472-3475), so the sealed number is inside what the cards said; the
prediction should have been the smaller of the two.

### s0074: batch 1-2, the routing's merge over pruned candidates (RKT)

The same stamps at batch 1 (position 700, layer 8, us since the block's A start, median): router seen 16.77, logits
loaded 17.31, candidates done 18.02, barrier passed 18.03, merge done 19.28, routing done 19.35. Of the routing's
2.58 us the merge is 1.25: ONE warp ranks two candidates per lane among all 48 (96 comparisons per lane), while the
other 15 warps wait. (At batch 8 after s0073: logits loaded 26.75, candidates done 28.27: 1.52 where it was 1.73.)
RKT: most of the 48 cannot be among the six best. The merge's order is: larger score first, on a tie the lower key
(key = 2 * slot for slots 0..31, 2 * (slot - 32) + 1 for slots 32..47; slot = 6 * warp + rank in the warp: this is
the tie order of the first engine's rounds, "lowest lane, first candidate before the second", and the bits of every
submission since depend on it). With R = the rank of a warp's best candidate among the 8 best ones in that order,
the candidate (warp, r) has at least R + r candidates in front of it: the best of the R better warps (on a tie their
key is lower than this warp's first key, which is lower than this candidate's) and its own r predecessors. That
needs the keys to ascend with r inside a warp, which holds for every warp but warp 5 (slots 30..35 straddle the two
key ranges). So lanes 0..20 take the 6 + 5 + 4 + 3 + 2 + 1 candidates with R + r < 6 of the warps other than 5, lanes
21..26 all six of warp 5; every lane ranks its one candidate among the 28 places (scores and keys in the area stage 1
used for its scores). A member of the six best has the same rank among the kept ones as among the 48 (nothing pruned
can be in front of it), so the sums are added in the same order.
gen_long batch 1: 1463.5 1463.1 1462.6 1462.6 / 1463.7 1463.2 1462.9 1463.1 against 1480.3 1479.3 1479.2 1479.7 /
1478.8 1478.6 1479.2 1479.1: -16 us (the count of comparisons said -35: the merge is a third of what it was in
instructions and gave back 0.34 of its 1.25 us). Batch 8 unchanged (3468); spill lines unchanged (144/44/44,
176/20/16: the batch 1-2 kernel's allocation survived).
Checks on the default build: bcheck 2..8 bitwise, digest_long 151d4654... / ccbb90c7..., digest1 ffef022d..., digest
da6376ff..., flake.py 200: 0 mismatches x4, flake_b (B=2 added: 0; 7, 5, 3, long generate: 0), api_stress, long_ctx,
wd_test, test_engine --check --long as before; serve_long on card 3: 1465 / 3468.
Exact ties are not in any of these workloads often enough to count on, so the tie order rests on the argument above.
Expected from the sealed evaluation: batch 1 1490 -> about 1474; batch 8 3405.
Result s0074: valid, 1484 us per token at batch 1 and 3408 us per step at batch 8 over HTTP (s0073: 1490 / 3405).
In process at batch 1: step 1365 (1382), generate on the baselines' workload 1464 (1479): -15 to -18 as predicted;
over HTTP only -6 (the HTTP figure of one engine has moved by 4 us between evaluations before: s0062..s0065 were
1494, 1497, 1496, 1492). Batch 8 is the same engine as s0073 (3405 -> 3408: that noise).

RMATCH (after s0074; not in the source, tools/mk_rmatch.cu.bak): the rank counts without the tie term in the loop.
Count the larger scores only (two instructions per comparison instead of five or six); threads with equal scores
have equal counts, so the tie term is `popc(__match_any_sync(count) & lower lanes)` where the tie order is the lane
order, and a rare second loop over the ids where it is not. Three parts, each on another card (tools/abd.sh: build
and gen_long under a time cap):
- bit 0, batch > 2 candidates (both rank counts): 3469.3 3469.6 3470.6 3470.6 on card 3, where s0074's build gives
  3467-3469 over the server: nothing, for about 160 of 390 instructions less per thread. Third time in this code that
  the instruction count did not predict the time (RSORT=1, RKT, this).
- bit 1, batch 1-2 candidates: spill line of the batch 1-2 kernel 176/24/24 instead of 176/20/16, batch 1 1495
  against 1465: +30 us from the allocation, as so often in that kernel.
- bit 2, batch 1-2 merge: spill line 192/48/84, and the run did not finish: my error, the match was written as
  `have && __match_any_sync(0xffffffff, ...)`, a warp-synchronous intrinsic with a full mask under a condition that
  only some lanes pass, which never completes. The first A/B job (all parts in one job, no time cap) held card 0 for
  its whole 1700 s because of it; every later variant test goes through tools/abd.sh.
Not pursued: no part is a gain. `engine/mk.cu` is the source of s0074.

After s0074, batch 1 (study build, same stamps, position 700): router seen 16.33, logits loaded 16.90, candidates
done and barrier passed 17.73, merge's ranks counted 18.30, merge done 18.54, routing done 18.61: the routing is 2.28
us of a 27.1 us layer (was 2.58 of 28.2), its merge 0.81 (was 1.25): rank counts 0.57, then sum, weights and units
0.24. Stamp sites for this: tools/mk_rclk.cu.bak with `MK_CLK_X=4` (tools/clk1.sh).
RTSW (in that file, off): batch 1-2, the weights of the six taken from shared memory instead of the local array
indexed by the lane (what RTS did for batch > 2, -28 us there). Spill line of the batch 1-2 kernel 160/52/88
instead of 176/20/16, batch 1 1519 against 1465 on the neighbouring card: +54 us. The batch 1-2 kernel keeps its
allocation only if nothing in it moves; RKT was lucky (176/20/16 before and after).
`engine/mk.cu` is the source of s0074 (checked against submissions/s0074/src.tgz).

Last measurements of the extension (no change to the engine).
- tools/sig_mono.cu: the router's sigmoid as the engine computes it (`1.0f / (1.0f + __expf(-z))`, the engine's
  compiler flags) over ALL finite float32 in ascending order: 0 inversions among 4,278,190,079 neighbours. It is
  monotone, so ranking raw logits and taking the sigmoid of the candidates only would be exact with a strict
  inequality as its guard (findings, "Where to continue" 00a). In 1 <= |z| < 8, 27.5 M of 50.3 M neighbouring floats
  have the SAME rounded sigmoid: ties between distinct logits are the rule at that distance, so the tie order
  (lower expert) has to be applied to the sigmoids, not to the logits.
- The candidate stage at batch 8 in four parts (tools/mk_rclk5.cu.bak, `tools/clk1.sh 8 5`; its build has the spill
  line 144/52/160, so the whole is 1.26 us where the engine's is about 1.5): sigmoids, sort and the store of the best
  score 0.41 us; rank among the 32 and the 18 stores 0.57; rank among the 18 and the six stores 0.28. Then merge and
  union 1.21.
- RMATCH had taken the tie terms out of both rank counts for nothing, so the 0.57 are not the comparisons. TRIBF
  (the 18 stores without branches: scores that are not candidates go to four places behind the 18): spill line of
  the batch > 2 kernel 144/52/112 instead of 144/44/44, gen_long batch 8 3487.1 3487.6 3487.8 3488.1 / 3486.9 3487.7
  3487.9 3487.9 against 3478.1 3469.6 3469.4 3469.4 / 3468.6 3469.6 3469.4 3469.3 on the same card: +18 us. Not kept
  (the variant is the five lines described here).
- Given RMATCH (160 instructions less per thread: nothing), idea 00a (75 less, 4 candidates more, a fallback path)
  is not expected to pay either; what the stage is bound by is not known. The next step there is the disassembly
  of the stage (tools/sass.sh), not another rewriting.

### s0075: the union's last pass on all 16 warps (UNI4)

SIG4 first (tools/mk_sig4.cu.bak, not kept): the disassembly of the candidate stage (tools/sass_route.py: 472
instructions from the sigmoids to the barrier) shows a branch region per sigmoid division (range test of the divisor,
MUFU.RCP and a Newton step by two FFMA on the fast path, a call otherwise). The fast path written out, with ONE test
for the four divisions: bit-equal to the compiled division for all 2^32 inputs (tools/sig_rcp.cu: 0 differ), and
3467.1 3467.6 3467.8 3467.7 / 3466.8 3468.3 3467.8 3468.2 against 3468.0 3468.8 3469.4 3469.6 / 3468.9 3469.4 3469.5
3469.6: -1.5 us, not worth a line of asm in the engine.
The rest of the batch-8 routing by the cycle counter (tools/mk_rclk6.cu.bak, `tools/clk1.sh 8 6`; this study build
has the engine's spill line): barrier after the candidates 28.04, merge done 28.57, its barrier 28.58, ballots and
barrier 28.67, routing done 29.22. The last pass of the union takes 0.55 us: threads 0..255 (8 warps) each look at
one expert and, for the 40 or so in use, walk the 8 sequences (weights, the two byte tables of the down phase).
UNI4: all 16 warps; threads tid and tid + 256 load the same expert's table entry and count its unit the same way;
the first writes the unit and sequences 0..3, the second sequences 4..7.
gen_long batch 8 on one card: 3468.7 3462.5 3463.3 3463.7 / 3461.6 3463.0 3463.1 3463.3 against 3476.1 3472.2 3469.4
3469.1 / 3468.4 3468.8 3469.6 3469.5: -6 us; spill lines unchanged; bcheck 2..8 and the three digests as before.
Expected from the sealed evaluation: batch 8 3405-3408 -> about 3400 (inside the spread of an evaluation; the
submission is made because the local A/B is clear and the bits are the same); batch 1 1484.
Checks on the default build (tools/vfy.sh, one part per card): bcheck 2..8 bitwise, digest_long 151d4654... /
ccbb90c7..., digest1 ffef022d..., flake.py 200: 0 mismatches x4, flake_b (2, 7, 5, 3, long generate): 0, api_stress,
long_ctx, wd_test, test_engine --check --long as before. gen_long on card 0: 1464 / 3467.5 (s0074's build there:
3468.0-3468.8, so -1 on that card, -6 on cards 2 and 3: serve_long 3462 against 3468).
Result s0075: valid, 1478 us per token at batch 1 and 3397 us per step at batch 8 over HTTP (s0074: 1484 / 3408;
in process: generate 3403, step 3408; batch 1 1364, the same kernel as s0074). Expected about 3400: held. Speedup
against vLLM's trial figures 1.334 / 1.117. `engine/` as committed is the source of s0075.
Step time by position for s0075 (tools/pos_time.py, free-running text, chunks of 128 from position 0), batch 1:
1361 1371 1426 1451 1520 1528 1549 1554 1580; batch 8: 3317 3358 3411 3417 3489 3500 3529 3530 3553 (s0071: 3349
3386 3442 3448 3518 3528 3557 3559 3582: 30 us less at every context length, as a change of the routing should be).

### s0076: the routing weights off the merge's path (UNI5)

After s0075 the merge stage of the batch > 2 routing (0.53 us per layer, warps 0..7, warps 8..15 idle) still did
four things in a row: rank the 12 candidates, add the six scores, divide (a branch region), and write the union's
table bytes, which the next stage (the ballots) waits for. The table bytes need only the rank and the expert, and
both are in the lane that holds a winning candidate. UNI5: that lane writes the byte itself, the merge ends there;
the sum (same order: rank 0 first) and the six weights of sequence b are computed by warp 8 + b in the stage of the
ballots, whose barrier the weights' reader (the union's last pass) is behind anyway.
gen_long batch 8 on one card, A/B/A/B: 3466.4 3460.8 3460.7 3460.3 / 3458.5 3460.1 3460.2 3460.0 against 3469.0
3465.4 3463.0 3463.0 / 3461.2 3461.8 3463.1 3463.2: -3 us (medians 3460.2 / 3463.0); spill lines unchanged; bcheck
2..8 and the three digests as before.
Expected from the sealed evaluation: batch 8 3397 -> about 3394, which one evaluation cannot tell from 3397 (spread
about 3 us at batch 8 between evaluations of one kernel in this series); batch 1 1478-1484. Submitted because the
local A/B is consistent, the bits are the same, and `engine/` has to be a submitted source.
Checks on the default build (tools/vfy.sh on four cards): bcheck 2..8, digest_long, digest1 as before; flake.py 200:
0 mismatches x4; flake_b: 0; api_stress, long_ctx, wd_test, test_engine --check --long as before; gen_long on card 0
1463 / 3464 (s0075's build there 3467.5), serve_long on card 3 1464 / 3458-3462.
Result s0076: valid, 1483 us per token at batch 1 and 3394 us per step at batch 8 over HTTP (s0075: 1478 / 3397; in
process: generate 3402, step 3405; batch 1 1364, the same batch-1 kernel since s0074, whose three HTTP figures are
1484, 1478, 1483). Expected about 3394: held, for what a 3 us step is worth in one evaluation. Speedup against
vLLM's trial figures 1.329 / 1.118.

## End of the extension (36.00 agent-hours)

`engine/` as committed is the source of s0076: track A, the bits of s0058 (digest_long 151d4654... / ccbb90c7...,
digest1 ffef022d...), 1483 us per token at batch 1 and 3394 us per step at batch 8 over HTTP; the goal of the
extension was 3470 at batch 8 without giving up batch 1 (1520 at its start). The last five submissions (s0072-s0076:
3426 -> 3394, 1493 -> 1478-1484) are all in the routing's top-6 selection and all came from cycle-counter stamps
inside it during the last three hours. What is left there, and what was tried and dropped (RT8, RSORT=1, RMATCH,
RTSW, TRIBF, SIG4), is in findings 15m and "Where to continue" 00.

The final engine's batch-8 routing by the cycle counter (s0076's source with the stamps of tools/mk_rclk6.cu.bak,
`tools/clk1.sh 8 6`, position 700, layer 8, median block; this study build keeps the engine's spill line 144/44/44):
router seen 25.63, barrier after the candidates 27.78, merge done 28.15, ballots (and weights) done and barrier
passed 28.27, routing done 28.81: 3.18 us (3.46 before s0072), of which load and candidates 2.15, merge 0.37 (0.53
before UNI5), ballots 0.11, last pass and its barrier 0.54. The last pass did not get shorter by these stamps
although gen_long gave 6 us per step for it on two cards: the stamps are thread 0's, and what the 16-warp split
shortens is the walk of the warps that hold experts in use. Its remaining parts: the prefix of the 8 per-warp counts
(a loop of up to 7 loads on warps 7 and 15), the divergent block of the experts in use, the final barrier.
The layer: D start 20.68, expert phase done 53.76, down phase done 69.42.

### s0077: the union's last pass without its loop over the per-warp counts (UNI6)

In the last pass every thread adds the counts of the warps before its own: a loop with a trip count that differs
per warp (up to 7 loads, compares and branches on warps 7 and 15, the ones the final barrier waits for). UNI6: two
vector loads of the 8 counts and seven selects. gen_long batch 8 on one card, A/B/A/B: 3462.0 3455.5 3455.8 3455.8 /
3454.7 3456.4 3456.0 3456.1 against 3465.7 3462.9 3459.8 3460.2 / 3458.9 3460.1 3459.8 3460.0: -4 us; spill lines
unchanged; bcheck 2..8 and the three digests as before.
Expected from the sealed evaluation: batch 8 3394 -> about 3390; batch 1 1478-1484 (kernel not touched).
Checks on the default build (tools/vfy.sh, four cards): bcheck 2..8, digest_long, digest1 as before; flake.py 200: 0
mismatches x4; flake_b: 0; api_stress, long_ctx, wd_test, test_engine --check --long as before; gen_long on card 0
1466 / 3464 (s0076's build there: 3464: nothing on that card), serve_long on card 3 1465 / 3455 (3458-3462).
Result s0077: valid, 1479 us per token at batch 1 and 3397 us per step at batch 8 over HTTP (s0076: 1483 / 3394).
In process: generate 3400.4 (3401.6), step 3403.9 (3405.4). Expected 3390 over HTTP: NOT confirmed; the headline
moved by +3, the in-process figures by -1.5. A change of 4 us on one card and 0 on another is below what the sealed
evaluation resolves; s0076 and s0077 are the same engine to its precision. `engine/` as committed is the source of
s0077 (the last valid submission; bits of s0058). Best figures of this engine family over HTTP: batch 1 1478
(s0075), batch 8 3394 (s0076).

## Final state (clock 35.90 of 36.00)

`engine/` = source of s0077 (checked against submissions/s0077/src.tgz), supersedes the "End of the extension"
paragraph above, which was written one submission earlier. Track A, bits of s0058. Sealed, over HTTP: s0075 1478 /
3397, s0076 1483 / 3394, s0077 1479 / 3397: one engine to the evaluation's precision, 1.33x / 1.117x of vLLM's
trial figures (1971 / 3796), 73-76 us inside the extension's goal of 3470 at batch 8. Nothing is running in the machine.
