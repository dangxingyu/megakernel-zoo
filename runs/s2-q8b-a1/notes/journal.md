# Journal

One dated entry per idea: what you expected, what you measured, what you concluded.

## 2026-11-09 — s0001 starter (HF reference)

Expected: valid, ~20-30 ms. Measured: valid, 21051 us B=1 (MBU 0.215), 20836 us B=8; 2217 launches/step,
84% GPU idle. HF bf16 error vs float64: rel L2 median 0.0101, max 0.078; my limit is 2x that (median 0.0203,
max 0.156). Target: 6081 us (B=1), 6434 us (B=8). Bandwidth bound 4520 us.

## The exact computation of one decode step (Qwen3-8B, from HF modeling_qwen3.py + config)

Config: hidden H=4096, layers L=36, heads 32, kv heads 8 (GQA group 4), head_dim 128, MLP inner 12288,
vocab 151936, rms eps 1e-6, rope theta 1e6, no biases anywhere, lm_head NOT tied (separate 151936x4096 bf16).
All weights bf16.

Per sequence b at position t (t = number of tokens already seen), token id tok:

    x = embed[tok]                                      # [4096]
    for l in 0..35:
      h  = RMSNorm(x; w_in[l])                          # y = w * (x * rsqrt(mean(x^2) + 1e-6)); HF: x->f32, norm, ->bf16, * w
      q  = Wq h   [4096]  -> 32 heads x 128
      k  = Wk h   [1024]  ->  8 heads x 128
      v  = Wv h   [1024]  ->  8 heads x 128
      q  = RMSNorm_128(q_head; w_qn[l])   per head (same [128] weight for every head)
      k  = RMSNorm_128(k_head; w_kn[l])   per head
      RoPE on q,k (per head, d=128): inv_freq[i] = theta^(-i/64), i=0..63; ang = t*inv_freq[i]
          out[i]    = x[i]*cos(ang_i)    - x[i+64]*sin(ang_i)
          out[i+64] = x[i+64]*cos(ang_i) + x[i]*sin(ang_i)       (rotate_half = cat(-x2, x1))
      append k,v to cache at position t -> K[l][b][0..t], V[l][b][0..t]   (8 heads x 128 each)
      for head hq in 0..31 (kv head g = hq // 4):
          s[j]  = (q[hq] . K[g][j]) / sqrt(128),   j = 0..t     (causal mask trivial for one new token)
          p     = softmax(s) (float32 in HF, then cast to bf16)
          a[hq] = sum_j p[j] * V[g][j]                            # [128]
      x = x + Wo a                                       # a: [4096] = 32 heads x 128 concatenated
      h2 = RMSNorm(x; w_post[l])
      g  = Wgate h2 [12288];  u = Wup h2 [12288]
      m  = silu(g) * u                                   # silu(z) = z * sigmoid(z)
      x  = x + Wdown m                                   # Wdown: [4096, 12288]
    logits = Wlm RMSNorm(x; w_final)                     # [151936]

Bytes read per step: per layer q 33.5 MB + k 8.4 + v 8.4 + o 33.5 + gate 100.7 + up 100.7 + down 100.7 = 386 MB;
x36 = 13.9 GB; lm_head 1.245 GB; total 15.14 GB. KV at context 192: 28 MB (B=1), 226 MB (B=8).

Precision plan: keep the residual stream x in float32, accumulate every matvec in float32, round the *inputs*
of each matvec to bf16 (that is what the bf16 model does anyway: norm output, attention output, silu*up are
bf16 in HF). This should be at or below the HF bf16 error (HF additionally rounds every matvec output and the
residual stream to bf16).

Structure of the cost: 4 matvec phases per layer (QKV fused: 6144x4096 sharing one input; O: 4096x4096;
gate+up fused: 24576x4096 sharing one input; down: 4096x12288) + lm_head = 145 serial phases; everything
else is tiny. Each phase needs the complete output of the previous one. So the game is: (1) read weights at
close to HBM speed inside a phase; (2) lose as little as possible at the 145 phase boundaries.

Matvec design decision: tensor-core MMA m16n8k16 (bf16 x bf16 -> f32 accumulate). A = 16 output rows x 16
input dims of W, B = 16 input dims x 8 batch columns: the N=8 of the instruction is exactly the batch. One
code path for B=1 and B=8 (unused columns zero), each column computed independently of the others -> batch
invariance bitwise by construction. Weights re-laid out at load so a warp reads 512 contiguous bytes per MMA
(each thread one 16-byte load).

## 2026-11-09 — experiment e0005: raw HBM streaming (tests/bw.cu)

8 GB buffer, one launch. Plain 16-byte loads + xor: 132 blocks x 1024 threads = 3005 GB/s (89.7% of 3350);
2 blocks/SM x 1024 = 3100 GB/s (92.5%). The MMA matvec (16-row tile per warp, x fragments in shared memory)
reaches 3020 GB/s (90.2%) with 16 or 32 warps per SM. So the raw ceiling for this design is ~90% MBU = 5.0 ms;
the target (6081 us) leaves ~1 ms for the 181 phase boundaries, attention and small ops.
Throughput scales with the number of warps that have a load in flight (not with loads per warp): a single warp
streams only 1-4 GB/s, so every phase has to be spread over all warps of all SMs, including the small
matrices (O is 33 MB = 11 us at full rate; one warp alone would need 35 us for ONE 16-row tile).
Decision: "block team" matvec: a block (SM) takes whole 16-row tiles, its warps split the K range of the tile,
partial accumulators are summed in shared memory in a fixed order (deterministic, independent of batch).

## 2026-11-09 — s0002: own CUDA operators, one launch per operator (rung `correct`)

Design as built (engine/mk.cu): 132 blocks x 1024 threads, 229 KB shared memory per block. Per phase: entry
(RMSNorm computed by every block for itself from the f32 residual stream, or copy of bf16 fragments written by
the previous phase) puts the matvec input into shared memory as MMA B-fragments; 31 loader warps split K of
each 16-row tile, warp 31 sums the 31 partial accumulators in fixed order and runs the epilogue (store qkv /
residual add / silu(gate)*up -> bf16 fragments / logits). gate and up rows interleaved 8+8 in one tile so the
MLP activation happens inside one thread. Attention: block per (sequence, kv head); K cache stored as MMA
A-fragments (scores = MMA over K), V cache stored transposed (output = MMA over V with bf16 probabilities).
Reads of data written by other blocks use ld.cg (L1 is not coherent across SMs).
Unit test (tests/test_engine.py, vs HF float32): rel L2 median 0.0042 (HF bf16: 0.0093), bitwise equal alone
vs batch of 8.
Expected: valid, 7000-7500 us. Measured: valid, 7366 us B=1 (MBU 0.615), 8062 us B=8; rel L2 median 0.0041
vs float64 (limit 0.0203), top-1 0.988 (HF bf16 0.951), batch bitwise equal. Launch gaps cost about 0.9 ms
compared with the single-launch mode measured in the unit test (6459 us).
max_context of the evaluator: see engine_stats.

## 2026-11-09 — s0003: one launch per step (rung `single_launch`)

Expected 6450 us B=1 / 7300 B=8. Measured 6465 / 7353 (MBU 0.700 / 0.624). Prediction confirmed.
Beats MPK and SGLang at B=1, not yet vLLM at B=8 (7077). Evaluator max_context = 1024.
Ideal for this matvec at raw streaming rate is ~5010 us, so ~1450 us (B=1) and ~2340 us (B=8) are lost at
the 181 phase boundaries, in entries/epilogues and attention. Next: measure per phase inside the kernel
(bench kernel with an op mask) instead of guessing.

## 2026-11-10 — per-phase costs, L2 behaviour, and a tooling trap

Bench kernel (tests/bench_ops.py, op mask inside one launch), B=1, us per phase above the barrier cost
(ideal at 2905 GB/s in brackets): qkv 24.0 (17.3), attention 8.2 (0), o 15.2 (11.5), gate+up 77.0 (69.3),
down 41.2 (34.6), lm 429 (429); barrier 0.71 us each (129 us/step). B=8: qkv 35, attn 10.1, o 16.4,
gate+up 89, down 43.4. So steady-state streaming is fine (lm_head runs at 2905 GB/s); the loss is ~36 us
per layer at phase boundaries, and at B=8 the per-block RMSNorm entry costs ~12 us (x2 per layer).

tests/bw2.cu: (1) streaming is capped globally (~2.7-3.0 TB/s) but one SM alone can pull 34 GB/s (1.5x its
fair share of 22.8) -> uneven tile counts cost little; (2) L2 hits: 57 GB/s per SM when every SM reads its own
region (total <= 35 MB), 84 GB/s when all read the same 128 KB; a 69 MB working set already misses.
So L2 can hold about one O matrix (33.5 MB), and re-reading from L2 is only ~2.5x faster than HBM per SM.
(3) cache operators ca/cg/cs/lu make no difference for streaming.

Trap: runtime ablation branches in the kernel (skip entry / matvec / epilogue) slowed the whole step by
420 us on their own, and the first version of the zero-fill removal looked 500 us slower because of them.
Disassembly shows why the code is fragile: per phase every one of the 32 warps runs a 64-bit division call,
I2F/RCP divisions and a remainder-unrolled loop prologue. An SM issues few instructions per cycle, so
per-phase scalar overhead executed by all 1024 threads is real time (1 us = 1700 cycles).
Rule from now on: whole-step A/B only (tests/ab.sh builds variants and times them interleaved, same
protocol as the evaluator; it reproduces s0003 to 0.1%).

## s0004: down entry without zero fill
Expected -80..-120 us at B=1, B=8 unchanged. Measured 6385 (-80) / 7340 (-13). Confirmed.

## s0005: lean phases (v2a)
Expected 6135 / 7320. Measured 6139 / 7320 (MBU 0.737). Confirmed.
Numerics note: this build differs from s0004 only in the (fixed) order in which partial sums are added, yet
its logits differ from s0004's by rel L2 4e-3, as much as either differs from float32. Cause: every matvec
input is rounded to bf16, so a 1e-7 perturbation flips a few bf16 roundings (4e-3 each) and that noise is
what the error consists of. Both builds are deterministic and bitwise batch-invariant. Consequence: error vs
float64 is a noise floor (median ~0.0042, max 0.02-0.04 on the evaluator's sequences, limit 0.156); every
build that changes summation order draws a new sample of it. Compare builds by timing, not by tiny error moves.

## s0006: RMSNorm fused into the previous epilogue (v2b)
Expected 5950 / 6420. Measured 5951 / 6422 (MBU 0.761 / 0.714). Confirmed; rungs parity and h0_step passed
(1.124x SGLang at B=1, 1.102x vLLM at B=8 — B=8 margin is thin).
Next: rung device_loop (generate() with argmax + feedback in the kernel), then phase boundaries.

## s0007: generate() — device loop
Expected generate_step_us 5900 / 6350, step path unchanged. Measured generate 5886 / 6340, step 5947 / 6418;
gate generate: 8/8 sequences identical to step-by-step greedy. All ladder rungs passed. Headline is now the
generate path. Trace (step path, B=1): 1 launch/step, 0.0078 host syncs/step, GPU idle 0.06%.

Where the remaining time is (B=1, per layer ~151 us vs 128 ideal at 2905 GB/s): attention 8.2 us during
which HBM is idle; ~3 us of non-streaming time at each of the 4 matvec boundaries (drain, reduce, barrier,
entry); barriers 0.7 us x 5.
Plan: (C) while the 8B attention blocks work, the other blocks touch their O tiles (prefetch into L2: O is
33.5 MB, L2 holds it), and the O tile table gives attention blocks fewer tiles; (D) the same idea at the other
boundaries (first tile of the next phase requested before the barrier).

## 2026-11-10 — O prefetch during attention: small gain, and what the phases really cost (not submitted)

Hypothesis: idle SMs touch their O tiles (L2 prefetch) during attention, attention blocks get fewer O tiles;
expected -250 us. Measured (tests/ab.sh): B=1 5946 -> 5907 (-40), B=8 +80..+110 with 3 tiles per idle block,
-75 with 2. Hypothesis mostly wrong.
Why (phase-repeat builds, -DREP_x=2 runs phase x twice per layer; delta/36 = in-place cost, B=1 / B=8):
attention 6.7 / 7.3 us; O second run (its weights are then in L2) 9.1 / 10.6 us; QKV 20.3 / 22.1;
gate+up 71.2 / 73.6; down 37.3 / 41.9. So O costs 9 us even with every byte in L2: 2 tiles from L2 are
~4.8 us, the other ~4 us is a fixed latency chain per phase (drain, reduce, epilogue with L2 round trips,
fence, atomic, poll, entry copy, latency of the first loads). 4 matvec phases x 4 us x 36 = 580 us per step.

tests/bw3.cu: a warp continues to execute while its load is in flight (compute fully overlaps the load), but
loads of ONE warp are serialized (8 back-to-back loads take 8 x 130 ns unloaded). An SM issues ~5
warp-instructions per cycle at most.

Decision: software-pipeline the weights across the phase boundary: after its last partial sum a loader warp
immediately requests its first 8 ksteps of the NEXT phase's first tile into registers; the data flows during
reduce / barrier / entry and the first MMA of the next phase finds it there. Needs everything inlined in one
function (32 registers live across the barrier).

## 2026-11-10 — register preload across the barrier: wrong (not submitted); software clock trace

v3 (tests/mk_v3_preload.cu): everything inlined, loaders request the next phase's first tile before the
barrier. Expected -400 us; measured +90 us (B=1 6044 vs 5952), +60 of it from the cross-phase preload itself.
Explanation consistent with the data: the SM's load path is shared; the coordinator's small loads (epilogue,
atomic, poll, entry copy) queue behind 31 warps x 8 pending weight loads, so the latency chain is not hidden
but pushed behind the preload. Reverted to the s0007 structure.

Trace tool (build -DTRACE, tests/trace.py): the last SM runs a counter in global memory (17.6 ns/tick), each
block's coordinator thread timestamps 6 points per phase in one layer. B=1, layer 17, us:
  qkv:     first tile 5.9, rest 12.5, reduce+epilogue+fence 0.74, atomicAdd 0.92 (!), wait 1.6;
           last block done 20.5 -> barrier released 23.5
  attn:    body up to 9.3 on the 8 attention blocks; phase 21.7 -> 32.2
  o:       entry 0.9, tiles 6.0 + 6.0, reduce+epilogue+fence 1.54, atomic 0.83; done 47.2 -> released 50.0
  gate+up: entry 1.7, 68 us of tiles, r+e+f 1.05, atomic 1.0; blocks with 11 tiles finish 6 us early (spread 13.6)
  down:    entry 1.8 (5.6 at B=8), tiles 17.5 + 15.6, r+e+f 1.7, atomic 1.35; done 163.1 -> released 165.3
  layer 165 us in the traced build (131 working blocks) vs 133 ideal.
Findings: (1) the barrier costs 2-3 us after the LAST block has finished: 131 atomicAdds on one counter
serialize (0.9 us mean) and polling adds more; (2) epilogues with an L2 round trip (residual read) take 1.5-1.7;
(3) entries 0.9-1.8 us at B=1, 2.4-5.6 at B=8; (4) attention 9.3 us.
Next: barrier without atomics (one byte flag per block, plain store; the coordinator warp polls all flags
with one 128-byte warp load + vote).

## s0008: SMs idle during attention stream their first O tile into shared memory
Path there: (1) flag barrier instead of the atomic counter: 50 us slower (and my first version had a real
deadlock: `==` instead of "has reached" on the flags; the trace build caught it); (2) register preload of the
first O tile: +400 us, the compiler spilled the 32 registers to local memory (LDL/STL = memory traffic);
(3) same bytes into shared memory instead (the 131 KB behind the K=4096 fragments is free outside the down
projection): works. Unused-batch-column lanes of the fragments may hold garbage: a column of the MMA only
depends on its own column of B, and unused output columns are never stored -> removed the scratch wipe.
Expected generate 5810 / 6330; measured 5810 / 6332 (step 5873 / 6413). Confirmed.
Trace: O's first tile now takes 1.3 us instead of 5.7, but attention got ~2 us slower: its K/V loads (HBM)
compete with 124 SMs flooding HBM. Net -2.1 us/layer.
Next: attention loads — copy the K/V history into shared memory with all 32 warps right at the start
(6.5 sequential loads per warp instead of 8 + 4, overlapping the q/k norm work), MMAs then run from shared
memory; delay the stash flood slightly so that attention's small loads go first.

## 2026-11-10 — attention K/V image copy (dropped); steady-state streaming is NOT the problem

Attention variant: all 32 warps copy the K/V history into shared memory first, MMAs from shared memory.
Bitwise identical to the old path (also beyond the image capacity, 420 positions), but 5951 vs 5872 us.
Dropped (tests/mk_att_image.cu).

lm_head-only benchmark (tests/bench_lm.py, -DBENCH_LM): engine matvec 2.942 TB/s; without reduce/epilogue
2.984; without the per-tile block barrier 2.988; whole tile per warp 2.942. So the block-team loop loses
nothing to its per-tile barrier and runs at the practical HBM limit (~3.0 TB/s = 89% of 3350). At that
rate a layer is 129 us; measured 151 us. The whole remaining loss (~22 us/layer, ~790 us/step) is phase
boundaries (4 x ~2.5 us of latency chain with HBM idle), attention (~9 us, 4.4 of it now hidden by the O
stash) and uneven tile counts (~3.8 us).
Plan: (a) coordinator requests the epilogue inputs (residual rows, next norm weights) before the tile's
barrier instead of after; (b) per-warp private input copy issued together with the first weight loads (no
block barrier after the entry); (c) fractional tiles for balance.

## s0009: epilogue inputs requested before the tile barrier
Expected generate 5795 / 6235. Measured step 5859 / 6315 (as predicted from the A/B), but generate 5890 at
B=1 (+80 vs s0008), so the headline did not improve. Prediction for generate wrong: I had only A/B-timed step().
tests/time_step.py now times both paths.

## 2026-11-10 — why equivalent builds differ by 50-400 us: local memory is an L2 round trip here
Chasing the s0009 generate regression: same algorithm, k_step and k_generate copies of the same functions,
timings between 5811 and 5896. Experiments:
- control state (layer, barrier index, step) moved to shared memory so the top level keeps nothing across calls:
  no gain by itself (5889/5896);
- ONE generic phase function for all matvecs (runtime ks/epilogue): correct, small, 7906 us: 50 LDL / 58 STL,
  i.e. the compiler spilled in the hot loop because loaders and the coordinator shared one register allocation;
- loader loop and coordinator loop as separate functions: 6726 (3 LDL per tile in the loader); register diet
  -> zero spills: 6048; groups of 8 loads spill again (6237), groups of 2: 6012;
- phase descriptor computed once by the coordinator thread and kept in shared memory instead of ~100
  instructions per thread per call: 5920;
- no calls at all (one interpreter loop over descriptors, everything inlined once): 5781 / 5774. Best so far.
Mechanism: shared memory takes 229 of the SM's 256 KB, so L1 is nearly gone and every local-memory access
(spill, saved return address of a call) goes to L2, queued behind the warp's own weight loads; done by 1024
threads at every phase boundary it is on the critical path. The "+420 us from ablation flags" mystery of the
first day was the same thing.
Rules: (1) check `tests/sass_spills.py` after every change: no LDL/STL outside init; (2) no calls on the
per-phase path; (3) per-phase work that every thread repeats must be a handful of instructions: compute
once in the coordinator thread, publish in shared memory; (4) roles with different register needs get
different code.

## s0010: interpreter kernel
Expected headline 5775, B=8 6260. Measured: generate 5771 / 6236, step 5779 / 6258 (MBU 0.783).
1.159x SGLang at B=1, 1.135x vLLM at B=8.

## s0011 / s0012 and the attention experiments
s0011 (1/rms on warps without copy loads): expected 5755, measured 5756. s0012 (idle SMs touch the layer's
K/V history into L2 before the O stash): expected 5697 / 6195, measured 5694 / 6207 (step 5702 / 6228).
Trace inside attention (attention blocks, incl. ~0.5-1 us per timestamp read): q/k/v prep 2-4 us, scores
~1.2, softmax ~2, V sum + store ~1.9. The K/V "image copy into shared memory" variant retested with K/V in
L2: still slower (5735 vs 5706); removed for good. New prep stage: one warp per vector, float4 per lane,
RMSNorm by warp reduction and the RoPE partner by shuffle (lane ^ 16): no block barrier in the stage;
local A/B -35 us at B=1.

## A numerical tail to keep in mind (not a bug)
My unit-test set has one position (code text, row 1, pos 27) where every build is at 0.13-0.16 rel L2 while
HF bf16 is at 0.03-0.07. Layer-by-layer comparison against HF float32 (tests/layer_err.py, last_layer.py):
through 35 layers my residual error is half of HF's (0.009 vs 0.009.. 0.016); in the LAST layer one MLP
hidden unit is huge, and the bf16 rounding of silu(gate)*up (HF rounds the same quantity) shows up as an
error along that unit's output column (dims 2109, 2284, 2652: mine -15/-13/+10, HF +7/+5/-5: same direction,
both inside half an ulp of the big value). After the last layer nothing averages it out. So the max over
positions of my error is a draw from bf16 rounding of rare huge activations; the evaluator's set has shown
max 0.022-0.040 against a limit of 0.156 in every submission. If that max ever comes close to the limit:
give the last layer's down projection a second pass with the low part m - bf16(m) (two bf16 fragments;
+34 us/step), or correct only the outlier units.

## s0013, s0014 and two dead ends
s0013 (attention prep without block barriers): expected 5660, measured 5659 (B=8 gen 6223).
s0014 (input copy: early exit / single-load path / loads in pairs): expected 5653 / 6160, measured 5651.
Dead end 1: per-SM streaming rates. Trace shows the same blocks 10-19% slower in every phase of a run
(corr 0.9), but the pattern changes between engine instances and tile counts proportional to a measured
table made things 100 us WORSE. Not a static SM property I can calibrate; would need dynamic claiming.
Dead end 2: requesting 4 or 6 copy loads at once: +60..+185 us. The unrolled conditional code is executed by
every thread at every phase. Calibration from this: ~20 extra instructions per thread per phase cost
~0.5 us per phase (~100 us per step). An SM does NOT issue 5 warp-instructions per cycle in this regime;
closer to one. So the per-phase "latency chain" is to a large part plain instruction count on 32 warps:
next step is an instruction audit of everything all threads execute at a phase boundary.

## s0015: weight loads with the last-use cache operator
While auditing the SASS of the loader loop (weight loads were generic `LD` because the pointer comes from
shared memory) I tried explicit cache operators on the weight loads: ca 5649 / 6184 (same as generic),
cg 5618 / 6128, cs 5637 / 6100, lu 5602 / 6047 (generate us/step, B=1 / B=8). The early microbenchmark had
shown "no difference between operators" for raw streaming, which is true, but in the kernel the stream
competes in L2 with the data the phases share (fragments, residual, K/V). Expected 5602 / 6047; measured
5595 (step 5600 / 6061). lu on the other single-use loads: no measurable effect.

Trace after it: tiles are faster for most SMs, but a few stragglers set the end of every phase: block
finish spread 13-22 us, mean wait at the barrier 8-11 us per phase. Static assignment is now the largest
loss. Plan: dynamic claiming. Every block starts with tile = block index (no claim storm, and the O stash
knows its tile), the coordinator claims the next unit with an atomic add on a per-phase counter while the
loaders stream (the claim is pipelined, off the critical path). The down projection (K = 12288, 393 KB
tiles) is claimed in K-thirds, so its unit equals a K=4096 tile and ONE loader loop serves all matrices;
the three thirds of a row tile are combined through a small global buffer in fixed order by whichever
block publishes last (CAS on a tagged 3-bit mask).

## 2026-11-10 — dynamic tile claiming: works, but no gain (not submitted)
Built it fully (tests/mk_dyn3r.cu, mk_dyn1r.cu): first unit of a block fixed (= its index), further units
claimed by the coordinator with an atomic add while the loaders stream (double-buffered slot in shared
memory), single loader loop for all matrices. Correct on the first run (bitwise batch equality, generate
identical). Results, generate us/step B=1 / B=8 (s0015 static: 5597 / 6043):
- down projection claimed as K-thirds with cross-block partials (store 512 B + fence + CAS per unit): 5730 / 6126
- claims mapped through a stride permutation to spread the active units over the matrix: worse than a
  contiguous moving window by 60-75 us (locality helps, the opposite of my guess)
- down projection as whole tiles in three rounds, window order: 5618 / 6018
- same with one round of 24 loads per unit: 5731 (the generic loop spills)
- the new loop with static ranges but without the special O table: 5730 (attention blocks then have 2 cold
  O tiles: that table is worth ~130 us)
So claiming is worth +21 / -25 us: nothing. Reason: static counts were already balanced to within one tile
and a claim-based scheme also ends with every SM finishing one last unit; the tail can only shrink with
units smaller than a tile at the end of a phase.
Bugs found on the way: (1) my single-run A/B script compiled the current source whatever SRC said, which
made s0015 look 140 us slower for an hour ("environment drift" that did not exist); (2) the phase
descriptor grew into the control block in shared memory (hang at step 1): layout now has static_asserts.
Also: the software trace clock is not a stable time base (tick rate depends on memory traffic); the pod was
replaced once (epoch 2) and reproduces the same timings.

## s0016: pipelining across the tile barrier
New target on the board since today: 5515 us at B=1 / 6215 at B=8 (1.10x SGLang 0.5.18 + torch.compile).
Long-context check (tests/test_long.py, 700 positions vs HF float32): median error 40% of HF bf16's,
top-1 >= 0.994, bitwise equal alone vs batch.
Experiment: at a tile's block barrier the loaders' load queues run empty. Requesting part of the next tile
before the barrier (loads do not block a warp): 2 loads + the extra kstep: 5600 -> 5525 / 6060 -> 5962.
Every attempt to go deeper spills: 3 or 4 loads in separate registers (11-22 spill instructions, 5583..5670),
a ring of 8 registers refilled after each MMA (hundreds of spills, 6086), 8 loop-carried registers refilled
after the group (139 spills, 5957). The spills land in attention and coordinator code, not in the loader: the
kernel is one function and the allocator's choices are global. Isolation by a call costs ~0.8 us per phase
(measured earlier), more than the deeper pipeline can give. Expected 5520; measured 5527.7 (step 5531 / 5975).

## s0017: pipeline after the stash tile
Expected 5510; measured 5514.5 (step 5518 / 5960): under the new target by half a microsecond; the evaluator
reads ~5 us higher than my harness. Need margin.
Next: no block barrier inside a phase at all. Loader warps write their partial, bump an arrival counter in
shared memory (one per partial buffer) and go on with the next tile; the coordinator waits for the counter,
reduces, releases the buffer by publishing how many tiles it has reduced; a loader only waits if it is two
tiles ahead of the coordinator. One block barrier per phase remains (phase end).

## 2026-11-10 — two more structural experiments without gain (not submitted)
On top of s0017 (5509.5 / 5948 generate):
- no block barrier inside a phase (arrival counters in shared memory, loaders never wait for each other;
  tests/mk_nobarrier.cu): bitwise identical results, 5505 / 5975. Sleeping instead of spinning in the
  coordinator's wait (`__nanosleep`): worse the longer the sleep (+7 .. +138 us).
- input copy private to each warp, no barrier after the entry (tests/mk_private_entry.cu): 5508 / 6080.
Conclusion: with the 2+extra pipeline in place, the tile barrier and the entry barrier cost nothing
measurable. What is left per layer (~12 us over the 129 us of pure streaming): attention net of the stash,
the end-of-phase tail (blocks waiting ~2 us on average for the slowest SM) and the chain reduce -> stores ->
fence -> atomic -> poll -> input copy -> first loads.
Next: the tail. SM speed differences are stable within an engine instance (corr 0.9) but differ between
instances, so: measure barrier wait per block at engine start and give slow SMs fewer gate+up / lm_head
tiles (those two phases have enough tiles per block for whole-tile corrections). Any tile-to-block table
gives bitwise the same logits, because a tile's arithmetic does not depend on the block that runs it.

## 2026-11-10 — barrier wait statistics (tests/calib.py)
New instrumentation: poll iterations of each block's coordinator at the grid barrier, per phase kind (recorded
only when calibration is switched on). One gate+up tile is 15.4 polls, so a poll is ~0.38 us. Mean wait per
phase at B=1: qkv 2.3 us, attention 7.5 us (the stash blocks), o 0.9 us, gate+up 5.3 us, down 3.4 us.
gate+up: blocks with 11 tiles wait 9 us, blocks with 12 tiles 3.1 us (2.0 .. 4.5 us). Moving whole tiles from
blocks that wait least to blocks that wait most did not change the step time: with 11 vs 12 tiles the grain
IS the imbalance. So the remaining lever is fractional tiles: every block gets total/132 tiles to within 1/31
tile (units = warp columns), boundary tiles are computed by two blocks and joined through global memory.
Expected: gate+up -2 us, down -1 us, qkv -0.5 us per layer, about -130 us per step.

## 2026-11-10 — fractional tiles: correct, no gain (not submitted) — imbalance is not a loss
tests/mk_fractional.cu: unit ranges of 1/31 tile per block, boundary tiles computed by two blocks and joined
through a 512-byte global record with a phase tag. Bitwise batch equality and generate identical. Timing:
5588 / 6055 with fractional tables, 5581 / 6048 with whole-tile tables in the same code (and that code is
70 us slower than s0017 because of the per-tile ownership tests in the loader loop).
With the dynamic-claiming result this closes the question: block finish times differ by several us, but
that is not lost time. HBM is the shared limit; when some SMs finish, the rest get their bandwidth. The
"wait for all" numbers of the trace measure slack, not waste. Wrong hypothesis, twice; ~5 hours.
What is left is real idle time of HBM at phase transitions and during attention.

## s0018: end-of-phase L2 prefetch
`prefetch.global.L2` (inline PTX) produces no value, so the warp does not wait; the disassembler prints it as
NOP, and a discarded `ld` is removed by the compiler, which is why I had not found a non-blocking load before.
At the end of a phase each loader warp prefetches its first ksteps of the next matvec phase (address
published by the coordinator in the phase descriptor). Depth 1 / 2 / 3 / 4 / 8: 5478 / 5449 / 5455 / 5476 /
5612 us (5520 without): two is best; more makes the coordinator's chain wait behind the prefetches (the
same effect that sank the register preload on day 1). Prefetching more of the NEXT TILE before a tile
barrier, beyond the 2+extra in registers: 2 -> -4 us, 4 -> +35, 6 -> +107: no use.
Expected 5455; measured 5454.8 (step 5459 / 5937). MBU 0.829.

## s0019, s0020: prefetch in the other idle windows
s0019: stash blocks prefetch 4 ksteps of their second O tile during attention (-36 us at B=1), attention
blocks prefetch 8 ksteps of their cold O tile when done (-34 us at B=8), 4 ksteps of the first q/k/v tile
during the embedding phase (-3). Expected 5413; measured 5421.6 (the evaluator reads ~10 us above my harness).
Not helpful: attention blocks prefetching their own K/V history (worse than the idle blocks' blocking touch:
5501 vs 5445), the K/V touch as prefetch instead of load (+17 at B=1, -14 at B=8), prefetching the q/k norm
weights and RoPE row before attention (0), other depths per transition (2 is the optimum for all three).
s0020: prefetch while waiting at the grid barrier (coordinator warp, per poll). Expected 5400.

## s0020 and a negative result at B=8
s0020 (prefetch while waiting at the grid barrier): expected 5400, measured 5398.2 (step 5401 / 5882).
Negative: with several sequences the input copy is microseconds of L2 traffic (down projection at B=8:
5.4 us for 196 KB per SM) during which HBM idles; prefetching 2..14 more ksteps of the first tile before the
copy made B=8 WORSE by 10..40 us. The copy and the prefetch evidently compete (L2 fill vs L2 read).
tests/test_batches.py: every batch size 1..8 bitwise equal to the single run, also in another slot; reset;
generate()/step() mixes reproduce step+argmax. All pass.

## s0021: fast-math
nvcc flags: -O2 / ptxas -O4: no change. -use_fast_math: 5396 -> 5378 / 5881 -> 5860 (reciprocal and rsqrt
instructions instead of IEEE division / sqrt with slow-path calls in the scalar epilogue code). Stated in the
hypothesis as a precision change; all tests pass, batch equality stays bitwise. Expected 5385; measured
5382.1 (B=8 generate 5853); error vs float64 unchanged (median 0.0042, max 0.034).
Next experiment: 16 warps per block instead of 32 (15 loaders + coordinator). Reasons: (1) 512 threads may
use 128 registers each, so the spill limit that capped the pipeline depth disappears; (2) everything that
all threads execute at a phase boundary is halved; (3) the microbenchmark of day 1 showed 16 streaming warps
per SM saturate HBM as well as 32.

## s0022: 24 warps per block
Made the warp count a build parameter (tile split, reduction tree, softmax slices, V segments, embedding).
16 warps (15 loaders, 128 registers): same speed as 32 at B=1 with the 2-load pipeline, and an 8-load pipeline
now fits without spills: 5350 / 5878. 24 warps (23 loaders, 80 registers): pipeline 2 / 3 / 4 / 6 loads ->
5360 / 5340 / 5343 / 5406 (6 spills). The prefetch depths tuned for 32 warps are still the optimum.
Expected 5346; measured 5340.2 (step 5344 / 5850, generate B=8 5841). MBU 0.847.

## 2026-11-10 — context length, and the like-for-like workload
tests/time_ctx.py (prompt 64, then generate() up to position 1088, the workload of the reference engines):
B=1: 5331 us/step at positions 64-192, rising to 5523 at 960-1088; mean over the 1024 steps 5427 us
(SGLang + torch.compile: 6066 -> 1.118x; the 1.10x line is 5515). B=8: 5796 -> 6343, mean 6065 us (6836 ->
1.127x; line 6215). The evaluator times positions 64-320 (mean context 192), where the numbers are 5340 / 5841.
Growth with context: at B=8 about 70% of it is the K/V bytes themselves (1.2 GB per step at 1000 positions);
at B=1 the K/V bytes are only ~50 us, the other ~140 us are the 8 attention blocks loading 516 KB of K and V
per layer through their own load path (21 sequential L2 loads per warp for K and again for V) while 124 SMs
wait. Splitting a unit's positions over several blocks (local max / sum / unnormalised V sum, then a combine
stage) would fix that at the price of one more barrier per layer; not done.
Also tried: attention's complete K tiles requested into registers before the q/k/v prep (132 spills,
5772 us): holding 32 registers across that stage breaks the allocator even with 80 registers.

## 2026-11-10 — small things after s0022 (no submission)
- Source cleanup (settled experiment switches removed): same logits; +5 us (layout noise). The first "cleaned"
  build had not compiled and build.sh silently kept the old library: tests now `rm` the library first.
- End-of-phase prefetch issued before the last tile's barrier instead of after: +36 us. lm_head prefetching
  layer 0's first q/k/v tile for the next launch (step path): -6 us, kept.
- Cost of doing the barrier right: the `__threadfence()` after a phase's last epilogue (stores visible to the
  other SMs before the counter is incremented) costs 47 us per step (measured by removing it: 5341 -> 5294).
  Kept: without it the order "data, then counter" is not guaranteed by the memory model, and a stale read
  would be a silent wrong answer. A cheaper ordering primitive would be worth 0.9%.

## s0023, s0024
s0023 (cleaned source + lm_head prefetch for the next launch): expected ~5345, measured 5336.3 (best B=1).
s0024 (wide fragment layout for 5+ sequences; the input copy becomes a block copy): expected 5345 / 5690,
measured 5346.8 / 5674 (generate; step 5351 / 5685). B=8 -167 us; B=1 +10 us although the new path is never
executed at B=1 (code layout; an out-of-line version of the wide copy spilled at its call sites: 5478).
3 loads requested per thread before the stores: 4 spill.

## 2026-11-10 — store merging (no gain) and where the design stands
Merging the 2-byte fragment stores of neighbouring lanes into whole 8-byte / 4-byte entries (half the store
instructions and addresses in the residual and gate+up epilogues): bitwise identical, +3..+5 us. Store count is
not a cost. Reverted.

Budget at B=1 now: 5336 us = 36 x ~136.6 us + lm_head ~416 us. At the practical HBM rate (2.99 TB/s) a layer
is 129.1 us, so ~7.5 us per layer are left: three matvec-to-matvec transitions (~1.5-2 us each: last
epilogue, fence 0.33, atomic + poll ~0.6, input copy, first loads; partly filled by the L2 prefetches) and
~1.5 us around attention. The fence alone is 47 us per step. Everything I tried to make loaders start before
the grid barrier (register preload, deeper prefetch, prefetch before the last tile barrier) loses to the
coordinator's chain queueing behind it.
The idea I did not build: no grid barrier at all, each loader warp waiting only for the ~9 producer tiles of
its own K slice (completion flags per tile). Estimated gain ~1 us per transition (the blocks that produce a
slice's inputs are 4-5 of 132, but with 11-vs-12-tile blocks most slices still depend on a late block, so it
would also need fractional tiles); it touches every phase and the barrier-free variants so far were neutral.

## 2026-11-10 — clock64 breakdown, then the big one: 12-bit weights (s0025)
`clock64()` works in device code (per-SM cycle counter, 1830 ticks/us; tests/clk.cu). `-DTIMING` build +
tests/timing.py: the coordinator thread of every block stamps the parts of each phase. It supersedes the software
trace clock. Caveat: the TIMING build itself has spills and runs ~150 us slower; trust differences, not totals.
Result on s0024 (B=1, us per layer): qkv 17.5, attention 7.7, o 8.3, gate+up 68.2, down 34.9; inside a matvec
phase: copy 0.16, reduce+epilogue+fence 0.7-0.95, atomic 0.25, gap to next phase 0.68, and a mean wait of
1.4-3.5 us at the grid barrier (blocks with one tile less) that costs almost nothing because HBM is shared.
Conclusion: everything that is not "bytes / 3.0 TB/s" was down to ~7 us per layer. The step is the bytes.

So read fewer bytes. Weight statistics (tests/wstats.py, wstats2.py): bf16 = sign, 8-bit exponent, 7-bit mantissa;
the exponents of a row sit in a window of 16 binades for all but ~1e-4 of the elements once every row is scaled
by its own power of two (exact). Fixed window for the model: 1.4e-4..4.8e-4 outside, worst tile 2385; per-tile
shift: worst 553; per-row shift: 0.6e-4..1.2e-4 outside, worst tile 206 (gate/up rows differ a lot in scale).
Format (engine/api.py pack_rows, tests/test_pack.py proves it lossless):
- code = sign<<11 | (exponent - 109)<<7 | mantissa, 12 bits; 8 codes = 96 bits per lane and kstep; a "group" of
  4 ksteps = 3 loads of 512 B per warp (same contiguous 512 B per load as before) -> 4 MMAs.
- decode per MMA register (2 codes): spread 24 -> 2x12 bits, move the sign bit, add the exponent base: ~8 integer
  instructions, ~35 per MMA.
- row r of the stream holds W[r] * 2^shift[r]; the coordinator multiplies the row's accumulated sum by 2^-shift
  (exact) before the epilogue.
- elements outside the window ("exceptions", 825104 of 7.57e9): stored as code 0 (= +2^-18) in the stream; the
  coordinator runs one extra MMA per 8 exceptions of the tile: k-slot e holds the exact value in its row, slot
  e+8 holds -2^-18, B = x[col] gathered from the fragments in shared memory. Every weight enters with its
  exact value; only the f32 summation order of those few terms differs.
- per-tile table entry of 256 B at a computable address: chunk count, 16 row shifts (int8), 3 inline chunks (24
  exceptions); overflow chunks in a separate array (tile order, prefetched one tile ahead).
Microbenchmark first (tests/bw12.cu, same block-team shape): 16-bit 3134 GB/s; 12-bit with decode 3120 GB/s of
packed bytes = 4.16 us per tile against 5.51. So an SM decodes as fast as HBM delivers; with fewer blocks
active one SM reaches 42 GB/s of packed bytes (compute limit, 1.8x its fair share).

Hypothesis s0025: bytes per step 15.1 -> 11.4 GB at the same rate; expected 5340 -> ~4520 (B=1), 5674 -> ~4830.
Measured: 4522.0 / 4832.5 (generate; step 4527 / 4841). Valid; numeric gates unchanged (median 0.0041, max
0.0486, bitwise batch equality, generate 8/8). MBU as the evaluator defines it (16-bit bytes) is now 1.00.
What it took after the first working version (4657 us):
- the coordinator's exception chain was two dependent cold loads (table entry -> chunk) per tile; under the
  weight flood a cold load takes microseconds, and on the first tile of a phase nothing hides it (first tile
  4.3 us instead of 0.7). Inline chunks at a known address + prefetch of the next tile's / next phase's entry.
- spills again: the coordinator's register copy of the descriptor and the running argmax of generate() were
  pushed to local memory by the exception code (9-18 LDL/STL, some per tile). Descriptor read in place through a
  volatile reference, argmax state / exception partial / row scales parked in shared memory -> 0 spills,
  4547 -> 4511.
- prefetch address bug carried from the 31-loader days (min(w, base) instead of min(w, extra)) fixed on the way.
Left on the table (measured with -DNOTAB, no table and no exceptions: 4434): the table/exception path costs
~80 us (B=1) / ~120 us (B=8). And the byte-ideal at 3.0 TB/s would be ~3800 + overhead: per layer the timing
build shows qkv 15.0 (bytes 12.6), attention+o 16.2 (8.4), gate+up 56 (50.3), down 28.7 (25.2).

## s0026: exact per-column scales (host only)
Where s0025's exception cost sits (tests/time_stop.py, step time with the first n layers; tests/abl.sh): not in
lm_head (the 16% "intercept" was an artifact of fitting a line through layers 12..36) but in layers 1-3:
+18 / +19 / +13 us against a build without exceptions, +1.7 us per 8 layers afterwards. Their gate/up/down
tiles have 128 exceptions on average (max 553): the coordinator runs up to 70 dependent load + MMA rounds per
tile and the loaders wait for it.
tests/wstats3.py: those exceptions are a column effect (a few input dimensions with tiny weights everywhere).
Scaling column j by 2^cs[j] (alternating row/column fit, 3 rounds) brings L1-3 gate+up from 6.6e-4..7.6e-4 to
1.5e-4..2.0e-4 and down from 5.4e-4..6.4e-4 to 0.7e-4..1.0e-4 outside the window; worst tile 44.
The input of the matrix must then be multiplied by 2^-cs[j], which costs nothing in the kernel:
- qkv, gate+up, lm_head read bf16(w_norm * x): the scale goes into the RMSNorm weight (float32, exact);
- down reads bf16(silu(gate) * up): the scale goes into the row scale of the up rows of gate+up.
(o would need the V rows of qkv scaled, tied over 4 heads; its gain is 3%, not done.)
Every rounding (bf16 of the fragments, the MMA sums) scales exactly, so the model is unchanged.
Check (tests/cmp_cols.py): residual after 1 layer with vs without column scales: rel diff 2.7e-8, 97% of the
elements bitwise equal (the difference is the f32 summation of a different exception set); after 2 layers
5e-8, after 4 layers 2e-5, after 36 layers 2.6e-3. Lesson worth remembering: one float32 ulp early in the stack
flips a bf16 rounding somewhere, each flip moves the next layer by ~1e-4 and flips 2% of its roundings: by the
end of the step ANY two builds are two independent draws of the bf16 noise. That is why the unit-test numbers
of the sensitive position (row 1, pos 27) jump between 0.03 and 0.22 from build to build; medians +-10%.
Expected 4460 / 4775; measured 4463.5 / 4781 (step) — valid, numeric gates as before.
Dead end on the way (kept under -DHOLD2=1): non-attention blocks holding their second o tile in registers across
the attention barrier (so the o projection needs no HBM after attention): 63 spills, +250 us. Would need the
tile in shared memory, and there is no room (229 KB used).

## s0027: shared memory under the L1 threshold for up to 4 sequences
Looking for why gate+up runs at 2.7 TB/s in the engine when the microbenchmark does 3.12: tests/bw12.cu with
extra (unused) shared memory. 12-bit stream, 132 blocks: 3124 GB/s up to 199680 B of shared memory, 2997 GB/s
from 200704 B on (per-SM maximum 34 -> 29 GB/s); the 16-bit stream does not care. The L1/shared-memory split
has discrete configurations; in the last one (the engine used 229 KB) the SM has ~28 KB of L1 left.
Layout rework: [misc 2 KB][partial sums 24.6 KB][sx]; sx last so that its size can depend on the launch.
Up to 4 sequences ("narrow"): fragments use 16 lanes = 128 B per kstep (lanes 16..31 of the MMA read their
neighbour's bytes, columns unused), K=4096 fragments behind a 98 KB scratch region (stash / attention scratch),
K=12288 fragments over it: 157.8 KB. 5..8 sequences ("wide"): unchanged 223 KB (196 KB of fragments alone).
Loader loop templated on the stride; exception records carry (kstep, offset) instead of a shared-memory index.
Expected 4463 -> ~4360 at B=1, B=8 unchanged; measured 4365.8 (generate; step 4369) and 4808 at B=8 (+0.6%,
code layout). Like-for-like workload (prompt 64 + 1024 tokens): B=1 mean 4440 us, B=8 5035 us.
Next for B=8: the fragments cannot fit, so read them from L2 instead (microbenchmark mode 2: B operands loaded
from global memory with a small shared memory: 4.20 us per tile against 4.32 now, and no input copy).

## s0028: bit layout for a cheap decode
Phase breakdown after s0027 (timing build, B=1, us per layer; bytes at 3.12 TB/s in brackets): qkv 14.6 (12.1),
attention 9.4 + o 7.9 (8.1), gate+up 53.6 (48.4), down 27.6 (24.2). The largest excess is attention + o.
Tried: request the block's second o tile before computing the stashed one (3 loads: -0.2 us/layer; all 9 loads:
662 spills) and prefetch the whole second tile into L2 during attention (+0.4 us/layer). So the o phase is
not waiting for memory. It is compute: tests/bw12.cu with few blocks shows an SM cannot do a tile faster than
2.3 us (43 GB/s of packed bytes): ~31 integer instructions of decode per MMA on 23 warps.
New layout: exponent window base 0x70 instead of 109 (all rows shifted by 2^3 more, exact), so that
bf16 = sign << 15 | 0b0111 << 11 | code[10:0]. Stream word r (r = 0..2) = MMA register r with the 4 constant
exponent bits of each half replaced by 4 bits of the fourth register's codes:
  a_r = (w_r & 0x87ff87ff) | 0x38003800;  a_3 = 4 shifted+masked pieces of w0, w1, w2 | 0x38003800.
11 ops per MMA; compute limit 1.59 us per tile (62 GB/s per SM).
Expected 4366 -> ~4270 / 4808 -> ~4725; measured 4270.3 / 4714 (generate; step 4273 / 4722). Valid; the
logits are bitwise those of s0027.

## s0029: adaptive stash at phase ends (and what the memory pipeline of an SM is like)
Budget after s0028 (B=1): 109.8 us per layer against 92.8 us of bytes at 3.12 TB/s. Inside a phase the stream
runs at full rate (gate+up "further tiles": 3.2 TB/s); the 17 us are the five transitions, ~3.3 us each: last
block's epilogue+fence 0.9, counter 0.25, the others notice 0.3, block barrier + dispatch 0.6, input copy 0.2,
then the first loads have to travel.
Plan "stash always": like the o projection during attention, every block streams its first tile of the next
phase into shared memory while its coordinator finishes. What happened, in order:
1. Full-tile stash at the end of every phase: 35 spills (fixed: thread index re-read through inline asm so that
   lane/warp are not kept alive through the tile loop; descriptor read through volatile; the copy buffers of
   the input copy initialised - conditionally assigned arrays were living in local memory). Then: no gain.
   The timing build shows why: first tile 5.4 -> 2.0 us, but the last epilogue+fence of the SAME block went from
   1.0 to 2.8 us: its stores and fence queue behind the block's stash loads.
2. Block barrier after the coordinator's fence, stash behind it: epilogue back to 0.9, but now EVERY block waits
   >= 3.3 us at the grid barrier. An SM's memory responses come back in order: the coordinator's read of the
   barrier counter is answered only after the 98 KB its loaders asked for. (This also explains two old
   observations: deeper end-of-phase prefetch was worse, and prefetching before the last tile barrier +36 us.)
3. Adaptive stash: loads issued 2 per warp at a time; a flag set by the coordinator when the grid barrier is
   complete stops the stash; the number of loads stashed per warp is left in shared memory and the first tile
   of the next phase takes load u from the stash or from HBM accordingly. Cap per warp: 4 loads (narrow), 6
   (wide), 3 for K=12288 (room). Same-source A/B, 3 interleaved rounds (run-to-run noise +-1 us; differences
   between builds of the same logic +-15 us): no stash 4280 / 4737, cap 4: 4197 / 4690, cap 6: 4217 / 4679,
   full: 4244 / 4678, 1 load at a time: 4242.
Also fixed on the way: loaders read the phase descriptor (next-phase pointers) after the coordinator might have
rewritten it (since s0018; it only steered a prefetch). They now read it before the new barrier.
Expected 4270 -> ~4200 / 4714 -> ~4685; measured 4198.1 / 4680 (generate; step 4202 / 4690). Valid.
Why the gain is only 1.7%: the block that finishes last has nothing stashed and still sets the pace; what the
others fetch ahead only pays back at the end of the next phase, when the late blocks get more of the HBM.

## s0030: 3+ sequences read their input fragments from L2
B=8 was 11% behind B=1 (4680 vs 4198): input copy of 64-196 KB per phase (1.1 / 1.1 / 1.1 / 2.5 us per
layer = 5.8 us) and 223 KB of shared memory (stream 4% slower, s0027). tests/bw12.cu mode 2: loader warps that
take their MMA B operands from global memory (two extra 512 B loads per group, L2 hits) lose 1% of stream rate.
So for 3+ sequences: no copy; global fragment layout [group][2][lane][8 bf16] (one load = ksteps 2j, 2j+1 of
all 32 lanes); loaders load fragments + weights of 2 groups back to back; the coordinator's exception MMAs
gather their x from global memory (L2) as well; shared memory 158 KB at every batch size; the K=12288 stash
now exists for every batch size. Lanes of unused sequences load nothing.
Spills again, and two rules learned: (1) with plain `threadIdx.x` the compiler computes lane/warp once at the
top of the kernel and keeps them alive across everything; under register pressure they go to local memory and
are reloaded at ~20 places (one slot, 59 LDL). Reading the thread index through inline asm (`mov.u32 %0,
%tid.x`) makes every use site recompute it. (2) an array assigned under a condition (`if (extra) a[u] = load`)
is kept in local memory; initialise it unconditionally first.
Crossover (tests/time_b.py, us/step B=1..8): copy path up to 4: 4180 4277 4336 4369 | L2 path from 3:
.. .. 4315 4316 4328 4334 4375; from 2: 4291 for B=2 (worse). So copy for 1-2, L2 for 3-8.
GBW=3 (3 groups back to back) spills (49); stash cap 4 loads also best here.
Expected 4680 -> ~4380 at B=8; measured 4379 (step) / 4371 (generate); B=1 4184.3. Valid, bitwise batch
equality, 700-position test bitwise equal between the two paths (tests/test_long.py, batch of 6 vs 2).

## 2026-11-10 — new requirements from the operator (read at 17.2 h)
(1) Second workload: 8192-token prompt + 1024 new tokens (context to 9216) at B=1 and B=8; the attention scratch
stopped at 3904. (2) Two tracks, never mixed: track A = BF16 weights as stored, BF16 activations / KV where the
general engines use BF16, same operators in the same order (layout and kernel organisation free); track B =
anything lossless on top (12-bit weights, power-of-two rescaling). Every hypothesis starts with the track and the
formats that differ; a build switch must keep the best track-A engine buildable.
Everything from s0025 on is track B. s0001-s0024 used BF16 weights but a float32 residual stream and the fused
RMSNorm (scale by 1/rms after the matvec), which is not "same operators in the same order": no submission so
far is a clean track A. Plan: long context first (both tracks need it), then a track-A build.

## s0031 (track B): attention to 12288 positions, split path for 1-2 sequences
Scratch: scores [4][T] f32 + probability fragments [T/16][32][4] bf16 = 32 B per position. Now 16 B: the
probabilities are written as bf16 into the low half of their score's slot and the V pass builds the MMA B operand
from two 8-byte loads. 9216 positions need 179 KB of shared memory in total (still under the L1 threshold); the
size is chosen from max_context at engine creation (unchanged up to 8000 positions).
What the one-block-per-(sequence, kv head) attention costs with context (tests/time_ctx.py, us/step): B=1 4230
at 64-640 -> 6434 at 9000 (62 us per layer: 8 SMs read 4.7 MB each while 124 wait); B=8 4476 -> 8456 (64 SMs,
close to the HBM bound: 302 MB of K/V per layer = 97 us).
Split path (B <= 2, from 72 position tiles = 1136 positions, i.e. beyond the prompt-64 workload):
- sub-phase 1: NB/(8B) blocks per unit, each computes the scores of a chunk of position tiles into global memory
  (every block recomputes the unit's q fragments; the last chunk's block appends K and V); grid barrier;
- sub-phase 2: one block per (unit, dim tile of 16): copies the unit's scores from L2, softmax (same code), V
  pass with one warp per sub-segment; the other blocks stash their o tile.
Bitwise batch invariance needs one summation order for both paths: from 72 tiles on, the kstep range is cut into
24 sub-segments (each one MMA chain), a third = sequential sum of its 8, result (T0+T1)+T2. The one-block path
marks sub-segment starts with byte flags in shared memory; below 72 tiles the old single chain per third is used
(one load per MMA: at short context every instruction on the 24 busy warps costs 6 cycles, a batched/masked
loop was 0.5-0.9 us per layer slower, measured with the new clock points inside the attention body).
Bugs on the way: a missing block barrier before the grid barrier in sub-phase 1 (coordinator thread counted the
block in while other warps were still storing scores: logits off by 1-2% - caught by the bitwise comparison
between batch sizes, not by the tolerance); blockIdx.x hoisted and spilled like threadIdx.x (now read through
inline asm as well).
K/V "touch" by idle blocks now only while all of it fits in L2 (it was read twice at long context).
Results (us/step): evaluator B=1 4186 / B=8 4366 (s0030: 4184 / 4371). Prompt-64 workload (64..1088): 4267.5 /
4575. B=1 at 9000 positions: 5066 (was 6434). B=8 at 9000: ~8300. Correct at 9216 positions (tests/test_ctx.py:
B=1 and B=8 against HF f32 at 159 positions, row 0 bitwise equal); generate() across the switch (tests/test_cross.py).

## s0032 (track A): the aligned build
Build switch: `engine/TRACK` (A or B; `MK_TRACK` overrides), read by build.sh (-DTRACK_A) and reported by the
library (`mk_track()`), so api.py cannot disagree with the kernel. Same kernel organisation as track B; what
changes under TRACK_A:
- weights: the checkpoint's BF16 values, permuted into MMA A-fragment order (pack_matrix); a group of 4 ksteps is
  4 loads instead of 3 (LPG); no per-tile table, no scales, no exceptions; stash tile 131 KB (shared memory 193 KB).
- activations: RB(x) = bf16 round wherever the HF BF16 model holds a BF16 tensor: q/k/v, o, gate, up, SiLU(gate),
  the product, down, both residual adds, q/k norm (two steps), RoPE (cos/sin rounded on the host, two products and
  the sum), attention scores (matmul, then * scale), probabilities, attention output, logits. Float32 only inside
  RMSNorm (statistics), softmax and the matvec accumulators, as in the reference.
- RMSNorm before the matvec, y = bf16(w * bf16(x * 1/rms)), not fused. The residual add publishes bf16(x)
  fragments (xraw) and the per-tile sums of x^2. With 1-2 sequences the input copy normalises (norm weights
  pre-permuted into fragment order, 8 KB per norm); with 3+ sequences, where the loaders read the input from L2,
  a phase of its own (PH_NORM: every block normalises 31 fragment entries, grid barrier) writes the normalised
  fragments: 254 grid barriers per step instead of 181.
Bitwise batch invariance holds between the two norm paths (same rms_inv and norm2 code).
Found on the way: (1) every warp computing 1/rms for itself (132 x 24 warps loading the same 8 lines at phase
start) cost 4 us per norm phase; one warp per sequence + a block barrier: 1.1 us. (2) the stash first-tile
path with 12 loads + 4 of the next tile does not fit in 80 registers: batched like the main loop under TRACK_A.
Stage 1 (16-bit weights only, fused norm, f32 residual): 5358 / 5478 us. Final: 5382 / 5630 locally.
Expected ~5380 / ~5630; measured 5386.0 (generate B=1; step 5388) and 5639 (step B=8). Valid.
Numerics: error ~1.2x HF BF16's own on my tests (limit 2x); see result.json for the evaluator's.
Bytes bound for 16-bit weights at 3.12 TB/s: 4850 us; overhead now 11%.

## 2026-11-10 — both workloads, both tracks (tests/workload.py)
us per token, generate() / step() + argmax + host read of the token; prefill by step():
| workload | track B (s0031) B=1 | B=8 | track A (s0032) B=1 | B=8 |
| prompt 64 + 1024 | 4264 / 4306 | 4571 / 4615 | 5465 / 5504 | 5828 / 5870 |
| prompt 8192 + 1024 | 5047 / 5085 | 8067 / 8107 | 6202 / 6238 | 9447 / 9487 |
Prompt 64 against SGLang 0.5.18 + torch.compile (6066 / 6836): track A 1.110x / 1.173x (target 1.10x: met, B=1
with 0.9% to spare), track B 1.42x / 1.50x. No baseline yet for the 8192 prompt.
At 9216 positions and 8 sequences the K/V cache read per step is 10.9 GB (36 layers x 302 MB): more than the
12-bit weights (11.4 GB) and not compressible in either track.

## 2026-11-11 — attention fine-tuning round (mostly dead ends; both tracks)
With clock points inside the attention body (ATS in the TIMING build), B=1, 192 positions: prep 3.0 us, scores
1.8, softmax 1.0, V 0.8, tail 1.0 (7.6 us; HBM is only half used during it).
- prep 3.0 -> 2.0: the K/V rows of the new token are stores into lines last touched one step ago; the fence
  behind them waited for those lines to come from HBM. The appending block now prefetches the position tile
  (and its norm weights / RoPE row) at the start of the q/k/v phase. The time moved into the score pass though
  (K loads wait for the same instant either way): -9 us per step only.
- score pass with 8 loads back to back instead of load/MMA pairs: nothing at short context, -0.7% at 9000
  positions with 8 sequences.
- K/V history prefetched during the q/k/v phase by the other blocks instead of "touched" at the start of
  attention: +55 us. L2 does not keep it through 12 us of weight stream.
- attention blocks prefetching their own K/V behind (or before) the prep's loads: +10 / +20 us; without the idle
  blocks' touch: +80 us.
- split path: K chunk prefetched during the prep, V slice during the softmax: +54 us at 9000 positions.
  (Pattern: `prefetch.global.L2` ahead of time has helped exactly twice, both times for a few loads a few
  hundred ns ahead; anything further ahead is gone or in the way.)
- split path for 8 sequences (2 blocks per unit, 4 dim tiles each, from 4096 positions): 8728 us per step at
  9000 positions against 8173: the 64 one-block units already run at 2.86 TB/s. Kept for 1-2 sequences only.
- what does matter: the idle blocks' stash flood delays the attention blocks' K loads (L2 hits!) by 1.8 us
  (body 7.5 us with the stash, 5.6 without). The idle blocks now start stashing when all attention blocks have
  stamped their score pass as done (a hint array, bounded polling; 1-2 sequences only): B=1 -17 us; with 8
  sequences it loses (+14 us), so not there.

## How to rebuild and re-measure either track (for whoever picks this up)
- Track switch: `engine/TRACK` contains `A` or `B` (default B); `MK_TRACK=A bash engine/build.sh` overrides it.
  build.sh rebuilds when the track changes (`engine/.built_track`). The library reports its track
  (`mk_track()`), `Engine.stats()["track"]` and `["formats_that_differ_from_the_checkpoint"]` say what it is.
- Track A = -DTRACK_A: BF16 weights as stored (4 loads per 4 ksteps), RB() rounding of every tensor that is BF16
  in the reference model, RMSNorm before the matvec (normalising input copy for 1-2 sequences, PH_NORM phase
  for 3+). Track B = 12-bit weight code + exact scales + exception MMAs, float32 residual, fused norm.
- Tests (all through `mk exec`): `bash tests/run_all.sh A|B` runs test_engine (vs HF f32, B=1/8, generate),
  test_pack (B only: the 12-bit stream is lossless), test_batches (B=1..8 bitwise), test_long (700 positions),
  test_cross (generate across the split-attention threshold), test_edge (small engines, limits, B=3/5 at
  1250 positions), soak (11k steps repeatable), test_ctx 2400 (split path vs one-block path bitwise).
  `tests/test_ctx.py 9216 1 8` is the full long-context check (about 3 minutes).
- Workloads: `tests/workload.py label PROMPT NEW B...` (generate() and step()+argmax+host read),
  `tests/time_ctx.py` (us per step against context, `MAXC=9280 BS=8`), `tests/time_step.py` (evaluator-like).
- Diagnosis: `-DTIMING` + tests/timing.py (cycle counter per phase part and inside the attention body),
  tests/sass_spills.py (must print no LDL/STL), tests/abn.sh N "label:-Ddefs" ... (interleaved A/B).

## s0033 / s0034, s0035 / s0036: the same source on both tracks
s0033 (B) / s0034 (A): attention scheduling (prefetch of the K/V append tile, stash start signalled by the
attention blocks for 1-2 sequences, K loads 8 at a time). Expected B 4186 -> ~4160, A 5386 -> ~5340;
measured 4161.5 / 4366 (B, generate B=1 / B=8) and 5343.0 / 5627 (A).
s0035 (B) / s0036 (A): the coordinator increments the grid barrier's counter right behind the fence of its
last epilogue, before the block barrier that releases its loaders and before the next descriptor is written.
Expected B ~4150 / ~4340, A ~5330 / ~5615; measured 4154.0 / 4340.7 and 5331.6 / 5615.1.
Tried and dropped in this round: the last block(s) to arrive skip the end-of-phase stash (atomicAdd's return
value tells): +2..+15 us; softmax loops unrolled: no effect at 9000 positions (so the softmax is not
instruction-overhead bound); track A: requesting the normalising copy's loads before the 1/rms barrier: no
effect; coordinator-only normalising copy while the loaders keep stashing: not built (its loads would queue
behind the stash loads, one round trip per batch).
Edge-case tests added (tests/test_edge.py): engines with max_batch 1 / 5 and max_context 64 / 9216 give
bitwise the same logits as the (8, 1400) engine; B=3 and B=5 at 1250 positions equal B=1 (split path) bitwise.

## 2026-11-11 — dependency-started phases and run-ahead loaders: built, correct, no gain (not submitted)
The idea the skill lists first ("dependency counters instead of grid-wide barriers"), built in two stages
(tests/mk_dep_ahead.cu, switch DEP):
1. DEP: gate+up, down, q/k/v of layers 1.. and lm_head no longer start behind a grid barrier. Each block goes
   on when its own coordinator has fenced; a loader warp requests its first tile at once, waits only for the
   blocks that produce its slice of the input (a stamp per block, host table of producer ranges per warp),
   copies that slice privately (1-2 sequences) or reads it from L2 (3+), and multiplies; the coordinator waits
   for the grid counter before its first epilogue (1/rms needs all partial sums). Correct on the first run and
   bitwise equal to the barrier version for B = 1, 2, 3, 8 and generate (tests/cmp_so.py).
   Time: 104.05 us per layer against 103.68. The waits at the barriers vanish, the first tiles grow by the same
   amount, and the lag is collected at the next real barrier (before attention).
2. Run-ahead on top: two descriptor slots and a slot index per warp, so that the loaders enter the next phase
   right after their last tile's barrier while their coordinator still runs the last epilogue (a guard stamp
   protects the partial-sum buffer). Also correct on the first run. Time: 108.5 us per layer: the coordinator's
   last epilogue + fence takes 1.8 us instead of 0.9 because it queues behind its own block's fresh loads (the
   in-order pipeline again), and nothing is gained elsewhere.
Why neither can pay - the measurement I should have made first (timing build, sum per phase against bytes at
3.12 TB/s, B=1): q/k/v 12.5 us (12.1), gate+up 50.9 (48.4), down 25.9 (24.2): the three matvec phases
that follow a matvec already run at 94-97% of the HBM bound, transitions included (the adaptive stash keeps
HBM busy across them). The loss is elsewhere: attention 9.0 us + o 7.1 us for 8.1 us worth of o bytes. That
region is a chain of latencies on a few SMs (prep, K loads, softmax, V, two fenced stores, then two tiles of
decode + MMA per block and the epilogue chain), not a barrier problem.
So the remaining 12% above the byte bound is: ~8 us per layer around attention, ~4.5 us in the three matvec
transitions, ~0.5 us per layer of lm_head/embedding overhead. Source reverted to s0035/s0036.

## s0037 (B) / s0038 (A): gentle stash during attention
After the negative result above, the attention + o region again: the idle blocks' stash (2 loads per warp in
flight on 124 SMs) saturates HBM, and whatever the attention blocks read meanwhile - even L2 hits - takes
~1.8 us longer (first seen on the K loads; with the stash delayed until the score pass is done, the V loads
took the hit instead: V pass 0.9 -> 2.5 us). Requesting the first loads of the stash one at a time (ATT_POLITE,
1-2 sequences only) leaves room: B=1 4161 -> 4099..4115 for 2..6 polite loads, 4142 for all 9; with 8
sequences (64 attention blocks, 68 stashing) it loses 10-20 us, so it stays off there.
Expected B 4154 -> ~4100, A 5332 -> ~5290; measured: see the board (s0037: 4105.1).

## s0039 (B) / s0040 (A): o projection rebalanced
Phase sums at B=8 (timing build): attention 8.4 us, o 10.7 us for 25 MB (8.1 us of bytes), of which 3.85 us
are the attention blocks waiting: they had one cold tile each, the 68 others 2.8 tiles (one stashed, then
two cold ones in a row). Contrary to my earlier "imbalance is not a loss" (true for long phases), a phase of
2-3 tiles is bound by the block with the most cold tiles in a row. Now the attention blocks take two tiles
whenever possible: us/step B=1..8 before: 4100 4206 4283 4287 4304 4302 4333 4359; after: 4088 4155 4246 4248
4265 .. 4294 4315. Plus (12-bit path) an L2 prefetch of the rest of the second tile at the start of the o
phase (-18 us at B=1; this one is a prefetch a few hundred ns ahead of its use, the kind that works).
Expected B 4105 -> ~4090 / 4343 -> ~4300; measured 4087.9 / 4322 (step B=8). Track A: see board (s0040).

## s0041 (B) / s0042 (A) and the workloads at this point
s0041 / s0042: with 6+ sequences the stash blocks prefetch all of their second o tile (B=8 -10..-20 us;
with fewer sequences it costs). Measured (evaluator): B 4093.1 / 4282 (generate B=1 / B=8), A 5303.7 / ~5575.
Workloads (tests/workload.py, us per token, generate() / step() + argmax + host read):
| workload | track B B=1 | B=8 | track A B=1 | B=8 |
| prompt 64 + 1024 | 4175 / 4223 | 4511 / 4566 | 5386 / 5425 | 5812 / 5854 |
| prompt 8192 + 1024 | 4995 / 5037 | 8107 / 8148 | 6154 / 6194 | 9470 / 9509 |
Prompt 64 against SGLang 0.5.18 + torch.compile (6066 / 6836): track A 1.126x / 1.176x (generate), 1.118x /
1.168x (step with host read); track B 1.453x / 1.515x.
Where the o phase stands (timing build, NOTAB / NOEXC variants): the table + exception path costs 0.2 us of
its first tile and 0.7 us per layer in total; the first (stashed) tile takes 2.2 us of pure compute. All
microbenchmarks agree on a floor of ~1.5 us per tile per SM whatever the decode costs: about one HMMA per
10 cycles per SM. The o matrix is 65.5k MMAs = 2.7 us on the whole GPU at that rate, so the o phase (now
6.9 us at B=1) cannot go below ~5 us; attention + o are ~14 us per layer for 8 us of bytes, and that is
most of what separates the engine from the byte bound.

## 2026-11-11 — long-context attention: where its time goes, and three cuts (s0043 / s0044)
Timing points inside the split path (B=1, 9000 positions, per layer): q prep 1.5 us, scores out at 8.4 us
(147 KB of K per block at the fair HBM share), then barrier + score copy + softmax 6.0 us + V + stores: the
attention blocks' part ends at 27.1 us. Bytes: 12 us.
1. Score copy in the V blocks: 12 loads per thread, two at a time behind stores = 8 round trips (~3 us). Now
   all loads of two heads are requested before the first store, and the maximum per head is taken during the
   copy (exact in any order), so the softmax loses its first pass: 27.1 -> 25.6 us; B=1 prompt-8192 workload
   4995 -> 4938 us per token (track A 6154 -> 6116).
2. One-block path: maximum per head during the score pass as well (4 fmax per tile and a shuffle reduction per
   warp instead of a pass over all scores).
3. Long contexts (>= 72 position tiles), both paths: no third softmax pass. The exponentials stay in shared
   memory; the V pass builds its MMA B operand as bf16(e * (1 / Z)) while it waits for memory anyway (one
   definition, pfrag_e, for both paths: bitwise equality between split and one-block path holds, and the
   values are those of the old e / Z under fast-math).
   Workload prompt 8192 + 256 tokens, track B: B=1 4952 -> 4853 us per token, B=8 ~8050 -> 7931.
Short contexts unchanged (4090 / 4287). Track A build: 2 local-memory instructions in the split V function
(one 8-byte value saved once per attention phase, long-context split mode only) - left.
Tool note: `mk exec` output only arrives when the job ends; for long jobs read
runs/.../execs/<id>/out.log, which is written live.

## s0043 (B) / s0044 (A): results, and two more long-context dead ends
Evaluator: s0043 4084.7 us (B=1, generate; step 4097) / 4306 (B=8 step); s0044 5286.3 / ~5570.
Workloads (us per token, generate() / step() + argmax + host read):
| workload | track B B=1 | B=8 | track A B=1 | B=8 |
| prompt 64 + 1024 | 4173 / 4219 | 4505 / 4549 | 5374 / 5413 | 5795 / 5839 |
| prompt 8192 + 1024 | 4881 / 4928 | 8138 / 8179 | 6061 / 6100 | 9452 / 9494 |
The softmax changes pay with 1 sequence at long context (B=1 prompt-8192: 4995 -> 4881, track A 6154 -> 6061)
and not with 8 (8107 -> 8138, within the +-1% run-to-run spread of that measurement): there the 64 attention
blocks are bound by their K and V bytes, and the V pass now carries the conversion.
A/B at 8192 positions (prompt 8192 + 256 tokens): V prefetch during the softmax (VPF=2): B=1 4844 -> 4884,
B=8 8008 -> 8060 - one more prefetch-ahead that loses. Conversion as a softmax pass of its own (ONFLY=0)
with 8 sequences: 8578 without the prefetch, 8008 with it; the on-the-fly version without prefetch (the
submitted one) is 8008. VPF stays 0, ONFLY 1.

## s0045 (B) / s0046 (A): long-context K touch; stress test
Stress test (tests/stress.py): random batch size 1-8, random start position up to 9216 (incl. around the
split threshold), step / generate / mixed segments of 1-40 tokens, every segment run twice from the same state
and compared bitwise. 20 minutes per track: B 6499 segments / 257952 steps, A 5114 / 202936: all identical
and finite. (The pod was replaced once in the middle; the job re-ran by itself.)
Long context with 8 sequences (prompt 8192 + 256 tokens, us per token, repeatable to +-3): base 8010;
idle blocks pull the first 16 K tiles of every unit into L2 at the start of attention (TOUCH_KL=128): 7954;
32 tiles: 7955; pulling the start of V during the softmax behind the "K done" stamps: 14300-14600 (1564
warps polling for 47 us and then loading in the attention blocks' way). K touch kept, V touch removed.
Expected: evaluator unchanged; measured s0045 4094.9 (B=1) - within the spread of equal builds.

## Correction: the long-context K touch (s0045 / s0046) does not hold up - switched off again
Same-pod A/B over the full workload (prompt 8192 + 1024 tokens, 8 sequences, two runs each): without the
touch 8178 / 8176 us per token, with it 8202 / 8202. It had won over the first 256 tokens (8010 -> 7954) and
loses over the rest (context 8448..9216), so the 256-token A/B was not representative. TOUCH_KL is 0 again:
the source now behaves as s0043 / s0044 (the two submissions s0045 / s0046 only differ at 3+ sequences beyond
4096 positions). Also tried and left off: idle blocks prefetching their first gate+up tile during attention
(NPF_GU: +15..+49 us at B=1).
Note on comparing numbers: after the pod was replaced (epoch 6 -> 7) the same binary measures ~0.5% slower at
8 sequences / long context (8138 -> 8177); A/B only within one pod.

## Where the run stands (2026-11-11, 27.5 of 40 hours)
One source (engine/mk.cu + engine/api.py), two builds (engine/TRACK):
- track B (default; model-specific, lossless): 12-bit weight code with exact power-of-two row/column scales and
  exception MMAs, float32 residual, RMSNorm fused into the matvecs.
  Evaluator (context 64-320): 4085-4095 us per step at B=1, 4290-4310 at B=8 (s0043, s0045).
- track A (aligned with the general engines): BF16 weights as stored, BF16 activations, RMSNorm before the
  matvec. Evaluator: 5286-5288 us at B=1, 5565-5575 at B=8 (s0044, s0046); numeric error 1.17x HF BF16's own
  (limit 2x), top-1 0.9507 against HF's 0.9512.
Both: one launch per step (step) or per generate() call, no host sync in the loop, bitwise batch invariance,
generate identical to step + argmax, context up to 12288 positions.
Workloads, us per token, generate() / step() + argmax + host read (tests/workload.py, pod epoch 6):
| | track B B=1 | B=8 | track A B=1 | B=8 |
| prompt 64 + 1024 | 4173 / 4219 | 4505 / 4549 | 5374 / 5413 | 5795 / 5839 |
| prompt 8192 + 1024 | 4881 / 4928 | 8138 / 8179 | 6061 / 6100 | 9452 / 9494 |
Fastest general engine on prompt 64 (SGLang 0.5.18 + torch.compile, 6066 / 6836): track A 1.129x / 1.180x,
track B 1.454x / 1.517x.
What the time is made of (track B, B=1, per layer of 104.5 us; bytes at the measured HBM rate of 3.12 TB/s
are 92.8): q/k/v 12.1 (bytes 12.1), attention 7.3 + o 6.9 (8.1), gate+up 50.6 (48.4), down 25.5 (24.2),
block barriers / dispatch ~2; lm_head + embedding 316 us per step (299).
What made the difference, in order of size: (1) reading fewer bytes - 12-bit lossless weights -15%, track B
only; (2) everything that keeps HBM streaming: block team with a software pipeline across the tile barrier,
last-use loads, stash of the first o tile during attention and adaptive stash at phase ends; (3) the L1
threshold of the shared-memory size and the cheap decode layout; (4) not spilling (opaque thread / block
index reads, descriptors in shared memory); (5) attention scheduling (gentle stash, two o tiles for the
attention blocks, split over all SMs at long context).
What did not: dynamic tile claiming, fractional tiles, every prefetch issued more than a few hundred ns before
its use, dependency-started phases and run-ahead loaders (the matvec phases were already at 94-97% of the HBM
bound), barrier-free tile loop, deeper register pipelines.

## s0047 (B) / s0048 (A): split attention, V pulled into L2 by the idle blocks
Timeline of the split path with clock points that now land in the right slot (B=1, 9000 positions, per layer):
q prep 1.5 | scores out 8.4 (K: 147 KB per block on all 132 SMs, HBM-bound) | barrier + score copy 12.6 |
softmax 15.6 | V + stores ~23.9 | all arrived 25.0 us. Between 8.4 and 15.6 HBM carried only the o stash of
the 68 idle blocks (6.7 MB), and then the 64 V blocks read 17.8 MB of V at ~26 GB/s per SM.
Change (TOUCH_VS=4): in sub-phase 2 the idle blocks first load the units' V history (8 loads of 512 B per warp
in flight, 3 rounds at 9000 positions; skipped with 2 sequences, where only 4 blocks idle), then stash.
Expected: evaluator unchanged; prompt-8192 workload at B=1 4881 -> ~4810; mid contexts in split mode -55 us.
Measured: V + stores end at 22.3 us (was 23.95). Workload prompt 8192 + 1024, B=1, generate / step + host read:
track B 4807 / 4850 us per token (was 4881 / 4928), track A 5970 / 6010 (was 6061 / 6100). Context 1216-1500 at
B=1: 4265-4280 us per step (was 4321-4331). Evaluator: s0047 4085.6 (B=1 generate) / 4295.5 (B=8 generate),
s0048 5292.1 / 5569.6 - unchanged, as predicted. Results bitwise as before (whole suite on both tracks passes).
The V pass itself is now bound by its instructions (544 ksteps per block at ~23 cycles: MMA 10 + building
the bf16 probability operand), not by memory: touching V after the stash instead of before gives the same.

Things measured on the way that did not help (all same-pod A/B; none in the source any more):
- V loads in flight in the split V pass: 1 / 2 / 3 / 4 / 6 / 8 / 12 per warp -> 4839 / 4903 / 4863 / 4850 / 4863 /
  4918 / 5132 us (prompt 8192 + 256): no trend below 6, worse above. Stays 4.
- V loads 2 or 4 in flight below 72 position tiles (one-block path): +8 us over the prompt-64 context range.
- Split threshold 48 / 32 / 24 tiles instead of 72 (before the V touch): the split costs ~3 us per layer fixed
  and wins only from ~1150 positions; with the V touch the crossover is ~1000, a gain of < 1 us per step over the
  prompt-64 workload: left at 72.
- B=8: K + V touch up to 640 tile-sequences (was 256): equal to 576 positions, +100 us at 1088; smaller limits
  (128 / 64 / 0) lose at short context (0: +50 us at 64-320). V-only touch for 256..768 tile-sequences: +10 to
  +40 us at 700-1088 positions. The 64 attention blocks then compete with the touch for HBM.
- Stash at once with all loads one at a time (ATT_WAIT=0, ATT_POLITE=9): +45 us at B=1.
- Second o tile requested only after the stashed tile's MMAs: +45 us. The stashed first o tile costs 2.1 us of
  compute whatever is in flight; the o phase is bound by its second tile's bytes.
Where attention + o stand at short context (B=1, 192 positions, per layer): the phase ends when the idle blocks'
stash ends = score pass done (3.0 us) + 0.4 + 3.9 us of bytes; the attention chain alone needs 5.5 us on a quiet
GPU (prep 1.55, scores 1.07, softmax 0.85, V 0.95, stores + fence 1.06) + 1.1 barrier. One SM takes K/V from L2
at ~80 GB/s and from HBM at ~45, so the chain grows by ~1 us per 100 positions (8.5 us at 576, 11 at 1100).

## Dead end: K loads before the prep (EARLY_K)
Idea: the warps without prep work request their first K position tile at the start of attention (8 loads held
in registers; the tile of the new row excepted, tiles re-assigned so that the six prep warps come last), in both
attention paths, so that K travels while q is prepared. Bitwise equal to the committed build.
Measured (B=1): 192 positions: prep 1.56 -> 1.96 us (its L2 loads now queue behind the SM's own HBM misses),
scores done 3.04 -> 2.80, phase 7.57 -> 7.43; 576 positions: phase 9.77 -> 10.14; split path at 9000: prep
1.5 -> 4.2, scores out 8.36 -> 8.09. Per step: +5 us at 64-192, +10 to +35 us from 192 to 1500, +2 at 8192;
track B build with 16 spills on top. One more case of "what an SM has in flight delays everything else it asks
for". Reverted.

## Dead ends after s0048 (same-pod A/B, us per step B=1 / B=8)
- V loads of the first VE ksteps requested before the score pass (held in registers; with nt <= 3 VE the
  "K done" stamp is set right after the K loads): VE=4/5/6: +55..+110 / +90..+170, and 39-55 spills. Reverted.
- 3+ sequences, idle blocks pull the attention blocks' first o tile into L2 after their stash (TOUCH_O=2):
  B=8 +45 us. HBM is not idle there, the 64 attention blocks read K/V.
- 3+ sequences, gentle start of the attention-phase stash (ATT_POLITE for B > 2): 5 loads +20, 9 loads +7 at B=8.
- 1-2 sequences: first 1-3 stash loads requested before waiting for the "K done" stamps: +6..+25 at B=1.
- Probe for B=8 at 9000 positions: idle blocks loading half of every unit's K while the attention blocks score
  the other half (timing only, wrong results): score pass 57 -> 53 us of 117. The one-block units are HBM-bound
  already; not worth a third attention path.

## s0049 (track A): RMSNorm inside the wide loaders - the norm phase is gone
Track A with 3+ sequences had a grid phase of its own for every RMSNorm in front of a matvec (q/k/v, gate/up,
lm_head: 73 extra grid barriers per step), because the loaders read the input fragments from L2 and the BF16
reference rounds x / rms before multiplying by the weight. Timing probe (norm phase skipped, wrong results):
B=3 5515 -> 5320, B=8 5567 -> 5363. Now: at the phase start every block copies the norm weights (8 KB, wide
fragment order) into shared memory (the narrow-fragment region, unused in wide mode) and loader warps 12-19
compute 1/rms per sequence (rms_inv: same order as everywhere else); the loaders then meet at a named barrier
of their own behind their first weight loads (bar.sync 1, the coordinator does not wait), and normalise the two
fragment loads of each group right before its MMAs: y = bf16(w * bf16(x / rms)), with cvt.rn.bf16x2 for both
roundings (same values as the integer rounding used before; norm2 of the narrow copy now uses it too).
Bitwise equal to s0048 at B=1, 2, 3, 8 (tests/cmp_so.py), whole suite passes, 0 spills (was 2).
Expected: B=8 5570 -> ~5450, B=1 unchanged. Measured: s0049 5278.2 / 5447.7 us generate (step 5282 / 5460).
Workloads (generate / step + host read): prompt 64 + 1024 B=1 5371 / 5410, B=8 5672 / 5716 (was 5795 / 5839,
now 1.205x SGLang+compile's 6836 through step()); prompt 8192 + 1024 B=8 9328 / 9370 (was 9452 / 9494).
Of the probe's 200 us, ~100 remain: the per-tile normalisation is repeated for every tile of a block (12 tiles of
gate/up), and the 1/rms round trip at every norm phase start.

## s0050 (track A): normalised fragments kept in shared memory after the first tile
A warp normalises its groups' input fragments in the block's first tile and keeps them in its own, already
consumed part of the stash image (first 1 KB of each group's slot) for the block's other tiles (gate/up: 12
tiles per block, lm_head: 72). Bitwise equal. Expected B=8 5448 -> ~5395; measured s0050 5283.7 (B=1 generate,
noise) / 5403 (B=8 step; s0049 5460). Prompt-64 workload B=8: 5625 / 5668 (generate / step + host read).
The same for every K=4096 phase in track B (input fragments from shared memory instead of L2 after the first
tile, XS_ALL=1): B=3 +70, B=8 +55 us; in track A for the o phase: neutral. Off.

## s0051 (track B): 6 V loads in flight per warp in the one-block attention path (3+ sequences, long context)
With 8 sequences at 9000 positions the score pass reads K at ~85% and the V pass V at ~83% of the HBM rate
(24 warps x 4 loads of 512 B in flight per SM in the V pass, half of the score pass's 8).
A/B (prompt 8192 + 256 tokens, B=8, us per token): VBATCH 4 / 6 / 8: 8015 / 7945 / 7970. The split path (1-2
sequences) keeps 4 (VBATCH_S; 6 and 8 measured slower there). Pipelining the next tile's first K loads before
the current tile's MMAs (KPIPE 2 / 4 / 8): +100 / +140 / +940 us per token and 0 / 28 / 58 spills - removed.
Workload prompt 8192 + 1024 at B=8: track B 8114 / 8160 (generate / step + host read; was 8138 / 8179),
track A 9215 / 9256 (was 9328 / 9370 in s0049). Evaluator: s0051 4094.3 / ~4290 - unchanged (the change
does not touch contexts below 1136), as predicted.

## s0052 (B) / s0053 (A): no fence behind the K/V append
The attention block that appends the step's K and V rows fenced them (__threadfence) before its block barrier.
Only the block itself reads the new row in this phase, and a block barrier makes the block's global stores
visible to all its threads (CUDA: __syncthreads; PTX: bar.sync has membar.cta semantics; the .cg cache operator
of the score loads does not change the memory model); other blocks read the row only after the grid barrier,
before which the block fences its attention output (same for the split path: fence at the end of
att_k_chunk). Same-pod A/B (3 rounds): 4090 -> 4080 (B=1), 4298 -> 4287 (B=8); bitwise equal; whole suite and
the randomised stress test (6 min track B: 1988 segments / 77906 steps; 4 min track A: 1069 / 41706) pass.
Pod epoch 8 from here (the pod was replaced; absolute numbers shift by ~0.5% between pods).
Measured: s0052 4082.8 / 4283.9 us generate (B=1 / B=8; step 4087.6 / 4296.0) - new best on track B.
s0053 (track A, with s0051's V batching): 5229.7 / 5414.8 generate (step 5235 / 5429): B=1 -54 us against
s0050 (more than track B gained: the track A prep has more rounding work between the stores and the fence).

## Dead end: split-path scores stored as bf16 between the sub-phases
The scores are bf16 values (RB), so storing them as bf16 in global memory would halve the V blocks' score copy
(144 -> 72 KB per block at 9000 positions). A/B prompt 8192 + 256 at B=1: 4776 -> 4735 and 4753 -> 4755 us
(within noise), and tests/test_ctx.py 2400 failed the bitwise comparison between the split path (B=1, 2: equal
to each other) and the one-block path (B=8), reproducibly, while the same source with the fp32 copy passes
(also with the K/V append fence removed: so s0052 is not involved). Cause not found in reading; with no
measurable gain, removed rather than debugged. (PTX: bar.sync guarantees that prior memory accesses are
performed relative to all threads of the CTA, whatever the cache operator of the later loads - the argument
for s0052; the long-context bitwise test passes with and without the fence.)

## Long context, 8 sequences: L2 prefetch one round ahead in the attention loops; K/V address sensitivity
- K: prefetch of the warp's next position tile (4 KB, one instruction, KPF=1) before the MMAs of the current
  one: 3 runs at prompt 8192 + 256, B=8: -124, +55, +100 us per token. Two rounds ahead +125. Not kept.
- V: prefetch of the next round's 6 ksteps (one instruction: lane = 4 x kstep + 128-byte part, VPF2=1):
  full workload prompt 8192 + 1024 at B=8 (engine sized 9216): 8107 -> 7919, 8107 -> 7919, 8107 -> 7916 us per
  token (generate; 3 interleaved pairs, -2.3%); 4 sequences: -50 us from 1500 positions on; but 8 sequences at
  1136-2600 positions +40, 3500 +12, 5500 +18, 7000 +46 (engines sized to the workload). Two rounds ahead:
  mixed. Kept, gated: up to 4 sequences always, more only from 480 position tiles (7680 positions) on.
- The K/V layout matters a lot at 8 sequences and long context: padding the K/V region of each (layer,
  sequence, head) by 1 / 2 / 3 / 4 / 8 position tiles (4 KB each) changes prompt 8192 + 256 at B=8 by 0 / -195
  / +575 (engine sized 8448) / +60 / +60 us per token. With 64 units streaming at the same pace their relative
  addresses decide how the reads spread over the HBM channels. Not tuned further (the best padding depends on
  the engine's max_context; a choice fitted to one workload size could be the worst for another). KV_PAD = 0.
Correction (same day): the V prefetch does not hold up. Rebuilt with the gate (and fewer spills), the same
full workload measured 8066 (without) vs 8157 (with), and prompt 8192 + 256 with the engine sized 9216: 7941 /
7943 vs 7987 / 7998. The -190 us belonged to one particular build of the ungated variant (6 spills, otherwise
identical logic); with the K/V address sensitivity above, a build or layout change moves this workload by
+-100 us. Both prefetches removed; KV_PAD stays as a switch (0).
- Tile-major K/V layout (KV_IL=1: [layer][position tile][unit] instead of [layer][unit][position tile], so that
  the units, which stream through their positions at the same pace, read neighbouring 4 KB blocks): bitwise
  equal, 0 spills; prompt 8192 + 256: B=8 7812 -> 7866, B=1 4769 -> 4823 us per token; short contexts equal.
  Off (switch kept; addressing now goes through KVU / KVT).

## Workloads at the current source (= s0052 track B / s0053 track A logic), pod epoch 8, 33 h
us per token, generate() / step() + argmax + host read of the token (tests/workload.py):
| workload | track B B=1 | track B B=8 | track A B=1 | track A B=8 |
| prompt 64 + 1024 | 4155 / 4199 | 4511 / 4557 | 5351 / 5393 | 5639 / 5684 |
| prompt 8192 + 1024 | 4784 / 4830 | 8017 / 8060 | 5959 / 5997 | 9198 / 9240 |
Against SGLang 0.5.18 + torch.compile on prompt 64 (6066 / 6836, server end to end), through step() with the host
read: track A 1.125x (B=1) / 1.203x (B=8), track B 1.445x / 1.500x. Since the summary of 27.5 h: track A B=8
5795 -> 5639 (norm phase gone), prompt 8192: track A 6061 -> 5959 (B=1), 9452 -> 9198 (B=8); track B 4881 ->
4784 (B=1), 8138 -> 8017 (B=8). No baseline for prompt 8192 on the board yet.

## s0054 (track B): each loader warp fetches its own slice of the input (1-2 sequences)
With 1-2 sequences every phase began with a block-wide copy of the input fragments into shared memory and a
block barrier (timing build, B=1: down projection 0.77 us, the K=4096 phases ~0.17 us). A loader warp only
multiplies its own ksteps, so now it requests just that slice (compact global layout [n][kstep][16 bf16]; at
most 5 loads per lane), issues its first weight loads (stash / first group), stores the slice into the usual
shared layout and syncs the warp - no block copy, no entry barrier (1/rms is read by the coordinator only after
the first tile's barrier). The coordinator's exception MMAs, which may need any kstep before that barrier, read
the compact global copy instead (one L2 round trip, in parallel with the loaders). Track A: the same for the
phases without a norm (o, down); norm phases keep the block-wide normalising copy.
gridDim.x is now read through inline asm like the thread and block index (the compiler had hoisted it to the
kernel start and spilled it: 17 local-memory instructions).
Bitwise equal (cmp_so B=1/2/3/8, both tracks), 0 spills on both tracks, suite + 6-minute stress test pass.
Same-pod A/B: track B B=1 4082 -> 4062, B=2 4157 -> 4121, B=8 unchanged; track A B=1 5240 -> 5229, B=2
5344 -> 5317. Measured: s0054 4061.1 (B=1 generate, new best; step 4065.5) / 4297.1 (B=8; within the build
spread of s0052's 4283.9).

## s0055 (track A): per-warp input slices for o and down (1-2 sequences)
Measured 5233.2 / 5403.1 us generate (B=1 / B=8; step 5237 / 5416). B=1 about equal to s0053's 5229.7 (the
same-pod A/B had -11), B=8 -12.

## No block barrier at the start of a phase with 3+ sequences (ENTRY_NB)
With 3+ sequences nothing is copied at a phase start (the loaders read the input from L2), but the entry still
ended with a block barrier: only the 1/rms of norm phases (loader warps 12-19, one per sequence) is produced
there, and the coordinator reads it only behind the first tile's barrier. Without that barrier (same-pod, 2
rounds, us per step B=1 / B=2 / B=8): 4059 / 4117 / 4286 and 4063 / 4119 / 4300 -> 4061 / 4118 / 4247 and
4056 / 4113 / 4249. Bitwise equal.
Also tried: 1/rms computed by the coordinator warp (loads of 4 sequences at a time) so that loader warps 12-19
start streaming at once: +25..35 us at B=1-2 (the coordinator is then late for its first tile), and with the
entry barrier kept +80 at B=8. COORD_RMS = 0.
- s0056 measured 4057.2 / 4244.6 us generate (B=1 / B=8; step 4061.6 / 4256.9): new best at both batch sizes.
- Stash filled with cp.async (no registers; a warp leaving its end-of-phase stash does not wait for its last
  loads, it waits in the next phase's first tile behind its memory loads; pacing kept): bitwise equal, 0 spills,
  B=1 equal, B=2 +7, B=8 +30. The ~1.1 us between "all arrived" and the next phase's start is not the drain of
  the stash loads. Reverted.

## s0057 (track B): q/k/v, gate+up and down on 128 of the 132 SMs
384 / 1536 / 256 tiles over 132 blocks: some blocks own one tile more than others (gate+up: 84 x 12 and 48 x 11;
the timing build showed 3.4 us of mean wait at the end of gate+up), and in the tail of a phase fewer SMs stream
than HBM needs. 128 divides all three counts (3 / 12 / 2 tiles per block): all blocks finish together; the 4
remaining SMs own no tile in these phases (empty ranges; the narrow loop's stashed-first-tile branch now checks
that the block has a tile). lm_head (9496 tiles: 124 x 72 + 8 x 71) and o (attention blocks first) keep all
132. Bitwise equal (a tile's arithmetic does not depend on the block that runs it).
Same-pod A/B (us per step, B=1 / B=2 / B=8): 4059 / 4122 / 4248 -> 4045 / 4101 / 4222 (two rounds, equal).
Track A: 5230 / 5313 / 5425 -> 5250 / 5340 / 5398: worse with 1-2 sequences, better with 8; so the table is
chosen per width now (NB_MAT for 1-2 sequences: 128 in track B, all blocks in track A; NB_MAT_W = 128 for 3+).
Measured: s0057 4042.8 / 4212.1 us generate (step 4047.7 / 4224.1) - new best at both batch sizes.
- s0058 (track A: per-width tables, no entry barrier with 3+ sequences): 5235.2 / 5400.7 us generate (B=1 / B=8;
  step 5240.8 / 5413.3).
- Track A, 1-2 sequences, norm phases: each loader warp normalising its own input slice (1/rms by warps 12-13,
  their partial sums loaded first, loaders meeting at a named barrier; norm weights of the slice loaded next to
  the input) instead of the block-wide normalising copy: bitwise equal, but 32 spills and B=1 +34 us. Reverted.
- First 1-2 V loads of each warp requested before the score pass (below 72 position tiles; no spills this time):
  B=1 +2..+30, B=8 +5..+19 us per step. Not kept.

## Workloads at s0057 (track B) / s0058 (track A), pod epoch 8, 35.5 h
us per token, generate() / step() + argmax + host read (tests/workload.py):
| workload | track B B=1 | track B B=8 | track A B=1 | track A B=8 |
| prompt 64 + 1024 | 4126 / 4168 | 4447 / 4491 | 5344 / 5384 | 5615 / 5658 |
| prompt 8192 + 1024 | 4705 / 4753 | 7909 / 7951 | 5918 / 5958 | 9162 / 9202 |
Against SGLang 0.5.18 + torch.compile on prompt 64 (6066 / 6836) through step() with the host read: track A
1.127x / 1.208x, track B 1.455x / 1.522x.
- Attention -> o without a grid barrier: the attention blocks stamp "output stored" (one int each, unique per
  phase) and every block's coordinator warp polls the 8B stamps (no counter increment; host barrier accounting
  adjusted). Bitwise equal, 0 spills, but +35..+45 us at B=1, +40..+60 at B=2 and B=8: 132 warps polling 8-64
  lines cost more than one atomic counter. Reverted.
- Re-sweep of the stash / prefetch parameters with the 128-block tables (2 rounds, us per step B=1 / B=8):
  base 4047 / 4225; CAP_H 6: 4077 / 4222; CAP_H 9: 4099 / 4234; CAP_HW 6: 4052 / 4236; WPF 12: 4045 / 4221;
  SG 3: 4051 / 4215. All within the spread; parameters left as they are.

## Where the run stands (35 of 40 hours) - final overview
Engine: one persistent kernel per step() call or per generate() call (132 blocks x 24 warps, 23 loader warps +
1 coordinator per block, call-free interpreter of phase descriptors in shared memory; 5 grid barriers per layer,
+1 for split attention, + the embedding / lm_head phases). Decode only; the prompt goes through step().
Two builds of the same source (engine/TRACK, or MK_TRACK=A|B for build.sh):
- track A (aligned with vLLM / SGLang): BF16 weights as stored (permuted into MMA fragment order), BF16
  activations and K/V cache, every tensor rounded to BF16 where the HF BF16 model has one, RMSNorm computed per
  step before each matvec (normalising input copy with 1-2 sequences, normalised by the loaders with 3+).
  Best evaluator results: s0055 / s0058 5233 / 5401 us per step (generate, B=1 / B=8).
- track B (lossless model-specific formats): 12-bit weight code (sign, 4-bit exponent window, 7-bit mantissa)
  with exact power-of-two row and column scales and exception MMAs for the weights outside the window; float32
  residual stream; RMSNorm fused into the matvecs. Best: s0057 4042.8 / 4212.1 us (generate; step 4047.7 /
  4224.1).
Correctness on both tracks: evaluator gates passed by every one of the 58 submissions; bitwise equality of a
sequence alone and inside a batch of 8 (also against the split attention path at long context); generate() equal
to step() + argmax; randomised stress test (batch 1-8, positions up to 9216, step / generate segments repeated
from the same state) bitwise repeatable.
Like-for-like workloads (us per token; the table above, pod epoch 8):
  prompt 64 + 1024:   track A 5344 / 5384 (B=1), 5615 / 5658 (B=8); track B 4126 / 4168, 4447 / 4491
  prompt 8192 + 1024: track A 5918 / 5958, 9162 / 9202;               track B 4705 / 4753, 7909 / 7951
  (generate / step + argmax + host read). Fastest general engine measured by the operators on prompt 64: SGLang
  0.5.18 + torch.compile 6066 / 6836 -> track A 1.13x / 1.21x, track B 1.46x / 1.52x. No baseline for the
  8192-token prompt was posted.
Since 27.5 h (this session): split-attention V touch (prompt 8192 B=1 -1.5%), track A norm inside the wide
loaders (B=8 -3%), no fence behind the K/V append, per-warp input slices for 1-2 sequences, no entry barrier for
3+ sequences, 128-block tile tables; dead ends listed in the sections above (V/K prefetch inside the attention
loops, bf16 split scores, tile-major K/V, cp.async stash, flag barrier, coordinator 1/rms, early K/V loads,
per-warp normalisation in track A, gentle stash for 3+ sequences, o-tile touch).
Where the remaining time goes (track B, B=1, per layer ~104 us against 92.8 us of bytes): attention + o ~13.4
us for 8.1 us of o bytes (the attention chain itself is ~5.3 us at short context and grows ~1 us per 100
positions), the barrier/transition cost of 5 phases, and the per-SM bandwidth spread within a phase (2-3 us
between equal blocks, not static: calibrating tile counts did not help).
- Probe (kernel recording %smid, 4 runs of 128 steps): with the 128-block tables the barrier waits at the end of
  gate+up average 5.5 polls (~2 us) with a spread of only 1.4 polls (~0.5 us) between blocks, and the
  block -> SM mapping is the same in every launch. What remains at the end of a phase is the barrier's own
  latency (last arrival -> counter -> polls), not imbalance; tail work-stealing would have little to win now.

## Only the attention blocks wait at the end of q/k/v (QKV_NOWAIT, one-block attention)
The other blocks need nothing that q/k/v produces before the end of attention (their attention-phase work is
the o stash and the K/V touch), so their coordinator counts the block in and goes on without polling. The
first version was wrong (cmp_so: logits differ): a block that skips the wait can add its attention-barrier
increment before every block has counted in for q/k/v, and that increment then completes the q/k/v barrier for
the attention blocks too early. Fixed by having those blocks check that the q/k/v barrier is complete before
they count in at the end of attention (by then it always is: one poll). Bitwise equal again.
Same-pod A/B (us per step B=1 / B=8, 2 x 2 rounds): 4047 / 4225 and 4040 / 4218 -> 4036 / 4198 and 4028 / 4206.
Measured s0059 (track B): 4047.0 / 4196.2 us generate (step 4051.8 / 4207.9): B=8 -16 against s0057, B=1 +4
(the same-pod A/B had -12..-19 at B=1; within the build spread). Suites and stress tests pass on both tracks.
Track A (3 rounds): B=1 5231 -> 5261 (+30), B=2 equal, B=8 5407 -> 5380 (-27): in track A it is used only with
3+ sequences (QKV_NW).
Measured s0060 (track A, QKV_NOWAIT for 3+ sequences): 5234.3 / 5377.8 us generate (step 5239.3 / 5390.2): B=8
-23 against s0058, B=1 equal.

## Workloads at s0059 (track B) / s0060 (track A) - the final source - pod epoch 8, ~36.5 h
us per token, generate() / step() + argmax + host read (tests/workload.py):
| workload | track B B=1 | track B B=8 | track A B=1 | track A B=8 |
| prompt 64 + 1024 | 4125 / 4170 | 4415 / 4459 | 5347 / 5385 | 5638 / 5681 |
| prompt 8192 + 1024 | 4747 / 4789 | 7967 / 8010 | 5948 / 5989 | 9131 / 9177 |
(Track A prompt 8192 was measured twice: the first run gave 6005 / 9772 at B=1 / B=8, the re-run of the same
library 5948 / 9131; the second is in the table. An A/B of QKV_NOWAIT at prompt 8192 + 256, B=8, engine sized
9216, showed no difference on either track (A 8998 / 8987, B 7792 / 7804), so the first run was an outlier.
The prompt-8192 B=8 workload moves by up to +-100 us between equal-logic builds, see the K/V layout notes.)
Against SGLang 0.5.18 + torch.compile on prompt 64 (6066 / 6836, server end to end) through step() with the host
read: track A 1.126x (B=1) / 1.203x (B=8), track B 1.455x / 1.533x.
Evaluator, best per track: track B s0057 4042.8 (B=1) / s0059 4196.2 (B=8) us per step (generate); track A
s0053 5229.7 (B=1) / s0060 5377.8 (B=8).
- Final randomised stress tests (12 minutes each, final libraries): track B 3975 segments / 158068 steps, track A
  3125 / 125134 - all bitwise repeatable and finite.
- Attention blocks waiting only for the blocks that own their q/k/v tiles (each block stamps its q/k/v rows as
  stored; an attention block polls the ~17 stamps of its producers instead of the barrier counter): bitwise
  equal, 0 spills, but B=1 +60..+83, B=2 +54..+82, B=8 +50..+56 us. Like the flag barrier after attention:
  polling per-block stamps costs more than the one counter. Reverted. (Pod epoch 9 from here.)
  Second version with the table entries and the stamps of each lane loaded together (one round trip per poll):
  +25..+35 us (6 spills). Dropped.
- Track A, 1-2 sequences, 128-block tables per phase (same-pod, us per step B=1): all on 132 blocks 5233;
  gate+up on 128: 5251; down on 128: 5234; q/k/v on 128: 5234. Track A keeps 132 blocks with 1-2 sequences
  (16-bit weights: each SM must stream 33% more bytes per tile; with 4 SMs fewer the per-SM rate limits).
- Track A parameter sweep (same-pod, us per step B=1 / B=8): base 5233 / 5380; CAP_H 3: 5250 / 5391; CAP_H 6:
  5253 / 5381; O2PF 9: 5233 / 5380; NPF 3: 5233 / 5388; SG 3: 5275 / 5389. Unchanged.
- Clean rebuild through engine/build.sh in a fresh pod (epoch 9) for both tracks: builds, test_engine passes,
  stats() reports the track and its formats.
- Track B, attention-phase stash wait for more sequences (ATT_WAIT_MAX_B 4 / 8; same pod, B=8): 4185 / 4232 us
  generate against 4185; with ATT_POLITE 0 at 8: 4207. Unchanged (2). (B=1 varied by +-25 us between these
  builds although its code path is the same: codegen spread.)
- Split threshold 64 / 60 position tiles instead of 72 (prompt-64 range, B=1, 2 rounds each): 4102 / 4103,
  4105 / 4126 against 4127 / 4127 - the differences appear from 64 positions on, where the threshold plays no
  role. Left at 72.
- Measurement spread: the same library, six separate processes on one pod: time_step generate B=1 4038-4053,
  B=8 4185-4212 us. Process-to-process (allocation-dependent) spread is ~15 us at B=1 and ~25 at B=8; A/B
  differences below that in this journal are not significant on their own.
- QKV_NOWAIT confirmed with 6 interleaved rounds (track B, us per step, mean): B=1 4050.5 -> 4040.9, B=8
  4227.7 -> 4204.7.
- Final long-context check (tests/test_ctx.py 9216 1 2 8, final source, both tracks, against HF float32 at 159
  checked positions incl. the last 64): CTX PASS on both; row 0 bitwise equal alone / in 2 / in 8 (split path
  at B=1-2, one-block path at B=8); track B median rel. L2 0.005-0.009 (HF BF16 0.011-0.021), track A
  0.013-0.021 (HF BF16 0.011-0.021), all finite, top-1 vs float32 0.99-1.00.
- Wide-loop group batching GBW 1 / 3 (4 rounds, B=8 generate min): 4180 / 4396 against 4185; GBW 1 again with 6
  rounds: 4186.5 against 4188.9 (B=1 equal). Left at 2.

## Addendum to "How to rebuild and re-measure" (state at the end of the run)
- Track A with 3+ sequences no longer has a PH_NORM phase: the loaders normalise (NORM_LOAD, NORM_NB, NORM_SX).
- Scheduling switches added in the last hours, all on by default: XSLICE (per-warp input slices, 1-2 sequences),
  ENTRY_NB (no entry barrier, 3+), NB_MAT / NB_MAT_W (128-block tables; track A: 132 with 1-2 sequences),
  QKV_NOWAIT (attention-free blocks skip the q/k/v wait; track A only with 3+), PREP_FENCE 0, TOUCH_VS 4,
  VBATCH 6 / VBATCH_S 4. Switches kept off for reference: KV_IL, KV_PAD, COORD_RMS, XS_ALL.
- tests/abw.sh N "PROMPT NEW B..." "label:defs"... and tests/abctx.sh: interleaved A/B on a workload or over the
  context range; tests/workload.py takes MAXCTX=... to size the engine like a longer workload.
- A/B differences under ~15 us (B=1) / ~25 us (B=8) are within the process-to-process spread.

## Ideas from the other runs (progress/, read at 37.6 h)
- s4-north-h100-a1 (another model, H100) found that under a saturated HBM queue every atomic / poll / fence waits
  for the bytes in flight, and went to 2 loads in flight per thread. Here the narrow loop requests GB = 3 groups
  (9 loads) back to back; GB 2 / 1 (3 rounds): B=1 4041 -> 4104 / 4109 us per step (B=8, which uses GBW, within
  noise). This engine already limits what is in flight where it matters (adaptive end-of-phase stash with 2
  loads per warp, gentle attention-phase stash, last-use weight loads); in the streaming loop deep queues pay.
- Their attention for one sequence is split into 8 self-contained position chunks merged at the end (flat in
  context). Here the split path is used from 1136 positions; at shorter contexts its extra grid barrier costs
  more than it saves. A chunk-merge formulation without that barrier would need the same arithmetic in the
  one-block path (bitwise batch invariance) - a rewrite of both attention paths, not attempted in the time left.
- s3-q8b-online-a1 (same model, online serving): wgmma + bulk copies for its prefill GEMMs and attention; no
  decode-matvec technique that applies to a phase already at 94-97% of the HBM rate.
- Attention units on every S-th block (S = 132 / 8B: blocks 0, 16, ... at B=1; even blocks at B=8) instead of
  blocks 0..8B-1, in case neighbouring blocks share a GPC's bandwidth: bitwise equal, but (3 rounds) B=1 4052 ->
  4067, B=2 4092 -> 4118, B=8 4217 -> 4228 us per step. Reverted.
- One cold o tile (not two) for the attention blocks with 6+ sequences (others then up to 3): B=8 4191 -> 4253
  us per step (3 rounds). Two stays.
- s0061 = s0059's source evaluated again (pod epoch 9): 4026.1 / 4193.9 us generate (step 4030.4 / 4205.2)
  against s0059's 4047.0 / 4196.2 on epoch 8: the headline of one unchanged engine moves by ~20 us between pods.
- s0062 = s0060's source (track A) evaluated again: 5236.6 / 5372.3 us generate (step 5243.1 / 5385.1),
  against 5234.3 / 5377.8.
- o-phase prefetch sweep (3 rounds): NPF_ATT 9 / 3, O2PF 3, NPF_STW 12 against base; then NPF_ATT 2 / 3 / 4
  against 6 with 5 rounds (us per step, mean B=1 / B=8): 4048 / 4204 -> 4040 / 4193, 4043 / 4195, 4043 / 4196.
  The attention blocks' L2 prefetch of their cold o tile is now 3 loads per warp (was 6).
Measured s0063 (track B, NPF_ATT 3): 4020.0 / 4180.3 us generate (step 4025.1 / 4192.2) against s0061's
4026.1 / 4193.9 on the same pod epoch - new best at both batch sizes.
- Track A, NPF_ATT 6 / 3 / 2 (4 rounds, generate min B=1 / B=8): 5228 / 5374, 5228 / 5356, 5226 / 5357. Measured
  s0064 (track A, NPF_ATT 3): 5235.2 / 5370.2 us generate (step 5241.8 / 5383.4) - equal to s0062.
- Further prefetch depths, track B (4 rounds, generate min B=1 / B=8): base 4019 / 4182; NPF_ST 2: 4037 / 4180;
  NPF_ST 6: 4021 / 4181; NPF_DN 2: 4020 / 4185; WPF_DEPTH 2: 4031 / 4183. Unchanged.

## Workloads at s0063 (track B) / s0064 (track A) - the final source - pod epoch 9, ~38.7 h
us per token, generate() / step() + argmax + host read:
| workload | track B B=1 | track B B=8 | track A B=1 | track A B=8 |
| prompt 64 + 1024 | 4120 / 4163 | 4407 / 4452 | 5343 / 5384 | 5636 / 5680 |
| prompt 8192 + 1024 | 4732 / 4777 | 7863 / 7905 | (5948 / 5989) | (9131 / 9177) |
(track A prompt 8192 from the s0060 source, same logic apart from a prefetch depth.)
Against SGLang 0.5.18 + torch.compile on prompt 64 (6066 / 6836) through step() with the host read: track A
1.127x / 1.204x, track B 1.457x / 1.535x.
Caution on prompt 8192 at B=8: about one process in four runs it ~7% slower (track B 8580 instead of
7860-7970; track A 9772 instead of 9131) with the same library; re-runs in new processes, in either batch
order, give the normal value. Not order-dependent (B=1 then B=8 in one process: 4732 / 7863). It matches the
K/V-address sensitivity measured with padding (one padding gave +575 us): the physical placement of the K/V
cache, which differs between processes, decides how 64 units streaming at the same pace spread over the HBM
channels. A layout that is robust to this (rather than one tuned to a size) is the open item for the long-context
B=8 case.
Evaluator, best per track: track B s0063 4020.0 / 4180.3 us per step (generate, B=1 / B=8); track A s0053
5229.7 (B=1) / s0064 5370.2 (B=8).
- The prompt-8192 B=8 outliers, sampled: 8 processes each (prompt 8192 + 256, engine sized 9216, generate us per
  token): KV_PAD 0: 8392, 7795, 7796, 7745, 7795, 7794, 7795, 7707; KV_PAD 2: 7698, 7736, 7786, 7786, 8401,
  7792, 7791, 8398. Two modes (~7750-7800 and ~8400), about one process in five or six in the slow one, with
  or without padding: the slow mode is a property of where the process's buffers land, not of the stride. The
  earlier "padding 2 gives -195 us" was one sample of each mode. Not resolved; KV_PAD stays 0.
- Ten more processes (tests/kvaddr.py: prompt 8192 + 128, B=8, engine sized 9216; mk_stat 4-6 now returns the
  K/V and weight buffer addresses): 7691, 7782, 7781, 7732, 7692, 7692, 7781, 7779, 7777, 7732 us per token -
  no slow-mode sample this time, but three small modes (~7690 / ~7730 / ~7780) that do not follow the virtual
  addresses (offsets mod 1 GB of the K/V buffers vary independently of the mode; physical placement is not
  visible from here). Left as an open item: about 1 in 6 processes earlier at ~8400, otherwise 7690-7800.
- s0065 = the committed final source (s0063's kernel + host-side diagnostic export), track B, same pod epoch 9:
  4047.3 / 4174.4 us generate (step 4051.2 / 4186.5) against s0063's 4020.0 / 4180.3: B=1 +27 with the same
  kernel - another sample of the ~20-30 us evaluator spread between evaluations of one engine.
- s0066 = the committed final source built as track A: 5235.7 / 5367.5 us generate (step 5241.5 / 5380.5),
  equal to s0064. Both final builds are on the board as s0065 (B) and s0066 (A).
- Last check of the committed source (track B, pod epoch 9): whole suite passes, 0 spills; time_step generate
  4019.8 / 4179.8 us (B=1 / B=8).
- Same for track A: whole suite passes, 0 spills; time_step generate 5228.9 / 5366.9 us.
- Last stress tests of the committed source (4 min each): track B 1356 segments / 52758 steps, track A 1071 /
  41786 - all bitwise repeatable and finite. End of the run.

Final summary of the run: notes/SUMMARY.md (best: s0063, 4020.0 us per step at B=1, track B).
