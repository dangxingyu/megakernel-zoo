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

# Run 2: from the decode kernel to an online engine (2026-11-10 ...)

Workload: replay of the Mooncake tool & agent trace, 189 requests in 120 s, 1.73 M prompt tokens (mean 9166,
median 6608, nine at the 32768 cut), 38 k output tokens (median 34). With an ideal unlimited prefix cache on
512-token blocks 30.5% of the prompt tokens repeat (1.204 M unique). Headline = share of requests with first
token <= 5 s and mean time between tokens <= 0.1 s. Lines to beat: SGLang 0.672, vLLM 0.64.

## s0001: inherited engine, unchanged
Expected: offline valid (~5.35 ms), online stops at the `online_api` gate. Measured: offline valid
(5351 us B=1, 5688 us B=8, generate 5348 / 5678), but the online run stopped EARLIER than I said: the engine
cannot even be constructed with max_batch=64, max_context=34816 (`mk_create: unsupported`; the kernel has
MAXB = 8 and a context limit of 3904 positions set by its shared-memory attention scratch). Status incorrect.

## Decision: new kernels for the pass (engine/eng.cu), the inherited kernel is parked (tests/inherited/)
Arithmetic of the workload first. 1.2-1.7 M prompt tokens in 120 s is 10-14 k tokens/s of prefill; at
1.39e10 flop per token (2 x 6.95e9 layer weights; lm_head only for the last column) that is 140-200 TFLOPS of
bf16 GEMM sustained if nothing is reused, plus attention, which grows with the square of the prompt. The inherited kernel is a
matrix-VECTOR product with at most 8 columns: its loop reads each weight once and uses it for 8 columns, and
all of its engineering (229 KB of shared memory for one column set, block-team tiles) is about streaming. A
prefill needs the opposite regime: the weights of a row block stay in L2/shared memory and are used for
hundreds of columns. So the pass is new code: GEMM over T token columns, attention over runs of consecutive
positions, KV in pages. Decode rows are just columns of the same pass (rung `prefill_with_decode` by
construction). Price, stated plainly: the offline step of this engine is 18.5 ms (B=1 and B=8) instead of
5.35 ms, because the GEMM at small T streams weights at 1.2 TB/s through shared memory and a pass is ~190
launches. That is a regression in steady decode that I accept for now and must win back (the pass costs at
least 16-18 ms even when it carries only decode rows).

Canonical arithmetic (so that batch composition, chunking and prefix reuse cannot change a token): a matvec
output is one f32 accumulator per (row, column), ksteps ascending; attention is cut at ABSOLUTE positions
into segments of 128, each segment gives (max, sum, weighted V sum) and segments are folded in ascending
order. A query's result therefore does not depend on which pass, which chunk or which neighbour it had.
tests/test_eng.py: step alone == step in batch of 8 == prefill in chunks of 1 / all / 7,33,64 / several rows
together: all bitwise equal; error vs HF float32 median 0.0043-0.0062 (HF bf16: 0.011-0.013).

## GEMM kernel: what bounds it (tests/gemm_bench.cu, tests/mma_rate.cu)
- tensor cores alone, operands from shared memory: ~650 TFLOPS (any warp count 4..32, any tile shape).
- v1 (16 warps x 32x32 register tile, 128x128 block, 8 loads per thread up front per chunk): 255 TFLOPS at
  T=1024. Ablation: no weight loads 286, no input loads 284, neither 323, neither + no smem stores 369,
  + no block barrier 418. So staging (global load -> shared memory) costs as much as the multiply.
- v2a (8 warps x 64x64 tile, 128x256 block, ONE load requested and stored around each kstep): 239. One load
  in flight per warp is latency bound, same lesson as the decode kernel's pipeline.
- v2 (same, two groups of 12 loads per chunk): 324-334 TFLOPS. Kept.
- v3 (16 warps x 32x64, three groups of 4): 292. More warps did not help; saved as tests/eng_gemm_v3_16warps.cu.txt.
- v2 ablation: no loads 432-460, no loads/stores 492-534, no barrier 542-579. Still ~40% of the time is
  staging. Ideas not yet tried: larger K chunk per barrier, 2 blocks sharing A, cp.async-like prefetch.
- cache operator on the weight loads (cs / lu): +2% at T=1024, kept default.
A microbenchmark bug cost 10 minutes: my first mma_rate.cu indexed shared memory out of bounds and printed
millions of TFLOPS; the numbers above are from the fixed version.

## Attention kernel
v1 (scores to shared memory, softmax by threads, V sum as second MMA from shared memory, 128 queries x one
kv head per block): 43% of a pass, 1446 us per launch at 1024 columns and context 2-4k. v2: a warp holds 16
query rows (4 positions x 4 heads) and keeps the scores of a 128-position segment in registers; the bf16
softmax weights of a tile ARE the A fragment of the weights x V product; K and V are stored as B fragments.
No shared-memory round trip, no barrier. Prefill path: one warp per (4 positions, kv head), segments in
sequence; decode path: one block per (row, kv head), 8 warps take 8 segments in parallel, warp 0 folds in
order (same code, same result). Measured pass of 1024 prefill columns: context 0: 77 -> 54 ms, context 4096:
~170 -> 78 ms. Ablation (all K/V reads from the first tile, L1-resident): 69 ms at context 4096, so K/V
streaming is ~10% and the rest is compute.
Kernel cost model fitted from tests/bench_pass.py (used by the scheduler): pass = 4 ms + 52 us per column
+ 5.6 ns per (query, key) pair + 25 ns per (decode row, key); a pass costs at least ~16-18 ms.

## s0002: first online engine (FIFO, 1024 columns per pass, no reuse)
Expected: valid if the backlog drains within the 90 s after the last arrival; slo share 0.02-0.05.
Measured: INCORRECT, liveness: 163 of 189 finished when the driver stopped at 210 s (26 still running),
slo 2/189, TTFT p50 39 s, time between tokens p50 113 ms. All token gates passed (probes 12/12, reference
8/8 + 8/8 identical, pages conserved). My estimate of the total work (175-195 s) was too low: 1.59 M
prefill columns in 1585 passes of ~125 ms = 8.0 k tokens/s, and attention at long context costs more than
I budgeted. Conclusion: without prefix reuse and without a policy the engine cannot even finish; throughput
and policy both have to change. The quick evaluation (40 s of trace) had passed and hid this: the quick run
is not a liveness test for the full one.

## s0003: scheduler + prefix page cache on the first kernels (invalid: the engine stalls itself)
engine/sched.py (pure Python, also runs against a cost model in tests/sim.py): KV pages of 512 positions with
reference counts; complete prompt pages registered under the hash of the whole prefix; a request adopts cached
leading pages, waits for a page another running request is computing, computes the rest; unused cached pages
are evicted leaf first, least recently used; an evicted prefix is recomputed by whoever needs it next. Prefill
order: Moore-Hodgson over the estimated remaining work (the set of requests that can still make 5 s, largest
demoted first). tests/test_online.py: 12 requests with shared prefixes, pool of 12 pages (41 evictions, 22
recomputed pages), every freed page poisoned with NaN: 12/12 identical to the request alone, pages conserved.
Expected: valid, slo share 0.6-0.8 (simulation 0.81). Measured: INVALID: the replay had not finished 300 s
after the driver gave up (the next phase then tripped over leftover tokens: KeyError in the runner).
Cause (tests/replay.py reproduces the evaluator's replay with a stall watchdog): my cost model said
4 ms + 52 us per column; real passes of ~200 columns cost 3x that. The scheduler multiplied its whole
estimate by a measured correction (up to 2.5), so the budget shrank, passes got smaller, and small passes are
the expensive ones: 2543 passes of 197 columns and 78 ms. A model with a multiplicative correction and a
wrong fixed term is unstable. Two wrong predictions in a row (s0002, s0003) came from the same place: I
had fitted the cost at one shape (1024 columns, context <= 8k) and extrapolated.

## Measuring the pass instead of guessing (tests/calib_cost.py)
Grid over prefill columns T x context, decode rows D x context, mixes. What it showed (first kernels):
- GEMM cost is quantized: a column block of 256 costs the same full or empty (staging dominates a unit), and
  units are dealt to the 132 SMs in rounds, so 768 columns cost MORE than 1024 (60 vs 54 ms), 272 as much as
  512, 1536 as much as 2040. Until the GEMM is balanced, the scheduler composes passes for size classes
  256 / 512 / 1024 / 2048 columns in total (24.9 / 35.3 / 53.4 / 103.4 ms at context 0).
- prefill attention: flat 70 ms for T <= 256 at context 16384, 109 ms at 28672: a FLOOR of 3.1 us x context
  per pass, because one warp streamed the whole K/V of a (4 positions, kv head) task with serialized loads
  (11 us per 128-key segment). Any pass that touched a long prompt lasted > 100 ms whatever its size. With
  this floor no policy finishes the trace (simulation with the measured costs: 47 unfinished; real replay
  with 250 ms passes: 182/189): this was the reason for the invalid runs, not the policy.

## Attention v3 / v4 (the change that made the engine viable)
Canonical arithmetic, second level: positions are also cut into super-segments of 4096 (32 segments). A
(query unit, kv head, super-segment) part runs anywhere and leaves (M, Z, O); parts are folded in ascending
order (k_attn_fold; with one super-segment the fold is the identity and is skipped). Two kernels, same
arithmetic per row: k_attn_b: a block takes 32 consecutive positions x kv head x super-segment, all 256
threads copy the segment's K and V (64 KB) into shared memory once, double buffered, 8 warps compute 4
positions each; k_attn_w: a warp takes (unit, kv head, super-segment) and reads K/V itself (decode rows,
short runs). Tasks claimed from a counter.
v3 measured: floor gone (T=64 at context 16384: 69 -> 27 ms), 6.0 -> 3.0 ns per (query, key) pair.
Ablation of v3 (T=1024, context 16384, attention = 52 ms of 105): no exp -3, no bf16 conversion -4.5, no
score MMAs +2, no V MMAs +7.5, no staging -12, no fold -16, all four -22. So neither the tensor cores nor
memory bound it (per-pair cost is flat from context 4k to 28k; ordering block tasks so that concurrent
blocks share K/V in L2 changed nothing): it is instruction count. SASS of the segment loop: ~1400
instructions per thread and segment, of which 842 were register MOVES: the MMA needs its A operand in four
consecutive registers and my q fragment was assembled from two 8-byte loads, so the compiler copied it
before every MMA pair (256 moves per segment), plus the fold through shared memory (LDS+FMUL+FMA+STS x 64).
v4: (1) q is stored by k_prep as ready A fragments per unit (one 16-byte load, no moves), with the score
scale log2(e)/sqrt(128) folded in; (2) the softmax reference of a segment is the running maximum
R = max(M, m), so the weights are already in the scale of the state and the weighted V sum accumulates by
MMA directly into O in registers (64 multiplies by 2^(M-R), exactly 1 when the maximum did not move,
instead of 256 instructions of fold); (3) Z is one more MMA per tile against a column of ones instead of
64 shifts + 64 adds; (4) cvt.rn.bf16.f32 and the compiler's F2FP pack for the weights. 4160 -> 3152 static
instructions, attention 52 -> 40.5 ms: 2.35 ns per pair. Now ~40% of a block-segment is MMA at the tensor
cores' own rate (2112 MMAs at 1.2 per ns).
tests/scalar_rate.cu, for the record: an SM executes 3-13 simple warp-instructions per ns (LDS+FADD: 13 with
16+ warps, 6.9 with 8), ~3 per ns for a mix with EX2/CVT; the "one warp-instruction per cycle" rule of
thumb in the first run's journal does not hold in general, only for load-latency chains.
Correctness (tests/test_long_eng.py, prompt of 34000 tokens, HF float32 chunked with a KV cache as
reference, NaN in every free page): rel L2 median 0.0031 / max 0.0038 at 41 positions up to 33999 (HF bf16
itself: 0.0072 / 0.0130), top-1 41/41; a second run with 108 chunks of other sizes (1..2040 columns,
block kernel, warp kernel, ragged groups, super-segment and page boundaries) is bitwise identical at all 41
positions. A 34000-token prompt alone prefills in 3.3 s.

## s0004: attention rewrite + size-class scheduler
Local replay before submitting (tests/replay.py, same driver loop as the evaluator): 189/189 finished at
152 s, 157/189 = 0.831 within the service levels, TTFT p50 2.25 s / p90 14.7 s, time between tokens p50 73 /
p99 81 ms, longest gap 90 ms, probes 4/4 identical, pages conserved, cost model within 1.3% (153.4 s
estimated, 151.5 s measured).
What the misses are: all 32 are long prompts (mean 24k tokens, 44% of all prompt tokens). The trace's
arrivals are quantized to 10 s: 11-22 requests arrive at the same instant, 60-245k prompt tokens per
burst, and a burst has 5 s. So the headline is "how many tokens of a burst can be prefilled in 5 s while
the streams keep their 100 ms": about 65-75k at the current 13-16k tokens/s. Policy is at its optimum for a
single machine (Moore-Hodgson); the rest is raw speed: GEMM (334 TFLOPS, 42 us per token, now 2/3 of a
pass) first.
Measured (s0004, sealed evaluation): VALID. slo_share_r2 = 0.831 (157/189; SGLang 0.672, vLLM 0.64), TTFT p50
2.25 s / p90 14.6 s, time between tokens p50 73 / p99 83 ms, longest gap 90 ms (vLLM 0.5 s, SGLang 7.1 s),
replay drained at 152 s. Schedule gate 12/12 probes identical, online_vs_step 8/8 alone and 8/8 together,
pages conserved at every look. Prediction (0.78-0.85) confirmed; the local replay predicted the evaluator to
the third digit, so tests/replay.py is a trustworthy stand-in from now on.
Offline, for the record: 18.4 ms per step at B=1 and B=8 (the inherited kernel: 5.35 / 5.69 ms). The pass
engine pays ~19-21 ms for a pass with only decode rows: weights streamed through shared memory at 1.2 TB/s
plus ~190 launches. To win back later (small-T path of the GEMM).

## Where the time went after s0004 (nsys over a 40 s replay)
k_gemm 60.7% of kernel time, decode-row attention (k_attn_w) 18.7%, prefill attention (k_attn_b) 17.4%, prep 1.9%,
fold 1.0%; GPU busy 93% under the profiler.
Decode-row attention: a warp reads K/V itself at 10 KB/us (one 512-byte load per ~50 ns, strictly one after
the other), so a super-segment of one (row, kv head) takes ~105-140 us per layer whatever else happens; with
real (distinct) sequences it is bounded by HBM anyway (147 KB per token of context per decode token:
0.5 ms per row at 10k context). Tried: the block kernel for decode rows (8 warps stage K/V at 52 KB/us, one
computes): better only up to ~4-8 rows (5 -> 3 ms per pass), worse beyond; kept for passes whose decode
tasks fit one round of blocks. Smaller super-segments (2048, 1024): 2 ms better at 12 rows, worse with many
rows and for prefill (more parts to fold); kept 4096 (now a build parameter SS_LOG).

## GEMM: four designs at 255-334 TFLOPS, and what the limit was
Everything below at T = 1024 columns, 24576 x 4096 (gate+up), per SM.
- v2 (8 warps x 64x64 register tile, block 128x256, two bursts of 12 global loads per chunk): 334.
- v3/v7 (16 warps x 32x64, bursts of 4 / of 8+4): 292 / 301. v6 (one load at a time between MMA groups): 195.
- v2 with the stores right after the loads, bursts staggered between even and odd warps: 255, stagger irrelevant.
- Per-unit time is the same with 4 or 132 SMs active (103 us per 128x256x4096 unit): nothing shared between
  SMs limits it (my first reading of the bench, "57 us with few SMs", compared different block shapes).
- tests/copy_rate.cu: global (L2) -> shared memory copy of an SM: 10 KB/us per warp that loads, linear up to
  ~55 KB/us at 8+ warps, independent of how many loads a thread requests back to back.
- tests/roles_rate.cu: loader warps next to MMA-only warps: the chunk takes copy time + multiply time, as
  with one set of warps. tests/l1_stage.cu: no shared memory at all, operands read through L1 with or without
  touch-ahead by an owner warp: 322 TFLOPS, the same. tests/l1_rate.cu: L1 hits 600 TFLOPS, L1 misses 170,
  prefetch.global.L1 useless.
  So on this SM a warp-driven copy and the tensor cores do not overlap: 96 KB per chunk at 55 KB/us = 1.8 us
  plus 2048 MMAs at 1.2 per ns = 1.7 us gives the 3.2-3.5 us per chunk every design landed on. I spent about
  two hours producing variations of one design against a wall I had not identified.
- What identified it: asking what the vendor GEMM does. torch.matmul on the same shapes: 786-820 TFLOPS
  (tests/cublas_ref.py; a ceiling reference, never part of the engine). nsys shows its launches: kernels
  "nvjet_sm90_..._256x128_64x4_..", 132 blocks x 384 threads, 168 registers, 213 KB shared memory. That is the
  Hopper path: wgmma (warpgroup MMA: 4 warps, 64 x N x 16 per instruction, both operands read from shared
  memory through descriptors) and cp.async.bulk (the copy engine moves bytes global -> shared memory; no warp
  executes a load) with mbarriers. I did not know these instructions well enough to reach for them unprompted;
  the instruction forms are from the CUTLASS headers (read in the pod; nothing of CUTLASS is compiled in).
New kernel (engine/eng.cu k_gemm, generator engine/gen_wg.py): block = 2 consumer warpgroups + 1 producer warp,
128 rows x 2 NB columns, NB in {16, 32, 64, 128} by the number of columns, 4 stages of 64 K in shared memory,
"full" / "empty" mbarriers per stage. Weights and activations are stored as [k/64][row/8][(k%64)/8][row%8][k%8]:
a K chunk of a row block is one contiguous range for the copy engine and, unchanged, the canonical K-major
wgmma operand without swizzle. Epilogues unchanged (the accumulator registers have the m16n8k16 structure).
tests/wg_bench.cu: 738 TFLOPS (4096 x 12288: 794), weights at 2.2-2.8 TB/s for T <= 64 (old kernel 1.2).
Numerically the engine's logits are bit-identical to the m16n8k16 version on the unit test (the tensor
cores evidently sum the 16 products of a K step the same way in both instructions), and a column's result
does not depend on NB or on its place in the block: all bitwise tests pass unchanged.
Pass costs (tests/calib_cost.py): 1024 columns at context 0: 53.4 -> 28.9 ms; 2040: 103 -> 57.6; 256: 24.9
-> 15.9; decode-only pass: 21 -> 9.6 ms; 1024 columns at context 16384: 93.7 -> 69.4 (attention is now the
larger part there). A column block still costs the same full or empty (1024 + 16 columns: 46.6 ms), so the
scheduler keeps the total a multiple of 256 (classes 256 .. 2048).

## s0005: GEMM on the Hopper path
Expected (local replay): 171/189 = 0.905, TTFT p50 1.34 s / p90 4.74 s, time between tokens p99 82 ms,
drained at 132 s; offline step 8-10 ms.
Measured (s0005): VALID, slo_share_r2 = 0.905 (171/189), TTFT p50 1.37 s / p90 4.76 s, time between tokens
p50 67 / p99 82 ms, longest gap 96 ms, drained at 132 s; probes 12/12, reference 8/8 + 8/8. Offline step
7.66 ms at B=1, 7.86 ms at B=8 (18.4 before; the inherited decode kernel: 5.35 / 5.69). Prediction confirmed
to the request.
The 18 misses are prompts of 15k-32k tokens in the four largest bursts (141k-246k prompt tokens arriving at
one instant); six of them miss by less than a second (5.1, 5.1, 5.2, 5.9, 6.0, 6.0 s).

## s0006: attention staging by the copy engine
After s0005 prefill attention was the largest part of a long-context pass (1024 columns at context 16k:
40 of 69 ms). Two changes in the block kernel: K and V of a segment come into shared memory by two
cp.async.bulk copies (they are contiguous 32 KB each in a page) with an mbarrier per buffer, instead of 16
loads + 16 stores per thread; the 68 multiplies that rescale the running state are skipped when no row of
the warp got a new maximum (a factor of exactly 1). 2.35 -> 1.63 ns per (query, key) pair: 1024 columns at
16k 69.4 -> 57.1 ms, at 28k 98.9 -> 77.7 ms. Same bits everywhere (unit, chunking, 34000-position, poison).
Scheduler: requests already past their first-token deadline are served shortest first; recomputing the
on-time set at every pass (instead of keeping "late" sticky) and a less conservative GPU share in the
estimates (0.85 instead of 0.7) were tried on the real replay: all four combinations give 177/189.
Expected 0.93-0.94; measured (s0006): VALID, 0.937 (177/189), TTFT p50 1.24 s / p90 3.93 s, time between
tokens p50 50 / p99 81 ms, longest gap 99 ms, offline step 7.53 / 7.73 ms.

## s0007: attention on the warpgroup MMA
One attention kernel for prefill and decode rows (the warp kernel is gone): K, V and q are stored as wgmma
operands; scores = 8 instructions m64n128k16 per warpgroup (16 positions x 4 heads x 128 keys), softmax in
registers, the packed weights of 16 positions are the REGISTER A operand of the value product, m64n136k16:
the 17th dim group of the V buffer is ones and gives Z in the same instruction. Tasks of two units (two
warpgroups) or one unit (decode rows, odd ends). First version (v6): no faster than the m16n8k16 kernel
(56.3 vs 57.1 ms). Ablation: tensor instructions 14.4 ms, scalar 6.8, copies + barriers ~6 of 27.5 ms: the
three ran one after the other. v7/v8: software pipeline (scores of segment s+1 issued before the scalar work
of segment s, two score register sets used in turn), no condition around any wgmma (ptxas warns "C7519 wgmma
instructions are serialized due to program dependence ... in divergent path" otherwise: inactive warpgroups
and groups past the end of the sequence run with weights 0 instead, an exact identity). 1.63 -> 1.36 ns per
(query, key) pair: 1024 columns at 16k 57.1 -> 52.3 ms, at 28k 77.7 -> 69.1 ms. Less than the ablation
suggested (I expected ~45 ms): removing any one of scores / values / scalar work still gives ~49 ms, so the
stages now overlap and the rest is something I have not isolated (per-segment barrier, mbarrier waits).
Bit-identical logits again (unit test, 34000-position test, poison test).
Decode rows with DISTINCT sequences (tests/dec_bench.py; my earlier decode numbers used rows aliasing one
sequence, which hides the memory traffic): 24 rows at context 16k cost 28.6 ms per pass, 12 rows 18.7 ms,
old and new kernel alike: that is 24 x 16k x 147 KB = 56 GB of K/V at ~3 TB/s, i.e. HBM-bound. Nothing to
gain there without storing less.
Expected 0.942 (local replay 178/189); measured (s0007): VALID, 0.942, TTFT p50 1.18 s / p90 3.70 s, time
between tokens p50 43 / p99 83 ms, longest gap 98 ms; offline step 7.67 / 7.88 ms.
Kernel shares now (1024-column passes, context 0-3k): GEMM 63% (at 738 TFLOPS, the vendor GEMM does 790),
attention 10%, k_prep 6%, the rest small. A 1024-column pass is 28.9 ms = 35k prompt tokens/s.

## Rejected: letting streams sit out a burst (not submitted)
The service level is on the MEAN time between tokens (measured: mean 47 ms, worst request 80 ms), so streams
own slack. Variant: while prompts that can still make 5 s are waiting, a stream that can afford it (or whose
request has already missed its first token) skips the pass. Simulation +1 request, real replay +1 (179/189),
but the longest gap between two tokens of a stream goes from 96 ms to 3.6 s. A prefill stalling the decoding
streams is exactly what this engine is supposed not to do (vLLM's longest stall on this trace: 0.5 s). One
request is not worth that; the code stays (MK_SKIP_DEC=1), default off.
What the 11 misses are now: prompts of 20-33k tokens in the 6 largest bursts; five of them miss by 0.15 to
1.4 s. In the simulation +10% speed is worth one or two requests: the curve is flat here.

## Load sweep (local replay, s0007 engine)
nominal 1 req/s: 97/103 = 0.942 within service levels (TTFT p50 1.16 s, time between tokens p99 77 ms);
2 req/s: 178/189 = 0.942; 3 req/s: 266/284 = 0.937 (TTFT p50 1.37 s); 4 req/s: 328/368 = 0.891 (TTFT p50
1.45 s, p90 5.7 s; drained at 147 s). Time between tokens p99 77-81 ms and longest gap 86-98 ms at every
rate: the pass budget holds under overload, the misses are all first-token misses of long prompts.

## Scheduler on the device (rung `no_host_scheduler`)
Until s0007 the host (Python) composed every pass, launched ~330 kernels for it and blocked on a token read:
one host sync per pass, and the GPU idle while the host thought (0.3 ms per pass).
Step 1: the scheduler as C++ (engine/dsched.h, ~500 lines, no allocation, fixed tables): request ring, slot
table, page free list + reference counts, prefix cache (open-addressing hash table with backward-shift
deletion, LRU list threaded through the pages, incremental count of reclaimable pages, hit counts cached per
request under a cache epoch), admission, Moore-Hodgson order, size-class choice, the plan of the pass (columns,
pages, attention units and tasks, selected columns), registration of finished prompt pages, deferred release
of finished requests. The same source builds for the host (tests/dsched_host.cpp) and tests/sim.py
--impl both steps it next to the Python reference at every pass: identical runs and estimates on all 5137
passes of the simulated replay, on 5669 passes with a 300-page pool (2035 evictions), and at 4 req/s. The
first comparison disagreed at pass 399: the simulator's Python side still had stream skipping on. No bug
found in the port by this test; the consistency check of the incremental counters also holds.
Step 2: k_sched (one block): all threads hash the pages of the prompts that arrived (the prefix chain is
built from page hashes), one thread runs ds_step. The pass kernels read their sizes from the plan header and
are launched with full grids; a kernel without work returns at once. Two GEMM variants (up to 64 columns /
more) are both launched, one returns.
Step 3: who launches. First the host launched step + pass without waiting (results by event query): same
headline, 0 syncs, but 5 ms of host time per pass for ~510 launches. Then tests/graph_while.cu: CUDA graphs
with conditional nodes work on this stack; a WHILE node loops 500 kernels at 0.91 us per kernel with no host
involved (host launches: 1.94 us); IF nodes cost 5.7 us each, so conditional kernels are plain kernels that
return early. The serving loop is now ONE graph: WHILE (a request is waiting, running or arriving)
{ k_sched; pass kernels; k_commit (tokens into the result block, step published, loop condition) }.
The host: submit = three asynchronous copies on a second stream (prompt into the arena, request record,
ring count); poll = write its clock (the device has none), read the count of finished steps and their
result blocks, and start the loop if it is not running (3 times in the whole replay: 5419 steps).
If the host does not read results for ~250 steps the loop stops itself instead of overwriting them.
Control words the host writes while kernels run are read with ld.cg (never from L1).
Local replay: 178/189 = 0.942 (as s0007), TTFT p50 1.08 s, time between tokens p99 80 ms, longest gap 93 ms,
poll at most 1-2 ms, probes 6/6 identical, NaN-poison test with eviction 12/12, pages conserved.
What the host still does, to be plain about it: it owns the clock, the arena allocation for prompts, the
rid <-> ring-index map, and a correction factor of the cost model computed from the times it observes.
Offline step()/generate() still use host-written plans (a fixed plan with a position offset).
Measured (s0008, sealed evaluation): VALID, slo_share_r2 0.942 (178/189), TTFT p50 1.13 s / p90 3.70 s, time
between tokens p50 44 / p99 80 ms, longest gap 94 ms; poll() at most 3 ms; the graph was launched 3 times for
5390 steps (5385 passes); host syncs 0; probes 12/12, reference 8/8 + 8/8, pages conserved. As predicted: the
headline did not move, the loop moved.

## A latent page-accounting bug, found by shrinking the ring (after s0008; not in any evaluated run: errors = 0 there)
tests with a request ring of 8 entries and a prompt arena of 9000 tokens (build flags -DDS_RING=8
-DARENA_TOK=9000) and a pool of 12 pages: 2 of 12 requests got wrong tokens and one page leaked.
Cause, in the scheduler logic itself (Python and C++ alike): admission counts every cached page that nobody
uses as reclaimable, but eviction only took LEAVES of the prefix tree. A request can hold a cached child
page without holding its parent (it computed the parent itself while another request registered that page
first, then adopted the child). The parent is then unused, not a leaf, and not evictable: the reservation
says a page is available, the allocator finds none. The Python reference raises in that case; the C++ port
returned page 0, i.e. handed a live page to a second request. Fix: eviction falls back to the least recently
used unused page even if it has cached children (their users keep valid data; new requests simply cannot reach
them any more). After the fix: 12/12 identical, errors 0, pages conserved in that configuration; the
simulator equivalence (Python vs C++) still holds on every pass. The allocator still counts an error instead
of failing loudly if the reservation is ever wrong; tests assert the counter is 0, and stats() reports it.
Lesson: the 12-page test with the normal ring had passed for the wrong reason (timing); the equivalence test
cannot find a bug both implementations share. tests/run_all.sh now runs all configurations.

## s0009: 64-row GEMM blocks + 8 stages for passes of up to 64 columns (and the eviction fix)
Profile of the offline step at B=1 (7.72 ms): GEMM 6.44 ms = 2.35 TB/s of weights, attention 0.38, k_prep 0.36,
rnorm 0.10, gaps between the ~190 launches ~0.4. Per matrix (tests/gemm_small.py, T=1): the 4096-row matrices
ran on 32 SMs only (32 blocks of 128 rows) at 1.85-2.1 TB/s. 64-row blocks: qkv 2.25, o 2.12, down 2.10 TB/s;
with 8 stages of look-ahead instead of 4: qkv 2.44, o 2.24, down 2.50, gate+up 2.88, lm_head 3.06 TB/s
(12 stages: no further gain). GEMMs of a step 6.44 -> 5.66 ms.
Expected: headline unchanged (0.942), offline step ~6.9 ms. Measured (s0009): VALID, 0.942 (178/189), TTFT p50
1.15 s, time between tokens p99 83 ms, longest gap 103 ms (was 94: one pass ran 3 ms over the 100 ms mark;
p99 is unchanged); offline step 6.82 ms at B=1, 6.99 ms at B=8 (was 7.75 / 7.96; inherited decode kernel
5.35 / 5.69). Rung beats_mpk is back; parity with the general engines on steady decode (6.07 ms at B=1)
is not: what is left above the 5.2 ms of pure streaming is ~4 us of start-up per small-matrix launch, two
tiny kernels per layer for the decode row (k_prep, k_attn: ~10 us each) and the gaps between ~190 launches.

## Attention: what "async" wgmma needs, and that it did not pay (not submitted)
The assembler tells when it turns wgmma.mma_async into a synchronous instruction; I had been filtering its remarks
out of my build output. Three of them:
  C7519 "serialized due to program dependence on compiler-inserted WG.AR in divergent path": a condition around a wgmma;
  C7514 "serialized due to non wgmma instructions reading accumulator registers of a wgmma between start and
        end of the pipeline stage": the submitted kernel (s0007, wait_group 1 pipeline) has this one, so its
        "pipeline" was synchronous all along, which is why the stages added up;
  C7512 "serialized due to insufficient register resources": a pipeline on wait_group 0 only, with two score
        sets + O + Z + the packed weights live (228 registers of 255).
v9 (wait_group 0 only, packed weights through shared memory): no remark, but 30 spills per segment: 56.1 ms
(1024 columns at context 16k) against 52.3. v10 (softmax segments of 64 instead of 128: 32 + 32 + 68
registers, no remark, no spill, bitwise invariances intact): 51.6 ms. So with wgmma really asynchronous the
kernel is no faster. Ablation of v10 (attention = 22.9 ms of the pass): skeleton (copies, barriers) 8.0;
scalar work alone (tensor instructions cut to 1 + 1) +3.5; tensor instructions alone (constant weights, no
maxima) +1.5; both together +14.9. The two cost 10 ms more together than apart: the scalar softmax
instructions and the tensor instructions of the same warps do not run at the same time in any arrangement I
found. I stop here: s0007's kernel stays (v10 is in tests/att_v10_seg64.cu.txt). 1.36 ns per (query, key)
pair is where prefill attention is.

## s0010: pass overheads
Where a 1024-column pass (28.7 ms) went besides the bare multiply (19.4 ms at 733 TFLOPS): GEMM epilogues 4.4,
k_prep 2.4, attention inside the chunk 1.2-3.9, the rest < 0.5.
- residual-add epilogue: the residual values of a column pair are requested together before anything is
  stored: -1.2 ms;
- k_prep: q and K written as one 8-byte store per lane instead of four 2-byte stores: k_prep 2.4 -> 1.2 ms
  (more blocks per SM for k_prep: no effect);
- GEMM: one stage of wgmma kept in flight (wait_group 1; no accumulator is read before the final wait 0, and
  the assembler has no remark): nothing at 1024 columns, +4% at 2048.
Pass of 1024 columns 28.8 -> 26.2 ms, 2040 columns 56.5 -> 52.3, decode-only pass 9.4 -> 8.5.
Not kept: GEMM units claimed from a counter by the producer (tests/gemm_dynamic_claim.cu.txt). The q/k/v
matrix has 192 units for 132 SMs at 1024 columns and takes two unit times (571 TFLOPS against 750 for the
others); I expected claiming to fix that. It cannot: units are indivisible, 192 jobs on 132 machines take two
job times however they are dealt (measured 92.9 us against 90.2). Smaller units (64 rows) are copy-bound.
Simulated sensitivity of the headline (device scheduler in the simulator): speed x1.05 -> 180/189, x1.1 ->
181, x1.2 -> 183, x1.4 -> 185. About one request per 5%.
Expected 0.947-0.952 (local replay 180/189, TTFT p50 1.10 s, time between tokens p99 78 ms, longest gap 99 ms).
Measured (s0010): VALID, slo_share_r2 0.952 (180/189), TTFT p50 1.10 s / p90 3.76 s, time between tokens p50 35 /
p99 78 ms, longest gap 101 ms; offline step 6.84 / 6.97 ms. Prediction confirmed (+2 requests).

## Cost-model correction only slows down (committed, rides with the next submission)
The measured/estimated correction factor was clamped to [0.8, 1.5]; below 1 it plans passes ABOVE the target
when the kernels ran faster than the model for a while, and the longest gap (101 ms in s0010) came from such a
pass. Now [1.0, 1.5]. Local replay twice: 180/189 both, longest gap 88-92 ms.

## What the board asks for besides the share (operator note on `mk board`, read at 6.3 h)
"The share alone will not be accepted as the result": (1) time between tokens p50 <= 25 ms and TTFT p50 / p90
no worse than the engines' (1.6 s / 4.9 s) at this rate; (2) offline step within 5% of the inherited kernel
(5.35 / 5.69 ms -> at most 5.62 / 5.97 ms): a lone decode row must not pay for a prefill-sized pass.
Where s0010 stands: TTFT p50 1.10 s / p90 3.76 s (fine), time between tokens p50 35 ms (not), offline step
6.84 / 6.97 ms (not). Both open points are the same thing: the fixed cost of a pass.

## generate(): the loop over the steps on the device (not yet submitted)
generate() was a host loop of eng_pass (190 launches per step). Now one CUDA graph per (batch, plan counts):
WHILE (step < end) { pass; k_gen_commit: position and step advance }, the position offset and the output step
read from device cells; the host launches once per stretch of steps with the same plan (it changes when the
rows enter a new super-segment of 4096 positions). tests/test_gen.py: generate == step-by-step greedy for
B = 1, 3, 8, with a host step in between, and across position 4096. Expected 6.6 ms; measured generate
6.40 ms at B=1, 6.52 at B=8 (step(): 6.80 / 6.94, unchanged: it returns logits to the host).

## Pacing the streams by policy: simulated, rejected
tests/sim.py reproduces s0010 (180/189, tbt p50 36 ms, TTFT p90 3.4 s), so I tried policies there first.
- After a pass longer than the pace (22-25 ms), passes of the streams alone until the mean is back: 159-164 of
  189 (0.84-0.87), TTFT p50 2.0-2.5 s, p90 6.3-9 s. The GPU is busy 98% of the replay; every decode-only pass
  is 8.6 ms of weights streamed for no prefill.
- Smaller passes throughout (target 30 ms instead of 80): 172/189, tbt p50 28 ms, TTFT p90 4.7 s; 25 ms:
  166/189, tbt p50 26 ms, TTFT p90 5.9 s. Better than pacing, still 8-14 requests lost, because a pass of 256
  columns costs 14.2 ms where 1024 cost 25.3.
Conclusion: tbt p50 <= 25 ms needs passes of 256-512 columns that are efficient, not a policy.

## GEMM block shapes per matrix and pass size (the mid-size passes were half empty)
tests/gemm_small.py over T: the GEMMs of a pass took 6.0 ms at 64 columns, 7.8 at 128, 11.1 at 256, 12.5 at
512, 19.5 at 1024. Not linear, and the reason is occupancy: a unit was 128 rows x 256 columns, so a 4096-row
matrix at 256 columns has 32 units for 132 SMs (o: 43.7 us at 256, 512, 768 and 1024 columns alike), and q/k/v
at 768-1024 columns has 144-192 units: two rounds.
The kernel body is a template over (NB columns per warpgroup, NT 64-row tiles); the arithmetic of an output
element does not depend on it (tests/gemm_shapes.py: all shapes, same bits). Measured every shape on every
matrix for T = 64..2048 (12 shapes; 128 x 3 tiles runs out of registers: C7511, 20x slower). One model fits
all of it within ~10%: a stage (64 of K) of a block takes 0.19 us + 0.072 us per 64 x 64 of block area, and a
matrix takes (units / 132, rounded up) x stage time x K / 64. gemm_shape() evaluates that for 6 shapes
(32x1, 64x1, 64x2, 64x3, 64x4, 128x2) on the host or, in the serving loop, in the kernel (one kernel per
epilogue with a switch over the bodies: no assembler remark; it costs 0.8 us more per launch than a kernel
with one body, which is what the two gated kernels cost before, so step()/generate() keep single-shape
kernels for <= 64 columns).
GEMMs of a pass, before -> after (ms): 96 columns 7.7 -> 5.6, 128: 7.8 -> 5.7, 192: 10.9 -> 7.0, 256: 11.1 ->
7.2, 384: 12.5 -> 9.6, 512: 13.2 -> 10.8, 640: 18.3 -> 14.4, 768: 18.4 -> 16.5, 1024: 21.2 -> 19.8, 1536: 34.2
-> 31.0, 2048: no change. Whole pass (calib_cost.py): 256 columns 14.2 -> 9.8 ms, 512: 16.2 -> 14.7.
What the fit says about small passes, checked afterwards: the 0.19 us per stage is the latency of four
dependent wgmma (45 ns + 0.58 ns per column each, one accumulator), so at <= 64 columns a unit takes
K / 16 x ~50 ns however little it copies: 12.3 us for K = 4096, 38 us for K = 12288, plus 2.6 us start-up per
kernel. Tried and dropped for <= 64 columns (all same bits): 8 columns per warpgroup (kept: 1-3%), look-ahead
of 12 / 16 stages (0), L2 prefetch by the idle consumers (slower), units of half / quarter tiles so that 128 /
256 SMs-worth of units stream the 4096-row matrices (0: it is the chain, not the copy rate).

## Pass target 80 -> 27 ms (the board's condition on time between tokens)
With 256-512 columns per pass no longer wasteful, shorter passes cost less. Simulator with the new classes
(128: 7.7 ms, 256: 9.65, 384: 12.9, 512: 14.3, 768: 22.2, 1024: 25.6, 1536: 39.3, 2048: 50.3) and replay:
  target 80 ms: sim 180/189, tbt p50 36 | replay 180/189, TTFT p50 1.04 / p90 3.49 s, tbt p50 34 / p99 76 ms
  target 32 ms: replay 175/189, TTFT p50 1.27 / p90 4.35 s, tbt p50 27 / p99 33 ms
  target 27 ms: sim 175/189, tbt p50 24 | replay 173/189, TTFT p50 1.36 / p90 4.62 s, tbt p50 23 / p99 28 ms,
                longest gap 30 ms
Adaptive targets in the simulator (27 ms unless a first-token deadline has < 0.5-3 s of slack, or unless the
prefill backlog is > 0.25-2 s; then 80 ms): 175-176/189 with tbt p99 back at 71-77 ms. Not worth it.
Decision: 27 ms. It gives up 5-7 requests of the share (0.952 -> ~0.92) for a median time between tokens of 23
ms instead of 34 and a p99 of 28 instead of 76, which is what the board asks for; TTFT p50 / p90 stay inside
1.6 / 4.9 s. Where the replay's 126 s of GPU time go (model): about 7000 passes x 7 ms of weights streamed
once per pass whatever it carries, decode rows reading their own K/V (147 KB per token of context per token),
prefill GEMM ~22 s, prefill attention ~17 s. The cost of short passes is the per-pass part.

## s0011: block shapes + generate() loop on the device, policy unchanged
Expected: headline unchanged 0.947-0.952 at the 80 ms target, tbt p50 ~34 / p99 ~76 ms, longest gap < 95 ms;
generate 6.3 / 6.45 ms per step, step() ~6.7-6.8 / 6.9.
Measured (s0011): VALID, 0.952 (180/189), TTFT p50 1.04 s / p90 3.49 s, tbt p50 34 / p99 76 ms, longest gap 88 ms
(was 101); offline step 6.73 / 6.87 ms, generate 6.32 / 6.44 ms (was 6.84 / 6.97 both). All as predicted; all
gates pass (generate == greedy steps included). Next: the pass target (one change, scheduling).

## s0012: pass target 27 ms
Expected: 0.915-0.926 (173-175/189), tbt p50 23 / p99 28 ms, longest gap ~30 ms, TTFT p50 1.3-1.4 s / p90 4.6-4.7 s.
Measured (s0012): VALID, 0.915 (173/189), TTFT p50 1.36 s / p90 4.63 s, time between tokens p50 22 / p90 27 / p99
28 ms, longest gap 30 ms; offline 6.77 / 6.90 (step), 6.32 / 6.44 (generate). Prediction confirmed.
The board's condition (1) is met (tbt p50 <= 25 ms, TTFT p50 / p90 <= 1.6 / 4.9 s) at 0.915 of the requests
within the service levels (engines: 0.64 / 0.67). The best share stays s0011's 0.952 at a median of 34 ms: the
two submissions are the two ends of the trade, and the knob is MK_TARGET_MS.
Open: condition (2), the offline step (6.32 ms generate, 6.77 ms step, against 5.62 allowed), and winning the
share back at 27 ms: both are the fixed cost of a small pass (sim: +10% speed = +2 requests).

## The fixed cost of a small pass (offline step 6.73 / 6.32 -> ~6.0 ms; not yet submitted)
nsys over decode steps (B = 1, context 64-320): per step k_gemm 5.50 ms (145 calls, 38 us mean), k_attn 0.38
(10.6 us per layer), k_prep 0.35 (9.8 us per layer, 3.0 to 17.1!), k_rnorm 0.11, the rest 0.03. Gaps between
kernels inside a graph: median 0.06 us; the cost of a kernel boundary is inside the kernel times (the smallest
kernels take 1.0-1.5 us).
1. step() as one graph launch (it was 190 launches): the graph is built per (batch, plan counts) and the token
   pointer of its embedding node and the logits pointer of its LM-head node are set per call
   (cudaGraphExecKernelNodeSetParams). step 6.72 -> 6.31 ms at B=1, 6.86 -> 6.44 at B=8.
   A bug of mine found by the test on the way: the position cell came from a ring of 64 pinned host entries; the
   host queues steps faster than they run, so the ring wrapped (test with 70 queued steps failed). The cells now
   travel by value in a 1-block kernel launched before the graph. (generate() had the same ring; the evaluator
   synchronizes after each generate, so s0011/s0012 were not affected.)
2. k_prep's 3-17 us: the last column of a run zero-fills the rest of its group of 16 V positions with 2-byte
   stores to memory that is not in the cache: 0.9 us per position. A run of one column (a decode row) that does
   not start a group needs none of it: the run before it cleared the same group. And when it does start a
   group, the 16 positions of a dim are two 16-byte stores. step 6.31 -> 6.08 ms; decode-only pass of 16 rows
   8.61 -> 8.00 ms. All NaN-poison tests pass.
3. GEMM at <= 64 columns, what bounds it (tests/gemm_shapes.py, one column, us per kernel; HBM floor at 3.07
   TB/s: qkv 16.4, o 10.9, gate+up 65.6, down 32.8): qkv 20.5, o 15.4, gate+up 69.9, down 41.4.
   - not the accumulation chain: K as two halves on two accumulators interleaved in a warpgroup: o 16, down 44;
   - not the consumer's fence / commit / wait: two stages per group: slower at every size (256 columns 7.2 ->
     7.8 ms);
   - the copy REQUESTS: a stage without the request for the activations: o 12.8, down 38.8; with the weights
     as two requests (three in all): o 17.5, down 47. About 40 ns per request, issued one after the other by
     one thread, on top of ~0.12 us per stage;
   - so: 4 producer threads (one per warp, stages dealt in turn; blocks of 384 threads) for the 64-row
     shapes: one column 5.30 -> 5.09 ms for the GEMMs of a step, 128 columns 5.87 -> 5.60, 256 columns 7.49 ->
     7.07. Shapes of 128 rows and more (4 stages): 0-3% slower with more producers, they keep one.
   - prefetch of the next GEMM kernel's first 4 stages into L2 by the consumers when their last unit is
     computed: step 6.09 -> 6.02 ms (2 stages 6.04, 8 stages 6.02).
   In the pass: step 6.08 -> 5.98 ms (less than the isolated kernels promise).
Not done, with reasons: K split over the two warpgroups of a block (a new canonical order for all shapes, and
with 64 streams the 4096-row matrices are then bound by HBM at ~2.6 TB/s anyway).

## Same-run comparisons only (a lesson), and the producer loop
Numbers from different pod commands differ by up to 5% on the GEMM benchmark (20.5 vs 19.5 ms at 1024 columns for
the same code), and a variant I had built as "the old behaviour" (one producer) was 0.38 ms per step slower
than the committed build: my first multi-producer loop (`if (gc % NPROD != prod) continue;` inside the loop over
all stages) cost that much by itself, and four producers were only hiding it. From here on every A/B is
library against library in one command, alternating, twice (tests/time_step2.py and tests/calib_cost.py with
MK_SO). Loop rewritten (producer p walks chunks p, p + NPROD, ..):
  committed build:              step 6.08 / 6.21 ms, pass of 128 columns 7.76, 256: 9.80, 512: 14.70, 16 decode rows 8.09
  one producer, new loop:       6.11 / 6.24, 7.76, 9.80, 14.87, 8.07    (= committed)
  four producers + prefetch:    5.98 / 6.12, 7.62, 9.50, 14.83, 7.89
Also measured this way: the refit of the shape costs that I made from the isolated benchmark (64x1 0.26, 64x2
0.52 us per stage) is worse in the pass (512 columns 15.0 -> 15.7 ms, 1024 + 16 rows 40.7 -> 43.8): reverted.

## Decode-row attention with the operand roles swapped (correct, bit-identical; gain small)
Where the simulator says the replay's GPU time goes at the 27 ms target: pass base (weights) 70.6 s, prefill
attention 18.0 s, decode-row attention 37.9 s (12.0 s in passes bound by the longest task, 25.9 s in passes
where all blocks are busy); decode rows read 76 TB of K/V (25 s at 3 TB/s). 5.7 decode rows per pass on average.
Concurrent decode rows share 3% of their pages: nothing to merge.
The general attention kernel computes 64 query rows x 128 keys per instruction; a decode row has 4 real rows.
tests/att_roles.cu: with the roles swapped (keys or dims on the 64-row side, the 4 heads as 8 columns) the
tensor cores give the same bits: 0 of 1024 scores, 0 of 1032 values/Z differ. So a unit of one position now
runs att_dec(): 16 + 24 instructions of 8 columns per segment instead of 16 of 128 / 136, the maxima per head
through shared memory, the weights written to shared memory as the B operand, output or part in the layout
k_attn_fold expects. All bitwise tests pass unchanged (prefill vs step, 34000-token context with folds).
Expected: per segment 2.8 -> ~1 us, decode attention 2-3x faster when few rows decode.
Measured: 1 row at context 16k: pass 9.05 -> 8.95 ms; 8 rows: 11.96 -> 11.75; 16 rows: 17.0 -> 16.6; a
64-320 token context (offline step): 5.98 -> 6.03 ms (slightly slower: two block barriers per segment).
Wrong: the arithmetic was never the limit. A block gets K and V of a segment (64 KB) from HBM at ~30 KB/us:
2.1 us per segment whatever is computed. Splitting each 32 KB copy into 2 / 4 / 8 requests: slower (1 row
9.10 / 9.36 / 9.56 ms).
Tried on the way: L2 prefetch of K/V 2 / 3 / 5 segments ahead by the block's idle threads: slower in every regime
(1 row 9.1 -> 9.3 ms; 32 rows at 30k: 53 -> 55 / 57 / 71 ms). Three K and three V buffers (two segments of
look-ahead) on one warpgroup: 1 row at 16k 9.10 -> 8.93 ms, i.e. still ~2.5 us per segment.
tests/kv_stride.cu then measured what a block can get: 8 blocks copying 64 KB segments three deep: 0.52 us per
segment (127 KB/us per block); 32 blocks 0.73 us (2.9 TB/s in all); 132 blocks 2.8 us (3.1 TB/s). The same
whether the segments are contiguous, 73.7 MB apart (the page pool), or anywhere: no address-translation
effect. So the data was never late with few rows; my swapped kernel was slow because it issues 40 instructions
per segment on ONE warpgroup, and an instruction costs a warpgroup ~40 ns whatever its width (16 x 8 columns is
no cheaper than 8 x 128).
Fix: the tiles go to three warpgroups side by side (blocks of 384 threads): scores 8 instructions on each of two
warpgroups (a key tile each), values 8 on each of three (two dim tiles and the ones). Same bits (all tests).
Same-run timings against the general kernel on one warpgroup (ms per pass; rows x context):
  offline step 5.98 -> 5.86 (B=1), 6.12 -> 6.01 (B=8)
  1 row: 2k 7.67 -> 7.13, 8k 9.10 -> 8.15, 16k 9.10 -> 8.17, 30k 9.11 -> 8.42
  16 rows: 8k 12.4 -> 10.6, 16k 17.0 -> 14.0;  32 rows: 8k 17.9 -> 14.5, 16k 27.3 -> 23.5, 30k 54.0 -> 55.8
  256 columns + 16 rows at 8k: 18.6 -> 17.1; 512 + 16: 24.9 -> 23.6; 256 + 32 rows at 16k: 33.7 -> 29.9
(the calibration's decode rows share pages, so the many-row numbers flatter both; with all blocks busy the
kernel is at what HBM gives.) One row at long context still pays ~1.9 us per segment (two block barriers and
16 instructions): 36 x 32 x 1.9 us = 2.2 ms per pass for a super-segment of 4096 positions.
Cost model: the task-bound term of the decode rows is 0.68 of the old coefficient (dsched.h DS_DLAT, sched.py).
Replay with it: 175/189, TTFT p50 1.32 / p90 4.49 s, tbt p50 22 / p99 30 ms. Per-pass log (MK_PASSLOG=1,
estimated vs measured): decode-only passes 12.7 vs 10.7 ms (c_small 8.6 was 2 ms too high: now 6.6), passes
with prompt columns 23-25 vs 24-27 ms (5-8% over). The correction factor averaged over all passes, so the
decode-only passes held it at 1.0: it now follows the passes with prompt columns only.

## s0013: step() as one graph launch; decode rows stop re-clearing their V group
Expected: step 6.08 / 6.21 ms, generate 6.09 / 6.22; online 0.915-0.926, tbt p50 22 / p99 28 ms, TTFT p50 1.3-1.4 s.
Measured (s0013): VALID, 0.921 (174/189), TTFT p50 1.36 s / p90 4.59 s, tbt p50 22 / p99 28 ms, longest gap 30 ms;
step 6.09 / 6.22 ms, generate 6.10 / 6.22 (s0012: 6.77 / 6.90 and 6.32 / 6.44). As predicted.

## s0014: four producer threads for 64-row blocks; prefetch of the next kernel's first stages
Expected: step 5.98-6.00 / 6.12 ms; online 0.915-0.926 (at most one request better), tbt p50 22 / p99 28 ms.
Measured (s0014): VALID, 0.921 (174/189), TTFT p50 1.37 s / p90 4.63 s, tbt p50 22 / p99 28 ms; step 5.99 / 6.13
ms, generate 6.00 / 6.13. As predicted (no change of the share).

## 1/rms inside the next GEMM up to 64 columns (two kernels per layer less)
k_rnorm (one warp per column: 256 partial sums -> 1/rms) ran before the q/k/v and the gate/up GEMM: 73 kernels of
~1.4 us per step. Up to 64 columns every block of those GEMMs needs all columns anyway, so its 8 consumer warps
compute the same sums (same order: same bits) into shared memory before the block's first barrier; the host
does not launch k_rnorm then, and in the serving loop k_rnorm returns at once.
Same-run: step 5.86 -> 5.81 ms (B=1), 6.01 -> 5.95 (B=8); decode pass of 16 rows 7.40 -> 7.35. All tests pass.
Batch 8 is now inside the board's 5% (5.97 ms); batch 1 is at 5.81 against 5.62.

## s0015: decode-row attention with the roles swapped, on three warpgroups
Expected: 0.921-0.931 (174-176), tbt p50 22-23 / p99 29-30 ms, TTFT p50 1.3-1.35 s / p90 4.5-4.6 s; step 5.86 / 6.01.
Measured (s0015): VALID, 0.921 (174/189), TTFT p50 1.37 s / p90 4.61 s, tbt p50 22 / p99 29 ms, longest gap 33 ms;
step 5.86 / 6.01 ms. The offline numbers as predicted; the share at the low end of the range (the decode rows of
the replay are mostly bound by HBM, as the simulator said: +1 request at best).

## s0016: 1/rms inside the next GEMM up to 64 columns
Expected: step 5.80-5.82 / 5.95-5.96 ms; online unchanged (174 +- 1).
Measured (s0016): VALID, 0.921 (174/189), tbt p50 23 / p99 29 ms; step 5.80 / 5.95 ms, generate 5.82 / 5.97.
As predicted. Offline since s0012: 6.77 -> 5.80 (B=1), 6.90 -> 5.95 (B=8); the board's lines are 5.62 / 5.97.

## What bounds the GEMM, measured apart (tests/wg_scale.cu, tests/kv_stride.cu)
tests/wg_scale.cu: warpgroups issuing wgmma on shared memory, nothing else (ns per instruction per warpgroup;
all 132 SMs):
  columns   1 warpgroup   2            3            4         TFLOPS (2 or more warpgroups)
     8        13.2 ns     19.7        29.6         39.4         219
    32        16.8        26.3        39.4         52.5         658
    64        21.7        35.0        52.5         74.2         988
   128        42.6        71.8       107.6        143.5         965
An SM's tensor throughput is reached with two warpgroups (a third or fourth adds nothing) and is ~965-988
TFLOPS over the GPU from 64 columns on; an 8-column instruction takes 13 ns alone, not 40. So:
- the engine's GEMM at 1024+ columns (750 TFLOPS) is NOT at the tensor limit. It is at the HBM limit: a block
  holds 128 x 256 accumulators, so the weights are read once per 256 columns: 4 x 15.1 GB per 1024-column pass
  = 20 ms at 3 TB/s (measured 19.8 ms); 2048 columns: 40 ms (measured 39-42); 512: 10 ms (measured 10.8).
  Wider blocks read the weights less often but copy more activations per stage (64 KB for 512 columns), and
  the copy engine of an SM carries 64-127 KB/us: a 64 x 512 block would need 126 KB/us to keep the tensor cores
  busy. The accumulators (128 registers x 256 threads) fix the block's area; HBM bounds it on one side, the
  copy engine on the other. Four consumer warpgroups (twice the accumulators: 128 x 512 per block) would give
  ~870-900 TFLOPS at 1024+ columns by my arithmetic; not built: the passes of the 27 ms target have 128-512
  columns, where the 4096-row matrices need small blocks to occupy the SMs at all.
- at one column the stage's 4 instructions are 0.08 us of its 0.15-0.19 us; `down` runs at 2.7 TB/s on 64
  streams and `o` at its floor: what is left above the weights is ~2 us per GEMM kernel start.
- the "40 ns per instruction" that I read into the decode-row attention was its two block barriers and
  mbarrier waits per segment, not the instructions (16 x 13-20 ns).

## Chained GEMM kernel for decode-size passes: tried, slower, dropped
k_chain: the o, gate/up and down GEMMs of a layer and the q/k/v GEMM of the next in ONE launch (the inherited
megakernel's structure), the phases separated by a barrier over the grid (fence, block barrier, one atomic add,
spin). Same bits (unit and generate tests). Expected ~1 us per replaced kernel boundary (0.1 ms per step).
Measured, alternating in one command: step 5.80 / 5.95 ms -> 5.90-5.99 / 6.07-6.14. A kernel boundary inside a
CUDA graph is already as cheap as a grid barrier, and in the chain all 132 blocks start every phase although o
and down have 64 units. What would be left to win is streaming the next phase's weights across the barrier;
the L2 prefetch of the next kernel's first stages already takes part of that. Not kept.
Where a decode step (5.80 ms) is now: GEMM kernels 5.43 (weights at 3.07 TB/s would be 4.93; ~2 us per
kernel start, and q/k/v, o, down have 96 / 64 / 64 streams), attention 0.25, k_prep 0.13.

## Who misses the first token at the 27 ms target, and a deadline boost
(A build-script bug of mine first: build.sh rebuilt only when eng.cu changed, so a change in dsched.h alone ran
with the old library in my local replays: two "no effect" results were of code that was not compiled. Fixed
(dsched.h is in the check). Submissions were never affected: the evaluator builds from scratch.)
Per-request records of the replay (tests/replay.py --out): all requests of a 10 s window arrive at once; the
15-16 misses are prompts of 16.6k-32.8k tokens in the windows with 165k-245k prompt tokens (t = 10 s: 5 misses,
50 s: 3, 110 s: 3). Three of them miss by 20-110 ms (5.02-5.11 s), every time. They are the last of the set the
deadline planner (Moore-Hodgson) keeps on time: its estimate of when they finish is ~0.4 s optimistic over 5 s.
- constants (share 0.45-0.7, margin 0.3-0.7): 173-174 of 189 in six replays, no effect.
- share from the state: the part of the GPU that prompts get is what the pass target leaves after the fixed cost
  of a pass (6 ms) and the streams decoding now, capped by `share` (at 27 ms with twenty long streams: a
  fifth, not 0.7). Simulator: 175 -> 176, TTFT p90 4.5 -> 4.15 s. Replay: 173, 173 (the same three near-misses):
  a better estimate alone only changes who is called late.
- deadline boost: while an on-time prompt has less than `boost_slack_s` to spare by the estimates, passes aim
  at `target_hi_s` (45 ms) instead of 27: larger passes carry more prompt columns per second. Replay: slack
  0.6 s: 174 (tbt p99 30 ms, longest gap 44); slack 1.0 s: 175 (two of the three near-misses at 4.69 / 4.87 s;
  tbt p50 23, p99 41 ms, longest gap 47 ms). Simulator (no estimation error): share unchanged, tbt p99 34-41.
  Sweep (replay): slack 1.5 s / 45 ms: 175, tbt p50 25; 2.0 s / 45: 175, tbt p50 26 (over the board's 25);
  1.0 s / 60 ms: 175, tbt p99 57; 1.5 s / 60: 175, p50 25, p99 58; 1.0 s / 35 ms: 174, p99 34.
  One of the three (a 32768-token prompt at t = 100 s) stays at 5.05-5.2 s in every setting.
Decision: boost_slack_s 1.0, target_hi_s 45 ms (MK_BOOST_S, MK_TARGET_HI_MS). It trades the tail of the time
between tokens (p99 29 -> 41 ms, longest gap 33 -> 47 ms, both far inside the 100 ms service level; the median
stays 23 ms) for two requests.

## s0017: deadline planner (share from the state, 45 ms passes while a prompt is short of slack)
Expected: 0.926 (175/189), TTFT p50 1.33 s / p90 4.5 s, tbt p50 23 ms, p99 41-43 ms, longest gap ~48 ms.
Measured (s0017): VALID, 0.926 (175/189), TTFT p50 1.32 s / p90 4.50 s, tbt p50 23 / p90 27 / p99 43 ms, longest
gap 48 ms; offline 5.80 / 5.95 ms. As predicted.
Also simulated and dropped: the streams of requests that already missed their first token sit out during a
boost: +0-1 request, and those streams stall 3.3-4.1 s.
State of the two ends of the trade: s0011 0.952 at a median time between tokens of 34 ms (80 ms passes);
s0017 0.926 at 23 ms (27 / 45 ms passes), with the offline step at 5.80 / 5.95 ms.

## Load sweep with the evaluator (tools/online_eval.py --rates "4 1", s0017's engine; not a submission)
All gates pass at both rates.
  nominal 4 req/s (368 requests in 120 s, 3.07 req/s): 0.837 within the service levels, TTFT p50 1.74 s / p90
  9.2 s / max 108 s (the bursts no longer fit: prompts queue), time between tokens p50 27 / p99 44 ms, longest
  gap 49 ms, drained 48 s after the last arrival;
  nominal 1 req/s (103 requests): TTFT p50 1.17 s / p90 4.27 s, time between tokens p50 21 / p99 37 ms.
Under overload the streams keep their pace (the pass target bounds them); what gives is the first token of
the prompts the deadline planner gives up on.

## q, K and V prepared in the q/k/v GEMM's epilogue up to 16 columns (k_prep leaves the decode step)
k_prep (per column 48 vectors of 128: RMSNorm of the head, RoPE, stores) was 3.5 us per layer of a decode step,
and the q/k/v values went through global memory in f32 to reach it. Up to 16 columns the q/k/v GEMM now runs
with units of 128 rows = ONE head of every column per block (48 units, 8 stages, 4 producers), writes the
block's vector of each column into shared memory instead of global memory, and after a block barrier a warp
per column does exactly what k_prep's warp did (the same function, prep_vec). Same bits (unit, generate,
long-context and online tests). In the serving loop k_prep returns at once for such passes.
Same-run, switch MK_PREPQ: step 5.775 -> 5.669 ms (B=1), 5.906 -> 5.825 (B=8). With it: 12 instead of 8 stages
of look-ahead for the 8- and 16-column shapes (-0.02 ms; 16 stages: +0.03).
Tried: L2 prefetch of the attention's first K/V segments from that epilogue: no effect, removed.
Replay unchanged: 175/189.

## s0018: q, K, V prepared in the q/k/v GEMM's epilogue up to 16 columns
Expected: step 5.66-5.68 / 5.82-5.83 ms, generate 0.02 above; online unchanged (175 +- 1).
Measured (s0018; evaluated after two pod replacements): VALID, 0.926 (175/189), tbt p50 23 / p99 41 ms; step
5.68 / 5.84 ms, generate 5.70 / 5.84. As predicted. The board's lines for the offline step are 5.62 / 5.97.

## Shortest first among prompts that arrive together (TTFT p50 1.33 -> 0.25 s)
The deadline order was (arrival, request id): inside a burst, trace order. All requests of a burst have the same
deadline, so any order keeps the same set on time; the shortest first gives the median request its first token
much earlier. Simulator: 176/189 either way, TTFT p50 1.20 -> 0.22 s.
First replay: no change at all (TTFT p50 1.33 s). The driver submits a burst's requests microseconds apart, so
no two arrival times are equal; the simulator submits them at the same instant. Now the order compares arrival
in quarters of a second (DS_ARRIVAL_Q), then the estimated time to the first token, then ring order; device and
Python schedulers agree pass by pass in four configurations.
Replay: TTFT p50 0.24-0.26 s, p90 4.4-4.6 s; 174 of 189 twice at boost slack 1.0 s (was 175): the early
finishers decode during the burst, and the last prompt of the on-time set at t = 10 s (16609 tokens) moves from
4.85 s to 5.0-5.1 s. Boost slack 1.2 s: 174, 175; 1.3 s: 175, 175 (that prompt at 4.95 / 4.99 s), tbt p50 24,
p99 43-45 ms; 1.5 s: 175 with tbt p50 26 (over the board's 25).
Also tried on the GEMM start (not kept): each mbarrier initialized by its own thread, and 1/rms computed by
every warp after the block barrier: step 5.68 -> 5.73 ms, 5.83 -> 6.08 at B=8.
Two more replays at 1.3 s: 175 and 174 with tbt p50 25 ms both times: no margin on the board's 25 ms.
Decision: shortest first with boost slack 1.2 s (tbt p50 24 ms). One request stands on the 5 s edge, so the share
is 174 or 175.
A latent bug found by the test suite on the new pod: eng_read_tokens (only used with MK_DEVSCHED=0, the Python
scheduler on the host) had no ctypes argument types, so the 64-bit engine handle went through a C int and the
call only worked while the process heap was below 4 GB; on this pod it crashed. Declared. The evaluated path
(device scheduler) never calls it.

## s0019: shortest first among prompts that arrive together
Expected: 0.921-0.926 (174-175), TTFT p50 0.25 s / p90 4.4-4.5 s, tbt p50 24 / p90 37-38 / p99 42-43 ms.
Measured (s0019): VALID, 0.926 (175/189), TTFT p50 0.237 s / p90 4.43 s (s0018: 1.33 / 4.52), tbt p50 24 / p90 38 /
p99 42 ms, longest gap 50 ms; offline 5.68 / 5.83 ms. As predicted (the request on the edge made it this time).

## Prefill attention: the two warpgroups of a block take their phases in different orders
From tests/wg_scale.cu: the tensor cores of an SM serve two warpgroups at 72 ns per 128-column instruction each
and one alone at 43 ns. In k_attn<2> both warpgroups did the same thing at the same time: scores of the next
segment (tensor), scalar work (maxima, weights: the tensor cores idle), values (tensor). And the task order is
(super-segment, kv head) major, so the 16 pair tasks of a chunk read the same K/V at the same time and the pass
is not bound by HBM (2.75 us per segment whether 64 or 520 tasks run): it is bound by this sequence.
Now warpgroup 0 does scores(s+1), scalar(s), values(s) and warpgroup 1 scalar(s), values(s), scores(s+1): one
warpgroup's instructions run while the other is in its scalar part. Each batch is waited for (the kernel's
"pipeline" was serialized by the assembler anyway, C7514), and the scores batch after the last segment is no
longer issued. Same bits (unit, long-context, online tests).
Same-run against the committed kernel (ms per pass, columns at context): 512 at 4096: 18.4 -> 17.5; 256 at 16k:
15.4 -> 14.5; 512 at 16k: 28.7 -> 25.6; 1024 at 16k: 53.5 -> 51.2; 512 at 28k: 35.7 -> 34.0; 1024 at 28k: 67.7 ->
64.2. The attention part: -8 to -15%. Cost model: c_seg 2.6 -> 2.35 us per segment, c_pair 1.45 -> 1.3 ns.
Replays (with s0019's order): 176 of 189 four times; the two requests that stood on the edge finish at 4.72-
4.83 s; TTFT p50 0.23 s / p90 4.15-4.2 s; tbt p50 22-23 ms, p99 38-40 ms. Boost slack 1.0 and 1.2 s give the
same share, 1.0 the better pace (tbt p90 31 against 34): back to 1.0.

## s0020: prefill attention with staggered warpgroups
Expected: 0.931 (176/189), TTFT p50 0.23-0.24 s / p90 4.15-4.2 s, tbt p50 22 / p90 31 / p99 39 ms; offline unchanged.
Measured (s0020): VALID, 0.931 (176/189), TTFT p50 0.246 s / p90 4.17 s, tbt p50 22 / p90 31 / p99 39 ms, longest gap
47 ms. Online as predicted. Offline 5.71 / 5.89 ms against 5.68 / 5.83 in s0019: NOT as predicted (I said
unchanged); to be checked same-run.

## Why s0020's offline step went up, and the fix: decode rows get a kernel of their own
Same-run A/B confirmed the evaluator: step 5.675 -> 5.712 ms (B=1), 5.823 -> 5.888 (B=8) with s0020's kernel.
The assembler's remarks told why: k_attn<1> (384 threads since the decode path moved onto three warpgroups) now
had "C7512: wgmma serialized due to insufficient register resources for the function": 384 threads leave 170
registers per thread and the general path wants ~228; its new phase order made the assembler serialize EVERY
wgmma of that kernel, the decode path's too. (Before, the kernel had C7514, which was confined to the general
path.)
Fix: two kernels. k_attn<1> is the general path again on 128 threads (odd ends of prompt runs); k_attn_dec is
the decode rows' kernel (384 threads, a dozen registers of state). The single-unit tasks of the plan are listed
with the one-position units first and their number in header element 6 (device planner and host plan alike), so
each kernel has its own range and none skips tasks; in the serving loop both are launched and the empty one
returns before touching shared memory. No assembler remark is left on any kernel.
Same-run: step 5.674 / 5.818 (s0019's kernel) vs 5.677 / 5.827 now. Decode passes: equal or slightly faster,
except the calibration's 16 / 32 rows at 16k context (14.0 -> 14.9, 23.7 -> 26.5 ms); those rows alias the same
pages, and three real replays are better than s0020's: 176/189 each, TTFT p90 4.07 s (4.17), tbt p90 27-28 ms
(31), p99 35-36 ms (39), longest gap 45 ms.

## s0021: decode-row attention in its own kernel
Expected: step 5.67-5.68 / 5.81-5.83 ms; 0.931 (176), TTFT p50 0.23-0.25 s / p90 4.05-4.1 s, tbt p50 22, p99 35-36 ms.
Measured (s0021): VALID, 0.931 (176/189), TTFT p50 0.247 s / p90 4.09 s, tbt p50 22 / p90 28 / p99 36 ms, longest
gap 44 ms; step 5.67 / 5.81 ms, generate 5.68 / 5.81. As predicted.

## k_attn_fold: threads without a position of the unit do nothing
nsys over 40 s of the replay, per pass: k_gemm 10.5 ms, k_attn_dec 4.9 (decode rows reading K/V), k_attn 3.6, k_prep
0.54, k_attn_fold 0.50, k_rnorm 0.15, k_sched 0.10. The fold kernel folded all 64 rows of a unit: for a decode
row 4 rows are real, and the other threads' part entries were not written in the pass (stale memory, from HBM).
A thread with no position of the unit now skips. Same-run: 8 rows at 8k context 8.32 -> 8.23 ms, 32 rows at
16k 26.45 -> 25.95, 256 columns + 16 rows 16.98 -> 16.85. Long-context and online tests pass.
Tried before it: the scores batch left in flight during the same warpgroup's scalar work: no change, not kept.

## s0022: fold kernel skips threads without a position
Expected: 0.931 (176), TTFT p50 0.23-0.25 s / p90 4.05-4.1 s, tbt p50 22 / p99 35-37 ms; offline unchanged.
Measured (s0022): VALID, 0.931 (176/189), TTFT p50 0.239 s / p90 4.05 s, tbt p50 22 / p90 27 / p99 35 ms, longest gap
46 ms; step 5.67 / 5.81 ms. As predicted.
Simulator with the current cost model: pass targets 27 / 29 / 31 ms and boosts 45 / 55 / 60 ms all give 177 of
189: the twelve misses left are prompts of 20k-33k tokens in bursts of 165k-245k prompt tokens that would need
15-25% more capacity in the 5 s after the burst; no setting of the policy reaches them.

## Soak and sweeps on s0022's engine (not submissions)
- two replays with 12 probes each: 176/189 both, probes identical 12/12 both;
- the same with freed KV pages poisoned with NaN (MK_POISON=1): all requests finish, probes identical 12/12 at
  nominal 2 req/s and at 4 req/s (368 requests, 0.856 within the service levels, tbt p99 43 ms);
- the evaluator at nominal 3 req/s (284 requests): valid, 0.912, TTFT p50 0.30 s, tbt p99 42 ms.
- prefix reuse is not limited by the page budget: the simulator gives the same 1032 hit pages with 600 and
  with 875 pages (evictions 1687 / 1412, recomputed 3 / 1).
- k_prep with V written as 16-byte stores per group of 8 positions: same bits, 0.5% of a prefill pass: not kept.
- pass target 28 ms: 176 three times (tbt p50 23); 29 ms: 177 with tbt p50 25 (no margin); 30 ms: 177 with 26.
  Stays at 27. Large-pass setting with the current kernels (MK_TARGET_MS=80 MK_BOOST_S=0): 181 of 189 = 0.958,
  TTFT p50 0.21 s / p90 3.34 s, tbt p50 39 / p99 74 ms; 40 ms: 178, tbt p50 32. Not submitted: the board asks
  for a median under 25 ms, and that end of the trade is already on it (s0011).
- simulator: streams ride every second pass while a prompt is short of slack: 177 either way, tbt p99 38 -> 72 ms.

## State at 16.2 h (s0022 is the engine in the workspace)
                                   s0010 (start of this stretch)   s0022
  within the service levels        0.952 (180/189)                 0.931 (176/189)     engines: 0.64 / 0.67
  TTFT p50 / p90                   1.10 s / 3.76 s                 0.24 s / 4.05 s     engines: 1.6-1.8 / 4.9-5.2 s
  time between tokens p50 / p99    35 / 78 ms                      22 / 35 ms          board asks p50 <= 25 ms
  longest gap of a stream          101 ms                          46 ms
  offline step B=1 / B=8           6.84 / 6.97 ms                  5.67 / 5.81 ms      board's lines: 5.62 / 5.97
What moved it: GEMM block shapes per matrix and pass size (short passes became affordable), the 27 ms pass
target with a 45 ms boost near a deadline, shortest-first inside a burst, decode-row attention with swapped
operand roles on three warpgroups, staggered warpgroups in prefill attention, q/K/V prepared in the GEMM
epilogue, 1/rms inside the GEMM, four producer threads, step()/generate() as graph launches.
What is open: the offline step at batch 1 is 0.9% above the board's line (GEMM kernels 5.43 of 5.67 ms; the
weights at 3.07 TB/s are 4.93); twelve prompts of 20k-33k tokens in the largest bursts miss 5 s at any policy
I could simulate.
What bounds the engine now (measured, see above): HBM at ~3 TB/s for the weights of every pass (15 GB) and for
the K/V of every decode row (147 KB per token of context per token: 76 TB in the replay); the register file
of an SM (two warpgroups x 128 accumulators: the GEMM block's area, and two warpgroups of attention state);
the copy engine for the activations of wider blocks.

## Two operator notes on the board that I had not read (posted 2026-11-10 17:10 and 19:00; read at 16.5 h)
I had been printing the tail of `mk board` only. Both notes are requirements, and s0012-s0022 were made without them.
1. Tracks. Track A: the number formats and data flow of the general engines (BF16 weights as stored, BF16
   activations and KV cache where they use BF16, same operators in the same order, nothing precomputed);
   layout and kernel organisation free. Track B: anything lossless in addition. Every hypothesis must start with
   the track and say which formats differ from the checkpoint's; the best track-A engine must stay buildable.
   Where this engine stands (every submission of this run, s0002-s0022, same arithmetic): weights BF16 as stored
   (permuted layout), KV cache BF16 (K after norm and RoPE, V), inputs of every matvec BF16. It differs from
   the engines' data flow in: (a) the residual stream and all matvec outputs stay in f32 between operators
   (they round to BF16 after every operator); (b) RMSNorm: the weight is multiplied in before the matvec and
   the 1/rms after it (a scalar per column commuted through the linear map); (c) q is stored multiplied by
   log2(e)/sqrt(128) and the softmax is taken in base 2 against a running maximum, in segments of 128 keys,
   with the weights rounded to BF16 before they multiply V. So it is not track A; I label it track B and list
   (a)-(c). A track-A build switch is owed.
2. The first-token tail. s0012 over HTTP, 300 s: TTFT p99 15.0 s at 1.61 req/s (engines 5.9 s), 84.9 s at 2.39
   req/s (9.5 / 10.1 s): "new prompts wait while streams that already decode are served"; required: TTFT p99 no
   worse than the better engine (about 6 s and 10 s) without giving back the stall-free decoding.
   My reading: two causes. The deadline planner gives up on the largest prompts of a burst and serves them
   last, and under sustained load nothing ever brings them back (starvation: the 85 s). And the 27 ms pass
   target halves the prompt throughput exactly when a burst has to be cleared (s0022: TTFT max 13.5 s; the
   80 ms setting: 9.5 s). The share within 5 s rewards the first and hides both.

## The first-token tail: a third pass target while somebody is heading for a late first token
Simulator (TTFT p99 / max, share, tbt p50 / p99; nominal 2, 3 req/s):
  s0022's policy:             9.8 / 13.1 s, 177, 24 / 38 ms  |  18.9 / 32.0 s, 260 of 284, 29 / 47 ms
  80 ms passes always:        7.6 /  9.3 s, 181, 39 / 78 ms  |  10.1 / 10.6 s, 273,        67 / 84 ms
  FIFO at 80 ms:              7.6 /  9.2 s, 162, 28 / 77 ms  |   8.5 /  9.9 s, 251,        48 / 84 ms
The tail is the time it takes to clear a burst, and that is a matter of prompt throughput: 27 ms passes halve
it. Tail guard: every step the planner walks its service order and adds up the estimates; if any prompt is
heading for a first token later than tail_guard_s after its arrival, the pass aims at target_tail_s instead.
  guard 8 s, 80 ms (sim):     7.9 /  9.5 s, 179, 26 / 76 ms  |  10.1 / 10.8 s, 268,        50 / 82 ms
(guard 5-6 s: p99 7.7 with tbt p50 28-29; guard 10-12 s: p99 9.6; 60 ms: p99 8.1 / 10.6; an age cap that puts
prompts older than 8 s first: the share at 4 req/s falls from 0.90 to 0.79 for p99 37 -> 26 s: not taken.)
Device scheduler and Python reference agree pass by pass. Replays (TTFT p90 / p99 / max, share, tbt p50 / p99,
longest gap):
  nominal 2:  s0022       4.06 / 13.1 / 13.6 s, 176 of 189, 22 / 35 ms, 46 ms
              8 s, 80 ms  3.81 /  7.9 /  9.7 s, 179,        23 / 73 ms, 88 ms
              8 s, 70 ms  3.86 /  8.0 /  9.8 s, 179,        24 / 65 ms, 80 ms
  nominal 3:  s0022       4.52 / 26.6 / 32.9 s, 258 of 284, 27 / 43 ms, 48 ms
              8 s, 80 ms  3.46 / 10.2 / 10.7 s, 267,        45 / 76 ms, 87 ms
              8 s, 70 ms  3.60 / 10.5 / 11.1 s, 266,        44 / 66 ms, 75 ms
Decision: guard 8 s, 80 ms. The board's HTTP measurement of the engines: p99 5.9 s at 1.61 req/s, 9.5-10.1 s at
2.39 (with time between tokens p99 0.3-1.7 s). This gets 7.9 s and 10.2 s in my replay with every gap under
90 ms. Below ~8 s at the standard rate I cannot go: the largest burst (245k prompt tokens at once) takes 9.7 s
to clear at full prompt throughput with the streams kept alive.

## s0023: tail guard (track B)
Expected: 0.947 (179), TTFT p50 0.21-0.24 s / p90 3.8 s / p99 7.9 s, tbt p50 23-24 / p99 73-74 ms, longest gap 88-90 ms.
Measured (s0023): VALID, 0.947 (179/189), TTFT p50 0.229 s / p90 3.75 s / p99 7.92 s / max 9.5 s (s0022: 0.239 / 4.05 /
13.1 / 13.5), tbt p50 23 / p90 65 / p99 74 ms, longest gap 91 ms; offline 5.67 / 5.84 ms. As predicted.

## Track A: a build switch (`engine/TRACK`, `-DTRACK_A=1`)
What the general engines do that this engine did not (the three differences listed above), now switchable:
- every operator's output rounded to BF16 (`ta_r`): residual stream after each add, q/k/v, gate, up, SiLU, their
  product, logits;
- RMSNorm as x * rsqrt(mean x^2 + eps) in f32 -> BF16 -> times weight -> BF16, before the matvec: `k_rnorm` writes
  that vector (on track B the residual-add epilogue writes bf16(w x) and the 1/rms goes behind the matvec);
- RoPE on BF16 values with BF16 cos / sin, products and sum rounded; q stored unscaled, scores scaled in f32.
Attention is unchanged and I count it as aligned: vLLM and SGLang call flash-attention style kernels, which is
this data flow (f32 scores, softmax against a running maximum over blocks of keys, weights cast to BF16 for the
product with V, f32 output rounded once). Nothing is precomputed that they compute per step.
Accuracy against HF float32 (tests/test_eng.py, relative L2 of logits, median / max; top-1 agreement):
  track B   0.0044 / 0.027 (B=1), 0.0047 / 0.116 (B=8); 1.000 / 0.998
  track A   0.0115 / 0.037,       0.0114 / 0.154;       0.983 / 0.983
  HF BF16   0.0106 / 0.052,       0.0109 / 0.332;       0.981
Track A has the engines' own error, as it should. All invariances hold bitwise on track A too (batch, chunking,
prefix reuse, kernel variants, generate vs steps, 34000 positions, online under the four configurations).
Cost, and what it took:
1. First build: step 6.59 / 7.61 ms (track B 5.67 / 5.81). The assembler serialized every wgmma of the
   residual-add GEMM (remark C7514): with the rounding in the epilogue the compiler moved integer work on the
   accumulators in front of the wgmma wait. An empty asm statement that names the accumulators after the wait
   (`WG_TOUCH_N*`) pins the order: 6.29 / 7.32.
2. `k_rnorm` with one warp per column: 16 dependent round trips to memory per column. 8 warps per column:
   5.96 / 6.12.
3. Prefill passes were 11-15% slower than track B's (512 columns: 16.8 vs 14.6 ms). Split by builds without the
   roundings and without the norm kernel: the kernel was 22 us at 512 columns, 103 us at 2040 (0.5 TB/s: three
   round trips in a row per item), the roundings 0.65 ms per 512-column pass. All loads of an item requested
   before anything is computed, no wasted fragment stores in the add epilogue, no 1/rms loads: 15.7 ms at 512
   (+7.7%), 28.1 at 1024 (+7.2%), 55.8 at 2040 (+6.5%); decode steps 5.90 / 6.07 ms (track B 5.68 / 5.83, same
   run). More blocks per SM for the norm kernel: no difference.
What remains is the engines' own work: 73 more kernels per pass (about 4 us each at small sizes) and 5 integer
operations per rounded value.
Replay, same run (share, TTFT p50 / p90 / p99 / max, tbt p50 / p90 / p99, longest gap):
  track A   178 of 189 = 0.942, 0.23 / 3.99 / 8.3 / 10.7 s, 25 / 70 / 75 ms, 89 ms
  track B   179 of 189 = 0.947, 0.23 / 3.78 / 7.9 /  9.6 s, 24 / 65 / 73 ms, 87 ms
The cost model is track B's; on track A the correction factor settles at 1.08-1.10 and absorbs it.

## s0024: the track-A build (engine/TRACK = A)
Expected: 0.942 (178), TTFT p50 0.23 / p90 4.0 / p99 8.3 s, tbt p50 25 / p99 75 ms, longest gap 89 ms, step 5.90 / 6.07 ms.
Measured (s0024): VALID, 0.947 (179/189), TTFT p50 0.221 / p90 4.02 / p99 8.34 / max 10.7 s, tbt p50 24.8 / p90 70 /
p99 76 ms, longest gap 89 ms; offline step 5.90 / 6.07 ms, generate 5.91 / 6.07; logits vs float64: rel L2 median
0.0098 (single) / 0.0101 (batched), long context 0.0155, top-1 agreement 0.997 (HF BF16 0.999).
As predicted (one request more within the levels than in my replay). The two tracks side by side:
  track B (s0023): 0.947, TTFT 0.229 / 3.75 / 7.92 s, tbt 23 / 65 / 74 ms, step 5.67 / 5.84 ms
  track A (s0024): 0.947, TTFT 0.221 / 4.02 / 8.34 s, tbt 25 / 70 / 76 ms, step 5.90 / 6.07 ms
The alignment costs 4% of decode speed and 7% of prefill speed, and no request. The workspace default goes back
to B (`engine/TRACK`); from here on every kernel change is measured on both.
Next: the prefill attention. At 2040 columns and context 28672 a pass is 124 ms, 71 ms of it attention, and by
my instruction count the tensor cores are busy about half of that time. Ablation first (ATT_ABL switches).

## Prefill attention: what bounds it, and a three-warpgroup kernel that did not help
Ablation of k_attn<2> (timing-only switches ATT_ABL: 1 no K/V copies after the first, 2 no scalar work, 4 no
tensor instructions), pass of 2040 columns at context 28672 (attention 65 ms of 124): tensor instructions alone
38 ms (531 TFLOPS of useful work over the pass, the SM's limit at two warpgroups), scalar work alone ~17,
copies ~10, and they nearly add up (all three: 65). At 512 columns / 16k: 28.8 ms with attention, 16.2 without.
- k_attn3: groups of three units, three warpgroups in rotation so that two of them are on the tensor cores
  while the third does its scalar part. Fitting it into 168 registers took: the task's scalars in shared
  memory, an empty asm naming the accumulators after each wait (otherwise C7518, wgmma serialized because a
  branch of one warpgroup reads them), the weights carried in the score registers or not at all (C7511). Same
  bits. Measured: no faster at any size (2040 / 28672: 124.3 vs 124.6 ms; 512 / 8192: 22.0 vs 20.3).
- Why (tests/wg_mix.cu): a warpgroup doing the scalar part slows the tensor warpgroups of its SM. One tensor
  warpgroup and one scalar warpgroup calibrated to 6.3 ms each take 8.3 ms together, two and one 11.2 -> 15.2.
  The softmax's issue slots and register reads compete with wgmma's; overlap gains ~2/3, not all.
- tests/sc_ops.cu (per 64 scores per thread): max + ex2 + cvt 0.81 us per warpgroup alone; ex2 is half of it;
  a polynomial 2^x on the FMA units 2.2x slower; cvt.rn.bf16x2 pairs are what ptxas already emits.
- Page ids of a task in shared memory instead of a lookup before each copy: no change.
Dead end for now; the attention stays at k_attn<2>.

## Offline decode step under the board's line (B=1: 5.67 -> 5.58 ms; the line is 5.62)
nsys of a B=1 step (tests/nsys_step.py on the trace's sqlite: per-position kernel times of a repeated step):
q/k/v with preparation 20.5 us (48 blocks, 2.45 TB/s), decode attention 6.9 us, o 14.4 (2.33 TB/s), gate/up
69.2 (2.91), down 35.7 (2.82) per layer; LM head 404 us (3.08 TB/s); embedding 6.9 us; gaps ~0.
tests/stream_rate.cu: bare streams of the same sizes with cp.async.bulk take as long (32 MB on 64 blocks
14.5 us, on 128 blocks 14.4; 96 MB 36 us): the GEMMs are at the streaming rate of a kernel of their size, and
more blocks or stages (o: 12 / 20; q/k/v: 8 / 10 / 12 stages) give ~nothing (q/k/v 10 stages: -10..25 us, kept).
What a kernel boundary costs is its ramp: tests/stream_pdl.cu, the 144 streams of a step back to back in one
graph: 4730 us; with programmatic edges 4692; with the next kernel launched early and asking for its first
stages before it waits 4583.
Kept, each measured in one run against the build before (tests/ab_defs.sh, tests/ab_head.sh):
- PDL (programmatic edges, griddepcontrol.wait at the top of every kernel): 5.68 / 5.84 -> 5.65 / 5.80.
- the GEMM producers trigger the successor's launch after their last copy request, and k_gemm1 requests 3
  weight stages before its wait: -> 5.63 / 5.78 (8 stages 5.67, 12 5.70: the first stages' inputs queue
  behind them; every kernel triggering at its start: +0.2 ms at B=8; L2 prefetch of 8 next stages: -5 us).
- decode attention with one block per task needs no counter (step, generate: the grid is the number of tasks):
  two round trips per layer less, and the first task's plan is read under the set-up: -> 5.58 / 5.72.
- embedding: all loads before the stores; k_gather computes the final 1/rms (no k_rnorm before the LM head);
  the step's position travels in the embedding node's arguments (no k_setcell launch): ~-5 us together.
Not kept: L2 prefetch of the o weights by idle attention blocks (+30..50 us), page ids in shared memory, a
counter-free first task with the trailing counter round trip still there (noise).
A bug on the way, caught by tests/run_all.sh (online, device loop: 10/12 identical) and the replay's probes:
the counter-free path ran tasks past the decode rows when the grid (the device loop's: all SMs) is larger.

## s0025: decode step (track B)
Expected: step 5.58 / 5.72 ms, online unchanged (0.942-0.947, TTFT p99 7.9-8.2 s, tbt p50 23-25 ms).
Measured (s0025): VALID, 0.947 (179/189); offline step 5.583 / 5.721 ms, generate 5.597 / 5.726 (s0023: 5.67 /
5.84). Both under the board's lines now (5.62 / 5.97). TTFT p50 0.230 / p90 3.80 / p99 7.91 / max 9.6 s, tbt
p50 24.1 / p90 66 / p99 74 ms, longest gap 84 ms. As predicted.
Open from the board: TTFT p99 at 1.61 req/s (engines ~6 s; here 7.9 s).

## Prefill GEMM: weights multicast in clusters of two (dead end), and what bounds the large shapes
Idea: the 128 x 2 shape reads the weights once per 256 columns; two blocks in a cluster sharing the row block
and each copying half of a stage into both (cp.async.bulk .multicast::cluster) would read them once per 512.
Built (gemm_body<..., MC>: cluster rank picks the column block, `empty` takes the arrivals of both blocks'
consumers, a handshake keeps a block until the other is done with it): same bits, 1.8x slower (gate+up at 1024
columns 282 -> 504 us). tests/mc_rate.cu explains it: 132 blocks pulling 48 KB stages (16 KB weights + 32 KB
activations) take 0.87 us per stage with no work at all, i.e. ~56 KB/us per SM, 7.3 TB/s of shared-memory fill
in total; weights multicast 1.06 us, activations multicast 0.99. The large shapes are bound by that fill rate,
not by HBM, and multicast lowers it. Reverted (mk note).
tests/cluster_occ.cu: clusters of 2 keep all 132 SMs (clusters of 4: 120).

## Scheduling knobs against the board's time-between-tokens line
Replays (share, TTFT p99, tbt p50): tail guard 8 s: 179, 7.9 s, 24 ms (s0025's setting); 6 s: 179, 7.9, 25;
5.5 s: 179, 7.8, 25; 5 s: 181 (0.958), 7.8, 26-27; 5 s with 70 ms tail passes: 181, 7.9, 27; with 60 ms: 180,
8.1, 26; normal target 25 or 26 ms with guard 5: 181, 7.8, 27-29 (slower prefill puts more time in the tail
mode). Every setting that keeps two more first tokens under 5 s costs the median stream 2-3 ms per token, over
the board's 25 ms. Not taken. Decode rows sharing KV pages within a pass (cascade attention): 3.6% of their page
reads (sim over the replay) - not worth a kernel. Decode attention itself is at the HBM roofline (32 rows at
30k context: 47 ms for 141 GB).
Track A without k_rnorm before the matvecs (the producer warps writing the normalized input into the stages):
same bits, slower in three variants (6.12-6.45 ms at B=1 against 5.84 with k_rnorm): a warp fills one stage
per round trip to L2. Not kept (mk note).
Longer replays (240 s, 368 / 585 requests; tests/data/trace_head.jsonl now holds 800 rows of the trace):
standard rate: 0.962, TTFT p99 7.7 s, max 9.6, tbt p50 22 ms; 1.5x the rate: 0.930, TTFT p99 13.2 s, max 22.5,
tbt p50 58 ms (the GPU is busy 99% of the time; the simulator agrees: p99 17 s). At that rate the engine is
past its capacity with every stream kept under 85 ms.

## Light decode passes in the serving loop (SWITCH node)
tests/online_dec.py (B requests decoding together in the device loop, context 300-700) against offline steps:
B=1 6.54 ms per pass vs 5.58, B=8 6.73 vs 5.72. nsys of the device loop: the general pass carries kernels with
nothing to do in a decode pass (k_attn<2>, k_attn<1>, k_prep, two 1/rms kernels per layer: 180+ launches) and
the merged GEMM kernels (+4..5 us each against the per-shape ones). Now the WHILE body is k_sched -> SWITCH ->
k_commit: k_sched sets the branch from the plan header (all attention tasks units of one position, at most 16 /
32 / 64 columns -> light pass 1 / 2 / 3, else the general pass). A light pass: per-shape GEMM kernels (columns
read from the plan), q/k/v prepared in the GEMM up to 16 columns, decode attention and fold only. A conditional
node costs ~6 us (tests/graph_while.cu), once per pass. Same bits (online tests under four configurations,
probes); decode passes: B=1 6.56 -> 5.75 ms, B=8 6.75 -> 5.94, B=16 7.15 -> 6.34, B=24 7.48 -> 6.97,
B=48 8.85 -> 8.27. Replay: decode-only passes 11.4 -> 10.3 ms measured; 179 of 189, TTFT p99 7.9 s, tbt p50
23 ms (24 before). With tail guard 5 s: 181 (0.958) but tbt p50 26: still over the line.

## s0026: light decode passes (track B)
Expected: 0.947 (179), TTFT p50 0.22-0.24 / p90 3.8 / p99 7.9 s, tbt p50 23 ms, offline 5.58 / 5.72 ms.
Measured (s0026): VALID, 0.947 (179/189), TTFT p50 0.221 / p90 3.76 / p99 7.94 / max 9.45 s, tbt p50 23.0 / p90
66 / p99 74 ms, longest gap 86 ms; offline step 5.568 / 5.706 ms. As predicted.
After s0026 (pod epoch 7):
- Prefill attention with warpgroup 0's scores of the next segment left in flight during its scalar work
  (ATT_ASYNC=1): same bits, no change at any size (2040 / 28672: 125.3 vs 126.3 ms). Switch kept, off.
- Tail guard with the light passes (track B): 8 s 179 / tbt 23, 6 s 179 / 24, 5.5 s 179 / 25, 5.25 s 179 / 25,
  5 s 181 / 26. A second, milder level (55 ms passes from 5 s, 80 ms from 8 s; simulator): 180, tbt +3 ms.
  Guard stays at 8 s.
- Track A on this pod: offline 5.98 / 6.10 ms (the s0025 code: 6.01 / 6.12, so the pod, not the code; track B
  5.57 / 5.71 here as before). Replay: 175-176 of 189, tbt p50 31, TTFT p99 9.3 s; large passes run 1.18x the
  cost model (track B 1.07). Light passes help track A too (decode pass B=8 6.91 -> 6.38 ms; replay 175 -> 176).
  The last track-A submission (s0024: 0.947) stays its record; not resubmitted.
- A trap of mine: replays after switching engine/TRACK ran the old library (replay.py does not run build.sh);
  three guard replays of track A taken for track B, caught by their numbers and rerun.

## Early weight copies that are really early
The "early" weight stages of s0025 went out only after the wait for the kernel before: the producers met the
consumers at the block barrier that follows the barrier set-up, and the consumers waited before it. Now the
barriers are set up first, the consumers wait after the block barrier (and meet at a named barrier of their own
for the 1/rms), the producers ask for the first stages and then wait. The merged GEMM kernel of the serving loop
does the same (it reads the number of columns from the plan before the wait: the plan is older than the kernel
before; a block that ends without work still waits, else its successor could run alongside its predecessor).
Same run (tests/ab_defs.sh): early stages 0 / 3 / 6 / 12: step B=1 5.533 / 5.456 / 5.453 / 5.511 ms, B=8
5.667 / 5.592 / 5.596 / 5.648 (s0026's code: 5.568 / 5.706). Device loop decode passes B=1 5.75 -> 5.64, B=8
5.95 -> 5.84 ms. Replay: 179, TTFT p99 8.1 s, tbt p50 24 ms (same as s0026 within noise). All tests pass.
s0027 (track B): expected offline 5.46 / 5.59 ms, online as s0026. Measured: VALID, 0.947 (179/189), offline step
5.453 / 5.591 ms, generate 5.469 / 5.596; TTFT p50 0.231 / p90 3.74 / p99 7.92 / max 9.52 s, tbt p50 23.0 / p99
73 ms, longest gap 85 ms. As predicted.
Then: decode attention sets up (barriers, the ones, its first task from the plan) before its wait: step B=1 5.456
-> 5.431 ms; other kernels than GEMMs let their successor launch at their start (PDL_TRIG 4): -> 5.411 / 5.548.
Device loop decode passes (tests/online_dec.py, twice, same run): trigger 3 vs 4: B=1 5.62 vs 5.80 ms (worse:
there the attention runs on all SMs, its idle blocks hold them while they wait), B=8 5.82 vs 5.80, B=32 7.16 vs
7.11. Replay with 4: 179, decode-only passes 10.17 ms. Kept 4 (the replay's decode passes have 8-40 rows).
s0028 (track B): expected offline 5.41 / 5.55 ms, online as s0027. Measured: VALID, 0.947 (179/189), offline step
5.412 / 5.548 ms (generate 5.428 / 5.553): milestone h0_step passed (>= 1.10x the general engines at B=1 and
B=8); TTFT p50 0.221 / p90 3.76 / p99 7.94 / max 9.53 s, tbt p50 23.1 / p99 74 ms, longest gap 84 ms.
Then: a block without work leaves at once instead of waiting first (block 0 of a kernel with work always has
some and waits, so the kernel still cannot end before its predecessor): device-loop decode B=1 5.80 -> 5.61 ms
(the regression of trigger mode 4 gone), B=8 / B=32 and offline unchanged. All tests pass, replay 179.

## Track A after the launch work; a near-miss trigger
Track A (this pod): offline 5.67 / 5.84 ms (s0024: 5.90 / 6.07; the line 5.62 / 5.97: B=8 in, B=1 0.9% over).
Online it is short of capacity: replay 176 (0.931), tbt p50 30 ms with guard 8; guard 12: 174 / 31; no guard:
169 / 26; normal target 24 or 22 ms: 173 / 33-37 (slower prompt passes leave more time in the tail mode).
Track A cannot meet the tbt line with this engine; the record stays s0024.
Near-miss trigger (MK_NEAR_S, both schedulers, parity checked in the simulator): passes also aim at the tail
target while some prompt is estimated to get its first token within near_s after 5 s, i.e. while big passes
can still rescue somebody (the tail guard fires on hopeless prompts too). Replays (track B): near 0.5 s: 179 /
tbt 25; 1.0: 180 (0.952) / 26; 1.5: 180 / 26. Like guard 5 (181 / 26): one or two requests for 2-3 ms of tbt
p50, over the line. Off by default.

## State at 26.7 h (the engine in the workspace = s0028 + idle blocks leave early + MK_NEAR_S, off)
Against the board's requirements (track B, the evaluator's numbers for s0028):
- share within service levels 0.947 (179/189); TTFT p50 0.22 s / p90 3.76 s (engines 1.6 / 4.9 s): met.
- time between tokens p50 23.1 ms (<= 25): met; p99 74 ms, longest gap 84 ms (engines 270-2600 ms / 0.45-7 s).
- offline step 5.41 / 5.55 ms (line 5.62 / 5.97): met; h0_step milestone (>= 1.10x the engines) passed.
- TTFT p99 7.9 s at 1.57 req/s (engines ~6 s): not met. 1.5x the rate, 240 s: p99 13.2 s (engines ~10 s):
  not met. What I know about it: the largest burst (245k prompt tokens at once) takes ~9.5 s to clear with
  every stream kept under ~85 ms; p99 is the third-largest first-token time, i.e. the end of that burst minus
  the two longest prompts. Every knob that moves it (guard 5, near-miss, longer tail passes) costs the median
  stream 2-3 ms per token, past the 25 ms line, and at 1.5x the rate the GPU is busy 99% of the time.
  Prefill throughput is what would move it; the GEMM at large shapes is bound by the SMs' shared-memory fill
  (multicast did not help), attention by tensor + scalar issue (three warpgroups, async scores: no help), decode
  attention by HBM. The engine is near what this design gives.
Track A (engine/TRACK = A): record s0024 (0.947, 5.90 / 6.07 ms); now 5.67 / 5.84 ms offline, online short of
the tbt line on this pod (journal above).

## Super-segments of 8192 positions instead of 4096 (SS_LOG 6): no
nsys of a 2040-column pass at context 16k: GEMMs 52%, attention 40%, k_prep 2.8%, k_attn_fold 2.5% (the parts of
5 super-segments per unit, f32, ~174 MB per layer). SS_LOG 6 halves the parts. Same accuracy, all invariances
hold. Prompt passes 2040 / 16k 95.1 -> 92.4 ms, 2040 / 28k 125.3 -> 122.9; tests/calib_cost.py's decode passes
looked much faster (32 rows at 16k: 26.1 -> 18.8 ms) - but its rows share one sequence's pages (L2 hits), so its
decode numbers measure task overheads, not HBM. The replay says otherwise: decode-only passes 9.99 -> 10.29 ms
(half the tasks for rows with their own pages: less parallel reading), share and TTFT equal. Kept 5.
(Decode-only passes of the replay are now 9.99 ms, from 11.4 before the light passes.)
Tensor-map copies (cp.async.bulk.tensor, tests/tma_mc.cu) behave like the plain bulk copies: 0.88 us per 48 KB
stage alone, multicast of the weights 1.16 (no work) / 1.41 us (0.58 us of work per stage); multicast of both
operands 0.88 / 1.35. Multicast is no way around the fill rate here. Closed.

## Prefill attention, ping-pong (FlashAttention-3's schedule): same speed
k_attn_pp (ATT_PP=1, kept, off): per warpgroup and segment G(s) = values of s and scores of s + 1 issued
together, then S(s + 1) = the scalar work; the two warpgroups take turns to issue G (named barriers), no block
barrier per segment (a buffer is refilled by whichever warpgroup releases it second, a counter in shared
memory), the scores past the last segment computed and dropped (no condition around an MMA). 209 registers, no
spills or serialization, same bits (unit, long context). Time: 2040 columns at 28k 126.2 vs 125.2 ms, at 16k
94.5 vs 94.9, 512 at 16k 28.4 vs 29.8. (A first version with a producer warp: 288 threads, ptxas capped it at
168 registers and spilled.)
Reading the old ablation again: tensor alone 38 ms, copies alone 14 ms, scalar alone 3.7 ms, all 66 ms (2040 /
28k). The copies are not small: every pair task streams its kv head's K/V through shared memory (64 KB per
segment), ~1 us per segment per SM at the fill rate measured for the GEMM; with the tensor time per segment of
the same order, how the warpgroups are scheduled does not matter much. More query rows per K/V load would, but
two units per warpgroup need ~264 registers, and three warpgroups (k_attn3) did not help either.

## Per-shape GEMM kernels for prompt passes too (bucket branches in the serving loop)
Host passes (tests/calib_cost.py, same run): the merged GEMM kernel against per-shape ones for the large shapes:
192 columns 9.42 -> 9.07 ms, 256 9.68 -> 9.27, 384 13.60 -> 12.73, 512 15.02 -> 14.11, 768 23.69 -> 22.49, 1024
26.83 -> 25.93, 2040 53.45 -> 52.37 (context 0); at 16k 512: 25.59 -> 24.90. Same bits. Host passes now use them
(MK_GEMM_SPLIT=2, default). Serving loop: gemm_shape gives 18 runs of numbers of columns over 1..2048 with one set
of four shapes each; the SWITCH gets one branch per run (the general pass with per-shape kernels, columns from
the plan, grid for the run's largest), k_sched picks it from the plan header. Graph build at the first request
0.02 s. All tests pass. Replays: 179, 179, 178 (same as before within noise); prompt passes 129-256 columns
24.9 -> 24.5 ms, 385-512 23.4 -> 23.2: ~1% (in the serving loop the passes carry decode rows and several runs).
s0029 (track B): expected 0.942-0.947, TTFT p99 7.9-8.2 s, tbt p50 23-24 ms, offline 5.41 / 5.55. Measured: VALID,
0.947 (179/189), TTFT p50 0.226 / p90 3.77 / p99 7.90 / max 9.49 s, tbt p50 23.2 / p99 74 ms, longest gap 88 ms;
offline step 5.414 / 5.552 ms. As predicted.

## The evaluator at three rates (local, s0029's engine), and where the first-token tail comes from
tools/online_eval.py --rates "2 4 1": r2 (1.57 req/s) 0.947, TTFT p50 0.23 / p90 3.77 / p99 7.93 / max 9.54 s,
tbt p50 23.2 / p99 74 ms, longest gap 86 ms; r4 (3.07 req/s) 0.910 (engines 0.42 / 0.43), TTFT p99 29 s, max
42 s, tbt p50 65 ms, gap 85 ms; r1 (0.86 req/s) 0.942 (engines 0.73 / 0.79), TTFT p99 7.93 s, max 9.26 s, tbt
p50 22.2 ms. All valid, every gate passed.
The tail does not depend on the rate: the replay's arrival times are those of the trace scaled, and the trace's
requests come in groups with the same timestamp. At r1 (replay --rate 1 --out): the 22 requests arriving at
t = 20.01 s carry ~250k prompt tokens (three of 26k-33k); the last of them gets its first token after 9.3 s,
the second-to-last after 7.6 s; t = 100.05: 7.9 s. Clearing such a group is ~27k prompt tokens per second
(prompt passes at long context run at 16k tokens/s, at short at 39k) with every stream kept under ~85 ms.

## Soak and last sweeps (s0029's engine)
Three poisoned replays with 8 probes: 179, 179, 179 (TTFT p99 8.2 s, tbt p50 23.7-23.8 ms), probes identical,
pages conserved; generate() as plain launches PASS; serving with host-launched steps (40 s) 66/70, probes
identical. Simulator sweeps, all 179 of 189: deadline margin 0.1 / 0.3 / 0.5 s, boost slack 0.5 / 1 / 1.5 / 2 s,
boost target 40 / 45 / 55 ms; pass cap 4096 instead of 2048 (with larger classes): no change at 80 ms tail
passes, and tail passes of 120-240 ms cost 13-17 requests their time between tokens (0.86-0.88). Big prompt
passes in the simulator: 74.5 ms, of which the decode rows' K/V (10.6 rows, 117k positions) ~5.8 ms; running
that attention on a separate set of SMs would take those SMs from the prompt attention, which is bound per SM:
no gain by arithmetic, not built. The ten misses are capacity: the engine is at what this design gives.

## GEMM epilogues: what a bare 128 x 256 kernel reaches, and the epilogue's loads
tests/gemm3.cu, the engine's tile and layout without any epilogue: gate/up at 2048 columns 450 us = 917 TFLOPS
(the engine: ~600 us). With a gate/up epilogue in fast math (silu, bf16 stores): 490 us; staged in shared memory
with one bulk store: 478 us. A 192 x 256 tile on three warpgroups: 901 TFLOPS, no better. So the engine's large
GEMMs lose ~25% in their epilogues. Timing-only ablation (EPI_ABL) in the engine: without the 1/rms loads and
the residual loads (one round trip per column group, 16 in a row per unit) a 2040-column pass at context 0 takes
43.3 instead of 51.1 ms, at 16k 81.7 instead of 94.3.
- The 1/rms of a unit's columns into shared memory at the unit's start (EPI_RNU, same bits): 2040 / 0 50.5 ->
  49.7 ms, 2040 / 28k 125.3 -> 123.7, 512 / 0 14.0 -> 13.8. Kept.
- The residual add's accumulators started at the residual (EPI_XINIT, track B; other bits, same accuracy, every
  invariance holds): > 512 columns 2-3.5% faster, decode steps -6 us; but ptxas serializes the 32 x 1 and 64 x 1
  shapes with it (C7515, whatever pins the accumulators), +5% for passes of 65-256 columns and +4.6% for decode
  passes of 33-64 rows; other shapes for those sizes are slower still. Net ~-0.8% of the replay's GPU time and a
  loss at higher concurrency: off (switch kept).
Replay with EPI_RNU: 179, tbt p50 23.6 ms, prompt passes 129-256 24.5 ms, 385-512 23.1, > 512 60.6 (ratio to
the cost model 1.06-1.10); offline 5.41 / 5.55 ms. All tests pass.
s0030 (track B): expected as s0029 within noise. Measured: VALID, 0.947 (179/189), TTFT p50 0.218 / p90 3.71 / p99
7.91 / max 9.39 s, tbt p50 23.0 / p99 74 ms, longest gap 87 ms; offline 5.414 / 5.550 ms.
Correction to the epilogue entry above: tests/gemm3.cu filled its operands with zeros, and the tensor cores run
faster on zeros (power). With random values its bare main loop takes 501 us (822 TFLOPS), with a gate/up
epilogue 529-538 us; the engine's per-shape kernel without stores 538 us. The engine is at this tile's rate;
there is no 25% to win in the epilogues, only the few percent found (1/rms staging). Of the ablation's gains a
part was the residual's L2 traffic, which neither an L2 prefetch nor read-only loads changed.

## The near-miss trigger on (MK_NEAR_S 0.75)
With the faster engine (light passes, per-shape GEMMs, staged 1/rms) the near-miss trigger's price in time
between tokens dropped: replays, near 0.75 s, normal target 27 ms: 180, 180, 180, 180 of 189 (0.952) with tbt
p50 23.9, 24.2, 24.1, 24.2 ms (target 25 ms: 180 / 24.0-24.3); off: 179 / 23.0-23.8. TTFT p99 7.9 s either way.
Default now 0.75 s.
s0031 (track B): expected 0.952 (180), tbt p50 23.5-24.5 ms, TTFT p99 7.9 s. Measured: VALID, 0.952 (180/189) - the
board's best share, now with every stream under 90 ms (s0010/s0011 had 0.952 with 80 ms passes always, tbt p50
34 ms) - TTFT p50 0.222 / p90 3.60 / p99 7.88 / max 9.35 s, tbt p50 23.5 / p99 74 ms, longest gap 87 ms;
offline 5.414 / 5.550 ms. As predicted.
Then guard 5 s with the near-miss window (the faster engine again): replays 181, 181, 181 (0.958), tbt p50 25.1,
24.7, 24.7 ms, TTFT p99 7.7 s (guard 5.5 s: 181 / 25.2, 24.8; guard 6: 180 / 24.1; near 1.0 / 1.25 with guard 8:
180 / 24.6-24.7). The evaluator's tbt p50 has been ~0.5 ms below my replays (s0031: 23.5 against 23.9-24.2).
Guard 5 s now default; if the evaluator's median comes out above 25 ms, back to 8 s.
s0032 (track B): expected 0.958 (181), tbt p50 24.2-24.8 ms. Measured: VALID, 0.958 (181/189), TTFT p50 0.219 / p90
3.37 / p99 7.82 / max 9.48 s, but tbt p50 25.28 ms - over the board's 25 ms line (this time the evaluator's
median came out 0.3 ms ABOVE my replays). Offline 5.454 / 5.568. Not compliant: s0031's setting (guard 8 s,
0.952, tbt 23.5) stays the one that meets every line; looking for 181 with more margin, else guard back to 8.
Margin search (pod epoch 9): guard 5 with 70 ms tail passes 181 / tbt 24.8, 24.8; guard 5 with a 26 ms normal
target 181 / 25.1, 25.2; guard 6 180 / 25.2, 25.2 (on this pod the same settings run ~0.5-1 ms higher than on the
last one). No setting gives 181 with a safe margin under 25 ms. Default back to guard 8 s (s0031's setting).
Overload level (simulator): passes of 140-200 ms while the estimated backlog is past 12-15 s: at 1.5x the rate
TTFT p99 15.7 -> 19.5-20.3 s and 100+ streams over the 100 ms line; at the standard rate 180 -> 173. No.
k_prep with a warp per (column, 8 vectors): no faster (mk note).
Guard 5 s with shorter tail passes: 60 ms 180, 180 / tbt 25.4, 25.3; 65 ms 180, 181 / 24.9, 25.2: the tail mode
then lasts longer, the median stream pays the same. GEMM shape picks against per-shape kernels at 128-1024
columns (tests/gemm_shapes.py, MK_GEMM_SPLIT=3): every pick within 0.7% of the best.

## State at 31 h (the engine in the workspace = s0031: track B, guard 8 s, near-miss 0.75 s)
Evaluator (s0031): 0.952 (180 of 189), TTFT p50 0.22 / p90 3.60 / p99 7.88 s, time between tokens p50 23.5 /
p99 74 ms, longest gap 87 ms, offline step 5.41 / 5.55 ms (generate 5.43 / 5.56), all gates passed, milestones
up to h0_step and online_beats_engines. s0032 showed 0.958 at a median of 25.3 ms (guard 5 s): the frontier
between the two lines, one request per ~1-2 ms of median time between tokens.
Open from the board: TTFT p99 (engines ~6 s / ~10 s at 1.61 / 2.39 req/s; here 7.9 s / ~13 s): the clearing
time of the trace's same-timestamp groups at full prefill speed with stall-free decoding.
Track A (engine/TRACK = A): builds and passes every test; record s0024.

## Age cap (MK_AGE_S 10, both schedulers)
Prompts that have waited longer than 10 s go first, oldest first (after the deadline order). Parity: the device
keeps arrival times in float32, so the Python reference now does the same in this check and in the first-token
estimate (0 mismatches at 1, 1.5 and 2x the rate). Standard rate: never triggers (180, same TTFT). 1.5x the rate,
240 s: TTFT p99 10.8 s either way (13.2 s with s0029's engine: the light passes, PDL and per-shape kernels moved
it), max 22.8 -> 19.7 s, share 544 -> 545 of 585. Default on.
Time between tokens, run to run (pod epoch 9, three replays each): near 0.75 s / target 27 ms: 180, tbt p50 24.1,
25.0, 25.2; near 0.75 / target 25: 180, 24.1, 24.6, 24.7; near off / target 27: 179, 23.3, 23.1, 23.1. The median
moves by ~1 ms between runs and pods (s0031's evaluator run: 23.5). Default now target 25 ms with near 0.75 s
(more margin at the same share); submitted to see the evaluator's number.
s0033 (track B): expected 0.952, tbt p50 23.6-24.7 ms. Measured: VALID, 0.952 (180/189), TTFT p50 0.215 / p90 3.63
/ p99 7.95 / max 9.53 s, tbt p50 24.77 / p99 74 ms, longest gap 87 ms; offline 5.453 / 5.564 (this pod). Under
the line, with a thin margin (s0031's 23.5 was the other end of the same spread). If a re-measurement needs the
margin, MK_NEAR_S=0 gives 179 of 189 at ~23 ms.
Final validation of HEAD (s0033's engine, pod epoch 9): tests/run_all.sh passes; tools/online_eval.py --rates
"2 4 1": valid, every milestone up to h0_step / online_beats_engines; r2 0.952, TTFT p99 7.94 s, tbt p50 24.1 ms;
r1 (0.86 req/s) 0.951 (s0029's engine: 0.942), tbt p50 20.1; r4 (3.07 req/s) 0.894 (0.910 with the 27 ms target:
the shorter normal passes cost prompt throughput under overload), TTFT p99 23.7 s.
At 1.5x the rate (2.36 req/s, the board's second rate is 2.39): 120 s: 0.951 (270 of 284), TTFT p99 10.1 s,
max 10.7 s, tbt p50 48.5 ms (target 27: 270, 10.2 s, 46.7 ms); 240 s: 544 / 547 of 585, p99 10.8 s either way.
The engines' TTFT p99 there was 9.5 / 10.1 s (HTTP, 300 s): about on par now (s0012: 85 s; s0023's engine 13.2
s). At 1.61 req/s the gap stays (7.9 against 5.9 s). Target 25 ms kept.
Super-segments of 2048 positions (SS_LOG 4): replay decode-only passes 10.07 vs 10.04 ms, big prompt passes 69.8
vs 68.5 ms (more parts to fold), 179 vs 180. With 8192 (SS_LOG 6, earlier) the decode passes were slower. 4096
stays. The 1.61 req/s first-token tail in detail (replay, the t = 10 s group, 22 requests, 245k prompt tokens,
served shortest first): the last three (26k, 33k, 33k tokens) get their first tokens at 5.96, 7.73, 9.47 s;
a 33k prompt takes ~1.75 s at the end of a burst (its late chunks attend to 16-33k positions). The second-to-last
is the p99; no order makes it smaller than the end of the group minus the longest prompt.
Ping-pong attention with q as register operands of the scores (wgmma RS, A from registers; 238 registers, same
bits): 2040 / 28k 125.5 vs 126.3 (ping-pong) vs 125.4 ms (k_attn<2>); 512 / 16k 28.9 vs 28.3 vs 27.8. Shared-
memory reads are not what bounds it either. (Kept in k_attn_pp, which stays off.)

## State at 32.5 h (the engine in the workspace = s0033)
Track B, defaults: normal target 25 ms, boost 45 ms, tail guard 8 s / 80 ms, near-miss window 0.75 s, age cap
10 s; light decode passes and per-shape bucket branches in the serving loop; PDL with early weight stages.
Evaluator: s0033 0.952 (180/189), TTFT p50 0.215 / p90 3.63 / p99 7.95 s, time between tokens p50 24.8 / p99
74 ms, longest gap 87 ms, offline 5.45 / 5.56 ms (s0031, same scheduling at 27 ms: 0.952, 23.5 ms, 5.41 / 5.55).
At 2.36 req/s (replay): 0.951, TTFT p99 10.1 s; at 0.86 req/s: 0.951; at 3.07 req/s: 0.894.
Against the board's lines: share, TTFT p50 / p90, tbt p50 <= 25 ms, offline <= 5.62 / 5.97 ms: met. TTFT p99:
~on par with the engines at 2.39 req/s, 7.9 vs ~6 s at 1.61 req/s (the clearing time of the largest arrival
group, see above). Track A builds (engine/TRACK) and passes every test; its record is s0024.

## The other runs (operator note of 2026-11-11, read at 32.5 h)
runs/: s2-q8b-a1 (same model and card, offline step_us_b1), s4-north-h100-a1 (another
model, HTTP step), two GB200 runs. All of them optimise the decode step at B=1..8; none serves prompts.
What they found, and what it means here:
- The largest win in both H100 runs is a lossless 12-bit code of the BF16 weights (sign + 4-bit exponent code
  + 7-bit mantissa, exact power-of-two row/column scales, rare escapes patched by extra MMAs): s2 5340 -> 4522 us
  (-15%), 4020 us at the end. Their BF16 track-B step before it (5340 us, one persistent kernel with grid
  barriers) is where ours is (5410 us with PDL kernels), so the gap between the runs is that code, not the
  kernel structure.
  Here it would pay only where the pass streams weights faster than it computes: decode-only passes (~10 ms in
  the replay, ~5 of it weights: -0.8 ms) and the offline step (a milestone already met). The share is set by the
  first tokens of the bursts (prompt passes at 300-2000 columns, compute-bound: GEMM ~556 TFLOP/s, attention
  ~466) and tbt p50 by the pass target (25 ms, the passes almost always carry prompt chunks at 2 req/s). A
  second weight copy costs 12 GB of the K/V pages (the t = 10 s group alone holds 245k prompt tokens), and the
  wgmma pipeline would need a decode stage. Not taken: no measurable effect on this metric expected for ~5 h of
  work with the time left.
- Layer 0 q/k/v from a per-token table (north): 1/36 of the q/k/v GEMM, ~0.5% of a decode pass. Not taken.
- Removing waits that are not dependencies, keeping HBM busy across boundaries: here PDL with early weight
  stages, the static attention path and idle blocks that exit without waiting did that (s0026-s0030).
Next: scheduling knobs that buy tbt margin at the same share (the line is 25 ms, s0033 measured 24.77).

## Near-miss pass target of its own (MK_TARGET_NEAR_MS, default = the tail target: no change)
Expected: the near-miss window (some prompt estimated in (5, 5.75] s) runs 80 ms passes; a shorter target there
(45-60 ms) might keep the 180th first token on time with less time between tokens. Parity of the device and Python
schedulers with the new knob: 0 mismatching passes (sim --impl both, target-near 45). Replays (pod epoch 9):
80 ms 180 / 180 (tbt p50 23.9 / 24.1), 45 ms 179 / 179 (24.0 / 23.8), 60 ms 180 / 180 (24.1 / 24.1). The
window is too short to move the median; 45 ms loses the near miss it is there for. Default stays 80 (knob kept).
Normal pass target below 25 ms (replays, pod epoch 9): 23 ms 179 / 179 (tbt p50 23.1 / 23.1), 24 ms 179 / 179
(24.0 / 23.1), 25 ms 180 / 180 (24.0 / 24.0). The 180th request needs the 25 ms passes' prompt throughput; 25 is
the knee. Default stays 25 ms.

## Track A: k_rnorm stores whole lines (more than 64 columns)
Track A prompt passes were ~9% slower than track B's (6 passes of 1024 columns at context 0-5k: 27.8-35.5 ms
against 25.1-33.0). nsys put the difference in k_rnorm (41 us per norm against track B's fused 1/rms), but its
grid was not the limit (1, 4, 8 blocks per SM: same pass times). ncu: 82 us cold, 0.55 TB/s. The stores were:
a warp per (column, eighth of the vector), each lane writing 16 bytes into its own 128-byte line of the operand
layout, the line's other 7 columns written by 7 other blocks. Ablation with contiguous (wrong-layout) stores:
-2 ms per pass. Now (T > 64) a warp takes a group of 8 columns and an eighth of the vector, lane (c, r) =
(lane & 7, lane >> 3): one store instruction fills four whole lines; each of the 8 columns' 1/rms is the same
lane-partial sum + butterfly as before (8 butterflies per warp). Bits unchanged: tests/test_eng.py --steps 200
(prefill in one 200-column chunk and in 32-column chunks of three rows bitwise equal to the steps, which keep
the old path) PASS.
Pass times (tests/ab_ta_pass.sh, track A, two rounds): old 27.85-35.5 ms, new 25.98-33.6 (-1.8 ms, -6%), with
4 blocks per SM 26.2-34.3 (TA_RN_BIG stays 1). Offline steps unchanged (T <= 8: old path) 5.73 / 5.86 ms.
Track A replay (two runs): 178 of 189 (0.942; before 177), TTFT p50 0.25 s (0.36), p90 4.0 s (4.24), p99 8.3
s (9.1), tbt p50 25.2 / 26.5 ms (26.4 / 27.7). Track B untouched (the change is inside TRACK_A).
Track A with the near-miss window off (replays): 177 / 178, tbt p50 23.9 / 23.9 ms (target 24 ms with the window:
178 / 178 at 27.7 / 27.9). Default for track A now near 0 (the library reports its track, eng_const(12)).
s0034 (track A, engine/TRACK=A for the submission only): expected 0.937-0.947, tbt p50 23.5-25, offline ~5.73 /
5.86. Measured: VALID, 0.942 (178/189), TTFT p50 0.253 / p90 4.05 / p99 8.36 / max 10.2 s, tbt p50 23.85 / p99
75.6 ms, longest gap 87 ms, offline 5.738 / 5.873 ms, cost-model correction 1.089. First track-A result under
the tbt line and within TTFT p50 / p90 of the engines; offline B=1 is 2% over its line (5.62), B=8 under.
Next on track A: the 73 k_rnorm kernels of a decode step (~3.9 us each against track B's fused 1/rms).

## Track A: the norm inside the ADD GEMM for small passes (no k_rnorm): slower, reverted
Expected: the 73 k_rnorm kernels of a track-A decode step cost ~3.8 us each (5.73 against track B's 5.45 ms);
doing the norm in the ADD kernel should save ~1.5-2.5 us each and bring B=1 under the 5.62 line.
(1) The last ADD block to finish (atomic count) normalizes the T columns: B=1 5757, B=8 6551 us (one SM, T x 8
items in a row). (2) All ADD blocks meet at a grid barrier (count + generation, a block per SM) and share the
columns: 5827 / 5991 us. Against 5727 / 5863 with k_rnorm. Bits equal in both (test_eng --steps 200, test_online).
Why: with PDL the next GEMM is resident (its producers streaming weights) as soon as the ADD blocks leave their
SMs; k_rnorm needs no shared memory and runs alongside. A norm inside the ADD kernel keeps the SMs' shared memory
until it is done, so the next matrix starts streaming later. The k_rnorm kernels cost less than they appear to;
track A's B=1 offline gap (5.73 against 5.62) stays.

## Track A: the norm applied to the GEMM's input stages (no k_rnorm up to 64 columns): slower, reverted
Timing ablation first (wrong results): track A without its norm kernels up to 64 columns: 5451 / 5556 us, i.e.
the 73 k_rnorm kernels are the whole track-A gap at B=1 and B=8 (5726 / 5864).
Then: the ADD epilogue stores bf16(x) into the operand buffer (exact: track A's residual is BF16-valued), the
QKV / GU consumers compute the columns' 1/rms (the track-B code path) and turn each input stage into
bf16(w * bf16(x * 1/rms)) in shared memory before the MMA (fence.proxy.async + warpgroup barrier). Bits equal
(test_eng --steps 200, bitwise prefill/steps). One stage at a time: 7741 / 7804 us (a few hundred ns in every
chunk's path, against ~150 ns of weight streaming per chunk); batches of up to 8 stages, loads first: 6623 /
6772 us (waiting for a batch drains the weight pipeline). The input would have to be normalized once per block
for the whole K (128 KB at 16 columns, next to 96 KB of weight stages): a new stage layout, not in the time
left. Reverted; track A keeps its norm kernels (5.73 / 5.86 ms, s0034).
Final checks of HEAD (pod epoch 9): tests/run_all.sh passes on track B and on track A (engine/TRACK=A in the pod
only): unit/bitwise, GEMM shapes, generate, 34000 positions, online (device loop, host steps, Python scheduler,
small ring/arena).
Normal target 24.5 ms (track B): 180 / 180, tbt p50 24.2 / 23.8 (25 ms in the same run: 180, 24.0). Within the
run-to-run spread of the median; 25 stays. (A first set of these replays ran a stale track-A library left in the
pod by the track-A suite run: build.sh decided by mtimes and the workspace sync restores TRACK's old mtime. It
now also records the track and flags it built with, engine/libeng.so.flags, and rebuilds when they differ.)

## State at 34.7 h
Track B (default, engine = s0033 + no-op knobs): evaluator s0033 0.952 (180/189), TTFT 0.215 / 3.63 / 7.95 s,
tbt p50 24.8 / p99 74 ms, offline 5.45 / 5.56 ms; best share s0032 0.958 (tbt p50 25.3: over the line).
Track A (engine/TRACK=A): s0034 0.942 (178/189), TTFT 0.25 / 4.05 / 8.4 s, tbt p50 23.9 / p99 76 ms, offline
5.74 / 5.87 ms (B=1 2% over its line: the 73 norm kernels are the whole gap, and the two ways of removing them
were slower, see above).
Since 32.5 h: the other runs read (12-bit lossless weights: not for this metric), near-miss target knob (no
lever), normal target 23 / 24 / 24.5 / 25 ms (25 is the knee), track A k_rnorm whole-line stores (-6% prompt
pass, same bits) and near-miss off for track A, build.sh rebuilds on a track change.

## k_prep: V of 8 aligned positions in one 16-byte store per dim
Ablations (timing only, wrong results; six 1024-column passes at context 0-5k, tests/ab_pass.sh): no k_prep work
at all -1.0 ms at context 0; no V stores -0.4..-1.0 ms; no K stores -0.1, no q stores -0.2 (at context 0; at
longer context K / q garbage changes the attention's data-dependent work, so only context 0 counts).
The V layout keeps the 8 positions of a dim in 16 contiguous bytes; k_prep wrote them as 2-byte stores, one per
(column, dim), from 8 warps of different blocks (the pattern that made track A's k_rnorm slow). Now V is done by
groups of 8 columns (prep_v8, a warp per (group, v head)): if the 8 columns are positions p .. p + 7 (p a multiple
of 8) of one run, a lane writes one 16-byte store per dim (and the 16-byte zero tail when the run ends at a
group's middle, as before); otherwise prep_vec column by column. Same values: tests/run_all.sh passes (bitwise
prefill/steps, 34000 positions with 108 chunks bitwise, online with NaN-poisoned pages).
Passes: 24.69-33.13 ms against 25.41-33.81 (PREP_V8=0), ~-0.5 ms (-2%) at short context.
Replays: three with it 179 / 179 / 179 (tbt p50 24.6-24.8); then in one run both builds: 179 / 179 (24.7 / 24.7)
with the same 10 misses (all 26-33k-token prompts, TTFTs within 0.06 s) - the earlier 180s of the day were the
other side of the same knife edge, which moved within the session. Neutral on the share, faster: kept (PREP_V8 1).
Then the committed engine before this change against it, same session, alternating: before 180 / 180 (tbt p50
24.8 / 24.1, TTFT p90 3.63 / 3.62), with prep_v8 179 / 179 (24.7 / 24.6, p90 3.77 / 3.76). The PREP_V8=0 build
of the comparison above was not the old code (its items were reorganized) and also gave 179. So any change of
k_prep's timing moved the 180th request over 5 s here, although the passes are faster: the share sits on a knife
edge of the scheduler's dynamics (same misses, all 26-33k-token prompts). Reverted to the previous eng.cu; the
replay is the evaluator's stand-in and it says 180 before, 179 after. (mk note.)
Scheduler knobs, one replay each (track B, same session): margin 0.2 / 0.4 s: 180 / 180 (tbt 24.0 / 24.0);
prefill share 0.6: 179 (26.5), 0.8: 180 (24.1); boost slack 0.5 / 1.5 s: 180 / 180 (23.8 / 24.1). The default
is on a plateau at 180 in these directions; no setting gave 181 under the tbt line.
Which request flips (per-request records, same session, previous engine against prep_v8): request 10 of the t = 0
group (26.9k tokens) gets its first token at 4.91 s before and 5.55 s with prep_v8 (request 8, 23.1k: 3.46 ->
4.10 s); every other miss is the same. Pass logs: both runs put the same columns through in the first 6 s (116.7k)
in the same GPU time (5992 / 5996 ms), but prep_v8's run did it in 270 passes instead of 238 (its estimates were
3.7% under the measured times against 1.7%: the cost model's per-column terms were calibrated with the old
k_prep, and smaller passes cost more per column). The 2% kernel gain did not reach the burst; the pass sizes did
change. A recalibration (tests/calib_cost.py) would be the way to take such a gain; not done in the time left.
Track A, more weight stages requested before the wait (PDL_EARLY, pod epoch 10): 3: 5676 / 5840 us, 6: 5668 /
5825, 10: 5682 / 5828. -8 / -15 us: not worth a switch (and the default would move track B's knife edge).

## What I would do next (35.8 h)
- Recalibrate the scheduler's cost model (tests/calib_cost.py -> sched.py / dsched.h constants and the pass
  classes) for the current kernels, then take kernel gains that the old model turned into smaller passes:
  k_prep's packed V stores (-2% per prompt pass, bits equal; commit 9bd1106) lost the 180th request because the
  passes got smaller, not slower.
- Track A, B=1 line: the 73 norm kernels are the whole gap (5.73 against 5.45 ms without them). The input of
  QKV / GU would have to be normalized once per block for the whole K and kept in shared memory (a new stage
  layout for up to 16 columns); per-stage normalization and norms inside the ADD kernel were slower.
- Prefill: the 12-bit lossless weights of the other runs do not help prompt passes; the large GEMMs are at
  ~650-710 TFLOP/s and attention at ~466; the first-token tail at 1.61 req/s is bounded by the largest burst's
  work (~9.5 s for 245k tokens).
HEAD on the new pod (epoch 10), clean build: test_eng PASS, offline 5414 / 5550 us, replays 180 / 180 (tbt p50
24.2 / 24.2 ms, TTFT p90 3.62 / 3.63 s). Submitted as the final track-B source (s0033's engine; since then only
track-A code, a default-preserving knob and build.sh's flags check).
s0035 (track B, final source): expected 0.952, tbt p50 23.5-25, offline ~5.41 / 5.55. Measured: VALID, 0.952
(180/189), TTFT p50 0.223 / p90 3.63 / p99 7.97 / max 9.51 s, tbt p50 24.1 / p99 73.8 ms, longest gap 86 ms,
offline 5.413 / 5.551 ms; every milestone up to h0_step. As expected.
The operator's TTFT p99 condition, measured their way (300 s of arrivals; tests/replay.py --duration 300, final
engine, pod epoch 10): 1.61 req/s: 383 requests, 0.977 within, TTFT p50 0.20 / p90 2.46 / p99 7.6 / max 9.3 s,
tbt p50 20.4 ms; 2.39 req/s: 585 requests, 0.959, TTFT 0.22 / 3.12 / 7.9 / 10.6 s, tbt p50 35.3 ms, longest gap
88-91 ms. Against the engines' HTTP numbers (p99 ~6 s at 1.61, 9.5 / 10.1 s at 2.39): better at 2.39 req/s,
still 1.6 s behind at 1.61 (the same burst-clearing bound).
The misses at 1.575 req/s by burst (replay records, final engine): every one is the last one or two prompts of a
group that arrives at once: t = 0 (15 requests, 141k tokens), 10 (22, 246k), 50 (20, 212k), 90 (16, 169k), 100
(15, 165k), 110 (16, 197k); the groups clear at ~25-35k prompt tokens/s, and the closest calls are 4.91 s (made
it) and 5.64 s (missed). One more request within 5 s needs ~10-15% more long-context prefill throughput.

## Final state (36.3 h)
Track B (default): s0035 = the committed source, 0.952 (180/189), TTFT 0.22 / 3.63 / 7.97 s, tbt p50 24.1 ms,
offline 5.41 / 5.55 ms; best share on the board s0032 0.958 (tbt p50 25.3, over the line). Track A
(engine/TRACK=A): s0034, 0.942 (178/189), tbt p50 23.9 ms, offline 5.74 / 5.87. Both pass tests/run_all.sh.

## prep_v8 once more, with the fallback fixed: rejected
Pass logs of the replays (previous engine against prep_v8): passes of 257-600 columns, which carry the decode
rows, were SLOWER with prep_v8 (22.6 against 22.2 ms at the same columns; estimates 4% under). A group of 8
columns that is not one aligned run (decode rows) fell back to one warp doing 8 V vectors in a row, each behind
three dependent loads (plan slot / position, page table). Fixed: lanes 0-8 load the plan entries and pages of the
8 columns and of the column before at once, shuffles hand them out (no dependent loads left in the fallback).
Bits equal (test_eng --steps 200, test_online with poisoned pages).
Same session, alternating (tests/ab_prev.sh builds the committed engine and the working one): 512 + 32 decode
columns at context 0-3.5k: no difference beyond noise (24.8-26.6 ms both); replays: previous 180 / 180 (tbt p50
24.0 / 24.1), prep_v8 180 / 180 (25.0 / 24.8). The median time between tokens moves to the line: rejected for
good; eng.cu stays as committed.

## Track A, offline steps: the input normalized inside QKV / GU (PB), the norm kernels gone at B=1
Expected: the 73 norm kernels are the whole track-A gap (ablation: 5.45 ms without them); the per-stage transform
failed because it sat in every chunk's path. Now (gemm_body template flag PB, offline passes of at most TA_PB_T
columns): the consumers compute the columns' 1/rms (the track-B code path), then write bf16(w * bf16(x * 1/rms))
of the whole K (4096) once per block into a resident 64 KB region of shared memory in the stage layout (fence.proxy
.async, barrier), and the stages carry only weights (q/k/v with preparation: 9 stages instead of 10 to fit 221 KB).
The ADD kernels are unchanged (x and the per-tile sums of squares as before); the first norm of layer 0 and the
final norm stay kernels.
Measured (pod epoch 10): with 8 columns: B=1 5676 -> 5526 us, B=8 5840 -> 6055 (the normalization of 8 columns is
16 vectors per thread in a row); so only for 1 column (TA_PB_T 1): 5529 / 5838 us. Track A's offline lines (5.62 /
5.97 ms) are both met now. Bits equal: tests/run_all.sh with engine/TRACK=A passes (bitwise prefill/steps, GEMM
shapes, generate, 34000 positions, online). Track B unchanged (all of it under TRACK_A): 5415 / 5551 us, PASS.
The serving loop's passes (device, columns from the plan) keep the norm kernels: track A's online numbers stay.
s0036 (track A, engine/TRACK=A for the submission only): expected offline ~5.53 / 5.84 ms, online as s0034.
Measured: VALID, 0.942 (178/189), TTFT p50 0.243 / p90 4.02 / p99 8.33 s, tbt p50 23.9 / p99 75.7 ms, longest gap
87 ms, offline 5.535 / 5.841 ms. Track A now meets the tbt line and both offline lines (5.62 / 5.97); TTFT p50 /
p90 under the engines' 1.6 / 4.9 s.
PB prologue with the weights loaded once per thread and x for 4 columns at a time: B=8 with 8 columns in PB 5964
us (6055 before, 5838 with the norm kernels), B=1 5550 (5529 with the committed prologue). Not kept; TA_PB_T stays 1.

## Final state (36.9 h)
Track B (default): s0035, 0.952 (180/189), TTFT 0.22 / 3.63 / 7.97 s, tbt p50 24.1 ms, offline 5.41 / 5.55 ms.
Track A (engine/TRACK=A): s0036, 0.942 (178/189), TTFT 0.24 / 4.02 / 8.33 s, tbt p50 23.9 ms, offline 5.54 / 5.84
ms: every line of the board met on track A except the TTFT p99 at 1.61 req/s, which neither track meets.
The workspace HEAD is both: engine/TRACK = B; the track-A build is the same source with TRACK=A.

## GEMM epilogues of prompt passes (36.9 h)
gemm_shapes.py (raw products, random data) at 1024 columns: q/k/v 81 us (634 TFLOP/s), o 52 (667), gate/up 295
(698), down 150 (687): the main loops are near this tile's rate. Ablations of a 1024-column pass (tests/ab_pass.sh,
timing only): no residual loads in the ADD epilogue (EPI_ABL 2) 25.2 -> 23.0 ms (-2.1 ms, 8%); no GU / ADD
epilogue at all (EPI_ABL 8) -8.7 ms (contaminated: the following kernels read stale data, but the size says the
epilogues, not the main loops, are where a prompt pass loses). The ADD kernel uses 166 of its 168 registers (no
room to prefetch the residual into registers). L2 prefetch of the residual a few stages ahead (EPI_XPF 8 / 24):
-0.2 ms; read-only path (EPI_XNC): no change.
EPI_XST: on the block's last unit (every block at 512-1024 columns), after both warpgroups' last MMA, each consumer
thread requests all its residual values of the unit at once by cp.async into the idle stage buffers, one wait,
then the column-group loop reads them from shared memory. Same bits (tests/run_all.sh passes). Passes: -0.4 ms at
context 0 (24.8 against 25.2), mixed beyond: so the round trips were ~0.4 of the 2.1 ms; the rest is the traffic
itself competing with the weight stream.
Same session, alternating (tests/ab_prev.sh): committed engine 180 / 180 (tbt p50 24.1 / 24.2 ms, TTFT max 9.5 s),
EPI_XST 180 / 180 (23.9 / 23.9 ms, 9.4 s). Kept (default on, both tracks).
s0037 (track B): expected 0.952, tbt p50 23.5-24.5. Measured: VALID, 0.952 (180/189), TTFT p50 0.219 / p90 3.62 /
p99 7.95 / max 9.39 s, tbt p50 23.8 / p99 74.5 ms, longest gap 86 ms, offline 5.414 / 5.551 ms. As expected; the
tbt margin grows (s0035: 24.1).
Correction: the "no epilogue" ablations (EPI_ABL 8, and a GU-only variant: -4.0 ms per 1024-column pass) do not
measure the epilogues: without them the products are unused and the compiler drops the MMAs (a GU kernel would
otherwise run above the tensor cores' peak). The residual-load ablation (2) keeps the products and stands: 2.1 ms,
of which EPI_XST recovered ~0.4; the rest is the traffic.

## Final state (37.6 h)
Track B (default): s0037, 0.952 (180/189), TTFT 0.22 / 3.62 / 7.95 s, tbt p50 23.8 ms, offline 5.41 / 5.55 ms.
Track A (engine/TRACK=A, same source): s0036, 0.942 (178/189), tbt p50 23.9 ms, offline 5.54 / 5.84 ms (the
residual staging applies to it too; its offline numbers are unchanged, its suite passes).
Left on the table, measured: the residual traffic of the ADD epilogue (~1.7 ms of a 1024-column pass), the
staging for a block's non-last units at 2040 columns (stage buffers busy with the next unit; a 32 KB cp.async ring
would fit next to the stages, worth ~0.2 ms per such pass), ~0.55 ms of launch gaps per pass (216 kernels).
300 s of arrivals with s0037's engine: 1.61 req/s 0.977 (374/383), TTFT p99 7.5 s (s0035's: 7.6), tbt p50 20.4 ms;
2.39 req/s 0.961 (562/585; before 561), TTFT p99 8.0 s (7.9), tbt p50 32.4 ms (35.3).
L2 persistence window over the residual stream (an MK_L2X knob, access-policy window on every kernel / graph
node; H100 allows 31 MB set aside): 1024-column passes at context 0-5k: off 24.8-33.4 ms, 16 MB 24.7-32.9 (noise),
32 MB 26.1-34.7 (worse); 2048-column passes to context 30k: 32 MB +1-3 ms per pass (attention's K/V lose the L2).
Not kept (code reverted).
tools/online_eval.py --rates "2 4 1" with s0037's engine (pod epoch 10): valid; r2 (1.57 req/s) 180/189 = 0.952,
tbt p50 23.9 ms, TTFT p50 0.22 / p99 7.97 s; r1 (0.86 req/s) 98/103 = 0.951, tbt p50 20.0 ms, TTFT p99 7.9 s;
r4 (3.07 req/s) 327/368 = 0.889, tbt p50 62 ms, TTFT p50 0.91 / p99 24.3 s (past capacity).
Track A with EPI_XST, same session, alternating against s0036's source: before 178 / 177 (tbt p50 23.8 / 23.9),
after 179 / 178 (23.9 / 23.9). Submitted as track A.
s0038 (track A): expected 0.942-0.947. Measured: VALID, 0.947 (179/189), TTFT p50 0.246 / p90 4.02 / p99 8.25 s,
tbt p50 23.7 / p99 75.4 ms, offline 5.533 / 5.843 ms. New track-A record; it meets every line of the board but
the TTFT p99 at 1.61 req/s (as track B).
Residual staging for narrower blocks (EPI_XST_NB 16 instead of 64; light decode passes of 17-64 columns), device
loop decode passes (tests/online_dec.py, tests/ab_dec2.sh): B=24 6855 / 6853 us both, B=48 8189 -> 8198 us. No
gain (one or two column groups per thread there); the threshold stays 64.

## Final state (38.7 h)
HEAD, clean build on pod epoch 10: test_eng --steps 200 PASS, test_online (poisoned pages) PASS, offline 5414 /
5551 us. Track B = s0037 (0.952, 180/189, tbt p50 23.8 ms); track A (TRACK=A) = s0038 (0.947, 179/189, tbt p50
23.7 ms, offline 5.53 / 5.84 ms).
Tail guard with s0037's engine (one session): guard 6 s 180 / 180 (tbt p50 24.0 / 24.3, TTFT p90 3.37 / 3.35, p99
7.7 / 7.8 s); guard 7 s 180 / 180 (23.9 / 23.8, p90 3.60 / 3.61, p99 7.8 / 7.7). Guard 8 (default): 180, 23.8-24.1,
p90 3.62, p99 7.95. The faster prompt passes leave room for an earlier guard; checking the TTFT p99 over 300 s.
300 s at 1.61 req/s (the operator's measurement): guard 7 s 0.977 (374/383), TTFT p90 2.45 / p99 6.9 s, tbt p50
20.7 ms; guard 6 s 0.977, p99 6.9 s, tbt 20.9 ms; guard 8 (earlier, same engine) 0.977, p99 7.5 s, tbt 20.4 ms.
Default guard now 7 s (same share and tbt at the standard rate, the first-token tail 0.6 s shorter). Submitted.
s0039 (track B, guard 7 s): expected 0.952, tbt 23.5-24.5, p99 ~7.7. Measured: VALID, 0.952 (180/189), TTFT p50
0.225 / p90 3.61 / p99 7.76 / max 9.37 s, tbt p50 23.9 / p99 73.3 ms, longest gap 88 ms, offline 5.414 / 5.552 ms.
Track A takes the same default (api.py); its record stays s0038 (measured with guard 8).
300 s at 2.39 req/s with guard 7 s: 0.962 (563/585), TTFT p50 0.21 / p90 3.04 / p99 7.2 / max 10.5 s, tbt p50 33.5
ms (guard 8: 0.961, p99 8.0 s). Engines: 9.5 / 10.1 s.

## Final state (39.5 h)
Track B (default): s0039, 0.952 (180/189), TTFT 0.23 / 3.61 / 7.76 s, tbt p50 23.9 ms, offline 5.41 / 5.55 ms;
over 300 s: TTFT p99 6.9 s at 1.61 req/s (engines ~6 s), 7.2 s at 2.39 req/s (engines 9.5-10.1 s).
Track A (engine/TRACK=A): s0038, 0.947 (179/189), tbt p50 23.7 ms, offline 5.53 / 5.84 ms.
Track A with the 7 s guard (replay): 179 / 189, tbt p50 24.1 ms, TTFT p90 4.04 / p99 8.3 s: as s0038 (guard 8);
not resubmitted.
Tail target 100 ms (guard 7 s): standard replay 180, tbt p50 23.9 / p99 85 ms, longest gap 107 ms (over the 0.1 s
stall-free line the board asks to keep), TTFT p99 7.6 s; 300 s at 1.61 req/s: TTFT p99 6.9 s (same as 80 ms), gap
109 ms. No gain on the tail, worse gaps: 80 ms stays.

## Run summary
See notes/SUMMARY.md (best: s0039, track B, slo_share_r2 0.952; track A s0038, 0.947).
