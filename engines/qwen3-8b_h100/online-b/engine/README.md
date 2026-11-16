# The engine

One model (Qwen3-8B, bf16), one GPU (H100 80 GB), no inference engine around the kernels.
History, measurements and dead ends: `notes/journal.md`.

## Where it stands (s0037, the evaluator, track B)

| | this engine | vLLM | SGLang |
|---|---|---|---|
| requests within the service levels, 1.57 req/s | 0.952 (180 of 189) | 0.64 | 0.67 |
| TTFT p50 / p90 / p99 | 0.22 / 3.6 / 8.0 s | 1.8 / 4.9 / - s | 1.6 / 5.2 / - s |
| time between tokens p50 / p99, longest gap | 23.8 / 74 ms, 86 ms | - / 321 ms, 503 ms | - / 2598 ms, 7050 ms |
| offline decode step B=1 / B=8 | 5.41 / 5.55 ms | | |

(Engine columns from the board's baselines; the same engine measured 0.952 with tbt p50 23.5 (s0031), 24.8 (s0033)
24.1 ms (s0035), and with the residual staging 23.8 ms (s0037).) Track A
(`engine/TRACK` = A, s0038): 0.947 (179 of 189), TTFT 0.25 / 4.02 / 8.2 s, tbt p50 23.7 / p99 75 ms, offline
5.74 / 5.87 ms (now 5.53 / 5.84 ms: offline steps of one column normalize inside QKV / GU, `TA_PB`). 300 s of arrivals (as the board's HTTP measurement): 1.61 req/s 0.977, TTFT p99 7.5 s (engines ~6 s);
2.39 req/s 0.961, TTFT p99 8.0 s (engines 9.5 / 10.1 s), tbt p50 32 ms.
The first-token tail is the time it takes to clear the trace's
groups of same-timestamp requests (up to ~250k prompt tokens at once) with every stream kept under ~90 ms; see
the journal's last entries for what was tried against it. At 0.86 req/s: 0.951; at 3.07 req/s: 0.889 (s0037's engine, tools/online_eval.py).

## What runs

A **pass** takes T token columns once over the weights: every decode row and every chunk of every prompt that is
being prefilled ride the same pass. The arithmetic of a column does not depend on the other columns, on where a
prompt is cut into chunks, on whether its prefix K/V came from the cache, or on which kernel variant computed it
(the canonical arithmetic is at the top of `eng.cu`). That is what the `schedule` and `online_vs_step` gates test.

| file | what |
|---|---|
| `eng.cu` | every kernel and the C entry points |
| `dsched.h` | the scheduler, C++ compiled for the device (kernel `k_sched`) and for the host (tests) |
| `sched.py` | the same policy in Python: reference for the simulator and for `MK_DEVSCHED=0` |
| `api.py` | `Engine`: offline `reset / step / generate`, online `submit / poll / idle / kv_pages / drop_cache / stats` |
| `gen_wg.py` | generates `wg_gen.h`, the wgmma instruction macros |
| `build.sh` | builds `libeng.so` (run by the evaluator; rebuilds when any source is newer) |

### Kernels of a pass (per layer)

- `k_gemm<epilogue>`: Y = W X with wgmma (operands in shared memory), `cp.async.bulk` (the copy engine brings weights
  and activations) and mbarriers. A block = 2 consumer warpgroups + up to 4 producer threads; it computes units of
  64 NT rows x 2 NB columns. `gemm_shape()` picks (NB, NT) per matrix and number of columns from a small cost model;
  every shape gives the same bits (`tests/gemm_shapes.py`). Epilogues: q/k/v (up to 16 columns it also prepares
  q, K and V: RMSNorm, RoPE, stores), residual add + next norm's input, gate/up with SiLU, LM head (logits or
  arg max). Up to 64 columns the kernel computes the 1/rms of its input norm itself.
- `k_prep`: q, K, V of the columns the GEMM did not prepare. K and V go to the sequence's KV pages.
- `k_attn<2>`: attention for pairs of units (a unit = up to 16 consecutive positions of a sequence), per kv head
  and super-segment of 4096 positions; K/V segments staged by the copy engine. The two warpgroups of a block take
  scores / scalar work / values in different orders, so that one uses the tensor cores while the other is in its
  scalar part. `k_attn<1>`: the same for the single units left over at the end of a prompt run.
- `k_attn_dec`: units of one position (decode rows): the same products with the operand roles swapped (keys or
  dims on the 64-row side, the 4 heads as columns), on three warpgroups (`tests/att_roles.cu`: same bits). Its
  tasks are listed first among the single-unit tasks of the plan (header element 6).
- `k_attn_fold`: folds the super-segments of a unit in order.
- once per pass: `k_embed`, `k_gather`, `k_argmax`; in the serving loop `k_sched` (admission, pages, prefix cache,
  eviction, the plan of the pass) and `k_commit` (tokens to the result ring, loop condition).

Consecutive kernels of a graph are joined by programmatic edges (PDL): every kernel waits for the one before
with `griddepcontrol.wait`; a GEMM lets its successor launch once its producers have asked for every copy, and a
per-shape GEMM asks for its first weight stages before it waits.
In the residual-add GEMMs (o, down) a block's last unit stages its residual values in the idle stage buffers by
`cp.async`, all at once (`EPI_XST`): the epilogue's column groups no longer wait one round trip each.

### Who drives

- **Serving** (`submit` / `poll`): one CUDA graph `WHILE (work) { k_sched; SWITCH { ... }; k_commit }` runs on the
  device. k_sched picks the branch from the plan it just wrote: a pass whose columns are all units of one position
  (decode rows) takes a light pass (<= 16 / 32 / 64 columns) without the kernels that would have nothing to do
  (prompt attention, k_prep up to 16 columns, the 1/rms kernels); any other pass takes the branch of its run of
  sizes (18 runs over 1..2048 columns with one set of GEMM shapes each). Every branch uses per-shape GEMM kernels
  (the number of columns read from the plan): a decode pass ~0.5-0.8 ms less than with the merged kernel and the
  empty ones, a prompt pass ~1% (`MK_LIGHT=0`: always the general pass with the merged GEMM kernel).
  The host copies prompts into an arena and a request ring, ticks a clock word, and copies result blocks out. It
  never waits for the device and the device never waits for it (`stats()['host_syncs']` stays 0).
- **Offline** (`step`, `generate`): the host writes a plan of B decode rows; `step` is one graph launch per call,
  `generate` one graph `WHILE (step < end) { pass }` per stretch of steps.

### Scheduler policy (`dsched.h`, mirrored in `sched.py`)

KV pages of 512 positions with reference counts; prefix cache keyed by chained page hashes; a request whose next
page another request is computing waits for it; LRU eviction of unused pages; admission by page reservation.
Prompts are served in Moore-Hodgson order over their first-token deadlines; among prompts that arrive together
(within a quarter of a second) the shortest first; those that cannot make their deadline last, shortest first. The size of a pass comes from a cost model (`classes`, `c_seg`, `c_dseg`, `c_pair` in `sched.py`,
calibrated with `tests/calib_cost.py`): as many prompt columns as fit the pass target next to the decode rows.

## Knobs (environment)

| variable | default | meaning |
|---|---|---|
| `MK_TARGET_MS` | 25 | pass time aimed at while prompts are prefilled: the time between tokens of the streams |
| `MK_BOOST_S`, `MK_TARGET_HI_MS` | 1.0, 45 | while an on-time prompt has less slack than this, passes aim at the higher target |
| `MK_TAIL_S`, `MK_TARGET_TAIL_MS` | 7, 80 | while some prompt is heading for a first token later than this (by the estimates), passes aim at the tail target |
| `MK_NEAR_S` | 0.75 (track A: 0) | passes also aim at the tail target while some prompt is heading for a first token within this many seconds after 5 s (a near miss; 0: off) |
| `MK_AGE_S` | 10 | prompts that have waited longer than this come first, oldest first (0: off) |
| `MK_SHARE`, `MK_MARGIN_S` | 0.7, 0.3 | cap of the GPU share assumed for prompts; safety margin on the first-token deadline |
| `MK_DEVSCHED`, `MK_GRAPH` | 1, 1 | 0: Python scheduler on the host / host-launched steps (tests) |
| `MK_GEN_GRAPH` | 1 | 0: `step` / `generate` as plain kernel launches |
| `MK_LIGHT` | 1 | 0: the serving loop always runs the general pass (no light decode passes) |
| `MK_PREPQ`, `MK_GEMM_SPLIT` | 1, 1 | 0: always `k_prep` / always the general GEMM kernel (A/B switches) |
| `MK_POISON` | 0 | 1: freed KV pages are filled with NaN (tests) |
| `MK_PASSLOG` | unset | record (columns, selected, estimated ms, measured ms) per pass (`tests/replay.py`) |
| `MK_PAGES`, `MK_RESERVE_GB`, `MK_SO` | | KV pages, memory left free, another library to load |

Three pass targets: 27 ms normally (the streams' pace), 45 ms while a prompt that can still make its first token in
5 s is short of slack, 80 ms while any prompt is heading for a first token later than 7 s (the tail guard, MK_TAIL_S; 5 s
gives 181 of 189 but a median time between tokens just over 25 ms: s0032) or just past 5 s (the near-miss
window, MK_NEAR_S). With
`MK_TAIL_S=0` the tail is not guarded (TTFT p99 13 s instead of 8 s at the standard rate, 27 s instead of 10 s at
1.5x the rate) and the streams never see a pass above 45 ms.

## Track

`engine/TRACK` holds `A` or `B` (no file: B); `build.sh` passes it on as `-DTRACK_A=1`. The scheduler, the memory
layout and the kernels are the same on both; the switch changes what is rounded where (`ta_r`, `prep_vec`,
`k_rnorm`, the epilogues of `k_gemm`). Build the other track with `echo A > engine/TRACK; bash engine/build.sh`
(replay.py and the tests load `engine/libeng.so` as it is: rebuild after switching). On both tracks the weights are the checkpoint's BF16 values (permuted
layout), the KV cache is BF16 (K after norm and RoPE, V) and every matvec input is BF16.

- **Track A** (the general engines' formats and data flow): every operator's output is rounded to BF16 as a BF16
  tensor would be: the residual stream after each add, q / k / v, gate, up, SiLU, their product, the logits.
  RMSNorm is x * rsqrt(mean x^2 + eps) in f32, rounded, times the weight, rounded, before the matvec (`k_rnorm`
  writes that vector). RoPE on BF16 values with BF16 cos / sin, each product and the sum rounded. q is stored
  unscaled. Attention as in the flash-attention kernels those engines call: scores accumulated in f32, scaled in
  f32, softmax against a running maximum over blocks of keys (here 128), weights rounded to BF16 before they
  multiply V, the output accumulated in f32 and rounded once.
- **Track B** (default; lossless additions): (a) the residual stream and the matvec outputs stay in f32 between
  operators; (b) RMSNorm: the weight is multiplied in before the matvec and the 1/rms after it, and up to 64
  columns the GEMM computes the 1/rms itself (no `k_rnorm`); (c) q is stored multiplied by log2(e)/sqrt(128).

Against the float32 reference (tests/test_eng.py, relative L2 of the logits, median): track B 0.0044, track A
0.0115, HF in BF16 0.0106. Steady decode: track A is about 5% slower per step (one more kernel per norm):
offline step 5.73 / 5.86 ms against 5.45 / 5.57 (B=1 / B=8, pod epoch 9). Above 64 columns its `k_rnorm` has a
warp per 8 columns so that every store fills whole 128-byte lines of the operand layout (prompt passes -6%).
Track A runs without the near-miss window (`MK_NEAR_S` defaults to 0 when the library is track A): replay 177-178
of 189, tbt p50 23.9 ms (with the window: 178, 25.2-26.5 ms).

## Tests (in the pod, `$MK_PY`)

`bash tests/run_all.sh` (about 8 minutes): float32 reference and bitwise batch / chunking invariance, every GEMM
shape against the others, `generate` against step-by-step greedy, 34000 positions against the reference with NaN in
free pages, and the online contract with shared prefixes, eviction and recomputation under four configurations
(device loop, host-launched steps, Python scheduler, a request ring of 8 entries).

- `tests/replay.py`: the evaluator's replay with a per-request report (`--out`), `tests/sim.py`: the same replay on
  the CPU against the cost model (`--impl both` compares the Python and the device scheduler pass by pass).
- `tests/calib_cost.py`, `tests/time_step2.py`, `tests/gemm_shapes.py`: pass, step and GEMM timings. Compare builds
  in ONE command with `MK_SO`, alternating: numbers from different commands differ by a few percent.
- `tests/wg_scale.cu`, `tests/kv_stride.cu`, `tests/att_roles.cu`: what the tensor cores, HBM and the copy engine
  give, and that swapping operand roles keeps the bits. `tests/stream_rate.cu`, `tests/stream_pdl.cu`,
  `tests/mc_rate.cu`, `tests/tma_mc.cu`, `tests/wg_mix.cu`, `tests/sc_ops.cu`: weight streams per kernel size,
  with PDL, multicast, tensor/scalar overlap, softmax operations.
- `tests/ab_defs.sh`, `tests/ab_att.sh`, `tests/ab_head.sh`, `tests/online_dec.py`, `tests/nsys_step.py`: A/B of
  build flags (decode steps, prompt passes), against git HEAD, decode passes in the device loop, per-kernel
  times of a repeated step from an nsys trace.
