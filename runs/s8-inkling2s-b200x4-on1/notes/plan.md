# Plan and open items (run s8-inkling2s-b200x4-on1)

State: engine/ = s0043 (extension: the score is the share at 8 req/s, 120 s in process: **0.9974**, 778 of 780; 2 req/s
1.000 with TTFT p50 177 ms / p99 4.03 s, time between tokens p50 25.2 / p99 60 ms, longest gap 86 ms; steady decode
10.9 ms per step at B=1, 13.7 at B=8; 8 req/s over 300 s 0.991, 6 req/s 0.997, 10 req/s 0.956).
notes/summary.md says
what the engine is, notes/journal.md how it got there, notes/findings.md (F10 - F28) what was measured.
prev/ holds three attention kernels that were measured and not adopted (each with the reason).

## How to check a change (all through `mk exec`, pod python, MK_SCHED=host unless noted)
- `LAYER=5 T=2400 python tools/t_pe_layer.py` (also LAYER=2 T=1500, LAYER=41 T=300): one layer against the
  reference's attention / MoE on the same inputs (attention + o_proj ~0.002, NVFP4 MoE median ~0.003, MXFP8 ~3e-5).
- `L=12 R=2 T=40 CH=7 python tools/t_pe_ref.py` (and `R=1 T=2500 CH=501 STEPWISE=0`, `R=3 T=600 CH=64 STEPWISE=0`,
  `R=8 T=24 CH=3`): chunked == stepwise == chunks, bitwise; the printed rel err against the reference must not move
  unless the arithmetic was meant to change (0.1299845278263092 for the first, 0.10401666164398193 for the second
  since s0023).
- `python tools/t_long.py ref 6000` then `eng 6000 3584` and `eng 6000 700`: 6000-token prompt against the reference
  (rel L2 0.052 / 0.081, top-1 13 of 16), bitwise equal across chunk sizes.
- `MK_SCHED=host python tools/t_dev.py` then `MK_SCHED=device python tools/t_dev.py`: device loop == host tokens.
- `MK_SCHED=device python tools/t_cache.py 13`: eviction, recomputation, dedup, conservation with 13 pages;
  `MK_NPAGES=420 python tools/t_replay_online.py --probes 6`: the replay under memory pressure (recomputes > 0).
- `python tools/t_stress.py 150 700` (device loop; also `MODE=edges ... 26 300`, `MODE=flood ... 1500 300`, `MODE=slow ... 64 300`, and
  `150 150` for a store of two full-length requests): liveness, schedule invariance and conservation in a crowd.
- `python tools/t_mix.py` (online, offline, online on one engine), `python tools/t_close.py` (close under load),
  `python tools/t_gap.py` (decode-only gaps through the device loop); on the host, without a GPU:
  `g++ -O2 -std=c++17 -o /tmp/sched_fuzz tools/sched_fuzz.cpp && /tmp/sched_fuzz 200 1` (the scheduler's books after
  every step of random scenarios; run it after any change of engine/sched.h) and `python3 tools/t_ring.py` (the
  host's rings).
- `python tools/t_pe_one.py` (NS=..., P=..., DEC=1, MK_FIXED_GRIDS=1 for the device loop's grids) and
  `mk profile nsys -- env MODE=prefill N=2560 P=4096 REPS=4 python tools/t_pe_prof.py` + `python tools/nsys_names.py
  $DIR 20`: pass times and kernels by template name. A part of a pass is measured only by launching an idempotent
  kernel twice or by timestamps in the kernel: stubbing arithmetic changes the routing and with it every expert kernel.
- `python tools/t_replay_online.py --probes 12 [--rate R] [--duration S]` (device loop by default; `MK_SCHED=host
  MK_PASS_LOG=1` writes pass_log.json for the cost model fit): the evaluator's replay and schedule probes.
- `python3 tools/sim.py [--tmax N]` (`SIM_TABLE=1` for measured pass times, `MK_SOFT_ROWS`) and `g++ -O2 -std=c++17
  -o /tmp/sched_sim tools/sched_sim.cpp && /tmp/sched_sim tools/trace400.txt [npages tmax soft_rows]`: the Python and
  the device scheduler on a virtual clock; they must agree with each other. They are pessimistic by 0.3-0.5 s for the
  last request of the largest burst (linear pass model).
  Their pass-time defaults (13.5 ms + 17.7 us per row; decode-only 12.0 + 0.166 per row) are those of s0023: against
  the s0035 engine they are pessimistic by ~0.4 s on TTFT p99 and ~4 ms on the time between tokens p50 (also with
  16.0 us per row: 4.40 s / 29.3 ms against the device's 4.02 s / 25.2 ms). Refit base, row and the decode line from
  a pass log before tuning policy with them; for the schedule itself (rows, hits, evictions) they are exact.

## Where a 2560-row pass goes now (s0032; per card, nsys, position 4096, 61.8 ms under nsys, ~56 ms in the replay)
expert GEMMs 13.0 ms (they read every expert of the model once per pass: 376 GB, 11.8 ms at the node's peak rate),
sum of a row's expert outputs on its home card 8.6 (fetch ~4 at ~420 GB/s; waiting for the other cards' experts 4
without rebalancing, ~1.4 in the replay), attention 7.2, q/k/v 4.7, activation + quantization 4.3, attention-input
fetch 3.25, bias GEMM 2.8, o_proj 2.4, post_mix 1.9, post_attn 1.9, router 1.75, route 1.5, gather 1.3 (s0034),
fold 1.3, routing fetch 1.15 (s0035), prep 1.0, assign 0.9, packx 0.8, convolution state 0.55 (s0033); the scheduler
kernel 0.13.
To see where a change went: profile both builds (`mk profile nsys -- env MODE=prefill N=3584 P=4096 REPS=8 python
tools/t_pe_prof.py`, then `python tools/nsys_names.py $DIR 36`) and compare kernel by kernel; an A/B of whole
passes cannot tell a gain in one kernel from a loss in another (s0029 / s0031).

## Capacity (extension; findings F26)
- 128 slots (S_NSLOT, engine/sched.h and engine/api.py), admission by priority from the ring window, and a page store
  of 2439 pages (sized from the free memory after all other allocations; 7.7 GB per card stay free: tools/t_mem.py).
  3000 pages would be worth 27 -> 23 late requests of 2120 at 8 req/s over 300 s (simulated); a page is 28 MB per
  card (K and V of 48 layers, convolution snapshots).
- The pressure rule's margin is 2 s while 32 or more sequences decode (S_LOAD_ND), 0.3 s below (s0042: 775 -> 778 of
  780). The two that stay late are 32,768-token prompts in the heaviest stretch (65 - 70 s); they need prompt rows 20%
  cheaper (simulated), not another order of service.
- Order of service (s0043): shortest-first among the prompts still in time, except that one prompt per step goes
  first when it would miss in that order and serving it first costs nobody (S_PM1 / S_PM2, 0.3 s each; counter
  `promotions`). Earlier triggers save more requests at 8 req/s over 300 s (17 -> 10 late of 2120, simulated) and
  cost 2 req/s its TTFT p90 (1.9 -> 3.4 s): a trigger that depends on the load is the open question there.
- The scheduler's first-token clock starts when a step first sees the request (up to a pass after submit()); the
  previous step's time would be the honest stamp (simulated: 0.05 s for the request at the limit; not built).
- The ring of waiting requests (S_RING 1024) fills at 10 req/s behind requests given up (admitted ones stay in the
  window until the tail passes them): 4096 entries give 103 -> 86 late of 2608 (simulated, s0042 scheduler); needs
  engine/sched.h and engine/api.py (S_RING, the shared structure) together.
- To ask "what if" at a rate: `. tools/simenv.sh; /tmp/sched_sim tools/tr8_300.txt 2156 3584 2560 0.0160 16384`
  (traces tr8_120 / tr8_300 / tr2_300 in tools/; the model in the environment is fitted to s0036 at 8 req/s and
  predicted s0037 / s0038 and the 300 s runs to the third digit; at ~120 decoding sequences, 10 req/s, it is
  optimistic: 0.946 for a measured 0.891). It prints the drain time after the last arrival: the driver allows 90 s,
  and past ~12 req/s no order of service keeps that.
- What a decode row costs in a pass that carries prompts: 0.11 ms at position 8192 (profile: +76 us per layer in the
  attention kernel for 64 rows = 16 us per (row, KV head, page) unit with 6 of 128 q rows in use; everything else
  ~0). 1290 decode rows/s at 8 req/s are ~9% of the engine's time.

## What is still worse than the general engine, and why
- TTFT p99 (4.02 s against 2.2 s): the 26k-32k-token prompts of a burst of 209k new prompt tokens. A pass costs
  ~13 ms + 17.7 us per row and every decoding request needs a token per pass, so with a mean gap of 90 ms the prompt
  throughput is at most 48k rows/s (the engine: 46.5k in 3584-row passes). Halving it needs half the cost per row.
- Time between tokens p50 (25 ms against 10 ms): a decode-only pass is 11.3 ms at one row (48 layers x 23 kernels,
  each bound by a serial chain or a latency), +0.16 ms per row of the replay (+0.24 - 0.36 for rows with unrelated
  tokens: ~8 experts of its own per row), and a request's gaps include the passes of the burst it arrived in.
- Long runs: the page store (2156 pages, 27.9 MB per page and card with the convolution snapshots) turns over in
  ~2 minutes of this trace: 9% of the missed blocks of a 20-minute replay are recomputations.

## Open, by expected effect
- (extension) Global attention at long contexts: a unit copies its K and V half pages (131 KB per half) for every
  group of 21 rows; one copy per page and several groups would save about a third of the far units' time (~4% of a
  pass at position 24,000, more for 32k prompts, which are the late requests at 8 req/s). Needs a record per half
  page (the two halves of a page cannot both stay in a block's shared memory) and a fold over half pages: a new
  rounding order, and by estimate most of the gain is lost again (the fold would read twice as much: +2.8 ms against
  -3.5 ms at position 24,000). Only worth it together with a smaller record.
- (extension) Median time between tokens under load: `-DS_BUSY_ND=48 -DS_BUSY_ROWS=2048` was measured (59 -> 54 ms at
  8 req/s for one request of 780); the code is not in the engine, the numbers are in the journal.
- Fetches between cards: full data parallelism for attention (a sequence's K/V on one card, q/k/v for all heads on the
  home card) would remove the attention-input fetch and the gather (~4.5 ms + their latency per pass) but needs
  per-card page stores, homing of requests and copying of cached pages between cards. q/k/v alone on the home card:
  ~2% (journal, 15.5 - 16.1 h).
- Rows of small passes with many different experts (0.24 - 0.36 ms per row): every expert tile costs 3888 MMAs
  whatever its columns. An MMA with the weights as the B operand (N = 256 weight rows) and the pairs as A would
  halve the operations; whether two weight units and their scale atoms concatenate into one 256-column operand is
  not known (tools/bench/tc5.cu has the single-unit layouts).
- cudaMemcpyPeer reached 560-740 GB/s in tools/bench/p2p.cu against ~300-420 GB/s for kernel reads, but its sizes
  must be known when the pass is enqueued (the device loop enqueues blindly) and its sources contiguous.
- Kernel boundaries of a small pass: 23 kernels per layer at >= 1.4 us each (post_mlp + next mix are fused since
  s0029). The two convolution-state kernels cannot simply be merged (the snapshot job reads the slot state that
  the sequence job overwrites); the two gather launches of the fixed program differ in threads per block.
- The scheduler kernel hashes a burst's prompts on one thread (3-5 ms once per burst): per-block hashes by all
  threads and a chain over them would make it ~0.1 ms.
- The bias GEMM is all per-unit overhead (one K step per unit, 17 us per unit); activation + NVFP4 quantization
  (4.3 ms per pass) is ~100 instructions per element pair (two tanh, one exp, a division).

## Measured dead ends (do not repeat without a new idea)
See the journal entries "experiments after s0019", "what the attention kernel's time is", "where the two remaining
losses are bound", "small-pass and pass-size experiments" and findings F15, F19, F22: `__expf` in the softmax, a
second launch program for small passes, two expert row groups per block at 5-8 experts, largest passes while nobody
decodes, expert tiles of 256 columns, a hot expert on every card, pass sizes aligned to column tiles, row loops
in the prep and fold kernels and in the routing fetch (F24, s0031), one barrier per round in the routing
kernel, the MoE-input quantization moved into the post-attention kernel, bulk copies (`cp.async.bulk`) for the
expert sum's fetch (F25); and
256-column dense tiles at full passes, scale-copy sharing, alternating scale columns, staged or register-held mix
sources, 28 softmax warps, pipelined attention, six row groups for small expert tiles, K halves in one block, interior
softmax path, batched id loads beyond the assignment kernel's, loads-first fetch kernels, smaller normal passes.
