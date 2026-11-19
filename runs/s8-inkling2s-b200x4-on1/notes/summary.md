# Summary of run s8-inkling2s-b200x4-on1 (track A): from the decode megakernel to a serving engine

(The summary of the run that built the inherited decode megakernel is notes/s7_summary.md; its plan notes/s7_plan.md.)

**Final state (end of the extension, 32 h): engine/ = s0043.** At 8 req/s 778 of 780 requests within the service
levels (`slo_share_r8` 0.9974, the same as s0042, with one request more than the s0042 source on the same pod; SGLang
0.823 over HTTP, 0.854 on the board), at 2 req/s 189 of 189 with the stall and tail figures of s0035. The section "Extension" below has the path from s0035 (0.336 at 8 req/s over HTTP) and the
figures at every rate; the section that follows here is the state at the end of the first 24 hours (s0035).

## Result at the end of the first 24 hours (s0035)
- Headline (share of the replay's 189 requests within the service levels, nominal 2 req/s): **1.000 (s0019 - s0035)**:
  all 189; SGLang on the same requests 0.825, the run's target line 0.925. engine/ = s0035.
- The five figures next to the general engine (s0035 / SGLang): TTFT p50 161 ms / 625 ms, TTFT p99 4.02 s / 2.2 s,
  mean time between a request's tokens p50 25 ms / 10 ms, p99 72 ms / 788 ms, longest stall ~87 ms / 2.4 s.
  Better on share, TTFT p50, time between tokens p99 and stall; worse on TTFT p99 and time between tokens p50.
  Both losses are bound by the cost of a pass (journal, "where the two remaining losses are bound"): a pass over the
  weights costs 13.5 ms + 17.7 us per row, the decoding requests need a token per pass, so the prompt throughput
  under a 90 ms mean gap is at most 48k rows/s (the engine: 46.5k) and the largest burst (209k new prompt tokens at
  once) needs 4.5 s; a decode-only pass is 12 ms + 0.16 per row.
- Load sweep with the replay tool (120 s of the trace at each nominal rate; s0033 source, the rows at 0.5, 1 and 3
  req/s with the s0035 source; requests / within the service levels / TTFT p50, p99, max / mean time between tokens
  p50, p99 / longest gap):

  | rate | requests | within | TTFT p50 / p99 / max | time between tokens p50 / p99 | longest gap |
  |---|---|---|---|---|---|
  | 0.5 req/s | 53 | 100% | 0.23 / 4.66 / 4.66 s | 20.7 / 70.3 ms | 75 ms |
  | 1 req/s | 103 | 100% | 0.21 / 4.01 / 4.66 s | 21.2 / 70.7 ms | 76 ms |
  | 2 req/s | 189 | 100% | 0.16 / 4.05 / 4.63 s | 25.4 / 72.8 ms | 87 ms |
  | 3 req/s | 284 | 100% | 0.16 / 4.00 / 4.69 s | 34.5 / 71.6 ms | 87 ms |
  | 4 req/s | 368 | 100% | 0.12 / 3.43 / 4.64 s | 30.7 / 73.6 ms | 88 ms |
  | 5 req/s | 483 | 100% | 0.16 / 3.50 / 4.90 s | 38.6 / 73.5 ms | 88 ms |
  | 6 req/s | 585 | 99.8% | 0.17 / 3.80 / 5.05 s | 48.0 / 75.6 ms | 89 ms |
  | 8 req/s | 780 | 83% | 1.58 / 7.54 / 9.66 s | 51.4 / 77.6 ms | 90 ms |

  SGLang (the board's reference runs): 0.5 req/s 94%, TTFT 0.9 / 2.6 s, 8 / 622 ms, stall 2.4 s; 2 req/s 82%, 0.6 /
  2.2 s, 10 / 788 ms, 2.4 s. The engine holds every request within the service levels up to 2.5 times the headline
  rate and 584 of 585 at three times (one first token 49 ms late); past its capacity (8 req/s) only first tokens
  are late: every request still finishes and no request's mean gap exceeds 80 ms. The largest burst of the window
  (22 requests, 209k new prompt tokens at once) is in every run, so the longest first token is ~4.65 s at every
  rate up to 4 req/s. (The same sweep with the s0024 source: 89% at 6 req/s and 47% at 8; with s0028 99.7% and 68%:
  at those rates the page store is full early, and until s0026 the scheduler kernel walked the whole cache table
  for every evicted page.)
- Longer windows: 400 s of the trace 643 of 643 within the service levels (s0035 source: TTFT p99 3.33 s, time
  between tokens p50 35.3 ms); 20 minutes (s0033 source: TTFT p99 3.31 s, time between tokens p50 49 ms) 2119 of 2120 (one first token at 5.47 s), 18,697 evictions, 1901
  recomputations, every ring wrapped, probes 12 of 12.
- Every submission since s0002 passes every gate with all sequences identical: schedule 12/12, online_vs_step 8/8 +
  8/8, kv_conservation, liveness; offline rel L2 0.078 / 0.081 (limit 0.179), batch gate bitwise equal.
- Robustness beyond the evaluation (tools/t_stress.py, s0024): 150 requests at once with a page store of 700 or 150
  pages (up to 1582 evictions, 472 recomputations), lengths at every limit up to 34,815 + 1 and 34,304 + 512
  positions, 1500 requests at once (more than the request ring holds), drop_cache while the crowd runs, a host that
  does not poll for 47 s (s0025), close() with 40 requests open: every request finishes with its tokens, sampled
  requests equal the same request alone, free == total afterwards. This test found the one liveness bug the replay
  could not show (prompts of more than 33,279 tokens, s0024). The scheduler code itself runs under a fuzz harness on
  the host (tools/sched_fuzz.cpp: 24,000 random scenarios with its books recomputed after every step; it finds
  eight seeded bugs, among them that of s0024), the host's rings under tools/t_ring.py.
- Steady decode (offline, in process): 11.0 ms per step at B=1, 13.7 ms at B=8. The inherited megakernel did 3.3 /
  5.4 ms: it is kept (engine/ink.cu, `MK_ENGINE=megakernel`, `MK_BUILD_MEGAKERNEL=1`) but cannot serve: see below.
- The serving loop runs on the devices (scheduler kernel on card 0, 0 host syncs per pass); the host writes prompts
  into page-locked memory, reads tokens from it and enqueues the same pass program again and again; submit() and
  poll() make no call into a card.

## Submissions
| id | share | what |
|---|---|---|
| s0001 | incorrect (expected) | inherited engine: no idle / kv_pages / drop_cache |
| s0002 | 0.085 | pass engine: rows of all requests through one tcgen05 pass over the weights, FIFO |
| s0003 | 0.868 | scheduler: passes sized by time, shortest-remaining-first within the 5 s class |
| s0004 | 0.889 | pass length from the decoding requests' gap credit; cheaper small passes |
| s0005 | 0.921 | experts move between cards with the traffic; expert outputs summed on one card |
| s0006 | 0.926 | tcgen05 chains: one issuing thread per row group |
| s0007 | 0.937 | dense GEMMs: four row groups per block on large passes |
| s0008 | 0.926 | no host scheduler: admission, page store, prefix cache, pass composition on the device |
| s0009 | 0.931 | expert placement decided on the device too |
| s0010 | 0.958 | attention on tensor cores (scores, softmax from tensor memory, value product in one block) |
| s0011 | 0.963 | GEMM chains: fewer, larger operand copies per stage |
| s0012 | 0.979 | a home card per pass row: residual, mixes, o_proj, router once instead of on every card |
| s0013 | 0.984 | passes of up to 2560 rows |
| s0014 | 0.984 | attention: far keys without bias or mask, fold with 16-byte loads |
| s0015 | 0.984 | submit() / poll() without a call into the cards: prompts copied in by the scheduler kernel |
| s0016 | 0.984 | small passes: three expert row groups per block when one per block needs more rounds |
| s0017 | 0.995 | GEMM read-out by 16 warps instead of 4 |
| s0018 | 0.995 | assignment kernel: 16-byte id loads, four rows ahead |
| s0019 | 1.000 | largest passes (3584 rows) only while a waiting prompt would miss its first-token time |
| s0020 | 1.000 | recomputations counted on the device; source comments |
| s0021 | 1.000 | softmax: first-pass scores kept in tensor memory, zeros through exp |
| s0022 | 1.000 | pressure rule with 0.3 s of slack (longest first token 4.9 -> 4.6 s) |
| s0023 | 1.000 | dense GEMMs sum their K range as two halves: on two SMs in small passes (decode pass 13.2 -> 11.3 ms) |
| s0024 | 1.000 | scheduler: key table overrun for prompts of more than 33,279 tokens (found by a stress test); submit guards |
| s0025 | 1.000 | the ring of token records cannot lap a host that does not poll; token ids checked at submit() |
| s0026 | 1.000 | scheduler kernel: averages, swap search and row tables by all threads, eviction without a table scan (prompt passes -2 ms; 400 s replay 642 -> 643 of 643) |
| s0027 | 1.000 | cache table keeps its empty slots over hours of evictions (3.3 simulated hours: 0 -> 13,876 empty slots); a waiting prompt is hashed once |
| s0028 | 1.000 | unembedding of small passes in the large-pass form (decode pass -0.12 ms; steady decode 11.07 / 13.83 ms) |
| s0029 | 1.000 | small passes: end of a layer fused with the next mix, fetch kernels loop over rows (decode pass -0.2 ms, largest prompt passes +0.4 ms) |
| s0030 | 1.000 | step() / generate() after submit() / poll() on one engine: the device loop is stopped and restarted (was: two composers of passes) |
| s0031 | 1.000 | routing fetch back to one block per row (its row loop had cost large passes 1.2 ms): 2560 rows 61.7 -> 60.6 ms, TTFT p99 4.16 -> 4.10 s |
| s0032 | 1.000 | routing fetch: a row's scales, ids, weights and outer scale as one contiguous piece (67 -> 53 us per layer; 2560 rows 61.0 -> 60.4 ms, TTFT p99 4.07 s) |
| s0033 | 1.000 | convolution-state fetch with 576 threads per row (two remote loads in series instead of six): prompt passes -0.3 ms, TTFT p99 4.04 s |
| s0034 | 1.000 | attention-output gather: one launch with 768 threads for every pass size (one remote load per thread): prompt passes -0.4 ms, TTFT p99 4.03 s |
| s0035 | 1.000 | routing fetch: the packed piece read by threads that carry no codes (one remote load per thread): 3584 rows 77.5 -> 77.25 ms |

## What the engine is (engine/pe.cu, engine/sched.h, engine/api.py)
- A **pass** = up to 2560 rows, 3584 while a waiting prompt would otherwise miss its first-token time (a row = one
  position of one sequence: every decoding sequence's next token plus chunks of prompts) through the 48 layers on
  all four cards. Attention is split by heads (12 q / 2 KV heads per
  card), experts by card (96 per card, moved between cards by weight swaps as the traffic changes).
- **One arithmetic** for every pass size: a row's result is bitwise independent of batching, chunking, scheduling,
  cache hits and of which card is its home (tests: stepwise == chunked == chunks of 7 / 64 / 100 / 501).
- GEMMs: tcgen05 with the checkpoint's formats (bf16 kind::f16; NVFP4 kind::mxf4nvf4 with block scales; MXFP8
  kind::mxf8f6f4), operands streamed through a ring in shared memory by bulk copies.
- Attention: one block per (21 rows x 6 q heads, KV head, page of 512 keys): Q K^T on tensor cores with the K page
  stored as an operand, softmax by 16 warps from tensor memory, P V with the V page as operand.
- K/V in pages of 512 positions with reference counts, a prefix cache (hash chain of 512-token blocks, LRU
  eviction, in-flight dedup, hits picked up at page boundaries), short-convolution state snapshots per page.
- Scheduler (engine/sched.h, runs in `k_sched` on card 0 and, compiled for the host, in tools/sched_sim.cpp):
  passes sized by a cost model, prompts ordered by (can still answer within 5 s, shortest remaining first), a
  pass as long as the decoding request with the least gap credit allows.
- Each pass row has a **home card** (the cards own quarters of a pass): residual state, mixes, norms, o_proj,
  router, routing only there; the others fetch what attention and the experts need. Passes of up to 64 rows are
  replicated instead.

## What worked, in the order of its effect on the share
1. Time-sized mixed passes with decode rows riding prompt passes, SJF within the deadline class, gap credit
   (0.085 -> 0.889).
2. Removing work that every card repeated (home card per row): 72.7 -> 59.5 ms per 2048 rows (0.963 -> 0.979).
3. Attention on tensor cores: attention 20.6 -> 8.1 ms per card and pass; context term 0.84 -> 0.30 us per row and
   1000 positions (0.931 -> 0.958).
4. Expert placement following the traffic (pairs per card 1.48x -> 1.07x apart) and summing a row's expert outputs
   in one place (0.889 -> 0.921).
5. tcgen05 chain engineering: 128-column tiles with shared B units, one issuing thread per row group, stages of
   several K steps (2048-row pass 139 -> 73 ms over s0003-s0011).
6. The pressure rule (largest passes only while a waiting prompt would miss its first-token time, with 0.3 s of
   slack): 0.995 -> 1.000 (s0019, s0022).

After the share was 1.000 (s0024 - s0033), what moved the other figures and the engine's reach:
- Tests the replay cannot be: a crowd with a small page store, lengths at every limit, a host that stops reading, the
  two APIs in turns, a fuzz of the scheduler code with its books recomputed after every step, replays of 400 s and
  20 minutes. They found four defects no evaluation had shown (F22): a key table one entry short for prompts over
  33,279 tokens (requests stood forever), a token ring a slow host could lap, a cache table that loses its empty
  slots over hours, offline calls composing passes next to the device loop.
- The scheduler kernel's own time (F23): 2.1 ms of every prompt pass on one thread, and a walk over the cache
  table per evicted page. In parallel / without the walk: capacity at 6 req/s 89% -> 99.8% of requests in time,
  400 s replay 642 -> 643 of 643, TTFT p99 at the headline rate 4.28 -> 4.15 s.
- Kernel boundaries of small passes and the exchanges between cards, kernel by kernel in profiles of two builds
  (F24, F25): steady decode 11.2 -> 11.0 ms per step, a 2560-row pass 61.4 -> 59.5 ms, TTFT p99 4.15 -> 4.02 s, time
  between tokens p50 26.0 -> 25.2 ms.

## What did not work (measured)
- The inherited megakernel as a serving engine: two code paths with different rounding (it has a batch-1 kernel
  and a batched kernel) diverge within ~8 generated tokens; the schedule gate needs one arithmetic.
- Attention as three kernels (score GEMM, softmax kernel, value GEMM) through global memory: correct, slower
  (prev/tc_attention). A softmax with four warps per SM: correct, not faster (16 warps are).
- More threads per row for small passes (another reduction order): breaks bitwise equality.
- L2 prefetch of weights; residual-mix sources kept in registers or staged in shared memory; B scales copied once
  per stage; 256-column dense tiles at 2048 rows (helps below 1024 rows only); a larger row cap before the home-card
  change (q/k/v and o_proj needed a second round of blocks above 2105 rows).
- Reading another card's data in scattered 16-byte pieces (tile layout): 80 GB/s; row-contiguous: ~300 GB/s.
- After s0023 (journal, findings F22 - F25): the fast exponential in the softmax, a second launch program for small
  passes, two expert row groups per block, largest passes while nobody decodes, expert tiles of 256 columns and a
  hot expert on every card (estimates), row loops in the prep / fold kernels and in the routing fetch, one barrier
  per round in the routing kernel, the MoE-input quantization in the post-attention kernel, bulk copies for the
  expert sum's fetch.

## Where the time of a 2560-row pass goes now (s0035; per card, nsys, one sequence at position 4096, no rebalancing: 61.0 ms under nsys, ~55 in the replay)
expert GEMMs 13.0 ms, sum of a row's expert outputs on its home card 8.6 (fetch over the links ~4, waiting for the
other cards' experts: 4 here, ~1.4 in the replay where the placement follows the traffic), attention 7.2, q/k/v 4.7,
activation + quantization 4.3, attention-input fetch 3.25, bias GEMM 2.8, o_proj 2.4, end of layer + next mix 1.9,
post_attn 1.9, router 1.75, route 1.5, gather 1.3, fold 1.3, routing fetch 1.15, prep 1.0, assign 0.9, packing 0.8,
convolution state 0.55; the scheduler kernel 0.13. Expert GEMMs read every expert of the model once per pass (376
GB; 11.8 ms at the node's peak rate): they are at the memory floor; the two large fetches (expert outputs, attention
input) are at the links' rate. A 1-row pass: 11.3 ms (48 layers x 22 kernels).

## At the end of the run
- **Best submission: s0035** (= engine/, valid, every gate with all sequences identical): share of the replay's 189
  requests within the service levels **1.000** (SGLang 0.825, target line 0.925); TTFT p50 161 ms / p99 4.02 s, mean
  time between a request's tokens p50 25.2 ms / p99 72 ms, longest stall 87 ms; steady decode 10.98 ms per step at
  B=1, 13.74 at B=8. The share has been 1.000 since s0019; s0024 - s0035 hold it and move the other figures.
- **What worked**: the list above ("What worked", 1-6, and "After the share was 1.000"). In one line each: mixed
  passes sized by time with shortest-first prompts and gap credit; a home card per row; attention on tensor cores;
  expert placement that follows the traffic; the pressure rule; then tests the replay cannot be (four defects), the
  scheduler kernel's serial work done in parallel, and the exchanges between cards trimmed kernel by kernel.
- **What did not**: the list above ("What did not work") and notes/plan.md ("Measured dead ends"). The two figures
  still worse than the general engine, TTFT p99 (4.02 s against 2.2 s) and time between tokens p50 (25 ms against
  10 ms), did not yield to anything tried: both are bound by the cost of a pass (about 13 ms + 17.7 us per row; 11.3
  ms for a one-row pass), and steady decode stays 2.3x slower than SGLang's and 3.3x slower than the inherited
  megakernel's, which cannot serve.
- **What I would try next** (details and estimates in notes/plan.md, "Open"):
  1. Full data parallelism for attention: a sequence's K/V on one card and q/k/v for all heads on the row's home
     card. Removes the attention-input fetch and the gather (~4.5 ms of a 2560-row pass, both at the links' rate
     now); needs per-card page stores and homing of requests. The largest step left for prompt throughput.
  2. A small-pass path with fewer kernel boundaries: 22 kernels per layer at >= 1.4 us each plus serial tensor-core
     chains make the 11.3 ms decode pass; a persistent per-layer kernel with the same arithmetic (flags instead of
     launches) is the only route to a decode pass well under 10 ms without a second arithmetic.
  3. Expert tiles with the weights as the B operand (256 weight rows per MMA) for passes with many experts and few
     columns: would halve the tensor-core operations of decode rows with unrelated tokens (0.24 - 0.36 ms per row);
     whether two weight units and their scale atoms concatenate into one operand has to be measured first.
  4. Expert placement that also balances the number of active experts per card in decode passes (the expert
     exchange waits ~20 us per layer for the card with the most experts).
  5. Refit the simulators' pass model to the s0035 engine before any further tuning of the scheduling rules, and a
     larger page store or cheaper cached pages for long runs (9% of missed blocks are recomputations after 20 min).

## Extension (2026-11-15/16, +8 agent-hours): capacity at 8 req/s
- The operator measured s0035 over HTTP for 300 s: 100% within the service levels at 1-2 req/s, 99.9% at 4, **33.6% at
  8 req/s** (SGLang 82.3%). The score became the share at 8 req/s (`slo_share_r8`, 120 s in process in the evaluator).
- **Where the capacity went** (findings F26): not into the kernels first. The scheduler had 64 slots and admitted in
  arrival order; at 8 req/s ~65-75 sequences decode at any moment (1290 output tokens/s, one token per sequence and
  pass, passes of 51-60 ms), so every request waited behind the head of the ring for a slot.
- Submissions of the extension (all valid; 2 req/s stays 1.000 with TTFT p99 4.03 s, time between tokens 25 / 64 ms,
  longest gap 87 ms):

  | id | slo_share_r8 (120 s) | what |
  |---|---|---|
  | s0036 | 0.836 | the audit's two defects: per-slot buffers sized by the scheduler's slots at any max_batch; prefix-cache matches verified token by token (and by the parent link) |
  | s0037 | 0.990 | 128 sequences in flight instead of 64 |
  | s0038 | 0.990 | admission by priority: requests that can still answer in time first, cheapest first; the others when there is room |
  | s0039 | 0.990 | largest passes for a prompt nobody waits behind (a lone 32,000-token prompt 808 -> 752 ms) |
  | s0040 | 0.990 | audit doubts: a failed page allocation surfaces as an error; an open request id is refused |
  | s0041 | 0.994 | page store sized from the memory really free: 2149 -> 2439 pages (7.7 GB per card stay free) |
  | s0042 | **0.997** | the largest passes 2 s (not 0.3 s) before a projected miss while 32 or more sequences decode |
  | s0043 | **0.997** | a prompt goes ahead of the shortest-first order when that saves it and costs nobody (A/B on one pod: 777 -> 778 of 780) |

- 300 s at 8 req/s in process (the operator's window): s0035 0.336 (over HTTP) -> 128 slots 0.947 -> priority admission
  0.982 -> larger page store 0.987 -> load-dependent pressure margin **0.991** (2101 of 2120; TTFT p50 0.18 s, p90 1.5 s,
  p99 4.9 s; time between tokens p50 60 / p99 82 ms; longest gap 91 ms). Schedule probes at 8 req/s over 300 s: 12 of 12 identical. Over 15 minutes at 8 req/s
  (6471 requests, s0042 source): 0.992 (6418), every request finishes, TTFT p99 4.8 s, time between tokens p50 61 / p99
  80 ms, longest gap 94 ms: the share does not decay. 6 req/s 0.997; 10 req/s 0.956 on the s0042 source (0.89 with s0038; arrival order would give 0.17
  in the simulator).
  All of these are in process; the HTTP calibre could not be run from the pod (no aiohttp in its Python).
- The tool that made this quick: tools/sched_sim.cpp runs the scheduler's own code on a virtual clock; with a pass
  model fitted to one measured result it predicted the next three device results to the third digit.
- Still true: median time between tokens (58 ms at 8 req/s, 25 ms at 2; SGLang 17.5 / 9-10 ms), TTFT p99, steady
  decode (11 ms per token) and a 32,000-token prompt's first token (0.76 s against 0.41 s) are bound by the cost of a
  pass, as before; kernel speed now buys the tail at 8 req/s (prompt-row cost -10%: 38 -> 21 late of 2120), not the
  share.

### The operator's matrix, in process (300 s of the trace per rate; 1 req/s on the s0041 source, the others s0042)
| rate | within service levels | TTFT p50 / p99 | time between tokens p50 / p99 | longest stall |
|---|---|---|---|---|
| 1 req/s | 238 of 238 (SGLang over HTTP 83.5%) | 0.18 / 3.9 s (0.55 / 2.2 s) | 28 / 70 ms (9 / 697 ms) | 0.08 s (2.3 s) |
| 2 req/s | 483 of 483 (85.4%) | 0.17 / 3.4 s (0.57 / 2.2 s) | 28 / 73 ms (9 / 704 ms) | 0.09 s (2.4 s) |
| 4 req/s | 1046 of 1046 (82.7%) | 0.13 / 3.4 s (0.62 / 2.2 s) | 46 / 74 ms (10 / 796 ms) | 0.09 s (2.8 s) |
| 6 req/s | 1593 of 1598 = 99.7% | 0.15 / 3.3 s | 55 / 76 ms | 0.09 s |
| 8 req/s | 2101 of 2120 = 99.1% (82.3%; s0035 33.6%) | 0.18 / 4.9 s (0.62 / 2.1 s) | 60 / 82 ms (17.5 / 787 ms) | 0.09 s (4.4 s) |
| 10 req/s | 2494 of 2608 = 95.6% | 0.28 / 110 s | 69 / 82 ms | 0.09 s |

(In parentheses the operator's SGLang figures over HTTP. At 1 - 4 req/s the in-process figures of this engine equal
the operator's HTTP figures for s0035 - 0.18 / 3.9 s, 0.17 / 3.4 s, 0.17 / 3.4 s - so the calibres agree there. At 8
req/s the TTFT p99 is now inside the service level; at 10 req/s the 4% the scheduler gives up on wait until the
arrivals end. s0042's rule changes nothing below 32 decoding sequences, which 2 req/s does not reach (27 in flight
at most over 300 s): the evaluator's 2 req/s figures are those of s0041. Of two runs at 2 req/s one had a longest
gap of 102 ms: a single poll() call of 89 ms in the unpinned driver process of my own run (journal); the evaluator's
runs have polls of at most 0.5 ms and a longest gap of 86 ms.)

### At the end of the extension
- **Best submission: s0043** (= engine/): `slo_share_r8` **0.9974** (778 of 780 requests at 8 req/s within the service
  levels, 120 s in process; SGLang 0.823 and s0035 0.336 in the operator's 300 s over HTTP); TTFT p50 177 ms / p90
  1.66 s / p99 4.35 s, time between tokens p50 59.7 / p99 81.3 ms, longest gap 91 ms. At 2 req/s 1.000 with TTFT p50
  176 ms / p99 4.02 s, time between tokens p50 25.1 / p99 58.6 ms, longest gap 87 ms. Steady decode 10.72 / 13.57
  ms. s0042 has the same number on the board (evaluated on another pod); on one pod the s0042 source gives 777 and
  s0043 778. Over 300 s at 8 req/s 0.991 and over 15 minutes 0.992 with the s0042 source, in process.
- **How firm the third digit is**: one request of the 780 (a 32,768-token prompt in the burst at 2.5 s) answers after
  4.95 s in the simulator, inside the 5 s on one pod and after 5.01 - 5.02 s on the next. Its burst needs 4.9 s at
  the largest passes: no order of service or margin moves it by more than 0.09 s. The score is 777 - 779 of 780
  depending on the pod; only a cheaper pass makes it firm.
- **What worked**: finding the limit before touching a kernel (the simulator with the scheduler's own code and one
  fitted pass model); 128 slots (0.336 -> 0.947 at 300 s); admission by priority (-> 0.982; graceful past capacity:
  0.89 at 10 req/s); the page store sized from measured free memory (-> 0.987); the largest passes earlier under
  decode load (-> 0.991; the TTFT p99 at 8 req/s over 300 s 9.7 -> 4.9 s); promotion ahead of the shortest-first
  order (one more request at 8 req/s, 6 -> 3 late of 1598 at 6 req/s simulated); the fuzz, which caught the priority
  admission's starvation on small stores before any device run; the audit's defects and two of its doubts fixed.
- **What did not**: everything aimed at the pass itself in these hours (64-cell tensor-memory reads, the far-key
  softmax without its maximum pass, home-card mode from 33 rows, the feasibility test with a realistic rate, other
  pass caps, refreshed cost-model constants): no gain; a shorter normal pass under load: median gap -7% for one
  request of the score. The engine's median time between tokens, TTFT p99, steady decode (11 ms per token) and the
  first token of a 32,000-token prompt (752 ms, from 808) are still 2-4x the general engine's: the cost of a pass.
- **Next** (notes/plan.md, "Capacity" and "Open"): the late requests at 8 req/s are the 32,768-token prompts at the
  end of their bursts, so prompt throughput at long positions (global attention: K/V operands copied for every
  group, records per page) is the lever for the last 0.3% (simulated: the two late requests of 780 need prompt rows
  20% cheaper; no order of service saves them); for the median gap and steady decode, a small-pass path
  with fewer kernel boundaries and the same arithmetic; a cheaper page (snapshots on one card, 2.1 of 28 MB) for
  long runs.
