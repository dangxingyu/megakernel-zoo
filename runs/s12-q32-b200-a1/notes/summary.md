# s12-q32-b200-a1: Qwen3-32B (BF16) on one B200 — final summary (board hour ~23.6)

## Final result
- **Best submission: s0041 (track B, exact 12-bit weights): 8948 us/token at B=1 and 9404 us/step at B=8 over HTTP =
  1.279x / 1.243x SGLang 0.5.23** (11440 / 11692); best B=8 s0038, 9390 (1.245x). `engine/` = s0041 exactly.
- **Best track A (BF16 weights as stored): s0040, 10012 / 10410 us = 1.143x / 1.123x** (best B=8: s0037, 10404).
- The tree (engine/TRACK = B) is s0040/s0041's source; `MK_TRACK=A` builds s0040's track-A engine.
- Both tracks pass the project's 1.10x claim at both batch sizes. All 36 submissions valid. Track B bit-identical to
  track A of the same source (tools/kit/bitcheck.py, B=1 and B=8); B=2..8 bitwise equal to B=1; all 31,983,534,080
  12-bit weights re-verified against the checkpoint on the final tree (0 mismatches).

## What worked (in order of effect)
1. Persistent cooperative megakernel: one launch per pass/cohort, 512 threads x 128 registers, weights pre-permuted
   into MMA fragment order, split-K in quads, cut tiles handed over through a sentinel slot (s0001-s0007).
2. Exact 12-bit weight storage for track B (per-row 16-exponent window, 3 words/lane/k-step, 3.3M escapes):
   48 GB instead of 64 GB per step (s0004 onwards).
3. Dataflow counters instead of grid barriers around attention; split grid barrier with the first weight batch
   requested before the wait, its state in shared memory (s0019, s0021: -120..-180 us).
4. Attention: one-warp-per-part softmax, 3 K blocks per round, two-half fold, DSMEM combine at B=7-8, the combine
   split over all blocks of a unit at B=1-6, K/V L2 prefetch at B<=2 (s0016-s0018, s0028).
5. Track B memory parallelism: 3 quads in flight with inputs read from shared memory just in time; cluster-split norm
   prep at B=7-8; packed bf16 norm arithmetic; parallel escape patching (s0024-s0026, s0032, s0034).
6. Runtime tuning measured on the serving workload's context mix (tests/wl.sh), e.g. s0036's attention cap and idle
   prefetch (-65 us in process at B=1).
7. Latency chains found with ordered timer stamps (extension, hours 24-27): the GEMM epilogue's critical warp issued the
   next op's prefetch (moved to the last warp, s0037); every thread did 64-bit divisions by the block count on the
   path after each barrier (32-bit, s0038: track B -60 / -47 us); the attention combine's arrival took two round trips
   (one, s0040).

## What did not work (journal.md has each with numbers)
Prefetch warps or per-warp bulk prefetch of the weight stream; more prefetch during attention; per-k-step RMSNorm
inside the 12-bit loop; register-carried barrier state in the large kernel (-8% streams); cp.async rings, L1-cached
or DSMEM-shared inputs for track B's o/down at B=8; narrowed block-reduction scans; many micro-changes that only moved
register allocation (the 7-8 sequence instance swings 30-40 us with any code compiled into it).

## What I would try next
- Paired-tile stream partition (one warp uses each input fragment for two tiles, both tracks): track B's o / down at
  B=8 are limited by the per-SM L2 read rate (~66 GB/s; weights ~43 + inputs ~28 GB/s). Estimated ~2.5% at B=8.
- The remaining latency chains at B=1 (tests/prof_att.py, prof_epi.py, prof_prep.py): attention ~11 us per layer
  (q/k/v readiness 1.5, prep 0.8, scores 0.9, softmax + P x V 2.0, arrival ~1, fold 1.7, signal to o 1.9); GEMM
  epilogue ~1.6-2.3 us after each stream (bounded by the cut-tile hand-over); norm prep ~1.8 us (its L2 reads slowed
  by the barrier-time prefetch, which still pays off overall).
- Track B's escapes (layers 1-5, ~50 us/step) need a different code: they are spread over every tile.

## Results (HTTP, the evaluator's client; us per token at B=1 / us per step at B=8)
| sub | track | what | B=1 | B=8 | vs SGLang (11440 / 11692) |
|---|---|---|---|---|---|
| s0001 | A | own persistent megakernel v1 (1024 threads x 64 regs) | 11488 | 11763 | 0.996x / 0.994x |
| s0002 | A | 512-thread blocks (128 regs), split attention | 10631 | 11480 | 1.076x / 1.018x |
| s0003 | A | dataflow counters q/k/v -> attention -> o | 10586 | 11290 | 1.081x / 1.036x |
| s0004 | B | exact 12-bit weights (bit-identical to A) | 9625 | 10407 | 1.189x / 1.124x |
| s0005 | A | split in quads (shared with B: bit-identity) | 10489 | 11175 | 1.091x / 1.046x |
| s0006 | A | RMSNorm per k-step in the stream at B>=3 | 10522 | 10994 | 1.087x / 1.063x |
| s0007 | A | cut tiles: sentinel hand-over | 10334 | 10833 | 1.107x / 1.079x |
| s0008 | B | track B of s0007 | 9467 | 10277 | 1.208x / 1.138x |
| s0009 | A | B>=7 DSMEM attention combine, 2 kernel instances | 10323 | 10768 | 1.108x / 1.086x |
| s0010 | B | track B of s0009 (large instance spills) | 9444 | 10356 | 1.211x / 1.129x |
| s0011 | B | track B on the small instance at all B | 9453 | 10265 | 1.210x / 1.139x |
| s0012 | B | escapes patched via smem, last-warp prefetch | 9402 | 10049 | 1.217x / 1.163x |
| s0013 | A | three kernel instances, per-warp argmax keys | 10351 | 10773 | 1.105x / 1.085x |
| s0014 | A | final source of hour 7 (cohort-tagged tokens, 20 ms gather) | 10354 | 10766 | 1.105x / 1.086x |
| s0015 | B | track B of s0014 | 9402 | 10071 | 1.217x / 1.161x |
| s0016 | A | K/V bulk L2 prefetch before the q/k/v wait (B<=2) | 10299 | 10763 | 1.111x / 1.086x |
| s0017 | A | B=8 attention: one warp per part softmax, 3 K blocks per warp round | 10304 | 10742 | 1.110x / 1.088x |
| s0018 | A | combine in two halves (cluster blocks fold in parallel) | 10344 | 10715 | 1.106x / 1.091x |
| s0019 | A | split grid barrier, first weight batch before the wait (B<=2) | 10230 | 10718 | 1.118x / 1.091x |
| s0020 | B | track B of s0019 | 9301 | 9943 | 1.230x / 1.176x |
| s0021 | A | split barrier in every instance, its state in shared memory | 10284 | 10539 | 1.112x / 1.109x |
| s0022 | B | track B of s0021, 7-8 sequences on the cluster-combine instance | 9336 | 9918 | 1.225x / 1.179x |
| s0023 | A | first weight batch also across the norm prep (B<=2) | 10265 | 10535 | 1.114x / 1.110x |
| s0024 | B | 3 quads in flight per warp in the norm GEMMs (inputs from smem just in time) | 9250 | 9780 | 1.237x / 1.196x |
| s0025 | B | B=1: o / down input column staged in smem, 3 quads in flight | 9197 | 9840 | 1.244x / 1.188x |
| s0026 | B | norm prep in packed bf16 arithmetic (same bits) | 9156 | 9714 | 1.249x / 1.204x |
| s0027 | A | track A build of s0026's source | 10249 | 10537 | 1.116x / 1.110x |
| s0028 | A | attention combine split over all blocks of a unit (1-6 sequences) | 10141 | 10533 | 1.128x / 1.110x |
| s0029 | B | track B of s0028 | 9063 | 9703 | 1.262x / 1.205x |
| s0030 | A | attention prep constants before the q/k/v wait (the tree) | 10131 | 10528 | 1.129x / 1.111x |
| s0031 | B | track B of s0030 | 9049 | 9683 | 1.264x / 1.207x |
| s0032 | B | 7-8 sequences: norm prep split over the cluster pair (DSMEM) | 9035 | 9495 | 1.266x / 1.231x |
| s0033 | A | s0032's source; 7-8 instance allocation (unused prep path with the cluster prep) | **10121** | **10440** | **1.130x / 1.120x** |
| s0034 | B | escapes of a quad patched by their lanes in parallel | 9016 | **9433** | 1.269x / **1.240x** |
| s0035 | B | at most 10 blocks per attention unit at B=1 (in process -29, HTTP +9) | 9025 | 9449 | 1.268x / 1.237x |
| s0036 | B | idle blocks prefetch 128 KB of o during attention | 8995 | 9439 | 1.272x / 1.239x |
| s0037 | A | next op's prefetch from the last warp (warp 0 finishes the cut tile) | 10083 | **10404** | 1.135x / **1.124x** |
| s0038 | B | GEMM block ranges / prefetch plans with 32-bit divisions | 8956 | **9390** | 1.277x / **1.245x** |
| s0039 | A | track A of s0038's source | 10061 | 10407 | 1.137x / 1.123x |
| s0040 | A | attention combine arrival: release + poll of a cumulative count | **10012** | 10410 | **1.143x** / 1.123x |
| s0041 | B | track B of s0040's source (= engine/) | **8948** | 9404 | **1.279x** / 1.243x |

Both tracks pass the project's 1.10x claim at both batch sizes: track A with s0033 (1.130x / 1.120x, the source in the tree), track B with
s0036 (1.272x / 1.239x; B=8 best in s0034, 1.240x); the tree (engine/TRACK = B) is s0036: s0034's kernel with two
runtime defaults changed for track B (attention cap 10, idle prefetch 128 KB). The tree's track-A build is the same library as s0033's
(the later changes are in the 12-bit path only; identical spills). All submissions valid; every track-B result is bit-identical to the track-A build of its source
(tools/kit/bitcheck.py, 1152 positions, B=1 and B=8).

## The engine (engine/mk.cu, engine/api.py, engine/w12.py)
- One cooperative persistent kernel per pass / per served cohort (generate loop, argmax and token feedback on the
  device, tokens streamed to host-mapped memory for submit/poll). 152 blocks x 512 threads (128 registers).
- Per layer: q/k/v GEMM -> attention -> o GEMM (+residual) -> gate|up GEMM (+SiLU*up) -> down GEMM (+residual).
  Grid barriers only o->gate|up, gate|up->down, down->q/k/v; q/k/v->attention and attention->o are dataflow
  counters (per kv group tile counts; per unit completions).
- GEMM: weights pre-permuted into m16n8k16 A-fragment order, each block a contiguous range of (tile, quad of 4
  k-steps), split over 16 warps; 8 x 16 B loads in flight per thread (ld.global.cs); partial sums reduced in warp
  order; tiles cut between blocks handed over through a sentinel slot (no atomics). The median block streams at
  the byte floor (gate|up 69.05 us vs 69.0 at 7.6 TB/s).
- RMSNorm: B<=2 normalise the residual into shared memory per block (1/rms from per-tile sums of squares written
  by the residual epilogues); B>=3 normalise per k-step in the stream (no prep), bit-identical.
- Attention: 64-position parts (a function of the context only: batch-invariant), unit (sequence, kv head) spread
  over nblk/(8B) blocks; tensor-core scores (Q as A, 8 heads in rows 0-7; K as B) and P x V (P straight from the
  score fragments); per-part softmax state combined by the last arriving block (global) or, at B=7-8 where a unit
  is exactly 2 blocks, through DSMEM in a cluster of 2 (sequential per-part fold everywhere).
- Grid barrier split in two: arrive at the end of an op, wait inside the next GEMM after each warp has requested its
  first 8 weight k-steps (the o projection likewise before waiting for attention); thread 0 keeps the barrier's L2
  prefetch range in shared memory (carried in registers it slowed the per-k-step norm streams ~8%).
- Track B at 7-8 sequences: the norm prep split over the cluster pair (each block reads and normalizes half of the
  residual, writes it into both blocks' shared memory over DSMEM): the prep was limited by each block's L2 reads.
- Attention combine at 1-6 sequences: the last block of a unit to arrive releases a per-unit epoch flag and every
  block of the unit folds 1024/nact of the (head, dim) outputs over all parts (the single combiner was bandwidth-
  limited: 4.2 us at 10 parts); the 7-8 instance keeps its cluster combine compiled unchanged.
- Attention rounds at B=8: softmax statistics one warp per 64-position part (4 lanes per head), 3 K blocks per warp
  round at 7-8 sequences, the combine in two halves (the two cluster blocks fold in parallel; the global combine
  uses the same split: B=1..8 bitwise equal). At 1-2 sequences the cached K/V is bulk-prefetched into L2 before the
  wait for this step's q/k/v.
- Three kernel instances (1-2, 3-6, 7-8 sequences): register allocation is the dominant source of variance; small
  code changes moved the large instance by hundreds of us (journal).
- Track B stream: the 12-bit loop needs more bytes in flight than one 2-quad batch per warp (streams were at 75-93%
  of their floor); in the norm GEMMs (and o / down at B=1, input column staged in shared memory) the inputs are read
  from shared memory just in time and 3 quads per warp are in flight (gate|up and down now 95-97% of the floor at B=1).
  HTTP at B=8 varies by ~100 us between evaluations of engines of the same structure (journal, s0025).
- Track B (engine/TRACK = B): every streamed matrix in an exact 12-bit code (notes/checkpoint.md): per-row 16-exponent
  window, rows scaled by 2^(128 - lo), layout with three words per lane and k-step decoded in ~12 ops, 3.33M escapes
  patched through a per-warp shared-memory buffer, a quad's escapes by their lanes in parallel. 47.98 GB instead of 63.97 GB; all 31,983,534,080 weights
  verified (MK_W12_VERIFY=1).

## What the checkpoint allows (notes/checkpoint.md)
No zeros, duplicates or structure; 10.54 bits/weight of entropy; a 16-exponent per-row window leaves 1.03e-4 of
weights outside (concentrated in layers 1-5's MLP, spread over rows); 11 bits would leave 1.7%.

## What did not work (journal.md, gap.md)
Hour 8-18: fused softmax into the P x V tasks, K/V prefetch at B>=3, more o prefetch during attention, per-warp
bulk L2 prefetch of the 12-bit stream, per-k-step norm in the 12-bit loop, split cluster output, register-carried
barrier state in the large instances (-8% on every norm stream), more loads in flight in track A (8 is right),
track B o / down inputs through a cp.async ring, through L1, or split over the cluster pair via DSMEM (all slower).
Prefetch warp walking the weights through L2 (deep queue delays every critical load), more/earlier L2 prefetch,
warp-distributed prefetch, dataflow gate|up -> down, on-the-fly norm at 64 registers, K/V preloads (spills), TMA ring
for K/V (no faster than register rounds in a microbenchmark), L1-cached GEMM inputs, evict_last K/V, cluster
combine at B=1 (2 blocks per unit), ptxas option search, smaller shared memory.

## Where the time goes (gap.md, final accounting)
Overheads over the byte floor (7.6 TB/s): track A ~1.7 ms (B=1) / ~2.0 ms (B=8), track B ~2.7 / ~3.2 ms. Per GEMM
op: slowest block's tail 1-2.6 us + epilogue 1.5-2 us + barrier 1.2 us; norm prep ~2 us (B=1); attention chain
~8.5 us/layer (B=1) / ~11 (B=8). The common limit behind the norm prep, the attention combine and track B's o /
down at B=8 is the per-SM L2 read rate: ~66 GB/s at most, ~20-40 GB/s beside the weight stream (bench/l2bw.cu).

## Checks (tests/)
check.py (vs HF, batch, generate), cmp32.py (vs HF float32), bitcheck (A vs B), binv.py (B=2..8 vs B=1 bitwise to
position 2039), stress.py (repetitions), robust.py (limits, serving), serve.py.
