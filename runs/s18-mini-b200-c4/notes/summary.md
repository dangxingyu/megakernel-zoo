# Run s18-mini-b200-c4 (track C, Inkling-2-Mini, 1x B200; direction: batch 1, fused) — summary

## Result (best: s0012)
- http_step_us_b1 **1260.0** us/token on the evaluator's prompts (start s0001: 1521.7; vLLM + DSpark 1544 -> 1.23x)
- natural_b1 **810.2** (start 1015.7; vLLM + DSpark 871 -> 1.075x)
- s0012 adds run s18-mini-b200-c3's 16-row verify library and draft rules (n-gram copies of 15 tokens in a copying
  regime) on top of this run's verify kernels, with this run's w2 bank-conflict fix applied to c3's library
  (V(16) 9.5 -> 7.9-8.4 ms) and this run's verify-cost table in c3's draft-length rule.
- b8 5075 / natural_b8 5470: no speculation at B >= 2 (the s17 a1 kernel; another run's direction)
- tokens per event 3.04 (evaluator) / 6.69 (natural) since s0012 (2.87 / 5.22 before); every gate passes in every submission; generate /
  http_tokens strict (track C): identical in all.
- step() = the s17 s0014 track-A function, unchanged (engine/a2 at B=1, engine/a1 at B>=2); the verify kernels'
  rows are bit-identical to it (tests/c/xcheck.py: 0 differing positions in every build, mixed K and K=8).

| sub | change | b1 | natural | tok/event |
|---|---|---|---|---|
| s0001 | s17 engine + c1 s0022 verify kernel / draft for B=1 requests | 1521.7 | 1015.7 | 2.88 / 5.22 |
| s0002 | verify kernel = 3-way merge of c1's verify path onto a2 s0025 + s17 | 1456.8 | 954.2 | 2.88 / 5.22 |
| s0003 | w2 staged inputs padded (8-way smem bank conflicts at 4 tokens) | 1403.5 | 890.3 | 2.88 / 5.22 |
| s0004 | w2 items on 3 warps (the step's path) | 1380.6 | 876.9 | 2.88 / 5.22 |
| s0005 | FP8 draft layers; MXFP8 pieces 3 deep; router polls together | 1374.1 | 872.3 | 2.87 / 5.22 |
| s0006 | w2: two items per 3-warp group per round (12 units in flight) | 1353.2 | 855.4 | 2.87 / 5.22 |
| s0007 | DSpark kernels in a CUDA-graph IF node (skipped on n-gram cycles) | 1350.3 | 848.8 | 2.87 / 5.22 |
| s0008 | one verify kernel per row count K (8 libraries; 7 return at entry) | 1318.3 | 834.4 | 2.87 / 5.22 |
| s0009 | per-K compile switches (c/kflags.txt) | 1310.5 | 830.7 | 2.87 / 5.22 |
| s0010 | verify via a graph SWITCH node over the 8 kernels (no empty launches) | 1312.4 | 830.7 | 2.87 / 5.22 |
| s0011 | committed source re-evaluated (= s0009 + off switches) | 1317.5 | 834.1 | 2.87 / 5.22 |
| s0012 | + c3's 16-row n-gram verify passes (library padded, own cost table, IF node) | 1260.0 | 810.2 | 3.04 / 6.69 |

## The verify pass vs one decode step (natural text, ctx 600)
V(K) K=1..8 ~2890 3424 3574 3813 4196 4482 4615 4680 us; the step 2790: V(1) = step + 100, V(2) = 1.23x,
V(8) = 1.68x (start of run: c1's kernel 3401 .. 5523 at random-token context, ~3420 .. 6170 on real text).

## What paid, and why
1. Bank conflicts (s0003): the w2 input rows in shared memory had a 576-word stride, all 8 B-operand columns of a
   4-token expert on one bank. Found by timing w2 items per block by token count (tests/c/w2blk.py).
2. Register allocation (s0008): the same kernel compiled with the row count K as a constant spills 40% less and is
   50-200 us faster per pass. Eight libraries, one per K; the draft decides K on the device, so all eight are
   launched and seven return at entry. Every attempt to add code to the generic kernel lost more in spills than it
   saved (TMA overflow slots, kv_conv4, split k/v tasks, certified INT8 argmax inline, mark sites).
3. Load depth through structure, not knobs (s0006): two w2 items per warp batch (two independent accumulator
   chains) doubled bytes in flight; the same for w13 did not pay (its chains were already 12 units).

## Robustness
tests/c/robust.py and soak.py (prompt lengths 1-1000, n 1-1984, vocabulary-edge ids, loops, past position 1024,
fallback for requests too long for the speculative path, mixed B=1 / B=8 cohorts): every stream identical to
step-by-step greedy. Note: the evaluator's http_tokens gate sends 8 prompts at once (the B=8 path), so the speculative
B=1 path is checked by these tests, not by the evaluator's gates.

## Dead ends (measured; journal.md has the numbers)
L2 / L1 prefetch of expert weights, allocating loads, W13 smem overflow via TMA (+ extra smem costs L1), staggered
groups, w13 two-items-per-warp, certified INT8 argmax (net 0 after its out-of-line cost), split k/v conv tasks,
theta 0.3-0.55 (flat), FG_ROWS, draft GEMV prefetch (register pressure), SWITCH-node verify (no HTTP gain).

## For a next run
- The 16-row library (c1 s0013 lineage) lacks most of this run's kernel work: a careful merge (c3 re-indexed tagged
  buffers to 16 rows; code added since has no base lines) could bring V(16) from ~8 ms towards ~6.5 ms.
- The attention record holding the pass's own positions (k/v convolutions with per-channel polls) is the largest
  dense loss left at K >= 2; any fix must be code-neutral for register allocation (or live in its own library).
- B=8 speculation (other run).
