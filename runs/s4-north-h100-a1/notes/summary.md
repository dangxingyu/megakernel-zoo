# Summary: North-Mini-Code-1.0 on one H100 80GB

Details, every measurement and the prediction made before each result: notes/journal.md. Findings about the checkpoint
and the workload: notes/checkpoint.md.

## Result

| | http_step_us_b1 | B=8 (us/step) |
|---|---|---|
| vLLM 0.32.0 | 5049 | 9277 |
| Cohere's megakernel | 3616 | 7176 |
| project claim (1.10x vLLM) | 4590 | 8434 |
| **track B, s0021** (best headline) | **2146** | 6028 |
| **track B, s0023** (= `engine/` now, byte-identical to its source) | 2150 | **5989** |
| **track A, s0022** | **2533** | **6893** |

23 submissions, all valid. s0021 and s0023 differ by one scheduling constant (attention's per-K-tile delay in the
gate|up share model): equal at batch 1 within the run-to-run spread of +-10 us, s0023 39 us better at batch 8, so
`engine/` holds s0023. Track A: `echo A > engine/TRACK` builds it (verified from a clean archive, e0385).
Every change after s0012 left the outputs bit for bit unchanged (logit/token hash, e0387).

## The engine

One persistent kernel per serving cohort (submit/poll), 132 blocks x 512 threads, a static schedule per layer:
norm -> A (q|k|v, router rows) -> B (attention on 32 blocks; gate|up of the union of the chosen experts, shares by
measured start delay; o_proj after a block's gate|up share) -> C (down of every chosen expert for the block's
residual row tile, residual add). Tensor-core GEMV with weights in mma-fragment order; batch = mma columns; every
tile summed in a fixed order (single == batched, bitwise).

## What worked (track B unless noted; us at B=1 over HTTP)

- Checkpoint-specific, lossless:
  - 12-bit weights: per-row exponent window, a shift per expert neuron for the 1,210 near-dead neurons, exact
    escapes (12.07 bits/weight, verified bit for bit on all 30.5e9 weights): 2933 -> 2628.
  - High bits laid out so a register of two weights is PRMT + SHF + LOP3 (the 12-bit decode had capped the stream):
    -46.
  - Layer 0 from per-token tables (its input is the embedding row): -30 (mostly B=8).
  - Certified int8 sketch of the tied head on the greedy path (541 instead of 805 MB; ~1.3 tiles of 16384
    recomputed exactly per sequence; argmax identical in 4608/4608 tests): -98.
- Kernel organisation (also in track A):
  - Weight loads with L1::no_allocate and the zero operand from shared memory, so stage C's re-read x operands stay
    in the 28 KB of L1: -113 (the largest single step after the 12-bit format).
  - Gate|up shares from measured start delays; the leader signals stage A before routing; union published first: -62.
  - o_proj inside stage B (after the L1 change; before it, it was worse at B=1): -65.
  - Removing dependent round trips: escape gather after the first quantum, local routing details at B=1, union load
    issued with the last stage-A quantum (its own kernel instantiation), exchanges instead of store+fence: -60 in all.
  - RMSNorm with cvt.rn.bf16x2 (bit-identical): -50 at B=8.

## What did not work (all measured; journal sections 7, 10, 15-19)

- Staging the next stage's weights during barriers (prefetch.global.L2 does not stage; cp.async/bulk copies delay
  the barrier's own round trips by as much as they save: the memory queue is first in, first out).
- Per-block routing, rank-count top-8, several polling warps, stamped attention partials or stamped residual
  (polling floods the queue), fewer attention blocks per task at B=1, 11-bit format (2.6% escapes), SM-aware role
  mapping (the "fast SMs" exist only with less shared memory than the engine uses), many share-constant tweaks.
- Run-to-run noise is +-10 us at B=1 (where roles land on the card): single runs mislead; tools/ab.sh alternates
  two builds on one card and was used for every decision after s0015.

## What I would try next

Exact draft-and-verify with drafts copied from the context (journal sections 19-20). On this workload the greedy
continuation loops: 8-token copied drafts are accepted for 6.2 tokens on average (three of eight sequences at
96-99% full acceptance, one prose-like at 0.2), and an 8-position pass of one sequence costs 2.49x a batch-1 step
(e0383). Estimate ~860 us/token at B=1 (~2.5x), more with 16-32 columns. Requirements: per-column positions in
attention, K/V of earlier columns of the pass taken from qkvo, logits of a P-position pass bit-identical to P single
steps, a short pass after a miss, and its own kernel instantiation (adding per-column offsets to the shared kernel
cost B=8 70 us). After that: the B=8 stage C reads ~56 MB/layer of x operands from L2 (row-tile decomposition).
