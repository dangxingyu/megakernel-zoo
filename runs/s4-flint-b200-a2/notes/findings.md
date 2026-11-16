# What is specific to Flint on a B200, with the measurement behind each point

Status legend: [measured] number in hand; [built] in the engine; [open] not settled.
Engine numbers are from the sealed evaluations unless a tool is named (see notes/journal.md for every step).

## Where the engine stands (sealed evaluations, HTTP, us per token at batch 1 / us per step at batch 8)

| | batch 1 | batch 8 | against vLLM 0.33.0 (1971 / 3796) |
|---|---|---|---|
| track A (formats and data flow of the general engines), s0077 = `engine/` as committed (1479 / 3397; one count loop less than s0076, the same to the evaluation's precision), best figures: s0075 at batch 1, s0076 at batch 8 | 1478-1483 | 3394-3397 | 1.33x / 1.118x (1.34x / 1.122x of the same-card rematch, 1990 / 3810) |
| the same track before the routing was taken apart with the cycle counter, s0070 / s0071 | 1493 | 3426 | 1.32x / 1.108x |
| the same track at the end of the first 24 hours, s0058 | 1520 | 3877 | 1.30x / 0.98x |
| float32 residual stream and hi+lo router operands, s0057 (was called track B; not bit-identical to track A, see below) | 1475 | 3950 | 1.34x / 0.96x |

(Extension, 2026-11-15: 12 more agent-hours for batch 8 on track A; target 3470 = 1.10x of the operator's same-card
rematch. Nineteen valid submissions s0059-s0077, all track A, all with the bits of s0058: tools/digest_long.py and
tools/digest1.py unchanged throughout. The audit of the first 24 hours found that the float32-stream build is not
bit-identical to its track-A build (0 of 14,336 positions), so it is not track B as this project defines it; it is
not submitted any more. Evaluations of the same kernel differ by about 6 us at batch 1 and 30 us at batch 8.)

What the extension found, in one place (items 15a-15g below, journal under "Extension"):
- 3877 -> 3426 at batch 8 and 1520 -> 1493 at batch 1 came from ten changes, each first seen in a measurement that
  did not need the profiling build: the code generation of a conditional load (15a, -56), the thread that polls
  (15b, -18 at batch 1), K/V tile batches that the compiler had dissolved (15c, -50 and -20), four loads in flight
  in the copy of the attention output (15e, -43), one fragment copy per down phase instead of one per round
  (15f, -146), two rewrites that took values off the stack (15h, -44 and -28), and the output projection of the 8
  main blocks with an attention pair computed by 8 other blocks (15i, -39).
- The scored run is a long-context run (positions 64..1088; 15d); what is measured in process at positions under 330
  can have the opposite sign there (HOSTC).
- The profiling build misleads where the question is "who waits for whom": its markers are loads, and a load waits
  for what its warp issued before it. Timing studies in the normal build (WHATIF, 15g) replaced it.
- In the last three hours the cycle counter (15l) was pointed into the routing, which nothing had profiled without
  disturbing it: five more submissions (15m), 3426 -> 3394 at batch 8 and 1493 -> 1478-1484 at batch 1 (in process
  1382 -> 1364), each a rewriting of the top-6 selection that gives the same order and ties.
- The goal of 3470 is met by 76 us (s0076: 3394; s0075: 3397; s0070: 3426); the spread of one evaluation is about 30. What is left at batch 8 on the scored workload, by timing study: routing computation
  about 200 us per step, context-dependent attention 230, the one fragment copy of the down phase 120, the copy of
  the attention output 90, the sentinels of the expert phase 80, the norms' sequential loads 45; the streaming itself
  is at the card's rate.

(Final engine of the extension: 0 mismatches in 399 repetitions of tools/flake.py per workload, and in 59 at batch 7,
5 and 3; step by position at batch 8, tools/pos_time.py: 3349 at positions under 128 to 3582 at 1024-1152.)

All of it strict (no result depends on timing: 0 mismatches in 299 repetitions of tools/flake.py per track and
batch size on the final build, and in every repetition run since s0017; s0008, the earlier headline of 1685, was a racy kernel) and bitwise the same for a
sequence alone and in any batch (tools/bcheck.py; the evaluator records the same). The claim of 1.10x holds at
batch 1 on both tracks (1.30x on track A) and not at batch 8 (0.98x: 82 us per step short of the general engine).
Track B (float32 residual stream, two-operand router: built for routing experiments that did not pay, item 14) is
the faster track at batch 1 since its batch 1-2 kernel got the good register allocation (item 13i) and the slower
one at batch 8, where its stream costs bytes and instructions; what this run found applies to both tracks.
What the time at batch 1 is made of (in process 1382-1402 us per token = about 27 us per layer for 107 MB = 15 us of
memory time, plus 70 us of head; items 8a-8c, 13f): the wait of the 128 weight SMs for the attention (about 4 us),
the router and routing (2.3), and the latency of six hand-overs per layer (norms, epilogues, counters: about 6).
What batch 8 lacks (about 76 us per layer for 310 MB = 41 us of memory time): the expert phases run at the card's
rate, but the 8 attention pairs that have no SM of their own (items 6, 13d) make every main block wait 9-15 us per
layer, and the down phase needs 22.6 us for 81 MB (5 rounds; three other organisations of it were slower as a
whole, item 13b and journal). The step also grows with the position (item 8b, 13c, 13f): the scored workload runs
to position 1088, where a token costs 200 us more than at position 64 at batch 1.

What was learned in the last third of the run, in one place (items 13c-13g): on an SM that runs 16 warps the
attention and the norms are bound by instructions (4 ns each per warp), not by memory, so a float division or a
recomputed scale is on the critical path; and the address of a 4-KB page that every block polls is worth 1% of the
step, so the engine times itself at start-up and places its counters and three small buffers.

Gains since the first valid engine, in the order found (track B HTTP batch 1 / batch 8):
  shared-memory reservation 224 -> 128 KB (item 9)                          1968 -> 1782 / 6273 -> 5245
  two kernel instances, batch > 2 organisation, head batches (item 13)      1772 -> 1733 / 4158 -> 4100   (after the first race fix)
  strict hand-overs: sentinel words (item 12d)                              1733 -> 1733 / 4100 -> 4268   (the price of strictness at batch 8)
  ranks instead of rounds, epilogues off the critical warps (items 8, 8a)   1733 -> 1699 / 4268 -> 4221
  K/V cache tiles in batches at long contexts (item 8b)                     1699 -> 1646
  batch > 2: rank merge, router tiles preloaded (items 8, 8c)                              4227 -> 4123
  generate/serve: the blocks' best tokens gathered in one trip instead of 302 (journal)    1650 -> 1606
  the same for batch > 2: 16 strides instead of one chain of 302 loads (journal)                       4116 -> 4088
  softmax weights by reciprocal + exact correction instead of divisions (item 13c)         1606 -> 1560 / 4088 -> 4055
  batch 7-8: early poll on main blocks with a pair (13d); norm scales once per warp (13e)          4055 -> 4039 -> 4021
  sentinel reads without the abort word in front of every reload (13g; track A 1546 -> 1527)   1559 -> 1566 / 4021 -> 3979
  counters on the fastest of 16 pages, timed at start-up (13g; track A 3914 -> 3870)          1566 -> 1560 / 3979 -> 3936
  three more buffers placed for the batch-1 kernel (13g; track A 1527 -> 1516)                1560 -> 1557 / (3936 -> 3950: noise)
  the kernel body as an inlined function: another register allocation (13i; track B only)     1557 -> 1473
Gains of the extension (track A, HTTP batch 1 / batch 8):
  fragment getter without a conditional load, batch > 2 kernel (15a)                         1522 -> 1531 / 3898 -> 3842
  hand-over counters polled from warp 6 instead of the leader warp 0 (15b)                    1531 -> 1514 / 3842 -> 3829
  attention K/V batches pinned (volatile asm), used in the batch > 2 kernel (15c)             1514 -> 1494 / 3829 -> 3784
  copy of the attention output with 4 loads in flight (15e)                                   1494 -> 1497 / 3784 -> 3735
  down phase: one fragment copy up front, rounds stream tiles only (15f)                      1497 -> 1496 / 3735 -> 3577
  (s0065: the same kernel again)                                                                       1492 / 3571
  that copy without its pointer array: spills 52/88 -> 24/24 bytes (15h)                      1492 -> 1493 / 3571 -> 3527
  routing merge without lane-indexed stack arrays (15h)                                       1493 -> 1493 / 3527 -> 3499
  batch 7-8: output projection of the 8 pair blocks hosted by 8 other main blocks (15i)       1493 -> 1493 / 3499 -> 3460
  the hosts on blocks 24..31 instead of 8..15 (15i)                                           1493 -> 1493 / 3460 -> 3448
  copy of the attention output: all pending words of a batch requested again per attempt (15e) 1493 -> 1493 / 3448 -> 3426
  routing union with two barriers fewer (15m)                                                 1493 -> 1491 / 3426 -> 3412
  routing candidates (batch > 2) by rank counts instead of shuffle rounds (15m)               1491 -> 1490 / 3412 -> 3405
  routing merge (batch 1-2) over 27 pruned candidates instead of 48 (15m)                     1490 -> 1484 / 3405 -> 3408
  routing union: the last pass on all 16 warps (15m)                                          1484 -> 1478 / 3408 -> 3397
  routing merge: table bytes from the ranking lanes, weights on idle warps (15m)              1478 -> 1483 / 3397 -> 3394
  routing union: the per-warp counts without a loop (15m; -4 in process on one card, 0 on another) 1483 -> 1479 / 3394 -> 3397

## The checkpoint

1. **Nothing to exploit in the weight values.** [measured, harness census e0003, all 39.79 B weights] No zeros, no
   denormals, no identical tensors, no zero rows/columns; exponent entropy 2.56 bits, mantissa 6.97 bits; a 12-bit
   fixed code has 6.3e-5 escapes (12.002 bits/weight), an 11-bit code 1.3% escapes (11.42 bits). Allows: W12 as on
   dense models. Does not allow: anything better than W12 with a fixed-width code.

2. **Routing is close to uniform and hard to predict.** [measured, tools/hf_study.py: 8 own sequences x 1152
   tokens, HF BF16] Per layer the 256 experts are used with entropy 6.7-7.6 bits (uniform = 8); 12-24% of a token's 6
   experts were used by the previous token (2% in the first MoE layer); no correlation of expert ids between
   consecutive layers (2.3% = chance). Allows: nothing based on locality (no expert cache, no temporal predictor).
   Over a batch of 8 a layer uses 38.4 distinct experts on average (min 6, max 48): the batch-8 step reads about
   3.26 + 1.77 * 38.4 / 6 = 14.6 GB, close to the "all route apart" count. A unit (expert used in a batch) has 1.25
   member sequences on average, so batch-8 expert work is 6.4x the batch-1 work, not 8x and not 1x.

3. **The router's top-6 is a near-tie most of the time.** [measured] margin between 6th and 7th sigmoid score: median
   0.011; BF16 and float32 pick different sets in 48% of (layer, token) pairs on my sequences. Consequences:
   (a) any reordering of a sum that feeds the router changes some expert choices, so the top-1 gate (limit 91.9%)
   moves by +-1.5 points between equally exact summation orders: track A 91.6% (fail, s0003), 93.8% (s0006), 93.1%
   (s0008); track B 95.65% (s0005), 93.2% (s0007). On my own reference the same variants are 88.6-91.3% where
   Hugging Face BF16 itself is 90.2   (b) The batch-of-8 result must use the same summation tree as the single-sequence one to stay bitwise equal, and
   the same multiply-add fusion: the attention is compiled in two places (attention blocks; main blocks at batch
   7-8), the compiler fused a*b + c*d differently in the two, and sequences 6-7 differed by a rounding step between
   alone and in the batch (s0017-s0019, track B; tools/bcheck.py checks every batch size and sequence, the gate
   checks one). Fixed by writing the attention's multiply-adds with explicit intrinsics (s0020).

4. **What the Hugging Face BF16 path really rounds** [measured element-exact, tools/hf_semantics.py]: linear =
   float32 accumulate + one rounding; SwiGLU = bf16(silu(g)) * u; the 6 expert outputs are multiplied by float32
   routing weights and SUMMED IN FLOAT32, then rounded once (the grouped implementation in integrations/moe.py, not
   the index_add loop in modeling_flint.py); residual adds round to BF16. Track A reproduces these points.
   Track B (lossless in exact arithmetic) keeps the residual stream in float32 and gives the router hi+lo BF16
   operands; it costs about 6% in speed (two mma passes on the router blocks, 4-byte residual) and does not remove
   the near-tie lottery of item 3.

## The architecture meets the card

5. **GQA ratio 8 = the n of mma.m16n8k16.** [built] 8 query heads share one K/V head. With q as the A operand (rows =
   heads) and a K-cache tile as the B operand (8 keys), the scores of one head for keys (2t, 2t+1, 2t+8, 2t+9) land
   in one thread; that is exactly the register layout of the B operand of the next mma (V tile x weights), so the
   softmax output is never stored or shuffled: it is born as the operand of the V mix.

6. **One (sequence, K/V head) pair = one SM.** [built] 4 pairs per sequence; a pair's whole visible K of a
   sliding-window layer is 132 KB. 24 SMs are not needed for weight streaming (128 blocks saturate the memory), so
   batch 1-6 has an idle SM per pair that also stages the next layer's K/V tiles (16 tiles in registers, 12 more key
   groups in shared memory). At batch 7-8 there are 32 pairs: the 8 extra ones run on main blocks while those wait
   (tools/test_engine.py, track B: batch 8 4767 -> 4338 us per step, 64+1024 workload 5278 -> 4525 us, against two
   pairs in sequence on 8 of the aux blocks).

7. **Batch 8 = the 8 columns of the mma B operand.** [built] Every shared matrix (attention, shared expert, dense,
   head) costs the same bytes at batch 8 as at batch 1; only the experts multiply (item 2).

8. **MoE phase structure**: the router (1 MB) runs on 16 otherwise idle SMs while the 128 weight-streaming SMs do the
   shared expert's gate/up; top-6 is then computed redundantly in every block (no extra hand-over). [built]
   An expert's gate/up is 64 row groups of 128 tiles; a block's share is cut into pieces of 16 tiles so that every warp
   has a full 16-tile batch in flight and the next batch is requested before the partial sums are folded.
   At batch 1-2 the top 6 are found by ranks (every thread counts the scores of its warp that beat its own: 8 loads
   from shared memory; then the 48 candidates the same way) instead of 12 rounds of warp maxima: bit-identical,
   17 us per step (s0021).

8a. **Where a layer's time goes, and what could be moved** [measured, tools/prof.py; s0021-s0023]. A batch-1 layer
   is about 32 us for 107 MB (15 us of memory time). The rest is a chain of small steps in which ONE warp or ONE
   block is what everybody waits for. Three places where that warp was doing something nobody was waiting for:
   (a) batch 1 has 3 expert row groups per block: slot 1 uses 16 warps, slot 2 only warps 0-7, and warp 0 wrote the
   hidden units of slot 1 (SiLU, atomics: 1 us) before its tiles of slot 2; warps 8-9, idle in slot 2, do it now
   (16 us per step). (b) The shared expert's hidden units (read only in the down phase) were written by warp 0
   before the routing; now by warp 15 during it (14 us). (c) The attention wrote this position's K and V into the
   cache before computing the scores; now after its output (4 us). All bit-identical.
   What cannot be moved: the wait of the 128 main blocks for the attention (4.5 us per layer at batch 1, 11 us at
   batch 8, where 8 pairs run on main blocks after their own phase A): its inputs are the last thing phase A
   produces. Staging those pairs' K/V in the main block's registers, or batching their loads, does not help (slower).

8c. **The 16 router SMs are on the critical path** [measured; s0031 and the speculative-routing build]. At batch 1
   the real logits arrive about 1 us after the main blocks have finished phase D. Their 1 MB of router tiles sitting
   in the staging registers from the layer's start is worth 159 us per step (measured by taking it away). At batch
   > 2 the router blocks also run an attention pair and used their registers for the next layer's K/V right after
   it: requesting the router tiles first and staging after the router duty is worth 32 us per step at batch 8
   (HTTP 4188 -> 4123, s0031); after that the router is 1.5 us ahead of the main blocks there.

8b. **The attention at long contexts** [measured; s0025-s0028]. The score is measured to position 1088, the
   in-process step at positions below 320. At position 900 the attention took 13.8 us against 7.2 at position 192:
   beyond the 16 key groups staged in registers (+12 K groups in shared memory) the cache tiles were loaded one per
   trip to memory inside dependent loops. Requesting a key group's 8 K tiles together, and the next 8 V levels
   together and as soon as the K registers are free (the staging registers are reused: a separate array made the
   kernel spill, 2195 us), is worth 53 us per token over HTTP on track B and 63 on track A at batch 1. In the batch
   > 2 kernel the same change is neutral to negative (measured both for its attention blocks and for the pairs on
   main blocks) and is not applied.

## The card (B200) as measured here

9. **The shared-memory reservation sets the speed of every global load** [measured, tools/bench12.cu; built].
   L1 and shared memory share 256 KB per SM. A batch of 16 tile loads per warp takes 0.76 us with 128 KB or less
   reserved, 0.83 us up to 160 KB, 1.13 us up to 192 KB and 2.27 us above; the saturated streaming rate falls from
   7.70 to 5.95 TB/s. The engine reserved 224 KB (K/V staging) until s0006. Cutting it to 128 KB: HTTP 1968 -> 1782 us
   at batch 1, 6273 -> 5245 at batch 8 (track B, s0005 -> s0007); track A 1901 -> 1685 and 6081 -> 4796 (s0006 ->
   s0008). This also explains why a 1024-thread kernel, W12 and shared-memory tile staging all looked bad earlier.

10. **Loads**: a warp's loads return in the order they were issued, and several can be in flight: unloaded, 8 tile
    loads take 0.42 us and 16 take 0.76 us; a load-then-use pair alone takes 0.7 us. 128 blocks x 16 warps with
    batches of 8 or 16 saturate the memory at 7.4-7.7 TB/s (tools/bench10/11/12). Consequences: everything a phase
    needs is requested as one batch as early as possible; small loads (activations) must be requested BEFORE the
    weight tiles of the same warp or they wait behind them. Placement does not matter: blocks reading one small
    region or far-apart regions, or cycling over 128 regions 2-8 MB apart, run at the same rate (no TLB effect).

11. **L2** is 129 MiB. Re-reading up to 96 MB runs at 17-22 TB/s (2.5-3x DRAM); `prefetch.global.L1/L2` behave like
    blocking loads. [measured, tools/bench11.cu] [open] Using idle SMs to pull the next phase's weights into L2.

12. **Hand-overs between blocks** [measured, built]: one atomic counter per kind of event; a waiting block spins on
    it with thread 0. Two rules, both found the hard way:
    (a) Readers: L1 is not coherent across SMs. Plain loads of data another SM wrote are stale in 2.8% of the cases,
        `ld.global.cg` loads in none (tools/bench8.cu); a plain poll of the counter sees it 0.78 us late, a
        `ld.global.cg` poll 0.105 us (tools/bench14.cu).
    (b) Writers: a plain store is NOT ordered before a later atomic, not from the same warp and not with
        `st.global.cg`: about 8% of 600-token generations differed between processes (tools/flake_gen1.py, 9 of 120;
        s0005-s0008 have this). `__threadfence` in every storing warp fixes it (0 of 120) and costs 11%.
    (c) Writing the data with `atomicExch` and bumping from the same warp afterwards (s0009-s0016) is NOT strict
        either: an exchange whose result is not used does not hold the warp up and the bump can overtake it
        (tools/flake.py: 1 of 357 repetitions on the s0015 kernel, 6% with more traffic; tools/zdump.py caught an
        attention head that had read a stale q). Waiting for the exchanges' results before the bump is strict
        (0 of 237) and costs 5-7%: one more trip through the memory queue per hand-over.
    (d) **The sentinel hand-over** [built, s0017/s0018]: the writer exchanges a NaN sentinel into its words at the
        START of its phase (every reader of the old content is done), reads the results back while its first loads
        are in flight, and bumps the counter at once; at the end it only exchanges the data in; a reader loads a
        word again while it holds the sentinel. Strict (a reader that saw the counter finds the sentinel or the new
        data, never the old), 0 mismatches in 386 repetitions, and the signal is off the critical path: step
        1577 -> 1595 us (track B, batch 1) against 1765 for settling at the phase end; batch 8 3961 -> 4106.
    (e) The memory queue is one FIFO for every SM (tools/bench17.cu: a hand-over with data takes 1.8 us on an idle
        card and 4.1 us when every warp has 16 tile loads in flight). Every dependent step of a hand-over (signal,
        poll, load of the data) is one trip through it; the sentinel protocol needs one (the data exchange, then
        the reader's own load) where counter-after-data needs three.

12e. **Hand-over reads are strong loads** [built, s0057/s0058; from another run of this project]. The run on
    Qwen3.8-27B on the same card (progress/s4-qwen38-27b-b200-a1, its finding C11) took the sentinel idea from here
    and found that a spin with a weak load (ld.global.cg) can miss another SM's later plain store: 16-35% stale
    reads in its scheme, none with ld.relaxed.gpu ("a cache hint is not a coherence guarantee on this two-die
    card"). Here every writer uses atomic exchanges and no stale read was ever seen (0 mismatches in 299
    repetitions per track and batch size, items 12c/12d), but the readers were weak loads of the same kind. With
    ld.relaxed.gpu for every read of a word another block writes: same spill lines, same timings (1401 / 3744 and
    1382 / 3805 us in process), same digests; HTTP 1475 / 3950 (B) and 1520 / 3877 (A), i.e. unchanged. Not
    hardened: the counter polls (volatile loads of counters that only grow: a stale value makes a block wait
    longer, never go early).

13. **Registers are the scarce resource of the main blocks** [measured]: one block per SM with 512 threads has 128
    registers per thread, 64 of them hold the 16 staged tiles. Every attempt to add in-flight values to a main-block
    phase (fragment operands in registers, all sequence pairs of a norm, batched K/V tiles) made the compiler spill
    (144 to 1150 bytes) and slowed the whole step by 5-40%, also in paths that were not touched. Compiling the
    kernel twice, for batches of 1-2 and for larger ones, each without the other's paths, removed that coupling
    (batch 1 serve 1753 -> 1728 us, bit-identical; s0011/s0012).
    Corollaries met later: a result carried in a register across a phase costs more than the wait it avoids
    (sentinel results, item 12d: 1577 -> 1621 us); 4 loads in flight in a copy loop made every phase 15% slower;
    code that is compiled in but never runs (a switch, a second inlined copy) moves the batch-8 step by 0.6%;
    and the two tracks, compiled from the same source, answer the same bit-identical reordering with opposite signs
    of 10-15 us. Differences below that are the register picture, not the idea.

13c. **Arithmetic is not free on an SM that runs 16 warps: the compiler's float division costs 38 ns per warp**
    [measured, tools/bench19.cu, tools/pos_time.py; built, s0041/s0042]. The step grew by 250 us (batch 1) and 375-400
    us (batch 8) between position 64 and 1100, and the fine profile of an attention block put most of the growth in
    the softmax stages, not in the K/V loads: at a full window of 513 keys a (sequence, K/V head) pair needs 4224
    weights e / sum, 8-12 IEEE divisions per thread (moving the scores from the stack to shared memory changed
    nothing: it was not memory). A division is 38.5 ns per warp with 16 warps on the SM; q = e*r, q + fma(-q, sum,
    e)*r with r = 1/sum once per thread is 12.2 ns and is the same float (Markstein's division step: the correctly
    rounded quotient when the remainder is exact, i.e. for every exponential of 2^-102 and more; 0 mismatches in
    6e10 random quotients and over all 2^23 mantissas; below 2^-102 the weight, itself under 1e-30, can be one unit
    in the last place off). Same digests as before on both tracks. Worth, over HTTP: track B 1606 -> 1560 / 4088 ->
    4055, track A 1608 -> 1548 / 4028 -> 3928. (A guard that sends exponentials below 2^-100 to the real division
    keeps every input bit-identical but made the batch > 2 kernel of track A spill, +50 us per step: switch
    SOFTMAX_GUARD, off.) What it does not cover: the router's sigmoid (one division per logit; 4 per thread at batch
    > 2, about 0.15 us per layer) has no exact shortcut that keeps the ties of the top-6.

13d. **At batch 7-8 the step waits for 8 attention pairs that have no SM of their own, and their SM's transfer rate
    is the bound** [measured, profile and switches; s0043/s0044]. The dedicated attention blocks are done about 6 us
    after phase A, the 8 pairs on main blocks 10 us (position 150) to 15 us (position 1000) after it, and 120 main
    blocks idle meanwhile: 12-18% of the step. What was tried on that chain (64+1024 workload, batch 8, same card):
    the output projection's tiles requested behind the attention's last stores instead of after its barrier: +130 us
    (as with the request at its start: tiles the attention does not need delay it by their whole transfer);
    the first K group requested in the trip of q/k/v: 0; K and 8 V levels in that trip: -45 us at positions below
    256, +100 over the workload (16 more tile registers live through the attention: spills in its long-context
    loops); the sentinel of the post-attention stream issued before the attention: 0 (an exchange whose result is
    not used does not hold the warp); the counter of phase A asked by an idle warp during the leaders' stores instead
    of by thread 0 after them: -7 to -20 us locally, HTTP 4055 -> 4039 (B), 3928 -> 3920 (A): kept. The same early
    poll between phases C and D of every main block: 0 (there the answer is not old news: the poll is as long as the
    data needs to arrive). Conclusion: a tile costs its 512 bytes of the SM's 58 GB/s whenever it is requested;
    merging trips buys nothing, only tiles that are already on the SM (registers, shared memory) are free, and a
    main block has no room for 264 KB of K/V next to its own tiles. A pair needs 0.9 us (position 150) to 4.6 us
    (full window) of its SM for K/V alone. Not solved: it would take 8 more SMs without weight duty, and the 128
    main blocks are tied to the 16 rows of the residual stream each of them owns.

13e. **The residual-stream norm at batch > 2** [measured; s0045/s0046]. It runs twice per layer on every main block
    and once on the router blocks, and takes 4-5 us at batch 8 against 1.8 at batch 1.
    (a) Not the loads: the sequences are loaded pair by pair, each pair after the previous one was used, which looked
    like 4 trips in a row. Requesting all pairs together (landing in staging registers the phase does not use: with
    any other landing place the kernel spilled 700-11000 bytes, it has no register left; and with the words that
    still hold the sentinel retried one at a time, because a retry of all pairs moves 64 KB per block and round)
    shortens the norm by 0.4 us in the fine profile and the step not at all (+170 us in the form tried: the phase
    after it got slower). The norm's time is the wait for the stream's last writer plus arithmetic. Not kept
    (tools/mk_normbatch_v*.cu.bak).
    (b) Arithmetic: every thread added the 16 partial sums and took the rsqrt for each of the 8 sequences. Once per
    warp (lane b for sequence b, read by shuffle; same sums, same order, same bits): track B step 3905 -> 3880 at
    batch 8, HTTP 4039 -> 4021; track A 3782 -> 3780, HTTP 3920 -> 3912 (noise): on track A the norms are not on the
    critical path, on track B the router blocks' norm is (float32 stream: twice the bytes, two fragments per value).
    (c) A fragility found on the way: the sum of squares a*a + b*b + c*c + d*d is left to the compiler's fusion, and
    a restructured copy of the norm fused it differently: sequences 2 and 6 no longer bitwise equal to the same
    sequence alone (tools/bcheck.py; deterministic, not a race). On track A the form cannot matter (the squares of
    BF16 values are exact in float32, and the mean is a multiplication by 2^-11): all 12 written-out forms give the
    reference digests. On track B (float32 stream) each of the 5 inlined copies has its own form today (none of the
    12 forms applied to all copies reproduces the digest), they agree between the two kernel instances, and bcheck
    is what guards it. Writing one form out everywhere (ss4, mode 10) makes batch invariance hold by construction
    but changes track B's bits (rounding noise of the norm), so it is not in the submitted engine: the gate on the
    maximum error (1.67-1.76 against 1.98) is decided by which near-tie routings flip, and a new set of bits is a
    new draw. Anyone who edits norm_frag on track B must run bcheck.

13f. **What time is made of on an SM that runs 16 warps** [measured, tools/bench20.cu, tools/bench21.cu, and the
    experiments of the journal's last sections]. One block of 512 threads per SM; all 16 warps busy:
    - a plain instruction costs every warp about 4 ns (four roundings to BF16, 5 instructions each: 82 ns; a warp sum
      of 5 shuffles: 86 ns; 4 exp: 73 ns; the compiler's float division: 38 ns; a barrier all warps reach together:
      about 20 ns). 250 instructions per thread on a critical block are a microsecond per layer, 49 us per token.
    - shared memory has 32 banks of 4 bytes and conflicts cost: store + load of a word 21 ns without conflict, 251 ns
      with 16 lanes in a bank, 502 ns with all 32: about 7.5 ns per lane and bank. An 8-byte access by 32 lanes has 2
      lanes per bank at best (47 ns per pair), a 16-byte access 4.
    - an mma is an instruction like the others here (16 in 73-80 ns); the dependent chain on one accumulator is not
      what bounds the attention.
    What follows, and what was tried with it (all bit-identical to the engine, none kept unless said):
    * the softmax divisions (13c, kept) and the norm's scales (13e, kept) are this effect;
    * 2 accumulators per half in the scores and 4 in the V mix (ATT_ACC): -9 us per token at short contexts, +45 at
      long ones at batch 1 and +285 at batch 8 (more registers live, the mix loop of the main-block pairs no longer
      pipelined);
    * one mma per K tile instead of two (KSWAP: the stored tile is also a 16 x 16 operand with rows = keys, registers
      x, z, y, w, and q as stored an 8-column operand; the 128 sums of a tile come out with the same bits, digests
      identical): the 8 mma saved per key group are 36 ns, the 4 scalar stores that carry the scores back to the
      layout the softmax needs have 4 lanes per bank and cost 157 ns: +73 us per token at batch 1;
    * 17 K groups and one V level of a full window staged in shared memory instead of 12 and 0 (KV_SM2: the weights
      move into 24 KB the attention blocks never used): the trips it removes are worth 11-15 us per token beyond
      position 512 on track B, the run-time layout variables cost 7-13 below it; track A +3 over the workload;
    * the copy of the attention output into shared memory at batch > 2 has 16 lanes per bank in each of its 8 stores
      per thread; assigning (sequence, half) places to threads instead of chunks brings that to 4 (FRAG_NB), but the
      global loads of one instruction then come from 16 cache lines instead of 4: +15 (track A) to +43 us (track B)
      per step at batch 8. Contiguity of the global loads is worth more than the banks.
    What it does not give: a way to shorten the attention itself. Its long-context growth that is left (batch 1:
    +200 us per token between position 64 and 1100) is the K tiles of key groups 28-32 and the V levels beyond the
    staged ones passing through one SM at 58 GB/s, plus about 50 instructions per key group and thread.

13g. **The address of a hot word is part of the step time** [measured, tools/padsweep.sh, tools/padsweep2.sh,
    tools/serve_long.py; built, s0047-s0050]. Found by accident: a control build that only allocated two more buffers
    was 12 us faster at batch 1 and 40 us slower at batch 8. Padding in front of single buffers (host code only, the
    kernel untouched) showed that only the counter array's address matters, and inside it:
    (a) the watchdog's abort word. A reader that finds the sentinel reloads its word until the data are there, and
    each attempt first loaded the abort word, the reload waiting for it: a second trip through the memory per
    attempt in every hand-over. With the word looked at on every 16th attempt only (LDH_AB): batch 1 in process 1442
    -> 1418 us (track A), 1471 -> 1458 (track B); the scored path locally (submit/poll, 64+1024) 1540 -> 1515 and
    1571 -> 1556; batch 8 track B 3880 -> 3854. HTTP: track A 1546 -> 1527 (s0048); track B 1559 -> 1566 / 4021 ->
    3979 (s0047: its batch-1 figure had been flattered by where the word lay in the server's process, see (b)).
    (b) the page of the counters. With (a) in place the batch-1 step still has two levels 14 us apart, and the level
    is the parity of address bits 12-15 and 17-21 of the page (sweep of single bits; bit 16 and the bits below 12 do
    not count); batch 8 has several levels over 40 us with another dependence. Same on both tracks and on every card.
    It follows the physical placement, not the virtual address: the same low 28 address bits in a slab of 256 MB
    instead of 4 MB kept batch 1 on its level and moved batch 8 by +35 us. So it cannot be fixed at build time, and
    the HTTP server's process need not be on the level the in-process timings are on. Built: each of the two kernel
    instances has its own counter array (mk_set_ctr), and the engine, when it starts, times a few greedy steps of
    fixed tokens with the array on each of 16 pages and keeps the fastest (api._place_counters, 2.4 s; pages and
    timings in stats()). The choice is reproducible (batch 8: the same page in every run on both tracks). In process
    batch 8 3779 -> 3748 (A), 3846 -> 3823 (B); HTTP batch 8 3914 -> 3870 (A, s0050), 3979 -> 3936 (B, s0049);
    batch 1 unchanged where the default page was a fast one already (HTTP 1527 A, 1566 -> 1560 B).
    What the bits are (an L2 slice? a memory channel?) is not known; what is known is the size of the effect of ONE
    4-KB page that every block polls and bumps: 1% of the step.
    (c) other one-page buffers. At batch 1 the router logits (1 KB), the post-attention stream and the attention
    output (one or two pages each) show the same levels, smaller: 5-10 us each on track A, 2-4 on track B (padding
    by pages from the environment, then the start-up timing: for the attention output 1388 against 1397 us by page).
    The buffers of many pages (layer input, q/k/v, hidden units) show nothing. The batch 1-2 instance now has copies
    of its own of the three and places each on the fastest of 16 pages: track A in process 1422 -> 1402, HTTP 1527 ->
    1516 (s0052, s0054); track B 1459 -> 1453, HTTP 1560 -> 1557 (s0053). 13 s at start-up.
    (d) what the timing can and cannot find. Choosing the batch-8 counter page among 64 instead of 16 was 20 us better
    by the timing and in generate() and 25-28 us worse over HTTP on both tracks (s0051, s0052); back at the 16 pages
    and addresses of s0049/s0050 the HTTP figures returned (3878 track A, 3950 track B; the three HTTP repeats of one
    evaluation spread over 30-80 us at batch 8). The timing runs generate(); the scored path is the served cohort,
    where the host also reads the progress word on the counters' page. A search over many candidates fits the
    timing's own path. Kept: 16 candidates.
    (e) limits. Placing three more buffers of the batch-1 kernel (hidden units, q/k/v, layer input: 5 us each when
    moved alone) did not add up: track A 1401 -> 1409, track B 1452 -> 1448 in process. Below about 8 us the outcome
    is the draw of the candidate pages a process gets, not the search over them.
    Not done: the three counters on pages of their own
    (moved alone, the attention counter was worth 9 us at batch 8 and the router counter 6 us at batch 1), because
    that needs their offsets at run time and the kernel has no register to spare.

13h. **One kernel, not two** [measured, tools/mk_two_kernels.cu.bak]. The kernel has no register to spare, and code
    for one role (attention blocks) changes the allocation of the other (main blocks): this is what stopped the split
    attention pair of 13d (correct, same bits, but phases A and C of every main block 2-4 us slower when its code is
    inlined, and 450 bytes of stack when it is a call). Launching the two roles as two kernels on two streams gives
    each its own allocation, and the tokens are identical, but the pair runs 20% slower than the single kernel
    (generate 1676 against 1402 us at batch 1, 4128 against 3797 at batch 8): hand-overs between two resident
    kernels cost about 5 us per layer more than inside one. Part of that was the order of the two launches (the
    attention/router kernel first puts the light roles on the first places, 13j): with the main kernel first batch
    8 is 3900 against 3794, batch 1 still 1641 against 1402. So the roles have to share a kernel and its registers,
    and the 8 pairs without an SM stay the open item at batch 8.

13i. **The register allocation of the batch-1 kernel is worth 5% of its step, and it is a draw** [measured; built,
    s0055]. The kernel uses all 128 registers of every thread, and what the compiler spills, it spills in the hot
    loops. Writing the kernel's body as an inlined function that the kernel calls (the same code, BODY_WRAP) changes
    the draw: the batch 1-2 instance then has 36/16 bytes of spill stores/loads on track B (68/84 without) and 68/84
    on track A (36/16 without). The instance with 36/16 is the fast one on either track: track B 1453 -> 1381 us in
    process, serve path 1550 -> 1481, HTTP 1557 -> 1473 (s0055); track A with the wrapper 1402 -> 1434, so track A
    keeps the plain form (s0056: unchanged, 1516). Same digests in every form. Found as the control of an experiment
    that failed (the attention/router blocks' path as a function of its own inside the one kernel: 320 bytes of
    stack around the call, 1669 us, and on track B another fusion of the norm's sum of squares, 13e (c)).
    What it means: every comparison of two kernel sources in this run carried this draw (the journal's "register
    picture"), 13 us up to 70 us at batch 1; a mechanism was only believed when its control build had the same spill
    lines or when it was host-side. And the two tracks' batch-1 difference had been mostly this draw: with the same
    allocation track B (float32 stream, no BF16 conversions in the norms) is 20 us faster than track A, not 50 slower.
    Not done: a search over neutral rewritings for a better draw of the batch > 2 kernels (56/40 on track A, 64/88 on
    track B) and for anything better than 36/16.

13j. **A block's index is a place in the machine: the light roles belong on the last 24 blocks, and the last main
    blocks are always last** [measured, tools/bt.py (profiling build), tools/mk_rolemap.cu.bak (the engine)]. Main
    blocks 112..127 reach every phase boundary 3.5-5 us after the other 112 (batch 1; more at long contexts), the
    same lag in every phase and step, and every barrier waits for the last block; not the counters' place, not the
    idle blocks' polling (both tried). And in the engine itself, moving the 24 attention/router roles from physical
    blocks 128..151 to 112..135, 104..127, 64..87 or 1..24 changes the step from 1402 / 3744 us (batch 1 / 8) to
    1405 / 4176, 1447 / 3962, 1453 / 4058, 1467 / 4498: up to 20% at batch 8. When the memory is busy, blocks are
    served in an order tied to their index, so the bandwidth-critical roles must not sit on the last places; the
    engine had this right by luck. A smaller share of the expert phase for the last 16 main blocks does not help
    (equal shares 1405 / 3754, any unequal split 13-20 us worse at batch 1): they are late because they are served
    last, and the phase would not end earlier without them. Rotating the roles within the last 24 blocks is worse
    at batch 8 (3903-4492).
    tools/bench22.cu (every block of a grid streams 16 MB of cold memory): the total rate is 2.35 TB/s with 16
    blocks (147 GB/s per SM), 6.44 with 64, 7.14 with 128, 7.36 with all 152; the "58 GB/s per SM" of items 13d and
    13f is therefore one SM's share when 128 stream, and an SM that streams while the others wait gets up to 2.5
    times that. The blocks finish in classes by index (12 blocks among the first 17, about 60 among the last 50,
    in pairs and fours that repeat irregularly): a fixed inequality of service, but not the engine's late group.
    [open] Why main blocks 112..127 as a whole are the late ones in the engine's profile.

## The extension (2026-11-15): batch 8 on track A

15a. **A conditional load in the mma operand decided the "register draw" of the batch > 2 kernel** [built, s0060].
   `xget` was `lane < Ls ? xf[chunk * Ls + lane] : 0`. Disassembly (tools/sass.sh, sass_seg.py: runs of HMMA with the
   instructions in and between them): in fast builds the load was predicated (5 instructions per tile), in others a
   branch was left around every load with the shared-memory address recomputed inside (20-23 per tile; a 16-tile run
   of the expert phase 87 -> 346 instructions); which one a phase got changed with unrelated code elsewhere. Now
   every lane loads: a lane without a column reads a zero word in front of the buffer with stride 0 (2 instructions
   per tile in every build seen). Batch > 2 kernel only (the batch 1-2 kernel had the predicated form and got 47 us
   slower with the new one). In process 3746 -> 3689; sealed 3898 -> 3842. This is a large part of what item 13i
   called the draw: it was never only spills (builds with fewer spill bytes were slower more than once).
15b. **A warp's memory operations complete in the order issued, exchanges and strong loads included** [built, s0061].
   Seen three ways at batch 8: a main block's two sentinel exchanges issued in front of its attention delayed the
   attention by 1.9 us; a counter load behind them returned 1.9 us later; requesting tiles before a phase's result
   exchanges delayed the exchanges by the tiles' 2.3 us (FPRE_EARLY, +26 us). An exchange whose result nobody reads
   does not stop the warp's instructions, but its next load. Consequences used: the thread that polls a hand-over
   counter is in warp 6, not in the leader warp 0 (batch 1 1403 -> 1389 in process; sealed 1531 -> 1514); nothing
   that others wait for is issued behind a batch of tiles. Not usable: plain stores instead of exchanges for values
   (hidden units: +25 us; residual rows: -4 to -10, not taken: a plain store has no order against the next sentinel).
15c. **Batches of plain C loads can be dissolved by the compiler; volatile asm pins them** [built, s0062]. Past the
   staged key groups the attention loads K and V tiles in batches of 8 (one trip to memory per batch). In the batch
   1-2 kernel part of that had been moved down to the uses; in the batch > 2 kernel the batching had measured
   "nothing" twice while the profiling build showed the V mix going from 3.06 to 1.47 us. With `ld.global` as
   volatile asm for the batch loads (KV_ASM; the key groups staged in shared memory keep plain loads: a global load
   of a shared address faults): batch 8 -30 to -55 us in every 128-position chunk (attention blocks and main-block
   pairs together; on the attention blocks alone +15: the other 8 pairs then trail), batch 1 -7 to -25. Sealed
   3829 -> 3784 and 1514 -> 1494. Every tile load of the kernel as volatile asm (TLOAD_ASM): +4 to +10, off; the
   staged K/V tiles pinned as well (KV_STG_ASM): nothing.
15d. **The scored run is long-context** [measured]. tools/pos_time.py, us per step by 128-position chunk at batch 8
   (s0061 build): 3737 3790 3835 3866 3906 3925 3953 3962 3998; batch 1: 1380 1392 1458 1485 1550 1559 1584 1591
   1617. 85% of the scored positions are beyond 256, where a sliding layer has its full 33 key groups and the 14
   full-attention layers keep growing. tools/gen_long.py (prompt 64 from tools/ids.npy, 1024 greedy tokens, in
   process) and tools/serve_long.py (the same through submit/poll) are the workloads to decide on; free-running
   text from arbitrary tokens (pos_time) and positions under 330 (test_engine --time) gave a different sign twice.
15e. **Sequential strong loads of a copy** [built, s0063]. The copy of the attention output loaded, tested and stored
   one sequence after the other (8 trips to memory per thread at batch 8, all still to come when the last output
   arrives). Four loads in flight, in registers; a word that holds the sentinel is waited for alone and the later
   ones are requested again together. In process 3805 -> 3763; sealed 3784 -> 3735. Width 2: +18; width 8: +40 (16
   more registers than the phase has); with the values passed through an array to a wait function: +77 (stack).
   The same for the norms' four loads: +33 (four in flight), +7 (two): the norm runs with all 16 staging registers
   full; the sixth attempt at this. Later (s0070): every pending word of a batch is requested again in each attempt
   instead of waiting for one alone, -13 (sealed 3448 -> 3426); the same in the down phase's copy: +36.
15f. **One fragment copy for the whole down phase** [built, s0064]. Each round copied its 8 items' hidden-unit
   fragments with strong loads between two half batches of tiles (8 tiles, copy, 8 tiles): the tile stream stopped
   once per round, 3.2 us per round where the expert phase needs 2.65. Every sequence is a member of exactly 6 units
   and of the shared expert, so the phase needs 8 B column slots whatever the routing: 64 slots of 32 chunks x 4
   lanes = the 64 KB fragment buffer at batch 8. The copy runs once (thread = sequence, chunk, half: 8 loads, 4 in
   flight), the rounds stream 16 tiles per warp, and a lane's operand is its sequence's slot or the zero word with
   stride 0. Two byte tables from the routing's union stage (rank of a unit in each member sequence; unit of each
   rank). In process 3763 -> 3621; sealed 3735 -> 3577. (F_PIPE, the copy requested together with the tiles each
   round, had been +69 to +169 us: the copy's own code stood between the mma and the tile requests.)
15g. **Timing studies in the normal build** [measured; `-DWHATIF=bits`, results wrong on purpose]. Bit 0 forces a
   fixed routing (36 units, so that garbage cannot change what is streamed); every other bit removes one piece of
   work. Batch 8, scored workload, us per step against forced routing alone (3452 on the s0064 kernel): no routing
   computation -206 (candidate extraction -110, merge of the 12 candidates -44, union -53); attention without cached
   keys -232 (scores -66, exponentials and weights -37, V mix -96); the down phase without its one copy -121; no
   copy of the attention output -89; no sentinels at the start of the expert phase -81; norms loading one pair of
   sequences instead of four -45. On the s0063 kernel: the down phase's copies after the first round -211 (this
   became 15f). Each figure carries the allocation of its own build (spill lines differ), +-30 or so.
   What the profiling build had suggested and the scored workload refuted: the 8 main blocks with an attention pair
   as the thing everybody waits for (HOSTC: their output projection computed by another main block: -1.4 to -2.0 us
   per layer in the profile, -60 us in process at positions under 256, +10 on the scored workload; a weighted split
   of the expert phase in their favour, 11:12 to 3:4: nothing).

15h. **New code written for clarity cost registers; the plain rewrites paid** [built, s0066, s0067]. The one-copy
   code kept the four source addresses of a batch in an array of pointers for its sentinel path: recomputed where
   they are used, the spill line of the batch > 2 kernel went from 176/52/88 to 160/24/24 bytes and the step from
   3616 to 3576 in process (sealed 3571 -> 3527). The routing's merge read "its own" candidate from two local arrays
   by lane index, which put them on the stack (14 stack accesses per thread): read from shared memory instead,
   frame 160 -> 144 bytes, 3589 -> 3577 (sealed 3527 -> 3499). tools/sass_spill.py lists what is left on the stack:
   the attention's per-key-group scores (a 7 x 4 array indexed by the round), which measured no better in shared
   memory (EXS: +14 at batch 8, +30 at batch 1).
15i. **The 8 main blocks with an attention pair, hosted (HOSTC)** [built, s0068]. At batch 7-8 a pair block comes out
   of its attention when every other main block has long had its output-projection tiles; its own output projection
   then made every block wait. Block 24 + o (first block 8 + o; which 8 blocks host is worth 10 us, the last main
   blocks being the worst: journal) now computes block o's 16 rows first (same tiles, fragments and sums) and
   its own second; the owner hands its rows over with its sentinel (it waits for the exchange to return instead of
   reading back, the host waits for the sentinel before it stores) and reads them back. Measured +10 us on the
   scored workload while the down phase was where everybody waited (before 15f), -63 after (3576 -> 3513 in process;
   sealed 3499 -> 3460). A lesson about order: a negative result is a statement about the kernel it was measured
   on; of five earlier negatives measured again after 15f, this one had changed sign.

15j. **Where a batch of loads pays and where it does not** [measured]. Loads of a hand-over are strong loads that
   may find the sentinel. Requested together they pay when the data is already there: the copy of the attention
   output after its wait (15e), the down phase's fragments (15f). They cost when the data is still arriving: the
   norm starts while the slowest blocks are writing their rows; one after the other, the later pairs of sequences
   find their values, together they all find the sentinel and each costs a wait call and a second load (seven
   attempts; the last, NORM_B=3, with the same spill line as the base: +27 us). For the same reason the down phase's
   copy waits for one word alone (all pending words per attempt: +36) while the attention-output copy, whose wait is
   short, asks for all of them (-13).
15k. **A counter is not a hand-over** [measured]. E_CTR: no sentinels for the expert phase's hidden units; every warp
   bumps a counter after its own exchanges and the readers wait for the counter, then read with strong loads. Same
   digests, -17 us, and 4 mismatches in 398 batch-8 repetitions of tools/flake.py (none in 199 repetitions of each
   of the submitted builds). The order of one warp's atomics at the L2 does not make "counter seen" imply "data
   seen" when 32 lanes exchange and one lane bumps. Every counter left in the engine is an early signal ("the
   sentinels are in place"); the data is always waited for itself. A change to a hand-over needs the 200-repetition
   run, not the digests.

15l. **Where a batch-8 layer goes, measured without memory operations** [measured; tools/bt_clk.py, build with
   `MK_EXTRA=-DMK_CLK`: the normal kernel with %clock64 stamps of every main block's phase boundaries in layer 8].
   Final kernel, teacher-forced text, position 700, us from a block's own A start (median of 128 blocks): A done 6.8;
   every pair begun 11.2; output projection done 19.9 (blocks with a pair 21.2, their hosts 21.7); D start 21.0;
   router seen 25.9; routing done 29.6; expert phase done 54.5; down phase done 70.2 (position 150: 68.0, the
   attention and output projection 5.2 instead of 8.7). So: 40 us in the two streamed phases (about 35 us of memory
   time at 7.4 TB/s for 41 units) and 30 us in front of them: six hand-overs through memory at 2-2.5 us from the
   writer's exchange to the reader's value (residual rows, q/k/v, attention output, post-attention rows, router
   logits, hidden units), the attention (5-9 us of instructions and K/V trips) and the routing (3.7). The cycle
   counter has different bases on different SMs: only intervals inside one block mean anything.
   Inside the phases (same tool, four more stamps): the norm of phase A takes 4.6 us and that of phase D 4.3 (each
   waits for the rows of the block that finishes last, then loads its pairs of sequences one after the other);
   between the routing and the first expert mma lie 3.8 us (2.3 the first tiles' trip, the rest the sentinel
   exchanges: issued for every slot by idle blocks instead, HS_AUX, +71 us; by a counter instead, E_CTR, not
   strict); the down phase's fragment copy takes 2.7 us (it waits for the last block's hidden units) and its rounds
   then run at the memory's rate (5.8 rounds in 12.0 us).

15m. **The top-6 selection of the routing was 5% of a layer at batch 8 and 9% at batch 1, in warp-level chains**
   [measured; stamps inside `route_local`, study source tools/mk_rclk.cu.bak, `MK_CLK_X=3 python tools/bt_clk.py`].
   Every main block computes the routing itself (no hand-over), so its time is on every block's path. Batch 8, per
   layer: load of the logits 0.64 us, candidate stage 1.73 (four sigmoids per thread, six rounds of warp maxima),
   merge of the 12 candidates per sequence 0.57, union over the batch 0.51. Batch 1: load 0.54, candidates 0.71,
   merge of 48 candidates on one warp 1.25, rest 0.07.
   - A barrier of a 16-warp block costs about 0.1 us here: the union had five in a row, two went by moving its byte
     table out of the candidates' way (s0072, -14 us per step sealed). Replacing a third barrier by seven ballots per
     warp is worth nothing.
   - A round of "warp maximum, lowest lane, kill" is bound by its chain of five shuffles and a ballot, not by the
     instructions around it: sorting each thread's four scores first (no local maximum, no slot shuffle: 50 of 340
     instructions less) gave 2.5 us per step, a fifth of the count's promise. Counting ranks through shared memory
     has no chain: the thread's best score is ranked among the 32 of its warp, and since a thread's j-th score of a
     thread with rank R has at least R + j scores in front of it, only 4 + 4 + 4 + 3 + 2 + 1 = 18 scores can be among the
     warp's six best; 18 lanes rank them (s0073: candidate stage 1.73 -> 1.52 us, -14 us per step in process, -7
     sealed).
   - The same bound prunes the batch-1 merge from 48 candidates (two per lane, 96 comparisons) to 27 (one per lane):
     s0074, merge 1.25 -> about 0.9 us, -16 us per token in process (-18 in the sealed step path, -6 over HTTP).
     The old merge's tie order (lower key, key = 2 * slot or 2 * (slot - 32) + 1) is part of the bits; the bound holds
     in it for every warp but the one whose slots straddle the two key ranges, which is kept in full.
   - The union's last pass (unit tables: 0.55 us per layer, 8 warps walking 8 sequences per expert in use) on all 16
     warps with 4 sequences per thread: s0075, -6 us per step in process, 3408 -> 3397 sealed.
   - The merge stage kept the next stage waiting for a sum and a division it did not need: the union's table bytes
     written by the ranking lanes themselves, the weights computed by the idle warps 8 + b during the ballots:
     s0076, -3 us per step in process, 3397 -> 3394 sealed.
   - The four sigmoid divisions each compile to a branch region (range test, MUFU.RCP + Newton step, or a call);
     the fast path written out behind one test is bit-equal for all 2^32 inputs (tools/sig_rcp.cu) and worth 1.5 us:
     not taken.
   - One warp per sequence with 8 scores per thread (no merge, one barrier fewer; RT8): +13 us per step. The stage is
     bound by what ONE warp has to do; idle warps do not help it.
   - Not gains, same bits: the tie terms out of the rank loops with one warp match (RMATCH: nothing at batch 8, +30
     us at batch 1 from the allocation); the 18 candidate stores without branches (TRIBF: +18 us, allocation); the
     batch-1 weights from shared memory (RTSW: +54 us, allocation). A `*_sync` warp intrinsic with a full mask under
     a condition that only some lanes pass never returns: that variant held a card until its job's time limit.
   Allows: any later change to the selection has a 3-minute measurement of its four stages. Does not allow: taking
   the instruction count as the cost (twice here the gain was a fifth to a half of what the count said).

## Ideas measured and not in the default build

(The extension's experiments are behind switches in `engine/mk.cu`, each with its verdict in a comment, and in
`tools/mk_rts_s.cu.bak`, `mk_normb3.cu.bak`, `mk_hostc_v5.cu.bak`, `mk_afirst_v1.cu.bak`; the journal has every
measurement. Kept off: AFIRST, F_PIPE, MATTN_TAIL, MATTN_CPRE, HSTG, NORM_B, KV_DB, VPAD, EXS, TLOAD_ASM, KV_STG_ASM,
FPRE_EARLY, APRE_EARLY, XPLAIN, HS_PLAIN, HST_PLAIN, HS_EARLY, E_SETTLE_LATE, E_CTR (not strict), ES_WM/ES_WO.)

14a. **The experts can be predicted right after the attention, but not used earlier.** [measured;
    tools/cert_route_study.py, tools/spec_agree.py, tools/mk_spec.cu.bak] z = R' x + M a with ONE BF16 limb of
    M = R' W_o (2 MB per layer) gives the exact top-6 set in 99.83% of 11520 layer-tokens, two limbs in 100%; with
    error bounds from the same tiles the set is certified in 55.6% (one limb) and 99.62% (two limbs). In the engine
    the predicted unit list differs from the real one in under 0.7% of the layers at batch 1 on track B (1-2% on
    track A). Using it to request expert tiles before the real router is done does not pay: the 16 router SMs are
    the critical path between attention and expert phase at batch 1 (their router tiles waiting in registers are
    worth 159 us per step), the predictor competes with the real router for them, and predictor + list take as long
    as the main blocks' own C + D phases. Step 1500 -> 1899 us with the predictor running and unused.

13a. **The expert stream is not slowed by where the experts lie.** [measured, tools/bench18.cu, 80 GB buffer, the
    engine's slots and barriers] 6 random 4 MB units per iteration: 6.42 TB/s against 6.41 contiguous; 38 units:
    7.32 against 7.30; storing the units striped over a 1 GB layer changes nothing (by row group) or costs 3-7% (by
    tile). Storing the down matrices block-major (a block's slices of all experts adjacent) changed nothing in the
    engine either. Allows: nothing to gain from the layout of the 76 GB of experts. It also shows that the batch-8
    expert phase (25-27 us for 154 MB) is 4-5 us of routing tables, sentinels and epilogue around a stream that
    runs at the card's rate.
13b. **A different down phase for batch > 2** (compact copy of the member sequences' hidden units, barrier-free
    stream of half-items, one fold): the phase itself 22.6 -> 20 us per layer, the step 4041 -> 4203 us. Not kept
    (tools/mk_downstream.cu.bak; notes/journal.md).

14. **The expert choice is known earlier than the model's code computes it.** The router ranks
    sigmoid(R . norm2(x_mid)) with x_mid = x + W_o a (a = the attention output). The norm divides by one positive
    number per token and the sigmoid is monotonic, so the top-6 SET is the top-6 of z = R' x_mid with R' = R diag(w2),
    and z = R' x + (R' W_o) a is linear in a. R' x can be computed while the attention runs (x is the layer's input),
    and M = R' W_o (256 x 4096 per layer, 4 MB) can be built at load time. So the experts of a layer could be
    decided one hand-over after the attention instead of after C (output projection), the norm and the router: about
    4.5 us earlier per layer at batch 1 on today's timeline (33 us). Only the set is early; the routing WEIGHTS need
    the norm's scale and are needed only in the down phase.
    What it would allow: expert tiles requested during C and D (they wait 2-5 us for nothing today), routing off the
    main blocks' critical path (2.7 us at batch 1, 7.9 us at batch 8), and one continuous tile stream after C.
    What it does not allow by itself: track A (the data flow differs), and M is a product, so it has to be stored to
    more than float32 precision (several BF16 limbs) for the result to stay "identical in exact arithmetic"; near-ties
    will flip as with any other summation order (finding 3).
    [built as stage A, tools/mk_er1.cu.bak, -DEARLY_ROUTE; measured] Numerically right (9216 of 9216 sets equal to the
    float64 formula), but slower at batch 1 (1635 against 1577 us): M is 8 MB per layer as 4 BF16 limbs and the 16
    router SMs read it at about 1 TB/s, most of it after the attention; and 4 limbs of M are a 3e-11 approximation,
    not exact arithmetic. Off by default.
    Cheaper variant, valid on track A because it only schedules loads: PREDICT the set from R' x alone (drop the
    attention term) and prefetch. [measured, tools/spec_study.py, HF BF16, 8 x 384 tokens] The real 6 experts are
    among the predicted top-k with recall 82.6% (k=6), 90.2% (8), 93.5% (10), 95.3% (12), 97.1% (16); 76%@6 in the
    first MoE layer, 90-95%@6 in layers 32-40. Batch 8, union of the predicted top-6: 85% of the units used are
    covered and 84% of what would be fetched is used.
15. **There is no register-free L2 fill.** [measured, tools/bench13.cu] `prefetch.global.L1/L2` gain 0.3 us on a
    2.2 us read of 16 tiles and loads into a dead register gain nothing. A look-ahead therefore needs real loads by
    warps that have free registers and no critical load queued behind them (a warp's loads return in issue order):
    in practice the 20 SMs that do not stream weights at batch 1.

## Where to continue (after the extension; ranked by the timing studies of item 15g)

00. **After s0076 (1483 / 3394): what the routing still holds** (15m). Per layer at batch 8 (3.18 us, measured on
   the final source): load and candidates 2.15, merge 0.37, ballots 0.11, the union's last pass and its barrier 0.54; at batch 1: load 0.54, candidates 0.71, merge 0.81.
   a. The four sigmoids per thread (about 100 instructions of the candidate stage at batch 8): rank the raw logits
      and take the sigmoid of the candidates only. This needs (i) the sigmoid as compiled (`1/(1+__expf(-z))`) to be
      monotone in z: testable exhaustively over all 2^32 floats in seconds, not done; (ii) the seventh score per
      warp as a guard (22 candidates instead of 18: a tie between the sixth and the seventh rounded sigmoid falls
      back to the present path, in every block alike); (iii) ties of the rounded sigmoid broken by the expert id in
      the merge. (i) is settled: tools/sig_mono.cu, 0 inversions among all 4.28e9 neighbouring finite floats; and
      55% of neighbouring floats in 1 <= |z| < 8 have the same rounded sigmoid, so (iii) is needed. But the count
      is not the cost in this stage: RMATCH took 160 of 390 instructions out of its two rank counts for no gain,
      and its parts measure 0.41 us (sigmoids, sort), 0.57 (rank among 32, 18 stores), 0.28 (rank among 18) in a
      study build. Read the stage's disassembly (tools/sass.sh) before rewriting it again.
   b. (Done or measured: the union's last pass on 16 warps is s0075; the batch-1 weights from shared memory, RTSW,
      cost +54 us through the batch 1-2 kernel's allocation; the batch-8 merge without its sum and division is s0076.)
      Open: the four barriers of the batch > 2 routing (0.1 us each), of which only the one behind the candidates
      separates work of all 16 warps.
   c. The 2.8 us between routing and first expert tiles at batch 1 (sentinel exchanges and the tiles' trip) is the
      next piece of the batch-1 layer that the cycle counter shows and nothing has shortened.

0. **Batch 8 was at 3426 here (s0070): 1.108x of the trial figure, 44 us inside the goal.** The streaming is at the card's rate (15g, tools/bench23.cu); what
   is left is the chain in front of the routed stream. In the order of what a timing study says is there:
   a. the routing computation in every main block (200 us per step): the candidate extraction is six rounds of a warp
      maximum over 128 scores per warp after four sigmoids per thread; ranking the raw logits (the sigmoid is monotone;
      ties of the rounded sigmoid need a fallback to stay bit-identical) and fewer barriers in the union are worth
      perhaps 40;
   b. the attention at long contexts (230): its stages are bound by instructions per key group now, not by trips to
      memory (two V batches in flight: -5 at batch 8, +38 at batch 1);
   c. the copies (down phase 120, attention output 90) and the expert phase's sentinels (80): each is strong loads or
      exchanges that the tile stream of the same warp has to wait for, and each contains one trip to memory that no
      protocol removes. A hand-over by counter instead of sentinels was tried for the expert phase's hidden units
      (E_CTR): -17 us on the final kernel, and NOT strict: 4 mismatches in 398 batch-8 repetitions of tools/flake.py
      (every warp bumps after its own exchanges; a reader still got the counter before some exchange had landed).
      Finding C11 of the Qwen3.8-27B run says what a counter hand-over needs on this card (release and acquire, not
      the order of one warp's atomics).
   A profiler that does not load from memory (%clock64 per block, finding C12 of that run) would have saved this run
   most of its wrong turns at batch 8.

(The list below is the one written at the end of the first 24 hours; items 1 and 2 are answered in part by 15a, 15g.)


1. **Batch 7-8: the 8 attention pairs without an SM** (13d, 13h). Splitting a pair (scores and softmax on its main
   block, V mix on an attention block that is done early) is written and gives the same bits (tools/mk_split_v2.cu.bak);
   it takes the main blocks' 3 us of post-attention work off the path everyone waits on (about 150 us per step, which
   is the distance to the general engine at batch 8). It needs the two roles compiled with separate register
   allocations in ONE launch without a call; the three ways tried (inlined, called, two kernels) each lose more than
   the split gains.
2. **The register allocation** (13i). Three allocations of the batch-1 kernel were seen, 50-290 us apart; nothing says
   36/16 bytes of spills is the best there is, and the batch-8 kernels were never searched. A neutral rewriting that
   frees a few registers in the hot loops (the staging array is 64 of the 128) would also end the "any change is a
   draw" problem that cost this run most of its negatives.
3. **Hand-over latency at batch 1** (six per layer, about 1 us each; 13f, 13g). Each counter on a page of its own
   chosen at start-up (needs their offsets at run time: two registers), and fewer barriers in the attention.
4. **The order in which blocks are served** (13j): why main blocks 112..127 are always last, and what the phase rates
   (3.8 TB/s in the expert phase at batch 1 against 7.1 for 128 blocks streaming freely, tools/bench22.cu) leave on
   the table.
5. Track B at batch 8: its float32 stream costs 60 us per step there and wins 20 at batch 1; nothing forces one
   container for both kernels of a track.

