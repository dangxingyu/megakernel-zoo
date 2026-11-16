# s4-flint-b200-a2: Flint on GB200

Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 77 submissions so far, 76 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 629
- 1.10x the fastest general engine: 1792
- vLLM 0.33.0 (general engine): 1971

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: First megakernel: one launch per step, tensor-core tile GEMV (track A)

- status: **valid**, step_us_b1 = 2679, at 0.77 agent-hours, category structure
- hypothesis: track A: formats as the checkpoint and the HF BF16 path (BF16 weights re-laid into mma tiles, BF16 activations, f32 norms/router/scores; logits returned as unrounded f32). One persistent kernel per step with 293 grid barriers; local profile says overheads (attention 665us, routing 560us, barriers ~500us) dominate, so I expect valid gates but about 2700 us at B=1 and 7200 us at B=8, i.e. slower than vLLM; this submission is to see the gates on the real reference.
- evaluator: valid: 2679 us/step at B=1 (MBU 0.236), 7425 us/step at B=8

### #2 s0002: OPERATOR: s0001 again, unchanged (evaluated again under the HTTP headline, harness 418e7ad)

- status: **valid**, http_step_us_b1 = 3003 (B=8: 7816), at 0.90 agent-hours, parent s0001, category structure
- hypothesis: OPERATOR re-evaluation of s0001 (evaluated again under the HTTP headline, harness 418e7ad); source unchanged, sha256 851717f0c9e66d18. The agent's hypothesis for s0001 was: track A: formats as the checkpoint and the HF BF16 path (BF16 weights re-laid into mma tiles, BF16 activations, f32 norms/router/scores; logits returned as unrounded f32). One persistent kernel per step with 293 grid barriers; local profile says overheads (attention 665us, routing 560us, barriers ~500us) dominate, so I expect valid gates but about 2700 us at B=1 and 7200 us at B=8, i.e. slower than vLLM; this submission is to see the gates on the real reference.
- evaluator: valid: 3003 us/step at B=1 and 7816 at B=8 over HTTP (offline (step through the cohort adapter)); in process 2676 us/step at B=1 (step, MBU 0.236)

### #3 s0003: 512-thread blocks, tiles staged in registers, L1-bypass loads, submit/poll serving

- status: **incorrect**, no score, at 1.64 agent-hours, parent s0002, category structure
- hypothesis: track A: formats as the checkpoint and the HF BF16 path (weights re-laid into mma tiles, BF16 activations, f32 norms/router/scores, logits unrounded f32). Kernel rebuilt around measured card facts (one outstanding load per warp, spills and smem staging are expensive): 16 warps x 128 registers with the next phase's tiles in registers, counters instead of a global barrier, ld.global.cg readers instead of fences, and a submit/poll path that runs a whole cohort (prompt + 1024 tokens) in one launch while tokens are polled. Local: 1993 us/step and 2010 us/token in the device loop at B=1, 6.9 ms at B=8. Expect valid, http_step_us_b1 about 2050 (still ~4% slower than vLLM), b8 about 7000.
- evaluator: incorrect: failed tokens
- gates that failed: tokens

### #4 s0004: Track B precision mode: float32 residual stream, hi+lo router input (same kernel as s0003)

- status: **valid**, http_step_us_b1 = 2194 (B=8: 7294), at 1.77 agent-hours, parent s0003, category correctness
- hypothesis: track B: weights in the checkpoint's BF16 format (re-laid into mma tiles); differs from the HF BF16 path only in precision that is lossless in exact arithmetic: the residual stream is kept in float32 instead of being rounded to BF16 at every add, sums that feed float32 code are not rounded, rotary tables are float32, and the router takes its input as hi+lo BF16 operands so the top-6 is chosen on float32-accurate logits. Same kernel and speed as s0003 (which failed only the tokens gate at 91.6%). On my own reference top-1 agreement rises from 89.9% to 91.6% (HF BF16 90.2%), so I expect the tokens gate to pass at about 93.5%, rel-L2 median about 0.06, and the first HTTP numbers: about 2190 us/token at B=1 (0.90x vLLM) and 7250 us/step at B=8.
- evaluator: valid: 2194 us/step at B=1 and 7294 at B=8 over HTTP (online (submit/poll)); in process 2013 us/step at B=1 (step, MBU 0.314)

### #5 s0005: Attention mix by dim-chunk warps, parallel routing finals, K/V staging levels, head on 152 SMs

- status: **valid**, http_step_us_b1 = 1968 (B=8: 6273), at 2.63 agent-hours, parent s0004, category attention
- hypothesis: track B: same formats as s0004 (BF16 weights in mma tiles; residual stream float32, router input hi+lo BF16). Changes since s0004, all structural: attention V-mix reorganised so a warp holds one dim chunk over the key groups of one parity (partial sums 8 KB instead of 64 KB of shared memory), softmax weights handed over as B operands, K and V of contexts over 256 staged in shared memory while the block idles, leader warps bump the counter themselves; router top-6 finals for all sequences in parallel; split mma chains; five leader warps in the q/k/v phase; the head streamed by all 152 SMs. Local: 1834 us/token (ctx 64-320), 1964 us/token on the HTTP workload at B=1, 6200 at B=8. Expect http_step_us_b1 about 1965 (1.00x vLLM), b8 about 6250, gates as s0004 (tokens gate about 93%).
- evaluator: valid: 1968 us/step at B=1 and 6273 at B=8 over HTTP (online (submit/poll)); in process 1810 us/step at B=1 (step, MBU 0.349)

### #6 s0006: Track A build of the s0005 kernel (pure BF16 arithmetic, 2 accumulators per chain)

- status: **valid**, http_step_us_b1 = 1901 (B=8: 6081), at 2.72 agent-hours, parent s0005, category correctness
- hypothesis: track A: every number format as the HF BF16 path (BF16 weights re-laid into mma tiles, BF16 activations and residual stream with HF's rounding points, f32 norms/router/scores; logits returned as unrounded f32). Same kernel as s0005 with the track-B precision switched off. The tokens gate is a lottery for BF16-faithful arithmetic (s0001 93.6%, s0003 91.6%); on my own reference three summation orders (1, 2, 4 accumulators per mma chain) give top-1 agreement 88.6%, 91.3%, 89.5% (HF BF16 90.2%), so I submit the 2-accumulator order and expect about 93.5% on the hidden set (pass), http_step_us_b1 about 1940 (track A is 1.5% faster than track B: no hi+lo router pass) and b8 about 6200.
- evaluator: valid: 1901 us/step at B=1 and 6081 at B=8 over HTTP (online (submit/poll)); in process 1757 us/step at B=1 (step, MBU 0.359)

### #7 s0007: Shared-memory reservation 224 -> 128 KB: L1 left for global loads

- status: **valid**, http_step_us_b1 = 1782 (B=8: 5245), at 3.14 agent-hours, parent s0005, category memory
- hypothesis: track B: formats that differ from the checkpoint's: residual stream in float32 and router input as hi+lo BF16 (as s0005); weights BF16 as stored. Change: the block's shared-memory reservation goes from 224 KB to 128 KB (no V tiles staged in shared memory, K stage 20 -> 12 key groups; also expert gate/up in 16-tile batches). tools/bench12.cu: a batch of 16 tile loads takes 2.27 us when 196+ KB are reserved and 0.76 us at 128 KB or less, because L1 and shared memory share one pool. Expect http B=1 1968 -> about 1855 us (1.06x vLLM), B=8 6273 -> about 5300 us; gates as s0005 (tokens about 95%).
- evaluator: valid: 1782 us/step at B=1 and 5245 at B=8 over HTTP (online (submit/poll)); in process 1634 us/step at B=1 (step, MBU 0.386)

### #8 s0008: Track A build of the s0007 kernel (128 KB reservation, 16-tile expert batches)

- status: **valid**, http_step_us_b1 = 1685 (B=8: 4796), at 3.21 agent-hours, parent s0006, category memory
- hypothesis: track A: no format differs from the checkpoint's or from what the general engines use (BF16 weights as stored, BF16 activations and K/V, float32 where the Hugging Face path uses float32). Same kernel as s0007 without the track-B precision mode; 2 accumulators per mma chain (the summation order whose top-1 agreement on my own reference is 90.8%, where Hugging Face BF16 has 90.2%). Expect http B=1 about 1720 us (track A was 3.5% faster than B in s0006 vs s0005), B=8 about 5100 us; tokens gate about 93% against the 91.9% limit, with the usual near-tie risk.
- evaluator: valid: 1685 us/step at B=1 and 4796 at B=8 over HTTP (online (submit/poll)); in process 1535 us/step at B=1 (step, MBU 0.411)

### #9 s0009: Race-free hand-overs (atomic writes), batch-8 routing and attention placement

- status: **valid**, http_step_us_b1 = 1772 (B=8: 4158), at 5.15 agent-hours, parent s0007, category sync
- hypothesis: track B: formats that differ from the checkpoint's: residual stream in float32 and router input as hi+lo BF16 (as s0007); weights BF16 as stored. Change: (1) every value another block reads is written with atomicExch and every writing warp bumps the hand-over counter itself; with plain stores about 8% of 600-token generations differed between runs (a reader could see the counter before the data), with this 0 of 108. (2) batch 8: the 8 attention pairs beyond the 24 aux blocks run on main blocks while those wait, and the router top-6 handles all sequences at once. (3) 2 accumulators per mma chain (new summation order). Expect http B=1 about 1750 us (s0007: 1782), B=8 about 4100 us (s0007: 5245, vLLM 3796); tokens gate about 94% (my reference: 93.2% against 91.1% for the s0007 order).
- evaluator: valid: 1772 us/step at B=1 and 4158 at B=8 over HTTP (online (submit/poll)); in process 1633 us/step at B=1 (step, MBU 0.387)

### #10 s0010: Track A build of the s0009 kernel (race-free hand-overs, batch-8 changes)

- status: **valid**, http_step_us_b1 = 1780 (B=8: 4235), at 5.19 agent-hours, parent s0008, category sync
- hypothesis: track A: no format differs from the checkpoint's or from what the general engines use (BF16 weights as stored, BF16 activations and K/V, float32 where the Hugging Face path uses float32). Same kernel as s0009 without the track-B precision mode, 2 accumulators per mma chain; its outputs are bit-identical to s0008 on my four test workloads, so the gates should repeat s0008 (tokens 93.07%). Expect http B=1 about 1710 us (s0008: 1685; the atomic writes cost about 1.5%), B=8 about 4250 us (s0008: 4796) from the attention placement and the parallel router top-6.
- evaluator: valid: 1780 us/step at B=1 and 4235 at B=8 over HTTP (online (submit/poll)); in process 1608 us/step at B=1 (step, MBU 0.393)

### #11 s0011: Two kernel instances (batch <=2, >2), down-phase loop of s0008 restored

- status: **valid**, http_step_us_b1 = 1745 (B=8: 4152), at 5.43 agent-hours, parent s0009, category structure
- hypothesis: track B: formats that differ from the checkpoint's: residual stream in float32 and router input as hi+lo BF16 (as s0009); weights BF16 as stored. Change: the kernel is compiled twice, for batches of 1-2 and for larger ones, each without the other's code paths (all 128 registers are in use, so unused paths cost spills in the used ones), and the down-phase loop goes back to the s0008 form (the pipelined fragment copy of s0009 was a loss). Outputs are bit-identical to s0009 on my test workloads. Local, serve path on track A: batch 1 1779 -> 1729 us, batch 8 4348 -> 4168 us. Expect http B=1 about 1725 us (s0009: 1772), B=8 about 3990 us (s0009: 4158); gates as s0009.
- evaluator: valid: 1745 us/step at B=1 and 4152 at B=8 over HTTP (online (submit/poll)); in process 1578 us/step at B=1 (step, MBU 0.400)

### #12 s0012: Track A build of the s0011 kernel (two instances, s0008 down-phase loop)

- status: **valid**, http_step_us_b1 = 1734 (B=8: 4079), at 5.47 agent-hours, parent s0010, category structure
- hypothesis: track A: no format differs from the checkpoint's or from what the general engines use (BF16 weights as stored, BF16 activations and K/V, float32 where the Hugging Face path uses float32). Same kernel as s0011 without the track-B precision mode, 2 accumulators per mma chain; outputs bit-identical to s0008 and s0010 on my test workloads, so the gates should repeat (tokens 93.07%). Local serve path: batch 1 1779 -> 1729 us, batch 8 4348 -> 4168 us against the s0010 kernel. Expect http B=1 about 1730 us (s0010: 1780; s0008, which has the race: 1685), B=8 about 4070 us (s0010: 4235).
- evaluator: valid: 1734 us/step at B=1 and 4079 at B=8 over HTTP (online (submit/poll)); in process 1559 us/step at B=1 (step, MBU 0.405)

### #13 s0013: Batch >2 kernel: split tile requests in the down phase, norm scratch, deferred writes

- status: **valid**, http_step_us_b1 = 1746 (B=8: 4101), at 6.22 agent-hours, parent s0011, category scheduling
- hypothesis: track B: formats that differ from the checkpoint's: residual stream in float32 and router input as hi+lo BF16 (as s0011); weights BF16 as stored. Change, batch > 2 instance only, no new registers: the down phase requests a round's tiles as 8 + 8 with the fragment copy in between (a warp's loads return in issue order), the norm keeps the later sequence pairs in shared memory instead of loading them twice, the hidden units are written after the last expert slot by all 16 warps. Outputs bit-identical to s0011 on my workloads. Local serve path: batch 8 4168 -> 4074 us on track A. Expect http B=8 about 4060 us (s0011: 4152), B=1 unchanged at about 1745 us.
- evaluator: valid: 1746 us/step at B=1 and 4101 at B=8 over HTTP (online (submit/poll)); in process 1577 us/step at B=1 (step, MBU 0.400)

### #14 s0014: Track A build of the s0013 kernel (batch >2: split tile requests, norm scratch)

- status: **valid**, http_step_us_b1 = 1734 (B=8: 3993), at 6.26 agent-hours, parent s0012, category scheduling
- hypothesis: track A: no format differs from the checkpoint's or from what the general engines use (BF16 weights as stored, BF16 activations and K/V, float32 where the Hugging Face path uses float32). Same kernel as s0013 without the track-B precision mode; outputs bit-identical to s0012 on my workloads, so the gates repeat. Local serve path: batch 8 4168 -> 4074 us. Expect http B=8 about 3990 us (s0012: 4079, vLLM 3796), B=1 unchanged at about 1734 us.
- evaluator: valid: 1734 us/step at B=1 and 3993 at B=8 over HTTP (online (submit/poll)); in process 1559 us/step at B=1 (step, MBU 0.405)

### #15 s0015: Head in 16-tile batches (staging registers)

- status: **valid**, http_step_us_b1 = 1733 (B=8: 4100), at 6.77 agent-hours, parent s0013, category gemm
- hypothesis: track B: formats that differ from the checkpoint's: residual stream in float32 and router input as hi+lo BF16 (as s0013); weights BF16 as stored. Change: the output head requests 16 tiles per load round trip into the 16 staging registers (free at that point) instead of 8 into its own. Outputs bit-identical to s0013 on my workloads. Local, track A: batch-1 step 1552 -> 1538 us, serve mean 1727 -> 1719 us; batch 8 unchanged. Expect http B=1 about 1738 us (s0013: 1746), B=8 about 4100 us.
- evaluator: valid: 1733 us/step at B=1 and 4100 at B=8 over HTTP (online (submit/poll)); in process 1567 us/step at B=1 (step, MBU 0.403)

### #16 s0016: Track A build of the s0015 kernel (head in 16-tile batches)

- status: **valid**, http_step_us_b1 = 1724 (B=8: 3989), at 6.81 agent-hours, parent s0014, category gemm
- hypothesis: track A: no format differs from the checkpoint's or from what the general engines use (BF16 weights as stored, BF16 activations and K/V, float32 where the Hugging Face path uses float32). Same kernel as s0015 without the track-B precision mode; outputs bit-identical to s0014 on my workloads. Expect http B=1 about 1725 us (s0014: 1734), B=8 about 3993 us (unchanged).
- evaluator: valid: 1724 us/step at B=1 and 3989 at B=8 over HTTP (online (submit/poll)); in process 1543 us/step at B=1 (step, MBU 0.409)

### #17 s0017: Strict hand-overs: sentinel words, signal at phase start, readers poll the data

- status: **valid**, http_step_us_b1 = 1786 (B=8: 4298), at 9.01 agent-hours, parent s0015, category sync
- hypothesis: track B: float32 residual stream and hi+lo BF16 router operands as in s0015, weights BF16 as stored; arithmetic unchanged. Found: with data written by atomicExch and the counter bumped afterwards by the same warp (s0009-s0016) the bump can still overtake the exchanges (tools/flake.py: 1 of 357 repetitions differed on the s0015 kernel, 6% on a build with more memory traffic; one caught case = one attention head read a stale q). New protocol for every per-layer hand-over (x, q/k/v, attention output, hidden units, router logits): the writer exchanges a NaN sentinel into its words at the START of its phase (all readers of the old content are done), reads the results back, bumps the counter at once; at the end it only exchanges the data in; a reader loads again while a word holds the sentinel. Expect: logits identical to s0015, 0 mismatches (0 of 237 locally), HTTP +1% at batch 1 (1733 -> about 1755 us) and +4% at batch 8 (4100 -> about 4250 us) because the sentinel doubles the atomic writes.
- evaluator: valid: 1786 us/step at B=1 and 4298 at B=8 over HTTP (online (submit/poll)); in process 1589 us/step at B=1 (step, MBU 0.397)

### #18 s0018: Sentinels issued a phase ahead, confirmed by read-back; second stream buffer

- status: **valid**, http_step_us_b1 = 1733 (B=8: 4268), at 9.57 agent-hours, parent s0017, category sync
- hypothesis: track B: float32 residual stream and hi+lo BF16 router operands as in s0015/s0017, weights BF16 as stored; arithmetic unchanged. s0017 read the sentinel exchange's result back at the start of the phase and that stalled the writing warp (measured locally: 46 us per step at batch 1; without the read-back the kernel is faster than the racy s0015 one). Now every sentinel is exchanged in one phase ahead, in time the block spends waiting anyway (q/k/v at the end of the previous layer, the post-attention stream and shared-expert units before the wait for the attention, the layer output at the end of phase C, the attention output after all blocks copied the last one), its result is dropped, and it is confirmed by a read-back load issued with the phase's own first loads. That needs the post-attention stream in a buffer of its own (x and x_mid no longer overwrite each other). In the expert phase the tile requests go out before the sentinel exchanges. Expect HTTP 1786 -> about 1735 us at batch 1 (s0015: 1733) and 4298 -> about 4255 at batch 8; logits identical; 0 mismatches (0 of 297 locally).
- evaluator: valid: 1733 us/step at B=1 and 4268 at B=8 over HTTP (online (submit/poll)); in process 1553 us/step at B=1 (step, MBU 0.406)

### #19 s0019: Track A build of the s0018 kernel (strict sentinel hand-overs)

- status: **valid**, http_step_us_b1 = 1783 (B=8: 4179), at 9.61 agent-hours, parent s0016, category sync
- hypothesis: track A: BF16 residual stream, BF16 activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0016); same kernel source as s0018 with the track switch. The hand-over protocol is the only change against s0016: sentinel words exchanged in a phase ahead, read-back confirm, signal at the phase start, readers re-load words that hold the sentinel; post-attention stream in its own buffer. Expect logits identical to s0016, 0 mismatches between runs (0 of 59 locally; s0016's protocol was not strict), HTTP 1724 -> about 1757 us at batch 1 and 3989 -> about 4200 at batch 8 (local: 1539 -> 1572 and 3855 -> 3925 in process; the batch-8 loss is the doubled number of atomic writes).
- evaluator: valid: 1783 us/step at B=1 and 4179 at B=8 over HTTP (online (submit/poll)); in process 1569 us/step at B=1 (step, MBU 0.402)

### #20 s0020: Attention multiply-adds written out: bitwise batch invariance at batch 7-8

- status: **valid**, http_step_us_b1 = 1733 (B=8: 4233), at 10.43 agent-hours, parent s0018, category correctness
- hypothesis: track B: float32 residual stream and hi+lo BF16 router operands as in s0018, weights BF16 as stored. Found with a new check (tools/bcheck.py: every batch size, every sequence, bitwise against the single-sequence run): in s0017-s0019 sequences 6-7 of a batch of 7 or 8 differ from the same sequence alone by one rounding step of a softmax weight. Their attention pairs run on main blocks, attention() is inlined there a second time, and the compiler fused a*b + c*d into an FMA differently in the two copies (track B rotary: no BF16 rounding between products and sum). Change: the five multiply-add expressions of attention() use explicit __fmaf_rn/__fmul_rn/__fadd_rn, with the fusion the attention-block copy had (batch-1 digest unchanged). Expect: every batch size bitwise equal to batch 1 (128 steps locally), all gate numbers as s0018, speed unchanged: HTTP about 1733 us at batch 1 and 4268 at batch 8 (+-15).
- evaluator: valid: 1733 us/step at B=1 and 4233 at B=8 over HTTP (online (submit/poll)); in process 1550 us/step at B=1 (step, MBU 0.407)

### #21 s0021: Router top-6 by ranks instead of six rounds of warp maxima (batch 1-2)

- status: **valid**, http_step_us_b1 = 1715 (B=8: 4231), at 10.77 agent-hours, parent s0020, category scheduling
- hypothesis: track B: float32 residual stream and hi+lo BF16 router operands as in s0020, weights BF16 as stored; arithmetic and choices bit-identical to s0020 (tools/digest1.py and tools/bcheck.py unchanged). In the batch 1-2 kernel every main block picked the 6 largest of 256 router scores by six rounds of (warp maximum by 5 shuffles, ballot) in 8 warps and six more rounds over the 48 candidates in one warp: about 90 dependent warp operations on the path to the expert phase. Now every thread computes the rank of its score among the 32 of its warp from shared memory (8 loads), and the 48 candidates are ranked the same way (12 loads); ties are ordered exactly as the rounds ordered them. Local: step 1550 -> 1533 us at batch 1, 64+1024 workload 1734 -> 1720. Expect HTTP 1733 -> about 1717 us at batch 1; batch 8 unchanged (4233 +-15: its kernel is not touched).
- evaluator: valid: 1715 us/step at B=1 and 4231 at B=8 over HTTP (online (submit/poll)); in process 1535 us/step at B=1 (step, MBU 0.411)

### #22 s0022: Expert phase, batch 1-2: idle warps fold and write the first slot's hidden units

- status: **valid**, http_step_us_b1 = 1701 (B=8: 4224), at 10.81 agent-hours, parent s0021, category scheduling
- hypothesis: track B: formats as s0021 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results. At batch 1 a block has 3 row groups of expert gate/up: slot 1 = 2 row groups on 16 warps, slot 2 = 1 row group on warps 0-7. The leader of the first half (warp 0) folded the 8 partial sums of slot 1 and wrote the hidden units (SiLU, 2 atomic exchanges per lane: about 1 us in the profile) BEFORE its own 16 tiles of slot 2, and the block's next barrier waited for it. Now, when the next slot has work for the first half only, warps 8 and 9 (idle in that slot) do the two epilogues of the current slot; the sentinel of those words was confirmed at the start of the phase, so any warp may write the data. Local: step 1535.6 -> 1519.9 us at batch 1, workload 1719 -> 1705. Expect HTTP 1715 -> about 1700 us at batch 1; batch 8 unchanged (4231 +-15).
- evaluator: valid: 1701 us/step at B=1 and 4224 at B=8 over HTTP (online (submit/poll)); in process 1524 us/step at B=1 (step, MBU 0.414)

### #23 s0023: Shared-expert epilogue and K/V cache stores moved off the critical path

- status: **valid**, http_step_us_b1 = 1699 (B=8: 4221), at 10.84 agent-hours, parent s0022, category scheduling
- hypothesis: track B: formats as s0022 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results. Two epilogues of the same kind, nobody waits for their data: (1) the shared expert's hidden units (read only in the down phase) were folded and written by warp 0 before the routing could start; now at batch 1 warp 15 does it while warps 0-7 rank the router scores, and at batch > 2 warp 0 does it after requesting its first expert tiles. (2) The attention wrote this position's K and V into the cache (plain 16-bit stores, read at the next step) before computing the scores; now after its output is on the way. Local A/B: (1) step 1521.5 -> 1507.9 us at batch 1 (workload 1704 -> 1694), batch 8 workload 4217 -> 4194; (2) 1551 -> 1547 (within noise). Expect HTTP 1701 -> about 1690 us at batch 1 and 4224 -> about 4205 at batch 8.
- evaluator: valid: 1699 us/step at B=1 and 4221 at B=8 over HTTP (online (submit/poll)); in process 1508 us/step at B=1 (step, MBU 0.419)

### #24 s0024: Track A build of the s0023 kernel (ranks, epilogues off the critical path)

- status: **valid**, http_step_us_b1 = 1732 (B=8: 4111), at 10.88 agent-hours, parent s0019, category scheduling
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0019); same kernel source as s0023 with the track switch. Against s0019 it has the explicit multiply-adds of the attention (no change of track-A results: tools/digest1.py and tools/bcheck.py are unchanged and pass on track A before and after), the rank-based top-6 of the batch 1-2 kernel, the expert-phase epilogues by idle warps, the shared-expert epilogue and K/V cache stores off the critical path. Local track A: step 1563 -> 1505 us at batch 1, 3871 -> 3883 at batch 8; 64+1024 workload 1769 -> 1717 and 4117 -> 4108. Expect HTTP 1783 -> about 1735 us at batch 1 and 4179 -> about 4170 at batch 8; gates as s0019.
- evaluator: valid: 1732 us/step at B=1 and 4111 at B=8 over HTTP (online (submit/poll)); in process 1494 us/step at B=1 (step, MBU 0.422)

### #25 s0025: Attention at long contexts: K/V cache tiles 8 per request, not one per trip

- status: **valid**, http_step_us_b1 = 1663 (B=8: 4222), at 11.24 agent-hours, parent s0023, category attention
- hypothesis: track B: formats as s0023 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results (tools/digest_long.py to position 1100, digest1, bcheck). Profile at position 900: the attention takes 13.8 us against 7.2 at position 192. Beyond the 16 key groups staged in registers (and 12 K groups in shared memory) the K tiles were loaded inside the score loop and the V tiles inside the dependent chain of the mix, one load per trip to memory (0.3-0.4 us each, up to 8 + 20 in a row). Now a key group's 8 K tiles and the next 8 V levels are requested together, into the staging registers that are free by then (a separate register array made the whole kernel spill: 2195 us). Batch 1-2 kernel only: in the batch > 2 kernel the same change cost 1.4% at batch 8 (measured), left as it was. Local: 64+1024 workload 1691 -> 1656 us at batch 1; short contexts unchanged. Expect HTTP 1699 -> about 1662 us at batch 1; batch 8 unchanged (4221 +-15).
- evaluator: valid: 1663 us/step at B=1 and 4222 at B=8 over HTTP (online (submit/poll)); in process 1508 us/step at B=1 (step, MBU 0.419)

### #26 s0026: Track A build of the s0025 kernel (K/V cache tiles 8 per request)

- status: **valid**, http_step_us_b1 = 1679 (B=8: 4116), at 11.28 agent-hours, parent s0024, category attention
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0024); same kernel source as s0025 with the track switch; bit-identical to s0024 (digest_long, digest1, bcheck on track A). The attention of the batch 1-2 kernel requests the K/V cache tiles beyond the staged ones 8 at a time. Local track A: 64+1024 workload 1716 -> 1662 us at batch 1, batch 8 unchanged. Expect HTTP 1732 -> about 1680 us at batch 1; batch 8 unchanged (4111 +-15).
- evaluator: valid: 1679 us/step at B=1 and 4116 at B=8 over HTTP (online (submit/poll)); in process 1498 us/step at B=1 (step, MBU 0.421)

### #27 s0027: Attention: first extra batch of V tiles requested right after the scores

- status: **valid**, http_step_us_b1 = 1646 (B=8: 4227), at 11.61 agent-hours, parent s0025, category attention
- hypothesis: track B: formats as s0025 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results (digest_long, digest1, bcheck). In s0025 the 8 V levels beyond the staged ones were requested at the start of the mix, i.e. after the softmax; they do not depend on it. Now they are requested as soon as the K registers are free (end of the score stage) and arrive during the softmax (batch 1-2 kernel). Local, one card: 64+1024 workload 1655 -> 1637 us at batch 1 on track B (track A 1661 -> 1657), short contexts 1507 -> 1498. Expect HTTP 1663 -> about 1647 us at batch 1; batch 8 unchanged (4222 +-15).
- evaluator: valid: 1646 us/step at B=1 and 4227 at B=8 over HTTP (online (submit/poll)); in process 1499 us/step at B=1 (step, MBU 0.421)

### #28 s0028: Track A build of the s0027 kernel (V tiles requested after the scores)

- status: **valid**, http_step_us_b1 = 1669 (B=8: 4118), at 11.66 agent-hours, parent s0026, category attention
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0026); same kernel source as s0027 with the track switch; bit-identical to s0026 (digest_long, digest1, bcheck on track A). The attention of the batch 1-2 kernel requests the first batch of V levels beyond the staged ones right after the scores. Local track A: 64+1024 workload 1661 -> 1655 us at batch 1 (the gain is smaller than on track B: 1655 -> 1638), batch 8 unchanged. Expect HTTP 1679 -> about 1673 us at batch 1; batch 8 unchanged (4116 +-15).
- evaluator: valid: 1669 us/step at B=1 and 4118 at B=8 over HTTP (online (submit/poll)); in process 1506 us/step at B=1 (step, MBU 0.419)

### #29 s0029: Batch > 2: the 12 routing candidates of a sequence merged by ranks

- status: **valid**, http_step_us_b1 = 1646 (B=8: 4188), at 12.23 agent-hours, parent s0027, category scheduling
- hypothesis: track B: formats as s0027 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results (bcheck for batches 2..8, digest_long). In the batch > 2 kernel every main block merged a sequence's 12 candidates (6 from each of its two warps) by six rounds of (maximum over the warp by 5 shuffles, ballot, shuffle); now every lane computes the rank of its candidate from 3 loads of shared memory, as the batch 1-2 kernel does since s0021 (ties in the same order). Local, same card: step 4074 -> 4042 us at batch 8 (track A 3883 -> 3851), 64+1024 workload 4204 -> 4165. Expect HTTP 4227 -> about 4190 us at batch 8; batch 1 unchanged (1646 +-10: its kernel is not touched).
- evaluator: valid: 1646 us/step at B=1 and 4188 at B=8 over HTTP (online (submit/poll)); in process 1499 us/step at B=1 (step, MBU 0.421)

### #30 s0030: Track A build of the s0029 kernel (batch > 2 routing merge by ranks)

- status: **valid**, http_step_us_b1 = 1671 (B=8: 4056), at 12.27 agent-hours, parent s0028, category scheduling
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0028); same kernel source as s0029 with the track switch; bit-identical to s0028 (bcheck on track A). The batch > 2 kernel merges a sequence's 12 routing candidates by ranks. Local track A: step 3883 -> 3851 us at batch 8, workload 4106 -> 4086. Expect HTTP 4118 -> about 4095 us at batch 8; batch 1 unchanged (1669 +-10).
- evaluator: valid: 1671 us/step at B=1 and 4056 at B=8 over HTTP (online (submit/poll)); in process 1506 us/step at B=1 (step, MBU 0.419)

### #31 s0031: Batch > 2: router blocks preload router tiles, stage K/V after the router

- status: **valid**, http_step_us_b1 = 1646 (B=8: 4123), at 12.41 agent-hours, parent s0029, category scheduling
- hypothesis: track B: formats as s0029 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results (bcheck for batches 2..8, digest_long). At batch > 2 the 16 router blocks also run an attention pair; right after it they staged the next layer's K/V tiles in their registers, so the 1 MB of router tiles was requested only when the router needed it, through the busy memory queue, on the path the main blocks wait on (at batch 1 the same tiles waiting in registers were measured to be worth 159 us per step). Now these blocks request the router tiles into the staging registers right after their attention and stage the next layer's K/V after the router duty (it has 40 us until the next attention). Local, same card: step 4042 -> 4010 us at batch 8 (track A 3851 -> 3819), workload 4171 -> 4125. Expect HTTP 4188 -> about 4150 us at batch 8; batch 1 unchanged (1646 +-10).
- evaluator: valid: 1646 us/step at B=1 and 4123 at B=8 over HTTP (online (submit/poll)); in process 1499 us/step at B=1 (step, MBU 0.421)

### #32 s0032: Track A build of the s0031 kernel (router tiles preloaded at batch > 2)

- status: **valid**, http_step_us_b1 = 1671 (B=8: 4085), at 12.44 agent-hours, parent s0030, category scheduling
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0030); same kernel source as s0031 with the track switch; bit-identical to s0030 (bcheck on track A). At batch > 2 the router blocks request the router tiles right after their attention and stage the next layer's K/V after the router duty. Local track A: step 3851 -> 3819 us at batch 8, workload 4081 -> 4071. Expect HTTP 4056 -> about 4035 us at batch 8; batch 1 unchanged (1671 +-10).
- evaluator: valid: 1671 us/step at B=1 and 4085 at B=8 over HTTP (online (submit/poll)); in process 1504 us/step at B=1 (step, MBU 0.420)

### #33 s0033: Attention: the k/v-row loads are issued together with the q-row loads

- status: **valid**, http_step_us_b1 = 1652 (B=8: 4120), at 12.93 agent-hours, parent s0031, category attention
- hypothesis: track B: formats as s0031 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical results (digest_long, digest1, bcheck). In the attention's first stage the threads that also read the raw k and v rows issued that load only after their q rows had arrived and had been used (norm sum, shuffles); now both loads are issued first. A by-product of an experiment that failed (output-projection tiles requested inside the attention of the pairs on main blocks: 3991 -> 4071 us, off). Local, same card: batch 8 step 4008 -> 3992 us on track B, batch 1 unchanged (1500); track A batch 1 1516 -> 1506, batch 8 unchanged. Expect HTTP batch 8 4123 -> about 4105 us, batch 1 unchanged (1646 +-10): a small change, inside the 30 us by which the batch-8 figure moves between evaluations.
- evaluator: valid: 1652 us/step at B=1 and 4120 at B=8 over HTTP (online (submit/poll)); in process 1502 us/step at B=1 (step, MBU 0.420)

### #34 s0034: Track A build of the s0033 kernel (attention load order)

- status: **valid**, http_step_us_b1 = 1671 (B=8: 4097), at 12.97 agent-hours, parent s0032, category attention
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0032); same kernel source as s0033 with the track switch; bit-identical to s0032 (digest_long, digest1, bcheck on track A). The attention issues its k/v-row loads together with the q-row loads. Local track A: step 1516 -> 1506 us at batch 1, workload 1657 -> 1649; batch 8 unchanged. Expect HTTP 1671 -> about 1663 us at batch 1, batch 8 4085 +-30 (on track B the same change moved nothing over HTTP: s0033).
- evaluator: valid: 1671 us/step at B=1 and 4097 at B=8 over HTTP (online (submit/poll)); in process 1499 us/step at B=1 (step, MBU 0.421)

### #35 s0035: Host sees the kernel's watchdog; contexts above 2048 positions refused

- status: **valid**, http_step_us_b1 = 1650 (B=8: 4117), at 13.14 agent-hours, parent s0033, category correctness
- hypothesis: track B: formats as s0033 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); kernel arithmetic and scheduling unchanged (digest_long, digest1, bcheck identical). Two robustness changes on the host side: (1) the flag the kernel sets when one of its waits gives up was cleared before every launch and never read; now poll() reads it with the progress word and raises, reset() and stats() read it, and the host re-bases its counters so the engine works again after an event (tested with an injected unreachable wait: tools/wd_test.py); nothing is read inside a decode step. (2) Engine() accepted any max_context, the attention's shared-memory layout holds 2048 positions; larger values are capped and longer requests refused. Expect every gate and every timing as s0033 within the noise of the evaluation: HTTP 1652 +-10 us at batch 1 and 4120 +-30 at batch 8; trace: still no host synchronization per decode step.
- evaluator: valid: 1650 us/step at B=1 and 4117 at B=8 over HTTP (online (submit/poll)); in process 1499 us/step at B=1 (step, MBU 0.421)

### #36 s0036: Track A build of the s0035 engine (watchdog visible, context cap)

- status: **valid**, http_step_us_b1 = 1672 (B=8: 4099), at 13.18 agent-hours, parent s0034, category correctness
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0034); same source as s0035 with the track switch; kernel arithmetic and scheduling unchanged. Host side: watchdog events are reported by poll(), reset() and stats() and the engine recovers; max_context is capped at the kernel's 2048 positions. Expect every gate and timing as s0034 within noise: HTTP 1671 +-10 us at batch 1, 4097 +-30 at batch 8.
- evaluator: valid: 1672 us/step at B=1 and 4099 at B=8 over HTTP (online (submit/poll)); in process 1498 us/step at B=1 (step, MBU 0.422)

### #37 s0037: generate/serve at batch 1-2: the 152 blocks' best tokens gathered in one trip

- status: **valid**, http_step_us_b1 = 1606 (B=8: 4116), at 13.72 agent-hours, parent s0035, category launch
- hypothesis: track B: formats as s0035 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); logits and tokens bit-identical (digest_long, digest1, bcheck, generate == step-by-step). After the head, block 0 collected the 152 blocks' (best value, index) pairs with one thread per sequence, 302 loads one after the other (generate() was 21-26 us per token slower than step() although it writes no logits). Now, in the batch 1-2 kernel, all threads of block 0 load up to 3 pairs each in one trip and the result is combined over lanes and warps (largest value, lowest index on a tie); as a function of its own, because inlined it slowed every phase. The batch > 2 kernel keeps the loop (there the call costs more than it saves). Local, same card: generate 1528 -> 1487 us per token, 64+1024 workload 1644 -> 1606, serve 1530 -> 1486 at batch 1; batch 8 unchanged. Expect HTTP 1650 -> about 1612 us at batch 1; batch 8 unchanged (4117 +-30).
- evaluator: valid: 1606 us/step at B=1 and 4116 at B=8 over HTTP (online (submit/poll)); in process 1485 us/step at B=1 (step, MBU 0.425)

### #38 s0038: Track A build of the s0037 kernel (token gather in one trip)

- status: **valid**, http_step_us_b1 = 1606 (B=8: 4094), at 13.76 agent-hours, parent s0036, category launch
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0036); same source as s0037 with the track switch; bit-identical logits and tokens (digest_long, digest1, bcheck on track A). In the batch 1-2 kernel block 0 gathers the 152 blocks' best tokens in one trip instead of 302 serial loads. Local track A, same card: generate 1526 -> 1477 us per token, 64+1024 workload 1645 -> 1593, serve 1526 -> 1479 at batch 1; batch 8 unchanged. Expect HTTP 1672 -> about 1622 us at batch 1; batch 8 unchanged (4099 +-30).
- evaluator: valid: 1606 us/step at B=1 and 4094 at B=8 over HTTP (online (submit/poll)); in process 1473 us/step at B=1 (step, MBU 0.429)

### #39 s0039: generate/serve at batch 3-8: the blocks' best tokens gathered in 16 strides

- status: **valid**, http_step_us_b1 = 1606 (B=8: 4088), at 14.00 agent-hours, parent s0037, category launch
- hypothesis: track B: formats as s0037 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); logits and tokens bit-identical (digest_long equal to s0037's kernel, bcheck, generate == step-by-step at batch 8). The batch > 2 kernel still gathered the 152 blocks' best (value, index) pairs with one thread per sequence, 302 loads one after the other (the function used at batch 1-2 cost more than it saved there). Now 16 threads per sequence each walk every 16th block (10 loads in a row, not 151), written inline with no shuffle so the register picture of the kernel stays the same (spill lines of the build unchanged). Local, same card: generate 3954 -> 3926 us per step, 64+1024 workload 4082 -> 4061, serve 3956 -> 3928 at batch 8; batch 1 unchanged. Expect HTTP batch 8 4116 -> about 4090 (noise +-30), batch 1 unchanged at 1606 +-6.
- evaluator: valid: 1606 us/step at B=1 and 4088 at B=8 over HTTP (online (submit/poll)); in process 1486 us/step at B=1 (step, MBU 0.425)

### #40 s0040: Track A build of the s0039 kernel (batch 3-8 token gather in 16 strides)

- status: **valid**, http_step_us_b1 = 1608 (B=8: 4028), at 14.03 agent-hours, parent s0039, category launch
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0038); same source as s0039 with the track switch; bit-identical logits and tokens (digest_long equal to s0038's kernel, bcheck on track A). In the batch > 2 kernel 16 threads per sequence each walk every 16th block's best (value, index) pair instead of one thread doing 302 loads in a row. Local track A, same card: generate 3889 -> 3867 us per step, 64+1024 workload 4074 -> 4052, serve 3895 -> 3865 at batch 8; batch 1 unchanged. Expect HTTP batch 8 4094 -> about 4070 (noise +-30), batch 1 unchanged at 1606 +-6.
- evaluator: valid: 1608 us/step at B=1 and 4028 at B=8 over HTTP (online (submit/poll)); in process 1473 us/step at B=1 (step, MBU 0.429)

### #41 s0041: Softmax weights: one reciprocal and an exact FMA correction, not 4224 divisions

- status: **valid**, http_step_us_b1 = 1560 (B=8: 4055), at 14.78 agent-hours, parent s0039, category attention
- hypothesis: track B: formats as s0039 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); logits bit-identical to s0039 on my tests (digest_long 1100 steps at batch 1 and 8, bcheck every batch size). The attention's softmax divided every exponential by the sum with the compiler's IEEE division: 4 per key group and thread, 4224 quotients per (sequence, K/V head) pair at a full window, 38 ns per quotient and warp on a busy SM (tools/bench19.cu). Now one division per thread (r = 1/sum) and q = e*r, q + fma(-q, sum, e)*r: Markstein's division step, the correctly rounded quotient whenever the remainder is exact (every exponential of 2^-102 and more; 0 mismatches in 6e10 quotients in bench19; below that the weight, itself under 1e-30, can be one unit in the last place off), 12 ns. Float32 as before, no format changes. Local, same card, 64+1024 workload: generate 1604 -> 1578 us per token at batch 1 and 4064 -> 4012 per step at batch 8 (the gain grows with the context: at positions over 1024, 1714 -> 1659 and 4311 -> 4268). Expect HTTP 1606 -> about 1578 at batch 1 and 4088 -> about 4040 at batch 8.
- evaluator: valid: 1560 us/step at B=1 and 4055 at B=8 over HTTP (online (submit/poll)); in process 1471 us/step at B=1 (step, MBU 0.429)

### #42 s0042: Track A build of the s0041 kernel (softmax weights by reciprocal + correction)

- status: **valid**, http_step_us_b1 = 1548 (B=8: 3928), at 14.82 agent-hours, parent s0041, category attention
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0040); same source as s0041 with the track switch; logits bit-identical to s0040 on my tests (digest_long, bcheck on track A). The softmax weights e/sum are computed as q = e*r, q + fma(-q, sum, e)*r with r = 1/sum (one division per thread instead of 4 per key group and thread): the correctly rounded float32 quotient for every exponential of 2^-102 and more (tools/bench19.cu), still float32. Local track A, same card, 64+1024 workload: generate 1591 -> 1541 us per token at batch 1 and 4048 -> 3969 per step at batch 8. Expect HTTP 1608 -> about 1558 at batch 1 and 4028 -> about 3950 at batch 8.
- evaluator: valid: 1548 us/step at B=1 and 3928 at B=8 over HTTP (online (submit/poll)); in process 1442 us/step at B=1 (generate, MBU 0.438)

### #43 s0043: Batch 7-8: main blocks with an attention pair poll phase A's counter early

- status: **valid**, http_step_us_b1 = 1558 (B=8: 4039), at 15.56 agent-hours, parent s0041, category sync
- hypothesis: track B: formats as s0041 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck), hand-over still strict (the fact 'every block has started A' is established by one thread and passed on by the block's barrier; flake 0 of 59). At batch 7-8 eight (sequence, K/V head) pairs run on main blocks, and every other block waits for them. After its own q/k/v stores such a block asked the counter of phase A (thread 0, one trip to memory, then a barrier) before it could request q/k/v, although the answer had been yes for microseconds. Now a warp that is idle while the five leader warps reduce and store asks, and the barrier that follows the leaders anyway passes it on. Local, same card: generate 3908 -> 3887 us per step at batch 8, 64+1024 workload 4013 -> 3999 (track A: 3838 -> 3832, 3974 -> 3968). Expect HTTP batch 8 4055 -> about 4040 (within the noise of +-30 of one evaluation; the local runs agree in sign in all four measures), batch 1 unchanged at 1560 +-6.
- evaluator: valid: 1558 us/step at B=1 and 4039 at B=8 over HTTP (online (submit/poll)); in process 1470 us/step at B=1 (step, MBU 0.429)

### #44 s0044: Track A build of the s0043 kernel (early poll of phase A's counter, batch 7-8)

- status: **valid**, http_step_us_b1 = 1546 (B=8: 3920), at 15.60 agent-hours, parent s0043, category sync
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0042); same source as s0043 with the track switch; bit-identical logits (digest_long, bcheck on track A), flake 0 of 59. On the eight main blocks that run an attention pair at batch 7-8 the counter of phase A is asked by an idle warp during the leaders' stores instead of by thread 0 after them. Local track A, same card: generate 3838 -> 3832 us per step at batch 8, 64+1024 workload 3974 -> 3968. Expect HTTP batch 8 3928 -> about 3920 (inside the noise of one evaluation), batch 1 unchanged at 1548 +-6.
- evaluator: valid: 1546 us/step at B=1 and 3920 at B=8 over HTTP (online (submit/poll)); in process 1442 us/step at B=1 (step, MBU 0.438)

### #45 s0045: Batch > 2: the norm's scale of each sequence computed once per warp, not by every thread

- status: **valid**, http_step_us_b1 = 1559 (B=8: 4021), at 16.60 agent-hours, parent s0043, category scheduling
- hypothesis: track B: formats as s0043 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck every batch size): the same 16 partial sums added in the same order and the same rsqrt, by lane b of each warp instead of by all 32 lanes for each of the 8 sequences, read by a shuffle (264 instructions per thread and norm become 45). On an SM with 16 warps instructions are time (finding 13c). The norm runs twice per layer on every main block and once on the 16 router blocks; on track B the router blocks (two fragments per value, reloads in the second pass) are on the critical path at batch 8. Local, same card: step 3905 -> 3880 us at batch 8, generate 3885 -> 3862, 64+1024 workload 4003 -> 3970; batch 1 uses the other kernel instance and is unchanged. Expect HTTP batch 8 4039 -> about 4010, batch 1 1558 +-6.
- evaluator: valid: 1559 us/step at B=1 and 4021 at B=8 over HTTP (online (submit/poll)); in process 1471 us/step at B=1 (step, MBU 0.429)

### #46 s0046: Track A build of the s0045 kernel (norm scales once per warp at batch > 2)

- status: **valid**, http_step_us_b1 = 1546 (B=8: 3912), at 16.63 agent-hours, parent s0045, category scheduling
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0044); same source as s0045 with the track switch; bit-identical logits (digest_long, bcheck on track A). The per-sequence scale of the norms is computed once per warp and read by shuffle at batch > 2. On track A the norms are not on the critical path (BF16 stream: half the bytes, one fragment per value): local step 3782 -> 3780 us at batch 8, 64+1024 workload 3966 -> 3964. Expect HTTP unchanged within the noise: 1546 +-6 at batch 1, 3920 +-30 at batch 8.
- evaluator: valid: 1546 us/step at B=1 and 3912 at B=8 over HTTP (online (submit/poll)); in process 1443 us/step at B=1 (step, MBU 0.438)

### #47 s0047: Sentinel reads: the abort word is loaded on every 16th retry, not before each reload

- status: **valid**, http_step_us_b1 = 1566 (B=8: 3979), at 18.45 agent-hours, parent s0045, category sync
- hypothesis: track B: formats as s0045 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck), same hand-over protocol (only how often a spinning reader looks at the watchdog's abort word changes; tools/wd_test.py still sees an injected hang in 0.9 s). A reader that finds the sentinel reloads its word until the data are there. Each attempt first loaded the abort word and the reload waited for that load: two trips through the memory per attempt in every hand-over of every layer. Found by accident: moving the counter array by 4 KB changed the batch-1 step by 17 us, and it was the abort word's address that did it. Local, same card: batch 1 step 1473 -> 1460 us, 64+1024 workload 1571 -> 1559; batch 8 step 3880 -> 3854, workload 3975 -> 3950. Expect HTTP 1559 -> about 1546 at batch 1 and 4021 -> about 3995 at batch 8.
- evaluator: valid: 1566 us/step at B=1 and 3979 at B=8 over HTTP (online (submit/poll)); in process 1458 us/step at B=1 (step, MBU 0.433)

### #48 s0048: Track A build of the s0047 kernel (abort word loaded on every 16th retry)

- status: **valid**, http_step_us_b1 = 1527 (B=8: 3914), at 18.56 agent-hours, parent s0047, category sync
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0046); same source as s0047 with the track switch; bit-identical logits (digest_long, bcheck on track A). A reader spinning on a sentinel loads the watchdog's abort word on every 16th attempt instead of in front of every reload. s0047 (track B) gave in-process 1471 -> 1458 and HTTP batch 8 4021 -> 3979 as expected but HTTP batch 1 1559 -> 1566, which my local run of the same path (tools/serve_long.py: submit/poll, 64+1024, three poll cadences) does not show: track B 1571 -> 1556, track A 1540 -> 1515 us per token at batch 1; batch 8 track A 3970 -> 3974. Expect HTTP batch 1 1546 -> about 1522 if the local measurement is right (about 1550 if s0047's figure was not an outlier), batch 8 3912 +-30.
- evaluator: valid: 1527 us/step at B=1 and 3914 at B=8 over HTTP (online (submit/poll)); in process 1418 us/step at B=1 (step, MBU 0.445)

### #49 s0049: Hand-over counters placed by a start-up timing: fastest of 16 pages per kernel

- status: **valid**, http_step_us_b1 = 1560 (B=8: 3936), at 19.34 agent-hours, parent s0047, category memory
- hypothesis: track B: formats as s0047 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck): only the address of the counter array changes. Measured: where the counters lie changes the step. Batch 1 has two levels 14 us apart, decided by the parity of address bits 12-15 and 17-21 of the counters' page; batch 8 has several levels over 40 us; the same on both tracks and every card, and it follows the physical placement, so the HTTP server's process need not be on the level my in-process timings are on (its batch-1 figure has been 12 us above my local run of the same path). Now each of the two kernel instances has its own counter array, and when the engine starts it times a few greedy steps of fixed tokens with the array on each of 16 pages and keeps the fastest (2.4 s; the pages and the timings are in stats()). Local, same card, same process: batch 8 step 3846 -> 3823, 64+1024 workload 3951 -> 3912, serve path 3957 -> 3915; batch 1 unchanged here (1460: the default page happens to be a fast one in this process). Expect HTTP batch 8 3979 -> about 3945; batch 1 1566 -> about 1553 if the server's process is on the slow level now, unchanged if not.
- evaluator: valid: 1560 us/step at B=1 and 3936 at B=8 over HTTP (online (submit/poll)); in process 1459 us/step at B=1 (step, MBU 0.433)

### #50 s0050: Track A build of the s0049 engine (counters placed by a start-up timing)

- status: **valid**, http_step_us_b1 = 1527 (B=8: 3870), at 19.37 agent-hours, parent s0049, category memory
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0048); same source as s0049 with the track switch; bit-identical logits (digest_long, bcheck on track A). Each kernel instance's counter array is put on the fastest of 16 pages by a timing of a few steps at start-up. s0049 (track B) gave HTTP 1566 -> 1560 at batch 1 and 3979 -> 3936 at batch 8. Local track A, same card and process: batch 8 step 3779 -> 3748, 64+1024 workload 3957 -> 3929..3940, serve path 3974 -> 3938; batch 1 unchanged (1421). Expect HTTP batch 8 3914 -> about 3880 and batch 1 1527 -> 1515..1527 (the lower end if the server's process is on the slow level now).
- evaluator: valid: 1527 us/step at B=1 and 3870 at B=8 over HTTP (online (submit/poll)); in process 1422 us/step at B=1 (step, MBU 0.444)

### #51 s0051: Start-up placement extended: 3 more buffers at batch 1, 64 counter pages at batch 8

- status: **valid**, http_step_us_b1 = 1555 (B=8: 3961), at 19.89 agent-hours, parent s0049, category memory
- hypothesis: track B: formats as s0049 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck): only addresses change. s0049 placed the counter array of each kernel instance on the fastest of 16 pages by a timing at start-up. The same two-level dependence on the page exists, smaller, for three buffers that are one or two pages at batch 1 and that every block reads (router logits, post-attention stream, attention output: 5-10 us each on track A, 2-4 on track B), and at batch 8 the best of 64 counter pages is better than the best of 16. Now the batch 1-2 instance also places those three buffers (16 pages each), and the batch > 2 instance chooses among 64 counter pages; 13 s at start-up instead of 2.4. Local, same card and build, against the s0049 placement: track B batch 8 step 3837 -> 3821, 64+1024 workload 3930 -> 3910, batch 1 unchanged (1460); track A batch 1 1418 -> 1410 and batch 8 3772 -> 3751. Expect HTTP batch 8 3936 -> about 3920 and batch 1 1560 +-6 on this track.
- evaluator: valid: 1555 us/step at B=1 and 3961 at B=8 over HTTP (online (submit/poll)); in process 1458 us/step at B=1 (step, MBU 0.433)

### #52 s0052: Track A build of the s0051 engine (extended start-up placement)

- status: **valid**, http_step_us_b1 = 1516 (B=8: 3898), at 19.94 agent-hours, parent s0051, category memory
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0050); same source as s0051 with the track switch; bit-identical logits (digest_long, bcheck on track A). The batch 1-2 kernel also places router logits, post-attention stream and attention output on the fastest of 16 pages each, the batch > 2 kernel picks its counter page among 64. s0051 (track B): in-process batch 8 3942 -> 3934, HTTP 1560 -> 1555 at batch 1; HTTP batch 8 3936 -> 3961 with its three repeats between 3918 and 4000 (the HTTP median at batch 8 moves by 25 us on its own). Local track A, same card and build, against the s0050 placement: batch 1 step 1418 -> 1410, 64+1024 workload 1516 -> 1509; batch 8 step 3772 -> 3751, workload 3954 -> 3934. Expect HTTP batch 1 1527 -> about 1520 and batch 8 3870 -> about 3850 +-30.
- evaluator: valid: 1516 us/step at B=1 and 3898 at B=8 over HTTP (online (submit/poll)); in process 1409 us/step at B=1 (step, MBU 0.448)

### #53 s0053: Placement: batch 8 back to the 16 counter pages of s0049; batch 1 buffers on own slabs

- status: **valid**, http_step_us_b1 = 1557 (B=8: 3950), at 20.16 agent-hours, parent s0051, category memory
- hypothesis: track B: formats as s0051 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck): only addresses change. s0051/s0052 chose the batch > 2 counter page among 64 by a generate() timing: 20 us better in generate(), but 25-28 us worse over HTTP on both tracks (3936 -> 3961, 3870 -> 3898): the serve path's best page is not the one that timing finds among many. Now the batch > 2 instance has exactly the allocations and the 16 candidate pages of s0049/s0050 again (it picks page 8 as there), and the batch 1-2 instance's three placed buffers (router logits, post-attention stream, attention output) are copies of their own, allocated after everything else. Local, same card: batch 1 step 1459 -> 1452 and serve path 1557 -> 1555; batch 8 serve path 3913 -> 3907. Expect HTTP batch 8 back to about 3936 and batch 1 1555 -> about 1550.
- evaluator: valid: 1557 us/step at B=1 and 3950 at B=8 over HTTP (online (submit/poll)); in process 1453 us/step at B=1 (generate, MBU 0.434)

### #54 s0054: Track A build of the s0053 engine (batch-8 counters as s0050, batch-1 buffers on own slabs)

- status: **valid**, http_step_us_b1 = 1516 (B=8: 3878), at 20.20 agent-hours, parent s0053, category memory
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0052); same source as s0053 with the track switch; bit-identical logits (digest_long, bcheck on track A). The batch > 2 instance has the allocations and the 16 candidate counter pages of s0050 again; the batch 1-2 instance places its counters and its own copies of router logits, post-attention stream and attention output. s0053 (track B): HTTP 1557 / 3950 (s0051: 1555 / 3961, s0049: 1560 / 3936: at batch 8 the three are within the spread of the HTTP repeats on that track). Local track A, same card: batch 1 step 1410 -> 1401, serve path 1509 -> 1501; batch 8 serve path 3933 -> 3924. Expect HTTP batch 1 1516 -> about 1508; batch 8 3898 -> 3870..3890 if the 64-page choice was what cost 28 us in s0052.
- evaluator: valid: 1516 us/step at B=1 and 3878 at B=8 over HTTP (online (submit/poll)); in process 1402 us/step at B=1 (step, MBU 0.450)

### #55 s0055: Track B: kernel body wrapped in an inlined function: better register allocation

- status: **valid**, http_step_us_b1 = 1473 (B=8: 3963), at 21.34 agent-hours, parent s0053, category other
- hypothesis: track B: formats as s0053 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck): the source of the kernel is the same, it is only written as an inlined function that the kernel calls. Found as the control of a failed experiment (attention/router blocks as a function of their own: 320 bytes of stack, 1669 us): with the wrapper alone the compiler's register allocation of the batch 1-2 instance has 36/16 bytes of spills instead of 68/84 on this track (on track A it is the other way round, so track A keeps the unwrapped form: its build is unchanged). The kernel uses all 128 registers and the spills sit in its hot loops. Local, same card: batch 1 step 1453 -> 1381 us, 64+1024 workload 1552 -> 1480, serve path 1550 -> 1481; batch 8 3813 -> 3805, serve path 3903 -> 3916. Expect HTTP batch 1 1557 -> about 1490 and batch 8 3950 +-30.
- evaluator: valid: 1473 us/step at B=1 and 3963 at B=8 over HTTP (online (submit/poll)); in process 1382 us/step at B=1 (step, MBU 0.457)

### #56 s0056: Track A build of the s0055 source (wrapper off on this track: build unchanged)

- status: **valid**, http_step_us_b1 = 1516 (B=8: 3874), at 21.38 agent-hours, parent s0055, category other
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0054); same source as s0055 with the track switch; bit-identical logits (digest_long on track A). The body wrapper of s0055 is switched by the track (BODY_WRAP = track B): on track A the unwrapped kernel is the one whose batch 1-2 instance has the good register allocation (36/16 bytes of spills; with the wrapper 68/84 and 1434 us instead of 1402). The track-A build has the same spill lines and timings as s0054 (local 1402 / 3747). Expect HTTP unchanged within the noise: 1516 +-6 at batch 1, 3878 +-30 at batch 8.
- evaluator: valid: 1516 us/step at B=1 and 3874 at B=8 over HTTP (online (submit/poll)); in process 1402 us/step at B=1 (generate, MBU 0.450)

### #57 s0057: Hand-over reads as strong loads (ld.relaxed.gpu) instead of ld.global.cg

- status: **valid**, http_step_us_b1 = 1475 (B=8: 3950), at 23.56 agent-hours, parent s0055, category correctness
- hypothesis: track B: formats as s0055 (float32 residual stream, hi+lo BF16 router operands, weights BF16 as stored); bit-identical logits (digest_long, bcheck). A correctness hardening, not a speed-up: every read of a word written by another block (the sentinel protocol's first loads and retries) was a weak load with a cache hint; another run of this project on the same card found that such a load in a spin can miss another SM's later plain store (its finding C11). Here the writers use atomic exchanges and no stale read was ever seen (0 mismatches in 299 repetitions), but the strong form costs nothing: same spill lines, same local timings (1382 / 3805 us), same digests. Expect HTTP unchanged within the noise: 1473 +-6 at batch 1, 3963 +-30 at batch 8.
- evaluator: valid: 1475 us/step at B=1 and 3950 at B=8 over HTTP (online (submit/poll)); in process 1389 us/step at B=1 (step, MBU 0.454)

### #58 s0058: Track A build of the s0057 source (hand-over reads as strong loads)

- status: **valid**, http_step_us_b1 = 1520 (B=8: 3877), at 23.60 agent-hours, parent s0057, category correctness
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0056); same source as s0057 with the track switch; bit-identical logits (digest_long, bcheck on track A). The reads of words written by other blocks are strong loads at gpu scope instead of weak loads with a cache hint (a hardening; s0057 on track B: 1475 / 3950, unchanged). Local track A: same spill lines and timings (1401 / 3744 us). Expect HTTP unchanged within the noise: 1516 +-6 at batch 1, 3874 +-30 at batch 8.
- evaluator: valid: 1520 us/step at B=1 and 3877 at B=8 over HTTP (online (submit/poll)); in process 1402 us/step at B=1 (step, MBU 0.450)

### #59 s0059: Audit defects: over-long served requests and batch > max_batch now raise

- status: **valid**, http_step_us_b1 = 1522 (B=8: 3898), at 23.98 agent-hours, parent s0058, category correctness
- hypothesis: track A: BF16 residual stream and activations, float32 where the Hugging Face path uses float32, weights BF16 as stored (as s0058); kernel unchanged, logits bit-identical (digest_long). Two defects found by the audit of s0054: submit() accepted a request whose prompt + answer do not fit the context and poll() returned it as finished with no tokens; step() and generate() did not check the batch size against max_batch. Both now raise ValueError (tools/api_stress.py tests them). No speed involved: expect HTTP 1520 +-6 at batch 1 and 3877 +-30 at batch 8.
- evaluator: valid: 1522 us/step at B=1 and 3898 at B=8 over HTTP (online (submit/poll)); in process 1403 us/step at B=1 (generate, MBU 0.450)

### #60 s0060: Fragment getter without a conditional load (batch > 2 kernel)

- status: **valid**, http_step_us_b1 = 1531 (B=8: 3842), at 24.80 agent-hours, parent s0059, category gemm
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. The B operand of every mma was a conditional shared-memory load (lane < Ls ? xf[..] : 0) that the compiler predicated in some builds and left as a branch with the address recomputed in others (5 to 23 instructions per tile, seen in the disassembly). Now an unconditional load (lanes without a column read a zero word with stride 0), in the batch > 2 kernel only: 2 instructions per tile. Same bits. Expect batch 8 over HTTP 3898 -> about 3835 us per step (local same-card A/B: -62 us), batch 1 unchanged at 1522 +-6.
- evaluator: valid: 1531 us/step at B=1 and 3842 at B=8 over HTTP (online (submit/poll)); in process 1404 us/step at B=1 (generate, MBU 0.447)

### #61 s0061: Hand-over counters polled from warp 6, not from the leader warp 0

- status: **valid**, http_step_us_b1 = 1514 (B=8: 3829), at 26.13 agent-hours, parent s0060, category sync
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. A warp's load returns behind the exchanges and counter bumps the warp issued before it (about 2 us each, measured in the batch-8 profile), and the thread that polls a hand-over counter was thread 0, in the warp that issues them at the end of most phases. Polling from thread 192 (warp 6, never a leader) removes that wait from the hand-overs. Same bits. Expect batch 1 over HTTP 1531 -> about 1513 us per token (+-6) and batch 8 3842 -> about 3834 (+-30); local same-card A/B over HTTP: -18 and -8.
- evaluator: valid: 1514 us/step at B=1 and 3829 at B=8 over HTTP (online (submit/poll)); in process 1390 us/step at B=1 (step, MBU 0.454)

### #62 s0062: Attention K/V batches pinned with volatile asm; used in the batch > 2 kernel

- status: **valid**, http_step_us_b1 = 1494 (B=8: 3784), at 26.81 agent-hours, parent s0061, category attention
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. Past the staged key groups the attention loads K and V tiles; requested 8 at a time they cost one trip to memory per batch instead of one per tile, but written as plain C loads the compiler moved them down to their uses (batch 1-2 kernel: partly; batch > 2 kernel: the batching had measured nothing). The batch loads are now volatile asm, and the batch > 2 kernel uses them on attention blocks and on main-block pairs. Same bits. The scored run is long-context (positions 64..1088), where this matters. Expect batch 8 over HTTP 3829 -> about 3780 (+-30) and batch 1 1514 -> about 1504 (+-6); local same-card HTTP A/B: -51 and -10.
- evaluator: valid: 1494 us/step at B=1 and 3784 at B=8 over HTTP (online (submit/poll)); in process 1384 us/step at B=1 (generate, MBU 0.456)

### #63 s0063: Copy of the attention output with 4 loads in flight (batch > 2 kernel)

- status: **valid**, http_step_us_b1 = 1497 (B=8: 3735), at 27.70 agent-hours, parent s0062, category sync
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. A main block copies the attention output of the 8 sequences into shared memory one sequence after the other (load, sentinel test, store); when the last output arrives up to seven loads are still to come, one trip to memory each (a timing study with one sequence instead of eight: -100 us per step). Now four loads are issued together and held in registers. Same bits. Expect batch 8 over HTTP 3784 -> about 3741 (+-30), batch 1 unchanged at 1494 +-6; local same-card HTTP A/B: -43.5 and 0.
- evaluator: valid: 1497 us/step at B=1 and 3735 at B=8 over HTTP (online (submit/poll)); in process 1384 us/step at B=1 (step, MBU 0.456)

### #64 s0064: Down phase: one fragment copy up front (compact slots), rounds stream tiles only

- status: **valid**, http_step_us_b1 = 1496 (B=8: 3577), at 28.03 agent-hours, parent s0063, category gemm
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. Batch > 2: each round of the down projections copied its items' hidden-unit fragments with strong loads between two half batches of tiles, so the tile stream stopped once per round (timing study: 210 us per step). Now every fragment of the phase is copied once before the first round into 8 slots per sequence (its 6 experts and the shared expert: 64 slots = the 64 KB buffer at batch 8), and a round only streams 16 tiles per warp; a lane reads its sequence's slot or a zero word. Same bits. Expect batch 8 over HTTP 3735 -> about 3590 (+-30), batch 1 unchanged at 1497 +-6; local same-card HTTP A/B: -146 and 0.
- evaluator: valid: 1496 us/step at B=1 and 3577 at B=8 over HTTP (online (submit/poll)); in process 1384 us/step at B=1 (step, MBU 0.456)

### #65 s0065: Same engine as s0064; the later experiments behind switches, all off

- status: **valid**, http_step_us_b1 = 1492 (B=8: 3571), at 29.05 agent-hours, parent s0064, category cleanup
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. The source of s0064 plus the experiments made after it (timing-study hooks, seven mechanisms that did not pay), each behind a compile-time switch that is off and documented. The default build has the same spill lines, the same bits (digests) and the same local timing as s0064 (HTTP path on one card 1483 / 3619 against 1483 / 3619). Expect no change: batch 1 1496 +-6, batch 8 3577 +-30.
- evaluator: valid: 1492 us/step at B=1 and 3571 at B=8 over HTTP (online (submit/poll)); in process 1385 us/step at B=1 (step, MBU 0.456)

### #66 s0066: Down-phase copy without its pointer array (fewer spills in the batch > 2 kernel)

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3527), at 29.29 agent-hours, parent s0065, category other
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. The one-copy code of s0064 kept the four source addresses of a load batch in an array of pointers for its sentinel path; recomputing them where they are used frees registers (spill line of the batch > 2 kernel 176/52/88 -> 160/24/24 bytes). Same function, same bits. Expect batch 8 over HTTP 3571 -> about 3531 (+-30), batch 1 unchanged at 1492 +-6; local same-card HTTP A/B: -40 and 0.
- evaluator: valid: 1493 us/step at B=1 and 3527 at B=8 over HTTP (online (submit/poll)); in process 1381 us/step at B=1 (generate, MBU 0.457)

### #67 s0067: Routing merge: the lane's candidate from shared memory, not from a stack array

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3499), at 29.69 agent-hours, parent s0066, category other
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. Batch > 2: the merge of a sequence's 12 routing candidates kept the six best in two local arrays that were then read by lane index, so the compiler put them on the stack (14 stack accesses per thread and layer). The lane now reads its candidate from shared memory, where the rank merge left it. Same bits. Expect batch 8 over HTTP 3527 -> about 3512 (+-30), batch 1 unchanged at 1493 +-6; local same-card HTTP A/B: -16 and 0.
- evaluator: valid: 1493 us/step at B=1 and 3499 at B=8 over HTTP (online (submit/poll)); in process 1381 us/step at B=1 (step, MBU 0.457)

### #68 s0068: Batch 7-8: output projection of the 8 pair blocks computed by 8 other main blocks

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3460), at 29.94 agent-hours, parent s0067, category scheduling
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. At batch 7-8 eight attention pairs run on main blocks, which then come out of their attention when every other main block has long had its output-projection tiles and needs only the attention output; their own output projection (tiles, copy, mma, residual) made every block wait. Now block 8+o computes the 16 rows of block o first (same tiles, same fragments, same sums) and its own second; the owner hands its rows over with its sentinel and reads them back. Same bits. This measured +10 us before the down phase was rebuilt (s0064) and -63 after. Expect batch 8 over HTTP 3499 -> about 3432 (+-30), batch 1 unchanged at 1493 +-6; local same-card HTTP A/B: -69 and 0.
- evaluator: valid: 1493 us/step at B=1 and 3460 at B=8 over HTTP (online (submit/poll)); in process 1381 us/step at B=1 (generate, MBU 0.457)

### #69 s0069: Hosted output projection: hosts moved from blocks 8..15 to blocks 24..31

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3448), at 30.79 agent-hours, parent s0068, category scheduling
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. Which 8 main blocks compute the output projection of the 8 pair blocks (s0068) is free; the place of a role on the card matters here (findings 13j). Measured by first host block on the scored workload: 8: 3512, 24: 3503, 120: 3533, the others 3508-3516. Hosts are now blocks 24..31. Same bits. Expect batch 8 over HTTP 3460 -> about 3450 (+-30), batch 1 unchanged at 1493 +-6; local same-card HTTP A/B: -10 and 0.
- evaluator: valid: 1493 us/step at B=1 and 3448 at B=8 over HTTP (online (submit/poll)); in process 1379 us/step at B=1 (generate, MBU 0.457)

### #70 s0070: Attention-output copy: all pending words of a batch re-requested per attempt

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3426), at 31.15 agent-hours, parent s0069, category sync
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. Batch > 2: in the 4-wide copy of the attention output a word that still held the sentinel was waited for alone and the later words of the batch were requested again afterwards, one more trip to memory after the last output is in. Now every pending word of the batch is requested again in each attempt. Same bits. Expect batch 8 over HTTP 3448 -> about 3435 (+-30), batch 1 unchanged at 1493 +-6; local same-card HTTP A/B: -14 and 0.
- evaluator: valid: 1493 us/step at B=1 and 3426 at B=8 over HTTP (online (submit/poll)); in process 1381 us/step at B=1 (generate, MBU 0.457)

### #71 s0071: Same engine as s0070; cycle-counter study build and two off switches added

- status: **valid**, http_step_us_b1 = 1493 (B=8: 3426), at 32.60 agent-hours, parent s0070, category cleanup
- hypothesis: track A: checkpoint BF16 weights, BF16 activations and K/V, float32 where HF uses it; no format differs. The source of s0070 plus code that the default build does not compile or run: the MK_CLK study build (phase stamps from the cycle counter, #ifdef only) and two experiments behind switches that are off (router blocks staging keys in shared memory, two K batches in flight for the pairs on main blocks). Same spill lines, same bits, same local timing as s0070 (HTTP path on one card 1481 / 3495 against 1479 / 3495). Expect no change: batch 1 1493 +-6, batch 8 3426 +-30.
- evaluator: valid: 1493 us/step at B=1 and 3426 at B=8 over HTTP (online (submit/poll)); in process 1380 us/step at B=1 (generate, MBU 0.457)

### #72 s0072: Routing union: two barriers fewer (table zeroed in stage 1, written by stage 2)

- status: **valid**, http_step_us_b1 = 1491 (B=8: 3412), at 33.61 agent-hours, parent s0071, category sync
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. Batch > 2: the union of the six experts per sequence took five barriers per layer in every main block; its byte table moves to the free partial-sum area, is zeroed while stage 1 runs and is written by stage 2 itself, which removes two barriers and one pass (about 0.2 us per layer on the path between the router and the first expert tiles). Expected: batch 8 3426 -> about 3416 us per step (-10, as on the scored workload in process and over the local server); batch 1 unchanged at 1493 (its kernel is not touched).
- evaluator: valid: 1491 us/step at B=1 and 3412 at B=8 over HTTP (online (submit/poll)); in process 1381 us/step at B=1 (step, MBU 0.457)

### #73 s0073: Routing candidates by rank counts in shared memory, no shuffle rounds (batch>2)

- status: **valid**, http_step_us_b1 = 1490 (B=8: 3405), at 34.09 agent-hours, parent s0072, category other
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. The cycle counter puts the candidate stage of the routing (four sigmoids and six rounds of warp maxima per thread, in every main block) at 1.73 us of the 70 us of a layer at batch 8; sorting the four scores first changed almost nothing, so the rounds are bound by their chain of shuffles. Now, batch > 2: each thread sorts its four scores, ranks its best among the 32 of the warp through shared memory, the six best threads write 18 candidates and 18 lanes rank them; same order and ties as the rounds. Expected: batch 8 3412 -> about 3398 us per step (-14, as on the scored workload in process and over the local server); batch 1 unchanged at 1491 (its kernel is not touched).
- evaluator: valid: 1490 us/step at B=1 and 3405 at B=8 over HTTP (online (submit/poll)); in process 1382 us/step at B=1 (generate, MBU 0.457)

### #74 s0074: Batch 1-2: routing merge ranks at most 27 pruned candidates, one per lane

- status: **valid**, http_step_us_b1 = 1484 (B=8: 3408), at 34.41 agent-hours, parent s0073, category other
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. Batch 1-2: the cycle counter puts the merge of the routing (one warp ranks two candidates per lane among all 48, in every main block) at 1.25 us of the 28 us of a layer at batch 1. With the rank R of a warp's best candidate among the 8 best ones, a candidate of rank r in its warp has at least R + r in front of it, so only 21 of the 48 can be among the six best (plus the six of the one warp whose slots straddle the two key ranges of the old tie order, kept in full): one candidate per lane, 28 comparisons instead of 96, same order and ties. Expected: batch 1 1490 -> about 1474 us per token (-16, as on the scored workload in process and over the local server); batch 8 unchanged at 3405 (its kernel is not touched).
- evaluator: valid: 1484 us/step at B=1 and 3408 at B=8 over HTTP (online (submit/poll)); in process 1363 us/step at B=1 (generate, MBU 0.463)

### #75 s0075: Routing union: last pass (unit tables) on all 16 warps, 4 sequences per thread

- status: **valid**, http_step_us_b1 = 1478 (B=8: 3397), at 35.35 agent-hours, parent s0074, category other
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. Batch > 2: the cycle counter puts the last pass of the routing's union (threads 0..255, one expert each, walking the 8 sequences of every expert in use to write weights and the down phase's byte tables) at 0.55 us of a 70 us layer, with 8 of the 16 warps idle. Now threads tid and tid + 256 take the same expert, count its unit the same way and write sequences 0..3 and 4..7. Expected: batch 8 3405-3408 -> about 3400 us per step (-6 in process on two cards, -1 on a third: inside the spread of one evaluation); batch 1 unchanged at 1484 (its kernel is not touched).
- evaluator: valid: 1478 us/step at B=1 and 3397 at B=8 over HTTP (online (submit/poll)); in process 1364 us/step at B=1 (generate, MBU 0.463)

### #76 s0076: Routing merge: table bytes from the ranking lanes; weights on idle warps (B>2)

- status: **valid**, http_step_us_b1 = 1483 (B=8: 3394), at 35.63 agent-hours, parent s0075, category other
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. Batch > 2: the merge stage of the routing (0.53 us per layer on warps 0..7, warps 8..15 idle) ranked the 12 candidates, summed the six scores, divided, and only then wrote the union's table bytes that the next stage waits for. Now the lane that holds a winning candidate writes its table byte itself and the stage ends; the sum (same order) and the weights of sequence b are computed by warp 8 + b during the ballots. Expected: batch 8 3397 -> about 3394 us per step (-3 in process, A/B/A/B on one card: one evaluation cannot tell this from no change); batch 1 unchanged, 1478-1484 (its kernel is not touched).
- evaluator: valid: 1483 us/step at B=1 and 3394 at B=8 over HTTP (online (submit/poll)); in process 1363 us/step at B=1 (generate, MBU 0.463)

### #77 s0077: Routing union, last pass: counts of the earlier warps without a loop (B>2)

- status: **valid**, http_step_us_b1 = 1479 (B=8: 3397), at 35.89 agent-hours, parent s0076, category other
- hypothesis: Track A: formats as the checkpoint's (BF16 weights, BF16 activations and K/V, float32 where the Hugging Face path uses float32); bits identical to s0058. Batch > 2: in the last pass of the routing's union every thread adds the per-warp counts of the warps before its own in a loop whose trip count differs per warp (up to 7 loads and branches on the warps the final barrier waits for); now two vector loads and seven selects. Expected: batch 8 3394 -> about 3390 us per step (-4 in process, A/B/A/B on one card; about the spread between two evaluations of one kernel); batch 1 unchanged, 1478-1484 (its kernel is not touched).
- evaluator: valid: 1479 us/step at B=1 and 3397 at B=8 over HTTP (online (submit/poll)); in process 1363 us/step at B=1 (generate, MBU 0.463)

