# Journal (s15-mini-b200-b1)

## Design (track A base)
One persistent cooperative kernel per decode step (engine/mk.cu), 152 blocks x 512 threads, grid barriers between
phases. Per layer application (48 per token): [A, B gate phases in loop passes 2-3] C q|gate|r|sink|k|v GEMV ->
D attention (64-position units, last unit combines) -> E O-proj (+ epilogue: attention conv, MoE-input dot partials)
-> F MoE input + router -> G routing + W13 (NVFP4/MXFP8 decoded to f16, BF16 inputs scaled exactly into f16) ->
H W2 (+ epilogue: finalize the next entering partial X and the next mix's dot partials). Head: final mix + 200058-row
unembedding. Weights repacked into m16n8k16 fragment order (engine/pack.py), values unchanged.

## Entries
- s0001 (quick): gates pass (numeric median 0.0249, KL median 0.0006); failed timed_steps because step() returned
  the same buffer each call (the timed loop keeps probe tensors). Fixed: fresh output tensor per step.
- Profiling (tests/t_prof.py, t_stamp.py: globaltimer stamps per block before/after every grid barrier):
  barrier ~1.1 us; at B=1 every vector phase 3-10x slower than its loads/compute suggest.
  * xred (transpose reduction) took 8000 cycles: the compiler turned the lane-dependent select into a branch, so the
    shuffles ran divergent (slow WARPSYNC path). Fixed with PTX selp: 550 cycles.
  * Per-thread descriptor arrays (Src S[5] etc.) lived in local memory; moved to shared memory.
  * GEMV inner loops spilled the streamed tiles (STL/LDL at the load line) at the 128-register cap: made the strip
    functions __noinline__ -> QKV GEMV 15 -> 10.5 us, W13 17 -> 10 us (standalone the QKV GEMV is 6.8 us).
  * L2 bulk prefetch of the next GEMV's weights during vector work: slower (vector loads queue behind it). Off.
  * Epilogue-fused residual mixes (W2/O-proj epilogues produce X / attn rows and dot partials): B=1 5.47 -> 5.22 ms,
    B=8 14.2 -> 10.9 ms (in kernel).
- s0002 (full): VALID, 5496 us B=1 / 12427 us B=8 over HTTP (in process 5283). But numeric median rose 0.025 -> 0.040
  (limit 0.054), numeric_long 0.020 -> 0.036 (limit 0.0475), KL median 0.0006 -> 0.0022: an accuracy regression
  between s0001 and s0002 to be found (local per-application test running).
- Root cause of slow straight-line code (measured): inside reduce_parts a 40-iteration thread-0 loop takes 2670
  cycles and 9 independent loads 4700 cycles; the same pattern standalone takes 0.6 us. Hypothesis "cold instruction
  fetch" REFUTED: a second back-to-back call (warm code) was not faster (tests/t_rp2.py); TLB misses also refuted
  (bench/hot.cu with 4000 pages touched: 0.63 us). Partial explanation found later: these per-thread loops do runtime
  integer divisions and serial thread-0 work (thousands of instructions); spills/callee saves add local traffic.
- s0003 (correctness): layer 0's attention-input RMS used the raw embedding's dots (q/k/v scaled ~4% wrong; V's own
  norm hid it at step 0). Fixed: numeric median 0.040 -> 0.026, numeric_long 0.036 -> 0.020, KL 0.0022 -> 0.0008.
  5200 / 11836 us. Lesson: per-application error medians over many positions (tests/t_perapp.py) found it where the
  per-token step-0 check did not.
- Register-allocation lottery (128 regs at 512 threads): unrelated edits moved B=1 by +-5% and the untouched head by
  +110 us. Measured: non-inlined per-strip calls spill ~80 callee-saved registers per call (~160 KB/SM of stack
  traffic per GEMV phase); inlining spills the streamed tiles; splitting phases into functions: 6.8 ms. Only
  same-process A/B on the scored workload (tests/ab.py, +-5 us) decides; separate runs differ by ~100 us.
- s0004: attention K/V/bias loads issued before the q/r prep, per-warp f16 input prep, runtime-indexed arrays removed:
  5041 / 10981 us (A/B predicted 5010 / 10398 at random tokens; real text routes wider at B=8).
- s0005/s0006: device greedy loop (in-kernel argmax keys, token fed back on the device, submit/poll server thread,
  tokens via host-mapped memory): 4959 / 10936 us (s0005 died: poll() before the first submit).
- Expert GEMVs are instruction-bound at ~4 TB/s (bench tests/t_nvbench.py: same speed with a fake cheap decode, any U,
  B=1 or 8). Gate down-projection phases (A, F1) cost only 250 us/step at B=1 when skipped outright (upper bound
  for the loop-gate linearization idea): deprioritized.
- Track B sketch head: certified int8 unembedding on the greedy path. First f16-MMA version barely helped (decode-
  bound); integer MMA (m16n8k32 s8) against two int8 digits of the hidden state streams at 6.8 TB/s; B=8 had 8-way
  smem bank conflicts on the digit rows (stride 4608 B) and a division-heavy candidate scan (25k cycles). Head
  147 -> 105 us (B=1), 157 -> 120 us (B=8); greedy tokens identical at 11000/11000 positions; step() logits
  bit-identical TRACK A vs B (bitcheck). Submitted as s0007 (TRACK=A base) and s0008 (TRACK=B).
- s0007 (track A base, TRACK=A): 4922 / 10836. s0008 (track B, sketch head): 4882 / 10813: -40 us B=1 (predicted
  -40), -23 us B=8 (predicted -40; generate shows -32). All gates incl. hidden pass; step() logits bit-identical
  to s0007's source with TRACK=A.
- s0009 (track A, attention chunk 64/128/256 per step via kernel templates, host picks from the known position):
  4942 / 10527 (B=8 -2.9% as predicted; B=1 +20 us, a separate compile of the same 64 path). A device-side
  dispatch (3 call sites) cost +2.6% at B=1 (kernel stack 656 -> 880 B); a noinline dispatcher lost the B=8 gain.
- Measured and dropped (same-process A/B): 32-position chunks at B=1 (+7 us); deeper GEMV unrolls (W13 U=8 +6%, QKV
  U=12 +0.8% B=8, O U=12 +4%); L1 prefetch of the first QKV round (+0.9% / +1.5%); 2-chunk build per thread (+3% B=8);
  18-load reduction batches (+1%); two accumulators in the NVFP4 strip (0 in engine, +15% standalone); cp.async ring
  for NVFP4 (standalone same 4.1 TB/s as registers: the strip is bound by its per-warp dependency chain).
- Layer-0 token table (track B, TRACK word l0tab): bit-identical to TRACK=A (bitcheck), but only -4 us B=1 / -12 us
  B=8: not submitted.
- ncu source-level stall samples (e0153, B=1): ~32% of warp samples at grid barriers (after attention 8.6%: few blocks
  have units; after W2 6.4%; after O-proj 5.4%); B=8 (e0154): expert decode lines ~20% (waiting for weight loads).
- s0010 (track A): reduce_parts slot lists from constant tables instead of a serial thread-0 loop: 4852 / 10232
  (A/B predicted 4847 / 9745 in process; expected HTTP ~4865 / ~10370). New best of both tracks.
