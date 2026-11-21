# Plan / handover (rewritten at ~11 agent-hours)

Final summary of the run: notes/summary.md (best: s0052, 2880 / 6486 us).

## Final state
- Best track B: s0047 "B MSGZ:N SKH L0T:N K1 KN EXQ:N NG2NI:N BF_UNR=3:N BPX=4:N BPG:1 NI_FINAL=__noinline__:1
  GLIN:1 BPX=8:1 KVC4=0:N RTMASK=0:1 FP4_UNR=6:1": **2879 us (B=1) / 6509 us (B=8)** over HTTP; final s0052 adds BPX=132:N: 2880 / 6486.
- Best track A: s0048 "A K1 BPG:1 BPX=8:1 KN EXQ:N NG2NI:N KVC4=0:N RTMASK=0:1 FP4_UNR=6:1": 2949 / 6730.
- vLLM 3692 / 5077. B=1: 1.28x (target 1.10x passed). B=8: 0.78x.
- Track B against track A: -70 us at B=1 (GLIN, the sketch head), -221 us at B=8 (MSGZ, L0T, sketch head).
- engine/TRACK holds the track-B string; the track-A build is the same tree with the track-A string. Every token is
  documented in engine/build.sh (NAME[=V]:1 = batch-1 kernel only, :N = generic B >= 2 kernel only).

## How the engine is organised now
Two kernels from one source (K1): a batch-1 kernel (constant batch; 99 KB of shared memory, SM1) and a generic
kernel for B >= 2 (KN; 133 KB, SMTRIM). Both under a shared-memory carve-out step (the footprint alone costs: +76 KB
was +658 us at B=8). The generic kernel's experts run from a per-block work queue (EXQ) with the two-column-group
code out of line (NG2NI).

## Where the time is (notes/gap.md)
B=1 ~2.9 ms: gate chain of the 24 gated applications ~0.75 ms (latency), weight phases ~26 us per application
(~15 at bandwidth), hand-offs. B=8 ~6.4 ms: w13 / w2 critical path ~60 us per application (w2 waits for the last
experts' activations), attention ~24 us per application, mixes and gates.

## Open items worth a later run
1. w13 queue interleaved across experts so that all activations complete together (W13IL: the mapping must change
   in both the queue and the round path; the first attempt hung).
2. Attention at B=8 with a second K/V buffer inside the existing 133 KB (masked gain of ATP / ATT2 was ~-100..-300 us).
3. Two-level head sketch (int6 then int8 on surviving strips): -25% of the head's bytes, ~19 us at B=1.
4. a1's B=8 design (TMA weight rings, w2 fixed at compile time) reaches 5608 us at B=8: the structural answer.
