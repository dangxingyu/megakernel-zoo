# Run s17-mini-b200-mb (merge, track B): summary

| sid | track | B=1 HTTP | B=8 HTTP | what | idea from |
|---|---|---|---|---|---|
| s0001 | A | 2900 | 5860 | a2 s0025 engine unchanged (merge base) | s15-mini-b200-a2 s0025 |
| s0002 | B | 2949 | 6511 | b2 s0047 unchanged (calibration: this card 0 is slower at B=1) | s15-mini-b200-b2 s0047 |
| s0003 | A | 3190 | 5701 | K1 (batch-1 compile: +384 us, a cliff) + ATT3 | b2 s0020 (K1), this run (ATT3) |
| s0004 | A | 2911 | 5701 | ATT3 in the single kernel: B>=3 attention by 3 teams per block, bit-identical | this run |
| s0005 | A | 2888 | 5464 | dual engine: a2 lineage at B=1, a1 s0025 at B>=2 | s17-mini-b200-ma s0002 (structure), a1 s0025 |
| s0006 | B | 2867 | 5480 | + SKH head sketch on the B=1 engine (fast-math-safe certificate) | b2 s0009 (idea), ported here |
| s0007 | B | 2862 | 5364 | a1 kernel with ma's gate phase 0 / message phase | s17-mini-b200-ma s0004 |
| s0008 | A | 2892 | 5349 | track A of the s0007 tree (best track A) | |
| s0009 | B | 2838 | 5352 | start-up placement tuning of the B=1 engine on the evaluating card | b2 s0010 (method) |
| s0010 | B | 2838 | 5347 | final-source check of s0009 (reproduced) | |
| s0011 | B | 2837 | 5377 | + placement tuning of the a1 engine (TUNE8): 11 us spread on card 0, no gain, off | |
| s0012 | B | 2839 | 5335 | + SKH in the a1 engine (B>=2) too | b2 s0009 (idea), ported here |
| s0013 | B | 2764 | 5366 | B=1 engine compiled for batch 1 with inactive mark sites (KB1+MKPROF); fix of an unguarded profile write in SKH: final | s17-mini-b200-ma s0003 |
| s0014 | A | 2799 | 5369 | track A of the s0013 tree | |
| s0015 | B | 2767 | 5265 | a1 kernel + ma's expert decode changes (NVFP4 scales on FMA pipe, MXFP8 to bf16 bits) | s17-mini-b200-ma s0006/s0007 |
| s0016 | B | 2767 | 5240 | faster sketch-digit stage in the a1 head (same digits) | this run (subagent) |
| s0017 | A | 2800 | 5221 | track A of the s0016 tree (SKH:8 does not pay at B=8 in this layout) | |
| s0018 | B | 2757 | 5167 | a1 kernel + ma s0009 (NVFP4 scale pairs by one e4m3x2 conversion); SKH only at B=1 | s17-mini-b200-ma s0009 |
| s0019 | A | 2801 | 5170 | track A of the s0018 tree (best evaluated track A) | |
| s0020 | B | **2770** | **5128** | + a1 settings PMAXC 2 / SCNT_REL 1 (attention pieces of <= 2 chunks): final | s17-mini-b200-ma s0010 |
| s0021 | A | 2806 | 5155 | track A of the final tree | |

vLLM 3707 / 5105, SGLang 4329 / 5934. Final track B s0020: 2770 / 5128 = 1.34x vLLM at B=1 (target 1.10x passed), 0.996x at B=8 (not passed;
4641 needed). Track B minus track A on the final tree (s0020 vs s0021): -36 us at B=1 (SKH); at B=8 the engines are identical
(-27 is evaluation noise).

Track-B items measured and not taken: GLIN (+86..110 us at B=1 on this base), MSGZ / L0T (moot once B>=2 is the a1
kernel, which already computes the messages per pass; L0T on a1 +28..33), K1 (+384), placement tuning of a1 (no
spread). New exact property work in this run: none beyond b2's list; the weights study of b2 (weights_study.md)
found no exact redundancy worth its decode, and this run's measurements agree that the step is latency, not bytes.
How to build: engine/TRACK "B SKH KB1 MKPROF PMAXC=2:8 SCNT_REL=1:8" (best, s0020) or "A KB1 MKPROF PMAXC=2:8 SCNT_REL=1:8" (track A of the final tree, s0021); see engine/build.sh.

## Final
- **Best submission: s0020 (track B), 2770 us/token at B=1 and 5128 at B=8 over HTTP** (1.34x / 0.996x vLLM).
  engine/ holds exactly this (TRACK "B SKH KB1 MKPROF PMAXC=2:8 SCNT_REL=1:8"). The board's lowest B=1 headline is
  s0018 (2757 / 5167): same B=1 engine, so the 13 us difference is evaluation noise; s0020 is kept for its real
  -39 us at B=8. Track A of the same tree: s0021, 2806 / 5155.

## What worked (with the measured gain)
- Dispatch by batch size to two engines, a2's kernel at B=1 and a1's at B>=2 (structure from s17-mini-b200-ma
  s0002): -237 us at B=8 against a2 with ATT3 (s0004 -> s0005).
- The sibling run's a1 kernel changes (ma s0004 / s0006 / s0007 / s0009 / s0010): 5464 -> 5128 at B=8 together.
- KB1 + the inactive mkprof mark sites for the B=1 kernel (ma s0003): -75 us at B=1 (s0013). This is a layout
  effect: it needed a fix of an unguarded profile write in the SKH port.
- SKH head sketch in the B=1 engine (b2 s0009, ported with a fast-math-safe certificate): -21..-36 us at B=1,
  exact (greedy tokens and logits identical to track A).
- Start-up placement trials of the B=1 engine's polled buffers (b2 s0010 method): -24 us at B=1 on card 0.
- ATT3 (this run: three 8-warp teams per block for B>=3 attention, bit-identical): -158 us at B=8 on the a2
  kernel (s0004). Superseded once a1 served B>=2.

## What did not work (measured)
- K1, the plain batch-1 compile of the a2 kernel: +384 us (a code-generation cliff).
- GLIN (linearised attention gate): +86..+110 us at B=1.
- MSGZ / L0T on a1: a1 already computes the messages once per pass; L0T +28..+33 us.
- SKH in the a1 head (SKH:8): exact, but no gain against its track-A twin (s0016 5240 vs s0017 5221).
- Placement trials for a1 (11 us spread on card 0); MKPROF marks in a1 (+315 us).
- More bytes in flight in a2's w2 (W2WG / W2_UNR). Load-depth sweeps of the final B=1 kernel: all worse, and three
  of them fall into the slow code-generation mode (~3140 us).

## What I would try next
1. **B=8 (0.5 ms short of the 1.10x target):** a pipelined MoE in the a1 kernel, starting each expert's w2 as soon
   as its w13 strips are done, via a work queue. w13 + w2 are ~50 us per application, and w2 runs at under half
   w13's per-SM rate because it waits for the end of w13.
2. **B=1:** find out what sets the two code-generation modes of the batch-1 kernel (diff the hot loops' SASS and
   their instruction-cache sets). Then shorten the loop gate chains (~0.7 ms per step), where every precompute
   attempt so far (GLIN, MSGZ, a2's message precompute) made F/G and the norm slower. A test of polling
   contention there is the first step.
3. Make the B=1 head's exact pass speculative (each block recomputes its best strip before the barrier; ~-10 us).
