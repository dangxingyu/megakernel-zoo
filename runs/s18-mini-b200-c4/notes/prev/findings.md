# Findings of the merge run (s17-mini-b200-ma), each with its measurement

## Which idea came from where (track A throughout)
| engine part | source |
|---|---|
| B=1 kernel (engine/a2) | s15-mini-b200-a2 s0025 (tagged dataflow, 152 x 768) |
| B>=2 kernel (engine/a1) | s15-mini-b200-a1 s0025 (ring stages 9 x 2; base s0024) |
| one engine with both kernels, chosen per cohort (engine/api.py) | this run, s0001 |
| KB1: batch-1 kernel with B a compile-time constant | idea of s15-mini-b200-b2 s0020/s0021 ("K1"); this run s0003 needed the inactive mark sites to make it pay |
| GZ0 (gate phase 0 without mixer blocks), MSGW (warp-per-message phase M) | this run, s0004; MSGW is the a1 form of b2's MSGZ idea (message work once per pass, organised for all sequences) |
| not taken: a2 s0026 (B>=3 attention in a2's kernel), a1 s0026/s0027/s0029/s0030 (a1's B=1 kernel), a1 s0028 (128-key sub-chunks at B=8: 5466 vs 5443 over HTTP), b2's track-B items | |

## The engine's speed follows the register allocator (a1 B=8 kernel and a2 B=1 kernel)
- a1 B=8 kernel, NVFP4 expert decode loop (cuobjdump of mk_kernel_t<8>, tests/sass_hm.py): 540 SASS instructions per
  40 HMMA in the base; 583 with GZ0, 567-604 with MSGW; MXFP8 loop 476 vs 578. In the long form every block scale
  goes through one register (PRMT R5 -> LEA R5 -> FFMA, next PRMT R5) and ~20 `@!UPT UIADD3 URZ` fillers appear:
  the loop is register-starved. No spills are near the loops. Effect: w13 32.0 -> 39.0 us, w2 18.5 -> 21.1 us per
  application, +450..+510 us per step, with the changed code itself bitwise equal (GZ0) and elsewhere.
- Not the cause: code placement (dead padding of 4..33 stores before phase E: +-30 us, no effect on GZ0),
  -maxrregcount 224..248 (loops unchanged), the size of the new code (out-of-line versions also starve the loops).
- Remedy found: keep the replaced path compiled behind a runtime condition that is never true
  (`p.prof_cap == -77777`): the loops return to 540 / 476 and the new code's gain appears (GZ0 -78 us, MSGW -45 us
  more). For MSGW one more variant was needed (no first-row preload). tests/sasssearch.sh: compile 8 variants and
  print their decode-loop lengths in ~20 s; time only those that match.
- a2 B=1 kernel: KB1 (hd.B constant 1) is bitwise equal and has fewer spills (116 -> 80 B) but 3120 vs 2743 us;
  compiled with the (inactive) mkprof mark sites it is 2724 us. Compiler memory barriers at the mark sites do not
  reproduce this. Other cliffs met: BF_UNR 3 (+445), -Xcicc -O2 (+420).

## B=8 accounting by ablation (before GZ0/MSGW; ablations also move the allocation, so they bound, not measure)
Skipping all loop gates -1393 us (58 us per gated application), gate phase 0 -1300, phase 1 -900, phase M -280,
head -142 (base 5718, pos 320-576). a2's kernel spends ~22 us extra per gated application at B=8 (crit path).

## Numerics of the merged engine
- B=1 (a2) vs B=8 slot 0 (a1) on the same tokens: step-0 logits rel 4.0e-3; greedy streams diverge only at
  near-ties (first divergence at token 9 of 40: top-2 margins 0.032 / 0.114 logits; tests/tie.py).
- MSGW vs the old phase M: pass-2 messages rel 1e-7, zm 5e-6 (kernel stopped after the first M phase); end logits
  differ by 2.2e-2 median, the bf16 cascade of the 48 applications. Evaluator s0004: batched median 0.0234 (limit
  0.054), long 0.0198 (0.0475), top-1 0.9712 (HF BF16 0.9619).

## Where the B=8 kernel's warps spend their time (ncu on one step at position 300, s0004 build; tests/ncu8.py)
Whole kernel: issue slots busy 24.7%, 2 warps per scheduler, 8.1 cycles per issued instruction; stall cycles per
instruction: barrier 1.81, long scoreboard 1.48, wait (fixed latency) 1.13, no_instruction (i-cache) 0.97,
short scoreboard 0.39, math throttle 0.35, branch 0.29. Average DRAM throughput 2.36 TB/s.
Warp samples by code region (tests/ncu_buckets.py; the kernel is laid out in phase order): w13 28.7%, w2 + F 17.9%,
o_proj + D + router + top-8 13.0%, gates + attention-input mixes 12.3%, head + out-of-line helpers 11.9%,
attention 7.7%, qkv 7.4%. Inside them the NVFP4 decode loops hold 19% of all samples (MXFP8 loops 3.6%); the
largest single points are warps parked at block barriers behind a thread that spins on a grid barrier or flag
(up to 2.7% each), mbarrier try-wait loops (~5% together), the barrier after the w13 tasks (1.8%: 4 vs 5 tasks
per warp) and the completion counters' MEMBAR.SC.GPU (0.4%).
Reading: the B=8 step is not limited by any one loop; the decode loops are a fifth of it, the rest is a long tail
of hand-offs, block barriers and waits, as in the B=1 kernel.

## The B=1 (a2) kernel under ncu (one step at position 300, tests/ncu1.py)
Stall cycles per issued instruction: barrier 9.57, long scoreboard 8.35 (the tagged polls and the loads behind
them), wait 2.10, not selected 1.23, branch 0.93, no_instruction 0.89, math throttle 0.74, short scoreboard 0.65.
The 24-warp dataflow kernel spends its time waiting on hand-offs (block barriers around polls) and memory
round trips, as its own gap accounting says (~10 hand-offs per application).

## How to work on this engine (for a later run)
- build: engine/build.sh (both kernels in parallel, ~40 s); B=1 kernel flags `-DKB1=1 -DMKPROF` (MK_K1PLAIN=1 for
  the plain a2 s0025 build). The B>=2 kernel's merge-run switches: GZ0/GZK, MSGW/MSGK/MSGV (on); study switches
  in tests/a1_mk_study_variants*.cu.txt (NFW, GZV, W2E, ATM2, NOWAIT13, PADN, CTNI; all measured, all off).
- one-engine-per-process timing: tests/t1.py (B=1 kernel), tests/t8.py / t8tf.py (B>=2 kernel), tests/var1.sh /
  var8.sh (alternating variants), tests/check8.sh (+ bitwise comparison), tests/serve.py (the wrapper's submit/poll).
- before timing any change to the B>=2 kernel: tests/sasssearch.sh / sassall.sh (decode loops 540 / 476 SASS
  instructions per 40 / 18 HMMA; otherwise expect +300..+500 us from the allocation alone).
- correctness: tests/t_ref8.py (vs tests/inkling2_ref.py), tests/limits.py, tests/tie.py, tests/msgstop.py.

## The B=8 NVFP4 decode loop is ALU-pipe limited; moving the scale build to the FMA pipe pays (s0006)
ncu: ALU pipe 788 M warp instructions per B=8 step vs FMA 425 M, tensor 64 M; inside the NVFP4 loop 19% of stalls
are math-pipe throttle. Per item the ALU carried 16 F2FP + 8 PRMT + 8 LEA. The block scale (b << 20) + (120 << 23)
compiled to LEA (ALU); as IMAD (FMA pipe) it is bitwise the same. ptxas rewrites any multiply by a known power of
two into LEA/SHF, even from a once-stored __shared__ variable, so the multiplier must be opaque (a volatile mov of a
__device__ variable, once per consume_tasks call). Result: -76 us at B=8 over HTTP (s0006: 5279).
Multiply-high byte extraction (IMAD.HI) is slow on this card (+55..+420 us).

## MXFP8 weights straight to bf16 bits (s0007, exact; tests/fp8d/fp8d_check.cu)
For an E4M3 byte b = s eeee mmm, the bf16 bit pattern (b << 4) + (b & 0x80) x 240 (the added 0x7800 carries the
sign from bit 11 to bit 15) encodes exactly value(b) x 2^-120: normals as bf16 normals with exponent eeee, subnormals
as bf16 subnormals. Multiplied (mul.rn.bf16x2) by the UE8M0 block scale written as 2^(e-7) it gives the same bits
as the former f16 route (cvt e4m3 -> f16, shift into bf16 x 2^-112, scale 2^(e-15)) for every byte pair and every
scale exponent 100..130 (exhaustive GPU check, 0 mismatches; NaN codes excluded, none occur): bf16 subnormal
inputs survive mul.rn.bf16x2. Cost per weight pair: PRMT + LOP3 on the ALU pipe and 2 IMAD on the FMA pipe, instead
of F2FP + SHF + 2 LOP3 on the ALU pipe (loop 476 -> 417 SASS instructions per 18 HMMA). B=8 -48 us over HTTP.
The same construction does not carry over to E2M1 (NVFP4): its subnormal-like codes (0, +-0.5) map to bf16
subnormals scaled by 2^-126, and no exact compensation fits in the bf16/fp32 exponent range before the MMA.

## NVFP4 block-scale pairs in one conversion (s0009)
The scales of rows g and g+8 for one k16 block are adjacent bytes of the lane's scale word. One exact
cvt.rn.f16x2.e4m3x2 of the word's half (selected by mov.b32 {l,h}, which ptxas folds: no shift) gives both;
f16 -> f32 is HADD2.F32 on the FMA pipe. Versus PRMT + IMAD per scale: NVFP4 loop 541 -> 500 instructions per
40 HMMA, ALU 229 -> 191; bitwise equal; B=8 -90 us over HTTP. Together s0006 + s0007 + s0009 took the B=8 decode
loops from 540 / 476 to 500 / 417 instructions and B=8 from 5355 to 5141 us, all bit-identical.
