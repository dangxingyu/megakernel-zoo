# Plan (merge run s17-mini-b200-ma), written at ~0.7 agent-h / 06:10

State: s0002 = a2 s0025 kernel at B=1 (2752 us HTTP, 1.34x vLLM) + a1 s0025 kernel at B>=2 (5450 us, 0.94x vLLM).
B=8 target 4641 needs -810 us (15%).

## The three largest B=8 losses (gap.md; ablations in journal.md)
1. Loop gates: 1393 us/step (58 us per gated application; a2's kernel ~22). Phase 0 (mixer-block mix -> flag -> z rows
   -> barrier) is the bulk.
   - (a) every block computes its sequence's mix itself (19 blocks per sequence) and its own z rows: removes the
     mixer -> flag hand-off and the xm L2 round trip (~3 us x 48 = ~150 us). Cost: code in the 255-register kernel.
   - (b) a2's dataflow gate (tagged hand-offs, no grid barriers) in a1: ~-35 us x 24 = -800 us if it carried over;
     a rewrite of 4 phases in a kernel whose allocation moves with any added code (MSGW: +200..300 us from 90 lines).
2. MoE streaming: ~3000 us/step, ~1200 above the decode/bandwidth floor (w13 at ~43 items/us/SM; decode-bound at
   8 warps/SM with 52% issue).
   - (a) more warps per SM for the expert phases (a2's 24 warps: w13 28.4 vs 32 us/app on the same text): needs a
     different kernel shape.
   - (b) tcgen05 MMA from shared memory (decoded BF16 weights staged in smem, TMEM accumulators): fewer register-file
     instructions per item; a large rewrite.
3. Attention 600 us/step (12.4 us/app at ctx 576; K/V bytes ~3 us at bandwidth).
   - (a) software-pipelined PV(k) / scores(k+1) (a1 estimated <= 1.5 us/app); (b) more pieces per (seq, head) with
     fewer sub-chunks each.

## Decision
Every structural change measured so far in either kernel moved its register allocation more than it saved (MSGW,
KB1). With ~3 h left, no rewrite of a phase is affordable to debug; the next work is:
1. sweep of a1's existing switches after the RI=9 change (cheap, background on card 1);
2. ports of any newer a1 / a2 submissions (re-read ledgers at ~07:00);
3. if time remains: gate phase-0 variant (1a) behind a switch, A/B'd one engine per process.

## Status at ~2.6 agent-h (06:40): s0004 = 2722 / 5355 us (B=8 0.95x vLLM; target 4641 not reached)
Done since the plan: KB1 (s0003, -20 at B=1), GZ0 + MSGW with the old paths kept compiled (s0004, -116 at B=8).
The code-generation finding changes what is affordable: a structural change in the a1 kernel is worth trying if
tests/sasssearch.sh shows the decode loops at 540 / 476 (with the replaced path kept compiled if needed).
Next (in order, if time): (1) gated phase 1 + norms at B=8 (largest remaining latency chain), (2) attention
pieces at B=8, (3) MoE w13 -> w2 hand-off (F stage). Each: SASS check first, then one-engine A/B.

## Final state (~2.7 agent-h, 06:40)
Best track A = s0004 (re-evaluated as s0005): HTTP 2722 / 5355 us (B=1 1.36x vLLM, B=8 0.95x vLLM; the 1.10x claim
holds at B=1 only). engine/ is byte-identical to s0004; study variants are in tests/a1_mk_study_variants*.cu.txt.
Tried after s0004 and not kept: NFW (+21..+100), EISS early refill at B=8 (+324), deeper rings (+280), GZV 64 (+6..+28),
ATM2 (+19), B=1 build points (noise). For a later run, the open structural items are unchanged (gap.md): the MoE
decode loop at ~30% issue (needs more warps per SM or a different MMA path), the gated chains (~1080 us at B=8),
attention (~600 us at B=8); any of them must pass the decode-loop SASS check (tests/sasssearch.sh) first.

## Final state (~3.5 agent-h): best overall s0007 = 2729 / 5231 us (B=1 1.36x vLLM, B=8 0.976x vLLM)
(board's B=1 best is s0006 at 2720: same B=1 kernel, the 9 us is evaluator noise). engine/ = s0007.
What moved B=8 in the second half: reading ncu pipe counts (ALU vs FMA) rather than stall totals; SCIM 4 and FP8D
moved decode work from the ALU to the FMA pipe, bitwise equal. Next for a later run: the B=8 MoE is now issue-bound
at ~54 instructions per NVFP4 item with 2 warps per scheduler; the remaining large item is the barrier/hand-off
structure (35-45% barrier stalls in the non-MoE regions), which needs a dataflow (a2-style) rather than a
barrier-per-phase organisation of the B=8 kernel.

## Final state (~4.5 agent-h): s0010 = 2726 / 5119 us (B=1 1.36x vLLM, B=8 0.997x vLLM, 14 us short of parity)
engine/ = s0010. s0009 (decode) and s0010 (a1's PMAXC 2 + SCNT_REL 1 re-measured) came last.

## Final (~4.5 agent-h): engine/ = s0011 (2727 / 5116 us; best B=8: s0011 5116; best B=1 on the board s0006 2720,
same B=1 kernel). B=8 0.998x vLLM (11 us short of parity), B=1 1.36x.

## Final (4.75 agent-h): engine/ = s0014: 2728 / 5093 us. B=1 1.36x vLLM; B=8 1.002x the vLLM image (5105), 0.997x the
vLLM source build (5077). s0011-s0013 were the s0010 kernel (FMA2 define missing); s0014 is the first with FMA2.
