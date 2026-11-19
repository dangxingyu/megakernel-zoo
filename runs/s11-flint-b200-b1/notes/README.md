# Notes of run s11-flint-b200-b1

- summary.md: results per submission, what was built, what was dropped, where the time is, method notes.
- headsketch.md: the certified int8 sketch of lm_head (track B): format, bound, verification, cost.
- gap.md: the step accounted (measurements and probe upper bounds, with the job ids).
- plan.md: the last step-back and the handoff (leads measured but not taken).
- journal.md: every experiment in order: expected, measured, concluded (items 1-211).
- prev/: the notes of the runs that built the inherited engine (s4-flint-b200-a2).

engine/ = s0027 (engine/TRACK = B: the certified int8 head sketch on s0026's kernel; bit-identical to track A): 1352.6 us/token at B=1, 3333 at B=8 over HTTP. TRACK=A builds s0026 (1393.1 / 3339: the best track-A figure).
Build switches: engine/TRACK = A (checkpoint formats), B (A + head sketch on the greedy paths, bit-identical logits),
F (the previous run's float32 stream, not bit-identical). MK_HSK_OFF=1: track B with the exact head (test).
MK_PAD (api.py, default 512): bytes allocated on the device before the engine's own (buffer layout, journal 62-73).

## How to measure (learned the hard way in this run)
1. A/B two builds in ONE job on ONE card, alternated, calibration on: tools/ab2.sh (in-process generate) or
   tools/gab.sh (served path the way the evaluation's engine process runs it: numpy prompts, nothing allocated on
   the device before the Engine). /tmp is shared by the pod's cards: every script keys its files by $MK_CARD
   (journal 49).
2. Batch 1: gab.sh (GL_B=1). Prompt rows do not matter at batch 1 (journal 90).
3. Batch 8: gab.sh with GL_B=8 GL_CLIENT=<seed> (tools/client_like.py: prompts built like the HTTP client's).
   tools/ids.npy rows (random tokens) route differently and mislead at batch 8 (s0011, journal 100-101). Effects
   under ~15 us at batch 8 do not survive one evaluation (s0012, journal 104).
4. Same-card baselines differ by up to ~10 us between cards 1-3; compare only inside a job.
5. Numerics-changing edits: tools/chk.sh (bcheck B=2..8 bitwise, digests), tools/vfy.sh 0/1/2/3 before a submission
   (vfy.sh builds the defaults: put the change in the source, not in MK_EXTRA).
6. Timelines: tools/clkcmp.sh (MK_CLK, one layer, slowest blocks); tools/bt_clk.py with MK_IDS for other text;
   tools/bt_gen.py: the same stamps in the generate() path (one launch), with where the layer starts in the step.
8. Stall sampling per SASS instruction: ncu through mk exec on tools/ncu_step.py (journal 167); a pmevent marker
   (PMTRIG in SASS) finds a loop in the 14k-instruction kernel.
7. Allocation: tools/res.sh / tools/spill_lines.sh show frames and where local memory is used; a frame jump
   (176 -> 192 B at batch 1) predicts a slow build.

Kept experiment sources and patches: tools/mk_ah2_asplit.cu.bak, mk_ah3_asplit3.cu.bak (attention split),
mk_epf.cu.bak, mk_qpfx.cu.bak (prefetch variants), vdef.patch, kqt2.patch, carve.patch, xplace.patch, naccd.patch,
nacc_phase.patch, sdpre.patch; tools/mkprobe.py (timing probes: WSKEL bits).
