# Plan (rewritten 2026-11-21 ~5.6 agent-h; previous at 3.6 h; status updated ~7.7 h)

Direction (operator 21:40): batch 8 is the main target, batch 1 must stay valid.
Best: s0006 HTTP 3961 (B=1), 7291 (B=8); in process at ctx 64-320: 3874 / 6860 (HEAD ~6590 at B=8).
Targets 3356 / 4615. a2 (other run): 3305 / 7257.

## B=8 step at pos 300 (natural text, mkprof, HEAD), per application (48)
MoE ~75 us: w13 35 (U ~37 experts, 110 MB: ~23 us at 5.5 TB/s + ramp; warp imbalance 10-15%), w2 20 (incl. barrier),
top-8 + routing 8, F stage/accum/fin 11. Attention 14 (grows ~2 us per 100 context at B=8: HTTP mean ctx ~576).
Mixes: A norm 11, D mix 7.5; loop gates (24 applications): A0 18, D0 19.5, A1 7.4, D1 10.3 (~1.3 ms per step).
qkv 10, o_proj 8.5.

## Facts that decide what is worth doing
- The decode ring at 12 items x 2 stages: 71 NVFP4 items/us/SM from L2, ~58 at full grid from HBM (5-6 TB/s
  steady); B=8 time follows the decode rate (per-stage overhead), not bytes in flight. L2 prefetch of any kind
  (cp.async.bulk.prefetch) hurts every time (three separate experiments).
- The kernel sits at 255 registers: changes to call sites (e.g. mixnorm's argument passing) move spills into hot
  loops (+100..+600 us) even when the changed function itself gets faster in isolation. Changes inside a phase's own
  code (new F path, E fin callback) were safe.
- mixnorm in the engine: 6.8 us, of which 2.9 us is the load phase (source codes / query pointers from stack arrays =
  local memory in L2 -> two dependent round trips). Fixing it needs the arguments out of local memory without
  touching register allocation elsewhere (not achieved yet).

## Next (in order)
1. w13 warp balance: half-unit tasks for the remainder units (or all) with a shared-memory completion count;
   expect -3 us per application at B=8 (-150 us).
2. Attention at long context, B=8: 256-key items are 2 rounds at ctx 1000; per-item fixed costs (q/r loads, clip,
   partial write + atomic, merge). HTTP B=8 is ~430 us above the ctx 64-320 figure.
3. Gate phase 1 at B=8 (5.4 us vs 2.3 at B=1) and z rows (3 us): cheap restructuring inside the gate code.
4. mixnorm arguments (see above) only with a way to keep mk_kernel's allocation stable (e.g. a per-application
   argument table in shared memory written once at the application start).

## Status at ~7.7 agent-h (s0015: HTTP 3573 B=1 = 1.033x vLLM, 5645 B=8 = 0.90x vLLM; a2 at 3012 / 6231)
Done since the rewrite: w2 as (slot, rowtile) units at B=8 + per-slot counters (s0007), attention pieces in one round
(s0008), no stack arrays on critical paths (s0009/s0010), fast math (s0011), B=1 deep w13 (s0012), kernel specialised
by batch size (s0013: the largest single gain since s0006), mixnorm <1,1> + z loads at B>1 (s0014), small (s0015).
What is left is structural: the B=8 MoE runs near the ring's ~5 TB/s streaming rate with a ~12% warp imbalance that
every splitting scheme lost to overheads; mixes are bounded by mixnorm's ~6.4 us latency (icache-bound code) times
~97 per step at B=8; B=1 is a chain of ~8 barriers and short latency-bound phases per application.
Ideas not attempted (too large for the time left): speculative w13 prefetch of the previous pass's experts into
shared memory (5/8 overlap in loop passes), a distributed mix (partials + 3 exchanges) for B=8, two attention items
per block on warp halves.

## Final state (~8.3 agent-h): s0017 HTTP B=1 3515 us (1.050x vLLM, 1.23x SGLang), B=8 5645 us (0.90x vLLM)
HEAD = s0017 + the B=8 round-planning removal (noise level, not submitted separately). Build: engine/build.sh
(fast math; MK_PRECISE=1 for IEEE), three kernel instantiations (B=1, B=8, generic).
Remaining structure-bound costs: B=8 MoE ~50 us per application at the ring's streaming rate; ~97 mixer-bound mixes
per step at B=8 (~8 us each); B=1 per application ~72 us of latency chains (~8 barriers, attention item chain ~8 us
+ merge, routing 2.3 us + w13 decode ~4 us after ~2.5 us latency).

## Update (~9.5 agent-h): s0020 HTTP 3489 (B=1, 1.058x vLLM) / 5593 (B=8, 0.91x vLLM). With one engine per process
the B=8 A/B is exact to +-1 us; every remaining knob and the earlier B=8 rejections re-checked with it hold.

## Update (~11.6 agent-h): s0027 HTTP 3350 (B=1, 1.102x vLLM: target met) / 5455 (B=8, 0.93x vLLM)
Since s0020: mixer relocation, router K split, gate F/G rotation, ring 9 x 2 (B=8), per-kernel ring stages (B=1 6 x 2),
early ring slot refill in the B=1 kernel (-70 lone-process: B=1's streams are refill-latency bound; B=8's are decode
bound: every refill/stage-depth variant there lost). s0028 (pending): 128-key attention sub-chunks at B=8.
B=8 still 0.93x vLLM: MoE at the ring's decode rate (w13 31 + w2 18 us per application), loop gates A0/D0 ~16 us
each per loop application (mixer-latency chains), mixes ~9 us. Not reachable in the time left: a distributed mix,
a different B=8 MoE decode (2 rowtiles per item sharing the x loads: halves the B-operand shared-memory reads).

## Final state (~12.3 agent-h): s0030 HTTP B=1 3307 us (1.117x vLLM, 1.31x SGLang; target 3356 met); s0029 B=8 5457 us
(0.93x vLLM, 1.09x SGLang; target 4615 not met). HEAD = s0031 (MIXT 0): B=1 3313, B=8 5435 (best B=8).
HEAD builds with engine/build.sh (track A only: checkpoint formats, BF16 activations, unquantized expert inputs).
Per-kernel switches that matter: RIB1 (B=1 ring stage items, 9), EISS (early refill bit per kind per kernel, 0x06),
CH2 (B>1 attention sub-chunk, 128), RKQ (router K parts per kernel), MIXHI, GROT.
For whoever continues on B=8: the MoE streams are decode-rate bound at 9-item stages (stage-count and refill
variants all lost); the remaining big items are the loop gates A0/D0 (~16 us each per loop application: a mixer
chain + flag hand-off + z rows + barrier) and the ~9 us mixes; both need a distributed mix rather than tuning.
