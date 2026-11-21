# Run summary (s15-mini-b200-a1, Inkling-2-Mini decode on one B200, track A throughout)

## Best submissions (HTTP, the score; all 32 submissions valid)
| | B=1 us/step | vs vLLM | B=8 us/step | vs vLLM | vs SGLang (B=1 / B=8) |
|---|---|---|---|---|---|
| target (1.10x vLLM) | 3356 | | 4615 | | |
| s0030 (best B=1) | 3307 | 1.117x | 5495 | 0.92x | 1.31x / 1.08x |
| **s0031 (best B=8; = engine/ HEAD)** | 3313 | 1.114x | **5435** | 0.934x | 1.31x / 1.09x |

engine/ at HEAD builds the same kernels as s0031 (the only source change since is RKQ8 kept at 2, which gives the
same B=8 kernel); rebuilt from scratch on the pod: serve B=1 3313, B=8 5441, t_det / t_serve pass.
Track A: checkpoint formats (NVFP4 experts layers 0-19, MXFP8 20-23, BF16 elsewhere), BF16 activations,
unquantized expert inputs. Build: engine/build.sh (fast math; MK_PRECISE=1 for IEEE).
B=1 target met; B=8 target not met.

## Engine
One persistent cooperative kernel per step (152 blocks x 256 threads, grid barriers), instantiated per batch size
(B=1, B=8, generic). Weights stream through per-warp TMA rings (9 items x 2 stages) into mma.sync m16n8k16 with exact
NVFP4 / MXFP8 decode in registers. At B=8 each distinct expert (union of the batch's top-8) is streamed once per
application with the tokens as MMA columns; w13 and w2 as (slot, rowtile) units with per-slot completion counters.
Attention: one grid round of (sequence, kv head, piece) items, 128-key sub-chunks with online softmax. Loop layers
6-17: the message half of the rank-64 gate computed once per pass for all four gates.

## What worked (largest first; details in journal.md)
- Kernel specialised by batch size (s0013) and removing dead paths per kernel (s0017-s0019).
- B=8 MoE as unit-parallel w13/w2 plus per-slot counters instead of barriers (s0007).
- No stack arrays on critical paths (local memory lives in L2: -250 / -290 us), fast math (-121 / -239).
- B=1 deep w13 stream (3 stages in flight), router K split, mixer blocks moved off the o_proj/combine blocks.
- Last stretch: ring stages of 9 items (B=8 -70: no partial stages); in the B=1 kernel, loading a full stage
  into registers and refilling the slot before its MMAs, on the expert streams only (B=1 3442 -> 3307 together with
  9-item stages); dropping the 1-query mixnorm copy in the B>1 kernels (MIXT 0, s0031: B=8 -32 one engine).

## What did not work
- At B=8 the expert streams are limited by the decode rate, not by refill latency: early refill, smaller stages,
  3-stage rings, L2 prefetch of any kind, and splitting units for warp balance all lost.
- Weight arena / memory-placement tricks: lone-process gains that did not transfer to the evaluator.
- Reusing expert weights across the loop passes: between passes the other 11 loop layers stream more than the
  126 MB L2 holds, and one expert does not fit in shared memory.
- Many switches inside the 255-register kernel regress through register allocation / layout alone.
- Measurement: B=8 HTTP repeats spread +-65 us within one submission, and lone-process A/B at B=8 has a ~30 us
  floor tied to the binary's placement. Changes under ~30 us at B=8 cannot be resolved.

## What I would try next (B=8)
1. A distributed residual mix: partial sums over many blocks plus a few exchanges, instead of one mixer block per
   sequence (~6.5 us latency chain, ~97 mixes per step).
2. Restructure the loop gates A0/D0 (~16 us each per loop application at B=8: mixer chain, flag hand-off,
   z rows, barrier), e.g. fuse the z rows into the mix blocks.
3. A different B=8 MoE decode: two rowtiles per item sharing the activation (B operand) loads, which would halve
   the shared-memory reads of the decode loop.
4. Attention sub-chunk size chosen by context (128 keys below ~512, 256 above), if the code growth can be contained.
