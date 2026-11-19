# Run s9-qwen38-27b-b200-f1: Qwen3.8-27B on one B200, track A past 1.10x at both batch sizes

**Final: track A s0022: 8445 us per token at B=1 and 8878 us per step at B=8 over HTTP (1.136x / 1.156x SGLang 0.5.23
on the same card; the target was 1.10x = 8720 / 9330). Track B s0021, exact 12-bit weights: 7665 / 8102 (1.252x /
1.267x); s0023 is the 12-bit build of the final source, unchanged in behaviour from s0021 and measured at 7668 / 8114 (the evaluator's repeatability: 2 us at B=1, 12 us at B=8).** `engine/` is the
s0022 source (`engine/TRACK` = `A`; `A W12` builds s0023).

Inherited: s5's s0016 (track A, 8886 / 9843 us per token at B=1 / B=8 over HTTP in the operator's same-card rematch)
and s0017 (track B, 8117 / 8864). SGLang 0.5.23 on the same card: 9594 / 10265; the track-A target: 8720 / 9330.
The earlier runs' notes are in `*_s5.md` and `*_s4.md`; this run's journal is `journal.md`, its findings `findings.md`
(S1-S12).

## Result (HTTP, the evaluator's client; us per token at B=1 / us per step at B=8)

| submission | track | what changed | B=1 | B=8 | vs SGLang rematch |
|---|---|---|---|---|---|
| s0001 | A | inherited engine, stats() label fixed | 8885 | 9836 | 1.080x / 1.044x |
| s0002 | A | weight-stream loads with `ld.global.cs` instead of `.cg` | 8701 | 9453 | 1.103x / 1.086x |
| s0003 | A | B=8 recurrent op in one pass per block (lmix_tri) | 8694 | 9222 | 1.104x / 1.113x |
| s0004 | A | barrier-time prefetch re-tuned: near 160 KB, far 64 KB | 8641 | 9145 | 1.110x / 1.122x |
| s0005 | A | served tokens written by the kernel into host-mapped memory | 8628 | 9134 | 1.112x / 1.124x |
| s0006 | A | prefetch schedule: consecutive far pieces, idle blocks take their whole share | 8587 | 9114 | 1.117x / 1.126x |
| s0007 | B | the s0006 source built with exact 12-bit weight storage | 7847 | 8360 | 1.223x / 1.228x |
| s0008 | A | attention layer as one op (a block per 256-key chunk, sentinel hand-overs) | 8492 | 9077 | 1.130x / 1.131x |
| s0009 | B | the s0008 source built with exact 12-bit weight storage | 7706 | 8253 | 1.245x / 1.244x |
| s0010 | A | B=7-8: RMSNorm partials by the four prefetched warps, r finished early | 8494 | 8982 | 1.130x / 1.143x |
| s0011 | B | the s0010 source built with exact 12-bit weight storage | 7709 | 8184 | 1.245x / 1.254x |
| s0012 | A | B=7-8 recurrent op: next state column's loads in flight during the current one | 8492 | 8956 | 1.130x / 1.146x |
| s0013 | B | the s0012 source built with exact 12-bit weight storage | 7708 | 8166 | 1.245x / 1.257x |
| s0014 | A | B=7-8: conv-history rows prefetched during in_proj; far piece 80 KB | 8492 | 8948 | 1.130x / 1.147x |
| s0015 | B | the s0014 source built with exact 12-bit weight storage | 7708 | 8193 | 1.245x / 1.253x |
| s0016 | A | B >= 4: attention blocks ask for their K chunk at the barrier before the op | 8494 | 8905 | 1.130x / 1.153x |
| s0017 | B | the s0016 source with 12-bit storage (conv-history prefetch off there) | 7710 | 8131 | 1.244x / 1.262x |
| s0018 | A | grid barrier: the block's prefetch requests after its counter increment | 8466 | 8899 | 1.133x / 1.154x |
| s0019 | B | the s0018 source built with exact 12-bit weight storage | 7689 | 8116 | 1.248x / 1.265x |
| s0020 | A | grid barrier: one prefetch piece per poll, none after the release | 8455 | 8880 | 1.135x / 1.156x |
| s0021 | B | the s0020 source built with exact 12-bit weight storage | **7665** | **8102** | **1.252x / 1.267x** |
| s0022 | A | far prefetch piece 128 KB for the BF16 engine (host constant, swept on the evaluator's card) | **8445** | **8878** | **1.136x / 1.156x** |
| s0023 | B | 12-bit build of the s0022 source (behaviour of s0021: a repeat measurement) | 7668 | 8114 | 1.251x / 1.265x |

All 23 submissions are valid. Every track-A submission's logits are bit-identical to the inherited engine's (hash of
20 steps at B=1..8 before each submission; the gates of every evaluation show the same numbers: numeric max 0.14 of
0.38 allowed, numeric_long max 0.457 of 0.592). The predictions written before each result were within 12 us at
B=1 except s0002 (+26), s0008 (-43) and s0020 (+21), and within 30 us at B=8 except s0015 (+38: see below). Track B is bit-identical to track A by construction and by check (tools/w12_full.sh on the submitted
source: all 25,621,954,560 stored weights, all 257 GEMM outputs at B=1 and B=8, logits of 48 steps).

## What worked, in order of size

1. **`ld.global.cs` for the weight streams (S2): -2.1 % / -3.9 %.** One token in the load instruction. With `.cg` the
   51 GB of weights a step reads pass through the 129 MB L2 as ordinary lines; with the cache-streaming operator
   they stop displacing what a step touches once (activation buffers, state, descriptors). It also removed the B=8
   penalty the earlier runs had seen in gate_up and not explained (S1: the write-back of 24.6 MB of dirty recurrent
   state per layer competing with the next weight stream).
2. **The B=8 recurrent op in one pass per block (S5): -2.4 % at B=8.** 384 units on 152 blocks ran in two rounds, each
   with its own input phase and four half-block barriers. Now: all inputs in one phase, a third unit shared by the two
   halves of a block (one state column pass each), one norm phase. Own kernel entry point, so B=1 code is untouched.
3. **Prefetch, re-measured in the cs regime (S6): -1.2 % / -0.3 %, then -0.5 % at B=1.** Larger near piece, a far piece
   before small ops, consecutive pieces instead of the same first bytes, and at B <= 3 the last 32 blocks without
   recurrent work fetch their whole out_proj share. Host tables only.
4. **One op per attention layer (S8): -1.1 % / -0.4 % over HTTP.** Only the fourth form paid: a block per 256-key chunk,
   sentinel hand-overs, relaxed read-back of the K/V append.
5. **Tokens straight into host-mapped memory: -0.15 %.** No device-to-host copy and event between two served passes.
6. **Start-of-GEMM work on the prefetched warps (S9): -1.0 % at B=8.** At 5-8 columns every warp loaded its RMSNorm
   partials before streaming; now warps 0-3 (4-6 us of slack: their shares are the block's prefetched bytes) load all
   of them and warp 0 finishes r at once. In the B >= 7 kernel only.
7. **The B=8 recurrent op again (S10): -0.3 %, then -0.1 %.** The next state column's loads in flight during the
   current column's arithmetic; the conv-history rows asked from L2 during the preceding in_proj (BF16 engine only:
   on the 12-bit engine it costs 25 us, which s0015 showed before I had timed it there).
8. **The attention blocks' K chunk requested one barrier earlier (S11): -0.5 % at B=8.** Host table only.
9. **The grid barrier's own prefetch out of the way of its signalling (S12): -0.5 % at B=1, -0.3 % at B=8.** A block's
   prefetch requests used to be issued alongside its counter increment and all at once; now after the increment, one
   64 KB piece per poll, none after the release. Found from the final per-op profile (an op that had become 2 us
   longer when the barrier after it got a larger request).

## What did not work (each measured; details in findings S3, S4, S7, S8 and the journal)

- A relaxed atomic add in the grid barrier (S7): -157 / -196 us and identical logits in a short test, but bench/rel.cu shows
  the increment overtaking the block's own stores when the memory system is idle (864 stale reads in 1.25e11; none
  with the release add). Not used. The release is ~0.45 us per barrier (390 per step) and cannot be dropped.
- Deferred commits of the recurrent state (state stored every 4th step, exact replay of saved rank-1 updates,
  bit-identical): -1.5 % at B=8 with `.cg` loads, -0.7 % with `.cs`, less than lmix_tri and not combinable with it
  as written; left in the source behind MK_DEFER (off).
- Per-warp early prefetch of the next GEMM (+4.4 %), next-op descriptor preload (+1.4 %), prefetch of the next
  epilogue's gain and output lines, prefetch pieces from several warps at once, a kernel pruned to its own GEMM
  instances (+33 % in every GEMM through register spills), compact one-column activations for BF16, cs loads for the
  K/V caches (L1 behaviour unverified) and other cache operators for the recurrent state.
- Attention as one op with one block per (sequence, kv head) (the block streams 0.6-1.1 MB), and with acquire/release
  counters between blocks (each strong atomic ~3 us); its score warps pre-loading K during the query prep (448 bytes
  of spill, +65 us).
- Around the B=8 recurrent op: conv-history rows as 24 separate requests from one warp (+85 us), conv weights
  prefetched (no gain), the state asked for one barrier earlier (+18 to +28 us); norm gains copied to shared memory
  for the residual epilogue (-8 / -3 us with +6 for the code: dropped); R_ASYNC_MAX = 8 (+115 us at B=8).
- A mistake of mine: s0015 (track B) carried the conv-history prefetch, which I had checked on the 12-bit build for
  bit-identity but not timed there: +27 us at B=8 against s0013. Timed afterwards (+25 us), compiled out of the W12
  build, s0017 is the corrected track-B build. Every later change was timed on both builds before it was submitted.
- Not submitted: the attention hub reading its own chunk's scores from shared memory and the V strips requested after
  the score loads (bench/variants/mk_hubown_vlate.cu): bit-identical, B=8 -15 to -24 us, but B=1 -12 us on one card,
  +7 on another, and +10 to +16 on the 12-bit build.

## What limits it now (measured, B=1, context ~620, 8.47 ms per step in process)

- Weight streams at the card's rate: 51.25 GB at ~7.3 GB/ms is ~7.0 ms.
- ~3.3 us of fixed cost per GEMM op x 257 (the stream-end spread inside a block, the cut-tile hand-over, the
  barrier's release and poll): ~0.85 ms. Of this ~0.45 us per barrier is the release (S7).
- The recurrent op: 8.7 us x 48 at B=1 (12.6 us at B=8); the attention layer: ~17 us x 16 at B=1 (21 at B=8);
  lm_head 350 us.

## Next

- The recurrent op at B=8 is still ~4 us per layer above B=1 (S10): its input phase (3.3 us) and the 3.4 us of DRAM its
  24.6 MB of state need per layer are what is left.
- B=1 moved only through changes that are not B=1 code paths of their own (prefetch tables, the attention op, the
  barrier): the same ideas that paid at B=8 (slack warps, history prefetch) cost there what they gain.
- Per GEMM at B=1 the time from the bulk of the warps ending their stream to the op's end is still 5.7-9 us (final
  profile in journal.md): slowest warp of a block +1.3 to +2.1, slowest block +0.7 to +2.8, hand-over and epilogue
  +2.2, last arrival to release +1.8 to +2.1. The last item is what S12 touched; the others need a way to rebalance
  work that keeps the summation order.
- The attention op is 3-4x its arithmetic: ~9 dependent memory round trips; a phase timer that does not use global
  atomics (they distort a 15 us op) would say which.
- Anything that shortens the per-GEMM tail without touching the summation order (bit-identity kept every gate where
  it was, which is why no rounding was changed in this run).

## Building and checking

- `engine/TRACK` = `A`: the track-A engine. `A W12`: track B from the same source.
- Before a track-A submission: `tools/abn.sh` (same-card alternating events + logits hash, `HASH_BS=1,2,3,4,5,6,7,8
  HASH_T=20`), `tools/att_check.py` (one-op attention against the four ops, 1150 steps), `tools/test_engine.py --time
  128`, `tools/online_time.py` (served path against greedy), `tools/stress.py` (determinism). Track B in addition:
  `tools/w12_full.sh` with `engine/TRACK` = `A W12`.
- Environment switches (defaults in api.py): MK_PF_NEAR / MK_PF_FAR / MK_PF_CUM / MK_PF_IDLE_N / MK_PF_FAR_ATT
  (prefetch table), MK_HOSTMAP, MK_ATT_FUSE, MK_DEFER; build switches in mk.cu: LDW_K, LM_TRI_OFF, LMPF_EARLY.
- Experiments that were not kept: `bench/variants/` (mk_pfw.cu, mk_cold.cu, mk_attall.cu + api_attall.py, mk_triprof.cu,
  mk_ringpf2.cu, mk_rpart_inline.cu with their runners), `bench/rel.cu`. Sources of the submissions as they were:
  mk_attdist.cu (s0008), mk_s0010.cu, mk_s0012.cu.
