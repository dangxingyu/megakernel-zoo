# s16-mini-b200-c1: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 10.0 agent-hours. 22 submissions so far, 22 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine with speculative decoding: 1404
- vLLM (inkling-2-mini image) + DSpark (general engine, track C): 1544
- vLLM (PR #69028 source build) + DSpark (general engine, track C): 1552
- SGLang (PR #51926 image) + DSpark (general engine, track C): 2828
- vLLM (inkling-2-mini image) without speculative decoding (track A line, for scale): 3694
- SGLang (PR #51926 image) without speculative decoding (track A line, for scale): 4389
- vLLM (PR #69028 source build) without speculative decoding (track A line, for scale): 3720

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track C v1: DSpark draft (torch graphs) + 8-row bit-exact verify pass, B=1 only

- status: **valid**, http_step_us_b1 = 2877 (B=8: 7055), at 0.57 agent-hours, category structure
- hypothesis: track C (verifier = track-A step(), checkpoint formats unchanged; draft head BF16 as stored). B=1 cohorts run draft-and-verify: a verify pass of 8 rows (bit-identical per row to step(), vcheck 0/1200 positions differ) + the DSpark draft as torch ops in a CUDA graph (2.1 ms), 3 cycles queued. Expect http_step_us_b1 ~3200 (evaluator prompts, ~2.6 tokens per cycle of ~9.3 ms), natural_b1 ~1.9 ms, b8 unchanged ~7.1 ms (no speculation at B=8); tokens_per_event ~2.7 / ~4.5.
- evaluator: valid: 2877 us/step at B=1 and 7055 at B=8 over HTTP (online (submit/poll)); in process 3182 us/step at B=1 (generate, MBU 0.217)

### #2 s0002: Track C v2: fused CUDA draft + n-gram copy drafts + shared-K/V attention, w2a 1-warp

- status: **valid**, http_step_us_b1 = 2158 (B=8: 7045), at 1.40 agent-hours, parent s0001, category structure
- hypothesis: track C (verifier = track-A step() unchanged; draft head BF16 as stored). Draft side 2150 -> ~540 us per cycle (fused CUDA kernels), n-gram copy drafts when the suffix matches >= 3 tokens (+4-7% tokens/cycle on evaluator-like prompts, ~+10% natural), 8-row verify pass ~6.5 -> ~6.0 ms (shared attention records, w2a one warp per item; bit-identical, vcheck). Expect http_step_us_b1 ~2200 (from 2877), natural_b1 ~1180 (from 1595), b8 unchanged ~7050; tokens_per_event ~2.95 / ~5.3.
- evaluator: valid: 2158 us/step at B=1 and 7045 at B=8 over HTTP (online (submit/poll)); in process 3174 us/step at B=1 (generate, MBU 0.217)

### #3 s0003: Track C v3: dynamic draft length (cumulative draft probability >= 0.4 sets K rows)

- status: **valid**, http_step_us_b1 = 1764 (B=8: 7043), at 1.68 agent-hours, parent s0002, category scheduling
- hypothesis: track C (verifier = track-A step() unchanged; draft head BF16 as stored). The verify pass now runs K = 1 + kd rows (kd = longest draft prefix whose product of the draft's own top-1 probabilities is >= 0.4; 8 rows for n-gram proposals), read on the device from vpos[1]; rows cost ~300-750 us each, rejected rows were most of them on evaluator-like prompts. Served in-process: eval-like 2729 -> 2200, natural 1367 -> 1255. Expect http_step_us_b1 ~1750 (from 2158), natural_b1 ~1120 (from 1221), b8 unchanged ~7050; tokens_per_event slightly lower (~2.9 / ~5.3).
- evaluator: valid: 1764 us/step at B=1 and 7043 at B=8 over HTTP (online (submit/poll)); in process 3168 us/step at B=1 (generate, MBU 0.218)

### #4 s0004: Track C v4: K=2 redundant mixes, w2a work-balanced ranges, n-gram in one kernel

- status: **valid**, http_step_us_b1 = 1705 (B=8: 7048), at 2.09 agent-hours, parent s0003, category scheduling
- hypothesis: track C (verifier = track-A step() unchanged; draft head BF16 as stored). Three small cycle-time cuts, all bit-identical (vcheck): verify passes of 2 rows mix in every block (no designated hop, K=2 4280 -> 4114 us); w2a ranges balanced by work when an expert has > 4 tokens (K=8 -100 us); n-gram drafts in one CUDA block (72 -> ~5 us per cycle). In-process served: eval-like 2200 -> 2127, natural 1255 -> 1204. Expect http_step_us_b1 ~1705 (from 1764), natural_b1 ~1145 (from 1194), b8 unchanged.
- evaluator: valid: 1705 us/step at B=1 and 7048 at B=8 over HTTP (online (submit/poll)); in process 3156 us/step at B=1 (generate, MBU 0.219)

### #5 s0005: Track C v5: draft early exit + faster draft GEMVs, n-gram min match 5

- status: **valid**, http_step_us_b1 = 1697 (B=8: 7044), at 2.48 agent-hours, parent s0004, category other
- hypothesis: track C (verifier = track-A step() unchanged; draft head BF16 as stored). Draft side ~556 -> ~470-520 us per cycle (Markov steps past the truncation point skipped, gate_up 4 strips/block), n-gram copies only for suffix matches >= 5 tokens (offline and served: eval-like -1.8%, natural equal). In-process served: eval-like 2127 -> 2079, natural 1204 -> 1195. Expect http_step_us_b1 ~1665 (from 1705), natural_b1 ~1135 (from 1143), b8 unchanged.
- evaluator: valid: 1697 us/step at B=1 and 7044 at B=8 over HTTP (online (submit/poll)); in process 3157 us/step at B=1 (generate, MBU 0.219)

### #6 s0006: Track A: B>=2 step kernel instance with one-warp, work-balanced w2a (bit-identical)

- status: **valid**, http_step_us_b1 = 3227 (B=8: 6356), at 2.83 agent-hours, parent s0005, category gemm
- hypothesis: track A (checkpoint formats unchanged; no speculation: engine/TRACK = A). The step kernel for B >= 2 is now its own template instance whose w2a runs one warp per item as three chains in block_strips' order, with block ranges balanced by work when an expert has > 4 tokens; the B = 1 kernel's code is unchanged. Bit-identical to s15-mini-b200-a2 s0016 at B=1 and B=8 (tools/kit/bitcheck.py, 1152 positions, batch invariance). In process (768 greedy steps): B=8 6595 -> 5961 us. Expect http_step_us_b1 unchanged ~3255, b8 7089 -> ~6450.
- evaluator: valid: 3227 us/step at B=1 and 6356 at B=8 over HTTP (online (submit/poll)); in process 3154 us/step at B=1 (generate, MBU 0.219)

### #7 s0007: Track C v7: shared-attention raw k/v in smem, draft PDL, B>=2 step kernel (s0006)

- status: **valid**, http_step_us_b1 = 1673 (B=8: 6352), at 3.37 agent-hours, parent s0006, category attention
- hypothesis: track C (verifier = the track-A step() of s0006, bit-identical to s0016; draft head BF16 as stored). Verify passes: the in-pass positions' raw k/v kept in smem for the cache/ring writes (K=8 5918 -> 5772 us, bit-identical); draft kernels chained with programmatic dependent launch (-25 us/cycle); B=8 cohorts use s0006's B>=2 step kernel. In-process served: eval-like 2079 -> 2050, natural 1195 -> 1173, B=8 776 -> 720 us/token/stream. Expect http_step_us_b1 ~1673 (from 1697), natural_b1 ~1108 (from 1129), b8 ~6356 (from 7044), natural_b8 ~6580.
- evaluator: valid: 1673 us/step at B=1 and 6352 at B=8 over HTTP (online (submit/poll)); in process 3154 us/step at B=1 (generate, MBU 0.219)

### #8 s0008: Track C v8: FP8 draft copies, n-gram inside the draft cycle, designated final rows

- status: **valid**, http_step_us_b1 = 1619 (B=8: 6351), at 3.90 agent-hours, parent s0007, category other
- hypothesis: track C (verifier = s0006 track-A step(); draft head: FP8 E4M3 per-row-scaled copies of the base unembedding and fold used by the DRAFT only, everything else BF16 as stored; the target is unchanged). Draft cycle: FP8 copies (-56..68 us/cycle, tokens/cycle -0.3%), the n-gram rule runs inside the draft and skips the DSpark block/Markov kernels when used (121 vs ~400 us in those cycles; traces identical to draft + mk_ngram_k), final-norm rows by designated blocks (K>=3, bit-identical). In-process served: eval-like 2050 -> 2004, natural 1173 -> 1128. Expect http_step_us_b1 ~1636 (from 1673), natural_b1 ~1064 (from 1106), b8 unchanged ~6350.
- evaluator: valid: 1619 us/step at B=1 and 6351 at B=8 over HTTP (online (submit/poll)); in process 3161 us/step at B=1 (generate, MBU 0.218)

### #9 s0009: Track A: current position's raw k/v for the conv ring from smem (no re-poll)

- status: **valid**, http_step_us_b1 = 3206 (B=8: 6359), at 4.13 agent-hours, parent s0006, category attention
- hypothesis: track A (checkpoint formats unchanged; no speculation: engine/TRACK = A). In the attention item of the current position, the conv-history ring writes take the raw k/v the k/v warps already loaded (smem) instead of polling the tagged qkv output again. Bit-identical to s15 s0016 (bitcheck 1152 positions, B=1 and B=8). In process: B=1 3270 -> 3242 us, B=8 -0.2%. Expect http_step_us_b1 3227 -> ~3200, b8 ~6340.
- evaluator: valid: 3206 us/step at B=1 and 6359 at B=8 over HTTP (online (submit/poll)); in process 3148 us/step at B=1 (generate, MBU 0.219)

### #10 s0010: Track C v10: gate D mixes loop messages first in verify passes

- status: **valid**, http_step_us_b1 = 1617 (B=8: 6339), at 4.39 agent-hours, parent s0009, category scheduling
- hypothesis: track C (verifier = s0009 track-A step(); draft as s0008). In verify passes the gate D items mix the loop messages (ready since the previous pass) before polling the current partials, taking that mix off the critical path; bit-identical (vcheck). Verify K=2 -13, K=3 -40, K=8 ~-18 us. Expect http_step_us_b1 ~1612 (from 1619), natural_b1 ~1063 (from 1067), b8 ~6350.
- evaluator: valid: 1617 us/step at B=1 and 6339 at B=8 over HTTP (online (submit/poll)); in process 3143 us/step at B=1 (generate, MBU 0.219)

### #11 s0011: Track C v11: verify passes poll only the qkv input streams a block uses

- status: **valid**, http_step_us_b1 = 1598 (B=8: 6343), at 4.84 agent-hours, parent s0010, category sync
- hypothesis: track C (verifier = s0009 track-A step(); draft as s0008). After the designated blocks publish the normalized q/k/v input rows (gated applications and global layers, K >= 3), each block polls only the streams its own qkv strips read instead of all three; bit-identical (vcheck 1100). Verify K=3 -60, K=4 -95, K=8 -80 us. Expect http_step_us_b1 ~1595 (from 1617), natural_b1 ~1050 (from 1064), b8 ~6340.
- evaluator: valid: 1598 us/step at B=1 and 6343 at B=8 over HTTP (online (submit/poll)); in process 3148 us/step at B=1 (step, MBU 0.220)

### #12 s0012: Track C v12: hybrid mixes for 2-row verify passes

- status: **valid**, http_step_us_b1 = 1588 (B=8: 6366), at 5.00 agent-hours, parent s0011, category sync
- hypothesis: track C (verifier = s0009 track-A step(); draft as s0008). In 2-row verify passes every block mixes row 0 itself while the last block mixes, normalizes and publishes row 1 (the others poll it), at the attention-input, gate-norm, MLP-input and MLP-gate sites: one mix + one hop instead of two serial mixes; bit-identical (vcheck). K=2 4110 -> 4028 us, K>=3 unchanged. Expect http_step_us_b1 ~1585 (from 1598; K=2 is the commonest pass on evaluator prompts), natural_b1 ~1047.
- evaluator: valid: 1588 us/step at B=1 and 6366 at B=8 over HTTP (online (submit/poll)); in process 3146 us/step at B=1 (generate, MBU 0.219)

### #13 s0013: Track C v13: designated mix rows on the last blocks for 3-4 row passes

- status: **valid**, http_step_us_b1 = 1587 (B=8: 6376), at 5.31 agent-hours, parent s0012, category scheduling
- hypothesis: track C (verifier = s0009 track-A step(); draft as s0008). In verify passes of 3-4 rows the designated mix / norm / final rows run on the last blocks (no w2b / o_proj strips, free earlier) instead of blocks 0..K-1; bit-identical (vcheck). K=3 4252 -> 4224 us, other K unchanged. Expect http_step_us_b1 ~1584 (from 1588; within noise), natural unchanged ~1050.
- evaluator: valid: 1587 us/step at B=1 and 6376 at B=8 over HTTP (online (submit/poll)); in process 3137 us/step at B=1 (generate, MBU 0.220)

### #14 s0014: Track C final: s0013 + off-by-default switches (INT8 certified argmax off)

- status: **valid**, http_step_us_b1 = 1585 (B=8: 6376), at 6.11 agent-hours, parent s0013, category cleanup
- hypothesis: track C (verifier = s0009 track-A step(); draft as s0008). Same function and default code paths as s0013; adds compile/env switches that are off by default (certified INT8 argmax MK_UN8=0, HYB_GN_MAX 2, W2_1W_MIN 0) and a preprocessor-guard fix. Confirms the final committed source. Expect http_step_us_b1 ~1587 (s0013, within noise), natural ~1050.
- evaluator: valid: 1585 us/step at B=1 and 6376 at B=8 over HTTP (online (submit/poll)); in process 3139 us/step at B=1 (generate, MBU 0.220)

### #15 s0015: Track A: bf16 GEMV with 3 units of loads in flight (BF_UNR 3)

- status: **valid**, http_step_us_b1 = 3169 (B=8: 6353), at 6.47 agent-hours, parent s0009, category gemm
- hypothesis: track A (checkpoint formats unchanged; no speculation: engine/TRACK = A). The bf16 GEMVs (qkv, router, gate D, unembedding) keep 3 instead of 2 64-k units of weight loads in flight per warp; same mma order, bit-identical to s15 s0016 (bitcheck 1152 positions, B=1 and B=8). In process B=1 3244 -> 3215, B=8 -0.5..-2%. Expect http_step_us_b1 3206 -> ~3180, b8 ~6330.
- evaluator: valid: 3169 us/step at B=1 and 6353 at B=8 over HTTP (online (submit/poll)); in process 3115 us/step at B=1 (generate, MBU 0.222)

### #16 s0016: Track C v15: bf16 GEMV 3 deep (s0015) + verify-kernel row polls 2 deep

- status: **valid**, http_step_us_b1 = 1559 (B=8: 6353), at 6.73 agent-hours, parent s0015, category sync
- hypothesis: track C (verifier = s0015 track-A step(), bit-identical to s0016; draft as s0008). The bf16 GEMVs keep 3 units of weight loads in flight (BF_UNR 3, from s0015) and the verify kernel's tagged row polls run 2 instead of 8 deep (all blocks poll the same published rows; fewer outstanding polls contend less); bit-identical (vcheck). Verify K=2 4028 -> 3945, K=8 5666 -> 5488 us. Expect http_step_us_b1 ~1550 (from 1585), natural_b1 ~1020 (from 1048), b8 ~6330.
- evaluator: valid: 1559 us/step at B=1 and 6353 at B=8 over HTTP (online (submit/poll)); in process 3120 us/step at B=1 (generate, MBU 0.221)

### #17 s0017: Track C v16: shallower merge / w2a staging polls in verify passes

- status: **valid**, http_step_us_b1 = 1541 (B=8: 6350), at 6.94 agent-hours, parent s0016, category sync
- hypothesis: track C (verifier = s0015 track-A step(); draft as s0008). In the verify kernel the shared-attention merge items poll their records 1 deep (was 8) and w2a stages its inputs 2 deep (was 8): fewer outstanding polls on data that many blocks poll at once; bit-identical (vcheck). Verify K=2 3945 -> 3855, K=3 4105 -> 4027, K=8 5488 -> 5440 us. Expect http_step_us_b1 ~1530 (from 1559), natural_b1 ~1023 (from 1031).
- evaluator: valid: 1541 us/step at B=1 and 6350 at B=8 over HTTP (online (submit/poll)); in process 3121 us/step at B=1 (generate, MBU 0.221)

### #18 s0018: Track A: shallower tagged polls (row polls, w2a staging 2 deep)

- status: **valid**, http_step_us_b1 = 3141 (B=8: 6300), at 7.39 agent-hours, parent s0015, category sync
- hypothesis: track A (checkpoint formats unchanged; no speculation: engine/TRACK = A). The step kernels poll published rows (B >= 2) and stage w2a inputs 2 instead of 8 loads deep per thread: fewer outstanding polls on data many blocks poll at once. Bit-identical to s15 s0016 (bitcheck 1152 positions, B=1 and B=8). In process B=1 3220 -> 3172, B=8 -20..-50 us. Expect http_step_us_b1 3169 -> ~3125, b8 ~6300.
- evaluator: valid: 3141 us/step at B=1 and 6300 at B=8 over HTTP (online (submit/poll)); in process 3076 us/step at B=1 (generate, MBU 0.224)

### #19 s0019: Track C v19: s0018 step kernel (shallower polls) + sequential router-logit polls

- status: **valid**, http_step_us_b1 = 1540 (B=8: 6310), at 7.47 agent-hours, parent s0018, category sync
- hypothesis: track C (verifier = s0018 track-A step(), bit-identical to s0016; draft as s0008). B=8 cohorts and the step path get s0018's shallower tagged polls (2 deep); the verify kernel polls the router logits one at a time (-3 us ctx 600, -12 us ctx 1100); bit-identical (vcheck). Expect http_step_us_b1 ~1538 (from 1541, within noise), natural ~1024, b8 ~6300 (from 6350).
- evaluator: valid: 1540 us/step at B=1 and 6310 at B=8 over HTTP (online (submit/poll)); in process 3078 us/step at B=1 (generate, MBU 0.224)

### #20 s0020: Track C v20: no L2 prefetch of qkv/router weights in verify passes

- status: **valid**, http_step_us_b1 = 1531 (B=8: 6302), at 7.67 agent-hours, parent s0019, category memory
- hypothesis: track C (verifier = s0018 track-A step(); draft as s0008). With the shallower tagged polls the verify kernel's L2 prefetch of the next qkv / router strips during the mixes now costs more than it saves; dropped (PF_MIN 9). Data and arithmetic unchanged. Verify K=3 4027 -> 3951, K=4 4225 -> 4176, K=8 -8 us. Expect http_step_us_b1 ~1525 (from 1540), natural ~1018.
- evaluator: valid: 1531 us/step at B=1 and 6302 at B=8 over HTTP (online (submit/poll)); in process 3072 us/step at B=1 (generate, MBU 0.224)

### #21 s0021: Track C v21: 3-4 row verify passes poll published rows 1 deep

- status: **valid**, http_step_us_b1 = 1531 (B=8: 6314), at 8.21 agent-hours, parent s0020, category sync
- hypothesis: track C (verifier = s0018 track-A step(); draft as s0008). Verify passes of 3-4 rows poll the published rows 1 deep (others 2); bit-identical (vcheck). Verify K=3 3955 -> 3932, K=4 4180 -> 4156 us, K=2/8 unchanged. Expect http_step_us_b1 ~1527 (from 1531, near noise), natural ~1019.
- evaluator: valid: 1531 us/step at B=1 and 6314 at B=8 over HTTP (online (submit/poll)); in process 3082 us/step at B=1 (generate, MBU 0.224)

### #22 s0022: Track C final: s0021 source + off-by-default switches (W13_CONTIG, FG_ROWS)

- status: **valid**, http_step_us_b1 = 1528 (B=8: 6302), at 9.39 agent-hours, parent s0021, category cleanup
- hypothesis: track C (verifier = s0018 track-A step(); draft as s0008). Same default code paths as s0021 (new switches W13_CONTIG and FG_ROWS are off: both measured slower); confirms the final committed source. Expect http_step_us_b1 ~1531 (noise), natural ~1020, b8 ~6310.
- evaluator: valid: 1528 us/step at B=1 and 6302 at B=8 over HTTP (online (submit/poll)); in process 3077 us/step at B=1 (generate, MBU 0.224)

