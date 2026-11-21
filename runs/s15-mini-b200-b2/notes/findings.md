# Findings (card: B200 in GB200 node, 152 SMs, 129 MB L2)

## Card microbenchmarks (bench/)
- Grid barrier, 152 blocks x 512 threads (bench/bar.cu, e0014): reset-by-last-arriver + generation flag 2.24 us;
  monotonic counter with red.release.gpu + ld.acquire spin 1.17 us; fence+atomicAdd+relaxed poll 1.29 us.
- Load latency, one thread pointer chase (bench/lat.cu, lat2.cu, e0023/e0037): L2 hit 310 cycles (0.16 us);
  line last written by other SMs 500 cycles (0.25 us); first touch 860 cycles. Dependent shfl+add 43 cycles.
- 40 MB contiguous read by 152x512 threads, 4-8 x 16 B loads in flight per thread (bench/pf.cu, e0040):
  cold 9.2 us (4.6 TB/s); after cp.async.bulk.prefetch.L2 of the same bytes 6.5 us (6.2 TB/s); second read of
  L2-resident data 7.5 us (5.6 TB/s). L2 staging gains ~1.4x on a 40 MB GEMV, not more.
- FP4 decode cost (bench/dec.cu, e0032): cvt.rn.bf16x2.e2m1x2 (F2FP.BF16.E2M1, one instruction per 2 values) +
  bf16x2 mul by the E4M3 block scale (cvt.rn.bf16x2.e4m3x2) + mma: 1.8 ns per SM per warp-kstep; the whole
  B=1 w13 decode of a layer (41472 units) is ~2 us GPU-wide: MMA-issue bound, not a limit.
- **Instruction cache** (bench/icache_gen.py, e0043): straight-line code up to 8000 instructions (128 KB) runs
  at 2.0-2.4 cycles/instr (8 warps); 12000+ instructions: 6.0 cycles/instr (fetched from L2). A persistent
  kernel larger than ~8K instructions refetches every phase's code: several us per phase.
- Local memory: 960 B stack x 512 threads = 480 KB per SM > L1: every call-frame fill / spill is an L2 round trip.

## Engine v1 (s0003, valid): 6578 us B=1 / 12658 B=8 over HTTP; in process 6384 us.
Phase spans at B=1 (us, mkprof e0013): w13 22, att 19, w2 15, qkv 15, oproj 14, mixa 12, mixm 11, router 8;
loop passes add gz/gx 8-19 each. Single-block phases with 0.2 us of arithmetic took 5-10 us.

## Checkpoint facts (measured)
- BF16 census (tools/kit/weights_census.py over all 590 BF16 matrices, 2.244e9 weights, e0102): no zeros, no
  denormals, no zero rows/cols, no identical tensors; exponent entropy 2.59 bits; per-row 16-binade window (12-bit
  code sign + 4-bit exponent offset + 7 mantissa): 7.64e-5 escapes (12.0024 bits/weight); 8-binade (11-bit) 1.33%.
- Routing (harness inkling2_ref fork precision, 4 own texts x 384 tokens, tests/routing_study.py, e0102): experts of a
  loop layer shared between pass 1 and pass 2: 4.3-5.4 of 8 (layer 17: 3.0); pass 2-3: 3.9-5.1; pass 1-3: 3.1-4.4.
  Consecutive tokens of one application share 3.27 of 8 (1.7-5.8). Union over 4 sequences at one position: 25.5.
  Distinct experts used per application over 1536 tokens: min 76, mean 122 of 128.
- Per step the BF16 weights are 3.66 GB of 5.54 GB (attention/router/gates of 48 applications + unembedding 0.92 GB),
  experts 1.84 GB.
