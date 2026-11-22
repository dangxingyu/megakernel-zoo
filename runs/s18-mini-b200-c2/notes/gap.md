# Gap accounting, B=8 track C (run s18) — updated after s0004 build

Headline b8 = cycle time / tokens per sequence per cycle. s0003: evaluator prompts 7181 us/token at 2.66 tokens per
event (cycle ~19 ms); natural 4476 us at 4.64 (~20.8 ms per cycle incl. ragged ends). vLLM + DSpark: 2711 / 1851.

## Cycle (in process, B=8 natural prompts, N=512, MK_CYC_TIMING=1, e0063)
  verify (gather + wide pass + scatter) 14.9 ms (93%) + 8 drafts 1.06 ms (on 8 side streams; 1.68 ms serially).
  Rows per cycle ~40-50 (dynamic draft length, theta 0.4; caps 3/4/5 worse than 8, theta 0.2/0.6 within 3%: e0058, e0064).

## Wide pass time vs rows (tests/w/wtime.py, ctx 600; wab.py same-job A/B)
  8 rows 8.1-9.2 ms, 32 rows 13.6, 64 rows 18.9-20.7 ms. Megakernel B=8 step (8 rows): 6.5 ms. Bound at 64 rows:
  MoE bytes ~550 MB/app ~ 3.6 ms per pass + dense 57 MB/app ~0.4 ms + unembedding 0.13 ms -> ~4.2 ms at 7.3 TB/s.

## Per application (nsys, R=64 / R=8, before PDL: kernel durations add up to the pass; after PDL they overlap)
  w13 147 / 52, w2 64 / 25 us: ~2.6-3.4 TB/s. Diagnostic builds (e0057): memory only (no dequant/MMA) 140 / 53,
    compute only (no loads) 149 / 50 -> both limits are near the full time and overlap poorly. Stream microbench
    (bench/stream.cu): 7.2 TB/s with 16 warps x 3 x 2 KB per SM, so the 16-B cp.async ring + dequant per 576-B unit
    (~76 instructions per warp) is the loss, not HBM.
  attention 73-91 / 35-41 us: per item ~10-12 us latency chain (q/r prep 3 us, bias 2.6 us behind the K/V copy) and a
    slow last-arriver merge; fixed by staging (s0003) and multi-chunk items with double-buffered K/V (s0004: -4%).
  dense GEMMs qkv 36-46 / 18-21, gate D 33 / 10-13, o_proj 23-31 / 8-11, router 17-21 / 6-8 us: latency-bound
    (B staging + weight prefetch depth); 128-register cap with depth-4 prefetch still spills 0.3-1 KB in qkv / o_proj.
  row jobs (mixes, norms) 12-15 us each, small kernels (kv, w2sum, w2conv) 3-7 us each: kernel ramp/tail; PDL (s0003)
    took 10.7 -> 9.2 ms at 8 rows.
  unembedding 483-571 us at 64 rows (0.93 GB: 130 us at 7.3 TB/s).

## Named losses at 64 rows (per pass, approximate)
1. MoE ~10 ms vs ~3.6 ms of bytes (dequant issue + 16-B async copies; one unit per 576 B).
2. Fixed per-kernel cost: ~15 kernels x 48 applications, 3-5 us each -> ~2.5-3.5 ms.
3. Attention ~2-3 ms (latency per item; 24 query rows x 128 keys per chunk is tiny work).
4. Dense GEMMs ~4 ms vs ~0.4 ms of bytes (latency-bound staging, spills).

## Update (after s0006): instruction throughput on this card (bench/cvt.cu, thread-ops per clock per SM)
cvt.rn.f16x2.e2m1x2 15.8, cvt.rn.f16x2.e4m3x2 21.1, mul.rn.f16x2 62.6, prmt / lop3 31.6 (incl. one extra op each).
NVFP4 unit (1024 weights) = 512 e2m1x2 + 128 e4m3x2 conversions per warp -> ~37 cycles per unit per SM: w13 at 64 rows
(4068 units per SM) >= ~73 us, w2 ~37 us; measured ~85 / ~45 (item loops): the MoE is within ~20% of its conversion
floor. An integer (PRMT lookup) decode costs ~11 ALU ops per f16x2 pair: no better. Fewer decodes need fewer
(expert, chunk) items (larger token groups per expert) or a different MMA format (not exact for BF16 inputs).
