#!/usr/bin/env python
"""Memory bandwidth of this Apple-silicon GPU, measured with MLX: the figure eval/modelinfo.py CARDS uses for MBU on a
Mac (the counterpart of the H100 read-rate microbenchmark).

    .venv-mac/bin/python scripts/mac_membw.py [--gb 4] [--iters 20]

Three kernels over a buffer much larger than the GPU's caches, each timed with time.perf_counter around `iters`
launches (each evaluated before the next) after a full synchronize, best of five rounds:
  read   a Metal kernel (mx.fast.metal_kernel) that reads every byte once as uint4 vectors and writes one word per
         threadgroup: what a decode step does with its weights
  sum    mx.sum over the buffer viewed as float32 and as bfloat16 (MLX's own reductions; read only)
  copy   y = x + 0 (read + write; bytes counted as read + written)
Prints one MK_MEMBW JSON line. `read_gb_s` (the best read-only kernel) is what MBU uses.
"""
import argparse
import json
import time

import mlx.core as mx

READ_SRC = """
    uint tid = thread_position_in_grid.x;
    uint n = threads_per_grid.x;
    const device uint4* xv = reinterpret_cast<const device uint4*>(x);
    uint c = count[0];
    uint4 a0 = uint4(0), a1 = uint4(0), a2 = uint4(0), a3 = uint4(0);
    uint i = tid;
    for (; i + 3 * n < c; i += 4 * n) {  // four independent loads in flight per thread
        a0 ^= xv[i]; a1 ^= xv[i + n]; a2 ^= xv[i + 2 * n]; a3 ^= xv[i + 3 * n];
    }
    for (; i < c; i += n) a0 ^= xv[i];
    uint4 acc = a0 ^ a1 ^ a2 ^ a3;
    // one write per thread, cheap next to the reads; keeps the loads alive
    out[tid] = acc.x ^ acc.y ^ acc.z ^ acc.w;
"""


def timed(fn, iters: int, rounds: int = 5) -> float:
    mx.eval(fn())
    mx.synchronize()
    best = float("inf")
    for _ in range(rounds):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):  # one evaluation per launch: a copy's output is as large as its input
            mx.eval(fn())
        mx.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    nbytes = int(args.gb * 1e9) // 64 * 64
    x8 = mx.arange(nbytes // 4, dtype=mx.uint32)
    mx.eval(x8)
    vec = x8.reshape(-1, 4)  # uint4 elements
    n_vec = vec.shape[0]
    kernel = mx.fast.metal_kernel(name="mk_membw_read", input_names=["x", "count"], output_names=["out"], source=READ_SRC)
    count = mx.array([n_vec], dtype=mx.uint32)
    info = mx.device_info()
    results = {}
    for threads in (1 << 16, 1 << 17, 1 << 18):
        def read():
            return kernel(inputs=[vec, count], grid=(threads, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(threads,)], output_dtypes=[mx.uint32])[0]
        results[f"read_t{threads}"] = nbytes / timed(read, args.iters) / 1e9
    results["sum_f32"] = nbytes / timed(lambda: mx.sum(x8.view(mx.float32)), args.iters) / 1e9
    results["sum_bf16"] = nbytes / timed(lambda: mx.sum(x8.view(mx.bfloat16)), args.iters) / 1e9
    results["copy_rw"] = 2 * nbytes / timed(lambda: x8 + 0, args.iters) / 1e9
    read_best = max(v for k, v in results.items() if k.startswith(("read", "sum")))
    out = {"device": info.get("device_name"), "architecture": info.get("architecture"), "memory_gb": info.get("memory_size", 0) / 1e9,
           "buffer_gb": nbytes / 1e9, "gb_s": {k: round(v, 1) for k, v in results.items()}, "read_gb_s": round(read_best, 1),
           "mlx": mx.__version__}
    print("MK_MEMBW " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
