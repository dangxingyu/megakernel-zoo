import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import mk_kernels as kn
w = mx.load((MODEL_DIR + "/model-00001-of-00001.safetensors"))
e = "model.llm.layers.0.mlp.experts."
w13, s13, w2, s2 = (w[e + n] for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"))
g13 = w[e + "w13_weight_amax"] / w[e + "w13_weight_outer_scale_denominator"]
print("g13 range", g13.min().item(), g13.max().item())
K = 256
mx.random.seed(0)
x = (mx.random.normal((K, 2304)) * 3).astype(mx.float16)
idx = mx.random.randint(0, 128, (K, 8)).astype(mx.int32)
flat = idx.reshape(-1); order = mx.argsort(flat); es = flat[order]
xs = x[order // 8][:, None, :]
ref = mx.gather_qmm(xs.astype(mx.float32), w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=True)[:, 0] * g13[es][:, None]
print("raw max (f32 path, before g)", (mx.abs(ref) / g13[es][:, None]).max().item(), "scaled max", mx.abs(ref).max().item())
def bench(f):
    for _ in range(2): mx.eval(f())
    t = time.perf_counter()
    for _ in range(10): mx.eval(f())
    return (time.perf_counter() - t) / 10 * 1e3
try:
    f = lambda: mx.gather_qmm(xs, w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=16, bits=4, mode="nvfp4", global_scale=g13, sorted_indices=True)
    o = f()[:, 0].astype(mx.float32)
    print("global_scale: rel", (mx.linalg.norm(o - ref) / mx.linalg.norm(ref)).item(), "ms", bench(f))
except Exception as ex: print("global_scale err", ex)
f32 = lambda: mx.gather_qmm(xs.astype(mx.float32), w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=True)
print("f32 ms", bench(f32))
f16 = lambda: mx.gather_qmm(xs, w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=True)
print("f16 ms", bench(f16))
bf = lambda: mx.gather_qmm(xs.astype(mx.bfloat16), w13.view(mx.uint32), s13, None, rhs_indices=es, transpose=True, group_size=16, bits=4, mode="nvfp4", sorted_indices=True)
o = bf()[:, 0].astype(mx.float32) * g13[es][:, None]
print("bf16 rel", (mx.linalg.norm(o - ref) / mx.linalg.norm(ref)).item(), "ms", bench(bf))
