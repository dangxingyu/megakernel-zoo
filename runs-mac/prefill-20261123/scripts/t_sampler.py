import time, numpy as np, mlx.core as mx
V = 200058
U32 = mx.uint32
def u32(v): return mx.array(v, dtype=U32)
def noise(seed, q, n=V):
    """Gumbel noise for (seed, position q): a counter-based hash of the token index, 24 uniform bits each."""
    key = (seed ^ (q.astype(U32) * u32(0x85EBCA6B))) * u32(0xC2B2AE35) + u32(0x27D4EB2F)
    x = mx.arange(n, dtype=U32) * u32(0x9E3779B1) + key
    x = x ^ (x >> 16); x = x * u32(0x85EBCA6B); x = x ^ (x >> 13); x = x * u32(0xC2B2AE35); x = x ^ (x >> 16)
    u = ((x >> 8).astype(mx.float32) + 0.5) * (1.0 / 16777216.0)
    return -mx.log(-mx.log(u))
def sample(lg, sp, seed, q, filt):
    z = lg * sp[0]
    score = z + noise(seed, q)[None]
    if filt:
        zs = -mx.sort(-z, axis=-1)
        ps = mx.softmax(zs, axis=-1)
        before = mx.cumsum(ps, axis=-1) - ps
        n_p = (before < sp[1]).sum(axis=-1)
        k = sp[2].astype(mx.int32)
        n = mx.where(k > 0, mx.minimum(k, n_p), n_p)
        thr = mx.take_along_axis(zs, (n - 1).reshape(1, 1), axis=-1)
        score = mx.where(z >= thr, score, -mx.inf)
    tok = mx.argmax(score, axis=-1).astype(mx.int32)
    p = mx.exp(mx.take_along_axis(z, tok[:, None], axis=-1)[:, 0] - mx.logsumexp(z, axis=-1))
    return tok, p
mx.random.seed(0)
lg = mx.random.normal((1, V)) * 3.0
lg[0, 5] = 14.0; lg[0, 77] = 13.0
seed = u32([1234])
for filt in (False, True):
    f = mx.compile(lambda lg, sp, seed, q: sample(lg, sp, seed, q, filt))
    sp = mx.array([1 / 0.8, 0.95, 0.0])
    for _ in range(3): mx.eval(f(lg, sp, seed, mx.array([7], dtype=mx.int32)))
    t = time.perf_counter()
    for i in range(50): mx.eval(f(lg, sp, seed, mx.array([i], dtype=mx.int32)))
    print("filt", filt, f"{(time.perf_counter() - t) / 50 * 1e3:.2f} ms per row")
# distribution check against the exact probabilities (no filter, then top-p)
z = np.array(lg[0]) / 0.8; pr = np.exp(z - z.max()); pr /= pr.sum()
f = mx.compile(lambda lg, sp, seed, q: sample(lg, sp, seed, q, True))
for name, sp, expect in (("temp 0.8", [1 / 0.8, 1.0, 0.0], None), ("top_p 0.9", [1 / 0.8, 0.9, 0.0], 0.9), ("top_k 2", [1 / 0.8, 1.0, 2.0], None)):
    toks = [int(f(lg, mx.array(sp), seed, mx.array([i], dtype=mx.int32))[0].item()) for i in range(4000)]
    c = np.bincount(toks, minlength=V)
    order = np.argsort(-pr)[:6]
    print(name, "top tokens: expected", np.round(pr[order], 3), "observed", np.round(c[order] / 4000, 3), "distinct", (c > 0).sum())
# determinism: same (seed, q) -> same token; u never 0 or 1
a = [int(f(lg, mx.array([1.25, 0.95, 0.0]), seed, mx.array([i], dtype=mx.int32))[0].item()) for i in range(50)]
b = [int(f(lg, mx.array([1.25, 0.95, 0.0]), seed, mx.array([i], dtype=mx.int32))[0].item()) for i in range(50)]
g = noise(seed, mx.array([3], dtype=mx.int32)); print("deterministic", a == b, "noise finite", bool(mx.isfinite(g).all().item()), "mean %.3f (0.577)" % g.mean().item())
