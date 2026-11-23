"""Wide prefill vs passes of 8 rows: same K/V, conv windows, next token? and timing."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import mk_target as mt
from mk_target import Target
MD = MODEL_DIR
N = int(os.environ.get("NP", "1630")); CH = int(os.environ.get("CH", "256"))
tg = Target(MD, 4096)
rng = np.random.default_rng(0)
prompt = [int(t) for t in rng.integers(1000, 150000, N)]

def old():
    tg.reset(); i = 0; last = None
    while i < N:
        K = min(8, N - i)
        lg = tg.forward(mx.array(prompt[i:i+K], dtype=mx.int32), tok_only=True); tg.commit(K)
        last = lg[K-1:K]; mx.async_eval(last, *tg.state()); i += K
    mx.eval(last); return last

def buckets(n):
    K = 32
    while K < n: K *= 2
    return K

def wide():
    tg.reset(); i = 0; last = None
    while i < N:
        n = min(CH, N - i); K = buckets(n)
        last, _ = tg.prefill(prompt[i:i+n], K); mx.async_eval(last, *tg.state()); i += n
    mx.eval(last); return last

def snap():
    tg.flush(); p = tg.pos
    return (np.stack([np.array(k[0, :p]) for k in tg.kc]), np.stack([np.array(k[0, :p]) for k in tg.vc]), [np.array(c) for c in tg.cw])

mode = mt.PREFILL
t = time.perf_counter(); a = old(); t_old = time.perf_counter() - t
t = time.perf_counter(); a = old(); t_old = time.perf_counter() - t
sa = snap()
for r in range(3):
    t = time.perf_counter(); b = wide(); t_w = time.perf_counter() - t
    print(f"run {r}: wide ({mode}, chunk {CH}) {t_w:.3f} s", flush=True)
sb = snap()
print(f"N={N}: 8-row passes {t_old:.3f} s, wide {t_w:.3f} s ({N/t_w:.0f} tok/s)")
print("next token", a.tolist(), b.tolist())
for nm, x, y in (("kc", sa[0], sb[0]), ("vc", sa[1], sb[1])):
    xf = x.view(np.float16).astype(np.float32); yf = y.view(np.float16).astype(np.float32)
    print(nm, "identical" if (x == y).all() else f"differs: {np.mean(x != y):.4f} of entries, rel {np.linalg.norm(xf - yf) / np.linalg.norm(xf):.2e}")
print("conv windows", ["identical" if (u == v).all() else f"rel {np.linalg.norm(u - v) / np.linalg.norm(u):.2e}" for u, v in zip(sa[2], sb[2])])
