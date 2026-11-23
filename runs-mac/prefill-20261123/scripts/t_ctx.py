"""Long context: a wide exact prefill against 8-row passes at T = 32,768 (same bits?), and the time of passes at long
positions (prefill chunk, decode step) in both modes."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
import mk_target as mt
from mk_target import Target
T = int(os.environ.get("CTX", "32768"))
t = time.time(); tg = Target(MODEL_DIR, T); print(f"load {time.time()-t:.1f} s, T={tg.T}, tau={tg.tau}, snap={tg.snap}", flush=True)
rng = np.random.default_rng(0)
ids = [int(x) for x in rng.integers(1000, 150000, 40000)]
def st(p0, p1):
    tg.flush(); return np.stack([np.array(k[0, p0:p1]) for k in tg.kc]), [np.array(c) for c in tg.cw]
if os.environ.get("EQ", "1") == "1":
    # same bits at short positions with the long-context kernels
    N = 300
    tg.reset(); i = 0
    while i < N:
        K = min(8, N - i); lg = tg.forward(mx.array(ids[i:i+K], dtype=mx.int32), tok_only=True); tg.commit(K); mx.eval(lg, *tg.state()); i += K
    a = st(0, N)
    tg.reset(); tok, _ = tg.prefill(ids[:256], 256); mx.eval(tok, *tg.state()); tok, _ = tg.prefill(ids[256:300], 64); mx.eval(tok, *tg.state())
    b = st(0, N)
    print(f"{mt.PREFILL} wide vs 8-row at T={T}: kc", "identical" if (a[0] == b[0]).all() else f"differs ({np.mean(a[0] != b[0]):.3f})", flush=True)
for P in [int(x) for x in os.environ.get("POSS", "0,4096,8192,16384,30000").split(",")]:
    if P + 300 > T: continue
    tg.pos = tg.base = P
    ts = []
    for r in range(3):
        tg.pos = tg.base = P; t = time.perf_counter(); tok, _ = tg.prefill(ids[:256], 256); mx.eval(tok, *tg.state()); ts.append(time.perf_counter() - t)
    w = min(ts[1:])
    ds = []
    for r in range(12):
        t = time.perf_counter(); lg = tg.forward(mx.array(ids[r:r+1], dtype=mx.int32), tok_only=True); tg.commit(1); mx.eval(lg, *tg.state()); ds.append(time.perf_counter() - t)
    d8 = []
    for r in range(6):
        t = time.perf_counter(); lg = tg.forward(mx.array(ids[:8], dtype=mx.int32), tok_only=True); tg.commit(8); mx.eval(lg, *tg.state()); d8.append(time.perf_counter() - t)
    print(f"position {P}: 256-row {mt.PREFILL} pass {w*1e3:.0f} ms ({w/256*1e3:.2f} ms/token); 1-row step {np.median(ds)*1e3:.1f} ms; 8-row pass {np.median(d8)*1e3:.1f} ms; nan {bool(mx.isnan(tg.kr.view(mx.float16).astype(mx.float32)).any().item())}", flush=True)
print("peak memory GB", round(mx.get_peak_memory() / 1e9, 1))
