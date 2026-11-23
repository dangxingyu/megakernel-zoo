import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
import mlx.core as mx
import mk_target as mt, mk_kernels as kn
from mk_target import Target
tg = Target(MODEL_DIR, 4096)
bad = []
def chk(name, f):
    def g(*a, **k):
        y = f(*a, **k)
        yy = y.view(mx.float16).astype(mx.float32) if y.dtype == mx.uint16 else y
        m = mx.abs(yy).max().item(); n = mx.isnan(yy).any().item()
        if n or m > 6e4 or not np.isfinite(m): bad.append((name, m, n))
        return y
    return g
od = mt.dense
mt.dense = lambda L, nm, x: chk(f"dense {L['i']} {nm}", od)(L, nm, x)
kn.experts_mm = chk("experts", kn.experts_mm)
Target._attn_wide = chk("attn", Target._attn_wide)
# also check h copies
for L in tg.layers:
    for nm in ("qgrs", "wk", "wv", "wo"):
        w = L[nm + "h"]; m = mx.abs(w.astype(mx.float32)).max().item()
        if not np.isfinite(m): print("inf in half copy", L["i"], nm)
rng = np.random.default_rng(0)
toks = [int(t) for t in rng.integers(1000, 150000, 256)]
tg.reset(); t, _ = tg.prefill(toks, 256); mx.eval(t)
print(bad[:20], len(bad))
