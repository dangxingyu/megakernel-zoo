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
tg = Target(MODEL_DIR, 4096)
K = int(os.environ.get("K", "256")); POS = int(os.environ.get("POS", "768"))
toks = list(range(1000, 1000 + K))
def run():
    tg.pos = tg.base = POS
    t, _ = tg.prefill(toks, K); mx.eval(t, *tg.state())
for _ in range(2): run()
ts = []
for _ in range(4):
    t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
m = float(np.median(ts))
print(f"{mt.PREFILL} no={os.environ.get('MK_WHATIF_NO','')} K={K} pos={POS}: {m*1e3:.1f} ms = {m/K*1e3:.3f} ms/token")
