import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time
sys.path.insert(0, os.getcwd())
import mlx.core as mx
from mk_target import Target
tg = Target(MODEL_DIR, 4096)
for K in (256, 256, 224, 224, 96, 96, 512, 512):
    tg.reset(); t = time.perf_counter()
    tok, _ = tg.prefill(list(range(1000, 1000 + K)), K); mx.eval(tok, *tg.state())
    print(K, f"{(time.perf_counter() - t) * 1e3:.0f} ms", flush=True)
