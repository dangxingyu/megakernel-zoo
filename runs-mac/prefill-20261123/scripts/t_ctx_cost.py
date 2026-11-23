"""What a larger max_context costs with a short prompt: decode ms per token (greedy, evaluator prompts, 512 tokens)
and memory."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, (ZOO + "eval"))
import mlx.core as mx
from api import Engine
import prompts as P
MD = MODEL_DIR; CTX = int(sys.argv[2])
t = time.time(); e = Engine(MD, 1, CTX); load = time.time() - t
tok = P.load(MD); res = []
for seed in (0, 1, 2):
    prompt = P.make(tok, 64, seed)["ids"]
    for rep in range(2):
        e.submit("r", prompt, 512); out, first = [], None
        while True:
            ev = e.poll()
            for rid, toks, done in ev:
                if first is None: first = time.perf_counter()
                out += toks
            if ev and ev[-1][2]: break
        ms = (time.perf_counter() - first) / (len(out) - 1) * 1e3
    res.append(ms)
print(f"max_context {CTX}: load {load:.1f} s; decode {np.mean(res):.2f} ms/token ({', '.join('%.2f' % r for r in res)}); "
      f"memory active {mx.get_active_memory() / 1e9:.1f} GB, peak {mx.get_peak_memory() / 1e9:.1f} GB, cache {mx.get_cache_memory() / 1e9:.1f} GB", flush=True)
