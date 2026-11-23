"""Engine-level: TTFT of submit/poll for several prompt lengths, and (exact mode) the online tokens against step()."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
from api import Engine
t = time.time(); e = Engine(MODEL_DIR, 1, int(os.environ.get("CTX", "4096"))); print(f"load+warm {time.time() - t:.1f} s", flush=True)
rng = np.random.default_rng(1)
from tokenizers import Tokenizer
tok = Tokenizer.from_file((MODEL_DIR + "/tokenizer.json"))
import json
import glob
R = ZOO
files = [R + "runs-mac/deviation/README.md", R + "eval/README.md", R + "mkserve/README.md", R + "README.md"] + sorted(glob.glob(R + "eval/*.py")) + sorted(glob.glob(R + "mkserve/*.py"))
ids = tok.encode("\n\n".join(open(f).read() for f in files)).ids
print("text tokens", len(ids))
for L in [int(a) for a in sys.argv[2].split(",")]:
    prompt = ids[:L]
    for r in range(2):
        e.submit("r", prompt, 64); t = time.perf_counter(); out = []; ttft = None
        while True:
            ev = e.poll()
            for rid, toks, done in ev:
                if ttft is None: ttft = time.perf_counter() - t
                out += toks
            if ev and ev[-1][2]: break
        tot = time.perf_counter() - t
    print(f"prompt {L}: first token {ttft*1e3:.0f} ms ({L/ttft:.0f} tok/s), 64 tokens {tot:.2f} s", flush=True)
    if os.environ.get("CHECK") == "1":
        e.reset()
        for t_ in prompt[:-1]: e.step(np.array([[t_]]))
        last = np.array([[prompt[-1]]]); ref = []
        for _ in range(64):
            lg = e.step(last); nt = int(mx.argmax(lg[0] if lg.ndim == 2 else lg, axis=-1).item()); ref.append(nt); last = np.array([[nt]])
        print("   online == step greedy:", out == ref, flush=True)
