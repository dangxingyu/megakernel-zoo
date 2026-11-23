"""Prefix cache: a request whose prompt starts like the cached sequence. Tokens with the cache against tokens after a
full reset (must be identical in exact mode), and the time to the first token of each."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, numpy as np
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
from api import Engine
from tokenizers import Tokenizer
MD = MODEL_DIR
e = Engine(MD, 1, int(os.environ.get("CTX", "4096")))
tok = Tokenizer.from_file(MD + "/tokenizer.json")
R = ZOO
ids = tok.encode(open(R + "runs-mac/deviation/README.md").read() + open(R + "eval/README.md").read()).ids
ids2 = tok.encode(open(R + "mkserve/README.md").read()).ids

def run(prompt, n, fresh):
    if fresh:
        e._hard_reset()
    e.submit("r", prompt, n); t = time.perf_counter(); out, ttft = [], None
    while True:
        ev = e.poll()
        for rid, toks, done in ev:
            if ttft is None: ttft = time.perf_counter() - t
            out += toks
        if ev and ev[-1][2]: break
    return out, ttft

N = 96
cases = []
p1 = ids[:1500]
a, _ = run(p1, N, True)
turn2 = p1 + a + ids2[:120]                       # the next turn of a chat: the old prompt, the reply, a new message
cases.append(("turn 2 = turn 1 + reply + 120 new", p1, turn2))
cases.append(("same prompt again", p1, p1))
cases.append(("prefix of the cached prompt (700)", p1, p1[:700]))
cases.append(("diverges after 1000 tokens", p1, p1[:1000] + ids2[:400]))
cases.append(("nothing shared", p1, ids2[:800]))
cases.append(("3,000-token prompt, then + reply + 60", ids[:3000], None))
ok = True
for name, first, second in cases:
    r1, _ = run(first, N, True)
    if second is None:
        second = first + r1 + ids2[:60]
    hits0, kept0 = e.counters["prefix_hits"], e.counters["prefix_tokens"]
    c, tc = run(second, N, False)
    kept = e.counters["prefix_tokens"] - kept0
    f, tf = run(second, N, True)
    same = c == f
    ok &= same or os.environ.get("MK_PREFILL") == "fast"
    print(f"{name}: prompt {len(second)}, kept {kept}; first token {tc*1e3:.0f} ms with the cache, {tf*1e3:.0f} ms without; "
          f"tokens identical: {same}" + ("" if same else f" (first difference at {next(i for i in range(N) if c[i] != f[i])})"), flush=True)
print("ALL IDENTICAL" if ok else "DIFFERENCES", e.stats()["spec"])
