"""Sampling through draft-and-verify: are the tokens of submit()/poll() with sampling parameters exactly the tokens a
step-by-step loop draws with the same seed (step() logits + Engine.sample at each position)? And how fast is it."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, time, glob, json, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, (ZOO + "eval"))
import mlx.core as mx
from api import Engine
from tokenizers import Tokenizer
MD = MODEL_DIR
e = Engine(MD, 1, 4096)
tok = Tokenizer.from_file(MD + "/tokenizer.json")
R = ZOO
N = int(os.environ.get("N", "256"))
prompts = {}
ids = tok.encode(open(R + "runs-mac/deviation/README.md").read()).ids
prompts["notes, 300 tokens"] = ids[:300]
prompts["python source, 200 tokens"] = tok.encode(open(R + "eval/prompts.py").read()).ids[:200]
chat = "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle.<|im_end|>\n<|im_start|>assistant\n"
try:
    import prompts as P
    prompts["evaluator prompt, 64 tokens"] = P.make(P.load(MD), 64, 0)["ids"]
except Exception as ex:
    print("no evaluator prompt:", ex)
prompts["story request"] = tok.encode("Write a short story about a lighthouse keeper who finds a message in a bottle.\n\n").ids

def online(prompt, cfg):
    c0 = dict(e.counters)
    e.submit("r", prompt, N, sampling=cfg); t = time.perf_counter(); out, first = [], None
    while True:
        ev = e.poll()
        for rid, toks, done in ev:
            if first is None: first = time.perf_counter()
            out += toks
        if ev and ev[-1][2]: break
    dt = time.perf_counter() - first
    cyc = e.counters["cycles"] - c0["cycles"]
    return out, dt / max(1, len(out) - 1) * 1e3, (len(out) - 1) / max(1, cyc)

def stepwise(prompt, cfg):
    e.reset()
    for t_ in prompt[:-1]: e.step(np.array([t_]))
    last, out = prompt[-1], []
    for i in range(N):
        lg = e.step(np.array([last]))
        last = e.sample(lg, len(prompt) + i, cfg) if cfg else int(mx.argmax(lg.reshape(-1)).item())
        out.append(last)
    return out

cfgs = [("greedy", None), ("T 0.7", {"temperature": 0.7, "seed": 11}), ("T 1.0 top_p 0.95", {"temperature": 1.0, "top_p": 0.95, "seed": 12}),
        ("T 0.8 top_k 40", {"temperature": 0.8, "top_k": 40, "seed": 13}), ("T 0.3", {"temperature": 0.3, "seed": 14})]
ok = True
for pname, prompt in prompts.items():
    for cname, cfg in cfgs:
        a, ms, tpc = online(prompt, cfg)
        a2, ms2, _ = online(prompt, cfg)                     # again: compiled, and the same seed must give the same tokens
        b = stepwise(prompt, cfg)
        same = a == b
        ok &= same and a == a2
        print(f"{pname} | {cname}: online == stepwise {same}" + ("" if same else f" (first difference at {next(i for i in range(N) if a[i] != b[i])})")
              + f"; same seed again {a == a2}; {ms2:.2f} ms/token, {tpc:.2f} tokens/cycle, {len(set(a))} distinct", flush=True)
    c1 = online(prompt, {"temperature": 0.7, "seed": 99})[0]
    print(f"{pname} | another seed differs: {c1 != online(prompt, {'temperature': 0.7, 'seed': 11})[0]}", flush=True)
print("ALL EXACT" if ok else "MISMATCH")
