"""The engine at long positions against the float32 reference of long_reference.py (9,600 tokens; positions every
1,024 up to 8,191 and every 64 from 8,191 on, where the log scaling of the global-attention queries starts).
MK_PREFILL=exact: wide exact passes (the bits of one-row steps). MK_PREFILL=fast: the fast wide passes' own rows, and
8-row step passes over a prompt prefilled fast up to position 8,000."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, (ZOO + "scripts"))
import mlx.core as mx
import mk_target as mt, mk_kernels as kn
from mk_target import Target
from mac_make_reference import kl, rel_l2, spread
rd = Path.home() / os.environ.get("REF", "mk-ref/inkling-2-mini-m5max-long")
ids = [int(t) for t in np.load(rd / "ids.npy")[0]]; pos = [int(p) for p in np.load(rd / "positions.npy")]
ref = torch.from_numpy(np.load(rd / "logits.npy")).float()
tg = Target(str(Path.home() / "models/Inkling-2-Mini"), 32768)
def rows_logits(fo16):
    z = kn.gemv(tg.unembed, fo16, tg.vocab); return mx.tanh(z * (1.0 / tg.cap)) * tg.cap
def wide(upto, store):
    i = tg.pos
    while i < upto:
        n = min(256, upto - i); K = next(k for k in (32, 64, 128, 256) if k >= n)
        tg.flush()
        toks = mx.array(ids[i:i + n] + [0] * (K - n), dtype=mx.int32)
        m_full = mx.array([tg.pos, 0, tg.pos], dtype=mx.int32); m_loc = mx.array([tg.pos, 1024, tg.pos], dtype=mx.int32)
        fo16, feats, knew, vnew, wins = tg._pure(toks, m_full, m_loc, tg.kc, tg.vc, tg.kr, tg.vr, tg.cw)
        rows = [p - i for p in pos if i <= p < i + n]
        if rows and store is not None:
            lg = rows_logits(fo16[mx.array(rows)]); mx.eval(lg)
            for r, row in enumerate(rows): store[i + row] = np.array(lg[r])
        tg.commit_wide(n, knew, vnew, wins); mx.eval(*tg.state()); i += n
def steps(upto, store):
    i = tg.pos
    while i < upto:
        K = min(8, upto - i); need = [p for p in pos if i <= p < i + K]
        lg = tg.forward(mx.array(ids[i:i + K], dtype=mx.int32), tok_only=not need); tg.commit(K)
        for p in need: store[p] = np.array(lg[p - i])
        mx.eval(lg, *tg.state()); i += K
def report(name, store):
    for label, sel in (("positions below 8,191", [p for p in pos if p < 8191 and p in store]), ("positions 8,191 and above", [p for p in pos if p >= 8191 and p in store])):
        if not sel: continue
        z = torch.from_numpy(np.stack([store[p] for p in sel])).float(); rr = ref[[pos.index(p) for p in sel]]
        print(json.dumps({"mode": name, "where": label, "n": len(sel), "rel_l2": spread(rel_l2(z, rr)), "kl": spread(kl(z, rr)),
                          "top1": (z.argmax(-1) == rr.argmax(-1)).float().mean().item()}), flush=True)
tg.reset(); st = {}; wide(len(ids), st); report(f"{mt.PREFILL} wide passes, their own rows", st)
if mt.PREFILL == "fast":
    P0 = int(os.environ.get("P0", "8000")); tg.reset(); st = {}; wide(P0, None); steps(len(ids), st); report(f"fast prefill of {P0}, then 8-row step passes", st)
if os.environ.get("NOTAU") == "1":          # what the error would be without the log scaling
    tg.tau = None; tg._fns = {}
    tg.reset(); st = {}; wide(len(ids), st); report(f"{mt.PREFILL} wide passes WITHOUT log scaling", st)
