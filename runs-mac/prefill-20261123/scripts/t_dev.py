"""Deviation from the float32 reference at its long positions (128 .. 1144, every 8th):
 step  : the engine's own exact passes over the whole sequence (what the gates measure)
 fastP : the first P tokens by fast wide prefill, the rest by exact passes (what serving a P-token prompt does)
 wide  : the fast wide passes' own logits (the first token of a reply comes from one of these rows)"""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, (ZOO + "scripts"))
import mlx.core as mx
import mk_target as mt, mk_kernels as kn
from mk_target import Target
from mac_make_reference import kl, rel_l2, spread
rd = (HOME + "/mk-ref/inkling-2-mini-m5max-fp32/")
ids = [int(t) for t in np.load(rd + "ids.npy")[0]]
lp = np.load(rd + "long_positions.npy")
ref = torch.from_numpy(np.load(rd + "long_logits64.npy")[0]).float()
tg = Target(MODEL_DIR, 4096)
def logits_rows(fo16):
    z = kn.gemv(tg.unembed, fo16, tg.vocab)
    return mx.tanh(z * (1.0 / tg.cap)) * tg.cap
def exact_from(P, store):
    i = P
    while i < len(ids):
        K = min(8, len(ids) - i)
        lg = tg.forward(mx.array(ids[i:i + K], dtype=mx.int32)); tg.commit(K)
        for r in range(K):
            if i + r in store: store[i + r] = np.array(lg[r])
        mx.eval(*tg.state()); i += K
def report(name, store, sel):
    z = torch.from_numpy(np.stack([store[p] for p in sel])).float()
    rr = ref[[int(np.where(lp == p)[0][0]) for p in sel]]
    e = rel_l2(z, rr)
    print(json.dumps({"mode": name, "n": len(sel), "rel_l2": spread(e), "kl": spread(kl(z, rr)),
                      "top1": (z.argmax(-1) == rr.argmax(-1)).float().mean().item()}), flush=True)
mode = mt.PREFILL
tg.reset(); st = {int(p): None for p in lp}; exact_from(0, st)
base = dict(st)
for P in (256, 512, 1024):
    sel = [int(p) for p in lp if p >= P]
    report(f"step (positions >= {P})", base, sel)
    tg.reset(); i = 0
    while i < P:
        t, _ = tg.prefill(ids[i:i + 256], 256); mx.eval(t, *tg.state()); i += 256
    st = {p: None for p in sel}; exact_from(P, st)
    report(f"{mode} prefill {P} + steps", st, sel)
# the wide passes' own rows
tg.reset(); st = {}; i = 0
while i < 1152:
    n = min(256, 1152 - i)
    toks = mx.array(ids[i:i + n] + [0] * (256 - n), dtype=mx.int32)
    m_full = mx.array([tg.pos, 0, tg.pos], dtype=mx.int32); m_loc = mx.array([tg.pos, 1024, tg.pos], dtype=mx.int32)
    tg.flush()
    fo16, feats, knew, vnew, wins = tg._pure(toks, m_full, m_loc, tg.kc, tg.vc, tg.kr, tg.vr, tg.cw)
    rows = [int(p) - i for p in lp if i <= p < i + n]
    lg = logits_rows(fo16[mx.array(rows)]); mx.eval(lg)
    for r, row in enumerate(rows): st[i + row] = np.array(lg[r])
    tg.commit_wide(n, knew, vnew, wins); mx.eval(*tg.state())
    i += n
report(f"{mode} wide rows", st, [int(p) for p in lp])
report("step (all)", base, [int(p) for p in lp])
