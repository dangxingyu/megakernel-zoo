#!/usr/bin/env python
"""Where the BF16 path's error against the float32 reference comes from (eval/inkling2_ref.py on the CPU).

    python scripts/mac_error_attribution.py [--ref-dir ~/mk-ref/inkling-2-mini-m5max-fp32] [--out FILE.json]

The reference's positions (the first sequence: 128 check positions and every 8th position after them) are computed in:
  high              float32 everywhere (must reproduce the stored reference, logits64.npy);
  fork              the engines' rounding points (BF16 storage) everywhere: the BF16 path, the evaluator's tolerance;
  fork, high routes fork with every MoE layer's expert choice forced to high's: the error without routing flips;
  only:<family>     float32 everywhere except one family of operations, which rounds as the engines do (with high's
                    routes): that family's own share of the error.
Families: attention (projections, short convolutions, L2 clips, attention, output projection), moe (router, experts),
attnres (attention over depth and the residual partial sums), sconv (the short convolutions after attention and MoE),
norms (the RMS norms before attention and MoE), messages (the loop layers' message gates), unembed (the final norm and
the logits). Prints one JSON line per configuration: rel-L2 spread, top-1 agreement, KL, routing agreement.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

zoo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(zoo / "eval"))
sys.path.insert(0, str(zoo / "scripts"))
import inkling2_ref as ref  # noqa: E402
from mac_make_reference import kl, rel_l2, spread  # noqa: E402

FAMILIES = {
    "attention": [(ref, "attention")],
    "moe": [(ref, "moe")],
    "attnres": [(ref, "block_attn_res"), (ref.AttnResState, "finalize")],
    "sconv": [(ref, "static_sconv")],
    "norms": [(ref.Inkling2, "layer_inputs"), (ref.Inkling2, "mlp_input")],
    "messages": [(ref, "message_gate")],
    "unembed": [(ref.Inkling2, "logits")],
}
SEL: set = set()
STATE: dict = {}


def install():
    """Each family's functions run with the selected precision (model.p and any Prec argument) when the family is in
    SEL; calls nested inside inherit it."""
    for fam, sites in FAMILIES.items():
        for owner, name in sites:
            fn = getattr(owner, name)

            def g(*a, _fn=fn, _fam=fam, **k):
                if _fam not in SEL:
                    return _fn(*a, **k)
                m, ps = STATE["model"], STATE["psel"]
                old, m.p = m.p, ps
                a = [ps if isinstance(x, ref.Prec) else x for x in a]
                k = {kk: ps if isinstance(v, ref.Prec) else v for kk, v in k.items()}
                try:
                    return _fn(*a, **k)
                finally:
                    m.p = old
            setattr(owner, name, g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path.home() / "models/Inkling-2-Mini"))
    ap.add_argument("--ref-dir", default=str(Path.home() / "mk-ref/inkling-2-mini-m5max-fp32"))
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default=",".join(FAMILIES))
    args = ap.parse_args()
    rd = Path(args.ref_dir)
    meta = json.loads((rd / "meta.json").read_text())
    ids = torch.from_numpy(np.load(rd / "ids.npy")[:1]).long()
    T = meta["check_tokens"]
    pos = torch.cat([torch.arange(T), torch.from_numpy(np.load(rd / "long_positions.npy"))])
    stored = torch.cat([torch.from_numpy(np.load(rd / "logits64.npy")[0]), torch.from_numpy(np.load(rd / "long_logits64.npy")[0])])

    install()
    t0 = time.time()
    model = ref.Inkling2(args.model_dir, devices=[torch.device("cpu")], mode="fork")
    STATE["model"] = model
    high, fork = ref.Prec("high", torch.float32), ref.Prec("fork", torch.float32)
    STATE["psel"] = fork
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    def run(base, sel=(), routes=None):
        SEL.clear()
        SEL.update(sel)
        model.p = base
        model.force_routes = routes
        rec = {}
        with torch.no_grad():
            h = model.hidden(ids, record=rec)
            z = model.logits(h[:, pos]).float()[0]
        model.force_routes = None
        return z, rec

    rows = []

    def report(name, z, rec, z_ref, rec_ref):
        e = rel_l2(z, z_ref)
        flips = [(rec[s] != rec_ref[s]).any(-1).float().mean().item() for s in sorted(rec_ref)]
        row = {"config": name, "rel_l2": spread(e), "rel_l2_check_median": e[:T].median().item(),
               "rel_l2_long_median": e[T:].median().item(),
               "top1": (z.argmax(-1) == z_ref.argmax(-1)).float().mean().item(), "kl": spread(kl(z, z_ref)),
               "tokens_with_a_route_flip_per_layer": float(np.mean(flips))}
        rows.append(row)
        print(json.dumps(row), flush=True)

    t1 = time.time()
    zh, rh = run(high)
    print(f"high {time.time() - t1:.0f}s; against the stored reference: rel_l2 median "
          f"{rel_l2(zh, stored).median().item():.2e} max {rel_l2(zh, stored).max().item():.2e}", flush=True)
    # force_routes is indexed by layer; a loop layer is applied once per pass, so Routes hands out each application's
    # choices in turn
    apps = model.cfg.applications()
    per_layer = {}
    for i, _, s in apps:
        per_layer.setdefault(i, []).append(s)
    single = {i: rh[ss[0]] for i, ss in per_layer.items() if len(ss) == 1}
    multi = sorted(i for i, ss in per_layer.items() if len(ss) > 1)

    class Routes(dict):
        """force_routes[i] for a layer applied several times (the loop): the choices of each application in turn."""
        def __init__(self):
            super().__init__()
            self.count = {}

        def __getitem__(self, i):
            if i in single:
                return single[i]
            n = self.count.get(i, 0)
            self.count[i] = n + 1
            return rh[per_layer[i][n % len(per_layer[i])]]

        def __contains__(self, i):
            return True

    def forced():
        return Routes()

    zf, rf = run(fork)
    report("fork", zf, rf, zh, rh)
    zfr, rfr = run(fork, routes=forced())
    report("fork, high routes", zfr, rfr, zh, rh)
    for fam in args.only.split(","):
        z, r = run(high, (fam,), routes=forced())
        report(f"only:{fam}", z, r, zh, rh)
    z, r = run(high, tuple(FAMILIES), routes=forced())
    report("only:all families", z, r, zh, rh)
    print(f"multi-application layers (loop): {multi}; total {time.time() - t0:.0f}s", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "loop_layers": multi}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
