#!/usr/bin/env python
"""Layer by layer: mlx-lm's Inkling-2-Mini (float32 parameters) against eval/inkling2_ref.py mode high (float32, CPU),
on the first --tokens tokens of the reference's first sequence. After every layer application (48 for Mini: the loop
layers once per pass) both give the residual stream's partial sum; prints its rel-L2 per application, and the expert
choices' agreement.

    python scripts/mac_mlxlm_layer_diff.py [--tokens 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

zoo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(zoo / "eval"))
import inkling2_ref as ref  # noqa: E402


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm(dim=-1) / b.norm(dim=-1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path.home() / "models/Inkling-2-Mini"))
    ap.add_argument("--ref-dir", default=str(Path.home() / "mk-ref/inkling-2-mini-m5max-fp32"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no-kernels", action="store_true", help="mlx-lm's own Metal kernels off (its plain-MLX path)")
    ap.add_argument("--apps", type=int, default=48)
    ap.add_argument("--stepwise", action="store_true", help="one token per forward (with mlx-lm's cache): every product is a "
                    "one-row GEMV, float32-accurate; MLX's GPU float32 GEMMs of more rows are not on the M5 Max (~7.7e-4 "
                    "relative, as TF32)")
    args = ap.parse_args()
    ids = np.load(Path(args.ref_dir) / "ids.npy")[:1, : args.tokens]

    # mlx-lm
    from mlx_lm import load
    from mlx_lm.models.inkling2 import layers as L
    taps_m, routes_m = [], []
    orig = L.DecoderLayer.__call__

    def tap(self, *a, **k):
        out = orig(self, *a, **k)
        taps_m.append(out[0])
        return out
    L.DecoderLayer.__call__ = tap
    orig_router = L.TopKRouter.__call__

    def tap_router(self, x):
        out = orig_router(self, x)
        routes_m.append(out)
        return out
    L.TopKRouter.__call__ = tap_router
    if args.no_kernels:
        L._use_kernels = lambda: False
    model, _ = load(args.model_dir)
    if args.dtype == "float32":
        model.set_dtype(mx.float32)
    if args.stepwise:
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(model)
        outs = []
        for t in range(args.tokens):
            outs.append(model(mx.array(ids[:, t:t + 1]), cache=cache)[0])
            mx.eval(outs[-1], taps_m, routes_m)
        logits_m = mx.concatenate(outs)
        A = len(taps_m) // args.tokens
        taps_m = [mx.concatenate(taps_m[j::A], axis=1) for j in range(A)]
        routes_m = [tuple(mx.concatenate([r[k] for r in routes_m[j::A]], axis=-2) for k in range(2)) for j in range(A)]
    else:
        logits_m = model(mx.array(ids))[0]
    mx.eval(logits_m, taps_m)
    taps_m = [torch.from_numpy(np.array(t.astype(mx.float32)))[0] for t in taps_m]
    rm = []
    for r in routes_m:
        idx = [x for x in r if x.dtype in (mx.int32, mx.uint32, mx.int64)]
        rm.append(torch.from_numpy(np.array(idx[0]).astype(np.int64)).reshape(args.tokens, -1) if idx else None)
    logits_m = torch.from_numpy(np.array(logits_m.astype(mx.float32)))
    del model
    mx.clear_cache()

    # reference
    m = ref.Inkling2(args.model_dir, devices=[torch.device("cpu")], mode="high", dtype=torch.float32)
    taps_r, rec = [], {}
    with torch.no_grad():
        h = m.hidden(torch.from_numpy(ids).long(), record=rec, taps=taps_r)
        logits_r = m.logits(h).float()[0]
    taps_r = [t[0] for t in taps_r]
    slots = sorted(rec)
    print(f"applications: mlx-lm {len(taps_m)}, reference {len(taps_r)}; routers: mlx-lm {len(rm)}", flush=True)
    for j, (a, b) in enumerate(list(zip(taps_m, taps_r))[: args.apps]):
        e = rel(a, b)
        line = {"app": j, "rel_median": round(e.median().item(), 6), "rel_max": round(e.max().item(), 6),
                "pos_of_max": int(e.argmax())}
        if j < 18 and rm[j] is not None:  # slots 0-17 are the first 18 applications (later passes have slots past 24)
            ra, rb = rm[j].sort(-1).values, rec[slots[j]][0].reshape(args.tokens, -1).sort(-1).values
            line["route_rows_differ"] = int((ra != rb).any(-1).sum())
        print(json.dumps(line), flush=True)
    e = rel(logits_m - logits_m.mean(-1, keepdim=True), logits_r - logits_r.mean(-1, keepdim=True))
    print(json.dumps({"logits_rel_median": e.median().item(), "max": e.max().item(),
                      "top1": (logits_m.argmax(-1) == logits_r.argmax(-1)).float().mean().item()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
