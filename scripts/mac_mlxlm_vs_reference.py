#!/usr/bin/env python
"""mlx-lm's Inkling-2-Mini against the float32 reference, in its own BF16 and with every floating parameter in
float32 (model.set_dtype; the quantized expert codes and their scales are unchanged).

    python scripts/mac_mlxlm_vs_reference.py [--ref-dir ~/mk-ref/inkling-2-mini-m5max-fp32]

mlx-lm takes its activation dtype from the embedding's: in BF16 it rounds where the engines do; in float32 it should
agree with eval/inkling2_ref.py mode high to float32 accuracy (up to routing flips) if the two implement one function.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import load

zoo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(zoo / "scripts"))
from mac_make_reference import kl, rel_l2, spread  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path.home() / "models/Inkling-2-Mini"))
    ap.add_argument("--ref-dir", default=str(Path.home() / "mk-ref/inkling-2-mini-m5max-fp32"))
    args = ap.parse_args()
    rd = Path(args.ref_dir)
    meta = json.loads((rd / "meta.json").read_text())
    T = meta["check_tokens"]
    ids = np.load(rd / "ids.npy")[:1]
    pos = np.concatenate([np.arange(T), np.load(rd / "long_positions.npy")])
    stored = torch.cat([torch.from_numpy(np.load(rd / "logits64.npy")[0]), torch.from_numpy(np.load(rd / "long_logits64.npy")[0])])

    model, _ = load(args.model_dir)
    for name in ("bfloat16", "float32"):
        if name == "float32":
            model.set_dtype(mx.float32)
        z = model(mx.array(ids))[0][mx.array(pos)]
        mx.eval(z)
        z = torch.from_numpy(np.array(z.astype(mx.float32)))
        e = rel_l2(z, stored)
        print(json.dumps({"mlxlm": name, "rel_l2": spread(e), "rel_l2_check_median": e[:T].median().item(),
                          "top1": (z.argmax(-1) == stored.argmax(-1)).float().mean().item(), "kl": spread(kl(z, stored))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
