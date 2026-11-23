#!/usr/bin/env python
"""Reference data for the kernel evaluator, made on this Mac (no CUDA, no float64 on the GPU).

    python scripts/mac_make_reference.py --out REF_DIR [--sequences 1] [--device cpu] [--dtype float32]

eval/make_reference.py's output (ids.npy, logits64.npy, long_positions.npy, long_logits64.npy, hf_bf16_err.npy,
hf_bf16_long_err.npy, meta.json), computed by eval/inkling2_ref.py, the repository's plain-PyTorch Inkling-2-Mini:
  reference   mode "high" in --dtype (float32 by default): the checkpoint's stored codes and scales dequantized per
              expert, everything between the discrete steps in float32 (24 significant bits against BF16's 8).
  BF16 path   mode "fork": the serving engines' rounding points (BF16 storage, FP32 math where they use it). Its
              error against the reference sets the evaluator's tolerance, as the Hugging Face BF16 path does in
              make_reference.py (there is no Hugging Face Inkling-2-Mini).
The token sequences are the first --sequences rows of the repository's eval/refs/<model>/ids.npy (the B200
reference's sequences: 1,152 tokens each), so the positions are the same as on the GPU machines.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

zoo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(zoo / "eval"))
import inkling2_ref  # noqa: E402


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    a = a.double() - a.double().mean(-1, keepdim=True)
    ref = ref.double() - ref.double().mean(-1, keepdim=True)
    return (a - ref).norm(dim=-1) / ref.norm(dim=-1)


def kl(a: torch.Tensor, ref: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    a2, r2 = a.reshape(-1, a.shape[-1]), ref.reshape(-1, ref.shape[-1])
    out = torch.empty(a2.shape[0], dtype=torch.float64)
    for i in range(0, a2.shape[0], chunk):
        lr = torch.log_softmax(r2[i:i + chunk].double(), -1)
        la = torch.log_softmax(a2[i:i + chunk].double(), -1)
        out[i:i + chunk] = (lr.exp() * (lr - la)).sum(-1)
    return out.reshape(a.shape[:-1])


def spread(x: torch.Tensor) -> dict:
    flat = x.flatten().sort().values
    return {"median": flat[len(flat) // 2].item(), "p99": flat[int(0.99 * len(flat))].item(), "max": flat[-1].item()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(Path.home() / "models/Inkling-2-Mini"))
    ap.add_argument("--ids", default=str(zoo / "eval/refs/thinkingmachines-inkling-2-mini/ids.npy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequences", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=0, help="use only the first N tokens of each sequence (0: all; a timing test)")
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--check-tokens", type=int, default=128)
    ap.add_argument("--long-every", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = torch.from_numpy(np.load(args.ids)[: args.sequences]).long()
    if args.tokens:
        ids = ids[:, : args.tokens]
    R, TL = ids.shape
    T = min(args.check_tokens, TL)
    long_positions = torch.arange(T, TL, args.long_every)
    pos = torch.cat([torch.arange(T), long_positions])
    print(f"MK_REF ids {tuple(ids.shape)}; check {T} positions, long {len(long_positions)} (every {args.long_every}); "
          f"device {args.device}, threads {torch.get_num_threads()}", flush=True)

    t0 = time.time()
    model = inkling2_ref.Inkling2(args.model_dir, devices=[torch.device(args.device)], mode="fork")
    print(f"MK_REF loaded in {time.time() - t0:.0f}s (act_quant={model.act_quant})", flush=True)
    logits = {}
    for mode in ("fork", "high"):
        model.p = inkling2_ref.Prec(mode, getattr(torch, args.dtype))
        t1 = time.time()
        rows = []
        with torch.no_grad():
            for r in range(R):
                h = model.hidden(ids[r:r + 1].to(args.device), {})
                rows.append(model.logits(h[:, pos.to(h.device)]).float().cpu()[0])
                del h
                print(f"MK_REF mode={mode} sequence {r + 1} of {R} at {time.time() - t1:.0f}s", flush=True)
        logits[mode] = torch.stack(rows)  # [R, len(pos), V]
        print(f"MK_REF mode={mode} done in {time.time() - t1:.0f}s", flush=True)

    ref, bf = logits["high"], logits["fork"]
    ref_check, ref_long = ref[:, :T].contiguous(), ref[:, T:].contiguous()
    bf_check, bf_long = bf[:, :T], bf[:, T:]
    err, err_long = rel_l2(bf_check, ref_check), rel_l2(bf_long, ref_long)
    np.save(out / "ids.npy", ids.numpy().astype(np.int64))
    np.save(out / "logits64.npy", ref_check.numpy())
    np.save(out / "long_positions.npy", long_positions.numpy().astype(np.int64))
    np.save(out / "long_logits64.npy", ref_long.numpy())
    np.save(out / "hf_bf16_err.npy", err.numpy().astype(np.float32))
    np.save(out / "hf_bf16_long_err.npy", err_long.numpy().astype(np.float32))
    cfg = json.loads((Path(args.model_dir) / "generation_config.json").read_text())
    eos = cfg.get("eos_token_id")
    meta = {"model_dir": args.model_dir, "sequences": R, "prompt_tokens": args.prompt_tokens, "check_tokens": T, "long_tokens": TL,
            "vocab": int(ref.shape[-1]), "ids_source": args.ids + f" (first {R} rows)",
            "reference_dtype": args.dtype, "reference_impl": f"eval/inkling2_ref.py Inkling2 mode high ({args.dtype}, act_quant={model.act_quant}) on {args.device}",
            "bf16_path": "eval/inkling2_ref.py mode fork (engines' BF16 rounding points); the hf_bf16_* keys hold its error",
            "banned_tokens": sorted([eos] if isinstance(eos, int) else (eos or [])),
            "hf_bf16_rel_l2": spread(err), "hf_bf16_top1_agreement": (bf_check.argmax(-1) == ref_check.argmax(-1)).float().mean().item(),
            "long_every": args.long_every, "long_positions": int(len(long_positions)),
            "hf_bf16_rel_l2_long": spread(err_long),
            "hf_bf16_top1_agreement_long": (bf_long.argmax(-1) == ref_long.argmax(-1)).float().mean().item(),
            "hf_bf16_kl": spread(kl(bf_check, ref_check)), "hf_bf16_kl_long": spread(kl(bf_long, ref_long)),
            "prompt_source": "eval/refs ids (B200 reference sequences)", "torch": torch.__version__,
            "machine": platform.platform(), "made_unix": time.time(), "seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("MK_REF " + json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
