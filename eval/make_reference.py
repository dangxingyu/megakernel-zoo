#!/usr/bin/env python
"""Reference data for the kernel evaluator, made once per model on a GPU with room for the reference's weights.

    python eval/make_reference.py --model-dir DIR --out REF_DIR [--dtype auto|float64|float32]

The reference is the Hugging Face model in float64 (8 bytes per parameter). When that does not fit the card it is
float32 (`--dtype auto` decides from the text decoder's parameter count and the card's memory; meta.json records
`reference_dtype`). float32 keeps 24 significant bits where BF16 keeps 8, so its own rounding error is about 2^16
times smaller than the error the gate measures; the tolerance (a multiple of the BF16 path's own error against the
reference) is unchanged. A model whose float32 copy does not fit either gets no reference: the script stops.

What it writes:
  ids.npy        int64 [R, T_long]   token sequences: a P-token prompt, then the model's own greedy continuation
                                      (Hugging Face, BF16). Every later check feeds these tokens (teacher forcing).
  logits64.npy   float32 [R, T, V]   logits of the float64 model for the first T positions (stored as float32);
                                      position t holds the logits after reading token t, i.e. the prediction of t+1.
  long_positions.npy, long_logits64.npy   int64 [L], float32 [R, L, V]: the same for every `--long-every`-th
                                      position from T to T_long, so that long contexts and every part of a timed
                                      window can be checked (an audit found that nothing past T was ever compared).
  meta.json      shapes, versions, and how far the Hugging Face BF16 path itself is from float64 on these
                 positions (relative L2 error per position, top-1 agreement). The evaluator's tolerance is a
                 multiple of that, so it follows the model instead of being a hand-picked number.

Why float64: a BF16 reference has its own rounding error of the size we want to bound; comparing a kernel with
it cannot tell a kernel that is more accurate from one that is less (ForgeMegakernel's oracle does the same).
The file and key names keep "64" whatever the reference's precision is.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_text  # noqa: E402
import modelinfo  # noqa: E402
import prompts  # noqa: E402

FILL = {"float64": 0.86, "float32": 0.90}  # largest share of the card's memory the reference's weights may take


def choose_dtype(model_dir: str, wanted: str, cards: int = 1) -> tuple[str, dict]:
    """float64 when the text decoder's weights fit the card(s) in float64, else float32, else nothing."""
    params = modelinfo.weights(model_dir)["text_parameters"]
    total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(cards))
    need = {"float64": params * 8, "float32": params * 4}
    why = {"text_parameters": params, "gpu_memory_gb": total / 1e9, "cards": cards, "float64_gb": need["float64"] / 1e9, "float32_gb": need["float32"] / 1e9}
    if wanted != "auto":
        return wanted, why
    for name in ("float64", "float32"):
        if need[name] <= FILL[name] * total:
            return name, why
    raise SystemExit("MK_REF no reference: even float32 weights (%.0f GB) do not fit this card (%.0f GB)" % (need["float32"] / 1e9, total / 1e9))


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per position: ||a - ref|| / ||ref|| over the vocabulary, both centred (logits are defined up to a shift)."""
    a = a.double() - a.double().mean(-1, keepdim=True)
    ref = ref.double() - ref.double().mean(-1, keepdim=True)
    return (a - ref).norm(dim=-1) / ref.norm(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--check-tokens", type=int, default=128, help="T: positions with float64 logits")
    parser.add_argument("--long-tokens", type=int, default=1152, help="T_long: sequence length available for timing")
    parser.add_argument("--long-every", type=int, default=8, help="float64 logits are kept at every N-th position past T")
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--ids", default=None, help="int64 [R, T_long] .npy: use these token sequences instead of generating them. The continuation of a prompt "
                        "differs between machines at near-ties, and with it the reference; fixed sequences make the evaluation the same everywhere")
    parser.add_argument("--dtype", choices=["auto", "float64", "float32"], default="auto")
    parser.add_argument("--cards", type=int, default=1, help="spread the reference model over this many cards (layer after layer) when "
                        "it fits no single card; the BF16 path always runs on card 0 alone")
    args = parser.parse_args()
    import transformers

    ref_dtype, sizing = choose_dtype(args.model_dir, args.dtype, args.cards)
    print("MK_REF dtype " + json.dumps({"reference_dtype": ref_dtype, **sizing}), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = prompts.load(args.model_dir)
    R, P, T, TL = args.sequences, args.prompt_tokens, args.check_tokens, args.long_tokens
    start = torch.tensor([prompts.make(tok, P, args.seed + i)["ids"] for i in range(R)], dtype=torch.long, device="cuda")

    t0 = time.time()
    model = hf_text.load(args.model_dir, torch.bfloat16)
    loaded = hf_text.describe(model)
    print("MK_REF bf16 model " + json.dumps(loaded), flush=True)
    banned = set()
    for config in (model.config, getattr(model, "generation_config", None)):
        eos = getattr(config, "eos_token_id", None)
        banned |= set() if eos is None else ({eos} if isinstance(eos, int) else set(eos))
    banned = sorted(banned)
    given = torch.from_numpy(np.load(args.ids)).cuda() if args.ids else None
    if given is not None:
        R, TL = given.shape
    with torch.no_grad():
        ids = start
        past = None
        step_in = ids
        for _ in range(0 if given is not None else TL - P):
            result = model(input_ids=step_in, past_key_values=past, use_cache=True)
            past = result.past_key_values
            logits = result.logits[:, -1].float()
            logits[:, banned] = -float("inf")  # the sequences must not end: the evaluator needs every position
            step_in = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, step_in], dim=1)
        if given is not None:
            ids = given
        else:
            del past, result
        long_positions = torch.arange(T, TL, args.long_every)
        full = [model(input_ids=ids[r:r + 1]).logits[0].float().cpu() for r in range(R)]  # one pass over each whole sequence
        bf16_logits = torch.stack([f[:T] for f in full])
        bf16_long = torch.stack([f[long_positions] for f in full])
        del full
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"MK_REF bf16 done in {time.time() - t0:.0f}s ids {tuple(ids.shape)}", flush=True)

    t0 = time.time()
    model64 = hf_text.load(args.model_dir, getattr(torch, ref_dtype), cards=args.cards)
    loaded64 = hf_text.describe(model64)
    print("MK_REF reference model " + json.dumps({**loaded64, "gpu_allocated_gb": [torch.cuda.memory_allocated(i) / 1e9 for i in range(args.cards)]}), flush=True)
    logits64 = torch.empty(R, T, bf16_logits.shape[-1], dtype=torch.float32)
    long64 = torch.empty(R, len(long_positions), bf16_logits.shape[-1], dtype=torch.float32)
    with torch.no_grad():
        for r in range(R):
            whole = model64(input_ids=ids[r:r + 1]).logits[0]
            logits64[r] = whole[:T].float().cpu()
            long64[r] = whole[long_positions.to(whole.device)].float().cpu()
            del whole
    del model64
    torch.cuda.empty_cache()
    print(f"MK_REF {ref_dtype} done in {time.time() - t0:.0f}s", flush=True)

    err = rel_l2(bf16_logits, logits64)  # [R, T]
    agree = (bf16_logits.argmax(-1) == logits64.argmax(-1)).float().mean().item()
    np.save(out / "ids.npy", ids.cpu().numpy())
    np.save(out / "logits64.npy", logits64.numpy())
    np.save(out / "long_positions.npy", long_positions.numpy())
    np.save(out / "long_logits64.npy", long64.numpy())
    np.save(out / "hf_bf16_err.npy", err.numpy().astype(np.float32))                         # [R, T]: the BF16 path's error at each position,
    np.save(out / "hf_bf16_long_err.npy", rel_l2(bf16_long, long64).numpy().astype(np.float32))  # [R, long positions]: for reading a failed gate
    long_flat = rel_l2(bf16_long, long64).flatten().sort().values
    long_agree = (bf16_long.argmax(-1) == long64.argmax(-1)).float().mean().item()
    flat = err.flatten().sort().values
    meta = {"model_dir": args.model_dir, "sequences": R, "prompt_tokens": P, "check_tokens": T, "long_tokens": TL,
            "vocab": int(logits64.shape[-1]), "seed": args.seed,
            "reference_dtype": ref_dtype, "reference_sizing": sizing, "hf_class": loaded["class"], "hf_bf16_parameter_dtypes": loaded["parameter_dtypes"],
            "reference_parameter_dtypes": loaded64["parameter_dtypes"], "banned_tokens": banned,
            "hf_bf16_rel_l2": {"median": flat[len(flat) // 2].item(), "p99": flat[int(0.99 * len(flat))].item(), "max": flat[-1].item()},
            "hf_bf16_top1_agreement": agree,
            "long_every": args.long_every, "long_positions": int(len(long_positions)),
            "hf_bf16_rel_l2_long": {"median": long_flat[len(long_flat) // 2].item(), "p99": long_flat[int(0.99 * len(long_flat))].item(),
                                    "max": long_flat[-1].item()},
            "hf_bf16_top1_agreement_long": long_agree,
            "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
            "made_unix": time.time()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("MK_REF " + json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
