#!/usr/bin/env python
"""What the evaluator has to know about a checkpoint and a card, read from the checkpoint itself (config.json and
the JSON headers of its safetensors shards; no weights are read) and from a small table of cards.

    python eval/modelinfo.py MODEL_DIR [--hw h100|b200] [--batch 1 8] [--context 192]

Standard library only: the loop's worker and the dashboard import it too.

Bytes one decode step has to read (`decode_bytes`). The count follows the model's structure:

  rest       every weight of the text decoder outside the groups below: read once per step
  head       the output head: `lm_head` (Inkling 2: `unembed`) when the checkpoint stores one, else the token embedding table (a tied head
             is the table). A stored table that is only a lookup is NOT counted (one row per sequence is)
  experts    weights under an `experts` module of a mixture-of-experts layer (bytes as stored: MXFP4 experts of
             gpt-oss count their U8 code blocks, two 4-bit codes per byte, plus one U8 scale per 32 codes, and BF16 biases). A token reads k of N experts per
             layer, so a step at batch B reads between k/N (all sequences route alike) and min(1, B*k/N) (all
             route apart) of them: `low` and `high`. At B=1 they are equal. Which experts a token uses depends
             on the token; equal expert sizes are assumed, the real routing is not measured here
  vision     vision tower / projector of a multimodal checkpoint: not read when decoding text
  aux        multi-token-prediction head stored next to the model (`mtp.*`): not used in plain decoding
  kv         K and V of earlier positions (BF16): `context` positions in a full-attention layer, at most the
             window in a sliding-window layer, none in a linear-attention (recurrent) layer
  state      recurrent state of linear-attention layers (float32 in the Hugging Face path): value heads x key
             head dim x value head dim, plus the short convolution state; Inkling 2's short convolutions keep
             the last kernel-1 inputs of four convolutions per layer

A looped model (Inkling-2-Mini, `loop_config`: layers core_start..core_end-1 run num_passes times with shared
weights, transformers#51050) is counted per layer APPLICATION: a looped layer's weights are read once per pass
(they do not fit any cache), its experts are chosen anew in every pass (k of N per application), and every pass
has its own K/V and short-convolution caches. The message gates of a looped layer (`controllers.*`) are used only
in the passes after the first. `applications` in the result says how many layer applications one token makes.

Activations are not counted.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

# Peak memory bandwidth per card, GB/s, used for MBU and the bandwidth bound. H100: the figure ForgeMegakernel
# uses (our read-rate microbenchmark, dashboard/membw.json, reaches 3085). B200: derived from NVIDIA's system
# figures (two cards: 16 TB/s), not published per card and not measured by us.
CARDS = {
    "h100": {"gpu": "NVIDIA H100 80GB HBM3", "request": "1 × H100", "hbm_gb_s": 3350.0, "memory_gb": 85.5, "sms": 132, "arch": "x86_64",
             "hbm_source": "vendor peak for HBM3 as used by ForgeMegakernel; our microbenchmark reads 3085 GB/s"},
    "b200": {"gpu": "NVIDIA B200 (GB200 node)", "request": "1 × B200", "hbm_gb_s": 8000.0, "memory_gb": 198.7, "sms": None, "arch": "aarch64",
             "hbm_source": "derived from NVIDIA's GB200 system figures (16 TB/s for two cards); not measured"},
    # A whole node as the target (the engine owns every card): the figures are the sum over its cards.
    "b200x4": {"gpu": "4 x NVIDIA B200 (one GB200 node)", "request": "4 × B200", "hbm_gb_s": 4 * 8000.0, "memory_gb": 4 * 198.7,
               "sms": None, "arch": "aarch64", "cards": 4,
               "hbm_source": "4 x the B200 figure (8000 GB/s per card, derived from NVIDIA's GB200 system figures, not measured); "
                             "cards are joined by NVLink (NV18 between every pair, about 777 GB/s measured peer copy)"},
}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}


def parameter_count(name: str, dtype: str, nbytes: int) -> int:
    """Parameters a stored tensor holds. MXFP4 (gpt-oss): `*_blocks` are U8 bytes with two 4-bit codes each (two
    parameters per byte) and `*_scales` one shared power-of-two exponent per 32 codes (no parameters of their own);
    their bytes are counted as stored either way. ModelOpt NVFP4 / MXFP8 experts (Inkling 2): a U8 `*_weight` holds
    two E2M1 codes per byte; `*_weight_scale` (E4M3 per 16 codes, or U8 = UE8M0 per 32 for MXFP8), `*_amax` and
    `*_outer_scale_denominator` are scales, not parameters."""
    if name.endswith("_blocks") and dtype == "U8":
        return 2 * nbytes
    if name.endswith("_scales") and dtype in ("U8", "F8_E8M0"):
        return 0
    if ".experts." in name:
        if name.endswith(("_weight_scale", "_amax", "_outer_scale_denominator")):
            return 0
        if name.endswith("_weight") and dtype == "U8":
            return 2 * nbytes
    return nbytes // DTYPE_BYTES.get(dtype, 2)


def card_count(hw: str | None) -> int:
    return int(CARDS.get(hw or "", {}).get("cards", 1))


def hw_of_gpu_name(name: str) -> str | None:
    low = name.lower()
    return "h100" if "h100" in low else "b200" if ("b200" in low or "gb200" in low) else None


def shards(model_dir: str | Path) -> list[Path]:
    """The checkpoint's shards: the files its index names when there is one (some repositories also carry a second
    copy of the weights in another layout, e.g. `consolidated.safetensors`), else every .safetensors file."""
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if files and all((root / f).exists() for f in files):
            return [root / f for f in files]
    return sorted(root.glob("*.safetensors"))


def tensors(model_dir: str | Path):
    """(name, dtype, shape, bytes) of every tensor of the checkpoint, from the shard headers."""
    for path in shards(model_dir):
        with open(path, "rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        for name, entry in header.items():
            if name != "__metadata__":
                yield name, entry["dtype"], entry["shape"], entry["data_offsets"][1] - entry["data_offsets"][0]


def category(name: str) -> str:
    low = name.lower()
    if low.startswith("mtp."):
        return "aux"
    if any(part in low for part in ("visual", "vision", "projector", "multi_modal", "image")):
        return "vision"
    if "lm_head" in low or low.endswith(".unembed.weight"):
        return "head"
    if "embed_tokens" in low or low.endswith("wte.weight") or "token_embedding" in low or low.endswith(".embed.weight"):
        return "input_embedding"
    if ".experts." in low or low.endswith(".experts") or ".experts_" in low:
        return "experts"
    return "rest"


def text_config(model_dir: str | Path) -> dict:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    return config.get("text_config", config)


def layer_applications(config: dict) -> list[int]:
    """How often one token applies each decoder layer: 1, or `num_passes` for the layers of a loop (`loop_config`
    of Inkling-2-Mini: core_start..core_end-1)."""
    n = config["num_hidden_layers"]
    loop = config.get("loop_config") or {}
    passes = int(loop.get("num_passes") or 1)
    start, end = int(loop.get("core_start", 0)), int(loop.get("core_end", 0))
    return [passes if start <= i < end else 1 for i in range(n)]


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def layer_of(name: str) -> int | None:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else None


def weights(model_dir: str | Path) -> dict:
    """Bytes and parameter counts per group, and what the text decoder holds in total."""
    size = {k: 0 for k in ("rest", "head", "experts", "input_embedding", "vision", "aux")}
    params = dict(size)
    for name, dtype, shape, nbytes in tensors(model_dir):
        group = category(name)
        size[group] += nbytes
        params[group] += parameter_count(name, dtype, nbytes)
    text = [k for k in size if k not in ("vision", "aux")]
    return {"bytes": size, "parameters": params, "text_parameters": sum(params[k] for k in text),
            "text_bytes": sum(size[k] for k in text), "stored_head": size["head"] > 0}


def decode_bytes(model_dir: str | Path, batch: int, context: float) -> dict:
    """Bytes one decode step at this batch size and context has to read. `total` uses the low expert count and is
    what MBU is computed from; `total_high` uses the high one (equal unless the model has experts and batch > 1)."""
    info = weights(model_dir)
    size = info["bytes"]
    config = text_config(model_dir)
    hidden = config["hidden_size"]
    head = size["head"] if info["stored_head"] else size["input_embedding"]
    apps = layer_applications(config)
    looped = any(a > 1 for a in apps)
    expert_reads = float(size["experts"])  # expert bytes times the applications of their layer
    if looped:  # extra reads of the looped layers' weights: once more per pass after the first
        extra_rest = 0
        expert_reads = 0.0
        for name, _, _, nbytes in tensors(model_dir):
            group, layer = category(name), layer_of(name)
            reads = apps[layer] if group in ("rest", "experts") and layer is not None else 1
            if group == "rest" and ".controllers." in name:
                reads = max(reads - 1, 0)  # message gates: passes after the first only (stored once, read P-1 times)
            if group == "rest":
                extra_rest += (reads - 1) * nbytes
            elif group == "experts":
                expert_reads += reads * nbytes
        size = {**size, "rest": size["rest"] + extra_rest}
    fixed = size["rest"] + head
    n_experts = config.get("num_experts") or config.get("n_routed_experts") or config.get("num_local_experts")  # gpt-oss: num_local_experts
    k_experts = config.get("num_experts_per_tok") or config.get("experts_per_token")
    how = ["every text-decoder weight outside the input embedding table once"
           + ("" if info["stored_head"] else " (the table is counted: it is the tied output head)")]
    experts_low = experts_high = 0.0
    if looped:
        loop = config["loop_config"]
        how.append(f"loop: layers {loop['core_start']}-{loop['core_end'] - 1} run {loop['num_passes']} passes with shared weights, "
                   f"{sum(apps)} layer applications per token from {len(apps)} layers: their weights are counted once per pass "
                   "(message gates once per pass after the first), their experts chosen anew in each pass, their K/V and "
                   "convolution state kept per pass")
    if size["experts"] and n_experts and k_experts:
        share = k_experts / n_experts
        experts_low, experts_high = expert_reads * share, expert_reads * min(1.0, batch * share)
        how.append(f"experts: {k_experts} of {n_experts} per layer and token, so {share:.4f} of the expert weights when all {batch} "
                   f"sequences route alike (low) and {min(1.0, batch * share):.4f} when they all route apart (high); equal expert "
                   "sizes assumed, real routing not measured")
    elif size["experts"]:
        experts_low = experts_high = float(size["experts"])
        how.append("experts: no num_experts / num_experts_per_tok in the config, all expert weights counted")
    head_dim = config.get("head_dim") or hidden // config["num_attention_heads"]
    per_position = 2 * config["num_key_value_heads"] * head_dim * 2  # K and V of one layer, BF16
    layers = config["num_hidden_layers"]
    types = config.get("layer_types")
    if not types and config.get("local_layer_ids") is not None:  # Inkling 2: the local layers are listed
        local = set(config["local_layer_ids"])
        types = ["sliding_attention" if i in local else "full_attention" for i in range(layers)]
    types = types or ["full_attention"] * layers
    window = config.get("sliding_window") or config.get("sliding_window_size") or 0
    kv = state = 0.0
    for kind in [t for t, reads in zip(types, apps) for _ in range(reads)]:  # a looped layer has one cache per pass
        if kind == "linear_attention":
            v_heads, k_dim, v_dim = (config.get(k, 0) for k in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
            conv = (2 * config.get("linear_num_key_heads", 0) * k_dim + v_heads * v_dim) * max(config.get("linear_conv_kernel_dim", 1) - 1, 0)
            state += v_heads * k_dim * v_dim * 4 + conv * 2
        elif kind == "sliding_attention" and window:
            kv += min(context, window) * per_position
        else:
            kv += context * per_position
    if config.get("sconv_kernel_size"):  # Inkling 2: short convolutions on K, V, attention output and MoE output
        lags = config["sconv_kernel_size"] - 1
        kv_dim = config["num_key_value_heads"] * head_dim
        state += sum(apps) * lags * (2 * kv_dim + 2 * hidden) * 2
        how.append(f"short-convolution state: the last {lags} inputs of the K, V, attention-output and MoE-output "
                   "convolutions per layer (BF16)")
    counts = {k: sum(a for t, a in zip(types, apps) if t == k) for k in sorted(set(types))}
    if len(counts) > 1 or "full_attention" not in counts:
        how.append("K and V per layer type " + json.dumps(counts) + (f", sliding window {window}" if window else "")
                   + ("; linear-attention layers have a float32 recurrent state instead of K/V" if "linear_attention" in counts else ""))
    embedding_rows = batch * hidden * 2
    total = fixed + experts_low + embedding_rows + batch * (kv + state)
    return {"weights": fixed + experts_low, "weights_fixed": fixed, "experts_low": experts_low, "experts_high": experts_high,
            "embedding_rows": embedding_rows, "kv": batch * kv, "state": batch * state,
            "total": total, "total_high": total + experts_high - experts_low, "applications": sum(apps), "how": "; ".join(how)}


def describe(model_dir: str | Path, hw: str | None = None, batches=(1, 8), context: float = 192.0) -> dict:
    info = weights(model_dir)
    config = text_config(model_dir)
    out = {"model_dir": str(model_dir), "shards": len(shards(model_dir)), **info,
           "config": {k: config.get(k) for k in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
                                                 "num_key_value_heads", "head_dim", "num_experts", "n_routed_experts", "num_local_experts",
                                                 "num_experts_per_tok", "sliding_window", "tie_word_embeddings")},
           "quantization": (json.loads((Path(model_dir) / "config.json").read_text()).get("quantization_config") or {}).get("quant_method"),
           "decode_bytes": {f"b{b}": decode_bytes(model_dir, b, context) for b in batches}, "context": context}
    if hw in CARDS:
        rate = CARDS[hw]["hbm_gb_s"] * 1e9
        out["card"] = {"hw": hw, **CARDS[hw]}
        out["bound_us"] = {b: {"low": v["total"] / rate * 1e6, "high": v["total_high"] / rate * 1e6} for b, v in out["decode_bytes"].items()}
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--hw", default=None)
    parser.add_argument("--batch", type=int, nargs="*", default=[1, 8])
    parser.add_argument("--context", type=float, default=192.0)
    args = parser.parse_args()
    print("MK_MODELINFO " + json.dumps(describe(args.model_dir, args.hw, args.batch, args.context)))
