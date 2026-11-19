"""Load the text decoder of a Hugging Face checkpoint, in a given precision, straight onto the GPU.

    model = hf_text.load(model_dir, torch.bfloat16)       # forward(input_ids=..., past_key_values=..., use_cache=...)

Most checkpoints load through AutoModelForCausalLM, including multimodal ones whose family has a text-only class
(Qwen3.8: the `...ForConditionalGeneration` checkpoint loads as `Qwen3_5ForCausalLM`; the vision tower and the
multi-token-prediction head are left on disk). A family without one loads as its image-text model; it is then only
ever called with token ids, and the vision tower just sits in memory.

`device_map` places every tensor on the card as it is read, so a model larger than the pod's host memory can be
loaded (a float32 Flint is 159 GB); without `accelerate` the model goes through host memory instead.

`cards` > 1 spreads the layers over that many cards (device_map "auto", layer after layer: one card works at a time),
for a reference whose weights fit no single card (gpt-oss-120b in float32 is 468 GB). A checkpoint stored in MXFP4
(gpt-oss's experts) is loaded in its own format for BF16 (the MXFP4 kernels when transformers finds them) and
dequantized for any other precision: every MXFP4 value (4-bit code times a power of two) is exact in BF16, float32
and float64, so the higher-precision model has exactly the checkpoint's weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load(model_dir: str, dtype: torch.dtype, device: str = "cuda", cards: int = 1):
    import transformers

    extra = {}
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if (config.get("quantization_config") or {}).get("quant_method") == "mxfp4" and dtype != torch.bfloat16:
        extra["quantization_config"] = transformers.Mxfp4Config(dequantize=True)
    text = config.get("text_config", config)
    if dtype == torch.float64 and any(text.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        extra["experts_implementation"] = "eager"  # the default grouped matrix product has no float64 kernel
    if cards > 1:
        free = {i: int(torch.cuda.mem_get_info(i)[0] * 0.92) for i in range(cards)}
        return transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map="auto", max_memory=free, **extra).eval()
    kinds = [transformers.AutoModelForCausalLM]
    if hasattr(transformers, "AutoModelForImageTextToText"):
        kinds.append(transformers.AutoModelForImageTextToText)
    errors = []
    for kind in kinds:
        for placed in (True, False):
            try:
                if placed:
                    model = kind.from_pretrained(model_dir, dtype=dtype, device_map={"": device}, **extra)
                else:
                    model = kind.from_pretrained(model_dir, dtype=dtype, **extra).to(device)
                return model.eval()
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                errors.append(f"{kind.__name__} device_map={placed}: {type(error).__name__}: {str(error)[:300]}")
                if "Unrecognized configuration class" in str(error):
                    break  # this Auto class does not know the checkpoint's family: try the next
                torch.cuda.empty_cache()  # else once more through host memory (no accelerate, or a device_map quirk)
    raise RuntimeError("could not load " + model_dir + ": " + " | ".join(errors))


def describe(model) -> dict:
    """What was loaded: class, parameter count, the dtypes its parameters ended up in."""
    dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        dtypes[str(parameter.dtype).removeprefix("torch.")] = dtypes.get(str(parameter.dtype).removeprefix("torch."), 0) + parameter.numel()
    return {"class": type(model).__name__, "parameters": sum(dtypes.values()), "parameter_dtypes": dtypes}
