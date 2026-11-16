"""Load the text decoder of a Hugging Face checkpoint, in a given precision, straight onto the GPU.

    model = hf_text.load(model_dir, torch.bfloat16)       # forward(input_ids=..., past_key_values=..., use_cache=...)

Most checkpoints load through AutoModelForCausalLM, including multimodal ones whose family has a text-only class
(Qwen3.8: the `...ForConditionalGeneration` checkpoint loads as `Qwen3_5ForCausalLM`; the vision tower and the
multi-token-prediction head are left on disk). A family without one loads as its image-text model; it is then only
ever called with token ids, and the vision tower just sits in memory.

`device_map` places every tensor on the card as it is read, so a model larger than the pod's host memory can be
loaded (a float32 Flint is 159 GB); without `accelerate` the model goes through host memory instead.
"""

from __future__ import annotations

import torch


def load(model_dir: str, dtype: torch.dtype, device: str = "cuda"):
    import transformers

    kinds = [transformers.AutoModelForCausalLM]
    if hasattr(transformers, "AutoModelForImageTextToText"):
        kinds.append(transformers.AutoModelForImageTextToText)
    errors = []
    for kind in kinds:
        for placed in (True, False):
            try:
                if placed:
                    model = kind.from_pretrained(model_dir, dtype=dtype, device_map={"": device})
                else:
                    model = kind.from_pretrained(model_dir, dtype=dtype).to(device)
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
