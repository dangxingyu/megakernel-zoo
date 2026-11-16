"""Starting point: the Hugging Face model behind the engine contract. Correct and slow (one PyTorch call per
layer operation, the host in every step). Everything here is yours to replace; only the contract must hold
(eval/_engine_runner.py in the harness):

    Engine(model_dir, max_batch, max_context)
    reset()                      forget every sequence
    step(tokens[B] int64 cuda)   append one token per sequence, return next-token logits [B, vocab] on cuda
    close(), stats()             optional
"""

import torch
from transformers import AutoModelForCausalLM


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
        self.past = None

    def reset(self) -> None:
        self.past = None

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=tokens[:, None], past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return out.logits[:, -1]

    def close(self) -> None:
        self.model = None
        self.past = None

    def stats(self) -> dict:
        return {"implementation": "huggingface-reference"}
