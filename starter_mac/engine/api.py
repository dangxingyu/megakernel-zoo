"""Starting point on Apple silicon: Inkling-2-Mini through mlx-lm's `inkling2` model (ml-explore/mlx-lm#2086) behind the
engine contract. Correct and slow (one MLX graph per token built in Python, the host in every step, no speculation).
Everything here is yours to replace; only the contract must hold (eval/_engine_runner.py and eval/_device.py in the
harness, MK_DEVICE=metal):

    Engine(model_dir, max_batch, max_context)
    reset()                          forget every sequence
    step(tokens)                     tokens: NumPy int64 [B], the next input token of each sequence (B fixed between
                                     resets); returns the next-token logits [B, 200058] as an mx.array (float32 here;
                                     bf16 is fine), possibly still lazy: the harness evaluates it
    close(), stats()                 optional
    generate(tokens, n_steps)        optional: greedy loop inside the engine; returns integers [B, n_steps]
    submit(rid, prompt_ids, n), poll()   optional, the online contract (eval/serve_engine.py): where a track-C engine
                                     drafts with the checkpoint's DSpark head (mtp.safetensors) and verifies

The weights stay in the checkpoint's stored formats: mlx-lm maps the NVFP4 experts (U8 codes, E4M3 scales per 16,
per-expert amax / outer-scale denominator -> one float32 tensor scale per expert) and the MXFP8 experts of four layers
(E4M3 codes, UE8M0 scales per 32) onto mx.gather_qmm's nvfp4 / mxfp8 modes, and keeps every other weight in BF16; the
expert inputs are not quantized (as in the harness's reference). The vision tower is not loaded. About 19 GB.

This file uses mlx-lm, a general engine's model code: fine for a starting point (the CUDA starter uses transformers),
not in the timed path of a submission.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        from mlx_lm.utils import load_model

        t0 = time.time()
        self.max_batch, self.max_context = int(max_batch), int(max_context)
        info = mx.device_info()
        # keep the weights resident (what mlx_lm.generate does): unified memory is otherwise free to page them out
        mx.set_wired_limit(info["max_recommended_working_set_size"])
        # text only: vision_config None drops the vision tower (its weights are skipped by the loader's sanitize)
        self.model, self.config = load_model(Path(model_dir), lazy=False, model_config={"vision_config": None})
        self.vocab = int(self.model.args.text.unpadded_vocab_size)
        self.cache = None
        self.position = 0
        self.steps = 0
        self.load_seconds = time.time() - t0

    def reset(self) -> None:
        self.cache = None
        self.position = 0

    def step(self, tokens) -> mx.array:
        tokens = np.asarray(tokens, dtype=np.int64)
        B = tokens.shape[0]
        if B > self.max_batch:
            raise ValueError(f"batch {B} > max_batch {self.max_batch}")
        if self.position >= self.max_context:
            raise ValueError(f"context limit {self.max_context} reached")
        if self.cache is None:
            self.cache = self.model.make_cache()
        x = mx.array(tokens.astype(np.int32))[:, None]          # [B, 1]
        logits = self.model(x, cache=self.cache)[:, -1]          # [B, vocab] float32, soft-capped, unpadded vocabulary
        mx.async_eval(logits)                                     # start the step; the caller synchronizes when it reads
        self.position += 1
        self.steps += 1
        return logits

    def close(self) -> None:
        self.model = None
        self.cache = None

    def stats(self) -> dict:
        return {"implementation": "mlx-lm inkling2 (ml-explore/mlx-lm#2086) behind the contract", "steps": self.steps,
                "load_seconds": round(self.load_seconds, 1), "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}
