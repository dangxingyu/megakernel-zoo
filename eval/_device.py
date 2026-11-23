"""Where the engine under test runs, and how the harness hands it tokens and reads back what it returns.

    MK_DEVICE=cuda           an NVIDIA card through PyTorch (the default everywhere but macOS)
    MK_DEVICE=metal          an Apple-silicon GPU (aliases: mps, mlx; the default on macOS)

The engine contract differs only in the types that cross it:

    cuda   step(tokens) takes an int64 CUDA tensor [B] and returns a CUDA tensor [B, vocab] (bf16 or float32);
           generate(tokens, n) returns an int64 CUDA tensor [B, n]
    metal  step(tokens) takes a NumPy int64 array [B] (on the host: unified memory, no copy to speak of; it may be a
           strided view, np.ascontiguousarray it if you need it packed) and returns an mx.array or a NumPy array
           [B, vocab] (float32, bfloat16 or float16). An mx.array may be lazy: the harness calls mx.async_eval on
           what step() returns (every step's logits are computed, as on the GPU path) and synchronizes with mx.eval
           + mx.synchronize() before it stops a clock. generate(tokens, n) returns an mx.array or NumPy array of
           integers [B, n]

Everything else (reset, close, stats, submit / poll with Python lists of token ids) is the same on both. Greedy
choices made by the harness are the first index of the maximum on both (torch.argmax, mx.argmax and np.argmax agree on
ties). No torch is needed on the metal path, no MLX on the cuda path.
"""

from __future__ import annotations

import contextlib
import os
import sys

_ALIASES = {"cuda": "cuda", "gpu": "cuda", "metal": "metal", "mps": "metal", "mlx": "metal", "apple": "metal"}


def device_name() -> str:
    wanted = (os.environ.get("MK_DEVICE") or ("metal" if sys.platform == "darwin" else "cuda")).strip().lower()
    if wanted not in _ALIASES:
        raise SystemExit(f"MK_DEVICE={wanted!r}: expected cuda or metal (mps, mlx)")
    return _ALIASES[wanted]


class Cuda:
    """The PyTorch / CUDA path, exactly the calls the harness made before it knew about other devices."""
    name = "cuda"

    def __init__(self):
        import torch
        self.torch = torch

    def ids(self, array):                       # [R, T] int64 from the reference, on the device
        return self.torch.from_numpy(array).cuda()

    def tensor(self, rows):                     # a list (of lists) of token ids, on the device
        return self.torch.tensor(rows, device="cuda")

    def reverse(self, n: int, like):            # index that reverses the slot order
        return self.torch.arange(n - 1, -1, -1, device=like.device)

    def host(self, logits):                     # logits -> NumPy float32 (a copy)
        return logits.float().cpu().numpy()

    def argmax(self, logits):                   # greedy choice as float32, on the device
        return logits.float().argmax(-1)

    def argmax_raw(self, logits):               # greedy choice in the logits' own dtype, on the device
        return logits.argmax(-1)

    def tokens_host(self, tokens):              # token ids -> NumPy (a copy on the host)
        return tokens.cpu().numpy()

    def to_host(self, tokens):                  # the host read a streaming server makes every step
        return tokens.cpu()

    def tolist(self, tokens) -> list:
        return tokens.tolist()

    def finite(self, logits) -> bool:
        return bool(self.torch.isfinite(logits.float()).all())

    def launch(self, value) -> None:            # CUDA work is already queued when step() returns
        return None

    def sync(self, value=None) -> None:
        self.torch.cuda.synchronize()

    def range_push(self, name: str) -> None:
        self.torch.cuda.nvtx.range_push(name)

    def range_pop(self) -> None:
        self.torch.cuda.nvtx.range_pop()

    def no_grad(self):
        return self.torch.no_grad()

    def memory(self) -> dict:                   # NVML is read by the runner itself on this path
        return {}


class Metal:
    """Apple silicon through MLX: tokens are NumPy int64 on the host, logits mx.array (or NumPy)."""
    name = "metal"

    def __init__(self):
        import mlx.core as mx
        import numpy as np
        self.mx, self.np = mx, np

    def ids(self, array):
        return self.np.ascontiguousarray(array, dtype=self.np.int64)

    def tensor(self, rows):
        return self.np.asarray(rows, dtype=self.np.int64)

    def reverse(self, n: int, like):
        return self.np.arange(n - 1, -1, -1)

    def _is_mx(self, value) -> bool:
        return isinstance(value, self.mx.array)

    def host(self, logits):
        np, mx = self.np, self.mx
        if self._is_mx(logits):
            return np.array(logits.astype(mx.float32) if logits.dtype != mx.float32 else logits, copy=True)
        if hasattr(logits, "detach"):           # a torch tensor (an engine that uses torch on the CPU or MPS)
            return logits.detach().float().cpu().numpy()
        return np.array(logits, dtype=np.float32, copy=True)

    def argmax(self, logits):
        np, mx = self.np, self.mx
        if self._is_mx(logits):
            return np.array(mx.argmax(logits, axis=-1)).astype(np.int64)
        if hasattr(logits, "detach"):
            return logits.detach().float().argmax(-1).cpu().numpy().astype(np.int64)
        return np.asarray(logits).argmax(-1).astype(np.int64)

    argmax_raw = argmax

    def tokens_host(self, tokens):
        np = self.np
        if self._is_mx(tokens):
            return np.array(tokens).astype(np.int64)
        if hasattr(tokens, "detach"):
            return tokens.detach().cpu().numpy().astype(np.int64)
        return np.asarray(tokens, dtype=np.int64)

    to_host = tokens_host

    def tolist(self, tokens) -> list:
        return [int(t) for t in self.tokens_host(tokens).reshape(-1)]

    def finite(self, logits) -> bool:
        if self._is_mx(logits):
            return bool(self.mx.all(self.mx.isfinite(logits)).item())
        return bool(self.np.isfinite(self.host(logits)).all())

    def launch(self, value) -> None:
        """Start computing what step() returned without waiting for it (the GPU path's asynchronous launch)."""
        if self._is_mx(value):
            self.mx.async_eval(value)

    def sync(self, value=None) -> None:
        if self._is_mx(value):
            self.mx.eval(value)
        self.mx.synchronize()

    def range_push(self, name: str) -> None:
        return None

    def range_pop(self) -> None:
        return None

    def no_grad(self):
        return contextlib.nullcontext()

    def memory(self) -> dict:
        mx = self.mx
        info = mx.device_info()
        return {"gpu_memory_used_gb": mx.get_peak_memory() / 1e9, "gpu_memory_active_gb": mx.get_active_memory() / 1e9,
                "gpu_memory_note": "MLX allocations of the engine's process: peak and at the end (mx.get_peak_memory / get_active_memory)",
                "gpu_name": info.get("device_name"), "gpu_working_set_gb": info.get("max_recommended_working_set_size", 0) / 1e9}


_DEVICE = None


def get():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = Metal() if device_name() == "metal" else Cuda()
    return _DEVICE
