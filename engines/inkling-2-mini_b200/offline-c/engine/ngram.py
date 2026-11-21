"""Context-lookup drafts (prompt lookup / n-gram copy) next to the DSpark draft, for exact draft-and-verify.

After the draft head has written its proposal (vtok = [anchor, 7 drafts] at position c = vpos), this module looks
for the longest earlier occurrence (up to 8 tokens) of the committed text's suffix ending at the anchor; if it is at
least MIN_N tokens long, the 7 tokens that followed that occurrence replace the draft head's proposal (periodic
extension when the occurrence is less than 8 tokens back: a loop of period p continues as itself). Ties: the most
recent occurrence. Measured on logged cycles (tests/c/ngram_sim.py, same anchors): tokens per cycle 2.74 -> 3.09
(evaluator prompts), 4.54 -> 4.96 (natural text) with MIN_N = 3.

Everything is device tensors with static shapes (captured in a CUDA graph): the committed tokens are kept in
hist[position] (prompt at prefill; per cycle the 8 target tokens of the verify pass at positions c_old + 1 ..
c_old + 8, of which the first emit[0] are committed and the rest are overwritten before they are read).
"""
from __future__ import annotations

import torch

import os
MIN_N, MAX_N, NB = int(os.environ.get("MK_NGRAM_MIN", "5")), 8, 8


class NGram:
    """lib: the engine's libmk.so (mk_ngram: the same rule as step() below in one CUDA block); without it, torch."""

    def __init__(self, maxctx: int, dev, lib=None):
        self.lib = lib if lib is not None and hasattr(lib, "mk_ngram") else None
        self.T = maxctx + 2 * NB
        self.dev = dev
        self.hist = torch.full((self.T,), -1, dtype=torch.int32, device=dev)
        self.pos = torch.arange(self.T, device=dev)
        self.ar8 = torch.arange(NB, device=dev)
        self.k8 = torch.arange(MAX_N, device=dev)
        self.ar7 = torch.arange(NB - 1, device=dev)
        self.used = torch.zeros(1, dtype=torch.int32, device=dev)     # 1 if the last proposal came from here
        self.graph = None

    def prefill(self, prompt: torch.Tensor, anchor: torch.Tensor):
        """prompt int32 [P] (device), anchor int32 [1] (the target's token after the prompt, at position P)."""
        P = prompt.shape[0]
        self.hist.fill_(-1)
        self.hist[:P].copy_(prompt)
        self.hist[P:P + 1].copy_(anchor)

    def step(self, emit: torch.Tensor, vpos: torch.Tensor, vtok: torch.Tensor, record: bool = True):
        """emit [1 + 8] int32 (n, target tokens), vpos [1] int32 (= c_new), vtok [8] int32 (in/out)."""
        c = vpos[0:1].long()                                                   # [1] anchor position
        if record:
            c_old = c - emit[0:1].long()
            self.hist.index_copy_(0, c_old + 1 + self.ar8, emit[1:])
        T = self.T
        # L[i] = length of the common suffix of hist[..i] and hist[..c] (up to MAX_N), for i < c
        suf = self.hist.index_select(0, (c - self.k8).clamp_min(0))           # [MAX_N]: hist[c - k]
        idx = self.pos[None, :] - self.k8[:, None]                             # [MAX_N, T]: i - k
        vals = self.hist.index_select(0, idx.clamp_min(0).flatten()).view(MAX_N, T)
        eq = (vals == suf[:, None]) & (idx >= 0) & ((c - self.k8)[:, None] >= 0)
        L = eq.int().cumprod(0).sum(0)                                         # [T]
        L = torch.where(self.pos < c, L, torch.zeros_like(L))
        score = L.long() * T + self.pos                                        # longest, then most recent
        best = score.max(0, keepdim=True)
        bi = best.values % T                                                   # [1]
        bl = best.values // T
        p = (c - bi).clamp_min(1)
        cand = self.hist.index_select(0, (bi + 1 + (self.ar7 % p)).clamp(0, T - 1))   # [7] (used only if ok)
        ok = bl >= MIN_N
        self.used.copy_(ok.int())
        vtok[1:].copy_(torch.where(ok, cand, vtok[1:]))
        if vpos.numel() >= 2:   # an n-gram proposal is verified in full (8 rows)
            vpos[1:2].copy_(torch.where(ok, torch.full_like(vpos[1:2], NB), vpos[1:2]))

    def launch(self, emit, vpos, vtok, record: bool = True):
        import ctypes
        r = self.lib.mk_ngram(ctypes.c_void_p(self.hist.data_ptr()), ctypes.c_int(self.T), ctypes.c_void_p(emit.data_ptr()),
                              ctypes.c_void_p(vpos.data_ptr()), ctypes.c_void_p(vtok.data_ptr()), ctypes.c_int(int(record)),
                              ctypes.c_int(MIN_N), ctypes.c_int(NB), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert r == 0, "mk_ngram launch failed"

    def run(self, emit, vpos, vtok, record: bool = True):
        """One n-gram step on the current stream (the CUDA kernel, or the captured torch graphs)."""
        if self.lib is not None:
            self.launch(emit, vpos, vtok, record)
        else:
            (self.graph if record else self.graph_np).replay()

    def capture(self, emit, vpos, vtok):
        if self.lib is not None:
            return
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        save = self.hist.clone(), vtok.clone()
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step(emit, vpos, vtok, record=False)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step(emit, vpos, vtok)
        self.graph_np = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph_np):
            self.step(emit, vpos, vtok, record=False)
        self.hist.copy_(save[0])
        vtok.copy_(save[1])
        torch.cuda.synchronize()
