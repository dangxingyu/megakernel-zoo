"""Inkling-2-Mini decode engine for one B200 (track A): two persistent-kernel engines behind one contract.

- batch 1: a2/ (s15-mini-b200-a2 s0025: tagged-dataflow persistent kernel, 152 x 768 threads)
- batch >= 2: a1/ (s15-mini-b200-a1 s0024: persistent kernel specialised by batch size, TMA weight rings)

Both keep the checkpoint's formats (BF16; NVFP4 codes + E4M3 block scales + per-expert amax / denominator; MXFP8
codes + UE8M0 scales) and unquantized BF16 expert inputs; only layout and kernel organisation differ. Each holds its
own copy of the weights in its own layout. The batch size is fixed between resets, so one engine serves a cohort.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# track C (exact draft-and-verify for single-sequence cohorts behind submit() / poll()) is chosen by engine/TRACK
# itself (the run's evaluator sets MK_TRACK=c for every submission; a track-A submission writes A there); MK_SPEC=0
# turns it off for tests. step() and generate() are the track-A engines in every case.
_TRF = [x.upper() for x in (HERE / "TRACK").read_text().replace(",", " ").split()] if (HERE / "TRACK").exists() else ["A"]
SPEC = "C" in _TRF and os.environ.get("MK_SPEC") != "0"


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Engine:
    GATHER_S = 0.05

    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        t0 = time.time()
        # both kernels keep K/V for at most 4096 positions (a1: 64 chunks of 64; a2: maxctx cap); a1 refuses a larger
        # max_context, so both get the effective limit, which submit() / step() / generate() enforce
        self.max_batch, self.max_context = max_batch, min(int(max_context), 4096)
        max_context = self.max_context
        self.e8 = None
        want8 = max_batch >= 2 and os.environ.get("MK_B8", "a1") == "a1"
        first8 = os.environ.get("MK_ORDER", "18") == "81"    # which engine allocates first (placement)

        def load8():
            m8 = _load_mod("mk_a1eng", HERE / "a1" / "eng.py")
            self.e8 = m8.Engine(model_dir, max_batch, max_context)
            self.e8.gather_s = 0.0
        if want8 and first8:
            load8()
        m1 = _load_mod("mk_a2eng", HERE / "a2" / "eng.py")
        self.e1 = m1.Engine(model_dir, max_batch, max_context, verbose=False)
        self.e1.GATHER_S = 0.0          # the cohort is formed here
        if want8 and not first8:
            load8()
        self.ec = None                  # track C: the speculative engine (engine/c: verify kernel + DSpark draft)
        if SPEC:
            mc = _load_mod("mk_ceng", HERE / "c" / "eng.py")
            self.ec = mc.Engine(model_dir, max_batch, max_context, verbose=False)
            self.ec.GATHER_S = 0.0
        self.cur = None                 # engine of the current batch (fixed between resets)
        self.pending = []
        self.active = None              # (engine, {rid: finished?})
        print(f"MK_ENGINE ready in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    def _pick(self, B):
        return self.e1 if B == 1 or self.e8 is None else self.e8

    # ---------------------------------------------------------------------------------------- contract
    def reset(self) -> None:
        for e in (self.e1, self.e8, self.ec):
            if e is not None:
                e.reset()
        self.cur = None

    def _sel(self, B):
        e = self._pick(B)
        if self.cur is None:
            self.cur = e
        elif self.cur is not e:
            raise ValueError(f"MK_ENGINE batch {B} differs from the batch fixed since the last reset")
        return e

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._sel(int(tokens.shape[0])).step(tokens)

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        return self._sel(int(tokens.shape[0])).generate(tokens, n_steps)

    # ---------------------------------------------------------------------------------------- serving
    def submit(self, rid, prompt, n) -> None:
        prompt, n = list(prompt), int(n)
        lim = self.max_context
        if len(prompt) < 1 or n < 1 or len(prompt) + n - 1 > lim:
            raise ValueError(f"MK_ENGINE request {rid!r}: prompt {len(prompt)} + {n} new tokens exceeds {lim} positions")
        if min(prompt) < 0 or max(prompt) >= 201024:
            raise ValueError(f"MK_ENGINE request {rid!r}: token id out of range")
        self.pending.append((rid, prompt, n, time.perf_counter()))

    def poll(self):
        events = []
        if self.active is None:
            if not self.pending:
                return events
            first = self.pending[0]
            Pn = len(first[1])
            same = [r for r in self.pending if len(r[1]) == Pn]
            if len(same) < self.max_batch and time.perf_counter() - first[3] < self.GATHER_S:
                return events
            cohort = same[:self.max_batch]
            self.pending = [r for r in self.pending if all(r is not q for q in cohort)]
            e = self._pick(len(cohort))
            if len(cohort) == 1 and self._spec_ok(cohort[0]):
                e = self.ec
            self.cur = None
            for rid, prompt, n, _ in cohort:
                e.submit(rid, prompt, n)
            self.active = (e, {r[0]: False for r in cohort})
        e, left = self.active
        for ev in e.poll():
            events.append(ev)
            if ev[2]:
                left[ev[0]] = True
        if all(left.values()):
            e.poll()                    # lets the sub-engine retire its cohort
            self.active = None
            self.cur = None
        return events

    def _spec_ok(self, req):
        ec = self.ec
        if ec is None or ec.draft is None:
            return False
        return len(req[1]) + req[2] + getattr(ec, "vmax", 8) * (ec.SPEC_DEPTH + 2) <= ec.maxctx

    def close(self) -> None:
        for e in (self.e1, self.e8, self.ec):
            if e is not None:
                e.close()

    def stats(self) -> dict:
        s = {"implementation": "two persistent kernels: a2 s0025 (B=1), a1 s0024 (B>=2)" +
             ("; track C: B=1 requests by draft-and-verify (engine/c)" if self.ec is not None else ""),
             "track": "C" if self.ec is not None else "A"}
        if self.ec is not None:
            s["spec"] = self.ec.stats().get("spec")
        s["b1"] = self.e1.stats()
        if self.e8 is not None:
            s["b8"] = self.e8.stats()
        return s
