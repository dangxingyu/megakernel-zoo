"""Flint decode engine: one persistent CUDA kernel per step (engine/mk.cu). This file only loads the checkpoint,
re-lays the weights into the kernel's tile order (a permutation of the BF16 values; no value changes) and launches.

    Engine(model_dir, max_batch, max_context); reset(); step(tokens) -> logits [B, vocab] float32; generate(tokens, n)
"""

import ctypes
import json
import os
import struct
import subprocess
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HID, NLAYER, VOCAB, NEXP, MAXB = 2048, 49, 150144, 256, 8


def pack_tiles(w: torch.Tensor) -> torch.Tensor:
    """int16 [M, K] (BF16 bits) -> tiles [M/16][K/16][32 lanes][4 regs][2 halves] in mma A-operand order."""
    M, K = w.shape
    return w.reshape(M // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous().view(-1)


def interleave(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """[I, K] gate and up -> [2I, K] where every 16-row group holds 8 gate rows followed by the same 8 up rows."""
    I, K = gate.shape
    return torch.stack([gate.reshape(I // 8, 8, K), up.reshape(I // 8, 8, K)], dim=1).reshape(2 * I, K)


class Checkpoint:
    """The safetensors shards, moved to the GPU one at a time; tensors are int16 views of the raw bytes."""

    def __init__(self, model_dir: str):
        self.dir = model_dir
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.files = sorted(set(index["weight_map"].values()))
        self.where = {}
        self.data_start = []
        for si, name in enumerate(self.files):
            with open(os.path.join(model_dir, name), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            self.data_start.append(8 + n)
            for key, e in header.items():
                if key == "__metadata__":
                    continue
                assert e["dtype"] == "BF16", (key, e["dtype"])
                self.where[key] = (si, e["data_offsets"][0], e["data_offsets"][1], e["shape"])
        self.raw = {}

    def shard_of(self, key: str) -> int:
        return self.where[key][0]

    def load_shard(self, si: int) -> None:
        if si in self.raw:
            return
        path = os.path.join(self.dir, self.files[si])
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=self.data_start[si])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.raw[si] = torch.from_numpy(mm).cuda()
        del mm

    def get(self, key: str) -> torch.Tensor:
        si, a, b, shape = self.where[key]
        t = self.raw[si][a:b]
        if a % 2:
            t = t.clone()
        return t.view(torch.int16).view(shape)

    def f32(self, key: str) -> torch.Tensor:
        return self.get(key).contiguous().view(torch.bfloat16).float().contiguous()


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        assert max_batch <= MAXB, "this engine is built for batches of at most 8"
        t0 = time.time()
        lib_path = os.path.join(HERE, "build", "libmk_prof.so" if os.environ.get("MK_PROF") else "libmk.so")
        if not os.path.exists(lib_path):
            subprocess.run(["bash", os.path.join(HERE, "build.sh")], check=True)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.mk_create.restype = ctypes.c_void_p
        self.lib.mk_create.argtypes = [ctypes.c_void_p]
        self.lib.mk_run.restype = ctypes.c_int
        self.lib.mk_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.mk_poll.restype = ctypes.c_int
        self.lib.mk_poll.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mk_read_dbg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mk_check.restype = ctypes.c_int
        self.lib.mk_check.argtypes = [ctypes.c_void_p]
        torch.cuda.init()
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        assert cfg["hidden_size"] == HID and cfg["num_hidden_layers"] == NLAYER and cfg["vocab_size"] == VOCAB
        # The kernel's attention keeps at most 128 key groups of 16 positions per warp set (shared-memory weights area,
        # 8 groups per warp): contexts above 2048 positions are refused here instead of being computed wrongly.
        self.max_context = min(int(max_context), 2048)
        self.kgmax = (self.max_context + 15) // 16
        self.keep = []
        slots = np.zeros(NLAYER * 20 + 8, dtype=np.int64)
        self.track = open(os.path.join(HERE, "TRACK")).read().strip() if os.path.exists(os.path.join(HERE, "TRACK")) else "A"
        self.track = os.environ.get("MK_TRACK", self.track)
        # early routing (track B only, experimental): engine/EARLY_ROUTE or MK_EARLY=1; the kernel is built with -DEARLY_ROUTE then
        self.early = self.track == "B" and (os.path.exists(os.path.join(HERE, "EARLY_ROUTE")) or bool(os.environ.get("MK_EARLY")))
        # speculative routing (both tracks): engine/SPEC_ROUTE or MK_SPEC=1; the kernel is built with -DSPEC_ROUTE then
        self.spec = (not self.early) and (os.path.exists(os.path.join(HERE, "SPEC_ROUTE")) or bool(os.environ.get("MK_SPEC")))
        ck = Checkpoint(model_dir)

        def keep(t: torch.Tensor) -> int:
            self.keep.append(t)
            return t.data_ptr()

        def layer_keys(l: int) -> list:
            p = f"model.layers.{l}."
            return [k for k in ck.where if k.startswith(p)]

        items = {l: layer_keys(l) for l in range(NLAYER)}
        items["g"] = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        last = {i: max(ck.shard_of(k) for k in ks) for i, ks in items.items()}
        need = {si: {i for i, ks in items.items() if any(ck.shard_of(k) == si for k in ks)} for si in range(len(ck.files))}
        done = set()
        cache_bytes = max_batch * 4 * self.kgmax * 8 * 512
        for si in range(len(ck.files)):
            ck.load_shard(si)
            for i in [i for i in items if i not in done and last[i] <= si]:
                if i == "g":
                    base = NLAYER * 20
                    slots[base + 0] = keep(ck.get("model.embed_tokens.weight").contiguous().clone())
                    slots[base + 1] = keep(pack_tiles(ck.get("lm_head.weight")))
                    slots[base + 2] = keep(ck.f32("model.norm.weight"))
                else:
                    self._pack_layer(ck, i, cfg, slots, keep, cache_bytes)
                done.add(i)
            for sj in list(ck.raw):
                if need[sj] <= done:
                    del ck.raw[sj]
        assert len(done) == len(items)
        # rotary tables, exactly as the Hugging Face module computes them (float32 on the device, cast to BF16)
        inv_freq = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
        freqs = torch.arange(self.max_context, device="cuda")[:, None].float() * inv_freq.cuda()[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        base = NLAYER * 20
        if self.track == "B":     # float32 tables (the Hugging Face BF16 path rounds them to BF16)
            slots[base + 3] = keep(emb.cos().contiguous())
            slots[base + 4] = keep(emb.sin().contiguous())
        else:
            slots[base + 3] = keep(emb.cos().to(torch.bfloat16).float().contiguous())
            slots[base + 4] = keep(emb.sin().to(torch.bfloat16).float().contiguous())
        slots[base + 5] = self.max_context
        self.debug = bool(os.environ.get("MK_DEBUG"))
        slots[base + 6] = 1 if self.debug else 0
        torch.cuda.synchronize()
        self.h = self.lib.mk_create(slots.ctypes.data_as(ctypes.c_void_p))
        if not self.h:
            raise RuntimeError("mk_create failed")
        self.watchdog = 0
        self.max_batch = int(max_batch)
        if os.environ.get("MK_FENCE"):
            self.lib.mk_set_fence.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.mk_set_fence(self.h, 1)
        self._pending = []
        self._cohort = None
        self.pos = 0
        self.steps = 0
        self._tok = None
        self.ctr_pages = {}
        self.ctr_cal_us = {}
        self._place_counters()
        self.load_seconds = time.time() - t0

    def _pack_layer(self, ck, l, cfg, slots, keep, cache_bytes):
        p = f"model.layers.{l}."
        s = l * 20
        a = p + "self_attn."
        slots[s + 0] = keep(pack_tiles(torch.cat([ck.get(a + "q_proj.weight"), ck.get(a + "k_proj.weight"), ck.get(a + "v_proj.weight")], 0)))
        slots[s + 1] = keep(pack_tiles(ck.get(a + "o_proj.weight")))
        moe = l not in cfg["mlp_only_layers"]
        m = p + "mlp."
        if not moe:
            slots[s + 2] = keep(pack_tiles(interleave(ck.get(m + "gate_proj.weight"), ck.get(m + "up_proj.weight"))))
            slots[s + 3] = keep(pack_tiles(ck.get(m + "down_proj.weight")))
        else:
            slots[s + 4] = keep(pack_tiles(ck.get(m + "gate.weight")))
            if self.early or self.spec:
                # Early routing: the router's top 6 are those of z = R' x + M a with R' = R diag(w2) and M = R' W_o
                # (x: the layer's input, a: the attention output before o_proj). Both matrices are built here in float64
                # and stored as BF16 limbs: R' (products of two BF16 numbers) is exact in 2 limbs, M is carried to 4
                # limbs = 32 bits, more than the float32 accumulation of the router itself.
                R = ck.get(m + "gate.weight").contiguous().view(torch.bfloat16).double()
                w2 = ck.f32(p + "post_attention_layernorm.weight").double()
                Wo = ck.get(a + "o_proj.weight").contiguous().view(torch.bfloat16).double()
                Rp = R * w2[None, :]
                M = Rp @ Wo
                def limbs(x, n):
                    out, r = [], x.clone()
                    for _ in range(n):
                        h = r.to(torch.bfloat16)
                        out.append(h); r = r - h.double()
                    return out, r
                rl_, rr = limbs(Rp, 2)
                assert float(rr.abs().max()) == 0.0, "R' is not exact in two BF16 limbs"
                ml_, mr = limbs(M, 1 if self.spec else 4)       # (the predictor of speculative routing: one limb)
                self.fused_resid = max(getattr(self, "fused_resid", 0.0), float((mr.abs() / M.abs().clamp_min(1e-30)).max()))
                slots[s + 18] = keep(torch.cat([pack_tiles(x.contiguous().view(torch.int16)) for x in rl_]))
                slots[s + 19] = keep(torch.cat([pack_tiles(x.contiguous().view(torch.int16)) for x in ml_]))
                if os.environ.get("MK_FUSED_HASH"):     # debugging: are the fused matrices the same in every process?
                    hs = getattr(self, "fused_hash", 0)
                    for x in ml_ + rl_:
                        v = x.contiguous().view(torch.int16).to(torch.int64)
                        hs = (hs * 1000003 + int((v * (torch.arange(v.numel(), device=v.device).view(v.shape) % 65521 + 1)).sum().item())) % (1 << 61)
                    self.fused_hash = hs
                del R, Wo, Rp, M, rl_, ml_, rr, mr
            slots[s + 5] = keep(pack_tiles(interleave(ck.get(m + "shared_expert.gate_proj.weight"), ck.get(m + "shared_expert.up_proj.weight"))))
            slots[s + 6] = keep(pack_tiles(ck.get(m + "shared_expert.down_proj.weight")))
            gate = torch.stack([ck.get(f"{m}experts.{e}.gate_proj.weight") for e in range(NEXP)])
            up = torch.stack([ck.get(f"{m}experts.{e}.up_proj.weight") for e in range(NEXP)])
            gu = torch.stack([gate.view(NEXP, 64, 8, HID), up.view(NEXP, 64, 8, HID)], dim=2).view(NEXP * 1024, HID)
            del gate, up
            slots[s + 7] = keep(pack_tiles(gu))
            del gu
            down = torch.stack([ck.get(f"{m}experts.{e}.down_proj.weight") for e in range(NEXP)]).view(NEXP * HID, 512)
            slots[s + 8] = keep(pack_tiles(down))
            del down
        slots[s + 9] = keep(torch.zeros(cache_bytes // 2, dtype=torch.int16, device="cuda"))
        slots[s + 10] = keep(torch.zeros(cache_bytes // 2, dtype=torch.int16, device="cuda"))
        slots[s + 11] = keep(ck.f32(p + "input_layernorm.weight"))
        slots[s + 12] = keep(ck.f32(p + "post_attention_layernorm.weight"))
        slots[s + 13] = keep(ck.f32(a + "q_norm.weight"))
        slots[s + 14] = keep(ck.f32(a + "k_norm.weight"))
        slots[s + 15] = keep(ck.f32(a + "softmax_offset"))
        slots[s + 16] = 1 if cfg["layer_types"][l] == "sliding_attention" else 0
        slots[s + 17] = 1 if moe else 0

    def _place_counters(self) -> None:
        # Where a small buffer that every block polls lies in memory changes the step time: at batch 1 the counter
        # array has two levels 14 us apart (the parity of a few address bits of its page decides), the router logits,
        # the post-attention stream and the attention output 5-8 us each; at batch 8 the counters have several levels
        # over 40 us. Which page is good depends on the physical placement of this process's buffers
        # (notes/findings.md, 13g). So the engine times itself when it starts: a few greedy steps of fixed tokens with
        # the buffer on each candidate page, the fastest page is kept. Nothing of the result is kept but page numbers.
        if os.environ.get("MK_NOCAL") or os.environ.get("MK_PROF"):
            return
        lib = self.lib
        lib.mk_set_off.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int][:4]
        names = ("counters", "router_logits", "post_attention_stream", "attention_output")
        # (kernel instance, batch size, buffer, candidate pages, steps per timing)
        plan = [(0, 1, 0, 16, 12), (0, 1, 1, 16, 24), (0, 1, 2, 16, 24), (0, 1, 3, 16, 24), (1, min(self.max_batch, 8), 0, 16, 6)]
        # (batch 8, 64 counter pages instead of 16: 20 us better by this timing and in generate(), 25 us worse over
        #  HTTP on both tracks, s0051/s0052: the serve path's best page is not the one a generate() timing finds among
        #  many; the 16 pages of s0049/s0050 are kept, at the addresses they had there)
        if os.environ.get("MK_CAL_SMALL"):
            plan = [plan[0], (1, min(self.max_batch, 8), 0, 16, 6)]
        for c, B, which, npages, n in plan:
            if B < (3 if c else 1) or n >= self.max_context:
                continue
            tok = torch.arange(B, device="cuda", dtype=torch.int64) * 977 + 1000
            res = []
            for page in range(npages):
                if lib.mk_set_off(self.h, c, which, page):
                    raise RuntimeError("mk_set_off failed")
                best = None
                for rep in range(4):
                    self.pos = 0
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    self.generate(tok, n)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t1
                    if rep and (best is None or dt < best):
                        best = dt
                res.append(best / n * 1e6)
            page = min(range(npages), key=lambda i: res[i])
            if lib.mk_set_off(self.h, c, which, page):
                raise RuntimeError("mk_set_off failed")
            key = ("b1_" if c == 0 else "b8_") + names[which]
            self.ctr_pages[key] = page
            self.ctr_cal_us[key] = [round(r, 1) for r in res]
        if self.lib.mk_check(self.h):
            raise RuntimeError("a wait inside the kernel gave up during start-up")
        self.pos = 0
        self.steps = 0
        if os.environ.get("MK_ADDR"):
            print("MK_CAL pages %s" % (self.ctr_pages,), flush=True)
            for k, v in self.ctr_cal_us.items():
                print("MK_CAL %s: min %.1f max %.1f us: %s" % (k, min(v), max(v), v), flush=True)

    def _check(self) -> None:
        # Every wait inside the kernel has a watchdog (a few seconds of spinning); if one fired, what the kernel returned
        # since is not valid. Checked where the host waits for the device anyway, never inside a decode step.
        if self.lib.mk_check(self.h):
            self.watchdog += 1
            raise RuntimeError("a wait inside the kernel gave up (watchdog): the results of the last run are not valid")

    def reset(self) -> None:
        if self.pos:
            self._check()
        self.pos = 0

    def _check_batch(self, B: int) -> None:
        # the K/V caches and every per-sequence buffer are sized for max_batch sequences
        if B < 1 or B > self.max_batch:
            raise ValueError("batch size %d: this engine was created for 1..%d sequences" % (B, self.max_batch))

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos >= self.max_context:
            raise RuntimeError("context is full")
        logits = torch.empty((B, VOCAB), dtype=torch.float32, device="cuda")
        self._tok = tokens
        if self.lib.mk_run(self.h, tokens.data_ptr(), tokens.stride(0), logits.data_ptr(), None, B, self.pos, 0, 1, 0):
            raise RuntimeError("kernel launch failed")
        self.pos += 1
        self.steps += 1
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos + n_steps > self.max_context:
            raise RuntimeError("context is full")
        out = torch.empty((B, n_steps), dtype=torch.int64, device="cuda")
        self._tok = tokens
        if self.lib.mk_run(self.h, tokens.data_ptr(), tokens.stride(0), None, out.data_ptr(), B, self.pos, 0, n_steps, 1):
            raise RuntimeError("kernel launch failed")
        self.pos += n_steps
        self.steps += n_steps
        return out

    # ---- online contract: requests are gathered into a cohort (same prompt length and length of answer); the whole
    # cohort (prompt steps without the head, then the greedy steps) is ONE kernel launch; poll() copies the tokens
    # the kernel has finished so far through a second stream while it runs.
    GATHER_S = 0.05

    def submit(self, rid, prompt, n) -> None:
        # A request that cannot be served is refused here, loudly (it used to come back from poll() as finished with
        # no tokens): prompt + answer must fit the engine's context (max_context, at most 2048 positions).
        prompt = [int(t) for t in prompt]
        n = int(n)
        if len(prompt) < 1 or n < 1:
            raise ValueError("submit: a request needs at least one prompt token and one new token")
        if len(prompt) - 1 + n > self.max_context:
            raise ValueError("submit: %d prompt tokens + %d new tokens do not fit the context of %d positions"
                             % (len(prompt), n, self.max_context))
        self._pending.append((rid, prompt, n, time.perf_counter()))

    def poll(self) -> list:
        events = []
        if self._cohort is None:
            if not self._pending:
                return events
            if len(self._pending) < self.max_batch and time.perf_counter() - self._pending[-1][3] < self.GATHER_S:
                return events
            first = self._pending[0]
            P, n = len(first[1]), first[2]
            if P < 1 or P - 1 + n > self.max_context:      # (submit() refuses these; never silently "finished")
                self._pending.pop(0)
                raise RuntimeError("request %r does not fit the context" % (first[0],))
            cohort = [r for r in self._pending if len(r[1]) == P and r[2] == n][:self.max_batch]
            self._pending = [r for r in self._pending if all(r is not c for c in cohort)]
            ids = torch.tensor([r[1] for r in cohort], dtype=torch.int64, device="cuda")
            B = len(cohort)
            if self.lib.mk_run(self.h, ids.data_ptr(), P, None, None, B, 0, P - 1, n, 2):
                raise RuntimeError("kernel launch failed")
            self.pos = P - 1 + n
            self.steps += P - 1 + n
            self._cohort = {"rids": [r[0] for r in cohort], "n": n, "sent": 0, "ids": ids, "buf": np.zeros((B, n), dtype=np.int32)}
            return events
        c = self._cohort
        done = self.lib.mk_poll(self.h, c["buf"].ctypes.data_as(ctypes.c_void_p))
        if done < 0:
            self._cohort = None
            if done == -2:
                self.lib.mk_check(self.h); self.watchdog += 1
                raise RuntimeError("a wait inside the kernel gave up (watchdog): this cohort's tokens are not valid")
            raise RuntimeError("poll failed")
        if done > c["sent"]:
            for i, rid in enumerate(c["rids"]):
                events.append((rid, c["buf"][i, c["sent"]:done].tolist(), done == c["n"]))
            c["sent"] = done
            if done == c["n"]:
                self._cohort = None
        return events

    def read_debug(self) -> np.ndarray:
        torch.cuda.synchronize()
        out = np.zeros((NLAYER + 1, MAXB, HID), dtype=np.float32)
        self.lib.mk_read_dbg(self.h, out.ctypes.data_as(ctypes.c_void_p))
        return out

    def close(self) -> None:
        torch.cuda.synchronize()
        self.keep = []

    def stats(self) -> dict:
        return {"implementation": "flint megakernel (one persistent kernel per step or cohort, tensor-core tile GEMV)", "track": self.track,
                "steps": self.steps, "load_seconds": round(self.load_seconds, 1), "watchdog_events": self.watchdog,
                "placed_pages": dict(self.ctr_pages), "placement_us": dict(self.ctr_cal_us)}
