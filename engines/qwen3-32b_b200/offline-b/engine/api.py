"""Decode engine for Qwen3-32B (BF16) on one B200: a persistent megakernel (engine/mk.cu), one launch per model
pass (step) or per decoded sequence of passes (generate, serving). Python loads the weights, lays them out in MMA
fragment order and launches the kernel; nothing of the model runs in PyTorch.

Contract (eval/_engine_runner.py): Engine(model_dir, max_batch, max_context), reset(), step(tokens), generate(),
submit()/poll(), stats(), close().
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w12  # noqa: E402

HERE = Path(__file__).resolve().parent
HID, VOCAB, FF, NL, QKV_M, QDIM, HD = 5120, 151936, 25600, 64, 10240, 8192, 128
MAXB = 8
NWARP = 16          # warps per block (mk.cu NWARP; checked at load)
NTHR = 1024
OPW = 16
OP_GEMM, OP_ATTN = 1, 2
EP_QKV, EP_RESID, EP_GATEUP, EP_LOGITS = 1, 2, 3, 4
IN_NORM, IN_ATTN, IN_ACT = 1, 2, 3
F_LOGITS, F_ARGMAX = 1, 2
TRACK = (os.environ.get("MK_TRACK") or ((HERE / "TRACK").read_text().split()[0] if (HERE / "TRACK").exists() else "A")).strip().upper()


class Params(ctypes.Structure):
    _fields_ = [
        ("ops", ctypes.c_void_p), ("nops", ctypes.c_int),
        ("B", ctypes.c_int), ("nsteps", ctypes.c_int), ("flags", ctypes.c_int), ("nblk", ctypes.c_int),
        ("pos0", ctypes.c_int * MAXB), ("nforced", ctypes.c_int * MAXB),
        ("forced", ctypes.c_void_p), ("fstride", ctypes.c_int),
        ("embed", ctypes.c_void_p), ("final_norm", ctypes.c_void_p),
        ("x", ctypes.c_void_p), ("qkv", ctypes.c_void_p), ("attn", ctypes.c_void_p), ("act", ctypes.c_void_p),
        ("logits", ctypes.c_void_p), ("gpart", ctypes.c_void_p), ("tcnt", ctypes.c_void_p), ("gkey", ctypes.c_void_p),
        ("tok_out", ctypes.c_void_p), ("tok_ld", ctypes.c_int),
        ("host_tok", ctypes.c_void_p), ("host_prog", ctypes.c_void_p),
        ("bar", ctypes.c_void_p), ("bar_base", ctypes.c_uint),
        ("rope", ctypes.c_void_p), ("max_ctx", ctypes.c_int),
        ("kc", ctypes.c_void_p), ("vc", ctypes.c_void_p),
        ("pf_bytes", ctypes.c_int), ("pf_idle_bytes", ctypes.c_int),
        ("dcnt", ctypes.c_void_p), ("pf_warp15", ctypes.c_int), ("host_tag", ctypes.c_int), ("big_min_b", ctypes.c_int), ("clus", ctypes.c_int), ("att_flag", ctypes.c_int), ("xf", ctypes.c_void_p), ("normf_min_b", ctypes.c_int), ("ssq", ctypes.c_void_p), ("apart", ctypes.c_void_p), ("aml", ctypes.c_void_p), ("acnt", ctypes.c_void_p),
        ("dbg", ctypes.c_int), ("att_smax", ctypes.c_int), ("pf_att", ctypes.c_int), ("pf_first", ctypes.c_int), ("pf_piece", ctypes.c_int),
        ("prof", ctypes.c_void_p), ("prof2", ctypes.c_void_p),
    ]


def _lib():
    so = Path(os.environ["MK_LIB"]) if os.environ.get("MK_LIB") else HERE / "build" / "libmk.so"
    if not so.exists():
        subprocess.run(["bash", str(HERE / "build.sh")], check=True)
    lib = ctypes.CDLL(str(so))
    lib.mk_launch.argtypes = [ctypes.POINTER(Params), ctypes.c_void_p]
    lib.mk_launch.restype = ctypes.c_int
    lib.mk_params_size.restype = ctypes.c_int
    lib.mk_smem_bytes.argtypes = [ctypes.c_int]
    lib.mk_smem_bytes.restype = ctypes.c_int
    lib.mk_host_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    lib.mk_host_alloc.restype = ctypes.c_int
    assert lib.mk_params_size() == ctypes.sizeof(Params), (lib.mk_params_size(), ctypes.sizeof(Params))
    assert lib.mk_nwarp() == NWARP
    if chr(lib.mk_track()) != TRACK:
        raise RuntimeError(f"library built for track {chr(lib.mk_track())}, engine runs track {TRACK}: rebuild (engine/build.sh with MK_TRACK={TRACK})")
    return lib


def to_frag(W: torch.Tensor) -> torch.Tensor:
    """[M, K] bf16 -> [M/16][K/16][32 lanes][8] in m16n8k16 A-operand order (a0: row g, k 2t..; a1: row g+8;
    a2: row g, k 2t+8..; a3: row g+8, k 2t+8..)."""
    M, K = W.shape
    x = W.view(torch.int16).view(M // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return x.view(-1)


class Weights:
    def __init__(self, model_dir: str):
        from safetensors import safe_open
        idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}
        self.where = idx
        self.dir = Path(model_dir)
        self.open = safe_open

    def get(self, name: str) -> torch.Tensor:
        f = self.where[name]
        if f not in self.files:
            self.files[f] = self.open(str(self.dir / f), framework="pt", device="cuda")
        return self.files[f].get_tensor(name)


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        assert 1 <= max_batch <= MAXB, "max_batch above 8 is not supported"
        self.lib = _lib()
        self.Bmax = max_batch
        self.mc = (max(int(max_context), 16) + 15) // 16 * 16
        smem = self.lib.mk_smem_bytes(self.mc)
        if smem > 227 * 1024:
            raise ValueError(f"max_context {max_context} too large for the attention scratch ({smem} B)")
        self.dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        self.nblk = int(os.environ.get("MK_NBLK", props.multi_processor_count))
        self.keep = []
        self.weight_bytes = 0
        t0 = time.time()
        self._load(model_dir)
        self.load_s = time.time() - t0
        self._buffers()
        self.pos = 0
        self.B = None
        self.bar_base = 0
        self.pf_bytes = int(os.environ.get("MK_PF", "196608"))
        # (track B with its attention cap of 10 leaves more blocks idle during attention: 128 KB each, e0397)
        self.pf_idle = int(os.environ.get("MK_PF_IDLE", "131072" if TRACK == "B" else "262144"))
        self.pf_first = int(os.environ.get("MK_PF1", "65536"))
        self.pf_att = int(os.environ.get("MK_PF_ATT", "0"))
        self.pf_piece = int(os.environ.get("MK_PFP", "65536"))
        # serving
        self.waiting = []
        self.cohort_tag = 0
        self.cohort = None
        self.first_wait = None
        # a cohort starts when max_batch requests wait or 20 ms after the first arrived (only time to first token pays)
        self.gather_s = float(os.environ.get("MK_GATHER_MS", "20")) / 1e3

    # -------------------------------------------------------------------------------------------- loading
    def _load(self, model_dir: str):
        wts = Weights(model_dir)
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        assert cfg["hidden_size"] == HID and cfg["num_hidden_layers"] == NL and cfg["vocab_size"] == VOCAB
        assert cfg["intermediate_size"] == FF and cfg["num_attention_heads"] == 64 and cfg["num_key_value_heads"] == 8
        keep = self.keep

        def K(t):
            keep.append(t)
            return t.data_ptr()

        self.w12 = TRACK == "B"
        self.w12_stats = {"weights": 0, "escapes": 0, "verified_weights": 0, "mismatches": 0, "semantic_mismatches": 0}
        verify = os.environ.get("MK_W12_VERIFY") == "1"

        def frag(W):
            """The matrix in the engine's storage: [ptr, w12 flag, row factors, escape starts, escape entries]."""
            W = W.contiguous()
            M, Kd = W.shape
            t = to_frag(W)
            if not self.w12:
                self.weight_bytes += t.numel() * 2
                return [K(t), 0, 0, 0, 0]
            lo = w12.window_rows(W)
            words, esc = w12.pack(t, M, Kd, lo)
            if verify:   # every weight decoded with the kernel's arithmetic against the checkpoint
                dec = w12.unpack(words, M, Kd, esc)
                ref = w12.scaled_bits(t, M, Kd, lo)
                self.w12_stats["mismatches"] += int((dec != ref).sum())
                NT, KQ = M // 16, Kd // 64
                lane = torch.arange(32, device=W.device).view(32, 1); e = torch.arange(8, device=W.device).view(1, 8)
                lof = lo.view(NT, 16)[:, (lane >> 2) + 8 * ((e >> 1) & 1)].view(NT, 1, 1, 32, 8)
                val = dec.to(torch.int16).view(torch.bfloat16).double() * torch.ldexp(torch.ones_like(lof, dtype=torch.float64), lof - w12.L0)
                self.w12_stats["semantic_mismatches"] += int((val != t.view(NT, KQ, 4, 32, 8).view(torch.bfloat16).double()).sum())
                self.w12_stats["verified_weights"] += W.numel()
                del dec, ref, val, lof
            hdr, ent = w12.escape_lists(esc, M, Kd, self.nblk, NWARP)
            fs = torch.ldexp(torch.ones(M, dtype=torch.float32, device=W.device), (lo - w12.L0).to(torch.int32))
            fs = torch.cat([fs, fs.new_ones(64)]).contiguous()
            self.weight_bytes += words.numel() * 4
            self.w12_stats["weights"] += W.numel()
            self.w12_stats["escapes"] += int(esc[0].numel())
            del t
            return [K(words), 1, K(fs), K(hdr), K(ent)]

        def bfw(name):
            return K(wts.get(name).to(torch.bfloat16).contiguous())

        def gemm_row(ep, NT, KS, f, normw, inmode, layer):
            r = [OP_GEMM, ep, NT, KS, f[0], normw, 0, inmode, layer, 0, 0, 0, f[1], f[2], f[3], f[4]]
            return r

        self.embed = wts.get("model.embed_tokens.weight").contiguous()
        keep.append(self.embed)
        ops = []
        gemm_rows = []   # index of every GEMM op (for the prefetch chain)
        for l in range(NL):
            p = f"model.layers.{l}."
            Wqkv = torch.cat([wts.get(p + "self_attn.q_proj.weight"), wts.get(p + "self_attn.k_proj.weight"),
                              wts.get(p + "self_attn.v_proj.weight")], 0)
            ops.append(gemm_row(EP_QKV, QKV_M // 16, HID // 16, frag(Wqkv), bfw(p + "input_layernorm.weight"), IN_NORM, l))
            del Wqkv
            ops.append([OP_ATTN, 0, 0, 0, 0, bfw(p + "self_attn.q_norm.weight"), bfw(p + "self_attn.k_norm.weight"), 0, l])
            ops.append(gemm_row(EP_RESID, HID // 16, QDIM // 16, frag(wts.get(p + "self_attn.o_proj.weight")), 0, IN_ATTN, l))
            g = wts.get(p + "mlp.gate_proj.weight").view(FF // 8, 8, HID)
            u = wts.get(p + "mlp.up_proj.weight").view(FF // 8, 8, HID)
            Wgu = torch.cat([g, u], 1).view(2 * FF, HID)
            del g, u
            ops.append(gemm_row(EP_GATEUP, 2 * FF // 16, HID // 16, frag(Wgu), bfw(p + "post_attention_layernorm.weight"), IN_NORM, l))
            del Wgu
            ops.append(gemm_row(EP_RESID, HID // 16, FF // 16, frag(wts.get(p + "mlp.down_proj.weight")), 0, IN_ACT, l))
        ops.append(gemm_row(EP_LOGITS, VOCAB // 16, HID // 16, frag(wts.get("lm_head.weight")), bfw("model.norm.weight"), IN_NORM, NL))
        torch.cuda.synchronize()
        wts.files.clear()
        a = np.zeros((len(ops), OPW), dtype=np.int64)
        for i, r in enumerate(ops):
            a[i, :len(r)] = r
        gem = [i for i in range(len(ops)) if a[i, 0] == OP_GEMM]
        for i in range(len(ops)):   # next GEMM (weights to prefetch), wrapping to layer 0's qkv
            nxt = next((j for j in gem if j > i), gem[0])
            w12f = a[nxt, 12]
            a[i, 9], a[i, 10], a[i, 11] = a[nxt, 4], a[nxt, 2] * a[nxt, 3] // 4, 1536 if w12f else 2048   # quads
        self.ops = torch.from_numpy(a).to(self.dev)
        self.nops = len(ops)
        self.weight_bytes += VOCAB * HID * 0   # (embedding not streamed)

    def _buffers(self):
        d = self.dev
        z = lambda *s, dt=torch.uint16: torch.zeros(*s, dtype=dt, device=d)
        self.x = z(MAXB * HID, dt=torch.int16)
        self.qkv = z(MAXB * QKV_M, dt=torch.int16)
        self.attn = z(QDIM * MAXB, dt=torch.int16)
        self.act = z(FF * MAXB, dt=torch.int16)
        self.logits = torch.zeros(MAXB, VOCAB, dtype=torch.float32, device=d)
        ntile = VOCAB // 16
        self.gpart = torch.full((ntile * 32 * 4,), 0x7fbadbad, dtype=torch.int32, device=d).view(torch.float32)   # cut-tile slots, sentinel = empty
        self.tcnt = torch.zeros(ntile, dtype=torch.int32, device=d)
        self.gkey = torch.zeros(self.nblk * MAXB, dtype=torch.int64, device=d)
        self.tok_ld = self.mc + 8
        self.tok_out = torch.zeros(MAXB, self.tok_ld, dtype=torch.int64, device=d)
        self.forced = torch.zeros(MAXB, self.tok_ld, dtype=torch.int64, device=d)
        self.bar = torch.zeros(4, dtype=torch.int32, device=d)
        maxp = self.mc // 64
        self.apart = torch.zeros(64 * maxp * 8 * 128, dtype=torch.float32, device=d)
        self.aml = torch.zeros(64 * maxp * 16, dtype=torch.float32, device=d)
        self.acnt = torch.zeros(64, dtype=torch.int32, device=d)
        self.ssq = torch.zeros(MAXB * 320, dtype=torch.float32, device=d)
        self.dcnt = torch.zeros(16 + 64, dtype=torch.int32, device=d)   # [0] attention, [1..8] kv groups, [16 + unit] combine flags
        self.clus = int(os.environ.get('MK_CLUS', '1')) if self.nblk % 2 == 0 else 0   # clusters of 2 blocks
        # large-batch kernel instances from this batch size on (3-6: per-k-step norm; 7-8: DSMEM combine, fewer
        # attention rounds). Track B: only the 7-8 instance (without the per-k-step norm: B=8 workload 9927 -> 9843 us,
        # e0254; at s0010 that instance had spilled in the 12-bit build)
        self.big_min_b = int(os.environ.get('MK_BIG_MINB', '7' if TRACK == 'B' else '3'))
        self.xf = torch.zeros(MAXB * HID, dtype=torch.int16, device=d)
        # RMSNorm per k-step in the stream from this batch size on (track A: from 3; track B's decode leaves no issue slots)
        self.normf_min_b = int(os.environ.get('MK_NORMF_MINB', '99' if TRACK == 'B' else '3'))
        self.att_flag = int(os.environ.get('MK_ATT_FLAG', '3'))   # 1: o waits on attention counters, 2: attention on q/k/v tile counters
        kvn = NL * MAXB * 8 * self.mc * HD
        self.kc = torch.zeros(kvn, dtype=torch.int16, device=d)
        self.vc = torch.zeros(kvn, dtype=torch.int16, device=d)
        # RoPE table exactly as Qwen3RotaryEmbedding computes it (float32, then bf16)
        inv = 1.0 / (1000000.0 ** (torch.arange(0, HD, 2, dtype=torch.int64, device=d).float() / HD))
        pos = torch.arange(self.mc, device=d).float()
        fr = pos[:, None] * inv[None, :]
        emb = torch.cat([fr, fr], -1)
        self.rope = torch.stack([emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)], 1).contiguous()  # [mc][2][128]
        # host-mapped token output for serving
        hp, dp = ctypes.c_void_p(), ctypes.c_void_p()
        n = MAXB * self.tok_ld
        rc = self.lib.mk_host_alloc(n * 4 + 64, ctypes.byref(hp), ctypes.byref(dp))
        assert rc == 0, rc
        self.host_tok_dev = dp.value
        self.host_tok = np.ctypeslib.as_array((ctypes.c_int * n).from_address(hp.value))
        self.host_prog_dev = dp.value + n * 4

    # -------------------------------------------------------------------------------------------- launch
    def _launch(self, B, nsteps, flags, forced_ptr, fstride, nforced, host=False):
        P = Params()
        P.ops = self.ops.data_ptr(); P.nops = self.nops
        P.B = B; P.nsteps = nsteps; P.flags = flags; P.nblk = self.nblk
        for b in range(MAXB):
            P.pos0[b] = self.pos
            P.nforced[b] = nforced[b] if b < len(nforced) else 0
        P.forced = forced_ptr; P.fstride = fstride
        P.embed = self.embed.data_ptr(); P.final_norm = 0
        P.x = self.x.data_ptr(); P.qkv = self.qkv.data_ptr(); P.attn = self.attn.data_ptr(); P.act = self.act.data_ptr()
        P.logits = self.logits.data_ptr(); P.gpart = self.gpart.data_ptr(); P.tcnt = self.tcnt.data_ptr()
        P.gkey = self.gkey.data_ptr()
        P.tok_out = self.tok_out.data_ptr(); P.tok_ld = self.tok_ld
        P.host_tok = self.host_tok_dev if host else 0
        P.host_tag = self.cohort_tag
        P.host_prog = self.host_prog_dev if host else 0
        P.bar = self.bar.data_ptr(); P.bar_base = self.bar_base & 0xFFFFFFFF
        P.rope = self.rope.data_ptr(); P.max_ctx = self.mc
        P.kc = self.kc.data_ptr(); P.vc = self.vc.data_ptr()
        P.pf_bytes = self.pf_bytes; P.pf_idle_bytes = self.pf_idle
        P.pf_first = min(self.pf_first, self.pf_bytes); P.pf_piece = self.pf_piece
        P.pf_att = self.pf_att
        P.dbg = int(os.environ.get('MK_DBG', '0'))
        # most blocks per attention unit; track B at B=1: 10 (workload -30 us vs 19, e0393; results unchanged: the parts and
        # the combine's split do not depend on it)
        P.att_smax = int(os.environ.get('MK_ATT_SMAX', '10' if TRACK == 'B' else '1000'))
        P.dcnt = self.dcnt.data_ptr(); P.att_flag = self.att_flag; P.clus = self.clus; P.big_min_b = self.big_min_b; P.pf_warp15 = int(os.environ.get('MK_PFW15', '1'))
        P.xf = self.xf.data_ptr(); P.normf_min_b = self.normf_min_b
        self.dcnt.zero_()
        P.ssq = self.ssq.data_ptr(); P.apart = self.apart.data_ptr(); P.aml = self.aml.data_ptr(); P.acnt = self.acnt.data_ptr()
        P.prof = self.prof.data_ptr() if getattr(self, "prof", None) is not None else 0
        P.prof2 = self.prof2.data_ptr() if getattr(self, "prof2", None) is not None else 0
        rc = self.lib.mk_launch(ctypes.byref(P), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        if rc != 0:
            raise RuntimeError(f"mk_launch failed: {rc}")
        nb_step = 1 + self.nops - (NL if self.att_flag & 1 else 0) - (NL if self.att_flag & 2 else 0)   # grid barriers per step
        self.bar_base = (self.bar_base + nsteps * nb_step * self.nblk) & 0xFFFFFFFF
        self.pos += nsteps

    def _check(self, tokens: torch.Tensor, nsteps: int):
        B = int(tokens.shape[0])
        if not 1 <= B <= self.Bmax:
            raise ValueError(f"batch {B} outside 1..{self.Bmax}")
        if self.B is None:
            self.B = B
        elif B != self.B:
            raise ValueError("batch size changed without reset()")
        if self.pos + nsteps > self.mc:
            raise ValueError(f"context full: {self.pos} + {nsteps} > {self.mc}")
        if bool(((tokens < 0) | (tokens >= VOCAB)).any()):
            raise ValueError("token id outside the vocabulary")
        return B

    # -------------------------------------------------------------------------------------------- contract
    def reset(self) -> None:
        self.pos = 0
        self.B = None

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.to(device=self.dev, dtype=torch.int64).contiguous()
        B = self._check(tokens, 1)
        self.forced[:B, 0].copy_(tokens)
        self._launch(B, 1, F_LOGITS, self.forced.data_ptr(), self.tok_ld, [1] * B)
        return self.logits[:B].clone()

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        tokens = tokens.to(device=self.dev, dtype=torch.int64).contiguous()
        B = self._check(tokens, n_steps)
        self.forced[:B, 0].copy_(tokens)
        self._launch(B, n_steps, F_ARGMAX, self.forced.data_ptr(), self.tok_ld, [1] * B)
        return self.tok_out[:B, :n_steps].clone()

    def close(self) -> None:
        torch.cuda.synchronize()

    def stats(self) -> dict:
        return {"implementation": "megakernel", "track": TRACK, "w12": self.w12_stats if self.w12 else None, "weight_bytes": self.weight_bytes, "nblk": self.nblk,
                "load_s": round(self.load_s, 1), "max_context": self.mc}

    # -------------------------------------------------------------------------------------------- serving
    def submit(self, rid, prompt, n) -> None:
        prompt = [int(t) for t in prompt]
        n = int(n)
        if not prompt or any(t < 0 or t >= VOCAB for t in prompt):
            raise ValueError("prompt token outside the vocabulary (or empty)")
        if n < 1 or len(prompt) + n - 1 > self.mc:
            raise ValueError(f"request needs {len(prompt) + n - 1} positions, context is {self.mc}")
        self.waiting.append((rid, prompt, n))
        if self.first_wait is None:
            self.first_wait = time.perf_counter()

    def _start_cohort(self):
        co = self.waiting[:self.Bmax]
        self.waiting = self.waiting[self.Bmax:]
        self.first_wait = time.perf_counter() if self.waiting else None
        B = len(co)
        plen = [len(p) for _, p, _ in co]
        nsteps = max(pl + n - 1 for (_, _, n), pl in zip(co, plen))
        f = torch.zeros(MAXB, self.tok_ld, dtype=torch.int64)
        for b, (_, p, _) in enumerate(co):
            f[b, :len(p)] = torch.tensor(p, dtype=torch.int64)
        self.forced.copy_(f.to(self.dev))
        self.cohort_tag = (self.cohort_tag + 1) & 0xFFF   # a late write of the previous cohort cannot pass for a token of this one
        self.host_tok[:] = -1
        self.reset()
        self.B = B
        self._launch(B, nsteps, F_ARGMAX, self.forced.data_ptr(), self.tok_ld, plen, host=True)
        self.cohort = {"reqs": co, "plen": plen, "sent": [0] * B, "B": B}

    def poll(self):
        out = []
        if self.cohort is not None:
            co = self.cohort
            done = True
            for b, (rid, p, n) in enumerate(co["reqs"]):
                k = co["sent"][b]
                base = b * self.tok_ld + co["plen"][b] - 1
                new = []
                while k < n:
                    v = int(self.host_tok[base + k])
                    if v < 0 or (v >> 18) != self.cohort_tag:
                        break
                    new.append(v & 0x3FFFF)
                    k += 1
                if new:
                    out.append((rid, new, k == n))
                co["sent"][b] = k
                if k < n:
                    done = False
            if done:
                self.cohort = None
        if self.cohort is None and self.waiting:
            if len(self.waiting) >= self.Bmax or time.perf_counter() - self.first_wait >= self.gather_s:
                self._start_cohort()
        return out
