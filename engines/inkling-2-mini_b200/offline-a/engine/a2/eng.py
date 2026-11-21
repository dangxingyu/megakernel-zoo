"""Inkling-2-Mini decode engine for one B200: one persistent cooperative CUDA kernel per decode step (engine/mk.cu).

Weights keep the checkpoint's formats (BF16; NVFP4 codes + E4M3 block scales + per-expert amax / denominator; MXFP8
codes + UE8M0 scales); only their layout changes (MMA fragment order). Expert inputs are not quantized: BF16
activations are carried exactly as f16 hi + lo * 2^-24 with a power-of-two scale, weights are decoded exactly.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import threading
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
H, NLW, NAPP, NE, MAXB = 2304, 24, 48, 128, 8
VOCAB_REAL = 200058
VOCAB_ROWS = 201024
PREFIX = "model.llm."
CORE0, CORE1, NPASS = 6, 18, 3
_TRF = HERE.parent / "TRACK"
TRACK = [x for x in re.split(r"[\s_,+]+", os.environ.get("MK_TRACK") or (_TRF.read_text() if _TRF.exists() else "A")) if x]

P = ctypes.c_void_p


class W12M(ctypes.Structure):
    _fields_ = [("data", P), ("base", P), ("hdr", P), ("esc", P)]


USE_W12 = "W12" in TRACK
W12_STATS = {"weights": 0, "escapes": 0, "matrices": 0}


def w12_matrix(w: torch.Tensor, keep: list) -> W12M:
    """Exact 12-bit code of a BF16 matrix (engine/w12.py), verified bit for bit against the matrix it encodes."""
    import sys; sys.path.insert(0, str(HERE)); import w12
    data, base, hdr, esc, ne = w12.pack(w)
    u = w12.unpack(data, base, hdr, esc, w.shape[0], w.shape[1])
    assert torch.equal(u.view(torch.int16), w.contiguous().view(torch.int16)), "w12 round trip mismatch"
    if esc.numel() == 0:
        esc = torch.zeros(1, dtype=torch.int32, device=w.device)
    m = W12M()
    for name, t in (("data", data), ("base", base), ("hdr", hdr), ("esc", esc)):
        t = t.contiguous(); keep.append(t); setattr(m, name, t.data_ptr())
    W12_STATS["weights"] += w.numel(); W12_STATS["escapes"] += ne; W12_STATS["matrices"] += 1
    return m


class LayerW(ctypes.Structure):
    _fields_ = [("wqkv", P), ("wo", P), ("wr", P), ("rbias", P), ("w13", P), ("w2", P), ("f13", P), ("f2", P),
                ("norm", P * 3), ("mlp_norm", P), ("ksc", P), ("vsc", P), ("attn_sc", P), ("mlp_sc", P),
                ("qscale", P), ("rscale", P), ("relp", P), ("aq", P), ("mq", P), ("cD", P), ("cFG", P),
                ("qkv12", W12M), ("wo12", W12M),
                ("nsk", ctypes.c_int), ("mx", ctypes.c_int), ("glob", ctypes.c_int), ("naq", ctypes.c_int)]


class App(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("layer", "pass_", "slot", "bnd", "nblk")] + [("blk", ctypes.c_int * 4)] + \
               [(n, ctypes.c_int) for n in ("pin", "nmsg")] + [("msg", ctypes.c_int * 2)] + \
               [(n, ctypes.c_int) for n in ("loopl", "out", "pad")]


class Ctx(ctypes.Structure):
    _fields_ = [("L", LayerW * NLW), ("A", App * NAPP),
                ("embed", P), ("embed_norm", P), ("oq", P), ("fnorm", P), ("unembed", P), ("loopq", P), ("un12", W12M),
                ("nfin", ctypes.c_int), ("fin", ctypes.c_int * 4), ("fin_pin", ctypes.c_int), ("maxctx", ctypes.c_int),
                ("pbt", P), ("kc", P), ("vc", P), ("hk", P), ("hv", P), ("ha", P), ("hm", P),
                ("qkvt", P), ("arec", P), ("aoutt", P), ("attst", P), ("rlogt", P), ("act16t", P),
                ("part", P), ("w2p", P), ("cnt", P), ("bar", P),
                ("xgt", P), ("mgt", P), ("ubt", P), ("gvt", P), ("xpt", P), ("xst", P), ("xrt", P), ("sht", P), ("w2ct", P), ("ntt", P),
                ("amax", P), ("next_tok", P), ("prof", P)]


class Hdr(ctypes.Structure):
    _fields_ = [("B", ctypes.c_int), ("pos", ctypes.c_int), ("flags", ctypes.c_int), ("nsteps", ctypes.c_int),
                ("tokens", P), ("logits", P), ("out_tok", P), ("out_stride", ctypes.c_int), ("tag0", ctypes.c_uint),
                ("prompt", P), ("plen", ctypes.c_int), ("tok_tag", ctypes.c_uint)]


# ------------------------------------------------------------------------------------------------ layouts
def frag_bf16(w: torch.Tensor) -> torch.Tensor:
    """[N, K] bf16 -> bytes [N/16][K/64][4][32 lanes][16 B] in m16n8k16 A-fragment order."""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0, (n, k)
    x = w.contiguous().view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)
    return x.contiguous().view(torch.uint8).flatten()


def frag_nvfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K/2] u8 (low nibble first), scales [E, N, K/16] (e4m3 as u8) -> [E][N/16][K/64][576 B]."""
    e, n, kb = codes.shape
    k = kb * 2
    s, kg = n // 16, k // 64
    c = codes.view(e, s, 2, 8, kg, 4, 2, 4).permute(0, 1, 4, 3, 7, 5, 6, 2).reshape(e, s, kg, 512)
    sc = scales.view(torch.uint8).view(e, s, 2, 8, kg, 4).permute(0, 1, 4, 3, 5, 2).reshape(e, s, kg, 64)
    return torch.cat([c, sc], dim=-1).contiguous().flatten()


def frag_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K] (e4m3 as u8), scales [E, N, K/32] u8 -> [E][N/16][K/64][1056 B] for the f16 m16n8k16 path:
    lane (g, t) holds 32 B ordered (j, kh, r, e): element (row 8r + g, k 16j + 8kh + 2t + e); then the scales
    [g][blk][r] (k32 block blk of the unit)."""
    e, n, k = codes.shape
    s, u = n // 16, k // 64
    c = codes.view(torch.uint8).view(e, s, 2, 8, u, 4, 2, 4, 2).permute(0, 1, 4, 3, 7, 5, 6, 2, 8).reshape(e, s, u, 1024)
    sc = scales.view(torch.uint8).view(e, s, 2, 8, u, 2).permute(0, 1, 4, 3, 5, 2).reshape(e, s, u, 32)
    return torch.cat([c, sc], dim=-1).contiguous().flatten()


def pad_rows(w: torch.Tensor, rows: int) -> torch.Tensor:
    if w.shape[0] == rows:
        return w
    return torch.cat([w, w.new_zeros(rows - w.shape[0], *w.shape[1:])])


def applications():
    """(layer, pass, cache slot) of the 48 applications (as eval/inkling2_ref.py Config.applications)."""
    out = [(i, 0, i) for i in range(CORE0)]
    for p in range(NPASS):
        out += [(i, p, i if p == 0 else NLW + (p - 1) * (CORE1 - CORE0) + (i - CORE0)) for i in range(CORE0, CORE1)]
    return out + [(i, 0, i) for i in range(CORE1, NLW)]


def app_table():
    """pbuf index 0 = h0, index a + 1 = the partial after application a."""
    apps = applications()
    idx_of = {}
    for a, (i, p, _) in enumerate(apps):
        idx_of[(i, p)] = a + 1
    P5 = idx_of[(5, 0)]
    rows = []
    for a, (i, p, slot) in enumerate(apps):
        d = dict(layer=i, pass_=p, slot=slot, bnd=int(i % 6 == 0), out=a + 1, nmsg=0, msg=[0, 0], loopl=max(0, i - CORE0))
        d["pin"] = a  # output of the previous application (h0 for a = 0)
        if CORE0 <= i < CORE1 and p >= 1 and i == CORE0:
            d["pin"] = P5
        # blocks before the partial (attention-input mix)
        if i == 0:
            blk = []
        elif i < 6:
            blk = [0]
        elif i < 12:
            blk = [0] if i == 6 else [0, P5]
        elif i < 18:
            P11 = idx_of[(11, p)]
            blk = [0, P5] if i == 12 else [0, P5, P11]
        else:
            P11, P17 = idx_of[(11, NPASS - 1)], idx_of[(17, NPASS - 1)]
            blk = [0, P5, P11] if i == 18 else [0, P5, P11, P17]
        d["blk"] = blk
        if CORE0 <= i < CORE1 and p >= 1:
            d["nmsg"] = 2
            d["msg"] = [idx_of[(11, p - 1)], idx_of[(17, p - 1)]]
        rows.append(d)
    fin = [0, P5, idx_of[(11, NPASS - 1)], idx_of[(17, NPASS - 1)]]
    return rows, fin, NAPP


class Loader:
    def __init__(self, model_dir: str, dev):
        from safetensors import safe_open
        self.dir = Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        files = sorted(set(idx.values()))
        self.where = idx
        self.files = {f: safe_open(str(self.dir / f), framework="pt", device=str(dev)) for f in files}

    def get(self, name):
        n = PREFIX + name
        return self.files[self.where[n]].get_tensor(n)


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int, verbose: bool = True):
        t0 = time.time()
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        assert max_batch <= MAXB, f"max_batch {max_batch} > {MAXB}"
        self.max_batch, self.max_context = max_batch, int(max_context)
        self.maxctx = min(4096, (self.max_context + 63) // 64 * 64)
        self.dev = torch.device("cuda", 0)
        torch.cuda.set_device(0)
        self.lib = ctypes.CDLL(os.environ.get("MK_LIB1") or str(HERE.parent / "build" / "libmk1.so"))
        self.lib.mk_host_alloc.restype = ctypes.c_void_p
        assert self.lib.mk_ctx_bytes() == ctypes.sizeof(Ctx), (self.lib.mk_ctx_bytes(), ctypes.sizeof(Ctx))
        assert self.lib.mk_layer_bytes() == ctypes.sizeof(LayerW)
        assert self.lib.mk_app_bytes() == ctypes.sizeof(App)
        assert self.lib.mk_hdr_bytes() == ctypes.sizeof(Hdr)
        assert self.lib.mk_setup() == 0
        assert self.lib.mk_w12() == int(USE_W12), "the library was built for another weight storage than TRACK says"
        self.nblk = torch.cuda.get_device_properties(0).multi_processor_count
        self.keep = []
        ld = Loader(model_dir, self.dev)
        local = set(self.cfg["local_layer_ids"])
        ctx = Ctx()
        k = self.keep

        def keep(t):
            t = t.contiguous()
            k.append(t)
            return t.data_ptr()

        for i in range(NLW):
            p = f"layers.{i}."
            Lw = LayerW()
            glob = i not in local
            g = lambda n: ld.get(p + n)  # noqa: E731
            qg = pad_rows(torch.cat([g("attn.wq_du.weight"), g("attn.gate_attn.weight"), g("attn.wr_du.weight"),
                                     g("attn.attn_sink_proj.weight")]), 6560)
            if glob:
                kg = torch.cat([g("attn.wk_dv.weight"), g("attn.k_sconv.weight_proj")])
                vg = torch.cat([g("attn.wv_dv.weight"), g("attn.v_sconv.weight_proj")])
            else:
                kg, vg = g("attn.wk_dv.weight"), g("attn.wv_dv.weight")
            Lw.nsk = kg.shape[0] // 16
            wqkv = torch.cat([qg, kg, vg])
            if USE_W12:
                Lw.qkv12 = w12_matrix(wqkv, k)
            else:
                Lw.wqkv = keep(frag_bf16(wqkv))
            del wqkv
            del qg, kg, vg
            if USE_W12:
                Lw.wo12 = w12_matrix(g("attn.wo_ud.weight"), k)
            else:
                Lw.wo = keep(frag_bf16(g("attn.wo_ud.weight")))
            Lw.wr = keep(frag_bf16(g("mlp.gate.weight")))
            Lw.rbias = keep(g("mlp.gate.bias").float())
            w13 = g("mlp.experts.w13_weight")
            Lw.mx = int(w13.dtype == torch.float8_e4m3fn)
            if not Lw.mx:
                Lw.w13 = keep(frag_nvfp4(w13, g("mlp.experts.w13_weight_scale")))
                Lw.w2 = keep(frag_nvfp4(g("mlp.experts.w2_weight"), g("mlp.experts.w2_weight_scale")))
                for nm, attr in (("w13", "f13"), ("w2", "f2")):
                    amax = g(f"mlp.experts.{nm}_weight_amax").double()
                    den = g(f"mlp.experts.{nm}_weight_outer_scale_denominator").double()
                    setattr(Lw, attr, keep((amax / den).float()))
            else:
                Lw.w13 = keep(frag_mxfp8(w13, g("mlp.experts.w13_weight_scale")))
                Lw.w2 = keep(frag_mxfp8(g("mlp.experts.w2_weight"), g("mlp.experts.w2_weight_scale")))
            del w13
            if glob:
                for j, nm in enumerate(("q", "k", "v")):
                    Lw.norm[j] = keep(g(f"{nm}_input_norm.weight"))
                Lw.ksc = keep(g("attn.k_sconv.static_w"))     # [4][1024], row 0 = current token
                Lw.vsc = keep(g("attn.v_sconv.static_w"))
            else:
                n0 = keep(g("attn_norm.weight"))
                Lw.norm[0] = Lw.norm[1] = Lw.norm[2] = n0
                Lw.ksc = keep(g("attn.k_sconv.weight")[:, 0, :])   # [1024][4], tap 3 = current token
                Lw.vsc = keep(g("attn.v_sconv.weight")[:, 0, :])
            Lw.mlp_norm = keep(g("mlp_norm.weight"))
            Lw.attn_sc = keep(g("attn_sconv.weight")[:, 0, :])
            Lw.mlp_sc = keep(g("mlp_sconv.weight")[:, 0, :])
            Lw.qscale = keep(g("attn.q_l2_clip.scale"))
            Lw.rscale = keep(g("attn.rel_l2_clip.scale"))
            Lw.relp = keep(g("attn.rel_logits_proj.proj"))       # [16][1024]
            if i > 0:
                aq = ld.get(f"attn_res.attn_queries.{i - 1}").float()
                Lw.naq = 1 if aq.dim() == 1 else aq.shape[0]
                Lw.aq = keep(aq.reshape(-1))
            Lw.mq = keep(ld.get(f"attn_res.mlp_queries.{i}").float())
            Lw.glob = int(glob)
            if CORE0 <= i < CORE1:
                names = ("attn", "k", "v", "mlp")
                D = torch.cat([g(f"controllers.{n}.down_weight") for n in names])        # [256, 4608]
                Lw.cD = keep(frag_bf16(D))
                FG = torch.stack([torch.stack([g(f"controllers.{n}.forget_weight"), g(f"controllers.{n}.gain_weight")]) for n in names])
                Lw.cFG = keep(FG)                                                         # [4][2][2304][64]
            ctx.L[i] = Lw
        rows, fin, _ = app_table()
        for a, d in enumerate(rows):
            A = App()
            A.layer, A.pass_, A.slot, A.bnd, A.nblk = d["layer"], d["pass_"], d["slot"], d["bnd"], len(d["blk"])
            for j, v in enumerate(d["blk"]):
                A.blk[j] = v
            A.pin, A.nmsg, A.loopl, A.out = d["pin"], d["nmsg"], d["loopl"], d["out"]
            A.msg[0], A.msg[1] = d["msg"]
            ctx.A[a] = A
        ctx.nfin = len(fin)
        for j, v in enumerate(fin):
            ctx.fin[j] = v
        ctx.fin_pin = NAPP
        ctx.embed = keep(ld.get("embed.weight"))
        ctx.embed_norm = keep(ld.get("embed_norm.weight"))
        ctx.oq = keep(ld.get("attn_res.output_query").float())
        ctx.fnorm = keep(ld.get("norm.weight"))
        qm = ld.get("middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
        kv = ld.get("middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
        ctx.loopq = keep(torch.stack([qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]], dim=1))   # [12][4][H]
        un = ld.get("unembed.weight")[:VOCAB_REAL]
        if USE_W12:
            ctx.un12 = w12_matrix(pad_rows(un, 12504 * 16), k)
        else:
            ctx.unembed = keep(frag_bf16(pad_rows(un, 12504 * 16)))
        del un
        ld = None
        # runtime buffers
        z = lambda *shape, dt=torch.bfloat16: keep(torch.zeros(*shape, dtype=dt, device=self.dev))  # noqa: E731
        ctx.maxctx = self.maxctx
        ctx.pbt = z(NAPP + 1, MAXB, H // 2, 2, dt=torch.int32)
        self.pbt = k[-1]
        ctx.kc = z(NAPP, MAXB, 8, self.maxctx, 128)
        ctx.vc = z(NAPP, MAXB, 8, self.maxctx, 128)
        self.hist = []
        for nm, w in (("hk", 1024), ("hv", 1024), ("ha", H), ("hm", H)):
            setattr(ctx, nm, z(NAPP, MAXB, 3, w))
            self.hist.append(k[-1])
        u64 = lambda *shape: z(*shape, 2, dt=torch.int32)  # noqa: E731  (value, tag) pairs
        ctx.qkvt = u64(2, MAXB, 8672)
        ctx.arec = u64(MAXB * 24 * 64 * 130)
        ctx.aoutt = u64(2, MAXB, 1536)
        ctx.attst = u64(2, MAXB, H // 2)
        ctx.rlogt = u64(2, MAXB, NE)
        ctx.act16t = u64(2, 64, MAXB, 2, 576)
        ctx.part = z(64 * 144 * 2 * 6 * 32 * 4, dt=torch.float32)
        ctx.w2p = z(MAXB * 8 * 3 * 144 * 16, dt=torch.float32)
        ctx.cnt = z(self.lib.mk_cn_total(), dt=torch.int32)
        ctx.bar = z(4, dt=torch.int32)
        self.ctx_bar = k[-1]   # [2]: NaN-in-router flags (debug)
        ctx.xgt = u64(2, MAXB, 4, H)
        ctx.mgt = u64(2, MAXB, 4, H)
        ctx.ubt = u64(2, MAXB, 4, H)
        ctx.gvt = u64(2, MAXB, 4, 64)
        ctx.xpt = u64(2, MAXB, 4, H)
        ctx.xst = u64(2, 3, MAXB, H // 2)
        ctx.xrt = u64(2, MAXB, H // 2)
        ctx.sht = u64(2, MAXB)
        ctx.w2ct = u64(2, MAXB, 8, H)
        ctx.ntt = u64(MAXB)
        ctx.amax = z(MAXB, dt=torch.int64)
        ctx.next_tok = z(MAXB, dt=torch.int32)
        self.next_tok = k[-1]
        if os.environ.get('MK_PROF'):
            ctx.prof = z(self.nblk * 4096 * 2, dt=torch.int64)
            self.prof = k[-1]
        self.next_tok = k[-1]
        self.ctx = ctx
        self.ctx_dev = torch.empty(ctypes.sizeof(Ctx), dtype=torch.uint8, device=self.dev)
        buf = (ctypes.c_uint8 * ctypes.sizeof(Ctx)).from_buffer_copy(ctx)
        self.ctx_dev.copy_(torch.frombuffer(bytearray(buf), dtype=torch.uint8))
        self.ctx_host = buf    # the kernel takes the context by value when the library says so (mk_ctx_param)
        self.ctx_arg = ctypes.addressof(buf) if hasattr(self.lib, "mk_ctx_param") else self.ctx_dev.data_ptr()
        torch.cuda.synchronize()
        self.stream = torch.cuda.current_stream()
        self.pos = 0
        self.B = None
        self._keep_toks = []
        self._on = None
        self.launches = 0
        if verbose:
            print(f"MK_ENGINE ready in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 1e9:.1f} GB; "
                  f"smem {self.lib.mk_smem_bytes()} B", flush=True)

    # -------------------------------------------------------------------------------------------- driving
    def reset(self) -> None:
        torch.cuda.synchronize()
        for t in self.hist:
            t.zero_()
        torch.cuda.synchronize()
        self.pos = 0
        self.B = None
        self._keep_toks = []

    def _launch(self, tok_ptr, B, logits_ptr=0, out_tok=0, nsteps=0, prompt_ptr=0, plen=0):
        n = max(1, nsteps)
        if not 0 <= self.pos or self.pos + n > self.maxctx:
            raise RuntimeError(f"MK_ENGINE context limit: positions {self.pos}..{self.pos + n - 1}, limit {self.maxctx}")
        h = Hdr()
        h.B, h.pos, h.tokens, h.logits, h.out_tok = B, self.pos, tok_ptr, logits_ptr, out_tok
        h.nsteps, h.prompt, h.plen = nsteps, prompt_ptr, plen
        self.launches += 1
        h.tag0 = (self.launches * 64) & 0x7FFFFFFF
        self.launches += n - 1
        r = self.lib.mk_launch(ctypes.c_void_p(self.ctx_arg), ctypes.byref(h), self.nblk,
                               ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert r == 0, "launch failed"
        self.pos += n

    def _check(self, tokens: torch.Tensor):
        B = tokens.shape[0]
        if not 1 <= B <= self.max_batch or (self.B is not None and B != self.B):
            raise ValueError(f"MK_ENGINE batch {B} (fixed {self.B} since the last reset, max {self.max_batch})")
        if self.B is None:
            self.B = B
        return B

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = self._check(tokens)
        if self.pos + 1 > self.max_context or self.pos + 1 > self.maxctx:
            raise ValueError(f"MK_ENGINE context limit {min(self.max_context, self.maxctx)} reached")
        if not tokens.is_cuda and (int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB_ROWS):   # host tensors: free check
            raise ValueError("MK_ENGINE token id out of range")
        tok = tokens.to(device=self.dev, dtype=torch.int32).contiguous()   # device ids: clamped in the kernel (no sync)
        logits = torch.empty(B, VOCAB_REAL, dtype=torch.float32, device=self.dev)
        self._keep_toks.append(tok)
        if len(self._keep_toks) > 256:
            self._keep_toks = self._keep_toks[-128:]
        self._launch(tok.data_ptr(), B, logits.data_ptr())
        return logits

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = self._check(tokens)
        if self.pos + n_steps > min(self.max_context, self.maxctx):
            raise ValueError("MK_ENGINE context limit")
        if not tokens.is_cuda and (int(tokens.min()) < 0 or int(tokens.max()) >= VOCAB_ROWS):
            raise ValueError("MK_ENGINE token id out of range")
        tok = tokens.to(device=self.dev, dtype=torch.int32).contiguous().view(1, B)   # clamped in the kernel
        out = torch.zeros(n_steps, MAXB, dtype=torch.int32, device=self.dev)
        self._keep_toks.append(tok)
        self._launch(0, B, 0, out.data_ptr(), nsteps=n_steps, prompt_ptr=tok.data_ptr(), plen=1)
        return (out[:, :B] - 1).t().long()

    # -------------------------------------------------------------------------------------------- serving
    GATHER_S = 0.05

    def _online_init(self):
        if self._on is not None:
            return
        n = (self.max_context + 8) * MAXB
        ptr = self.lib.mk_host_alloc(ctypes.c_size_t(n * 4))
        assert ptr
        import numpy as np
        self._gen_host_ptr = ptr
        self._gen_np = np.ctypeslib.as_array((ctypes.c_int32 * n).from_address(ptr))
        self._on = {"pending": [], "active": None}

    def submit(self, rid, prompt, n) -> None:
        self._online_init()
        prompt, n = list(prompt), int(n)
        lim = min(self.max_context, self.maxctx)
        if len(prompt) < 1 or n < 1 or len(prompt) + n - 1 > lim:
            raise ValueError(f"MK_ENGINE request {rid!r}: prompt {len(prompt)} + {n} new tokens exceeds {lim} positions")
        if min(prompt) < 0 or max(prompt) >= VOCAB_ROWS:
            raise ValueError(f"MK_ENGINE request {rid!r}: token id out of range")
        self._on["pending"].append((rid, prompt, n, time.perf_counter()))

    def poll(self):
        self._online_init()
        on = self._on
        events = []
        act = on["active"]
        if act is None:
            if not on["pending"]:
                return events
            first = on["pending"][0]
            Pn = len(first[1])
            same = [r for r in on["pending"] if len(r[1]) == Pn]
            if len(same) < self.max_batch and time.perf_counter() - first[3] < self.GATHER_S:
                return events
            cohort = same[:self.max_batch]
            on["pending"] = [r for r in on["pending"] if all(r is not q for q in cohort)]
            act = self._start_cohort(cohort)
            on["active"] = act
        self._pump(act)
        g = self._gen_np
        B = act["B"]
        while act["read"] < act["nmax"]:
            kk = act["read"]
            row = g[kk * MAXB:kk * MAXB + B]
            if (row == 0).any():
                break
            for i, (rid, _, n, _) in enumerate(act["reqs"]):
                if kk < n:
                    events.append((rid, [int(row[i]) - 1], kk == n - 1))
            act["read"] += 1
        if act["read"] >= act["nmax"]:
            on["active"] = None
        return events

    def _start_cohort(self, cohort):
        """One launch for the whole cohort: the prompt fed one token per step, then greedy decoding with the chosen
        token fed back on the device; every step's tokens land in host-mapped memory (1 + id, 0 = not yet)."""
        B = len(cohort)
        Pn = len(cohort[0][1])
        self.reset()
        self.B = B
        self._gen_np[:] = 0
        prompt = torch.tensor([r[1] for r in cohort], dtype=torch.int32).contiguous().t().contiguous().to(self.dev)  # [P, B]
        nmax = max(r[2] for r in cohort)
        act = {"B": B, "P": Pn, "reqs": cohort, "nmax": nmax, "read": 0, "prompt": prompt}
        self._launch(0, B, 0, self._gen_host_ptr, nsteps=Pn + nmax - 1, prompt_ptr=prompt.data_ptr(), plen=Pn)
        return act

    def _pump(self, act):
        pass

    @property
    def pbuf(self):
        """[49][MAXB][H] bf16 view of the tagged partials (debugging)."""
        v = self.pbt[..., 0].contiguous()
        return v.view(torch.bfloat16).view(NAPP + 1, MAXB, H)

    def close(self) -> None:
        torch.cuda.synchronize()

    def stats(self) -> dict:
        return {"implementation": "persistent-cooperative-kernel", "track": " ".join(TRACK), "nblk": self.nblk,
                "w12": dict(W12_STATS) if USE_W12 else None,
                "context_limit": min(self.max_context, self.maxctx)}
