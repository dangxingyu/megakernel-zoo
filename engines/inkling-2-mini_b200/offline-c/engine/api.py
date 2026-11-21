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
TRACK = [x.upper() for x in re.split(r"[\s_,+]+", os.environ.get("MK_TRACK") or ((HERE / "TRACK").read_text() if (HERE / "TRACK").exists() else "A")) if x]

P = ctypes.c_void_p


class W12M(ctypes.Structure):
    _fields_ = [("data", P), ("base", P), ("hdr", P), ("esc", P)]


USE_W12 = "W12" in TRACK
# track C (draft-and-verify behind submit() / poll()) is chosen by engine/TRACK itself (the run's evaluator sets
# MK_TRACK=c for every submission; a track-A submission of the verifier alone writes A there), or MK_SPEC=1 (tests)
_TRACK_FILE = [x.upper() for x in re.split(r"[\s_,+]+", (HERE / "TRACK").read_text() if (HERE / "TRACK").exists() else "A") if x]
SPEC = ("C" in _TRACK_FILE and os.environ.get("MK_SPEC") != "0") or os.environ.get("MK_SPEC") == "1"
W12_STATS = {"weights": 0, "escapes": 0, "matrices": 0}


def w12_matrix(w: torch.Tensor, keep: list) -> W12M:
    """Exact 12-bit code of a BF16 matrix (engine/w12.py), verified bit for bit against the matrix it encodes."""
    import w12
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
                ("amax", P), ("next_tok", P), ("prof", P), ("fout", P), ("xft", P),
                ("un8", P), ("un8s", P), ("un8m", P), ("zhi", P), ("zlo", P)]


class Hdr(ctypes.Structure):
    _fields_ = [("B", ctypes.c_int), ("pos", ctypes.c_int), ("flags", ctypes.c_int), ("nsteps", ctypes.c_int),
                ("tokens", P), ("logits", P), ("out_tok", P), ("out_stride", ctypes.c_int), ("tag0", ctypes.c_uint),
                ("prompt", P), ("plen", ctypes.c_int), ("tok_tag", ctypes.c_uint), ("vpos", P)]
NB_SPEC = 8   # rows of a verify pass (the draft's block)
RING = int(os.environ.get("MK_RING", "16"))   # short-convolution history slots (mk.cu RING)


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


def int8_unembed(w: torch.Tensor):
    """Verify passes' certified argmax: INT8 copy of the unembedding [N, K] bf16 with one FP32 scale per row
    (s = max|w| / 127, q = round(w / s)), in piece_i8's fragment order ([N/16][K/64][32 lanes][4 tiles][8 B], the
    element order of frag_bf16), and per row the error coefficient (max_k |w - s q| + 6e-4 max|w|) * (1 + 1e-3)
    (rounded up), computed in float64 from the stored weights."""
    n, k = w.shape
    qs, ss, ms = [], [], []
    for r0 in range(0, n, 8192):
        wd = w[r0:r0 + 8192].double()
        amax = wd.abs().amax(1)
        s = (amax / 127.0).float()
        sd = s.double().clamp_min(1e-30)
        q = torch.round(wd / sd[:, None]).clamp(-127, 127)
        err = (wd - sd[:, None] * q).abs().amax(1)
        m = ((err + 6e-4 * amax) * (1 + 1e-3)).float()
        m = torch.nextafter(m, torch.full_like(m, float("inf")))
        qs.append(q.to(torch.int8)); ss.append(s); ms.append(m)
    q = torch.cat(qs)
    x = q.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()       # [N/16][K/16][32][8]
    x = x.view(n // 16, k // 64, 4, 32, 8).permute(0, 1, 3, 2, 4).contiguous().view(torch.uint8).flatten()
    return x, torch.cat(ss).contiguous(), torch.cat(ms).contiguous()


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
        self.lib = ctypes.CDLL(os.environ.get("MK_LIB") or str(HERE / "build" / "libmk.so"))
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
        self.embed_w = ld.get("embed.weight")
        ctx.embed = keep(self.embed_w)
        self.embed_norm_w = ld.get("embed_norm.weight")
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
        self.unembed_rm = un if SPEC else None    # row-major copy for the draft head
        if SPEC and os.environ.get("MK_UN8", "0") == "1":
            q8, s8, m8 = int8_unembed(pad_rows(un, 12504 * 16))
            ctx.un8, ctx.un8s, ctx.un8m = keep(q8), keep(s8), keep(m8)
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
            setattr(ctx, nm, z(NAPP, MAXB, RING, w))
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
        ctx.fout = z(MAXB, H)
        self.fout = k[-1]
        ctx.xft = u64(MAXB, H // 2)
        ctx.zhi = z(MAXB, 12504 * 16, dt=torch.float32)
        ctx.zlo = z(MAXB, dt=torch.int32)
        if os.environ.get('MK_PROF'):
            ctx.prof = z(self.nblk * 4096 * 2, dt=torch.int64)
            self.prof = k[-1]
        self.ctx = ctx
        self.ctx_dev = torch.empty(ctypes.sizeof(Ctx), dtype=torch.uint8, device=self.dev)
        buf = (ctypes.c_uint8 * ctypes.sizeof(Ctx)).from_buffer_copy(ctx)
        self.ctx_dev.copy_(torch.frombuffer(bytearray(buf), dtype=torch.uint8))
        self.ctx_host = buf    # the kernel takes the context by value when the library says so (mk_ctx_param)
        self.ctx_arg = ctypes.addressof(buf) if hasattr(self.lib, "mk_ctx_param") else self.ctx_dev.data_ptr()
        self.draft = None
        if SPEC:
            import sys
            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            import draft as _draft
            self.draft = _draft.Draft(model_dir, self.maxctx, self.embed_w, self.embed_norm_w, self.unembed_rm, self.dev)
            self.unembed_rm = None
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

    def _launch_verify(self, tok_ptr, K, vpos_ptr, logits_ptr=0):
        """One verify pass: rows b < vpos[1] (K if that is not in 1..8) of sequence 0 at positions vpos[0] + b (vpos on
        the device), tokens tok_ptr [K]
        (int32, device); per row argmax -> next_tok[b], final-norm rows -> fout, partials -> pbt (draft features)."""
        h = Hdr()
        h.B, h.pos, h.tokens, h.logits, h.out_tok = K, 0, tok_ptr, logits_ptr, 0
        h.nsteps, h.prompt, h.plen, h.vpos = 0, 0, 0, vpos_ptr
        self.launches += 1
        h.tag0 = (self.launches * 64) & 0x7FFFFFFF
        r = self.lib.mk_launch(ctypes.c_void_p(self.ctx_arg), ctypes.byref(h), self.nblk,
                               ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        assert r == 0, "launch failed"

    FEAT_IDX = (5, 23, 41)   # pbt rows: partial at the start of applications 5, 23, 41

    def _feats(self, K=8):
        """[K, 4, H] bf16 target features of the last verify pass's rows (copies)."""
        p = [self.pbt[i, :K, :, 0].contiguous().view(torch.bfloat16) for i in self.FEAT_IDX]   # 3 x [K, H]
        return torch.stack(p + [self.fout[:K]], 1)

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
            if len(cohort) == 1 and self.draft is not None and Pn + cohort[0][2] + NB_SPEC * (self.SPEC_DEPTH + 2) <= self.maxctx:
                act = self._spec_start(cohort[0])
            else:
                act = self._start_cohort(cohort)
            on["active"] = act
        if act.get("spec"):
            self._spec_pump(act, events)
            if act["done"]:
                on["active"] = None
            return events
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

    # ---- track C: one sequence by draft-and-verify. A cycle = one verify pass of 8 rows (the anchor and 7 drafts) +
    # the draft graph (acceptance, the draft's context K/V, the next 7 drafts), all stream-ordered on the device; the
    # host keeps SPEC_DEPTH cycles queued and reads each cycle's committed tokens from pinned memory after its event.
    SPEC_DEPTH = int(os.environ.get("MK_SPEC_DEPTH", "3"))

    USE_NGRAM = os.environ.get("MK_NGRAM", "1") == "1"

    def _spec_graphs(self):
        if getattr(self.draft, "g_cyc", None) is None:
            if isinstance(self.draft, getattr(__import__("draft"), "FusedDraft", ())):
                self.draft.capture(lambda: self.next_tok[:NB_SPEC], lambda: self._feats(NB_SPEC), pbt=self.pbt, fout=self.fout)
            else:
                self.draft.capture(lambda: self.next_tok[:NB_SPEC], lambda: self._feats(NB_SPEC))
        if self.USE_NGRAM and getattr(self, "ngram", None) is None:
            import ngram as _ngram
            self.ngram = _ngram.NGram(self.maxctx, self.dev, None if os.environ.get("MK_NGRAM_TORCH") == "1" else self.lib)
            self.ngram.capture(self.draft.emit, self.draft.vpos, self.draft.vtok)

    def _spec_start(self, req):
        rid, prompt, n, _ = req
        d = self.draft
        self._spec_graphs()
        self.reset()
        P = len(prompt)
        pt = torch.tensor(prompt, dtype=torch.int32).to(self.dev)
        vp = torch.zeros(2, dtype=torch.int32, device=self.dev)   # [position, rows]
        pos, K = 0, 1
        for pos in range(0, P, NB_SPEC):          # the prompt in verify passes of up to 8 rows (bit-identical to step())
            K = min(NB_SPEC, P - pos)
            vp[0] = pos
            vp[1] = K
            self._launch_verify(pt[pos:pos + K].data_ptr(), K, vp.data_ptr())
            f = self._feats(K)
            d.ctx_append(f, torch.arange(pos, pos + K, device=self.dev))
        d.feats[:K].copy_(f)
        d.c_new.fill_(P)
        d.anchor.copy_(self.next_tok[K - 1:K].long())
        d.frow.fill_(K - 1)
        d.emit[0] = 1
        d.emit[1:2].copy_(self.next_tok[K - 1:K])
        if not hasattr(self, "_spec_host"):
            self._spec_host = torch.zeros(self.SPEC_DEPTH + 1, NB_SPEC + 1, dtype=torch.int32).pin_memory()
        if not hasattr(self, "_spec_stats"):
            self._spec_stats = {"requests": 0, "events": 0, "tokens": 0, "cycles": 0}
        self._spec_stats["requests"] += 1
        act = {"spec": True, "rid": rid, "n": n, "sent": 0, "done": False, "q": [], "k": 0, "keep": (pt, vp)}
        self._spec_record(act)                    # the first token (the prompt's greedy successor)
        if getattr(d, "handles_ngram", False):    # the draft's cycle does the n-gram step itself (draftk.cu k_ngram)
            d.ngram_prefill(pt, self.next_tok[K - 1:K])
        d.g_prop.replay()
        if self.USE_NGRAM and not getattr(d, "handles_ngram", False):
            self.ngram.prefill(pt, self.next_tok[K - 1:K])
            self.ngram.run(self.draft.emit, self.draft.vpos, self.draft.vtok, record=False)
        for _ in range(self.SPEC_DEPTH - 1):
            self._spec_cycle(act)
        return act

    def _spec_record(self, act):
        slot = act["k"] % (self.SPEC_DEPTH + 1)
        act["k"] += 1
        self._spec_host[slot].copy_(self.draft.emit, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        act["q"].append((slot, ev))

    def _spec_cycle(self, act):
        d = self.draft
        self._launch_verify(d.vtok.data_ptr(), NB_SPEC, d.vpos.data_ptr())
        d.g_cyc.replay()
        if self.USE_NGRAM and not getattr(d, "handles_ngram", False):
            self.ngram.run(self.draft.emit, self.draft.vpos, self.draft.vtok)
        self._spec_stats["cycles"] += 1
        self._spec_record(act)

    def _spec_pump(self, act, events):
        q = act["q"]
        while q and q[0][1].query():
            slot, _ = q.pop(0)
            row = self._spec_host[slot].tolist()
            toks = row[1:1 + row[0]]
            need = act["n"] - act["sent"]
            if need <= 0:
                continue
            toks = toks[:need]
            act["sent"] += len(toks)
            self._spec_stats["events"] += 1
            self._spec_stats["tokens"] += len(toks)
            events.append((act["rid"], toks, act["sent"] >= act["n"]))
            # keep SPEC_DEPTH cycles queued while tokens are still needed (each cycle commits at least one)
            if act["sent"] + 8 * len(q) < act["n"]:
                self._spec_cycle(act)
        if act["sent"] >= act["n"]:
            act["done"] = True

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
                "context_limit": min(self.max_context, self.maxctx),
                "spec": dict(getattr(self, "_spec_stats", {})) if self.draft is not None else None}
