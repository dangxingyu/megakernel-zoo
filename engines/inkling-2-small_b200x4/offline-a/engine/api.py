"""Inkling-2-Small decode engine for one GB200 node (4 x B200).

One persistent CUDA kernel per card per step (engine/ink.cu). Attention is split over the cards by heads (12 q heads,
2 KV heads per card), every expert is split over the cards by its intermediate dimension (576 of 2304 per card), the
router, the attention residuals, the short convolutions and the norms are replicated. Weights keep the checkpoint's
formats (BF16, NVFP4 codes + E4M3 scales, MXFP8 codes + UE8M0 scales); only their layout changes (MMA fragment order).
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
H, NL, NE, ILOC, NCARD, MAXB = 4608, 48, 384, 576, 4, 8
VOCAB_REAL = 200058
PREFIX = "model.llm."
VOCAB_ROWS = 201024  # rows of the embedding and of the unembedding (200058 of them are real tokens)
MAXSLOT = 48  # allocation bound (>= the kernel's MAXSLOT)


# ------------------------------------------------------------------------------------------------ ctypes mirror
P = ctypes.c_void_p


class W12M(ctypes.Structure):
    _fields_ = [("data", P), ("base", P), ("hdr", P), ("esc", P)]


TRACK = [x for x in re.split(r"[\s_,+]+", os.environ.get("MK_TRACK") or ((HERE / "TRACK").read_text() if (HERE / "TRACK").exists() else "A")) if x]
USE_W12 = "W12" in TRACK                       # build switch (-DW12): every dense matrix in the 12-bit code
USE_W12U = USE_W12 or "W12U" in TRACK          # load switch: only the unembedding in the exact 12-bit code
USE_L0T = "L0T" in TRACK                       # load switch: layer-0 token tables (Engine._build_l0_tables)
TC_MODE = 0


def w12_matrix(w: torch.Tensor, keep: list, verify: bool) -> W12M:
    import w12
    data, base, hdr, esc, nesc = w12.pack(w)
    if verify:
        u = w12.unpack(data, base, hdr, esc, w.shape[0], w.shape[1])
        assert torch.equal(u.view(torch.int16), w.view(torch.int16)), "w12 roundtrip mismatch"
    if esc.numel() == 0:
        esc = torch.zeros(1, dtype=torch.int32, device=w.device)
    m = W12M()
    for name, t in (("data", data), ("base", base), ("hdr", hdr), ("esc", esc)):
        t = t.contiguous(); keep.append(t); setattr(m, name, t.data_ptr())
    W12_STATS["values"] += w.numel(); W12_STATS["escapes"] += nesc; W12_STATS["matrices"] += 1
    return m


W12_STATS = {"values": 0, "escapes": 0, "matrices": 0}


class LayerW(ctypes.Structure):
    _fields_ = [("wqkv", P), ("wo", P), ("wr", P), ("rbias", P), ("w13", P), ("w2", P), ("f13", P), ("f2", P), ("w13t", P), ("w2t", P),
                ("norm", P * 3), ("mlp_norm", P), ("ksc", P), ("vsc", P), ("attn_sc", P), ("mlp_sc", P),
                ("qscale", P), ("rscale", P), ("relp", P), ("aq", P), ("mq", P),
                ("qkv12", W12M), ("wo12", W12M), ("wr12", W12M),
                ("glob", ctypes.c_int), ("mx", ctypes.c_int), ("qkv_strips", ctypes.c_int), ("qkv_skip", ctypes.c_int)]


class Hdr(ctypes.Structure):  # per-step header at the start of Ctx (copied every launch)
    _fields_ = [("card", ctypes.c_int), ("B", ctypes.c_int), ("nblk", ctypes.c_int), ("pos", ctypes.c_int),
                ("parity", ctypes.c_int), ("want_logits", ctypes.c_int), ("gen_idx", ctypes.c_int), ("gen_n", ctypes.c_int),
                ("xbase", ctypes.c_uint * 4), ("tag0", ctypes.c_uint), ("pad1", ctypes.c_uint),
                ("logits", P), ("tokens", P), ("gen_out", P)]


class Ctx(ctypes.Structure):
    _fields_ = [("h", Hdr), ("L", LayerW * NL),
                ("embed", P), ("embed_norm", P), ("oq", P), ("fnorm", P), ("unembed", P),
                ("un_strips", ctypes.c_int), ("un_row0", ctypes.c_int), ("un12", W12M),
                ("blocks", P), ("partial", P), ("attn_s", P), ("hist_attn", P), ("hist_mlp", P), ("hist_k", P),
                ("hist_v", P), ("kc", P), ("vc", P), ("maxctx", ctypes.c_int), ("pad0", ctypes.c_int),
                ("xfrag", P), ("x2frag", P), ("w13frag", P), ("w13sc", P), ("qkvo", P), ("rlog", P), ("ids", P),
                ("wts", P), ("uni", P), ("nuni", P), ("act", P), ("w2frag", P), ("w2sc", P), ("achunk", P),
                ("acnt", P), ("ofrag", P), ("hfrag", P), ("part", P), ("scnt", P), ("next_tok", P), ("bar", P),
                ("comm", P * NCARD), ("tprof", P), ("dbg", P), ("dbg_ids", P),
                ("amax_loc", P), ("ucnt", P), ("rll", P), ("tred", P), ("gll", P), ("ctag", P), ("prog", P),
                ("l0tab", P), ("l0emb", P)]


# ------------------------------------------------------------------------------------------------ layouts
def frag_bf16(w: torch.Tensor) -> torch.Tensor:
    """[N, K] bf16 -> bytes [N/16][K/64][4][32 lanes][16 B] in m16n8k16 A-fragment order."""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0, (n, k)
    x = w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)
    return x.contiguous().view(torch.uint8).flatten()


def frag_nvfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K/2] u8 (low nibble first), scales [E, N, K/16] (e4m3 as u8) -> [E][N/16][K/64][576 B]."""
    e, n, kb = codes.shape
    k = kb * 2
    s, kg = n // 16, k // 64
    c = codes.view(e, s, 2, 8, kg, 4, 2, 4).permute(0, 1, 4, 3, 7, 5, 6, 2).reshape(e, s, kg, 512)
    sc = scales.view(torch.uint8).view(e, s, 2, 8, kg, 4).permute(0, 1, 4, 3, 5, 2).reshape(e, s, kg, 64)
    return torch.cat([c, sc], dim=-1).contiguous().flatten()


def frag_nvfp4_tc(codes: torch.Tensor, scales: torch.Tensor, pairs: bool) -> torch.Tensor:
    """The same bytes for the tcgen05 block-scaled MMA: codes [E, N, K/2] u8, scales [E, N, K/16] u8 ->
    [E][N/128][K/64][4608 B]: 128 rows x 64 k of codes K-major without swizzle (row m, k: byte (m//8)*256 +
    ((k%64)//32)*128 + (m%8)*16 + (k%32)//2) and the scale-factor atom (row m, block j: 4096 + (m%32)*16 + (m//32)*4 + j).
    pairs: the N rows are (gate, up) pairs; a row group becomes 64 gate rows then the 64 up rows of the same pairs."""
    e, n, kb = codes.shape
    k = kb * 2
    rg, ks = n // 128, k // 64
    sc = scales.view(torch.uint8)
    if pairs:
        codes = codes.view(e, rg, 64, 2, kb).transpose(2, 3).reshape(e, n, kb)
        sc = sc.view(e, rg, 64, 2, k // 16).transpose(2, 3).reshape(e, n, k // 16)
    c = codes.view(e, rg, 16, 8, ks, 2, 16).permute(0, 1, 4, 2, 5, 3, 6).reshape(e, rg, ks, 4096)
    sf = sc.view(e, rg, 4, 32, ks, 4).permute(0, 1, 4, 3, 2, 5).reshape(e, rg, ks, 512)
    return torch.cat([c, sf], dim=-1).contiguous().flatten()


def frag_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K] (e4m3 as u8), scales [E, N, K/32] u8 -> [E][N/16][K/64][1056 B]."""
    e, n, k = codes.shape
    s, u = n // 16, k // 64
    c = codes.view(torch.uint8).view(e, s, 2, 8, u, 2, 2, 4, 4).permute(0, 1, 4, 5, 3, 7, 6, 2, 8).reshape(e, s, u, 1024)
    sc = scales.view(torch.uint8).view(e, s, 2, 8, u, 2).permute(0, 1, 4, 3, 5, 2).reshape(e, s, u, 32)
    return torch.cat([c, sc], dim=-1).contiguous().flatten()


def pad_rows(w: torch.Tensor, rows: int) -> torch.Tensor:
    if w.shape[0] == rows:
        return w
    return torch.cat([w, w.new_zeros(rows - w.shape[0], *w.shape[1:])])


# ------------------------------------------------------------------------------------------------ loading
class Loader:
    def __init__(self, model_dir: str):
        from safetensors import safe_open
        self.dir = Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = {k: v for k, v in idx.items() if k.startswith(PREFIX)}
        self._open = safe_open
        self.local = threading.local()

    def _f(self, file):
        cache = getattr(self.local, "files", None)
        if cache is None:
            cache = self.local.files = {}
        if file not in cache:
            cache[file] = self._open(str(self.dir / file), framework="pt")
        return cache[file]

    def get(self, name, sl=None):
        f = self._f(self.where[PREFIX + name])
        if sl is None:
            return f.get_tensor(PREFIX + name)
        return f.get_slice(PREFIX + name)[sl]


def load_card(ld: Loader, cfg: dict, c: int, layers: range, dev: torch.device, keep: list):
    """Build every weight of card c on dev. Returns dict of tensors (kept alive) and the per-layer LayerW values."""
    g = lambda n, sl=None: ld.get(n, sl)  # noqa: E731
    cu = lambda t: t.to(dev, non_blocking=False)  # noqa: E731
    local_ids = set(cfg["local_layer_ids"])
    out = {}
    lw = []
    for i in range(NL):
        Lw = LayerW()
        if i not in layers:
            lw.append(Lw)
            continue
        p = f"layers.{i}."
        glob = i not in local_ids
        hq = slice(1536 * c, 1536 * (c + 1))
        wq = cu(g(p + "attn.wq_du.weight", (hq,)))
        wg = cu(g(p + "attn.gate_attn.weight", (hq,)))
        wr = cu(g(p + "attn.wr_du.weight", (slice(192 * c, 192 * (c + 1)),)))
        ws = cu(g(p + "attn.attn_sink_proj.weight", (slice(12 * c, 12 * (c + 1)),)))
        grpA = pad_rows(torch.cat([wq, wg, wr, ws]), 3280)
        kvs = slice(256 * c, 256 * (c + 1))
        wk = cu(g(p + "attn.wk_dv.weight", (kvs,)))
        wv = cu(g(p + "attn.wv_dv.weight", (kvs,)))
        if glob:
            dk = cu(g(p + "attn.k_sconv.weight_proj", (slice(8 * c, 8 * c + 8),)))
            dv = cu(g(p + "attn.v_sconv.weight_proj", (slice(8 * c, 8 * c + 8),)))
            grpB, grpC = pad_rows(torch.cat([wk, dk]), 272), pad_rows(torch.cat([wv, dv]), 272)
        else:
            grpB, grpC = wk, wv
        wqkv = torch.cat([grpA, grpB, grpC])
        Lw.qkv_strips = wqkv.shape[0] // 16
        verify = bool(os.environ.get("MK_W12_VERIFY"))
        if USE_W12:
            Lw.qkv12 = w12_matrix(wqkv, keep, verify)
        else:
            t = frag_bf16(wqkv); keep.append(t); Lw.wqkv = t.data_ptr()
        del wq, wg, wr, ws, wk, wv, grpA, grpB, grpC, wqkv
        wo = cu(g(p + "attn.wo_ud.weight", (slice(None), hq)))
        if USE_W12:
            Lw.wo12 = w12_matrix(wo.contiguous(), keep, verify)
            Lw.wr12 = w12_matrix(cu(g(p + "mlp.gate.weight")), keep, verify)
        else:
            t = frag_bf16(wo.contiguous()); keep.append(t); Lw.wo = t.data_ptr()
            t = frag_bf16(cu(g(p + "mlp.gate.weight"))); keep.append(t); Lw.wr = t.data_ptr()
        t = cu(g(p + "mlp.gate.bias")).float().contiguous(); keep.append(t); Lw.rbias = t.data_ptr()
        # experts
        w13 = g(p + "mlp.experts.w13_weight", (slice(None), slice(1152 * c, 1152 * (c + 1))))
        Lw.mx = int(w13.dtype == torch.float8_e4m3fn)
        if not Lw.mx:
            s13 = g(p + "mlp.experts.w13_weight_scale", (slice(None), slice(1152 * c, 1152 * (c + 1))))
            w13, s13 = cu(w13), cu(s13)
            t = frag_nvfp4(w13, s13); keep.append(t); Lw.w13 = t.data_ptr()
            if TC_MODE & 1:   # the same bytes a second time in the layout of the tensor-core path (larger batches)
                t = frag_nvfp4_tc(w13, s13, True); keep.append(t); Lw.w13t = t.data_ptr()
            del w13, s13
            w2 = g(p + "mlp.experts.w2_weight", (slice(None), slice(None), slice(288 * c, 288 * (c + 1))))
            s2 = g(p + "mlp.experts.w2_weight_scale", (slice(None), slice(None), slice(36 * c, 36 * (c + 1))))
            w2, s2 = cu(w2.contiguous()), cu(s2.contiguous())
            t = frag_nvfp4(w2, s2); keep.append(t); Lw.w2 = t.data_ptr()
            if TC_MODE & 2:
                t = frag_nvfp4_tc(w2, s2, False); keep.append(t); Lw.w2t = t.data_ptr()
            del w2, s2
            for nm, attr in (("w13", "f13"), ("w2", "f2")):
                amax = g(p + f"mlp.experts.{nm}_weight_amax").double()
                den = g(p + f"mlp.experts.{nm}_weight_outer_scale_denominator").double()
                t = cu((amax / den).float().contiguous()); keep.append(t); setattr(Lw, attr, t.data_ptr())
        else:
            s13 = g(p + "mlp.experts.w13_weight_scale", (slice(None), slice(1152 * c, 1152 * (c + 1))))
            t = frag_mxfp8(cu(w13), cu(s13)); keep.append(t); Lw.w13 = t.data_ptr()
            del w13, s13
            w2 = g(p + "mlp.experts.w2_weight", (slice(None), slice(None), slice(576 * c, 576 * (c + 1))))
            s2 = g(p + "mlp.experts.w2_weight_scale", (slice(None), slice(None), slice(18 * c, 18 * (c + 1))))
            t = frag_mxfp8(cu(w2.contiguous()), cu(s2.contiguous())); keep.append(t); Lw.w2 = t.data_ptr()
            del w2, s2
        # small weights
        def small(name, sl=None, dtype=None):
            t = cu(g(p + name, sl))
            if dtype is not None:
                t = t.to(dtype)
            t = t.contiguous(); keep.append(t); return t.data_ptr()
        if glob:
            Lw.norm[0] = small("q_input_norm.weight"); Lw.norm[1] = small("k_input_norm.weight"); Lw.norm[2] = small("v_input_norm.weight")
            t = cu(g(p + "attn.k_sconv.static_w"))[:, 256 * c:256 * (c + 1)].contiguous(); keep.append(t); Lw.ksc = t.data_ptr()
            t = cu(g(p + "attn.v_sconv.static_w"))[:, 256 * c:256 * (c + 1)].contiguous(); keep.append(t); Lw.vsc = t.data_ptr()
        else:
            n0 = small("attn_norm.weight")
            Lw.norm[0] = Lw.norm[1] = Lw.norm[2] = n0
            t = cu(g(p + "attn.k_sconv.weight"))[256 * c:256 * (c + 1), 0, :].contiguous(); keep.append(t); Lw.ksc = t.data_ptr()
            t = cu(g(p + "attn.v_sconv.weight"))[256 * c:256 * (c + 1), 0, :].contiguous(); keep.append(t); Lw.vsc = t.data_ptr()
        Lw.mlp_norm = small("mlp_norm.weight")
        t = cu(g(p + "attn_sconv.weight"))[:, 0, :].contiguous(); keep.append(t); Lw.attn_sc = t.data_ptr()
        t = cu(g(p + "mlp_sconv.weight"))[:, 0, :].contiguous(); keep.append(t); Lw.mlp_sc = t.data_ptr()
        t = cu(g(p + "attn.q_l2_clip.scale"))[12 * c:12 * (c + 1)].contiguous(); keep.append(t); Lw.qscale = t.data_ptr()
        t = cu(g(p + "attn.rel_l2_clip.scale"))[12 * c:12 * (c + 1)].contiguous(); keep.append(t); Lw.rscale = t.data_ptr()
        t = cu(g(p + "attn.rel_logits_proj.proj")).t().contiguous(); keep.append(t); Lw.relp = t.data_ptr()   # [1024][16]
        if i > 0:
            t = cu(ld.get(f"attn_res.attn_queries.{i - 1}")).float().contiguous(); keep.append(t); Lw.aq = t.data_ptr()
        t = cu(ld.get(f"attn_res.mlp_queries.{i}")).float().contiguous(); keep.append(t); Lw.mq = t.data_ptr()
        Lw.glob = int(glob)
        lw.append(Lw)
    return lw


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int, layers: int = NL, verbose: bool = True,
                 debug: bool = False):
        t0 = time.time()
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        self.ndev = torch.cuda.device_count()
        assert self.ndev >= NCARD, "needs 4 cards"
        self.max_batch, self.max_context = max_batch, max_context
        self.red_ok = False   # B=1 redundant-block kernel: slower than the team phases since s0002
        self.nlayers = layers
        self.lib = ctypes.CDLL(os.environ.get("MK_LIB") or str(HERE / "build" / "libink.so"))
        assert self.lib.ink_ctx_bytes() == ctypes.sizeof(Ctx), (self.lib.ink_ctx_bytes(), ctypes.sizeof(Ctx))
        assert self.lib.ink_layer_bytes() == ctypes.sizeof(LayerW)
        assert self.lib.ink_un_w12() in (2, int(USE_W12U)), "the library was built for another unembedding storage than TRACK says"
        global TC_MODE
        TC_MODE = self.lib.ink_tc_mode()   # NVFP4 expert matrices in the tensor-core layout: 1 w13, 2 w2
        self.tc_minb = int(os.environ.get("MK_TC_MINB") or self.lib.ink_tc_minb())   # batches from here on: the tensor-core kernel
        self.devs = [torch.device("cuda", c) for c in range(NCARD)]
        for c in range(NCARD):
            torch.zeros(1, device=f"cuda:{c}")
        assert self.lib.ink_enable_peer(NCARD) == 0
        torch.cuda.set_device(0)
        for c in range(NCARD):
            assert self.lib.ink_setup(c) == 0
        self.nblk = torch.cuda.get_device_properties(0).multi_processor_count
        ld = Loader(model_dir)
        self.keep = [[] for _ in range(NCARD)]
        self.layerw = [None] * NCARD
        errs = []

        def work(c):
            try:
                with torch.cuda.device(c):
                    self.layerw[c] = load_card(ld, self.cfg, c, range(layers), self.devs[c], self.keep[c])
                    torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                errs.append(e)
        ths = [threading.Thread(target=work, args=(c,)) for c in range(NCARD)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        if errs:
            raise errs[0]
        if verbose:
            print(f"MK_ENGINE layers loaded in {time.time() - t0:.0f}s", flush=True)
        # globals and buffers
        self.ctx_host = []
        self.ctx_dev = []
        self.bufs = []
        self.comm = []
        comm_bytes = self.lib.ink_comm_bytes()
        for c in range(NCARD):
            with torch.cuda.device(c):
                cm = torch.zeros(comm_bytes, dtype=torch.uint8, device=self.devs[c])
                self.comm.append(cm)
        embed = ld.get("embed.weight")
        for c in range(NCARD):
            dev = self.devs[c]
            with torch.cuda.device(c):
                k = self.keep[c]
                ctx = Ctx()
                for i in range(NL):
                    ctx.L[i] = self.layerw[c][i]
                t = embed.to(dev); k.append(t); ctx.embed = t.data_ptr()
                t = ld.get("embed_norm.weight").to(dev); k.append(t); ctx.embed_norm = t.data_ptr()
                t = ld.get("attn_res.output_query").float().to(dev); k.append(t); ctx.oq = t.data_ptr()
                t = ld.get("norm.weight").to(dev); k.append(t); ctx.fnorm = t.data_ptr()
                z = lambda *shape, dt=torch.bfloat16: self._buf(k, dev, shape, dt)  # noqa: E731
                ctx.blocks = z(9, MAXB, H)
                ctx.partial = z(MAXB, H)
                ctx.attn_s = z(MAXB, H)
                ctx.hist_attn = z(2, NL, MAXB, 3, H)
                ctx.hist_mlp = z(2, NL, MAXB, 3, H)
                ctx.hist_k = z(NL, MAXB, 3, 256)
                ctx.hist_v = z(NL, MAXB, 3, 256)
                ctx.kc = z(NL, MAXB, 2, max_context, 128)
                ctx.vc = z(NL, MAXB, 2, max_context, 128)
                ctx.maxctx = max_context
                ctx.xfrag = z(3, 72 * 32 * 16)
                ctx.x2frag = z(72 * 32 * 16)
                ctx.w13frag = z(72 * 32 * 32, dt=torch.uint8)
                ctx.w13sc = z(144 * MAXB, dt=torch.float32)
                ctx.qkvo = z(MAXB, 3840)
                ctx.rlog = z(MAXB, NE, dt=torch.float32)
                ctx.ids = z(MAXB, 8, dt=torch.int32)
                ctx.wts = z(MAXB, 8, dt=torch.float32)
                ctx.uni = z(64, dt=torch.int32)
                ctx.nuni = z(4, dt=torch.int32)
                ctx.act = z(64, MAXB, ILOC, 4, dt=torch.int32)   # (gate, tag, up, tag) words of the routed pairs
                ctx.w2frag = z(64 * 9 * 32 * 32, dt=torch.uint8)
                ctx.w2sc = z(64 * 18 * MAXB, dt=torch.float32)
                ctx.achunk = z(MAXB * 2 * 17 * 6 * 130, dt=torch.float32)
                ctx.acnt = z(MAXB * 2, dt=torch.int32)
                ctx.ofrag = z(24 * 32 * 16)
                ctx.hfrag = z(72 * 32 * 16)
                ctx.part = z(64 * 72 * MAXSLOT * 128, dt=torch.float32)
                ctx.scnt = z(64 * 72 + 4096, dt=torch.int32)
                ctx.next_tok = z(MAXB, dt=torch.int32)
                ctx.bar = z(4, dt=torch.int32)
                self.bar_t = getattr(self, 'bar_t', []) + [k[-1]]   # [2], [3]: debug words of the kernel
                for d in range(NCARD):
                    ctx.comm[d] = self.comm[d].data_ptr()
                if os.environ.get("MK_TPROF"):
                    ctx.tprof = z((NL + 1) * 64 + 1184 + 152 * 16 + 64, dt=torch.int64)
                    self.tprof = getattr(self, "tprof", []) + [k[-1]]
                if debug:
                    ctx.dbg = z(NL, 2, MAXB, H)
                    ctx.dbg_ids = z(NL, MAXB, 8, dt=torch.int32)
                    self.dbg_t = getattr(self, "dbg_t", []) + [k[-2], k[-1]]
                ctx.amax_loc = z(64 * MAXB, dt=torch.int32)
                ctx.ucnt = z(64, dt=torch.int32)
                ctx.rll = z(MAXB * NE * 2, dt=torch.int32)
                ctx.tred = z(10 * MAXB * 32 * 18 * 2, dt=torch.int32)
                ctx.gll = z(MAXB * 1536 * 2, dt=torch.int32)
                ctx.ctag = z(MAXB * 2 * 17 * 2, dt=torch.int32)
                if not hasattr(self, "dbg_prog"):   # progress words of the tensor-core kernel (host-mapped)
                    self.lib.ink_host_alloc.restype = ctypes.c_void_p
                    self.dbg_prog = self.lib.ink_host_alloc(ctypes.c_size_t(4096))
                ctx.prog = self.dbg_prog
                # the unembedding last: the runtime buffers above are placed before it, so their placement does not
                # depend on how it is stored (BF16 fragments, or the exact 12-bit code with track word W12U)
                # (measured: the order makes no difference in this kernel, e0108)
                rows = 201024 // NCARD
                un = ld.get("unembed.weight", (slice(rows * c, rows * (c + 1)),)).to(dev)
                un = pad_rows(un, (rows + 15) // 16 * 16)
                if USE_W12U:
                    ctx.un12 = w12_matrix(un, k, bool(os.environ.get("MK_W12_VERIFY")))
                else:
                    t = frag_bf16(un); k.append(t); ctx.unembed = t.data_ptr()
                ctx.un_strips = un.shape[0] // 16
                ctx.un_row0 = rows * c
                del un
                ctx.h.card = c
                ctx.h.nblk = self.nblk
                dctx = torch.empty(ctypes.sizeof(Ctx), dtype=torch.uint8, device=dev)
                self.ctx_host.append(ctx)
                self.ctx_dev.append(dctx)
        del embed
        self.streams = [torch.cuda.current_stream(0)] + [torch.cuda.Stream(device=c) for c in range(1, NCARD)]
        self.hdr_pinned = [torch.empty(ctypes.sizeof(Hdr) * 64, dtype=torch.uint8).pin_memory() for _ in range(NCARD)]
        self.hdr_slot = 0
        for c in range(NCARD):
            buf = (ctypes.c_uint8 * ctypes.sizeof(Ctx)).from_buffer_copy(self.ctx_host[c])
            self.ctx_dev[c].copy_(torch.frombuffer(bytearray(buf), dtype=torch.uint8))
        for c in range(NCARD):
            torch.cuda.synchronize(c)
        self.xbase = [0, 0, 0, 0]
        self.step_count = 0
        import collections
        self._keep_toks = collections.deque(maxlen=4096)
        # progress words of the tensor-core kernel (host-mapped) and a report when a launch does not end
        self.lib.ink_host_alloc.restype = ctypes.c_void_p
        self._last_launch = 0.0
        threading.Thread(target=self._stall_report, daemon=True, name="mk-stall-report").start()
        self.pos = 0
        self.B = None
        # A sequence has at most ctx_limit positions: the K/V cache holds max_context of them and a global layer's
        # attention combines at most ink_max_positions() (2176). Beyond that the kernel would write into other
        # sequences' K/V rows and chunk records, so every launch is checked (_launch) and refused loudly.
        self.ctx_limit = min(int(max_context), int(self.lib.ink_max_positions()))
        self.l0mode = 0
        if verbose and USE_W12U:
            print(f"MK_ENGINE W12: {W12_STATS}", flush=True)
        if USE_L0T:
            assert layers == NL and not self.red_ok and self.lib.ink_l0t_ok(self.nblk), "layer-0 tables: not valid for this build / card"
            t1 = time.time()
            tabs, embn = self._l0_rows(MAXB, 0)
            self._install_l0_tables(tabs, embn)
            if verbose:
                print(f"MK_ENGINE layer-0 token tables built in {time.time() - t1:.1f}s", flush=True)
        if verbose:
            print(f"MK_ENGINE ready in {time.time() - t0:.0f}s; GB per card "
                  f"{[round(torch.cuda.memory_allocated(c) / 1e9, 1) for c in range(NCARD)]}", flush=True)

    @staticmethod
    def _buf(keep, dev, shape, dt):
        t = torch.zeros(*shape, dtype=dt, device=dev)
        keep.append(t)
        return t.data_ptr()

    # -------------------------------------------------------------------------------------------- driving
    def _sync_all(self):
        for c in range(NCARD):
            torch.cuda.synchronize(c)

    def reset(self) -> None:
        self._sync_all()
        for c in range(NCARD):
            ctx = self.ctx_host[c]
            for name in ("hist_attn", "hist_mlp", "hist_k", "hist_v", "blocks", "partial"):
                pass
            # zero sequence state (histories, caches are overwritten position by position; histories must be zero)
            for name, numel, es in (("hist_attn", 2 * NL * MAXB * 3 * H, 2), ("hist_mlp", 2 * NL * MAXB * 3 * H, 2),
                                    ("hist_k", NL * MAXB * 3 * 256, 2), ("hist_v", NL * MAXB * 3 * 256, 2)):
                self.lib_memset(c, getattr(ctx, name), numel * es)
        self._sync_all()
        self.pos = 0
        self.B = None

    def lib_memset(self, c, ptr, nbytes):
        with torch.cuda.device(c):
            torch.cuda.current_stream(c)
            t = _wrap(ptr, nbytes, self.devs[c])
            t.zero_()

    def _l0_rows(self, B, pos):
        """Layer 0 alone for every token id (mode 1), B ids per launch at position `pos`, through the kernel a step
        of that batch size uses. Returns per card the projection rows [vocab][2304] (this card's q rows, then its r,
        sink, padding, k, v rows: what the main qkv GEMV of layer 0 writes; the gate rows are computed beside the
        attention as before and are not tabulated: their summation order depends on batch and position) and the
        RMS-normed embedding [vocab][H]. Both depend on the token id only."""
        V = 201024
        assert self.l0mode == 0 and self.ctx_host[0].L[0].qkv_skip == 0
        tabs = [torch.empty(V, 2304, dtype=torch.bfloat16, device=self.devs[c]) for c in range(NCARD)]
        embn = [torch.zeros(V, H, dtype=torch.bfloat16, device=self.devs[c]) for c in range(NCARD)]
        n = (V + B - 1) // B * B
        ids = (torch.arange(n, dtype=torch.int32, device=self.devs[0]) % V).contiguous()
        self.reset()
        for c in range(NCARD):
            self.ctx_host[c].l0emb = embn[c].data_ptr()
        self._push_ctx()
        qk = [_wrap(self.ctx_host[c].qkvo, MAXB * 3840 * 2, self.devs[c]).view(torch.bfloat16).view(MAXB, 3840) for c in range(NCARD)]
        self.pos = pos
        for a in range(0, n, B):
            self._launch([ids.data_ptr() + 4 * a] * NCARD, B, 0, 0, lo=0, hi=1, mode=1)
            m = min(B, V - a)
            for c in range(NCARD):
                with torch.cuda.device(c), torch.cuda.stream(self.streams[c]):
                    tabs[c][a:a + m, :1536] = qk[c][:m, :1536]
                    tabs[c][a:a + m, 1536:] = qk[c][:m, 3072:]
        self._sync_all()
        for c in range(NCARD):
            self.ctx_host[c].l0emb = 0
        self._push_ctx()
        self.reset()
        return tabs, embn

    def _push_ctx(self):
        self._sync_all()
        for c in range(NCARD):
            buf = (ctypes.c_uint8 * ctypes.sizeof(Ctx)).from_buffer_copy(self.ctx_host[c])
            with torch.cuda.device(c):
                self.ctx_dev[c].copy_(torch.frombuffer(bytearray(buf), dtype=torch.uint8))
        self._sync_all()

    def _install_l0_tables(self, tabs, embn):
        """From here on every launch runs in mode 2: the embedding pointer is the normed embedding, a step copies its
        tokens' rows of the projection table, and layer 0's main qkv GEMV has no strips left (192 - 96 gate strips,
        which the blocks beside the attention still compute, - 96 query strips skipped)."""
        for c in range(NCARD):
            self.keep[c] += [tabs[c], embn[c]]
            ctx = self.ctx_host[c]
            ctx.embed = embn[c].data_ptr()
            ctx.l0tab = tabs[c].data_ptr()
            ctx.L[0].qkv_strips = 192
            ctx.L[0].qkv_skip = 96
        self._push_ctx()
        self.l0mode = 2
        self.reset()

    def _launch(self, tokens_ptrs, B, logits_ptr, want_logits, gen_out=0, gen_idx=0, gen_n=0, lo=0, hi=NL, mode=None):
        if not 0 <= self.pos < self.ctx_limit:
            raise RuntimeError(f"MK_ENGINE context limit: a step at position {self.pos} was asked for, the limit is "
                               f"{self.ctx_limit} positions per sequence (max_context {self.max_context}, kernel "
                               f"{self.lib.ink_max_positions()}); nothing was launched")
        hi = min(hi, self.nlayers)
        parity = self.step_count & 1
        for c in range(NCARD):
            h = Hdr()
            h.card, h.B, h.nblk, h.pos, h.parity, h.want_logits = c, B, self.nblk, self.pos, parity, want_logits
            h.gen_idx, h.gen_n = gen_idx, gen_n
            for k in range(4):
                h.xbase[k] = self.xbase[k] & 0xFFFFFFFF
            if os.environ.get("MK_TCDBG") and not gen_out:   # host-mapped debug words of a TC_DEBUG build
                if not hasattr(self, "dbg_host"):
                    self.lib.ink_host_alloc.restype = ctypes.c_void_p
                    self.dbg_host = self.lib.ink_host_alloc(ctypes.c_size_t(4096 * 4))
                gen_out = self.dbg_host
            h.logits, h.tokens, h.gen_out = logits_ptr, tokens_ptrs[c], gen_out
            # first and last layer of the launch (tensor memory set-up / release), and the layer-0 mode
            h.pad1 = lo | ((hi - 1) << 8) | ((self.l0mode if mode is None else mode) << 16)
            h.tag0 = (self.step_count * NL) & 0xFFFFFFFF
            s = self.streams[c]
            with torch.cuda.device(c):
                r = self.lib.ink_launch(c, ctypes.c_void_p(self.ctx_dev[c].data_ptr()), ctypes.byref(h), self.nblk,
                                        ctypes.c_void_p(s.cuda_stream), lo, hi,
                                        2 if B >= self.tc_minb else int(B == 1 and self.red_ok))
                assert r == 0, f"launch failed {r}"
        self._last_launch = time.time()
        nl = hi - lo
        ev = [0, 0, 0, 1 if hi == NL else 0]   # kinds 0-2 now travel as tagged data
        for k in range(4):
            self.xbase[k] += ev[k]
        self.step_count += 1

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        if self.B is None:
            self.B = B
        assert B == self.B and B <= MAXB
        self._check_room(1)
        tok = tokens.to(device=self.devs[0], dtype=torch.int32).contiguous()
        logits = torch.empty(B, VOCAB_REAL, dtype=torch.float32, device=self.devs[0])
        ev = torch.cuda.Event()
        ev.record(self.streams[0])
        for c in range(1, NCARD):
            self.streams[c].wait_event(ev)
        self._keep_toks.append(tok)   # alive until its launch has long run: steps may be queued without a host synchronisation
        self._launch([tok.data_ptr()] * NCARD, B, logits.data_ptr(), 1)
        self.pos += 1
        return logits

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        if self.B is None:
            self.B = B
        assert B == self.B and B <= MAXB
        self._check_room(n_steps)
        out = torch.empty(n_steps, B, dtype=torch.int32, device=self.devs[0])
        tok = tokens.to(device=self.devs[0], dtype=torch.int32).contiguous()
        ev = torch.cuda.Event()
        ev.record(self.streams[0])
        for c in range(1, NCARD):
            self.streams[c].wait_event(ev)
        nt = [self.ctx_host[c].next_tok for c in range(NCARD)]
        for k in range(n_steps):
            ptrs = [tok.data_ptr()] * NCARD if k == 0 else nt
            self._launch(ptrs, B, 0, 0)
            self.pos += 1
            out[k].copy_(_wrap(nt[0], B * 4, self.devs[0]).view(torch.int32), non_blocking=True)
        return out.t().long()

    def _check_room(self, n_steps, pos=None):
        """Refuse, before anything is launched, n_steps more steps that would pass the context limit."""
        pos = self.pos if pos is None else pos
        if n_steps < 0 or pos + n_steps > self.ctx_limit:
            raise ValueError(f"MK_ENGINE context limit: {n_steps} step(s) from position {pos} would reach position "
                             f"{pos + n_steps - 1}; a sequence has at most {self.ctx_limit} positions "
                             f"(max_context {self.max_context}, kernel {self.lib.ink_max_positions()})")

    # -------------------------------------------------------------------------------------------- online contract
    GATHER_S = 0.05

    def _online_init(self):
        if getattr(self, "_on", None) is not None:
            return
        self.lib.ink_host_alloc.restype = ctypes.c_void_p
        n = (self.max_context + 8) * MAXB
        ptr = self.lib.ink_host_alloc(ctypes.c_size_t(n * 4))
        assert ptr
        self._gen_host_ptr = ptr
        self._gen_host = (ctypes.c_int32 * n).from_address(ptr)
        import numpy as np
        self._gen_np = np.ctypeslib.as_array(self._gen_host)
        self._on = {"pending": [], "active": None, "first_t": None}

    def submit(self, rid, prompt, n) -> None:
        self._online_init()
        prompt, n = list(prompt), int(n)
        # a request of P prompt tokens and n new tokens takes the steps at positions 0 .. P + n - 2
        if len(prompt) < 1 or n < 1 or len(prompt) + n - 1 > self.ctx_limit:
            raise ValueError(f"MK_ENGINE context limit: request {rid!r} has {len(prompt)} prompt tokens and asks for {n} "
                             f"new ones; it needs 1 <= prompt, 1 <= n and prompt + n - 1 <= {self.ctx_limit} positions "
                             f"(max_context {self.max_context}, kernel {self.lib.ink_max_positions()}); not accepted")
        # token ids index the embedding (and, with the layer-0 tables, the tables): 201024 rows
        if min(prompt) < 0 or max(prompt) >= VOCAB_ROWS:
            raise ValueError(f"MK_ENGINE token id out of range: request {rid!r} holds an id outside [0, {VOCAB_ROWS}) "
                             f"(min {min(prompt)}, max {max(prompt)}); not accepted")
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
            P = len(first[1])
            same = [r for r in on["pending"] if len(r[1]) == P]
            if len(same) < MAXB and time.perf_counter() - first[3] < self.GATHER_S:
                return events
            cohort = same[:MAXB]
            on["pending"] = [r for r in on["pending"] if all(r is not q for q in cohort)]
            act = self._start_cohort(cohort)
            on["active"] = act
        # enqueue more steps (keep a window in flight)
        self._pump(act)
        g = self._gen_np
        B = act["B"]
        while act["read"] < act["nmax"]:
            k = act["read"]
            row = g[k * MAXB:k * MAXB + B]
            if (row == 0).any():
                break
            for i, (rid, _, n, _) in enumerate(act["reqs"]):
                if k < n:
                    events.append((rid, [int(row[i]) - 1], k == n - 1))
            act["read"] += 1
        if act["read"] >= act["nmax"]:
            on["active"] = None
        return events

    def _start_cohort(self, cohort):
        B = len(cohort)
        P = len(cohort[0][1])
        self.reset()
        self.B = B
        self._gen_np[:] = 0
        prompt = torch.tensor([r[1] for r in cohort], dtype=torch.int32).t().contiguous().to(self.devs[0])  # [P, B]
        nmax = max(r[2] for r in cohort)
        act = {"B": B, "P": P, "reqs": cohort, "nmax": nmax, "read": 0, "issued": 0, "prompt": prompt}
        # prefill: P-1 teacher-forced steps, all cards read the prompt tokens from card 0
        ev = torch.cuda.Event()
        ev.record(self.streams[0])
        for c in range(1, NCARD):
            self.streams[c].wait_event(ev)
        for t in range(P - 1):
            self._launch([prompt[t].data_ptr()] * NCARD, B, 0, 0)
            self.pos += 1
        return act

    def _pump(self, act, window=48):
        B = act["B"]
        target = min(act["nmax"], act["read"] + window)
        nt = [self.ctx_host[c].next_tok for c in range(NCARD)]
        while act["issued"] < target:
            k = act["issued"]
            ptrs = [act["prompt"][act["P"] - 1].data_ptr()] * NCARD if k == 0 else nt
            self._launch(ptrs, B, 0, 0, gen_out=self._gen_host_ptr, gen_idx=k, gen_n=1)
            self.pos += 1
            act["issued"] += 1

    def _stall_report(self):
        """A launch that has not ended 45 s after the last one was issued: say once, on stderr, which cards are
        still busy and where block 0 of each card's tensor-core kernel is (engine/ink.cu TC_PROG). No CUDA call is
        made while launches end normally."""
        reported = False
        while True:
            time.sleep(5)
            try:
                if not self._last_launch or time.time() - self._last_launch < 45:
                    reported = False
                    continue
                busy = []
                for c in range(NCARD):
                    with torch.cuda.device(c):
                        busy.append(not self.streams[c].query())
                if not any(busy):
                    reported = False
                    continue
                if reported:
                    continue
                reported = True
                v = (ctypes.c_int * (NCARD * 16)).from_address(self.dbg_prog)
                msg = [f"MK_ENGINE stall: no launch ended for {time.time() - self._last_launch:.0f} s; step_count {self.step_count} "
                       f"(tag0 of the last launch {((self.step_count - 1) * NL) & 0xFFFFFFFF}), pos {self.pos}, B {self.B}, cards busy {busy}"]
                for c in range(NCARD):
                    w = [v[c * 16 + k] & 0xFFFFFFFF for k in range(8)]
                    msg.append(f"  card {c}: started tag0 {w[0]}, ended tag0 {w[5]}; routing entered layer+1 {w[6]}, preps {w[7]}; "
                               f"gemv_tc entered layer+1 {w[1] & 255} w2 {(w[1] >> 8) & 255} nu {w[1] >> 16}, "
                               f"left layer+1 {w[2] & 255} (nsub {w[2] >> 16}), read-out progress {w[3] >> 8} of {w[3] & 255}")
                print("\n".join(msg), file=sys.stderr, flush=True)
            except Exception as ex:  # noqa: BLE001
                if not reported:
                    print(f"MK_ENGINE stall report failed: {ex}", file=sys.stderr, flush=True)
                reported = True

    def close(self) -> None:
        self._sync_all()

    def stats(self) -> dict:
        return {"implementation": "persistent-kernel-per-card-tp4", "nblk": self.nblk, "track": " ".join(TRACK),
                "w12": dict(W12_STATS) if USE_W12U else None, "l0_tables": self.l0mode == 2,
                "context_limit": self.ctx_limit}


def _wrap(ptr, nbytes, dev):
    """A uint8 tensor view of device memory owned elsewhere."""
    class _A:
        pass
    a = _A()
    a.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}
    with torch.cuda.device(dev):
        return torch.as_tensor(a, device=dev)


