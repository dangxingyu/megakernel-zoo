"""Inkling-2-Small serving engine for one GB200 node (4 x B200): the pass engine (engine/pe.cu, engine/sched.h).

A pass carries up to `tmax` rows (prompt chunks and decoding sequences together) through the 48 layers on all four
cards. Attention is split over the cards by heads, every expert lives whole on one card (96 per card), the rest of a
layer runs once per row on the row's home card. Weights keep the checkpoint's formats (BF16, NVFP4 codes + E4M3
scales, MXFP8 codes + UE8M0 scales); every expert GEMM's input is quantized as the model defines. K/V lives in pages
of 512 positions with reference counts; full prompt pages are kept in a prefix cache (hash chain of the 512-token
blocks) and evicted when memory is needed.

Online (submit / poll / idle / kv_pages / drop_cache / stats): by default the serving loop is on the devices
(MK_SCHED=device): the scheduler is a kernel on card 0 (engine/sched.h), this file only writes requests and prompts
into page-locked memory, reads tokens from it and keeps four threads enqueuing the fixed pass program.
MK_SCHED=host runs the same scheduling rules in Python (tests, pass logs for the cost model, tools/sim.py).
Offline (reset / step / generate): host-driven passes of the same kernels.
`MK_ENGINE=megakernel` selects the decode megakernel of run s7 (engine/mk_api.py + engine/ink.cu); it is kept for the
record and cannot serve (two kernels with different rounding).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:      # the scheduler can be simulated without a GPU (tools/sim.py)
    torch = None

HERE = Path(__file__).resolve().parent
H, NL, NE, NEC, NCARD, TOPK = 4608, 48, 384, 96, 4, 8
VOCAB_REAL = 200058
PREFIX = "model.llm."
PAGE, MAXPG, HS = 512, 68, 9728
AQ = 21                             # rows of an attention group (sched.h S_AQ)
PAGE_BYTES = 2 * 32 * 4096          # K (or V) of one layer of one page on one card
P = ctypes.c_void_p
I = ctypes.c_int


class PLayer(ctypes.Structure):
    _fields_ = [("wqkv", P), ("wo", P), ("wr", P), ("rbias", P), ("w13", P), ("w2", P), ("f13", P), ("f2", P),
                ("norm", P * 3), ("mlp_norm", P), ("ksc", P), ("vsc", P), ("attn_sc", P), ("mlp_sc", P),
                ("qscale", P), ("rscale", P), ("relp", P), ("aq", P), ("mq", P), ("kpg", P), ("vpg", P),
                ("glob", I), ("mx", I), ("qkv_strips", I), ("pad", I)]


HDR_INTS = ["T", "tag", "ngrp", "nseq", "nsnap", "nlog", "mode", "nprol", "unitsL", "unitsG", "o_slot", "o_pos", "o_tok",
            "o_back", "o_grp", "o_seq", "o_snap", "o_log", "o_prol", "o_pt", "lo", "hi", "tok_dev", "rg", "rgd", "pad2", "nswap", "o_swap"]


class PHdr(ctypes.Structure):
    _fields_ = [(n, I) for n in HDR_INTS] + [("logits", P), ("tokens", P)]


PEER_BUFS = ["xnat", "ids", "wts", "xouter", "cnt", "xq", "xs8", "attn_raw", "hfin"]      # PCtx.peer: enum PX_* in pe.cu
NPX = len(PEER_BUFS)
CTX_BUFS = ["tab", "blocks", "xin", "qkvo", "qn", "relb", "rt", "apart", "attn_out", "xo", "attn_raw", "attn_s", "mlp_in",
            "mlp_inx", "rlog", "ids", "wts", "pidx", "cnt", "emap", "xq", "xouter", "xs8", "ptok", "pslot", "items", "ginfo",
            "h13", "xt13", "aouter", "xt2", "pitem", "pairout", "mlp_raw", "hist", "snap", "hfin", "cand", "sig", "err", "cntg", "gate", "xnat", "kscr", "kflag"]


class PCtx(ctypes.Structure):
    _fields_ = ([("h", PHdr), ("L", PLayer * NL), ("card", I), ("npages", I), ("tmax", I), ("pmax", I),
                 ("eps", ctypes.c_float), ("umax", I), ("pad0", I), ("pad1", I),
                 ("embed", P), ("embed_norm", P), ("oq", P), ("fnorm", P), ("unembed", P), ("un_strips", I), ("un_row0", I)]
                + [(n, P) for n in CTX_BUFS]
                + [("attn_out_p", P * NCARD), ("pairout_p", P * NCARD), ("sig_p", P * NCARD), ("mlp_raw_p", P * NCARD), ("peer", (P * NPX) * NCARD),
                   ("sched", P), ("shr", P), ("arena", P), ("cand_p", P * NCARD), ("ctx_p", P * NCARD), ("tab_p", P * NCARD),
                   ("w13_p", (P * NCARD) * NL), ("w2_p", (P * NCARD) * NL), ("swtmp", P), ("harena", P)])


S_RING, S_OUT, S_NSIG = 1024, 1 << 17, NL * 4 + 24
S_NSLOT = 128                # engine/sched.h: slots of the device scheduler (checked against the library at load)


class SReq(ctypes.Structure):
    _fields_ = [("rid", I), ("plen", I), ("n_out", I), ("off", I)]


class SShared(ctypes.Structure):      # engine/sched.h: host-mapped; the host writes requests, the device tokens
    _fields_ = [(n, ctypes.c_uint) for n in ("ring_head", "ring_tail", "out_head", "cmd_drop", "ack_drop", "stop", "pass_no", "active",
                                             "free_pages", "total_pages", "cached_pages", "out_tail")] + \
               [("stats", ctypes.c_ulonglong * 32), ("ring", SReq * S_RING), ("out", (I * 2) * S_OUT)]


SC_NAMES = ["passes", "rows", "prefill_rows", "decode_rows", "cache_hit_blocks", "cache_miss_blocks", "late_hit_blocks", "evictions", "requests",
            "max_active", "cache_inserts", "dedup_waits", "first_tokens", "first_tokens_in_slo", "recomputes", "empty_passes", "expert_swaps", "rebalances",
            "page_alloc_failures", "promotions"]


class PassArgs(ctypes.Structure):
    _fields_ = [("dev", I), ("ctx", P), ("hdr", P), ("tab", P), ("tab_bytes", ctypes.c_long), ("tab_dev", P), ("stream", P), ("nsm", I), ("flags", I)]


# ------------------------------------------------------------------------------------------------ layouts
# Weights in the tensor core's operand layout (K-major, no swizzle): units of 128 rows x one K step, row m and
# element k of a step at byte (m // 8) * 256 + (k // KH) * 128 + (m % 8) * 16 + (k % KH) * B with KH elements per
# 16 bytes (bf16: 8, e4m3: 16, e2m1: 32). Block scales follow the unit as the 512-byte atom the tensor core's
# scale copy takes: row m, scale j at (m % 32) * 16 + (m // 32) * 4 + j. The bytes are the checkpoint's.
def tc_bf16(w: torch.Tensor) -> torch.Tensor:
    """[N, K] bf16 -> [N/128][K/16][4096 B]."""
    n, k = w.shape
    assert n % 128 == 0 and k % 16 == 0, (n, k)
    x = w.view(n // 128, 16, 8, k // 16, 2, 8).permute(0, 3, 1, 4, 2, 5)
    return x.contiguous().view(torch.uint8).flatten()


def tc_nvfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K/2] u8 (low nibble first), scales [E, N, K/16] (e4m3 as u8) -> [E][N/128][K/64][4608 B]."""
    e, n, kb = codes.shape
    rg, ks = n // 128, kb // 32
    c = codes.view(e, rg, 16, 8, ks, 2, 16).permute(0, 1, 4, 2, 5, 3, 6).reshape(e, rg, ks, 4096)
    sf = scales.view(torch.uint8).view(e, rg, 4, 32, ks, 4).permute(0, 1, 4, 3, 2, 5).reshape(e, rg, ks, 512)
    return torch.cat([c, sf], dim=-1).contiguous().flatten()


def tc_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """codes [E, N, K] (e4m3 as u8), scales [E, N, K/32] u8 (ue8m0) -> [E][N/128][K/32][4608 B] (scale in byte 0)."""
    e, n, k = codes.shape
    rg, ks = n // 128, k // 32
    c = codes.view(torch.uint8).view(e, rg, 16, 8, ks, 2, 16).permute(0, 1, 4, 2, 5, 3, 6).reshape(e, rg, ks, 4096)
    sc = scales.view(torch.uint8).view(e, rg, 4, 32, ks).permute(0, 1, 4, 3, 2)
    sf = torch.zeros(e, rg, ks, 32, 4, 4, dtype=torch.uint8, device=codes.device)
    sf[..., 0] = sc
    return torch.cat([c, sf.reshape(e, rg, ks, 512)], dim=-1).contiguous().flatten()


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


def load_card(ld: Loader, cfg: dict, c: int, layers: range, dev: torch.device, keep: list, wexp: dict):
    g = lambda n, sl=None: ld.get(n, sl)  # noqa: E731
    cu = lambda t: t.to(dev, non_blocking=False)  # noqa: E731
    local_ids = set(cfg["local_layer_ids"])
    lw = []
    es = slice(NEC * c, NEC * (c + 1))
    for i in range(NL):
        Lw = PLayer()
        if i not in layers:
            lw.append(Lw)
            continue
        p = f"layers.{i}."
        glob = i not in local_ids
        assert glob == (i % 6 == 5)
        hq = slice(1536 * c, 1536 * (c + 1))
        wq = cu(g(p + "attn.wq_du.weight", (hq,)))
        wg = cu(g(p + "attn.gate_attn.weight", (hq,)))
        wr = cu(g(p + "attn.wr_du.weight", (slice(192 * c, 192 * (c + 1)),)))
        ws = cu(g(p + "attn.attn_sink_proj.weight", (slice(12 * c, 12 * (c + 1)),)))
        grpA = pad_rows(torch.cat([wq, wg, wr, ws]), 26 * 128)
        kvs = slice(256 * c, 256 * (c + 1))
        wk = cu(g(p + "attn.wk_dv.weight", (kvs,)))
        wv = cu(g(p + "attn.wv_dv.weight", (kvs,)))
        if glob:
            dk = cu(g(p + "attn.k_sconv.weight_proj", (slice(8 * c, 8 * c + 8),)))
            dv = cu(g(p + "attn.v_sconv.weight_proj", (slice(8 * c, 8 * c + 8),)))
            grpB, grpC = pad_rows(torch.cat([wk, dk]), 384), pad_rows(torch.cat([wv, dv]), 384)
        else:
            grpB, grpC = wk, wv
        wqkv = torch.cat([grpA, grpB, grpC])
        Lw.qkv_strips = wqkv.shape[0] // 128
        assert Lw.qkv_strips == (32 if glob else 30)
        t = tc_bf16(wqkv); keep.append(t); Lw.wqkv = t.data_ptr()
        del wq, wg, wr, ws, wk, wv, grpA, grpB, grpC, wqkv
        t = tc_bf16(cu(g(p + "attn.wo_ud.weight")).contiguous()); keep.append(t); Lw.wo = t.data_ptr()
        t = tc_bf16(cu(g(p + "mlp.gate.weight"))); keep.append(t); Lw.wr = t.data_ptr()
        t = cu(g(p + "mlp.gate.bias")).float().contiguous(); keep.append(t); Lw.rbias = t.data_ptr()
        w13 = g(p + "mlp.experts.w13_weight", (es,))
        Lw.mx = int(w13.dtype == torch.float8_e4m3fn)
        assert Lw.mx == int(i >= 40)
        s13 = g(p + "mlp.experts.w13_weight_scale", (es,))
        w2 = g(p + "mlp.experts.w2_weight", (es,))
        s2 = g(p + "mlp.experts.w2_weight_scale", (es,))
        if not Lw.mx:
            t = tc_nvfp4(cu(w13), cu(s13)); keep.append(t); Lw.w13 = t.data_ptr(); wexp[i] = [t]
            t = tc_nvfp4(cu(w2), cu(s2)); keep.append(t); Lw.w2 = t.data_ptr(); wexp[i].append(t)
            for nm, attr in (("w13", "f13"), ("w2", "f2")):
                amax = g(p + f"mlp.experts.{nm}_weight_amax").double()
                den = g(p + f"mlp.experts.{nm}_weight_outer_scale_denominator").double()
                t = cu((amax / den).float().contiguous()); keep.append(t); setattr(Lw, attr, t.data_ptr())
        else:
            t = tc_mxfp8(cu(w13), cu(s13)); keep.append(t); Lw.w13 = t.data_ptr(); wexp[i] = [t]
            t = tc_mxfp8(cu(w2), cu(s2)); keep.append(t); Lw.w2 = t.data_ptr(); wexp[i].append(t)
        del w13, s13, w2, s2

        def small(name, sl=None):
            t = cu(g(p + name, sl)).contiguous(); keep.append(t); return t.data_ptr()
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
        assert t.numel() == 12 * 128, t.shape
        t = cu(g(p + "attn.rel_l2_clip.scale"))[12 * c:12 * (c + 1)].contiguous(); keep.append(t); Lw.rscale = t.data_ptr()
        assert t.numel() == 12 * 16, t.shape
        t = cu(g(p + "attn.rel_logits_proj.proj")).t().contiguous()   # [1024][16]
        assert tuple(t.shape) == (1024, 16), t.shape
        t = tc_bf16(t); keep.append(t); Lw.relp = t.data_ptr()
        if i > 0:
            t = (cu(ld.get(f"attn_res.attn_queries.{i - 1}")).float() * H ** -0.5).contiguous(); keep.append(t); Lw.aq = t.data_ptr()
            assert t.numel() == (3 if glob else 1) * H, (i, t.shape)
        t = (cu(ld.get(f"attn_res.mlp_queries.{i}")).float() * H ** -0.5).contiguous(); keep.append(t); Lw.mq = t.data_ptr()
        assert t.numel() == H
        Lw.glob = int(glob)
        lw.append(Lw)
    return lw


def _wrap(ptr, nbytes, dev):
    """A uint8 tensor view of device memory owned elsewhere."""
    class _A:
        pass
    a = _A()
    a.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}
    with torch.cuda.device(dev):
        return torch.as_tensor(a, device=dev)


class Seq:
    __slots__ = ("rid", "slot", "prompt", "plen", "n_out", "pos", "out", "pages", "keys", "ncached", "fresh", "prol_page",
                 "last_tok", "t_submit", "reserve", "emitted", "t_first")


class PassEngine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int, layers: int = NL, verbose: bool = True,
                 tmax: int | None = None, npages: int | None = None):
        t0 = time.time()
        self.cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        assert torch.cuda.device_count() >= NCARD, "needs 4 cards"
        self.max_batch, self.max_context = max_batch, max_context
        self.open_rids = set()
        self.nlayers = layers if isinstance(layers, int) else NL
        self.layer_set = range(layers) if isinstance(layers, int) else list(layers)   # a list: debugging single layers
        # slots: the device scheduler's table has S_NSLOT (128) whatever max_batch is (it serves as many requests as
        # arrive, up to that many at a time); every per-slot buffer is sized for them. (Sized by max_batch, an engine built with
        # max_batch = 8 wrote past its convolution-state buffer with more than 8 requests in flight.)
        self.nslot = max(max_batch, S_NSLOT)
        self.tmax = tmax or int(os.environ.get("MK_TMAX") or 3584)
        self.soft_rows = int(os.environ.get("MK_SOFT_ROWS") or 2560)      # rows of a pass unless a prompt would be late
        self.pmax = self.tmax * 8
        self.umax = 24576                  # attention units of a pass (a record of 128 q vectors each)
        self.lib = ctypes.CDLL(os.environ.get("MK_LIB") or str(HERE / "build" / "libpe.so"))
        assert self.lib.pe_nslot() == S_NSLOT, "engine/sched.h and engine/api.py disagree on the slot count"
        assert self.lib.pe_ctx_bytes() == ctypes.sizeof(PCtx), (self.lib.pe_ctx_bytes(), ctypes.sizeof(PCtx))
        assert self.lib.pe_layer_bytes() == ctypes.sizeof(PLayer)
        assert self.lib.pe_hdr_bytes() == ctypes.sizeof(PHdr)
        self.lib.pe_host_alloc.restype = ctypes.c_void_p
        self.devs = [torch.device("cuda", c) for c in range(NCARD)]
        for c in range(NCARD):
            torch.zeros(1, device=f"cuda:{c}")
        assert self.lib.pe_enable_peer(NCARD) == 0
        for c in range(NCARD):
            assert self.lib.pe_setup(c) == 0
        torch.cuda.set_device(0)
        self.nsm = torch.cuda.get_device_properties(0).multi_processor_count
        ld = Loader(model_dir)
        self.keep = [[] for _ in range(NCARD)]
        self.wexp = [dict() for _ in range(NCARD)]      # per card, per layer: [w13, w2] tensors, expert-major
        self.layerw = [None] * NCARD
        errs = []

        def work(c):
            try:
                with torch.cuda.device(c):
                    self.layerw[c] = load_card(ld, self.cfg, c, self.layer_set, self.devs[c], self.keep[c], self.wexp[c])
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
        embed = ld.get("embed.weight")
        self.vocab_rows = int(embed.shape[0])
        tm, pm = self.tmax, self.pmax
        # pages: what the cards can hold after the weights and the pass buffers
        fixed = (9 + 3 + 6) * tm * H * 2 + tm * 3824 * 2 + tm * 12 * 1152 * 2 + self.umax * 128 * 132 * 4 + pm * H * 7 + self.nslot * NL * 3 * HS * 2
        per_page = NL * 2 * PAGE_BYTES + NL * 3 * HS * 2
        want = int(os.environ.get("MK_NPAGES") or 0) or (max_batch * ((max_context + PAGE - 1) // PAGE) + 8 if max_batch <= 8 else 1 << 30)
        # (npages is None: the page store is sized below, from the memory that is free once everything else is allocated)
        self.ctx_host, self.ctx_dev, self.bufs = [], [], []
        for c in range(NCARD):
            dev = self.devs[c]
            with torch.cuda.device(c):
                k = self.keep[c]
                ctx = PCtx()
                for i in range(NL):
                    ctx.L[i] = self.layerw[c][i]
                t = embed.to(dev); k.append(t); ctx.embed = t.data_ptr()
                t = ld.get("embed_norm.weight").to(dev); k.append(t); ctx.embed_norm = t.data_ptr()
                t = (ld.get("attn_res.output_query").float() * H ** -0.5).to(dev).contiguous(); k.append(t); ctx.oq = t.data_ptr()
                assert t.numel() == H
                t = ld.get("norm.weight").to(dev); k.append(t); ctx.fnorm = t.data_ptr()
                rows = 201024 // NCARD
                un = ld.get("unembed.weight", (slice(rows * c, rows * (c + 1)),)).to(dev)
                un = pad_rows(un, (rows + 127) // 128 * 128)
                t = tc_bf16(un); k.append(t); ctx.unembed = t.data_ptr()
                ctx.un_strips = un.shape[0] // 128          # row groups
                assert ctx.un_strips == 393
                ctx.un_row0 = rows * c
                del un
                b = {}

                def z(name, nbytes):
                    t = torch.zeros(int(nbytes), dtype=torch.uint8, device=dev)
                    b[name] = t
                    setattr(ctx, name, t.data_ptr())
                z("tab", (16 * tm + 32 * self.nslot + self.nslot * MAXPG + 8192) * 4)
                z("blocks", 9 * tm * H * 2); z("xin", 3 * tm * H * 2); z("qkvo", tm * 3824 * 2); z("qn", tm * 12 * 128 * 2)
                z("relb", tm * 12 * 1024 * 2); z("rt", (tm * 12 // 128 + 1) * 4096); z("apart", self.umax * 128 * 132 * 4); z("attn_out", tm * 1536 * 2)
                z("xo", tm * 6144 * 2); z("attn_raw", tm * H * 2); z("attn_s", tm * H * 2); z("mlp_in", tm * H * 2)
                z("mlp_inx", tm * H * 2); z("rlog", tm * NE * 4); z("ids", tm * 8 * 4); z("wts", tm * 8 * 4); z("pidx", tm * 8 * 4)
                z("cnt", NL * NE * 4); z("emap", NL * NE * 4); z("xq", tm * H); z("xouter", tm * 4); z("xs8", tm * 384)
                z("ptok", 4 * pm * 4); z("pslot", 4 * pm * 4); z("items", 4 * pm * 16); z("ginfo", 64)
                mi = pm // 128 + NEC + 1                # items of a card: full tiles plus one partial tile per expert
                z("h13", pm * H * 2); z("xt13", mi * 144 * 4608); z("aouter", pm * 4); z("xt2", mi * 72 * 4608); z("pitem", pm * 4)
                z("pairout", pm * H * 2)
                z("mlp_raw", tm * H * 2); z("hist", self.nslot * NL * 3 * HS * 2)
                z("hfin", tm * H * 2); z("cand", tm * 8); z("sig", 4 * S_NSIG); z("err", 16); z("cntg", NL * NE * 4); z("gate", 4 * S_NSIG); z("xnat", 3 * tm * H * 2); z("kscr", 994 * 128 * 64 * 4); z("kflag", 994 * 4)
                ctx.card, ctx.tmax, ctx.pmax, ctx.umax = c, tm, pm, self.umax
                ctx.eps = float(self.cfg["rms_norm_eps"])
                self.ctx_host.append(ctx)
                self.bufs.append(b)
        del embed
        # The page store: what the cards can hold now that the weights and the pass buffers are allocated, less what
        # comes after it (the expert swap space, 0.45 GB per card; the scheduler's arrays on card 0, ~0.1 GB) and a
        # reserve. Measured: with 2.7 GB subtracted here, 7.5 GB per card are free when the engine serves (the driver
        # reports ~4.8 GB less free at this point than there is later). Estimated before the buffers were allocated,
        # the store had left 14.7 GB per card unused.
        if npages is None:
            torch.cuda.empty_cache()
            free = min(torch.cuda.mem_get_info(c)[0] for c in range(NCARD)) - 0.6e9 - 2.7e9
            npages = int(max(16, min(want, free // per_page)))
        self.npages = npages
        for c in range(NCARD):
            ctx, b, dev = self.ctx_host[c], self.bufs[c], self.devs[c]
            with torch.cuda.device(c):
                t = torch.zeros(npages * NL * 3 * HS * 2, dtype=torch.uint8, device=dev)
                b["snap"] = t; ctx.snap = t.data_ptr()
                for i in self.layer_set:
                    for nm in ("kpg", "vpg"):
                        t = torch.zeros(npages * PAGE_BYTES, dtype=torch.uint8, device=dev)
                        b[f"{nm}{i}"] = t
                        setattr(ctx.L[i], nm, t.data_ptr())
                ctx.npages = npages
        # where each expert's weights are: card | index on the card << 8 (experts 96 c .. 96 c + 95 start on card c)
        self.emap = ((np.arange(NE) // NEC) | ((np.arange(NE) % NEC) << 8)).astype(np.int32)[None].repeat(NL, 0).copy()
        for c in range(NCARD):
            self.bufs[c]["emap"].view(torch.int32).copy_(torch.from_numpy(self.emap.reshape(-1)).to(self.devs[c]))
        for c in range(NCARD):
            for d in range(NCARD):
                self.ctx_host[c].attn_out_p[d] = self.ctx_host[d].attn_out
                self.ctx_host[c].pairout_p[d] = self.ctx_host[d].pairout
                self.ctx_host[c].sig_p[d] = self.ctx_host[d].sig
                self.ctx_host[c].mlp_raw_p[d] = self.ctx_host[d].mlp_raw
                for k, nm in enumerate(PEER_BUFS):
                    self.ctx_host[c].peer[d][k] = getattr(self.ctx_host[d], nm)
        # the serving loop on the device (engine/sched.h): scheduler state, page arrays, prefix cache and the prompt
        # arena on card 0; the request ring and the token ring in host-mapped memory
        self.lib.pe_sched_bytes.restype = ctypes.c_int
        assert ctypes.sizeof(SShared) == self.lib.pe_shared_bytes(), (ctypes.sizeof(SShared), self.lib.pe_shared_bytes())
        self.ncache = 1 << max(12, int(npages * 4 - 1).bit_length())
        self.arena_ints = 16 << 20
        with torch.cuda.device(0):
            d0 = self.devs[0]
            self.dv = {"sched": torch.zeros(self.lib.pe_sched_bytes(), dtype=torch.uint8, device=d0),
                       "freep": torch.zeros(npages * 4, dtype=torch.uint8, device=d0), "page_ref": torch.zeros(npages * 4, dtype=torch.uint8, device=d0),
                       "cache": torch.zeros(self._cache_bytes(npages), dtype=torch.uint8, device=d0),      # the table, then the cached pages' tokens
                       "arena": torch.zeros(self.arena_ints * 4, dtype=torch.uint8, device=d0), "cfg": torch.zeros(128, dtype=torch.uint8, device=d0),
                       "eload": torch.zeros(NL * NE * 4, dtype=torch.uint8, device=d0), "emaster": torch.zeros(NL * NE * 4, dtype=torch.uint8, device=d0)}
        self.swtmp = []
        for c in range(NCARD):                        # room for the experts a card receives in one round of swaps
            with torch.cuda.device(c):
                self.swtmp.append(torch.zeros(12 * (36 * 144 * 4608 * 3 // 2), dtype=torch.uint8, device=self.devs[c]) if max_batch > 8 else None)
        self.shared_ptr = self.lib.pe_host_alloc(ctypes.c_size_t(ctypes.sizeof(SShared)))
        self.harena_ptr = self.lib.pe_host_alloc(ctypes.c_size_t(self.arena_ints * 4))      # arriving prompts, page-locked
        self.harena = np.ctypeslib.as_array((ctypes.c_int32 * self.arena_ints).from_address(self.harena_ptr))
        assert self.shared_ptr
        self.shared = SShared.from_address(self.shared_ptr)
        for c in range(NCARD):
            with torch.cuda.device(c):
                self.ctx_dev.append(torch.zeros(ctypes.sizeof(PCtx), dtype=torch.uint8, device=self.devs[c]))
        for c in range(NCARD):
            ctx = self.ctx_host[c]
            ctx.sched, ctx.shr, ctx.arena = self.dv["sched"].data_ptr(), self.shared_ptr, self.dv["arena"].data_ptr()
            ctx.harena = self.harena_ptr
            ctx.swtmp = self.swtmp[c].data_ptr() if self.swtmp[c] is not None else 0
            for i in range(NL):
                for d in range(NCARD):
                    ctx.w13_p[i][d] = self.ctx_host[d].L[i].w13
                    ctx.w2_p[i][d] = self.ctx_host[d].L[i].w2
            for d in range(NCARD):
                ctx.cand_p[d] = self.ctx_host[d].cand
                ctx.ctx_p[d] = self.ctx_dev[d].data_ptr()
                ctx.tab_p[d] = self.ctx_host[d].tab
        for c in range(NCARD):
            with torch.cuda.device(c):
                raw = bytes((ctypes.c_uint8 * ctypes.sizeof(PCtx)).from_buffer_copy(self.ctx_host[c]))
                self.ctx_dev[c].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(self.devs[c]))
        self.streams = [torch.cuda.current_stream(0)] + [torch.cuda.Stream(device=c) for c in range(1, NCARD)]
        for c in range(NCARD):
            torch.cuda.synchronize(c)
        # host staging: a ring of pinned (header + tables) buffers
        self.tab_ints = 16 * tm + 32 * self.nslot + self.nslot * MAXPG + 8192
        self.ring_n = 64
        hb = ctypes.sizeof(PHdr)
        self.hdr_bytes = hb
        self.stage_bytes = hb + self.tab_ints * 4
        self.stage_ptr = self.lib.pe_host_alloc(ctypes.c_size_t(self.stage_bytes * self.ring_n))
        assert self.stage_ptr
        self.stage = np.ctypeslib.as_array((ctypes.c_uint8 * (self.stage_bytes * self.ring_n)).from_address(self.stage_ptr))
        self.ring_i = 0
        self.tag = 0
        self.fixed_grids = int(os.environ.get("MK_FIXED_GRIDS", 0))
        self.sched_mode = os.environ.get("MK_SCHED", "device")       # device: the serving loop runs on the cards (k_sched)
        self.dev_on = False
        self.unsynced = 0
        self._keep_alive = deque(maxlen=256)
        self._host_state(npages)
        if verbose:
            print(f"MK_ENGINE ready in {time.time() - t0:.0f}s; pages {npages}; GB per card "
                  f"{[round(torch.cuda.memory_allocated(c) / 1e9, 1) for c in range(NCARD)]}", flush=True)

    def _host_state(self, npages: int) -> None:
        """Everything the host keeps: page store, prefix cache, slots, queues, the pass cost model."""
        self.npages = npages
        self.free_pages = list(range(npages - 1, -1, -1))
        self.page_ref = np.zeros(npages, dtype=np.int32)
        self.cache = OrderedDict()        # key -> [page, parent key, children, active users]
        self.cache_evictable = 0
        self.reserved = 0
        self.pt = np.zeros((self.nslot, MAXPG), dtype=np.int32)
        self.computing = {}               # block key -> the sequence that is computing it now
        self.free_slots = list(range(self.nslot - 1, -1, -1))
        self.queue = deque()
        self.active = []
        self.off_seqs = None
        self.pos = 0
        self._clock = time.perf_counter
        # pass time model in ms: base + dec * decode rows + sum over prompt rows of (row + ctx * position), times a
        # correction that follows the measured passes
        self.cost = {"base": float(os.environ.get("MK_COST_BASE", 13.5)), "dec": float(os.environ.get("MK_COST_DEC", 0.05)),
                     "row": float(os.environ.get("MK_COST_ROW", 0.0177)), "ctx": float(os.environ.get("MK_COST_CTX", 0.20e-6))}
        self.cost_scale = 1.0
        self.target_decode_ms = float(os.environ.get("MK_PASS_MS", 80.0))       # passes that carry decoding sequences
        self.target_idle_ms = float(os.environ.get("MK_PASS_IDLE_MS", 140.0))  # passes with prompts only
        self.ttft_slo = float(os.environ.get("MK_TTFT_SLO", 5.0))
        self.pass_log = [] if os.environ.get("MK_PASS_LOG") else None
        self.rebalance = int(os.environ.get("MK_REBALANCE", 1))
        self.eload = np.zeros((NL, NE))   # recent pairs per expert and layer (what the cards' expert load is made of)
        self.reb_seen = 0
        self.slack_passes = int(os.environ.get("MK_SLACK", 1))
        self.gap_limit = float(os.environ.get("MK_GAP_LIMIT", 0.090))             # mean gap a request is held to, s
        self.min_pass_ms = float(os.environ.get("MK_PASS_MIN_MS", 45.0))
        self.late_pass_ms = float(os.environ.get("MK_PASS_LATE_MS", 0.0))        # pass length while only late prompts wait (0: no cap)
        self.host_ms = float(os.environ.get("MK_HOST_MS", 3.0))                   # between two passes: readback, admission
        self.counters = {"passes": 0, "rows": 0, "prefill_rows": 0, "decode_rows": 0, "cache_hit_blocks": 0, "cache_miss_blocks": 0,
                         "late_hit_blocks": 0, "evictions": 0, "host_syncs": 0, "requests": 0, "max_active": 0, "max_queue": 0,
                         "pass_ms_sum": 0.0, "pass_ms_max": 0.0, "cache_inserts": 0, "dedup_waits": 0, "first_tokens": 0,
                         "first_tokens_in_slo": 0, "ttft_max_s": 0.0, "expert_swaps": 0, "rebalances": 0}

    # -------------------------------------------------------------------------------------------- pages
    def _alloc_page(self) -> int:
        if not self.free_pages:
            self._evict(1)
        p = self.free_pages.pop()
        self.page_ref[p] = 1
        return p

    def _evict(self, n: int) -> None:
        """Forget cached pages nobody uses, least recently used leaves first."""
        freed = 0
        while freed < n:
            victim = None
            for key, node in self.cache.items():
                if node[2] == 0 and node[3] == 0:
                    victim = key
                    break
            if victim is None:
                raise RuntimeError("KV store full: no evictable page")
            self._drop_node(victim)
            freed += 1
            self.counters["evictions"] += 1

    def _drop_node(self, key) -> None:
        page, parent, _, _ = self.cache.pop(key)
        if parent is not None and parent in self.cache:
            self.cache[parent][2] -= 1
        self.cache_evictable -= 1
        self._unref(page)

    def _unref(self, page: int) -> None:
        self.page_ref[page] -= 1
        if self.page_ref[page] == 0:
            self.free_pages.append(page)

    def _available(self) -> int:
        return len(self.free_pages) + self.cache_evictable - self.reserved

    def kv_pages(self):
        if self.dev_on:
            return (int(self.shared.free_pages), int(self.shared.total_pages))
        return (len(self.free_pages), self.npages)

    def drop_cache(self) -> None:
        """Forget every retained prefix that no running request uses (an idle engine: all of them)."""
        if self.dev_on:
            return self._dev_drop_cache()
        for key in list(self.cache.keys()):
            node = self.cache[key]
            if node[3] == 0:
                self.cache.pop(key)
                if node[1] is not None and node[1] in self.cache:
                    self.cache[node[1]][2] -= 1
                self._unref(node[0])
        self.cache_evictable = 0

    # -------------------------------------------------------------------------------------------- sequences
    @staticmethod
    def _block_keys(prompt: np.ndarray, nblocks: int):
        keys, h = [], b""
        for j in range(nblocks):
            h = hashlib.blake2b(h + prompt[j * PAGE:(j + 1) * PAGE].tobytes(), digest_size=16).digest()
            keys.append(h)
        return keys

    def _admit(self, rid, prompt, n_out, t_submit, use_cache=True) -> Seq | None:
        plen = len(prompt)
        total_pages = (plen + n_out + PAGE - 1) // PAGE
        keys = self._block_keys(prompt, plen // PAGE) if use_cache else []
        usable = min(len(keys), (plen - 1) // PAGE)      # the last prompt position is always computed
        hit = 0
        while hit < usable and keys[hit] in self.cache:
            hit += 1
        # pages the hit would pin are evictable now and will not be afterwards
        pinned = sum(1 for j in range(hit) if self.cache[keys[j]][3] == 0)
        need = total_pages - hit
        if not self.free_slots or self._available() - pinned < need:
            return None
        s = Seq()
        s.rid, s.prompt, s.plen, s.n_out, s.t_submit = rid, prompt, plen, n_out, t_submit
        s.slot = self.free_slots.pop()
        s.keys, s.ncached = keys, hit
        s.pages = []
        for j in range(hit):
            node = self.cache[keys[j]]
            if node[3] == 0:
                self.cache_evictable -= 1
            node[3] += 1
            self.cache.move_to_end(keys[j])
            self.page_ref[node[0]] += 1
            s.pages.append(node[0])
        s.pos = hit * PAGE
        s.fresh, s.prol_page = True, (s.pages[-1] if hit else -1)
        s.out, s.last_tok, s.emitted, s.t_first = [], -1, 0, None
        s.reserve = need
        self.reserved += need
        self.counters["cache_hit_blocks"] += hit
        self.counters["cache_miss_blocks"] += len(keys) - hit
        self.counters["requests"] += 1
        return s

    def _ensure_pages(self, s: Seq, upto_pos: int) -> None:
        """Pages for positions < upto_pos."""
        while len(s.pages) * PAGE < upto_pos:
            s.reserve -= 1
            self.reserved -= 1
            s.pages.append(self._alloc_page())
        self.pt[s.slot, :len(s.pages)] = s.pages

    def _release(self, s: Seq) -> None:
        for j, page in enumerate(s.pages):
            key = s.keys[j] if j < len(s.keys) else None
            node = self.cache.get(key) if key is not None else None
            if node is not None and node[0] == page:
                node[3] -= 1
                if node[3] == 0:
                    self.cache_evictable += 1
            self._unref(page)
        self.reserved -= s.reserve
        s.reserve = 0
        s.pages = []
        for key in [k for k, v in self.computing.items() if v is s]:
            del self.computing[key]
        self.free_slots.append(s.slot)

    def _cache_insert(self, s: Seq, j: int) -> None:
        """Block j of s (a full prompt page whose snapshot was written) becomes shareable."""
        key = s.keys[j]
        if key in self.cache:
            return
        parent = s.keys[j - 1] if j else None
        if parent is not None:
            pn = self.cache.get(parent)
            if pn is None:
                return                      # the chain below is not cached: this page could not be found again
            pn[2] += 1
        self.cache[key] = [s.pages[j], parent, 0, 1]
        self.page_ref[s.pages[j]] += 1
        self.counters["cache_inserts"] += 1

    def _late_hits(self, s: Seq) -> None:
        """At a page boundary of its prompt: take over blocks that were cached since the request was admitted."""
        if s.pos % PAGE or len(s.pages) * PAGE != s.pos:
            return
        usable = min(len(s.keys), (s.plen - 1) // PAGE)
        j = s.pos // PAGE
        got = 0
        while j < usable:
            node = self.cache.get(s.keys[j])
            if node is None:
                break
            if node[3] == 0:
                self.cache_evictable -= 1
            node[3] += 1
            self.cache.move_to_end(s.keys[j])
            self.page_ref[node[0]] += 1
            s.pages.append(node[0])
            j += 1
            got += 1
        if got:
            s.pos = j * PAGE
            s.fresh, s.prol_page = True, s.pages[-1]
            s.reserve -= got
            self.reserved -= got
            self.pt[s.slot, :len(s.pages)] = s.pages
            self.counters["late_hit_blocks"] += got

    @staticmethod
    def own_lo(T: int, c: int) -> int:
        """First pass row whose home is card c (pe.cu own_lo): the cards own consecutive quarters of a pass."""
        if T <= 64:                    # REPL_T: every card owns every row; card 0 stands for them
            return 0 if c == 0 else T
        return T if c >= NCARD else (T * c + 3) >> 2

    def home_rows(self, name: str, dtype, lead: int, width: int, T: int):
        """Rows [0, T) of a per-row buffer [lead][tmax][width], each row from its home card (tests)."""
        out = torch.empty(lead, T, width, dtype=dtype)
        for c in range(NCARD):
            lo, hi = self.own_lo(T, c), self.own_lo(T, c + 1)
            if hi > lo:
                out[:, lo:hi] = self.bufs[c][name].view(dtype)[: lead * self.tmax * width].view(lead, self.tmax, width)[:, lo:hi].cpu()
        return out

    # -------------------------------------------------------------------------------------------- one pass
    def _run_pass(self, plan, mode: int, logits_ptr: int = 0, tokens_ptr: int = 0, lo: int = 0, hi: int | None = None):
        """plan: list of (seq-like with .slot/.fresh/.prol_page/.pages, pos0, tokens (np.int32) or None, want_logit_last,
        snap list [(n_rows_before_boundary, page)]). Launches the pass on the four cards; returns nlog."""
        hi = self.nlayers if hi is None else hi
        T = sum(len(p[2]) if p[2] is not None else p[5] for p in plan)
        assert 0 < T <= self.tmax
        slot = np.empty(T, np.int32); pos = np.empty(T, np.int32); tok = np.zeros(T, np.int32); back = np.empty(T, np.int32)
        grp, seqj, snapj, logr, prol = [], [], [], [], []
        r = 0
        uL = uG = 0
        for s, p0, toks, want_last, snaps, nrow in plan:
            n = nrow
            slot[r:r + n] = s.slot
            pos[r:r + n] = np.arange(p0, p0 + n)
            if toks is not None:
                tok[r:r + n] = toks
            back[r:r + n] = np.minimum(np.arange(n), 3)
            for g0 in range(0, n, AQ):                 # attention groups: up to AQ consecutive rows
                gn = min(AQ, n - g0)
                gp = p0 + g0
                pgl0 = max(0, gp - 1023) >> 9
                npgl = ((gp + gn - 1) >> 9) - pgl0 + 1
                npgg = ((gp + gn - 1) >> 9) + 1
                grp.append((r + g0, gn, s.slot, gp, pgl0, npgl, uL, uG))
                uL += 2 * npgl
                uG += 2 * npgg
            seqj.append((s.slot, s.slot, r, n))
            for nb, page in snaps:
                snapj.append((page, s.slot, r, nb))
            if want_last == 1:
                logr.append(r + n - 1)
            elif want_last == 2:
                logr.extend(range(r, r + n))
            if s.fresh:
                prol.append((s.slot, s.prol_page))
                s.fresh = False
            r += n
        assert uG <= self.umax, (uG, self.umax)
        self.tag += 1
        hdr = PHdr()
        hdr.T, hdr.tag, hdr.ngrp, hdr.nseq, hdr.nsnap, hdr.nlog, hdr.mode, hdr.nprol = T, self.tag, len(grp), len(seqj), len(snapj), len(logr), mode, len(prol)
        hdr.unitsL, hdr.unitsG = uL, uG
        parts = [slot, pos, tok, back, np.asarray(grp, np.int32).reshape(-1), np.asarray(seqj, np.int32).reshape(-1),
                 np.asarray(snapj, np.int32).reshape(-1), np.asarray(logr, np.int32), np.asarray(prol, np.int32).reshape(-1),
                 self.pt.reshape(-1)]
        offs, o = [], 0
        for a in parts:
            offs.append(o)
            o += a.size
        assert o <= self.tab_ints
        (hdr.o_slot, hdr.o_pos, hdr.o_tok, hdr.o_back, hdr.o_grp, hdr.o_seq, hdr.o_snap, hdr.o_log, hdr.o_prol, hdr.o_pt) = offs
        hdr.lo, hdr.hi = lo, hi
        hdr.pad2 = self.tmax      # the grid of a fixed-program launch
        hdr.rg = 1 if T <= int(os.environ.get("MK_RG1_T", "16")) else 3
        # dense GEMMs: four row groups per block when three would need a second round of block-units (148 SMs)
        hdr.rgd = 1 if T <= int(os.environ.get("MK_RG2_T", "1024")) else 2
        hdr.tok_dev = 1 if tokens_ptr else 0
        hdr.logits, hdr.tokens = logits_ptr, tokens_ptr
        base = self.ring_i * self.stage_bytes
        self.ring_i = (self.ring_i + 1) % self.ring_n
        st = self.stage[base:base + self.stage_bytes]
        ctypes.memmove(self.stage_ptr + base, ctypes.byref(hdr), self.hdr_bytes)
        st[self.hdr_bytes:self.hdr_bytes + o * 4] = np.concatenate(parts).view(np.uint8)
        args = (PassArgs * NCARD)()
        for c in range(NCARD):
            a = args[c]
            a.dev, a.ctx, a.hdr = c, self.ctx_dev[c].data_ptr(), self.stage_ptr + base
            a.tab, a.tab_bytes, a.tab_dev = self.stage_ptr + base + self.hdr_bytes, o * 4, self.ctx_host[c].tab
            a.stream, a.nsm, a.flags = self.streams[c].cuda_stream, self.nsm, self.fixed_grids
        rc = self.lib.pe_pass(args, NCARD, int(os.environ.get('MK_LAUNCH_THREADS', '1')))
        assert rc == 0, f"pass launch failed {rc}"
        self.counters["passes"] += 1
        self.counters["rows"] += T
        return len(logr)

    def _sync(self):
        for c in range(NCARD):
            self.lib.pe_sync(c, ctypes.c_void_p(self.streams[c].cuda_stream))
        self.counters["host_syncs"] += 1
        self.unsynced = 0
        e = self.bufs[0]["err"].view(torch.int32)[0].item()
        if e:
            raise RuntimeError(f"pass engine: a card did not signal exchange {e - 1}")

    def _read_tokens(self, nlog: int) -> np.ndarray:
        best = None
        for c in range(NCARD):
            v = self.bufs[c]["cand"][:nlog * 8].cpu().numpy().view(np.uint64)
            best = v if best is None else np.maximum(best, v)
        return (0xFFFFFFFF - (best & np.uint64(0xFFFFFFFF))).astype(np.int64)

    # -------------------------------------------------------------------------------------------- offline contract
    class _Off:
        __slots__ = ("slot", "fresh", "prol_page", "pages", "keys", "reserve")

    def reset(self) -> None:
        self._sync()
        if self.off_seqs:
            for s in self.off_seqs:
                for page in s.pages:
                    self._unref(page)
                self.free_slots.append(s.slot)
        self.off_seqs = None
        self.pos = 0

    def _off_init(self, B):
        if self.dev_on:
            self._dev_leave()          # step() / generate() compose their passes on the host: the device loop has to stand
        if self.off_seqs is None:
            assert not self.active and not self.queue, "offline and online use do not mix"
            self.off_seqs = []
            for b in range(B):
                s = PassEngine._Off()
                s.slot, s.fresh, s.prol_page, s.pages, s.keys, s.reserve = self.free_slots.pop(), True, -1, [], [], 0
                self.off_seqs.append(s)
        assert len(self.off_seqs) == B

    def _off_pages(self):
        for s in self.off_seqs:
            while len(s.pages) * PAGE <= self.pos:
                s.pages.append(self._alloc_page())
            self.pt[s.slot, :len(s.pages)] = s.pages

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        self._off_init(B)
        self._off_pages()
        tok = tokens.to(device=self.devs[0], dtype=torch.int32).contiguous()
        logits = torch.empty(B, VOCAB_REAL, dtype=torch.float32, device=self.devs[0])
        ev = torch.cuda.Event()
        ev.record(self.streams[0])
        for c in range(1, NCARD):
            self.streams[c].wait_event(ev)
        self._keep_alive.append((tok, logits))
        plan = [(s, self.pos, None, 1, [], 1) for s in self.off_seqs]
        self._run_pass(plan, 1, logits.data_ptr(), tok.data_ptr())
        self.pos += 1
        self.unsynced += 1
        if self.unsynced >= 24:
            self._sync()
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        self._off_init(B)
        cur = tokens.to("cpu", torch.int64).numpy().astype(np.int32)
        out = np.empty((B, n_steps), np.int64)
        for k in range(n_steps):
            self._off_pages()
            plan = [(s, self.pos, cur[b:b + 1], 1, [], 1) for b, s in enumerate(self.off_seqs)]
            nlog = self._run_pass(plan, 2)
            self._sync()
            nxt = self._read_tokens(nlog)
            out[:, k] = nxt
            cur = nxt.astype(np.int32)
            self.pos += 1
        return torch.from_numpy(out).to(self.devs[0])

    # -------------------------------------------------------------------------------------------- online contract
    # ---- device-side serving loop: the host copies prompts in and tokens out
    def _dev_start(self) -> None:
        assert self.off_seqs is None and not self.active and not self.queue, "the device loop starts on an unused engine"
        self._sync()
        c = self.cost
        reb = int(os.environ.get("MK_REB_EVERY", 6)) if (self.rebalance and self.swtmp[0] is not None) else 0
        cfg = (ctypes.c_double * 12)(c["base"], c["dec"], c["row"], c["ctx"], self.target_decode_ms, self.target_idle_ms,
                                     self.ttft_slo - 0.1, self.gap_limit, self.min_pass_ms, 1.0, reb, self.soft_rows)
        rc = self.lib.pe_sched_init(P(self.ctx_dev[0].data_ptr()), self.npages, self.tmax, self.umax, self.ncache, P(self.dv["freep"].data_ptr()),
                                    P(self.dv["page_ref"].data_ptr()), P(self.dv["cache"].data_ptr()), cfg, P(self.dv["cfg"].data_ptr()),
                                    P(self.dv["eload"].data_ptr()), P(self.dv["emaster"].data_ptr()))
        assert rc == 0, rc
        # (a second start in the same process: the loop's counters begin at zero again, and no flag of the first
        # run may look like a flag of the second, whose pass tags repeat)
        sh = self.shared
        for f in ("ring_head", "ring_tail", "out_head", "out_tail", "cmd_drop", "ack_drop", "pass_no", "active", "free_pages", "total_pages", "cached_pages"):
            setattr(sh, f, 0)
        for cc in range(NCARD):                       # every card has "taken" the pass before the first
            for nm in ("sig", "gate", "kflag"):
                self.bufs[cc][nm].zero_()
            sig = self.bufs[cc]["sig"].view(torch.int32)
            sig[S_NSIG - 2] = 1 << 20
            sig[S_NSIG - 1] = 1 << 20
            torch.cuda.synchronize(cc)
        self.lib.pe_stream_create.restype = ctypes.c_void_p
        self.dev_streams = [self.lib.pe_stream_create(cc) for cc in range(NCARD)]
        self.copy_stream = self.lib.pe_stream_create(0)
        self._dev_hdr = PHdr()
        self._dev_hdr.pad2 = self.tmax
        self._dev_args = (PassArgs * NCARD)()
        for cc in range(NCARD):
            a = self._dev_args[cc]
            a.dev, a.ctx, a.hdr, a.tab, a.tab_bytes, a.tab_dev = cc, self.ctx_dev[cc].data_ptr(), ctypes.addressof(self._dev_hdr), 0, 0, 0
            a.stream, a.nsm, a.flags = self.dev_streams[cc], self.nsm, 3
        self.out_seen, self.n_sub, self.n_fin, self.arena_off, self.polls = 0, 0, 0, 0, 0
        self.live, self.hold = {}, deque()       # prompts the device may still read (rid -> offset); requests waiting for room
        self.sub_ms = 0.0
        self.wd_pass, self.wd_t = -1, time.time()
        self.shared.stop = 0
        rc = self.lib.pe_pump_start(self._dev_args, P(self.shared_ptr + SShared.stop.offset))
        assert rc == 0
        self.dev_on = True

    def _cache_bytes(self, npages: int) -> int:
        self.lib.pe_cache_bytes.restype = ctypes.c_long
        return int(self.lib.pe_cache_bytes(self.ncache, npages))

    def _dev_leave(self) -> None:
        # The offline calls after online use: allowed on an idle engine (the loop is stopped; the next submit() starts
        # it again with an empty page store), refused while requests are open (two composers of passes would not do).
        if self.n_sub != self.n_fin:
            raise RuntimeError("step() / generate() while submitted requests are open")
        self.shared.stop = 1
        rc = self.lib.pe_pump_stop()
        self.dev_on = False
        assert rc == 0, f"device loop did not stop cleanly ({rc})"
        self._sync()

    def _dev_submit(self, rid, prompt, n) -> None:
        if not self.dev_on:
            if self.off_seqs is not None:
                self.reset()           # online use after offline use: the offline sequences are forgotten
            self._dev_start()
        self.n_sub += 1
        if self.hold or not self._dev_put(rid, prompt, n):
            self.hold.append((rid, prompt, n))        # no room in the ring or the prompt memory right now: poll() retries

    def _dev_put(self, rid, prompt, n) -> bool:
        # The prompt goes into page-locked memory the cards can read; the scheduler kernel copies it into its arena at
        # the next pass boundary. No call into the device here: a copy issued from the host waits behind the passes
        # already enqueued (measured: up to 62 ms).
        # The memory is a ring: a prompt's place is in use until its request's first token has come back (decoding does
        # not read the prompt), and a new prompt never overlaps the places in use (they lie, in the order written,
        # between the oldest one's start and arena_off).
        t0 = time.perf_counter()
        sh = self.shared
        if ((sh.ring_head - sh.ring_tail) & 0xFFFFFFFF) >= S_RING - 1:
            return False
        nids = len(prompt)
        off = self.arena_off
        if self.live:
            lo = next(iter(self.live.values()))
            if lo < off:                                # in use: [lo, off)
                if off + nids + 4 > self.arena_ints:
                    if nids + 4 > lo:
                        return False
                    off = 0
            elif off + nids + 4 > lo:                   # in use: [lo, end) and [0, off)
                return False
        elif off + nids + 4 > self.arena_ints:
            off = 0
        self.harena[off:off + nids] = prompt
        self.live[rid] = off
        self.sub_ms = max(self.sub_ms, (time.perf_counter() - t0) * 1e3)
        e = sh.ring[sh.ring_head & (S_RING - 1)]
        e.rid, e.plen, e.n_out, e.off = int(rid), nids, int(n), off
        self.arena_off = (off + nids + 3) & ~3              # 16-byte pieces on the device
        sh.ring_head = sh.ring_head + 1
        return True

    def _dev_poll(self):
        if not self.dev_on:
            return []
        sh = self.shared
        head = sh.out_head
        while self.hold and self._dev_put(*self.hold[0]):
            self.hold.popleft()
        if head == self.out_seen:
            self.polls += 1
            if self.polls % 4096 == 0:
                # watchdog on host-mapped memory only (reading a card's memory here waits behind the enqueued passes:
                # up to 300 ms measured): the count of rows computed has to move while requests are open (the count
                # of passes does not say it: a loop that composes empty passes is as dead as one that stands)
                now, pn = time.time(), int(sh.stats[1])
                if pn != self.wd_pass or self.n_sub == self.n_fin:
                    self.wd_pass, self.wd_t = pn, now
                elif now - self.wd_t > 30.0:
                    e = [int(self.bufs[cc]["err"].view(torch.int32)[k]) for cc in range(NCARD) for k in range(2)]
                    raise RuntimeError(f"device loop: no row for 30 s with requests open; cards' error words {e}")
            return []
        ev, order = {}, []
        i = self.out_seen
        while i != head:
            rid, v = sh.out[i & (S_OUT - 1)]
            if rid not in ev:
                ev[rid] = [[], False]
                order.append(rid)
                self.live.pop(rid, None)             # its prompt has been read
            ev[rid][0].append(v & 0x7FFFFFFF)
            if v < 0:
                ev[rid][1] = True
                self.n_fin += 1
                self.open_rids.discard(rid)
            i = (i + 1) & 0xFFFFFFFF
        self.out_seen = head
        sh.out_tail = head                           # the device stops composing rows when the ring of records is nearly full
        if sh.stats[SC_NAMES.index("page_alloc_failures")]:
            raise RuntimeError("device loop: a page was needed that admission had not reserved (K/V of page 0 may be overwritten)")
        return [(rid, ev[rid][0], ev[rid][1]) for rid in order]

    def _dev_drop_cache(self) -> None:
        sh = self.shared
        want = (sh.cmd_drop + 1) & 0xFFFFFFFF
        sh.cmd_drop = want
        t0 = time.time()
        while sh.ack_drop != want:
            assert time.time() - t0 < 30, "device loop does not answer drop_cache"
            time.sleep(0.0002)
        p0 = sh.pass_no
        while sh.pass_no == p0:                      # the pass after the drop has published the page counts
            assert time.time() - t0 < 30
            time.sleep(0.0002)

    def submit(self, rid, prompt, n) -> None:
        # what the engine cannot serve is refused here: nothing past this point checks a request's size again
        # (the limit is the page table's, 68 pages per sequence, not the constructor's max_context)
        if len(prompt) < 1 or int(n) < 1 or len(prompt) + int(n) > MAXPG * PAGE:
            raise ValueError(f"request {rid}: {len(prompt)} prompt tokens + {n} new tokens outside 1 .. {MAXPG * PAGE}")
        if (len(prompt) + int(n) + PAGE - 1) // PAGE > self.npages:
            raise ValueError(f"request {rid}: needs more K/V pages than the store has ({self.npages})")
        prompt = np.asarray(prompt, dtype=np.int32)
        if int(prompt.min()) < 0 or int(prompt.max()) >= self.vocab_rows:       # a row of the embedding table, or nothing
            raise ValueError(f"request {rid}: token id outside 0 .. {self.vocab_rows - 1}")
        if rid in self.open_rids:                   # (two open requests with one id would share a token stream)
            raise ValueError(f"request id {rid} is already open")
        self.open_rids.add(rid)
        if self.sched_mode == "device":
            return self._dev_submit(rid, prompt, n)
        self.queue.append((rid, prompt, int(n), self._clock()))
        self.counters["max_queue"] = max(self.counters["max_queue"], len(self.queue))

    def idle(self) -> bool:
        if self.sched_mode == "device":
            return not self.dev_on or self.n_sub == self.n_fin
        return not self.queue and not self.active

    def poll(self):
        if self.sched_mode == "device":
            return self._dev_poll()
        # admission: in arrival order, while slots and pages last
        while self.queue:
            rid, prompt, n, ts = self.queue[0]
            s = self._admit(rid, prompt, n, ts)
            if s is None:
                assert self.active, "a request does not fit an empty engine"
                break
            self.queue.popleft()
            self.active.append(s)
        if not self.active:
            return []
        self.counters["max_active"] = max(self.counters["max_active"], len(self.active))
        t0 = self._clock()
        plan, decode, prefill, est = self._compose(t0)
        nlog = self._run_pass(plan, 2)
        self._sync()
        toks = self._read_tokens(nlog)
        now = self._clock()
        events = []
        li = 0
        for s in decode:
            t = int(toks[li]); li += 1
            s.pos += 1
            s.out.append(t); s.last_tok = t
        for s, n, wants, snapped in prefill:
            p0 = s.pos
            s.pos += n
            for j in range(p0 // PAGE, s.pos // PAGE):
                if j < len(s.keys):
                    if j in snapped:
                        self._cache_insert(s, j)
                    if self.computing.get(s.keys[j]) is s:
                        del self.computing[s.keys[j]]
            if wants:
                t = int(toks[li]); li += 1
                s.out.append(t); s.last_tok = t
                s.t_first = now
                ttft = now - s.t_submit
                self.counters["first_tokens"] += 1
                self.counters["first_tokens_in_slo"] += ttft <= self.ttft_slo
                self.counters["ttft_max_s"] = max(self.counters["ttft_max_s"], ttft)
        done = []
        for s in self.active:
            if len(s.out) > s.emitted:
                fin = len(s.out) >= s.n_out
                if fin:
                    self.open_rids.discard(s.rid)
                events.append((s.rid, s.out[s.emitted:], fin))
                s.emitted = len(s.out)
                if fin:
                    done.append(s)
        for s in done:
            self.active.remove(s)
            self._release(s)
        ms = (now - t0) * 1e3
        nrows = sum(p[1] for p in prefill)
        if self.rebalance and nrows >= 256:
            self._observe_experts()
        if est > 0 and nrows >= 128:      # follow the measured time of passes with prompt rows (slowly: one pass says little)
            self.cost_scale = min(4.0, max(0.25, self.cost_scale * (1.0 + 0.2 * (ms / (est * self.cost_scale) - 1.0))))
        self.counters["pass_ms_sum"] += ms
        self.counters["pass_ms_max"] = max(self.counters["pass_ms_max"], ms)
        if self.pass_log is not None:
            self.pass_log.append((round(t0, 4), len(decode), sum(p[1] for p in prefill), round(ms, 2), round(est, 2),
                                  sum(p[0].pos for p in prefill) // max(1, len(prefill))))
        return events

    # ---- expert placement: the cards' expert load follows the traffic, so the engine moves experts between cards
    def _observe_experts(self) -> None:
        """Pairs per expert of the pass that just ran; every few passes swap experts between the most and the least
        loaded card of a layer. A swap exchanges the weights of two experts (same bytes, other card) and their
        entries in the placement table: no result depends on it."""
        cnt = self.bufs[0]["cntg"].view(torch.int32).cpu().numpy().reshape(NL, NE)
        self.eload = 0.75 * self.eload + 0.25 * cnt
        self.reb_seen += 1
        if self.reb_seen % int(os.environ.get("MK_REB_EVERY", 6)):
            return
        swaps = []
        cap = int(os.environ.get("MK_REB_SWAPS", 12))
        for i in self.layer_set:
            w = (self.eload[i] > 0.05) + self.eload[i] / 128.0          # about the weight streams (128-pair tiles)
            card = self.emap[i] & 255
            load = np.bincount(card, weights=w, minlength=NCARD)
            for _ in range(3):
                M, m = int(load.argmax()), int(load.argmin())
                gap = load[M] - load[m]
                if gap < 0.15 * load.mean() + 1.0:
                    break
                on_m = np.nonzero(card == m)[0]
                b = on_m[np.argmin(w[on_m])]
                on_M = np.nonzero(card == M)[0]
                d = w[on_M] - w[b]
                d = np.where((d > 0) & (d < gap), d, -1.0)
                if d.max() <= 0:
                    break
                a = on_M[np.argmin(np.abs(np.where(d > 0, d, 1e9) - gap / 2))]
                dv = w[a] - w[b]
                swaps.append((i, int(a), int(b)))
                load[M] -= dv; load[m] += dv
                card[a], card[b] = m, M
            if len(swaps) >= cap:
                break
        if not swaps:
            return
        touched = set()
        for i, a, b in swaps:
            ca, ia = int(self.emap[i, a]) & 255, int(self.emap[i, a]) >> 8
            cb, ib = int(self.emap[i, b]) & 255, int(self.emap[i, b]) >> 8
            for t in (0, 1):
                A = self.wexp[ca][i][t].view(NEC, -1)[ia]
                B = self.wexp[cb][i][t].view(NEC, -1)[ib]
                tmp = A.to(self.devs[cb])
                A.copy_(B)
                B.copy_(tmp)
            self.emap[i, a], self.emap[i, b] = cb | (ib << 8), ca | (ia << 8)
            touched.add(i)
        for c in range(NCARD):
            em = self.bufs[c]["emap"].view(torch.int32).view(NL, NE)
            for i in touched:
                em[i].copy_(torch.from_numpy(self.emap[i]).to(self.devs[c]))
        for c in range(NCARD):
            torch.cuda.synchronize(c)
        self.counters["expert_swaps"] += len(swaps)
        self.counters["rebalances"] += 1

    def _units(self, pos: int, n: int) -> int:
        """Attention units of the global layers for n rows from pos: 2 KV heads x pages per group of AQ rows (bound)."""
        return 2 * ((n + AQ - 1) // AQ) * (((pos + n - 1) >> 9) + 1)

    def _compose(self, now: float):
        """One pass: every decoding sequence (one row each), then prompt rows while the pass stays within its time.
        Prompts in the order that lets most of them answer within the first-token limit: those that can still make
        it first, shortest remaining first; a prompt whose next block another request is computing waits for it
        (and that request is served now)."""
        c = self.cost
        plan, decode, prefill = [], [], []
        ubudget = self.umax
        for s in self.active:
            if s.pos >= s.plen:
                self._ensure_pages(s, s.pos + 1)
                plan.append((s, s.pos, np.asarray([s.last_tok], np.int32), 1, [], 1))
                decode.append(s)
                ubudget -= 2 * ((s.pos >> 9) + 1)
        nd = len(decode)
        self.counters["decode_rows"] += nd
        est = c["base"] + c["dec"] * nd
        target = self.target_decode_ms if nd else self.target_idle_ms
        if nd and self.slack_passes:
            # a decoding request needs a mean gap under the limit, not every gap: a pass may be as long as the
            # request with the least credit allows if its remaining gaps then run at the normal pass time; a request
            # that has used its credit up gets shorter passes until it has some again
            allow = self.target_idle_ms
            for s in decode:
                gaps = s.n_out - 1
                if gaps > 0:
                    left = s.n_out - len(s.out) - 1          # gaps after this pass
                    allow = min(allow, 1e3 * (self.gap_limit * gaps - (now - s.t_first)) - left * self.target_decode_ms)
            target = max(self.min_pass_ms, allow - self.host_ms)
            if self.late_pass_ms and not any((s.t_submit + self.ttft_slo - now) >= (s.plen - s.pos) * c["row"] * self.cost_scale / 1e3 + 0.03
                                               for s in self.active if s.pos < s.plen):
                # no prompt can still answer in time: nothing is gained for the service level by long passes
                target = min(target, self.late_pass_ms)
        target /= self.cost_scale
        rows_left = self.tmax - nd
        ms_row = c["row"]
        pre = [s for s in self.active if s.pos < s.plen]
        for s in pre:
            self._late_hits(s)
        per_s = 1000.0 / (ms_row * self.cost_scale)          # prompt rows per second the engine does now
        if self.soft_rows < self.tmax:
            # the largest passes only while some waiting prompt, shortest first at the normal pass size, would be late
            # (sched.h has the same rule)
            cap = self.soft_rows
            rate = 1000.0 * cap / ((c["base"] + (ms_row + c["ctx"] * 12000.0) * cap) * self.cost_scale)
            alive = [(s.plen - s.pos, s.t_submit, k, s.t_submit + self.ttft_slo - now) for k, s in enumerate(pre)
                     if (s.t_submit + self.ttft_slo - now) >= (s.plen - s.pos) / per_s + 0.03]
            cum = 0
            for rem, _, _, left in sorted(alive):
                cum += rem
                if cum / rate + 0.30 > left:
                    cap = self.tmax
                    break
            rows_left = cap - nd

        def order(s):
            rem = s.plen - s.pos
            late = (s.t_submit + self.ttft_slo - now) < rem / per_s + 0.03
            return (late, rem, s.t_submit)
        scheduled = set()

        def take(s, limit):
            """Rows of s into this pass: as many as the time and row budgets allow, at most `limit`."""
            nonlocal est, rows_left, ubudget
            cr = ms_row + c["ctx"] * s.pos
            n = min(s.plen - s.pos, rows_left, limit, int((target - est) / cr))
            while n > 16 and self._units(s.pos, n) > ubudget:
                n = max(16, n // 2)
            if n <= 0 or self._units(s.pos, n) > ubudget:
                return
            self._ensure_pages(s, s.pos + n)
            snaps, snapped = [], set()
            for j in range(s.pos // PAGE, (s.pos + n) // PAGE):
                if j < len(s.keys) and s.keys[j] not in self.cache:
                    snaps.append(((j + 1) * PAGE - s.pos, s.pages[j]))
                    snapped.add(j)
            for j in range(s.pos // PAGE, min((s.pos + n - 1) // PAGE + 1, len(s.keys))):
                if s.keys[j] not in self.cache and s.keys[j] not in self.computing:
                    self.computing[s.keys[j]] = s
            wants = s.pos + n == s.plen
            plan.append((s, s.pos, s.prompt[s.pos:s.pos + n], 1 if wants else 0, snaps, n))
            prefill.append((s, n, wants, snapped))
            scheduled.add(id(s))
            est += cr * n
            rows_left -= n
            ubudget -= self._units(s.pos, n)
            self.counters["prefill_rows"] += n

        for s in sorted(pre, key=order):
            if rows_left <= 0 or est + ms_row * 16 > target:
                break
            if id(s) in scheduled:
                continue
            j = s.pos // PAGE
            owner = self.computing.get(s.keys[j]) if (s.pos % PAGE == 0 and j < len(s.keys)) else None
            if owner is not None and owner is not s:
                # the block is on its way: serve the request that computes it (to the end of that block) and wait
                self.counters["dedup_waits"] += 1
                if id(owner) not in scheduled and owner.pos < owner.plen:
                    take(owner, (owner.pos // PAGE + 1) * PAGE - owner.pos)
                continue
            take(s, 1 << 30)
        if not plan:        # nothing fitted the time budget: the most urgent prompt gets a minimal chunk
            s = min(pre, key=order)
            target = 1e9
            take(s, 256)
        return plan, decode, prefill, est

    def close(self) -> None:
        if self.dev_on:
            self.shared.stop = 1
            rc = self.lib.pe_pump_stop()
            self.dev_on = False
            assert rc == 0, f"device loop did not stop cleanly ({rc})"
        self._sync()

    def stats(self) -> dict:
        if self.dev_on:
            sh = self.shared
            d = {n: int(sh.stats[k]) for k, n in enumerate(SC_NAMES)}
            d.update(scheduler="device (k_sched on card 0; the host copies prompts in and tokens out)", host_syncs=0, passes_enqueued=int(self.lib.pe_pump_launched(0)),
                     submit_ms_max=round(self.sub_ms, 2),
                     implementation="pass-engine-tp4-attention-ep4-experts", tmax=self.tmax, npages=self.npages, free_pages=int(sh.free_pages),
                     cached_pages=int(sh.cached_pages), active=int(sh.active), track="A")
            return d
        d = dict(self.counters)
        d.update(cost_scale=round(self.cost_scale, 3), implementation="pass-engine-tp4-attention-ep4-experts", tmax=self.tmax, npages=self.npages,
                 free_pages=len(self.free_pages), cached_pages=len(self.cache), queue=len(self.queue), active=len(self.active),
                 track="A")
        return d


if os.environ.get("MK_ENGINE") == "megakernel":
    from mk_api import Engine  # noqa: F401
else:
    Engine = PassEngine
