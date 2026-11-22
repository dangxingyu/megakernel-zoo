"""The checkpoint's DSpark draft head (mtp.safetensors, config mtp_config) for exact draft-and-verify (track C).

The draft only proposes tokens; the verify pass of the target (mk.cu, mk_step<true>) decides. Its function follows the
harness's description of vLLM's DSpark (BF16 storage and GEMMs, FP32 norms / RoPE / softmax / Markov gate math, the
hidden_out x unembed fold in float64 stored BF16), written here on static buffers so that one draft-and-verify cycle
is a fixed sequence of kernels (captured once as a CUDA graph):

  after a verify pass of 8 rows (x_c, d_1 .. d_7 at positions c .. c+7): the target's tokens tgt[0..7] (argmax per
  row), a = number of leading drafts equal to tgt, committed tokens tgt[0..a]; every row's four target features
  (partials at the start of applications 5, 23, 41 and the final-norm output) become the draft's context K/V at its
  position (rows > a are overwritten by later cycles before any read: the block attends to positions < c_new only);
  c_new = c + a + 1, anchor = tgt[a]; the block of 8 rows (embedding of the anchor, 7 mask rows) through the 3 layers,
  the Markov recurrence -> 7 drafts; the next verify pass reads (anchor, drafts) at position c_new.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

BF16 = torch.bfloat16
NB = 8          # block rows (the draft's block; verify rows of the 8-row kernel)
VR = 16         # verify rows of the 16-row library (engine/mk16.cu)


class TorchDraft:
    def __init__(self, model_dir: str, maxctx: int, embed: torch.Tensor, embed_norm: torch.Tensor, unembed: torch.Tensor,
                 dev):
        from safetensors.torch import load_file
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        mc, tc = cfg["mtp_config"], cfg["text_config"]
        assert mc["markov_head_type"] == "rnn_hidden" and mc["block_size"] == NB
        self.H, self.L = mc["hidden_size"], mc["num_hidden_layers"]
        self.nh, self.nkv, self.hd = mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
        self.eps, self.rank, self.window = mc["rms_norm_eps"], mc["markov_rank"], mc["sliding_window_size"]
        self.local = set(mc["local_layer_ids"])
        assert mc["target_layer_ids"] == [5, 23, 41, 48]
        self.vocab = tc["unpadded_vocab_size"]
        self.cap = tc["final_logit_softcapping"]
        self.dev = dev
        self.maxctx = maxctx
        w = load_file(str(Path(model_dir) / "mtp.safetensors"), device=str(dev))
        self.embed = embed                                   # [rows, H] bf16 (the target's, shared)
        self.embed_norm = embed_norm.float()
        self.unembed = unembed[: self.vocab].contiguous()    # [V, H] bf16 row-major
        self.w = []
        for l in range(self.L):
            p = f"layers.{l}."
            g = lambda k: w[p + k]  # noqa: E731
            self.w.append({
                "qkv": torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]).contiguous(),
                "kv_ctx": torch.cat([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")]).contiguous(),
                "o": g("self_attn.o_proj.weight"),
                "gate_up": torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")]).contiguous(),
                "down": g("mlp.down_proj.weight"),
                **{k: g(n).float() for k, n in (("in_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"),
                                                ("dflare", "dflare_norm.weight"), ("q_norm", "self_attn.q_norm.weight"),
                                                ("k_norm", "self_attn.k_norm.weight"), ("k_norm_ctx", "self_attn.k_norm_ctx.weight"))},
            })
        self.norm = w["norm.weight"].float()
        self.mask_embed = w["mask_embed"]
        self.alpha = torch.softmax(w["fuse_weight"].float(), dim=-1).to(BF16).float()   # [L + 1, 4]
        mh = "markov_head."
        self.m_norm = w[mh + "dflare_norm.weight"].float()
        self.w1 = w[mh + "markov_w1.weight"]                                              # [padded vocab, r]
        rnn_w = w[mh + "rnn_proj.weight"]
        r = self.rank
        self.w_rec = rnn_w[:, :2 * r].float().t().contiguous()                            # [2r, 3r]
        self.w_ctx = rnn_w[:, 2 * r:].float().t().contiguous()                            # [2H, 3r]
        self.rnn_b = w[mh + "rnn_proj.bias"].float()
        ho_w, ho_b = w[mh + "hidden_out.weight"], w[mh + "hidden_out.bias"]
        fw = torch.empty(self.vocab, r, dtype=BF16, device=dev)
        fb = torch.empty(self.vocab, dtype=BF16, device=dev)
        hw, hb = ho_w.double(), ho_b.double()
        for s in range(0, self.vocab, 8192):
            head = self.unembed[s:s + 8192].double()
            fw[s:s + 8192] = (head @ hw).to(BF16)
            fb[s:s + 8192] = (head @ hb).to(BF16)
        self.fold_w, self.fold_b = fw, fb
        del w
        self.inv_freq = 1.0 / (mc["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float32, device=dev) / self.hd))
        # context K/V cache (sequence 0) per layer: rows 0..maxctx-1 = positions, rows maxctx.. = the block's own 8 rows
        T = maxctx + NB
        self.kc = [torch.zeros(T, self.nkv, self.hd, dtype=BF16, device=dev) for _ in range(self.L)]
        self.vc = [torch.zeros(T, self.nkv, self.hd, dtype=BF16, device=dev) for _ in range(self.L)]
        self.tpos = torch.arange(maxctx, device=dev)
        self.ar8 = torch.arange(NB, device=dev)
        # cycle state (device)
        self.vtok = torch.zeros(NB, dtype=torch.int32, device=dev)     # verify input: anchor, 7 drafts
        self.vpos = torch.tensor([0, NB], dtype=torch.int32, device=dev)   # [c, rows K of the next verify pass]
        self.pconf = torch.zeros(NB - 1, dtype=torch.float32, device=dev)  # the draft's top-1 probability per step
        self.theta = float(os.environ.get("MK_CUM_TH", "0.3"))   # rule 1: Markov early exit only (draftk.cu DK_RULE)
        self.ar7 = torch.arange(NB - 1, device=dev)
        self.c_new = torch.zeros(1, dtype=torch.long, device=dev)
        self.anchor = torch.zeros(1, dtype=torch.long, device=dev)
        self.frow = torch.zeros(1, dtype=torch.long, device=dev)
        self.feats = torch.zeros(NB, 4, self.H, dtype=BF16, device=dev)  # target features of the verify rows
        self.emit = torch.zeros(NB + 1, dtype=torch.int32, device=dev)   # [n, tokens...] of the last cycle

    # ------------------------------------------------------------------------------------------------ pieces
    def _rms(self, x, weight):
        u = x.float()
        return (u * torch.rsqrt(u.square().mean(-1, keepdim=True) + self.eps) * weight).to(BF16)

    def _add_rms(self, x, residual, weight):
        s = x.float() + residual.float()
        u = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + self.eps) * weight
        return u.to(BF16), s.to(BF16)

    def _rope(self, x, pos):
        f = pos.to(torch.float32)[:, None] * self.inv_freq[None]
        cos, sin = f.cos()[:, None], f.sin()[:, None]
        h = self.hd // 2
        x1, x2 = x[..., :h].float(), x[..., h:].float()
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(BF16)

    def _mix(self, feats, row, weight):
        m = torch.einsum("tjh,j->th", feats.float(), self.alpha[row])
        return (m * torch.rsqrt(m.square().mean(-1, keepdim=True) + self.eps) * weight).to(BF16)

    def ctx_append(self, feats, pos):
        """feats [n, 4, H] bf16 of positions pos [n] (long, device) -> context K/V rows at those positions."""
        for l in range(self.L):
            w = self.w[l]
            u = self._mix(feats, l, w["dflare"])
            kv = F.linear(u, w["kv_ctx"])
            k, v = kv.split([self.nkv * self.hd, self.nkv * self.hd], -1)
            k = self._rms(k.reshape(-1, self.nkv, self.hd), w["k_norm_ctx"])
            self.kc[l].index_copy_(0, pos, self._rope(k, pos))
            self.vc[l].index_copy_(0, pos, v.reshape(-1, self.nkv, self.hd))

    def propose(self):
        """state c_new, anchor, frow (feature row of position c_new - 1) -> vtok = [anchor, 7 drafts], vpos = c_new."""
        n, H, T = NB, self.H, self.maxctx
        c = self.c_new
        e = self.embed.index_select(0, self.anchor).float()
        e = (e * torch.rsqrt(e.square().mean(-1, keepdim=True) + self.eps) * self.embed_norm).to(BF16)
        x = torch.cat([e, self.mask_embed[None].expand(n - 1, H)], 0)                      # [8, H]
        pos = c + self.ar8                                                                  # [8]
        valid_g = self.tpos < c
        valid_l = valid_g & (self.tpos >= c - self.window)
        ones = torch.ones(n, dtype=torch.bool, device=self.dev)
        mask_g = torch.cat([valid_g, ones])
        mask_l = torch.cat([valid_l, ones])
        residual = None
        g = self.nh // self.nkv
        for l in range(self.L):
            w = self.w[l]
            if residual is None:
                residual, h = x, self._rms(x, w["in_ln"])
            else:
                h, residual = self._add_rms(x, residual, w["in_ln"])
            qkv = F.linear(h, w["qkv"])
            q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
            q = self._rope(self._rms(q.reshape(n, self.nh, self.hd), w["q_norm"]), pos)
            k = self._rope(self._rms(k.reshape(n, self.nkv, self.hd), w["k_norm"]), pos)
            self.kc[l][T:] = k
            self.vc[l][T:] = v.reshape(n, self.nkv, self.hd)
            keys, vals = self.kc[l].float(), self.vc[l].float()                             # [T + 8, nkv, D]
            mask = mask_l if l in self.local else mask_g
            qf = q.float().view(n, self.nkv, g, self.hd)
            s = torch.einsum("qkgd,tkd->kgqt", qf, keys) * self.hd ** -0.5
            s = s.masked_fill(~mask, float("-inf"))
            a = torch.einsum("kgqt,tkd->qkgd", torch.softmax(s, -1), vals)
            o = F.linear(a.reshape(n, self.nh * self.hd).to(BF16), w["o"])
            h, residual = self._add_rms(o, residual, w["post_ln"])
            gu = F.linear(h, w["gate_up"]).float()
            gate, up = gu.chunk(2, -1)
            x = F.linear((F.silu(gate) * up).to(BF16), w["down"])
        hid, _ = self._add_rms(x, residual, self.norm)
        hid = hid[1:]                                                                       # [7, H]
        # Markov head
        r = self.rank
        f = self._mix(self.feats.index_select(0, self.frow), self.L, self.m_norm)          # [1, H]
        inp = torch.cat([hid, f.expand(n - 1, H)], -1)                                      # [7, 2H]
        context = inp.float() @ self.w_ctx + self.rnn_b                                     # [7, 3r]
        base = F.linear(hid, self.unembed)                                                  # [7, V] bf16
        state = torch.zeros(1, r, dtype=BF16, device=self.dev)
        prev = self.anchor
        rd = lambda t: t.to(BF16).float()  # noqa: E731
        drafts, probs = [], []
        for i in range(n - 1):
            raw = context[i:i + 1] + torch.cat([state, self.w1.index_select(0, prev)], -1).float() @ self.w_rec
            gt, cand, out = rd(raw[:, :r]), rd(raw[:, r:2 * r]), rd(raw[:, 2 * r:])
            gt = rd(1.0 / (1.0 + torch.exp(-gt)))
            cand = rd(torch.tanh(cand))
            state = (rd(gt * state.float()) + rd(rd(1.0 - gt) * cand)).to(BF16)
            latent = torch.tanh(out).to(BF16)
            bias = torch.addmm(self.fold_b[None], latent, self.fold_w.t())                  # [1, V] bf16
            z = (base[i:i + 1].float() + bias.float()).to(BF16).float()
            lg = (self.cap * torch.tanh(z / self.cap)).to(BF16)
            prev = lg.float().argmax(-1)
            lf = lg.float()
            probs.append(torch.exp(lf.max(-1).values - torch.logsumexp(lf, -1)))
            if getattr(self, "conf_log", None) is not None:   # tests: top-2 logits and log-sum-exp per position
                lf = lg.float()
                t2 = lf.topk(2, -1).values
                self.conf_log.append(torch.cat([t2[0], torch.logsumexp(lf, -1)]))
            drafts.append(prev)
        self.vtok[0:1].copy_(self.anchor)
        self.vtok[1:].copy_(torch.cat(drafts))
        p = torch.cat(probs)
        self.pconf.copy_(p)
        # draft length: the largest k with prod_{i<=k} p_i >= theta (k >= 1); next verify rows = k + 1
        kd = (torch.cumprod(p, 0) >= self.theta).long().cumprod(0).sum(0, keepdim=True).clamp_min(1)
        self.vpos[0:1].copy_(c)
        self.vpos[1:2].copy_(kd + 1)

    def accept(self, tgt, feats):
        """After a verify pass of 8 rows: tgt [8] (int32 target tokens), feats [8, 4, H] -> c_new, anchor, frow, emit."""
        self.feats.copy_(feats)
        t = tgt.long()
        kp = self.vpos[1:2].long().clamp(1, NB)          # rows of the verify pass that just ran (rows >= kp stale)
        match = ((self.vtok[1:].long() == t[:NB - 1]) & (self.ar7 < kp - 1)).long()
        a = match.cumprod(0).sum(0, keepdim=True)                                            # [1]
        c = self.vpos[0:1].long()
        self.ctx_append(self.feats, c + self.ar8)
        self.c_new.copy_(c + a + 1)
        self.anchor.copy_(t.index_select(0, a))
        self.frow.copy_(a)
        self.emit[0:1].copy_(a + 1)
        self.emit[1:].copy_(tgt)

    # ------------------------------------------------------------------------------------------------ graphs
    def capture(self, tgt_fn, feats_fn):
        """CUDA graphs of the draft side of a cycle: cyc = accept (reads the verify pass's tokens and features) +
        propose; prop = propose alone (after a prefill)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        save = [t.clone() for t in (self.vtok, self.vpos, self.c_new, self.anchor, self.frow)]
        with torch.cuda.stream(s):
            for _ in range(2):
                self.accept(tgt_fn(), feats_fn())
                self.propose()
        torch.cuda.current_stream().wait_stream(s)
        self.g_cyc = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_cyc):
            self.accept(tgt_fn(), feats_fn())
            self.propose()
        self.g_prop = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_prop):
            self.propose()
        for t, v in zip((self.vtok, self.vpos, self.c_new, self.anchor, self.frow), save):
            t.copy_(v)
        for l in range(self.L):
            self.kc[l].zero_(); self.vc[l].zero_()
        torch.cuda.synchronize()


# ==================================================================================================== fused (CUDA)
HERE = Path(__file__).resolve().parent


def fp8_rows(w: torch.Tensor):
    """[N, K] -> (E4M3 codes as u8 [N, K], per-row FP32 scale amax / 448; zero rows scale 1). Deterministic."""
    wf = w.float()
    s = wf.abs().amax(1) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (wf / s[:, None]).to(torch.float8_e4m3fn).view(torch.uint8), s.contiguous()


def frag_fp8(c: torch.Tensor) -> torch.Tensor:
    """[N, K] u8 codes -> bytes [N/16][K/32 tile pairs][32 lanes][2 tiles][8 B]: the bf16 A-fragment element order, one
    byte per element, two k16 tiles per 16-byte lane load."""
    n, k = c.shape
    assert n % 16 == 0 and k % 64 == 0, (n, k)
    x = c.contiguous().view(n // 16, 2, 8, k // 32, 2, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    return x.contiguous().flatten()


def frag_bf16(w: torch.Tensor) -> torch.Tensor:
    """[N, K] bf16 -> bytes [N/16][K/64][4][32 lanes][16 B] in m16n8k16 A-fragment order (as api.py)."""
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0, (n, k)
    x = w.contiguous().view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)
    return x.contiguous().view(torch.uint8).flatten()


_P = ctypes.c_void_p
_L_FIELDS = ["qkv", "kvc", "o", "gu", "dn", "in_ln", "post_ln", "dflare", "q_norm", "k_norm", "k_norm_ctx", "kc", "vc",
             "qkv8", "kvc8", "o8", "gu8", "dn8", "qkv_s", "kvc_s", "o_s", "gu_s", "dn_s"]
_D_FIELDS = ["wctx", "wrec", "un", "fold", "fold_b", "w1", "embed", "mask_embed", "rnn_b", "embed_norm", "norm", "m_norm",
             "alpha", "inv_freq", "vtok", "vpos", "tgt", "feats", "emit", "c_new", "anchor", "frow", "pos64", "st",
             "u", "f", "xres", "h", "kvraw", "qkv", "q", "att", "act", "xm", "base", "state", "part", "opart", "ctxo", "raw",
             "amax", "pbt", "fout", "lse", "pconf", "acnt", "cum"]


class _DkL(ctypes.Structure):
    _fields_ = [(n, _P) for n in _L_FIELDS]


class _Dk(ctypes.Structure):
    _fields_ = [("L", _DkL * 3)] + [(n, _P) for n in _D_FIELDS] + [(n, ctypes.c_longlong) for n in ("T", "mode", "nrows", "pad")] + \
               [(n, _P) for n in ("un8", "fold8", "un_s", "fold_s", "hist")] + \
               [(n, ctypes.c_longlong) for n in ("f8", "ngmin", "hT")] + [("theta", ctypes.c_double)] + \
               [(n, _P) for n in ("tgt16", "pbt16", "fout16")] + [(n, ctypes.c_longlong) for n in ("lim16", "ng16min", "ng16acc")]


class _Launch:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


class FusedDraft:
    """Same interface as TorchDraft (vtok, vpos, c_new, anchor, frow, feats, emit, kc / vc, ctx_append, capture ->
    g_cyc / g_prop), computed by engine/draftk.cu (dk_run: ~45 kernels per cycle, captured in a CUDA graph)."""
    VP = 12504 * 16

    def __init__(self, model_dir: str, maxctx: int, embed, embed_norm, unembed, dev):
        from safetensors.torch import load_file
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        mc, tc = cfg["mtp_config"], cfg["text_config"]
        assert mc["markov_head_type"] == "rnn_hidden" and mc["block_size"] == NB and mc["num_hidden_layers"] == 3
        assert mc["hidden_size"] == 2304 and mc["intermediate_size"] == 9216 and mc["markov_rank"] == 256
        assert mc["local_layer_ids"] == [0, 2] and mc["sliding_window_size"] == 1024 and mc["target_layer_ids"] == [5, 23, 41, 48]
        self.lib = ctypes.CDLL(os.environ.get("MK_DKLIB") or str(HERE.parent / "build" / "libdk.so"))
        assert self.lib.dk_struct_bytes() == ctypes.sizeof(_Dk), (self.lib.dk_struct_bytes(), ctypes.sizeof(_Dk))
        assert self.lib.dk_setup() == 0
        self.dev, self.maxctx, self.L, self.H = dev, maxctx, 3, 2304
        self.vocab = tc["unpadded_vocab_size"]
        H, V = 2304, self.vocab
        k = self.keep = []

        def kp(t):
            t = t.contiguous(); k.append(t); return t.data_ptr()

        w = load_file(str(Path(model_dir) / "mtp.safetensors"), device=str(dev))
        d = self.d = _Dk()
        self.kc, self.vc = [], []
        T = maxctx + NB      # context positions < maxctx, then the block's own 8 rows
        for l in range(3):
            p = f"layers.{l}."
            g = lambda n: w[p + n]  # noqa: E731
            Lw = _DkL()
            f8l = {"0": 0, "": 0, "base": 1, "fold": 2, "layers": 4, "all": 7}.get(os.environ.get("MK_DK_FP8", "all").lower(), 3) & 4
            mats = {"qkv": torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")]),
                    "kvc": torch.cat([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")]),
                    "o": g("self_attn.o_proj.weight"),
                    "gu": torch.cat([g("mlp.gate_proj.weight").view(-1, 8, H), g("mlp.up_proj.weight").view(-1, 8, H)], 1).reshape(-1, H),
                    "dn": g("mlp.down_proj.weight")}
            for nm, m in mats.items():
                setattr(Lw, nm, kp(frag_bf16(m)))
                if f8l:   # FP8 copy for the draft's own GEMVs (approximate draft; MK_DK_FP8=layers / all)
                    c8, sc = fp8_rows(m)
                    setattr(Lw, nm + "8", kp(frag_fp8(c8)))
                    setattr(Lw, nm + "_s", kp(sc))
            del mats
            for a, n in (("in_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"), ("dflare", "dflare_norm.weight"),
                         ("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight"), ("k_norm_ctx", "self_attn.k_norm_ctx.weight")):
                setattr(Lw, a, kp(g(n).float()))
            kc = torch.zeros(T, 8, 128, dtype=BF16, device=dev); vc = torch.zeros_like(kc)
            self.kc.append(kc); self.vc.append(vc)
            Lw.kc, Lw.vc = kp(kc), kp(vc)
            d.L[l] = Lw
        mh = "markov_head."
        rnn_w = w[mh + "rnn_proj.weight"]
        d.wctx = kp(frag_bf16(rnn_w[:, 512:]))
        d.wrec = kp(frag_bf16(rnn_w[:, :512]))
        un = unembed[:V]
        pad = lambda t, n: torch.cat([t, t.new_zeros(n - t.shape[0], *t.shape[1:])])  # noqa: E731
        d.un = kp(frag_bf16(pad(un, self.VP)))
        ho_w, ho_b = w[mh + "hidden_out.weight"].double(), w[mh + "hidden_out.bias"].double()
        fw = torch.empty(V, 256, dtype=BF16, device=dev)
        fb = torch.empty(V, dtype=BF16, device=dev)
        for s in range(0, V, 8192):
            head = un[s:s + 8192].double()
            fw[s:s + 8192] = (head @ ho_w).to(BF16)
            fb[s:s + 8192] = (head @ ho_b).to(BF16)
        d.fold = kp(frag_bf16(pad(fw, self.VP)))
        f8 = os.environ.get("MK_DK_FP8", "all").lower()   # default: FP8 base + fold copies (measured: tokens/cycle within 0.3%, -55..-68 us per cycle)
        self.f8 = {"0": 0, "": 0, "base": 1, "fold": 2, "layers": 4, "all": 7}.get(f8, 3)
        d.f8 = self.f8
        if self.f8 & 2:
            c8, sc = fp8_rows(pad(fw, self.VP))
            d.fold8, d.fold_s = kp(frag_fp8(c8)), kp(sc); del c8
        if self.f8 & 1:
            c8, sc = fp8_rows(pad(un, self.VP))
            d.un8, d.un_s = kp(frag_fp8(c8)), kp(sc); del c8
        del fw
        d.fold_b = kp(fb)
        d.w1 = kp(w[mh + "markov_w1.weight"])
        d.embed = kp(embed)
        d.mask_embed = kp(w["mask_embed"])
        d.rnn_b = kp(w[mh + "rnn_proj.bias"].float())
        d.embed_norm = kp(embed_norm.float())
        d.norm = kp(w["norm.weight"].float())
        d.m_norm = kp(w[mh + "dflare_norm.weight"].float())
        d.alpha = kp(torch.softmax(w["fuse_weight"].float(), dim=-1).to(BF16).float())
        d.inv_freq = kp(1.0 / (mc["rope_theta"] ** (torch.arange(0, 128, 2, dtype=torch.float32, device=dev) / 128)))
        del w
        z = lambda *s_, dt=BF16: torch.zeros(*s_, dtype=dt, device=dev)  # noqa: E731
        self.vtok = z(VR, dt=torch.int32)
        self.vpos = torch.tensor([0, NB], dtype=torch.int32, device=dev)   # [c, rows K of the next verify pass]
        self.pconf = z(NB - 1, dt=torch.float32)
        self.theta = float(os.environ.get("MK_CUM_TH", "0.3"))   # rule 1: Markov early exit only (draftk.cu DK_RULE)
        d.pconf, d.theta = self.pconf.data_ptr(), self.theta
        d.cum = kp(z(NB, dt=torch.float32))
        d.acnt = kp(z(NB, dt=torch.int32))
        d.lse = kp(z(7 * ((12504 + 7) // 8), dt=torch.float32))
        self.c_new = z(1, dt=torch.long); self.anchor = z(1, dt=torch.long); self.frow = z(1, dt=torch.long)
        self.pos64 = z(NB, dt=torch.long)
        self.feats = z(NB, 4, H)
        self.emit = z(VR + 1, dt=torch.int32)
        d.vtok, d.vpos, d.c_new, d.anchor, d.frow = (t.data_ptr() for t in (self.vtok, self.vpos, self.c_new, self.anchor, self.frow))
        d.pos64, d.feats, d.emit = self.pos64.data_ptr(), self.feats.data_ptr(), self.emit.data_ptr()
        d.tgt = 0
        for n, shape, dt in (("st", (32,), torch.int32), ("u", (3, VR, H), BF16), ("f", (H,), BF16), ("xres", (NB, H), BF16),
                             ("h", (NB, H), BF16), ("kvraw", (3, VR, 2048), BF16), ("qkv", (NB, 5120), BF16), ("q", (NB, 3072), BF16),
                             ("att", (NB, 3072), BF16), ("act", (NB, 9216), BF16), ("xm", (NB, 2 * H), BF16),
                             ("base", (NB, self.VP), BF16), ("state", (2, 256), BF16), ("part", (8 * 32 * 24 * 132,), torch.float32),
                             ("opart", (4, NB, H), torch.float32), ("ctxo", (2, NB, 768), torch.float32), ("raw", (768,), torch.float32),
                             ("amax", (NB,), torch.long)):
            setattr(d, n, kp(z(*shape, dt=dt)))
        d.T = maxctx
        # n-gram drafts inside the cycle (the rule of engine/ngram.py; the engine skips its own n-gram step)
        self.hist = torch.full((maxctx + 2 * VR,), -1, dtype=torch.int32, device=dev)
        d.hist, d.hT = self.hist.data_ptr(), self.hist.numel()
        ng_on = os.environ.get("MK_NGRAM", "1") == "1" and os.environ.get("MK_DK_NGRAM", "1") == "1"
        self.handles_ngram = ng_on
        d.ngmin = int(os.environ.get("MK_NGRAM_MIN", "7")) if ng_on else 99   # 7 since s0013 (5-6-token matches go to DSpark or the short-copy rule)
        d.tgt16 = d.pbt16 = d.fout16 = None
        d.lim16, d.ng16min, d.ng16acc = -1, 99, 99      # 16-row passes off until capture() gets the 16-row library's buffers
        self.g_cyc = self.g_prop = None
        torch.cuda.synchronize()

    def _run(self, mode, nrows=NB):
        self.d.mode, self.d.nrows = mode, nrows
        r = self.lib.dk_run(ctypes.byref(self.d), _P(torch.cuda.current_stream().cuda_stream))
        assert r == 0, f"dk_run failed {r}"

    def ngram_prefill(self, prompt, anchor):
        """prompt int32 [P] (device), anchor int32 [1]: the n-gram history before the first proposal."""
        P = prompt.shape[0]
        self.hist.fill_(-1)
        self.hist[:P].copy_(prompt)
        self.hist[P:P + 1].copy_(anchor)

    def ctx_append(self, feats, pos):
        n = feats.shape[0]
        self.feats[:n].copy_(feats)
        self.pos64[:n].copy_(pos)
        self._run(1, n)

    def propose(self):
        self._run(2)

    def capture(self, tgt_fn, feats_fn, pbt=None, fout=None, tgt16=None, pbt16=None, fout16=None, lim16=-1, pre_fn=None):
        """pbt / fout (the engine's tagged partials [49][8][H/2][2] int32 and final-norm rows [8][H] bf16): when given,
        the cycle reads the features straight from them (no copy into feats)."""
        tgt = tgt_fn()
        assert tgt.dtype == torch.int32 and tgt.is_contiguous()
        self.d.tgt = tgt.data_ptr()
        direct = pbt is not None and fout is not None
        if direct:
            assert pbt.dtype == torch.int32 and tuple(pbt.shape[1:]) == (NB, 1152, 2) and fout.dtype == BF16
            self.d.pbt, self.d.fout = pbt.data_ptr(), fout.data_ptr()
            feats_fn = None
            if tgt16 is not None:   # the 16-row library's outputs (engine/mk16.cu): n-gram copies of 15 tokens
                assert tgt16.dtype == torch.int32 and tgt16.numel() >= VR and tuple(pbt16.shape[1:]) == (VR, 1152, 2)
                self.d.tgt16, self.d.pbt16, self.d.fout16 = tgt16.data_ptr(), pbt16.data_ptr(), fout16.data_ptr()
                self.d.lim16 = int(lim16)
                self.d.ng16min = int(os.environ.get("MK_NG16_MIN", "5"))
                self.d.ng16acc = int(os.environ.get("MK_NG16_ACC", "7"))   # simulated: natural (12 prompts) -1.9%, headline prompts =; served: natural -1.5%
        self._feats_fn = feats_fn
        save = [t.clone() for t in (self.vtok, self.vpos, self.c_new, self.anchor, self.frow)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                if feats_fn: self.feats.copy_(feats_fn())
                self._run(0)
                self._run(2)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if os.environ.get("MK_DK_NOGRAPH") == "1":
            self.g_cyc = _Launch(lambda: (self._feats_fn and self.feats.copy_(self._feats_fn()), self._run(0)))
            self.g_prop = _Launch(lambda: self._run(2))
        else:
            self.g_cyc = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_cyc):
                if pre_fn: pre_fn()   # the cycle's verify pass (engine: SWITCH node over the specialized kernels)
                if feats_fn: self.feats.copy_(feats_fn())
                self._run(0)
            self.g_prop = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_prop):
                self._run(2)
        for t, v in zip((self.vtok, self.vpos, self.c_new, self.anchor, self.frow), save):
            t.copy_(v)
        for l in range(3):
            self.kc[l].zero_(); self.vc[l].zero_()
        torch.cuda.synchronize()


def Draft(model_dir, maxctx, embed, embed_norm, unembed, dev):
    """The draft the engine uses: fused CUDA (default) or the torch one (MK_DRAFTK=0)."""
    if os.environ.get("MK_DRAFTK", "1") == "1":
        return FusedDraft(model_dir, maxctx, embed, embed_norm, unembed, dev)
    return TorchDraft(model_dir, maxctx, embed, embed_norm, unembed, dev)
