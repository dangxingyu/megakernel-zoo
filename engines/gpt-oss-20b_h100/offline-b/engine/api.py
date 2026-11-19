"""gpt-oss-20b decode engine: one persistent CUDA kernel per step / generate call / served cohort (engine/mk.cu).
This file loads the checkpoint, re-lays the weights into the kernel's tile order (a permutation of the stored bytes:
BF16 values and MXFP4 codes/scales unchanged) and launches.

    Engine(model_dir, max_batch, max_context); reset(); step(tokens) -> logits [B, vocab] float32;
    generate(tokens, n) -> int64 [B, n]; submit(rid, prompt, n) / poll()
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
HID, NL, VOCAB, NE, MAXB = 2880, 24, 201088, 32, 8
NB = 132   # blocks of the persistent kernel (one per SM of an H100)
NQH, NKV, HD = 64, 8, 64


SKS = 90                      # k32-steps per sketch tile
SKT = SKS * 384 + 256         # bytes per sketch tile: 6-bit fragments (3 words per lane per k32-step), then per row pair g: s, s', en, en', qn, qn', 0, 0


def pack_sketch(w16: torch.Tensor) -> torch.Tensor:
    """Track B certified head sketch. Row i: s_i = max|w_i| / 127 (float32), q_i = round(w_i / s_i) (int8), stored in
    the A-fragment order of mma.m16n8k32.s8 (lane (g, t): a0 = row g, k 4t..4t+3; a1 = row g+8; a2/a3: k + 16).
    Header per row: en_i >= ||w_i - s_i q_i|| + 2^-10 ||w_i|| (the 2^-10 ||w|| ||x|| term covers the float32 rounding
    of the exact path: 2880 products, <= 2880 * 2^-23 = 3.4e-4) and qn_i >= ||s_i q_i||, from float64, rounded up."""
    V, K = w16.shape
    out = torch.empty(V // 16, SKT, dtype=torch.uint8, device="cuda")
    for r0 in range(0, V, 16 * 1024):
        w = w16[r0:r0 + 16 * 1024].view(torch.bfloat16).double()
        s = (w.abs().amax(1) / 31.0).float()          # 6-bit codes q in [-31, 31]
        s = torch.where(s > 0, s, torch.full_like(s, 1e-30))
        q = torch.round(w / s.double()[:, None]).clamp(-31, 31)
        e = (w - s.double()[:, None] * q).norm(dim=1) + 2.0 ** -10 * w.norm(dim=1)
        qn = (s.double()[:, None] * q).norm(dim=1)
        en32 = torch.nextafter(e.float(), torch.tensor(float("inf"), device="cuda"))
        qn32 = torch.nextafter(qn.float(), torch.tensor(float("inf"), device="cuda"))
        en32 = torch.where(en32.double() >= e, en32, torch.nextafter(en32, torch.tensor(float("inf"), device="cuda")))
        qn32 = torch.where(qn32.double() >= qn, qn32, torch.nextafter(qn32, torch.tensor(float("inf"), device="cuda")))
        n = w.shape[0] // 16
        q8 = q.to(torch.int64).view(n, 2, 8, SKS, 2, 4, 4)         # [tile][row half][g][m][k half][t][4]
        frag = q8.permute(0, 3, 2, 5, 4, 1, 6).contiguous()        # [tile][m][g][t][k half][row half][4] -> a0 a1 a2 a3
        c = (frag & 63).view(n, SKS, 32, 4, 4)                       # 6-bit two's complement codes [tile][m][lane][reg][byte]
        a = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)   # [tile][m][lane][4] 24-bit fields
        W0 = a[..., 0] | ((a[..., 3] & 0xFF) << 24)
        W1 = a[..., 1] | (((a[..., 3] >> 8) & 0xFF) << 24)
        W2 = a[..., 2] | (((a[..., 3] >> 16) & 0xFF) << 24)
        words = torch.stack([W0, W1, W2], 2)                       # [tile][m][3][32 lanes]
        words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
        out[r0 // 16:r0 // 16 + n, :SKS * 384] = words.contiguous().view(n, SKS * 96).view(torch.uint8)
        hdr = torch.zeros(n, 8, 8, dtype=torch.float32, device="cuda")
        sv, ev, qv = s.view(n, 2, 8), en32.view(n, 2, 8), qn32.view(n, 2, 8)
        hdr[:, :, 0], hdr[:, :, 1] = sv[:, 0], sv[:, 1]
        hdr[:, :, 2], hdr[:, :, 3] = ev[:, 0], ev[:, 1]
        hdr[:, :, 4], hdr[:, :, 5] = qv[:, 0], qv[:, 1]
        out[r0 // 16:r0 // 16 + n, SKS * 384:] = hdr.view(n, 64).view(torch.uint8)
    return out.view(-1)


SK5 = os.environ.get("MK_SK5", "1") == "1"   # track B: the B=1 build (-DSK5) reads the 5-bit group-scaled sketch; the general build the 6-bit one
SKT5 = (SKS // 2) * 672 + 256  # bytes per tile: per k32-step pair 5 code words + 1 scale word per lane, then the header


def pack_sketch5(w16: torch.Tensor) -> torch.Tensor:
    """Track B certified head sketch, 5-bit codes with group scales. Row i: base sb_i = max|w_i| / (15 * 255) (float32);
    group g (columns 32g..32g+31 = k32-step g): m_ig = ceil(max_g|w| / (15 sb_i)) in [0, 255], q = round(w / (sb_i m_ig))
    clamped to [-16, 15]; w^ = sb_i m_ig q exactly. Per k32-step pair p and lane (g, t): words N0..N3 hold the low 4 bits
    of the 32 codes of the lane's 8 A registers (registers 2i, 2i+1 in the low / high nibbles of N_i's bytes) and word
    P holds bit 4 of byte j of register r at bit 8j + r; the kernel decodes register r to bytes 8q (int8). Then one
    scale word per g: m(row g, 2p), m(row g+8, 2p), m(row g, 2p+1), m(row g+8, 2p+1). Header as pack_sketch with s = sb,
    en_i >= ||w_i - w^_i|| + 2^-10 ||w_i|| + 2^-15 ||w^_i|| (the last term covers the float32 group accumulation of
    the sketch: 90 terms) and qn_i >= ||w^_i|| (1 + 2^-15)."""
    V, K = w16.shape
    NP = SKS // 2
    out = torch.empty(V // 16, SKT5, dtype=torch.uint8, device="cuda")
    up = torch.tensor(float("inf"), device="cuda")
    for r0 in range(0, V, 16 * 1024):
        w = w16[r0:r0 + 16 * 1024].view(torch.bfloat16).double()
        R = w.shape[0]; n = R // 16
        sb = (w.abs().amax(1) / (15.0 * 255.0)).float()
        sb = torch.where(sb > 0, sb, torch.full_like(sb, 1e-30))
        ag = w.view(R, SKS, 32).abs().amax(2)                                       # [R, 90]
        m = torch.ceil(ag / (15.0 * sb.double()[:, None])).clamp(0, 255)
        sg = sb.double()[:, None] * m                                               # exact (float32 x 8-bit integer)
        q = torch.where(sg[:, :, None] > 0, torch.round(w.view(R, SKS, 32) / sg[:, :, None].clamp_min(1e-300)), torch.zeros_like(w.view(R, SKS, 32)))
        q = q.clamp(-16, 15)
        wh = (sg[:, :, None] * q).view(R, K)
        e = (w - wh).norm(dim=1) + 2.0 ** -10 * w.norm(dim=1) + 2.0 ** -15 * wh.norm(dim=1)
        qn = wh.norm(dim=1) * (1 + 2.0 ** -15)
        en32 = torch.nextafter(e.float(), up); en32 = torch.where(en32.double() >= e, en32, torch.nextafter(en32, up))
        qn32 = torch.nextafter(qn.float(), up); qn32 = torch.where(qn32.double() >= qn, qn32, torch.nextafter(qn32, up))
        q8 = q.to(torch.int64).view(n, 2, 8, SKS, 2, 4, 4)                         # [tile][row half][g][m][k half][t][j]
        frag = q8.permute(0, 3, 2, 5, 4, 1, 6).contiguous()                         # [tile][m][g][t][k half][row half][j]
        c5 = (frag & 31).view(n, NP, 2, 32, 4, 4).permute(0, 1, 3, 2, 4, 5).reshape(n, NP, 32, 8, 4)   # [tile][p][lane][r][j]
        lo4, hb = c5 & 15, c5 >> 4
        sh = (8 * torch.arange(4, device="cuda"))
        Nw = [((lo4[..., 2 * i, :] << sh) | (lo4[..., 2 * i + 1, :] << (sh + 4))).sum(-1) for i in range(4)]
        Pw = (hb << (sh[None, :] + torch.arange(8, device="cuda")[:, None])).sum((-1, -2))
        words = torch.stack(Nw + [Pw], 2)                                           # [tile][p][5][32 lanes]
        words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
        mt = m.to(torch.int64).view(n, 2, 8, NP, 2)                                 # [tile][row half][g][p][mm]
        sw = (mt[:, 0, :, :, 0] | (mt[:, 1, :, :, 0] << 8) | (mt[:, 0, :, :, 1] << 16) | (mt[:, 1, :, :, 1] << 24)).permute(0, 2, 1)   # [tile][p][g]
        sw = torch.where(sw >= 2 ** 31, sw - 2 ** 32, sw).to(torch.int32)
        blk = torch.cat([words.contiguous().view(n, NP, 160), sw.contiguous().view(n, NP, 8)], 2)    # [tile][p][168 words]
        out[r0 // 16:r0 // 16 + n, :NP * 672] = blk.contiguous().view(n, NP * 168).view(torch.uint8)
        hdr = torch.zeros(n, 8, 8, dtype=torch.float32, device="cuda")
        sv, ev, qv = sb.view(n, 2, 8), en32.view(n, 2, 8), qn32.view(n, 2, 8)
        hdr[:, :, 0], hdr[:, :, 1] = sv[:, 0], sv[:, 1]
        hdr[:, :, 2], hdr[:, :, 3] = ev[:, 0], ev[:, 1]
        hdr[:, :, 4], hdr[:, :, 5] = qv[:, 0], qv[:, 1]
        out[r0 // 16:r0 // 16 + n, NP * 672:] = hdr.view(n, 64).view(torch.uint8)
    return out.view(-1)


def pack_tiles(w: torch.Tensor) -> torch.Tensor:
    """int16 [M, K] (BF16 bits) -> tiles [M/16][K/16][32 lanes][4 regs][2 halves] in mma A-operand order."""
    M, K = w.shape
    return w.reshape(M // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous().view(-1)


def pack_fp4(blocks: torch.Tensor, scales: torch.Tensor):
    """MXFP4 [E, M, K/32, 16] u8 codes + [E, M, K/32] u8 scales -> codes [E][M/16][K/64][32 lanes][16 B] and
    scales [E][M/16][K/64][8 g][4 B]; a pure byte permutation (each byte = two e2m1 codes of columns 2i, 2i+1)."""
    E, M, G, _ = blocks.shape
    C = G // 2  # 64-column chunks
    c = blocks.reshape(E, M // 16, 2, 8, C, 4, 2, 4).permute(0, 1, 4, 3, 7, 5, 6, 2).contiguous().view(-1)
    # nibble order inside each 32-bit word for the decode in mk.cu (cvt4): register r = (nibble 3-r, nibble 7-r).
    # old register r = byte r (low nibble, high nibble). A permutation of the nibbles (codes unchanged).
    b = c.view(-1, 4)
    lo, hi = b & 15, b >> 4
    nb = torch.stack([lo[:, 3] | (lo[:, 2] << 4), lo[:, 1] | (lo[:, 0] << 4), hi[:, 3] | (hi[:, 2] << 4), hi[:, 1] | (hi[:, 0] << 4)], 1)
    c = nb.contiguous().view(-1)
    s = scales.reshape(E, M // 16, 2, 8, C, 2).permute(0, 1, 4, 3, 2, 5).contiguous().view(-1)
    return c, s


def fuse_items(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Codes [items][512 B] and scales [items][32 B] (pack_fp4) -> [items][544 B]: each item's codes then its scales
    (a pure byte permutation; one bulk copy per ring stage)."""
    n = c.numel() // 512
    assert s.numel() == n * 32
    return torch.cat([c.view(n, 512), s.view(n, 32)], 1).contiguous().view(-1)


def qk_perm() -> torch.Tensor:
    """Row order inside a q/k head: rowtile j holds dims 8j..8j+7 then 32+8j..32+8j+7 (RoPE pairs in one thread)."""
    r = torch.arange(64)
    j, i = r // 16, r % 16
    return torch.where(i < 8, 8 * j + i, 32 + 8 * j + (i - 8))


class Checkpoint:
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
                self.where[key] = (si, e["data_offsets"][0], e["data_offsets"][1], e["shape"], e["dtype"])
        self.raw = {}

    def shard_of(self, key):
        return self.where[key][0]

    def load_shard(self, si):
        if si in self.raw:
            return
        path = os.path.join(self.dir, self.files[si])
        mm = np.memmap(path, dtype=np.uint8, mode="r", offset=self.data_start[si])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.raw[si] = torch.from_numpy(mm).cuda()
        del mm

    def u8(self, key):
        si, a, b, shape, dt = self.where[key]
        assert dt == "U8", (key, dt)
        return self.raw[si][a:b].view(shape)

    def i16(self, key):
        si, a, b, shape, dt = self.where[key]
        assert dt == "BF16", (key, dt)
        t = self.raw[si][a:b]
        if a % 2:
            t = t.clone()
        return t.view(torch.int16).view(shape)

    def f32(self, key):
        return self.i16(key).contiguous().view(torch.bfloat16).float().contiguous()


class LayerP(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("wqkv", "bqkv", "wo", "bo", "wr", "br", "gu_c", "gu_s", "gu_b",
                                               "dn_c", "dn_s", "dn_b", "ln1", "ln2", "sinks", "kc", "vc")] + [("sliding", ctypes.c_int)]


class Params(ctypes.Structure):
    _fields_ = [("L", LayerP * NL)] + [(n, ctypes.c_void_p) for n in ("embed", "head", "lnf", "cosT", "sinT", "x", "xm", "q",
                                                                      "attn", "act", "rlog", "ypart", "amv", "ami", "bar")] + [("maxctx", ctypes.c_int), ("prof", ctypes.c_void_p), ("prof_cap", ctypes.c_uint), ("skel", ctypes.c_void_p), ("dcnt", ctypes.c_void_p), ("sk", ctypes.c_void_p), ("gth", ctypes.c_void_p), ("clist", ctypes.c_void_p)]


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        if max_batch > MAXB:
            raise ValueError("this engine serves at most %d sequences" % MAXB)
        t0 = time.time()
        # track: engine/TRACK ("A" default: checkpoint formats only; "B": + certified int8 head sketch on the greedy path)
        tf = os.path.join(HERE, "TRACK")
        self.track = (os.environ.get("MK_TRACK") or (open(tf).read().split()[0] if os.path.exists(tf) else "A")).upper()
        if self.track not in ("A", "B"):
            raise ValueError("TRACK must be A or B")
        # track B runs the RFOLD build (libmk_b.so), track A the plain one (libmk.so); MK_LIB overrides
        lib_path = os.path.join(HERE, "build", os.environ.get("MK_LIB") or os.environ.get("MK_MAINLIB") or ("libmk_b.so" if self.track == "B" else "libmk.so"))   # MK_MAINLIB: keeps the per-batch-size builds
        if not os.path.exists(lib_path):
            subprocess.run(["bash", os.path.join(HERE, "build.sh")], check=True)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.mk_create.restype = ctypes.c_void_p
        self.lib.mk_create.argtypes = [ctypes.c_void_p]
        self.lib.mk_run.restype = ctypes.c_int
        self.lib.mk_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
        self.lib.mk_poll.restype = ctypes.c_int
        self.lib.mk_poll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.mk_params_size.restype = ctypes.c_int
        self.lib.mk_take_err.restype = ctypes.c_int
        self.lib.mk_take_err.argtypes = [ctypes.c_void_p]
        assert self.lib.mk_params_size() == ctypes.sizeof(Params), (self.lib.mk_params_size(), ctypes.sizeof(Params))
        # RFOLD build (track B only): router weights folded with the post-attention norm weight (exact fp32 products)
        self.rfold = bool(self.lib.mk_rfold())
        self.fuse = bool(self.lib.mk_fuse()) if hasattr(self.lib, "mk_fuse") else False   # fused MXFP4 items (layout only)
        if self.rfold and self.track != "B":
            raise ValueError("the RFOLD build is a track B change")
        torch.cuda.init()
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        assert cfg["hidden_size"] == HID and cfg["num_hidden_layers"] == NL and cfg["vocab_size"] == VOCAB
        # the attention keeps the scores of a whole window in shared memory: at most 2048 positions
        self.max_context = min(int(max_context), 2048)
        self.maxctx = (self.max_context + 15) // 16 * 16
        self.max_batch = int(max_batch)
        self.keep = []
        P = Params()
        self.P = P
        ck = Checkpoint(model_dir)

        def keep(t):
            self.keep.append(t)
            return t.data_ptr()

        perm = qk_perm().cuda()
        qkv_rows = torch.cat([(h * 64 + perm) for h in range(NQH)] + [(4096 + h * 64 + perm) for h in range(NKV)] +
                             [torch.arange(4608, 5120, device="cuda")])
        items = {l: [k for k in ck.where if k.startswith(f"model.layers.{l}.")] for l in range(NL)}
        items["g"] = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
        last = {i: max(ck.shard_of(k) for k in ks) for i, ks in items.items()}
        need = {si: {i for i, ks in items.items() if any(ck.shard_of(k) == si for k in ks)} for si in range(len(ck.files))}
        done = set()
        cache_elems = MAXB * NKV * self.maxctx * HD
        for si in range(len(ck.files)):
            ck.load_shard(si)
            for i in [i for i in items if i not in done and last[i] <= si]:
                if i == "g":
                    P.embed = keep(ck.i16("model.embed_tokens.weight").contiguous().clone())
                    P.head = keep(pack_tiles(ck.i16("lm_head.weight")))
                    if self.track == "B":
                        P.sk = keep(pack_sketch(ck.i16("lm_head.weight").contiguous()))
                        self.sk5 = keep(pack_sketch5(ck.i16("lm_head.weight").contiguous())) if SK5 else None
                    P.lnf = keep(ck.f32("model.norm.weight"))
                else:
                    l = i
                    L = P.L[l]
                    a = f"model.layers.{l}.self_attn."
                    m = f"model.layers.{l}.mlp."
                    w = torch.cat([ck.i16(a + "q_proj.weight"), ck.i16(a + "k_proj.weight"), ck.i16(a + "v_proj.weight")], 0)
                    L.wqkv = keep(pack_tiles(w[qkv_rows].contiguous()))
                    bq = torch.cat([ck.f32(a + "q_proj.bias"), ck.f32(a + "k_proj.bias"), ck.f32(a + "v_proj.bias")], 0)
                    L.bqkv = keep(bq[qkv_rows].contiguous())
                    L.wo = keep(pack_tiles(ck.i16(a + "o_proj.weight")))
                    L.bo = keep(ck.f32(a + "o_proj.bias"))
                    if self.rfold:
                        wr = ck.f32(m + "router.weight")                        # [32, 2880]
                        ln2 = ck.f32(f"model.layers.{l}.post_attention_layernorm.weight")
                        rf = (wr * ln2[None, :]).t().contiguous()               # [2880, 32] fp32
                        assert torch.equal(rf.double(), (wr.double() * ln2.double()[None, :]).t()), "fold not exact"
                        L.wr = keep(rf)
                    else:
                        L.wr = keep(pack_tiles(ck.i16(m + "router.weight")))
                    L.br = keep(ck.f32(m + "router.bias"))
                    c, s = pack_fp4(ck.u8(m + "experts.gate_up_proj_blocks"), ck.u8(m + "experts.gate_up_proj_scales"))
                    if self.fuse:
                        c = fuse_items(c, s); s = c
                    L.gu_c = keep(c); L.gu_s = keep(s)
                    L.gu_b = keep(ck.f32(m + "experts.gate_up_proj_bias"))
                    c, s = pack_fp4(ck.u8(m + "experts.down_proj_blocks"), ck.u8(m + "experts.down_proj_scales"))
                    if self.fuse:
                        c = fuse_items(c, s); s = c
                    L.dn_c = keep(c); L.dn_s = keep(s)
                    L.dn_b = keep(ck.f32(m + "experts.down_proj_bias"))
                    L.ln1 = keep(ck.f32(f"model.layers.{l}.input_layernorm.weight"))
                    L.ln2 = keep(ck.f32(f"model.layers.{l}.post_attention_layernorm.weight"))
                    L.sinks = keep(ck.f32(a + "sinks"))
                    L.kc = keep(torch.zeros(cache_elems, dtype=torch.int16, device="cuda"))
                    L.vc = keep(torch.zeros(cache_elems, dtype=torch.int16, device="cuda"))
                    L.sliding = 1 if cfg["layer_types"][l] == "sliding_attention" else 0
                done.add(i)
            for sj in list(ck.raw):
                if need[sj] <= done:
                    del ck.raw[sj]
        assert len(done) == len(items)
        del ck
        # rotary tables exactly as the Hugging Face module computes them (float32 on the device, then BF16)
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
        hcfg = AutoConfig.from_pretrained(model_dir)
        rot = GptOssRotaryEmbedding(hcfg).cuda()
        dummy = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
        cos, sin = rot(dummy, torch.arange(self.maxctx, device="cuda")[None, :])
        P.cosT = keep(cos[0].float().contiguous())
        P.sinT = keep(sin[0].float().contiguous())
        assert cos.shape[-1] == 32
        P.x = keep(torch.zeros(MAXB * HID, dtype=torch.int16, device="cuda"))
        P.xm = keep(torch.zeros(MAXB * HID, dtype=torch.int16, device="cuda"))
        P.q = keep(torch.zeros(MAXB * 4096, dtype=torch.int16, device="cuda"))
        P.attn = keep(torch.zeros(MAXB * 4096, dtype=torch.int16, device="cuda"))
        P.act = keep(torch.zeros(32 * MAXB * HID, dtype=torch.int16, device="cuda"))
        P.rlog = keep(torch.zeros(MAXB * NB * (NE + 1) + MAXB * 180, dtype=torch.float32, device="cuda"))
        P.ypart = keep(torch.zeros(2 * 32 * MAXB * HID, dtype=torch.float32, device="cuda"))
        P.amv = keep(torch.zeros(NB * MAXB, dtype=torch.float32, device="cuda"))
        P.ami = keep(torch.zeros(NB * MAXB, dtype=torch.int32, device="cuda"))
        P.bar = keep(torch.zeros(64, dtype=torch.int32, device="cuda"))
        P.maxctx = self.maxctx
        self.prof_cap = int(os.environ.get("MK_PROF_CAP", "0"))
        if self.prof_cap:
            self.prof_buf = torch.zeros(NB * self.prof_cap * 2, dtype=torch.int64, device="cuda")
            P.prof = self.prof_buf.data_ptr(); P.prof_cap = self.prof_cap
        P.skel = keep(torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"))
        P.dcnt = keep(torch.zeros(256, dtype=torch.int32, device="cuda"))
        P.gth = keep(torch.zeros(NB * MAXB, dtype=torch.float32, device="cuda"))
        P.clist = keep(torch.zeros(2 * 12568, dtype=torch.int32, device="cuda"))
        self.hstep = 0
        torch.cuda.synchronize()
        self.h = self.lib.mk_create(ctypes.byref(P))
        if not self.h:
            raise RuntimeError("mk_create failed")
        # batch sizes 1 and 8 run on builds with B a compile-time constant (same arithmetic per sequence)
        self.libB = {}
        for bs in (1, 8):
            name = os.environ.get("MK_B%dLIB" % bs) or (("libmk_b%d.so" if self.track == "B" else "libmk_%d.so") % bs)
            pb = os.path.join(HERE, "build", name)
            if "MK_LIB" in os.environ or os.environ.get("MK_NO_B%d" % bs) is not None or not os.path.exists(pb):
                continue
            libx = ctypes.CDLL(pb)
            for fn in ("mk_create", "mk_run", "mk_poll", "mk_take_err"):
                getattr(libx, fn).restype = getattr(self.lib, fn).restype
                getattr(libx, fn).argtypes = getattr(self.lib, fn).argtypes
            libx.mk_params_size.restype = ctypes.c_int
            assert libx.mk_params_size() == ctypes.sizeof(Params)
            Px = P
            if bs == 1 and self.track == "B" and SK5:   # the B=1 track-B build reads the 5-bit sketch
                Px = Params.from_buffer_copy(P); Px.sk = self.sk5
            hx = libx.mk_create(ctypes.byref(Px))
            if not hx:
                raise RuntimeError("mk_create (B=%d build) failed" % bs)
            self.libB[bs] = (libx, hx)
        self.lib1, self.h1 = self.libB.get(1, (None, None))
        self.pos = 0
        self.steps = 0
        self._pending = []
        self._cohort = None
        self.load_seconds = time.time() - t0

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def _k(self, B):
        return self.libB.get(B, (self.lib, self.h))

    def reset(self) -> None:
        self.pos = 0

    def _check_err(self):
        # the kernel replaces out-of-range token ids by 0 (never reads past the embedding) and flags it; the flag is
        # read without a sync, so an error surfaces at the latest one call later
        if self.lib.mk_take_err(self.h) or any(l.mk_take_err(h) for l, h in self.libB.values()):
            raise ValueError("a token id outside [0, %d) was given (replaced by 0 on the device)" % VOCAB)

    def _check_batch(self, B):
        self._check_err()
        if B < 1 or B > self.max_batch:
            raise ValueError("batch size %d: this engine was created for 1..%d sequences" % (B, self.max_batch))

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos >= self.max_context:
            raise RuntimeError("context is full (%d positions)" % self.max_context)
        tokens = tokens.to(torch.int64).contiguous()
        logits = torch.empty((B, VOCAB), dtype=torch.float32, device="cuda")
        lib, h = self._k(B)
        if lib.mk_run(h, tokens.data_ptr(), 1, logits.data_ptr(), None, B, self.pos, 1, 1, 0, self._stream(), self.hstep):
            raise RuntimeError("kernel launch failed")
        self._tok = tokens
        self.pos += 1
        self.hstep += 1
        self.steps += 1
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos + n_steps > self.max_context:
            raise RuntimeError("context is full")
        tokens = tokens.to(torch.int64).contiguous()
        out = torch.empty((B, n_steps), dtype=torch.int64, device="cuda")
        lib, h = self._k(B)
        if lib.mk_run(h, tokens.data_ptr(), 1, None, out.data_ptr(), B, self.pos, 1, n_steps, 1, self._stream(), self.hstep):
            raise RuntimeError("kernel launch failed")
        self._tok = tokens
        self.pos += n_steps
        self.hstep += n_steps
        self.steps += n_steps
        return out

    # ---- online contract: requests with the same prompt length and answer length form a cohort; one kernel
    # launch runs the prompt and the greedy steps; poll() reads the tokens the kernel has written to host memory.
    GATHER_S = 0.05

    def submit(self, rid, prompt, n) -> None:
        prompt = [int(t) for t in prompt]
        n = int(n)
        if len(prompt) < 1 or n < 1:
            raise ValueError("submit: a request needs at least one prompt token and one new token")
        if len(prompt) - 1 + n > self.max_context:
            raise ValueError("submit: %d prompt tokens + %d new tokens do not fit the context of %d positions"
                             % (len(prompt), n, self.max_context))
        if min(prompt) < 0 or max(prompt) >= VOCAB:
            raise ValueError("submit: token id out of range")
        self._pending.append((rid, prompt, n, time.perf_counter()))

    def poll(self) -> list:
        events = []
        if self._cohort is None:
            if not self._pending:
                return events
            if len(self._pending) < self.max_batch and time.perf_counter() - self._pending[-1][3] < self.GATHER_S:
                return events
            first = self._pending[0]
            Pn, n = len(first[1]), first[2]
            cohort = [r for r in self._pending if len(r[1]) == Pn and r[2] == n][:self.max_batch]
            self._pending = [r for r in self._pending if all(r is not c for c in cohort)]
            ids = torch.tensor([r[1] for r in cohort], dtype=torch.int64, device="cuda")
            B = len(cohort)
            lib, h = self._k(B)
            if lib.mk_run(h, ids.data_ptr(), Pn, None, None, B, 0, Pn, n, 2, self._stream(), self.hstep):
                raise RuntimeError("kernel launch failed")
            self.pos = Pn - 1 + n
            self.hstep += n
            self.steps += Pn - 1 + n
            self._cohort = {"rids": [r[0] for r in cohort], "n": n, "sent": 0, "ids": ids, "B": B, "k": (lib, h),
                            "buf": np.zeros((B, n), dtype=np.int32)}
            return events
        c = self._cohort
        lib, h = c["k"]
        done = lib.mk_poll(h, c["buf"].ctypes.data_as(ctypes.c_void_p), c["B"], c["n"])
        if done > c["sent"]:
            for i, rid in enumerate(c["rids"]):
                events.append((rid, c["buf"][i, c["sent"]:done].tolist(), done == c["n"]))
            c["sent"] = done
            if done == c["n"]:
                self._cohort = None
        return events

    def close(self) -> None:
        torch.cuda.synchronize()
        self.keep = []

    def stats(self) -> dict:
        return {"implementation": "gpt-oss-20b megakernel v1 (grid barriers, sm_90)", "track": self.track, "head": "certified 6-bit sketch on the greedy path" if self.track == "B" else "exact", "steps": self.steps,
                "load_seconds": round(self.load_seconds, 1)}
