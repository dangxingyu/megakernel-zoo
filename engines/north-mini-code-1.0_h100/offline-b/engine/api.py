"""North-Mini-Code-1.0 decode engine: one persistent CUDA kernel per step (engine/mk.cu).

This file only loads the checkpoint into the kernel's memory layout and launches the kernel; no model code of
any library runs in the timed path.

Layout (track A: every weight keeps its BF16 bits; only the order in memory changes): each matrix is stored in
tensor-core fragment order, 16x16 tiles, 512 bytes per tile, so that one 16-byte load per thread feeds one mma.
"""
import ctypes
import numpy as np
import json
import os
import struct
import time

import torch

HID, NLAYER, TOPK, MAXB, MAXU, VOCAB = 2048, 49, 8, 8, 64, 262144
PART_ROWS, HMAX = 5248, 3072
HERE = os.path.dirname(os.path.abspath(__file__))


class LayerW(ctypes.Structure):
    _fields_ = [("qkv", ctypes.c_void_p), ("router", ctypes.c_void_p), ("o", ctypes.c_void_p), ("expert", ctypes.c_void_p),
                ("normw", ctypes.c_void_p), ("exp_stride", ctypes.c_uint64),
                ("gu_tiles", ctypes.c_int), ("down_K", ctypes.c_int), ("rope", ctypes.c_int), ("dense", ctypes.c_int)]


SKETCH_DEFAULT = "1"   # "1": the certified sketch head is used on the greedy path (track B)
PTRS = ["head", "embed", "normf", "rope", "kc", "vc", "x", "qkvo", "qkvp", "rl", "attn", "attp", "h", "logits", "candv",
        "candi", "curtok", "zeros", "ctl", "dyn", "l0qkv", "l0mlp", "l0m", "sk", "dbg_x", "force", "prof", "trace", "role"]


class Model(ctypes.Structure):
    _fields_ = [("L", LayerW * NLAYER)] + [(n, ctypes.c_void_p) for n in PTRS] + [("max_ctx", ctypes.c_int), ("maxK", ctypes.c_int)]


def frag(w: torch.Tensor) -> torch.Tensor:
    """[..., M, N] -> mma A-operand order [..., M/16, N/16, 32 lanes, 4 regs, 2]: lane = 4*(row%8) + (col%8)//2,
    reg = 2*((col%16)//8) + (row%16)//8."""
    lead = w.shape[:-2]
    M, N = w.shape[-2:]
    v = w.reshape(*lead, M // 16, 2, 8, N // 16, 2, 4, 2)
    n = len(lead)
    return v.permute(*range(n), n + 0, n + 3, n + 2, n + 5, n + 4, n + 1, n + 6)


def tiles_bf16(w: torch.Tensor) -> torch.Tensor:
    """Track A tiles of a [M, N] BF16 matrix: [M/16, N/16 * 512] bytes, the checkpoint's bits in fragment order."""
    M, N = w.shape
    return frag(w.contiguous().view(torch.int16)).reshape(M // 16, -1).contiguous().view(torch.uint8)


def exb(K: int) -> int:
    return 128 if K == 48 else 256


# High bits of a k-step pair (sign and exponent-code bits 1..3 of 16 weights per lane) as two 32-bit words. Register
# q of the thread (weights 2q, 2q + 1: bf16 halves 0 and 1) lives in word q >> 2, stored rotated so that the kernel
# gets its bits in place with one rotation and one mask: rotl(word, HI_ROT[q & 3]) & 0x87008700 = the sign (bits
# 15, 31) and code bits 1..3 (bits 8..10, 24..26) of both halves. The four rotations place the four registers of a
# word on disjoint bits.
HI_ROT = (0, 4, 8, 28)
HI_POS = ((8, 8), (9, 9), (10, 10), (15, 15))   # (bit of the bf16 value, bit of the half in the register), per field


def hi_pack(f: torch.Tensor) -> torch.Tensor:
    """f: [..., 16] int32 bf16 patterns of a lane's pair (value n = 2q + half) -> [..., 8] uint8."""
    words = torch.zeros(f.shape[:-1] + (2,), dtype=torch.int64, device=f.device)
    for n in range(16):
        q, h = n // 2, n % 2
        w, rot = q >> 2, HI_ROT[q & 3]
        v = f[..., n].to(torch.int64)
        for vb, rb in HI_POS:
            pos = (rb + 16 * h - rot) % 32
            words[..., w] |= ((v >> vb) & 1) << pos
    b = torch.stack([(words >> (8 * k)) & 0xFF for k in range(4)], -1)   # little endian
    return b.reshape(f.shape[:-1] + (8,)).to(torch.uint8)


def hi_unpack(hi8: torch.Tensor) -> torch.Tensor:
    """Inverse of hi_pack: [..., 8] uint8 -> [..., 16] int32 with only the sign and code bits 1..3 set."""
    b = hi8.to(torch.int64).reshape(hi8.shape[:-1] + (2, 4))
    words = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
    out = []
    for n in range(16):
        q, h = n // 2, n % 2
        w, rot = q >> 2, HI_ROT[q & 3]
        v = torch.zeros_like(words[..., 0])
        for vb, rb in HI_POS:
            pos = (rb + 16 * h - rot) % 32
            v |= ((words[..., w] >> pos) & 1) << vb
        out.append(v)
    return torch.stack(out, -1).to(torch.int32)


def tiles_w12(w: torch.Tensor, cshift: torch.Tensor | None = None, hdr_shift: torch.Tensor | None = None, verify: bool = False):
    """Track B tiles of a [M, N] BF16 matrix (see engine/mk.cu): [M/16, N/16 * 384 + exb] bytes and (escapes, weights).
    W[i][j] = 2^(r_i + c_j) V[i][j]; V keeps sign and mantissa and has the exponent code 1..15 (0: the weight is 0
    in the tile). cshift: [M, N]-broadcastable integer c_j per element (None: 0). hdr_shift: [M/16, 8] int8 written to
    header bytes 16..23 (the gate|up tiles carry their neurons' down-column shifts there).
    Header: bytes 0..15 r_i, 16..23 shifts, 24..31 the end of each row group's entries (entries are sorted by
    row % 8), then the entries [value:16 | k-step:8 | row:4 | col:4]; an entry's value is the true weight divided by
    2^c_j, as BF16 bits."""
    M, N = w.shape
    R, K = M // 16, N // 16
    EXB, dev = exb(K), w.device
    EXN = (EXB - 32) // 4
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    E = (bits >> 7) & 0xFF
    nz = (bits & 0x7FFF) != 0
    assert not bool(((E == 0) & nz).any()) and not bool((E == 255).any()), "denormal, infinite or NaN weight"
    Ea = E - cshift.to(torch.int32) if cshift is not None else E
    # the window of 15 exponent values that covers most of each row
    hist = torch.zeros((M, 512), dtype=torch.int32, device=dev)
    hist.scatter_add_(1, torch.where(nz, Ea + 128, torch.zeros_like(Ea)).long(), torch.ones_like(Ea))
    hist[:, 0] = 0
    cs = torch.cat([torch.zeros((M, 1), dtype=torch.int32, device=dev), hist.cumsum(1, dtype=torch.int32)], 1)
    lo = (cs[:, 15:] - cs[:, :-15]).argmax(1).to(torch.int32)          # window [lo, lo + 15) in the shifted index
    del hist, cs
    r = lo - 128 - 1                                                    # code = Ea - r in 1..15
    assert int(r.min()) >= 0 and int(r.max()) <= 222
    code = Ea - r[:, None]
    inw = nz & (code >= 1) & (code <= 15)
    scaled = torch.where(inw, (bits & 0x8000) | (code << 7) | (bits & 0x7F), torch.zeros_like(bits))
    esc = nz & ~inw
    idx = esc.nonzero()
    row, col = idx[:, 0], idx[:, 1]
    # within a tile the entries are sorted by row % 8 (the lanes of one row pair read only their own)
    order = torch.argsort((row // 16) * 8 + (row % 8), stable=True)
    row, col = row[order], col[order]
    tile = row // 16
    counts = torch.bincount(tile, minlength=R)
    assert int(counts.max()) <= EXN, f"a tile has {int(counts.max())} escapes, the header holds {EXN}"
    ea = Ea[row, col]
    assert idx.shape[0] == 0 or (int(ea.min()) >= 1 and int(ea.max()) <= 254)
    val = (bits[row, col] & 0x807F) | (ea << 7)
    entry = (val << 16) | ((col // 16) << 8) | ((row % 16) << 4) | (col % 16)
    rank = torch.arange(idx.shape[0], device=dev) - (counts.cumsum(0) - counts)[tile]
    hdr = torch.zeros((R, EXB), dtype=torch.uint8, device=dev)
    hdr[:, 0:16] = r.view(R, 16).to(torch.uint8)
    if hdr_shift is not None:
        hdr[:, 16:24] = hdr_shift.to(torch.int8).view(torch.uint8)
    grp = torch.bincount(tile * 8 + (row % 8), minlength=R * 8).view(R, 8)
    hdr[:, 24:32] = grp.cumsum(1).to(torch.uint8)                       # where each row group's entries end
    h32 = torch.zeros((R, EXN), dtype=torch.int32, device=dev)
    h32[tile, rank] = entry.to(torch.int32)
    hdr[:, 32:] = h32.view(torch.uint8)
    # per thread and pair of k-steps: 16 low bytes, 16 high nibbles
    f = frag(scaled.to(torch.int16)).reshape(R, K // 2, 2, 32, 8).permute(0, 1, 3, 2, 4).reshape(R, K // 2, 32, 16).to(torch.int32)
    lo8 = (f & 0xFF).to(torch.uint8)
    hi8 = hi_pack(f)
    out = torch.cat([torch.cat([lo8.reshape(R, K // 2, 512), hi8.reshape(R, K // 2, 256)], 2).reshape(R, K * 384), hdr], 1)
    if verify:
        back = untile_w12(out, M, N, cshift)
        assert bool((back == bits).all()), "12-bit tiles do not decode to the checkpoint's bits"
    return out, int(idx.shape[0]), int(nz.sum())


def untile_w12(t: torch.Tensor, M: int, N: int, cshift: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse of tiles_w12: the BF16 bit patterns [M, N] (int32), from the tile bytes alone (plus the column shifts)."""
    R, K = M // 16, N // 16
    EXB = exb(K)
    data, hdr = t[:, :K * 384].reshape(R, K // 2, 768), t[:, K * 384:]
    lo8 = data[:, :, :512].reshape(R, K // 2, 32, 16).to(torch.int32)
    hi8 = data[:, :, 512:].reshape(R, K // 2, 32, 8).to(torch.int32)
    sc = lo8 | hi_unpack(hi8)                                            # sign | code << 7 | mantissa
    # back from fragment order: [R, K/2, 32, 16] -> [R, K, 32, 4, 2] -> [M, N]
    v = sc.reshape(R, K // 2, 32, 2, 8).permute(0, 1, 3, 2, 4).reshape(R, K, 8, 4, 2, 2, 2)   # R, K, g, t, ch, rh, j
    sc = v.permute(0, 5, 2, 1, 4, 3, 6).reshape(M, N)
    r = hdr[:, 0:16].to(torch.int32).reshape(M)
    code = (sc >> 7) & 0xFF
    Ea = code + r[:, None]
    E = Ea + cshift.to(torch.int32) if cshift is not None else Ea
    bits = torch.where(code > 0, (sc & 0x807F) | (E << 7), torch.zeros_like(sc))
    h32 = hdr[:, 32:].contiguous().view(torch.int32)
    counts = hdr[:, 31].long()
    EXN = h32.shape[1]
    live = torch.arange(EXN, device=t.device)[None, :] < counts[:, None]
    tl, k = live.nonzero()[:, 0], live.nonzero()[:, 1]
    e = h32[tl, k].to(torch.int64) & 0xFFFFFFFF
    row = tl * 16 + ((e >> 4) & 15)
    col = ((e >> 8) & 255) * 16 + (e & 15)
    val = (e >> 16).to(torch.int32)
    ev = (val >> 7) & 0xFF
    if cshift is not None:
        ev = ev + cshift.to(torch.int32).expand(M, N)[row, col]
    bits[row, col] = (val & 0x807F) | (ev << 7)
    return bits


def read_shard(path: str, device: str) -> dict:
    """All tensors of one safetensors shard as BF16 views of a single device buffer."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        import numpy as np
        raw = np.fromfile(f, dtype=np.int16)
    buf = torch.from_numpy(raw).to(device)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "BF16", (name, info["dtype"])
        a, b = info["data_offsets"]
        out[name] = buf[a // 2:b // 2].view(torch.bfloat16).view(*info["shape"])
    return out


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        assert max_batch <= MAXB, "this engine is built for batches up to 8"
        t0 = time.time()
        self.dev = "cuda"
        torch.cuda.init()
        self.lib = ctypes.CDLL(os.path.join(HERE, "libmk.so"))
        assert self.lib.mk_sizeof_model() == ctypes.sizeof(Model), (self.lib.mk_sizeof_model(), ctypes.sizeof(Model))
        self.max_ctx = max(128, (max_context + 127) // 128 * 128)
        assert self.max_ctx <= 2048, "contexts beyond 2048 tokens are not implemented in this build"
        self.maxK = self.max_ctx // 16
        self.keep = {}
        m = Model()
        # ---- arena layout (bytes). Track A: 512 B per k-step; track B: 384 B per k-step + a header per tile.
        self.w12 = os.environ.get("MK_TRACK", open(os.path.join(HERE, "TRACK")).read().strip()) == "B"
        self.l0tab = self.w12 and os.environ.get("MK_L0TAB", "1") != "0"   # layer 0 from token tables (track B)
        self.skh = self.w12 and os.environ.get("MK_SKH", SKETCH_DEFAULT) != "0"       # certified int8 sketch of the head on the greedy path (track B)
        assert bool(self.lib.mk_is_w12()) == self.w12, "libmk.so was built for the other track: run engine/build.sh"
        verify = bool(os.environ.get("MK_VERIFY"))
        tby = (lambda K: K * 384 + exb(K)) if self.w12 else (lambda K: K * 512)
        geo = [(1, 384, 192)] + [(128, 96, 48)] * (NLAYER - 1)  # experts, gate|up tiles, down K
        off, lay = 0, []
        for l in range(NLAYER):
            ne, gt, dk = geo[l]
            stride = gt * tby(128) + 128 * tby(dk)
            d = {"qkv": off}; off += 320 * tby(128)
            d["router"] = off; off += 8 * tby(128) if l else 0
            d["o"] = off; off += 128 * tby(256)
            d["expert"] = off; off += ne * stride
            d["stride"] = stride
            lay.append(d)
        head_off = off; off += (VOCAB // 16) * tby(128)
        arena = torch.empty(off + 4096, dtype=torch.uint8, device=self.dev)
        self.keep["arena"] = arena
        base = arena.data_ptr()
        normw = torch.empty((NLAYER + 1, HID), dtype=torch.bfloat16, device=self.dev)
        embed = torch.empty((VOCAB, HID), dtype=torch.bfloat16, device=self.dev)
        self.keep.update(normw=normw, embed=embed)
        st = {"escapes": 0, "weights": 0, "dead_neurons": 0, "min_shift": 0}

        def put(o, w, cshift=None, hdr_shift=None):
            """Tiles of the [M, N] matrix w at arena offset o; returns the bytes written."""
            if self.w12:
                t, ne_, nw_ = tiles_w12(w, cshift, hdr_shift, verify)
                st["escapes"] += ne_; st["weights"] += nw_
            else:
                t = tiles_bf16(w)
            t = t.reshape(-1)
            arena[o:o + t.numel()] = t
            return t.numel()

        def layer_ready(l, have):
            p = f"model.layers.{l}."
            if p + "self_attn.q_proj.weight" not in have or p + "self_attn.o_proj.weight" not in have or p + "input_layernorm.weight" not in have:
                return False
            if p + "self_attn.k_proj.weight" not in have or p + "self_attn.v_proj.weight" not in have:
                return False
            if l == 0:
                return all(p + f"mlp.{k}_proj.weight" in have for k in ("gate", "up", "down"))
            if p + "mlp.gate.weight" not in have:
                return False
            return all(p + f"mlp.experts.{e}.{k}_proj.weight" in have for e in (0, 63, 127) for k in ("gate", "up", "down")) and \
                sum(1 for k in have if k.startswith(p + "mlp.experts.")) == 384

        def do_layer(l, have):
            p = f"model.layers.{l}."
            g = lambda n: have.pop(p + n)
            d = lay[l]
            qkv0 = torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")], 0)
            put(d["qkv"], qkv0)
            put(d["o"], g("self_attn.o_proj.weight"))
            normw[l] = g("input_layernorm.weight")
            ne, gt, dk = geo[l]
            if l == 0 and self.l0tab:
                self._layer0_tables(embed, normw[0], qkv0, have[p + "mlp.gate_proj.weight"], have[p + "mlp.up_proj.weight"], have[p + "mlp.down_proj.weight"])
            if l > 0:
                put(d["router"], g("mlp.gate.weight"))
            nn = gt * 8                                           # neurons per expert
            for e0 in range(0, ne, 32):                           # experts in groups of 32 (memory)
                es = range(e0, min(ne, e0 + 32))
                n = len(es)
                name = (lambda e, k: f"mlp.{k}_proj.weight") if l == 0 else (lambda e, k: f"mlp.experts.{e}.{k}_proj.weight")
                gate = torch.stack([g(name(e, "gate")) for e in es])
                up = torch.stack([g(name(e, "up")) for e in es])
                down = torch.stack([g(name(e, "down")) for e in es])       # [n, HID, nn]
                gu = torch.stack([gate.view(n, gt, 8, HID), up.view(n, gt, 8, HID)], 2).reshape(n * gt * 16, HID)
                del gate, up
                cd = hs = None
                if self.w12:
                    # one shift per neuron: its down column's median exponent against the expert's largest (1210 neurons in
                    # 14 experts have down columns 2^20 .. 2^62 below the rest: study/dead_neurons.py)
                    ed = ((down.view(torch.int16).to(torch.int32) >> 7) & 0xFF).float()
                    med = ed.median(1).values                              # [n, nn]
                    cdn = (med - med.max(1, keepdim=True).values).round().to(torch.int32).clamp(-90, 0)   # (h is stored times 2^(96 + shift))
                    del ed
                    st["dead_neurons"] += int((cdn <= -8).sum()); st["min_shift"] = min(st["min_shift"], int(cdn.min()))
                    cd = cdn[:, None, :].expand(n, HID, nn).reshape(n * HID, nn)
                    hs = cdn.reshape(n * gt, 8)
                tg = put_view(gu, None, hs)
                try:
                    td = put_view(down.reshape(n * HID, nn), cd, None)
                except AssertionError as err:
                    raise AssertionError(f"layer {l} experts {e0}..: {err}; shifts {int(cdn.min())}..{int(cdn.max())}")
                del gu, down
                blk = torch.cat([tg.view(n, -1), td.view(n, -1)], 1).reshape(-1)
                o = d["expert"] + e0 * d["stride"]
                arena[o:o + blk.numel()] = blk
                del blk, tg, td

        def put_view(w, cshift, hdr_shift):
            if self.w12:
                t, ne_, nw_ = tiles_w12(w, cshift, hdr_shift, verify)
                st["escapes"] += ne_; st["weights"] += nw_
                return t
            return tiles_bf16(w)

        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        shards = sorted(set(index.values()))
        have, todo = {}, list(range(NLAYER))
        for s in shards:
            have.update(read_shard(os.path.join(model_dir, s), self.dev))
            if "model.embed_tokens.weight" in have:
                e = have.pop("model.embed_tokens.weight")
                embed.copy_(e)
                if self.skh:
                    self._sketch_head(e)
                for r0 in range(0, VOCAB, 32768):
                    put(head_off + r0 // 16 * tby(128), e[r0:r0 + 32768])
                del e
            if "model.norm.weight" in have:
                normw[NLAYER] = have.pop("model.norm.weight")
            while todo and layer_ready(todo[0], have):
                do_layer(todo.pop(0), have)
        assert not todo and not have, (todo, list(have)[:5])
        del have
        torch.cuda.empty_cache()
        self.pack_stats = dict(st, verified_bit_for_bit=verify, arena_bytes=off)
        # ---- tables and scratch
        K = self.keep
        inv_freq = 1.0 / (50000 ** (torch.arange(0, 128, 2, dtype=torch.float, device=self.dev) / 128))
        fr = torch.arange(self.max_ctx, device=self.dev)[:, None].float() * inv_freq[None]
        K["rope"] = torch.stack([fr.cos().to(torch.bfloat16).float(), fr.sin().to(torch.bfloat16).float()], -1).contiguous()
        z = lambda *shape, dtype=torch.int16: torch.zeros(shape, dtype=dtype, device=self.dev)
        kvn = NLAYER * MAXB * 4 * self.maxK * 256 * 8
        K["kc"], K["vc"] = z(kvn), z(kvn)
        K["x"] = z(MAXB * HID)
        K["qkvo"] = z(MAXB * 5120)
        K["qkvp"] = z(2 * MAXB * 5120, dtype=torch.float32)
        K["rl"] = z(2 * MAXB * 128 * 2, dtype=torch.int32)
        K["attn"] = z(MAXB * 4096)
        K["attp"] = z(32 * 8 * 1040, dtype=torch.int64)   # (8-byte words: value and stamp, ATT_STAMP)
        K["h"] = z(MAXU * MAXB * HMAX + 4096)
        K["logits"] = z(MAXB * VOCAB, dtype=torch.float32)
        K["candv"] = z(MAXB * 512, dtype=torch.float32)   # exact candidates per block, then the sketch's lower bounds per block
        K["candi"] = z(MAXB * 256, dtype=torch.int32)
        K["curtok"] = z(MAXB, dtype=torch.int64)
        K["zeros"] = z(16384)
        K["ctl"] = z(self.lib.mk_sizeof_ctl(), dtype=torch.uint8)
        K["dyn"] = z(NLAYER * self.lib.mk_sizeof_dyn(), dtype=torch.uint8)
        K["dyn"][:4] = torch.tensor([1, 0, 0, 0], dtype=torch.uint8)  # layer 0 is dense: one "expert", slot 0
        K["modeldev"] = z(ctypes.sizeof(Model) + 64, dtype=torch.uint8)
        for l in range(NLAYER):
            ne, gt, dk = geo[l]
            L, d = m.L[l], lay[l]
            L.qkv, L.router, L.o, L.expert = base + d["qkv"], base + d["router"], base + d["o"], base + d["expert"]
            L.normw = normw[l].data_ptr()
            L.exp_stride = d["stride"]
            L.gu_tiles, L.down_K, L.dense = gt, dk, int(l == 0)
            L.rope = int(l == 0 or l % 4 != 0)
        m.head, m.embed, m.normf = base + head_off, embed.data_ptr(), normw[NLAYER].data_ptr()
        for n in PTRS[3:20]:  # rope .. dyn
            setattr(m, n, K[n].data_ptr())
        K["l0m"] = z(MAXB * HID)
        m.l0m = K["l0m"].data_ptr()
        m.l0qkv = K["l0qkv"].data_ptr() if self.l0tab else None
        m.l0mlp = K["l0mlp"].data_ptr() if self.l0tab else None
        m.sk = K["sk"].data_ptr() if self.skh else None
        m.dbg_x, m.force, m.prof, m.trace = None, None, None, None
        m.role = None
        m.max_ctx, m.maxK = self.max_ctx, self.maxK
        self.m = m
        self._commit()
        if os.environ.get("MK_ROLES", "0") != "0":
            self._roles()
        self.pos = 0
        self.pending, self.active, self.max_batch = [], None, max_batch
        self.calls = {"step": 0, "generate": 0}
        self.load_seconds = time.time() - t0

    @torch.no_grad()
    def _layer0_tables(self, embed, nw, wqkv, wg, wu, wd):
        """Track B: what layer 0 computes from the token alone, for every token. The layer's input is the embedding
        row, and attention and MLP both read its RMSNorm (parallel block), so q|k|v before RoPE and the whole MLP
        output are functions of the token id. Same rounding points as the kernel (float32 sums, BF16 activations)."""
        bf = torch.bfloat16
        tq = torch.empty((VOCAB, 5120), dtype=bf, device=self.dev)
        tm = torch.empty((VOCAB, HID), dtype=bf, device=self.dev)
        wqkv, wg, wu, wd, nwf = wqkv.float(), wg.float(), wu.float(), wd.float(), nw.float()
        for r0 in range(0, VOCAB, 8192):
            x = embed[r0:r0 + 8192].float()
            rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            xn = (nwf * (x * rs).to(bf).float()).to(bf).float()
            tq[r0:r0 + 8192] = (xn @ wqkv.T).to(bf)
            gt = (xn @ wg.T).to(bf).float()
            up = (xn @ wu.T).to(bf).float()
            h = ((gt / (1.0 + torch.exp(-gt))).to(bf).float() * up).to(bf).float()
            tm[r0:r0 + 8192] = (h @ wd.T).to(bf)
        self.keep["l0qkv"], self.keep["l0mlp"] = tq, tm

    @torch.no_grad()
    def _sketch_head(self, e):
        """Track B, greedy path: an int8 sketch of the head with a certified per-row error bound (the kernel's head
        stage says how it is used). Row i: s_i = max|w_i| / 127 as float32, q_i = round(w_i / s_i) in [-127, 127].
        Stored per 16-row tile in the fragment order of mma m16n8k32 (s8), found by bench/mma8_probe.cu: thread
        (g, t) of k-step m holds a0 = row g, columns 32 m + 8 t + 0..3; a1 = row g + 8, the same columns; a2 = row g,
        columns 32 m + 8 t + 4..7; a3 = row g + 8, those columns. Header (256 B): per row pair (g, g + 8): s, s', en, en', qn, qn', 0, 0 (float32), with
        en >= ||w - s q|| + 2^-10 ||w|| and qn >= ||s q||, all norms in float64 and rounded up."""
        T = VOCAB // 16
        sk = torch.empty((T, 64 * 512 + 256), dtype=torch.uint8, device=self.dev)
        inf = torch.tensor(float("inf"), device=self.dev)
        up32 = lambda v: torch.nextafter(v.float(), inf)          # a float32 not below the float64 value
        worst = 0.0
        for r0 in range(0, VOCAB, 16384):
            w = e[r0:r0 + 16384].double()
            R = w.shape[0]
            s = (w.abs().amax(1) / 127).float().clamp_min(1e-30)
            sd = s.double()[:, None]
            q = torch.round(w / sd).clamp(-127, 127)
            en = (w - sd * q).pow(2).sum(1).sqrt()
            wn = w.pow(2).sum(1).sqrt()
            qn = (sd * q).pow(2).sum(1).sqrt()
            worst = max(worst, float((en / wn.clamp_min(1e-30)).max()))
            en32 = up32(en * (1 + 1e-6) + wn * 2.0 ** -10)
            qn32 = up32(qn * (1 + 1e-6))
            t = R // 16
            # [tile, row half h, g, k-step m, t, column half c, i] -> [tile, m, g, t, register c * 2 + h, byte i]
            a = q.to(torch.int8).view(t, 2, 8, 64, 4, 2, 4).permute(0, 3, 2, 4, 5, 1, 6).reshape(t, 64 * 512)
            hd = torch.zeros((t, 8, 8), dtype=torch.float32, device=self.dev)
            for k, v in enumerate((s, en32, qn32)):
                v = v.view(t, 2, 8)
                hd[:, :, 2 * k], hd[:, :, 2 * k + 1] = v[:, 0], v[:, 1]
            sk[r0 // 16:r0 // 16 + t, :64 * 512] = a.view(torch.uint8)
            sk[r0 // 16:r0 // 16 + t, 64 * 512:] = hd.view(t, 64).view(torch.uint8).view(t, 256)
            del w, q, a
        self.keep["sk"] = sk
        self.sketch_rel_err = worst      # largest ||w - s q|| / ||w|| over the rows

    @torch.no_grad()
    def _roles(self):
        """Which SMs stream fastest on this card (they are a fixed set of %smid values), and the schedule's roles mapped
        onto SMs: the routing chain (leader, router blocks) on the fastest ones. Falls back to the block index when
        the measurement is not consistent (every launch must see the same 132 SM ids)."""
        W = self.keep["arena"]
        tpw = min(24, W.numel() // (2112 * 65536) - 1)
        out = torch.zeros(264, dtype=torch.int64, device=self.dev)
        speed, ids = {}, None
        for rep in range(6):
            assert self.lib.mk_calibrate(ctypes.c_void_p(W.data_ptr()), tpw, ctypes.c_void_p(out.data_ptr())) == 0
            o = out.cpu().numpy()
            sm, cyc = o[:132], o[132:].astype(np.float64)
            if ids is None: ids = set(sm.tolist())
            if set(sm.tolist()) != ids or len(ids) != 132 or max(ids) > 255:
                self.role_info = {"used": False, "reason": "SM ids differ between launches"}; return
            for s_, c in zip(sm.tolist(), cyc.max() / cyc): speed.setdefault(s_, []).append(c)
        sp = {s_: float(np.median(v)) for s_, v in speed.items()}
        order = sorted(sp, key=lambda s_: -sp[s_])           # fastest first
        mode = os.environ.get("MK_ROLES", "1")
        if mode == "2": order = sorted(sp)                   # fixed placements (experiments): by SM id, ascending,
        elif mode == "3": order = sorted(sp, reverse=True)   # descending,
        elif mode.startswith("s"): order = sorted(sp); np.random.default_rng(int(mode[1:])).shuffle(order)   # seeded shuffle
        med = float(np.median(list(sp.values())))
        fast = [s_ for s_ in order if sp[s_] > 1.10 * med]
        # roles that start the routing chain first: the leader (0), then the router half-tile blocks (116..131);
        # then everything else in its natural order
        critical = [0] + list(range(116, 132))
        rest = [r for r in range(132) if r not in critical]
        roles = critical + rest
        table = np.zeros(256, dtype=np.int16)
        for s_, r in zip(order, roles): table[s_] = r
        t = torch.from_numpy(table).to(self.dev)
        self.keep["role"] = t
        self.m.role = t.data_ptr()
        self._commit()
        self.role_info = {"used": True, "fast_sms": sorted(fast), "fast_speed": round(float(np.mean([sp[s_] for s_ in fast])) / med, 3) if fast else 1.0}

    def _commit(self):
        torch.cuda.synchronize()
        rc = self.lib.mk_init(ctypes.byref(self.m), ctypes.c_void_p(self.keep["modeldev"].data_ptr()), self.max_ctx)
        assert rc == 0, rc
        self.keep["ctl"].zero_()
        torch.cuda.synchronize()

    def reset(self) -> None:
        self.pos = 0

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        if tokens.dtype != torch.int64:
            tokens = tokens.to(torch.int64)
        logits = torch.empty((B, VOCAB), dtype=torch.float32, device=self.dev)
        rc = self.lib.mk_step(ctypes.c_void_p(tokens.data_ptr()), tokens.stride(0), ctypes.c_void_p(logits.data_ptr()), B, self.pos)
        assert rc == 0
        self.pos += 1
        self.calls["step"] += 1
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        if tokens.dtype != torch.int64:
            tokens = tokens.to(torch.int64)
        out = torch.empty((B, n_steps), dtype=torch.int64, device=self.dev)
        rc = self.lib.mk_generate(ctypes.c_void_p(tokens.data_ptr()), tokens.stride(0), ctypes.c_void_p(out.data_ptr()), B, self.pos, n_steps)
        assert rc == 0
        self.pos += n_steps
        self.calls["generate"] += 1
        return out

    # ---- the online contract of the HTTP server: the whole decode of a cohort is one kernel launch on the device
    GATHER_S = 0.03   # how long the first request of a cohort waits for others that arrive with it

    def submit(self, rid: int, prompt, n: int) -> None:
        self.pending.append((int(rid), [int(t) for t in prompt], int(n), time.perf_counter()))

    def poll(self):
        if self.active is None:
            if not self.pending:
                return []
            first = self.pending[0]
            group = [r for r in self.pending if len(r[1]) == len(first[1]) and r[2] == first[2]][:self.max_batch]
            if len(group) < self.max_batch and time.perf_counter() - first[3] < self.GATHER_S:
                return []
            self.pending = [r for r in self.pending if all(r is not g for g in group)]
            B, P, n = len(group), len(first[1]), first[2]
            assert 1 <= P and P + n <= self.max_ctx, "prompt + new tokens exceed the engine's context"
            prompt = torch.tensor([r[1] for r in group], dtype=torch.int64, device=self.dev)
            out = torch.empty((n, B), dtype=torch.int64, device=self.dev)
            torch.cuda.synchronize()
            rc = self.lib.mk_serve_start(ctypes.c_void_p(prompt.data_ptr()), P, ctypes.c_void_p(out.data_ptr()), B, P, n)
            assert rc == 0
            self.host = (ctypes.c_int64 * (n * B))()
            self.active = {"rids": [r[0] for r in group], "B": B, "n": n, "sent": 0, "prompt": prompt, "out": out}
            self.pos = P - 1 + n
            self.calls["serve"] = self.calls.get("serve", 0) + 1
            return []
        a = self.active
        prog = self.lib.mk_serve_poll()
        if prog <= a["sent"]:
            return []
        k0, B = a["sent"], a["B"]
        rc = self.lib.mk_serve_fetch(ctypes.c_void_p(a["out"].data_ptr()), ctypes.byref(self.host, k0 * B * 8), B, k0, prog)
        assert rc == 0
        done = prog == a["n"]
        events = [(rid, [self.host[k * B + i] for k in range(k0, prog)], done) for i, rid in enumerate(a["rids"])]
        a["sent"] = prog
        if done:
            self.active = None
        return events

    def close(self) -> None:
        torch.cuda.synchronize()
        self.keep = {}

    def stats(self) -> dict:
        impl = ("megakernel v7, track B: 12-bit weights (row exponents, neuron shifts, exact escapes)" + (", layer 0 from token tables" if self.l0tab else "") + (", certified int8 sketch head on the greedy path" if self.skh else "") if self.w12
                else "megakernel v6, track A: BF16 weights in mma fragment order")
        return {"implementation": impl, "load_seconds": self.load_seconds, "pack": self.pack_stats, "layer0_tables": self.l0tab, "sketch_head": self.skh, "roles": getattr(self, "role_info", None), **self.calls}
