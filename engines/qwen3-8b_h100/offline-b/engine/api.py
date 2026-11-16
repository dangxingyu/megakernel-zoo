"""Qwen3-8B decode engine: one CUDA megakernel (engine/mk.cu) behind the contract.

    Engine(model_dir, max_batch, max_context)
    reset()                      forget every sequence
    step(tokens[B] int64 cuda)   append one token per sequence, return next-token logits [B, vocab] on cuda

Python only loads the weights, re-lays them out for the kernel (once) and launches the kernel; no torch
module and no library call computes anything in a step.
"""

import ctypes
import json
import os
import struct
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H, NL, VOCAB, INNER = 4096, 36, 151936, 12288
MODE = int(os.environ.get("MK_MODE", "1"))  # 0: one launch per operator, 1: one launch per step


def _load_lib():
    so = os.environ.get("MK_SO") or os.path.join(HERE, "libmk.so")
    if not os.path.exists(so):
        subprocess.check_call(["bash", os.path.join(HERE, "build.sh")])
    lib = ctypes.CDLL(so)
    lib.mk_create.restype = ctypes.c_void_p
    lib.mk_create.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 8
    lib.mk_step.restype = None
    lib.mk_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mk_generate.restype = None
    lib.mk_generate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.mk_stat.restype = ctypes.c_longlong
    lib.mk_stat.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mk_destroy.restype = None
    lib.mk_destroy.argtypes = [ctypes.c_void_p]
    lib.mk_check.restype = ctypes.c_int
    lib.mk_track.restype = ctypes.c_int
    return lib


EXP0 = 112                         # exponent field of code n = 0; the 4-bit code covers binades EXP0 .. EXP0 + 15.
                                   # 112 = 0x70: the bf16 is then  sign << 15 | 0b0111 << 11 | low 11 bits of the code
NEG_V0 = 0x8000 | (EXP0 << 7)      # bf16 bits of -(value of the all-zero code)
COLMASK = int(os.environ.get("MK_COLMASK", "15"))   # debug: column scales for qkv (1), gate+up (2), down (4), lm_head (8)
TILE_ROWS_CHUNK = 4096             # rows packed at a time (bounds the temporaries; multiple of 16)


def pack_matrix(w: torch.Tensor) -> torch.Tensor:
    """Track A: [M, K] bf16 -> the same 16-bit values in MMA A-fragment order (a permutation, nothing else):
    [tile][kstep][lane = 4*g + t][ch][rh][e], element = W[tile*16 + rh*8 + g][kstep*16 + ch*8 + t*2 + e].
    A warp reads 512 contiguous bytes per MMA."""
    m, k = w.shape
    v = w.contiguous().view(torch.int16).view(m // 16, 2, 8, k // 16, 2, 4, 2)
    return v.permute(0, 3, 2, 5, 4, 1, 6).reshape(-1)


def _best_shift(e, valid, other):
    """e [M, K] exponent fields, other [K] shifts already applied along the other axis -> the shift per row
    that puts most of the row's valid elements into [EXP0, EXP0 + 15] (0 for a row without valid elements)."""
    v = (e + other[None, :] + 64).clamp_(0, 383).to(torch.int64)
    h = torch.zeros((e.shape[0], 384), dtype=torch.int32, device=e.device)
    h.scatter_add_(1, v, valid.to(torch.int32))
    del v
    c = torch.cumsum(torch.nn.functional.pad(h, (1, 0)), dim=1)
    m, start = (c[:, 16:] - c[:, :-16]).max(dim=1)
    return torch.where(m > 0, (EXP0 + 64) - start.to(torch.int32), torch.zeros_like(m))


def fit_shifts(w: torch.Tensor, columns: bool = True):
    """Exact power-of-two scales for bf16 weights [M, K]: row r is multiplied by 2**rs[r] and column j by
    2**cs[j] so that as many elements as possible fall into the 16 binades of the 12-bit code. The kernel
    undoes the row scale on the row's result; the column scale is undone on the INPUT of the matrix by whoever
    produces it (a norm weight or a row scale of the previous matrix, see Engine.__init__), all exact.
    Returns int32 rs [M], cs [K] (cs = 0 when columns is False or the matrix has values that cannot be shifted)."""
    bits = (w.contiguous().view(torch.int16).to(torch.int32)) & 0xFFFF
    e = (bits >> 7) & 0xFF
    valid = (e > 0) & (e < 255)
    special = ((e == 0) & ((bits & 0x7F) != 0)) | (e == 255)          # denormal / inf / nan: cannot be shifted
    del bits
    cs0 = torch.zeros(w.shape[1], dtype=torch.int32, device=w.device)
    big = torch.where(valid, e, torch.full_like(e, -1)).amax(dim=1)
    small = torch.where(valid, e, torch.full_like(e, 999)).amin(dim=1)
    rs0 = _best_shift(e, valid, cs0)
    rs0 = torch.maximum(torch.minimum(rs0, 254 - big), 1 - small)     # every shifted value stays a normal bf16
    rs0 = torch.where(special.any(dim=1) | (big < 0), torch.zeros_like(rs0), rs0)
    if not columns or bool(special.any()):
        return rs0, cs0
    rs, cs = rs0, cs0
    for _ in range(3):
        cs = _best_shift(e.t(), valid.t(), rs)
        rs = _best_shift(e, valid, cs)
    e2 = e + rs[:, None] + cs[None, :]
    if bool((valid & ((e2 < 1) | (e2 > 254))).any()) or int(rs.abs().max()) > 40 or int(cs.abs().max()) > 40:
        return rs0, cs0
    return rs, cs


def pack_rows(w: torch.Tensor, rs: torch.Tensor, cs: torch.Tensor):
    """Lossless 12-bit packing of bf16 weights [M, K] (M, K multiples of 16, K of 64) scaled by the exact
    powers of two of fit_shifts: element (r, j) is stored as W[r, j] * 2**(rs[r] + cs[j]).

    Elements whose exponent field is then in [EXP0, EXP0 + 15] are stored as a 12-bit code
    sign << 11 | (exponent - EXP0) << 7 | mantissa. The others (about 1e-4 of the elements: very small values,
    zeros) are stored as code 0 in the stream and returned as exceptions (row, col, exact bf16 bits of the
    scaled value): the kernel adds (value - value_of_code_0) * x for them, so every weight enters the product
    with its exact value.

    Stream layout (int32 words): [tile][group of 4 ksteps][j = 0..2][lane = 4*g + t][4]: per lane and group 12
    words; kstep m of the group = words 3m..3m+2 = 96 bits = 8 codes c0..c7 in the order of the MMA A fragment
    (i = (ch*2 + rh)*2 + e, element = W[tile*16 + rh*8 + g][kstep*16 + ch*8 + t*2 + e]). The bits are placed so
    that the kernel needs few operations: word r (r = 0..2) is MMA register r with the constant exponent bits
    (11..14 and 27..30) replaced by 4 bits of c6 resp. c7; register 3 is gathered from those 24 bits.
    A warp reads 512 contiguous bytes per load and decodes 4 MMAs from 3 loads."""
    m, k = w.shape
    bits = (w.contiguous().view(torch.int16).to(torch.int32)) & 0xFFFF
    e = (bits >> 7) & 0xFF
    e2 = e + rs[:, None] + cs[None, :]
    zero = (e == 0) | (e == 255)
    assert bool((zero | ((e2 >= 1) & (e2 <= 254))).all())
    n = e2 - EXP0
    inwin = (n >= 0) & (n <= 15) & ~zero
    code = torch.where(inwin, ((bits >> 15) << 11) | (n << 7) | (bits & 0x7F), torch.zeros_like(bits))
    xbits = torch.where(zero, bits, (bits & 0x807F) | (e2 << 7))      # exact bits of the scaled value
    exc = (~inwin).nonzero()                                          # [N, 2] (row, col), row-major order
    exc_bits = xbits[exc[:, 0], exc[:, 1]]
    del bits, e, e2, n, inwin, xbits, zero
    v = code.view(m // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(m // 16, k // 16, 32, 8)
    del code
    c0, c1, c2, c3, c4, c5, c6, c7 = v.unbind(dim=3)

    def half(c, nib):                                                 # 16 bits: code bits 10..0, 4 bits of the 4th register, sign
        return (c & 0x7FF) | (nib << 11) | ((c >> 11) << 15)

    def nibs(c):                                                      # a code of the 4th register, spread over the 3 words
        return c & 0xF, (c >> 4) & 0xF, ((c >> 8) & 7) | ((c >> 11) << 3)

    n6, n7 = nibs(c6), nibs(c7)
    words = torch.stack(
        [half(c0, n6[0]) | (half(c1, n7[0]) << 16), half(c2, n6[1]) | (half(c3, n7[1]) << 16), half(c4, n6[2]) | (half(c5, n7[2]) << 16)],
        dim=3,
    )                                                                 # [tile][kstep][lane][3]
    del v, c0, c1, c2, c3, c4, c5, c6, c7, n6, n7
    words = words.view(m // 16, k // 64, 4, 32, 3).permute(0, 1, 3, 2, 4).reshape(m // 16, k // 64, 32, 3, 4)
    words = words.permute(0, 1, 3, 2, 4).reshape(-1)
    return words, exc, exc_bits


def unpack_rows(words: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """Inverse of the stream layout (tests), written like the kernel's decoder: int32 [M, K] = bf16 bits."""
    w = words.view(m // 16, k // 64, 3, 32, 4).permute(0, 1, 3, 2, 4).reshape(m // 16, k // 64, 32, 4, 3)
    w = w.permute(0, 1, 3, 2, 4).reshape(m // 16, k // 16, 32, 3).to(torch.int64) & 0xFFFFFFFF
    w0, w1, w2 = w.unbind(dim=3)
    K, M = 0x38003800, 0x87FF87FF
    a0, a1, a2 = (w0 & M) | K, (w1 & M) | K, (w2 & M) | K
    a3 = ((w0 >> 11) & 0x000F000F) | ((w1 >> 7) & 0x00F000F0) | ((w2 >> 3) & 0x07000700) | ((w2 << 1) & 0x80008000) | K
    b = torch.stack([a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16, a2 & 0xFFFF, a2 >> 16, a3 & 0xFFFF, a3 >> 16], dim=3)
    b = b.view(m // 16, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(m, k)
    return b.to(torch.int32)


INLINE_CHUNKS = 3                  # exception chunks stored inside a tile's table entry


def build_tables(n_tiles, shifts, exc_tile, exc_row, exc_col, exc_bits):
    """Per-tile table and exception chunks for the kernel's coordinator warp.

    A chunk is one extra MMA for 8 exceptions of a tile: 8 records of 2 words,
      word 0 = exact bf16 bits of the (shifted) weight | NEG_V0 << 16 (cancels the stream's code-0 stand-in)
      word 1 = row in the tile | (position of x[col] inside its kstep's fragment: t*4 + slot) << 4 | kstep << 8
    Unused records are zero (weight 0, no cancellation).
    ttab int32 [n_tiles][64] (256 bytes, two cache lines, address known without a lookup):
      [0] first overflow chunk, [1] number of chunks, [4:8] the 16 row shifts as int8,
      [8:56] the tile's first INLINE_CHUNKS chunks.
    exc int32 [overflow chunks][16]: chunks beyond the inline ones, in tile order."""
    tile = torch.cat(exc_tile).to(torch.int64)
    row = torch.cat(exc_row).to(torch.int64)
    col = torch.cat(exc_col).to(torch.int64)
    bits = torch.cat(exc_bits).to(torch.int64)
    assert bool((tile[1:] >= tile[:-1]).all())
    dev = tile.device
    cnt = torch.bincount(tile, minlength=n_tiles)
    nch = (cnt + 7) // 8
    novf = (nch - INLINE_CHUNKS).clamp(min=0)
    off = torch.cumsum(novf, 0) - novf
    first = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(tile.numel(), device=dev) - first[tile]          # position of the exception in its tile
    offk = ((((col & 7) >> 1) << 2) + (col & 1) + 2 * ((col >> 3) & 1)) | ((col >> 4) << 4)   # t*4 + slot, kstep
    w0 = bits | (NEG_V0 << 16)
    w1 = row | (offk << 4)
    ttab = torch.zeros((n_tiles, 64), dtype=torch.int64, device=dev)
    ttab[:, 0] = off
    ttab[:, 1] = nch
    ttab[:, 4:8] = torch.cat(shifts).view(n_tiles, 16).contiguous().view(torch.int32).to(torch.int64)
    inl = pos < INLINE_CHUNKS * 8
    ttab[tile[inl], 8 + 2 * pos[inl]] = w0[inl]
    ttab[tile[inl], 9 + 2 * pos[inl]] = w1[inl]
    n_ovf = int(novf.sum())
    exc = torch.zeros(((n_ovf + 2) * 8, 2), dtype=torch.int64, device=dev)   # 2 chunks of padding: the kernel reads ahead
    o = ~inl
    rec = off[tile[o]] * 8 + (pos[o] - INLINE_CHUNKS * 8)
    exc[rec, 0] = w0[o]
    exc[rec, 1] = w1[o]
    return ttab.to(torch.int32).contiguous(), exc.to(torch.int32).contiguous(), int(tile.numel())


FORMATS = {
    "B": "weights: lossless 12-bit code of the BF16 values (sign, 4-bit exponent code, 7-bit mantissa) with exact "
    "power-of-two row/column scales folded into norm weights, out-of-window elements patched exactly; residual stream "
    "and accumulators float32; K/V cache BF16",
    "A": "none: weights BF16 as stored (permuted into the kernel's read order only); activations and K/V cache BF16 "
    "wherever the Hugging Face BF16 model has a BF16 tensor (float32 inside RMSNorm, matvec accumulators and the "
    "attention operator from the q.k accumulator through the softmax, as in the fused attention kernels: scores "
    "float32, probabilities BF16); RMSNorm applied before the matvec",
}


class _Weights:
    """Reads tensors from the safetensors shards straight to the GPU, one at a time."""

    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.headers = {}

    def get(self, name) -> torch.Tensor:
        fn = os.path.join(self.dir, self.map[name])
        if fn not in self.headers:
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.headers[fn] = (json.loads(f.read(n)), 8 + n)
        hdr, base = self.headers[fn]
        e = hdr[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        b0, b1 = e["data_offsets"]
        import numpy as np

        arr = np.memmap(fn, dtype=np.int16, mode="r", offset=base + b0, shape=((b1 - b0) // 2,))
        t = torch.from_numpy(np.array(arr)).cuda()
        return t.view(torch.bfloat16).view(e["shape"])


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        self.lib = _load_lib()
        self.max_batch, self.max_context = max_batch, max_context
        wts = _Weights(model_dir)
        self.track = "A" if self.lib.mk_track() else "B"
        if self.track == "A":
            self._load_track_a(wts)
        else:
            self._load_track_b(wts)
        self.embed = wts.get("model.embed_tokens.weight").contiguous()
        # RoPE table exactly as the reference computes it (float32 angles)
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
        ang = torch.arange(max_context, dtype=torch.float)[:, None] * inv_freq[None, :]
        self.rope = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous().cuda()
        if self.track == "A":      # the reference model casts cos / sin to the activations' dtype
            self.rope = self.rope.to(torch.bfloat16).float().contiguous()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.h = self.lib.mk_create(
            max_batch, max_context, self.W.data_ptr(), self.embed.data_ptr(), self.norm_w.data_ptr(),
            self.qk_norm_w.data_ptr(), self.rope.data_ptr(), self.ttab.data_ptr(), self.exc.data_ptr(), self.wfrag.data_ptr(),
        )
        if not self.h:
            raise RuntimeError("mk_create failed")
        self.pos = 0
        self.mode = MODE
        self._keep = None

    def _load_track_a(self, wts):
        """Track A: the checkpoint's BF16 weights, permuted into the kernel's read order; norm weights as stored."""
        layer_el = 6144 * 4096 + 4096 * 4096 + 24576 * 4096 + 4096 * 12288
        total = NL * layer_el + VOCAB * H
        self.W = torch.empty(total, dtype=torch.int16, device="cuda")
        self.norm_w = torch.empty((2 * NL + 1, H), dtype=torch.float32, device="cuda")
        self.qk_norm_w = torch.empty((NL, 2, 128), dtype=torch.float32, device="cuda")
        off = 0

        def put(w):
            nonlocal off
            p = pack_matrix(w)
            self.W[off : off + p.numel()] = p
            off += p.numel()

        for l in range(NL):
            pre = f"model.layers.{l}."
            q, k, v = (wts.get(pre + f"self_attn.{n}_proj.weight") for n in "qkv")
            put(torch.cat([q, k, v], dim=0))
            del q, k, v
            put(wts.get(pre + "self_attn.o_proj.weight"))
            g = wts.get(pre + "mlp.gate_proj.weight")
            u = wts.get(pre + "mlp.up_proj.weight")
            # tile i = gate rows 8i..8i+7 then up rows 8i..8i+7: silu(gate)*up is computed inside one thread
            put(torch.cat([g.view(INNER // 8, 8, H), u.view(INNER // 8, 8, H)], dim=1).reshape(2 * INNER, H))
            del g, u
            put(wts.get(pre + "mlp.down_proj.weight"))
            self.norm_w[2 * l] = wts.get(pre + "input_layernorm.weight").float()
            self.norm_w[2 * l + 1] = wts.get(pre + "post_attention_layernorm.weight").float()
            self.qk_norm_w[l, 0] = wts.get(pre + "self_attn.q_norm.weight").float()
            self.qk_norm_w[l, 1] = wts.get(pre + "self_attn.k_norm.weight").float()
        put(wts.get("lm_head.weight"))
        assert off == total
        self.norm_w[2 * NL] = wts.get("model.norm.weight").float()
        # norm weights in the order of the matvec input fragments (the two layouts of the kernel), BF16 as stored
        k = torch.arange(H, device="cuda")
        ks, t, slot = k >> 4, (k & 7) >> 1, (k & 1) + 2 * ((k >> 3) & 1)
        idx_c = ((ks * 4 + t) << 2) + slot
        idx_w = ((((ks >> 2) * 2 + ((ks & 3) >> 1)) * 4 + t) << 3) + (ks & 1) * 4 + slot
        self.wfrag = torch.zeros((2 * NL + 1, 2, H), dtype=torch.int16, device="cuda")
        names = [n for l in range(NL) for n in (f"model.layers.{l}.input_layernorm.weight", f"model.layers.{l}.post_attention_layernorm.weight")]
        for i, name in enumerate(names + ["model.norm.weight"]):
            wb = wts.get(name).contiguous().view(torch.int16)
            self.wfrag[i, 0, idx_c] = wb
            self.wfrag[i, 1, idx_w] = wb
        self.ttab = torch.zeros(64, dtype=torch.int32, device="cuda")     # unused in this track
        self.exc = torch.zeros(64, dtype=torch.int32, device="cuda")
        self.n_exceptions = 0

    def _load_track_b(self, wts):
        """Track B: 12-bit weight stream with exact row/column scales and exception tables."""
        layer_el = 6144 * 4096 + 4096 * 4096 + 24576 * 4096 + 4096 * 12288
        total = (NL * layer_el + VOCAB * H) * 3 // 8                       # int32 words: 12 bits per weight
        self.W = torch.empty(total, dtype=torch.int32, device="cuda")
        self.norm_w = torch.empty((2 * NL + 1, H), dtype=torch.float32, device="cuda")
        self.qk_norm_w = torch.empty((NL, 2, 128), dtype=torch.float32, device="cuda")
        off = 0
        tile0 = 0
        shifts, exc_tile, exc_row, exc_col, exc_bits = [], [], [], [], []

        def put(w, columns=True, out_shift=None):
            """Append a matrix to the stream. Returns its column shifts: the producer of the matrix's input must
            multiply input element j by 2**-cs[j]. out_shift [M]: extra exact scale 2**-out_shift on the row results
            (the column scale of the matrix that consumes this one's output)."""
            nonlocal off, tile0
            rs, cs = fit_shifts(w, columns)
            for r0 in range(0, w.shape[0], TILE_ROWS_CHUNK):
                words, exc, xb = pack_rows(w[r0 : r0 + TILE_ROWS_CHUNK], rs[r0 : r0 + TILE_ROWS_CHUNK], cs)
                self.W[off : off + words.numel()] = words
                off += words.numel()
                exc_tile.append(tile0 + (r0 + exc[:, 0]) // 16)
                exc_row.append((r0 + exc[:, 0]) % 16)
                exc_col.append(exc[:, 1])
                exc_bits.append(xb)
                del words, exc, xb
            sh = rs if out_shift is None else rs + out_shift
            assert int(sh.abs().max()) < 127
            shifts.append(sh.to(torch.int8))
            tile0 += w.shape[0] // 16
            return cs

        def scaled(norm_weight, cs):                                   # norm weight * 2**-cs, exact
            return torch.ldexp(norm_weight.float(), -cs)

        for l in range(NL):
            pre = f"model.layers.{l}."
            q, k, v = (wts.get(pre + f"self_attn.{n}_proj.weight") for n in "qkv")
            cs = put(torch.cat([q, k, v], dim=0), columns=bool(COLMASK & 1))
            self.norm_w[2 * l] = scaled(wts.get(pre + "input_layernorm.weight"), cs)
            del q, k, v
            put(wts.get(pre + "self_attn.o_proj.weight"), columns=False)
            g = wts.get(pre + "mlp.gate_proj.weight")
            u = wts.get(pre + "mlp.up_proj.weight")
            dn = wts.get(pre + "mlp.down_proj.weight")
            _, cs_dn = fit_shifts(dn, bool(COLMASK & 4))
            # tile i = gate rows 8i..8i+7 then up rows 8i..8i+7: silu(gate)*up is computed inside one thread.
            # The down projection's column scale j is undone on its input silu(gate[j]) * up[j] through row up[j].
            extra = torch.stack([torch.zeros_like(cs_dn).view(INNER // 8, 8), cs_dn.view(INNER // 8, 8)], dim=1).reshape(2 * INNER)
            cs = put(torch.cat([g.view(INNER // 8, 8, H), u.view(INNER // 8, 8, H)], dim=1).reshape(2 * INNER, H), columns=bool(COLMASK & 2), out_shift=extra)
            self.norm_w[2 * l + 1] = scaled(wts.get(pre + "post_attention_layernorm.weight"), cs)
            del g, u
            cs = put(dn, columns=bool(COLMASK & 4))
            assert bool((cs == cs_dn).all())
            del dn
            self.qk_norm_w[l, 0] = wts.get(pre + "self_attn.q_norm.weight").float()
            self.qk_norm_w[l, 1] = wts.get(pre + "self_attn.k_norm.weight").float()
        cs = put(wts.get("lm_head.weight"), columns=bool(COLMASK & 8))
        assert off == total
        self.ttab, self.exc, self.n_exceptions = build_tables(tile0, shifts, exc_tile, exc_row, exc_col, exc_bits)
        self.wfrag = torch.zeros(16, dtype=torch.int16, device="cuda")   # unused in this track
        del shifts, exc_tile, exc_row, exc_col, exc_bits
        self.norm_w[2 * NL] = scaled(wts.get("model.norm.weight"), cs)

    def reset(self) -> None:
        self.pos = 0

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.int64 or not tokens.is_contiguous():
            tokens = tokens.to(torch.int64).contiguous()
        b = tokens.shape[0]
        if self.pos >= self.max_context or b > self.max_batch:
            raise ValueError("step() beyond max_context or max_batch")
        logits = torch.empty((b, VOCAB), dtype=torch.float32, device="cuda")
        self.lib.mk_step(self.h, tokens.data_ptr(), logits.data_ptr(), b, self.pos, self.mode)
        self.pos += 1
        self._keep = tokens  # the kernel reads it asynchronously
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Greedy decode n_steps tokens in ONE kernel launch: argmax, token feedback and the loop are on the
        device. tokens [B] is the next input token; returns int64 [B, n_steps]."""
        if tokens.dtype != torch.int64 or not tokens.is_contiguous():
            tokens = tokens.to(torch.int64).contiguous()
        b = tokens.shape[0]
        if self.pos + n_steps > self.max_context or b > self.max_batch:
            raise ValueError("generate() would exceed max_context or max_batch")
        out = torch.empty((b, n_steps), dtype=torch.int64, device="cuda")
        self.lib.mk_generate(self.h, tokens.data_ptr(), out.data_ptr(), b, self.pos, n_steps)
        self.pos += n_steps
        self._keep = tokens
        return out

    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.mk_destroy(self.h)
            self.h = None
        self.W = self.embed = self.norm_w = self.qk_norm_w = self.rope = self.ttab = self.exc = self.wfrag = None

    def stats(self) -> dict:
        return {
            "implementation": "megakernel",
            "mode": "step(): one launch per step; generate(): one launch for all steps (argmax, feedback, loop on device)",
            "kernel_launches": int(self.lib.mk_stat(self.h, 0)),
            "steps": int(self.lib.mk_stat(self.h, 1)),
            "blocks": int(self.lib.mk_stat(self.h, 2)),
            "host_syncs_in_step": 0,
            "track": self.track,
            "formats_that_differ_from_the_checkpoint": FORMATS[self.track],
            "grid_barriers_per_step": "181 (5 per layer + 1); +36 with 1-2 sequences beyond 1136 positions (attention split "
            "over all SMs); track A with 3+ sequences: +73 (RMSNorm as a phase of its own)",
            "max_context_supported": 12288,
            "block_barriers_per_step": "one per 16-row tile (about 3460 per SM) + three per matvec phase",
            "sync_points_on_the_host": 0,
            "weight_bits": 12 if self.track == "B" else 16,
            "weight_exceptions": self.n_exceptions,
            "weight_bytes_read_per_step": int(self.W.numel()) * (4 if self.track == "B" else 2),
            "max_batch": self.max_batch,
            "max_context": self.max_context,
        }
