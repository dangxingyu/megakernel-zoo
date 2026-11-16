"""Decode engine for Qwen3.8-27B on one B200: a persistent megakernel (engine/mk.cu), one launch per model pass.

Contract (eval/_engine_runner.py): Engine(model_dir, max_batch, max_context), reset(), step(tokens), generate(),
submit()/poll(), stats(), close(). Python only loads the weights, lays them out and launches the kernel; nothing of
the model runs in PyTorch.

MK_TRACK=A (default B): submit()/poll() decode one token per pass, the same data flow as step().
MK_TRACK=B: submit()/poll() draft with the checkpoint's own MTP head and verify P positions per pass (exactly the
greedy tokens of step(): a verify pass computes every position with the arithmetic of a single step).
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HID, VOCAB, FF, NL = 5120, 248320, 17408, 64
NCM, MROW, RING, PMAX = 32, 16480, 64, 32
CHAIN_MAX = 4   # drafts made by the MTP head per pass (1 in the verify launch + chained launches); more only by context copy
(OP_EMBED, OP_GEMM, OP_LMIX, OP_ATT_PREP, OP_ATT_SCORE, OP_ATT_SOFTMAX, OP_ATT_VSUM, OP_ATT_OUT, OP_ARGMAX, OP_FINISH, OP_MTP_IN,
 OP_ACCEPT, OP_CHAIN_FIN) = range(1, 14)
OP_ATT_DIST = 15
# One-token passes run each full-attention layer as one op, one block per 256-key chunk of each (sequence, kv head)
# (mk.cu att_dist_work), while a unit's chunks fit its blocks; otherwise the four separate ops
ATT_FUSE = os.environ.get("MK_ATT_FUSE", "1") == "1"
EP_STORE, EP_RESID, EP_GATEUP, EP_LOGITS, EP_SET = 1, 2, 3, 4, 5
F_SPEC, F_LOGITS, F_FEEDBACK, F_CHAIN, F_DEFER = 1, 2, 4, 1024, 16384
# F_DEFER (mk.cu): the recurrent state is written to memory on every DEFER_N-th one-token step of a sequence; the steps
# in between save their rank-1 update and replay the saved ones (bit-identical state). 1: every step stores its state.
# Measured (same card): B=8 9779 -> 9634 us per step at N = 4 (N = 3: 9638, 5: ~9650); B=1 8874 -> 8894 (the replay costs
# more than 3 MB of state per layer saves): deferred from DEFER_MIN_B sequences on (B is fixed between resets).
DEFER_MIN_B = int(os.environ.get("MK_DEFER_MINB", "4"))
DEFER_N = max(1, min(5, int(os.environ.get("MK_DEFER", "1"))))   # at most 4 saved updates (lmix_dual stages one per thread role)
OPW = 20  # int64 words per op
DRAFT_ROWS = 32768  # the draft head scores token ids below this (first rows of lm_head, same buffer)
TRACK = (os.environ.get("MK_TRACK") or (HERE / "TRACK").read_text().split()[0]).upper()   # the build switch: A or B
RAMP_LATE = os.environ.get("MK_RAMP_LATE", "1") == "1"   # B=1, nothing in flight: a pass's length chosen at its launch
MTP_SKIP = os.environ.get("MK_MTP_SKIP", "1") == "1"   # verify passes without the MTP head while copies keep being accepted
W12_EXACT = os.environ.get("MK_W12_EXACT", "1") == "1"   # W12 patched into the MMA operands: bit-identical to BF16 storage
# W12 word layout: "prmt" = per k-step [sign|mantissa bytes 0-3][bytes 4-7][8 exponent nibbles] (decoded with a byte
# permute, must match the build flag -DW12_PRMT); "pack" = two 12-bit codes per 24 bits (s0004-s0010)
W12_FMT = os.environ.get("MK_W12_FMT", "prmt")
W12_DUAL = os.environ.get("MK_W12_DUAL", "1") == "1"   # BF16 copies of out_proj / o_proj for B = 1 passes (build: -DW12_MIXED)
# Exponent window per row: 16 values (code = exponent - lo) when every out-of-window weight is patched in from the
# lists (exact W12: code 0 needs no escape meaning); 15 (code 0 = escape, decodes to 2^(lo-1)) for the s0002 loops.
W12_WIN = int(os.environ.get("MK_W12_WIN", "16" if W12_EXACT else "15"))
W12 = (os.environ.get("MK_W12") or ("1" if "W12" in (HERE / "TRACK").read_text().upper().split() else "0")) == "1"   # track B only


class Ctx(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int) for n in ("B", "P", "NC", "NG", "nwarps", "nblocks", "max_ctx", "flags", "pos_ub", "NGq",
                                             "tok_stride", "na_max", "pos_off", "src_P", "src_NG", "src_sel", "next_P",
                                             "defer_n", "pad1", "tok_tag")]
                + [("bar_base", ctypes.c_uint64)]
                + [(n, ctypes.c_void_p) for n in ("bar", "pos", "nacc", "tok_in", "h", "mbuf", "ssq", "part", "cnt", "logits",
                                                  "amax_val", "amax_idx", "tok_res", "tok_out", "tokbuf", "qf", "sc", "pf",
                                                  "vpart", "rope_cos", "rope_sin", "limit", "mtp_res", "cnt_out", "tok_next", "hist",
                                                  "lkf", "pft", "ts", "wts")]
                + [("wts_op", ctypes.c_int), ("pad3", ctypes.c_int)]
                + [(n, ctypes.c_void_p) for n in ("cst", "nst", "sst", "tokbuf_next", "bts")])


def _lib():
    so = Path(os.environ["MK_LIB"]) if os.environ.get("MK_LIB") else HERE / "build" / "libmk.so"   # MK_LIB: A/B tests
    if not so.exists():
        subprocess.run(["bash", str(HERE / "build.sh")], check=True)
    lib = ctypes.CDLL(str(so))
    lib.mk_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(Ctx), ctypes.c_int]
    lib.mk_launch.restype = ctypes.c_int
    if hasattr(lib, "mk_host_alloc"):   # (older libraries in A/B tests have none)
        lib.mk_host_alloc.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        lib.mk_host_alloc.restype = ctypes.c_int
    return lib


def to_frag(W: torch.Tensor) -> torch.Tensor:
    """[M, K] bf16 -> int16 buffer in m16n8k16 A-operand order: [tile][kstep][lane][reg][half]."""
    M, K = W.shape
    x = W.view(torch.int16).view(M // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return x.view(-1)


def w12_window_rows(W: torch.Tensor) -> torch.Tensor:
    """Per output row, the first of the 15 consecutive exponent fields that hold the most of the row's weights
    (exact counts). Per-row windows matter on this checkpoint: a few rows of down_proj / out_proj / o_proj (the
    residual channels with large weights) hold thousands of above-window weights each under one window per tensor."""
    M = W.shape[0]
    e = (W.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    h = torch.zeros(M, 256, dtype=torch.int32, device=W.device)
    h.scatter_add_(1, e.long(), torch.ones_like(e))
    cs = torch.cat([torch.zeros(M, 1, dtype=torch.int64, device=W.device), h.long().cumsum(1)], 1)
    return (cs[:, W12_WIN:] - cs[:, :-W12_WIN]).argmax(1)   # [M] int64


def w12_rows_frag(lo_rows: torch.Tensor, NT: int) -> torch.Tensor:
    """lo per element in fragment order, broadcastable to [NT][KQ][4][32][8]: row = T*16 + lane/4 + 8*((slot/2)&1)."""
    lane = torch.arange(32, device=lo_rows.device).view(32, 1)
    slot = torch.arange(8, device=lo_rows.device).view(1, 8)
    r = (lane >> 2) + 8 * ((slot >> 1) & 1)                 # [32][8]
    return lo_rows.view(NT, 16)[:, r].view(NT, 1, 1, 32, 8)


def w12_pack(F: torch.Tensor, NT: int, KQ: int, lo_rows: torch.Tensor):
    """Fragment-ordered bf16 bits [NT][KQ][4 k-steps][32 lanes][8] -> 12-bit codes, 48 bytes per lane and quad:
    [NT][KQ][3 planes][32 lanes][4 words]. Code [sign][exponent - lo_row + 1][7 mantissa bits]. Returns the packed words
    (int32) and the positions [n, 5] and bf16 bits [n] of the weights the code cannot express (exponent outside
    [lo_row, lo_row + 14]); they carry code 0."""
    b = F.view(NT, KQ, 4, 32, 8).to(torch.int32) & 0xFFFF
    e = (b >> 7) & 0xFF
    lo = w12_rows_frag(lo_rows, NT).to(torch.int32)
    inw = (e >= lo) & (e <= lo + W12_WIN - 1)
    code = torch.where(inw, ((b >> 15) << 11) | ((e - lo + 16 - W12_WIN) << 7) | (b & 0x7F), torch.zeros_like(b))
    out = (~inw).nonzero()
    vals = b[~inw]
    if W12_FMT == "prmt":   # [byte 0-3][byte 4-7][nibbles: weight k at bit 4 (k >> 1) + 16 (k & 1)] per lane and k-step
        cc = torch.where(inw, e - lo, torch.zeros_like(e)).to(torch.int64)   # escapes: any code (patched)
        by = (((b >> 8) & 0x80) | (b & 0x7F)).to(torch.int64)                # [sign][7 mantissa bits]
        del e, inw, b, lo, code
        sh = torch.tensor([0, 8, 16, 24], device=F.device)
        wA = (by[..., 0:4] << sh).sum(-1); wB = (by[..., 4:8] << sh).sum(-1)
        nsh = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=F.device)
        wE = (cc << nsh).sum(-1)
        u = torch.stack([wA, wB, wE], -1)                                    # [NT][KQ][4][32][3]
        del by, cc, wA, wB, wE
        u = u.permute(0, 1, 3, 2, 4).reshape(NT, KQ, 32, 3, 4).permute(0, 1, 3, 2, 4)
        u = (u - ((u >> 31) << 32)).to(torch.int32).contiguous()
        return u.view(-1), out, vals
    del e, inw, b, lo
    c = code.to(torch.int64).view(NT, KQ, 4, 32, 4, 2)
    del code
    x = c[..., 0] | (c[..., 1] << 12)                      # two codes = 24 bits
    del c
    M = 0xFFFFFFFF
    u = torch.stack([(x[..., 0] | (x[..., 1] << 24)) & M, ((x[..., 1] >> 8) | (x[..., 2] << 16)) & M,
                     ((x[..., 2] >> 16) | (x[..., 3] << 8)) & M], -1)   # [NT][KQ][4][32][3]
    del x
    u = u.permute(0, 1, 3, 2, 4).reshape(NT, KQ, 32, 3, 4).permute(0, 1, 3, 2, 4)
    u = (u - ((u >> 31) << 32)).to(torch.int32).contiguous()
    return u.view(-1), out, vals


def w12_ctab(lo_rows: torch.Tensor) -> torch.Tensor:
    """Per row the decode base C * 0x10001 (C = (lo - 16 + W12_WIN) << 7) as int32: the kernel adds it to two codes at once.
    Padded by 32 rows: the kernel reads the next tile's bases ahead."""
    c = ((lo_rows.to(torch.int64) - (16 - W12_WIN)) << 7) * 0x10001
    c = torch.cat([c, c.new_zeros(64)])
    return (c - ((c >> 31) << 32)).to(torch.int32).contiguous()


def w12_escapes(out: torch.Tensor, vals: torch.Tensor, tab: torch.Tensor, NT: int, KQ: int):
    """Per-warp lists of the out-of-window weights in stream order: hdr[w] = start | min(n + 1, 32) << 20 (n entries
    and the all-ones sentinel) and entries (position, bf16 bits); position = [quad in the warp's share][lane (5 bits)]
    [k-step (2)][slot (3)]."""
    nw = tab.numel() - 1
    keep = out[:, 0] < NT
    out, vals = out[keep], vals[keep]
    Q = out[:, 0] * KQ + out[:, 1]
    w = torch.searchsorted(tab, Q, right=True) - 1
    epos = ((Q - tab[w]) << 10) | (out[:, 3] << 5) | (out[:, 2] << 3) | out[:, 4]
    counts = torch.bincount(w, minlength=nw)
    start = torch.cat([torch.zeros(1, dtype=torch.int64, device=tab.device), counts.cumsum(0)])[:nw] + torch.arange(nw, device=tab.device)
    assert int(start.max()) < (1 << 20)
    ent = torch.full((out.shape[0] + nw + 32, 2), -1, dtype=torch.int32, device=tab.device)   # +32: lanes read 32 entries
    ent[torch.arange(out.shape[0], device=tab.device) + w, 0] = epos.to(torch.int32)
    ent[torch.arange(out.shape[0], device=tab.device) + w, 1] = vals
    hdr = start | (torch.clamp(counts + 1, max=32) << 20)
    return hdr.to(torch.int32).contiguous(), ent.view(-1).contiguous()


def w12_escapes_fixed(out: torch.Tensor, vals: torch.Tensor, tab: torch.Tensor, NT: int, KQ: int):
    """Exact (bit-identical) W12: per-warp lists at a fixed stride CAP (no header to load first), entries in stream
    order (position = [quads left in the warp's share][k-step (2)][lane (5)][slot (3)], bf16 bits), each list ends
    with a zero entry. Returns the entries (int32, [nw * CAP + 16][2] flattened) and CAP (a multiple of 16)."""
    nw = tab.numel() - 1
    keep = out[:, 0] < NT
    out, vals = out[keep], vals[keep]
    Q = out[:, 0] * KQ + out[:, 1]
    w = torch.searchsorted(tab, Q, right=True) - 1
    # quad field = quads left in the warp's share at that quad (its loop counter), >= 1; 0 = end of list
    epos = ((tab[w + 1] - Q) << 10) | (out[:, 2] << 8) | (out[:, 3] << 3) | out[:, 4]   # nonzero() order = this order
    counts = torch.bincount(w, minlength=nw)
    CAP = (int(counts.max()) + 1 + 15) // 16 * 16   # CAP = 16: the kernel reads 16 entries, else 32 and refills ahead
    first = torch.cat([torch.zeros(1, dtype=torch.int64, device=tab.device), counts.cumsum(0)])[:-1]
    rank = torch.arange(out.shape[0], device=tab.device) - first[w]
    ent = torch.zeros((nw * CAP + 48, 2), dtype=torch.int32, device=tab.device)   # + the window's read-ahead; 0 = end
    ent[w * CAP + rank, 0] = epos.to(torch.int32)
    ent[w * CAP + rank, 1] = vals
    return ent.view(-1).contiguous(), CAP


def w12_unpack_fixed(packed: torch.Tensor, NT: int, KQ: int, lo_rows: torch.Tensor, ent: torch.Tensor, CAP: int, tab: torch.Tensor):
    """What the exact W12 kernel feeds the MMA: the dense decode with every listed weight patched in (bf16 bits
    [NT][KQ][4][32][8]), for the bit-for-bit check."""
    u = (packed.view(NT, KQ, 3, 32, 4).permute(0, 1, 3, 2, 4).reshape(NT, KQ, 32, 4, 3).to(torch.int64)) & 0xFFFFFFFF
    C = (w12_ctab(lo_rows)[:NT * 16].to(torch.int64) & 0xFFFFFFFF)
    lane = torch.arange(32, device=packed.device)
    Cg = C.view(NT, 16)[:, lane >> 2].view(NT, 1, 32, 1)
    Cg8 = C.view(NT, 16)[:, (lane >> 2) + 8].view(NT, 1, 32, 1)
    pair = torch.arange(4, device=packed.device).view(1, 1, 1, 4)
    Cp = torch.where((pair & 1) == 0, Cg, Cg8).view(NT, 1, 32, 1, 4)
    if W12_FMT == "prmt":   # the kernel's dec8p: byte permute with sign replication, mask, exponent nibbles, base
        wA, wB, wE = u[..., 0], u[..., 1], u[..., 2]                       # [NT][KQ][32][4 k-steps]
        def half(word, byte):  # [sign replicated][byte] -> the 16-bit half before masking
            bb = (word >> (8 * byte)) & 0xFF
            return bb | (((bb >> 7) * 0xFF) << 8)
        X = torch.stack([half(wA, 0) | (half(wA, 1) << 16), half(wA, 2) | (half(wA, 3) << 16),
                         half(wB, 0) | (half(wB, 1) << 16), half(wB, 2) | (half(wB, 3) << 16)], -1)
        E = torch.stack([(wE << 7) & 0x07800780, (wE << 3) & 0x07800780, (wE >> 1) & 0x07800780, (wE >> 5) & 0x07800780], -1)
        r = ((X & 0x807F807F) + E + Cp) & 0xFFFFFFFF
    else:
        u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]
        x = torch.stack([u0, (u0 >> 24) | (u1 << 8), (u1 >> 16) | (u2 << 16), u2 >> 8], -1) & 0xFFFFFF
        t = x + 15 * (x & 0xFFF000)
        r = (t + Cp + ((t >> 11) & 0x10001) * 0x7800) & 0xFFFFFFFF
    bits = torch.stack([r & 0xFFFF, r >> 16], -1).reshape(NT, KQ, 32, 4, 8).permute(0, 1, 3, 2, 4).contiguous()
    nw = tab.numel() - 1
    e = ent.view(-1, 2)[:nw * CAP].view(nw, CAP, 2).to(torch.int64)
    real = e[..., 0] != 0
    assert bool((~real[:, -1]).all())                          # every list has its sentinel inside the slot
    assert bool((real[:, 1:] <= real[:, :-1]).all())           # entries, then sentinels only
    wi = torch.arange(nw, device=packed.device).view(nw, 1).expand(nw, CAP)[real]
    epos, val = e[..., 0][real] & 0xFFFFFFFF, e[..., 1][real]
    Q = tab[wi + 1] - (epos >> 10)
    flat = ((Q * 4 + ((epos >> 8) & 3)) * 32 + ((epos >> 3) & 31)) * 8 + (epos & 7)
    bits.view(-1)[flat] = val
    return bits.to(torch.int32)


def w12_unpack(packed: torch.Tensor, NT: int, KQ: int, lo_rows: torch.Tensor, hdr: torch.Tensor, ent: torch.Tensor, tab: torch.Tensor):
    """What the kernel computes, for the bit-for-bit check: the bf16 bits [NT][KQ][4][32][8] of the dense decode
    (code + C_row + sign * 0x7800) with every listed weight put in its place (the kernel adds w x - v0 x for those,
    v0 being the decode of code 0, which equals putting w in place in exact arithmetic)."""
    u = (packed.view(NT, KQ, 3, 32, 4).permute(0, 1, 3, 2, 4).reshape(NT, KQ, 32, 4, 3).to(torch.int64)) & 0xFFFFFFFF
    u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]
    x = torch.stack([u0, (u0 >> 24) | (u1 << 8), (u1 >> 16) | (u2 << 16), u2 >> 8], -1) & 0xFFFFFF   # [NT][KQ][lane][k-step][pair]
    t = x + 15 * (x & 0xFFF000)
    C = (w12_ctab(lo_rows)[:NT * 16].to(torch.int64) & 0xFFFFFFFF)                    # per row
    lane = torch.arange(32, device=packed.device)
    Cg = C.view(NT, 16)[:, lane >> 2].view(NT, 1, 32, 1)                              # rows g (registers 0, 2)
    Cg8 = C.view(NT, 16)[:, (lane >> 2) + 8].view(NT, 1, 32, 1)                       # rows g + 8 (registers 1, 3)
    pair = torch.arange(4, device=packed.device).view(1, 1, 1, 4)
    Cp = torch.where((pair & 1) == 0, Cg, Cg8).view(NT, 1, 32, 1, 4)
    r = (t + Cp + ((t >> 11) & 0x10001) * 0x7800) & 0xFFFFFFFF
    bits = torch.stack([r & 0xFFFF, r >> 16], -1).reshape(NT, KQ, 32, 4, 8).permute(0, 1, 3, 2, 4).contiguous()   # [NT][KQ][k-step][lane][slot]
    e = ent.view(-1, 2)[:-32].to(torch.int64)   # the last 32 rows are padding for the lane-parallel read
    sentinel = e[:, 0] == -1
    w = torch.cumsum(sentinel.to(torch.int64), 0) - sentinel.to(torch.int64)   # the warp of every entry
    assert int(sentinel.sum()) == tab.numel() - 1
    sidx = torch.nonzero(sentinel).flatten()
    st = hdr.to(torch.int64) & 0xFFFFF
    assert int(st[0]) == 0 and bool((st[1:] == sidx[:-1] + 1).all()) and bool(((hdr.to(torch.int64) >> 20) == torch.clamp(sidx - st + 1, max=32)).all())
    keep = ~sentinel
    epos, val, w = e[keep, 0] & 0xFFFFFFFF, e[keep, 1], w[keep]
    Q = tab[w] + (epos >> 10)
    flat = ((Q * 4 + ((epos >> 3) & 3)) * 32 + ((epos >> 5) & 31)) * 8 + (epos & 7)
    assert bool((bits.view(-1)[flat] == ((C.view(NT * 16)[(Q // KQ) * 16 + ((epos >> 5) & 31) // 4 + 8 * ((epos >> 1) & 1)]) & 0xFFFF)).all())   # code 0 decodes to v0 of the row
    bits.view(-1)[flat] = val
    return bits.to(torch.int32)


def ngroups(nc: int) -> int:
    return 1 if nc <= 8 else 2 if nc <= 16 else 4


class Weights:
    """Reads the checkpoint's BF16 tensors straight onto the card."""

    def __init__(self, model_dir: str):
        self.where = {}
        for f in sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors"))):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = json.loads(fh.read(n))
            for k, v in head.items():
                if k != "__metadata__":
                    self.where[k] = (f, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0], v["shape"])
        self.maps = {}

    def get(self, name: str) -> torch.Tensor:
        f, off, size, shape = self.where[name]
        if f not in self.maps:
            self.maps[f] = np.memmap(f, dtype=np.uint8, mode="r")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = torch.from_numpy(self.maps[f][off:off + size])  # read-only memmap: only copied to the card
        return raw.cuda().view(torch.bfloat16).view(shape)


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        t0 = time.time()
        torch.cuda.init()
        self.lib = _lib()
        sm_count, max_smem = ctypes.c_int(), ctypes.c_int()
        if self.lib.mk_init(ctypes.byref(sm_count), ctypes.byref(max_smem)):
            raise RuntimeError("mk_init failed")
        self.nblocks = sm_count.value
        self.max_smem = max_smem.value
        self.nwarps = self.nblocks * 32
        self.Bmax = max(1, int(max_batch))
        self.mc = ((int(max_context) + 2 * PMAX + 255) // 256) * 256
        self.keep = []           # every device buffer the kernel uses
        self.launches = 0
        self.dev = torch.device("cuda")
        self.spec = TRACK == "B"
        # Served one-token decode (submit / poll): the kernel's last op writes the chosen tokens straight into page-locked
        # host memory mapped for the device (MK_HOSTMAP=0: a device buffer and one device-to-host copy per pass, as inherited)
        self.hmap = None
        self.hmap_tag = 0
        if os.environ.get("MK_HOSTMAP", "1") == "1" and hasattr(self.lib, "mk_host_alloc"):
            hp, dp = ctypes.c_void_p(), ctypes.c_void_p()
            n = (max_context + 8) * max(max_batch, 8)
            if self.lib.mk_host_alloc(n * 8, ctypes.byref(hp), ctypes.byref(dp)) == 0:
                self.hmap = np.ctypeslib.as_array((ctypes.c_int64 * n).from_address(hp.value))
                self.hmap_dev = dp.value
        self.defer_n = DEFER_N   # change only right after reset()
        self._buffers()
        self._load(model_dir)
        self.pos_ub = 0
        self.bar_base = 0
        self._hold = None
        self.counters = {"passes": 0, "tokens": 0, "chain_launches": 0}
        self.load_seconds = time.time() - t0
        self.waiting, self.cohort, self.first_wait = [], None, None

    # ------------------------------------------------------------------ buffers
    def _z(self, *shape, dtype=torch.float32):
        t = torch.zeros(*shape, dtype=dtype, device=self.dev)
        self.keep.append(t)
        return t

    def _buffers(self):
        B, mc, nw = self.Bmax, self.mc, self.nwarps
        i16, i32, i64 = torch.int16, torch.int32, torch.int64
        b = {}
        b["bar"] = self._z(2, dtype=i64)
        b["pos"] = self._z(max(B, 8), dtype=i32)
        b["nacc"] = self._z(3 * NCM, dtype=i32)   # accepted, to replay, committed state buffer (per sequence)
        b["limit"] = self._z(max(B, 8), dtype=i32)
        b["lkf"] = self._z(max(B, 8), dtype=i32)
        b["hist"] = self._z(max(B, 8) * mc, dtype=i32)
        b["h"] = self._z(NCM * HID, dtype=i16)
        b["mbuf"] = self._z(NCM * MROW, dtype=i16)
        b["ssq"] = self._z(320 * NCM)
        b["part"] = self._z((VOCAB // 16 + self.nblocks + 1) * 4 * 32 * 4)
        b["part"].view(torch.int32).fill_(0x7fbadbad)   # cut-tile slots start as sentinels (mk.cu SENT_BITS)
        b["cnt"] = self._z(VOCAB // 16, dtype=i32)
        b["amax_val"] = self._z(VOCAB // 16 * NCM)
        b["amax_idx"] = self._z(VOCAB // 16 * NCM, dtype=i32)
        b["tok_res"] = self._z(NCM, dtype=i32)
        b["mtp_res"] = self._z(NCM, dtype=i32)
        b["amax_main"] = self._z(NCM, dtype=i64)    # running argmax keys of the heads (folded in the GEMM)
        b["amax_draft"] = self._z(NCM, dtype=i64)
        b["tokbuf"] = self._z(NCM, dtype=i64)
        self.tokbuf2 = self._z(NCM, dtype=i64)          # second draft buffer (per-sequence layouts)
        # ring of per-pass layouts: [slot][current B+1 | next B+1 | recurrent units per block (4 each, -1 = none)]
        self.layw = 32 + self.nblocks * 4
        self.lay = self._z(16, self.layw, dtype=i32)
        self.lay_host = torch.zeros(16, self.layw, dtype=torch.int32).pin_memory()
        self.utab_cache = {}
        self.lay_i = 0
        NGQ = (6 * PMAX + 7) // 8
        b["qf"] = self._z(B * 4 * 16 * NGQ * 128, dtype=i16)
        b["sc"] = self._z(B * 24 * PMAX * mc)
        b["pf"] = self._z(B * 4 * (mc // 16) * NGQ * 128, dtype=i16)
        b["vpart"] = self._z(B * 4 * 16 * (mc // 256) * NGQ * 32 * 4)
        pos = torch.arange(mc, dtype=torch.float32, device=self.dev)
        inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float) / 64)).to(self.dev)
        freqs = pos[:, None] * inv_freq[None, :]
        b["rope_cos"] = freqs.cos().to(torch.bfloat16).float().contiguous()
        b["rope_sin"] = freqs.sin().to(torch.bfloat16).float().contiguous()
        self.keep += [b["rope_cos"], b["rope_sin"]]
        self.buf = b

    # ------------------------------------------------------------------ weights and programs
    def _load(self, model_dir: str):
        cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        types = cfg["layer_types"]
        assert cfg["hidden_size"] == HID and cfg["vocab_size"] == VOCAB and len(types) == NL
        wts = Weights(model_dir)
        B, mc = self.Bmax, self.mc
        pre = "model.language_model."
        f32 = lambda t: t.float().contiguous()
        keep = self.keep
        bf = self.buf
        self.weight_bytes = 0

        def K(t):
            keep.append(t)
            return t.data_ptr()

        def gain(name):  # RMSNorm weights are stored centred
            return K(f32(1.0 + wts.get(name).float()))

        def frag(W, bf=False):   # bf: keep this matrix in BF16 storage in a W12 engine (MK_W12_BF_OUT=1: out_proj, o_proj)
            M, Kc = W.shape
            t = to_frag(W)
            if not self.w12 or bf:
                self.weight_bytes += t.numel() * 2
                return K(t)
            lo = w12_window_rows(W)
            NT, KQ = M // 16, Kc // 64
            packed, out, vals = w12_pack(t, NT, KQ, lo)
            self.weight_bytes += packed.numel() * 4 + M * 4
            self.w12_stats["weights"] += t.numel()
            if os.environ.get("MK_W12_VERIFY") == "1":   # decode as the kernel does and compare every weight
                partition(NT, KQ)
                if W12_EXACT:
                    ent, CAP = w12_escapes_fixed(out, vals, tabs[(NT, KQ)], NT, KQ)
                    back = w12_unpack_fixed(packed, NT, KQ, lo, ent, CAP, tabs[(NT, KQ)])
                else:
                    hdr, ent = w12_escapes(out, vals, tabs[(NT, KQ)], NT, KQ)
                    back = w12_unpack(packed, NT, KQ, lo, hdr, ent, tabs[(NT, KQ)])
                ref = t.view(NT, KQ, 4, 32, 8).to(torch.int32) & 0xFFFF
                bad = int((back != ref).sum())
                self.w12_stats["verified_weights"] = self.w12_stats.get("verified_weights", 0) + t.numel()
                self.w12_stats["mismatches"] = self.w12_stats.get("mismatches", 0) + bad
                self.w12_stats["verified_tensors"] = self.w12_stats.get("verified_tensors", 0) + 1
                del back, ref
            return {"ptr": K(packed), "ctab": K(w12_ctab(lo)), "out": out, "vals": vals}

        tables, tabs = {}, {}
        flags = (HERE / "build" / "flags").read_text() if (HERE / "build" / "flags").exists() else ""
        self.w12 = W12   # engine/TRACK "A W12": the one-token data flow with 12-bit weight storage (the run's track B)
        if not self.w12 and "-DW12_ONLY" in (HERE / "build" / "flags").read_text() if (HERE / "build" / "flags").exists() else False:
            raise RuntimeError("this build has no BF16 stream loops (W12_ONLY): set MK_W12=1 or rebuild with MK_NO_W12_ONLY=1")
        self.w12_stats = {"weights": 0, "escapes": 0}
        self.alt_rows = {}

        def partition(NT, KQ):
            """Equal contiguous shares per warp (in quads of 4 k-steps, tile-major) and, per tile, the first and
            the last block that hold a piece of it."""
            if (NT, KQ) not in tables:
                nw, UQ, nb = self.nwarps, NT * KQ, self.nblocks
                assert UQ >= nw
                tab = (np.arange(nw + 1, dtype=np.int64) * UQ) // nw
                first = np.searchsorted(tab, np.arange(NT, dtype=np.int64) * KQ, side="right") - 1
                last = np.searchsorted(tab, (np.arange(NT, dtype=np.int64) + 1) * KQ - 1, side="right") - 1
                tiles = (first // 32) | ((last // 32) << 16)
                tables[(NT, KQ)] = (K(torch.from_numpy(tab.astype(np.int32)).to(self.dev)),
                                    K(torch.from_numpy(tiles.astype(np.int32)).to(self.dev)))
                tabs[(NT, KQ)] = torch.from_numpy(tab).to(self.dev)
            return tables[(NT, KQ)]

        def row(type_, ep=0, NT=0, KQ=0, use_r=0, M=0, i6=0, i7=0, W=0, xf=0, out=0, gnext=0, p=()):
            if type_ == OP_GEMM:
                p = partition(NT, KQ)
                if isinstance(W, dict) and W12_EXACT:   # exact W12: fixed-stride lists, patched into the MMA operands
                    ent, CAP = w12_escapes_fixed(W["out"], W["vals"], tabs[(NT, KQ)], NT, KQ)
                    if os.environ.get("MK_W12_DROP_ESC") == "1":   # timing experiment only: wrong results
                        ent.fill_(0)
                    self.w12_stats["escapes"] += int((ent.view(-1, 2)[:, 0] != 0).sum())
                    self.w12_stats["list_bytes"] = self.w12_stats.get("list_bytes", 0) + ent.numel() * 4
                    p = p + (0, K(ent), W["ctab"])
                    i7 = (1 << 32) | CAP
                    W = W["ptr"]
                elif isinstance(W, dict):   # 12-bit storage: code base, per-warp lists of the out-of-window weights
                    start, ent = w12_escapes(W["out"], W["vals"], tabs[(NT, KQ)], NT, KQ)
                    self.w12_stats["escapes"] += ent.numel() // 2 - self.nwarps - 32
                    if os.environ.get("MK_W12_STATS") == "1":
                        st = (start & 0xFFFFF).cpu(); cw = torch.cat([st[1:], torch.tensor([ent.numel() // 2 - 32], dtype=st.dtype)]) - st - 1
                        self.w12_stats.setdefault("per_op", []).append((NT, KQ, int(cw.sum()), int(cw.max()), float(cw.float().quantile(0.99))))
                    p = p + (K(start), K(ent), W["ctab"])
                    i7 = 1 << 32
                    W = W["ptr"]
            return [type_, ep, NT, KQ, use_r, M, i6, i7, W, xf, out, gnext] + list(p) + [0] * (8 - len(p))

        XF, OUT = 9, 10      # columns of an op row
        latest = {}          # the newest x buffer of each width, in program order

        def xbuf(ksteps):
            return self._z(ksteps * 4 * 128, dtype=torch.int16).data_ptr()

        def wire(rows):
            """Every producer of activations writes a buffer of its own (so that a consumer may read it through
            L1: nobody read that address earlier in the launch); every GEMM reads the newest buffer of its width."""
            for r in rows:
                t = r[0]
                if t == OP_GEMM:
                    r[XF] = latest[r[3] * 4]
                    if r[1] in (EP_RESID, EP_SET):
                        r[OUT] = latest[320] = xbuf(320)
                    elif r[1] == EP_GATEUP:
                        r[OUT] = latest[1088] = xbuf(1088)
                elif t == OP_EMBED:
                    r[OUT] = latest[320] = xbuf(320)
                elif t == OP_LMIX or t == OP_ATT_OUT or (t == OP_ATT_VSUM and r[6]):
                    r[XF] = latest[384] = xbuf(384)
                elif t == OP_MTP_IN:
                    latest[640] = r[OUT]
            return rows

        def attention(prefix, ops):
            Win = torch.cat([wts.get(prefix + "q_proj.weight"), wts.get(prefix + "k_proj.weight"), wts.get(prefix + "v_proj.weight")], 0)
            ops.append(row(OP_GEMM, EP_STORE, NT=Win.shape[0] // 16, KQ=80, use_r=1, M=Win.shape[0], W=frag(Win),
                           out=bf["mbuf"].data_ptr()))
            del Win
            Kc = self._z(B * 4 * mc * 256, dtype=torch.int16)
            Vc = self._z(B * 4 * mc * 256, dtype=torch.int16)
            p = (Kc.data_ptr(), Vc.data_ptr(), gain(prefix + "q_norm.weight"), gain(prefix + "k_norm.weight"))
            if os.environ.get("MK_ATT_OUT_OP", "0") == "1":   # the old separate output op (A/B)
                for t in (OP_ATT_PREP, OP_ATT_SCORE, OP_ATT_SOFTMAX, OP_ATT_VSUM, OP_ATT_OUT):
                    ops.append(row(t, p=p))
            else:   # the value-sum op finishes the output itself (last chunk per output unit)
                for t in (OP_ATT_PREP, OP_ATT_SCORE, OP_ATT_SOFTMAX):
                    ops.append(row(t, p=p))
                ops.append(row(OP_ATT_VSUM, i6=1, p=p))
            return wts.get(prefix + "o_proj.weight")

        def mlp(lp, Wout, g_next, ops):
            gn = gain(lp + "post_attention_layernorm.weight")
            r = row(OP_GEMM, EP_RESID, NT=320, KQ=96, W=frag(Wout, bf=os.environ.get("MK_W12_BF_OUT", "0") == "1"), gnext=gn)
            ops.append(r)
            if self.w12 and W12_DUAL:   # a BF16 copy for one-column passes (B = 1): there out_proj / o_proj stream faster
                self.alt_rows[id(r)] = row(OP_GEMM, EP_RESID, NT=320, KQ=96, W=frag(Wout, bf=True), gnext=gn)   # in BF16 (6 quads per warp)
            gate, up = wts.get(lp + "mlp.gate_proj.weight"), wts.get(lp + "mlp.up_proj.weight")
            Wgu = torch.stack([gate.view(FF // 8, 8, HID), up.view(FF // 8, 8, HID)], 1).reshape(2 * FF, HID)
            del gate, up
            ops.append(row(OP_GEMM, EP_GATEUP, NT=2 * FF // 16, KQ=80, use_r=1, W=frag(Wgu)))
            del Wgu
            ops.append(row(OP_GEMM, EP_RESID, NT=320, KQ=FF // 64, W=frag(wts.get(lp + "mlp.down_proj.weight")), gnext=g_next))

        g_in = [gain(f"{pre}layers.{i}.input_layernorm.weight") for i in range(NL)] + [gain(f"{pre}norm.weight")]
        self.emb = wts.get(f"{pre}embed_tokens.weight").view(torch.int16).contiguous()
        keep.append(self.emb)
        body = [row(OP_EMBED, W=self.emb.data_ptr(), gnext=g_in[0])]
        self.layer_end = []
        for i in range(NL):
            lp = f"{pre}layers.{i}."
            if types[i] == "linear_attention":
                la = lp + "linear_attn."
                Win = torch.cat([wts.get(la + "in_proj_qkv.weight"), wts.get(la + "in_proj_z.weight"),
                                 wts.get(la + "in_proj_b.weight"), wts.get(la + "in_proj_a.weight")], 0)
                body.append(row(OP_GEMM, EP_STORE, NT=Win.shape[0] // 16, KQ=80, use_r=1, M=Win.shape[0], W=frag(Win),
                                out=bf["mbuf"].data_ptr()))
                del Win
                S = self._z(2 * B * 48 * 16384)   # two buffers: committed state, a speculative pass's final state
                ring = self._z(B * RING * 10240, dtype=torch.int16)
                convw = K(f32(wts.get(la + "conv1d.weight").view(10240, 4)))
                negA = K(f32(-(wts.get(la + "A_log").float().exp())))
                dtb = K(f32(wts.get(la + "dt_bias")))
                nwt = K(f32(wts.get(la + "norm.weight")))
                pk = self._z(B * 48 * PMAX * 128)
                pd = self._z(B * 48 * PMAX * 128)
                pdec = self._z(B * 48 * PMAX)
                if not hasattr(self, "lmx"):   # split recurrent op (one sequence): warp sums [48][PMAX][4] + flags [48][2] u64
                    self.lmx = self._z(48 * PMAX * 4 + 48 * 2 * 2)
                    self.lmx.view(torch.int32).fill_(0x7fbadbad)   # norm-sum slots start as sentinels (mk.cu SENT_BITS)
                body.append(row(OP_LMIX, W=pdec.data_ptr(), out=self.lmx.data_ptr(), M=B * 48 * 16384,
                                p=(S.data_ptr(), ring.data_ptr(), convw, negA, dtb, nwt, pk.data_ptr(), pd.data_ptr())))
                Wout = wts.get(la + "out_proj.weight")
            else:
                Wout = attention(lp + "self_attn.", body)
            mlp(lp, Wout, g_in[i + 1], body)
            del Wout
            self.layer_end.append(len(body))
        self.model_bytes = self.weight_bytes
        self.head_ptr = frag(wts.get("lm_head.weight"))
        fold = os.environ.get("MK_ARGMAX_FOLD", "1") == "1"
        am, ad = (bf["amax_main"].data_ptr(), bf["amax_draft"].data_ptr()) if fold else (0, 0)
        head = [row(OP_GEMM, EP_LOGITS, NT=VOCAB // 16, KQ=80, use_r=1, i6=1, W=self.head_ptr, out=am),
                row(OP_ARGMAX, NT=VOCAB // 16, out=bf["tok_res"].data_ptr(), p=(am,))]
        wire(body)
        x_main = latest[320]          # bf16(h (1+w_norm)) of the last layer: input of lm_head, source of the drafter
        wire(head)
        progs = {"full": body + head + [row(OP_FINISH, i6=1)], "nohead": body + [row(OP_FINISH)]}
        # "full2": every run (prep, score, softmax, value sum) of an attention layer replaced by one OP_ATT_DIST row (same
        # caches and gains, the value-sum row's output buffer). Its hand-over slots hold sentinels (mk.cu SENT_BITS)
        # between ops: (max, 1 / sum) per unit and reader chunk, 6 score rows per unit, value sums per (unit, dim tile, chunk)
        self.att_mi = self._z(32 * 10 * 12); self.att_sc = self._z(32 * 6 * mc); self.att_vp = self._z(32 * 16 * 10 * 32 * 4)
        for t_ in (self.att_mi, self.att_sc, self.att_vp):
            t_.view(torch.int32).fill_(0x7fbadbad)
        body2, i = [], 0
        while i < len(body):
            r = body[i]
            if r[0] == OP_ATT_PREP and [x[0] for x in body[i:i + 4]] == [OP_ATT_PREP, OP_ATT_SCORE, OP_ATT_SOFTMAX, OP_ATT_VSUM]:
                f = list(r); f[0] = OP_ATT_DIST; f[6] = 0; f[XF] = body[i + 3][XF]
                f[16] = self.att_mi.data_ptr(); f[17] = self.att_sc.data_ptr(); f[18] = self.att_vp.data_ptr()
                body2.append(f); i += 4
            else:
                body2.append(r); i += 1
        if len(body2) < len(body):
            progs["full2"] = body2 + head + [row(OP_FINISH, i6=1)]
        if self.alt_rows:   # one-column decode passes: the same program with the BF16 copies of out_proj / o_proj
            def alt(r):   # the BF16 twin of a wired row: same activation buffers (XF, OUT), BF16 weights and no lists
                a = self.alt_rows.get(id(r))
                if a is None:
                    return r
                a = list(a); a[XF] = r[XF]; a[OUT] = r[OUT]
                return a
            progs["full_b1"] = [alt(r) for r in progs["full"]]
            if "full2" in progs:
                progs["full2_b1"] = [alt(r) for r in progs["full2"]]

        if self.spec:
            # the checkpoint's multi-token-prediction head: mtp.fc, one full-attention layer, mtp.norm, shared lm_head
            mp = "mtp.layers.0."
            fc = frag(wts.get("mtp.fc.weight"))
            g_emb, g_hid = gain("mtp.pre_fc_norm_embedding.weight"), gain("mtp.pre_fc_norm_hidden.weight")
            fc_row = row(OP_GEMM, EP_SET, NT=320, KQ=160, W=fc, gnext=gain(mp + "input_layernorm.weight"))
            layer = []
            Wout = attention(mp + "self_attn.", layer)
            mlp(mp, Wout, gain("mtp.norm.weight"), layer)
            del Wout
            dhead = [row(OP_GEMM, EP_LOGITS, NT=DRAFT_ROWS // 16, KQ=80, use_r=1, W=self.head_ptr, out=ad),
                     row(OP_ARGMAX, NT=DRAFT_ROWS // 16, out=bf["mtp_res"].data_ptr(), p=(ad,))]

            x_fc = xbuf(640)
            mtp_rows = wire([row(OP_MTP_IN, out=x_fc), fc_row] + layer + dhead)[1:]
            x_mtp = latest[320]       # the drafter's own post-layer x: source of the next chained draft

            def mtp_in(src, idx=0, stride=0, hidden=x_main):
                return row(OP_MTP_IN, i6=src, i7=idx, M=stride, W=self.emb.data_ptr(), gnext=g_emb, out=x_fc, p=(g_hid, hidden))

            progs["verify"] = body + head + [mtp_in(0)] + mtp_rows + [row(OP_ACCEPT)]
            progs["verify_nomtp"] = body + head + [row(OP_ACCEPT, i6=1)]   # while copied drafts keep being accepted
            progs["prefill"] = body + [mtp_in(2)] + mtp_rows[:3] + [row(OP_FINISH)]
            for Pv in range(3, PMAX + 1):
                for i in range(2, min(Pv, CHAIN_MAX + 1)):
                    progs[f"chain{Pv}_{i}"] = [mtp_in(1, i - 1, Pv, x_mtp)] + mtp_rows + [row(OP_CHAIN_FIN, i6=i, M=Pv)]
        self.prog = {}
        self.pf = {}
        self.prog_np = {}      # the programs' rows on the host: the prefetch table of a batch size is made on first use
        for name, rows in progs.items():
            a = np.array(rows, dtype=np.int64)
            assert a.shape[1] == OPW
            for o in range(a.shape[0]):   # p5 of a GEMM: the conv history ring of the recurrent op right after it (its rows are
                if a[o, 0] == OP_GEMM:    # prefetched during the GEMM, mk.cu ring_prefetch), else 0
                    a[o, 13] = a[o + 1, 13] if (o + 1 < a.shape[0] and a[o + 1, 0] == OP_LMIX) else 0
            self.prog[name] = torch.from_numpy(a).to(self.dev)
            self.pf[name] = self._prefetch_table(a)
            self.prog_np[name] = a
        self.keep += list(self.prog.values()) + [t for t in self.pf.values() if t is not None]
        self.prog_full, self.prog_nohead = self.prog["full"], self.prog["nohead"]
        self.prog_b1 = self.prog.get("full_b1")
        self.pf_by_ptr = {self.prog[k].data_ptr(): self.pf[k] for k in self.prog}
        self.name_by_ptr = {self.prog[k].data_ptr(): k for k in self.prog}
        self.pf_b = {}
        torch.cuda.synchronize()
        wts.maps.clear()

    def _prefetch_table(self, a, B=0):
        """[nops][nblocks][2] (address, bytes): at the barrier after op o, block b asks L2 for the first bytes of its
        share of the next weight matrix: PF_NEAR bytes when that matrix is the very next op (covers the start of the
        stream while the slowest blocks finish), PF_FAR when small ops (recurrent state, attention) come first."""
        # Re-swept with the weight streams on ld.global.cs (s9, BF16 engine, same card, B=1 / B=8 us per step): near 128 KB
        # far 0: 8664 / 9221; 160 KB / 0: 8649 / 9174; 128 KB / 64 KB: 8649 / 9195; 144 KB / 64 KB: 8624 / 9155;
        # 160 KB / 64 KB: 8622 / 9140; 176 KB / 64 KB: 8633 / 9140; near 256 KB +105 / +70, far 512 KB +82 / +115.
        # Before (cg loads): near 128 KB, far 0 (BF16; far 1 MB +0.6 to +1.6 %) and far 96 KB (W12, s5).
        # W12 engine with cs loads (s9): 128 / 96 KB 7838 / 8372; 144 / 64 KB 7840 / 8350; 160 / 96 KB 7845 / 8357; 192 / 96 KB +47 / +25.
        near = int(os.environ.get("MK_PF_NEAR", "147456" if self.w12 else "163840"))
        # far piece, BF16 engine: 128 KB since the barrier issues one piece per poll and none after the release (s0020). On
        # the evaluator's card, B=1 / B=8 us per step: 64 / 80 KB (s0020) 8462 / 8858, 96 KB 8453 / 8851, 112 KB 8451 / 8851,
        # 128 KB 8448 / 8850, 160 KB 8454 / 8859. Card-specific at this level (another card of the pod: +8 / +58 at 128 KB).
        # 12-bit engine: 64 KB (96 KB -2 / 0, 128 KB +2 / +2).
        far = int(os.environ.get("MK_PF_FAR", "65536" if self.w12 else "131072"))
        if near <= 0 and far <= 0:
            return None
        n, nb, nw = a.shape[0], self.nblocks, self.nwarps
        t = np.zeros((n, nb, 2), dtype=np.int64)
        gemms = [i for i in range(n) if a[i, 0] == OP_GEMM]
        # MK_PF_CUM=1: the barriers before a GEMM that is not the next op (recurrent op, attention ops in between) ask
        # for consecutive pieces of the block's share instead of the same first bytes again
        # (same card, B=1 / B=8 us per step: 8621 -> 8605 / 9168 -> 9150 with 64 KB pieces; 96 KB 8612 / 9160, 128 KB 8612 / 9165)
        cum = os.environ.get("MK_PF_CUM", "1") == "1"
        # Idle blocks: in a one-token pass of B <= 3 sequences the recurrent op leaves blocks without a unit (B=1: 48 heads
        # on 2 x 48 of 152 blocks). At the barrier before it the last MK_PF_IDLE_N of them ask for their whole share of
        # the matrix after it (out_proj, 414 KB per block). B=1, same card: none 8612; the last 16: 8591, 24: 8578,
        # 32: 8579, 40: 8574, 48: 8583, all 56: 8588 (more prefetch traffic delays the recurrent op's own loads);
        # 192 or 256 KB instead of the whole share: 8590-8597. (MK_PF_IDLE=n: blocks n.. whatever the batch.)
        idle0 = int(os.environ.get("MK_PF_IDLE", "0"))
        if B > 0 and "MK_PF_IDLE" not in os.environ:   # the blocks without a recurrent unit in a one-token pass of B sequences
            u = B * 48
            busy = 2 * u if 2 * u <= nb else u if u <= nb else (u + 1) // 2 if u <= 2 * nb else nb
            k = min(nb - busy, int(os.environ.get("MK_PF_IDLE_N", "32"))) if B <= 3 else 0   # measured at B=1 only
            idle0 = nb - k if k > 0 else 0
        g_prev, off = -1, np.zeros(nb, dtype=np.int64)
        for o in range(n):
            nxt = [g for g in gemms if g > o]
            if not nxt:
                continue
            g = nxt[0]
            if g != g_prev:
                g_prev, off = g, np.zeros(nb, dtype=np.int64)
            budget = near if g == o + 1 else far
            if o + 1 < n and a[o + 1, 0] == OP_ATT_DIST:
                # the barrier before the one-op attention layer (few blocks work in it at small batches): the whole rest of
                # the block's o_proj share at B <= 3, the usual far piece otherwise. Same card, us per step: B=1 far piece
                # 8566, 128 KB 8557, 256 KB 8571, whole share 8520; B=8 9094, 9097, 9122, 9100.
                budget = int(os.environ.get("MK_PF_FAR_ATT", str(1 << 30 if 0 < B <= 3 else far)))
            if budget <= 0:
                continue
            NT, KQ, W = int(a[g, 2]), int(a[g, 3]), int(a[g, 8])
            qb = 1536 if (int(a[g, 7]) >> 32) else 2048
            UQ = NT * KQ
            starts = (np.arange(nb, dtype=np.int64) * 32 * UQ) // nw
            ends = (np.arange(1, nb + 1, dtype=np.int64) * 32 * UQ) // nw
            bud = np.full(nb, budget, dtype=np.int64)
            if idle0 > 0 and o + 1 < n and a[o + 1, 0] == OP_LMIX:
                bud[idle0:] = int(os.environ.get("MK_PF_IDLE_BYTES", str(1 << 30)))
            t[o, :, 0] = W + starts * qb + off
            t[o, :, 1] = np.clip(np.minimum((ends - starts) * qb - off, bud), 0, None) // 16 * 16
            if cum:
                off = off + bud
            if o + 1 < n and a[o + 1, 0] == OP_ATT_DIST and os.environ.get("MK_PF_ATT_K", "1" if B >= 4 else "0") == "1" and B > 0:
                # the blocks of the one-op attention layer (unit u, chunk ch -> block u S + ch) ask for their chunk of the
                # K cache (131 KB) at the barrier before it instead of a weights piece: the op's largest phase was the
                # scores waiting for K (phase timer: 4.5 us of 15 at B=1, 7 of 21 at B=8). Same card, BF16 engine:
                # B=8 8937 -> 8880 us per step; B=1 8530 -> 8526 (left off below 4 sequences)
                S_ = nb // (4 * B)
                for u in range(4 * B):
                    for ch in range(min(S_, int(os.environ.get("MK_PF_ATT_CH", "5")))):
                        t[o, u * S_ + ch, 0] = int(a[o + 1, 12]) + u * self.mc * 512 + ch * 131072
                        t[o, u * S_ + ch, 1] = 131072
                        off[u * S_ + ch] = 0   # nothing of the next matrix asked for yet by this block
        return torch.from_numpy(t).to(self.dev)

    # ------------------------------------------------------------------ launching
    def _launch(self, B, P, tok_ptr, tok_stride, flags, logits_ptr=0, tok_out_ptr=0, prog=None, nops=None, advance=None, **kw):
        if prog is None:
            fuse = (ATT_FUSE and P == 1 and B <= 8 and not (flags & (F_SPEC | F_CHAIN)) and "full2" in self.prog
                    and ((self.pos_ub + 256) >> 8) <= min(10, self.nblocks // (4 * B)))   # the unit's key chunks fit its blocks
            if self.prog_b1 is not None and B * P == 1:
                prog = self.prog["full2_b1"] if fuse else self.prog_b1
            else:
                prog = self.prog["full2"] if fuse else self.prog_full
        nops = prog.shape[0] if nops is None else nops
        NC = B * P
        cx = Ctx()
        cx.B, cx.P, cx.NC, cx.NG = B, P, NC, ngroups(NC)
        cx.nwarps, cx.nblocks, cx.max_ctx, cx.flags = self.nwarps, self.nblocks, self.mc, flags
        cx.pos_ub, cx.NGq, cx.tok_stride = self.pos_ub, (6 * P + 7) // 8, tok_stride
        cx.bar_base = self.bar_base
        for name, t in self.buf.items():
            setattr(cx, name, t.data_ptr())
        cx.tok_in, cx.logits, cx.tok_out = tok_ptr, logits_ptr, tok_out_ptr
        pft = self.pf_by_ptr.get(prog.data_ptr())
        if P == 1 and not (flags & (F_SPEC | F_CHAIN)) and prog.data_ptr() in self.name_by_ptr:
            key = (prog.data_ptr(), B)   # one-token passes: the table of this batch size (which blocks idle in the recurrent op)
            if key not in self.pf_b:
                self.pf_b[key] = self._prefetch_table(self.prog_np[self.name_by_ptr[prog.data_ptr()]], B)
            pft = self.pf_b[key]
        cx.pft = pft.data_ptr() if (pft is not None and nops == prog.shape[0]) else 0
        cx.ts = self.ts.data_ptr() if getattr(self, "ts", None) is not None else 0
        cx.wts = self.wts.data_ptr() if getattr(self, "wts", None) is not None else 0
        cx.wts_op = getattr(self, "wts_op", -1)
        cx.bts = self.bts.data_ptr() if getattr(self, "bts", None) is not None else 0
        cx.src_P, cx.src_NG = P, cx.NG
        cx.next_P = P
        for k, v in kw.items():
            setattr(cx, k, v)
        if self.defer_n > 1 and B >= DEFER_MIN_B and not self.spec and not (flags & (F_SPEC | F_CHAIN)) and nops == prog.shape[0]:
            cx.flags |= F_DEFER   # whole passes of the plain programs only (every one ends with OP_FINISH)
            cx.defer_n = self.defer_n
        cx.NG = ngroups(cx.NC)
        if "src_NG" not in kw:
            cx.src_NG = cx.NG
        if cx.pos_ub + P > self.mc:
            raise RuntimeError("context longer than max_context")
        # shared memory: context, r, then the work area (recurrent op: 2 x 5.7 KB; GEMM: partial sums of cut tiles)
        PM = 8 if P <= 8 and cx.na_max <= 8 else PMAX   # the replay stages the previous pass's accepted positions too
        lmb = (5 * PM * 128 + 10 * PM + 32) * 4
        if P == 1:   # the recurrent op may run two units per block at one position (lmix_dual): two layouts
            lmb = max(lmb, 2 * (5 * 8 * 128 + 10 * 8 + 32) * 4)
        smem = 1280 + max(lmb, 32 * 2 * cx.NG * 512 + int(os.environ.get("MK_HC", "6")) * 16 * NCM * 2 + (4096 if cx.NC > 4 else 0))   # + r partials (R_ASYNC_MAX)   # recurrent op (PM positions); tile-sum slots + residual rows
        if self.w12:   # the W12 area after the GEMM work area of this many column groups (W12S_F in mk.cu)
            smem = max(smem, 1280 + (32 * 2 * cx.NG * 512 + int(os.environ.get("MK_HC", "6")) * 16 * NCM * 2 + 4096) + 72 * 32 * 4)
        smem = max(smem, int(os.environ.get("MK_SMEM_MIN", "0")))   # A/B tests of the L1 / shared-memory split
        if self.lib.mk_launch(prog.data_ptr(), nops, ctypes.byref(cx), smem):
            raise RuntimeError("mk_launch failed")
        self.bar_base += nops * self.nblocks
        self.pos_ub += P if advance is None else advance
        self.launches += 1

    # ------------------------------------------------------------------ the offline contract
    def reset(self) -> None:
        self.buf["pos"].zero_()
        self.buf["nacc"].zero_()
        self.pos_ub = 0

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        logits = torch.empty(B, VOCAB, dtype=torch.bfloat16, device=self.dev)
        self._launch(B, 1, tokens.data_ptr(), tokens.stride(0), F_LOGITS, logits_ptr=logits.data_ptr())
        self._hold = (tokens, logits)
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        B = tokens.shape[0]
        tb = self.buf["tokbuf"]
        tb[:B] = tokens
        out = torch.empty(n_steps, B, dtype=torch.int64, device=self.dev)
        for i in range(n_steps):
            self._launch(B, 1, tb.data_ptr(), 1, F_FEEDBACK, tok_out_ptr=out[i].data_ptr())
        self._hold = (tokens, out)
        return out.t()

    def close(self) -> None:
        torch.cuda.synchronize()
        self.keep = []
        self.buf = {}

    def stats(self) -> dict:
        s = {"implementation": "megakernel, one launch per model pass, weights in MMA fragment order", "track": "C" if TRACK != "A" else ("B" if self.w12 else "A"),   # the project's names: A checkpoint BF16 storage, one token per pass; B: + exact 12-bit weight storage (bit-identical logits); C: draft-and-verify (MK_TRACK=B here)
             "launches": self.launches, "weight_bytes": self.weight_bytes, "sm_count": self.nblocks,
             "max_shared_memory": self.max_smem,
             "load_seconds": round(self.load_seconds, 1)}
        s.update(self.counters)
        if self.w12:
            s["w12"] = dict(self.w12_stats, bits_per_weight=round((self.weight_bytes * 8 + self.w12_stats["escapes"] * 64) / max(1, self.w12_stats["weights"]), 4))
        if self.counters["passes"]:
            s["tokens_per_verify_pass"] = round(self.counters["tokens"] / self.counters["passes"], 3)
        return s

    # ------------------------------------------------------------------ the online contract (HTTP server)
    def submit(self, rid, prompt, n) -> None:
        self.waiting.append((int(rid), list(prompt), int(n)))
        if self.first_wait is None:
            self.first_wait = time.perf_counter()

    def _take_cohort(self):
        length = len(self.waiting[0][1])
        cohort = [r for r in self.waiting if len(r[1]) == length][:self.Bmax]
        self.waiting = [r for r in self.waiting if all(r is not c for c in cohort)]
        self.first_wait = time.perf_counter() if self.waiting else None
        return cohort, length

    def _start_cohort(self):
        cohort, L = self._take_cohort()
        B = len(cohort)
        nmax = max(r[2] for r in cohort)
        ids = torch.tensor([r[1] for r in cohort], dtype=torch.int64, device=self.dev).t().contiguous()  # [L, B]
        tb = self.buf["tokbuf"]
        self.reset()
        # positions per verify pass: 8 columns fit one MMA; a single sequence goes to 16 while its drafts are
        # copied from the context (they are almost always accepted), a batch of 8 uses 4 (32 columns)
        Pv = int(os.environ.get("MK_PV", "0")) or (4 if B == 1 else 4 if B <= 8 else 2)   # B=1: 4 measured best on prose (5071 vs 5275 us at 8)
        Pv = max(1, min(Pv, PMAX, NCM // B))
        Phi = int(os.environ.get("MK_PV_HI", "0")) or (32 if B == 1 else Pv)
        Phi = max(Pv, min(Phi, PMAX, NCM // B))
        spec = self.spec and Pv >= 2 and L >= 2 and L + nmax + 2 * PMAX < self.mc
        co = {"reqs": cohort, "B": B, "n": nmax, "ids": ids, "queued": 0, "done": 0, "events": [], "spec": spec,
              "sent": [0] * B}
        if not spec:   # prompt in chunks of up to NCM // B positions (bit-identical to single steps)
            pc = max(1, min(int(os.environ.get("MK_PREFILL_CHUNK", str(NCM // B))), PMAX, NCM // B))
            for t0 in range(0, L - 1, pc):
                Pp = min(pc, L - 1 - t0)
                cur = ids[t0:t0 + Pp].t().contiguous()            # [B, Pp]: column c = b * Pp + p
                self._hold_prefill = cur
                self._launch(B, Pp, cur.data_ptr(), 1, 0, prog=self.prog_nohead)
            tb[:B] = ids[L - 1]
            co["hmap"] = self.hmap is not None and nmax * B <= self.hmap.size
            if co["hmap"]:
                self.hmap[:nmax * B] = -1   # not written yet (the previous cohort's passes have all ended)
                self.hmap_tag = self.hmap_tag % 1000000 + 1   # in the upper half of every token of this cohort: a slot
                co["tag"] = self.hmap_tag                     # counts only when it carries this cohort's tag
            else:
                co["out"] = torch.empty(nmax, B, dtype=torch.int64, device=self.dev)
                co["host"] = torch.empty(nmax, B, dtype=torch.int64).pin_memory()
        else:
            # prompt: the drafter's K/V history is built on the way. Chunks of up to NCM // B positions per pass
            # (a multi-position pass is bit-identical to single steps); MK_PREFILL_CHUNK=1: one position per pass
            pc = max(1, min(int(os.environ.get("MK_PREFILL_CHUNK", str(NCM // B))), PMAX, NCM // B))
            for t0 in range(0, L - 1, pc):
                Pp = min(pc, L - 1 - t0)
                if Pp == 1:
                    self._launch(B, 1, ids[t0].data_ptr(), 1, 0, prog=self.prog["prefill"], tok_next=ids[t0 + 1].data_ptr())
                    continue
                cur = ids[t0:t0 + Pp].t().contiguous()            # [B, Pp]: column c = b * Pp + p
                nxt = ids[t0 + 1:t0 + 1 + Pp].t().contiguous()
                self._hold_prefill = (cur, nxt)
                self._launch(B, Pp, cur.data_ptr(), 1, 0, prog=self.prog["prefill"], tok_next=nxt.data_ptr())
            tb.view(-1)[:B * Pv] = ids[L - 1].repeat_interleave(Pv)   # column 0: the last prompt token; no drafts yet
            lim = torch.tensor([L - 1 + r[2] for r in cohort], dtype=torch.int32, device=self.dev)
            self.buf["limit"][:B] = lim
            self.buf["lkf"].zero_()
            self.buf["hist"].view(-1, self.mc)[:B, :L] = ids.t().to(torch.int32)   # the drafter may copy from the context
            cap = nmax + 8
            if B > 1 and os.environ.get("MK_VARP", "1") == "1":
                hi = int(os.environ.get("MK_VAR_HI", "16")); lo = int(os.environ.get("MK_VAR_LO", "1"))   # columns to the copyable sequences: +1.5-2 % (B=8 policy A/B, hour 5.8)
                tb2 = self.tokbuf2
                tb2[:B * lo] = ids[L - 1].repeat_interleave(lo)
                tb.view(-1)[:B * lo] = ids[L - 1].repeat_interleave(lo)
                co.update(var=True, hi=hi, lo=lo, lo_max=int(os.environ.get("MK_VAR_LO_MAX", "4")), lkb=[False] * B,
                          cur_len=[lo] * B, par=0, tbs=[tb, tb2], Ls=[])
            co.update(Pv=Pv, Plo=Pv, Phi=Phi, Ps=[], lk=False, cap=cap, known_pos=L - 1, limit=L - 1 + nmax, prev_P=0,
                      out=torch.empty(cap, max(B * Phi, NCM), dtype=torch.int64, device=self.dev),   # per-sequence lengths: up to NCM columns
                      cnt=torch.empty(cap, 2 * B, dtype=torch.int32, device=self.dev),
                      host=torch.empty(cap, max(B * Phi, NCM), dtype=torch.int64).pin_memory(),
                      hcnt=torch.empty(cap, 2 * B, dtype=torch.int32).pin_memory())
        self.cohort = co

    def _queue_plain(self, co):
        tb = self.buf["tokbuf"]
        while co["queued"] < co["n"] and co["queued"] - co["done"] < 4:
            i = co["queued"]
            if co["hmap"]:
                self._launch(co["B"], 1, tb.data_ptr(), 1, F_FEEDBACK, tok_out_ptr=self.hmap_dev + i * co["B"] * 8, tok_tag=co["tag"])
            else:
                self._launch(co["B"], 1, tb.data_ptr(), 1, F_FEEDBACK, tok_out_ptr=co["out"][i].data_ptr())
                co["host"][i].copy_(co["out"][i], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                co["events"].append(ev)
            co["queued"] += 1

    def _depth(self, co):
        """Verify passes in flight: the first passes of a cohort one at a time, so that a copyable continuation found
        by a pass sets the length of the next pass but one instead of the one after that (one ramp pass less on
        looping text; the GPU waits ~0.2 ms for the host per pass while this lasts); then two."""
        return 1 if co["queued"] < int(os.environ.get("MK_RAMP_SINGLE", "8")) else 2

    def _queue_spec(self, co):
        tb, B, Phi = self.buf["tokbuf"], co["B"], co["Phi"]
        NG1 = ngroups(B)
        while co["queued"] < co["cap"] and co["queued"] - co["done"] < self._depth(co) and min(co["sent"]) < co["n"]:
            i = co["queued"]
            inflight = co["queued"] - co["done"]
            P = co["Pv"]                                   # this pass
            NP = co["Phi"] if co["lk"] else co["Plo"]      # the next one: deep while the context-copy drafter is on
            ramp = inflight == 0 and RAMP_LATE and B == 1
            if ramp:
                # nothing in flight: this pass's length follows the last pass's result (its accept op wrote up to
                # Phi - 1 copied drafts), and this pass again leaves room for Phi - 1 of them
                P = co["Phi"] if co["lk"] else co["Plo"]
                NP = co["Phi"]
            NGv = ngroups(B * P)
            # the MTP head only validates copies while a copied continuation keeps being accepted in full: skipped
            nomtp = co.get("loop_ok", False) and P == co["Phi"] and NP == co["Phi"] and MTP_SKIP
            self.pos_ub = min(co["known_pos"] + inflight * Phi, co["limit"])
            self._launch(B, P, tb.data_ptr(), 1, F_SPEC, prog=self.prog["verify_nomtp" if nomtp else "verify"],
                         tok_out_ptr=co["out"][i].data_ptr(),
                         cnt_out=co["cnt"][i].data_ptr(), na_max=co["prev_P"], next_P=NP, advance=0)
            self.counters["nomtp_passes"] = self.counters.get("nomtp_passes", 0) + int(nomtp)
            co["prev_P"] = P
            ub = min(co["known_pos"] + (inflight + 1) * Phi, co["limit"])
            for k in range(2, 2 if nomtp else min(co["Plo"] if ramp else NP, CHAIN_MAX + 1)):   # MTP drafts 2 .. (draft 1 came out of the verify pass)
                self.pos_ub = ub + k - 2
                self._launch(B, 1, 0, 1, F_CHAIN, prog=self.prog[f"chain{NP}_{k}"], pos_off=k - 2, advance=0,
                             src_P=(P if k == 2 else 1), src_NG=(NGv if k == 2 else NG1), src_sel=(1 if k == 2 else 0))
                self.counters["chain_launches"] += 1
            co["cnt"][i, B:2 * B] = self.buf["lkf"][:B]
            co["host"][i, :B * P].copy_(co["out"][i, :B * P], non_blocking=True)
            co["hcnt"][i].copy_(co["cnt"][i], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            co["events"].append(ev)
            co["Ps"].append(P)
            co["Pv"] = ((co["Phi"] if co["lk"] else co["Plo"]) if ramp else NP)
            co["queued"] += 1

    def _lengths(self, co):
        """Positions per sequence for the next pass: sequences whose drafts were copied from the context get up to HI,
        the others LO (one MTP draft: no chain launch); at most NCM columns; spare columns go to the others."""
        B, hi, lo = co["B"], co["hi"], co["lo"]
        if os.environ.get("MK_VAR_POL", "0") == "1":
            # the batch ends when its slowest (prose) sequence ends: finished sequences keep one column, sequences
            # with MTP drafts get columns first (up to lo_max), the copyable ones take the rest (up to hi)
            fin = [co["sent"][j] >= co["reqs"][j][2] for j in range(B)]
            L = [1] * B
            for cap, sel in ((co["lo_max"], [j for j in range(B) if not fin[j] and not co["lkb"][j]]),
                             (hi, [j for j in range(B) if not fin[j] and co["lkb"][j]])):
                grew = True
                while grew and sum(L) < NCM:
                    grew = False
                    for j in sel:
                        if L[j] < cap and sum(L) < NCM:
                            L[j] += 1; grew = True
            return L
        L = [hi if f else lo for f in co["lkb"]]
        while sum(L) > NCM:
            j = max(range(B), key=lambda k: L[k])
            L[j] -= 1
        k = 0
        while sum(L) < NCM and any(L[j] < co["lo_max"] for j in range(B) if not co["lkb"][j]):
            j = [j for j in range(B) if not co["lkb"][j]][k % max(1, sum(1 for f in co["lkb"] if not f))]
            if L[j] < co["lo_max"]:
                L[j] += 1
            k += 1
        return L

    def _layout(self, cur, nxt):
        i = self.lay_i % 16
        self.lay_i += 1
        cs = np.concatenate([[0], np.cumsum(cur)]).astype(np.int32)
        ns = np.concatenate([[0], np.cumsum(nxt)]).astype(np.int32)
        self.lay_host[i, 0:len(cs)] = torch.from_numpy(cs)
        self.lay_host[i, 16:16 + len(ns)] = torch.from_numpy(ns)
        self.lay_host[i, 32:] = self._unit_table(cur)
        self.lay[i].copy_(self.lay_host[i], non_blocking=True)
        base = self.lay.data_ptr() + i * self.layw * 4
        return base, base + 16 * 4, cs

    def _unit_table(self, cur):
        """Recurrent units (sequence, value head) to blocks for one pass: longest first onto the least loaded block
        (cost ~ 6 + 0.8 us per position), at most 4 per block. Which block computes a unit does not change its
        arithmetic; the round-robin order put up to three long units on one block while others had short ones."""
        key = tuple(int(x) for x in cur)
        t = self.utab_cache.get(key)
        if t is None:
            import heapq
            nb, B = self.nblocks, len(cur)
            cost = [6.0 + 0.8 * cur[b] for b in range(B)]
            heap = [(0.0, bl) for bl in range(nb)]
            n = [0] * nb
            tab = np.full((nb, 4), -1, dtype=np.int32)
            if os.environ.get("MK_UTAB", "1") == "0":   # A/B: the kernel's round-robin order
                for bl in range(nb):
                    for k in range(4):
                        if bl + k * nb < B * 48: tab[bl, k] = bl + k * nb
            for u in ([] if os.environ.get("MK_UTAB", "1") == "0" else sorted(range(B * 48), key=lambda u: (-cost[u // 48], u))):
                load, bl = heapq.heappop(heap)
                while n[bl] >= 4:
                    load, bl = heapq.heappop(heap)
                tab[bl, n[bl]] = u; n[bl] += 1
                heapq.heappush(heap, (load + cost[u // 48], bl))
            t = torch.from_numpy(tab.reshape(-1))
            self.utab_cache[key] = t
        return t

    def _queue_spec_var(self, co):
        B = co["B"]
        NG1 = ngroups(B)
        while co["queued"] < co["cap"] and co["queued"] - co["done"] < self._depth(co) and min(co["sent"]) < co["n"]:
            i = co["queued"]
            inflight = co["queued"] - co["done"]
            cur = co["cur_len"]
            nxt = self._lengths(co)
            cst, nst, cs = self._layout(cur, nxt)
            tb_cur, tb_nxt = co["tbs"][co["par"]], co["tbs"][1 - co["par"]]
            P = max(cur)
            # no MTP head when no sequence needs its drafts next time: 1 position, or copies accepted in full
            okq = co.get("okseq", [False] * B)
            nomtp = MTP_SKIP and all(nxt[j] == 1 or okq[j] for j in range(B))
            self.counters["nomtp_passes"] = self.counters.get("nomtp_passes", 0) + int(nomtp)
            self.pos_ub = min(co["known_pos"] + (inflight + 1) * PMAX, co["limit"])
            self._launch(B, P, tb_cur.data_ptr(), 1, F_SPEC, prog=self.prog["verify_nomtp" if nomtp else "verify"],
                         tok_out_ptr=co["out"][i].data_ptr(),
                         cnt_out=co["cnt"][i].data_ptr(), na_max=co["prev_P"], next_P=max(nxt), advance=0,
                         NC=int(cs[-1]), cst=cst, nst=nst, sst=cst, tokbuf=tb_cur.data_ptr(), tokbuf_next=tb_nxt.data_ptr())
            co["prev_P"] = P
            for k in range(2, 2 if nomtp else min(max(nxt), CHAIN_MAX + 1)):   # MTP drafts 2.. for the sequences without copied drafts
                self.pos_ub = min(co["known_pos"] + (inflight + 1) * PMAX + k, co["limit"])
                self._launch(B, 1, 0, 1, F_CHAIN, prog=self.prog[f"chain{PMAX}_{k}"], pos_off=k - 2, advance=0,
                             src_P=1, src_NG=(ngroups(int(cs[-1])) if k == 2 else NG1), src_sel=(1 if k == 2 else 0),
                             sst=(cst if k == 2 else 0), nst=nst,
                             tokbuf=tb_nxt.data_ptr())
                self.counters["chain_launches"] += 1
            co["cnt"][i, B:2 * B] = self.buf["lkf"][:B]
            co["host"][i, :int(cs[-1])].copy_(co["out"][i, :int(cs[-1])], non_blocking=True)
            co["hcnt"][i].copy_(co["cnt"][i], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            co["events"].append(ev)
            co["Ls"].append(cs)
            co["cur_len"] = nxt
            co["par"] ^= 1
            co["queued"] += 1

    def poll(self):
        if self.cohort is None:
            if not self.waiting:
                return []
            if len(self.waiting) < self.Bmax and time.perf_counter() - self.first_wait < 0.05:
                return []
            self._start_cohort()
        co = self.cohort
        events = []
        if not co["spec"]:
            self._queue_plain(co)
            while co["done"] < co["queued"]:
                i = co["done"]
                if co["hmap"]:   # a pass has ended when every sequence's slot holds a token id (8-byte stores from the device)
                    tokens = self.hmap[i * co["B"]:(i + 1) * co["B"]].tolist()
                    if not all((t >> 32) == co["tag"] and (t & 0xffffffff) < VOCAB for t in tokens):
                        break
                    tokens = [t & 0xffffffff for t in tokens]
                else:
                    if not co["events"][i].query():
                        break
                    tokens = co["host"][i].tolist()
                for j, (rid, _, n) in enumerate(co["reqs"]):
                    if i < n:
                        events.append((rid, [tokens[j]], i == n - 1))
                co["done"] += 1
            if co["done"] == co["n"]:
                self.cohort = None
            return events
        var = co.get("var", False)
        if var:
            self._queue_spec_var(co)
        else:
            self._queue_spec(co)
        B = co["B"]
        while co["done"] < co["queued"] and co["events"][co["done"]].query():
            i = co["done"]
            starts = co["Ls"][i] if var else None
            P = None if var else co["Ps"][i]
            cnt, tokens = co["hcnt"][i].tolist(), co["host"][i].tolist()
            if var:
                co["lkb"] = [bool(x) for x in cnt[B:2 * B]]
            per = self.counters.setdefault("per_seq", {})
            for j, (rid, _, n) in enumerate(co["reqs"]):
                a = min(cnt[j], n - co["sent"][j])
                if a > 0:
                    co["sent"][j] += a
                    c0 = int(starts[j]) if var else j * P
                    events.append((rid, tokens[c0:c0 + a], co["sent"][j] == n))
                    self.counters["tokens"] += a
                    st = per.setdefault(rid, [0, 0, 0])   # tokens, passes, passes with copied drafts
                    st[0] += a; st[1] += 1; st[2] += cnt[B + j]
            co["lk"] = all(cnt[B:2 * B])
            if var:   # per sequence: copied drafts accepted in full and the drafter still finds a copy
                co["okseq"] = [bool(cnt[B + j]) and cnt[j] >= int(starts[j + 1] - starts[j]) > 1 for j in range(B)]
            if not var:   # a full-length pass of copied drafts accepted in full, and the drafter still finds a copy
                co["loop_ok"] = bool(co["lk"]) and P == co["Phi"] and all(c >= P for c in cnt[:B])
            co["known_pos"] += max(cnt[:B]) if cnt else 0
            self.counters["passes"] += 1
            co["done"] += 1
        if all(s >= r[2] for s, r in zip(co["sent"], co["reqs"])) and co["done"] == co["queued"]:
            self.cohort = None
        elif co["done"] == co["queued"] and co["queued"] >= co["cap"]:
            self.cohort = None  # cannot happen: every pass emits at least one token per unfinished sequence
        return events
