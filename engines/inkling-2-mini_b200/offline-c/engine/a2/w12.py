"""Exact 12-bit storage of BF16 matrices (track B), in the engine's MMA fragment order.

A BF16 weight is sign(1) | exponent(8) | mantissa(7). Each row r stores a window base e0[r]; a weight whose exponent
lies in [e0, e0 + 15] is stored as the byte (sign << 7 | mantissa) plus the 4-bit offset exponent - e0. Others are
escapes: stored with offset 0 and listed exactly (the kernel patches the decoded register before the MMA), so the
decoded operand is bit-for-bit the BF16 weight.

Layout per (strip of 16 rows, unit of 64 k), per lane (32 lanes, the m16n8k16 A-fragment owner of 8 values per tile,
4 tiles per unit), 48 bytes in three 512-byte chunks:
  chunk 0: tiles 0, 1 value bytes (8 per tile, value order v = 2 * reg + half), chunk 1: tiles 2, 3,
  chunk 2: one 32-bit nibble word per tile: bits [4 r, 4 r + 4) = offset of (reg r, half 0), bits [16 + 4 r, ...) =
  offset of (reg r, half 1).
Escapes per (strip, unit): hdr = count | offset << 8 into a list of int32 lane << 27 | reg16 << 23 | half << 22 | bf16
(reg16 = 4 * tile + reg).
"""

import torch


def _frag_values(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> [S][U][4 tiles][32 lanes][8 values] int32 bf16 bits, in fragment order (as engine.api.frag_bf16)."""
    n, k = w.shape
    x = w.contiguous().view(torch.int16).view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)
    # [S][K/16][g][t][kh][r][e] -> lane = g*4 + t, values (kh, r, e) = reg (kh*2 + r) * 2 + e  (a0=(0,0) a1=(0,1) a2=(1,0) a3=(1,1))
    x = x.reshape(n // 16, k // 64, 4, 32, 8)
    return x.to(torch.int32) & 0xFFFF


def pack(w: torch.Tensor):
    """w [N, K] bf16 (cuda) -> (data uint8 [S*U*1536], base int32 [N], hdr int32 [S*U], esc int32 [E], n_escapes)."""
    assert w.dtype == torch.bfloat16 and w.dim() == 2
    n, k = w.shape
    assert n % 16 == 0 and k % 64 == 0
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    # per row: the 16-wide exponent window holding the most weights
    hist = torch.zeros(n, 256 + 16, dtype=torch.int32, device=w.device)
    hist.scatter_add_(1, e.long(), torch.ones_like(e))
    cs = torch.cat([torch.zeros(n, 1, dtype=torch.int64, device=w.device), hist.long().cumsum(1)], 1)
    win = cs[:, 16:16 + 241] - cs[:, 0:241]                     # windows starting at 0 .. 240
    e0 = win.argmax(1).to(torch.int32)                            # [N]
    v = _frag_values(w)                                           # [S][U][4][32][8]
    S, U = n // 16, k // 64
    # row of each fragment value: lane g = lane >> 2, value v = 2*reg + half, reg r: rows g (r = 0, 2) or g + 8 (r = 1, 3)
    lane = torch.arange(32, device=w.device).view(1, 1, 1, 32, 1)
    vi = torch.arange(8, device=w.device).view(1, 1, 1, 1, 8)
    reg = vi // 2
    strip = torch.arange(S, device=w.device).view(S, 1, 1, 1, 1)
    row = strip * 16 + (lane >> 2) + 8 * (reg % 2)
    base = e0[row.expand(S, U, 4, 32, 8)]
    ev = (v >> 7) & 0xFF
    off = ev - base
    esc = (off < 0) | (off > 15)
    off = torch.where(esc, torch.zeros_like(off), off)
    byte = ((v >> 8) & 0x80) | (v & 0x7F)                         # sign << 7 | mantissa
    data = torch.empty(S, U, 3, 32, 16, dtype=torch.uint8, device=w.device)
    by = byte.to(torch.uint8)                                     # [S][U][4][32][8]
    data[:, :, 0] = torch.cat([by[:, :, 0], by[:, :, 1]], dim=-1)
    data[:, :, 1] = torch.cat([by[:, :, 2], by[:, :, 3]], dim=-1)
    # nibble word per tile: (reg r, half h) at bit 16 h + 4 r
    r4 = (vi // 2) * 4 + (vi % 2) * 16
    nib = (off << r4).sum(-1).to(torch.int64) & 0xFFFFFFFF       # [S][U][4][32]
    nw = nib.to(torch.int32).permute(0, 1, 3, 2).contiguous()    # [S][U][32][4]
    data[:, :, 2] = nw.view(torch.uint8).view(S, U, 32, 16)
    # escapes
    idx = esc.nonzero()                                           # [E, 5]: s, u, tile, lane, value
    ne = idx.shape[0]
    su = idx[:, 0] * U + idx[:, 1]
    order = torch.argsort(su, stable=True)
    idx, su = idx[order], su[order]
    vals = v[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3], idx[:, 4]]
    reg16 = idx[:, 2] * 4 + idx[:, 4] // 2
    entry = (idx[:, 3] << 27) | (reg16 << 23) | ((idx[:, 4] % 2) << 22) | vals
    counts = torch.bincount(su, minlength=S * U)
    assert int(counts.max()) < 256 if ne else True, "too many escapes in one unit"
    offs = torch.cumsum(counts, 0) - counts
    assert ne < (1 << 23)
    hdr = (counts | (offs << 8)).to(torch.int32)
    return data.flatten(), e0, hdr, entry.to(torch.int32), ne


def unpack(data, base, hdr, esc, n, k) -> torch.Tensor:
    """The exact inverse (the kernel's decode in torch): -> [N, K] bf16."""
    S, U = n // 16, k // 64
    d = data.view(S, U, 3, 32, 16)
    by = torch.stack([d[:, :, 0, :, :8], d[:, :, 0, :, 8:], d[:, :, 1, :, :8], d[:, :, 1, :, 8:]], dim=2).to(torch.int32)  # [S][U][4][32][8]
    nw = d[:, :, 2].contiguous().view(torch.int32).view(S, U, 32, 4).permute(0, 1, 3, 2)              # [S][U][4][32]
    vi = torch.arange(8, device=data.device).view(1, 1, 1, 1, 8)
    r4 = (vi // 2) * 4 + (vi % 2) * 16
    off = (nw.unsqueeze(-1) >> r4) & 0xF
    lane = torch.arange(32, device=data.device).view(1, 1, 1, 32, 1)
    strip = torch.arange(S, device=data.device).view(S, 1, 1, 1, 1)
    row = strip * 16 + (lane >> 2) + 8 * ((vi // 2) % 2)
    e = base[row.expand(S, U, 4, 32, 8)] + off
    v = ((by & 0x80) << 8) | (e << 7) | (by & 0x7F)
    if esc.numel():
        cnt = hdr & 0xFF
        su = torch.repeat_interleave(torch.arange(S * U, device=data.device), cnt.long())
        ent = esc[: su.numel()].long()
        ln, rg, hf, val = (ent >> 27) & 31, (ent >> 23) & 15, (ent >> 22) & 1, ent & 0xFFFF
        v[su // U, su % U, rg // 4, ln, (rg % 4) * 2 + hf] = val.to(v.dtype)
    # back from fragment order: inverse of _frag_values
    x = v.to(torch.int16).view(S, U * 4, 8, 4, 2, 2, 2)          # [S][K/16][g][t][kh][r][e]
    x = x.permute(0, 5, 2, 1, 4, 3, 6).contiguous()               # [S][r][g][K/16][kh][t][e]
    return x.view(n, k).view(torch.bfloat16)
