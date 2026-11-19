"""Track B: exact 12-bit storage of the checkpoint's BF16 weights.

Per output row r a 16-exponent window [lo_r, lo_r + 15] (the one holding the most of the row's weights, exact counts).
Every row is stored scaled by 2^(128 - lo_r) (an exponent shift, exact), so an in-window weight has the exponent
128 + e' = 0x80 | e' (e' = exponent - lo_r in 0..15) and decodes as sign | (0x80 | e') << 7 | mantissa with no add;
the GEMM epilogue multiplies the row's sum by 2^(lo_r - 128) (a power of two: exact, the sums are scaled exactly).
Weights outside the window (1.03e-4 of the checkpoint) are written with e' = 0 and patched into the MMA operand
registers from per-warp lists with their exact scaled bits, so every product is the checkpoint's.

Layout per (16-row tile, quad of 4 k-steps), 1536 bytes: 3 loads of [32 lanes][16 B] = 12 words per lane; k-step j
of the quad uses words 3j, 3j+1, 3j+2 (P, Q, R). With code12 = sign << 11 | e' << 7 | mantissa of a value:
  P holds register a0 (two values) in place: bits 15 / 31 sign, [7:11) / [23:27) e', [0:7) / [16:23) mantissa,
  Q holds a1 and R holds a2 the same way; bits [11:15) / [27:31) of P, Q, R (free: the decoded exponent is 0x80 | e')
  carry code12 of a3's two values, 4 bits each: P bits 0-3, Q bits 4-7, R bits 8-11.
Decode: a0 = P & 0x87FF87FF | 0x40004000 (one LOP3), a1, a2 likewise; a3 from the three nibble fields.
"""
import torch

L0 = 128
WIN = 16


def window_rows(W: torch.Tensor) -> torch.Tensor:
    """Per row the first of the WIN consecutive exponent values holding the most weights of the row."""
    M = W.shape[0]
    e = (W.view(torch.int16).to(torch.int32) >> 7) & 0xFF
    h = torch.zeros(M, 256, dtype=torch.int32, device=W.device)
    h.scatter_add_(1, e.long(), torch.ones_like(e))
    cs = torch.cat([torch.zeros(M, 1, dtype=torch.int64, device=W.device), h.long().cumsum(1)], 1)
    lo = (cs[:, WIN:] - cs[:, :-WIN]).argmax(1)
    return lo.to(torch.int32)


def pack(F: torch.Tensor, M: int, K: int, lo: torch.Tensor):
    """F: fragment-ordered bf16 bits (int16, [M/16][K/16][32][8]). Returns the packed int32 words
    [NT][KQ][4 k-steps][3 words][32 lanes] stored as [NT][KQ][3 loads][32 lanes][4 words], and the escapes:
    unit index (tile * KQ + quad), lane, j, r, h, scaled bits."""
    NT, KQ = M // 16, K // 64
    b = F.view(NT, KQ, 4, 32, 8).to(torch.int64) & 0xFFFF
    lane = torch.arange(32, device=F.device).view(32, 1)
    eidx = torch.arange(8, device=F.device).view(1, 8)
    rowoff = (lane >> 2) + 8 * ((eidx >> 1) & 1)                          # row in tile of (lane, value e = 2r + h)
    lof = lo.view(NT, 16)[:, rowoff].view(NT, 1, 1, 32, 8).to(torch.int64)
    ex = (b >> 7) & 0xFF
    ep = ex - lof
    inw = (ep >= 0) & (ep < WIN)
    epc = torch.where(inw, ep, torch.zeros_like(ep))
    s, m = (b >> 15) & 1, b & 0x7F
    code = (s << 11) | (epc << 7) | m                                    # 12 bits
    inplace = (s << 15) | (epc << 7) | m                                 # bf16 positions, exponent bits 11-14 free
    # values e = 2r + h: r = 0..3 registers, h = half
    def word(r, chunk):
        lo_, hi_ = inplace[..., 2 * r], inplace[..., 2 * r + 1]
        c3lo, c3hi = (code[..., 6] >> (4 * chunk)) & 0xF, (code[..., 7] >> (4 * chunk)) & 0xF
        return lo_ | (hi_ << 16) | (c3lo << 11) | (c3hi << 27)
    Pw, Qw, Rw = word(0, 0), word(1, 1), word(2, 2)                      # [NT][KQ][4 j][32]
    wl = torch.stack([Pw, Qw, Rw], 3)                                    # [NT][KQ][4 j][3][32]
    wl = wl.permute(0, 1, 4, 2, 3).reshape(NT, KQ, 32, 12)               # per lane: word 3 j + i
    wl = wl.view(NT, KQ, 32, 3, 4).permute(0, 1, 3, 2, 4)                # [NT][KQ][3 loads][32][4]
    words = (wl - ((wl >> 31) << 32)).to(torch.int32).contiguous()
    # escapes
    pos = (~inw).nonzero()                                               # [n, 5]: tile, quad, j, lane, e
    eb = b[~inw]
    elo = lof.expand(NT, KQ, 4, 32, 8)[~inw]
    eex = (eb >> 7) & 0xFF
    se = eex + L0 - elo
    assert bool(((se >= 1) & (se <= 254)).all()), "escape exponent out of range after scaling"
    sval = (eb & 0x807F) | (se << 7)
    unit = pos[:, 0] * KQ + pos[:, 1]
    return words.view(-1), (unit, pos[:, 3], pos[:, 2], pos[:, 4] >> 1, pos[:, 4] & 1, sval)


def unpack(words: torch.Tensor, M: int, K: int, esc) -> torch.Tensor:
    """The kernel's decode (scaled bits), escapes patched: int32 [NT][KQ][4][32][8]."""
    NT, KQ = M // 16, K // 64
    w = words.view(NT, KQ, 3, 32, 4).to(torch.int64) & 0xFFFFFFFF
    w = w.permute(0, 1, 3, 2, 4).reshape(NT, KQ, 32, 4, 3)               # [NT][KQ][32][j][P, Q, R]
    P, Q, R = w[..., 0], w[..., 1], w[..., 2]                            # [NT][KQ][32][4 j]
    def reg(X):
        return (X & 0x87FF87FF) | 0x40004000
    t = ((P >> 11) & 0x000F000F) | ((Q >> 7) & 0x00F000F0) | ((R >> 3) & 0x0F000F00)
    r3 = (t & 0x07FF07FF) | ((t << 4) & 0x80008000) | 0x40004000
    regs = [reg(P), reg(Q), reg(R), r3]
    out = []
    for e in range(8):
        r, h = e >> 1, e & 1
        out.append((regs[r] >> (16 * h)) & 0xFFFF)
    dec = torch.stack(out, -1).permute(0, 1, 3, 2, 4).contiguous()      # [NT][KQ][4 j][32][8]
    unit, lane, j, r, h, sval = esc
    dec[unit // KQ, unit % KQ, j, lane, r * 2 + h] = sval.to(torch.int64)
    return dec.to(torch.int32)


def scaled_bits(F: torch.Tensor, M: int, K: int, lo: torch.Tensor) -> torch.Tensor:
    """bits of w * 2^(L0 - lo_row) in fragment order [NT][KQ][4][32][8] (exact: an exponent shift)."""
    NT, KQ = M // 16, K // 64
    b = F.view(NT, KQ, 4, 32, 8).to(torch.int32) & 0xFFFF
    lane = torch.arange(32, device=F.device).view(32, 1)
    eidx = torch.arange(8, device=F.device).view(1, 8)
    rowoff = (lane >> 2) + 8 * ((eidx >> 1) & 1)
    lof = lo.view(NT, 16)[:, rowoff].view(NT, 1, 1, 32, 8)
    ex = (b >> 7) & 0xFF
    assert bool((ex > 0).all()), "zero or subnormal weight: the exponent shift would not be exact"
    return (b & 0x807F) | ((ex + L0 - lof) << 7)


def escape_lists(esc, M: int, K: int, nblk: int, nwarp: int):
    """Per-warp lists for the kernel's partition (block b: units [U b / nblk, U (b+1) / nblk), warp w of it:
    an equal share): entries (unit, packed) as int32 pairs, each list ended by a sentinel (unit 0x7fffffff).
    packed = scaled bits << 16 | lane << 11 | j << 9 | r << 7 | h << 6. Returns (hdr [nblk * nwarp], entries)."""
    unit, lane, j, r, h, sval = esc
    dev = unit.device
    NT, KQ = M // 16, K // 64
    U = NT * KQ
    bs = (torch.arange(nblk + 1, device=dev, dtype=torch.int64) * U) // nblk
    b0, b1 = bs[:-1].view(nblk, 1), bs[1:].view(nblk, 1)
    w = torch.arange(nwarp, device=dev, dtype=torch.int64).view(1, nwarp)
    ws = (b0 + ((b1 - b0) * w) // nwarp).view(-1)                       # warp starts, global order
    gw = torch.searchsorted(ws, unit.to(torch.int64), right=True) - 1
    order = torch.argsort(gw * (U + 1) + unit.to(torch.int64), stable=True)
    gw, unit, lane, j, r, h, sval = gw[order], unit[order], lane[order], j[order], r[order], h[order], sval[order]
    nw = nblk * nwarp
    cnt = torch.bincount(gw, minlength=nw)
    start = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), (cnt + 1).cumsum(0)[:-1]])
    n = unit.numel()
    ent = torch.zeros(n + nw + 32, 2, dtype=torch.int64, device=dev)   # + 32: the kernel reads windows of 32 entries
    ent[:, 0] = 0x7FFFFFFF
    rank = torch.arange(n, device=dev) - torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), cnt.cumsum(0)[:-1]])[gw]
    idx = start[gw] + rank
    ent[idx, 0] = unit.to(torch.int64)
    ent[idx, 1] = (sval.to(torch.int64) << 16) | (lane.to(torch.int64) << 11) | (j.to(torch.int64) << 9) | (r.to(torch.int64) << 7) | (h.to(torch.int64) << 6)
    ent = (ent - ((ent >> 31) << 32)).to(torch.int32).contiguous()
    return start.to(torch.int32).contiguous(), ent.view(-1)
