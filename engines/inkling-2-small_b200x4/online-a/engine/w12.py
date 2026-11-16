"""Exact 12-bit code for BF16 weight matrices in m16n8k16 fragment order (track B storage).

Per row a base exponent E (window [E, E+15]); per value: sign + 7 mantissa bits (a byte) and a 4-bit exponent
offset. Values outside the window (and zeros of rows whose window does not start at 0) are escapes: stored exactly
(bf16) in a per-unit list and patched into the decoded registers. Layout per (strip of 16 rows, unit of 64 k), per
lane (32): 32 B of bytes + 16 B of nibbles (3 x 512 B per unit, lane-major), see engine/ink.cu run_w12.
  bytes: word q of a lane = registers 2q (bytes 0,1 = lo,hi value) and 2q+1 (bytes 2,3)
  nibbles: word n = registers 4n..4n+3: bits 4a.. = offset of lo value of reg 4n+a, bits 16+4a.. = hi value
  register r = j*4 + a (j = k-step in the unit, a = A-fragment register: rows g / g+8, k pair)
"""
import torch


def frag_values(w: torch.Tensor) -> torch.Tensor:
    """[N, K] bf16 -> int32 bits [S, U, 32 lanes, 16 regs, 2] in fragment order (same order as frag_bf16)."""
    n, k = w.shape
    x = w.view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6)  # [S, KS, g, t, kh, hi, e]
    x = x.reshape(n // 16, k // 64, 4, 32, 4, 2)  # [S, U, j, lane, a, e]
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(n // 16, k // 64, 32, 16, 2)  # [S, U, lane, r=j*4+a, e]
    return x.contiguous().view(torch.int16).int() & 0xFFFF


def row_base(w: torch.Tensor):
    """Best 16-exponent window per row (exponent field of nonzero values); all-zero rows get 0."""
    b = w.view(torch.int16).int() & 0xFFFF
    e = (b >> 7) & 0xFF
    nz = (b & 0x7FFF) != 0
    h = torch.zeros(w.shape[0], 256, dtype=torch.int32, device=w.device)
    h.scatter_add_(1, e.long(), nz.int())
    cs = torch.cat([torch.zeros_like(h[:, :1]), h.cumsum(1, dtype=torch.int32)], 1)
    inside = cs[:, 16:] - cs[:, :-16]  # windows lo = 0..240
    lo = inside.argmax(1)
    lo[nz.sum(1) == 0] = 0
    return lo.int()


def pack(w: torch.Tensor):
    """Returns (data uint8 [S*U*1536], base uint8 [N], esc_hdr int32 [S*U] (count | offset<<8), esc int32 [n])."""
    n, k = w.shape
    S, U = n // 16, k // 64
    E = row_base(w)  # [N]
    v = frag_values(w)  # [S, U, lane, r, e]
    lane = torch.arange(32, device=w.device)
    g = lane // 4
    r = torch.arange(16, device=w.device)
    hi = r % 2  # a = r % 4: a0 (row g), a1 (row g+8), a2 (g), a3 (g+8) -> hi = a & 1
    rows = (torch.arange(S, device=w.device)[:, None, None] * 16 + g[None, :, None] + 8 * hi[None, None, :])  # [S, lane, r]
    Er = E[rows].unsqueeze(1).unsqueeze(-1)  # [S, 1, lane, r, 1]
    s = (v >> 15) & 1
    ex = (v >> 7) & 0xFF
    m = v & 0x7F
    d = ex - Er
    ok = (d >= 0) & (d <= 15)
    zero = (v & 0x7FFF) == 0
    ok = ok & ~(zero & (Er != 0))  # a zero decodes as exponent E: only exact when E == 0
    dd = torch.where(ok, d, torch.zeros_like(d))
    byte = ((s << 7) | m).to(torch.uint8)  # [S, U, lane, r, e]
    # bytes: per lane 32 B: reg r, value e -> byte index r*2 + e
    by = byte.reshape(S, U, 32, 32)
    # nibbles: word n = r//4, bits 4a + 16e
    a = (r % 4)
    nib = (dd << (4 * a.view(1, 1, 1, 16, 1) + 16 * torch.arange(2, device=w.device).view(1, 1, 1, 1, 2)))
    words = nib.view(S, U, 32, 4, 4, 2).sum(dim=(4, 5)).to(torch.int64) & 0xFFFFFFFF  # [S, U, lane, 4]
    words = words.to(torch.int32).contiguous().view(torch.uint8).view(S, U, 32, 16)
    data = torch.cat([by[:, :, :, :16], by[:, :, :, 16:], words], dim=2)  # placeholder, fixed below
    # lane-major blocks: [S, U, 3 blocks, 32 lanes, 16 B]
    data = torch.stack([by[:, :, :, :16], by[:, :, :, 16:], words], dim=2).contiguous().view(-1)
    # escapes
    bad = ~ok
    idx = bad.nonzero()  # [n, 5]: S, U, lane, r, e
    unit = idx[:, 0] * U + idx[:, 1]
    entry = (idx[:, 2] << 27) | (idx[:, 3] << 23) | (idx[:, 4] << 22) | v[bad]
    order = torch.argsort(unit * 4096 + idx[:, 2] * 64 + idx[:, 3] * 2 + idx[:, 4])
    unit, entry = unit[order], entry[order]
    cnt = torch.bincount(unit, minlength=S * U)
    off = torch.cumsum(cnt, 0) - cnt
    assert int(cnt.max()) < 256 if cnt.numel() else True
    hdr = (cnt | (off << 8)).int()
    return data, E.to(torch.uint8), hdr, entry.int(), int(bad.sum())


def unpack(data, E, hdr, esc, n, k):
    """Inverse of pack (for verification): returns [N, K] bf16."""
    S, U = n // 16, k // 64
    d = data.view(S, U, 3, 32, 16)
    by = torch.cat([d[:, :, 0], d[:, :, 1]], dim=-1).int()  # [S, U, lane, 32] = r*2+e
    words = d[:, :, 2].contiguous().view(torch.int32).view(S, U, 32, 4)
    r = torch.arange(16, device=data.device)
    a = r % 4
    nibs = torch.stack([(words[..., r // 4] >> (4 * a + 16 * e)) & 0xF for e in range(2)], dim=-1)  # [S, U, lane, 16, 2]
    lane = torch.arange(32, device=data.device)
    g = lane // 4
    hi = r % 2
    rows = (torch.arange(S, device=data.device)[:, None, None] * 16 + g[None, :, None] + 8 * hi[None, None, :])
    Er = E.int()[rows].unsqueeze(1).unsqueeze(-1)
    b = by.view(S, U, 32, 16, 2)
    v = ((b >> 7) << 15) | ((nibs + Er) << 7) | (b & 0x7F)
    # escapes
    cnt = hdr & 0xFF
    unit = torch.repeat_interleave(torch.arange(S * U, device=data.device), cnt.long())
    ln, rr, ee, val = (esc >> 27) & 31, (esc >> 23) & 15, (esc >> 22) & 1, esc & 0xFFFF
    v = v.view(S * U, 32, 16, 2)
    v[unit.long(), ln.long(), rr.long(), ee.long()] = val.to(v.dtype)
    v = v.view(S, U, 32, 16, 2)
    # back to [N, K]
    x = v.view(S, U, 32, 4, 4, 2).permute(0, 1, 3, 2, 4, 5)  # [S, U, j, lane, a, e]
    x = x.reshape(S, U * 4, 8, 4, 2, 2, 2)  # [S, KS, g, t, kh, hi, e]
    x = x.permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)  # [S, hi, g, KS, kh, t, e]
    return (x.short()).view(torch.bfloat16)
