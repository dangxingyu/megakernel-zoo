"""Weight re-layout for the megakernel: permutations of the stored bytes into MMA fragment order (no value changes).

BF16 [N, K]           -> [N/16 rowtiles][K/16 ksteps][32 lanes][4 regs][2]           (512 B per kstep tile)
NVFP4 codes [E,N,K/2] -> [E][N/16][K/64 chunks][32 lanes][4 ksteps][4 regs] bytes    (512 B per item)
      scales [E,N,K/16] (E4M3) -> [E][N/16][K/64][8 g][4 ksteps][2 halves]           (64 B per item)
MXFP8 codes [E,N,K]   -> [E][N/16][K/32][32 lanes][2 ksteps][4 regs][2]               (512 B per item)
      scales [E,N,K/32] (UE8M0) -> [E][N/16][K/32][8 g][2 halves]                     (16 B per item)
Lane = 4 g + t; reg 0: (row g, cols 2t, 2t+1), 1: (row g+8, same), 2: (row g, cols 2t+8, +9), 3: (row g+8, ...).
"""

import torch


def tile_bf16(w: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    assert n % 16 == 0 and k % 16 == 0, w.shape
    return w.contiguous().view(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()


def untile_bf16(t: torch.Tensor, n: int, k: int) -> torch.Tensor:
    return t.view(n // 16, k // 16, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(n, k)


def tile_nvfp4(codes: torch.Tensor, scales: torch.Tensor):
    e, n, kb = codes.shape
    k = kb * 2
    r, kc = n // 16, k // 64
    c = codes.contiguous().view(e, r, 2, 8, kc, 4, 2, 4).permute(0, 1, 4, 3, 7, 5, 6, 2).contiguous()
    s = scales.contiguous().view(torch.uint8).view(e, r, 2, 8, kc, 4).permute(0, 1, 4, 3, 5, 2).contiguous()
    return c, s


def untile_nvfp4(c, s, e, n, k):
    r, kc = n // 16, k // 64
    codes = c.view(e, r, kc, 8, 4, 4, 2, 2).permute(0, 1, 7, 3, 2, 5, 6, 4).reshape(e, n, k // 2)
    scales = s.view(e, r, kc, 8, 4, 2).permute(0, 1, 5, 3, 2, 4).reshape(e, n, k // 16)
    return codes, scales


def tile_mxfp8(codes: torch.Tensor, scales: torch.Tensor):
    e, n, k = codes.shape
    r, kc = n // 16, k // 32
    c = codes.contiguous().view(torch.uint8).view(e, r, 2, 8, kc, 2, 2, 4, 2).permute(0, 1, 4, 3, 7, 5, 6, 2, 8).contiguous()
    s = scales.contiguous().view(torch.uint8).view(e, r, 2, 8, kc).permute(0, 1, 4, 3, 2).contiguous()
    return c, s


def untile_mxfp8(c, s, e, n, k):
    r, kc = n // 16, k // 32
    codes = c.view(e, r, kc, 8, 4, 2, 2, 2, 2).permute(0, 1, 7, 3, 2, 5, 6, 4, 8).reshape(e, n, k)
    scales = s.view(e, r, kc, 8, 2).permute(0, 1, 4, 3, 2).reshape(e, n, k // 32)
    return codes, scales


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_nvfp4(codes, scales, dtype=torch.float32):
    """[E, N, K/2] u8 codes, [E, N, K/16] e4m3 scales -> [E, N, K] of code x block scale (no per-expert factor)."""
    tab = torch.tensor(E2M1, dtype=dtype, device=codes.device)
    v = tab[torch.stack([codes & 0xF, codes >> 4], -1).flatten(-2).long()]
    sc = scales.view(torch.float8_e4m3fn).to(dtype)
    return (v.unflatten(-1, (-1, 16)) * sc.unsqueeze(-1)).flatten(-2)


def dequant_mxfp8(codes, scales, dtype=torch.float32):
    v = codes.view(torch.float8_e4m3fn).to(dtype)
    ex = torch.exp2(scales.view(torch.uint8).to(dtype) - 127.0)
    return (v.unflatten(-1, (-1, 32)) * ex.unsqueeze(-1)).flatten(-2)
