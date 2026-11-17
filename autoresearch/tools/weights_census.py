#!/usr/bin/env python
"""What is in a checkpoint's weights, bit by bit: the raw material for model-specific storage formats.

    python eval/weights_census.py --model-dir D --label NAME

For every BF16 matrix (2-D tensor) of the checkpoint, on the GPU, exactly (integer counts, no sampling):
  zeros, denormals; entropy of the 16-bit value, of the exponent field, of the mantissa, of the sign
  (order 0, one table per tensor: what a memoryless entropy code with that table would need);
  mean over rows of the exponent entropy within a row (what a per-row code could reach);
  how many low mantissa bits are zero (a coarser grid than BF16?);
  fixed-width codes "sign + k-bit exponent offset in a per-row window + 7 mantissa bits" for k = 4, 3, 2
  (12, 11, 10 bits): the share of weights outside the best window of each row (they need an exact escape);
  rows and columns that are entirely zero; tensors that are bit-identical to another tensor.
Prints one MK_CENSUS line per group of tensors (grouped by the last two name components, for example
`mlp.down_proj`) and one for the whole model. Non-BF16 and non-matrix tensors are only counted.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
dev = "cuda"


def entropy(counts: torch.Tensor) -> float:
    c = counts[counts > 0].double()
    p = c / c.sum()
    return float(-(p * p.log2()).sum())


def group_of(name: str) -> str:
    parts = [p for p in name.split(".") if not p.isdigit()]
    parts = [p for p in parts if p not in ("weight", "model", "layers", "language_model")]
    return ".".join(parts[-2:]) if parts else name


acc = defaultdict(lambda: defaultdict(float))
other = defaultdict(int)
hashes = defaultdict(list)
for path in sorted(Path(args.model_dir).glob("*.safetensors")):
    with safe_open(str(path), framework="pt", device=dev) as f:
        for name in f.keys():
            w = f.get_tensor(name)
            if w.dtype != torch.bfloat16 or w.dim() != 2 or min(w.shape) < 16:
                other[f"{str(w.dtype).replace('torch.', '')} {w.dim()}-D"] += w.numel()
                continue
            g = group_of(name)
            a = acc[g]
            bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
            n = bits.numel()
            e, m = (bits >> 7) & 0xFF, bits & 0x7F
            a["tensors"] += 1
            a["weights"] += n
            a["zeros"] += int(((bits & 0x7FFF) == 0).sum())
            a["denormals"] += int(((e == 0) & (m != 0)).sum())
            a["h_value_x_n"] += entropy(torch.bincount(bits.flatten(), minlength=65536)) * n
            ecount = torch.bincount(e.flatten(), minlength=256)
            a["h_exponent_x_n"] += entropy(ecount) * n
            a["h_mantissa_x_n"] += entropy(torch.bincount(m.flatten(), minlength=128)) * n
            a["h_sign_x_n"] += entropy(torch.bincount((bits >> 15).flatten(), minlength=2)) * n
            for k in range(1, 5):
                a[f"low{k}_zero"] += int(((m & ((1 << k) - 1)) == 0).sum())
            # per row: histogram of the exponent field (zeros and denormals have field 0 and are escapes anyway)
            rows = w.shape[0]
            hist = torch.zeros((rows, 256), dtype=torch.int32, device=dev)
            for r0 in range(0, rows, 8192):
                er = e[r0:r0 + 8192].long()
                hist[r0:r0 + er.shape[0]].scatter_add_(1, er, torch.ones_like(er, dtype=torch.int32))
            hist[:, 0] = 0
            hd = hist.double()
            tot = hd.sum(1, keepdim=True).clamp(min=1)
            pr = hd / tot
            a["h_exponent_in_row_x_n"] += float((-(pr * torch.where(pr > 0, pr.log2(), torch.zeros_like(pr))).sum(1) * tot.squeeze(1)).sum())
            csum = torch.cat([torch.zeros((rows, 1), dtype=torch.float64, device=dev), hd.cumsum(1)], dim=1)
            for k in (4, 3, 2):
                width = 1 << k
                best = (csum[:, width:] - csum[:, :-width]).max(1).values      # most weights a window of this width covers
                a[f"escapes_k{k}"] += n - float(best.sum())
            a["zero_rows"] += int(((bits & 0x7FFF) == 0).all(1).sum())
            a["zero_cols"] += int(((bits & 0x7FFF) == 0).all(0).sum())
            a["rows"] += rows
            a["cols"] += w.shape[1]
            hashes[hashlib.sha256(w.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()].append(name)
            del w, bits, e, m, hist, hd, pr, csum


def line(g: str, a: dict) -> dict:
    n = a["weights"]
    out = {"model": args.label, "group": g, "tensors": int(a["tensors"]), "weights": int(n), "zero_share": a["zeros"] / n, "denormal_share": a["denormals"] / n,
           "entropy_value_bits": a["h_value_x_n"] / n, "entropy_exponent_bits": a["h_exponent_x_n"] / n, "entropy_mantissa_bits": a["h_mantissa_x_n"] / n,
           "entropy_sign_bits": a["h_sign_x_n"] / n, "entropy_exponent_within_row_bits": a["h_exponent_in_row_x_n"] / n,
           "low_mantissa_bits_zero_share": {str(k): a[f"low{k}_zero"] / n for k in range(1, 5)},
           "zero_rows": int(a["zero_rows"]), "zero_cols": int(a["zero_cols"]), "rows": int(a["rows"])}
    for k in (4, 3, 2):
        esc = a[f"escapes_k{k}"] / n
        out[f"code_{8 + k}bit"] = {"escape_share": esc, "bits_per_weight_with_32bit_escapes": 8 + k + 32 * esc}
    return out


total = defaultdict(float)
for g, a in sorted(acc.items(), key=lambda kv: -kv[1]["weights"]):
    for key, value in a.items():
        total[key] += value
    print("MK_CENSUS " + json.dumps(line(g, a)), flush=True)
dups = [names for names in hashes.values() if len(names) > 1]
whole = line("ALL", total)
whole.update(other_tensors=dict(other), identical_tensor_sets=[[n[-60:] for n in names[:4]] for names in dups[:6]], identical_sets=len(dups))
print("MK_CENSUS " + json.dumps(whole), flush=True)
