"""Inkling-2-Mini decode engine: one persistent CUDA kernel per call (engine/mk.cu), weights re-laid at load time.

    Engine(model_dir, max_batch, max_context)
    reset()                      forget every sequence
    step(tokens[B] int64 cuda)   append one token per sequence, return next-token logits [B, 200058] float32
    generate(tokens, n_steps)    greedy decode on the device
    close(), stats()
"""

import ctypes
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack import tile_bf16, tile_mxfp8, tile_nvfp4  # noqa: E402

H, NL, MAXB, QKVW, VOCAB, VOCAB_ROWS = 2304, 24, 8, 542 * 16, 200058, 201024
NAPP, NSLOT, MAXCH, CH = 48, 48, 64, 64
PREFIX = "model.llm."
LAYER_FIELDS = ["wqkv", "wo", "wr", "rbias", "w13c", "w13s", "w2c", "w2s", "f13", "f2", "mx", "glob", "loop",
                "nq", "nk", "nv", "nm", "ksc", "vsc", "asc", "msc", "qscale", "rscale", "relp", "aq", "mq",
                "cD", "cF", "cG", "nrt", "relpT", "pad1"]
GLOBAL_FIELDS = ["embed", "embed_norm", "oq", "fnorm", "unembed", "lq", "vecs", "qkvo", "attn_out", "attn_s", "act",
                 "rlog", "zb", "xg", "msg", "msgir", "apart", "acnt", "kc", "vc", "rk", "rv", "ro", "rm", "bar",
                 "amv", "ami", "maxctx", "prof", "prof_cap", "skel", "zm", "xn", "xnd", "xm", "rrx", "rdbg", "yb"]


def _lib_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("MK_LIB8", os.path.join(here, "..", "build", "libmk8.so"))


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        assert 1 <= max_batch <= MAXB, "max_batch must be 1..8"
        self.max_batch = max_batch
        self.max_context = int(max_context)
        if self.max_context > MAXCH * CH:
            raise ValueError(f"max_context {max_context} > {MAXCH * CH}")
        self.dev = torch.device("cuda")
        self.lib = ctypes.CDLL(_lib_path())
        self.lib.mk_create.restype = ctypes.c_void_p
        self.lib.mk_run.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p] * 3
        self.lib.mk_host_tok.restype = ctypes.c_void_p
        self.lib.mk_host_prog.restype = ctypes.c_void_p
        self.lib.mk_host_tok.argtypes = [ctypes.c_void_p]
        self.lib.mk_host_prog.argtypes = [ctypes.c_void_p]
        self.lib.mk_poll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.mk_prog_reset.argtypes = [ctypes.c_void_p]
        assert self.lib.mk_layer_size() == 8 * len(LAYER_FIELDS)
        assert self.lib.mk_params_size() == 8 * (NL * len(LAYER_FIELDS) + len(GLOBAL_FIELDS))
        self.keep = []
        cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
        self.local = set(cfg["local_layer_ids"])
        self._load(model_dir)
        self._alloc()
        blob = np.array(self.fields, dtype=np.uint64)
        self.blob = blob
        self.h = self.lib.mk_create(blob.ctypes.data_as(ctypes.c_void_p))
        assert self.h, "mk_create failed"
        self.dev_host_tok = self.lib.mk_host_tok(self.h)
        self.dev_host_prog = self.lib.mk_host_prog(self.h)
        self.pending = []
        self.cohort = None
        self.gather_s = float(os.environ.get("MK_GATHER_MS", "3")) / 1e3
        self.stream = torch.cuda.current_stream().cuda_stream
        self.reset()

    # ------------------------------------------------------------------------------------------ loading
    def _ptr(self, t):
        if t is None:
            return 0
        t = t.contiguous()
        self.keep.append(t)
        return t.data_ptr()

    def _load(self, model_dir):
        from safetensors import safe_open
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(v for k, v in index.items() if k.startswith(PREFIX)))
        handles = [safe_open(os.path.join(model_dir, fn), framework="pt", device="cuda") for fn in files]
        where = {}
        for hd in handles:
            for k in hd.keys():
                if k.startswith(PREFIX):
                    where[k[len(PREFIX):]] = hd

        def get(name):
            return where[name].get_tensor(PREFIX + name)

        bf = torch.bfloat16
        layer_vals = []
        for i in range(NL):
            p = f"layers.{i}."
            glob = i not in self.local
            loop = 6 <= i < 18
            a = p + "attn."
            sink = torch.zeros(32, H, dtype=bf, device=self.dev)
            sink[:24] = get(a + "attn_sink_proj.weight")
            rows = [get(a + "wq_du.weight"), get(a + "wr_du.weight"), get(a + "gate_attn.weight"), sink, get(a + "wk_dv.weight")]
            if glob:
                kp = torch.zeros(32, H, dtype=bf, device=self.dev); kp[:32] = get(a + "k_sconv.weight_proj")
                vp = torch.zeros(32, H, dtype=bf, device=self.dev); vp[:32] = get(a + "v_sconv.weight_proj")
                rows += [kp, get(a + "wv_dv.weight"), vp]
            else:
                rows += [get(a + "wv_dv.weight")]
            wqkv = torch.cat(rows, 0)
            nrt = wqkv.shape[0] // 16
            assert nrt == (542 if glob else 538)
            v = {"wqkv": tile_bf16(wqkv), "nrt": nrt}
            del wqkv, rows
            v["wo"] = tile_bf16(get(a + "wo_ud.weight"))
            v["wr"] = tile_bf16(get(p + "mlp.gate.weight"))
            v["rbias"] = get(p + "mlp.gate.bias").float()
            e = p + "mlp.experts."
            w13, w2 = get(e + "w13_weight"), get(e + "w2_weight")
            if w13.dtype == torch.uint8:
                v["mx"] = 0
                s13, s2 = get(e + "w13_weight_scale").view(torch.uint8), get(e + "w2_weight_scale").view(torch.uint8)
                for sc in (s13, s2):  # the kernel's scale decode needs positive normal E4M3 (exponent field 1..15, not NaN)
                    ex = (sc >> 3) & 15
                    assert int(((sc & 0x80) != 0).sum()) == 0 and int((ex == 0).sum()) == 0 and int((sc == 0x7F).sum()) == 0, i
                v["w13c"], v["w13s"] = tile_nvfp4(w13, s13)
                v["w2c"], v["w2s"] = tile_nvfp4(w2, s2)
                v["f13"] = (get(e + "w13_weight_amax").float() / get(e + "w13_weight_outer_scale_denominator").float()).contiguous()
                v["f2"] = (get(e + "w2_weight_amax").float() / get(e + "w2_weight_outer_scale_denominator").float()).contiguous()
            else:
                v["mx"] = 1
                s13, s2 = get(e + "w13_weight_scale"), get(e + "w2_weight_scale")
                assert int(s13.view(torch.uint8).max()) <= 142 and int(s2.view(torch.uint8).max()) <= 142
                v["w13c"], v["w13s"] = tile_mxfp8(w13, s13)
                v["w2c"], v["w2s"] = tile_mxfp8(w2, s2)
                v["f13"] = v["f2"] = None
            del w13, w2
            v["glob"] = int(glob)
            v["loop"] = int(loop)
            if glob:
                v["nq"], v["nk"], v["nv"] = get(p + "q_input_norm.weight"), get(p + "k_input_norm.weight"), get(p + "v_input_norm.weight")
                v["ksc"], v["vsc"] = get(a + "k_sconv.static_w"), get(a + "v_sconv.static_w")
            else:
                n = get(p + "attn_norm.weight")
                v["nq"] = v["nk"] = v["nv"] = n
                v["ksc"], v["vsc"] = get(a + "k_sconv.weight").reshape(-1, 4), get(a + "v_sconv.weight").reshape(-1, 4)
            v["nm"] = get(p + "mlp_norm.weight")
            v["asc"] = get(p + "attn_sconv.weight").reshape(-1, 4)
            v["msc"] = get(p + "mlp_sconv.weight").reshape(-1, 4)
            v["qscale"] = get(a + "q_l2_clip.scale")
            v["rscale"] = get(a + "rel_l2_clip.scale")
            v["relp"] = get(a + "rel_logits_proj.proj")
            v["relpT"] = v["relp"].t().contiguous()
            v["aq"] = get(f"attn_res.attn_queries.{i - 1}").float().reshape(-1) if i > 0 else torch.zeros(3 * H, device=self.dev)
            v["mq"] = get(f"attn_res.mlp_queries.{i}").float()
            if loop:
                c = p + "controllers."
                v["cD"] = torch.stack([get(c + w + ".down_weight") for w in ("attn", "k", "v", "mlp")])
                v["cF"] = torch.stack([get(c + w + ".forget_weight") for w in ("attn", "k", "v", "mlp")])
                v["cG"] = torch.stack([get(c + w + ".gain_weight") for w in ("attn", "k", "v", "mlp")])
            else:
                v["cD"] = v["cF"] = v["cG"] = None
            layer_vals.append(v)
            torch.cuda.empty_cache()
        self.layer_vals = layer_vals
        g = {}
        g["embed"] = get("embed.weight")
        g["embed_norm"] = get("embed_norm.weight")
        g["oq"] = get("attn_res.output_query").float()
        g["fnorm"] = get("norm.weight")
        un = get("unembed.weight")
        g["unembed"] = tile_bf16(un[:12504 * 16])
        del un
        qm = get("middle_cycle_cross_pass_queries").float().unflatten(0, (-1, 2))
        kv = get("middle_cycle_cross_pass_kv_queries").float().unflatten(0, (-1, 2))
        g["lq"] = torch.stack([qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]], dim=1).contiguous()
        self.gvals = g
        if os.environ.get("MK_ARENA", "0") != "0":   # one weight arena: B=1 -50 us lone-process here, but +17 in the evaluator (s0021): off
            self._arena()

    def _arena(self):
        """All weight tensors copied into one contiguous allocation in the kernel's order of use (placement study)."""
        order = ["wqkv", "wo", "wr", "w13c", "w13s", "w2c", "w2s", "cD", "cF", "cG"]
        items = []
        for v in self.layer_vals:
            for k in order:
                if isinstance(v.get(k), torch.Tensor): items.append((v, k))
            for k, t in v.items():
                if isinstance(t, torch.Tensor) and k not in order: items.append((v, k))
        for k, t in self.gvals.items():
            if isinstance(t, torch.Tensor): items.append((self.gvals, k))
        mode = os.environ.get("MK_ARENA_ORDER", "use")
        if mode == "experts_first":
            ex = ("w13c", "w13s", "w2c", "w2s")
            items = [it for it in items if it[1] in ex] + [it for it in items if it[1] not in ex]
        elif mode == "reverse":
            items = items[::-1]
        al = int(os.environ.get("MK_ARENA_ALIGN", "65536"))
        offs, tot = [], 0
        skew = int(os.environ.get("MK_ARENA_SKEW", "16384"))
        for n, (d, k) in enumerate(items):
            t = d[k].contiguous()
            tot = (tot + al - 1) // al * al + (n % 16) * skew
            offs.append(tot); tot += t.numel() * t.element_size()
        arena = torch.empty(tot, dtype=torch.uint8, device=self.dev)
        for (d, k), o in zip(items, offs):
            t = d[k].contiguous()
            nb = t.numel() * t.element_size()
            view = arena[o:o + nb].view(t.dtype).view(t.shape)
            view.copy_(t)
            d[k] = view
        self.arena = arena
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    def _alloc(self):
        dev, bf, f32 = self.dev, torch.bfloat16, torch.float32
        C = self.max_context
        z = lambda *s, dt=bf: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        b = {}
        b["vecs"] = z(MAXB, 9, H)
        b["qkvo"] = z(MAXB, QKVW)
        b["attn_out"] = z(MAXB, 3072)
        b["attn_s"] = z(MAXB, H)
        b["act"] = z(64, MAXB, 1152)
        b["rlog"] = z(4, MAXB, 128, dt=f32)
        b["zb"] = z(MAXB, 4, 64, dt=f32)
        b["xg"] = z(MAXB, 4, H, dt=f32)
        b["msg"] = z(48, MAXB, H, dt=f32)
        b["msgir"] = z(48, MAXB, dt=f32)
        b["zm"] = z(MAXB, 48, 64, dt=f32)
        b["xn"] = z(MAXB, 3, H)
        b["xnd"] = z(MAXB, 1, H)
        b["xm"] = z(MAXB, 4, H, dt=f32)
        b["rrx"] = z(MAXB, 4, dt=f32)
        b["rdbg"] = torch.full((48, MAXB, 8), -1, dtype=torch.int32, device=dev)
        b["yb"] = torch.zeros((64, MAXB, 2304), dtype=torch.bfloat16, device=dev)   # w2 outputs per (slot, token) (B = 8 path)
        b["apart"] = z(MAXB, 8, MAXCH, 3, 136, dt=f32)
        b["acnt"] = z(MAXB * 8 + 8, dt=torch.int32)
        b["kc"] = z(NSLOT, MAXB, 8, C, 128)
        b["vc"] = z(NSLOT, MAXB, 8, C, 128)
        b["rk"] = z(NSLOT, MAXB, 4, 1024)
        b["rv"] = z(NSLOT, MAXB, 4, 1024)
        b["ro"] = z(NSLOT, MAXB, 4, H)
        b["rm"] = z(NSLOT, MAXB, 4, H)
        b["bar"] = z(512, dt=torch.int32)   # [0..63] barrier words, mixer flags from 64 at 128-byte stride
        b["amv"] = z(152, MAXB, dt=f32)
        b["ami"] = z(152, MAXB, dt=torch.int32)
        if os.environ.get("MK_BARENA", "0") != "0":   # placement study: buffers in one allocation
            al = int(os.environ.get("MK_BARENA_ALIGN", "65536")); skew = int(os.environ.get("MK_BARENA_SKEW", "16384"))
            offs, tot = [], 0
            keys = list(b.keys())
            for n, k in enumerate(keys):
                tot = (tot + al - 1) // al * al + (n % 16) * skew
                offs.append(tot); tot += b[k].numel() * b[k].element_size()
            ar = torch.zeros(tot, dtype=torch.uint8, device=dev)
            for k, o in zip(keys, offs):
                t = b[k]; nb = t.numel() * t.element_size()
                v = ar[o:o + nb].view(t.dtype).view(t.shape); v.copy_(t); b[k] = v
            self.barena = ar
            torch.cuda.empty_cache()
        self.bufs = b
        self.logits = torch.zeros(MAXB, VOCAB, dtype=f32, device=dev)
        fields = []
        for v in self.layer_vals:
            for name in LAYER_FIELDS:
                x = v.get(name, 0)
                fields.append(self._ptr(x) if isinstance(x, torch.Tensor) else int(x or 0))
        for name in GLOBAL_FIELDS:
            if name == "maxctx":
                fields.append(C)
            elif name == "prof":
                cap = int(os.environ.get("MK_PROF_CAP", "0"))
                self.prof = torch.zeros(152 * max(cap, 1) * 2, dtype=torch.int64, device=dev) if cap else None
                fields.append(self._ptr(self.prof) if cap else 0)
            elif name == "prof_cap":
                fields.append(int(os.environ.get("MK_PROF_CAP", "0")))
            elif name == "skel":
                self.skel = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
                fields.append(self._ptr(self.skel))
            elif name.startswith("pad"):
                fields.append(0)
            elif name in self.gvals:
                fields.append(self._ptr(self.gvals[name]))
            else:
                fields.append(self._ptr(b[name]) if b[name] is not None else 0)
        self.fields = fields

    # ------------------------------------------------------------------------------------------ contract
    def reset(self) -> None:
        if getattr(self, "cohort", None) is not None:
            torch.cuda.synchronize()
            self.cohort = None
        for k in ("rk", "rv", "ro", "rm", "acnt"):
            self.bufs[k].zero_()
        self.pos = 0
        self.B = None

    def _check_tokens(self, t):
        if t.numel() and (int(t.min()) < 0 or int(t.max()) >= VOCAB_ROWS):
            raise ValueError("token id out of range")

    def _run(self, tokens, nsteps, mode, out=None, stop_app=NAPP, nprompt=1):
        """tokens: [B] (nprompt == 1) or [B, nprompt] int64; the kernel runs nsteps steps from self.pos."""
        B = tokens.shape[0]
        if self.B is None:
            self.B = B
        if B != self.B or not 1 <= B <= self.max_batch:
            raise ValueError(f"batch {B} (expected {self.B}, max {self.max_batch})")
        if self.pos + nsteps > self.max_context:
            raise ValueError(f"context full: {self.pos} + {nsteps} > {self.max_context}")
        if not tokens.is_cuda:
            self._check_tokens(tokens)   # (device tensors are not read back: the kernel maps an id outside the table to 0)
        tok = tokens.to(device=self.dev, dtype=torch.int64).contiguous()
        rc = self.lib.mk_run(self.h, tok.data_ptr(), self.logits.data_ptr(), 0 if out is None else out.data_ptr(),
                             B, self.pos, nsteps, mode, stop_app, nprompt, self.dev_host_tok, self.dev_host_prog,
                             torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"mk_run failed ({rc})")
        self.pos += nsteps

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        self._finish_cohort()
        self._run(tokens, 1, 0)
        return self.logits[: tokens.shape[0]].clone()

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Greedy decode on the device: tokens [B] is the next input; returns int64 [B, n_steps]."""
        self._finish_cohort()
        out = torch.empty(tokens.shape[0], n_steps, dtype=torch.int64, device=self.dev)
        self._run(tokens, n_steps, 1, out=out)
        return out

    # ------------------------------------------------------------------------------------------ online contract
    def submit(self, rid, prompt, n) -> None:
        prompt = [int(t) for t in prompt]
        n = int(n)
        if not prompt or n < 1:
            raise ValueError("empty prompt or n < 1")
        if min(prompt) < 0 or max(prompt) >= VOCAB_ROWS:
            raise ValueError("token id out of range")
        if len(prompt) + n - 1 > self.max_context:
            raise ValueError(f"request needs {len(prompt) + n - 1} positions > max_context {self.max_context}")
        self.pending.append((rid, prompt, n, time.perf_counter()))

    def _start_cohort(self):
        P = len(self.pending[0][1])
        group = [r for r in self.pending if len(r[1]) == P][: self.max_batch]
        self.pending = [r for r in self.pending if all(r is not g for g in group)]
        B = len(group)
        nmax = max(r[2] for r in group)
        self.reset()
        self.lib.mk_prog_reset(self.h)
        prompts = torch.tensor([r[1] for r in group], dtype=torch.int64, device=self.dev)
        self.B = B
        self._run(prompts, P + nmax - 1, 2, nprompt=P)
        self.cohort = {"rids": [r[0] for r in group], "n": [r[2] for r in group], "sent": [0] * B, "B": B, "ngen": nmax,
                       "buf": np.zeros(B * nmax, dtype=np.int32)}

    def _finish_cohort(self):
        if self.cohort is not None:
            torch.cuda.synchronize()
            self.cohort = None
            self.B = None

    def poll(self):
        events = []
        c = self.cohort
        if c is not None:
            done = self.lib.mk_poll(self.h, c["buf"].ctypes.data, c["B"], c["ngen"])
            for i, rid in enumerate(c["rids"]):
                upto = min(done, c["n"][i])
                if upto > c["sent"][i]:
                    row = c["buf"][i * c["ngen"]: i * c["ngen"] + upto]
                    events.append((rid, row[c["sent"][i]:upto].tolist(), upto == c["n"][i]))
                    c["sent"][i] = upto
            if done >= c["ngen"]:
                torch.cuda.synchronize()
                self.cohort = None
                self.B = None
        if self.cohort is None and self.pending:
            now = time.perf_counter()
            if len(self.pending) >= self.max_batch or now - self.pending[0][3] >= self.gather_s:
                self._start_cohort()
        return events

    def close(self) -> None:
        torch.cuda.synchronize()
        self.keep = []
        self.bufs = {}

    def stats(self) -> dict:
        return {"implementation": "mk persistent kernel", "track": "A"}
