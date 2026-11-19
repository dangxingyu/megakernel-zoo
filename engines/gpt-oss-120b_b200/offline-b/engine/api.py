"""gpt-oss-120b decode engine: one persistent CUDA kernel per step / generate call / served cohort (engine/mk.cu).
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
HID, NL, VOCAB, NE, MAXB = 2880, 36, 201088, 128, 8
NQH, NKV, HD = 64, 8, 64


def pack_tiles(w: torch.Tensor) -> torch.Tensor:
    """int16 [M, K] (BF16 bits) -> tiles [M/16][K/16][32 lanes][4 regs][2 halves] in mma A-operand order."""
    M, K = w.shape
    return w.reshape(M // 16, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous().view(-1)


def pack_head_sketch(w: torch.Tensor) -> torch.Tensor:
    """Track B head sketch records from the BF16 head (int16 bits [V, 2880]). Per 16-row tile 94 records of 512 B:
    90 int8 A fragments (m16n8k32: lane (g, t) = rows g, g+8 x k 4t..4t+3 and 16+4t..16+4t+3), 3 BF16 A tiles of the
    error-block norms (45 blocks of 64 + 3 zero columns), 16 rows x 8 floats (s, |s q|, |w|, 0...).
    s = fp32(max|w| / 127), q = rint(w / s) in [-127, 127], e = w - s q (exact in float64); norms rounded up."""
    V, K = w.shape
    out = torch.empty(V // 16, 94, 512, dtype=torch.uint8, device=w.device)
    CH = 4096
    for r0 in range(0, V, CH):
        wb = w[r0:r0 + CH].view(torch.bfloat16).double()
        n = wb.shape[0]
        s = (wb.abs().amax(1) / 127.0).float()
        s = torch.where(s > 0, s, torch.ones_like(s)).double()
        q = torch.round(wb / s[:, None]).clamp_(-127, 127)
        e = wb - s[:, None] * q
        en = e.view(n, 45, 64).norm(dim=2) * (1 + 2.0 ** -30)
        enf = en.float()
        enf = torch.where(enf.double() < en, torch.nextafter(enf, torch.full_like(enf, float("inf"))), enf)
        u = enf.view(torch.int32)
        u = torch.where((u & 0xffff) != 0, (u + 0x10000) & ~0xffff, u)
        enb = (u >> 16).to(torch.int16)                                   # BF16 bits, >= the exact norm
        enp = torch.zeros(n, 48, dtype=torch.int16, device=w.device); enp[:, :45] = enb
        sqn = ((s[:, None] * q).norm(dim=1) * (1 + 2.0 ** -20)).float()
        wn = (wb.norm(dim=1) * (1 + 2.0 ** -20)).float()
        qi = q.to(torch.int8).view(n // 16, 2, 8, 90, 2, 4, 4).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
        out[r0 // 16:(r0 + n) // 16, :90] = qi.view(torch.uint8).view(n // 16, 90, 512)
        out[r0 // 16:(r0 + n) // 16, 90:93] = pack_tiles(enp).view(torch.uint8).view(n // 16, 3, 512)
        cst = torch.zeros(n, 8, dtype=torch.float32, device=w.device)
        cst[:, 0] = s.float(); cst[:, 1] = sqn; cst[:, 2] = wn
        out[r0 // 16:(r0 + n) // 16, 93] = cst.view(torch.uint8).view(n // 16, 512)
    return out.view(-1)


def pack_fp4(blocks: torch.Tensor, scales: torch.Tensor):
    """MXFP4 [E, M, K/32, 16] u8 codes + [E, M, K/32] u8 scales -> codes [E][M/16][K/64][32 lanes][16 B] and
    scales [E][M/16][K/64][8 g][4 B]; a pure byte permutation (each byte = two e2m1 codes of columns 2i, 2i+1)."""
    E, M, G, _ = blocks.shape
    C = G // 2  # 64-column chunks
    c = blocks.reshape(E, M // 16, 2, 8, C, 4, 2, 4).permute(0, 1, 4, 3, 7, 5, 6, 2).reshape(E, M // 16, C, 512)
    s = scales.reshape(E, M // 16, 2, 8, C, 2).permute(0, 1, 4, 3, 2, 5).reshape(E, M // 16, C, 32)
    # one 544-byte record per (rowtile, chunk): the 512 code bytes, then the 32 scale bytes
    rec = torch.cat([c, s], dim=3).contiguous().view(-1)
    return rec, rec


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
                                               "dn_c", "dn_s", "dn_b", "ln1", "ln2", "sinks", "ln1b", "ln2b", "kc", "vc", "wp")] + [("sliding", ctypes.c_int)]


class Params(ctypes.Structure):
    _fields_ = [("L", LayerP * NL)] + [(n, ctypes.c_void_p) for n in ("embed", "head", "lnf", "lnfb", "cosT", "sinT", "x", "xm", "q",
                                                                      "attn", "act", "rlog", "ypart", "ecnt", "apart", "acnt", "amv", "ami", "bar")] + [("maxctx", ctypes.c_int), ("prof", ctypes.c_void_p), ("prof_cap", ctypes.c_uint), ("skel", ctypes.c_void_p), ("plog", ctypes.c_void_p), ("hsk", ctypes.c_void_p), ("hsl", ctypes.c_void_p), ("hdbg", ctypes.c_void_p), ("opart", ctypes.c_void_p), ("adone", ctypes.c_void_p), ("cpart", ctypes.c_void_p), ("ccnt", ctypes.c_void_p), ("ypart1", ctypes.c_void_p), ("fcnt", ctypes.c_void_p)]


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        if max_batch > MAXB:
            raise ValueError("this engine serves at most %d sequences" % MAXB)
        t0 = time.time()
        lib_path = os.path.join(HERE, "build", os.environ.get("MK_LIB", "libmk.so"))
        if not os.path.exists(lib_path):
            subprocess.run(["bash", os.path.join(HERE, "build.sh")], check=True)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.mk_create.restype = ctypes.c_void_p
        self.lib.mk_create.argtypes = [ctypes.c_void_p]
        self.lib.mk_run.restype = ctypes.c_int
        self.lib.mk_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.mk_poll.restype = ctypes.c_int
        self.lib.mk_poll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.lib.mk_params_size.restype = ctypes.c_int
        self.lib.mk_flag.restype = ctypes.c_int
        self.lib.mk_flag.argtypes = [ctypes.c_void_p]
        self.lib.mk_clear_flag.argtypes = [ctypes.c_void_p]
        assert self.lib.mk_params_size() == ctypes.sizeof(Params), (self.lib.mk_params_size(), ctypes.sizeof(Params))
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
                    if self.lib.mk_hsk():
                        P.hsk = keep(pack_head_sketch(ck.i16("lm_head.weight")))
                    P.lnf = keep(ck.f32("model.norm.weight"))
                    P.lnfb = keep(ck.i16("model.norm.weight").contiguous().clone())
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
                    L.wr = keep(pack_tiles(ck.i16(m + "router.weight")))
                    L.br = keep(ck.f32(m + "router.bias"))
                    # expert predictor (steers L2 prefetch only): router rows scaled by ln2 / ln1, applied to the
                    # ln1-normed layer input that phase A already holds
                    w1 = ck.f32(f"model.layers.{l}.input_layernorm.weight")
                    w2 = ck.f32(f"model.layers.{l}.post_attention_layernorm.weight")
                    ratio = torch.where(w1.abs() > 0, w2 / torch.where(w1.abs() > 0, w1, torch.ones_like(w1)), torch.zeros_like(w1))
                    wpf = (ck.f32(m + "router.weight") * ratio[None, :]).to(torch.bfloat16).view(torch.int16)
                    L.wp = keep(pack_tiles(wpf.contiguous()))
                    c, s = pack_fp4(ck.u8(m + "experts.gate_up_proj_blocks"), ck.u8(m + "experts.gate_up_proj_scales"))
                    L.gu_c = keep(c); L.gu_s = keep(s)
                    L.gu_b = keep(ck.f32(m + "experts.gate_up_proj_bias"))
                    c, s = pack_fp4(ck.u8(m + "experts.down_proj_blocks"), ck.u8(m + "experts.down_proj_scales"))
                    L.dn_c = keep(c); L.dn_s = keep(s)
                    L.dn_b = keep(ck.f32(m + "experts.down_proj_bias"))
                    L.ln1 = keep(ck.f32(f"model.layers.{l}.input_layernorm.weight"))
                    L.ln2 = keep(ck.f32(f"model.layers.{l}.post_attention_layernorm.weight"))
                    L.sinks = keep(ck.f32(a + "sinks"))
                    L.ln1b = keep(ck.i16(f"model.layers.{l}.input_layernorm.weight").contiguous().clone())
                    L.ln2b = keep(ck.i16(f"model.layers.{l}.post_attention_layernorm.weight").contiguous().clone())
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
        P.rlog = keep(torch.zeros(8 * MAXB * NE, dtype=torch.float32, device="cuda"))
        P.plog = keep(torch.zeros(8 * MAXB * NE, dtype=torch.float32, device="cuda"))
        P.hsl = keep(torch.zeros(152 * MAXB + MAXB * HID // 2 + 1 + 12800, dtype=torch.float32, device="cuda"))   # + HSK_XG rows (bf16), HSK_RQ queue
        P.opart = keep(torch.zeros(112 * 16 * MAXB, dtype=torch.float32, device="cuda"))
        P.adone = keep(torch.zeros(16, dtype=torch.int32, device="cuda"))
        P.cpart = keep(torch.zeros(28 * 2 * 128, dtype=torch.float32, device="cuda"))
        P.ccnt = keep(torch.zeros(32, dtype=torch.int32, device="cuda"))
        P.fcnt = keep(torch.zeros(192, dtype=torch.int32, device="cuda"))
        self.hdbg = torch.zeros(4, dtype=torch.int32, device="cuda")
        P.hdbg = keep(self.hdbg)
        P.ecnt = keep(torch.zeros(32, dtype=torch.int32, device="cuda"))
        P.apart = keep(torch.zeros(64 * 32 * 8 * 68, dtype=torch.float32, device="cuda"))
        P.acnt = keep(torch.zeros(64, dtype=torch.int32, device="cuda"))
        P.ypart = keep(torch.zeros(32 * MAXB * HID, dtype=torch.float32, device="cuda"))
        P.ypart1 = keep(torch.zeros(32 * MAXB * HID, dtype=torch.float32, device="cuda"))
        P.amv = keep(torch.zeros(152 * MAXB, dtype=torch.float32, device="cuda"))
        P.ami = keep(torch.zeros(152 * MAXB, dtype=torch.int32, device="cuda"))
        P.bar = keep(torch.zeros(64, dtype=torch.int32, device="cuda"))
        P.maxctx = self.maxctx
        self.prof_cap = int(os.environ.get("MK_PROF_CAP", "0"))
        if self.prof_cap:
            self.prof_buf = torch.zeros(152 * self.prof_cap * 2, dtype=torch.int64, device="cuda")
            P.prof = self.prof_buf.data_ptr(); P.prof_cap = self.prof_cap
        P.skel = keep(torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()
        self.h = self.lib.mk_create(ctypes.byref(P))
        if not self.h:
            raise RuntimeError("mk_create failed")
        self.pos = 0
        self.steps = 0
        self._pending = []
        self._cohort = None
        self.load_seconds = time.time() - t0

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def _check_ids(self) -> None:
        # the kernel flags token ids outside the vocabulary (it reads them as id 0 so that nothing outside the tables
        # is touched); the flag is in host-mapped memory and checked here, without a device synchronisation
        flag = self.lib.mk_flag(self.h)
        if flag:
            self.lib.mk_clear_flag(self.h)
            raise ValueError("a token id outside the vocabulary (0..%d) was given; the results since are not valid" % (VOCAB - 1))

    def reset(self) -> None:
        self._check_ids()
        self.pos = 0

    def _check_batch(self, B):
        if B < 1 or B > self.max_batch:
            raise ValueError("batch size %d: this engine was created for 1..%d sequences" % (B, self.max_batch))

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        self._check_ids()
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos >= self.max_context:
            raise RuntimeError("context is full (%d positions)" % self.max_context)
        tokens = tokens.to(torch.int64).contiguous()
        logits = torch.empty((B, VOCAB), dtype=torch.float32, device="cuda")
        if self.lib.mk_run(self.h, tokens.data_ptr(), 1, logits.data_ptr(), None, B, self.pos, 1, 1, 0, self._stream()):
            raise RuntimeError("kernel launch failed")
        self._tok = tokens
        self.pos += 1
        self.steps += 1
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        self._check_ids()
        B = tokens.shape[0]
        self._check_batch(B)
        if self.pos + n_steps > self.max_context:
            raise RuntimeError("context is full")
        tokens = tokens.to(torch.int64).contiguous()
        out = torch.empty((B, n_steps), dtype=torch.int64, device="cuda")
        if self.lib.mk_run(self.h, tokens.data_ptr(), 1, None, out.data_ptr(), B, self.pos, 1, n_steps, 1, self._stream()):
            raise RuntimeError("kernel launch failed")
        self._tok = tokens
        self.pos += n_steps
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
            if self.lib.mk_run(self.h, ids.data_ptr(), Pn, None, None, B, 0, Pn, n, 2, self._stream()):
                raise RuntimeError("kernel launch failed")
            self.pos = Pn - 1 + n
            self.steps += Pn - 1 + n
            self._cohort = {"rids": [r[0] for r in cohort], "n": n, "sent": 0, "ids": ids, "B": B,
                            "buf": np.zeros((B, n), dtype=np.int32)}
            return events
        c = self._cohort
        done = self.lib.mk_poll(self.h, c["buf"].ctypes.data_as(ctypes.c_void_p), c["B"], c["n"])
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
        hs = bool(self.lib.mk_hsk())
        d = {"implementation": "gpt-oss-120b megakernel v1 (grid barriers)", "track": "B" if hs else "A", "steps": self.steps,
             "load_seconds": round(self.load_seconds, 1)}
        if hs:
            h = self.hdbg.tolist()
            d["head_sketch"] = {"greedy_heads": h[2], "candidate_rowtiles": h[0], "overflow_blocks": h[1]}
        return d
