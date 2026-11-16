"""Qwen3-8B online engine: own CUDA kernels (engine/eng.cu) behind the offline and the online contract.

A *pass* runs T token columns once over the weights: the decode rows of every running request and chunks of the
prompts being prefilled, together. Python here loads the weights, keeps the slot / page tables and composes the
plan of each pass; no torch module and no library computes anything of the model.

    offline:  reset(), step(tokens[B]) -> logits [B, vocab], generate(tokens[B], n) -> tokens [B, n]
    online:   submit(rid, prompt, max_new_tokens), poll() -> [(rid, tokens, finished)], idle(), kv_pages(),
              drop_cache(), stats()
"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sched  # noqa: E402
H, NL, VOCAB, INNER = 4096, 36, 151936, 12288
PAGE_TOK = 512
NSLOT, MAXPG = 64, 68
LASTBIT = 0x10000


def _load_lib():
    so = os.environ.get("MK_SO") or os.path.join(HERE, "libeng.so")
    if not os.path.exists(so):
        subprocess.check_call(["bash", os.path.join(HERE, "build.sh")])
    lib = ctypes.CDLL(so)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.eng_create.restype = vp
    lib.eng_create.argtypes = [ci] + [vp] * 5
    lib.eng_plan_ints.restype = ci
    lib.eng_tcap.restype = ci
    lib.eng_selcap.restype = ci
    lib.eng_page_bytes.restype = ctypes.c_size_t
    lib.eng_set_plan.restype = None
    lib.eng_set_plan.argtypes = [vp, vp, ci, ci]
    lib.eng_pass.restype = None
    lib.eng_pass.argtypes = [vp, ci, ci, ci, ci, ci, ci, vp, vp, vp, ci, ci]
    lib.eng_const.restype = ci
    lib.eng_const.argtypes = [ci]
    lib.eng_step.restype = None
    lib.eng_step.argtypes = [vp, ci, ci, ci, ci, ci, vp, vp]
    lib.eng_generate.restype = None
    lib.eng_generate.argtypes = [vp, ci, ci, ci, ci, ci, vp, vp, ci, ci, ci]
    cd, cl = ctypes.c_double, ctypes.c_longlong
    lib.eng_ds_init.argtypes = [vp, vp, ci, ci]
    lib.eng_ds_submit.restype = cl
    lib.eng_ds_submit.argtypes = [vp, vp, ci, ci, cd, cl, cl]
    lib.eng_ds_arena_off.restype = cl
    lib.eng_ds_arena_off.argtypes = [vp, cl]
    lib.eng_ds_tick.argtypes = [vp, cd, cd]
    lib.eng_ds_harvest.argtypes = [vp, vp, ci]
    lib.eng_ds_running.argtypes = [vp]
    lib.eng_ds_run.argtypes = [vp]
    lib.eng_ds_step_only.argtypes = [vp, ci]
    lib.eng_ds_counters.argtypes = [vp, vp]
    lib.eng_poison_page.restype = None
    lib.eng_poison_page.argtypes = [vp, ci]
    lib.eng_stat.restype = ctypes.c_longlong
    lib.eng_stat.argtypes = [vp, ci]
    lib.eng_sync.restype = None
    lib.eng_sync.argtypes = []
    lib.eng_read_tokens.restype = None
    lib.eng_read_tokens.argtypes = [vp, vp, ci]
    lib.eng_destroy.restype = None
    lib.eng_destroy.argtypes = [vp]
    return lib


def pack_matrix(w: torch.Tensor) -> torch.Tensor:
    """[M, K] bf16 -> int16 stream in the kernels' operand layout [k // 64][row // 8][(k % 64) // 8][row % 8][k % 8]:
    a K chunk of 64 for a block of rows is one contiguous byte range, and in shared memory it is the K-major
    operand of the warpgroup MMA as it is (see xidx in eng.cu)."""
    m, k = w.shape
    v = w.contiguous().view(torch.int16).view(m // 8, 8, k // 64, 8, 8)   # [n, i8, c, ko, e]
    return v.permute(2, 0, 3, 1, 4).reshape(-1)


class DsParams(ctypes.Structure):   # engine/dsched.h
    _fields_ = [(k, ctypes.c_double) for k in ("target_s", "ttft_s", "margin_s", "share", "c_small", "c_seg", "c_dseg", "c_col", "c_pair", "speed", "boost_slack_s", "target_hi_s", "tail_guard_s", "target_tail_s", "near_s", "age_s", "target_near_s")] + \
               [("class_base", ctypes.c_double * 8), ("class_size", ctypes.c_int * 8)] + \
               [(k, ctypes.c_int) for k in ("n_blocks", "reuse", "sticky", "fifo", "min_chunk", "tcap", "selcap", "n_pages", "n_slots", "ss_shift", "partcap", "pad")]


class DsResult(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int) for k in ("n_sel", "n_cols", "n_free", "n_running", "n_queued", "n_cached", "head", "step")] + \
               [("est_s", ctypes.c_double), ("req", ctypes.c_int * 256), ("fin", ctypes.c_int * 256), ("tok", ctypes.c_int * 256)]


DS_COUNTERS = ("admitted finished cache_hit_pages cache_registered evictions recomputed_pages waits_for_leader lent_turns demoted "
               "prefill_tokens decode_tokens dropped_cache passes queue_max pages_min_free errors columns running_max steps").split()


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
        arr = np.memmap(fn, dtype=np.int16, mode="r", offset=base + b0, shape=((b1 - b0) // 2,))
        t = torch.from_numpy(np.array(arr)).cuda()
        return t.view(torch.bfloat16).view(e["shape"])


class Engine:
    def __init__(self, model_dir: str, max_batch: int, max_context: int):
        self.lib = _load_lib()
        self.max_batch, self.max_context = max_batch, max_context
        wts = _Weights(model_dir)
        layer_i16 = (6144 * 4096 + 4096 * 4096 + 24576 * 4096 + 4096 * 12288)
        total = NL * layer_i16 + VOCAB * H
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
        self.embed = wts.get("model.embed_tokens.weight").contiguous()
        # RoPE table exactly as the reference computes it (float32 angles)
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
        ang = torch.arange(max_context, dtype=torch.float)[:, None] * inv_freq[None, :]
        self.rope = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous().cuda()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.n_slots = min(max_batch, NSLOT)
        self.max_pages_per_seq = min(MAXPG, -(-max_context // PAGE_TOK))
        page_bytes = self.lib.eng_page_bytes()
        free, _ = torch.cuda.mem_get_info()
        reserve = int(float(os.environ.get("MK_RESERVE_GB", "1.0")) * 2**30) + self.lib.eng_fixed_bytes()
        self.n_pages = int(min(self.n_slots * self.max_pages_per_seq, (free - reserve) // page_bytes))
        if os.environ.get("MK_PAGES"):
            self.n_pages = min(self.n_pages, int(os.environ["MK_PAGES"]))
        self.h = self.lib.eng_create(self.n_pages, self.W.data_ptr(), self.embed.data_ptr(), self.norm_w.data_ptr(),
                                     self.qk_norm_w.data_ptr(), self.rope.data_ptr())
        if not self.h:
            raise RuntimeError("eng_create failed")
        self.TCAP, self.SELCAP = self.lib.eng_tcap(), self.lib.eng_selcap()
        self.PL_SLOT, self.PL_POS, self.PL_TOK = 0, self.TCAP, 2 * self.TCAP
        (self.PL_UNIT, _, self.PL_BT, self.PL_FOLD, self.PL_SEL, self.PL_PT, self.MAXU, _, self.MAXBT,
         self.PARTCAP, self.PL_CU, self.SS) = (self.lib.eng_const(i) for i in range(12))
        self.plan = np.zeros(self.lib.eng_plan_ints(), dtype=np.int32)
        self.pt = self.plan[self.PL_PT : self.PL_PT + NSLOT * MAXPG].reshape(NSLOT, MAXPG)   # view: page table inside the plan
        self.poison = bool(int(os.environ.get("MK_POISON", "0")))  # tests: NaN in every page that is freed
        if self.poison:
            for p in range(self.n_pages):
                self.lib.eng_poison_page(self.h, p)
        env = os.environ.get
        # track A's prompt passes are slower: the near-miss window (80 ms passes) put its median time between tokens
        # over 25 ms (replays: 25.2 / 26.5 ms with it, 23.9 / 23.9 without, 178 / 177-178 of 189)
        self.track_a = self.lib.eng_const(12) == 1
        self.S = sched.Scheduler(self.n_pages, self.n_slots, self.TCAP, self.SELCAP,
                                 policy=env("MK_POLICY", "deadline"), reuse=bool(int(env("MK_REUSE", "1"))),
                                 target_s=float(env("MK_TARGET_MS", "25")) * 1e-3, margin_s=float(env("MK_MARGIN_S", "0.3")),
                                 min_chunk=int(env("MK_MIN_CHUNK", "64")), share=float(env("MK_SHARE", "0.7")),
                                 sticky=bool(int(env("MK_STICKY", "1"))), boost_slack_s=float(env("MK_BOOST_S", "1.0")), target_hi_s=float(env("MK_TARGET_HI_MS", "45")) * 1e-3,
                                 tail_guard_s=float(env("MK_TAIL_S", "7")), target_tail_s=float(env("MK_TARGET_TAIL_MS", "80")) * 1e-3, near_s=float(env("MK_NEAR_S", "0" if self.track_a else "0.75")), age_s=float(env("MK_AGE_S", "10")), target_near_s=float(env("MK_TARGET_NEAR_MS", env("MK_TARGET_TAIL_MS", "80"))) * 1e-3, skip_dec=bool(int(env("MK_SKIP_DEC", "0"))),
                                 tbt_s=float(env("MK_TBT_MS", "85")) * 1e-3, skip_margin_s=float(env("MK_SKIP_MARGIN_S", "0.3")))
        if self.poison:
            self.S.freed_hook = lambda page: self.lib.eng_poison_page(self.h, page)
        # The scheduler on the device (default). MK_DEVSCHED=0: the Python reference scheduler above drives the passes.
        self.dev = bool(int(env("MK_DEVSCHED", "1")))
        # MK_GRAPH=1 (default): the serving loop (scheduler step + pass, while there is work) is one CUDA graph that
        # runs on the device; the host copies prompts in and results out on another stream. MK_GRAPH=0: the host
        # launches every step and pass (same kernels), keeping MK_DEPTH of them queued.
        self.use_graph = bool(int(env("MK_GRAPH", "1")))
        self.depth = int(env("MK_DEPTH", "2"))
        assert ctypes.sizeof(DsParams) == self.lib.eng_ds_param_bytes() and ctypes.sizeof(DsResult) == self.lib.eng_ds_result_bytes()
        S = self.S
        self.ds_params = DsParams(S.target_s, S.ttft_s, S.margin_s, S.share, S.c_small, S.c_seg, S.c_dseg, S.c_col, S.c_pair, 1.0, S.boost_slack_s, S.target_hi_s, S.tail_guard_s, S.target_tail_s, S.near_s, S.age_s, S.target_near_s,
                                  (ctypes.c_double * 8)(*[b for _, b in S.classes]), (ctypes.c_int * 8)(*[c for c, _ in S.classes]),
                                  S.n_blocks, int(S.reuse), int(S.sticky), int(S.policy == "fifo"), S.min_chunk, S.tcap, S.selcap,
                                  self.n_pages, self.n_slots, self.SS, self.PARTCAP, 0)
        self.RING = self.lib.eng_ds_ring()
        if self.dev:
            self.lib.eng_ds_init(self.h, ctypes.byref(self.ds_params), int(self.poison), int(self.use_graph))
        self.res_buf = (DsResult * 16)()
        self.steps_launched = 0
        self.steps_done = 0
        self.rid_of = {}                 # ring count -> rid of the requests the host still waits for
        self.backlog = []                # requests that did not fit into the ring / the arena yet
        self.oldest = 0                  # ring count of the oldest request not finished
        self.submitted = 0
        self.dev_free = self.n_pages
        self.t_last_done = 0.0
        self.dev_stale = False           # requests may still sit in slots after their last token
        self.pending_events = []
        self.prev_cols = False
        self.gen_graph = bool(int(env("MK_GEN_GRAPH", "1")))   # generate(): the loop over the steps as a graph on the device
        self.adapt = bool(int(env("MK_ADAPT", "1")))        # correct the cost model by the measured pass times
        self.corr = 1.0
        self.pass_log = [] if env("MK_PASSLOG", "") else None      # (columns, selected, estimated ms, measured ms) per pass
        self.t0 = time.perf_counter()
        # offline state
        self.pos = 0
        self._off_B = 0
        self._off_ss = -1
        self._off_pages = [[] for _ in range(NSLOT)]
        self._keep = None
        self.counters = dict(passes=0, host_syncs=0, columns=0, pass_ms_max=0.0, pass_ms_sum=0.0, est_ms_sum=0.0, compose_ms_sum=0.0)

    # ------------------------------------------------------------------ pages (offline path and tests)
    def _alloc(self, n):
        if len(self.S.free) < n:
            return None
        return [self.S.free.pop() for _ in range(n)]

    def _free(self, pages):
        for p in pages:
            self.S._release(p)

    def _upload(self):
        self.lib.eng_set_plan(self.h, self.plan.ctypes.data, 0, self.plan.size)

    # ------------------------------------------------------------------ offline contract
    def reset(self) -> None:
        self.pos = 0
        for b in range(NSLOT):
            if self._off_pages[b]:
                self._free(self._off_pages[b])
                self._off_pages[b] = []
        self._off_B = 0

    def _off_prepare(self, B, pos0, last_pos):
        """slots 0..B-1 own pages up to position last_pos; the plan is the B columns at position pos_add"""
        need = last_pos // PAGE_TOK + 1
        changed = B != self._off_B
        for b in range(B):
            while len(self._off_pages[b]) < need:
                got = self._alloc(1)
                if got is None:
                    raise RuntimeError("out of KV pages")
                self.pt[b, len(self._off_pages[b])] = got[0]
                self._off_pages[b].append(got[0])
                changed = True
        if changed or (pos0 >> self.SS) != self._off_ss:
            self._off_ss = pos0 >> self.SS
            self._off_counts = self._fill_plan([(b, 0, None, 1) for b in range(B)], pos_add=pos0)
            self._off_B = B
            self._upload()

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.int64 or not tokens.is_contiguous():
            tokens = tokens.to(torch.int64).contiguous()
        B = tokens.shape[0]
        self._off_prepare(B, self.pos, self.pos)
        logits = torch.empty((B, VOCAB), dtype=torch.float32, device="cuda")
        n2, n1, nf = self._off_counts
        if self.gen_graph:      # one graph launch per step
            self.lib.eng_step(self.h, B, n2, n1, nf, self.pos, tokens.data_ptr(), logits.data_ptr())
        else:
            self.lib.eng_pass(self.h, B, n2, n1, nf, B, self.pos, tokens.data_ptr(), logits.data_ptr(), None, 0, 0)
        self.pos += 1
        self._keep = tokens  # the kernels read it asynchronously
        return logits

    def generate(self, tokens: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Greedy decode n_steps tokens: argmax, token feedback and the loop over the steps are on the device (one
        graph launch per stretch of steps with the same plan); the host does not wait."""
        if tokens.dtype != torch.int64 or not tokens.is_contiguous():
            tokens = tokens.to(torch.int64).contiguous()
        B = tokens.shape[0]
        if self.pos + n_steps > self.max_context:
            raise ValueError("generate() would exceed max_context")
        out = torch.empty((B, n_steps), dtype=torch.int64, device="cuda")
        i, ss = 0, 1 << self.SS
        while i < n_steps:
            run = min(n_steps - i, ss - ((self.pos + i) & (ss - 1)))   # the plan changes when the rows enter a new super-segment
            self._off_prepare(B, self.pos + i, self.pos + n_steps - 1)
            n2, n1, nf = self._off_counts
            if self.gen_graph:
                self.lib.eng_generate(self.h, B, n2, n1, nf, self.pos + i, tokens.data_ptr(), out.data_ptr(), n_steps, i, run)
            else:
                for k in range(i, i + run):
                    self.lib.eng_pass(self.h, B, n2, n1, nf, B, self.pos + k, tokens.data_ptr() if k == 0 else None, None, out.data_ptr(), n_steps, k)
            i += run
        self.pos += n_steps
        self._keep = (tokens, out)
        return out

    # ------------------------------------------------------------------ online contract
    # With the scheduler on the device the host does three things: copy a prompt in (submit), launch the next
    # pass while requests are outstanding, copy result blocks out (poll). It does not know what a pass contains.
    def submit(self, rid: int, prompt, max_new_tokens: int) -> None:
        now = time.perf_counter() - self.t0
        if not self.dev:
            self.S.submit(rid, prompt, max_new_tokens, now)
            return
        self.backlog.append((rid, np.ascontiguousarray(prompt, dtype=np.int32), int(max_new_tokens), now))
        self._feed()

    def _feed(self):
        while self.backlog:
            rid, ids, n_out, t = self.backlog[0]
            oldest_off = self.lib.eng_ds_arena_off(self.h, self.oldest) if self.oldest < self.submitted else 0
            count = self.lib.eng_ds_submit(self.h, ids.ctypes.data, ids.size, n_out, t, self.oldest, oldest_off)
            if count < 0:
                self.counters["backlog_waits"] = self.counters.get("backlog_waits", 0) + 1
                return
            self.backlog.pop(0)
            self.rid_of[count] = rid
            self.submitted = count + 1

    def idle(self) -> bool:
        if not self.dev:
            return self.S.idle()
        return not self.rid_of and not self.backlog    # a pass still in flight then carries nothing

    def _settle(self, cmd=2):
        """off the serving path: let the scheduler do its bookkeeping (cmd 1: drop the prefix cache). If the serving
        loop is still running the host keeps reading results until it ends; their events go out with the next poll."""
        if self.use_graph:
            while self.lib.eng_ds_running(self.h):
                self.pending_events.extend(self._harvest())
        else:
            while self.steps_launched > self.steps_done:
                self.pending_events.extend(self._harvest())
        self.lib.eng_ds_step_only(self.h, cmd)
        self.steps_launched += 1
        self.pending_events.extend(self._harvest())
        self.dev_stale = False

    def kv_pages(self):
        if not self.dev:
            return (self.S.pages_free(), self.n_pages)
        if self.dev_stale and not self.rid_of:
            self._settle()
        return (self.dev_free, self.n_pages)

    def drop_cache(self) -> None:
        if not self.dev:
            self.S.drop_cache()
        else:
            self._settle(1)

    def _fill_plan(self, runs, pos_add=0):
        """runs: [(slot, pos0, tokens or None (decode: the slot's last token), select)] -> the plan (host copy).
        Returns (attention tasks, units to fold); sets self._plan_cols, self._plan_sel."""
        p, SS = self.plan, self.SS
        col, sel, units, two, one, fold, nparts = 0, [], [], [], [], [], 0
        for ri, (slot, pos0, toks, select) in enumerate(runs):
            n = 1 if toks is None else len(toks)
            idx = np.arange(n, dtype=np.int32)
            p[self.PL_SLOT + col : self.PL_SLOT + col + n] = slot | ((idx & 15) << 17)
            p[self.PL_SLOT + col + n - 1] |= LASTBIT
            p[self.PL_POS + col : self.PL_POS + col + n] = pos0 + idx
            p[self.PL_TOK + col : self.PL_TOK + col + n] = -1 if toks is None else toks
            p[self.PL_CU + col : self.PL_CU + col + n] = len(units) + (idx >> 4)
            # attention units of at most 16 positions (one warpgroup); a unit that spans several super-segments
            # (4096 positions) leaves one part per super-segment in the scratch, folded afterwards
            u0 = len(units)
            for c0 in range(0, n, 16):
                nq = min(16, n - c0)
                nss = ((pos0 + pos_add + c0 + nq - 1) >> SS) + 1
                if nss > 1:
                    fold.append(len(units))
                    units.append((col + c0, nq, nparts, nss))
                    nparts += nss
                else:
                    units.append((col + c0, nq, -1, 1))
            # tasks, per super-segment and kv head: pairs of units that both reach the super-segment (two
            # warpgroups of a block), single units otherwise (decode rows, odd ends). Listed by (super-segment,
            # kv head, units) so that the blocks running at the same time read the same K/V.
            nu = len(units) - u0
            for ss in range(units[u0 + nu - 1][3]):
                first = next(u for u in range(u0, u0 + nu) if units[u][3] > ss)     # units below do not reach it
                live = list(range(first, u0 + nu))
                pairs2 = [live[i] for i in range(0, len(live) - 1, 2)]
                for g in range(8):
                    two.extend(u | (ss << 16) | (g << 22) for u in pairs2)
                    if len(live) & 1:
                        one.append(live[-1] | (ss << 16) | (g << 22))
            if select == 1:
                sel.append((col + n - 1, slot, ri))
            elif select == 2:   # tests: every column of the run
                sel.extend((col + i, slot if i == n - 1 else -1, ri) for i in range(n))
            col += n
        one_dec = [t for t in one if units[t & 0xfff][1] == 1]      # units of one position: the decode rows' kernel
        bt = two + one_dec + [t for t in one if units[t & 0xfff][1] != 1]
        p[p.size - 16 + 6] = len(one_dec)                            # plan header, element 6
        assert col <= self.TCAP and len(sel) <= self.SELCAP, (col, len(sel))
        assert len(units) <= self.MAXU and len(bt) <= self.MAXBT and nparts <= self.PARTCAP, (len(units), len(bt), nparts)
        p[self.PL_UNIT : self.PL_UNIT + 4 * len(units)] = np.asarray(units, dtype=np.int32).reshape(-1)
        p[self.PL_BT : self.PL_BT + len(bt)] = bt
        p[self.PL_FOLD : self.PL_FOLD + len(fold)] = fold
        if sel:
            p[self.PL_SEL : self.PL_SEL + 2 * len(sel)] = np.asarray([(c, s) for c, s, _ in sel], dtype=np.int32).reshape(-1)
        self._plan_cols, self._plan_sel = col, [ri for _, _, ri in sel]
        return len(two), len(one), len(fold)

    def _run_pass(self, runs, want_logits=False):
        """Fill and upload the plan of a pass, launch it. Returns (run index of every selected column, logits)."""
        n2, n1, nf = self._fill_plan(runs)
        col, sel = self._plan_cols, self._plan_sel
        self._upload()
        logits = torch.empty((len(sel), VOCAB), dtype=torch.float32, device="cuda") if want_logits else None
        self.lib.eng_pass(self.h, col, n2, n1, nf, len(sel), 0, None, logits.data_ptr() if want_logits else None, None, 0, 0)
        self.counters["passes"] += 1
        self.counters["columns"] += col
        return sel, logits

    def poll(self):
        if self.dev:
            return self._poll_dev()
        S = self.S
        t_a = time.perf_counter()
        if self.adapt:
            S.speed = self.corr
        plan = S.compose(t_a - self.t0)
        if plan is None:
            return []
        runs = []
        for r, slot, pos0, n, toks, select in plan.runs:
            self.pt[slot, :len(r.pages)] = r.pages
            runs.append((slot, pos0, toks, select))
        t_b = time.perf_counter()
        sel, _ = self._run_pass(runs)
        S.launched(plan)
        toks = np.empty(max(1, len(sel)), dtype=np.int32)
        if sel:
            self.lib.eng_read_tokens(self.h, toks.ctypes.data, len(sel))   # the one host sync of a pass
        else:
            self.lib.eng_sync()
        t_c = time.perf_counter()
        c = self.counters
        c["host_syncs"] += 1
        dt = t_c - t_b
        c["pass_ms_max"] = max(c["pass_ms_max"], dt * 1e3)
        c["pass_ms_sum"] += dt * 1e3
        c["est_ms_sum"] += plan.est_s * 1e3
        c["compose_ms_sum"] += (t_b - t_a) * 1e3
        if self.adapt and plan.n_pre:
            self.corr = min(1.5, max(1.0, 0.95 * self.corr + 0.05 * dt / plan.est_s))
        return S.complete(plan, toks, t_c - self.t0)

    def _harvest(self):
        """result blocks of the steps the device finished -> events"""
        events, c = [], self.counters
        while True:
            n = self.lib.eng_ds_harvest(self.h, self.res_buf, 16)
            for i in range(n):
                res = self.res_buf[i]
                now = time.perf_counter() - self.t0
                self.steps_done += 1
                if res.n_cols:
                    c["passes"] += 1
                    c["columns"] += res.n_cols
                    # a pass's duration as the host sees it: the time since the previous pass was read, when that
                    # one came alone and the device was not idle in between
                    if self.prev_cols and n == 1:
                        dur = now - self.t_last_done
                        c["pass_ms_max"] = max(c["pass_ms_max"], dur * 1e3)
                        c["pass_ms_sum"] += dur * 1e3
                        c["est_ms_sum"] += res.est_s * 1e3
                        if self.pass_log is not None:
                            self.pass_log.append((res.n_cols, res.n_sel, res.est_s * 1e3, dur * 1e3))
                        # the correction follows the passes that carry prompt columns: only their size is a choice
                        if self.adapt and res.est_s > 0 and dur < 3 * res.est_s and res.n_cols != res.n_sel:
                            self.corr = min(1.5, max(1.0, 0.95 * self.corr + 0.05 * dur / res.est_s))
                self.prev_cols = bool(res.n_cols) and n == 1
                self.t_last_done = now
                self.dev_free = res.n_free
                base = self.oldest - self.oldest % self.RING      # ring index -> count (the live window is shorter than the ring)
                for j in range(res.n_sel):
                    count = base + res.req[j]
                    if count < self.oldest:
                        count += self.RING
                    fin = bool(res.fin[j])
                    events.append((self.rid_of[count], [int(res.tok[j])], fin))
                    if fin:
                        del self.rid_of[count]
                        self.dev_stale = True
                while self.oldest < self.submitted and self.oldest not in self.rid_of:
                    self.oldest += 1
            if n < 16:
                return events

    def _poll_dev(self):
        lib, h = self.lib, self.h
        lib.eng_ds_tick(h, time.perf_counter() - self.t0, self.corr if self.adapt else 1.0)
        events = self.pending_events + self._harvest()
        self.pending_events = []
        if self.backlog:
            self._feed()
        if self.rid_of:
            if self.use_graph:
                if not lib.eng_ds_running(h):      # the loop ended (nothing was left when it looked) or was never started
                    lib.eng_ds_run(h)
                    self.counters["loop_starts"] = self.counters.get("loop_starts", 0) + 1
            else:
                while self.steps_launched - self.steps_done < self.depth:
                    lib.eng_ds_run(h)
                    self.steps_launched += 1
        return events

    # ------------------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.eng_destroy(self.h)
            self.h = None
        self.W = self.embed = self.norm_w = self.qk_norm_w = self.rope = None

    def stats(self) -> dict:
        s = {
            "implementation": "pass engine (eng.cu): decode rows and prefill chunks in one pass over the weights",
            "kernel_launches": int(self.lib.eng_stat(self.h, 0)),
            "passes": int(self.lib.eng_stat(self.h, 1)),
            "plan_uploads": int(self.lib.eng_stat(self.h, 3)),
            "token_reads": int(self.lib.eng_stat(self.h, 4)),
            "kv_pages_total": self.n_pages,
            "kv_pages_free": len(self.S.free),
            "kv_pages_cached": len(self.S.cache),
            "queue": len(self.S.queue),
            "running": len(self.S.running),
            "policy": self.S.policy,
            "prefix_reuse": self.S.reuse,
            "target_pass_ms": self.S.target_s * 1e3,
            "cost_model_correction": round(self.corr, 3),
            "host_syncs_per_pass": 1,
            "max_batch": self.max_batch,
            "max_context": self.max_context,
        }
        if self.dev:
            cnt = (ctypes.c_longlong * 24)()
            self.lib.eng_ds_counters(self.h, cnt)
            s.update({k: int(cnt[i]) for i, k in enumerate(DS_COUNTERS)})
            s.update(scheduler="on the device (k_sched, engine/dsched.h)", host_syncs_per_pass=0,
                     serving_loop="CUDA graph: WHILE(work) { step; pass; commit } on the device" if self.use_graph else "host launches every step",
                     kv_pages_free=self.dev_free, queue=len(self.backlog), running=len(self.rid_of),
                     graph_launches=int(cnt[23]), host_launched_steps=int(cnt[22]))
            s["cache_hit_tokens"] = s["cache_hit_pages"] * PAGE_TOK
        else:
            s.update(self.S.c)
            s.update(scheduler="host (engine/sched.py)")
        s.update(self.counters)
        return s
