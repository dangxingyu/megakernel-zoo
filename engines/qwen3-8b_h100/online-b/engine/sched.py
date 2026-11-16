"""Scheduler of the online engine: requests, slots, KV pages, prefix cache, and what goes into each pass.

Pure Python (no torch): the engine executes the plans, tests/sim.py runs the same code against a cost model.

A pass = every decode row + a time budget of prefill columns. Policy:
  * prefill order: requests that can still give their first token within the service level come first, in
    deadline order (Moore-Hodgson over the estimated remaining work: when the set cannot all be on time, the
    largest job is demoted); demoted and late requests follow, oldest first;
  * the prefill budget of a pass is what keeps the pass (= the time between two tokens of every decoding
    stream) near `target_s`, by a cost model of the kernels (per column, per query-key pair);
  * KV pages (512 positions) are reference counted. Complete prompt pages are registered in a prefix cache
    under the hash of the whole prefix up to and including them; a request adopts every leading page that is
    already there, waits for a page another running request is computing (and lends that request its turn),
    and computes the rest. Cached pages nobody uses are evicted leaf first, least recently used first, when a
    page is needed; an evicted prefix is simply computed again by the next request that needs it.
"""

from __future__ import annotations

import hashlib
import heapq
import struct
from collections import deque

import numpy as np

PAGE = 512


CFIX_S, SHARE_MIN = 0.006, 0.15   # deadline estimates: fixed cost of a pass; least share of the GPU assumed for prompts (dsched.h)
ARRIVAL_Q = 4.0   # the deadline order looks at arrival times in units of 1 / this seconds (dsched.h DS_ARRIVAL_Q)
DLAT = 0.68     # decode rows: a block alone on its task takes this part of c_dseg per segment (dsched.h DS_DLAT)


class Req:
    __slots__ = ("rid", "prompt", "n_in", "n_out", "t_arrive", "slot", "pos", "produced", "pages", "chain", "max_reuse",
                 "need_pages", "reserved", "late", "was_late", "t_first", "claims", "hits")

    def __init__(self, rid, prompt, n_out, now):
        self.rid = rid
        self.prompt = np.ascontiguousarray(prompt, dtype=np.int32)
        self.n_in, self.n_out = int(self.prompt.size), int(n_out)
        self.t_arrive = now
        self.slot, self.pos, self.produced = -1, 0, 0
        self.pages = []            # page id per page index (shared or own)
        self.chain = None          # prefix hash per complete prompt page
        self.max_reuse = (self.n_in - 1) // PAGE   # the last prompt token is always computed
        self.need_pages = (self.n_in + self.n_out - 1 + PAGE - 1) // PAGE
        self.reserved = 0          # pages this request may still take from the pool
        self.late = False
        self.was_late = False
        self.t_first = None
        self.claims = []           # prefix hashes this request is computing (pending)
        self.hits = 0              # pages adopted from the cache

    def hashes(self):
        if self.chain is None:
            self.chain, h = [], b""
            raw = self.prompt.tobytes()
            for k in range(self.n_in // PAGE):
                h = hashlib.blake2b(h + raw[k * PAGE * 4:(k + 1) * PAGE * 4], digest_size=16).digest()
                self.chain.append(h)
        return self.chain


class Plan:
    """one pass: runs = [(req or None, slot, pos0, n, tokens or None, select)]"""
    __slots__ = ("runs", "n_cols", "n_dec", "n_pre", "est_s", "sel")

    def __init__(self):
        self.runs, self.n_cols, self.n_dec, self.n_pre, self.est_s, self.sel = [], 0, 0, 0, 0.0, []


class Scheduler:
    def __init__(self, n_pages, n_slots, tcap=2048, selcap=256, **kw):
        self.n_pages, self.n_slots, self.tcap, self.selcap = n_pages, n_slots, tcap, selcap
        # policy
        self.target_s = kw.get("target_s", 0.060)        # pass time aimed at
        self.ttft_s = kw.get("ttft_s", 5.0)
        self.margin_s = kw.get("margin_s", 0.30)         # deadline margin: one pass + polling
        self.policy = kw.get("policy", "deadline")       # "deadline" | "fifo"
        self.reuse = kw.get("reuse", True)
        self.min_chunk = kw.get("min_chunk", 64)
        self.share = kw.get("share", 0.7)                # of the GPU that goes to prefill while streams decode (deadline estimates)
        self.sticky = kw.get("sticky", True)             # a request once judged late stays late
        self.skip_dec = kw.get("skip_dec", False)        # streams with slack sit out while on-time prompts wait (off: see journal)
        self.tbt_s = kw.get("tbt_s", 0.085)              # mean time between tokens aimed at (service level: 0.1)
        self.skip_margin_s = kw.get("skip_margin_s", 0.3)
        self.share_now = self.share
        # passes aim at target_hi_s while an on-time prompt has less than boost_slack_s to spare by the estimates (0: never)
        self.boost_slack_s = kw.get("boost_slack_s", 0.0)
        self.target_hi_s = kw.get("target_hi_s", self.target_s)
        self.min_slack = 1e9
        # the tail of the first-token times: passes aim at target_tail_s while some prompt is heading for a first token
        # later than tail_guard_s by the estimates (0: never)
        self.tail_guard_s = kw.get("tail_guard_s", 0.0)
        self.target_tail_s = kw.get("target_tail_s", 0.080)
        self.near_s = kw.get("near_s", 0.0)              # ... and while some prompt is heading for a first token within near_s after ttft_s
        self.age_s = kw.get("age_s", 0.0)                # prompts waiting longer than this come first, oldest first (0: never)
        self.target_near_s = kw.get("target_near_s", self.target_tail_s)   # the pass target in a near miss
        self.near_miss = False
        self.worst_ttft = 0.0
        # Cost model of the kernels (seconds), from tests/calib_cost.py:
        #  * GEMM and the rest by the total number of columns, which is kept a multiple of 256: a column block of the
        #    GEMM costs the same full or not (1040 columns cost 13 ms more than 1024). Per class: time at context 0
        #    without the attention inside the chunk. A pass with decode rows only costs `c_small`;
        #  * attention: a task = (32 positions of a prefill run or one decode row, kv head, super-segment of 32
        #    segments of 128 keys) on one block, 132 blocks: time = 36 layers x per-segment time x
        #    max(longest task, all tasks / 132); 2.35 us per segment for 32 positions, 2.8 us for a decode row
        #    (bounded by reading its K/V: 147 KB per token of context).
        self.classes = kw.get("classes", ((128, 0.0077), (256, 0.00965), (384, 0.0129), (512, 0.0143), (768, 0.0222),
                                          (1024, 0.0256), (1536, 0.0393), (2048, 0.0503)))
        self.c_small = kw.get("c_small", 0.0066)
        self.c_seg = kw.get("c_seg", 36 * 2.35e-6)
        self.c_dseg = kw.get("c_dseg", 36 * 2.8e-6)
        self.n_blocks, self.n_warps = 132, 1056
        self.c_col = kw.get("c_col", 25e-6)              # for the deadline estimates: per column
        self.c_pair = kw.get("c_pair", 1.3e-9)           # ... and per (query, key) pair
        self.speed = 1.0                                 # measured / modelled pass time (set by the engine)
        # state
        self.queue = deque()
        self.running = {}            # slot -> Req
        self.free_slots = list(range(n_slots - 1, -1, -1))
        self.free = list(range(n_pages - 1, -1, -1))
        self.ref = np.zeros(n_pages, dtype=np.int32)      # running requests using a page
        self.cache = {}              # prefix hash -> [page, parent hash, children, last use]
        self.page_hash = {}          # page -> prefix hash (cached pages)
        self.pending = {}            # prefix hash -> Req computing that page
        self.committed = 0           # pages running requests may still take
        self.clock = 0
        self.freed_hook = None       # called with a page id when it returns to the free list (tests: poison)
        self.c = dict(admitted=0, finished=0, queue_max=0, pages_min_free=n_pages, cache_hit_pages=0, cache_hit_tokens=0,
                      cache_registered=0, evictions=0, recomputed_pages=0, waits_for_leader=0, lent_turns=0, demoted=0,
                      prefill_tokens=0, decode_tokens=0, dropped_cache=0, stream_skips=0)
        self.evicted = set()         # prefix hashes evicted (to count recomputation)

    # ------------------------------------------------------------------ pages and cache
    def reclaimable(self):
        return sum(1 for h, e in self.cache.items() if self.ref[e[0]] == 0)

    def pages_free(self):
        """pages not held by any running request and not retained by the cache"""
        return len(self.free)

    def _evict_one(self):
        # the least recently used cached page nobody uses: a leaf if there is one, else any (see dsched.h)
        best = None
        for leaf_only in (True, False):
            for h, e in self.cache.items():
                if (e[2] == 0 or not leaf_only) and self.ref[e[0]] == 0 and (best is None or e[3] < self.cache[best][3]):
                    best = h
            if best is not None:
                break
        if best is None:
            return False
        self._uncache(best, free=True)
        self.evicted.add(best)
        self.c["evictions"] += 1
        return True

    def _uncache(self, h, free):
        page, parent, _, _ = self.cache.pop(h)
        del self.page_hash[page]
        if parent is not None and parent in self.cache and self.cache[parent][2] > 0:
            self.cache[parent][2] -= 1
        if free:
            self._release(page)

    def _release(self, page):
        if self.freed_hook:
            self.freed_hook(page)
        self.free.append(page)

    def _take_page(self):
        if not self.free and not self._evict_one():
            raise RuntimeError("out of KV pages (reservation accounting broken)")
        self.c["pages_min_free"] = min(self.c["pages_min_free"], len(self.free) - 1)
        return self.free.pop()

    def drop_cache(self):
        for h in list(self.cache):
            page = self.cache[h][0]
            self._uncache(h, free=self.ref[page] == 0)   # a page still used by a running request becomes private to it
            self.c["dropped_cache"] += 1
        self.evicted.clear()

    def _adopt(self, r):
        """take over every cached page at the request's position (page boundary); returns pages adopted"""
        n = 0
        if not self.reuse:
            return 0
        chain = r.hashes()
        while r.pos % PAGE == 0 and r.pos // PAGE < r.max_reuse and len(r.pages) == r.pos // PAGE:
            e = self.cache.get(chain[r.pos // PAGE])
            if e is None:
                break
            self.clock += 1
            e[3] = self.clock
            self.ref[e[0]] += 1
            r.pages.append(e[0])
            r.pos += PAGE
            r.hits += 1
            n += 1
            if r.reserved > 0:
                r.reserved -= 1
                self.committed -= 1
        if n:
            self.c["cache_hit_pages"] += n
            self.c["cache_hit_tokens"] += n * PAGE
        return n

    def _hit_count(self, r):
        """leading pages of a queued request that are in the cache now"""
        if not self.reuse:
            return 0, 0
        chain, k, idle = r.hashes(), 0, 0
        while k < r.max_reuse:
            e = self.cache.get(chain[k])
            if e is None:
                break
            idle += self.ref[e[0]] == 0
            k += 1
        return k, idle

    # ------------------------------------------------------------------ requests
    def submit(self, rid, prompt, n_out, now):
        self.queue.append(Req(rid, prompt, n_out, now))
        self.c["queue_max"] = max(self.c["queue_max"], len(self.queue))

    def idle(self):
        return not self.queue and not self.running

    def _try_admit(self, r):
        if not self.free_slots:
            return False
        hits, idle_hits = self._hit_count(r)
        need = r.need_pages - hits
        if need > len(self.free) + self.reclaimable() - idle_hits - self.committed:
            return False
        r.slot = self.free_slots.pop()
        r.reserved = r.need_pages
        self.committed += r.need_pages
        self.running[r.slot] = r
        self._adopt(r)
        self.c["admitted"] += 1
        return True

    def _finish(self, r):
        chain = r.hashes()
        for k, page in enumerate(r.pages):
            if page in self.page_hash:
                self.ref[page] -= 1
            else:
                self.ref[page] = 0
                self._release(page)
        for h in r.claims:
            if self.pending.get(h) is r:
                del self.pending[h]
        self.committed -= r.reserved
        r.reserved, r.pages = 0, []
        del self.running[r.slot]
        self.free_slots.append(r.slot)
        self.c["finished"] += 1

    # ------------------------------------------------------------------ cost model
    def cost_prefill(self, pos, n):
        return n * self.c_col + self.c_pair * n * (pos + 0.5 * n)

    def remaining_cost(self, r, pos=None):
        pos = r.pos if pos is None else pos
        return self.cost_prefill(pos, r.n_in - pos)

    def att_cost(self, max_segs, sum_segs):
        return self.c_seg * max(max_segs, sum_segs / self.n_blocks) if sum_segs else 0.0

    def dec_cost(self, positions):
        if not positions:
            return 0.0
        segs = [p // 128 + 1 for p in positions]
        return self.c_dseg * max(DLAT * min(32, max(segs)), sum(segs) * 8 / self.n_blocks)

    @staticmethod
    def _run_segs(pos, n):
        """(longest block task, all block tasks) in segments for a prefill run of n columns at pos"""
        return min(32, (pos + n - 1) // 128 + 1), ((n + 31) // 32) * 8 * ((pos + 0.5 * n) / 128 + 1)

    def _fill(self, cands, cap, budget):
        """Greedy fill of at most `cap` prefill columns in candidate order within an attention budget (seconds).
        Returns ([(req, n)], attention cost). No side effects."""
        out, mx, sm, cols = [], 0, 0.0, 0
        for r in cands:
            room = cap - cols
            if room < 4 or len(out) >= self.selcap - 70:
                break
            left = r.n_in - r.pos
            hi = min(left, room)
            if hi < left:
                hi -= hi % 4
            def cost(n):
                m, t = self._run_segs(r.pos, n)
                return self.att_cost(max(mx, m), sm + t)
            if cost(hi) <= budget:
                n = hi
            else:
                lo = 0
                while hi - lo > 4:            # largest n (multiple of 4) within the budget
                    mid = (lo + hi) // 8 * 4
                    if mid <= lo:
                        break
                    if cost(mid) <= budget:
                        lo = mid
                    else:
                        hi = mid
                n = lo
                if n < min(self.min_chunk, left):
                    if out:
                        continue                 # a later (shorter) candidate may still fit
                    n = min(self.min_chunk, left, room - room % 4 if room < left else left)   # always make progress
            if n <= 0:
                continue
            m, t = self._run_segs(r.pos, n)
            mx, sm, cols = max(mx, m), sm + t, cols + n
            out.append((r, n))
        return out, self.att_cost(mx, sm)

    # ------------------------------------------------------------------ one pass
    def _order(self, now):
        """prefill candidates (running and queued) in service order"""
        cands = [r for r in self.running.values() if r.pos < r.n_in] + list(self.queue)
        if self.policy == "fifo":
            self.min_slack, self.worst_ttft = 1e9, 0.0
            return sorted(cands, key=lambda r: r.t_arrive)
        # Moore-Hodgson over the requests in deadline order: add each to the on-time set; when the set cannot make
        # the last deadline, drop its largest job. Recomputed at every pass from the current estimates (a request
        # that looked hopeless can come back when the others went faster than estimated), unless `sticky`.
        heap, t, late = [], now, set()
        costs = {}
        for r in cands:
            if self.sticky and r.late:
                continue
            hits = self._hit_count(r)[0] if r.slot < 0 else 0
            costs[r.rid] = self.remaining_cost(r, max(r.pos, hits * PAGE)) * self.speed / self.share_now
        # deadline order by arrival in quarters of a second; prompts that arrive together (a burst): the shortest first
        cands.sort(key=lambda r: (int(r.t_arrive * ARRIVAL_Q), costs.get(r.rid, 0.0), r.rid))
        for r in cands:
            if self.sticky and r.late:
                late.add(r.rid)
                continue
            cost = costs[r.rid]
            t += cost
            heapq.heappush(heap, (-cost, r.rid, r))
            if t > r.t_arrive + self.ttft_s - self.margin_s:
                _, _, worst = heapq.heappop(heap)
                t -= costs[worst.rid]
                late.add(worst.rid)
        t2, self.min_slack = now, 1e9
        for r in cands:
            if r.rid not in late and r.rid in costs:
                t2 += costs[r.rid]
                self.min_slack = min(self.min_slack, r.t_arrive + self.ttft_s - self.margin_s - t2)
        for r in cands:
            is_late = r.rid in late
            if is_late and not r.was_late:
                r.was_late = True
                self.c["demoted"] += 1
            r.late = is_late
        first = [r for r in cands if not r.late]
        # the others have missed their first token anyway: shortest remaining work first
        rest = sorted((r for r in cands if r.late), key=lambda r: (costs.get(r.rid) or self.remaining_cost(r), r.rid))
        order = first + rest
        if self.age_s > 0:   # age cap (dsched.h, where arrival times are float32)
            f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
            old = sorted((r for r in order if now - f32(r.t_arrive) > self.age_s), key=lambda r: (f32(r.t_arrive), r.rid))
            if old:
                ids = {r.rid for r in old}
                order = old + [r for r in order if r.rid not in ids]
        # the worst first-token time anybody is heading for, by the estimates, in this order
        t3, self.worst_ttft, self.near_miss = now, 0.0, False
        f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]     # (dsched.h keeps arrival times in float32)
        for r in order:
            t3 += costs.get(r.rid) or self.remaining_cost(r) * self.speed / self.share_now
            w = t3 - f32(r.t_arrive)
            self.worst_ttft = max(self.worst_ttft, w)
            if self.near_s > 0 and self.ttft_s < w <= self.ttft_s + self.near_s:
                self.near_miss = True
        return order

    def compose(self, now):
        """Admit, then build the next pass. Returns a Plan or None (nothing to do)."""
        # the part of the GPU the prompts get: what the pass target leaves after the fixed cost of a pass and the
        # streams that decode now (capped by `share`)
        tgt = self.target_s / self.speed
        dec_now = self.dec_cost([r.pos for r in self.running.values() if r.pos >= r.n_in])
        self.share_now = min(self.share, max(SHARE_MIN, (tgt - CFIX_S - dec_now) / tgt))
        order = self._order(now)
        for r in order:
            if r.slot < 0 and self._try_admit(r):
                self.queue.remove(r)
        if not self.running:
            return None
        plan = Plan()
        dec_pos = []
        # Streams. The service level is on the MEAN time between a request's tokens, so a stream that has been
        # served faster than that owns slack. While prompts that can still make their first-token deadline are
        # waiting, a stream sits the pass out if it can afford it: its last token is due at first token +
        # tbt_s x (tokens - 1); it must be able to finish from here at one token per `target_s` pass with
        # `skip_margin_s` to spare. Streams of requests that already missed their first token have no service
        # level left and always sit out. Every stream rides again as soon as no such prompt is waiting.
        critical = self.skip_dec and any(not r.late for r in order if r.pos < r.n_in)
        for r in self.running.values():
            if r.pos >= r.n_in:
                if critical:
                    if r.t_first - r.t_arrive > self.ttft_s:
                        self.c["stream_skips"] += 1
                        continue
                    left = r.n_out - r.produced
                    if r.t_first + self.tbt_s * (r.n_out - 1) - now - left * self.target_s * 1.1 > self.skip_margin_s:
                        self.c["stream_skips"] += 1
                        continue
                plan.runs.append((r, r.slot, r.pos, 1, None, 1))
                dec_pos.append(r.pos)
        plan.n_dec = len(dec_pos)
        dec = self.dec_cost(dec_pos)
        # prefill candidates in service order: adopt cached pages; a request whose next page is being computed
        # by another one waits for it and lends that one its turn
        cands, seen = [], set()
        for r in order:
            if r.slot < 0 or r.pos >= r.n_in or r.rid in seen:
                continue
            self._adopt(r)
            if self.reuse and r.pos % PAGE == 0 and len(r.pages) == r.pos // PAGE and r.pos // PAGE < r.max_reuse:
                lead = self.pending.get(r.hashes()[r.pos // PAGE])
                if lead is not None and lead is not r and lead.slot >= 0 and lead.pos < lead.n_in:
                    self.c["waits_for_leader"] += 1
                    if lead.rid not in seen:
                        self.c["lent_turns"] += 1
                        seen.add(lead.rid)
                        cands.append(lead)
                    continue
            seen.add(r.rid)
            cands.append(r)
        best = None
        if cands:
            target = (self.target_tail_s if (self.tail_guard_s > 0 and self.worst_ttft > self.tail_guard_s) else self.target_near_s if self.near_miss else
                      self.target_hi_s if self.min_slack < self.boost_slack_s else self.target_s) / self.speed
            for size, base in self.classes:
                cap = min(size, self.tcap - 8) - plan.n_dec
                if cap < 4 or (best is not None and base + dec > target):
                    continue
                fill, att = self._fill(cands, cap, target - base - dec)
                cols = sum(n for _, n in fill)
                est = base + dec + att
                if not cols:
                    continue
                # the class with the most prefill per second among those within the target; else the cheapest
                key = (est <= target * 1.02, cols / est if est <= target * 1.02 else -est)
                if best is None or key > best[0]:
                    best = (key, fill, est)
        if best is not None:
            _, fill, est = best
            for r, n in fill:
                self._claim(r, r.pos, n)
                plan.runs.append((r, r.slot, r.pos, n, r.prompt[r.pos:r.pos + n], 1 if r.pos + n == r.n_in else 0))
                plan.n_pre += n
            plan.est_s = est
        else:
            plan.est_s = self.c_small + dec
        if not plan.runs:
            # nothing fits or everything waits (streams sitting out with no prefill run composed): let the streams ride
            for r in self.running.values():
                if r.pos >= r.n_in:
                    plan.runs.append((r, r.slot, r.pos, 1, None, 1))
            if not plan.runs:
                return None
            plan.n_dec = len(plan.runs)
            plan.est_s = self.c_small + self.dec_cost([x[2] for x in plan.runs])
        plan.n_cols = plan.n_dec + plan.n_pre
        # pages for every position written in this pass
        for r, slot, pos0, n, toks, sel in plan.runs:
            last = (pos0 + n - 1) // PAGE
            while len(r.pages) <= last:
                page = self._take_page()
                self.ref[page] = 1
                r.pages.append(page)
                r.reserved -= 1
                self.committed -= 1
        return plan

    def _claim(self, r, pos, n):
        if not self.reuse:
            return
        chain = r.hashes()
        for k in range(pos // PAGE, min((pos + n - 1) // PAGE + 1, len(chain))):
            h = chain[k]
            if h not in self.cache and h not in self.pending:
                self.pending[h] = r
                r.claims.append(h)
                if h in self.evicted:
                    self.evicted.discard(h)
                    self.c["recomputed_pages"] += 1

    def launched(self, plan):
        """the pass is on its way: positions advance; complete prompt pages become visible to later passes"""
        for r, slot, pos0, n, toks, sel in plan.runs:
            if toks is None:
                self.c["decode_tokens"] += 1
            else:
                self.c["prefill_tokens"] += n
                if self.reuse:
                    chain = r.hashes()
                    for k in range(pos0 // PAGE, (pos0 + n) // PAGE):
                        if k < len(chain) and chain[k] not in self.cache and r.pages[k] not in self.page_hash:
                            parent = chain[k - 1] if k > 0 else None
                            if parent is None or parent in self.cache:   # else its prefix was dropped or evicted meanwhile
                                self.clock += 1
                                self.cache[chain[k]] = [r.pages[k], parent, 0, self.clock]
                                self.page_hash[r.pages[k]] = chain[k]
                                if parent is not None:
                                    self.cache[parent][2] += 1
                                self.c["cache_registered"] += 1
                        if k < len(chain) and self.pending.get(chain[k]) is r:
                            del self.pending[chain[k]]
            r.pos = pos0 + n

    def complete(self, plan, tokens, now):
        """tokens: one per selected run, in order. Returns the events of poll()."""
        events, j = [], 0
        for r, slot, pos0, n, toks, sel in plan.runs:
            if not sel:
                continue
            tok = int(tokens[j])
            j += 1
            r.produced += 1
            if r.t_first is None:
                r.t_first = now
            done = r.produced >= r.n_out
            events.append((r.rid, [tok], done))
            if done:
                self._finish(r)
        return events
