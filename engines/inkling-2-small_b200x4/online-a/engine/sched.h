// The serving loop's scheduler: admission, the page store with its prefix cache, and the composition of a pass.
// Plain C++ without allocation or library calls: the same code runs in a kernel on card 0 (engine/pe.cu, k_sched:
// the slot table, the page free list, the cache and the request ring live in device memory and nothing waits for the
// host) and on the host for tests (tools/sched_sim.cpp replays the trace on a virtual clock).
#pragma once
#ifdef __CUDACC__
#define SFN __host__ __device__ inline
#else
#define SFN inline
#endif

#define S_NSLOT 128          // sequences in flight (prompts in prefill and decoding sequences). At 8 req/s of the trace ~6.5
                             // requests/s x 190 tokens x 50 ms per pass are ~60 decoding sequences alone
#define S_MAXPG 68
#define S_PAGE 512
#define S_RING 1024
#define S_OUT (1 << 17)
#define S_MAXE 192            // sequences with rows in one pass
#define S_GRPW 8
#define S_EVK 4096
#define S_ADMIT 16            // requests admitted per step at most (the best of the waiting ones)
#define S_KOUT 8              // an output token counts as this many prompt rows in a waiting request's cost
#define S_LATE_SLOTS 24       // a request that cannot answer in time any more is admitted only while this many slots ...
#define S_LATE_PAGES 256      // ... and this many pages stay free for those that can
#define S_PM1 0.3              // promotion: a prompt projected to miss by less than this in shortest-first order ...
#define S_PM2 0.3              // ... goes first if it and those it overtakes then keep this much to spare (seconds)
#define S_LOAD_ND 32           // "many sequences decode": the pressure rule's margin is 2 s instead of 0.3 s
#define S_STRIPES 256         // stripes of the cache table, each with its least recently used evictable leaf
#define S_AQ 21            // rows of an attention group: 21 x 6 q heads are the 128 rows of a tensor-core tile
#define S_SWMAX 12            // expert swaps per rebalance
#define S_NL 48
#define S_NE 384

struct PHdr {               // what a pass is: written before every pass (by the host, or by the scheduler kernel)
  int T, tag, ngrp, nseq, nsnap, nlog, mode, nprol;     // mode: 0 none, 1 logits to `logits`, 2 argmax candidates
  int unitsL, unitsG, o_slot, o_pos, o_tok, o_back, o_grp, o_seq, o_snap, o_log, o_prol, o_pt, lo, hi, tok_dev, rg;
  int rgd, pad2;            // rg: row groups per block of the expert GEMMs (1 or 3); rgd: of the dense ones (1 or 2)
  int nswap, o_swap;        // expert swaps between cards before this pass: (layer, a, b, card a, index a, card b, index b)
  float* logits;            // [nlog][VOCAB_REAL] on card 0 (mode 1)
  const int* tokens;        // when tok_dev: [T] int32 on card 0 (rows' input tokens)
};

struct SReq { int rid, plen, n_out, off; };     // off: the prompt's first token in the arena
struct SShared {                                // host-mapped memory: the only thing host and device share
  volatile unsigned ring_head;                  // host: requests written
  volatile unsigned ring_tail;                  // device: requests admitted
  volatile unsigned out_head;                   // device: token records written
  volatile unsigned cmd_drop, ack_drop;         // host asks for drop_cache; device has done it
  volatile unsigned stop;                       // host: leave the loop
  volatile unsigned pass_no;                    // device: passes composed
  volatile unsigned active, free_pages, total_pages, cached_pages;
  volatile unsigned out_tail;                   // host: token records read
  volatile unsigned long long stats[32];
  SReq ring[S_RING];
  int out[S_OUT][2];                            // rid, token | last << 31
};

struct SSlot {
  int state;                  // 0 free, 1 prompt, 2 decoding
  int rid, plen, n_out, pos, nout, last_tok, off;
  int fresh, prol_page, npages, reserve, nkeys, cb_end;   // cb_end: prompt blocks [pos / 512, cb_end) are being computed by this slot
  double t_submit, t_first;
  int pages[S_MAXPG];
  unsigned long long keys[S_MAXPG];     // one per full prompt block: up to 67 for a prompt of 34,815 tokens
};
struct SNode { unsigned long long key; int page, parent, children, users; unsigned lru; int pad; };   // key 0: empty, 1: deleted
struct SEntry { int slot, n, kind, li, p0, r0; unsigned long long snapmask; };    // kind 0 decode row, 1 prompt chunk; r0: its first row of the pass

enum { SC_PASSES, SC_ROWS, SC_PREFILL, SC_DECODE, SC_HIT, SC_MISS, SC_LATEHIT, SC_EVICT, SC_REQ, SC_MAXACT, SC_INSERT, SC_DEDUP,
       SC_FIRST, SC_FIRST_OK, SC_RECOMP, SC_EMPTY, SC_SWAPS, SC_REBAL, SC_NOPAGE, SC_PROMO, SC_N };

struct Sched {
  // configuration
  int npages, tmax, umax, ncache;               // ncache: cache table entries, a power of two
  double base, dec, row, ctx, target_decode_ms, target_idle_ms, ttft_slo, gap_limit, min_pass_ms, host_ms;
  // arrays owned by whoever holds the scheduler (device memory on card 0, or the host's heap in tests)
  int* freep; int* page_ref; SNode* cache;
  int* ptoks;                 // [npages][S_PAGE] the tokens of the block a cached page holds (behind the cache table)
  float* eload;               // [48][384] recent pairs per expert and layer
  int* emap;                  // [48][384] where each expert's weights are: card | index on the card << 8
  int reb_seen, reb_every;
  // state
  SSlot slot[S_NSLOT];
  int nfree, cache_evictable, cache_nodes, reserved, lru_tick, pass_no, ne, rows_last, pad;
  double cost_scale, t_last, est_last;
  SEntry ent[S_MAXE];
  unsigned long long counters[32];
  // requests whose prompts are in the arena (the scheduler kernel copies them from the host's page-locked memory):
  // admission stops at head_limit when has_limit is set
  unsigned head_limit, copied; int has_limit;
  int soft_rows;              // rows of a pass unless a waiting prompt would miss its first-token time (then tmax)
  unsigned long long evk[S_EVK];      // keys of evicted blocks (direct-mapped): a miss on one of them is a recomputation
  // Eviction without a scan of the table: per stripe of ncache / S_STRIPES nodes, a node that is (or was) its least
  // recently used evictable leaf, with the lru value it had. The recorded value is never above the stripe's true
  // minimum: a node that becomes an evictable leaf lowers it (s_cand); a record whose node is used again, removed or
  // replaced stays until it is looked at (s_evict_one checks it and rescans that stripe only).
  unsigned st_lru[S_STRIPES]; int st_idx[S_STRIPES];
  int lazy_rows;              // the caller writes the per-row tables from ent[] (the scheduler kernel: all its threads)
  // Requests in the ring that are not admitted yet (ring entries [ring_tail, head): admission is by priority, not in
  // arrival order): when each was first seen (its first-token clock starts then), its state, its block keys.
  unsigned seen;                                  // ring entries below this have their arrival stamped
  double w_t[S_RING];
  unsigned char w_state[S_RING];                  // 0 waiting, 1 waiting with its keys made, 2 admitted
  unsigned long long w_keys[S_RING][S_MAXPG];
};

// ---------------------------------------------------------------------------------------------- hash chain of the prompt
SFN unsigned long long s_mix(unsigned long long h) {
  h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
  return h;
}
SFN unsigned long long s_block_key(unsigned long long prev, const int* tok) {
  unsigned long long h = prev ^ 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < S_PAGE; i++) h = (h ^ (unsigned)tok[i]) * 0x100000001b3ull + (h >> 29);
  h = s_mix(h);
#ifdef S_WEAK_HASH
  h %= S_WEAK_HASH;               // (tests: collisions on purpose)
#endif
  return h < 2 ? h + 2 : h;       // 0 and 1 mark empty and deleted table entries
}
// nodes of the cache table's allocation: the table, then the pages' tokens
SFN long s_cache_alloc_nodes(int ncache, int npages) { return (long)ncache + ((long)npages * S_PAGE * 4 + sizeof(SNode) - 1) / sizeof(SNode); }

// ---------------------------------------------------------------------------------------------- prefix cache (open addressing)
SFN int s_find(const Sched* S, unsigned long long key) {
  const unsigned mask = S->ncache - 1;
  for (unsigned i = (unsigned)s_mix(key) & mask, n = 0; n < (unsigned)S->ncache; i = (i + 1) & mask, n++) {
    const unsigned long long k = S->cache[i].key;
    if (k == key) return (int)i;
    if (k == 0) return -1;
  }
  return -1;
}
// The cached node of a block: found by its key and then verified. A key is a 64-bit hash of the prompt up to the
// block; equal keys do not prove equal tokens. A match needs (1) the block's 512 tokens equal to those the cached
// page was computed from and (2) the node's parent to be the node already matched for the block before (`prev`, -1
// for the first block): by induction the whole prefix is the same token for token.
SFN int s_match(const Sched* S, unsigned long long key, const int* tok, int prev) {
  const int i = s_find(S, key);
  if (i < 0 || S->cache[i].parent != prev) return -1;
  const int* pt = S->ptoks + (long)S->cache[i].page * S_PAGE;
  for (int k = 0; k < S_PAGE; k++) if (pt[k] != tok[k]) return -1;
  return i;
}
SFN int s_insert_node(Sched* S, unsigned long long key, int page, int parent) {
  const unsigned mask = S->ncache - 1;
  for (unsigned i = (unsigned)s_mix(key) & mask, n = 0; n < (unsigned)S->ncache; i = (i + 1) & mask, n++) {
    if (S->cache[i].key < 2) {
      SNode& nd = S->cache[i];
      nd.key = key; nd.page = page; nd.parent = parent; nd.children = 0; nd.users = 1; nd.lru = ++S->lru_tick;
      S->cache_nodes++;
      return (int)i;
    }
  }
  return -1;
}
SFN void s_cand(Sched* S, int i) {              // node i has become an evictable leaf (no user, no child)
  const int st = i / (S->ncache / S_STRIPES);
  if (S->st_idx[st] < 0 || S->cache[i].lru < S->st_lru[st]) { S->st_idx[st] = i; S->st_lru[st] = S->cache[i].lru; }
}
SFN void s_unref(Sched* S, int page) {
  if (--S->page_ref[page] == 0) S->freep[S->nfree++] = page;
}
SFN void s_remove_node(Sched* S, int i) {       // a node nobody uses
  SNode& nd = S->cache[i];
  if (nd.parent >= 0 && S->cache[nd.parent].key >= 2) {
    SNode& pa = S->cache[nd.parent];
    if (--pa.children == 0 && pa.users == 0) s_cand(S, nd.parent);
  }
  s_unref(S, nd.page);
  nd.key = 1;
  S->cache_nodes--;
  S->cache_evictable--;
  // A deleted mark is needed only while a later slot of its run is in use: with an empty slot after it, it and the
  // deleted marks before it are empty again. Without this the table runs out of empty slots after a few hours of
  // evictions and every lookup of a key that is not there walks the whole table.
  const unsigned mask = S->ncache - 1;
  if (S->cache[((unsigned)i + 1) & mask].key == 0)
    for (unsigned j = (unsigned)i, n = 0; S->cache[j].key == 1 && n < (unsigned)S->ncache; j = (j - 1) & mask, n++) S->cache[j].key = 0;
}
SFN bool s_evict_one(Sched* S) {                // the least recently used leaf that nobody uses
  const int sw = S->ncache / S_STRIPES;
  for (;;) {
    int bs = -1;
    for (int s = 0; s < S_STRIPES; s++)
      if (S->st_idx[s] >= 0 && (bs < 0 || S->st_lru[s] < S->st_lru[bs])) bs = s;
    if (bs < 0) return false;
    const int best = S->st_idx[bs];
    const SNode& nd = S->cache[best];
    const bool valid = nd.key >= 2 && nd.users == 0 && nd.children == 0 && nd.lru == S->st_lru[bs];
    unsigned long long key = 0;
    if (valid) { key = nd.key; s_remove_node(S, best); }       // (its parent may become a candidate, also of this stripe)
    // the stripe's record is used up or stale: its true minimum
    int b = -1;
    for (int i = bs * sw; i < (bs + 1) * sw; i++) {
      const SNode& x = S->cache[i];
      if (x.key >= 2 && x.users == 0 && x.children == 0 && (b < 0 || x.lru < S->cache[b].lru)) b = i;
    }
    S->st_idx[bs] = b;
    if (b >= 0) S->st_lru[bs] = S->cache[b].lru;
    if (valid) {
      S->evk[key & (S_EVK - 1)] = key;
      S->counters[SC_EVICT]++;
      return true;
    }
  }
}
SFN int s_alloc_page(Sched* S) {
  if (S->nfree == 0) s_evict_one(S);
  // (cannot happen: admission reserved the page. If the books were ever wrong it must not pass unseen: the count is
  // published and the host stops serving with an error)
  if (S->nfree == 0) { S->counters[SC_NOPAGE]++; return 0; }
  const int p = S->freep[--S->nfree];
  S->page_ref[p] = 1;
  return p;
}
SFN void s_drop_cache(Sched* S) {
  // nodes without users form subtrees closed downwards: removing them leaves the parents' counts right
  for (int i = 0; i < S->ncache; i++)
    if (S->cache[i].key >= 2 && S->cache[i].users == 0) s_remove_node(S, i);
  // an emptied table: forget the deletion marks, so that probes stay short
  if (S->cache_nodes == 0) for (int i = 0; i < S->ncache; i++) S->cache[i].key = 0;
}
SFN void s_use_node(Sched* S, int i) {          // one more running request shares this page
  SNode& nd = S->cache[i];
  if (nd.users == 0) S->cache_evictable--;
  nd.users++;
  nd.lru = ++S->lru_tick;
  S->page_ref[nd.page]++;
}

// ---------------------------------------------------------------------------------------------- sequences
SFN void s_release(Sched* S, int si) {
  SSlot& s = S->slot[si];
  for (int j = 0; j < s.npages; j++) {
    const int page = s.pages[j];
    if (j < s.nkeys) {
      const int nd = s_find(S, s.keys[j]);
      if (nd >= 0 && S->cache[nd].page == page) {
        if (--S->cache[nd].users == 0) { S->cache_evictable++; if (S->cache[nd].children == 0) s_cand(S, nd); }
      }
    }
    s_unref(S, page);
  }
  S->reserved -= s.reserve;
  s.reserve = 0; s.npages = 0; s.state = 0; s.cb_end = 0;
}
SFN void s_ensure_pages(Sched* S, SSlot& s, int upto) {      // pages for positions < upto
  while (s.npages * S_PAGE < upto) {
    s.reserve--; S->reserved--;
    s.pages[s.npages++] = s_alloc_page(S);
  }
}
SFN void s_cache_insert(Sched* S, SSlot& s, int j, const int* arena) {     // block j of s (a full prompt page whose snapshot was written)
  const unsigned long long key = s.keys[j];
  if (s_find(S, key) >= 0) return;              // (the same block, or another one with the same key: one of them is cached)
  int parent = -1;
  if (j) {
    parent = s_find(S, s.keys[j - 1]);
    // the chain below must be cached, and be the very page this request uses (not just a node with the same key)
    if (parent < 0 || S->cache[parent].page != s.pages[j - 1]) return;
  }
  if (s_insert_node(S, key, s.pages[j], parent) < 0) return;
  if (parent >= 0) S->cache[parent].children++;
  S->page_ref[s.pages[j]]++;
  S->counters[SC_INSERT]++;
  const int* tok = arena + s.off + j * S_PAGE;
  int* pt = S->ptoks + (long)s.pages[j] * S_PAGE;
  for (int k = 0; k < S_PAGE; k++) pt[k] = tok[k];
}
SFN void s_late_hits(Sched* S, SSlot& s, const int* arena) {      // at a page boundary: take over blocks cached since admission
  if (s.pos % S_PAGE || s.npages * S_PAGE != s.pos) return;
  const int usable = s.nkeys < (s.plen - 1) / S_PAGE ? s.nkeys : (s.plen - 1) / S_PAGE;
  int j = s.pos / S_PAGE, got = 0;
  // the node of the block before: the one whose page this request uses (then its prefix is that node's chain)
  int prev = -1;
  if (j) {
    prev = s_find(S, s.keys[j - 1]);
    if (prev < 0 || S->cache[prev].page != s.pages[j - 1]) return;
  }
  while (j < usable) {
    const int nd = s_match(S, s.keys[j], arena + s.off + j * S_PAGE, prev);
    if (nd < 0) break;
    s_use_node(S, nd);
    s.pages[s.npages++] = S->cache[nd].page;
    prev = nd;
    j++; got++;
  }
  if (got) {
    s.pos = j * S_PAGE;
    s.fresh = 1; s.prol_page = s.pages[s.npages - 1];
    s.reserve -= got; S->reserved -= got;
    S->counters[SC_LATEHIT] += got;
  }
}
// keys: the request's block keys (made once, when it is first tried); t_arr: when it was first seen; spare_slots /
// spare_pages: what must stay free after it (0 for a request that can still answer in time)
SFN bool s_admit(Sched* S, const SReq& rq, const int* arena, double t_arr, const unsigned long long* keys, int spare_slots, int spare_pages) {
  int si = -1, nfree_slots = 0;
  for (int i = 0; i < S_NSLOT; i++) if (S->slot[i].state == 0) { if (si < 0) si = i; nfree_slots++; }
  if (si < 0 || nfree_slots <= spare_slots) return false;
  SSlot& s = S->slot[si];
  const int nkeys = rq.plen / S_PAGE;
  for (int j = 0; j < nkeys; j++) s.keys[j] = keys[j];
  const int usable = nkeys < (rq.plen - 1) / S_PAGE ? nkeys : (rq.plen - 1) / S_PAGE;    // the last position is always computed
  int hit = 0, pinned = 0, nodes[S_MAXPG];
  while (hit < usable) {
    const int nd = s_match(S, s.keys[hit], arena + rq.off + hit * S_PAGE, hit ? nodes[hit - 1] : -1);
    if (nd < 0) break;
    if (S->cache[nd].users == 0) pinned++;
    nodes[hit++] = nd;
  }
  const int total_pages = (rq.plen + rq.n_out + S_PAGE - 1) / S_PAGE;
  const int need = total_pages - hit;
  if (S->nfree + S->cache_evictable - S->reserved - pinned < need + spare_pages) return false;
  s.state = 1; s.rid = rq.rid; s.plen = rq.plen; s.n_out = rq.n_out; s.off = rq.off;
  s.nkeys = nkeys; s.npages = 0; s.nout = 0; s.last_tok = -1; s.cb_end = 0;
  for (int j = 0; j < hit; j++) { s_use_node(S, nodes[j]); s.pages[s.npages++] = S->cache[nodes[j]].page; }
  s.pos = hit * S_PAGE;
  s.fresh = 1; s.prol_page = hit ? s.pages[hit - 1] : -1;
  s.reserve = need; S->reserved += need;
  s.t_submit = t_arr; s.t_first = 0.0;
  S->counters[SC_HIT] += hit; S->counters[SC_MISS] += nkeys - hit; S->counters[SC_REQ]++;
  for (int j = hit; j < nkeys; j++)       // blocks that were cached once and evicted: computed again (counted once)
    if (S->evk[s.keys[j] & (S_EVK - 1)] == s.keys[j]) { S->evk[s.keys[j] & (S_EVK - 1)] = 0; S->counters[SC_RECOMP]++; }
  return true;
}
SFN int s_units(int pos, int n) {               // attention units of the global layers (bound): 2 per group of S_AQ rows and page
  return 2 * ((n + S_AQ - 1) / S_AQ) * (((pos + n - 1) >> 9) + 1);
}

// ---------------------------------------------------------------------------------------------- expert placement
// The cards' expert load follows the traffic: from the pairs per expert of the pass that just ran (cnt [48][384]),
// every few prompt passes swap experts between the most and the least loaded card of a layer. Returns the number of
// swaps written to sw (7 ints each: layer, a, b, card a, index a, card b, index b); the cards exchange the weights
// before the next pass (k_swap1 / k_swap2). No result depends on where an expert's weights are.
// One layer's swaps: up to three, 6 ints each (a, b, card a, index a, card b, index b). A function of the layer's
// loads and placement only, so the scheduler kernel computes the 48 layers side by side.
SFN int s_reb_layer(const float* el, const int* em, int* o) {
  float load[4] = {0.f, 0.f, 0.f, 0.f};
  for (int e = 0; e < S_NE; e++) load[em[e] & 255] += (el[e] > 0.05f ? 1.0f : 0.0f) + el[e] / 128.0f;
  int used[8], nu = 0, ns = 0;     // experts already swapped in this round: the cards move a round's swaps side by side
  for (int it = 0; it < 3; it++) {
    int M = 0, m = 0;
    for (int q = 1; q < 4; q++) { if (load[q] > load[M]) M = q; if (load[q] < load[m]) m = q; }
    const float gap = load[M] - load[m], mean = (load[0] + load[1] + load[2] + load[3]) * 0.25f;
    if (gap < 0.15f * mean + 1.0f) break;
    int b = -1; float wb = 0.f;
    for (int e = 0; e < S_NE; e++)
      if ((em[e] & 255) == m) {
        bool u = false;
        for (int k = 0; k < nu; k++) u |= used[k] == e;
        const float w = (el[e] > 0.05f ? 1.0f : 0.0f) + el[e] / 128.0f;
        if (!u && (b < 0 || w < wb)) { b = e; wb = w; }
      }
    int a = -1; float best = 0.f, da = 0.f;
    for (int e = 0; e < S_NE; e++)
      if ((em[e] & 255) == M) {
        bool u = false;
        for (int k = 0; k < nu; k++) u |= used[k] == e;
        const float d = (el[e] > 0.05f ? 1.0f : 0.0f) + el[e] / 128.0f - wb;
        if (!u && d > 0.f && d < gap) { const float x = d > gap * 0.5f ? d - gap * 0.5f : gap * 0.5f - d; if (a < 0 || x < best) { a = e; best = x; da = d; } }
      }
    if (a < 0 || b < 0) break;
    int* q = o + 6 * ns++;
    q[0] = a; q[1] = b; q[2] = em[a] & 255; q[3] = em[a] >> 8; q[4] = em[b] & 255; q[5] = em[b] >> 8;
    used[nu++] = a; used[nu++] = b;
    load[M] -= da; load[m] += da;
  }
  return ns;
}
#define S_PROPW 20            // a layer's proposals: count, then 3 x 6 ints
SFN bool s_reb_due(const Sched* S) { return (S->reb_seen + 1) % S->reb_every == 0; }
// pre: null (the average and the layers' swaps are made here), or the layers' proposals [S_NL][S_PROPW] made by the
// caller after it has averaged the loads (only when s_reb_due)
SFN int s_observe(Sched* S, const int* cnt, int* sw, const int* pre) {
  if (!pre) for (int i = 0; i < S_NL * S_NE; i++) S->eload[i] = 0.75f * S->eload[i] + 0.25f * (float)cnt[i];
  if (++S->reb_seen % S->reb_every) return 0;
  int ns = 0;
  for (int i = 0; i < S_NL && ns < S_SWMAX; i++) {
    int* em = S->emap + i * S_NE;
    int loc[S_PROPW];
    const int* pr = pre ? pre + i * S_PROPW : loc;
    if (!pre) loc[0] = s_reb_layer(S->eload + i * S_NE, em, loc + 1);
    for (int k = 0; k < pr[0] && ns < S_SWMAX; k++) {
      const int* q = pr + 1 + 6 * k;
      int* o = sw + 7 * ns++;
      o[0] = i; o[1] = q[0]; o[2] = q[1]; o[3] = q[2]; o[4] = q[3]; o[5] = q[4]; o[6] = q[5];
      const int t = em[q[0]]; em[q[0]] = em[q[1]]; em[q[1]] = t;
    }
  }
  if (ns) { S->counters[SC_SWAPS] += ns; S->counters[SC_REBAL]++; }
  return ns;
}

// ---------------------------------------------------------------------------------------------- one step of the loop
// cand[c][li]: the cards' candidates of the pass that just ran (argmax key: logit bits << 32 | ~token).
// Writes the tables of the next pass into `tab` and its header into *hdr. Returns the number of rows.
// pre: see s_observe. With S->lazy_rows the per-row tables (slot, position, token, rows back) are left to the caller,
// who has ent[] (slot, first row, rows, kind, first position) to write them from.
SFN bool s_obs_due(const Sched* S, const int* cnt) { return cnt && S->eload && S->rows_last >= 256; }
SFN int s_step(Sched* S, SShared* sh, const int* arena, const unsigned long long* const* cand, const int* cnt, int* tab, PHdr* hdr, double now,
               const int* pre = 0) {
  // ---- (1) the pass that just ran: tokens out, prompt progress, cache inserts, finished requests leave
  if (S->ne && S->rows_last >= 128 && S->est_last > 0) {     // the cost model follows the measured time of prompt passes
    const double ms = (now - S->t_last) * 1e3;
    double sc = S->cost_scale * (1.0 + 0.2 * (ms / (S->est_last * S->cost_scale) - 1.0));
    S->cost_scale = sc < 0.25 ? 0.25 : (sc > 4.0 ? 4.0 : sc);
  }
  unsigned oh = sh->out_head;
  for (int e = 0; e < S->ne; e++) {
    const SEntry& en = S->ent[e];
    SSlot& s = S->slot[en.slot];
    int tok = -1;
    if (en.li >= 0) {
      unsigned long long best = 0;
      for (int cc = 0; cc < 4; cc++) { const unsigned long long v = cand[cc][en.li]; best = v > best ? v : best; }
      tok = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFull));
    }
    if (en.kind == 0) s.pos++;
    else {
      const int p0 = s.pos;
      s.pos += en.n;
      for (int j = p0 / S_PAGE; j < s.pos / S_PAGE; j++)
        if (j < s.nkeys && ((en.snapmask >> (j - p0 / S_PAGE)) & 1)) s_cache_insert(S, s, j, arena);
      if (en.li >= 0) {
        s.state = 2; s.t_first = now;
        S->counters[SC_FIRST]++;
        if (now - s.t_submit <= S->ttft_slo) S->counters[SC_FIRST_OK]++;
      }
    }
    if (tok >= 0) {
      s.last_tok = tok; s.nout++;
      const bool last = s.nout >= s.n_out;
      sh->out[oh & (S_OUT - 1)][0] = s.rid;
      sh->out[oh & (S_OUT - 1)][1] = tok | (last ? (int)0x80000000 : 0);
      oh++;
      if (last) s_release(S, en.slot);
    }
  }
  sh->out_head = oh;
  S->ne = 0;
  // ---- (2) drop_cache on request
  if (sh->cmd_drop != sh->ack_drop) { s_drop_cache(S); sh->ack_drop = sh->cmd_drop; }
  // ---- (3) admission, by priority among the requests waiting in the ring: those that can still answer in time
  // first, the cheapest of them first (prompt rows + S_KOUT per output token); the others in arrival order, and only
  // while slots and pages stay free for requests that can. (In arrival order, a full engine made every request wait
  // behind the one at the head: at 8 req/s all of them were late.)
  {
    unsigned tail = sh->ring_tail;
    const unsigned head = S->has_limit ? S->head_limit : sh->ring_head;
    for (; S->seen != head; S->seen++) { S->w_t[S->seen & (S_RING - 1)] = now; S->w_state[S->seen & (S_RING - 1)] = 0; }
    const double per_s0 = 1000.0 / (S->row * S->cost_scale);
    unsigned cand[S_ADMIT]; double ckey[S_ADMIT]; int nc = 0;
    int nwait = 0;
    for (unsigned i = tail; i != head; i++) {
      const unsigned x = i & (S_RING - 1);
      if (S->w_state[x] == 2) continue;
      nwait++;
      const SReq& rq = sh->ring[x];
      const bool late = (now - S->w_t[x]) + rq.plen / per_s0 + 0.03 > S->ttft_slo;
      // one key: on-time requests by cost, late ones after them in arrival order
      const double key = late ? 1e12 + (double)(i - tail) : (double)rq.plen + (double)S_KOUT * rq.n_out;
      if (nc == S_ADMIT && key >= ckey[nc - 1]) continue;
      int k = nc < S_ADMIT ? nc++ : nc - 1;
      for (; k > 0 && ckey[k - 1] > key; k--) { ckey[k] = ckey[k - 1]; cand[k] = cand[k - 1]; }
      ckey[k] = key; cand[k] = i;
    }
    int nprompt = 0;
    for (int i = 0; i < S_NSLOT; i++) nprompt += S->slot[i].state == 1;
    for (int k = 0; k < nc; k++) {
      const unsigned x = cand[k] & (S_RING - 1);
      const SReq& rq = sh->ring[x];
      if (S->w_state[x] == 0) {
        unsigned long long h = 0;
        for (int j = 0; j < rq.plen / S_PAGE; j++) { h = s_block_key(h, arena + rq.off + j * S_PAGE); S->w_keys[x][j] = h; }
        S->w_state[x] = 1;
      }
      const bool late = ckey[k] >= 1e12;
      // What a late request must leave free: nothing when no prompt is being served (the engine has room now) or when
      // the ring is filling up with late requests (nothing new could be seen); never more than an eighth of the store.
      const bool relax = nprompt == 0 || nwait > S_RING * 3 / 4;
      const int sp = S->npages / 8 < S_LATE_PAGES ? S->npages / 8 : S_LATE_PAGES;
      if (!s_admit(S, rq, arena, S->w_t[x], S->w_keys[x], late && !relax ? S_LATE_SLOTS : 0, late && !relax ? sp : 0)) {
        if (late) break;           // (an on-time request that does not fit: a cheaper or smaller one behind it may)
        continue;
      }
      S->w_state[x] = 2;
    }
    while (tail != head && S->w_state[tail & (S_RING - 1)] == 2) tail++;
    sh->ring_tail = tail;
  }
  // ---- (4) the next pass: every decoding sequence, then prompt rows while the pass stays within its time
  int T = 0, ng = 0, nseq = 0, nsnap = 0, nlog = 0, nprol = 0, uL = 0, uG = 0, nd = 0, nactive = 0;
  int* t_slot = tab; int* t_pos = tab + S->tmax; int* t_tok = tab + 2 * S->tmax; int* t_back = tab + 3 * S->tmax;
  int* t_grp = tab + 4 * S->tmax;                               // groups: at most tmax (single rows)
  int* t_seq = t_grp + S_GRPW * S->tmax; int* t_snap = t_seq + 4 * S_MAXE; int* t_log = t_snap + 4 * (S_MAXE + 64);
  int* t_prol = t_log + S_MAXE; int* t_pt = t_prol + 2 * S_MAXE; int* t_swap = t_pt + S_NSLOT * S_MAXPG;
  const int nswap = s_obs_due(S, cnt) ? s_observe(S, cnt, t_swap, pre) : 0;
  int ubudget = S->umax;
  double est = S->base;
  // rows of one sequence into the tables
#define S_ROWS(si_, p0_, n_, kind_, want_)                                                                        \
  do {                                                                                                            \
    SSlot& q_ = S->slot[si_];                                                                                     \
    const int r0_ = T;                                                                                            \
    if (S->lazy_rows) T += (n_);                                                                                  \
    else for (int x_ = 0; x_ < (n_); x_++) {                                                                      \
      t_slot[T] = (si_); t_pos[T] = (p0_) + x_; t_back[T] = x_ < 3 ? x_ : 3;                                      \
      t_tok[T] = (kind_) ? arena[q_.off + (p0_) + x_] : q_.last_tok;                                              \
      T++;                                                                                                        \
    }                                                                                                             \
    for (int g0_ = 0; g0_ < (n_); g0_ += S_AQ) {       /* attention groups: up to S_AQ consecutive rows */          \
      const int gn_ = (n_) - g0_ < S_AQ ? (n_) - g0_ : S_AQ, gp_ = (p0_) + g0_;                                   \
      const int pgl0_ = (gp_ - 1023 > 0 ? gp_ - 1023 : 0) >> 9;                                                   \
      const int npgl_ = ((gp_ + gn_ - 1) >> 9) - pgl0_ + 1, npgg_ = ((gp_ + gn_ - 1) >> 9) + 1;                   \
      int* G_ = t_grp + S_GRPW * ng++;                                                                            \
      G_[0] = r0_ + g0_; G_[1] = gn_; G_[2] = (si_); G_[3] = gp_;                                                 \
      G_[4] = pgl0_; G_[5] = npgl_; G_[6] = uL; G_[7] = uG;                                                       \
      uL += 2 * npgl_; uG += 2 * npgg_;                                                                           \
    }                                                                                                             \
    t_seq[4 * nseq] = (si_); t_seq[4 * nseq + 1] = (si_); t_seq[4 * nseq + 2] = r0_; t_seq[4 * nseq + 3] = (n_);  \
    nseq++;                                                                                                       \
    if (q_.fresh) { t_prol[2 * nprol] = (si_); t_prol[2 * nprol + 1] = q_.prol_page; nprol++; q_.fresh = 0; }     \
    SEntry& en_ = S->ent[S->ne++];                                                                                \
    en_.slot = (si_); en_.n = (n_); en_.kind = (kind_); en_.p0 = (p0_); en_.snapmask = 0; en_.li = -1;            \
    en_.r0 = r0_;                                                                                                 \
    if (want_) { en_.li = nlog; t_log[nlog++] = r0_ + (n_) - 1; }                                                 \
  } while (0)

  // a host that has not taken the tokens of the last passes: no rows until it has (the ring of token records must
  // not lap what was not read; 64 sequences fill it in ~26 s)
  const bool hold = (unsigned)(sh->out_head - sh->out_tail) > (unsigned)(S_OUT - 16384);
  double allow = S->target_idle_ms;
  for (int i = 0; i < S_NSLOT; i++) {
    SSlot& s = S->slot[i];
    if (s.state) nactive++;
    if (s.state != 2 || hold) continue;
    s_ensure_pages(S, s, s.pos + 1);
    S_ROWS(i, s.pos, 1, 0, 1);
    nd++;
    ubudget -= 2 * ((s.pos >> 9) + 1);
    const int gaps = s.n_out - 1;
    if (gaps > 0) {       // a request needs a mean gap under the limit: the pass may be as long as its credit allows
      const int left = s.n_out - s.nout - 1;
      const double a = 1e3 * (S->gap_limit * gaps - (now - s.t_first)) - left * S->target_decode_ms;
      allow = a < allow ? a : allow;
    }
  }
  est += S->dec * nd;
  double target = S->target_idle_ms;
  if (nd) { target = allow - S->host_ms; if (target < S->min_pass_ms) target = S->min_pass_ms; }
  target /= S->cost_scale;
  int rows_left = S->tmax - nd, npre = 0;
  const double per_s = 1000.0 / (S->row * S->cost_scale);
  // prompts: those that can still answer in time first, shortest remaining first; blocks cached meanwhile are taken over
  unsigned long long done_mask[(S_NSLOT + 63) / 64];     // slots already looked at in this pass
  for (int k = 0; k < (S_NSLOT + 63) / 64; k++) done_mask[k] = 0;
#define DM_GET(i_) ((done_mask[(i_) >> 6] >> ((i_) & 63)) & 1)
#define DM_SET(i_) (done_mask[(i_) >> 6] |= 1ull << ((i_) & 63))
  for (int i = 0; i < S_NSLOT; i++) if (S->slot[i].state == 1) s_late_hits(S, S->slot[i], arena);
  if (S->soft_rows < S->tmax) {
    // Passes of S->soft_rows rows keep the decoding requests' gaps short; the largest passes (S->tmax rows, a few
    // percent cheaper per row) are taken only while some waiting prompt, in the shortest-first order and at the
    // normal pass size, would get its first token too late.
    int cap = S->soft_rows;
    if (nd == 0) {
      // No sequence is decoding, and no prompt would answer at the end of a normal pass (the shortest has more rows
      // left than that): nobody waits for this pass to end. The largest pass: one base cost less per ~1000 rows (a
      // lone 32,000-token prompt answers after 9 passes instead of 13).
      int shortest = 1 << 30;
      for (int i = 0; i < S_NSLOT; i++)
        if (S->slot[i].state == 1 && S->slot[i].plen - S->slot[i].pos < shortest) shortest = S->slot[i].plen - S->slot[i].pos;
      if (shortest > S->soft_rows && shortest < (1 << 30)) cap = S->tmax;
    }
    const double rate = 1000.0 * S->soft_rows / ((S->base + (S->row + S->ctx * 12000.0) * S->soft_rows) * S->cost_scale);
    for (int i = 0; i < S_NSLOT && cap < S->tmax; i++) {
      const SSlot& a = S->slot[i];
      if (a.state != 1) continue;
      const int rem = a.plen - a.pos;
      const double left = a.t_submit + S->ttft_slo - now;
      if (left < rem / per_s + 0.03) continue;               // too late whatever the pass size
      long cum = 0;
      for (int j = 0; j < S_NSLOT; j++) {
        const SSlot& b = S->slot[j];
        if (b.state != 1) continue;
        const int rj = b.plen - b.pos;
        if ((b.t_submit + S->ttft_slo - now) < rj / per_s + 0.03) continue;
        if (rj < rem || (rj == rem && (b.t_submit < a.t_submit || (b.t_submit == a.t_submit && j <= i)))) cum += rj;
      }
      // (the margin: 0.3 s; 2 s while many sequences decode. The projection counts neither their rows nor the
      // prompts still to arrive, and under load it held too long: at 8 req/s the largest passes two seconds
      // earlier bring 3 of the 5 late requests of 780 in, and leave 2 req/s as it is)
      if (cum / rate + (nd >= S_LOAD_ND ? 2.0 : 0.30) > left) cap = S->tmax;
    }
    rows_left = cap - nd;
  }
  // Promotion. Shortest-first serves a later burst's short prompts before an earlier burst's long one, and the long
  // one can miss its first token although serving it first would have let everybody answer in time. One prompt per
  // step goes ahead of the shortest-first order if (1) in that order, at the largest pass size, it is projected to
  // miss (margin S_PM1) and (2) with it served first, it and every prompt it overtakes are still projected in time
  // (margin S_PM2). If that cannot be had, the order stays: the longest are the ones given up, which is what loses
  // the fewest requests. Among several candidates: the earliest deadline.
  int urgent = -1;
  {
    int oi[S_NSLOT], orem[S_NSLOT], n = 0; double oleft[S_NSLOT], ot[S_NSLOT];
    for (int i = 0; i < S_NSLOT; i++) {
      const SSlot& a = S->slot[i];
      if (a.state != 1) continue;
      const int rem = a.plen - a.pos;
      const double left = a.t_submit + S->ttft_slo - now;
      if (left < rem / per_s + 0.03) continue;               // too late already: not in this order
      int k = n++;                                           // insertion in the order of service: rows left, arrival, slot
      for (; k > 0 && (orem[k - 1] > rem || (orem[k - 1] == rem && ot[k - 1] > a.t_submit)); k--) {
        oi[k] = oi[k - 1]; orem[k] = orem[k - 1]; oleft[k] = oleft[k - 1]; ot[k] = ot[k - 1];
      }
      oi[k] = i; orem[k] = rem; oleft[k] = left; ot[k] = a.t_submit;
    }
    const double rmax = 1000.0 * S->tmax / ((S->base + S->dec * nd + (S->row + S->ctx * 12000.0) * S->tmax) * S->cost_scale);
    long cum = 0; int best = -1;
    for (int k = 0; k < n; k++) {
      cum += orem[k];
      if (cum / rmax + S_PM1 > oleft[k] && (best < 0 || oleft[k] < oleft[best])) best = k;
    }
    if (best >= 0) {
      bool ok = orem[best] / rmax + S_PM2 <= oleft[best];
      cum = orem[best];
      for (int k = 0; k < best && ok; k++) { cum += orem[k]; if (cum / rmax + S_PM2 > oleft[k]) ok = false; }
      if (ok) { urgent = oi[best]; S->counters[SC_PROMO]++; }
    }
  }
  for (int round = 0; round < S_NSLOT + 1 && !hold; round++) {
    int bi = -1; int blate = 0, brem = 0; double bt = 0;
    const bool fallback = round == S_NSLOT;           // nothing fitted the time: the most urgent prompt gets a minimal chunk
    if (fallback) { if (T > 0) break; for (int k = 0; k < (S_NSLOT + 63) / 64; k++) done_mask[k] = 0; }
    else if (rows_left <= 0 || est + S->row * 16 > target) continue;
    for (int i = 0; i < S_NSLOT; i++) {
      const SSlot& s = S->slot[i];
      if (s.state != 1 || DM_GET(i)) continue;
      const int rem = s.plen - s.pos;
      const int late = (s.t_submit + S->ttft_slo - now) < rem / per_s + 0.03;
      const int cls = late * 2 + (i != urgent);              // in time and promoted, in time, late
      if (bi < 0 || cls < blate || (cls == blate && (rem < brem || (rem == brem && s.t_submit < bt)))) { bi = i; blate = cls; brem = rem; bt = s.t_submit; }
    }
    if (bi < 0) break;
    DM_SET(bi);
    int si = bi, limit = 1 << 30;
    if (fallback) { target = 1e9; limit = 256; }
    {   // a prompt whose next block another request is computing waits for it; that request is served now
      const SSlot& s = S->slot[bi];
      const int j = s.pos / S_PAGE;
      if (s.pos % S_PAGE == 0 && j < s.nkeys) {
        int owner = -1;
        for (int o = 0; o < S_NSLOT && owner < 0; o++) {
          const SSlot& a = S->slot[o];
          if (o != bi && a.state == 1 && a.pos / S_PAGE <= j && j < a.cb_end && j < a.nkeys && a.keys[j] == s.keys[j]) owner = o;
        }
        if (owner >= 0) {
          S->counters[SC_DEDUP]++;
          bool sched = false;
          for (int e = 0; e < S->ne; e++) if (S->ent[e].slot == owner) sched = true;
          if (sched) continue;
          si = owner; limit = (S->slot[owner].pos / S_PAGE + 1) * S_PAGE - S->slot[owner].pos;
          DM_SET(owner);
        }
      }
    }
    SSlot& s = S->slot[si];
    const double cr = S->row + S->ctx * s.pos;
    int n = s.plen - s.pos;
    if (n > rows_left) n = rows_left;
    if (n > limit) n = limit;
    const int fit = (int)((target - est) / cr);
    if (n > fit) n = fit;
    while (n > 16 && s_units(s.pos, n) > ubudget) n = n / 2 > 16 ? n / 2 : 16;
    if (n <= 0 || s_units(s.pos, n) > ubudget) continue;
    s_ensure_pages(S, s, s.pos + n);
    const bool wants = s.pos + n == s.plen;
    const int p0 = s.pos;
    S_ROWS(si, p0, n, 1, wants);
    SEntry& en = S->ent[S->ne - 1];
    for (int j = p0 / S_PAGE; j < (p0 + n) / S_PAGE; j++)
      if (j < s.nkeys && s_find(S, s.keys[j]) < 0) {
        en.snapmask |= 1ull << (j - p0 / S_PAGE);
        t_snap[4 * nsnap] = s.pages[j]; t_snap[4 * nsnap + 1] = si; t_snap[4 * nsnap + 2] = T - n; t_snap[4 * nsnap + 3] = (j + 1) * S_PAGE - p0;
        nsnap++;
      }
    {
      int ce = (p0 + n - 1) / S_PAGE + 1;
      if (ce > s.nkeys) ce = s.nkeys;
      if (ce > s.cb_end) s.cb_end = ce;
    }
    est += cr * n;
    rows_left -= n; npre += n;
    ubudget -= s_units(p0, n);
  }
#undef S_ROWS
#undef DM_GET
#undef DM_SET
  // ---- (5) the page table and the header
  for (int i = 0; i < S_NSLOT; i++) {
    const SSlot& s = S->slot[i];
    if (s.state) for (int j = 0; j < s.npages; j++) t_pt[i * S_MAXPG + j] = s.pages[j];
  }
  hdr->T = T; hdr->tag = ++S->pass_no; hdr->ngrp = ng; hdr->nseq = nseq; hdr->nsnap = nsnap; hdr->nlog = nlog; hdr->mode = 2;
  hdr->nprol = nprol; hdr->unitsL = uL; hdr->unitsG = uG;
  hdr->o_slot = 0; hdr->o_pos = (int)(t_pos - tab); hdr->o_tok = (int)(t_tok - tab); hdr->o_back = (int)(t_back - tab);
  hdr->o_grp = (int)(t_grp - tab); hdr->o_seq = (int)(t_seq - tab); hdr->o_snap = (int)(t_snap - tab); hdr->o_log = (int)(t_log - tab);
  hdr->o_prol = (int)(t_prol - tab); hdr->o_pt = (int)(t_pt - tab);
  hdr->lo = 0; hdr->hi = 48; hdr->tok_dev = 0;
  hdr->rg = T <= 16 ? 1 : 3; hdr->rgd = T <= 1024 ? 1 : 2; hdr->pad2 = S->tmax;
  hdr->logits = 0; hdr->tokens = 0;
  hdr->nswap = nswap; hdr->o_swap = (int)(t_swap - tab);
  S->t_last = now; S->est_last = npre ? est : 0.0; S->rows_last = npre;
  S->counters[SC_PASSES]++; S->counters[SC_ROWS] += T; S->counters[SC_PREFILL] += npre; S->counters[SC_DECODE] += nd;
  if (T == 0) S->counters[SC_EMPTY]++;
  if ((unsigned long long)nactive > S->counters[SC_MAXACT]) S->counters[SC_MAXACT] = nactive;
  // what the host may read (idle, kv_pages, stats)
  sh->active = nactive; sh->free_pages = S->nfree; sh->total_pages = S->npages; sh->cached_pages = S->cache_nodes;
  for (int k = 0; k < SC_N; k++) sh->stats[k] = S->counters[k];
  sh->pass_no = S->pass_no;
  return T;
}

// table sizes (ints) for a given tmax, and the scheduler's defaults
SFN int s_tab_ints(int tmax) { return (4 + S_GRPW) * tmax + 4 * S_MAXE + 4 * (S_MAXE + 64) + S_MAXE + 2 * S_MAXE + S_NSLOT * S_MAXPG + 7 * S_SWMAX; }
SFN void s_init(Sched* S, int npages, int tmax, int umax, int ncache, int* freep, int* page_ref, SNode* cache) {
  S->npages = npages; S->tmax = tmax; S->umax = umax; S->ncache = ncache;
  S->base = 13.5; S->dec = 0.05; S->row = 0.0177; S->ctx = 0.20e-6;
  S->target_decode_ms = 80.0; S->target_idle_ms = 140.0; S->ttft_slo = 5.0; S->gap_limit = 0.090; S->min_pass_ms = 45.0; S->host_ms = 3.0;
  S->freep = freep; S->page_ref = page_ref; S->cache = cache;
  S->ptoks = reinterpret_cast<int*>(cache + ncache);        // (the caller allocates s_cache_alloc_nodes(ncache, npages) nodes)
  S->eload = 0; S->emap = 0; S->reb_seen = 0; S->reb_every = 6;
  for (int i = 0; i < S_NSLOT; i++) { S->slot[i].state = 0; S->slot[i].npages = 0; S->slot[i].reserve = 0; S->slot[i].cb_end = 0; }
  S->nfree = npages;
  for (int i = 0; i < npages; i++) { freep[i] = npages - 1 - i; page_ref[i] = 0; }
  for (int i = 0; i < ncache; i++) cache[i].key = 0;
  S->cache_evictable = 0; S->cache_nodes = 0; S->reserved = 0; S->lru_tick = 0; S->pass_no = 1 << 20; S->ne = 0; S->rows_last = 0;
  S->cost_scale = 1.0; S->t_last = 0.0; S->est_last = 0.0;
  for (int k = 0; k < 32; k++) S->counters[k] = 0;
  S->head_limit = 0; S->copied = 0; S->has_limit = 0; S->soft_rows = tmax;
  for (int k = 0; k < S_EVK; k++) S->evk[k] = 0;
  for (int k = 0; k < S_STRIPES; k++) { S->st_idx[k] = -1; S->st_lru[k] = 0; }
  S->lazy_rows = 0; S->seen = 0;
}
