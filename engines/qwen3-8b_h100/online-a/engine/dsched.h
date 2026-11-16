// The scheduler of the engine as it runs ON THE DEVICE: request ring, slot table, KV page free list with reference
// counts, prefix cache (hash table + LRU), admission, service order, and the plan of the next pass.
// One function, ds_step(), executed by one thread of the scheduler kernel at every pass boundary. The same
// source compiles for the host (tests/dsched_host.cpp), where tests/sim.py checks it decision for decision
// against the Python reference engine/sched.py. Policy and cost model: see sched.py.
#pragma once
#include <stdint.h>

#ifdef __CUDACC__
#define DS_FN __host__ __device__ __forceinline__
#else
#define DS_FN static inline
#endif

#define DS_PAGE 512
#define DS_NSLOT 64
#define DS_MAXPG 68
#ifndef DS_RING
#define DS_RING 4096           // requests alive (queued or running) at any time, at most
#endif
#define DS_QMAX 192            // queued requests looked at per pass (in arrival order)
#define DS_CAND (DS_NSLOT + DS_QMAX)
#define DS_TAB 8192            // prefix hash table (cached pages + pages being computed)
#define DS_MAXPAGES 1024
#define DS_NCLASS 8
#define DS_SELCAP 256
#define DS_NCNT 24

// plan layout (ints), shared with the pass kernels (eng.cu defines the same offsets from these)
#define DS_TCAP 2048
#define DS_MAXU 512
#define DS_MAXBT 32768
#define DS_PL_SLOT 0
#define DS_PL_POS (DS_TCAP)
#define DS_PL_TOK (2 * DS_TCAP)
#define DS_PL_UNIT (3 * DS_TCAP)
#define DS_PL_BT (DS_PL_UNIT + 4 * DS_MAXU)
#define DS_PL_FOLD (DS_PL_BT + DS_MAXBT)
#define DS_PL_SEL (DS_PL_FOLD + DS_MAXU)
#define DS_PL_PT (DS_PL_SEL + 2 * DS_SELCAP)
#define DS_PL_CU (DS_PL_PT + DS_NSLOT * DS_MAXPG)
#define DS_PL_HDR (DS_PL_CU + DS_TCAP)     // [16]: columns, tasks of two units, tasks of one unit, units to fold, selected columns
#define DS_PL_INTS (DS_PL_HDR + 16)
#define DS_LASTBIT 0x10000
// decode rows: a block alone on its task takes this part of c_dseg per segment (1.9 us against the 2.8 us that HBM
// allows per block when all blocks read)
#define DS_DLAT 0.68
#define DS_ARRIVAL_Q 4.0      // the deadline order looks at arrival times in units of 1 / this (seconds)
// deadline estimates: fixed cost of a pass (weights streamed once), and the least share of the GPU assumed for prompts
#define DS_CFIX_S 0.006
#define DS_SHARE_MIN 0.15

enum { DS_C_ADMITTED, DS_C_FINISHED, DS_C_HIT_PAGES, DS_C_REGISTERED, DS_C_EVICTIONS, DS_C_RECOMPUTED, DS_C_WAITS, DS_C_LENT,
       DS_C_DEMOTED, DS_C_PREFILL_TOK, DS_C_DECODE_TOK, DS_C_DROPPED, DS_C_PASSES, DS_C_QUEUE_MAX, DS_C_MIN_FREE, DS_C_ERRORS,
       DS_C_COLUMNS, DS_C_RUN_MAX, DS_C_STEPS };

struct DsParams {
  double target_s, ttft_s, margin_s, share, c_small, c_seg, c_dseg, c_col, c_pair, speed;
  double boost_slack_s, target_hi_s;   // passes aim at target_hi_s while an on-time prompt has less than boost_slack_s to spare (0: never)
  double tail_guard_s, target_tail_s;  // passes aim at target_tail_s while some prompt is heading for a first token later than tail_guard_s (0: never)
  double near_s;                       // ... and while some prompt is heading for one within near_s after ttft_s (a near miss; 0: never)
  double age_s;                        // prompts waiting longer than this come first, oldest first (0: never)
  double target_near_s;                // the pass target in a near miss (when the tail guard does not hold)
  double class_base[DS_NCLASS];
  int class_size[DS_NCLASS];
  int n_blocks, reuse, sticky, fifo, min_chunk, tcap, selcap, n_pages, n_slots, ss_shift, partcap, pad;
};
struct DsReqHost { int n_in, n_out, prompt_off; float t_arrive; };   // written by the host, one per ring entry
struct DsReq {
  int state;                 // 0 unseen, 1 queued, 2 running, 3 done
  int slot, late, was_late, max_reuse, need_pages, nchain, hit_k, hit_idle, hit_epoch;
  double cost;               // estimated time to its first token from now, alone (this step's deadline ordering)
  uint64_t chain[64];        // hash of the prompt up to and including each complete page
};
struct DsSlot { int req, pos, produced, npages, reserved, n_in, n_out, prompt_off; double t_arrive; };
struct DsEnt { uint64_t key, parent; int state, page, owner, children; };   // state: 0 empty, 1 cached, 2 being computed by `owner`
struct DsResult {            // what the host reads after a pass
  int n_sel, n_cols, n_free, n_running, n_queued, n_cached, head, step;
  double est_s;
  int req[DS_SELCAP];        // ring index of the request of each token
  int fin[DS_SELCAP];        // 1: its last token
  int tok[DS_SELCAP];        // the tokens (written by the pass)
};
struct DsState {
  DsParams P;
  int head, seen;            // ring: first entry not done; entries seen (hashed) so far
  int n_free, committed, n_reclaim, epoch, n_free_slots, n_run, lru_head, lru_tail, n_fin, n_freed;
  int free_pages[DS_MAXPAGES], ref[DS_MAXPAGES], page_ent[DS_MAXPAGES], lru_prev[DS_MAXPAGES], lru_next[DS_MAXPAGES];
  int freed[DS_MAXPAGES];    // pages that went back to the free list in this step (tests: poison)
  int free_slots[DS_NSLOT], run_order[DS_NSLOT], fin_slots[DS_SELCAP];
  DsSlot slot[DS_NSLOT];
  long long cnt[DS_NCNT];
  unsigned evicted_bits[32768];  // statistics only: prefixes evicted (to count recomputation)
  // scratch of one step
  int cand[DS_CAND], ord[DS_CAND], heap[DS_CAND], c2[DS_CAND], fill_req[DS_CAND], fill_n[DS_CAND], best_req[DS_CAND], best_n[DS_CAND];
  double cost[DS_CAND];
  unsigned char is_late[DS_CAND], seen2[DS_NSLOT];
  DsEnt tab[DS_TAB];
  DsReq req[DS_RING];
};

DS_FN uint64_t ds_mix(uint64_t x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; x ^= x >> 31;
  return x;
}
// hash of one page of prompt tokens (any thread), and the chain over the pages of a prompt
DS_FN uint64_t ds_hash_page(const int* tok) {
  uint64_t h = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < DS_PAGE; i++) { h ^= (uint32_t)tok[i]; h *= 0x100000001b3ULL; h ^= h >> 29; }
  return h;
}
DS_FN void ds_chain(uint64_t* chain, int n) {
  uint64_t c = 0x243f6a8885a308d3ULL;
  for (int k = 0; k < n; k++) { c = ds_mix(c ^ chain[k]); if (c == 0) c = 1; chain[k] = c; }
}

// ---------------------------------------------------------------- prefix table (linear probing, backward-shift deletion)
DS_FN int ds_find(const DsState& S, uint64_t key) {
  int i = (int)(key & (DS_TAB - 1));
  while (S.tab[i].state != 0) {
    if (S.tab[i].key == key) return i;
    i = (i + 1) & (DS_TAB - 1);
  }
  return -1;
}
DS_FN int ds_insert(DsState& S, uint64_t key) {
  int i = (int)(key & (DS_TAB - 1));
  while (S.tab[i].state != 0) i = (i + 1) & (DS_TAB - 1);
  S.tab[i].key = key;
  return i;
}
DS_FN void ds_erase(DsState& S, int i) {
  int j = i;
  for (;;) {
    j = (j + 1) & (DS_TAB - 1);
    if (S.tab[j].state == 0) break;
    const int home = (int)(S.tab[j].key & (DS_TAB - 1));
    // can entry j move to the hole i? yes unless its home lies cyclically in (i, j]
    const bool stay = i <= j ? (home > i && home <= j) : (home > i || home <= j);
    if (!stay) {
      S.tab[i] = S.tab[j];
      if (S.tab[i].state == 1) S.page_ent[S.tab[i].page] = i;
      i = j;
    }
  }
  S.tab[i].state = 0;
}
DS_FN bool ds_cached(const DsState& S, uint64_t key, int& idx) { idx = ds_find(S, key); return idx >= 0 && S.tab[idx].state == 1; }

// ---------------------------------------------------------------- LRU over cached pages (head = least recently used)
DS_FN void ds_lru_remove(DsState& S, int page) {
  const int p = S.lru_prev[page], n = S.lru_next[page];
  if (p >= 0) S.lru_next[p] = n; else S.lru_head = n;
  if (n >= 0) S.lru_prev[n] = p; else S.lru_tail = p;
}
DS_FN void ds_lru_append(DsState& S, int page) {
  S.lru_prev[page] = S.lru_tail; S.lru_next[page] = -1;
  if (S.lru_tail >= 0) S.lru_next[S.lru_tail] = page; else S.lru_head = page;
  S.lru_tail = page;
}

// ---------------------------------------------------------------- pages
DS_FN void ds_release(DsState& S, int page) {
  S.free_pages[S.n_free++] = page;
  if (S.n_freed < DS_MAXPAGES) S.freed[S.n_freed++] = page;
}
DS_FN void ds_uncache(DsState& S, int idx, bool free_it) {
  const int page = S.tab[idx].page;
  const uint64_t parent = S.tab[idx].parent;
  if (parent) { int pi; if (ds_cached(S, parent, pi) && S.tab[pi].children > 0) S.tab[pi].children--; }
  ds_lru_remove(S, page);
  S.page_ent[page] = -1;
  if (S.ref[page] == 0) S.n_reclaim--;
  ds_erase(S, ds_find(S, S.tab[idx].key));
  S.epoch++;
  if (free_it) ds_release(S, page);
}
// Evict the least recently used cached page nobody uses: a leaf of the prefix tree if there is one. If every unused
// page still has a cached child (a request can hold a child page without its parent: it computed the parent
// itself while another request registered it), the least recently used unused page goes anyway: its descendants
// stay valid for their users and merely cannot be reached by new requests any more.
DS_FN bool ds_evict_one(DsState& S) {
  for (int pass = 0; pass < 2; pass++) {
    for (int page = S.lru_head; page >= 0; page = S.lru_next[page]) {
      const int idx = S.page_ent[page];
      if ((pass == 1 || S.tab[idx].children == 0) && S.ref[page] == 0) {
        const uint64_t key = S.tab[idx].key;
        S.evicted_bits[(key >> 5) & 32767] |= 1u << (key & 31);
        ds_uncache(S, idx, true);
        S.cnt[DS_C_EVICTIONS]++;
        return true;
      }
    }
  }
  return false;
}
DS_FN int ds_take_page(DsState& S) {
  if (S.n_free == 0 && !ds_evict_one(S)) { S.cnt[DS_C_ERRORS]++; return 0; }
  if (S.n_free - 1 < S.cnt[DS_C_MIN_FREE]) S.cnt[DS_C_MIN_FREE] = S.n_free - 1;
  return S.free_pages[--S.n_free];
}
// forget every retained prefix; a page still used by a running request becomes private to it
DS_FN void ds_drop_cache(DsState& S) {
  for (int i = 0; i < DS_TAB;) {
    if (S.tab[i].state == 1) {
      const int page = S.tab[i].page;
      ds_lru_remove(S, page);
      S.page_ent[page] = -1;
      if (S.ref[page] == 0) ds_release(S, page);
      S.cnt[DS_C_DROPPED]++;
      ds_erase(S, i);          // may move a later entry into i: look at i again
    } else i++;
  }
  S.n_reclaim = 0;
  S.epoch++;
  for (int i = 0; i < 32768; i++) S.evicted_bits[i] = 0;
}

// ---------------------------------------------------------------- requests
// leading pages of a request that are cached now (and how many of them nobody uses)
DS_FN void ds_hits(DsState& S, DsReq& r) {
  if (r.hit_epoch == S.epoch) return;
  int k = 0, idle = 0;
  if (S.P.reuse) {
    while (k < r.max_reuse) {
      int idx;
      if (!ds_cached(S, r.chain[k], idx)) break;
      idle += S.ref[S.tab[idx].page] == 0;
      k++;
    }
  }
  r.hit_k = k; r.hit_idle = idle; r.hit_epoch = S.epoch;
}
// take over every cached page at the slot's position (page boundary)
DS_FN void ds_adopt(DsState& S, int* pt, int s) {
  if (!S.P.reuse) return;
  DsSlot& sl = S.slot[s];
  DsReq& r = S.req[sl.req];
  while ((sl.pos & (DS_PAGE - 1)) == 0 && (sl.pos >> 9) < r.max_reuse && sl.npages == (sl.pos >> 9)) {
    int idx;
    if (!ds_cached(S, r.chain[sl.pos >> 9], idx)) break;
    const int page = S.tab[idx].page;
    ds_lru_remove(S, page); ds_lru_append(S, page);
    if (S.ref[page] == 0) { S.n_reclaim--; S.epoch++; }   // the idle-hit counts of other requests change
    S.ref[page]++;
    pt[s * DS_MAXPG + sl.npages++] = page;
    sl.pos += DS_PAGE;
    S.cnt[DS_C_HIT_PAGES]++;
    if (sl.reserved > 0) { sl.reserved--; S.committed--; }
  }
}
DS_FN bool ds_try_admit(DsState& S, const DsReqHost* rh, int* pt, int ri) {
  if (S.n_free_slots == 0) return false;
  DsReq& r = S.req[ri];
  ds_hits(S, r);
  const int need = r.need_pages - r.hit_k;
  if (need > S.n_free + S.n_reclaim - r.hit_idle - S.committed) return false;
  const int s = S.free_slots[--S.n_free_slots];
  DsSlot& sl = S.slot[s];
  sl.req = ri; sl.pos = 0; sl.produced = 0; sl.npages = 0; sl.reserved = r.need_pages;
  sl.n_in = rh[ri].n_in; sl.n_out = rh[ri].n_out; sl.prompt_off = rh[ri].prompt_off; sl.t_arrive = rh[ri].t_arrive;
  S.committed += r.need_pages;
  r.slot = s; r.state = 2;
  S.run_order[S.n_run++] = s;
  ds_adopt(S, pt, s);
  S.cnt[DS_C_ADMITTED]++;
  return true;
}
DS_FN void ds_finish(DsState& S, int* pt, int s) {
  DsSlot& sl = S.slot[s];
  DsReq& r = S.req[sl.req];
  for (int k = 0; k < sl.npages; k++) {
    const int page = pt[s * DS_MAXPG + k];
    if (S.page_ent[page] >= 0) { if (--S.ref[page] == 0) { S.n_reclaim++; S.epoch++; } }
    else { S.ref[page] = 0; ds_release(S, page); }
  }
  for (int k = 0; k < r.nchain; k++) {
    const int idx = ds_find(S, r.chain[k]);
    if (idx >= 0 && S.tab[idx].state == 2 && S.tab[idx].owner == s) ds_erase(S, idx);
  }
  S.committed -= sl.reserved;
  sl.reserved = 0; sl.npages = 0;
  int j = 0;
  for (int i = 0; i < S.n_run; i++) if (S.run_order[i] != s) S.run_order[j++] = S.run_order[i];
  S.n_run = j;
  S.free_slots[S.n_free_slots++] = s;
  r.state = 3; r.slot = -1;
  S.cnt[DS_C_FINISHED]++;
}

// ---------------------------------------------------------------- cost model
DS_FN double ds_cost_prefill(const DsParams& P, int pos, int n) { return n * P.c_col + P.c_pair * n * (pos + 0.5 * n); }
DS_FN double ds_att_cost(const DsParams& P, double mx, double sm) {
  if (sm == 0) return 0.0;
  const double b = sm / P.n_blocks;
  return P.c_seg * (mx > b ? mx : b);
}
DS_FN void ds_run_segs(int pos, int n, double& mx, double& tot) {
  const int m = (pos + n - 1) / 128 + 1;
  mx = m < 32 ? m : 32;
  tot = ((n + 31) / 32) * 8 * ((pos + 0.5 * n) / 128 + 1);
}

// Greedy fill of at most `cap` prefill columns in candidate order within an attention budget (seconds).
DS_FN int ds_fill(DsState& S, int ncand, int cap, double budget, int* out_slot, int* out_n, double& att) {
  const DsParams& P = S.P;
  int nout = 0, cols = 0;
  double mx = 0, sm = 0;
  for (int ci = 0; ci < ncand; ci++) {
    const DsSlot& sl = S.slot[S.c2[ci]];
    const int room = cap - cols;
    if (room < 4 || nout >= P.selcap - 70) break;
    const int left = sl.n_in - sl.pos;
    int hi = left < room ? left : room;
    if (hi < left) hi -= hi % 4;
    int n;
    double m, t;
    ds_run_segs(sl.pos, hi, m, t);
    if (ds_att_cost(P, mx > m ? mx : m, sm + t) <= budget) n = hi;
    else {
      int lo = 0;
      while (hi - lo > 4) {
        const int mid = (lo + hi) / 8 * 4;
        if (mid <= lo) break;
        ds_run_segs(sl.pos, mid, m, t);
        if (ds_att_cost(P, mx > m ? mx : m, sm + t) <= budget) lo = mid; else hi = mid;
      }
      n = lo;
      const int mc = P.min_chunk < left ? P.min_chunk : left;
      if (n < mc) {
        if (nout) continue;
        int alt = room < left ? room - room % 4 : left;
        n = mc < alt ? mc : alt;
      }
    }
    if (n <= 0) continue;
    ds_run_segs(sl.pos, n, m, t);
    if (m > mx) mx = m;
    sm += t; cols += n;
    out_slot[nout] = S.c2[ci]; out_n[nout] = n; nout++;
  }
  att = ds_att_cost(P, mx, sm);
  return nout;
}

DS_FN void ds_init(DsState& S, const DsParams& P) {
  S.P = P;
  S.head = 0; S.seen = 0;
  S.n_free = P.n_pages; S.committed = 0; S.n_reclaim = 0; S.epoch = 1; S.n_run = 0; S.lru_head = -1; S.lru_tail = -1; S.n_fin = 0; S.n_freed = 0;
  for (int i = 0; i < P.n_pages; i++) { S.free_pages[i] = P.n_pages - 1 - i; S.ref[i] = 0; S.page_ent[i] = -1; }
  S.n_free_slots = P.n_slots;
  for (int i = 0; i < P.n_slots; i++) S.free_slots[i] = P.n_slots - 1 - i;
  for (int i = 0; i < DS_NCNT; i++) S.cnt[i] = 0;
  S.cnt[DS_C_MIN_FREE] = P.n_pages;
  for (int i = 0; i < 32768; i++) S.evicted_bits[i] = 0;
  for (int i = 0; i < DS_TAB; i++) S.tab[i].state = 0;
  for (int i = 0; i < DS_RING; i++) S.req[i].state = 0;
}

// A request the scheduler sees for the first time (its chain hashes are in place)
DS_FN void ds_new_req(DsState& S, const DsReqHost* rh, int ri) {
  DsReq& r = S.req[ri];
  r.state = 1; r.slot = -1; r.late = 0; r.was_late = 0;
  r.max_reuse = (rh[ri].n_in - 1) / DS_PAGE;
  r.need_pages = (rh[ri].n_in + rh[ri].n_out - 1 + DS_PAGE - 1) / DS_PAGE;
  r.nchain = rh[ri].n_in / DS_PAGE;
  r.hit_k = 0; r.hit_idle = 0; r.hit_epoch = 0;
}

// ---------------------------------------------------------------- one step = one pass boundary
// tail: number of requests the host has put into the ring so far (absolute count; entry index = count % DS_RING).
// Writes the plan (ints, layout DS_PL_*) and the result header; returns the number of columns of the pass.
// cmd: 0 normal; 1 drop the prefix cache and plan nothing.
DS_FN int ds_step(DsState& S, const DsReqHost* rh, int* plan, DsResult& R, double now, int tail, int cmd) {
  const DsParams& P = S.P;
  int* pt = plan + DS_PL_PT;
  S.n_freed = 0;
  // ---- the requests whose last token was produced by the previous pass leave now
  for (int i = 0; i < S.n_fin; i++) ds_finish(S, pt, S.fin_slots[i]);
  S.n_fin = 0;
  while (S.head < S.seen && S.req[S.head % DS_RING].state == 3) { S.req[S.head % DS_RING].state = 0; S.head++; }
  // ---- new requests
  for (; S.seen < tail; S.seen++) ds_new_req(S, rh, S.seen % DS_RING);
  if (cmd == 1) ds_drop_cache(S);

  // ---- service order of the prompts: running ones that are still prefilling + the queue
  int nc = 0, nq = 0;
  for (int i = 0; i < S.n_run; i++) { const DsSlot& sl = S.slot[S.run_order[i]]; if (sl.pos < sl.n_in) S.cand[nc++] = sl.req; }
  for (int q = S.head; q < S.seen && nc < DS_CAND; q++) if (S.req[q % DS_RING].state == 1) { if (nq < DS_QMAX) S.cand[nc++] = q % DS_RING; nq++; }
  if (nq > S.cnt[DS_C_QUEUE_MAX]) S.cnt[DS_C_QUEUE_MAX] = nq;
  // The part of the GPU the prompts get: what the pass target leaves after the fixed cost of a pass and the
  // streams that decode now (at a 27 ms target with twenty long streams that is a fifth, not P.share).
  double share = P.share;
  {
    double dmax0 = 0, dsum0 = 0;
    int nd0 = 0;
    for (int i = 0; i < S.n_run; i++) {
      const DsSlot& sl = S.slot[S.run_order[i]];
      if (sl.pos >= sl.n_in) { const double segs = sl.pos / 128 + 1; if (segs > dmax0) dmax0 = segs; dsum0 += segs; nd0++; }
    }
    double dec0 = 0;
    if (nd0) { const double a = DS_DLAT * (dmax0 < 32 ? dmax0 : 32), b = dsum0 * 8 / P.n_blocks; dec0 = P.c_dseg * (a > b ? a : b); }
    const double tgt = P.target_s / P.speed;
    double sd = (tgt - DS_CFIX_S - dec0) / tgt;
    if (sd < DS_SHARE_MIN) sd = DS_SHARE_MIN;
    if (sd < share) share = sd;
  }
  // estimated time to the first token of each prompt, alone
  for (int i = 0; i < nc; i++) {
    const int ri = S.cand[i];
    DsReq& r = S.req[ri];
    r.cost = -1.0;
    if (P.fifo || (P.sticky && r.late)) continue;
    int pos = 0;
    if (r.slot < 0) { ds_hits(S, r); pos = r.hit_k * DS_PAGE; }
    else pos = S.slot[r.slot].pos;
    r.cost = ds_cost_prefill(P, pos, rh[ri].n_in - pos) * P.speed / share;
  }
  // sort by (arrival in quarters of a second, estimated time, ring order): insertion sort, the list is nearly sorted.
  // Prompts that arrive together (a burst; the driver submits them microseconds apart): the shortest first.
  // Ring order = distance from head.
#define DS_SEQ(ri) (((ri) - S.head % DS_RING + DS_RING) % DS_RING)
#define DS_KEY(ri) (S.req[ri].cost < 0 ? 0.0 : S.req[ri].cost)
#define DS_ARR(ri) ((long long)((double)rh[ri].t_arrive * DS_ARRIVAL_Q))
#define DS_BEFORE(x, y) (DS_ARR(x) < DS_ARR(y) || (DS_ARR(x) == DS_ARR(y) && (DS_KEY(x) < DS_KEY(y) || (DS_KEY(x) == DS_KEY(y) && DS_SEQ(x) < DS_SEQ(y)))))
  for (int i = 1; i < nc; i++) {
    const int x = S.cand[i];
    int j = i - 1;
    while (j >= 0 && DS_BEFORE(x, S.cand[j])) { S.cand[j + 1] = S.cand[j]; j--; }
    S.cand[j + 1] = x;
  }
#undef DS_BEFORE
#undef DS_KEY
#undef DS_ARR
  int no = 0;   // S.ord: the order
  double min_slack = 1e9;   // least time to spare (by the estimates) of the prompts that can still make their first token
  if (P.fifo) { for (int i = 0; i < nc; i++) S.ord[no++] = S.cand[i]; }
  else {
    // Moore-Hodgson: add each to the on-time set in deadline order; when the set cannot make the last deadline, drop its largest job.
    int nh = 0;
    double t = now;
    for (int i = 0; i < nc; i++) {
      const int ri = S.cand[i];
      DsReq& r = S.req[ri];
      S.is_late[i] = 0; S.cost[i] = -1.0;
      if (P.sticky && r.late) { S.is_late[i] = 1; continue; }
      const double cost = r.cost;
      S.cost[i] = cost;
      t += cost;
      // max-heap on (cost, then smaller ring order first: Python's heap of (-cost, rid) pops the smallest rid among equals)
      int h = nh++;
      S.heap[h] = i;
      while (h > 0) {
        const int par = (h - 1) / 2, a = S.heap[h], b = S.heap[par];
        if (S.cost[a] > S.cost[b] || (S.cost[a] == S.cost[b] && DS_SEQ(S.cand[a]) < DS_SEQ(S.cand[b]))) { S.heap[h] = b; S.heap[par] = a; h = par; } else break;
      }
      if (t > rh[ri].t_arrive + P.ttft_s - P.margin_s) {
        const int w = S.heap[0];
        S.heap[0] = S.heap[--nh];
        int h2 = 0;
        for (;;) {
          int l = 2 * h2 + 1, rr = l + 1, m = h2;
#define DS_GT(x, y) (S.cost[S.heap[x]] > S.cost[S.heap[y]] || (S.cost[S.heap[x]] == S.cost[S.heap[y]] && DS_SEQ(S.cand[S.heap[x]]) < DS_SEQ(S.cand[S.heap[y]])))
          if (l < nh && DS_GT(l, m)) m = l;
          if (rr < nh && DS_GT(rr, m)) m = rr;
#undef DS_GT
          if (m == h2) break;
          const int tmp = S.heap[h2]; S.heap[h2] = S.heap[m]; S.heap[m] = tmp; h2 = m;
        }
        t -= S.cost[w];
        S.is_late[w] = 1;
      }
    }
    double t2 = now;
    for (int i = 0; i < nc; i++) {
      if (S.is_late[i] || S.cost[i] < 0) continue;
      t2 += S.cost[i];
      const double slack = rh[S.cand[i]].t_arrive + P.ttft_s - P.margin_s - t2;
      if (slack < min_slack) min_slack = slack;
    }
    for (int i = 0; i < nc; i++) {
      DsReq& r = S.req[S.cand[i]];
      if (S.is_late[i] && !r.was_late) { r.was_late = 1; S.cnt[DS_C_DEMOTED]++; }
      r.late = S.is_late[i];
      if (!S.is_late[i]) S.ord[no++] = S.cand[i];
    }
    // the others have missed their first token anyway: shortest remaining work first
    const int n1 = no;
    for (int i = 0; i < nc; i++) {
      if (!S.is_late[i]) continue;
      const int ri = S.cand[i];
      double c = S.cost[i];
      if (!(c > 0)) { const int pos = S.req[ri].slot >= 0 ? S.slot[S.req[ri].slot].pos : 0; c = ds_cost_prefill(P, pos, rh[ri].n_in - pos); S.cost[i] = c; }
      int j = no - 1;
      // S.heap reused as the cand index of each late entry of ord (parallel array from n1)
      while (j >= n1 && (S.cost[S.heap[j]] > c || (S.cost[S.heap[j]] == c && DS_SEQ(S.ord[j]) > DS_SEQ(ri)))) { S.ord[j + 1] = S.ord[j]; S.heap[j + 1] = S.heap[j]; j--; }
      S.ord[j + 1] = ri; S.heap[j + 1] = i; no++;
    }
  }
  // age cap: whoever has waited longer than age_s goes first, oldest first (S.heap is free here)
  if (!P.fifo && P.age_s > 0) {
    int na = 0;
    for (int i = 0; i < no; i++) if (now - rh[S.ord[i]].t_arrive > P.age_s) S.heap[na++] = S.ord[i];
    if (na) {
      for (int i = 1; i < na; i++) {
        const int x = S.heap[i]; int j = i - 1;
        while (j >= 0 && (rh[S.heap[j]].t_arrive > rh[x].t_arrive || (rh[S.heap[j]].t_arrive == rh[x].t_arrive && DS_SEQ(S.heap[j]) > DS_SEQ(x)))) { S.heap[j + 1] = S.heap[j]; j--; }
        S.heap[j + 1] = x;
      }
      int k = na;
      for (int i = 0; i < no; i++) if (!(now - rh[S.ord[i]].t_arrive > P.age_s)) S.heap[k++] = S.ord[i];
      for (int i = 0; i < no; i++) S.ord[i] = S.heap[i];
    }
  }
  // the worst first-token time anybody is heading for, by the estimates, in this order
  double worst_ttft = 0;
  int near_miss = 0;
  if (!P.fifo) {
    double t3 = now;
    for (int i = 0; i < no; i++) {
      const int ri = S.ord[i];
      const DsReq& r = S.req[ri];
      double c = r.cost;
      if (!(c >= 0)) { const int pos = r.slot >= 0 ? S.slot[r.slot].pos : 0; c = ds_cost_prefill(P, pos, rh[ri].n_in - pos) * P.speed / share; }
      t3 += c;
      const double w = t3 - rh[ri].t_arrive;
      if (w > worst_ttft) worst_ttft = w;
      if (P.near_s > 0 && w > P.ttft_s && w <= P.ttft_s + P.near_s) near_miss = 1;
    }
  }
  // ---- admission, in service order
  for (int i = 0; i < no; i++) if (S.req[S.ord[i]].slot < 0) ds_try_admit(S, rh, pt, S.ord[i]);

  int cols = 0, nsel = 0, nu = 0, n2 = 0, n1t = 0, n1d = 0, nf = 0, nparts = 0, ndec = 0;
  double est = 0;
  if (S.n_run > 0 && cmd == 0) {
    // ---- streams: one column each
    double dmax = 0, dsum = 0;
    int run_slot[DS_SELCAP], run_n[DS_SELCAP], nruns = 0;
    for (int i = 0; i < S.n_run; i++) {
      const int s = S.run_order[i];
      const DsSlot& sl = S.slot[s];
      if (sl.pos >= sl.n_in) {
        run_slot[nruns] = s; run_n[nruns] = 0; nruns++;   // n = 0 marks a decode row
        const double segs = sl.pos / 128 + 1;
        if (segs > dmax) dmax = segs;
        dsum += segs;
      }
    }
    ndec = nruns;
    double dec = 0;
    if (ndec) { const double a = DS_DLAT * (dmax < 32 ? dmax : 32), b = dsum * 8 / P.n_blocks; dec = P.c_dseg * (a > b ? a : b); }
    // ---- prefill candidates in service order: adopt cached pages; a request whose next page is being computed by
    // another one waits for it and lends that one its turn
    int n2c = 0;
    for (int i = 0; i < P.n_slots; i++) S.seen2[i] = 0;
    for (int i = 0; i < no; i++) {
      const DsReq& r = S.req[S.ord[i]];
      const int s = r.slot;
      if (s < 0 || S.slot[s].pos >= S.slot[s].n_in || S.seen2[s]) continue;
      ds_adopt(S, pt, s);
      const DsSlot& sl = S.slot[s];
      if (P.reuse && (sl.pos & (DS_PAGE - 1)) == 0 && sl.npages == (sl.pos >> 9) && (sl.pos >> 9) < r.max_reuse) {
        const int idx = ds_find(S, r.chain[sl.pos >> 9]);
        if (idx >= 0 && S.tab[idx].state == 2) {
          const int ls = S.tab[idx].owner;
          if (ls != s && S.slot[ls].pos < S.slot[ls].n_in) {
            S.cnt[DS_C_WAITS]++;
            if (!S.seen2[ls]) { S.cnt[DS_C_LENT]++; S.seen2[ls] = 1; S.c2[n2c++] = ls; }
            continue;
          }
        }
      }
      S.seen2[s] = 1;
      S.c2[n2c++] = s;
    }
    // ---- size class of the pass: the most prefill per second among those within the target; else the cheapest
    int nbest = 0, have = 0, best_ok = 0;
    double best_val = 0, best_est = 0;
    if (n2c) {
      const double target = ((P.tail_guard_s > 0 && worst_ttft > P.tail_guard_s) ? P.target_tail_s : near_miss ? P.target_near_s : min_slack < P.boost_slack_s ? P.target_hi_s : P.target_s) / P.speed;
      for (int ci = 0; ci < DS_NCLASS; ci++) {
        const int size = P.class_size[ci];
        const double base = P.class_base[ci];
        const int cap = (size < P.tcap - 8 ? size : P.tcap - 8) - ndec;
        if (cap < 4 || (have && base + dec > target)) continue;
        double att;
        const int nfl = ds_fill(S, n2c, cap, target - base - dec, S.fill_req, S.fill_n, att);
        int fc = 0;
        for (int i = 0; i < nfl; i++) fc += S.fill_n[i];
        const double e = base + dec + att;
        if (!fc) continue;
        const int ok = e <= target * 1.02;
        const double val = ok ? fc / e : -e;
        if (!have || ok > best_ok || (ok == best_ok && val > best_val)) {
          have = 1; best_ok = ok; best_val = val; best_est = e; nbest = nfl;
          for (int i = 0; i < nfl; i++) { S.best_req[i] = S.fill_req[i]; S.best_n[i] = S.fill_n[i]; }
        }
      }
    }
    if (have) {
      for (int i = 0; i < nbest && nruns < DS_SELCAP; i++) { run_slot[nruns] = S.best_req[i]; run_n[nruns] = S.best_n[i]; nruns++; }
      est = best_est;
    } else est = P.c_small + dec;
    // ---- the plan: columns, pages, attention units and tasks, selected columns
    int* two = plan + DS_PL_BT;
    for (int ri = 0; ri < nruns; ri++) {
      const int s = run_slot[ri];
      DsSlot& sl = S.slot[s];
      DsReq& r = S.req[sl.req];
      const bool decode = run_n[ri] == 0;
      const int n = decode ? 1 : run_n[ri], pos0 = sl.pos;
      if (!decode && P.reuse) {   // claim the pages this run computes
        const int k1 = (pos0 + n - 1) / DS_PAGE + 1;
        for (int k = pos0 / DS_PAGE; k < (k1 < r.nchain ? k1 : r.nchain); k++) {
          if (ds_find(S, r.chain[k]) < 0) {
            const int idx = ds_insert(S, r.chain[k]);
            S.tab[idx].state = 2; S.tab[idx].owner = s; S.tab[idx].page = -1; S.tab[idx].children = 0; S.tab[idx].parent = 0;
            const uint64_t key = r.chain[k];
            if (S.evicted_bits[(key >> 5) & 32767] & (1u << (key & 31))) { S.evicted_bits[(key >> 5) & 32767] &= ~(1u << (key & 31)); S.cnt[DS_C_RECOMPUTED]++; }
          }
        }
      }
      while (sl.npages <= (pos0 + n - 1) / DS_PAGE) {
        const int page = ds_take_page(S);
        S.ref[page] = 1;
        pt[s * DS_MAXPG + sl.npages++] = page;
        sl.reserved--; S.committed--;
      }
      for (int i = 0; i < n; i++) {
        plan[DS_PL_SLOT + cols + i] = s | ((i & 15) << 17) | (i == n - 1 ? DS_LASTBIT : 0);
        plan[DS_PL_POS + cols + i] = pos0 + i;
        plan[DS_PL_TOK + cols + i] = decode ? -1 : sl.prompt_off + pos0 + i;
        plan[DS_PL_CU + cols + i] = nu + (i >> 4);
      }
      const int u0 = nu;
      for (int c0 = 0; c0 < n; c0 += 16) {
        const int nqq = n - c0 < 16 ? n - c0 : 16;
        const int nss = ((pos0 + c0 + nqq - 1) >> P.ss_shift) + 1;
        plan[DS_PL_UNIT + 4 * nu] = cols + c0; plan[DS_PL_UNIT + 4 * nu + 1] = nqq; plan[DS_PL_UNIT + 4 * nu + 3] = nss;
        if (nss > 1) { plan[DS_PL_FOLD + nf++] = nu; plan[DS_PL_UNIT + 4 * nu + 2] = nparts; nparts += nss; }
        else plan[DS_PL_UNIT + 4 * nu + 2] = -1;
        nu++;
      }
      // tasks per super-segment and kv head: pairs of units that both reach it, a single unit for the odd end.
      // The single-unit tasks are collected behind the last possible pair task (moved down afterwards).
      const int nun = nu - u0, nss_last = plan[DS_PL_UNIT + 4 * (nu - 1) + 3];
      for (int ss = 0; ss < nss_last; ss++) {
        int first = u0;
        while (plan[DS_PL_UNIT + 4 * first + 3] <= ss) first++;
        const int live = u0 + nun - first;
        for (int g = 0; g < 8; g++) {
          for (int i = 0; i + 1 < live; i += 2) two[n2++] = (first + i) | (ss << 16) | (g << 22);
          if (live & 1) two[DS_MAXBT - 1 - n1t++] = (first + live - 1) | (ss << 16) | (g << 22);
        }
      }
      if (decode || pos0 + n == sl.n_in) {
        plan[DS_PL_SEL + 2 * nsel] = cols + n - 1; plan[DS_PL_SEL + 2 * nsel + 1] = s;
        R.req[nsel] = sl.req;
        sl.produced++;
        const int done = sl.produced >= sl.n_out;
        R.fin[nsel] = done;
        if (done) S.fin_slots[S.n_fin++] = s;
        nsel++;
      }
      // the pass is on its way: complete prompt pages become visible to later passes; the position advances
      if (decode) S.cnt[DS_C_DECODE_TOK]++;
      else {
        S.cnt[DS_C_PREFILL_TOK] += n;
        if (P.reuse) {
          for (int k = pos0 / DS_PAGE; k < (pos0 + n) / DS_PAGE; k++) {
            if (k >= r.nchain) break;
            int idx = ds_find(S, r.chain[k]);
            const int page = pt[s * DS_MAXPG + k];
            if (!(idx >= 0 && S.tab[idx].state == 1) && S.page_ent[page] < 0) {
              int pi = -1;
              const bool parent_ok = k == 0 || ds_cached(S, r.chain[k - 1], pi);
              if (parent_ok) {
                if (idx >= 0) ds_erase(S, idx);          // it was "being computed" (by this or another slot)
                idx = ds_insert(S, r.chain[k]);
                S.tab[idx].state = 1; S.tab[idx].page = page; S.tab[idx].owner = -1; S.tab[idx].children = 0;
                S.tab[idx].parent = k ? r.chain[k - 1] : 0;
                S.page_ent[page] = idx;
                ds_lru_append(S, page);
                if (k) { ds_cached(S, r.chain[k - 1], pi); S.tab[pi].children++; }
                S.epoch++;
                S.cnt[DS_C_REGISTERED]++;
                idx = -1;
              }
            }
            idx = ds_find(S, r.chain[k]);
            if (idx >= 0 && S.tab[idx].state == 2 && S.tab[idx].owner == s) ds_erase(S, idx);
          }
        }
      }
      sl.pos = pos0 + n;
      cols += n;
    }
    // single-unit tasks behind the pair tasks, in the order they were listed: first those of units of one position
    // (decode rows: their own kernel), then the others
    for (int i = 0; i < n1t; i++) { const int tk = two[DS_MAXBT - 1 - i]; if (plan[DS_PL_UNIT + 4 * (tk & 0xfff) + 1] == 1) two[n2 + n1d++] = tk; }
    for (int i = 0, k = n1d; i < n1t; i++) { const int tk = two[DS_MAXBT - 1 - i]; if (plan[DS_PL_UNIT + 4 * (tk & 0xfff) + 1] != 1) two[n2 + k++] = tk; }
    S.cnt[DS_C_PASSES]++;
    S.cnt[DS_C_COLUMNS] += cols;
  }
  if (S.n_run > S.cnt[DS_C_RUN_MAX]) S.cnt[DS_C_RUN_MAX] = S.n_run;
  int* hdr = plan + DS_PL_HDR;
  hdr[0] = cols; hdr[1] = n2; hdr[2] = n1t; hdr[3] = nf; hdr[4] = nsel; hdr[5] = S.n_freed; hdr[6] = n1d;
  R.n_sel = nsel; R.n_cols = cols; R.n_free = S.n_free; R.n_running = S.n_run; R.n_cached = 0; R.head = S.head; R.est_s = est;
  R.step = (int)S.cnt[DS_C_STEPS]++;
  int nqueued = 0;
  for (int q = S.head; q < S.seen; q++) nqueued += S.req[q % DS_RING].state == 1;
  R.n_queued = nqueued;
  return cols;
}
