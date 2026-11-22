// mkprof.cuh: where the time goes INSIDE a persistent kernel (nsys sees one kernel; this sees its phases).
//
// One thread per block records a mark at every phase boundary: the global timer (ns, the same clock on every SM)
// and the SM's cycle counter, with a 16-bit tag. tools/mkprof/mkprof.py turns the dump into a table per phase:
// work time per block, time spent waiting (barrier), the span of the phase across the grid, imbalance between
// blocks, and, given the bytes a phase must read, the bandwidth it reached.
//
// Compiled out unless -DMKPROF: the marks cost nothing in a build without it.
//
//   #include "mkprof.cuh"
//   __global__ void k(..., MkProf prof) {
//     MKPROF_DECL(n);                         // the block's record counter (a register)
//     for (int step = 0; step < steps; ++step) {
//       if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_STEP);   // optional: start of a decode step
//       for (layer...) {
//         if (threadIdx.x == 0) MKPROF_MARK(prof, n, 1);           // phase 1 (e.g. qkv) begins
//         ... work ...
//         if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_WAIT); // entering the grid barrier
//         grid_barrier();
//         if (threadIdx.x == 0) MKPROF_MARK(prof, n, 2);           // phase 2 begins
//         ...
//   Host: buf = torch.zeros(grid * cap * 2, dtype=torch.int64, device="cuda"); MkProf{(uint64_t*)buf.data_ptr(), cap}
//         after the run: np.save("prof.npy", buf.view(grid, cap, 2).cpu().numpy())
//         python tools/mkprof/mkprof.py prof.npy --tags tags.json [--bytes bytes.json] [--card h100|b200]
//   Mark from ONE thread (threadIdx.x == 0) and keep the mark next to the code it times (after a __syncthreads()
//   when the phase's work is block-wide, or the mark measures only thread 0). Records beyond `cap` are dropped.
#pragma once
#include <cstdint>

struct MkProf {
  unsigned long long* buf;  // [gridDim.x][cap][2]: {globaltimer ns, (clock64 << 16) | tag}
  unsigned int cap;         // records per block
};

#define MKPROF_STEP 0xfffeu   // a decode step begins
#define MKPROF_WAIT 0xffffu   // the block starts waiting (grid barrier, flag, dependency counter)

#ifdef MKPROF
__device__ __forceinline__ unsigned long long mkprof_gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define MKPROF_DECL(n) unsigned int n = 0
#define MKPROF_MARK(P, n, tag)                                                                       \
  do {                                                                                               \
    if ((n) < (P).cap) {                                                                             \
      unsigned long long* r_ = (P).buf + 2ull * ((unsigned long long)blockIdx.x * (P).cap + (n));   \
      r_[0] = mkprof_gtime();                                                                        \
      r_[1] = ((unsigned long long)clock64() << 16) | ((unsigned long long)(tag) & 0xffffull);       \
    }                                                                                                \
    ++(n);                                                                                           \
  } while (0)
#else
#define MKPROF_DECL(n) do {} while (0)
#define MKPROF_MARK(P, n, tag) do {} while (0)
#endif
