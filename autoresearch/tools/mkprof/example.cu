// Example and self-test of mkprof.cuh: a persistent kernel with three phases per step and a grid barrier between
// phases. Phase 1 streams `bytes` of memory (known bandwidth), phase 2 is pure arithmetic, phase 3 is deliberately
// unbalanced (block b works proportionally to b % 4 + 1). Writes prof.npy, tags.json, bytes.json.
//   nvcc -O3 -arch=native -DMKPROF -o ex example.cu && ./ex && python mkprof.py prof.npy --tags tags.json --bytes bytes.json --card h100
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <string>
#include "mkprof.cuh"
namespace cg = cooperative_groups;

__global__ void persistent(const float4* __restrict__ src, size_t n4, float* sink, int steps, MkProf prof) {
  cg::grid_group grid = cg::this_grid();
  MKPROF_DECL(n);
  float acc = 0.f;
  for (int s = 0; s < steps; ++s) {
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_STEP);
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, 1);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
      float4 v = __ldcs(src + i); acc += v.x + v.y + v.z + v.w;
    }
    __syncthreads();
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_WAIT);
    grid.sync();
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, 2);
    for (int k = 0; k < 4000; ++k) acc = acc * 1.0000001f + 1e-7f;
    __syncthreads();
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_WAIT);
    grid.sync();
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, 3);
    for (int k = 0; k < 2000 * (blockIdx.x % 4 + 1); ++k) acc = acc * 1.0000001f + 1e-7f;
    __syncthreads();
    if (threadIdx.x == 0) MKPROF_MARK(prof, n, MKPROF_WAIT);
    grid.sync();
  }
  if (acc == 12345.f) sink[0] = acc;
}

int main() {
  int dev = 0, sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const size_t bytes = size_t(2) << 30, n4 = bytes / 16; const int steps = 10, threads = 512; const unsigned cap = 64;
  float4* src; float* sink; unsigned long long* buf;
  cudaMalloc(&src, bytes); cudaMemset(src, 0, bytes); cudaMalloc(&sink, 4);
  cudaMalloc(&buf, size_t(sms) * cap * 16); cudaMemset(buf, 0, size_t(sms) * cap * 16);
  MkProf prof{buf, cap}; int st = steps;
  void* argsv[] = {&src, (void*)&n4, &sink, &st, &prof};
  cudaError_t e = cudaLaunchCooperativeKernel((void*)persistent, sms, threads, argsv, 0, 0);
  cudaDeviceSynchronize();
  if (e != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
  std::vector<unsigned long long> h(size_t(sms) * cap * 2);
  cudaMemcpy(h.data(), buf, h.size() * 8, cudaMemcpyDeviceToHost);
  FILE* f = fopen("prof.npy", "wb");
  char hdr[128]; int len = snprintf(hdr, sizeof hdr, "{'descr': '<i8', 'fortran_order': False, 'shape': (%d, %u, 2), }", sms, cap);
  int pad = 64 - (10 + len + 1) % 64; std::string H(hdr); H += std::string(pad, ' '); H += '\n';
  unsigned short hl = (unsigned short)H.size(); fwrite("\x93NUMPY\x01\x00", 1, 8, f); fwrite(&hl, 2, 1, f); fwrite(H.data(), 1, H.size(), f);
  fwrite(h.data(), 8, h.size(), f); fclose(f);
  f = fopen("tags.json", "w"); fprintf(f, "{\"1\": \"stream\", \"2\": \"compute\", \"3\": \"unbalanced\"}\n"); fclose(f);
  f = fopen("bytes.json", "w"); fprintf(f, "{\"stream\": %zu}\n", bytes); fclose(f);
  printf("sms %d, wrote prof.npy\n", sms);
  return 0;
}
