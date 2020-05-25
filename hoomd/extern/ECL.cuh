/*
ECL-CC code: ECL-CC is a connected components algorithm. The CUDA
implementation thereof is very fast. It operates on graphs stored in
binary CSR format.

Copyright (c) 2017, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Jayadharini Jaiganesh and Martin Burtscher
*/

// Maintainer: jglaser

#include <hip/hip_runtime.h>

#pragma once

inline void ecl_connected_components(const int nodes,
    const int edges,
    const int *d_nidx,
    const int* d_nlist,
    const int *d_nstat,
    const int *d_wl,
    const hipDeviceProp_t& deviceProp);

#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <set>

static const int Device = 0;
static const int ThreadsPerBlock = 256;

#ifdef __HIP_PLATFORM_NVCC__
static const int warpsize = 32;
#else
static const int warpsize = 64;
#endif

static __device__ int topL, posL, topH, posH;

/* initialize with first smaller neighbor ID */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void init(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }

  if (from == 0) {topL = 0; posL = 0; topH = nodes - 1; posH = nodes - 1;}
}

/* intermediate pointer jumping */
inline __device__ int representative(const int idx, int* const __restrict__ nstat)
{
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/* process low-degree vertices at thread granularity and fill worklists */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute1(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int vstat = nstat[v];
    if (v != vstat) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx;
        if (deg <= 352) {
          idx = atomicAdd(&topL, 1);
        } else {
          idx = atomicAdd(&topH, -1);
        }
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

/* process medium-degree vertices at warp granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute2(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl)
{
  const int lane = threadIdx.x % warpsize;

  int idx;
  if (lane == 0) idx = atomicAdd(&posL, 1);

  #ifdef __HIP_PLATFORM_NVCC__
  idx = __shfl_sync(0xffffffff,idx, 0);
  #else
  idx = __shfl(idx,0);
  #endif
  while (idx < topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpsize) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(&posL, 1);

    #ifdef __HIP_PLATFORM_NVCC__
    idx = __shfl_sync(0xffffffff,idx, 0);
    #else
    idx = __shfl(idx,0);
    #endif
  }
}

/* process high-degree vertices at block granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute3(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl)
{
  __shared__ int vB;
  if (threadIdx.x == 0) {
    const int idx = atomicAdd(&posH, -1);
    vB = (idx > topH) ? wl[idx] : -1;
  }
  __syncthreads();
  while (vB >= 0) {
    const int v = vB;
    __syncthreads();
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&posH, -1);
      vB = (idx > topH) ? wl[idx] : -1;
    }
    __syncthreads();
  }
}

/* link all vertices to sink */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void flatten(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}

inline void ecl_connected_components(const int nodes,
    const int edges,
    const int *d_nidx,
    const int *d_nlist,
    int *d_nstat,
    int *d_wl,
    const hipDeviceProp_t& deviceProp)
    {
    const int SMs = deviceProp.multiProcessorCount;
    const int mTSM = deviceProp.maxThreadsPerMultiProcessor;

    const int blocks = SMs * mTSM / ThreadsPerBlock;
    hipLaunchKernelGGL(init, dim3(blocks), dim3(ThreadsPerBlock), 0, 0, nodes, d_nidx, d_nlist, d_nstat);
    hipLaunchKernelGGL(compute1, dim3(blocks), dim3(ThreadsPerBlock), 0, 0, nodes, d_nidx, d_nlist, d_nstat, d_wl);
    hipLaunchKernelGGL(compute2, dim3(blocks), dim3(ThreadsPerBlock), 0, 0, nodes, d_nidx, d_nlist, d_nstat, d_wl);
    hipLaunchKernelGGL(compute3, dim3(blocks), dim3(ThreadsPerBlock), 0, 0, nodes, d_nidx, d_nlist, d_nstat, d_wl);
    hipLaunchKernelGGL(flatten, dim3(blocks), dim3(ThreadsPerBlock), 0, 0, nodes, d_nidx, d_nlist, d_nstat);
    }


#if 0
struct GPUTimer
{
  hipEvent_t beg, end;
  GPUTimer() {hipEventCreate(&beg);  hipEventCreate(&end);}
  ~GPUTimer() {hipEventDestroy(beg);  hipEventDestroy(end);}
  void start() {hipEventRecord(beg, 0);}
  double stop() {hipEventRecord(end, 0);  hipEventSynchronize(end);  float ms;  hipEventElapsedTime(&ms, beg, end);  return 0.001 * ms;}
};

static void computeCC(const int nodes, const int edges, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  hipSetDevice(Device);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  int* nidx_d;
  int* nlist_d;
  int* nstat_d;
  int* wl_d;

  if (hipSuccess != hipMalloc((void **)&nidx_d, (nodes + 1) * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate nidx_d\n\n");  exit(-1);}
  if (hipSuccess != hipMalloc((void **)&nlist_d, edges * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate nlist_d\n\n");  exit(-1);}
  if (hipSuccess != hipMalloc((void **)&nstat_d, nodes * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate nstat_d,\n\n");  exit(-1);}
  if (hipSuccess != hipMalloc((void **)&wl_d, nodes * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate wl_d,\n\n");  exit(-1);}

  if (hipSuccess != hipMemcpy(nidx_d, nidx, (nodes + 1) * sizeof(int), hipMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}
  if (hipSuccess != hipMemcpy(nlist_d, nlist, edges * sizeof(int), hipMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n\n");  exit(-1);}

  hipFuncSetCacheConfig(init, hipFuncCachePreferL1);
  hipFuncSetCacheConfig(compute1, hipFuncCachePreferL1);
  hipFuncSetCacheConfig(compute2, hipFuncCachePreferL1);
  hipFuncSetCacheConfig(compute3, hipFuncCachePreferL1);
  hipFuncSetCacheConfig(flatten, hipFuncCachePreferL1);

  const int blocks = SMs * mTSM / ThreadsPerBlock;
  GPUTimer timer;
  timer.start();
  init<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
  compute1<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute2<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute3<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  flatten<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
  double runtime = timer.stop();

  printf("compute time: %.4f s\n", runtime);
  printf("throughput: %.3f Mnodes/s\n", nodes * 0.000001 / runtime);
  printf("throughput: %.3f Medges/s\n", edges * 0.000001 / runtime);

  if (hipSuccess != hipMemcpy(nstat, nstat_d, nodes * sizeof(int), hipMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n\n");  exit(-1);}

  hipFree(wl_d);
  hipFree(nstat_d);
  hipFree(nlist_d);
  hipFree(nidx_d);
}

static void verify(const int v, const int id, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  if (nstat[v] >= 0) {
    if (nstat[v] != id) {fprintf(stderr, "ERROR: found incorrect ID value\n\n");  exit(-1);}
    nstat[v] = -1;
    for (int i = nidx[v]; i < nidx[v + 1]; i++) {
      verify(nlist[i], id, nidx, nlist, nstat);
    }
  }
}

int main(int argc, char* argv[])
{
  printf("ECL-CC v1.0 (%s)\n", __FILE__);
  printf("Copyright 2017 Texas State University\n");

  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file_name\n\n", argv[0]);  exit(-1);}

  ECLgraph g = readECLgraph(argv[1]);

  int* nodestatus = NULL;
  hipHostAlloc(&nodestatus, g.nodes * sizeof(int), hipHostAllocDefault);
  if (nodestatus == NULL) {fprintf(stderr, "ERROR: nodestatus - host memory allocation failed\n\n");  exit(-1);}

  printf("input graph: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * g.edges / g.nodes);
  int mindeg = g.nodes;
  int maxdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    int deg = g.nindex[v + 1] - g.nindex[v];
    mindeg = std::min(mindeg, deg);
    maxdeg = std::max(maxdeg, deg);
  }
  printf("minimum degree: %d edges\n", mindeg);
  printf("maximum degree: %d edges\n", maxdeg);

  computeCC(g.nodes, g.edges, g.nindex, g.nlist, nodestatus);

  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(nodestatus[v]);
  }
  printf("number of connected components: %d\n", s1.size());

  /* verification code (may need extra runtime stack space due to deep recursion) */

  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
      if (nodestatus[g.nlist[i]] != nodestatus[v]) {fprintf(stderr, "ERROR: found adjacent nodes in different components\n\n");  exit(-1);}
    }
  }

  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] < 0) {fprintf(stderr, "ERROR: found negative component number\n\n");  exit(-1);}
  }

  std::set<int> s2;
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (nodestatus[v] >= 0) {
      count++;
      s2.insert(nodestatus[v]);
      verify(v, nodestatus[v], g.nindex, g.nlist, nodestatus);
    }
  }
  if (s1.size() != s2.size()) {fprintf(stderr, "ERROR: number of components do not match\n\n");  exit(-1);}
  if (s1.size() != count) {fprintf(stderr, "ERROR: component IDs are not unique\n\n");  exit(-1);}

  printf("all good\n\n");

  hipFreeHost(nodestatus);
  return 0;
}
#endif
