// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file test_warp_tools.cu
 * \brief CUDA kernels for testing warp-level primitives.
 */

#include "test_warp_tools.cuh"
#include "hoomd/WarpTools.cuh"

#define BLOCK_SIZE 32

//! Performs an iterative warp reduction on a data set using \a tpp threads per row.
/*!
 * \param d_data Data to scan as a N x width matrix.
 * \param d_reduce Output of the reduction at each step.
 * \param d_sum Total sum for each row of data.
 * \param N Number of rows in data.
 * \param width Number of entries to scan.
 * \param reduce_idx Indexer for saving intermediate results of reduction.
 * \tparam tpp Number of threads to use per row in \a d_data .
 *
 * The kernel is launched with \a tpp threads working per row in \a d_data, which has \a N rows and \a width entries
 * per row. This sub-warp group then iterates through the data in the row, performing a reduction at each iteration.
 * The result of the reduction is saved into \a d_reduce for each iteration. The total sum is also accumulated
 * into \a d_sum.
 *
 * This test kernel is more complicated than the basic tests that CUB runs for WarpReduce. The reason for this is to
 * emulate a use-case in HOOMD, namely the force accumulation using multiple threads per particle.
 */
template<int tpp>
__global__ void warp_reduce_kernel(const int* d_data,
                                   int* d_reduce,
                                   int* d_sum,
                                   const unsigned int N,
                                   const unsigned int width,
                                   const Index2D reduce_idx)
    {
    // thread id in the global grid
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // row of data that this thread operates on
    const unsigned int idx = tid / tpp;
    // index of thread within the sub warp
    const unsigned int cta_idx = threadIdx.x % tpp;
    if (idx >= N) return;

    int sum(0), cntr(0);
    unsigned int offset = cta_idx;
    bool done = false;
    while (!done)
        {
        // load in data
        int thread_data;
        if (offset < width)
            {
            thread_data = d_data[idx * width + offset];
            }
        else
            {
            thread_data = 0;
            done = true;
            }
        offset += tpp;

        // only scan if sub warp still has work to do
        done = hoomd::detail::WarpScan<bool,tpp>().Broadcast(done, 0);
        if (!done)
            {
            // scan the thread data
            int sum_iter = hoomd::detail::WarpReduce<int,tpp>().Sum(thread_data);

            // save reduce result for this iteration
            if (cta_idx == 0)
                d_reduce[reduce_idx(idx,cntr)] = sum_iter;

            // accumulate total sum
            sum += sum_iter;

            ++cntr;
            }
        }

    // thread 0 writes out accumulated sum
    if (cta_idx == 0)
        {
        d_sum[idx] = sum;
        }
    }

// Dispatch for warp reduction based on requested threads per particle.
/*!
 * \param params Reduction parameters.
 * \tparam tpp Number of threads to try to launch.
 *
 * This recursive template compiles the kernel for all valid threads per particle (powers of 2 from 1 to 32), and only
 * executes the kernel for the number of threads that is equal to the value specified in \a params.
 */
template<int tpp>
void warp_reduce_launcher(const reduce_params& params)
    {
    if (tpp == params.tpp)
        {
        dim3 grid((params.N*tpp+BLOCK_SIZE-1)/BLOCK_SIZE);
        warp_reduce_kernel<tpp><<<grid, BLOCK_SIZE>>>(params.data, params.reduce, params.sum, params.N, params.width, params.reduce_idx);
        }
    else
        {
        warp_reduce_launcher<tpp/2>(params);
        }
    }
//! Terminates the recursive template.
template<>
void warp_reduce_launcher<0>(const reduce_params& params)
    {
    }

/*!
 * \params Scan parameters.
 *
 * The scan results are first memset to zero.
 */
void warp_reduce(const reduce_params& params)
    {
    cudaMemset(params.reduce, 0, params.reduce_idx.getNumElements() * sizeof(int));
    cudaMemset(params.sum, 0, params.N * sizeof(int));
    warp_reduce_launcher<32>(params);
    }

//! Performs an iterative warp scan on a data set using \a tpp threads per row.
/*!
 * \param d_data Data to scan as a N x width matrix.
 * \param d_scan Output of the scan at each step of sum.
 * \param d_sum Total sum for each row of data.
 * \param N Number of rows in data.
 * \param width Number of entries to scan.
 * \param scan_idx Indexer for saving intermediate results of scan.
 * \tparam tpp Number of threads to use per row in \a d_data .
 *
 * The kernel is launched with \a tpp threads working per row in \a d_data, which has \a N rows and \a width entries
 * per row. This sub-warp group then iterates through the data in the row, performing an exclusive sum at each iteration.
 * The result of the scan is saved into \a d_scan for each thread along with the aggregate at each iteration. The total
 * sum is also accumulated into \a d_sum.
 *
 * This test kernel is more complicated than the basic tests that CUB runs for WarpScan. The reason for this is to
 * emulate a use-case in HOOMD, namely the neighbor list generation using multiple threads per particle.
 */
template<int tpp>
__global__ void warp_scan_kernel(const int* d_data,
                                 int* d_scan,
                                 int* d_sum,
                                 const unsigned int N,
                                 const unsigned int width,
                                 const Index3D scan_idx)
    {
    // thread id in the global grid
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // row of data that this thread operates on
    const unsigned int idx = tid / tpp;
    // index of thread within the sub warp
    const unsigned int cta_idx = threadIdx.x % tpp;
    if (idx >= N) return;

    int sum(0), cntr(0);
    unsigned int offset = cta_idx;
    bool done = false;
    while (!done)
        {
        // load in data
        int thread_data;
        if (offset < width)
            {
            thread_data = d_data[idx * width + offset];
            }
        else
            {
            thread_data = 0;
            done = true;
            }
        offset += tpp;

        // only scan if sub warp still has work to do
        done = hoomd::detail::WarpScan<bool,tpp>().Broadcast(done, 0);
        if (!done)
            {
            // scan the thread data
            int sum_iter(0);
            hoomd::detail::WarpScan<int,tpp>().ExclusiveSum(thread_data, thread_data, sum_iter);

            // save scan result for this iteration
            d_scan[scan_idx(idx,cta_idx,cntr)] = thread_data;
            if (cta_idx == 0)
                d_scan[scan_idx(idx,tpp,cntr)] = sum_iter;

            // accumulate total sum
            sum += sum_iter;

            ++cntr;
            }
        }

    // thread 0 writes out accumulated sum
    if (cta_idx == 0)
        {
        d_sum[idx] = sum;
        }
    }

// Dispatch for warp scan based on requested threads per particle.
/*!
 * \param params Scan parameters.
 * \tparam tpp Number of threads to try to launch.
 *
 * This recursive template compiles the kernel for all valid threads per particle (powers of 2 from 1 to 32), and only
 * executes the kernel for the number of threads that is equal to the value specified in \a params.
 */
template<int tpp>
void warp_scan_launcher(const scan_params& params)
    {
    if (tpp == params.tpp)
        {
        dim3 grid((params.N*tpp+BLOCK_SIZE-1)/BLOCK_SIZE);
        warp_scan_kernel<tpp><<<grid, BLOCK_SIZE>>>(params.data, params.scan, params.sum, params.N, params.width, params.scan_idx);
        }
    else
        {
        warp_scan_launcher<tpp/2>(params);
        }
    }
//! Terminates the recursive template.
template<>
void warp_scan_launcher<0>(const scan_params& params)
    {
    }

/*!
 * \params Scan parameters.
 *
 * The scan results are first memset to zero.
 */
void warp_scan(const scan_params& params)
    {
    cudaMemset(params.scan, 0, params.scan_idx.getNumElements() * sizeof(int));
    cudaMemset(params.sum, 0, params.N * sizeof(int));
    warp_scan_launcher<32>(params);
    }
