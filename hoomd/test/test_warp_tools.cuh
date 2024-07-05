// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file test_warp_tools.cuh
 * \brief Supporting data structures and CUDA kernels for testing warp-level primitives.
 */

#ifndef HOOMD_TEST_WARP_TOOLS_CUH_
#define HOOMD_TEST_WARP_TOOLS_CUH_

#include "hoomd/Index1D.h"

namespace hoomd
    {
namespace test
    {
//! Arguments for warp reduce tests
struct reduce_params
    {
    reduce_params(int* data_,
                  int* reduce_,
                  int* sum_,
                  unsigned int N_,
                  unsigned int width_,
                  unsigned int tpp_,
                  const Index2D& reduce_idx_)
        : data(data_), reduce(reduce_), sum(sum_), N(N_), width(width_), tpp(tpp_),
          reduce_idx(reduce_idx_)
        {
        }

    const int* data;          //!< Data to scan as a N x width matrix
    int* reduce;              //!< Output of the reduction at each step
    int* sum;                 //!< Total sum for each row of data
    const unsigned int N;     //!< Number of rows in data
    const unsigned int width; //!< Number of entries to reduce
    const unsigned int tpp;   //!< Number of threads to scan with per particle
    const Index2D reduce_idx; //!< Indexer for saving intermediate results of reduction
    };

//! Calls the warp reduce kernel
void warp_reduce(const reduce_params& params);

//! Arguments for warp scan tests
struct scan_params
    {
    scan_params(int* data_,
                int* scan_,
                int* sum_,
                unsigned int N_,
                unsigned int width_,
                unsigned int tpp_,
                const Index3D& scan_idx_)
        : data(data_), scan(scan_), sum(sum_), N(N_), width(width_), tpp(tpp_), scan_idx(scan_idx_)
        {
        }

    const int* data;          //!< Data to scan as a N x width matrix
    int* scan;                //!< Output of the scan at each step of sum
    int* sum;                 //!< Total sum for each row of data
    const unsigned int N;     //!< Number of rows in data
    const unsigned int width; //!< Number of entries to scan
    const unsigned int tpp;   //!< Number of threads to scan with per particle
    const Index3D scan_idx;   //!< Indexer for saving intermediate results of scan
    };

//! Calls the warp scan kernel
void warp_scan(const scan_params& params);

    } // end namespace test
    } // end namespace hoomd

#endif // HOOMD_TEST_TEST_WARP_TOOLS_CUH_
