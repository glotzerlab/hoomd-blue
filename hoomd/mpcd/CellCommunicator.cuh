// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellCommunicator.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellCommunicator
 */

#ifndef MPCD_CELL_COMMUNICATOR_CUH_
#define MPCD_CELL_COMMUNICATOR_CUH_

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <cuda_runtime.h>

namespace mpcd
{
namespace gpu
{

//! Kernel driver to pack cell communication buffer
template<typename T, class PackOpT>
cudaError_t pack_cell_buffer(typename PackOpT::element *d_send_buf,
                             const T *d_props,
                             const unsigned int *d_send_idx,
                             const PackOpT op,
                             const unsigned int num_send,
                             unsigned int block_size);

//! Kernel driver to unpack cell communication buffer
template<typename T, class PackOpT>
cudaError_t unpack_cell_buffer(T *d_props,
                               const unsigned int *d_cells,
                               const unsigned int *d_recv,
                               const unsigned int *d_recv_begin,
                               const unsigned int *d_recv_end,
                               const typename PackOpT::element *d_recv_buf,
                               const PackOpT op,
                               const unsigned int num_cells,
                               const unsigned int block_size);

#ifdef NVCC

namespace kernel
{
//! Kernel to pack cell communication buffer
/*!
 * \param d_send_buf Send buffer to pack (output)
 * \param d_props Cell property to pack
 * \param d_send_idx Cell indexes to pack into the buffer
 * \param op Pack operator
 * \param num_send Number of cells to pack
 * \param block_size Number of threads per block
 *
 * \tparam T Type of data to pack (inferred)
 * \tparam PackOpT Pack operation type (inferred)
 *
 * \b Implementation details:
 * Using one thread per cell to pack, each cell to send has its properties read
 * and packed into the send buffer.
 */
template<typename T, class PackOpT>
__global__ void pack_cell_buffer(typename PackOpT::element *d_send_buf,
                                 const T *d_props,
                                 const unsigned int *d_send_idx,
                                 const PackOpT op,
                                 const unsigned int num_send)
    {
    // one thread per buffer cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_send)
        return;

    const unsigned int send_idx = d_send_idx[idx];
    d_send_buf[idx] = op.pack(d_props[send_idx]);
    }

//! Kernel to unpack cell communication buffer
/*!
 * \param d_props Cell property array to unpack into
 * \param d_cells List of unique cells to unpack
 * \param d_recv Map of unique cells onto the receive buffer
 * \param d_recv_begin Beginning index into \a d_recv for cells to unpack
 * \param d_recv_end End index (exclusive) into \a d_recv for cells to unpack
 * \param d_recv_buf Received buffer from neighbor ranks
 * \param op Packing operator
 * \param num_cells Number of cells to unpack
 *
 * \tparam T Data type to unpack (inferred)
 * \tparam PackOpT Pack operation type (inferred)
 *
 * \b Implementation details:
 * Using one thread per cell to unpack, each cell iterates over the cells it
 * has received, and applies the unpacking operator to each in turn.
 */
template<typename T, class PackOpT>
__global__ void unpack_cell_buffer(T *d_props,
                                   const unsigned int *d_cells,
                                   const unsigned int *d_recv,
                                   const unsigned int *d_recv_begin,
                                   const unsigned int *d_recv_end,
                                   const typename PackOpT::element *d_recv_buf,
                                   const PackOpT op,
                                   const unsigned int num_cells)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_cells)
        return;

    const unsigned int cell_idx = d_cells[idx];
    const unsigned int begin = d_recv_begin[idx];
    const unsigned int end = d_recv_end[idx];

    // loop through all received data for this cell, and unpack it iteratively
    T val = d_props[cell_idx];
    for (unsigned int i = begin; i < end; ++i)
        {
        val = op.unpack(d_recv_buf[d_recv[i]], val);
        }

    // save the accumulated unpacked value
    d_props[cell_idx] = val;
    }

} // end namespace kernel

/*!
 * \param d_send_buf Send buffer to pack (output)
 * \param d_props Cell property to pack
 * \param d_send_idx Cell indexes to pack into the buffer
 * \param op Pack operator
 * \param num_send Number of cells to pack
 * \param block_size Number of threads per block
 *
 * \tparam T Type of data to pack (inferred)
 * \tparam PackOpT Pack operator type
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::pack_cell_buffer
 */
template<typename T, class PackOpT>
cudaError_t pack_cell_buffer(typename PackOpT::element *d_send_buf,
                             const T *d_props,
                             const unsigned int *d_send_idx,
                             const PackOpT op,
                             const unsigned int num_send,
                             unsigned int block_size)
    {
    // determine runtime block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::pack_cell_buffer<T,PackOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_send / run_block_size + 1);
    mpcd::gpu::kernel::pack_cell_buffer<<<grid, run_block_size>>>(d_send_buf,
                                                                  d_props,
                                                                  d_send_idx,
                                                                  op,
                                                                  num_send);

    return cudaSuccess;
    }

/*!
 * \param d_props Cell property array to unpack into
 * \param d_cells List of unique cells to unpack
 * \param d_recv Map of unique cells onto the receive buffer
 * \param d_recv_begin Beginning index into \a d_recv for cells to unpack
 * \param d_recv_end End index (exclusive) into \a d_recv for cells to unpack
 * \param d_recv_buf Received buffer from neighbor ranks
 * \param op Packing operator
 * \param num_cells Number of cells to unpack
 *
 * \tparam T Data type to unpack (inferred)
 * \tparam PackOpT Pack operation type (inferred)
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::unpack_cell_buffer
 */
template<typename T, class PackOpT>
cudaError_t unpack_cell_buffer(T *d_props,
                               const unsigned int *d_cells,
                               const unsigned int *d_recv,
                               const unsigned int *d_recv_begin,
                               const unsigned int *d_recv_end,
                               const typename PackOpT::element *d_recv_buf,
                               const PackOpT op,
                               const unsigned int num_cells,
                               const unsigned int block_size)
    {
    // determine runtime block size
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::unpack_cell_buffer<T,PackOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }
    const unsigned int run_block_size = min(block_size, max_block_size);

    dim3 grid(num_cells / run_block_size + 1);
    mpcd::gpu::kernel::unpack_cell_buffer<<<grid, run_block_size>>>(d_props,
                                                                    d_cells,
                                                                    d_recv,
                                                                    d_recv_begin,
                                                                    d_recv_end,
                                                                    d_recv_buf,
                                                                    op,
                                                                    num_cells);

    return cudaSuccess;
    }
#endif // NVCC

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_CELL_COMMUNICATOR_CUH_
