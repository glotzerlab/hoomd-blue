// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellCommunicator.cuh
 * \brief Declaration of CUDA kernels for mpcd::CellCommunicator
 */

#ifndef MPCD_CELL_COMMUNICATOR_CUH_
#define MPCD_CELL_COMMUNICATOR_CUH_

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

namespace mpcd
{
namespace gpu
{

//! Kernel driver to pack cell communication buffers
template<typename T, class PackOpT>
cudaError_t pack_cell_buffer(typename PackOpT::element *d_left_buf,
                             typename PackOpT::element *d_right_buf,
                             const Index3D& left_idx,
                             const Index3D& right_idx,
                             const uint3& right_offset,
                             const T *d_props,
                             PackOpT pack_op,
                             const Index3D& ci,
                             unsigned int block_size);

//! Kernel driver to unpack cell communication buffers
template<typename T, class PackOpT>
cudaError_t unpack_cell_buffer(T *d_props,
                               PackOpT pack_op,
                               const Index3D& ci,
                               const typename PackOpT::element *d_left_buf,
                               const typename PackOpT::element *d_right_buf,
                               const Index3D& left_idx,
                               const Index3D& right_idx,
                               const uint3& right_offset,
                               const unsigned int block_size);

#ifdef NVCC

namespace kernel
{
//! Kernel to pack cell communication buffers
/*!
 * \param d_left_buf Buffer to pack for sending the left direction (output)
 * \param d_right_buf Buffer to pack for sending the right direction (output)
 * \param left_idx Indexer into left buffer
 * \param right_idx Indexer into right buffer
 * \param right_offset Offset of where the begin reading the right buffer out of the cell data
 * \param d_props Cell property to pack
 * \param ci Cell list indexer
 * \param N_tot Total number of cells to pack
 *
 * \tparam T Type of data to pack (inferred)
 *
 * \b Implementation details:
 * Using one thread per cell to pack, the 1d thread index is converted back to
 * a 3d cell index. This index is in turn used to read the data to pack.
 */
template<typename T, class PackOpT>
__global__ void pack_cell_buffer(typename PackOpT::element *d_left_buf,
                                 typename PackOpT::element *d_right_buf,
                                 const Index3D left_idx,
                                 const Index3D right_idx,
                                 const uint3 right_offset,
                                 const T *d_props,
                                 PackOpT pack_op,
                                 const Index3D ci,
                                 const unsigned int N_tot)
    {
    // one thread per buffer cell
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    // left buffer
    if (idx < left_idx.getNumElements())
        {
        uint3 cell = left_idx.getTriple(idx);
        d_left_buf[idx] = pack_op.pack(d_props[ci(cell.x, cell.y, cell.z)]);
        }
    else // right buffer
        {
        idx -= left_idx.getNumElements();
        uint3 cell = right_idx.getTriple(idx);
        d_right_buf[idx] = pack_op.pack(d_props[ci(right_offset.x+cell.x,right_offset.y+cell.y,right_offset.z+cell.z)]);
        }
    }

//! Kernel to unpack cell communication buffers
/*!
 * \param d_props Cell property array to unpack into
 * \param reduction_op Commutative reduction operator to use when applying unpack
 * \param ci Cell list indexer
 * \param d_left_buf Buffer from left face to unpack
 * \param d_right_buf Buffer from right face to unpack
 * \param left_idx Indexer into left face buffer
 * \param right_idx Indexer into right face buffer
 * \param right_offset Offset for mapping right buffer into cell list
 * \param N_tot Total number of cells to unpack
 *
 * \tparam T Data type to unpack (inferred)
 * \tparam ReductionOpT Commutative reduction operator type (inferred)
 *
 * \b Implementation details:
 * Using one thread per cell to unpack, the 1d thread index is converted to a
 * buffer cell. This cell is mapped into the 3d cell list, and the \a reduction_op
 * is applied to the current value of the cell.
 *
 */
template<typename T, class PackOpT>
__global__ void unpack_cell_buffer(T *d_props,
                                   PackOpT pack_op,
                                   const Index3D ci,
                                   const typename PackOpT::element *d_left_buf,
                                   const typename PackOpT::element *d_right_buf,
                                   const Index3D left_idx,
                                   const Index3D right_idx,
                                   const uint3 right_offset,
                                   const unsigned int N_tot)
    {
    // one thread per particle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_tot)
        return;

    // left buffer
    unsigned int target;
    typename PackOpT::element add_val;
    if (idx < left_idx.getNumElements())
        {
        uint3 cell = left_idx.getTriple(idx);
        target = ci(cell.x,cell.y,cell.z);
        add_val = d_left_buf[idx];
        }
    else // right buffer
        {
        idx -= left_idx.getNumElements();
        uint3 cell = right_idx.getTriple(idx);
        target = ci(right_offset.x+cell.x,right_offset.y+cell.y,right_offset.z+cell.z);
        add_val = d_right_buf[idx];
        }

    // write out
    const T cur_val = d_props[target];
    d_props[target] = pack_op.unpack(add_val, cur_val);
    }

} // end namespace kernel

/*!
 * \param d_left_buf Buffer to pack for sending the left direction (output)
 * \param d_right_buf Buffer to pack for sending the right direction (output)
 * \param left_idx Indexer into left buffer
 * \param right_idx Indexer into right buffer
 * \param right_offset Offset of where the begin reading the right buffer out of the cell data
 * \param d_props Cell property to pack
 * \param ci Cell list indexer
 * \param block_size Number of threads per block
 *
 * \tparam T Type of data to pack (inferred)
 *
 * \returns cudaSuccess on completion
 *
 * \sa mpcd::gpu::kernel::pack_cell_buffer
 */
template<typename T, class PackOpT>
cudaError_t pack_cell_buffer(typename PackOpT::element *d_left_buf,
                             typename PackOpT::element *d_right_buf,
                             const Index3D& left_idx,
                             const Index3D& right_idx,
                             const uint3& right_offset,
                             const T *d_props,
                             PackOpT pack_op,
                             const Index3D& ci,
                             unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::pack_cell_buffer<T,PackOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // compute number of threads and blocks required
    const unsigned int N_tot = left_idx.getNumElements() + right_idx.getNumElements();
    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);

    mpcd::gpu::kernel::pack_cell_buffer<<<grid, run_block_size>>>(d_left_buf,
                                                                     d_right_buf,
                                                                     left_idx,
                                                                     right_idx,
                                                                     right_offset,
                                                                     d_props,
                                                                     pack_op,
                                                                     ci,
                                                                     N_tot);

    return cudaSuccess;
    }

/*!
 * \param d_props Cell property array to unpack into
 * \param reduction_op Commutative reduction operator to use when applying unpack
 * \param ci Cell list indexer
 * \param d_left_buf Buffer from left face to unpack
 * \param d_right_buf Buffer from right face to unpack
 * \param left_idx Indexer into left face buffer
 * \param right_idx Indexer into right face buffer
 * \param right_offset Offset for mapping right buffer into cell list
 * \param block_size Number of threads per block
 *
 * \tparam T Data type to unpack (inferred)
 * \tparam ReductionOpT Commutative reduction operator type (inferred)
 *
 * \returns cudaSuccess on completion
 *
 * The \a reduction_op should be a commutative, binary functor. Refer to
 * ReductionOperators.h for examples.
 *
 * \sa mpcd::gpu::kernel::unpack_cell_buffer
 */
template<typename T, class PackOpT>
cudaError_t unpack_cell_buffer(T *d_props,
                               PackOpT pack_op,
                               const Index3D& ci,
                               const typename PackOpT::element *d_left_buf,
                               const typename PackOpT::element *d_right_buf,
                               const Index3D& left_idx,
                               const Index3D& right_idx,
                               const uint3& right_offset,
                               const unsigned int block_size)
    {
    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void*)mpcd::gpu::kernel::unpack_cell_buffer<T,PackOpT>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // compute number of threads and blocks required
    const unsigned int N_tot = left_idx.getNumElements() + right_idx.getNumElements();
    unsigned int run_block_size = min(block_size, max_block_size);
    dim3 grid(N_tot / run_block_size + 1);

    mpcd::gpu::kernel::unpack_cell_buffer<<<grid, run_block_size>>>(d_props,
                                                                    pack_op,
                                                                    ci,
                                                                    d_left_buf,
                                                                    d_right_buf,
                                                                    left_idx,
                                                                    right_idx,
                                                                    right_offset,
                                                                    N_tot);

    return cudaSuccess;
    }
#endif // NVCC

} // end namespace gpu
} // end namespace mpcd

#endif // MPCD_CELL_COMMUNICATOR_CUH_
