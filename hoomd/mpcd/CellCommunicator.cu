// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellCommunicator.cu
 * \brief Explicit instantiation of kernel drives for mpcd::CellCommunicator
 */

#include "CellCommunicator.cuh"
#include "ReductionOperators.h"

namespace mpcd
{
namespace gpu
{

//! Explicit template instantiation of pack for test/cell_communicator_mpi.cc
template cudaError_t pack_cell_buffer(int3 *d_left_buf,
                                      int3 *d_right_buf,
                                      const Index3D& left_idx,
                                      const Index3D& right_idx,
                                      const uint3& right_offset,
                                      const int3 *d_props,
                                      const Index3D& ci,
                                      unsigned int block_size);

//! Explicit template instantiation of unpack for test/cell_communicator_mpi.cc
template cudaError_t unpack_cell_buffer(int3 *d_props,
                                        mpcd::ops::Sum reduction_op,
                                        const Index3D& ci,
                                        const int3 *d_left_buf,
                                        const int3 *d_right_buf,
                                        const Index3D& left_idx,
                                        const Index3D& right_idx,
                                        const uint3& right_offset,
                                        const unsigned int block_size);

} // end namespace gpu
} // end namespace mpcd
