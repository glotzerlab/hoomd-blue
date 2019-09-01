// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

#ifndef __NEIGHBORLISTGPUTREE_CUH__
#define __NEIGHBORLISTGPUTREE_CUH__

/*! \file NeighborListGPUTree.cuh
    \brief Declares GPU kernel code for neighbor list tree traversal on the GPU
*/

#include <cuda_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

#include "hoomd/neighbor/InsertOps.h"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

struct NullOp
    {
    #ifdef NVCC
    DEVICE neighbor::BoundingBox get(const unsigned int idx) const
        {
        const Scalar3 p = make_scalar3(0,0,0);
        return neighbor::BoundingBox(p,p);
        }
    #endif

    HOSTDEVICE unsigned int size() const
        {
        return 0;
        }
    };

struct PointMapInsertOp : public neighbor::PointInsertOp
    {
    PointMapInsertOp(const Scalar4 *points_, const unsigned int *map_, unsigned int N_)
        : neighbor::PointInsertOp(points_, N_), map(map_)
        {}

    #ifdef NVCC
    DEVICE neighbor::BoundingBox get(const unsigned int idx) const
        {
        const Scalar4 point = points[map[idx]];
        const Scalar3 p = make_scalar3(point.x, point.y, point.z);

        // construct the bounding box for a point
        return neighbor::BoundingBox(p,p);
        }
    #endif

    const unsigned int *map;
    };

const unsigned int NeigborListTypeSentinel = 0xffffffff;

//! Kernel driver to generate morton code-type keys for particles and reorder by type
cudaError_t gpu_nlist_mark_types(unsigned int *d_types,
                                 unsigned int *d_indexes,
                                 unsigned int *d_lbvh_errors,
                                 const Scalar4 *d_pos,
                                 const unsigned int N,
                                 const unsigned int nghosts,
                                 const BoxDim& box,
                                 const Scalar3 ghost_width,
                                 const unsigned int block_size);

uchar2 gpu_nlist_sort_types(void *d_tmp,
                            size_t &tmp_bytes,
                            unsigned int *d_types,
                            unsigned int *d_sorted_types,
                            unsigned int *d_indexes,
                            unsigned int *d_sorted_indexes,
                            const unsigned int N);

cudaError_t gpu_nlist_count_types(unsigned int *d_first,
                                  unsigned int *d_last,
                                  const unsigned int *d_types,
                                  const unsigned int ntypes,
                                  const unsigned int N,
                                  const unsigned int block_size);

#undef DEVICE
#undef HOSTDEVICE

#endif //__NEIGHBORLISTGPUTREE_CUH__
