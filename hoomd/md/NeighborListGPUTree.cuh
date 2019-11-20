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

#include "hoomd/extern/neighbor/neighbor/BoundingVolumes.h"
#include "hoomd/extern/neighbor/neighbor/InsertOps.h"
#include "hoomd/extern/neighbor/neighbor/TransformOps.h"

#ifdef NVCC
#define DEVICE __device__ __forceinline__
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define DEVICE
#define HOSTDEVICE
#endif

struct SkippableBoundingSphere : public neighbor::BoundingSphere
    {
    HOSTDEVICE SkippableBoundingSphere() : skip(true) {}

    #ifdef NVCC
    DEVICE SkippableBoundingSphere(const Scalar3& o, const Scalar r)
        : neighbor::BoundingSphere(o,r)
        {
        skip = !(r > Scalar(0));
        }

    DEVICE bool overlap(const neighbor::BoundingBox& box) const
        {
        if (!skip)
            {
            return neighbor::BoundingSphere::overlap(box);
            }
        else
            {
            return false;
            }
        }
    #endif

    bool skip;
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

template<bool use_body, bool use_diam>
struct ParticleQueryOp
    {
    ParticleQueryOp(const Scalar4 *positions_,
                    const unsigned int *bodies_,
                    const Scalar *diams_,
                    const unsigned int* map_,
                    unsigned int N_,
                    unsigned int Nown_,
                    const Scalar rcut_,
                    const Scalar rlist_)
        : positions(positions_), bodies(bodies_), diams(diams_), map(map_),
          N(N_), Nown(Nown_), rcut(rcut_), rlist(rlist_)
          {}

    #ifdef NVCC
    struct ThreadData
        {
        HOSTDEVICE ThreadData(Scalar3 position_,
                              int idx_,
                              unsigned int body_,
                              Scalar diam_)
            : position(position_), idx(idx_), body(body_), diam(diam_)
            {}

        Scalar3 position;
        int idx;
        unsigned int body;
        Scalar diam;
        };
    typedef SkippableBoundingSphere Volume;

    DEVICE ThreadData setup(const unsigned int idx) const
        {
        const unsigned int pidx = map[idx];

        const Scalar4 position = positions[pidx];
        const Scalar3 r = make_scalar3(position.x, position.y, position.z);

        unsigned int body(0xffffffff);
        if (use_body)
            {
            body = __ldg(bodies + pidx);
            }
        Scalar diam(1.0);
        if (use_diam)
            {
            diam = __ldg(diams + pidx);
            }

        return ThreadData(r, pidx, body, diam);
        }

    DEVICE Volume get(const ThreadData& q, const Scalar3& image) const
        {
        return Volume(q.position+image, (q.idx < Nown) ? rlist : -1.0);
        }

    DEVICE bool overlap(const Volume& v, const neighbor::BoundingBox& box) const
        {
        return v.overlap(box);
        }

    DEVICE bool refine(const ThreadData& q, const int primitive) const
        {
        bool exclude = (q.idx == primitive);

        // body exclusion
        if (use_body && !exclude && q.body != 0xffffffff)
            {
            const unsigned int body = __ldg(bodies + primitive);
            exclude |= (q.body == body);
            }

        // diameter exclusion
        if (use_diam && !exclude)
            {
            const Scalar4 position = positions[primitive];
            const Scalar3 r = make_scalar3(position.x, position.y, position.z);
            const Scalar diam = diams[primitive];

            // compute factor to add to base rc
            const Scalar delta = (q.diam + diam) * Scalar(0.5) - Scalar(1.0);
            Scalar rc2 = (rcut+delta);
            rc2 *= rc2;

            // compute distance and wrap back into box
            const Scalar3 dr = r - q.position;
            const Scalar drsq = dot(dr,dr);

            // exclude if outside the sphere
            exclude |= drsq > rc2;
            }

        return !exclude;
        }
    #endif

    HOSTDEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4 *positions;
    const unsigned int *bodies;
    const Scalar *diams;
    const unsigned int *map;
    unsigned int N;
    unsigned int Nown;
    Scalar rcut;
    Scalar rlist;
    };

struct NeighborListOp
    {
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int* new_max_neigh_,
                   const unsigned int* first_neigh_,
                   unsigned int max_neigh_)
        : nneigh(nneigh_), new_max_neigh(new_max_neigh_),
          first_neigh(first_neigh_), max_neigh(max_neigh_)
        {
        neigh_list = reinterpret_cast<uint4*>(neigh_list_);
        }

    #ifdef NVCC
    //! Thread-local data
    struct ThreadData
        {
        DEVICE ThreadData(const unsigned int idx_,
                          const unsigned int first_,
                          const unsigned int num_neigh_,
                          const uint4 stack_)
            : idx(idx_), first(first_), num_neigh(num_neigh_)
            {
            stack[0] = stack_.x;
            stack[1] = stack_.y;
            stack[2] = stack_.z;
            stack[3] = stack_.w;
            }

        unsigned int idx;       //!< Index of primitive
        unsigned int first;     //!< First index to use for writing neighbors
        unsigned int num_neigh; //!< Number of neighbors for this thread
        unsigned int stack[4];
        };

    template<class QueryDataT>
    DEVICE ThreadData setup(const unsigned int idx, const QueryDataT& q) const
        {
        const unsigned int first = __ldg(first_neigh + q.idx);
        const unsigned int num_neigh = nneigh[q.idx]; // no __ldg, since this is writeable

        // prefetch from the stack if current number of neighbors does not align with a boundary
        uint4 stack = make_uint4(0,0,0,0);
        if (num_neigh % 4 != 0)
            {
            stack = neigh_list[(first+num_neigh-1)/4];
            }

        return ThreadData(q.idx, first, num_neigh, stack);
        }

    DEVICE void process(ThreadData& t, const int primitive) const
        {
        if (t.num_neigh < max_neigh)
            {
            // push primitive into the stack of 4, pre-increment
            const unsigned int offset = t.num_neigh % 4;
            t.stack[offset] = primitive;
            // coalesce writes into chunks of 4
            if (offset == 3)
                {
                neigh_list[(t.first+t.num_neigh)/4] = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
                }
            }
        ++t.num_neigh;
        }

    DEVICE void finalize(const ThreadData& t) const
        {
        nneigh[t.idx] = t.num_neigh;
        if (t.num_neigh > max_neigh)
            {
            atomicMax(new_max_neigh, t.num_neigh);
            }
        else if (t.num_neigh % 4 != 0)
            {
            // write partial (leftover) stack, counting is now post-increment so need to shift by 1
            // only need to do this if didn't overflow, since all neighbors were already written due to alignment of max
            neigh_list[(t.first+t.num_neigh-1)/4] = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
            }
        }
    #endif

    uint4* neigh_list;                  //!< Neighbors of each sphere
    unsigned int* nneigh;               //!< Number of neighbors per search sphere
    unsigned int* new_max_neigh;        //!< New maximum number of neighbors
    const unsigned int* first_neigh;    //!< Index of first neighbor
    unsigned int max_neigh;             //!< Maximum number of neighbors allocated
    };

const unsigned int NeighborListTypeSentinel = 0xffffffff;

//! Kernel driver to generate morton code-type keys for particles and reorder by type
cudaError_t gpu_nlist_mark_types(unsigned int *d_types,
                                 unsigned int *d_indexes,
                                 unsigned int *d_lbvh_errors,
                                 Scalar4 *d_last_pos,
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
                            const unsigned int N,
                            const unsigned int num_bits);

cudaError_t gpu_nlist_count_types(unsigned int *d_first,
                                  unsigned int *d_last,
                                  const unsigned int *d_types,
                                  const unsigned int ntypes,
                                  const unsigned int N,
                                  const unsigned int block_size);

cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size);

#undef DEVICE
#undef HOSTDEVICE

#endif //__NEIGHBORLISTGPUTREE_CUH__
