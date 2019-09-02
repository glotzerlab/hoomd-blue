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

#include "hoomd/neighbor/BoundingVolumes.h"
#include "hoomd/neighbor/InsertOps.h"
#include "hoomd/neighbor/TransformOps.h"

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

struct ParticleQueryOp
    {
    ParticleQueryOp(const Scalar4 *positions_,
                    const unsigned int *bodies_,
                    const Scalar *diams_,
                    const unsigned int* map_,
                    unsigned int N_,
                    const Scalar *rcut_,
                    const Scalar rbuff_,
                    const Scalar rpad_,
                    const Index2D type_index_,
                    const unsigned int lbvh_type_)
        : positions(positions_), bodies(bodies_), diams(diams_), map(map_), N(N_),
          rcut(rcut_), rbuff(rbuff_), rpad(rpad_), type_index(type_index_), lbvh_type(lbvh_type_)
        {}

    #ifdef NVCC
    struct ThreadData
        {
        HOSTDEVICE ThreadData(Scalar3 position_,
                              Scalar R_,
                              int idx_,
                              unsigned int type_,
                              unsigned int body_,
                              Scalar diam_,
                              Scalar rc_)
            : position(position_), R(R_), idx(idx_), type(type_), body(body_), diam(diam_), rc(rc_)
            {}

        Scalar3 position;
        Scalar R;
        int idx;
        unsigned int type;
        unsigned int body;
        Scalar diam;
        Scalar rc;
        };
    typedef SkippableBoundingSphere Volume;

    DEVICE ThreadData setup(const unsigned int idx) const
        {
        const unsigned int pidx = map[idx];

        const Scalar4 position = positions[pidx];
        const Scalar3 r = make_scalar3(position.x, position.y, position.z);
        const unsigned int type = __scalar_as_int(position.w);

        Scalar rc = rcut[type_index(type,lbvh_type)];
        Scalar rl;
        if (rc > Scalar(0.0))
            {
            rc += rbuff;
            rl = rc + rpad;
            }
        else
            {
            rc = Scalar(-1.0);
            rl = Scalar(-1.0);
            }

        const unsigned int body = (bodies != NULL) ? __ldg(bodies + pidx) : 0xffffffff;
        const Scalar diam = (diams != NULL) ? __ldg(diams + pidx) : Scalar(1.0);

        return ThreadData(r, rl, pidx, type, body, diam, rc);
        }

    DEVICE Volume get(const ThreadData& q, const Scalar3& image) const
        {
        return Volume(q.position+image,q.R);
        }

    DEVICE bool overlap(const Volume& v, const neighbor::BoundingBox& box) const
        {
        return v.overlap(box);
        }

    DEVICE bool refine(const ThreadData& q, const int primitive) const
        {
        bool exclude = (q.idx == primitive);

        // body exclusion
        if (bodies != NULL && q.body != 0xffffffff)
            {
            const unsigned int body = __ldg(bodies + primitive);
            exclude |= (q.body == body);
            }

        // diameter exclusion
        if (diams != NULL && !exclude)
            {
            const Scalar4 position = positions[primitive];
            const Scalar3 r = make_scalar3(position.x, position.y, position.z);
            const Scalar diam = diams[primitive];

            // compute factor to add to base rc
            const Scalar delta = (q.diam + diam) * Scalar(0.5) - Scalar(1.0);
            const Scalar sqshift = (delta + Scalar(2.0) * q.rc) * delta;

            // compute distance and wrap back into box
            const Scalar3 dr = r - q.position;
            const Scalar drsq = dot(dr,dr);

            // exclude if outside the sphere
            exclude |= drsq > (q.rc*q.rc + sqshift);
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
    const Scalar *rcut;
    Scalar rbuff;
    Scalar rpad;
    Index2D type_index;
    unsigned int lbvh_type;
    };

struct NeighborListOp
    {
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int* new_max_neigh_,
                   const unsigned int* first_neigh_,
                   const unsigned int* max_neigh_)
        : neigh_list(neigh_list_), nneigh(nneigh_), new_max_neigh(new_max_neigh_),
          first_neigh(first_neigh_), max_neigh(max_neigh_)
        {}

    //! Thread-local data
    struct ThreadData
        {
        HOSTDEVICE ThreadData(const unsigned int idx_, const unsigned int first_, const unsigned int max_neigh_)
            : idx(idx_), first(first_), num_neigh(0), max_neigh(max_neigh_)
            {}

        unsigned int idx;       //!< Index of primitive
        unsigned int first;     //!< First index to use for writing neighbors
        unsigned int num_neigh; //!< Number of neighbors for this thread
        unsigned int max_neigh; //!< Maximum number of neighbors
        };

    template<class QueryDataT>
    HOSTDEVICE ThreadData setup(const unsigned int idx, const QueryDataT& q) const
        {
        return ThreadData(q.idx, first_neigh[q.idx], max_neigh[q.type]);
        }

    HOSTDEVICE void process(ThreadData& t, const int primitive) const
        {
        if (t.num_neigh < t.max_neigh)
            neigh_list[t.first+t.num_neigh] = primitive;
        ++t.num_neigh;
        }

    #ifdef NVCC
    DEVICE void finalize(const ThreadData& t) const
        {
        nneigh[t.idx] = t.num_neigh;
        if (t.num_neigh > t.max_neigh)
            {
            atomicMax(new_max_neigh, t.num_neigh);
            }
        }
    #endif

    unsigned int* neigh_list;   //!< Neighbors of each sphere
    unsigned int* nneigh;       //!< Number of neighbors per search sphere
    unsigned int* new_max_neigh;    //!< New maximum number of neighbors
    const unsigned int* first_neigh;  //!< Index of first neighbor
    const unsigned int* max_neigh;     //!< Maximum number of neighbors allocated
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

cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size);

unsigned int gpu_nlist_remove_ghosts(unsigned int *d_traverse_order,
                                     const unsigned int N,
                                     const unsigned int N_own);

#undef DEVICE
#undef HOSTDEVICE

#endif //__NEIGHBORLISTGPUTREE_CUH__
