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

// forward declaration
namespace neighbor
    {
    class LBVH;
    class LBVHTraverser;
    }

//! Sentinel for an invalid particle (e.g., ghost)
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

//! Kernel driver to sort particles by type
uchar2 gpu_nlist_sort_types(void *d_tmp,
                            size_t &tmp_bytes,
                            unsigned int *d_types,
                            unsigned int *d_sorted_types,
                            unsigned int *d_indexes,
                            unsigned int *d_sorted_indexes,
                            const unsigned int N,
                            const unsigned int num_bits);

//! Kernel driver to count particles by type
cudaError_t gpu_nlist_count_types(unsigned int *d_first,
                                  unsigned int *d_last,
                                  const unsigned int *d_types,
                                  const unsigned int ntypes,
                                  const unsigned int N,
                                  const unsigned int block_size);

//! Kernel driver to rearrange primitives for faster traversal
cudaError_t gpu_nlist_copy_primitives(unsigned int *d_traverse_order,
                                      const unsigned int *d_indexes,
                                      const unsigned int *d_primitives,
                                      const unsigned int N,
                                      const unsigned int block_size);

//! Wrapper around the neighbor::LBVH class
/*!
 * This wrapper only exposes data types that are natively supported in HOOMD
 * so that all neighbor-specific templates and structs can be handled only
 * in CUDA code.
 */
class LBVHWrapper
    {
    public:
        //! Constructor
        LBVHWrapper();

        //! Setup the LBVH
        void setup(const Scalar4* points,
                   const unsigned int* map,
                   unsigned int N,
                   cudaStream_t stream);

        //! Build the LBVH
        void build(const Scalar4* points,
                   const unsigned int* map,
                   unsigned int N,
                   const Scalar3& lo,
                   const Scalar3& hi,
                   cudaStream_t stream,
                   unsigned int block_size);

        //! Get the underlying LBVH
        std::shared_ptr<neighbor::LBVH> get()
            {
            return lbvh_;
            }

        //! Get the number of particles in the LBVH
        unsigned int getN() const;

        //! Get the sorted order of the primitives from the LBVH
        const unsigned int* getPrimitives() const;

        //! Get the list of tunable parameters
        std::vector<unsigned int> getTunableParameters() const;

    private:
        std::shared_ptr<neighbor::LBVH> lbvh_;  //!< Underlying neighbor::LBVH
    };

//! Wrapper around the neighbor::LBVHTraverser class
/*!
 * This wrapper only exposes data types that are natively supported in HOOMD
 * so that all neighbor-specific templates and structs can be handled only
 * in CUDA code.
 */
class LBVHTraverserWrapper
    {
    public:
        //! Structure to group together all the parameters needed for a traversal
        struct TraverserArgs
            {
            // map
            unsigned int* map;

            // particle query
            Scalar4* positions;
            unsigned int* bodies;
            Scalar* diams;
            unsigned int* order;
            unsigned int N;
            unsigned int Nown;
            Scalar rcut;
            Scalar rlist;
            BoxDim box;

            // neighbor list
            unsigned int* neigh_list;
            unsigned int* nneigh;
            unsigned int* new_max_neigh;
            unsigned int* first_neigh;
            unsigned int max_neigh;
            };

        //! Constructor
        LBVHTraverserWrapper();

        //! Setup the LBVH traverser
        void setup(const unsigned int* map,
                   neighbor::LBVH& lbvh,
                   cudaStream_t stream);

        //! Traverse the LBVH
        void traverse(TraverserArgs& args,
                      neighbor::LBVH& lbvh,
                      const Scalar3* images,
                      const unsigned int Nimages,
                      cudaStream_t stream,
                      unsigned int block_size);

        //! Get the list of tunable parameters
        std::vector<unsigned int> getTunableParameters() const;

    private:
        std::shared_ptr<neighbor::LBVHTraverser> trav_; //!< Underlying neighbor::LBVHTraverser
    };

#endif //__NEIGHBORLISTGPUTREE_CUH__
