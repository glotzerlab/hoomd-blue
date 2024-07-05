// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __NEIGHBORLISTGPUTREE_CUH__
#define __NEIGHBORLISTGPUTREE_CUH__

/*! \file NeighborListGPUTree.cuh
    \brief Declares GPU kernel code for neighbor list tree traversal on the GPU
*/

#include <hip/hip_runtime.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/ParticleData.cuh"

// forward declaration
namespace neighbor
    {
class LBVH;
class LBVHTraverser;
    } // namespace neighbor

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Sentinel for an invalid particle (e.g., ghost)
const unsigned int NeighborListTypeSentinel = 0xffffffff;

//! Kernel driver to generate morton code-type keys for particles and reorder by type
hipError_t gpu_nlist_mark_types(unsigned int* d_types,
                                unsigned int* d_indexes,
                                unsigned int* d_lbvh_errors,
                                Scalar4* d_last_pos,
                                const Scalar4* d_pos,
                                const unsigned int N,
                                const unsigned int nghosts,
                                const BoxDim& box,
                                const Scalar3 ghost_width,
                                const unsigned int block_size);

//! Kernel driver to sort particles by type
uchar2 gpu_nlist_sort_types(void* d_tmp,
                            size_t& tmp_bytes,
                            unsigned int* d_types,
                            unsigned int* d_sorted_types,
                            unsigned int* d_indexes,
                            unsigned int* d_sorted_indexes,
                            const unsigned int N,
                            const unsigned int num_bits);

//! Kernel driver to count particles by type
hipError_t gpu_nlist_count_types(unsigned int* d_first,
                                 unsigned int* d_last,
                                 const unsigned int* d_types,
                                 const unsigned int ntypes,
                                 const unsigned int N,
                                 const unsigned int block_size);

//! Kernel driver to rearrange primitives for faster traversal
hipError_t gpu_nlist_copy_primitives(unsigned int* d_traverse_order,
                                     const unsigned int* d_indexes,
                                     const unsigned int* d_primitives,
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

    /// Destructor
    ~LBVHWrapper();

    //! Setup the LBVH
    void setup(const Scalar4* points, const unsigned int* map, unsigned int N, hipStream_t stream);

    //! Build the LBVH
    void build(const Scalar4* points,
               const unsigned int* map,
               unsigned int N,
               const Scalar3& lo,
               const Scalar3& hi,
               hipStream_t stream,
               unsigned int block_size);

    //! Get the underlying LBVH
    neighbor::LBVH* get()
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
    // Storing a bare pointer here because CUDA 11.5 fails to compile
    // std::shared_ptr<neighbor::LBVH>

    /// Underlying neighbor::LBVH
    neighbor::LBVH* lbvh_;
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
        size_t* first_neigh;
        unsigned int max_neigh;
        };

    //! Constructor
    LBVHTraverserWrapper();

    /// Destructor
    ~LBVHTraverserWrapper();

    //! Setup the LBVH traverser
    void setup(const unsigned int* map, neighbor::LBVH& lbvh, hipStream_t stream);

    //! Traverse the LBVH
    void traverse(TraverserArgs& args,
                  neighbor::LBVH& lbvh,
                  const Scalar3* images,
                  const unsigned int Nimages,
                  hipStream_t stream,
                  unsigned int block_size);

    //! Get the list of tunable parameters
    std::vector<unsigned int> getTunableParameters() const;

    private:
    // Storing a bare pointer here because CUDA 11.5 fails to compile
    // std::shared_ptr<neighbor::LBVHTraverser>

    neighbor::LBVHTraverser* trav_; //!< Underlying neighbor::LBVHTraverser
    };

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__NEIGHBORLISTGPUTREE_CUH__
