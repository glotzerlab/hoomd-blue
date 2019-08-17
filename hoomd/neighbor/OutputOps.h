// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_OUTPUT_OPS_H_
#define NEIGHBOR_OUTPUT_OPS_H_

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define HOSTDEVICE
#endif

namespace neighbor
{

//! Count the number of neighbors
/*!
 * Operation to count the number of neighbors during a traversal.
 * This operation is used as a template for neighbor::gpu::lbvh_traverse_ropes.
 *
 * This operation can serve as an example for how to write your own custom
 * traversal operations. Each operation must supply the following:
 *
 *  - ThreadData: a lightweight structure for thread-local data during traversal.
 *  - setup(): a method called at the beginning of the traversal kernel.
 *  - process(): a method to process an intersected primitive.
 *  - finalize(): a method called at the end of the traversal kernel.
 *
 * See each method below for additional details of the type of functionality typically used.
 */
struct CountNeighborsOp
    {
    //! Constructor
    /*!
     * \param nneigh Number of neighbors array (output)
     *
     * The constructor is called on the host. The caller can use member variables of
     * the operation to stash pointers needed for sophisticated output workflows.
     */
    CountNeighborsOp(unsigned int* nneigh_)
        : nneigh(nneigh_)
        {}

    //! Lightweight data structure for thread-local data during traversal.
    /*!
     * Each thread needs to know which primitive it is processing and how many
     * neighbors have been found. This struct is instantiated per thread by setup(),
     * modified by process(), and saved to global memory by finalize().
     */
    struct ThreadData
        {
        //! Constructor
        /*!
         * \param idx_ The thread index of the search sphere processed by the thread.
         */
        HOSTDEVICE ThreadData(const unsigned int idx_)
            : idx(idx_), num_neigh(0)
            {}

        int idx;                //!< Index of the search sphere being processed
        unsigned int num_neigh; //!< Number of numbers found
        };

    //! Setup the thread data before traversal begins.
    /*!
     * \param idx The thread index.
     * \returns The initialized ThreadData
     *
     * Setup functions may do additional processing of variables if needed.
     */
    HOSTDEVICE ThreadData setup(const unsigned int idx) const
        {
        return ThreadData(idx);
        }

    //! Process a new primitive that is overlapped.
    /*!
     * \param t The ThreadData being operated on.
     * \param primitive The new primitive index to add.
     *
     * This method simply adds a new neighbor to the thread data. More sophisticated
     * processing (additional computations, additional memory transactions) are allowed.
     * Note that this processing step occurs deep in the traversal, and so it is advised
     * to avoid unnecessarily divergent execution.
     */
    HOSTDEVICE void process(ThreadData& t, const int primitive) const
        {
        ++t.num_neigh;
        }

    //! Finalize output operations.
    /*!
     * \param t The ThreadData being operated on.
     *
     * This method writes out the total number of neighbors found to global memory.
     * It is called at the very end of the traversal kernel, and so allows additional
     * custom output operations to be injected without significant cost during the traversal.
     */
    HOSTDEVICE void finalize(const ThreadData& t) const
        {
        nneigh[t.idx] = t.num_neigh;
        }

    unsigned int *nneigh;   //!< Number of neighbors per-search sphere
    };

//! Generate a neighbor list
/*!
 * The indexes of primitives overlapping the search spheres are saved into a list.
 * It is assumed that each particle is allowed the same maximum number of neighbors.
 * Neighbors are only written to the list if they will fit within the allocated memory.
 * Regardless, the number of neighbors is still counted, even if not all neighbors fit
 * in the list.
 */
struct NeighborListOp
    {
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int max_neigh_)
        : neigh_list(neigh_list_), nneigh(nneigh_), max_neigh(max_neigh_)
        {}

    //! Thread-local data
    struct ThreadData
        {
        HOSTDEVICE ThreadData(const unsigned int idx_, unsigned int first_)
            : idx(idx_), first(first_), num_neigh(0)
            {}

        int idx;                //!< Index of primitive
        unsigned int first;     //!< First index to use for writing neighbors
        unsigned int num_neigh; //!< Number of neighbors for this thread
        };

    //! Initialize the local ThreadData
    /*!
     * \param idx Index of search sphere.
     */
    HOSTDEVICE ThreadData setup(const unsigned int idx) const
        {
        return ThreadData(idx, max_neigh*idx);
        }


    //! Process a new primitive that is overlapped.
    /*!
     * \param t The ThreadData being operated on.
     * \param primitive The new primitive index to add.
     *
     * The new primitive is inserted into the neighbor list if it fits within
     * the allocated bounds. This involves an immediate transaction to global memory.
     * The number of neighbors is incremented regardless, but writing is defered until
     * finalize().
     */
    HOSTDEVICE void process(ThreadData& t, const int primitive) const
        {
        if (t.num_neigh < max_neigh)
            neigh_list[t.first+t.num_neigh] = primitive;
        ++t.num_neigh;
        }

    //! Finalize output operations.
    /*!
     * \param t The ThreadData being operated on.
     *
     * The number of neighbors is written to global memory.
     *
     * \todo This method should also alert the caller if the number of neighbors
     * exceeded the allocation.
     */
    HOSTDEVICE void finalize(const ThreadData& t) const
        {
        nneigh[t.idx] = t.num_neigh;
        if (t.num_neigh > max_neigh)
            {
            // write overflow condition
            }
        }

    unsigned int* neigh_list;   //!< Neighbors of each sphere
    unsigned int* nneigh;       //!< Number of neighbors per search sphere
    unsigned int max_neigh;     //!< Maximum number of neighbors allocated per sphere
    };

} // end namespace neighbor

#undef HOSTDEVICE
#endif // NEIGHBOR_OUTPUT_OPS_H_
