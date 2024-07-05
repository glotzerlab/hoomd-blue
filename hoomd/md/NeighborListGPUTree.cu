// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborListGPUTree.cuh"
#include "hip/hip_runtime.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hipcub/hipcub.hpp>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/remove.h>
#pragma GCC diagnostic pop

#include <neighbor/neighbor.h>

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel to mark particles by type
/*!
 * \param d_types Type of each particle.
 * \param d_indexes Index of each particle.
 * \param d_lbvh_errors Error flag for particles.
 * \param d_last_pos Last position of particles at neighbor list update.
 * \param d_pos Current position of particles.
 * \param N Number of locally owned particles.
 * \param nghosts Number of ghost particles.
 * \param box Simulation box for the local rank.
 * \param ghost_width Size of the ghost layer.
 *
 * Using one thread each, the positions of all particles (local + ghosts) are first
 * loaded. The particle coordinates are made into a fraction inside the box + ghost layer.
 * If a local particle lies outside this box, an error is marked. If a ghost particle is
 * outside this box, it is silently flagged as being off rank by setting its type to a sentinel.
 * The particle type (or this sentinel) is then stored along with the particle index for subsequent
 * sorting.
 *
 * The last position of each particle is also saved during this kernel, which is used to later
 * check for when the neighbor list should be rebuilt.
 */
__global__ void gpu_nlist_mark_types_kernel(unsigned int* d_types,
                                            unsigned int* d_indexes,
                                            unsigned int* d_lbvh_errors,
                                            Scalar4* d_last_pos,
                                            const Scalar4* d_pos,
                                            const unsigned int N,
                                            const unsigned int nghosts,
                                            const BoxDim box,
                                            const Scalar3 ghost_width)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N + nghosts)
        return;

    // acquire particle data
    Scalar4 postype = d_pos[idx];
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);

    // get position in simulation box
    uchar3 periodic = box.getPeriodic();
    Scalar3 f = box.makeFraction(pos, ghost_width);

    /*
     * check if the particle is inside the unit cell + ghost layer.
     * we silently ignore ghosts outside of this width, since they could be in bonds, etc.
     */
    if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001))
        || (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001))
        || (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
        {
        // error for owned particle
        if (idx < N)
            {
            atomicMax(d_lbvh_errors, idx + 1);
            return;
            }
        else // silent for ghosts
            {
            type = NeighborListTypeSentinel;
            }
        }

    d_types[idx] = type;
    d_indexes[idx] = idx;
    // record as "last" position of owned particles
    if (idx < N)
        {
        d_last_pos[idx] = postype;
        }
    }

/*!
 * \param d_types Type of each particle.
 * \param d_indexes Index of each particle.
 * \param d_lbvh_errors Error flag for particles.
 * \param d_last_pos Last position of particles at neighbor list update.
 * \param d_pos Current position of particles.
 * \param N Number of locally owned particles.
 * \param nghosts Number of ghost particles.
 * \param box Simulation box for the local rank.
 * \param ghost_width Size of the ghost layer.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa gpu_nlist_mark_types_kernel
 */
hipError_t gpu_nlist_mark_types(unsigned int* d_types,
                                unsigned int* d_indexes,
                                unsigned int* d_lbvh_errors,
                                Scalar4* d_last_pos,
                                const Scalar4* d_pos,
                                const unsigned int N,
                                const unsigned int nghosts,
                                const BoxDim& box,
                                const Scalar3 ghost_width,
                                const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(gpu_nlist_mark_types_kernel));
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(block_size, max_block_size);
    const unsigned int num_blocks = ((N + nghosts) + run_block_size - 1) / run_block_size;
    hipLaunchKernelGGL(gpu_nlist_mark_types_kernel,
                       dim3(num_blocks),
                       dim3(run_block_size),
                       0,
                       0,
                       d_types,
                       d_indexes,
                       d_lbvh_errors,
                       d_last_pos,
                       d_pos,
                       N,
                       nghosts,
                       box,
                       ghost_width);
    return hipSuccess;
    }

/*!
 * \param d_tmp Temporary memory for sorting.
 * \param tmp_bytes Number of temporary bytes for sorting.
 * \param d_types The particle types to sort.
 * \param d_sorted_types The sorted particle types.
 * \param d_indexes The particle indexes to sort.
 * \param d_sorted_indexes The sorted particle indexes.
 * \param N Number of particle types to sort.
 * \param num_bits Number of bits required to sort the type (usually a small number).
 * \returns A pair of flags saying if the output data needs to be swapped with the input.
 *
 * The sorting is done using CUB with the DoubleBuffer. On the first call, the temporary memory
 * is sized. On the second call, the sorting is actually performed. The sorted data may not actually
 * lie in \a d_sorted_* because of the double buffers, but this sorting seems to be more efficient.
 * The user should accordingly swap the input and output arrays if the returned values are true.
 */
uchar2 gpu_nlist_sort_types(void* d_tmp,
                            size_t& tmp_bytes,
                            unsigned int* d_types,
                            unsigned int* d_sorted_types,
                            unsigned int* d_indexes,
                            unsigned int* d_sorted_indexes,
                            const unsigned int N,
                            const unsigned int num_bits)
    {
    hipcub::DoubleBuffer<unsigned int> d_keys(d_types, d_sorted_types);
    hipcub::DoubleBuffer<unsigned int> d_vals(d_indexes, d_sorted_indexes);

    // we counted number of bits to sort, so the range of bit indexes is [0,num_bits)
    hipcub::DeviceRadixSort::SortPairs(d_tmp, tmp_bytes, d_keys, d_vals, N, 0, num_bits);

    uchar2 swap = make_uchar2(0, 0);
    if (d_tmp != NULL)
        {
        // mark that the gpu arrays should be flipped if the final result is not in the sorted array
        // (1)
        swap.x = (d_keys.selector == 0);
        swap.y = (d_vals.selector == 0);
        }
    return swap;
    }

//! Kernel to count the number of each particle type.
/*!
 * \param d_first First index of particles of a given type.
 * \param d_last Last index of particles of a given type.
 * \param d_types Type of each particle.
 * \param ntypes Number of particle types.
 * \param N Number of particles.
 *
 * This kernel actually marks the beginning (first) and end (last) range of the
 * particles of each type. Counting can then be done by taking the difference between
 * first and last. This demarcations are done by looking left/right from the current
 * particle, using appropriate sentinels for the end of each run.
 *
 * The kernel relies on the data being prefilled correctly. These fillings should be
 * NeighborListTypeSentinel for \a d_first and 0 for \a d_last so that the default (if no
 * particle of a given type is found) is that there are 0 of them in the list.
 */
__global__ void gpu_nlist_count_types_kernel(unsigned int* d_first,
                                             unsigned int* d_last,
                                             const unsigned int* d_types,
                                             const unsigned int ntypes,
                                             const unsigned int N)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // one thread per particle
    if (idx >= N)
        return;

    // my type
    const unsigned int type = d_types[idx];
    // look to left if not first
    const unsigned int left = (idx > 0) ? d_types[idx - 1] : NeighborListTypeSentinel;
    // look to right if not last
    const unsigned int right = (idx < N - 1) ? d_types[idx + 1] : NeighborListTypeSentinel;

    // if left is not same as self (or idx == 0 by use of sentinel), this is the first index in the
    // type
    if (left != type && type < ntypes)
        {
        d_first[type] = idx;
        }
    // if right is not the same as self (or idx == N-1 by use of sentinel), this is the last index
    // in the type
    if (right != type && type < ntypes)
        {
        d_last[type] = idx + 1;
        }
    }

/*!
 * \param d_first First index of particles of a given type.
 * \param d_last Last index of particles of a given type.
 * \param d_types Type of each particle.
 * \param ntypes Number of particle types.
 * \param N Number of particles.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa gpu_nlist_count_types_kernel
 */
hipError_t gpu_nlist_count_types(unsigned int* d_first,
                                 unsigned int* d_last,
                                 const unsigned int* d_types,
                                 const unsigned int ntypes,
                                 const unsigned int N,
                                 const unsigned int block_size)

    {
    // initially, fill all types as empty
    thrust::fill(thrust::device, d_first, d_first + ntypes, NeighborListTypeSentinel);
    hipMemset(d_last, 0, sizeof(unsigned int) * ntypes);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(gpu_nlist_count_types_kernel));
    max_block_size = attr.maxThreadsPerBlock;

    int run_block_size = min(block_size, max_block_size);
    hipLaunchKernelGGL(gpu_nlist_count_types_kernel,
                       dim3(N / run_block_size + 1),
                       dim3(run_block_size),
                       0,
                       0,
                       d_first,
                       d_last,
                       d_types,
                       ntypes,
                       N);
    return hipSuccess;
    }

//! Kernel to copy the particle indexes into traversal order
/*!
 * \param d_traverse_order List of particle indexes in traversal order.
 * \param d_indexes Original indexes of the sorted primitives.
 * \param d_primitives List of the primitives (sorted in LBVH order).
 * \param N Number of primitives.
 *
 * The primitive index for this thread is first loaded. It is then mapped back
 * to its original particle index, which is stored for subsequent traversal.
 */
__global__ void gpu_nlist_copy_primitives_kernel(unsigned int* d_traverse_order,
                                                 const unsigned int* d_indexes,
                                                 const unsigned int* d_primitives,
                                                 const unsigned int N)
    {
    // one thread per particle
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    const unsigned int primitive = d_primitives[idx];
    d_traverse_order[idx] = __ldg(d_indexes + primitive);
    }

/*!
 * \param d_traverse_order List of particle indexes in traversal order.
 * \param d_indexes Original indexes of the sorted primitives.
 * \param d_primitives List of the primitives (sorted in LBVH order).
 * \param N Number of primitives.
 * \param block_size Number of CUDA threads per block.
 *
 * \sa gpu_nlist_copy_primitives_kernel
 */
hipError_t gpu_nlist_copy_primitives(unsigned int* d_traverse_order,
                                     const unsigned int* d_indexes,
                                     const unsigned int* d_primitives,
                                     const unsigned int N,
                                     const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(gpu_nlist_copy_primitives_kernel));
    max_block_size = attr.maxThreadsPerBlock;

    int run_block_size = min(block_size, max_block_size);
    hipLaunchKernelGGL(gpu_nlist_copy_primitives_kernel,
                       dim3(N / run_block_size + 1),
                       dim3(run_block_size),
                       0,
                       0,
                       d_traverse_order,
                       d_indexes,
                       d_primitives,
                       N);
    return hipSuccess;
    }

/////////////////////////////////////
// neighbor program and wrappers
/////////////////////////////////////

#define DEVICE __device__ __forceinline__

//! A bounding sphere that can skip traversal
/*!
 * Extends the base neighbor::BoundingSphere to be skipped if the input
 * radius is negative. This is useful for the traversal when a particle is
 * a ghost so that we don't have to do an extra sort to filter them out.
 */
struct SkippableBoundingSphere : public neighbor::BoundingSphere
    {
    //! Default constructor, always skip
    DEVICE SkippableBoundingSphere() : skip(true) { }

    //! Constructor
    /*!
     * \param o Center of sphere.
     * \param r Radius of sphere.
     *
     * The sphere is made to be skipped if \a r is negative. This
     * is stored in a separate variable because the base class squares
     * the radius.
     */
    DEVICE SkippableBoundingSphere(const Scalar3& o, const Scalar r)
        : neighbor::BoundingSphere(o, r)
        {
        skip = !(r > Scalar(0));
        }

    //! Overlap test
    /*!
     * \param box Box to test overlap with.
     * \returns True if the sphere overlaps the \a box, and the sphere is
     *          not made to be skipped.
     */
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

    bool skip; //!< Flag to skip traversal of sphere
    };

//! Insert operation for a point under a mapping.
/*!
 * Extends the base neighbor::PointInsertOp to insert a point primitive
 * subject to a mapping of the indexes. This is useful for reading from
 * the array of particles that is pre-sorted by type so that the original
 * particle data does not need to be shuffled.
 */
struct PointMapInsertOp
    {
    //! Constructor
    /*!
     * \param points_ List of points to insert (w entry is unused).
     * \param map_ Map of the nominal index to the index in \a points_.
     * \param N_ Number of primitives to insert.
     */
    PointMapInsertOp(const Scalar4* points_, const unsigned int* map_, unsigned int N_)
        : points(points_), map(map_), N(N_)
        {
        }

    //! Construct bounding box
    /*!
     * \param idx Nominal index of the primitive [0,N).
     * \returns A neighbor::BoundingBox corresponding to the point at map[idx].
     */
    DEVICE neighbor::BoundingBox get(const unsigned int idx) const
        {
        const Scalar4 point = points[map[idx]];
        const Scalar3 p = make_scalar3(point.x, point.y, point.z);

        // construct the bounding box for a point
        return neighbor::BoundingBox(p, p);
        }

    __host__ DEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4* points;
    const unsigned int* map; //!< Map of particle indexes.
    const unsigned int N;
    };

//! Neighbor list particle query operation.
/*!
 * \tparam use_body If true, use the body fields during query.
 *
 * This operation specifies the neighbor list traversal scheme. The
 * query is between a SkippableBoundingSphere and the bounding boxes in
 * the LBVH. The template parameters can be activated to engage body-filtering
 * which is defined elsewhere in HOOMD.
 *
 * All spheres in the traversal are given the same search radius. This is compatible
 * with a traversal-per-type-per-type scheme. It was found that passing this radius
 * as a constant to the traversal program decreased register pressure in the kernel
 * from a traversal-per-type scheme.
 *
 * The particles are traversed using a \a map. Ghost particles can be included
 * in this map, and they will be neglected during traversal.
 */
template<bool use_body> struct ParticleQueryOp
    {
    //! Constructor
    /*!
     * \param positions_ Particle positions.
     * \param bodies_ Particle body tags.
     * \param map_ Map of the particle indexes to traverse.
     * \param N_ Number of particles (total).
     * \param Nown_ Number of locally owned particles.
     * \param rcut_ Cutoff radius for the spheres.
     * \param rlist_ Total search radius for the spheres (differs under shifting).
     */
    ParticleQueryOp(const Scalar4* positions_,
                    const unsigned int* bodies_,
                    const unsigned int* map_,
                    unsigned int N_,
                    unsigned int Nown_,
                    const Scalar rcut_,
                    const Scalar rlist_,
                    const BoxDim& box_)
        : positions(positions_), bodies(bodies_), map(map_), N(N_), Nown(Nown_), rcut(rcut_),
          rlist(rlist_), box(box_)
        {
        }

    //! Data stored per thread for traversal
    /*!
     * The body tags are only actually set if these are specified
     * by the template parameters. The compiler might be able to optimize them
     * out if they are unused.
     */
    struct ThreadData
        {
        DEVICE ThreadData(Scalar3 position_, int idx_, unsigned int body_)
            : position(position_), idx(idx_), body(body_)
            {
            }

        Scalar3 position;  //!< Particle position
        int idx;           //!< True particle index
        unsigned int body; //!< Particle body tag (may be invalid)
        };

    // specify that the traversal Volume is a bounding sphere
    typedef SkippableBoundingSphere Volume;

    //! Loads the per-thread data
    /*!
     * \param idx Nominal primitive index.
     * \returns The ThreadData required for traversal.
     *
     * The ThreadData is loaded subject to a mapping. The particle position
     * is always loaded. The body is only loaded if the template
     * parameter requires it.
     */
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

        return ThreadData(r, pidx, body);
        }

    //! Return the traversal volume subject to a translation
    /*!
     * \param q The current thread data.
     * \param image The image vector for traversal.
     * \returns The traversal bounding volume.
     *
     * The ThreadData is converted to a search volume. The search sphere is
     * made to be skipped if this is a ghost particle.
     */
    DEVICE Volume get(const ThreadData& q, const Scalar3& image) const
        {
        return Volume(q.position + image, (q.idx < Nown) ? rlist : -1.0);
        }

    //! Perform the overlap test with the LBVH
    /*!
     * \param v Traversal volume.
     * \param box Box in LBVH to intersect with.
     * \returns True if the volume and box overlap.
     *
     * The overlap test is implemented by the sphere.
     */
    DEVICE bool overlap(const Volume& v, const neighbor::BoundingBox& box) const
        {
        return v.overlap(box);
        }

    //! Refine the rough overlap test with a primitive
    /*!
     * \param q The current thread data.
     * \param primitive Index of the intersected primitive.
     * \returns True If the volumes still overlap after refinement.
     *
     * HOOMD's neighbor lists require additional filtering. This first ensures
     * that the overlap is not with itself. If body filtering is enabled,
     * particles in the same body do not overlap.
     */
    DEVICE bool refine(const ThreadData& q, const int primitive) const
        {
        bool exclude = (q.idx == primitive);

        // body exclusion
        if (use_body && !exclude && q.body != 0xffffffff)
            {
            const unsigned int body = __ldg(bodies + primitive);
            exclude |= (q.body == body);
            }

        return !exclude;
        }

    //! Get the number of primitives
    __host__ DEVICE unsigned int size() const
        {
        return N;
        }

    const Scalar4* positions;   //!< Particle positions
    const unsigned int* bodies; //!< Particle bodies
    const unsigned int* map;    //!< Mapping of particles to read
    unsigned int N;             //!< Total number of particles in map
    unsigned int Nown;          //!< Number of particles owned by the local rank
    Scalar rcut;                //!< True cutoff radius + buffer
    Scalar rlist;               //!< Maximum cutoff (may include shifting)
    const BoxDim box;           //!< Box dimensions
    };

//! Operation to write the neighbor list
/*!
 * The neighbor list is assumed to be aligned to multiples of 4. This enables
 * coalescing writes into packets of 4 neighbors without adding much register pressure.
 * This object maintains an internal stack to do this, and it can restart from a previous
 * traversal without losing information.
 */
struct NeighborListOp
    {
    //! Constructor
    /*!
     * \param neigh_list_ Neighbor list (aligned to multiple of 4)
     * \param nneigh_ Neighbor of neighbors per particle
     * \param new_max_neigh_ Maximum number of neighbors to allocate if overflow occurs.
     * \param first_neigh_ First index for the current particle index in the neighbor list.
     * \param max_neigh_ Maximum number of neighbors to allow per particle.
     *
     * The \a neigh_list_ pointer is internally cast into a uint4 for coalescing.
     */
    NeighborListOp(unsigned int* neigh_list_,
                   unsigned int* nneigh_,
                   unsigned int* new_max_neigh_,
                   const size_t* first_neigh_,
                   unsigned int max_neigh_)
        : nneigh(nneigh_), new_max_neigh(new_max_neigh_), first_neigh(first_neigh_),
          max_neigh(max_neigh_)
        {
        neigh_list = reinterpret_cast<uint4*>(neigh_list_);
        }

    //! Thread-local data
    /*!
     * The thread-local data constitutes a stack of neighbors to write, the index of the current
     * primitive, the first index to write into, and the current number of neighbors found for this
     * thread.
     */
    struct ThreadData
        {
        //! Constructor
        /*!
         * \param idx_ The index of this particle.
         * \param first_ The first neighbor index of this particle.
         * \param num_neigh_ The current number of neighbors of this particle.
         * \param stack_ The initial values for the stack (can be all 0s if \a num_neigh_ is aligned
         * to 4).
         */
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
        size_t first;           //!< First index to use for writing neighbors
        unsigned int num_neigh; //!< Number of neighbors for this thread
        unsigned int stack[4];  //!< Internal stack of neighbors
        };

    //! Setup the thread data
    /*!
     * \param idx Index of this thread.
     * \param q Thread-local query data.
     * \returns The ThreadData for output.
     *
     * \tparam Type of QueryData.
     *
     * This setup function can poach data from the query data in order to save loads.
     * In this case, it makes use of the particle index mapping.
     */
    template<class QueryDataT>
    DEVICE ThreadData setup(const unsigned int idx, const QueryDataT& q) const
        {
        const size_t first = __ldg(first_neigh + q.idx);
        const unsigned int num_neigh = nneigh[q.idx]; // no __ldg, since this is writeable

        // prefetch from the stack if current number of neighbors does not align with a boundary
        /* NOTE: There seemed to be a compiler error/bug when stack was declared outside this if
                 statement, initialized with zeros, and then assigned inside (so that only
                 one return statement was needed). It went away using:

                 uint4 tmp = neigh_list[...];
                 stack = tmp;

                 But this looked funny, so the structure below seems more human readable.
         */
        if (num_neigh % 4 != 0)
            {
            uint4 stack = neigh_list[(first + num_neigh - 1) / 4];
            return ThreadData(q.idx, first, num_neigh, stack);
            }
        else
            {
            return ThreadData(q.idx, first, num_neigh, make_uint4(0, 0, 0, 0));
            }
        }

    //! Processes a newly intersected primitive.
    /*!
     * \param t My output thread data.
     * \param primitive The index of the primitive to process.
     *
     * If the neighbor will fit into the allocated memory, it is pushed onto the stack.
     * The stack is written to memory if it is full. The number of neighbors found for this
     * thread is incremented, regardless.
     */
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
                neigh_list[(t.first + t.num_neigh) / 4]
                    = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
                }
            }
        ++t.num_neigh;
        }

    //! Finish the output job once the thread is ready to terminate.
    /*!
     * \param t My output thread data
     *
     * The number of neighbors found for this thread is written. If this value
     * exceeds the current allocation, this value is atomically maximized for
     * reallocation. Any values remaining on the stack are written to ensure the
     * list is complete.
     */
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
            // only need to do this if didn't overflow, since all neighbors were already written due
            // to alignment of max
            neigh_list[(t.first + t.num_neigh - 1) / 4]
                = make_uint4(t.stack[0], t.stack[1], t.stack[2], t.stack[3]);
            }
        }

    uint4* neigh_list;           //!< Neighbors of each sphere
    unsigned int* nneigh;        //!< Number of neighbors per search sphere
    unsigned int* new_max_neigh; //!< New maximum number of neighbors
    const size_t* first_neigh;   //!< Index of first neighbor
    unsigned int max_neigh;      //!< Maximum number of neighbors allocated
    };

//! Host function to convert a double to a float in round-down mode
float double2float_rd(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) > x)
        {
        xf = std::nextafterf(xf, -std::numeric_limits<float>::infinity());
        }
    return xf;
    }

//! Host function to convert a double to a float in round-up mode
float double2float_ru(double x)
    {
    float xf = static_cast<float>(x);
    if (static_cast<double>(xf) < x)
        {
        xf = std::nextafterf(xf, std::numeric_limits<float>::infinity());
        }
    return xf;
    }

/*!
 * Initializes the shared pointer for the underlying LBVH.
 */
LBVHWrapper::LBVHWrapper()
    {
    lbvh_ = new neighbor::LBVH();
    }

LBVHWrapper::~LBVHWrapper()
    {
    delete lbvh_;
    }

/*!
 * \param points Particle positions
 * \param map Mapping of particles for insertion
 * \param N Number of particles
 * \param stream CUDA stream for execution
 */
void LBVHWrapper::setup(const Scalar4* points,
                        const unsigned int* map,
                        unsigned int N,
                        hipStream_t stream)
    {
    PointMapInsertOp insert(points, map, N);
    lbvh_->setup(stream, insert);
    }

/*!
 * \param points Particle positions
 * \param map Mapping of particles for insertion
 * \param N Number of particles
 * \param lo Lower bound of box
 * \param hi Upper bound of box
 * \param stream CUDA stream for execution
 * \param block_size CUDA block size for execution
 *
 * If HOOMD is using double-precision Scalars, then the lo and hi bounds of the
 * box are internally converted to floats using round-down and round-up modes,
 * respectively, which conserves the original box.
 */
void LBVHWrapper::build(const Scalar4* points,
                        const unsigned int* map,
                        unsigned int N,
                        const Scalar3& lo,
                        const Scalar3& hi,
                        hipStream_t stream,
                        unsigned int block_size)
    {
#if HOOMD_LONGREAL_SIZE == 64
    float3 lof = make_float3(double2float_rd(lo.x), double2float_rd(lo.y), double2float_rd(lo.z));
    float3 hif = make_float3(double2float_ru(hi.x), double2float_ru(hi.y), double2float_ru(hi.z));
#else
    float3 lof = lo;
    float3 hif = hi;
#endif

    PointMapInsertOp insert(points, map, N);
    lbvh_->build(neighbor::LBVH::LaunchParameters(block_size, stream), insert, lof, hif);
    }

unsigned int LBVHWrapper::getN() const
    {
    return lbvh_->getN();
    }

/*!
 * The internal neighbor data is converted to a raw device pointer that can be
 * used in host code.
 */
const unsigned int* LBVHWrapper::getPrimitives() const
    {
    return lbvh_->getPrimitives().get();
    }

std::vector<unsigned int> LBVHWrapper::getTunableParameters() const
    {
    return lbvh_->getTunableParameters();
    }

/*!
 * Initializes the shared pointer for the underlying LBVHTraverser.
 */
LBVHTraverserWrapper::LBVHTraverserWrapper()
    {
    trav_ = new neighbor::LBVHTraverser();
    };

LBVHTraverserWrapper::~LBVHTraverserWrapper()
    {
    delete trav_;
    }

/*!
 * \param map Mapping operation for the primitives for efficient traversal
 * \param lbvh LBVH to traverse
 * \param stream CUDA stream for execution
 */
void LBVHTraverserWrapper::setup(const unsigned int* map, neighbor::LBVH& lbvh, hipStream_t stream)
    {
    neighbor::MapTransformOp mapop(map);
    trav_->setup(stream, lbvh, mapop);
    }

/*!
 * \param args Pack of traversal arguments used to build operations
 * \param lbvh LBVH to traverse
 * \param images List of image vectors
 * \param Nimages Number of image vectors
 * \param stream CUDA stream for execution
 * \param block_size CUDA block size for execution
 *
 * The traversal arguments are forwarded to various neighbor operations
 * for traversal, including the transform operation, the output operation,
 * and the query operation. The images are forwarded directly, using the
 * Scalar precision.
 *
 * As a microoptimization, the query operation is templated on whether
 * body filtering is enabled. These switches
 * are determined by checking if the body data pointers
 * are NULL. It is the callers job to set rcut and rlist in the TraverserArgs
 * to compatible with those modes.
 */
void LBVHTraverserWrapper::traverse(TraverserArgs& args,
                                    neighbor::LBVH& lbvh,
                                    const Scalar3* images,
                                    const unsigned int Nimages,
                                    hipStream_t stream,
                                    unsigned int block_size)
    {
    neighbor::MapTransformOp map(args.map);

    NeighborListOp nlist_op(args.neigh_list,
                            args.nneigh,
                            args.new_max_neigh,
                            args.first_neigh,
                            args.max_neigh);

    neighbor::ImageListOp<Scalar3> translate(images, Nimages);

    if (args.bodies == NULL)
        {
        ParticleQueryOp<false> query(args.positions,
                                     NULL,
                                     args.order,
                                     args.N,
                                     args.Nown,
                                     args.rcut,
                                     args.rlist,
                                     args.box);
        trav_->traverse(neighbor::LBVHTraverser::LaunchParameters(block_size, stream),
                        lbvh,
                        query,
                        nlist_op,
                        translate,
                        map);
        }
    else if (args.bodies != NULL)
        {
        ParticleQueryOp<true> query(args.positions,
                                    args.bodies,
                                    args.order,
                                    args.N,
                                    args.Nown,
                                    args.rcut,
                                    args.rlist,
                                    args.box);
        trav_->traverse(neighbor::LBVHTraverser::LaunchParameters(block_size, stream),
                        lbvh,
                        query,
                        nlist_op,
                        translate,
                        map);
        }
    }

std::vector<unsigned int> LBVHTraverserWrapper::getTunableParameters() const
    {
    return trav_->getTunableParameters();
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
