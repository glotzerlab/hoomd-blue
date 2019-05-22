// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _INTEGRATOR_HPMC_CUH_
#define _INTEGRATOR_HPMC_CUH_


#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <cassert>

#include "HPMCCounters.h"

#ifdef NVCC
#include "HPMCPrecisionSetup.h"
#include "Moves.h"
#include "hoomd/TextureTools.h"
#endif

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoGPU.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_up
/*! \ingroup hpmc_data_structs */
struct hpmc_args_t
    {
    //! Construct a pair_args_t
    hpmc_args_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                hpmc_counters_t *_d_counters,
                const unsigned int *_d_cell_idx,
                const unsigned int *_d_cell_size,
                const unsigned int *_d_excell_idx,
                const unsigned int *_d_excell_size,
                const Index3D& _ci,
                const Index2D& _cli,
                const Index2D& _excli,
                const uint3& _cell_dim,
                const Scalar3& _ghost_width,
                const unsigned int *_d_cell_set,
                const unsigned int _n_active_cells,
                const unsigned int _N,
                const unsigned int _num_types,
                const unsigned int _seed,
                const Scalar* _d,
                const Scalar* _a,
                const unsigned int *_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _move_ratio,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const Scalar3 _ghost_fraction,
                const bool _domain_decomposition,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const bool _has_orientation,
                const unsigned int _max_n,
                const cudaDeviceProp& _devprop,
                bool _update_shape_param,
                cudaStream_t _stream,
                unsigned int *_d_active_cell_ptl_idx = NULL,
                unsigned int *_d_active_cell_accept = NULL,
                unsigned int *_d_active_cell_move_type_translate = NULL)
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_counters(_d_counters),
                  d_cell_idx(_d_cell_idx),
                  d_cell_size(_d_cell_size),
                  d_excell_idx(_d_excell_idx),
                  d_excell_size(_d_excell_size),
                  ci(_ci),
                  cli(_cli),
                  excli(_excli),
                  cell_dim(_cell_dim),
                  ghost_width(_ghost_width),
                  d_cell_set(_d_cell_set),
                  n_active_cells(_n_active_cells),
                  N(_N),
                  num_types(_num_types),
                  seed(_seed),
                  d_d(_d),
                  d_a(_a),
                  d_check_overlaps(_check_overlaps),
                  overlap_idx(_overlap_idx),
                  move_ratio(_move_ratio),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  ghost_fraction(_ghost_fraction),
                  domain_decomposition(_domain_decomposition),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  has_orientation(_has_orientation),
                  max_n(_max_n),
                  devprop(_devprop),
                  update_shape_param(_update_shape_param),
                  stream(_stream),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    hpmc_counters_t *d_counters;      //!< Move accept/reject counters
    const unsigned int *d_cell_idx;   //!< Index data for each cell
    const unsigned int *d_cell_size;  //!< Number of particles in each cell
    const unsigned int *d_excell_idx; //!< Index data for each expanded cell
    const unsigned int *d_excell_size;//!< Number of particles in each expanded cell
    const Index3D& ci;                //!< Cell indexer
    const Index2D& cli;               //!< Indexer for d_cell_idx
    const Index2D& excli;             //!< Indexer for d_excell_idx
    const uint3& cell_dim;            //!< Cell dimensions
    const Scalar3& ghost_width;       //!< Width of the ghost layer
    const unsigned int *d_cell_set;   //!< List of active cells
    const unsigned int n_active_cells;//!< Number of active cells
    const unsigned int N;             //!< Number of particles
    const unsigned int num_types;     //!< Number of particle types
    const unsigned int seed;          //!< RNG seed
    const Scalar* d_d;                //!< Maximum move displacement
    const Scalar* d_a;                //!< Maximum move angular displacement
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer into interaction matrix
    const unsigned int move_ratio;    //!< Ratio of translation to rotation moves
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    const unsigned int select;        //!< Current selection
    const Scalar3 ghost_fraction;     //!< Width of the inactive layer
    const bool domain_decomposition;  //!< Is domain decomposition mode enabled?
    const unsigned int block_size;    //!< Block size to execute
    const unsigned int stride;        //!< Number of threads per overlap check
    const unsigned int group_size;    //!< Size of the group to execute
    const bool has_orientation;       //!< True if the shape has orientation
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    bool update_shape_param;          //!< If true, update size of shape param and synchronize GPU execution stream
    cudaStream_t stream;              //!< The CUDA stream associated with the update kernel
    unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell (ignore if NULL)
    unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise (ignore if NULL)
    unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    };

cudaError_t gpu_hpmc_excell(unsigned int *d_excell_idx,
                            unsigned int *d_excell_size,
                            const Index2D& excli,
                            const unsigned int *d_cell_idx,
                            const unsigned int *d_cell_size,
                            const unsigned int *d_cell_adj,
                            const Index3D& ci,
                            const Index2D& cli,
                            const Index2D& cadji,
                            const unsigned int block_size);

template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *params);

cudaError_t gpu_hpmc_shift(Scalar4 *d_postype,
                           int3 *d_image,
                           const unsigned int N,
                           const BoxDim& box,
                           const Scalar3 shift,
                           const unsigned int block_size);

#ifdef NVCC
/*!
 * Definition of function templates and templated GPU kernels
 */

//! Device function to compute the cell that a particle sits in
__device__ inline unsigned int computeParticleCell(const Scalar3& p,
                                                   const BoxDim& box,
                                                   const Scalar3& ghost_width,
                                                   const uint3& cell_dim,
                                                   const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    if (f.x >= Scalar(0.0) && f.x < Scalar(1.0) && f.y >= Scalar(0.0) && f.y < Scalar(1.0) && f.z >= Scalar(0.0) && f.z < Scalar(1.0))
        return ci(ib,jb,kb);
    else
        return 0xffffffff;
    }

//! HPMC  update kernel
/*! \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_counters Acceptance counters to increment
    \param d_cell_idx Particle index stored in the cell list
    \param d_cell_size The size of each cell
    \param d_excell_idx Indices of particles in extended cells
    \param d_excell_size Number of particles in each extended cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param excli Extended cell list indexer
    \param cell_dim Dimensions of the cell list
    \param ghost_width Width of the ghost layer
    \param d_cell_set List of active cells
    \param n_active_cells Number of active cells
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param d_d Array of maximum move displacements
    \param d_a Array of rotation move sizes
    \param d_check_overlaps Interaction matrix
    \parma overlap_idx Indexer into interaction matrix
    \param move_ratio Ratio of translation moves to rotation moves
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param select Current index within the loop over nselect selections (for RNG generation)
    \param ghost_fraction Width of the inactive layer in MPI domain decomposition simulations
    \param domain_decomposition True if executing with domain decomposition
    \param d_params Per-type shape parameters

    MPMC in its published form has a severe limit on the number of parallel threads in 3D. This implementation launches
    group_size threads per cell (1,2,4,8,16,32). Each thread in the group performs the same trial move on the same
    particle, and then checks for overlaps against different particles from the extended cell list. The entire extended
    cell list is covered in a batched loop. The group_size is autotuned to find the fastest performance. Smaller systems
    tend to run fastest with a large group_size due to the increased parallelism. Larger systems tend to run faster
    at smaller group_sizes because they already have the parallelism from the system size - however, even the largest
    systems benefit from group_size > 1 on K20. Shared memory is used to set an overlap flag to 1 if any of the threads
    in the group detect an overlap. After all checks are complete, the master thread in the group applies the trial move
    update if accepted.

    No __syncthreads is needed after the overlap checks because the group_size is always chosen to be a power of 2 and
    smaller than the warp size. Only a __threadfence_block() is needed to ensure memory consistency.

    Move stats are tallied in local memory, then totaled in shared memory at the end and finally a single thread in the
    block runs an atomicAdd on global memory to get the system wide total. This isn't as good as a reduction, but it
    is only a tiny fraction of the compute time.

    In order to simplify indexing and boundary checks, a list of active cells is determined on the host and passed into
    the kernel. That way, only a linear indexing of threads is needed to handle any geometry of active cells.

    Heavily divergent warps are avoided by pre-building a list of all particles in the neighboring region of any given
    cell. Otherwise, extremely non-uniform cell lengths (i.e. avg 1, max 4) don't cause massive performance degradation.

    **Indexing**
        - threadIdx.y indexes the current group in the block
        - threadIdx.x is the offset within the current group
        - blockIdx.x runs enough blocks so that all active cells are covered

    \ingroup hpmc_kernels
*/
template< class Shape >
__global__ void gpu_hpmc_mpmc_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_cell_idx,
                                     const unsigned int *d_cell_size,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const unsigned int *d_cell_set,
                                     const unsigned int n_active_cells,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const Scalar* d_d,
                                     const Scalar* d_a,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int move_ratio,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const Scalar3 ghost_fraction,
                                     const bool domain_decomposition,
                                     unsigned int *d_active_cell_ptl_idx,
                                     unsigned int *d_active_cell_accept,
                                     unsigned int *d_active_cell_move_type_translate,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes)
    {
    // flags to tell what type of thread we are
    bool active = true;
    unsigned int group;
    unsigned int offset;
    unsigned int group_size;
    bool master;
    unsigned int n_groups;

    if (Shape::isParallel())
        {
        // use 3d thread block layout
        group = threadIdx.z;
        offset = threadIdx.y;
        group_size = blockDim.y;
        master = (offset == 0 && threadIdx.x == 0);
        n_groups = blockDim.z;
        }
    else
        {
        group = threadIdx.y;
        offset = threadIdx.x;
        group_size = blockDim.x;
        master = (offset == 0);
        n_groups = blockDim.y;
        }

    unsigned int err_count = 0;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_translate_accept_count;
    __shared__ unsigned int s_translate_reject_count;
    __shared__ unsigned int s_rotate_accept_count;
    __shared__ unsigned int s_rotate_reject_count;
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];

    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    Scalar *s_d = (Scalar *)(s_pos_group + n_groups);
    Scalar *s_a = (Scalar *)(s_d + num_types);
    unsigned int *s_check_overlaps = (unsigned int *) (s_a + num_types);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_overlap =   (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);
    unsigned int *s_type_group = (unsigned int*)(s_queue_gid + max_queue_size);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }

        for (unsigned int cur_offset = 0; cur_offset < num_types; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_types)
                {
                s_a[cur_offset + tidx] = d_a[cur_offset + tidx];
                s_d[cur_offset + tidx] = d_d[cur_offset + tidx];
                }
            }

        unsigned int ntyppairs = overlap_idx.getNumElements();

        for (unsigned int cur_offset = 0; cur_offset < ntyppairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < ntyppairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_type_group + n_groups);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_translate_accept_count = 0;
        s_translate_reject_count = 0;
        s_rotate_accept_count = 0;
        s_rotate_reject_count = 0;
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_queue_size = 0;
        s_still_searching = 1;
        }
    if (master)
        {
        s_overlap[group] = 0;
        }

    // identify the active cell that this thread handles
    unsigned int active_cell_idx = 0;
    active_cell_idx = blockIdx.x * n_groups + group;

    // this thread is inactive if it indexes past the end of the active cell list
    if (active_cell_idx >= n_active_cells)
        active = false;

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;
    if (active)
        {
        my_cell = d_cell_set[active_cell_idx];
        my_cell_size = d_cell_size[my_cell];
        }

    // need to deactivate if there are no particles in this cell
    if (my_cell_size == 0)
        active = false;

    __syncthreads();

    // initial implementation just moves one particle per cell (nselect=1).
    // these variables are ugly, but needed to get the updated quantities outside of the scope
    unsigned int i;
    unsigned int overlap_checks = 0;
    bool move_type_translate = false;
    bool move_active = true;
    int ignore_stats = 0;

    if (active)
        {
        // one RNG per cell
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCMonoTrialMove, seed, my_cell, select, timestep);

        // select one of the particles randomly from the cell
        unsigned int my_cell_offset = hoomd::UniformIntDistribution(my_cell_size-1)(rng);
        i = __ldg(d_cell_idx + cli(my_cell_offset, my_cell));

        // read in the position and orientation of our particle.
        Scalar4 postype_i = __ldg(d_postype + i);
        Scalar4 orientation_i = make_scalar4(1,0,0,0);

        unsigned int typ_i = __scalar_as_int(postype_i.w);
        Shape shape_i(quat<Scalar>(orientation_i), s_params[typ_i]);

        if (shape_i.hasOrientation())
            orientation_i = __ldg(d_orientation + i);

        shape_i.orientation = quat<Scalar>(orientation_i);

        // if this looks funny, that is because it is. Using ignore_stats as a bool setting ignore_stats = ...
        // causes a compiler bug.
        if (shape_i.ignoreStatistics())
            ignore_stats = 1;

        vec3<Scalar> pos_i = vec3<Scalar>(postype_i);

        // for domain decomposition simulations, we need to leave all particles in the inactive region alone
        // in order to avoid even more divergence, this is done by setting the move_active flag
        // overlap checks are still processed, but the final move acceptance will be skipped
        if (domain_decomposition && !isActive(make_scalar3(postype_i.x, postype_i.y, postype_i.z), box, ghost_fraction))
            move_active = false;

        // make the move
        unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
        move_type_translate = !shape_i.hasOrientation() || (move_type_select < move_ratio);

        if (move_type_translate)
            {
            move_translate(pos_i, rng, s_d[typ_i], dim);

            // need to reject any move that puts the particle in the inactive region
            if (domain_decomposition && !isActive(vec_to_scalar3(pos_i), box, ghost_fraction))
                move_active = false;
            }
        else
            {
            move_rotate(shape_i.orientation, rng, s_a[typ_i], dim);
            }

        // stash the trial move in shared memory so that other threads in this block can process overlap checks
        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_i.x, pos_i.y, pos_i.z);
            s_type_group[group] = typ_i;
            s_orientation_group[group] = quat_to_scalar4(shape_i.orientation);
            }
        }

    // sync so that s_postype_group and s_orientation are available before other threads might process overlap checks
    __syncthreads();

    // counters to track progress through the loop over potential neighbors
    unsigned int excell_size;
    unsigned int k = offset;
    if (active)
        {
        excell_size = d_excell_size[my_cell];
        overlap_checks += excell_size;
        }

    // loop while still searching
    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (active)
            {
            // prefetch j
            unsigned int j, next_j = 0;
            if (k < excell_size)
                {
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                }

            // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
            // and as long as no overlaps have been found
            while (!s_overlap[group] && s_queue_size < max_queue_size && k < excell_size)
                {
                if (k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j;
                    vec3<Scalar> r_ij;

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[s_type_group[group]]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        {
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = __ldg(d_postype + j);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[__scalar_as_int(postype_j.w)]);

                    // put particle j into the coordinate system of particle i
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = j;
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        }

                    } // end if k < excell_size
                } // end while (s_queue_size < max_queue_size && k < excell_size)
            } // end if active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group_size*group;  // z component is for Shape parallelism

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j = s_queue_j[tidx_1d];
            Scalar4 postype_j;
            Scalar4 orientation_j;
            vec3<Scalar> r_ij;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            unsigned int type_i = s_type_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[type_i]);

            // build shape j from global memory
            postype_j = __ldg(d_postype + check_j);
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<Scalar>(__ldg(d_orientation + check_j));

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(type_i, type_j)] && test_overlap(r_ij, shape_i, shape_j, err_count))
                {
                atomicAdd(&s_overlap[check_group], 1);
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;

        if (active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

    // update the data if accepted
    if (master)
        {
        if (active && move_active)
            {
            // first need to check if the particle remains in its cell
            Scalar3 xnew_i = s_pos_group[group];
            unsigned int new_cell = computeParticleCell(xnew_i, box, ghost_width, cell_dim, ci);
            bool accepted=true;
            if (s_overlap[group])
                accepted=false;
            if (new_cell != my_cell)
                accepted=false;

            if (accepted)
                {
                // write out the updated position and orientation
                d_postype[i] = make_scalar4(xnew_i.x, xnew_i.y, xnew_i.z, __int_as_scalar(s_type_group[group]));
                d_orientation[i] = s_orientation_group[group];
                }

            if (d_active_cell_accept)
                {
                // store particle index
                d_active_cell_ptl_idx[active_cell_idx] = i;
                }

            if (d_active_cell_accept)
                {
                // store accept flag
                d_active_cell_accept[active_cell_idx] = accepted ? 1 : 0;
                }

            if (d_active_cell_move_type_translate)
                {
                // store move type
                d_active_cell_move_type_translate[active_cell_idx] = move_type_translate ? 1 : 0;
                }

            // if an auxiliary array was provided, defer writing out statistics
            if (d_active_cell_ptl_idx)
                {
                ignore_stats = 1;
                }

            if (!ignore_stats && accepted && move_type_translate)
                atomicAdd(&s_translate_accept_count, 1);
            if (!ignore_stats && accepted && !move_type_translate)
                atomicAdd(&s_rotate_accept_count, 1);
            if (!ignore_stats && !accepted && move_type_translate)
                atomicAdd(&s_translate_reject_count, 1);
            if (!ignore_stats && !accepted && !move_type_translate)
                atomicAdd(&s_rotate_reject_count, 1);
            }
        else // active && move_active
            {
            if (d_active_cell_ptl_idx && active_cell_idx < n_active_cells)
                {
                // indicate that no particle was selected
                d_active_cell_ptl_idx[active_cell_idx] = UINT_MAX;
                }
            }

        // count the overlap checks
        atomicAdd(&s_overlap_checks, overlap_checks);
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->translate_accept_count, s_translate_accept_count);
        atomicAdd(&d_counters->translate_reject_count, s_translate_reject_count);
        atomicAdd(&d_counters->rotate_accept_count, s_rotate_accept_count);
        atomicAdd(&d_counters->rotate_reject_count, s_rotate_reject_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        }
    }

//! Kernel driver for gpu_update_hpmc_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_update(const hpmc_args_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_d);
    assert(args.d_a);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);
    assert(args.stride >= 1);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_mpmc_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // the new block size might not fit the group size and stride, decrease group size until it is
    group_size = args.group_size;

    unsigned int stride = min(block_size, args.stride);
    while (stride*group_size > block_size)
        {
        group_size--;
        }

    unsigned int n_groups = block_size / (group_size * stride);
    unsigned int max_queue_size = n_groups*group_size;
    unsigned int shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                                max_queue_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                                args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
                                args.overlap_idx.getNumElements() * sizeof(unsigned int);

    unsigned int min_shared_bytes = args.num_types * (sizeof(typename Shape::param_type) + 2*sizeof(Scalar)) +
               args.overlap_idx.getNumElements() * sizeof(unsigned int);

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;
        if (block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        // the new block size might not fit the group size and stride, decrease group size until it is
        stride = args.stride;
        group_size = args.group_size;

        unsigned int stride = min(block_size, args.stride);
        while (stride*group_size > block_size)
            {
            group_size--;
            }

        n_groups = block_size / (group_size * stride);
        max_queue_size = n_groups*group_size;
        shared_bytes = n_groups * (sizeof(unsigned int)*2 + sizeof(Scalar4) + sizeof(Scalar3)) +
                       max_queue_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                       min_shared_bytes;
        }

    static unsigned int base_shared_bytes = UINT_MAX;
    bool shared_bytes_changed = base_shared_bytes != shared_bytes + attr.sharedSizeBytes;
    base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

    unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
    static unsigned int extra_bytes = UINT_MAX;
    if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
        {
        // required for memory coherency
        cudaDeviceSynchronize();

        // determine dynamically requested shared memory
        char *ptr = (char *)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].load_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;
        }

    shared_bytes += extra_bytes;

    // setup the grid to run the kernel
    dim3 threads;
    if (Shape::isParallel())
        {
        // use three-dimensional thread-layout with blockDim.z < 64
        threads = dim3(stride, group_size, n_groups);
        }
    else
        {
        threads = dim3(group_size, n_groups,1);
        }

    dim3 grid( args.n_active_cells / n_groups + 1, 1, 1);

    gpu_hpmc_mpmc_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                 args.d_orientation,
                                                                 args.d_counters,
                                                                 args.d_cell_idx,
                                                                 args.d_cell_size,
                                                                 args.d_excell_idx,
                                                                 args.d_excell_size,
                                                                 args.ci,
                                                                 args.cli,
                                                                 args.excli,
                                                                 args.cell_dim,
                                                                 args.ghost_width,
                                                                 args.d_cell_set,
                                                                 args.n_active_cells,
                                                                 args.N,
                                                                 args.num_types,
                                                                 args.seed,
                                                                 args.d_d,
                                                                 args.d_a,
                                                                 args.d_check_overlaps,
                                                                 args.overlap_idx,
                                                                 args.move_ratio,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.ghost_fraction,
                                                                 args.domain_decomposition,
                                                                 args.d_active_cell_ptl_idx,
                                                                 args.d_active_cell_accept,
                                                                 args.d_active_cell_move_type_translate,
                                                                 params,
                                                                 max_queue_size,
                                                                 max_extra_bytes);

    return cudaSuccess;
    }

#endif //NVCC

}; // end namespace detail

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_CUH_

