// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _HPMC_IMPLICIT_NEW_CUH_
#define _HPMC_IMPLICIT_NEW_CUH_

#include "HPMCCounters.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

#include <curand_kernel.h>

#include <cassert>

#ifdef NVCC
#include "HPMCPrecisionSetup.h"
#include "Moves.h"
#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"
#endif

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicit.cuh
    \brief Declaration of CUDA kernels drivers
*/

//! Wraps arguments to gpu_hpmc_implicit_update
/*! \ingroup hpmc_data_structs */
struct hpmc_implicit_args_new_t
    {
    //! Construct a hpmc_implicit_args_new_t
    hpmc_implicit_args_new_t(Scalar4 *_d_postype,
                Scalar4 *_d_orientation,
                const Scalar4 *_d_postype_old,
                const Scalar4 *_d_orientation_old,
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
                const unsigned int *_d_check_overlaps,
                const Index2D& _overlap_idx,
                const unsigned int _timestep,
                const unsigned int _dim,
                const BoxDim& _box,
                const unsigned int _select,
                const unsigned int _block_size,
                const unsigned int _stride,
                const unsigned int _group_size,
                const bool _has_orientation,
                const unsigned int _max_n,
                const cudaDeviceProp& _devprop,
                curandState_t *_d_state_cell,
                curandState_t *_d_state_cell_new,
                const unsigned int _depletant_type,
                hpmc_counters_t *_d_counters,
                hpmc_implicit_counters_t *_d_implicit_count,
                const curandDiscreteDistribution_t *_d_poisson,
                unsigned int *_d_overlap_cell,
                unsigned int _groups_per_cell,
                const unsigned int *_d_active_cell_ptl_idx,
                const unsigned int *_d_active_cell_accept,
                const unsigned int *_d_active_cell_move_type_translate,
                const Scalar *_d_d_min,
                const Scalar *_d_d_max,
                bool _update_shape_param,
                cudaStream_t _stream
                )
                : d_postype(_d_postype),
                  d_orientation(_d_orientation),
                  d_postype_old(_d_postype_old),
                  d_orientation_old(_d_orientation_old),
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
                  d_check_overlaps(_d_check_overlaps),
                  overlap_idx(_overlap_idx),
                  timestep(_timestep),
                  dim(_dim),
                  box(_box),
                  select(_select),
                  block_size(_block_size),
                  stride(_stride),
                  group_size(_group_size),
                  has_orientation(_has_orientation),
                  max_n(_max_n),
                  devprop(_devprop),
                  d_state_cell(_d_state_cell),
                  d_state_cell_new(_d_state_cell_new),
                  depletant_type(_depletant_type),
                  d_counters(_d_counters),
                  d_implicit_count(_d_implicit_count),
                  d_poisson(_d_poisson),
                  d_overlap_cell(_d_overlap_cell),
                  groups_per_cell(_groups_per_cell),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate),
                  d_d_min(_d_d_min),
                  d_d_max(_d_d_max),
                  update_shape_param(_update_shape_param),
                  stream(_stream)
        {
        };

    Scalar4 *d_postype;               //!< postype array
    Scalar4 *d_orientation;           //!< orientation array
    const Scalar4 *d_postype_old;     //!< old postype array
    const Scalar4 *d_orientation_old; //!< old orientation array
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
    const unsigned int *d_check_overlaps; //!< Interaction matrix
    const Index2D& overlap_idx;       //!< Indexer for interaction matrix
    const unsigned int timestep;      //!< Current time step
    const unsigned int dim;           //!< Number of dimensions
    const BoxDim& box;                //!< Current simulation box
    const unsigned int select;        //!< Current selection
    const unsigned int block_size;    //!< Block size to execute
    const unsigned int stride;        //!< Number of threads per overlap check
    const unsigned int group_size;    //!< Size of the group to execute
    const bool has_orientation;       //!< True if the shape has orientation
    const unsigned int max_n;         //!< Maximum size of pdata arrays
    const cudaDeviceProp& devprop;    //!< CUDA device properties
    curandState_t *d_state_cell;        //!< RNG state per cell
    curandState_t *d_state_cell_new;    //!< RNG state per cell
    const unsigned int depletant_type; //!< Particle type of depletant
    hpmc_counters_t *d_counters;      //!< Acceptance/rejection counters
    hpmc_implicit_counters_t *d_implicit_count; //!< Active cell acceptance/rejection counts
    const curandDiscreteDistribution_t *d_poisson; //!< Handle for precomputed poisson distribution (per type)
    unsigned int *d_overlap_cell;     //!< Overlap flag per active cell
    unsigned int groups_per_cell;     //!< Number of groups to process in parallel per cell
    const unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell
    const unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise
    const unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    const Scalar *d_d_min;             //!< Minimum insertion diameter for depletants (per type)
    const Scalar *d_d_max;             //!< Maximum insertion diameter for depletants (per type)
    bool update_shape_param;           //!< True if this is the first iteration
    cudaStream_t stream;               //!< CUDA stream for kernel execution
    };

template< class Shape >
cudaError_t gpu_hpmc_insert_depletants_queue(const hpmc_implicit_args_new_t &args, const typename Shape::param_type *d_params);

template< class Shape >
cudaError_t gpu_hpmc_implicit_accept_reject_new(const hpmc_implicit_args_new_t &args, const typename Shape::param_type *d_params);

#ifdef NVCC
/*!
 * Definition of function templates and templated GPU kernels
 */

template< class Shape >
__global__ void gpu_hpmc_insert_depletants_queue_kernel(Scalar4 *d_postype,
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
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     const unsigned int *d_active_cell_ptl_idx,
                                     const unsigned int *d_active_cell_accept,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes,
                                     unsigned int depletant_type,
                                     const Scalar4 *d_postype_old,
                                     const Scalar4 *d_orientation_old,
                                     unsigned int *d_overlap_cell,
                                     hpmc_implicit_counters_t *d_implicit_counters,
                                     curandState_t *d_state_cell,
                                     curandState_t *d_state_cell_new,
                                     const curandDiscreteDistribution_t *d_poisson,
                                     const Scalar *d_d_min,
                                     const Scalar *d_d_max,
                                     bool reinitialize_curand)
    {
    // flags to tell what type of thread we are
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
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;
    __shared__ unsigned int s_n_inserted;

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    __shared__ unsigned int s_reject;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_queue_j = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);

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
    char *s_extra = (char *)(s_queue_gid + max_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;
        s_n_inserted = 0;
        s_reject = 0;
        }

    __syncthreads();

    // identify the active cell that this thread handles
    unsigned int active_cell_idx = blockIdx.x;

    // pull in the index of our cell
    unsigned int my_cell = d_cell_set[active_cell_idx];
    unsigned int my_cell_size = d_cell_size[my_cell];

    // need to handle if there are no particles in this cell
    if (my_cell_size == 0)
        return;

    // load updated particle index
    unsigned int i = d_active_cell_ptl_idx[active_cell_idx];

    // if the move was not performed or has been rejected before, nothing to do here
    if (i == UINT_MAX || !d_active_cell_accept[active_cell_idx]) return;

    // load updated particle position
    Scalar4 postype_i = __ldg(d_postype + i);
    unsigned int type_i = __scalar_as_int(postype_i.w);

    curandState_t local_state;
    if (reinitialize_curand)
        {
        curand_init((unsigned long long)(seed+active_cell_idx), (unsigned long long)timestep, select, &local_state);
        }
    else
        {
        // load RNG state per cell
        local_state = d_state_cell[active_cell_idx];
        }

    // for every active cell, draw a poisson random number
    unsigned int n_depletants = curand_discrete(&local_state, d_poisson[type_i]);

    // save RNG state per cell
    if (master && (group == 0))
        {
        d_state_cell_new[active_cell_idx] = local_state;
        }

    unsigned int overlap_checks = 0;
    unsigned int n_inserted = 0;

    Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

    Scalar d_min = d_d_min[type_i];
    Scalar d_max = d_d_max[type_i];

    // one RNG per group
    hoomd::detail::Saru rng(active_cell_idx*n_groups+group, seed+select, timestep);

    // iterate over depletants
    for (unsigned int i_dep = 0; i_dep < n_depletants; i_dep += n_groups)
        {
        unsigned int i_dep_local = i_dep + group;

        bool active = i_dep_local < n_depletants;

        if (active)
            {
            n_inserted++;
            // draw a random vector in the excluded volume sphere of the large particle
            Scalar theta = Scalar(2.0*M_PI)*rng.template s<Scalar>();
            Scalar z = Scalar(2.0)*rng.template s<Scalar>()-Scalar(1.0);

            // random normalized vector
            vec3<Scalar> n(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(theta),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(theta),z);

            // draw random radial coordinate in test sphere
            Scalar r3 = rng.template s<Scalar>(fast::pow(d_min/d_max,Scalar(3.0)),Scalar(1.0));
            Scalar r = Scalar(0.5)*d_max*fast::pow(r3,Scalar(1.0/3.0));

            // test depletant position around old configuration
            Scalar4 postype_i_old = __ldg(d_postype_old + i);
            vec3<Scalar> pos_test = vec3<Scalar>(postype_i_old)+r*n;

            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng);
                }

            overlap_checks++;
            bool overlap_new = false;
                {
                Scalar4 orientation_i = make_scalar4(1,0,0,0);
                Shape shape_i(quat<Scalar>(orientation_i), s_params[type_i]);
                if (shape_i.hasOrientation())
                    {
                    orientation_i = __ldg(d_orientation + i);
                    shape_i.orientation = quat<Scalar>(orientation_i);
                    }

                // check depletant overlap with shape at new position
                vec3<Scalar> r_ij = vec3<Scalar>(postype_i) - pos_test;

                // test circumsphere overlap
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (s_check_overlaps[overlap_idx(type_i, depletant_type)]
                    && circumsphere_overlap
                    && test_overlap(r_ij, shape_test, shape_i, err_count))
                    {
                    overlap_new = true;
                    }
                }

            if (overlap_new) active = false;

            overlap_checks++;
            bool overlap_old = false;
            if (active)
                {
                // check depletant overlap with shape at old position
                vec3<Scalar> r_ij = vec3<Scalar>(postype_i_old) - pos_test;
                Scalar4 orientation_i_old = make_scalar4(1,0,0,0);
                Shape shape_i_old(quat<Scalar>(quat<Scalar>(orientation_i_old)), d_params[type_i]);
                if (shape_i_old.hasOrientation())
                    {
                    orientation_i_old = __ldg(d_orientation_old + i);
                    shape_i_old.orientation = quat<Scalar>(orientation_i_old);
                    }

                // test circumsphere overlap
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_i_old.getCircumsphereDiameter();
                bool circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                if (s_check_overlaps[overlap_idx(type_i, depletant_type)]
                    && circumsphere_overlap
                    && test_overlap(r_ij, shape_test, shape_i_old, err_count))
                    {
                    overlap_old = true;
                    }
                }

            if (! overlap_old) active = false;

            // stash the trial move in shared memory so that other threads in this block can process overlap checks
            if (master)
                {
                s_pos_group[group] = make_scalar3(pos_test.x, pos_test.y, pos_test.z);
                s_orientation_group[group] = quat_to_scalar4(shape_test.orientation);
                }
            }

        if (master && group == 0)
            {
            s_queue_size = 0;
            s_still_searching = 1;
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
                while (!s_reject && s_queue_size < max_queue_size && k < excell_size)
                    {
                    if (k < excell_size)
                        {
                        Scalar4 postype_j;
                        Scalar4 orientation_j;
                        vec3<Scalar> r_ij;

                        // build some shapes, but we only need them to get diameters, so don't load orientations
                        // build shape i from shared memory
                        Scalar3 pos_test = s_pos_group[group];
                        Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

                        // prefetch next j
                        k += group_size;
                        j = next_j;

                        if (k < excell_size)
                            {
                            next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                            }

                        // read in position, and orientation of neighboring particle
                        postype_j = __ldg(d_postype + j);
                        unsigned int type_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                        // put particle j into the coordinate system of particle i
                        r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                        r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));
                        // test circumsphere overlap
                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                        if (s_check_overlaps[overlap_idx(depletant_type, type_j)] && i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
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

            unsigned int tidx_1d = offset + group_size*group;

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
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

                // build shape j from global memory
                postype_j = __ldg(d_postype + check_j);
                orientation_j = make_scalar4(1,0,0,0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(__ldg(d_orientation + check_j));

                // put particle j into the coordinate system of particle i
                r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                if (test_overlap(r_ij, shape_test, shape_j, err_count))
                    {
                    s_reject = 1;
                    }
                }

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && !s_reject && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        // exit early if any depletant leads to rejection
        if (s_reject) break;
        } // end loop over depletants

    // update the data if accepted
    if (master && group == 0 && s_reject)
        {
        // count the overlap checks
        d_overlap_cell[active_cell_idx] = 1;
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    if (master)
        {
        // count the overlap checks
        atomicAdd(&s_overlap_checks, overlap_checks);

        // count inserted depletants
        atomicAdd(&s_n_inserted, n_inserted);
        }

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);

        // increment number of inserted depletants
        atomicAdd(&d_implicit_counters->insert_count, n_inserted);
        }
    }


//! Definition of kernel to set up cuRAND for the maximum kernel parameters
__global__ void gpu_curand_implicit_setup(unsigned int n_rng,
                                          unsigned int seed,
                                          unsigned int timestep,
                                          curandState_t *d_state);
/*!
 * Definition of templated GPU kernel drivers
 */

//! Kernel driver for gpu_update_hpmc_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_insert_depletants_queue(const hpmc_implicit_args_new_t& args, const typename Shape::param_type *params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_counters);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.d_check_overlaps);
    assert(args.group_size >= 1);
    assert(args.stride >= 1);

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static cudaFuncAttributes attr;
    if (max_block_size == -1)
        {
        cudaFuncGetAttributes(&attr, gpu_hpmc_insert_depletants_queue_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        }

    // might need to modify group_size to make the kernel runnable
    unsigned int group_size = args.group_size;

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

    // ensure block_size is a multiple of stride
    unsigned int stride = args.stride;
    while (block_size % stride)
        {
        stride--;
        }

    // the new block size might not be a multiple of group size, decrease group size until it is
    group_size = args.group_size;

    while (block_size % (stride*group_size))
        {
        group_size--;
        }

    unsigned int n_groups = block_size / (group_size * stride);
    unsigned int max_queue_size = n_groups*group_size;

    unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
               args.overlap_idx.getNumElements() * sizeof(unsigned int);

    unsigned int shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3)) +
                                max_queue_size*(sizeof(unsigned int) + sizeof(unsigned int)) +
                                min_shared_bytes;

    if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
        {
        block_size -= args.devprop.warpSize;
        if (block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        group_size = args.group_size;

        stride = min(block_size, args.stride);
        while (stride*group_size > block_size)
            {
            group_size--;
            }

        n_groups = block_size / (group_size * stride);
        max_queue_size = n_groups*group_size;
        shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3)) +
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
        char *ptr = (char *) nullptr;
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

    static dim3 last_threads = dim3(0,0,0);

    bool reinitialize_curand = false;
    if (threads.x != last_threads.x || threads.y != last_threads.y || threads.z != last_threads.z)
        {
        // thread configuration changed
        last_threads = threads;
        reinitialize_curand = true;
        }

    // 1 block per active cell
    dim3 grid( args.n_active_cells, 1, 1);

    // reset counters
    cudaMemsetAsync(args.d_overlap_cell,0, sizeof(unsigned int)*args.n_active_cells, args.stream);

    gpu_hpmc_insert_depletants_queue_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
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
                                                                 args.d_check_overlaps,
                                                                 args.overlap_idx,
                                                                 args.timestep,
                                                                 args.dim,
                                                                 args.box,
                                                                 args.select,
                                                                 args.d_active_cell_ptl_idx,
                                                                 args.d_active_cell_accept,
                                                                 params,
                                                                 max_queue_size,
                                                                 max_extra_bytes,
                                                                 args.depletant_type,
                                                                 args.d_postype_old,
                                                                 args.d_orientation_old,
                                                                 args.d_overlap_cell,
                                                                 args.d_implicit_count,
                                                                 args.d_state_cell,
                                                                 args.d_state_cell_new,
                                                                 args.d_poisson,
                                                                 args.d_d_min,
                                                                 args.d_d_max,
                                                                 reinitialize_curand);
    // advance per-cell RNG states
    cudaMemcpyAsync(args.d_state_cell, args.d_state_cell_new, sizeof(curandState_t)*args.n_active_cells, cudaMemcpyDeviceToDevice, args.stream);

    return cudaSuccess;
    }

//! Kernel to accept or reject moves on a per active cell basis
template<class Shape>
__global__ void gpu_implicit_accept_reject_new_kernel(
    unsigned int *d_overlap_cell,
    unsigned int n_active_cells,
    const unsigned int *d_cell_set,
    const unsigned int *d_cell_size,
    const unsigned int *d_cell_idx,
    Index2D cli,
    Scalar4 *d_postype,
    Scalar4 *d_orientation,
    const Scalar4 *d_postype_old,
    const Scalar4 *d_orientation_old,
    hpmc_counters_t *d_counters,
    const BoxDim box,
    const unsigned int *d_active_cell_ptl_idx,
    const unsigned int *d_active_cell_accept,
    const unsigned int *d_move_type_translate,
    curandState_t *d_state_cell,
    const typename Shape::param_type *d_params
    )
    {
    unsigned int active_cell_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (active_cell_idx >= n_active_cells) return;

    // ignore if no move was proposed in this cell
    unsigned int updated_ptl_idx = d_active_cell_ptl_idx[active_cell_idx];
    if (updated_ptl_idx == UINT_MAX)
        {
        return;
        }

    // whether the particle move was accepted (== generated no overlap)
    bool accept = d_active_cell_accept[active_cell_idx];

    if (accept)
        {
        unsigned int n_overlap = d_overlap_cell[active_cell_idx];

        // if no overlap of depletant in new configuration, accept
        accept = !n_overlap;
        }

    // the particle that was updated
    Scalar4 postype_i = d_postype[updated_ptl_idx];
    Shape shape_i(quat<Scalar>(), d_params[__scalar_as_int(postype_i.w)]);

    if (!accept)
        {
        // revert to old position and orientation
        d_postype[updated_ptl_idx] = d_postype_old[updated_ptl_idx];
        d_orientation[updated_ptl_idx] = d_orientation_old[updated_ptl_idx];

        if (!shape_i.ignoreStatistics())
            {
            // increment reject count
            if (d_move_type_translate[active_cell_idx])
                {
                atomicAdd(&d_counters->translate_reject_count, 1);
                }
            else
                {
                atomicAdd(&d_counters->rotate_reject_count, 1);
                }
            }
        }
    else
        {
        if (!shape_i.ignoreStatistics())
            {
            // increment accept count
            if (d_move_type_translate[active_cell_idx])
                {
                atomicAdd(&d_counters->translate_accept_count, 1);
                }
            else
                {
                atomicAdd(&d_counters->rotate_accept_count, 1);
                }
            }
        }
    }

//! Kernel driver for gpu_hpmc_implicit_accept_reject_new_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_implicit_accept_reject_new(const hpmc_implicit_args_new_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_cell_idx);
    assert(args.d_cell_size);
    assert(args.d_excell_idx);
    assert(args.d_excell_size);
    assert(args.d_cell_set);
    assert(args.group_size >= 1);
    assert(args.group_size <= 32);  // note, really should be warp size of the device

    // accept-reject on a per cell basis
    unsigned int block_size = 256;
    gpu_implicit_accept_reject_new_kernel<Shape><<<args.n_active_cells/block_size + 1, block_size, 0, args.stream>>>(
        args.d_overlap_cell,
        args.n_active_cells,
        args.d_cell_set,
        args.d_cell_size,
        args.d_cell_idx,
        args.cli,
        args.d_postype,
        args.d_orientation,
        args.d_postype_old,
        args.d_orientation_old,
        args.d_counters,
        args.box,
        args.d_active_cell_ptl_idx,
        args.d_active_cell_accept,
        args.d_active_cell_move_type_translate,
        args.d_state_cell,
        d_params);

    return cudaSuccess;
    }

#endif // NVCC

}; // end namespace detail

} // end namespace hpmc

#endif // _HPMC_IMPLICIT_CUH_
