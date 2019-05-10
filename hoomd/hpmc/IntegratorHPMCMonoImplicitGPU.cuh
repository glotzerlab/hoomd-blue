// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef _HPMC_IMPLICIT_CUH_
#define _HPMC_IMPLICIT_CUH_

#include "HPMCCounters.h"

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"

#include "hoomd/extern/util/mgpucontext.h"

#include <curand_kernel.h>

#include <cassert>

#ifdef NVCC
#include "hoomd/WarpTools.cuh"
#include "HPMCPrecisionSetup.h"
#include "Moves.h"
#include "hoomd/Saru.h"
#include "hoomd/TextureTools.h"
#include "hoomd/extern/kernels/segreducecsr.cuh"
#include "hoomd/extern/kernels/segreduce.cuh"
#include "hoomd/extern/kernels/intervalmove.cuh"
#include "hoomd/extern/kernels/scan.cuh"
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
struct hpmc_implicit_args_t
    {
    //! Construct a pair_args_t
    hpmc_implicit_args_t(Scalar4 *_d_postype,
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
                unsigned int *_d_overlap_cell_scan,
                unsigned int _groups_per_cell,
                const unsigned int *_d_active_cell_ptl_idx,
                const unsigned int *_d_active_cell_accept,
                const unsigned int *_d_active_cell_move_type_translate,
                float *_d_lnb,
                unsigned int *_d_n_success_zero,
                unsigned int _ntrial,
                unsigned int *_d_depletant_active_cell,
                unsigned int &_n_overlaps,
                unsigned int *_d_n_success_forward,
                unsigned int *_d_n_overlap_shape_forward,
                unsigned int *_d_n_success_reverse,
                unsigned int *_d_n_overlap_shape_reverse,
                float *_d_depletant_lnb,
                const Scalar *_d_d_min,
                const Scalar *_d_d_max,
                bool _update_shape_param,
                mgpu::ContextPtr _mgpu_context,
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
                  d_overlap_cell_scan(_d_overlap_cell_scan),
                  groups_per_cell(_groups_per_cell),
                  d_active_cell_ptl_idx(_d_active_cell_ptl_idx),
                  d_active_cell_accept(_d_active_cell_accept),
                  d_active_cell_move_type_translate(_d_active_cell_move_type_translate),
                  d_lnb(_d_lnb),
                  d_n_success_zero(_d_n_success_zero),
                  ntrial(_ntrial),
                  d_depletant_active_cell(_d_depletant_active_cell),
                  n_overlaps(_n_overlaps),
                  d_n_success_forward(_d_n_success_forward),
                  d_n_overlap_shape_forward(_d_n_overlap_shape_forward),
                  d_n_success_reverse(_d_n_success_reverse),
                  d_n_overlap_shape_reverse(_d_n_overlap_shape_reverse),
                  d_depletant_lnb(_d_depletant_lnb),
                  d_d_min(_d_d_min),
                  d_d_max(_d_d_max),
                  update_shape_param(_update_shape_param),
                  mgpu_context(_mgpu_context),
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
    unsigned int *d_overlap_cell_scan; //!< Overlap flag per active cell (scan result)
    unsigned int groups_per_cell;     //!< Number of groups to process in parallel per cell
    const unsigned int *d_active_cell_ptl_idx; //!< Updated particle index per active cell
    const unsigned int *d_active_cell_accept;//!< =1 if active cell move has been accepted, =0 otherwise
    const unsigned int *d_active_cell_move_type_translate;//!< =1 if active cell move was a translation, =0 if rotation
    float *d_lnb;                     //!< Logarithm of Boltzmann factor
    unsigned int *d_n_success_zero;   //!< If the number of successful re-insertion attempts is zero (new position)
    unsigned int ntrial;              //!< Number of reinsertion attempts
    unsigned int *d_depletant_active_cell; //!< Lookup of active-cell idx per depletant
    unsigned int &n_overlaps;          //!< Total number of inserted overlapping depletants
    unsigned int *d_n_success_forward; //!< successful reinsertions in forward move, per depletant
    unsigned int *d_n_overlap_shape_forward; //!< reinsertions into old colloid position, per depletant
    unsigned int *d_n_success_reverse; //!< successful reinsertions in reverse move, per depletant
    unsigned int *d_n_overlap_shape_reverse; //!< reinsertions into new colloid position, per depletant
    float *d_depletant_lnb;            //!< logarithm of configurational bias weight, per depletant
    const Scalar *d_d_min;             //!< Minimum insertion diameter for depletants (per type)
    const Scalar *d_d_max;             //!< Maximum insertion diameter for depletants (per type)
    bool update_shape_param;           //!< True if this is the first iteration
    mgpu::ContextPtr mgpu_context;    //!< ModernGPU context
    cudaStream_t stream;               //!< CUDA stream for kernel execution
    };

template< class Shape >
cudaError_t gpu_hpmc_implicit_count_overlaps(const hpmc_implicit_args_t &args, const typename Shape::param_type *d_params);

template< class Shape >
cudaError_t gpu_hpmc_implicit_accept_reject(const hpmc_implicit_args_t &args, const typename Shape::param_type *d_params);

#ifdef NVCC
/*!
 * Definition of function templates and templated GPU kernels
 */

//! HPMC implicit count overlaps kernel
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
    \param d_check_overlaps Interaction matrix
    \param overlap_idx Indexer for interaction matrix
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param select Current index within the loop over nselect selections (for RNG generation)
    \param d_params Per-type shape parameters

    \ingroup hpmc_kernels
*/
template< class Shape, unsigned int group_size >
__global__ void gpu_hpmc_implicit_count_overlaps_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     const Scalar4 *d_postype_old,
                                     const Scalar4 *d_orientation_old,
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
                                     curandState_t *d_state_cell,
                                     curandState_t *d_state_cell_new,
                                     const curandDiscreteDistribution_t *d_poisson,
                                     unsigned int depletant_type,
                                     unsigned int *d_overlap_cell,
                                     const unsigned int *d_active_cell_ptl_idx,
                                     const unsigned int *d_active_cell_accept,
                                     hpmc_counters_t *d_counters,
                                     hpmc_implicit_counters_t *d_implicit_counters,
                                     const unsigned int groups_per_cell,
                                     const Scalar *d_d_min,
                                     const Scalar *d_d_max,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_extra_bytes)
    {
    // flags to tell what type of thread we are
    unsigned int group;
    unsigned int offset;
    bool master;
    unsigned int n_groups;

    if (Shape::isParallel())
        {
        // use 3d thread block layout
        group = threadIdx.z;
        offset = threadIdx.y;
        master = (offset == 0 && threadIdx.x == 0);
        n_groups = blockDim.z;
        }
    else
        {
        group = threadIdx.y;
        offset = threadIdx.x;
        master = (offset == 0);
        n_groups = blockDim.y;
        }

    unsigned int err_count = 0;

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    unsigned int *s_check_overlaps = (unsigned int *)(s_params + num_types);

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
    char *s_extra = (char *)(s_check_overlaps + overlap_idx.getNumElements());

    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, max_extra_bytes);

    __syncthreads();

    // determine global CTA index
    unsigned int group_global = 0;
    group_global = blockIdx.x * n_groups + group;

    // active cell corresponding to this group
    unsigned int active_cell_idx = group_global / groups_per_cell;

    bool active = true;

    // this thread is inactive if it indexes past the end of the active cell list
    if (active_cell_idx >= n_active_cells)
        {
        active = false;
        }

    // pull in the index of our cell
    unsigned int my_cell = 0;
    unsigned int my_cell_size = 0;

    if (active)
        {
        my_cell = d_cell_set[active_cell_idx];
        my_cell_size = d_cell_size[my_cell];

        // ignore if there are no particles in this cell
        if (my_cell_size == 0)
            active = false;
        }

    // load updated particle index
    unsigned int idx_i;

    if (active)
        {
        idx_i = d_active_cell_ptl_idx[active_cell_idx];

        // if the move was not performed or has been rejected before, nothing to do here
        if (idx_i == UINT_MAX || !d_active_cell_accept[active_cell_idx])
            {
            active = false;
            }
        }

    Scalar4 postype_i;
    unsigned int type_i;
    vec3<Scalar> pos_i;
    Scalar4 orientation_i = make_scalar4(1,0,0,0);

    if (active)
        {
        // load updated particle position
        postype_i = __ldg(d_postype + idx_i);
        type_i = __scalar_as_int(postype_i.w);
        pos_i = vec3<Scalar>(postype_i);
        }

    // load RNG state per cell
    curandState_t local_state;

    if (active_cell_idx < n_active_cells)
        {
        local_state = d_state_cell[active_cell_idx];
        }

    unsigned int n_depletants;
    if (active)
        {
        // for every active cell, draw a poisson random number
        n_depletants = curand_discrete(&local_state, d_poisson[type_i]);
        }

    // save RNG state per cell
    if (active_cell_idx < n_active_cells && master && (group_global % groups_per_cell == 0))
        {
        d_state_cell_new[active_cell_idx] = local_state;
        }

    // we no longer need to do any processing for inactive cells after this point
    if (!active)
        return;

    Shape shape_i(quat<Scalar>(orientation_i), s_params[__scalar_as_int(postype_i.w)]);
    if (shape_i.hasOrientation())
        {
        orientation_i = __ldg(d_orientation + idx_i);
        shape_i.orientation = quat<Scalar>(orientation_i);
        }


    hoomd::detail::Saru rng(group_global, seed+select, timestep);

    unsigned int overlap_checks = 0;

    // number of overlapping depletants
    unsigned int n_overlap = 0;

    // number of depletants inserted
    unsigned int n_inserted = 0;

    // number of depletants in free volume
    unsigned int n_free_volume = 0;

    Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

    Scalar d_min = d_d_min[type_i];
    Scalar d_max = d_d_max[type_i];

    // iterate over depletants
    for (unsigned int i_dep = 0; i_dep < n_depletants; i_dep += groups_per_cell)
        {
        unsigned int i_dep_local = i_dep + group_global % groups_per_cell;

        unsigned int overlap = 0;

        vec3<Scalar> pos_test;

        if (i_dep_local < n_depletants)
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

            // test depletant position
            pos_test = pos_i+r*n;

            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng);
                }

            // check if depletant overlaps with the old configuration
            unsigned int excell_size = d_excell_size[my_cell];
            overlap_checks += excell_size;

            vec3<Scalar> r_ij;

            for (unsigned int k = 0; k < excell_size; k += group_size)
                {
                unsigned int local_k = k + offset;
                if (local_k < excell_size)
                    {
                    bool circumsphere_overlap = false;
                    unsigned int j;
                    Scalar4 postype_j;
                    do {
                        // read in position, and orientation of neighboring particle
                        j = __ldg(&d_excell_idx[excli(local_k, my_cell)]);

                        // check against neighbor
                        postype_j = __ldg(d_postype_old + j);
                        Shape shape_j(quat<Scalar>(), s_params[__scalar_as_int(postype_j.w)]);
                        if (shape_j.hasOrientation())
                            {
                            shape_j.orientation = quat<Scalar>(__ldg(d_orientation_old + j));
                            }

                        // test depletant in sphere around new particle position
                        r_ij = vec3<Scalar>(postype_j) - pos_test;
                        r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                        // test circumsphere overlap
                        OverlapReal rsq = dot(r_ij,r_ij);
                        OverlapReal DaDb = shape_test.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
                        circumsphere_overlap = (rsq*OverlapReal(4.0) <= DaDb * DaDb);

                        if (!circumsphere_overlap)
                            {
                            // fetch next element
                            local_k += group_size;
                            k += group_size;
                            }
                        } while(!circumsphere_overlap && (local_k < excell_size));

                    if (circumsphere_overlap)
                        {
                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(), s_params[typ_j]);
                        if (shape_j.hasOrientation())
                            {
                            shape_j.orientation = quat<Scalar>(__ldg(d_orientation_old + j));
                            }

                        if (s_check_overlaps[overlap_idx(depletant_type, typ_j)]
                            && test_overlap(r_ij, shape_test, shape_j, err_count))
                            {
                            overlap = 1;
                            }
                        }
                    }

                } // end loop over neighbors

            overlap_checks++;
            }

        hoomd::detail::WarpReduce<unsigned int, group_size> reducer;
        overlap = reducer.Sum(overlap);

        if (i_dep_local < n_depletants)
            {
            overlap_checks++;

            if (!overlap)
                {
                // depletant not overlapping in old configuration
                n_free_volume++;

                vec3<Scalar> r_ij = pos_i - pos_test;
                if (s_check_overlaps[overlap_idx(type_i, depletant_type)]
                    && test_overlap(r_ij, shape_test, shape_i, err_count))
                    {
                    // increase overlap count
                    if (master)
                        {
                        atomicAdd((unsigned int *)&d_overlap_cell[active_cell_idx], 1);
                        n_overlap++;
                        }
                    }
                }
            }
        } // end loop over depletants

    if (master)
        {
        // increment number of overlap checks
        atomicAdd(&d_counters->overlap_checks, overlap_checks);

        // increment number of overlap check errors
        atomicAdd(&d_counters->overlap_err_count, err_count);

        // increment number of inserted depletants
        atomicAdd(&d_implicit_counters->insert_count, n_inserted);

        // increment number of overlapping depletants
        atomicAdd(&d_implicit_counters->overlap_count, n_overlap);

        // increment number of depletants in free volume
        atomicAdd(&d_implicit_counters->free_volume_count, n_free_volume);
        }
    }

//! HPMC implicit depletant reinsertion kernel
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
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param select Current index within the loop over nselect selections (for RNG generation)
    \param d_params Per-type shape parameters

    \ingroup hpmc_kernels
*/
template< class Shape >
__global__ void gpu_hpmc_implicit_reinsert_kernel(Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     const Scalar4 *d_postype_old,
                                     const Scalar4 *d_orientation_old,
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
                                     Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     unsigned int depletant_type,
                                     const unsigned int *d_active_cell_ptl_idx,
                                     const unsigned int *d_active_cell_accept,
                                     hpmc_counters_t *d_counters,
                                     hpmc_implicit_counters_t *d_implicit_count,
                                     const unsigned int ntrial,
                                     unsigned int *d_n_success_forward,
                                     unsigned int *d_n_overlap_shape_forward,
                                     unsigned int *d_n_success_reverse,
                                     unsigned int *d_n_overlap_shape_reverse,
                                     const unsigned int *d_depletant_active_cell,
                                     unsigned int n_overlaps,
                                     const Scalar *d_d_min,
                                     const Scalar *d_d_max,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes)
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

    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);

    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *)(s_pos_group + n_groups);
    unsigned int *s_queue_j =   (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_overlap =   (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_overlap + n_groups);

    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;

    __shared__ unsigned int s_n_overlap_checks;
    __shared__ unsigned int s_n_overlap_errors;

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

        unsigned int num_typpairs = overlap_idx.getNumElements();
        for (unsigned int cur_offset = 0; cur_offset < num_typpairs; cur_offset += block_size)
            {
            if (cur_offset + tidx < num_typpairs)
                {
                s_check_overlaps[cur_offset + tidx] = d_check_overlaps[cur_offset + tidx];
                }
            }
        }

    // initialize the shared memory array for communicating overlaps
    if (group == 0 && master)
        {
        s_n_overlap_checks = 0;
        s_n_overlap_errors = 0;
        }

    __syncthreads();

    // initialize extra shared mem
    char *s_extra = (char *)(s_queue_gid + max_queue_size);

    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, max_extra_bytes);

    __syncthreads();

    // determine global CTA index
    unsigned int group_global = 0;
    group_global = blockIdx.x * n_groups + group;

    // is this thread active?
    bool active = true;

    unsigned int i_dep = group_global / (2*ntrial);

    if (i_dep >= n_overlaps)
        {
        active = false;
        }

    // active cell corresponding to this group
    unsigned int active_cell_idx = 0;

    if (active)
        {
        active_cell_idx = d_depletant_active_cell[i_dep];
        }

    // pull in the index of our cell
    unsigned int my_cell = 0;

    if (active)
        {
        my_cell = d_cell_set[active_cell_idx];
        }

    hoomd::detail::Saru rng(group_global, seed+select, timestep);

    unsigned int idx_i = UINT_MAX;

    if (active)
        {
        // load updated particle index
        idx_i = d_active_cell_ptl_idx[active_cell_idx];

        // if the move was not performed or has been rejected before, nothing to do here
        if (idx_i == UINT_MAX || !d_active_cell_accept[active_cell_idx])
            {
            active = false;
            }
        }

    Scalar4 postype_i = make_scalar4(0,0,0,0);
    Scalar4 orientation_i = make_scalar4(1,0,0,0);

    if (active)
        {
        // load updated particle position
        postype_i = __ldg(d_postype + idx_i);

        Shape shape_i(quat<Scalar>(), s_params[__scalar_as_int(postype_i.w)]);
        if (shape_i.hasOrientation())
            {
            orientation_i = __ldg(d_orientation + idx_i);
            }
        }

    vec3<Scalar> pos_i_new(postype_i);

    unsigned int overlap_checks = 0;

    Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

    vec3<Scalar> pos_i_old;
    if (active)
        {
        pos_i_old = vec3<Scalar>(__ldg(d_postype_old + idx_i));
        }

    unsigned int excell_size;

    if (active)
        {
        excell_size = d_excell_size[my_cell];
        }

    // we iterate over 2*ntrial attempts because for every attempt
    // we have to insert both in the new and in the old configuration
    unsigned int i_trial = group_global % (2*ntrial);

    // if we are inserting in the new configuration
    bool forward = (i_trial % 2 == 0);

    // the first depletant inserted is treated specially for the reverse move
    // because we have already inserted an overlapping depletant at the new position
    // (note we are doing forward and reverse, hence the factor of two)
    bool first_reverse = (i_trial < 2 && !forward);

    vec3<Scalar> pos_test;
    vec3<Scalar> r_ij;
    bool overlap_shape = false;

    if (master)
        {
        s_overlap[group] = 0;
        }

    if (active)
        {
        r_ij = pos_i_old - pos_i_new;
        Scalar d = fast::sqrt(dot(r_ij,r_ij));

        Scalar rmin(0.0);
        Scalar d_max = d_d_max[__scalar_as_int(postype_i.w)];
        Scalar rmax = Scalar(0.5)*d_max;

        Scalar ctheta_min(-1.0);
        bool do_rotate = false;

        // check against the updated particle
        Shape shape_i(quat<Scalar>(), s_params[__scalar_as_int(postype_i.w)]);
        Scalar R = max(shape_i.getInsphereRadius() - shape_test.getCircumsphereDiameter(),0.0);

        if (d > Scalar(0.0) && R > Scalar(0.0))
            {
            // draw a random direction in the bounded spherical shell
            Scalar ctheta = (R*R+d*d-d_max*d_max/Scalar(4.0))/(Scalar(2.0)*R*d);
            if (ctheta >= Scalar(-1.0) && ctheta < Scalar(1.0))
                {
                // true intersection, we can restrict angular sampling
                ctheta_min = ctheta;
                }

            // is there an intersection?
            if (Scalar(2.0)*d < d_max+Scalar(2.0)*R)
                {
                // sample in shell around smaller sphere
                rmin = R;
                rmax = d+d_max/Scalar(2.0);
                do_rotate = true;
                }
            }

        // draw random radial coordinate in a spherical shell
        Scalar r3 = rng.s(fast::pow(rmin/rmax,Scalar(3.0)),Scalar(1.0));
        Scalar r = rmax*fast::pow(r3,Scalar(1.0/3.0));

        // random direction in spherical shell
        Scalar z = rng.s(ctheta_min,Scalar(1.0));
        Scalar phi = Scalar(2.0*M_PI)*rng.template s<Scalar>();
        vec3<Scalar> n;
        if (do_rotate)
            {
            vec3<Scalar> u(r_ij/d);

            if (!forward)
                {
                u = -u;
                }

            // normal vector
            vec3<Scalar> v(cross(u,vec3<Scalar>(0,0,1)));
            if (dot(v,v) < EPSILON)
                {
                v = cross(u,vec3<Scalar>(0,1,0));
                }
            v *= fast::rsqrt(dot(v,v));

            quat<Scalar> q(quat<Scalar>::fromAxisAngle(u,phi));
            n = z*u+(fast::sqrt(Scalar(1.0)-z*z))*rotate(q,v);
            }
        else
            {
            n = vec3<Scalar>(fast::sqrt(Scalar(1.0)-z*z)*fast::cos(phi),fast::sqrt(Scalar(1.0)-z*z)*fast::sin(phi),z);
            }

        // test depletant position
        pos_test = r*n;

        if (forward^do_rotate)
            {
            // insert at old position of particle
            pos_test += pos_i_old;
            }
        else
            {
            // insert at new position of particle
            pos_test += pos_i_new;
            }

        if (shape_test.hasOrientation())
            {
            shape_test.orientation = generateRandomOrientation(rng);
            }

        // stash the trial move in shared memory so that other threads in this block can process overlap checks
        if (master)
            {
            s_pos_group[group] = make_scalar3(pos_test.x, pos_test.y, pos_test.z);
            s_orientation_group[group] = quat_to_scalar4(shape_test.orientation);
            }

        bool overlap_shape_old = false;
        bool overlap_shape_new = false;

        // check if depletant overlaps with particle at old position
        vec3<Scalar> r_ij = pos_i_old - pos_test;
        if (shape_i.hasOrientation())
            {
            shape_i.orientation = quat<Scalar>(__ldg(d_orientation_old + idx_i));
            }

        // if depletant can be inserted in excluded volume at old (new) position, success
        if (s_check_overlaps[overlap_idx(__scalar_as_int(postype_i.w), depletant_type)]
            && test_overlap(r_ij, shape_test, shape_i, err_count))
            {
            overlap_shape_old = true;
            }

        // check if depletant overlaps with particle at new position
        r_ij = pos_i_new - pos_test;
        if (shape_i.hasOrientation())
            {
            shape_i.orientation = quat<Scalar>(orientation_i);
            }

        // if depletant can be inserted in excluded volume at old (new) position, success
        if (s_check_overlaps[overlap_idx(__scalar_as_int(postype_i.w), depletant_type)]
            && test_overlap(r_ij, shape_test, shape_i, err_count))
            {
            overlap_shape_new = true;
            }

        // we count a possible trial insertion as one that overlaps with the old shape or the new one,
        // depending on which Rosenbluth weight we are computing
        overlap_shape = ((forward && overlap_shape_old) || (!forward && overlap_shape_new));

        if ((forward && overlap_shape_new) || (!forward && overlap_shape_old))
            {
            // shortcut, reject
            if (master) s_overlap[group] = 1;
            }

        // check if depletant overlaps with the old configuration
        if (master && overlap_shape && !s_overlap[group])
            {
            overlap_checks += excell_size;
            }
        }

    __syncthreads();

    // if the check for the updated particle fails, no need to check for overlaps with other particles
    bool trial_active = true;
    if (!overlap_shape || s_overlap[group])
        {
        trial_active = false;
        }

    if (group == 0 && master)
        {
        // initialize queue
        s_queue_size = 0;
        s_still_searching = 1;
        }
    __syncthreads();

    unsigned int k = offset;

    while (s_still_searching)
        {
        // stage 1, fill the queue.
        // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

        // active threads add to the queue
        if (trial_active)
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

                    // build some shapes, but we only need them to get diameters, so don't load orientations
                    // build shape i from shared memory
                    Scalar3 pos_i = s_pos_group[group];
                    Shape shape_i(quat<Scalar>(), s_params[depletant_type]);

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        {
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);
                        }

                    // read in position, and orientation of neighboring particle
                    postype_j = __ldg(d_postype_old + j);
                    Shape shape_j(quat<Scalar>(), s_params[__scalar_as_int(postype_j.w)]);

                    // put particle j into the coordinate system of depletant
                    r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
                    r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                    // test circumsphere overlap
                    OverlapReal rsq = dot(r_ij,r_ij);
                    OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                    if (idx_i != j && rsq*OverlapReal(4.0) <= DaDb * DaDb)
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
            } // end if trial_active

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        // when we get here, all threads have either finished their list, or encountered a full queue
        // either way, it is time to process overlaps
        // need to clear the still searching flag and sync first
        if (master && group == 0)
            s_still_searching = 0;

        unsigned int tidx_1d = offset + group*group_size;

        // max_queue_size is always <= block size, so we just need an if here
        if (tidx_1d < min(s_queue_size, max_queue_size))
            {
            // need to extract the overlap check to perform out of the shared mem queue
            unsigned int check_group = s_queue_gid[tidx_1d];
            unsigned int check_j = s_queue_j[tidx_1d];
            Scalar4 postype_j;
            Scalar4 orientation_j;

            // build shape i from shared memory
            Scalar3 pos_i = s_pos_group[check_group];
            Shape shape_i(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type]);

            // build shape j from global memory
            postype_j = __ldg(d_postype_old + check_j);
            orientation_j = make_scalar4(1,0,0,0);
            unsigned int typ_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(orientation_j), s_params[typ_j]);
            if (shape_j.hasOrientation())
                shape_j.orientation = quat<Scalar>(__ldg(d_orientation_old + check_j));

            // put particle j into the coordinate system of particle i
            r_ij = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_i);
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            if (s_check_overlaps[overlap_idx(depletant_type, typ_j)]
                && test_overlap(r_ij, shape_i, shape_j, err_count))
                {
                atomicAdd(&s_overlap[check_group], 1);
                }
            }

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            {
            s_queue_size = 0;
            }

        if (trial_active && !s_overlap[group] && k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();
        } // end while (s_still_searching)

    unsigned int overlap = s_overlap[group];

    // for every overlapping depletant
    if (active && master)
        {
        // tally into global mem

        if (forward)
            {
            if (overlap_shape)
                {
                if (!overlap)
                    {
                    atomicAdd(&d_n_success_forward[i_dep], 1);
                    }
                atomicAdd(&d_n_overlap_shape_forward[i_dep], 1);
                }
            }
        else
            {
            if (overlap_shape || first_reverse)
                {
                if (!overlap || first_reverse)
                    {
                    atomicAdd(&d_n_success_reverse[i_dep], 1);
                    }
                atomicAdd(&d_n_overlap_shape_reverse[i_dep], 1);
                }
            }

        // increment number of overlap checks
        atomicAdd(&s_n_overlap_checks, overlap_checks);

        // increment number of overlap check errors
        atomicAdd(&s_n_overlap_errors, err_count);
        }

    __syncthreads();

    if (master && group == 0)
        {
        // write out number of overlap checks
        atomicAdd(&d_counters->overlap_checks, s_n_overlap_checks);

        // write out number of overlap check errors
        atomicAdd(&d_counters->overlap_err_count, s_n_overlap_errors);
        }
    }

//! Definition of kernel to compute the configurational bias weights
__global__ void gpu_implicit_compute_weights_kernel(unsigned int n_overlaps,
             unsigned int *d_n_success_forward,
             unsigned int *d_n_overlap_shape_forward,
             unsigned int *d_n_success_reverse,
             unsigned int *d_n_overlap_shape_reverse,
             float *d_lnb,
             unsigned int *d_n_success_zero,
             unsigned int *d_depletant_active_cell);

//! Kernel to accept or reject moves on a per active cell basis
template<class Shape>
__global__ void gpu_implicit_accept_reject_kernel(
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
    const float *d_lnb,
    const unsigned int *d_n_success_zero,
    curandState_t *d_state_cell,
    unsigned int ntrial,
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

        if (n_overlap && ntrial)
            {
            // take into account reinsertion attempts

            // log of acceptance probability
            unsigned int n_success_zero = d_n_success_zero[active_cell_idx];

            Scalar lnb = d_lnb[active_cell_idx];

            if (! n_success_zero)
                {
                // load RNG state per cell
                curandState_t local_state = d_state_cell[active_cell_idx];

                // apply acceptance criterion
                accept = curand_uniform(&local_state) < expf(lnb);

                // store RNG state
                d_state_cell[active_cell_idx] = local_state;
                }
            else
                {
                accept = false;
                }
            }
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

//! Definition of kernel to set up cuRAND for the maximum kernel parameters
__global__ void gpu_curand_implicit_setup(unsigned int n_rng,
                                          unsigned int seed,
                                          unsigned int timestep,
                                          curandState_t *d_state);
/*!
 * Definition of templated GPU kernel drivers
 */

//! count overlaps kernel launcher
/*!
 * \tparam shape Shape class
 * \tparam group_size Number of threads to use per group, must be power of 2 and smaller than warp size
 *
 * Partial function template specialization is not allowed in C++, so instead we have to wrap this with a struct that
 * we are allowed to partially specialize.
 */
template<class Shape, unsigned int group_size>
struct CountOverlapsKernelLauncher
    {
    //! Launcher for the kernel
    static cudaError_t launch(const hpmc_implicit_args_t& args, const typename Shape::param_type *d_params)
        {
        if (group_size == args.group_size)
            {
            // determine the maximum block size and clamp the input block size down
            static int max_block_size = -1;
            static cudaFuncAttributes attr;
            if (max_block_size == -1)
                {
                cudaFuncGetAttributes(&attr, gpu_hpmc_implicit_count_overlaps_kernel<Shape, group_size>);
                max_block_size = attr.maxThreadsPerBlock;
                }

            // setup the grid to run the kernel
            unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
            unsigned int stride = min(block_size, args.stride);

            while (stride*group_size > block_size)
                {
                // use power-of-two block size because of warp shuffle
                return CountOverlapsKernelLauncher<Shape, group_size/2>::launch(args, d_params);
                }

            unsigned int n_groups = block_size/ (group_size * stride);

            static unsigned int n_active_cells = UINT_MAX;

            if (n_active_cells != args.n_active_cells)
                {
                // (re-) initialize cuRAND
                unsigned int block_size = 512;
                gpu_curand_implicit_setup<<<args.n_active_cells/block_size + 1,block_size, 0, args.stream>>>
                                                 (args.n_active_cells,
                                                  args.seed,
                                                  args.timestep,
                                                  args.d_state_cell);
                n_active_cells = args.n_active_cells;
                }

            unsigned int shared_bytes = args.num_types * (sizeof(typename Shape::param_type))
                + args.overlap_idx.getNumElements() * sizeof(unsigned int);

            static unsigned int base_shared_bytes = UINT_MAX;
            bool shared_bytes_changed = base_shared_bytes != shared_bytes;
            if (shared_bytes_changed != base_shared_bytes)
                base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

            unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;

            static unsigned int extra_bytes = UINT_MAX;
            if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
                {
                // required for memory coherency
                cudaDeviceSynchronize();

                // determine dynamically requested shared memory
                char *ptr_begin = nullptr;
                char *ptr =  ptr_begin;
                for (unsigned int i = 0; i < args.num_types; ++i)
                    {
                    d_params[i].load_shared(ptr, max_extra_bytes);
                    }
                extra_bytes = ptr - ptr_begin;
                }

            shared_bytes += extra_bytes;

            dim3 threads;
            if (Shape::isParallel())
                {
                // use three-dimensional thread-layout with blockDim.z < 64
                threads = dim3(stride, group_size, n_groups);
                }
            else
                {
                threads = dim3(group_size, n_groups, 1);
                }
            dim3 grid((args.n_active_cells*args.groups_per_cell)/n_groups+1, 1, 1);

            // reset counters
            cudaMemsetAsync(args.d_overlap_cell,0, sizeof(unsigned int)*args.n_active_cells, args.stream);

            // check for newly generated overlaps with depletants
            gpu_hpmc_implicit_count_overlaps_kernel<Shape, group_size><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                         args.d_orientation,
                                                                         args.d_postype_old,
                                                                         args.d_orientation_old,
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
                                                                         args.d_state_cell,
                                                                         args.d_state_cell_new,
                                                                         args.d_poisson,
                                                                         args.depletant_type,
                                                                         args.d_overlap_cell,
                                                                         args.d_active_cell_ptl_idx,
                                                                         args.d_active_cell_accept,
                                                                         args.d_counters,
                                                                         args.d_implicit_count,
                                                                         args.groups_per_cell,
                                                                         args.d_d_min,
                                                                         args.d_d_max,
                                                                         d_params,
                                                                         max_extra_bytes);

            // advance per-cell RNG states
            cudaMemcpyAsync(args.d_state_cell, args.d_state_cell_new, sizeof(curandState_t)*args.n_active_cells, cudaMemcpyDeviceToDevice, args.stream);
            }
        else
            {
            CountOverlapsKernelLauncher<Shape, group_size/2>::launch(args, d_params);
            }

        return cudaSuccess;
        }
    };

//! Template specialization to do nothing for the group_size = 0 case
template<class Shape>
struct CountOverlapsKernelLauncher<Shape, 0>
    {
    static cudaError_t launch(const hpmc_implicit_args_t& args, const typename Shape::param_type *d_params)
        {
        // do nothing
        return cudaSuccess;
        }
    };

// Kernel driver for gpu_hpmc_implicit_count_overlaps_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_implicit_count_overlaps(const hpmc_implicit_args_t& args, const typename Shape::param_type *d_params)
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
    assert(args.block_size%(args.stride*args.group_size)==0);

    // call launcher
    cudaError_t err = CountOverlapsKernelLauncher<Shape, 32>::launch(args, d_params);
    if (err != cudaSuccess)
        return err;

    // return total number of overlaps
    mgpu::Scan<mgpu::MgpuScanTypeExc>(args.d_overlap_cell, (int) args.n_active_cells, (unsigned int)0, mgpu::plus<unsigned int>(),
        (unsigned int *) 0, (args.ntrial ? &args.n_overlaps : (unsigned int *) 0), args.d_overlap_cell_scan, *args.mgpu_context);

    return cudaSuccess;
    }

//! Kernel driver for gpu_hpmc_implicit_reinsert_kernel() and gpu_hpmc_implicit_accept_reject_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_implicit_accept_reject(const hpmc_implicit_args_t& args, const typename Shape::param_type *d_params)
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

    #ifdef ENABLE_HPMC_REINSERT
    if (args.n_overlaps && args.ntrial > 0)
        {
        // construct the lookup of active cell idx per depletant
        mgpu::IntervalExpand(args.n_overlaps, args.d_overlap_cell_scan,
            mgpu::counting_iterator<unsigned int>(0), args.n_active_cells, args.d_depletant_active_cell,
            *args.mgpu_context);

        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static cudaFuncAttributes attr;
        if (max_block_size == -1)
            {
            cudaFuncGetAttributes(&attr, gpu_hpmc_implicit_reinsert_kernel<Shape>);
            max_block_size = attr.maxThreadsPerBlock;
            }

        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);
        unsigned int stride = min(block_size, args.stride);


        // the new block size might not be a multiple of group size, decrease group size until it is
        unsigned int group_size = args.group_size;

        while ((block_size%(stride*group_size)) != 0)
            {
            // decrease block_size further until it is a multiple of group_size
            // (because the kernel uses warp-shuffle instructions we cannot use non-power-of-two group sizes)
            block_size--;
            }
        // setup the grid to run the kernel
        unsigned int n_groups = block_size / group_size / stride;

        unsigned int shared_bytes = n_groups * (sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3)) +
                                    block_size*sizeof(unsigned int)*2 +
                                    args.num_types * (sizeof(typename Shape::param_type)) +
                                    args.overlap_idx.getNumElements() * sizeof(unsigned int);

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            block_size -= args.devprop.warpSize;

            // the new block size might not be a multiple of group size, decrease group size until it is
            group_size = args.group_size;

            while ((block_size%(stride*group_size)) != 0)
                {
                block_size--;
                }

            n_groups = block_size / group_size / stride;
            shared_bytes = n_groups * (sizeof(unsigned int) + sizeof(Scalar4) + sizeof(Scalar3)) +
                           block_size*sizeof(unsigned int)*2 +
                           args.num_types * (sizeof(typename Shape::param_type)) +
                           args.overlap_idx.getNumElements() * sizeof(unsigned int);
            }

        static unsigned int base_shared_bytes = UINT_MAX;
        bool shared_bytes_changed = base_shared_bytes != shared_bytes;
        if (shared_bytes_changed != base_shared_bytes)
            base_shared_bytes = shared_bytes + attr.sharedSizeBytes;

        unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;

        static unsigned int extra_bytes = UINT_MAX;
        if (extra_bytes == UINT_MAX || args.update_shape_param || shared_bytes_changed)
            {
            // required for memory coherency
            cudaDeviceSynchronize();

            // determine dynamically requested shared memory
            char *ptr_begin = nullptr;
            char *ptr =  ptr_begin;
            for (unsigned int i = 0; i < args.num_types; ++i)
                {
                d_params[i].load_shared(ptr, max_extra_bytes);
                }
            extra_bytes = ptr - ptr_begin;
            }

        shared_bytes += extra_bytes;

        // reset counters
        cudaMemsetAsync(args.d_n_success_forward,0, sizeof(unsigned int)*args.n_overlaps, args.stream);
        cudaMemsetAsync(args.d_n_overlap_shape_forward,0, sizeof(unsigned int)*args.n_overlaps, args.stream);
        cudaMemsetAsync(args.d_n_success_reverse,0, sizeof(unsigned int)*args.n_overlaps, args.stream);
        cudaMemsetAsync(args.d_n_overlap_shape_reverse,0, sizeof(unsigned int)*args.n_overlaps, args.stream);

        dim3 threads;
        if (Shape::isParallel())
            {
            // use three-dimensional thread-layout with blockDim.z < 64
            threads = dim3(stride, group_size, n_groups);
            }
        else
            {
            threads = dim3(group_size, n_groups, 1);
            }
        dim3 grid((args.n_overlaps*2*args.ntrial)/n_groups+1, 1, 1);

        // check for newly generated overlaps with depletants
        gpu_hpmc_implicit_reinsert_kernel<Shape><<<grid, threads, shared_bytes, args.stream>>>(args.d_postype,
                                                                     args.d_orientation,
                                                                     args.d_postype_old,
                                                                     args.d_orientation_old,
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
                                                                     args.depletant_type,
                                                                     args.d_active_cell_ptl_idx,
                                                                     args.d_active_cell_accept,
                                                                     args.d_counters,
                                                                     args.d_implicit_count,
                                                                     args.ntrial,
                                                                     args.d_n_success_forward,
                                                                     args.d_n_overlap_shape_forward,
                                                                     args.d_n_success_reverse,
                                                                     args.d_n_overlap_shape_reverse,
                                                                     args.d_depletant_active_cell,
                                                                     args.n_overlaps,
                                                                     args.d_d_min,
                                                                     args.d_d_max,
                                                                     d_params,
                                                                     block_size,
                                                                     max_extra_bytes);

        block_size = 256;

        // reset flags
        cudaMemsetAsync(args.d_n_success_zero,0, sizeof(unsigned int)*args.n_active_cells, args.stream);

        // compute logarithm of configurational bias weights per active cell
        gpu_implicit_compute_weights_kernel<<<args.n_overlaps/block_size+1,block_size, 0, args.stream>>>(args.n_overlaps,
             args.d_n_success_forward,
             args.d_n_overlap_shape_forward,
             args.d_n_success_reverse,
             args.d_n_overlap_shape_reverse,
             args.d_depletant_lnb,
             args.d_n_success_zero,
             args.d_depletant_active_cell);

        // do a segmented reduction
        mgpu::SegReduceCsr(args.d_depletant_lnb, args.d_overlap_cell_scan, args.n_overlaps,
            args.n_active_cells, true, args.d_lnb, 0.0f, mgpu::plus<float>(), *args.mgpu_context);
        }
    #endif

    // accept-reject on a per cell basis
    unsigned int block_size = 256;
    gpu_implicit_accept_reject_kernel<Shape><<<args.n_active_cells/block_size + 1, block_size, 0, args.stream>>>(
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
        args.d_lnb,
        args.d_n_success_zero,
        args.d_state_cell,
        args.ntrial,
        d_params);

    return cudaSuccess;
    }

#endif // NVCC

}; // end namespace detail

} // end namespace hpmc

#endif // _HPMC_IMPLICIT_CUH_

