// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BoxDim.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/Moves.h"
#include <hip/hip_runtime.h>

#include "hoomd/hpmc/HPMCCounters.h"

#include "GPUHelpers.cuh"
#include "HPMCMiscFunctions.h"

#include "IntegratorHPMCMonoGPUDepletants.cuh"
#include "IntegratorHPMCMonoGPUDepletantsAuxilliaryTypes.cuh"

#include <cassert>

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit compile time
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

#ifdef __HIPCC__
namespace kernel
    {
//! Kernel for computing the depletion Metropolis-Hastings weight  (phase 1)
template<class Shape, unsigned int max_threads>
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
    __global__ void hpmc_insert_depletants_phase1(const Scalar4* d_trial_postype,
                                                  const Scalar4* d_trial_orientation,
                                                  const unsigned int* d_trial_move_type,
                                                  const Scalar4* d_postype,
                                                  const Scalar4* d_orientation,
                                                  hpmc_counters_t* d_counters,
                                                  const unsigned int* d_excell_idx,
                                                  const unsigned int* d_excell_size,
                                                  const Index2D excli,
                                                  const uint3 cell_dim,
                                                  const Scalar3 ghost_width,
                                                  const Index3D ci,
                                                  const unsigned int N_local,
                                                  const unsigned int num_types,
                                                  const unsigned int seed,
                                                  const unsigned int* d_check_overlaps,
                                                  const Index2D overlap_idx,
                                                  const uint64_t timestep,
                                                  const unsigned int dim,
                                                  const BoxDim box,
                                                  const unsigned int select,
                                                  unsigned int* d_reject_out_of_cell,
                                                  const typename Shape::param_type* d_params,
                                                  unsigned int max_queue_size,
                                                  unsigned int max_extra_bytes,
                                                  unsigned int depletant_type_a,
                                                  hpmc_implicit_counters_t* d_implicit_counters,
                                                  const unsigned int* d_update_order_by_ptl,
                                                  const unsigned int* d_reject_in,
                                                  const unsigned int ntrial,
                                                  const unsigned int* d_tag,
                                                  const Scalar4* d_vel,
                                                  const Scalar4* d_trial_vel,
                                                  int* d_deltaF_int,
                                                  bool repulsive,
                                                  unsigned int work_offset,
                                                  unsigned int max_depletant_queue_size,
                                                  const unsigned int* d_n_depletants,
                                                  const unsigned int global_work_offset,
                                                  const unsigned int n_work_local,
                                                  const unsigned int max_n_depletants)
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0) && (threadIdx.x == 0);
    unsigned int n_groups = blockDim.y;

    unsigned int err_count = 0;

    // shared particle configuation
    __shared__ Scalar4 s_orientation_i_new;
    __shared__ Scalar4 s_orientation_i_old;
    __shared__ Scalar3 s_pos_i_new;
    __shared__ Scalar3 s_pos_i_old;
    __shared__ unsigned int s_type_i;

    // shared arrays for per type pair parameters
    __shared__ unsigned int s_overlap_checks;
    __shared__ unsigned int s_overlap_err_count;

    // shared queue variables
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_adding_depletants;
    __shared__ unsigned int s_depletant_queue_size;

    // per particle free energy in units of log(1+1/ntrial)
    __shared__ int s_deltaF_int;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED(char, s_data)
    typename Shape::param_type* s_params = (typename Shape::param_type*)(&s_data[0]);
    Scalar4* s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3* s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int* s_check_overlaps = (unsigned int*)(s_pos_group + n_groups);
    unsigned int* s_overlap_group
        = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int* s_queue_j = (unsigned int*)(s_overlap_group + n_groups);
    unsigned int* s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int* s_queue_didx = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int* s_queue_itrial = (unsigned int*)(s_queue_didx + max_depletant_queue_size);

        // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx
            = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        unsigned int block_size = blockDim.x * blockDim.y * blockDim.z;
        unsigned int param_size = num_types * sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int*)s_params)[cur_offset + tidx] = ((int*)d_params)[cur_offset + tidx];
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
    char* s_extra = (char*)(s_queue_itrial + max_depletant_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    // initialize the shared memory array for communicating overlaps
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;

        s_deltaF_int = 0;
        }

    __syncthreads();

    // identify the particle that this block handles
    unsigned int i = blockIdx.x + work_offset;

    // if this particle is rejected a priori because it has left the cell or is in
    // an inactive region, don't check overlaps
    if (d_reject_out_of_cell[i] || !d_trial_move_type[i])
        return;

    // load updated particle position
    if (master && group == 0)
        {
        Scalar4 postype_i_new = d_trial_postype[i];
        Scalar4 postype_i_old = d_postype[i];
        s_pos_i_new = make_scalar3(postype_i_new.x, postype_i_new.y, postype_i_new.z);
        s_pos_i_old = make_scalar3(postype_i_old.x, postype_i_old.y, postype_i_old.z);
        s_type_i = __scalar_as_int(postype_i_new.w);

        s_orientation_i_new = d_trial_orientation[i];
        s_orientation_i_old = d_orientation[i];
        }

    // sync so that s_pos_i_old etc. are available
    __syncthreads();

    unsigned int overlap_checks = 0;

    // find the cell this particle should be in
    unsigned int my_cell = computeParticleCell(s_pos_i_old, box, ghost_width, cell_dim, ci, false);

    // the order of this particle in the chain
    unsigned int update_order_i = d_update_order_by_ptl[i];
    unsigned int seed_i_old = __scalar_as_int(d_vel[i].x);
    unsigned int seed_i_new = __scalar_as_int(d_trial_vel[i].x);
    unsigned int tag_i = d_tag[i];

    if (master && group == 0)
        {
        s_depletant_queue_size = 0;
        s_adding_depletants = 1;
        }

    __syncthreads();

    // unpack the block index to find out which work element this group handles
    unsigned int block = blockIdx.z * gridDim.y + blockIdx.y;
    unsigned int grid = gridDim.y * gridDim.z;

    unsigned int local_work_idx
        = offset + group * group_size + n_groups * group_size * (block >> 1);
    unsigned int blocks_per_depletant = grid >> 1;

    // we always have a number of blocks that is a multiple of two
    bool new_config = block & 1;

    detail::OBB obb_i;
        {
        // get shape OBB
        Shape shape_i(quat<Scalar>(!new_config ? d_orientation[i] : d_trial_orientation[i]),
                      s_params[s_type_i]);
        obb_i = shape_i.getOBB(new_config ? vec3<Scalar>(s_pos_i_new) : vec3<Scalar>(s_pos_i_old));

        // extend by depletant radius
        Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);

        ShortReal r = 0.5 * shape_test_a.getCircumsphereDiameter();
        obb_i.lengths.x += r;
        obb_i.lengths.y += r;
        obb_i.lengths.z += r;
        }

    /*
     * Phase 1: insert into particle i's excluded volume
     */

    unsigned int n_depletants = 0;

    while (s_adding_depletants)
        {
        // compute global indices
        unsigned int global_work_idx = global_work_offset + local_work_idx;
        unsigned int i_dep = global_work_idx % max_n_depletants;
        unsigned int i_trial = global_work_idx / max_n_depletants;

        // load random number of depletants from Poisson distribution
        unsigned int n_depletants_i
            = d_n_depletants[i * 2 * ntrial + new_config * ntrial + i_trial];

        if (i_dep == 0)
            n_depletants += n_depletants_i;

        while (s_depletant_queue_size < max_depletant_queue_size && local_work_idx < n_work_local)
            {
            if (i_trial < ntrial && i_dep < n_depletants_i)
                {
                // one RNG per depletant and trial insertion
                hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletants, 0, 0),
                                           hoomd::Counter(new_config ? seed_i_new : seed_i_old,
                                                          i_dep,
                                                          i_trial,
                                                          depletant_type_a));

                // filter depletants overlapping with particle i
                vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));

                Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
                quat<Scalar> o;
                if (shape_test_a.hasOrientation())
                    {
                    o = generateRandomOrientation(rng, dim);
                    }
                if (shape_test_a.hasOrientation())
                    shape_test_a.orientation = o;

                Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
                if (shape_i.hasOrientation())
                    shape_i.orientation
                        = quat<Scalar>(new_config ? s_orientation_i_new : s_orientation_i_old);
                vec3<Scalar> r_ij = vec3<Scalar>(new_config ? s_pos_i_new : s_pos_i_old) - pos_test;
                overlap_checks++;
                bool overlap_i_a = (s_check_overlaps[overlap_idx(s_type_i, depletant_type_a)]
                                    && check_circumsphere_overlap(r_ij, shape_test_a, shape_i)
                                    && test_overlap(r_ij, shape_test_a, shape_i, err_count));

                if (overlap_i_a)
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_depletant_queue_size, 1);

                    if (insert_point < max_depletant_queue_size)
                        {
                        s_queue_didx[insert_point] = i_dep;
                        s_queue_itrial[insert_point] = i_trial;
                        }
                    else
                        {
                        // we will recheck and insert this on the next time through
                        break;
                        }
                    } // end if add_to_queue
                }

            // advance local depletant idx
            local_work_idx += group_size * n_groups * blocks_per_depletant;

            // compute new global index
            global_work_idx = global_work_offset + local_work_idx;
            i_dep = global_work_idx % max_n_depletants;
            i_trial = global_work_idx / max_n_depletants;
            n_depletants_i = d_n_depletants[i * 2 * ntrial + new_config * ntrial + i_trial];
            } // end while (s_depletant_queue_size < max_depletant_queue_size && i_dep <
              // n_depletants_i)

        __syncthreads();

        // process the queue, group by group
        if (master && group == 0)
            s_adding_depletants = 0;

        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // is this group processing work from the first queue?
        bool active = group < min(s_depletant_queue_size, max_depletant_queue_size);

        if (active)
            {
            // regenerate depletant using seed from queue, this costs a few flops but is probably
            // better than storing one Scalar4 and a Scalar3 per thread in shared mem
            unsigned int i_dep_queue = s_queue_didx[group];
            unsigned int i_trial_queue = s_queue_itrial[group];

            hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletants, 0, 0),
                                       hoomd::Counter(new_config ? seed_i_new : seed_i_old,
                                                      i_dep_queue,
                                                      i_trial_queue,
                                                      depletant_type_a));

            // depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));
            Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
            quat<Scalar> o;
            if (shape_test_a.hasOrientation())
                {
                o = generateRandomOrientation(rng, dim);
                }

            // store them per group
            if (master)
                {
                s_pos_group[group] = vec_to_scalar3(pos_test);
                s_orientation_group[group] = quat_to_scalar4(o);
                }
            }

        if (master)
            s_overlap_group[group] = 0;

        __syncthreads();

        // counters to track progress through the loop over potential neighbors
        unsigned int excell_size;
        unsigned int k = offset;

        if (active)
            {
            excell_size = d_excell_size[my_cell];

            if (master)
                overlap_checks += excell_size;
            }

        // loop while still searching
        while (s_still_searching)
            {
            // fill the neighbor queue
            // loop through particles in the excell list and add them to the queue if they pass the
            // circumsphere check

            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached
                // the end of our own list and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size && !s_overlap_group[group])
                    {
                    // build some shapes, but we only need them to get diameters, so don't load
                    // orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    unsigned int tag_j = d_tag[j];

                    // has j been updated? ghost particles are not updated

                    // these multiple gmem loads present a minor optimization opportunity for the
                    // future
                    bool j_has_been_updated = j < N_local
                                              && d_update_order_by_ptl[j] < update_order_i
                                              && !d_reject_in[j] && d_trial_move_type[j];

                    // true if particle j is in the old configuration
                    bool old = !j_has_been_updated;

                    // read in position of neighboring particle, do not need it's orientation for
                    // circumsphere check for ghosts always load particle data
                    Scalar4 postype_j = (old || j >= N_local) ? d_postype[j] : d_trial_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test_a(quat<Scalar>(s_orientation_group[group]),
                                       s_params[depletant_type_a]);

                    // put particle j into the coordinate system of the test particle
                    vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool insert_in_queue = i != j && (old || j < N_local) && tag_i < tag_j;

                    bool circumsphere_overlap
                        = s_check_overlaps[overlap_idx(depletant_type_a, type_j)]
                          && check_circumsphere_overlap(r_jk, shape_test_a, shape_j);

                    insert_in_queue &= circumsphere_overlap;

                    if (insert_in_queue)
                        {
                        // add this particle to the queue
                        unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                        if (insert_point < max_queue_size)
                            {
                            s_queue_gid[insert_point] = group;
                            s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
                            }
                        else
                            {
                            // or back up if the queue is already full
                            // we will recheck and insert this on the next time through
                            k -= group_size;
                            }
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            // when we get here, all threads have either finished their list, or encountered a full
            // queue either way, it is time to process overlaps need to clear the still searching
            // flag and sync first
            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size * group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j_flag = s_queue_j[tidx_1d];
                bool check_old = check_j_flag & 1;
                unsigned int check_j = check_j_flag >> 1;

                // build depletant shape from shared memory
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test_a(quat<Scalar>(s_orientation_group[check_group]),
                                   s_params[depletant_type_a]);

                // build shape j from global memory
                Scalar4 postype_j = check_old ? d_postype[check_j] : d_trial_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(check_old ? d_orientation[check_j]
                                                                 : d_trial_orientation[check_j]);

                // put particle j into the coordinate system of the test particle
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                bool overlap_j_a = s_check_overlaps[overlap_idx(depletant_type_a, type_j)]
                                   && test_overlap(r_jk, shape_test_a, shape_j, err_count);

                bool overlap_j_b = overlap_j_a;
                bool overlap_i_a = true;
                bool overlap_i_b = true;

                if ((overlap_i_a && overlap_j_b) || (overlap_i_b && overlap_j_a))
                    atomicAdd(&s_overlap_group[check_group], 1);
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (!s_overlap_group[group] && active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        if (active && master && s_overlap_group[group])
            {
            if ((new_config && !repulsive) || (!new_config && repulsive))
                atomicAdd(&s_deltaF_int, 1); // numerator
            else
                atomicAdd(&s_deltaF_int, -1); // denominator
            }

        // do we still need to process depletants?
        __syncthreads();
        if (master && group == 0)
            s_depletant_queue_size = 0;
        if (local_work_idx < n_work_local)
            atomicAdd(&s_adding_depletants, 1);
        __syncthreads();
        } // end loop over depletants

    // write out free energy for this particle
    if (master && group == 0)
        {
        atomicAdd(&d_deltaF_int[i], s_deltaF_int);
        }

    if (err_count > 0)
        atomicAdd(&s_overlap_err_count, err_count);

    // count the overlap checks
    atomicAdd(&s_overlap_checks, overlap_checks);

    __syncthreads();

    // final tally into global mem
    if (master && group == 0)
        {
#if (__CUDA_ARCH__ >= 600)
        atomicAdd_system(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd_system(&d_counters->overlap_checks, s_overlap_checks);
#else
        atomicAdd(&d_counters->overlap_err_count, s_overlap_err_count);
        atomicAdd(&d_counters->overlap_checks, s_overlap_checks);
#endif
        }

    if (n_depletants)
        {
        Shape shape_i(quat<Scalar>(quat<Scalar>()), s_params[s_type_i]);
        bool ignore_stats = shape_i.ignoreStatistics();
        if (!ignore_stats)
            {
// increment number of inserted depletants
#if (__CUDA_ARCH__ >= 600)
            atomicAdd_system(&d_implicit_counters[depletant_type_a].insert_count, n_depletants);
#else
            atomicAdd(&d_implicit_counters[depletant_type_a].insert_count, n_depletants);
#endif
            }
        }
    }

//! Launcher for hpmc_insert_depletants_phase1 kernel with templated launch bounds
template<class Shape, unsigned int cur_launch_bounds>
void depletants_launcher_phase1(const hpmc_args_t& args,
                                const hpmc_implicit_args_t& implicit_args,
                                const hpmc_auxilliary_args_t& auxilliary_args,
                                const typename Shape::param_type* params,
                                unsigned int max_threads,
                                detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds * MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        int max_block_size;
        hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero
            = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        hipFuncGetAttributes(
            &attr,
            reinterpret_cast<const void*>(
                &kernel::hpmc_insert_depletants_phase1<Shape,
                                                       launch_bounds_nonzero * MIN_BLOCK_SIZE>));
        max_block_size = attr.maxThreadsPerBlock;

        // choose a block size based on the max block size by regs (max_block_size) and include
        // dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp, block_size);
        tpp = std::min((unsigned int)args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
        unsigned int n_groups = block_size / tpp;

        unsigned int max_queue_size = n_groups * tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const size_t min_shared_bytes = args.num_types * sizeof(typename Shape::param_type)
                                        + args.overlap_idx.getNumElements() * sizeof(unsigned int);

        size_t shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int))
                              + max_queue_size * 2 * sizeof(unsigned int)
                              + max_depletant_queue_size * 2 * sizeof(unsigned int)
                              + min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of "
                                     "particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            block_size -= args.devprop.warpSize;
            if (block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");
            tpp = min(tpp, block_size);
            tpp = std::min((unsigned int)args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
            n_groups = block_size / tpp;

            max_queue_size = n_groups * tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int))
                           + max_queue_size * 2 * sizeof(unsigned int)
                           + max_depletant_queue_size * 2 * sizeof(unsigned int) + min_shared_bytes;
            }

        // determine dynamically requested shared memory
        unsigned int base_shared_bytes
            = static_cast<unsigned int>(shared_bytes + attr.sharedSizeBytes);
        unsigned int max_extra_bytes
            = static_cast<unsigned int>(args.devprop.sharedMemPerBlock - base_shared_bytes);
        char* ptr = (char*)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        unsigned int extra_bytes = max_extra_bytes - available_bytes;
        shared_bytes += extra_bytes;

        // setup the grid to run the kernel
        dim3 threads(1, n_groups, tpp);

        // iterate over particles of this rank only
        for (int idev = auxilliary_args.gpu_partition_rank.getNumActiveGPUs() - 1; idev >= 0;
             --idev)
            {
            auto range = auxilliary_args.gpu_partition_rank.getRangeAndSetGPU(idev);

            if (range.first == range.second)
                continue;

            unsigned int blocks_per_particle
                = auxilliary_args.nwork_local[idev]
                      / (implicit_args.depletants_per_thread * n_groups * tpp)
                  + 1;

            dim3 grid(range.second - range.first, 2 * blocks_per_particle, 1);

            if (blocks_per_particle > static_cast<unsigned int>(args.devprop.maxGridSize[1]))
                {
                grid.y = args.devprop.maxGridSize[1];
                grid.z = 2 * blocks_per_particle / args.devprop.maxGridSize[1] + 1;
                }

            assert(args.d_trial_postype);
            assert(args.d_trial_orientation);
            assert(args.d_trial_move_type);
            assert(args.d_postype);
            assert(args.d_orientation);
            assert(args.d_counters);
            assert(args.d_excell_idx);
            assert(args.d_excell_size);
            assert(args.d_check_overlaps);
            assert(args.d_reject_out_of_cell);
            assert(implicit_args.d_implicit_count);
            assert(args.d_update_order_by_ptl);
            assert(args.d_reject_in);
            assert(auxilliary_args.d_tag);
            assert(auxilliary_args.d_vel);
            assert(auxilliary_args.d_trial_vel);
            assert(auxilliary_args.d_deltaF_int);
            assert(auxilliary_args.d_n_depletants_ntrial);

            hipLaunchKernelGGL(
                (kernel::hpmc_insert_depletants_phase1<Shape,
                                                       launch_bounds_nonzero * MIN_BLOCK_SIZE>),
                dim3(grid),
                dim3(threads),
                shared_bytes,
                auxilliary_args.streams_phase1[idev],
                args.d_trial_postype,
                args.d_trial_orientation,
                args.d_trial_move_type,
                args.d_postype,
                args.d_orientation,
                args.d_counters + idev * args.counters_pitch,
                args.d_excell_idx,
                args.d_excell_size,
                args.excli,
                args.cell_dim,
                args.ghost_width,
                args.ci,
                args.N,
                args.num_types,
                args.seed,
                args.d_check_overlaps,
                args.overlap_idx,
                args.timestep,
                args.dim,
                args.box,
                args.select,
                args.d_reject_out_of_cell,
                params,
                max_queue_size,
                max_extra_bytes,
                implicit_args.depletant_type_a,
                implicit_args.d_implicit_count + idev * implicit_args.implicit_counters_pitch,
                args.d_update_order_by_ptl,
                args.d_reject_in,
                auxilliary_args.ntrial,
                auxilliary_args.d_tag,
                auxilliary_args.d_vel,
                auxilliary_args.d_trial_vel,
                auxilliary_args.d_deltaF_int,
                implicit_args.repulsive,
                range.first,
                max_depletant_queue_size,
                auxilliary_args.d_n_depletants_ntrial,
                auxilliary_args.work_offset[idev],
                auxilliary_args.nwork_local[idev],
                implicit_args.max_n_depletants[idev]);
            }
        }
    else
        {
        depletants_launcher_phase1<Shape>(args,
                                          implicit_args,
                                          auxilliary_args,
                                          params,
                                          max_threads,
                                          detail::int2type<cur_launch_bounds / 2>());
        }
    }

    } // end namespace kernel

//! Kernel driver for kernel::insert_depletants_phase1()
/*! \param args Bundled arguments
    \param implicit_args Bundled arguments related to depletants
    \param auxilliary_args Bundled arguments for auxillary variable depletants
    \param d_params Per-type shape parameters

    This templatized method is the kernel driver for HPMC update of any shape. It is instantiated
   for every shape at the bottom of this file.

    \ingroup hpmc_kernels
*/
template<class Shape>
void hpmc_depletants_auxilliary_phase1(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxilliary_args_t& auxilliary_args,
                                       const typename Shape::param_type* params)
    {
    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::depletants_launcher_phase1<Shape>(args,
                                              implicit_args,
                                              auxilliary_args,
                                              params,
                                              launch_bounds,
                                              detail::int2type<MAX_BLOCK_SIZE / MIN_BLOCK_SIZE>());
    }
#endif

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE

    } // end namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd
