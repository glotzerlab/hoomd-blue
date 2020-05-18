// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/BoxDim.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/hpmc/Moves.h"
#include "hoomd/GPUPartition.cuh"

#include "hoomd/hpmc/HPMCCounters.h"

#include "GPUHelpers.cuh"
#include "HPMCMiscFunctions.h"

#include "IntegratorHPMCMonoGPUDepletantsAuxilliaryTypes.cuh"
#include "IntegratorHPMCMonoGPUDepletants.cuh"

#include <cassert>

namespace hpmc {

namespace gpu {

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit the number of template instantiations
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

//! Driver for kernel::hpmc_insert_depletants_auxilliary_phase2()
template< class Shape >
void hpmc_depletants_auxilliary_phase2(const hpmc_args_t& args,
    const hpmc_implicit_args_t& implicit_args,
    const hpmc_auxilliary_args_t& auxilliary_args,
    const typename Shape::param_type *params);

#ifdef __HIPCC__
namespace kernel
{

//! Kernel for computing the depletion Metropolis-Hastings weight (phase 2)
template< class Shape, unsigned int max_threads, bool pairwise >
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
__global__ void hpmc_insert_depletants_phase2(const Scalar4 *d_trial_postype,
                                     const Scalar4 *d_trial_orientation,
                                     const unsigned int *d_trial_move_type,
                                     const Scalar4 *d_postype,
                                     const Scalar4 *d_orientation,
                                     hpmc_counters_t *d_counters,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const Scalar3 ghost_width,
                                     const Index3D ci,
                                     const unsigned int N_local,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int *d_check_overlaps,
                                     const Index2D overlap_idx,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     const unsigned int select,
                                     unsigned int *d_reject_out_of_cell,
                                     const typename Shape::param_type *d_params,
                                     unsigned int max_queue_size,
                                     unsigned int max_extra_bytes,
                                     unsigned int depletant_type_a,
                                     unsigned int depletant_type_b,
                                     const Index2D depletant_idx,
                                     hpmc_implicit_counters_t *d_implicit_counters,
                                     const unsigned int *d_update_order_by_ptl,
                                     const unsigned int *d_reject_in,
                                     const unsigned int ntrial,
                                     const unsigned int *d_tag,
                                     const Scalar4 *d_vel,
                                     const Scalar4 *d_trial_vel,
                                     int *d_deltaF_int,
                                     bool repulsive,
                                     unsigned int work_offset,
                                     const unsigned int *d_n_depletants)
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0) && (threadIdx.x==0);
    unsigned int n_groups = blockDim.z;

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
    HIP_DYNAMIC_SHARED( char, s_data)
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);
    Scalar4 *s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3 *s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int *s_j_group = (unsigned int *) (s_pos_group + n_groups);
    unsigned int *s_overlap_group = (unsigned int *)(s_j_group + n_groups);
    unsigned int *s_check_overlaps = (unsigned int *) (s_overlap_group + n_groups);
    unsigned int *s_queue_j = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int *s_queue_k = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int *s_queue_gid = (unsigned int*)(s_queue_k + max_queue_size);
    unsigned int *s_queue_tid = (unsigned int*)(s_queue_gid + max_queue_size);
    unsigned int max_depletant_queue_size = n_groups;
    unsigned int *s_queue_offset = (unsigned int *)(s_queue_tid + max_depletant_queue_size);
    unsigned int *s_queue_didx = (unsigned int *)(s_queue_offset + max_depletant_queue_size);

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
    char *s_extra = (char *)(s_queue_didx + max_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    // initialize shared memory
    if (master && group == 0)
        {
        s_overlap_checks = 0;
        s_overlap_err_count = 0;

        s_deltaF_int = 0;
        }

    __syncthreads();

    // identify the active cell that this thread handles
    unsigned int i = blockIdx.x + work_offset;

    // if this particle is rejected a priori because it has left the cell, don't check overlaps
    // and avoid out of range memory access when computing the cell
    if (d_reject_out_of_cell[i])
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
    unsigned int my_cell = computeParticleCell(s_pos_i_old, box, ghost_width,
        cell_dim, ci, false);

    // the order of this particle in the chain
    unsigned int update_order_i = d_update_order_by_ptl[i];
    unsigned int tag_i = d_tag[i];

    if (master && group == 0)
        {
        s_still_searching = 1;
        s_queue_size = 0;
        }

    __syncthreads();

    /*
     * Phase 2: insert into neighbor particle excluded volumes
     */

    // count depletant insertions
    unsigned int n_depletants = 0;

    // fill the neighbor queue for i using n_groups*group_size threads
    unsigned int k = offset;

    unsigned int excell_size = d_excell_size[my_cell];

    // unpack the block index
    unsigned int gconfig = (blockIdx.z >> 1)/ntrial;
    unsigned int dim_config = (gridDim.z >> 1)/ntrial;
    unsigned int gidx = gridDim.y*gconfig+blockIdx.y;
    unsigned int new_config = blockIdx.z & 1;
    unsigned int i_trial = (blockIdx.z >> 1) % ntrial;
    unsigned int blocks_per_depletant = gridDim.y*dim_config;

    // loop while still searching
    while (s_still_searching)
        {
        // one depletant per group of neighbors
        unsigned int i_dep = group + gidx*n_groups;

        // prefetch j
        unsigned int j, next_j = 0;
        if (k < excell_size)
            next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

        // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
        while (s_queue_size < max_queue_size && k < excell_size)
            {
            // prefetch next j
            k += group_size;
            j = next_j;

            if (k < excell_size)
                next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

            // has j been updated? ghost particles are not updated
            bool j_has_been_updated = j < N_local &&
                d_update_order_by_ptl[j] < update_order_i &&
                !d_reject_in[j] &&
                d_trial_move_type[j];

            // true if particle j is in the old configuration
            bool old = !j_has_been_updated;

            // read in position of neighboring particle, do not need it's orientation for circumsphere check
            // for ghosts always load particle data
            Scalar4 postype_j = (old || j >= N_local) ? d_postype[j] : d_trial_postype[j];
            unsigned int type_j = __scalar_as_int(postype_j.w);
            Shape shape_j(quat<Scalar>(), s_params[type_j]);

            // load test particle configuration from shared mem
            vec3<Scalar> pos_i(new_config ? s_pos_i_new : s_pos_i_old);
            Shape shape_i(quat<Scalar>(), s_params[s_type_i]);

            // put particle j into the coordinate system of particle i
            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i;
            r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

            bool insert_in_queue = i != j && (old || j < N_local);

            Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
            Shape shape_test_b(quat<Scalar>(), s_params[depletant_type_b]);

            OverlapReal rsq(dot(r_ij, r_ij));
            OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();
            DaDb += shape_test_a.getCircumsphereDiameter() + shape_test_b.getCircumsphereDiameter();
            bool excluded_volume_overlap = (s_check_overlaps[overlap_idx(depletant_type_a, type_j)] ||
                s_check_overlaps[overlap_idx(depletant_type_b, type_j)]) &&
                (OverlapReal(4.0)*rsq <= DaDb*DaDb);

            insert_in_queue &= excluded_volume_overlap;

            if (insert_in_queue)
                {
                // add this particle to the queue
                unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                if (insert_point < max_queue_size)
                    {
                    // store this group's depletant index
                    s_queue_j[insert_point] = (j << 1) | (old ? 1 : 0);
                    s_queue_didx[insert_point] = i_dep;
                    }
                else
                    {
                    // or back up if the queue is already full
                    // we will recheck and insert this on the next time through
                    k -= group_size;
                    }
                } // end if k < excell_size
            } // end while (s_queue_size < max_queue_size && k < excell_size)

        // sync to make sure all threads in the block are caught up
        __syncthreads();

        if (master && group == 0)
            {
            s_depletant_queue_size = 0;
            s_adding_depletants = 1;
            }

        // max_queue_size is always <= block size, so we just need an if here
        unsigned int tidx_1d = offset + group_size*group;
        bool active = tidx_1d < min(s_queue_size, max_queue_size);

        // a few variables
        unsigned int check_old;
        unsigned int check_j;
        unsigned int check_i_dep;
        unsigned int n_depletants_j;

        if (active)
            {
            // need to extract the j particle out of the shared mem queue
            unsigned int check_j_flag = s_queue_j[tidx_1d];
            check_old = check_j_flag & 1;
            check_j  = check_j_flag >> 1;
            check_i_dep = s_queue_didx[tidx_1d];

            // load number of depletants from Poisson distribution
            n_depletants_j = d_n_depletants[check_j*2*ntrial+(1-check_old)*ntrial+i_trial];

            if (check_i_dep == 0)
                n_depletants += n_depletants_j;
            }

        __syncthreads();

        while (s_adding_depletants)
            {
            // every active thread adds more depletants to the intermediate queue
            while (active && s_depletant_queue_size < max_depletant_queue_size && check_i_dep < n_depletants_j)
                {
                // one RNG per particle, depletant and trial insertion
                unsigned int seed_j = __scalar_as_int(check_old ? d_vel[check_j].x : d_trial_vel[check_j].x);
                unsigned int tag_j = d_tag[check_j];
                hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants, seed_j,
                    check_i_dep, i_trial, depletant_idx(depletant_type_a,depletant_type_b));

                // filter depletants overlapping with particle j and i (in both configurations)
                Scalar4 postype_j(check_old ? d_postype[check_j] : d_trial_postype[check_j]);
                unsigned int type_j = __scalar_as_int(postype_j.w);

                detail::OBB obb_j;
                    {
                    // get shape OBB
                    Shape shape_j(quat<Scalar>(check_old ?
                        d_orientation[check_j] : d_trial_orientation[check_j]), s_params[type_j]);
                    obb_j = shape_j.getOBB(vec3<Scalar>(postype_j));

                    // extend by depletant radius
                    Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
                    Shape shape_test_b(quat<Scalar>(), s_params[depletant_type_b]);

                    OverlapReal r = 0.5*detail::max(shape_test_a.getCircumsphereDiameter(),
                        shape_test_b.getCircumsphereDiameter());
                    obb_j.lengths.x += r;
                    obb_j.lengths.y += r;
                    obb_j.lengths.z += r;
                    }

                // regenerate depletant
                vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_j, dim));

                // check against j
                Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
                Shape shape_test_b(quat<Scalar>(), s_params[depletant_type_b]);
                quat<Scalar> o;
                if (shape_test_a.hasOrientation() || shape_test_b.hasOrientation())
                    {
                    o = generateRandomOrientation(rng, dim);
                    }
                if (shape_test_a.hasOrientation())
                    shape_test_a.orientation = o;
                if (shape_test_b.hasOrientation())
                    shape_test_b.orientation = o;

                Shape shape_j(quat<Scalar>(), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(check_old ?
                        d_orientation[check_j] : d_trial_orientation[check_j]);

                vec3<Scalar> pos_j(postype_j);
                vec3<Scalar> r_jk = pos_test - pos_j;
                overlap_checks ++;
                bool overlap_j_a = (s_check_overlaps[overlap_idx(type_j, depletant_type_a)]
                    && check_circumsphere_overlap(r_jk, shape_j, shape_test_a)
                    && test_overlap(r_jk, shape_j, shape_test_a, err_count));

                bool overlap_j_b = overlap_j_a;
                if (pairwise)
                    {
                    overlap_checks++;
                    overlap_j_b = (s_check_overlaps[overlap_idx(type_j, depletant_type_b)]
                        && check_circumsphere_overlap(r_jk, shape_j, shape_test_b)
                        && test_overlap(r_jk, shape_j, shape_test_b, err_count));
                    }

                // check against i in this and other config
                vec3<Scalar> pos_i(new_config ? s_pos_i_new : s_pos_i_old);
                Shape shape_i(quat<Scalar>(), s_params[s_type_i]);

                if (shape_i.hasOrientation())
                    shape_i.orientation = new_config ?
                        quat<Scalar>(s_orientation_i_new) : quat<Scalar>(s_orientation_i_old);

                vec3<Scalar> pos_i_other(!new_config ? s_pos_i_new : s_pos_i_old);
                Shape shape_i_other(quat<Scalar>(), s_params[s_type_i]);

                if (shape_i_other.hasOrientation())
                    shape_i_other.orientation = !new_config ?
                        quat<Scalar>(s_orientation_i_new) : quat<Scalar>(s_orientation_i_old);

                vec3<Scalar> r_i_test = pos_test - pos_i;
                r_i_test = vec3<Scalar>(box.minImage(vec_to_scalar3(r_i_test)));

                overlap_checks ++;
                bool overlap_i_a = (s_check_overlaps[overlap_idx(s_type_i, depletant_type_a)]
                    && check_circumsphere_overlap(r_i_test, shape_i, shape_test_a)
                    && test_overlap(r_i_test, shape_i, shape_test_a, err_count));

                bool overlap_i_b = overlap_i_a;
                if (pairwise)
                    {
                    overlap_checks++;
                    overlap_i_b = (s_check_overlaps[overlap_idx(s_type_i, depletant_type_b)]
                        && check_circumsphere_overlap(r_i_test, shape_i, shape_test_b)
                        && test_overlap(r_i_test, shape_i, shape_test_b, err_count));
                    }

                vec3<Scalar> r_i_test_other = pos_test - pos_i_other;
                r_i_test_other = vec3<Scalar>(box.minImage(vec_to_scalar3(r_i_test_other)));
                overlap_checks++;
                bool overlap_i_a_other = (s_check_overlaps[overlap_idx(s_type_i, depletant_type_a)]
                    && check_circumsphere_overlap(r_i_test_other, shape_i_other, shape_test_a)
                    && test_overlap(r_i_test_other, shape_i_other, shape_test_a, err_count));

                bool overlap_i_b_other = overlap_i_a_other;
                if (pairwise)
                    {
                    overlap_checks++;
                    overlap_i_b_other = (s_check_overlaps[overlap_idx(s_type_i, depletant_type_b)]
                        && check_circumsphere_overlap(r_i_test_other, shape_i_other, shape_test_b)
                        && test_overlap(r_i_test_other, shape_i_other, shape_test_b, err_count));
                    }

                bool overlap_ij = ((overlap_j_a && overlap_i_b) ||
                    (overlap_j_b && overlap_i_a)) && tag_i > tag_j;
                bool overlap_ij_other = ((overlap_j_a && overlap_i_b_other) ||
                    (overlap_j_b && overlap_i_a_other)) && tag_i > tag_j;

                bool insert_into_queue = overlap_ij && !overlap_ij_other;

                if (insert_into_queue)
                    {
                    // add this particle to the queue
                    unsigned int insert_point = atomicAdd(&s_depletant_queue_size, 1);

                    if (insert_point < max_depletant_queue_size)
                        {
                        s_queue_tid[insert_point] = tidx_1d;
                        s_queue_offset[insert_point] = check_i_dep;
                        }
                    else
                        {
                        // we will recheck and insert this on the next time through
                        break;
                        }
                    } // end if add_to_queue

                // advance depletant idx
                check_i_dep += n_groups*blocks_per_depletant;
                } // end while (active && s_depletant_queue_size < max_depletant_queue_size && (check_i_dep) < n_depletants_j)

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

            // is this group processing work from the intermediate queue?
            bool checking_overlaps = group < min(s_depletant_queue_size, max_depletant_queue_size);

            unsigned int tag_j;

            if (checking_overlaps)
                {
                // regenerate depletant using seed from queue
                unsigned int check_tidx = s_queue_tid[group];

                // extract neighbor info from first queue
                unsigned int check_j_flag = s_queue_j[check_tidx];
                unsigned int i_dep_queue = s_queue_offset[group];

                unsigned int check_old = check_j_flag & 1;
                unsigned check_j  = check_j_flag >> 1;

                // store in shared mem
                if (master)
                    s_j_group[group] = check_j_flag;

                // load tag for this group
                tag_j = d_tag[check_j];

                unsigned int seed_j = __scalar_as_int(check_old ? d_vel[check_j].x : d_trial_vel[check_j].x);
                hoomd::RandomGenerator rng(hoomd::RNGIdentifier::HPMCDepletants, seed_j,
                    i_dep_queue, i_trial, depletant_idx(depletant_type_a, depletant_type_b));

                detail::OBB obb_j;
                Scalar4 postype_j(check_old ? d_postype[check_j] : d_trial_postype[check_j]);
                unsigned int type_j = __scalar_as_int(postype_j.w);

                    {
                    // get shape OBB
                    Shape shape_j(quat<Scalar>(check_old ?
                        d_orientation[check_j] : d_trial_orientation[check_j]), s_params[type_j]);
                    obb_j = shape_j.getOBB(vec3<Scalar>(postype_j));

                    // extend by depletant radius
                    Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
                    Shape shape_test_b(quat<Scalar>(), s_params[depletant_type_b]);

                    OverlapReal r = 0.5*detail::max(shape_test_a.getCircumsphereDiameter(),
                        shape_test_b.getCircumsphereDiameter());
                    obb_j.lengths.x += r;
                    obb_j.lengths.y += r;
                    obb_j.lengths.z += r;
                    }

                // depletant position and orientation
                vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_j, dim));
                Shape shape_test_a(quat<Scalar>(), s_params[depletant_type_a]);
                Shape shape_test_b(quat<Scalar>(), s_params[depletant_type_b]);
                quat<Scalar> o;
                if (shape_test_a.hasOrientation() || shape_test_b.hasOrientation())
                    {
                    o = generateRandomOrientation(rng,dim);
                    }

                // store them per group
                if (master)
                    {
                    s_pos_group[group] = vec_to_scalar3(pos_test);
                    s_orientation_group[group] = quat_to_scalar4(o);
                    }
                }

            __syncthreads();

            // inner queue, every group is an overlapping depletant and neighbor pair
            unsigned int k = offset;

            if (checking_overlaps)
                {
                if (master)
                    overlap_checks += excell_size;
                }

            if (master)
                s_overlap_group[group] = 0;

            __syncthreads();

            // loop while still searching
            while (s_still_searching)
                {
                // fill the neighbor queue
                // loop through particles in the excell list and add them to the queue if they pass the circumsphere check

                // active threads add to the queue
                if (checking_overlaps)
                    {
                    // prefetch j
                    unsigned int j, next_j = 0;
                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    // add to the queue as long as the queue is not full, and we have not yet reached the end of our own list
                    // and as long as no overlaps have been found
                    while (s_queue_size < max_queue_size && k < excell_size && !s_overlap_group[group])
                        {
                        // build some shapes, but we only need them to get diameters, so don't load orientations

                        // prefetch next j
                        k += group_size;
                        j = next_j;

                        if (k < excell_size)
                            next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                        unsigned int tag_k = d_tag[j];

                        // has k been updated?
                        bool k_has_been_updated = j < N_local &&
                            d_update_order_by_ptl[j] < update_order_i &&
                            !d_reject_in[j] &&
                            d_trial_move_type[j];

                        // true if particle j is in the old configuration
                        bool old = !k_has_been_updated;

                        // read in position of neighboring particle, do not need it's orientation for circumsphere check
                        // for ghosts always load particle data
                        Scalar4 postype_j = (old || j >= N_local) ? d_postype[j] : d_trial_postype[j];
                        unsigned int type_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(), s_params[type_j]);

                        // load test particle configuration from shared mem
                        vec3<Scalar> pos_test(s_pos_group[group]);
                        Shape shape_test_a(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type_a]);
                        Shape shape_test_b(quat<Scalar>(s_orientation_group[group]), s_params[depletant_type_b]);

                        // put particle j into the coordinate system of particle i
                        vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                        r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                        bool insert_in_queue = i != j && (old || j < N_local) && tag_j < tag_k;

                        bool circumsphere_overlap = s_check_overlaps[overlap_idx(depletant_type_a, type_j)] &&
                            check_circumsphere_overlap(r_jk, shape_test_a, shape_j);

                        circumsphere_overlap |= pairwise &&
                            s_check_overlaps[overlap_idx(depletant_type_b, type_j)] &&
                            check_circumsphere_overlap(r_jk, shape_test_b, shape_j);

                        insert_in_queue &= circumsphere_overlap;

                        if (insert_in_queue)
                            {
                            // add this particle to the queue
                            unsigned int insert_point = atomicAdd(&s_queue_size, 1);

                            if (insert_point < max_queue_size)
                                {
                                s_queue_gid[insert_point] = group;
                                s_queue_k[insert_point] = (j << 1) | (old ? 1 : 0);
                                }
                            else
                                {
                                // or back up if the queue is already full
                                // we will recheck and insert this on the next time through
                                k -= group_size;
                                }
                            } // end if k < excell_size
                        } // end while (s_queue_size < max_queue_size && k < excell_size)
                    } // end if checking_overlaps

                // sync to make sure all threads in the block are caught up
                __syncthreads();

                if (master && group == 0)
                    s_still_searching = 0;

                // all threads processing overlaps
                if (tidx_1d < min(s_queue_size, max_queue_size))
                    {
                    // need to extract the overlap check to perform out of the shared mem queue
                    unsigned int check_group = s_queue_gid[tidx_1d];
                    unsigned int check_k_flag = s_queue_k[tidx_1d];
                    bool check_k_old = check_k_flag & 1;
                    unsigned int check_k  = check_k_flag >> 1;

                    // build depletant shape from shared memory
                    Scalar3 pos_test = s_pos_group[check_group];
                    Shape shape_test_a(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type_a]);
                    Shape shape_test_b(quat<Scalar>(s_orientation_group[check_group]), s_params[depletant_type_b]);

                    // build shape k from global memory
                    Scalar4 postype_k = check_k_old ? d_postype[check_k] : d_trial_postype[check_k];
                    unsigned int type_k = __scalar_as_int(postype_k.w);
                    Shape shape_k(quat<Scalar>(), s_params[type_k]);
                    if (shape_k.hasOrientation())
                        shape_k.orientation = quat<Scalar>(check_k_old ? d_orientation[check_k] : d_trial_orientation[check_k]);
                    // put particle k into the coordinate system of test particle
                    vec3<Scalar> r_k_test = vec3<Scalar>(pos_test) - vec3<Scalar>(postype_k);
                    r_k_test = vec3<Scalar>(box.minImage(vec_to_scalar3(r_k_test)));

                    bool overlap_k_a = s_check_overlaps[overlap_idx(depletant_type_a, type_k)] &&
                        test_overlap(r_k_test, shape_k, shape_test_a, err_count);

                    bool overlap_k_b = overlap_k_a;
                    bool overlap_j_a = true;
                    bool overlap_j_b = true;

                    if (pairwise)
                        {
                        overlap_k_b = s_check_overlaps[overlap_idx(depletant_type_b, type_k)] &&
                            test_overlap(r_k_test, shape_k, shape_test_b, err_count);

                        unsigned int check_j_flag = s_j_group[check_group];
                        bool check_old_j = check_j_flag & 1;
                        unsigned int check_j = check_old_j >> 1;

                        Scalar4 postype_j = check_old_j ? d_postype[check_j] : d_trial_postype[check_j];
                        vec3<Scalar> pos_j(postype_j);
                        unsigned int type_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(), s_params[type_j]);
                        if (shape_j.hasOrientation())
                            shape_j.orientation = quat<Scalar>(check_old_j ?
                                d_orientation[j] : d_trial_orientation[j]);

                        if (overlap_k_b)
                            {
                            // check depletant a against j
                            vec3<Scalar> r_j_test = vec3<Scalar>(pos_test) - pos_j;
                            overlap_j_a = (s_check_overlaps[overlap_idx(type_j, depletant_type_a)]
                                && check_circumsphere_overlap(r_j_test, shape_j, shape_test_a)
                                && test_overlap(r_j_test, shape_j, shape_test_a, err_count));
                            }

                        if (overlap_k_a)
                            {
                            // check depletant b against j
                            vec3<Scalar> r_j_test = vec3<Scalar>(pos_test) - pos_j;
                            overlap_j_b = (s_check_overlaps[overlap_idx(type_j, depletant_type_b)]
                                && check_circumsphere_overlap(r_j_test, shape_j, shape_test_b)
                                && test_overlap(r_j_test, shape_j, shape_test_b, err_count));
                            }
                        }

                    if ((overlap_j_a && overlap_k_b) || (overlap_j_b && overlap_k_a))
                        {
                        // store result
                        atomicAdd(&s_overlap_group[check_group],1);
                        }
                    } // end if (processing neighbor)

                // threads that need to do more looking set the still_searching flag
                __syncthreads();
                if (master && group == 0)
                    s_queue_size = 0;
                if (!s_overlap_group[group] && checking_overlaps && k < excell_size)
                    atomicAdd(&s_still_searching, 1);
                __syncthreads();

                } // end while (s_still_searching)

            // overlap checks have been processed for this group, accumulate free energy
            if (checking_overlaps && master && !s_overlap_group[group])
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
            if (active && check_i_dep < n_depletants_j)
                atomicAdd(&s_adding_depletants, 1);
            __syncthreads();
            } // end loop over depletants

        // threads that need to do more looking set the still_searching flag
        __syncthreads();
        if (master && group == 0)
            s_queue_size = 0;
        if (k < excell_size)
            atomicAdd(&s_still_searching, 1);
        __syncthreads();

        } // end while (s_still_searching)

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
            atomicAdd_system(&d_implicit_counters[depletant_idx(depletant_type_a,
                depletant_type_b)].insert_count, n_depletants);
            #else
            atomicAdd(&d_implicit_counters[depletant_idx(depletant_type_a,
                depletant_type_b)].insert_count, n_depletants);
            #endif
            }
        }
    }

//! Launcher for hpmc_insert_depletants_phase2 kernel with templated launch bounds
template< class Shape, bool pairwise, unsigned int cur_launch_bounds>
void depletants_launcher_phase2(const hpmc_args_t& args,
    const hpmc_implicit_args_t& implicit_args,
    const hpmc_auxilliary_args_t& auxilliary_args,
    const typename Shape::param_type *params,
    unsigned int max_threads,
    detail::int2type<cur_launch_bounds>)
    {
    if (max_threads == cur_launch_bounds*MIN_BLOCK_SIZE)
        {
        // determine the maximum block size and clamp the input block size down
        static int max_block_size = -1;
        static hipFuncAttributes attr;
        constexpr unsigned int launch_bounds_nonzero = cur_launch_bounds > 0 ? cur_launch_bounds : 1;
        if (max_block_size == -1)
            {
            hipFuncGetAttributes(&attr,
                reinterpret_cast<const void*>(&kernel::hpmc_insert_depletants_phase2<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE, pairwise>));
            max_block_size = attr.maxThreadsPerBlock;
            }

        // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp,block_size);
        unsigned int n_groups = block_size / tpp;

        // clamp blockDim.z
        n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);

        unsigned int max_queue_size = n_groups*tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes = args.num_types * sizeof(typename Shape::param_type) +
                   args.overlap_idx.getNumElements() * sizeof(unsigned int);

        unsigned int shared_bytes = n_groups *(sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(unsigned int)) +
                                    max_queue_size*4*sizeof(unsigned int) +
                                    max_depletant_queue_size*2*sizeof(unsigned int) +
                                    min_shared_bytes;

        if (min_shared_bytes >= args.devprop.sharedMemPerBlock)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

        while (shared_bytes + attr.sharedSizeBytes >= args.devprop.sharedMemPerBlock)
            {
            block_size -= args.devprop.warpSize;
            if (block_size == 0)
                throw std::runtime_error("Insufficient shared memory for HPMC kernel");
            tpp = min(tpp, block_size);
            n_groups = block_size / tpp;

            // clamp blockDim.z
            n_groups = std::min((unsigned int) args.devprop.maxThreadsDim[2], n_groups);

            max_queue_size = n_groups*tpp;
            max_depletant_queue_size = n_groups;

            shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + 2*sizeof(unsigned int)) +
                           max_queue_size*4*sizeof(unsigned int) +
                           max_depletant_queue_size*2*sizeof(unsigned int) +
                           min_shared_bytes;
            }


        // determine dynamically requested shared memory
        unsigned int base_shared_bytes = shared_bytes + attr.sharedSizeBytes;
        unsigned int max_extra_bytes = args.devprop.sharedMemPerBlock - base_shared_bytes;
        char *ptr = (char *) nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        unsigned int extra_bytes = max_extra_bytes - available_bytes;
        shared_bytes += extra_bytes;

        // setup the grid to run the kernel
        dim3 threads(1, tpp, n_groups);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            if (range.first == range.second)
                continue;

            unsigned int blocks_per_particle = (implicit_args.max_n_depletants[idev]) /
                (implicit_args.depletants_per_group*n_groups) + 1;
            dim3 grid( range.second-range.first, blocks_per_particle, 2*auxilliary_args.ntrial);

            if (blocks_per_particle > args.devprop.maxGridSize[1])
                {
                grid.y = args.devprop.maxGridSize[1];
                grid.z *= blocks_per_particle/args.devprop.maxGridSize[1]+1;
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

            hipLaunchKernelGGL((kernel::hpmc_insert_depletants_phase2<Shape, launch_bounds_nonzero*MIN_BLOCK_SIZE, pairwise>),
                dim3(grid), dim3(threads), shared_bytes, auxilliary_args.streams_phase2[idev],
                                 args.d_trial_postype,
                                 args.d_trial_orientation,
                                 args.d_trial_move_type,
                                 args.d_postype,
                                 args.d_orientation,
                                 args.d_counters + idev*args.counters_pitch,
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
                                 implicit_args.depletant_type_b,
                                 implicit_args.depletant_idx,
                                 implicit_args.d_implicit_count + idev*implicit_args.implicit_counters_pitch,
                                 args.d_update_order_by_ptl,
                                 args.d_reject_in,
                                 auxilliary_args.ntrial,
                                 auxilliary_args.d_tag,
                                 auxilliary_args.d_vel,
                                 auxilliary_args.d_trial_vel,
                                 auxilliary_args.d_deltaF_int,
                                 implicit_args.repulsive,
                                 range.first,
                                 auxilliary_args.d_n_depletants_ntrial);
            }
        }
    else
        {
        depletants_launcher_phase2<Shape, pairwise>(args,
            implicit_args,
            auxilliary_args,
            params,
            max_threads,
            detail::int2type<cur_launch_bounds/2>());
        }
    }

} // end namespace kernel

//! Kernel driver for kernel::insert_depletants_phase2()
/*! \param args Bundled arguments
    \param implicit_args Bundled arguments related to depletants
    \param auxilliary_args Arguments for auxilliary variable depletants
    \param d_params Per-type shape parameters

    \ingroup hpmc_kernels
*/
template< class Shape >
void hpmc_depletants_auxilliary_phase2(const hpmc_args_t& args,
    const hpmc_implicit_args_t& implicit_args,
    const hpmc_auxilliary_args_t& auxilliary_args,
    const typename Shape::param_type *params)
    {
    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    if (implicit_args.depletant_type_a == implicit_args.depletant_type_b)
        {
        kernel::depletants_launcher_phase2<Shape, false>(args,
            implicit_args,
            auxilliary_args,
            params,
            launch_bounds,
            detail::int2type<MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
        }
    else
        {
        kernel::depletants_launcher_phase2<Shape, true>(args,
            implicit_args,
            auxilliary_args,
            params,
            launch_bounds,
            detail::int2type<MAX_BLOCK_SIZE/MIN_BLOCK_SIZE>());
        }
    }
#endif

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE

} // end namespace gpu

} // end namespace hpmc
