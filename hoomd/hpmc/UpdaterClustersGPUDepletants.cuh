// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file UpdaterClustersGPUDepletants.cuh
    \brief Implements the depletant kernels for the geometric cluster algorithm the GPU
*/

#pragma once

#include <hip/hip_runtime.h>

#include "HPMCMiscFunctions.h"
#include "hoomd/BoxDim.h"
#include "hoomd/CachedAllocator.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/GPUHelpers.cuh"

#include "IntegratorHPMCMonoGPUDepletants.cuh"
#include "UpdaterClustersGPU.cuh"

#ifdef __HIP_PLATFORM_NVCC__
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 256 // a reasonable minimum to limit the number of template instantiations
#else
#define MAX_BLOCK_SIZE 1024
#define MIN_BLOCK_SIZE 1024 // on AMD, we do not use __launch_bounds__
#endif

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Kernel driver for kernel::hpmc_clusters_depletants()
template<class Shape>
void hpmc_clusters_depletants(const cluster_args_t& args,
                              const hpmc_implicit_args_t& depletants_args,
                              const typename Shape::param_type* params);

#ifdef __HIPCC__
namespace kernel
    {
//! Kernel to insert depletants on-the-fly
template<class Shape, unsigned int max_threads>
#ifdef __HIP_PLATFORM_NVCC__
__launch_bounds__(max_threads)
#endif
    __global__ void clusters_insert_depletants(const Scalar4* d_postype,
                                               const Scalar4* d_orientation,
                                               bool line,
                                               vec3<Scalar> pivot,
                                               quat<Scalar> q,
                                               const unsigned int* d_excell_idx,
                                               const unsigned int* d_excell_size,
                                               const Index2D excli,
                                               const uint3 cell_dim,
                                               const Scalar3 ghost_width,
                                               const Index3D ci,
                                               const unsigned int num_types,
                                               const unsigned int seed,
                                               const unsigned int* d_check_overlaps,
                                               const Index2D overlap_idx,
                                               const uint64_t timestep,
                                               const unsigned int dim,
                                               const BoxDim box,
                                               const typename Shape::param_type* d_params,
                                               unsigned int max_queue_size,
                                               unsigned int max_extra_bytes,
                                               unsigned int depletant_type,
                                               unsigned int* d_nneigh,
                                               unsigned int* d_adjacency,
                                               const unsigned int maxn,
                                               unsigned int* d_overflow,
                                               unsigned int work_offset,
                                               unsigned int max_depletant_queue_size,
                                               const unsigned int* d_n_depletants)
    {
    // variables to tell what type of thread we are
    unsigned int group = threadIdx.y;
    unsigned int offset = threadIdx.z;
    unsigned int group_size = blockDim.z;
    bool master = (offset == 0);
    unsigned int n_groups = blockDim.y;

    unsigned int err_count = 0;

    // shared particle configuation
    __shared__ Scalar4 s_orientation_i;
    __shared__ Scalar3 s_pos_i;
    __shared__ unsigned int s_type_i;
    __shared__ int3 s_img_i;

    // shared queue variables
    __shared__ unsigned int s_queue_size;
    __shared__ unsigned int s_still_searching;
    __shared__ unsigned int s_adding_depletants;
    __shared__ unsigned int s_depletant_queue_size;

    // load the per type pair parameters into shared memory
    HIP_DYNAMIC_SHARED(char, s_data)
    typename Shape::param_type* s_params = (typename Shape::param_type*)(&s_data[0]);
    Scalar4* s_orientation_group = (Scalar4*)(s_params + num_types);
    Scalar3* s_pos_group = (Scalar3*)(s_orientation_group + n_groups);
    unsigned int* s_reject_group = (unsigned int*)(s_pos_group + n_groups);
    unsigned int* s_check_overlaps = (unsigned int*)(s_reject_group + n_groups);
    unsigned int* s_queue_j = (unsigned int*)(s_check_overlaps + overlap_idx.getNumElements());
    unsigned int* s_queue_gid = (unsigned int*)(s_queue_j + max_queue_size);
    unsigned int* s_queue_didx = (unsigned int*)(s_queue_gid + max_queue_size);

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
    char* s_extra = (char*)(s_queue_didx + max_depletant_queue_size);

    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int cur_type = 0; cur_type < num_types; ++cur_type)
        s_params[cur_type].load_shared(s_extra, available_bytes);

    __syncthreads();

    // identify the active cell that this thread handles
    unsigned int i = blockIdx.x + work_offset;

    // load updated particle position
    if (master && group == 0)
        {
        Scalar4 postype_i = d_postype[i];
        s_pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        s_type_i = __scalar_as_int(postype_i.w);
        s_orientation_i = d_orientation[i];

        // get image of particle i after transformation
        vec3<Scalar> pos_i_transf(s_pos_i);
        if (line)
            {
            pos_i_transf = lineReflection(pos_i_transf, pivot, q);
            }
        else
            {
            pos_i_transf = pivot - (pos_i_transf - pivot);
            }
        s_img_i = box.getImage(pos_i_transf);
        }

    // sync so that s_pos_i etc. are available
    __syncthreads();

    // generate random number of depletants from Poisson distribution
    unsigned int n_depletants = d_n_depletants[i];

    unsigned int overlap_checks = 0;
    unsigned int n_inserted = 0;

    // find the cell this particle should be in
    unsigned int my_cell = computeParticleCell(s_pos_i, box, ghost_width, cell_dim, ci, false);

    detail::OBB obb_i;
        {
        // get shape OBB
        Shape shape_i(quat<Scalar>(d_orientation[i]), s_params[s_type_i]);
        obb_i = shape_i.getOBB(vec3<Scalar>(s_pos_i));

        // extend by depletant radius
        Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

        Scalar r = 0.5
                   * detail::max(shape_test.getCircumsphereDiameter(),
                                 shape_test.getCircumsphereDiameter());
        obb_i.lengths.x += r;
        obb_i.lengths.y += r;

        if (dim == 3)
            obb_i.lengths.z += r;
        else
            obb_i.lengths.z = ShortReal(0.5);
        }

    if (master && group == 0)
        {
        s_depletant_queue_size = 0;
        s_adding_depletants = 1;
        }

    __syncthreads();

    unsigned int gidx = gridDim.y * blockIdx.z + blockIdx.y;
    unsigned int blocks_per_particle = gridDim.y * gridDim.z;
    unsigned int i_dep = group_size * group + offset + gidx * group_size * n_groups;

    while (s_adding_depletants)
        {
        while (s_depletant_queue_size < max_depletant_queue_size && i_dep < n_depletants)
            {
            // one RNG per depletant
            hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantsClusters, timestep, seed),
                hoomd::Counter(i, i_dep, depletant_type));

            n_inserted++;
            overlap_checks += 2;

            // test depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));

            Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng, dim);
                }

            Shape shape_i(quat<Scalar>(), s_params[s_type_i]);
            if (shape_i.hasOrientation())
                shape_i.orientation = quat<Scalar>(s_orientation_i);
            vec3<Scalar> r_ij = vec3<Scalar>(s_pos_i) - pos_test;
            bool overlap = (s_check_overlaps[overlap_idx(s_type_i, depletant_type)]
                            && check_circumsphere_overlap(r_ij, shape_test, shape_i)
                            && test_overlap(r_ij, shape_test, shape_i, err_count));

            if (overlap)
                {
                // add this particle to the queue
                unsigned int insert_point = atomicAdd(&s_depletant_queue_size, 1);

                if (insert_point < max_depletant_queue_size)
                    {
                    s_queue_didx[insert_point] = i_dep;
                    }
                else
                    {
                    // we will recheck and insert this on the next time through
                    break;
                    }
                } // end if add_to_queue

            // advance depletant idx
            i_dep += group_size * n_groups * blocks_per_particle;
            } // end while (s_depletant_queue_size < max_depletant_queue_size && i_dep <
              // n_depletants)

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
            hoomd::RandomGenerator rng(
                hoomd::Seed(hoomd::RNGIdentifier::HPMCDepletantsClusters, timestep, seed),
                hoomd::Counter(i, i_dep_queue, depletant_type));

            // depletant position and orientation
            vec3<Scalar> pos_test = vec3<Scalar>(generatePositionInOBB(rng, obb_i, dim));
            Shape shape_test(quat<Scalar>(), s_params[depletant_type]);
            if (shape_test.hasOrientation())
                {
                shape_test.orientation = generateRandomOrientation(rng, dim);
                }

            // store them per group
            if (master)
                {
                s_pos_group[group] = vec_to_scalar3(pos_test);
                s_orientation_group[group] = quat_to_scalar4(shape_test.orientation);
                }
            }

        if (master)
            {
            s_reject_group[group] = 0;
            }

        __syncthreads();

        // counters to track progress through the loop over potential neighbors
        unsigned int excell_size;
        unsigned int k = offset;

        // first pass, see if this depletant is active
        if (active)
            {
            excell_size = d_excell_size[my_cell];
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
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load
                    // orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for
                    // circumsphere check for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]),
                                     s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap
                        = s_check_overlaps[overlap_idx(depletant_type, type_j)]
                          && check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    // upper triangular matrix
                    if (i < j && circumsphere_overlap)
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
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                Scalar3 pos_test = s_pos_group[check_group];
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]),
                                 s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)]
                    && test_overlap(r_jk, shape_test, shape_j, err_count))
                    {
                    s_reject_group[check_group] = 1;
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();

            } // end while (s_still_searching)

        __syncthreads();
        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // second pass, see if transformed depletant overlaps
        vec3<Scalar> pos_test_transf;
        quat<Scalar> orientation_test_transf;
        unsigned int other_cell;

        active &= !s_reject_group[group];

        if (active)
            {
            pos_test_transf = vec3<Scalar>(s_pos_group[group]);
            if (line)
                {
                pos_test_transf = lineReflection(pos_test_transf, pivot, q);
                }
            else
                {
                pos_test_transf = pivot - (pos_test_transf - pivot);
                }

            // wrap back into into i's image (after transformation)
            pos_test_transf = box.shift(pos_test_transf, -s_img_i);
            int3 img = make_int3(0, 0, 0);
            box.wrap(pos_test_transf, img);

            other_cell = computeParticleCell(vec_to_scalar3(pos_test_transf),
                                             box,
                                             ghost_width,
                                             cell_dim,
                                             ci,
                                             false);
            excell_size = d_excell_size[other_cell];
            }

        k = offset;

        // loop while still searching
        while (s_still_searching)
            {
            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, other_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached
                // the end of our own list and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load
                    // orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, other_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for
                    // circumsphere check for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    Shape shape_test(quat<Scalar>(), s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test_transf);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap
                        = s_check_overlaps[overlap_idx(depletant_type, type_j)]
                          && check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (circumsphere_overlap)
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
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size * group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                vec3<Scalar> pos_test_transf(s_pos_group[check_group]);

                quat<Scalar> orientation_test_transf(
                    q * quat<Scalar>(s_orientation_group[check_group]));
                if (line)
                    {
                    pos_test_transf = lineReflection(pos_test_transf, pivot, q);
                    }
                else
                    {
                    pos_test_transf = pivot - (pos_test_transf - pivot);
                    }

                // wrap back into into i's image (after transformation)
                pos_test_transf = box.shift(pos_test_transf, -s_img_i);
                int3 img = make_int3(0, 0, 0);
                box.wrap(pos_test_transf, img);

                Shape shape_test_transf(quat<Scalar>(orientation_test_transf),
                                        s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test_transf);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)]
                    && test_overlap(r_jk, shape_test_transf, shape_j, err_count))
                    {
                    s_reject_group[check_group] = 1;
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();
            }

        __syncthreads();
        if (master && group == 0)
            {
            // reset the queue for neighbor checks
            s_queue_size = 0;
            s_still_searching = 1;
            }

        __syncthreads();

        // third pass, record overlaps
        active &= !s_reject_group[group];

        if (active)
            {
            excell_size = d_excell_size[my_cell];
            }

        k = offset;

        // loop while still searching
        while (s_still_searching)
            {
            // active threads add to the queue
            if (active)
                {
                // prefetch j
                unsigned int j, next_j = 0;
                if (k < excell_size)
                    next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                // add to the queue as long as the queue is not full, and we have not yet reached
                // the end of our own list and as long as no overlaps have been found
                while (s_queue_size < max_queue_size && k < excell_size)
                    {
                    Scalar4 postype_j;
                    Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                    vec3<Scalar> r_jk;

                    // build some shapes, but we only need them to get diameters, so don't load
                    // orientations

                    // prefetch next j
                    k += group_size;
                    j = next_j;

                    if (k < excell_size)
                        next_j = __ldg(&d_excell_idx[excli(k, my_cell)]);

                    // read in position of neighboring particle, do not need it's orientation for
                    // circumsphere check for ghosts always load particle data
                    postype_j = d_postype[j];
                    unsigned int type_j = __scalar_as_int(postype_j.w);
                    Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);

                    // load test particle configuration from shared mem
                    vec3<Scalar> pos_test(s_pos_group[group]);
                    Shape shape_test(quat<Scalar>(s_orientation_group[group]),
                                     s_params[depletant_type]);

                    // put particle j into the coordinate system of particle i
                    r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                    r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                    bool circumsphere_overlap
                        = s_check_overlaps[overlap_idx(depletant_type, type_j)]
                          && check_circumsphere_overlap(r_jk, shape_test, shape_j);

                    if (circumsphere_overlap)
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
                        } // end if k < excell_size
                    } // end while (s_queue_size < max_queue_size && k < excell_size)
                } // end if active

            // sync to make sure all threads in the block are caught up
            __syncthreads();

            if (master && group == 0)
                s_still_searching = 0;

            unsigned int tidx_1d = offset + group_size * group;

            // max_queue_size is always <= block size, so we just need an if here
            if (tidx_1d < min(s_queue_size, max_queue_size))
                {
                // need to extract the overlap check to perform out of the shared mem queue
                unsigned int check_group = s_queue_gid[tidx_1d];
                unsigned int check_j = s_queue_j[tidx_1d];

                // build depletant shape from shared memory
                vec3<Scalar> pos_test(s_pos_group[check_group]);
                Shape shape_test(quat<Scalar>(s_orientation_group[check_group]),
                                 s_params[depletant_type]);

                // build shape j from global memory
                Scalar4 postype_j = d_postype[check_j];
                Scalar4 orientation_j = make_scalar4(1, 0, 0, 0);
                unsigned int type_j = __scalar_as_int(postype_j.w);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[type_j]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(d_orientation[check_j]);

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_jk = vec3<Scalar>(postype_j) - vec3<Scalar>(pos_test);
                r_jk = vec3<Scalar>(box.minImage(vec_to_scalar3(r_jk)));

                if (s_check_overlaps[overlap_idx(depletant_type, type_j)]
                    && test_overlap(r_jk, shape_test, shape_j, err_count))
                    {
                    // write out to global memory
                    unsigned int n = atomicAdd(&d_nneigh[i], 1);
                    if (n < maxn)
                        {
                        d_adjacency[n + i * maxn] = check_j;
                        }
                    }
                } // end if (processing neighbor)

            // threads that need to do more looking set the still_searching flag
            __syncthreads();
            if (master && group == 0)
                s_queue_size = 0;
            if (active && k < excell_size)
                atomicAdd(&s_still_searching, 1);
            __syncthreads();
            }

        // do we still need to process depletants?
        __syncthreads();
        if (master && group == 0)
            s_depletant_queue_size = 0;
        if (i_dep < n_depletants)
            atomicAdd(&s_adding_depletants, 1);
        __syncthreads();
        } // end loop over depletants

    if (master && group == 0)
        {
        // overflowed?
        unsigned int nneigh = d_nneigh[i];
        if (nneigh > maxn)
            {
#if (__CUDA_ARCH__ >= 600)
            atomicMax_system(d_overflow, nneigh);
#else
            atomicMax(d_overflow, nneigh);
#endif
            }
        }
    }

//! Launcher for clusters_insert_depletants kernel with templated launch bounds
template<class Shape, unsigned int cur_launch_bounds>
void clusters_depletants_launcher(const cluster_args_t& args,
                                  const hpmc_implicit_args_t& implicit_args,
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
                &kernel::clusters_insert_depletants<Shape,
                                                    launch_bounds_nonzero * MIN_BLOCK_SIZE>));
        max_block_size = attr.maxThreadsPerBlock;
        if (max_block_size % args.devprop.warpSize)
            // handle non-sensical return values from hipFuncGetAttributes
            max_block_size = (max_block_size / args.devprop.warpSize - 1) * args.devprop.warpSize;

        // choose a block size based on the max block size by regs (max_block_size) and include
        // dynamic shared memory usage
        unsigned int block_size = min(args.block_size, (unsigned int)max_block_size);

        unsigned int tpp = min(args.tpp, block_size);
        tpp = std::min((unsigned int)args.devprop.maxThreadsDim[2], tpp); // clamp blockDim.z
        unsigned int n_groups = block_size / tpp;
        unsigned int max_queue_size = n_groups * tpp;
        unsigned int max_depletant_queue_size = n_groups;

        const unsigned int min_shared_bytes
            = static_cast<unsigned int>(args.num_types * sizeof(typename Shape::param_type)
                                        + args.overlap_idx.getNumElements() * sizeof(unsigned int));

        size_t shared_bytes = n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int))
                              + max_queue_size * 2 * sizeof(unsigned int)
                              + max_depletant_queue_size * sizeof(unsigned int) + min_shared_bytes;

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

            shared_bytes = static_cast<unsigned int>(
                n_groups * (sizeof(Scalar4) + sizeof(Scalar3) + sizeof(unsigned int))
                + max_queue_size * 2 * sizeof(unsigned int)
                + max_depletant_queue_size * sizeof(unsigned int) + min_shared_bytes);
            }

        unsigned int base_shared_bytes;
        base_shared_bytes = static_cast<unsigned int>(shared_bytes + attr.sharedSizeBytes);

        unsigned int max_extra_bytes
            = static_cast<unsigned int>(args.devprop.sharedMemPerBlock - base_shared_bytes);
        unsigned int extra_bytes;
        // determine dynamically requested shared memory
        char* ptr = (char*)nullptr;
        unsigned int available_bytes = max_extra_bytes;
        for (unsigned int i = 0; i < args.num_types; ++i)
            {
            params[i].allocate_shared(ptr, available_bytes);
            }
        extra_bytes = max_extra_bytes - available_bytes;

        shared_bytes += extra_bytes;

        // setup the grid to run the kernel
        dim3 threads(1, n_groups, tpp);

        for (int idev = args.gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
            {
            auto range = args.gpu_partition.getRangeAndSetGPU(idev);

            if (range.first == range.second)
                continue;

            unsigned int blocks_per_particle
                = implicit_args.max_n_depletants[idev]
                      / (implicit_args.depletants_per_thread * n_groups * tpp)
                  + 1;

            dim3 grid(range.second - range.first, blocks_per_particle, 1);

            if (blocks_per_particle > static_cast<unsigned int>(args.devprop.maxGridSize[1]))
                {
                grid.y = args.devprop.maxGridSize[1];
                grid.z = blocks_per_particle / args.devprop.maxGridSize[1] + 1;
                }

            hipLaunchKernelGGL(
                (kernel::clusters_insert_depletants<Shape, launch_bounds_nonzero * MIN_BLOCK_SIZE>),
                dim3(grid),
                dim3(threads),
                shared_bytes,
                implicit_args.streams[idev],
                args.d_postype,
                args.d_orientation,
                args.line,
                args.pivot,
                args.q,
                args.d_excell_idx,
                args.d_excell_size,
                args.excli,
                args.cell_dim,
                args.ghost_width,
                args.ci,
                args.num_types,
                args.seed,
                args.d_check_overlaps,
                args.overlap_idx,
                args.timestep,
                args.dim,
                args.box,
                params,
                max_queue_size,
                max_extra_bytes,
                implicit_args.depletant_type_a,
                args.d_nneigh,
                args.d_adjacency,
                args.maxn,
                args.d_overflow,
                range.first,
                max_depletant_queue_size,
                implicit_args.d_n_depletants);
            }
        }
    else
        {
        clusters_depletants_launcher<Shape>(args,
                                            implicit_args,
                                            params,
                                            max_threads,
                                            detail::int2type<cur_launch_bounds / 2>());
        }
    }

    } // end namespace kernel

//! Kernel driver for kernel::hpmc_clusters_depletants()
template<class Shape>
void hpmc_clusters_depletants(const cluster_args_t& args,
                              const hpmc_implicit_args_t& depletants_args,
                              const typename Shape::param_type* params)
    {
    // select the kernel template according to the next power of two of the block size
    unsigned int launch_bounds = MIN_BLOCK_SIZE;
    while (launch_bounds < args.block_size)
        launch_bounds *= 2;

    kernel::clusters_depletants_launcher<Shape>(
        args,
        depletants_args,
        params,
        launch_bounds,
        detail::int2type<MAX_BLOCK_SIZE / MIN_BLOCK_SIZE>());
    }
#endif

    } // end namespace gpu
    } // end namespace hpmc
    } // end namespace hoomd

#undef MAX_BLOCK_SIZE
#undef MIN_BLOCK_SIZE
