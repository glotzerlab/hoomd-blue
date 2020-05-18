// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#include <hip/hip_runtime.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/GPUPartition.cuh"

namespace hpmc {

namespace gpu {

//! Driver for kernel::hpmc_depletants_accept
void hpmc_depletants_accept(
    const unsigned int seed,
    const unsigned int timestep,
    const unsigned int select,
    const int *d_deltaF_int,
    const Index2D depletant_idx,
    const unsigned int deltaF_pitch,
    const Scalar *d_fugacity,
    const unsigned int *d_ntrial,
    unsigned *d_reject_out,
    const GPUPartition& gpu_partition,
    const unsigned int block_size);

void generate_num_depletants_ntrial(const Scalar4 *d_vel,
                                    const Scalar4 *d_trial_vel,
                                    const unsigned int ntrial,
                                    const unsigned int depletant_type_a,
                                    const unsigned int depletant_type_b,
                                    const Index2D depletant_idx,
                                    const Scalar *d_lambda,
                                    const Scalar4 *d_postype,
                                    unsigned int *d_n_depletants,
                                    const GPUPartition& gpu_partition,
                                    const unsigned int block_size,
                                    const hipStream_t *streams);

void get_max_num_depletants_ntrial(const unsigned int ntrial,
                            unsigned int *d_n_depletants,
                            unsigned int *max_n_depletants,
                            const hipStream_t *streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc);

} // end namespace gpu

} // end namespace hpmc
