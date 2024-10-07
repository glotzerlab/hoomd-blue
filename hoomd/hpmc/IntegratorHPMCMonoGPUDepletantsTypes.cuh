// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/BoxDim.h"
#include "hoomd/CachedAllocator.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/HPMCCounters.h"
#include <hip/hip_runtime.h>

// base data types
#include "IntegratorHPMCMonoGPUTypes.cuh"

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Wraps arguments to kernel::hpmc_insert_depletants
/*! \ingroup hpmc_data_structs */
struct hpmc_implicit_args_t
    {
    //! Construct a hpmc_implicit_args_t
    hpmc_implicit_args_t(const unsigned int _depletant_type_a,
                         hpmc_implicit_counters_t* _d_implicit_count,
                         const unsigned int _implicit_counters_pitch,
                         const bool _repulsive,
                         const unsigned int* _d_n_depletants,
                         const unsigned int* _max_n_depletants,
                         const unsigned int _depletants_per_thread,
                         const hipStream_t* _streams)
        : depletant_type_a(_depletant_type_a), d_implicit_count(_d_implicit_count),
          implicit_counters_pitch(_implicit_counters_pitch), repulsive(_repulsive),
          d_n_depletants(_d_n_depletants), max_n_depletants(_max_n_depletants),
          depletants_per_thread(_depletants_per_thread), streams(_streams) { };

    const unsigned int depletant_type_a;        //!< Particle type of first depletant
    hpmc_implicit_counters_t* d_implicit_count; //!< Active cell acceptance/rejection counts
    const unsigned int implicit_counters_pitch; //!< Pitch of 2D array counters per device
    const bool repulsive;                       //!< True if the fugacity is negative
    const unsigned int* d_n_depletants;         //!< Number of depletants per particle
    const unsigned int*
        max_n_depletants; //!< Maximum number of depletants inserted per particle, per device
    unsigned int depletants_per_thread; //!< Controls parallelism (number of depletant loop
                                        //!< iterations per group)
    const hipStream_t* streams;         //!< Stream for this depletant type
    };

//! Driver for kernel::hpmc_insert_depletants()
template<class Shape>
void hpmc_insert_depletants(const hpmc_args_t& args,
                            const hpmc_implicit_args_t& implicit_args,
                            const typename Shape::param_type* params);

void generate_num_depletants(const uint16_t seed,
                             const uint64_t timestep,
                             const unsigned int select,
                             const unsigned int rank,
                             const unsigned int depletant_type_a,
                             const Scalar* d_lambda,
                             const Scalar4* d_postype,
                             unsigned int* d_n_depletants,
                             const unsigned int block_size,
                             const hipStream_t* streams,
                             const GPUPartition& gpu_partition,
                             const unsigned int ntypes);

void get_max_num_depletants(unsigned int* d_n_depletants,
                            unsigned int* max_n_depletants,
                            const hipStream_t* streams,
                            const GPUPartition& gpu_partition,
                            CachedAllocator& alloc);

void reduce_counters(const unsigned int ngpu,
                     const unsigned int pitch,
                     const hpmc_counters_t* d_per_device_counters,
                     hpmc_counters_t* d_counters,
                     const unsigned int implicit_pitch,
                     const hpmc_implicit_counters_t* d_per_device_implicit_counters,
                     hpmc_implicit_counters_t* d_implicit_counters,
                     const unsigned int ntypes);

    } // end namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd
