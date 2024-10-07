// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include <hip/hip_runtime.h>

#include "IntegratorHPMCMonoGPUDepletants.cuh"

namespace hoomd
    {
namespace hpmc
    {
namespace gpu
    {
//! Wraps arguments to kernel::hpmc_insert_depletants_phase(n)
/*! \ingroup hpmc_data_structs */
struct hpmc_auxilliary_args_t
    {
    //! Construct a hpmc_auxilliary_args_t
    hpmc_auxilliary_args_t(const unsigned int* _d_tag,
                           const Scalar4* _d_vel,
                           const Scalar4* _d_trial_vel,
                           const unsigned int _ntrial,
                           const unsigned int _nwork_local[],
                           const unsigned int _work_offset[],
                           const unsigned int* _d_n_depletants_ntrial,
                           int* _d_deltaF_int,
                           const hipStream_t* _streams_phase1,
                           const hipStream_t* _streams_phase2,
                           const unsigned int _max_len,
                           unsigned int* _d_req_len,
                           const bool _add_ghosts,
                           const unsigned int _n_ghosts,
                           const GPUPartition& _gpu_partition_rank)
        : d_tag(_d_tag), d_vel(_d_vel), d_trial_vel(_d_trial_vel), ntrial(_ntrial),
          nwork_local(_nwork_local), work_offset(_work_offset),
          d_n_depletants_ntrial(_d_n_depletants_ntrial), d_deltaF_int(_d_deltaF_int),
          streams_phase1(_streams_phase1), streams_phase2(_streams_phase2), max_len(_max_len),
          d_req_len(_d_req_len), add_ghosts(_add_ghosts), n_ghosts(_n_ghosts),
          gpu_partition_rank(_gpu_partition_rank) { };

    const unsigned int* d_tag;  //!< Particle tags
    const Scalar4* d_vel;       //!< Particle velocities (.x component is the auxilliary variable)
    const Scalar4* d_trial_vel; //!< Particle velocities after trial move (.x component is the
                                //!< auxilliary variable)
    const unsigned int ntrial;  //!< Number of trial insertions per depletant
    const unsigned int* nwork_local;           //!< Number of insertions this rank handles, per GPU
    const unsigned int* work_offset;           //!< Offset into insertions for this rank
    const unsigned int* d_n_depletants_ntrial; //!< Number of depletants per particle, depletant
                                               //!< type pair and trial insertion
    int* d_deltaF_int;                         //!< Free energy difference rescaled to integer units
    const hipStream_t* streams_phase1;         //!< Stream for this depletant type, phase1 kernel
    const hipStream_t* streams_phase2;         //!< Stream for this depletant type, phase2 kernel
    const unsigned int max_len;  //!< Max length of dynamically allocated shared memory list
    unsigned int* d_req_len;     //!< Requested length of shared mem list per group
    const bool add_ghosts;       //!< True if we should add the ghosts from the domain decomposition
    const unsigned int n_ghosts; //!< Number of ghost particles
    const GPUPartition& gpu_partition_rank; //!< Split of particles for this rank
    };

//! Driver for kernel::hpmc_insert_depletants_auxilliary_phase2()
template<class Shape>
void hpmc_depletants_auxilliary_phase2(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxilliary_args_t& auxilliary_args,
                                       const typename Shape::param_type* params);

//! Driver for kernel::hpmc_insert_depletants_auxilliary_phase1()
template<class Shape>
void hpmc_depletants_auxilliary_phase1(const hpmc_args_t& args,
                                       const hpmc_implicit_args_t& implicit_args,
                                       const hpmc_auxilliary_args_t& auxilliary_args,
                                       const typename Shape::param_type* params);

//! Driver for kernel::hpmc_depletants_accept
void hpmc_depletants_accept(const uint16_t seed,
                            const uint64_t timestep,
                            const unsigned int select,
                            const unsigned int rank,
                            const int* d_deltaF_int,
                            const unsigned int deltaF_pitch,
                            const Scalar* d_fugacity,
                            const unsigned int* d_ntrial,
                            unsigned* d_reject_out,
                            const GPUPartition& gpu_partition,
                            const unsigned int block_size,
                            const unsigned int ntypes);

void generate_num_depletants_ntrial(const Scalar4* d_vel,
                                    const Scalar4* d_trial_vel,
                                    const unsigned int ntrial,
                                    const unsigned int depletant_type_a,
                                    const Scalar* d_lambda,
                                    const Scalar4* d_postype,
                                    unsigned int* d_n_depletants,
                                    const unsigned int N_local,
                                    const bool add_ghosts,
                                    const unsigned int n_ghosts,
                                    const GPUPartition& gpu_partition,
                                    const unsigned int block_size,
                                    const hipStream_t* streams,
                                    const unsigned int ntypes);

void get_max_num_depletants_ntrial(const unsigned int ntrial,
                                   unsigned int* d_n_depletants,
                                   unsigned int* max_n_depletants,
                                   const bool add_ghosts,
                                   const unsigned int n_ghosts,
                                   const hipStream_t* streams,
                                   const GPUPartition& gpu_partition,
                                   CachedAllocator& alloc);
    } // end namespace gpu

    } // end namespace hpmc
    } // end namespace hoomd
