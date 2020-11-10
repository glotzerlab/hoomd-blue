// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepRATTLEBDGPU.cuh
    \brief Declares GPU kernel code for Brownian dynamics on the GPU. Used by TwoStepRATTLEBDGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "EvaluatorConstraintManifold.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/GPUPartition.cuh"

#ifndef __TWO_STEP_RATTLE_BD_GPU_CUH__
#define __TWO_STEP_RATTLE_BD_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_rattle_bd_step_one()
struct rattle_bd_step_one_args
    {
    Scalar *d_gamma;          //!< Device array listing per-type gammas
    unsigned int n_types;     //!< Number of types in \a d_gamma
    bool use_alpha;          //!< Set to true to scale diameters by alpha to get gamma
    Scalar alpha;            //!< Scale factor to convert diameter to alpha
    Scalar T;                 //!< Current temperature
    Scalar eta;                 
    unsigned int timestep;    //!< Current timestep
    unsigned int seed;        //!< User chosen random number seed
    };

//! Kernel driver for the first part of the Brownian update called by TwoStepRATTLEBDGPU
cudaError_t gpu_rattle_brownian_step_one(Scalar4 *d_pos,
                                  int3 *d_image,
                                  const BoxDim &box,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_rtag,
                                  const unsigned int *d_groupTags,
                                  const unsigned int group_size,
                                  const Scalar4 *d_net_force,
                                  const Scalar3 *d_f_brownian,
                                  const Scalar3 *d_gamma_r,
                                  Scalar4 *d_orientation,
                                  Scalar4 *d_torque,
                                  const Scalar3 *d_inertia,
                                  Scalar4 *d_angmom,
                                  const rattle_bd_step_one_args& rattle_bd_args,
                                  const bool aniso,
                                  const Scalar deltaT,
                                  const unsigned int D,
                                  const bool d_noiseless_r,
                                  const GPUPartition& gpu_partition);

//! Kernel driver for the first part of the Brownian update called by TwoStepRATTLEBDGPU
cudaError_t gpu_include_rattle_force_bd(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar4 *d_net_force,
                                  Scalar3 *d_f_brownian,
                                  Scalar *d_net_virial,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_rtag,
                                  const unsigned int *d_groupTags,
                                  const unsigned int group_size,
                                  const rattle_bd_step_one_args& rattle_bd_args,
			          EvaluatorConstraintManifold manifold,
                                  unsigned int net_virial_pitch,
                                  const Scalar deltaT,
                                  const bool d_noiseless_t,
                                  const GPUPartition& gpu_partition);

#endif //__TWO_STEP_RATTLE_BD_GPU_CUH__
