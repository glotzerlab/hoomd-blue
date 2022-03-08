// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file TwoStepLangevinGPU.cuh
    \brief Declares GPU kernel code for Langevin dynamics on the GPU. Used by TwoStepLangevinGPU.
*/

#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include <hip/hip_runtime.h>

#ifndef __TWO_STEP_LANGEVIN_GPU_CUH__
#define __TWO_STEP_LANGEVIN_GPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Temporary holder struct to limit the number of arguments passed to gpu_langevin_step_two()
struct langevin_step_two_args
    {
    Scalar* d_gamma;                //!< Device array listing per-type gammas
    unsigned int n_types;           //!< Number of types in \a d_gamma
    bool use_alpha;                 //!< Set to true to scale diameters by alpha to get gamma
    Scalar alpha;                   //!< Scale factor to convert diameter to alpha
    Scalar T;                       //!< Current temperature
    uint64_t timestep;              //!< Current timestep
    uint16_t seed;                  //!< User chosen random number seed
    Scalar* d_sum_bdenergy;         //!< Energy transfer sum from bd thermal reservoir
    Scalar* d_partial_sum_bdenergy; //!< Array used for summation
    unsigned int block_size;        //!<  Block size
    unsigned int num_blocks;        //!<  Number of blocks
    bool noiseless_t; //!<  If set true, there will be no translational noise (random force)
    bool noiseless_r; //!<  If set true, there will be no rotational noise (random torque)
    bool tally;       //!< Set to true is bd thermal reservoir energy tally is to be performed
    };

//! Kernel driver for the second part of the Langevin update called by TwoStepLangevinGPU
hipError_t gpu_langevin_step_two(const Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 Scalar3* d_accel,
                                 const Scalar* d_diameter,
                                 const unsigned int* d_tag,
                                 unsigned int* d_group_members,
                                 unsigned int group_size,
                                 Scalar4* d_net_force,
                                 const langevin_step_two_args& langevin_args,
                                 Scalar deltaT,
                                 unsigned int D);

//! Kernel driver for the second part of the angular Langevin update (NO_SQUISH) by
//! TwoStepLangevinGPU
hipError_t gpu_langevin_angular_step_two(const Scalar4* d_pos,
                                         Scalar4* d_orientation,
                                         Scalar4* d_angmom,
                                         const Scalar3* d_inertia,
                                         Scalar4* d_net_torque,
                                         const unsigned int* d_group_members,
                                         const Scalar3* d_gamma_r,
                                         const unsigned int* d_tag,
                                         unsigned int group_size,
                                         const langevin_step_two_args& langevin_args,
                                         Scalar deltaT,
                                         unsigned int D,
                                         Scalar scale);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif //__TWO_STEP_LANGEVIN_GPU_CUH__
