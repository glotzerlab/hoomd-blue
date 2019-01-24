// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepLangevinGPU.cuh
    \brief Declares GPU kernel code for Langevin dynamics on the GPU. Used by TwoStepLangevinGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

#ifndef __TWO_STEP_LANGEVIN_GPU_CUH__
#define __TWO_STEP_LANGEVIN_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_langevin_step_two()
struct langevin_step_two_args
    {
    Scalar *d_gamma;          //!< Device array listing per-type gammas
    unsigned int n_types;     //!< Number of types in \a d_gamma
    bool use_lambda;          //!< Set to true to scale diameters by lambda to get gamma
    Scalar lambda;            //!< Scale factor to convert diameter to lambda
    Scalar T;                 //!< Current temperature
    unsigned int timestep;    //!< Current timestep
    unsigned int seed;        //!< User chosen random number seed
    Scalar *d_sum_bdenergy;   //!< Energy transfer sum from bd thermal reservoir
    Scalar *d_partial_sum_bdenergy;  //!< Array used for summation
    unsigned int block_size;  //!<  Block size
    unsigned int num_blocks;  //!<  Number of blocks
    bool noiseless_t;         //!<  If set true, there will be no translational noise (random force)
    bool noiseless_r;         //!<  If set true, there will be no rotational noise (random torque)
    bool tally;               //!< Set to true is bd thermal reservoir energy tally is to be performed
    };

//! Kernel driver for the second part of the Langevin update called by TwoStepLangevinGPU
cudaError_t gpu_langevin_step_two(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar3 *d_accel,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar4 *d_net_force,
                                  const langevin_step_two_args& langevin_args,
                                  Scalar deltaT,
                                  unsigned int D);

//! Kernel driver for the second part of the angular Langevin update (NO_SQUISH) by TwoStepLangevinGPU
cudaError_t gpu_langevin_angular_step_two(const Scalar4 *d_pos,
                             Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             Scalar4 *d_net_torque,
                             const unsigned int *d_group_members,
                             const Scalar3 *d_gamma_r,
                             const unsigned int *d_tag,
                             unsigned int group_size,
                             const langevin_step_two_args& langevin_args,
                             Scalar deltaT,
                             unsigned int D,
                             Scalar scale
                            );


#endif //__TWO_STEP_LANGEVIN_GPU_CUH__
