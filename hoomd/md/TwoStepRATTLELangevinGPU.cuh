// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file TwoStepRATTLELangevinGPU.cuh
    \brief Declares GPU kernel code for RATTLELangevin dynamics on the GPU. Used by TwoStepRATTLELangevinGPU.
*/

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include "EvaluatorConstraintManifold.h"

#ifndef __TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__
#define __TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_rattle_langevin_step_two()
struct rattle_langevin_step_two_args
    {
    Scalar *d_gamma;          //!< Device array listing per-type gammas
    unsigned int n_types;     //!< Number of types in \a d_gamma
    bool use_alpha;          //!< Set to true to scale diameters by alpha to get gamma
    Scalar alpha;            //!< Scale factor to convert diameter to alpha
    Scalar T;                 //!< Current temperature
    Scalar eta;                 
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

//! Kernel driver for the second part of the RATTLELangevin update called by TwoStepRATTLELangevinGPU
cudaError_t gpu_rattle_langevin_step_two(const Scalar4 *d_pos,
                                  Scalar4 *d_vel,
                                  Scalar3 *d_accel,
                                  const Scalar *d_diameter,
                                  const unsigned int *d_tag,
                                  unsigned int *d_group_members,
                                  unsigned int group_size,
                                  Scalar4 *d_net_force,
                                  const rattle_langevin_step_two_args& rattle_langevin_args,
				  EvaluatorConstraintManifold manifold,
                                  Scalar deltaT,
                                  unsigned int D);

//! Kernel driver for the second part of the angular RATTLELangevin update (NO_SQUISH) by TwoStepRATTLELangevinGPU
cudaError_t gpu_rattle_langevin_angular_step_two(const Scalar4 *d_pos,
                             Scalar4 *d_orientation,
                             Scalar4 *d_angmom,
                             const Scalar3 *d_inertia,
                             Scalar4 *d_net_torque,
                             const unsigned int *d_group_members,
                             const Scalar3 *d_gamma_r,
                             const unsigned int *d_tag,
                             unsigned int group_size,
                             const rattle_langevin_step_two_args& rattle_langevin_args,
                             Scalar deltaT,
                             unsigned int D,
                             Scalar scale
                            );


#endif //__TWO_STEP_RATTLE_LANGEVIN_GPU_CUH__
