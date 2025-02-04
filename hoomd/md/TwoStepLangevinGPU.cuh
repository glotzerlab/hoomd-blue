// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
    langevin_step_two_args(Scalar* _d_gamma,
                           unsigned int _n_types,
                           Scalar _T,
                           uint64_t _timestep,
                           uint16_t _seed,
                           Scalar* _d_sum_bdenergy,
                           Scalar* _d_partial_sum_bdenergy,
                           unsigned int _block_size,
                           unsigned int _num_blocks,
                           bool _noiseless_t,
                           bool _noiseless_r,
                           bool _tally,
                           const hipDeviceProp_t& _devprop)
        : d_gamma(_d_gamma), n_types(_n_types), T(_T), timestep(_timestep), seed(_seed),
          d_sum_bdenergy(_d_sum_bdenergy), d_partial_sum_bdenergy(_d_partial_sum_bdenergy),
          block_size(_block_size), num_blocks(_num_blocks), noiseless_t(_noiseless_t),
          noiseless_r(_noiseless_r), tally(_tally), devprop(_devprop)
        {
        }

    Scalar* d_gamma;                //!< Device array listing per-type gammas
    unsigned int n_types;           //!< Number of types in \a d_gamma
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
    const hipDeviceProp_t& devprop; //!< Device properties.
    };

//! Kernel driver for the second part of the Langevin update called by TwoStepLangevinGPU
hipError_t gpu_langevin_step_two(const Scalar4* d_pos,
                                 Scalar4* d_vel,
                                 Scalar3* d_accel,
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
