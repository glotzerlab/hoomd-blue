// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include <hip/hip_runtime.h>

/*! \file PCNDAngleForceGPU.cuh
    \brief Declares GPU kernel code for calculating the PCND angle forces. Used by PCNDAngleForceComputeGPU.
*/

#ifndef __PCNDANGLEFORCEGPU_CUH__
#define __PCNDANGLEFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes angle forces for PCNDAngleForceComputeGPU
hipError_t gpu_compute_PCND_angle_forces(Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
					 const unsigned int* d_tag,
                                         const unsigned int N,
                                         const Scalar4* d_pos,
                                         const BoxDim& box,
                                         const group_storage<3>* atable,
                                         const unsigned int* apos_list,
                                         const unsigned int pitch,
                                         const unsigned int* n_angles_list,
                                         Scalar2* d_params,
                                         unsigned int n_angle_types,
                                         int block_size,
                                         uint64_t timestep,
                                         uint64_t PCNDtimestep);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
