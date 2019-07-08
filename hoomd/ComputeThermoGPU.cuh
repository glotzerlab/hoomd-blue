// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef _COMPUTE_THERMO_GPU_CUH_
#define _COMPUTE_THERMO_GPU_CUH_

#include <cuda_runtime.h>

#include "ParticleData.cuh"
#include "ComputeThermoTypes.h"
#include "HOOMDMath.h"
#include "GPUPartition.cuh"

/*! \file ComputeThermoGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoGPU
    */

//! Holder for arguments to gpu_compute_thermo
struct compute_thermo_args
    {
    Scalar4 *d_net_force;    //!< Net force / pe array to sum
    Scalar *d_net_virial;    //!< Net virial array to sum
    Scalar4 *d_orientation;  //!< Particle data orientations
    Scalar4 *d_angmom;    //!< Particle data conjugate quaternions
    Scalar3 *d_inertia;      //!< Particle data moments of inertia
    unsigned int virial_pitch; //!< Pitch of 2D net_virial array
    unsigned int ndof;      //!< Number of degrees of freedom for T calculation
    unsigned int D;         //!< Dimensionality of the system
    Scalar4 *d_scratch;      //!< n_blocks elements of scratch space for partial sums
    Scalar *d_scratch_pressure_tensor; //!< n_blocks*6 elements of scratch space for partial sums of the pressure tensor
    Scalar *d_scratch_rot;      //!< Scratch space for rotational kinetic energy partial sums
    unsigned int block_size;    //!< Block size to execute on the GPU
    unsigned int n_blocks;      //!< Number of blocks to execute / n_blocks * block_size >= group_size
    Scalar external_virial_xx;  //!< xx component of the external virial
    Scalar external_virial_xy;  //!< xy component of the external virial
    Scalar external_virial_xz;  //!< xz component of the external virial
    Scalar external_virial_yy;  //!< yy component of the external virial
    Scalar external_virial_yz;  //!< yz component of the external virial
    Scalar external_virial_zz;  //!< zz component of the external virial
    Scalar external_energy;     //!< External potential energy
    };

//! Computes the partial sums of thermodynamic properties for ComputeThermo
cudaError_t gpu_compute_thermo_partial(Scalar *d_properties,
                               Scalar4 *d_vel,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_args& args,
                               bool compute_pressure_tensor,
                               bool compute_rotational_energy,
                               const GPUPartition& gpu_partition
                               );

//! Computes the final sums of thermodynamic properties for ComputeThermo
cudaError_t gpu_compute_thermo_final(Scalar *d_properties,
                               Scalar4 *d_vel,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_args& args,
                               bool compute_pressure_tensor,
                               bool compute_rotational_energy
                               );

#endif
