// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ajs42

#ifndef _COMPUTE_THERMO_GPU_CUH_
#define _COMPUTE_THERMO_GPU_CUH_

#include <cuda_runtime.h>

#include "ParticleData.cuh"
#include "ComputeThermoHMATypes.h"
#include "HOOMDMath.h"
#include "GPUPartition.cuh"

/*! \file ComputeThermoHMAGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoHMAGPU
    */

//! Holder for arguments to gpu_compute_thermo
struct compute_thermo_hma_args
    {
    Scalar4 *d_net_force;    //!< Net force / pe array to sum
    Scalar *d_net_virial;    //!< Net virial array to sum
    unsigned int virial_pitch; //!< Pitch of 2D net_virial array
    unsigned int D;         //!< Dimensionality of the system
    Scalar3 *d_scratch;      //!< n_blocks elements of scratch space for partial sums
    unsigned int block_size;    //!< Block size to execute on the GPU
    unsigned int n_blocks;      //!< Number of blocks to execute / n_blocks * block_size >= group_size
    Scalar external_virial_xx;  //!< xx component of the external virial
    Scalar external_virial_yy;  //!< yy component of the external virial
    Scalar external_virial_zz;  //!< zz component of the external virial
    Scalar external_energy;     //!< External potential energy
    Scalar temperature;         //!< Simulation temperature
    Scalar harmonicPressure;    //!< Harmonic pressure
    };

//! Computes the partial sums of thermodynamic properties for ComputeThermo
cudaError_t gpu_compute_thermo_hma_partial(Scalar4 *d_pos,
                               Scalar3 *d_lattice_site,
                               int3 *d_image,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_hma_args& args,
                               const GPUPartition& gpu_partition
                               );

//! Computes the final sums of thermodynamic properties for ComputeThermo
cudaError_t gpu_compute_thermo_hma_final(Scalar *d_properties,
                               unsigned int *d_body,
                               unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_hma_args& args
                               );

#endif
