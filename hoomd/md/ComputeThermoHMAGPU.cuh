// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _COMPUTE_THERMO_GPU_CUH_
#define _COMPUTE_THERMO_GPU_CUH_

#include <hip/hip_runtime.h>

#include "ComputeThermoHMATypes.h"
#include "hoomd/GPUPartition.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

/*! \file ComputeThermoHMAGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoHMAGPU
    */

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Holder for arguments to gpu_compute_thermo
struct compute_thermo_hma_args
    {
    Scalar4* d_net_force;    //!< Net force / pe array to sum
    Scalar* d_net_virial;    //!< Net virial array to sum
    size_t virial_pitch;     //!< Pitch of 2D net_virial array
    unsigned int D;          //!< Dimensionality of the system
    Scalar3* d_scratch;      //!< n_blocks elements of scratch space for partial sums
    unsigned int block_size; //!< Block size to execute on the GPU
    unsigned int n_blocks;   //!< Number of blocks to execute / n_blocks * block_size >= group_size
    Scalar external_virial_xx; //!< xx component of the external virial
    Scalar external_virial_yy; //!< yy component of the external virial
    Scalar external_virial_zz; //!< zz component of the external virial
    Scalar external_energy;    //!< External potential energy
    Scalar temperature;        //!< Simulation temperature
    Scalar harmonicPressure;   //!< Harmonic pressure
    };

//! Computes the partial sums of thermodynamic properties for ComputeThermo
hipError_t gpu_compute_thermo_hma_partial(Scalar4* d_pos,
                                          Scalar3* d_lattice_site,
                                          int3* d_image,
                                          unsigned int* d_body,
                                          unsigned int* d_tag,
                                          unsigned int* d_group_members,
                                          unsigned int group_size,
                                          const BoxDim& box,
                                          const compute_thermo_hma_args& args,
                                          const GPUPartition& gpu_partition);

//! Computes the final sums of thermodynamic properties for ComputeThermo
hipError_t gpu_compute_thermo_hma_final(Scalar* d_properties,
                                        unsigned int* d_body,
                                        unsigned int* d_tag,
                                        unsigned int* d_group_members,
                                        unsigned int group_size,
                                        const BoxDim& box,
                                        const compute_thermo_hma_args& args);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
