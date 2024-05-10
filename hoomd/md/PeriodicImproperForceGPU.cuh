// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproper.h"
#include "hip/hip_runtime.h"
#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"

#pragma once

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes periodic improper forces for PeriodicImproperForceComputeGPU
hipError_t gpu_compute_periodic_improper_forces(Scalar4* d_force,
                                                Scalar* d_virial,
                                                const size_t virial_pitch,
                                                const unsigned int N,
                                                const Scalar4* d_pos,
                                                const BoxDim& box,
                                                const group_storage<4>* tlist,
                                                const unsigned int* improper_ABCD,
                                                const unsigned int pitch,
                                                const unsigned int* n_impropers_list,
                                                periodic_improper_params* d_params,
                                                unsigned int n_improper_types,
                                                int block_size,
                                                int warp_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
