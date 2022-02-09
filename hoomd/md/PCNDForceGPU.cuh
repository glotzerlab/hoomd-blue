// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: akohlmey

#include "hip/hip_runtime.h"
#include "hoomd/ParticleData.cuh"
#include "hoomd/Index1D.h"
#include "hoomd/HOOMDMath.h"

/*! \file PCNDForceGPU.cuh
    \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by PCNDForceComputeGPU.
*/

#ifndef __PCNDFORCEGPU_CUH__
#define __PCNDFORCEGPU_CUH__

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel driver that computes lj forces on the GPU for PCNDForceComputeGPU
hipError_t gpu_compute_pcnd_forces(const unsigned int group_size,
		                     Scalar4* d_force,
                                     Scalar* d_virial,
                                     uint64_t virial_pitch,
                                     const Scalar4 *d_pos,
                                     const BoxDim& box,
                                     const unsigned int *d_n_neigh,
                                     const unsigned int *d_nlist,
                                     const size_t *d_head_list,
                                     const Scalar4 *d_coeffs,
                                     //const unsigned int size_nlist,
                                     //const unsigned int coeff_width,
                                     const Scalar r_cutsq,
                                     const unsigned int coeff_width,
				     const unsigned int size_nlist,
				     const unsigned int block_size,
                                     const unsigned int max_tex1d_width);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
