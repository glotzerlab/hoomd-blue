// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file AllDriverAnisoPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of anisotropic pair forces on the
   GPU
*/

#ifndef __ALL_DRIVER_ANISO_POTENTIAL_PAIR_GPU_CUH__
#define __ALL_DRIVER_ANISO_POTENTIAL_PAIR_GPU_CUH__

#include "AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairALJ.h"
#include "EvaluatorPairDipole.h"
#include "EvaluatorPairGB.h"

//! Compute dipole forces and torques on the GPU with EvaluatorPairDipole

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces_gb(const a_pair_args_t&,
                                 const EvaluatorPairGB::param_type*,
                                 const EvaluatorPairGB::shape_type*);

hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces_dipole(const a_pair_args_t&,
                                     const EvaluatorPairDipole::param_type*,
                                     const EvaluatorPairDipole::shape_type*);

//! Compute anisotropic Lennard-Jones forces and torques on the GPU with EvaluatorPairALJ
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces_ALJ_2D(const a_pair_args_t&,
                                     const EvaluatorPairALJ<2>::param_type*,
                                     const EvaluatorPairALJ<2>::shape_type*);

hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces_ALJ_3D(const a_pair_args_t&,
                                     const EvaluatorPairALJ<3>::param_type*,
                                     const EvaluatorPairALJ<3>::shape_type*);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif
