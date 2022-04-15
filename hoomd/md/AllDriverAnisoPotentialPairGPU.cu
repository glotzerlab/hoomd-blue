// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AllDriverAnisoPotentialPairGPU.cuh"

/*! \file AllDriverAnisoPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of anisotropic pair forces on the
   GPU
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
template
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairGB>(const a_pair_args_t& pair_args,
                                            const EvaluatorPairGB::param_type* d_param,
                                            const EvaluatorPairGB::shape_type* d_shape_param);

template
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairDipole>(const a_pair_args_t& pair_args,
                                     const EvaluatorPairDipole::param_type* d_param,
                                     const EvaluatorPairDipole::shape_type* d_shape_param);

//! Compute anisotropic Lennard-Jones forces and torques on the GPU with EvaluatorPairALJ
template
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairALJ<2>>(const a_pair_args_t& pair_args,
                                     const EvaluatorPairALJ<2>::param_type* d_param,
                                     const EvaluatorPairALJ<2>::shape_type* d_shape_param);

template
hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairALJ<3>>(const a_pair_args_t& pair_args,
                                     const EvaluatorPairALJ<3>::param_type* d_param,
                                     const EvaluatorPairALJ<3>::shape_type* d_shape_param);
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
