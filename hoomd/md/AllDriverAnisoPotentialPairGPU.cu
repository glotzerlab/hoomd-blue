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
hipError_t gpu_compute_pair_aniso_forces_gb(const a_pair_args_t& pair_args,
                                            const EvaluatorPairGB::param_type* d_param,
                                            const EvaluatorPairGB::shape_type* d_shape_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairGB>(pair_args, d_param, d_shape_param);
    }

hipError_t
gpu_compute_pair_aniso_forces_dipole(const a_pair_args_t& pair_args,
                                     const EvaluatorPairDipole::param_type* d_param,
                                     const EvaluatorPairDipole::shape_type* d_shape_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairDipole>(pair_args, d_param, d_shape_param);
    }

hipError_t
gpu_compute_pair_aniso_forces_ALJ_2D(const a_pair_args_t& pair_args,
                                     const EvaluatorPairALJ<2>::param_type* d_param,
                                     const EvaluatorPairALJ<2>::shape_type* d_shape_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairALJ<2>>(pair_args, d_param, d_shape_param);
    }

hipError_t
gpu_compute_pair_aniso_forces_ALJ_3D(const a_pair_args_t& pair_args,
                                     const EvaluatorPairALJ<3>::param_type* d_param,
                                     const EvaluatorPairALJ<3>::shape_type* d_shape_param)
    {
    return gpu_compute_pair_aniso_forces<EvaluatorPairALJ<3>>(pair_args, d_param, d_shape_param);
    }
    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
