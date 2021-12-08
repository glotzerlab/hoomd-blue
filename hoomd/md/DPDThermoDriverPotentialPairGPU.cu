// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file DPDThermoDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairDPDThermo.h"
#include "PotentialPairDPDThermoGPU.cuh"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_dpdthermodpd_forces(const dpd_pair_args_t& args,
                                           const EvaluatorPairDPDThermo::param_type* d_params)
    {
    return gpu_compute_dpd_forces<EvaluatorPairDPDThermo>(args, d_params);
    }

hipError_t gpu_compute_dpdthermo_forces(const pair_args_t& pair_args,
                                        const EvaluatorPairDPDThermo::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairDPDThermo>(pair_args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
