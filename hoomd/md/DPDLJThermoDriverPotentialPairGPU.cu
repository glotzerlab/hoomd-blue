// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file DPDLJThermoDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairDPDLJThermo.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_dpdljthermodpd_forces(const dpd_pair_args_t& args,
                                             const EvaluatorPairDPDLJThermo::param_type* d_params)
    {
    return gpu_compute_dpd_forces<EvaluatorPairDPDLJThermo>(args, d_params);
    }

hipError_t gpu_compute_dpdljthermo_forces(const pair_args_t& args,
                                          const EvaluatorPairDPDLJThermo::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairDPDLJThermo>(args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
