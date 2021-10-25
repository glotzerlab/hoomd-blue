/*! \file TWFDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces
    on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairTWF.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_twf_forces(const pair_args_t& pair_args,
                                  const EvaluatorPairTWF::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairTWF>(pair_args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
