/*! \file CosineSquaredDriverPotentialPairGPU.cu
    \brief Defines the driver functions for computing all types of pair forces
    on the GPU
*/

#include "AllDriverPotentialPairGPU.cuh"
#include "EvaluatorPairCosineSquared.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
hipError_t gpu_compute_cosinesquared_forces(const pair_args_t& pair_args,
                                            const EvaluatorPairCosineSquared::param_type* d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairCosineSquared>(pair_args, d_params);
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
