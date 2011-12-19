#include "AllDriverPotentialPairExtGPU.cuh"
#include "EvaluatorPairLJ2.h"

// Every evaluator needs a function in this file. The functions are very simple, containing a one line call to
// a template that does all of the work. To add a additional function, copy and paste this one, change the
// template argument to the correct evaluator <EvaluatorPairMine>, and update the type of the 2nd argument to the
// param_type of the evaluator
cudaError_t gpu_compute_lj2_forces(const pair_args_t& pair_args,
                                   const float2 *d_params)
    {
    return gpu_compute_pair_forces<EvaluatorPairLJ2>(pair_args, d_params);
    }
