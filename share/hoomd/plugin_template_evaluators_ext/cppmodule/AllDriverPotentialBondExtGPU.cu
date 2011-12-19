#include "AllDriverPotentialBondExtGPU.cuh"
#include "EvaluatorBondHarmonicDPD.h"

// Every evaluator needs a function in this file. The functions are very simple, containing a one line call to
// a template that does all of the work. To add a additional function, copy and paste this one, change the
// template argument to the correct evaluator <EvaluatorBondMine>, and update the type of the 2nd argument to the
// param_type of the evaluator
cudaError_t gpu_compute_harmonic_dpd_forces(const bond_args_t& bond_args,
                                   const float4 *d_params,
                                   unsigned int *d_flags)
    {
    return gpu_compute_bond_forces<EvaluatorBondHarmonicDPD>(bond_args, d_params, d_flags);
    }
