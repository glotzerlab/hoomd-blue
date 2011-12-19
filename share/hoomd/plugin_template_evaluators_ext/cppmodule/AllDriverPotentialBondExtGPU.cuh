// Every Evaluator in this plugin needs a corresponding function call here. This function call is responsible for
// performing the bond force computation with that evaluator on the GPU. (See AllDriverBondExtGPU.cu)

#ifndef __ALL_DRIVER_POTENTIAL_BOND_EXT_GPU_CUH__
#define __ALL_DRIVER_POTENTIAL_BOND_EXT_GPU_CUH__

#include "hoomd/hoomd_config.h"
#include "hoomd/PotentialBondGPU.cuh"

//! Compute harmonic+DPD bond forces on the GPU with EvaluatorBondHarmonicDPD
cudaError_t gpu_compute_harmonic_dpd_forces(const bond_args_t& bond_args, const float4 *d_params, unsigned int *d_flags);
// NOTE: The argument d_params must be of the type param_type specified
// in the evaluator class
#endif
