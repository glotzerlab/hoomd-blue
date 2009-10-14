#ifndef _EXAMPLE_UPDATER_CUH_
#define _EXAMPLE_UPDATER_CUH_

// there is no convenient header to include all GPU related headers, we need to include those that are needed
#include <hoomd/hoomd_config.h>
// need to include the particle data definition
#include <hoomd/ParticleData.cuh>

// A C API call to run a CUDA kernel is needed for ExampleUpdaterGPU to call
//! Zeros velocities on the GPU
extern "C" cudaError_t gpu_zero_velocities(const gpu_pdata_arrays &pdata);

#endif // _EXAMPLE_UPDATER_CUH_

