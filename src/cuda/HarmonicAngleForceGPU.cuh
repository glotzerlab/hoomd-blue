#include "ForceCompute.cuh"
#include "AngleData.cuh"
#include "ParticleData.cuh"

/*! \file HarmonicAngleForceGPU.cuh
	\brief Declares GPU kernel code for calculating the harmonic angle forces. Used by HarmonicAngleForceComputeGPU.
*/

#ifndef __HARMONICANGLEFORCEGPU_CUH__
#define __HARMONICANGLEFORCEGPU_CUH__

//! Kernel driver that computes harmonic angle forces for HarmonicAngleForceComputeGPU
cudaError_t gpu_compute_harmonic_angle_forces(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_angletable_array &atable, float2 *d_params, unsigned int n_angle_types, int block_size);

#endif
