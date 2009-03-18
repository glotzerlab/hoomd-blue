#include "ForceCompute.cuh"
#include "AngleData.cuh"
#include "ParticleData.cuh"

/*! \file CGCMMAngleForceGPU.cuh
	\brief Declares GPU kernel code for calculating the CGCMM angle forces. Used by CGCMMAngleForceComputeGPU.
*/

#ifndef __CGCMMANGLEFORCEGPU_CUH__
#define __CGCMMANGLEFORCEGPU_CUH__

//! Kernel driver that computes harmonic angle forces for HarmonicAngleForceComputeGPU
cudaError_t gpu_compute_CGCMM_angle_forces(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_angletable_array &atable, float2 *d_params, float2 *d_CGCMMsr, float4 *d_CGCMMepow, unsigned int n_angle_types, int block_size);

#endif
