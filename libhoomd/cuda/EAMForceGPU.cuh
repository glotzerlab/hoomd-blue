/**
powered by:
Moscow group.
*/

#include "ForceCompute.cuh"
#include "NeighborList.cuh"
#include "ParticleData.cuh"

/*! \file EAMForceGPU.cu
	\brief Defines GPU kernels code for calculating the EAM forces. Used by EAMForceComputeGPU.
*/

#ifndef __EAMFORCEGPU_CUH__
#define __EAMFORCEGPU_CUH__
struct EAMData{
	int nr;
	int nrho;
	int block_size;
	float dr;
	float rdr;
	float drho;
	float rdrho;
	float r_cutsq;
	float r_cut;
};
struct EAMArrays{
	float* electronDensity; 
	float* pairPotential; 
	float* embeddingFunction; 
	float* derivativeElectronDensity; 
	float* derivativePairPotential; 
	float* derivativeEmbeddingFunction; 
	float* atomDerivativeEmbeddingFunction;
};

//! Kernel driver that computes eam forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_forces(
	const gpu_force_data_arrays& force_data, 
	const gpu_pdata_arrays &pdata, 
	const gpu_boxsize &box, 
	const gpu_nlist_array &nlist, 
	float2 *d_coeffs, 
	int coeff_width, 
	const EAMArrays& eam_arrays, 
	const EAMData& eam_data);

#endif
