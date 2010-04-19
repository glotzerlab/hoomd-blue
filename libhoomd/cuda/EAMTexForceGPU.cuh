/**
powered by:
Moscow group.
*/

#include "ForceCompute.cuh"
#include "NeighborList.cuh"
#include "ParticleData.cuh"

/*! \file EAMTexForceGPU.cuh
	\brief Declares GPU kernel code for calculating the eam forces. Used by EAMTexForceComputeGPU.
*/

#ifndef __EAMTexForceGPU_CUH__
#define __EAMTexForceGPU_CUH__
struct EAMTexData{
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
struct EAMTexArrays{
	float* atomDerivativeEmbeddingFunction;
};
struct EAMLinear{
	float* electronDensity; 
	float* pairPotential; 
	float* embeddingFunction; 
	float* derivativeElectronDensity; 
	float* derivativePairPotential; 
	float* derivativeEmbeddingFunction; 
	int size_electronDensity; 
	int size_pairPotential; 
	int size_embeddingFunction; 
	int size_derivativeElectronDensity; 
	int size_derivativePairPotential; 
	int size_derivativeEmbeddingFunction; 

};

//! Kernel driver that computes lj forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_linear_forces(
	const gpu_force_data_arrays& force_data, 
	const gpu_pdata_arrays &pdata, 
	const gpu_boxsize &box, 
	const gpu_nlist_array &nlist, 
	float2 *d_coeffs, 
	int coeff_width, 
	const EAMLinear& eam_linear, 
	const EAMTexArrays& eam_arrays, 
	const EAMTexData& eam_data);

#endif
