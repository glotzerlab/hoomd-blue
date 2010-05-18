/**
powered by:
Moscow group.
*/

#include "ForceCompute.cuh"
#include "NeighborList.cuh"
#include "ParticleData.cuh"

/*! \file EAMTexInterForceGPU.cuh
	\brief Declares GPU kernel code for calculating the eam forces. Used by EAMTexInterForceComputeGPU.
*/

#ifndef __EAMTexInterForceGPU_CUH__
#define __EAMTexInterForceGPU_CUH__
struct EAMTexInterData{
	int ntypes;
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
struct EAMTexInterArrays{
	float* atomDerivativeEmbeddingFunction;
};
struct EAMtex{
	cudaArray* electronDensity;
	cudaArray* pairPotential;
	cudaArray* embeddingFunction;
	cudaArray* derivativeElectronDensity;
	cudaArray* derivativePairPotential;
	cudaArray* derivativeEmbeddingFunction;

};

//! Kernel driver that computes lj forces on the GPU for EAMForceComputeGPU
cudaError_t gpu_compute_eam_tex_inter_forces(
	const gpu_force_data_arrays& force_data,
	const gpu_pdata_arrays &pdata,
	const gpu_boxsize &box,
	const gpu_nlist_array &nlist,
	const EAMtex& eam_tex,
	const EAMTexInterArrays& eam_arrays,
	const EAMTexInterData& eam_data);

#endif
