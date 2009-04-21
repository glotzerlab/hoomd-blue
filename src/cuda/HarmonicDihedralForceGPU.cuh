#include "ForceCompute.cuh"
#include "DihedralData.cuh"
#include "ParticleData.cuh"

/*! \file HarmonicDihedralForceGPU.cuh
	\brief Declares GPU kernel code for calculating the harmonic dihedral forces. Used by HarmonicDihedralForceComputeGPU.
*/

#ifndef __HARMONICDIHEDRALFORCEGPU_CUH__
#define __HARMONICDIHEDRALFORCEGPU_CUH__

//! Kernel driver that computes harmonic dihedral forces for HarmonicDihedralForceComputeGPU
cudaError_t gpu_compute_harmonic_dihedral_forces(const gpu_force_data_arrays& force_data, const gpu_pdata_arrays &pdata, const gpu_boxsize &box, const gpu_dihedraltable_array &ttable, float4 *d_params, unsigned int n_dihedral_types, int block_size);

#endif
