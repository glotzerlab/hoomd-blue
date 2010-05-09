/**
powered by:
Moscow group.
*/
#include "gpu_settings.h"
#include "EAMForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file EAMForceGPU.cu
	\brief Defines GPU kernels code for calculating the EAM forces. Used by EAMForceComputeGPU.
*/

//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
__constant__ float* electronDensity;
__constant__	float* pairPotential;
__constant__	float* embeddingFunction;
__constant__	float* derivativeElectronDensity;
__constant__	float* derivativePairPotential;
__constant__	float* derivativeEmbeddingFunction;
__constant__	float* atomDerivativeEmbeddingFunction;
__constant__	EAMData eam_data;
//!First kernel need to calculate a part of pair potential and to calculate
/*!  Derivative Embedding Function for each atom.
*/
extern "C" __global__ void gpu_compute_eam_forces_kernel(
	gpu_force_data_arrays force_data,
	gpu_pdata_arrays pdata,
	gpu_boxsize box,
	gpu_nlist_array nlist,
	float2 *d_coeffs)
	{
	// read in the coefficients
	extern __shared__ float2 s_coeffs[];


	// start by identifying which particle we are to handle
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_local >= pdata.local_num)
		return;

	int idx_global = idx_local + pdata.local_beg;

	// load in the length of the list (MEM_TRANSFER: 4 bytes)
	int n_neigh = nlist.n_neigh[idx_global];

	// read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
	// (MEM TRANSFER: 16 bytes)
	float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);

	// initialize the force to 0
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	// prefetch neighbor index
	int cur_neigh = 0;
	int next_neigh = nlist.list[idx_global];
	int typei  = __float_as_int(pos.w);

	// loop over neighbors

	#ifdef ARCH_SM13
	// sm13 offers warp voting which makes this hardware bug workaround less of a performance penalty
	#define neigh_for for (int neigh_idx = 0; __any(neigh_idx < n_neigh); neigh_idx++)
	#else
	#define neigh_for for (int neigh_idx = 0; neigh_idx < nlist.height; neigh_idx++)
	#endif
	float atomElectronDensity  = 0.0f;
	int nr = eam_data.nr;
	int nrho = eam_data.nrho;
	int ntypes = eam_data.ntypes;
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
		{
		if (neigh_idx < n_neigh)
			{
			// read the current neighbor index (MEM TRANSFER: 4 bytes)
			// prefetch the next value and set the current one
			cur_neigh = next_neigh;
			if (neigh_idx+1 < nlist.height)
				next_neigh = nlist.list[nlist.pitch*(neigh_idx+1) + idx_global];

			// get the neighbor's position (MEM TRANSFER: 16 bytes)
			float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);

			// calculate dr (with periodic boundary conditions) (FLOPS: 3)
			float dx = pos.x - neigh_pos.x;
			float dy = pos.y - neigh_pos.y;
			float dz = pos.z - neigh_pos.z;
			int typej  = __float_as_int(neigh_pos.w);
			// apply periodic boundary conditions: (FLOPS 12)
			dx -= box.Lx * rintf(dx * box.Lxinv);
			dy -= box.Ly * rintf(dy * box.Lyinv);
			dz -= box.Lz * rintf(dz * box.Lzinv);

			// calculate r squard (FLOPS: 5)
			float rsq = dx*dx + dy*dy + dz*dz;
			if (rsq < eam_data.r_cutsq)
				{
				 float position_float = sqrt(rsq) * eam_data.rdr;
				 float position = position_float;
				 unsigned int r_index = (unsigned int)position_float;
				 position -= r_index;
				 atomElectronDensity += electronDensity[r_index + nr * (typei * ntypes + typej)] + derivativeElectronDensity[r_index + nr * (typei * ntypes + typej)] * position * eam_data.dr;
				}
			}

		}

	float position = atomElectronDensity * eam_data.rdrho;
	unsigned int r_index = (unsigned int)position;
	position -= (float)r_index;
	atomDerivativeEmbeddingFunction[idx_global] = derivativeEmbeddingFunction[r_index + typei * eam_data.nrho];

	force.w += embeddingFunction[r_index + typei * nrho] + derivativeEmbeddingFunction[r_index + typei * nrho] * position * eam_data.drho;
	force_data.force[idx_local] = force;
	}
//!Second kernel need to calculate a second part of pair potential and to calculate
/*!  force  by using Derivative Embedding Function for each atom.
*/
extern "C" __global__ void gpu_compute_eam_forces_kernel_2(
	gpu_force_data_arrays force_data,
	gpu_pdata_arrays pdata,
	gpu_boxsize box,
	gpu_nlist_array nlist,
	float2 *d_coeffs)
	{
		// read in the coefficients
	extern __shared__ float2 s_coeffs[];


	// start by identifying which particle we are to handle
	int idx_local = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_local >= pdata.local_num)
		return;

	int idx_global = idx_local + pdata.local_beg;

	// loadj in the length of the list (MEM_TRANSFER: 4 bytes)
	int n_neigh = nlist.n_neigh[idx_global];

	// read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
	// (MEM TRANSFER: 16 bytes)
	float4 pos = tex1Dfetch(pdata_pos_tex, idx_global);
	// prefetch neighbor index
	float position;
	unsigned int r_index;
	//Now we can use only ony type of atom.
	int typei = __float_as_int(pos.w);

	int cur_neigh = 0;
	int next_neigh = nlist.list[idx_global];
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float fxi = 0.0;
	float fyi = 0.0;
	float fzi = 0.0;
	float m_pe = 0.0f;
	float pairForce = 0.0f;
	float virial = 0.0f;
	force.w = force_data.force[idx_local].w;
	int nr = eam_data.nr;
	int nrho = eam_data.nrho;
	int ntypes = eam_data.ntypes;
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
		{
		if (neigh_idx < n_neigh)
			{
			cur_neigh = next_neigh;
			if (neigh_idx+1 < nlist.height)
				next_neigh = nlist.list[nlist.pitch*(neigh_idx+1) + idx_global];

			// get the neighbor's position (MEM TRANSFER: 16 bytes)
			float4 neigh_pos = tex1Dfetch(pdata_pos_tex, cur_neigh);

			// calculate dr (with periodic boundary conditions) (FLOPS: 3)
			float dx = pos.x - neigh_pos.x;
			float dy = pos.y - neigh_pos.y;
			float dz = pos.z - neigh_pos.z;
			int typej = __float_as_int(neigh_pos.w);
			// apply periodic boundary conditions: (FLOPS 12)
			dx -= box.Lx * rintf(dx * box.Lxinv);
			dy -= box.Ly * rintf(dy * box.Lyinv);
			dz -= box.Lz * rintf(dz * box.Lzinv);

			// calculate r squard (FLOPS: 5)
			float rsq = dx*dx + dy*dy + dz*dz;

			if (rsq <= eam_data.r_cutsq)
				{
				float r = sqrt(rsq);
				float inverseR = 1.0 / r;
				position = r * eam_data.rdr;
				r_index = (unsigned int)position;
				position -= r_index;
				int shift = (typei>=typej)?(int)(0.5 * (2 * ntypes - typej -1)*typej + typei) * nr:(int)(0.5 * (2 * ntypes - typei -1)*typei + typej) * nr;
				float pair_eng = (pairPotential[r_index + shift] +
					derivativePairPotential[r_index + shift] * position * eam_data.dr) * inverseR;
				float derivativePhi = (derivativePairPotential[r_index + shift] - pair_eng) * inverseR;
				float derivativeRhoI = derivativeElectronDensity[r_index + typei * nr];
				float derivativeRhoJ = derivativeElectronDensity[r_index + typej * nr];
				float fullDerivativePhi = atomDerivativeEmbeddingFunction[idx_global] * derivativeRhoJ +
					atomDerivativeEmbeddingFunction[cur_neigh] * derivativeRhoI + derivativePhi;
				pairForce = - fullDerivativePhi * inverseR;

				fxi += dx * pairForce ;
				fyi += dy * pairForce ;
				fzi += dz * pairForce ;
				m_pe += pair_eng * 0.5f;
				//virial += float(1.0/6.0) * ( pos.x * fxi + pos.y * fyi + pos.z * fzi);
				virial += float(1.0/6.0) * (rsq) * pairForce;
				}
			}
		}
		force.x = fxi;
		force.y = fyi;
		force.z = fzi;
		force.w += m_pe;
		// now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
		force_data.force[idx_local] = force;
		force_data.virial[idx_local] = virial;

	}


cudaError_t gpu_compute_eam_forces(
	const gpu_force_data_arrays& force_data,
	const gpu_pdata_arrays &pdata,
	const gpu_boxsize &box,
	const gpu_nlist_array &nlist,
	float2 *d_coeffs,
	int coeff_width,
	const EAMArrays& eam_arrays,
	const EAMData& eam_data)
	{
	assert(d_coeffs);
	assert(coeff_width > 0);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.local_num / (double)eam_data.block_size), 1, 1);
    dim3 threads(eam_data.block_size, 1, 1);

	// bind the texture
	pdata_pos_tex.normalized = false;
	pdata_pos_tex.filterMode = cudaFilterModePoint;
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;
    cudaMemcpyToSymbol("electronDensity", &eam_arrays.electronDensity[0], sizeof(float*));
    cudaMemcpyToSymbol("pairPotential", &eam_arrays.pairPotential[0], sizeof(float*));
    cudaMemcpyToSymbol("embeddingFunction", &eam_arrays.embeddingFunction[0], sizeof(float*));
    cudaMemcpyToSymbol("derivativeElectronDensity", &eam_arrays.derivativeElectronDensity[0], sizeof(float*));
    cudaMemcpyToSymbol("derivativePairPotential", &eam_arrays.derivativePairPotential[0], sizeof(float*));
    cudaMemcpyToSymbol("derivativeEmbeddingFunction", &eam_arrays.derivativeEmbeddingFunction[0], sizeof(float*));
    cudaMemcpyToSymbol("atomDerivativeEmbeddingFunction", &eam_arrays.atomDerivativeEmbeddingFunction[0], sizeof(float*));
    cudaMemcpyToSymbol("eam_data", &eam_data, sizeof(eam_data));

    gpu_compute_eam_forces_kernel<<< grid, threads, sizeof(float2)*coeff_width*coeff_width >>>(force_data,
	pdata,
	box,
	nlist,
	d_coeffs);

	cudaThreadSynchronize();

	gpu_compute_eam_forces_kernel_2<<< grid, threads, sizeof(float2)*coeff_width*coeff_width >>>(force_data,
	pdata,
	box,
	nlist,
	d_coeffs);
	if (!g_gpu_error_checking)
		{
		return cudaSuccess;
		}
	else
		{
		cudaThreadSynchronize();
		return cudaGetLastError();
		}
	}

// vim:syntax=cpp
