/**
powered by:
Moscow group.
*/

#include "gpu_settings.h"
#include "EAMTexInterForceGPU.cuh"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file EAMTexInterForceGPU.cu
	\brief Defines GPU kernel code for calculating the eam forces. Used by EAMTexInterForceComputeGPU.
*/

//! Texture for reading particle positions
	/*
		cudaArray* electronDensity;
	cudaArray* pairPotential;
	cudaArray* embeddingFunction;
	cudaArray* derivativeElectronDensity;
	cudaArray* derivativePairPotential;
	cudaArray* derivativeEmbeddingFunction;
	*/
texture<float4, 1, cudaReadModeElementType> pdata_pos_tex;
texture<float, 1, cudaReadModeElementType> electronDensity_tex;
texture<float2, 1, cudaReadModeElementType> pairPotential_tex;
texture<float, 1, cudaReadModeElementType> embeddingFunction_tex;
texture<float, 1, cudaReadModeElementType> derivativeElectronDensity_tex;
//texture<float, 1, cudaReadModeElementType> derivativePairPotential_tex;
texture<float, 1, cudaReadModeElementType> derivativeEmbeddingFunction_tex;
texture<float, 1, cudaReadModeElementType> atomDerivativeEmbeddingFunction_tex;
__constant__ EAMTexInterData eam_data;

extern "C" __global__ void gpu_compute_eam_tex_inter_forces_kernel(
	gpu_force_data_arrays force_data,
	gpu_pdata_arrays pdata,
	gpu_boxsize box,
	gpu_nlist_array nlist,
	float* atomDerivativeEmbeddingFunction)
	{
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
	float virial = 0.0f;

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
	float m_pe = 0.0f;
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
				//Определяем индекс в таблице.
				 float position_float = sqrtf(rsq) * eam_data.rdr;
				 atomElectronDensity += tex1D(electronDensity_tex, position_float + nr * (typei * ntypes + typej) + 0.5f ); //electronDensity[r_index + eam_data.nr * typej] + derivativeElectronDensity[r_index + eam_data.nr * typej] * position * eam_data.dr;
				}
			}

		}


	//Определяем индекс в таблице.
	float position = atomElectronDensity * eam_data.rdrho;
	/*unsigned int r_index = (unsigned int)position;
	position -= (float)r_index;*/
	//Извлекаем элементы.Производим интерполяцию.
	atomDerivativeEmbeddingFunction[idx_global] = tex1D(derivativeEmbeddingFunction_tex, position + typei * eam_data.nrho + 0.5f);//derivativeEmbeddingFunction[r_index + typei * eam_data.nrho];

	force.w += tex1D(embeddingFunction_tex, position + typei * eam_data.nrho + 0.5f);//embeddingFunction[r_index + typei * eam_data.nrho] + derivativeEmbeddingFunction[r_index + typei * eam_data.nrho] * position * eam_data.drho;
	force_data.force[idx_local] = force;
	}
extern "C" __global__ void gpu_compute_eam_tex_inter_forces_kernel_2(
	gpu_force_data_arrays force_data,
	gpu_pdata_arrays pdata,
	gpu_boxsize box,
	gpu_nlist_array nlist,
	float* atomDerivativeEmbeddingFunction)
	{
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
	int typei = __float_as_int(pos.w);
	// prefetch neighbor index
	float position;
	int cur_neigh = 0;
	int next_neigh = nlist.list[idx_global];
	//float4 force = force_data.force[idx_local];
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	//force.w = force_data.force[idx_local].w;
	float fxi = 0.0f;
	float fyi = 0.0f;
	float fzi = 0.0f;
	float m_pe = 0.0f;
	float pairForce = 0.0f;
	float virial = 0.0f;
	force.w = force_data.force[idx_local].w;
	int nr = eam_data.nr;
	int nrho = eam_data.nrho;
	int ntypes = eam_data.ntypes;
	float adef = atomDerivativeEmbeddingFunction[idx_global];
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

			if (rsq > eam_data.r_cutsq) continue;

			float inverseR = rsqrtf(rsq);
            float r = 1.0f / inverseR;
			position = r * eam_data.rdr;
			int shift = (typei>=typej)?(int)(0.5f * (2 * ntypes - typej -1)*typej + typei) * nr:(int)(0.5f * (2 * ntypes - typei -1)*typei + typej) * nr;
            float2 pair_potential = tex1D(pairPotential_tex, position + shift + 0.5f);
			float pair_eng =  pair_potential.x * inverseR;

			float derivativePhi = (pair_potential.y - pair_eng) * inverseR;

			float derivativeRhoI = tex1D(derivativeElectronDensity_tex, position + typei * eam_data.nr + 0.5f);

			float derivativeRhoJ = tex1D(derivativeElectronDensity_tex, position + typej * eam_data.nr + 0.5f);

			float fullDerivativePhi = adef * derivativeRhoJ +
				atomDerivativeEmbeddingFunction[cur_neigh] * derivativeRhoI + derivativePhi;
			 pairForce = - fullDerivativePhi * inverseR;
			virial += float(1.0f/6.0f) * rsq * pairForce;

			fxi += dx * pairForce ;
			fyi += dy * pairForce ;
			fzi += dz * pairForce ;
			m_pe += pair_eng * 0.5f;
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

/*! \param force_data Force data on GPU to write forces to
	\param pdata Particle data on the GPU to perform the calculation on
	\param box Box dimensions (in GPU format) to use for periodic boundary conditions
	\param nlist Neighbor list stored on the gpu
	\param d_coeffs A \a coeff_width by \a coeff_width matrix of coefficients indexed by type
		pair i,j. The x-component is lj1 and the y-component is lj2.
	\param coeff_width Width of the \a d_coeffs matrix.
	\param eam_data.r_cutsq Precomputed r_cut*r_cut, where r_cut is the radius beyond which the
		force is set to 0
	\param block_size Block size to execute

	\returns Any error code resulting from the kernel launch

	This is just a driver for calcEAMForces_kernel, see the documentation for it for more information.
*/
cudaError_t gpu_compute_eam_tex_inter_forces(
	const gpu_force_data_arrays& force_data,
	const gpu_pdata_arrays &pdata,
	const gpu_boxsize &box,
	const gpu_nlist_array &nlist,
	const EAMtex& eam_tex,
	const EAMTexInterArrays& eam_arrays,
	const EAMTexInterData& eam_data)
	{
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)pdata.local_num / (double)eam_data.block_size), 1, 1);
    dim3 threads(eam_data.block_size, 1, 1);

	// bind the texture
	pdata_pos_tex.normalized = false;
	pdata_pos_tex.filterMode = cudaFilterModePoint;
	cudaError_t error = cudaBindTexture(0, pdata_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
	if (error != cudaSuccess)
		return error;

	electronDensity_tex.normalized = false;
	electronDensity_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(electronDensity_tex, eam_tex.electronDensity);
	if (error != cudaSuccess)
		return error;

	pairPotential_tex.normalized = false;
	pairPotential_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(pairPotential_tex, eam_tex.pairPotential);
	if (error != cudaSuccess)
		return error;

	embeddingFunction_tex.normalized = false;
	embeddingFunction_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(embeddingFunction_tex, eam_tex.embeddingFunction);
	if (error != cudaSuccess)
		return error;

	derivativeElectronDensity_tex.normalized = false;
	derivativeElectronDensity_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(derivativeElectronDensity_tex, eam_tex.derivativeElectronDensity);
	if (error != cudaSuccess)
		return error;
/*
	derivativePairPotential_tex.normalized = false;
	derivativePairPotential_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(derivativePairPotential_tex, eam_tex.derivativePairPotential);
	if (error != cudaSuccess)
		return error;
*/
	derivativeEmbeddingFunction_tex.normalized = false;
	derivativeEmbeddingFunction_tex.filterMode = cudaFilterModeLinear ;
	error = cudaBindTextureToArray(derivativeEmbeddingFunction_tex, eam_tex.derivativeEmbeddingFunction);
	if (error != cudaSuccess)
		return error;
    // run the kernel
    cudaMemcpyToSymbol("eam_data", &eam_data, sizeof(EAMTexInterData));

    gpu_compute_eam_tex_inter_forces_kernel<<< grid, threads>>>(force_data,
	pdata,
	box,
	nlist,
	eam_arrays.atomDerivativeEmbeddingFunction);

	gpu_compute_eam_tex_inter_forces_kernel_2<<< grid, threads>>>(force_data,
	pdata,
	box,
	nlist,
	eam_arrays.atomDerivativeEmbeddingFunction);
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
