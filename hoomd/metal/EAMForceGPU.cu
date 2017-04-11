// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: Lin Yang, Alex Travesset
// Previous Maintainer: Morozov

#include "EAMForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file EAMForceGPU.cu
 \brief Defines GPU kernel code for calculating the EAM forces. Used by EAMForceComputeGPU.
 */

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;
//! Texture for reading the neighbor list
texture<unsigned int, 1, cudaReadModeElementType> nlist_tex;
//! Texture for reading potential
scalar4_tex_t tex_F;
scalar4_tex_t tex_rho;
scalar4_tex_t tex_rphi;
scalar4_tex_t tex_dF;
scalar4_tex_t tex_drho;
scalar4_tex_t tex_drphi;

//! Storage space for EAM parameters on the GPU
__constant__ EAMTexInterData eam_data_ti;

//! Kernel for computing EAM forces on the GPU
template<unsigned char use_gmem_nlist>
__global__ void gpu_compute_eam_tex_inter_forces_kernel(Scalar4 *d_force,
														Scalar *d_virial,
														const unsigned int virial_pitch,
														const unsigned int N,
														const Scalar4 *d_pos,
														BoxDim box,
														const unsigned int *d_n_neigh,
														const unsigned int *d_nlist,
														const unsigned int *d_head_list,
														Scalar *atomDerivativeEmbeddingFunction,
														const Scalar4 *d_F,
														const Scalar4 *d_rho,
														const Scalar4 *d_rphi,
														const Scalar4 *d_dF,
														const Scalar4 *d_drho,
														const Scalar4 *d_drphi) {

	// start by identifying which particle we are to handle
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
	return;

	// load in the length of the list
	int n_neigh = d_n_neigh[idx];
	const unsigned int head_idx = d_head_list[idx];

	// read in the position of our particle.
	Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
	Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // index and remainder
    Scalar position;  // look up position, scalar
    unsigned int int_position;  // look up index for position, integer
    unsigned int idxs;  // look up index in F, rho, rphi array, considering shift, integer
    Scalar remainder;  // look up remainder in array, integer
	Scalar4 v, dv; // value, d(value)

	// initialize the force to 0
	Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

	// prefetch neighbor index
	int cur_neigh = 0;
	int next_neigh(0);
	if (use_gmem_nlist)
	{
		next_neigh = d_nlist[head_idx];
	}
	else
	{
		next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx);
	}
	int typei = __scalar_as_int(postype.w);
    
	// loop over neighbors
	Scalar atomElectronDensity = Scalar(0.0);
    int ntypes = eam_data_ti.ntypes;
    int nrho = eam_data_ti.nrho;
	int nr = eam_data_ti.nr;
    Scalar rdrho = eam_data_ti.rdrho;
    Scalar rdr = eam_data_ti.rdr;
    Scalar r_cutsq = eam_data_ti.r_cutsq;

	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
	{
		// read the current neighbor index (MEM TRANSFER: 4 bytes)
		// prefetch the next value and set the current one
		cur_neigh = next_neigh;
		if (use_gmem_nlist)
		{
			next_neigh = d_nlist[head_idx + neigh_idx + 1];
		}
		else
		{
			next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idx+1);
		}

		// get the neighbor's position
		Scalar4 neigh_postype = texFetchScalar4(d_pos, pdata_pos_tex, cur_neigh);
		Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

		// calculate dr (with periodic boundary conditions)
		Scalar3 dx = pos - neigh_pos;
		int typej = __scalar_as_int(neigh_postype.w);
		// apply periodic boundary conditions
		dx = box.minImage(dx);

		// calculate r squared
		Scalar rsq = dot(dx, dx);;
		if (rsq < r_cutsq)
		{
            position = sqrtf(rsq) * rdr;
            int_position = (unsigned int) position;
            int_position = min(int_position, nr-1);
            remainder = position - int_position;

			idxs = int_position + nr * (typej * ntypes + typei);
			v = texFetchScalar4(d_rho, tex_rho, idxs);
			atomElectronDensity += v.w + v.z * remainder + v.y * remainder * remainder+
								   v.x * remainder * remainder * remainder;
		}
	}

	position = atomElectronDensity * rdrho;
    int_position = (unsigned int) position;
    int_position = min(int_position, nrho - 1);
    remainder = position - int_position;

	idxs = int_position + typei * nrho;
	dv = texFetchScalar4(d_dF, tex_dF, idxs);
	v = texFetchScalar4(d_F, tex_F, idxs);
	atomDerivativeEmbeddingFunction[idx] = dv.z + dv.y * remainder + dv.x * remainder * remainder;
	force.w += v.w + v.z * remainder + v.y * remainder * remainder +
			v.x * remainder * remainder * remainder;

	d_force[idx] = force;

}

//! Second stage kernel for computing EAM forces on the GPU
template<unsigned char use_gmem_nlist>
__global__ void gpu_compute_eam_tex_inter_forces_kernel_2(Scalar4 *d_force,
														  Scalar *d_virial,
														  const unsigned int virial_pitch,
														  const unsigned int N,
														  const Scalar4 *d_pos,
														  BoxDim box,
														  const unsigned int *d_n_neigh,
														  const unsigned int *d_nlist,
														  const unsigned int *d_head_list,
														  Scalar *atomDerivativeEmbeddingFunction,
														  const Scalar4 *d_F,
														  const Scalar4 *d_rho,
														  const Scalar4 *d_rphi,
														  const Scalar4 *d_dF,
														  const Scalar4 *d_drho,
														  const Scalar4 *d_drphi) {

	// start by identifying which particle we are to handle
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
	return;

	// loadj in the length of the list (MEM_TRANSFER: 4 bytes)
	int n_neigh = d_n_neigh[idx];
	const unsigned int head_idx = d_head_list[idx];

	// read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
	// (MEM TRANSFER: 16 bytes)
	Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
	Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
	int typei = __scalar_as_int(postype.w);

    // index and remainder
    Scalar position;  // look up position, scalar
    unsigned int int_position;  // look up index for position, integer
    unsigned int idxs;  // look up index in F, rho, rphi array, considering shift, integer
    Scalar remainder;  // look up remainder in array, integer
	Scalar4 v, dv; // value, d(value)

    // prefetch neighbor index
	int cur_neigh = 0;
	int next_neigh(0);
	if (use_gmem_nlist)
	{
		next_neigh = d_nlist[head_idx];
	}
	else
	{
		next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx);
	}
	//Scalar4 force = force_data.force[idx];
	Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
	//force.w = force_data.force[idx].w;
	Scalar fxi = Scalar(0.0);
	Scalar fyi = Scalar(0.0);
	Scalar fzi = Scalar(0.0);
	Scalar m_pe = Scalar(0.0);
	Scalar pairForce = Scalar(0.0);
	Scalar virial[6];
	for (int i = 0; i < 6; i++)
	virial[i] = Scalar(0.0);

	force.w = d_force[idx].w;
	int ntypes = eam_data_ti.ntypes;
    int nr = eam_data_ti.nr;
    Scalar rdr = eam_data_ti.rdr;
    Scalar r_cutsq = eam_data_ti.r_cutsq;
	Scalar adef = atomDerivativeEmbeddingFunction[idx];
	for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
	{
		cur_neigh = next_neigh;
		if (use_gmem_nlist)
		{
			next_neigh = d_nlist[head_idx + neigh_idx + 1];
		}
		else
		{
			next_neigh = texFetchUint(d_nlist, nlist_tex, head_idx + neigh_idx+1);
		}

		// get the neighbor's position (MEM TRANSFER: 16 bytes)
		Scalar4 neigh_postype = texFetchScalar4(d_pos, pdata_pos_tex,cur_neigh);
		Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

		// calculate dr (with periodic boundary conditions) (FLOPS: 3)
		Scalar3 dx = pos - neigh_pos;
		int typej = __scalar_as_int(neigh_postype.w);
		// apply periodic boundary conditions: (FLOPS 12)
		dx = box.minImage(dx);

		// calculate r squared (FLOPS: 5)
		Scalar rsq = dot(dx, dx);

		if (rsq > r_cutsq) continue;

		Scalar inverseR = rsqrtf(rsq);
		Scalar r = Scalar(1.0) / inverseR;
		position = r * rdr;
        int_position = (unsigned int) position;
        int_position = min(int_position, nr-1);
        remainder = position - int_position;

        int shift = (typei>=typej)?(int)(0.5 * (2 * ntypes - typej -1)*typej + typei) * nr:
					(int)(0.5 * (2 * ntypes - typei -1)*typei + typej) * nr;

		idxs = int_position + shift;
		v = texFetchScalar4(d_rphi, tex_rphi, idxs);
		dv = texFetchScalar4(d_drphi, tex_drphi, idxs);
		Scalar aspair_potential = v.w + v.z * remainder + v.y * remainder * remainder +
				v.x * remainder * remainder * remainder;
		Scalar derivative_pair_potential = dv.z + dv.y * remainder + dv.x * remainder * remainder;

		Scalar pair_eng = aspair_potential * inverseR;
		Scalar derivativePhi = (derivative_pair_potential - pair_eng) * inverseR;

		idxs = int_position + typei * ntypes * nr + typej * nr;
		dv = texFetchScalar4(d_drho, tex_drho, idxs);
		Scalar derivativeRhoI = dv.z + dv.y * remainder + dv.x * remainder * remainder;

		idxs = int_position + typej * ntypes * nr + typei * nr;
		dv = texFetchScalar4(d_drho, tex_drho, idxs);
		Scalar derivativeRhoJ = dv.z + dv.y * remainder + dv.x * remainder * remainder;

		Scalar fullDerivativePhi = adef * derivativeRhoJ +
		atomDerivativeEmbeddingFunction[cur_neigh] * derivativeRhoI + derivativePhi;
		pairForce = - fullDerivativePhi * inverseR;
		Scalar pairForceover2 = Scalar(0.5) *pairForce;
		virial[0] += dx.x * dx.x *pairForceover2;
		virial[1] += dx.x * dx.y *pairForceover2;
		virial[2] += dx.x * dx.z *pairForceover2;
		virial[3] += dx.y * dx.y *pairForceover2;
		virial[4] += dx.y * dx.z *pairForceover2;
		virial[5] += dx.z * dx.z *pairForceover2;

		fxi += dx.x * pairForce;
		fyi += dx.y * pairForce;
		fzi += dx.z * pairForce;
		m_pe += pair_eng * Scalar(0.5);
	}

	force.x = fxi;
	force.y = fyi;
	force.z = fzi;
	force.w += m_pe;
	// now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
	d_force[idx] = force;
	for (int i = 0; i < 6; i++)
	d_virial[i*virial_pitch+idx] = virial[i];

}

cudaError_t gpu_compute_eam_tex_inter_forces(Scalar4 *d_force,
											 Scalar *d_virial,
											 const unsigned int virial_pitch,
											 const unsigned int N,
											 const Scalar4 *d_pos,
											 const BoxDim &box,
											 const unsigned int *d_n_neigh,
											 const unsigned int *d_nlist,
											 const unsigned int *d_head_list,
											 const unsigned int size_nlist,
											 const EAMTexInterArrays &eam_arrays,
											 const EAMTexInterData &eam_data,
											 const Scalar4 *d_F,
											 const Scalar4 *d_rho,
											 const Scalar4 *d_rphi,
											 const Scalar4 *d_dF,
											 const Scalar4 *d_drho,
											 const Scalar4 *d_drphi,
											 const unsigned int compute_capability,
											 const unsigned int max_tex1d_width) {

	cudaError_t error;

    // bind the texture
	if (compute_capability < 350 && size_nlist <= max_tex1d_width) {
		nlist_tex.normalized = false;
		nlist_tex.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, nlist_tex, d_nlist, sizeof(unsigned int) * size_nlist);
		if (error != cudaSuccess)
			return error;
	}

    if (compute_capability < 350)
    {
		tex_F.normalized = false;
		tex_F.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_F, d_F, sizeof(Scalar) * eam_data.nrho * eam_data.ntypes);
		if (error != cudaSuccess)
			return error;

		tex_dF.normalized = false;
		tex_dF.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_dF, d_dF, sizeof(Scalar) * eam_data.nrho * eam_data.ntypes);
		if (error != cudaSuccess)
			return error;

		tex_rho.normalized = false;
		tex_rho.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_rho, d_rho, sizeof(Scalar) * eam_data.nrho * eam_data.ntypes * eam_data.ntypes);
		if (error != cudaSuccess)
			return error;

		tex_drho.normalized = false;
		tex_drho.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_drho, d_drho, sizeof(Scalar) * eam_data.nrho * eam_data.ntypes * eam_data.ntypes);
		if (error != cudaSuccess)
			return error;

		tex_rphi.normalized = false;
		tex_rphi.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_rphi, d_rphi, sizeof(Scalar) * (int) (0.5 * eam_data.nr * (eam_data.ntypes + 1) * eam_data.ntypes));
		if (error != cudaSuccess)
			return error;

		tex_drphi.normalized = false;
		tex_drphi.filterMode = cudaFilterModePoint;
		error = cudaBindTexture(0, tex_drphi, d_drphi, sizeof(Scalar) * (int) (0.5 * eam_data.nr * (eam_data.ntypes + 1) * eam_data.ntypes));
		if (error != cudaSuccess)
			return error;
    }

	pdata_pos_tex.normalized = false;
	pdata_pos_tex.filterMode = cudaFilterModePoint;
	error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
	if (error != cudaSuccess)
	return error;

	// run the kernel
	cudaMemcpyToSymbol(eam_data_ti, &eam_data, sizeof(EAMTexInterData));

	if (compute_capability < 350 && size_nlist > max_tex1d_width) {
		static unsigned int max_block_size = UINT_MAX;
		if (max_block_size == UINT_MAX) {
			cudaFuncAttributes attr;
			cudaFuncGetAttributes(&attr, gpu_compute_eam_tex_inter_forces_kernel<1>);

			cudaFuncAttributes attr2;
			cudaFuncGetAttributes(&attr2, gpu_compute_eam_tex_inter_forces_kernel_2<1>);

			max_block_size = min(attr.maxThreadsPerBlock, attr2.maxThreadsPerBlock);
		}

		unsigned int run_block_size = min(eam_data.block_size, max_block_size);

		// setup the grid to run the kernel
		dim3 grid((int) ceil((double) N / (double) run_block_size), 1, 1);
		dim3 threads(run_block_size, 1, 1);

		gpu_compute_eam_tex_inter_forces_kernel<1> <<<grid, threads>>>(d_force,
				d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist,
				d_head_list, eam_arrays.atomDerivativeEmbeddingFunction,
		        d_F, d_rho, d_rphi, d_dF, d_drho, d_drphi);

		gpu_compute_eam_tex_inter_forces_kernel_2<1> <<<grid, threads>>>(d_force,
                d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist,
                d_head_list, eam_arrays.atomDerivativeEmbeddingFunction,
				d_F, d_rho, d_rphi, d_dF, d_drho, d_drphi);
	} else {
		static unsigned int max_block_size = UINT_MAX;
		if (max_block_size == UINT_MAX) {
			cudaFuncAttributes attr;
			cudaFuncGetAttributes(&attr, gpu_compute_eam_tex_inter_forces_kernel<0>);

			cudaFuncAttributes attr2;
			cudaFuncGetAttributes(&attr2, gpu_compute_eam_tex_inter_forces_kernel_2<0>);

			max_block_size = min(attr.maxThreadsPerBlock, attr2.maxThreadsPerBlock);
		}

		unsigned int run_block_size = min(eam_data.block_size, max_block_size);

		// setup the grid to run the kernel
		dim3 grid((int) ceil((double) N / (double) run_block_size), 1, 1);
		dim3 threads(run_block_size, 1, 1);

		gpu_compute_eam_tex_inter_forces_kernel<0> <<<grid, threads>>>(d_force,
				d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist,
				d_head_list, eam_arrays.atomDerivativeEmbeddingFunction,
				d_F, d_rho, d_rphi, d_dF, d_drho, d_drphi);

		gpu_compute_eam_tex_inter_forces_kernel_2<0> <<<grid, threads>>>(d_force,
                d_virial, virial_pitch, N, d_pos, box, d_n_neigh, d_nlist,
                d_head_list, eam_arrays.atomDerivativeEmbeddingFunction,
				d_F, d_rho, d_rphi, d_dF, d_drho, d_drphi);
	}

	return cudaSuccess;
}

// vim:syntax=cpp
