// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



// Maintainer: morozov

/**
powered by:
Moscow group.
*/

#include "EAMForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

/*! \file EAMForceGPU.cu
    \brief Defines GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
*/

//! Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;
//! Texture for reading the neighbor list
texture<unsigned int, 1, cudaReadModeElementType> nlist_tex;


#ifdef SINGLE_PRECISION
//! Texture for reading electron density
texture<Scalar, 1, cudaReadModeElementType> electronDensity_tex;
//! Texture for reading EAM pair potential
texture<Scalar2, 1, cudaReadModeElementType> pairPotential_tex;
//! Texture for reading the embedding function
texture<Scalar, 1, cudaReadModeElementType> embeddingFunction_tex;
//! Texture for reading the derivative of the electron density
texture<Scalar, 1, cudaReadModeElementType> derivativeElectronDensity_tex;
//! Texture for reading the derivative of the embedding function
texture<Scalar, 1, cudaReadModeElementType> derivativeEmbeddingFunction_tex;
//! Texture for reading the derivative of the atom embedding function
texture<Scalar, 1, cudaReadModeElementType> atomDerivativeEmbeddingFunction_tex;

#else
//! Texture for reading electron density
texture<int2, 1, cudaReadModeElementType> electronDensity_tex;
//! Texture for reading EAM pair potential
texture<int4, 1, cudaReadModeElementType> pairPotential_tex;
//! Texture for reading the embedding function
texture<int2, 1, cudaReadModeElementType> embeddingFunction_tex;
//! Texture for reading the derivative of the electron density
texture<int2, 1, cudaReadModeElementType> derivativeElectronDensity_tex;
//! Texture for reading the derivative of the embedding function
texture<int2, 1, cudaReadModeElementType> derivativeEmbeddingFunction_tex;
//! Texture for reading the derivative of the atom embedding function
texture<int2, 1, cudaReadModeElementType> atomDerivativeEmbeddingFunction_tex;

#endif

//! Storage space for EAM parameters on the GPU
__constant__ EAMTexInterData eam_data_ti;

//! Kernel for computing EAM forces on the GPU
template<unsigned char use_gmem_nlist>
__global__ void gpu_compute_eam_tex_inter_forces_kernel(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    BoxDim box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    Scalar* atomDerivativeEmbeddingFunction)
    {
    #ifdef SINGLE_PRECISION
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    int n_neigh = d_n_neigh[idx];
    const unsigned int head_idx = d_head_list[idx];

    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

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
    int typei  = __scalar_as_int(postype.w);
    // loop over neighbors

    Scalar atomElectronDensity  = Scalar(0.0);
    int nr = eam_data_ti.nr;
    int ntypes = eam_data_ti.ntypes;
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

        // get the neighbor's position (MEM TRANSFER: 16 bytes)
        Scalar4 neigh_postype = texFetchScalar4(d_pos, pdata_pos_tex, cur_neigh);
        Scalar3 neigh_pos = make_scalar3(neigh_postype.x, neigh_postype.y, neigh_postype.z);

        // calculate dr (with periodic boundary conditions) (FLOPS: 3)
        Scalar3 dx = pos - neigh_pos;
        int typej  = __scalar_as_int(neigh_postype.w);
        // apply periodic boundary conditions: (FLOPS 12)
        dx = box.minImage(dx);

        // calculate r squard (FLOPS: 5)
        Scalar rsq = dot(dx, dx);;
        if (rsq < eam_data_ti.r_cutsq)
            {
            Scalar position_scalar = sqrtf(rsq) * eam_data_ti.rdr;
            atomElectronDensity += tex1D(electronDensity_tex, position_scalar + nr * (typei * ntypes + typej) + Scalar(0.5) ); //electronDensity[r_index + eam_data_ti.nr * typej] + derivativeElectronDensity[r_index + eam_data_ti.nr * typej] * position * eam_data_ti.dr;
            }
        }

    Scalar position = atomElectronDensity * eam_data_ti.rdrho;
    /*unsigned int r_index = (unsigned int)position;
    position -= (Scalar)r_index;*/
    atomDerivativeEmbeddingFunction[idx] = tex1D(derivativeEmbeddingFunction_tex, position + typei * eam_data_ti.nrho + Scalar(0.5));//derivativeEmbeddingFunction[r_index + typei * eam_data_ti.nrho];

    force.w += tex1D(embeddingFunction_tex, position + typei * eam_data_ti.nrho + Scalar(0.5));//embeddingFunction[r_index + typei * eam_data_ti.nrho] + derivativeEmbeddingFunction[r_index + typei * eam_data_ti.nrho] * position * eam_data_ti.drho;
    d_force[idx] = force;
    #endif
    }

//! Second stage kernel for computing EAM forces on the GPU
template<unsigned char use_gmem_nlist>
__global__ void gpu_compute_eam_tex_inter_forces_kernel_2(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    BoxDim box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    Scalar* atomDerivativeEmbeddingFunction)
    {
    #ifdef SINGLE_PRECISION
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
    // prefetch neighbor index
    Scalar position;
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
    int nr = eam_data_ti.nr;
    int ntypes = eam_data_ti.ntypes;
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

        // calculate r squard (FLOPS: 5)
        Scalar rsq = dot(dx, dx);

        if (rsq > eam_data_ti.r_cutsq) continue;

        Scalar inverseR = rsqrtf(rsq);
        Scalar r = Scalar(1.0) / inverseR;
        position = r * eam_data_ti.rdr;
        int shift = (typei>=typej)?(int)((2 * ntypes - typej -1)*typej/2 + typei) * nr:(int)((2 * ntypes - typei -1)*typei/2 + typej) * nr;
        Scalar2 pair_potential = tex1D(pairPotential_tex, position + shift + Scalar(0.5));
        Scalar pair_eng =  pair_potential.x * inverseR;

        Scalar derivativePhi = (pair_potential.y - pair_eng) * inverseR;

        Scalar derivativeRhoI = tex1D(derivativeElectronDensity_tex, position + typei * eam_data_ti.nr + Scalar(0.5));

        Scalar derivativeRhoJ = tex1D(derivativeElectronDensity_tex, position + typej * eam_data_ti.nr + Scalar(0.5));

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

    #endif
    }

cudaError_t gpu_compute_eam_tex_inter_forces(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    const BoxDim& box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const unsigned int *d_head_list,
    const unsigned int size_nlist,
    const EAMtex& eam_tex,
    const EAMTexInterArrays& eam_arrays,
    const EAMTexInterData& eam_data,
    const unsigned int compute_capability,
    const unsigned int max_tex1d_width)
    {
    cudaError_t error;
    if (compute_capability < 35 && size_nlist <= max_tex1d_width)
        {
        nlist_tex.normalized = false;
        nlist_tex.filterMode = cudaFilterModePoint;
        error = cudaBindTexture(0, nlist_tex, d_nlist, sizeof(unsigned int)*size_nlist);
        if (error != cudaSuccess)
            return error;
        }

    // bind the texture
    #ifdef SINGLE_PRECISION
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
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

    derivativeEmbeddingFunction_tex.normalized = false;
    derivativeEmbeddingFunction_tex.filterMode = cudaFilterModeLinear ;
    error = cudaBindTextureToArray(derivativeEmbeddingFunction_tex, eam_tex.derivativeEmbeddingFunction);
    if (error != cudaSuccess)
        return error;
    #endif
    // run the kernel
    cudaMemcpyToSymbol("eam_data_ti", &eam_data, sizeof(EAMTexInterData));

    if (compute_capability < 35 && size_nlist > max_tex1d_width)
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_eam_tex_inter_forces_kernel<1>);

            cudaFuncAttributes attr2;
            cudaFuncGetAttributes(&attr2, gpu_compute_eam_tex_inter_forces_kernel_2<1>);

            max_block_size = min(attr.maxThreadsPerBlock, attr2.maxThreadsPerBlock);
            }

        unsigned int run_block_size = min(eam_data.block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( (int)ceil((double)N / (double)run_block_size), 1, 1);
        dim3 threads(run_block_size, 1, 1);

        gpu_compute_eam_tex_inter_forces_kernel<1><<< grid, threads>>>(d_force,
                                                                    d_virial,
                                                                    virial_pitch,
                                                                    N,
                                                                    d_pos,
                                                                    box,
                                                                    d_n_neigh,
                                                                    d_nlist,
                                                                    d_head_list,
                                                                    eam_arrays.atomDerivativeEmbeddingFunction);

        gpu_compute_eam_tex_inter_forces_kernel_2<1><<< grid, threads>>>(d_force,
                                                                      d_virial,
                                                                      virial_pitch,
                                                                      N,
                                                                      d_pos,
                                                                      box,
                                                                      d_n_neigh,
                                                                      d_nlist,
                                                                      d_head_list,
                                                                      eam_arrays.atomDerivativeEmbeddingFunction);
        }
    else
        {
        static unsigned int max_block_size = UINT_MAX;
        if (max_block_size == UINT_MAX)
            {
            cudaFuncAttributes attr;
            cudaFuncGetAttributes(&attr, gpu_compute_eam_tex_inter_forces_kernel<0>);

            cudaFuncAttributes attr2;
            cudaFuncGetAttributes(&attr2, gpu_compute_eam_tex_inter_forces_kernel_2<0>);

            max_block_size = min(attr.maxThreadsPerBlock, attr2.maxThreadsPerBlock);
            }

        unsigned int run_block_size = min(eam_data.block_size, max_block_size);

        // setup the grid to run the kernel
        dim3 grid( (int)ceil((double)N / (double)run_block_size), 1, 1);
        dim3 threads(run_block_size, 1, 1);

        gpu_compute_eam_tex_inter_forces_kernel<0><<< grid, threads>>>(d_force,
                                                                    d_virial,
                                                                    virial_pitch,
                                                                    N,
                                                                    d_pos,
                                                                    box,
                                                                    d_n_neigh,
                                                                    d_nlist,
                                                                    d_head_list,
                                                                    eam_arrays.atomDerivativeEmbeddingFunction);

        gpu_compute_eam_tex_inter_forces_kernel_2<0><<< grid, threads>>>(d_force,
                                                                      d_virial,
                                                                      virial_pitch,
                                                                      N,
                                                                      d_pos,
                                                                      box,
                                                                      d_n_neigh,
                                                                      d_nlist,
                                                                      d_head_list,
                                                                      eam_arrays.atomDerivativeEmbeddingFunction);
        }

    return cudaSuccess;
    }

// vim:syntax=cpp
