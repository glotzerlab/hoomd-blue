/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// Maintainer: morozov

/**
powered by:
Moscow group.
*/

#include "EAMForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

/*! \file EAMForceGPU.cu
    \brief Defines GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
*/

//!< Texture for reading particle positions
scalar4_tex_t pdata_pos_tex;

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
extern "C" __global__ void gpu_compute_eam_tex_inter_forces_kernel(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    BoxDim box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const Index2D nli,
    Scalar* atomDerivativeEmbeddingFunction)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list (MEM_TRANSFER: 4 bytes)
    int n_neigh = d_n_neigh[idx];

    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

    // initialize the force to 0
    Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // prefetch neighbor index
    int cur_neigh = 0;
    int next_neigh = d_nlist[nli(idx, 0)];
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
        next_neigh = d_nlist[nli(idx, neigh_idx+1)];

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
            #ifdef SINGLE_PRECISION
            atomElectronDensity += tex1D(electronDensity_tex, position_scalar + nr * (typei * ntypes + typej) + Scalar(0.5) ); //electronDensity[r_index + eam_data_ti.nr * typej] + derivativeElectronDensity[r_index + eam_data_ti.nr * typej] * position * eam_data_ti.dr;
            #endif
            }
        }

    Scalar position = atomElectronDensity * eam_data_ti.rdrho;
    /*unsigned int r_index = (unsigned int)position;
    position -= (Scalar)r_index;*/
    #ifdef SINGLE_PRECISION
    atomDerivativeEmbeddingFunction[idx] = tex1D(derivativeEmbeddingFunction_tex, position + typei * eam_data_ti.nrho + Scalar(0.5));//derivativeEmbeddingFunction[r_index + typei * eam_data_ti.nrho];

    force.w += tex1D(embeddingFunction_tex, position + typei * eam_data_ti.nrho + Scalar(0.5));//embeddingFunction[r_index + typei * eam_data_ti.nrho] + derivativeEmbeddingFunction[r_index + typei * eam_data_ti.nrho] * position * eam_data_ti.drho;
    #endif
    d_force[idx] = force;
    }

//! Second stage kernel for computing EAM forces on the GPU
extern "C" __global__ void gpu_compute_eam_tex_inter_forces_kernel_2(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    BoxDim box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const Index2D nli,
    Scalar* atomDerivativeEmbeddingFunction)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // loadj in the length of the list (MEM_TRANSFER: 4 bytes)
    int n_neigh = d_n_neigh[idx];

    // read in the position of our particle. Texture reads of Scalar4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    Scalar4 postype = texFetchScalar4(d_pos, pdata_pos_tex, idx);
    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    int typei = __scalar_as_int(postype.w);
    // prefetch neighbor index
    Scalar position;
    int cur_neigh = 0;
    int next_neigh = d_nlist[nli(idx, 0)];
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
        next_neigh = d_nlist[nli(idx, neigh_idx+1)];

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
        #ifdef SINGLE_PRECISION
        Scalar2 pair_potential = tex1D(pairPotential_tex, position + shift + Scalar(0.5));
        #else
        Scalar2 pair_potential = make_scalar2(Scalar(0.0), Scalar(0.0));
        #endif
        Scalar pair_eng =  pair_potential.x * inverseR;

        Scalar derivativePhi = (pair_potential.y - pair_eng) * inverseR;

        #ifdef SINGLE_PRECISION
        Scalar derivativeRhoI = tex1D(derivativeElectronDensity_tex, position + typei * eam_data_ti.nr + Scalar(0.5));

        Scalar derivativeRhoJ = tex1D(derivativeElectronDensity_tex, position + typej * eam_data_ti.nr + Scalar(0.5));
        #else
        Scalar derivativeRhoI = Scalar(0.0);
        Scalar derivativeRhoJ = Scalar(0.0);
        #endif

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

cudaError_t gpu_compute_eam_tex_inter_forces(
    Scalar4* d_force,
    Scalar* d_virial,
    const unsigned int virial_pitch,
    const unsigned int N,
    const Scalar4 *d_pos,
    const BoxDim& box,
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const Index2D& nli,
    const EAMtex& eam_tex,
    const EAMTexInterArrays& eam_arrays,
    const EAMTexInterData& eam_data)
    {
    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)eam_data.block_size), 1, 1);
    dim3 threads(eam_data.block_size, 1, 1);

    // bind the texture
    #ifdef SINGLE_PRECISION
    pdata_pos_tex.normalized = false;
    pdata_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_pos_tex, d_pos, sizeof(Scalar4)*N);
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

    gpu_compute_eam_tex_inter_forces_kernel<<< grid, threads>>>(d_force,
                                                                d_virial,
                                                                virial_pitch,
                                                                N,
                                                                d_pos,
                                                                box,
                                                                d_n_neigh,
                                                                d_nlist,
                                                                nli,
                                                                eam_arrays.atomDerivativeEmbeddingFunction);

    gpu_compute_eam_tex_inter_forces_kernel_2<<< grid, threads>>>(d_force,
                                                                  d_virial,
                                                                  virial_pitch,
                                                                  N,
                                                                  d_pos,
                                                                  box,
                                                                  d_n_neigh,
                                                                  d_nlist,
                                                                  nli,
                                                                  eam_arrays.atomDerivativeEmbeddingFunction);

    return cudaSuccess;
    }

// vim:syntax=cpp
