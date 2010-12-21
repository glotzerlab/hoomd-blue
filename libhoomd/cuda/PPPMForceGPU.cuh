/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: PPPMForceGPU.cuh 2587 2010-01-08 17:02:54Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/cuda/PPPMForceGPU.cuh $
// Maintainer: joaander

#include <cufft.h>
#include "HOOMDMath.h"
#include "ForceCompute.cuh"
#include "ParticleData.cuh"
#include "Index1D.h"

/*! \file HarmonicBondForceGPU.cuh
    \brief Declares GPU kernel code for calculating the harmonic bond forces. Used by HarmonicBondForceComputeGPU.
*/

#ifndef __PPPMFORCEGPU_CUH__
#define __PPPMFORCEGPU_CUH__

//! Kernel driver that computes harmonic bond forces for HarmonicBondForceComputeGPU
    cudaError_t gpu_compute_pppm_forces(const gpu_force_data_arrays& force_data,
                                        const gpu_pdata_arrays &pdata,
                                        const gpu_boxsize &box,
                                        int Nx,
                                        int Ny,
                                        int Nz,
                                        int order,
                                        float *GPU_rho_coeff,
                                        cufftComplex *GPU_rho_real_space,
                                        cufftHandle plan,
                                        cufftComplex *GPU_E_x,
                                        cufftComplex *GPU_E_y,
                                        cufftComplex *GPU_E_z,
                                        float3 *GPU_k_vec,
                                        float *GPU_green_hat,
                                        float3 *E_field,
                                        int block_size);

float2 gpu_compute_pppm_thermo(int Nx,
                               int Ny,
                               int Nz,
                               cufftComplex *GPU_rho_real_space,
                               float3 *GPU_vg,
                               float *GPU_green_hat,
                               float2 *o_data,
                               float2 *i_data,
                               int block_size);

cudaError_t reset_kvec_green_hat(const gpu_boxsize &box,
                                 int Nx,
                                 int Ny,
                                 int Nz,
                                 int nbx,
                                 int nby,
                                 int nbz,
                                 int order,
                                 float m_kappa,
                                 float3 *kvec,
                                 float *green_hat,
                                 float3 *vg,
                                 float *gf_b,
                                 int block_size);

cudaError_t fix_exclusions(const gpu_force_data_arrays& force_data,
                           const gpu_pdata_arrays &pdata,
                           const gpu_boxsize &box,
                           const unsigned int *d_n_ex,
                           const unsigned int *d_exlist,
                           const Index2D nex,
                           float m_kappa,
                           int block_size);
#endif

