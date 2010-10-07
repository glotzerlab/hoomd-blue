/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: morozov

/**
powered by:
Moscow group.
*/

#include "ForceCompute.cuh"
#include "ParticleData.cuh"
#include "Index1D.h"

/*! \file EAMForceGPU.cuh
    \brief Declares GPU kernel code for calculating the eam forces. Used by EAMForceComputeGPU.
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
    const unsigned int *d_n_neigh,
    const unsigned int *d_nlist,
    const Index2D& nli,
    const EAMtex& eam_tex,
    const EAMTexInterArrays& eam_arrays,
    const EAMTexInterData& eam_data);

#endif

