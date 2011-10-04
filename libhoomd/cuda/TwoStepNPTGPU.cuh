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

// Maintainer: joaander

#ifndef __TWOSTEPNPTGPU_CUH__
#define __TWOSTEPNPTGPU_CUH__

#include <cuda_runtime.h>

#include "ParticleData.cuh"

/*! \file TwoStepNPTGPU.cuh
    \brief Declares GPU kernel code for NPT integration on the GPU. Used by TwoStepNPTGPU.
*/

//! Kernel driver for the the first step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_step_one(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             bool partial_scale,
                             float Xi,
                             float Eta,
                             float deltaT);

//! Kernel driver to scale the particles into a new box on the GPU
cudaError_t gpu_npt_boxscale(const gpu_pdata_arrays &pdata,
                             const gpu_boxsize& box,
                             bool partial_scale,
                             float Eta,
                             float deltaT);

//! Kernel driver for the the second step of the computation called by NPTUpdaterGPU
cudaError_t gpu_npt_step_two(const gpu_pdata_arrays &pdata,
                             unsigned int *d_group_members,
                             unsigned int group_size,
                             float4 *d_net_force,
                             float Xi,
                             float Eta,
                             float deltaT);

#endif

