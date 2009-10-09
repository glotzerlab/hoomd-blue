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
// Maintainer: ndtrung

/*! \file NVTRigidUpdaterGPU.cuh
    \brief Declares GPU kernel code for NVT rigid body integration on the GPU. Used by NVTUpdaterGPU.
*/

#include "RigidData.cuh"

#ifndef __NVTRIGIDUPDATER_CUH__
#define __NVTRIGIDUPDATER_CUH__

//! Stores intermediate values for NVT integration
/*! NVT integration (NVTRigidUpdaterGPU) requires summing up the kinetic energy of the system.
    gpu_nvt_rigid_data stores the needed auxiliary data structure needed to do the standard reduction
    sum.

    \ingroup gpu_data_structs
*/
struct gpu_nvt_rigid_data
    {
    unsigned int n_bodies;
    
    float  eta_dot_t0;
    float  eta_dot_r0;
    float4 *conjqm;
    
    float *partial_Ksum_t; //!< NBlocks elements, each is a partial sum of m*v^2
    float *partial_Ksum_r; //!< NBlocks elements, each is a partial sum of L*w^2
    float *Ksum_t;  //!< fully reduced Ksum_t on one GPU
    float *Ksum_r;  //!< fully reduced Ksum_r on one GPU
    };

//! Kernel driver for the first part of the NVT update called by NVTUpdaterGPU
cudaError_t gpu_nvt_rigid_body_pre_step(const gpu_pdata_arrays& pdata, const gpu_rigid_data_arrays& rigid_data,
                                        const gpu_boxsize &box, const gpu_nvt_rigid_data& nvt_rigid_data, float deltaT);

//! Kernel driver for the Ksum reduction final pass called by NVTUpdaterGPU
cudaError_t gpu_nvt_rigid_reduce_ksum(const gpu_nvt_rigid_data &d_nvt_rigid_data);

//! Kernel driver for the second part of the NVT update called by NVTUpdaterGPU
cudaError_t gpu_nvt_rigid_body_step(const gpu_pdata_arrays &pdata, const gpu_rigid_data_arrays& rigid_data, float4 **force_data_ptrs, int num_forces,
                                    const gpu_boxsize &box, const gpu_nvt_rigid_data& nvt_rigid_data, float deltaT);

#endif

