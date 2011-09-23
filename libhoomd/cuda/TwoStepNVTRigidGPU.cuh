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

/*! \file TwoStepNVTRigidGPU.cuh
    \brief Declares GPU kernel code for NVT rigid body integration on the GPU. Used by TwoStepNVTRigidGPU.
*/

#include "ParticleData.cuh"
#include "RigidData.cuh"
#include "TwoStepNVERigidGPU.cuh"

#ifndef __TWO_STEP_NVT_RIGID_CUH__
#define __TWO_STEP_NVT_RIGID_CUH__
    
/*! Thermostat data structure
*/
struct gpu_nvt_rigid_data
    {
    unsigned int n_bodies;  //!< Number of rigid bodies
    
    float  eta_dot_t0;      //!< Thermostat translational velocity
    float  eta_dot_r0;      //!< Thermostat rotational velocity
        
    float *partial_Ksum_t;  //!< NBlocks elements, each is a partial sum of m*v^2
    float *partial_Ksum_r;  //!< NBlocks elements, each is a partial sum of L*w^2
    float *Ksum_t;          //!< fully reduced Ksum_t on one GPU
    float *Ksum_r;          //!< fully reduced Ksum_r on one GPU
    };

//! Kernel driver for the first part of the NVT update called by TwoStepNVTRigidGPU
cudaError_t gpu_nvt_rigid_step_one(const gpu_pdata_arrays& pdata, 
                                        const gpu_rigid_data_arrays& rigid_data,
                                        unsigned int *d_group_members,
                                        unsigned int group_size,
                                        float4 *d_net_force,
                                        const gpu_boxsize &box, 
                                        const gpu_nvt_rigid_data &nvt_rdata,
                                        float deltaT);

//! Kernel driver for the Ksum reduction final pass called by TwoStepNVTRigidGPU
cudaError_t gpu_nvt_rigid_reduce_ksum(const gpu_nvt_rigid_data &nvt_rdata);

//! Kernel driver for the second part of the NVT update called by TwoStepNVTRigidGPU
cudaError_t gpu_nvt_rigid_step_two(const gpu_pdata_arrays &pdata, 
                                    const gpu_rigid_data_arrays& rigid_data,
                                    unsigned int *d_group_members,
                                    unsigned int group_size,
                                    float4 *d_net_force,
                                    float *d_net_virial,
                                    const gpu_boxsize &box, 
                                    const gpu_nvt_rigid_data &nvt_rdata,
                                    float deltaT);

#endif // __TWO_STEP_NVT_RIGID_CUH__

