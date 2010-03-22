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

// $Id$
// $URL$
// Maintainer: joaander

/*! \file TwoStepBDNVTGPU.cuh
    \brief Declares GPU kernel code for BDNVT integration on the GPU. Used by TwoStepBDNVTGPU.
*/

#include "ParticleData.cuh"

#ifndef __TWO_STEP_BDNVT_GPU_CUH__
#define __TWO_STEP_BDNVT_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_bdnvt_step_two()
struct bdnvt_step_two_args
    {
    float *d_gamma;         //!< Device array listing per-type gammas
    unsigned int n_types;   //!< Number of types in \a d_gamma
    bool gamma_diam;        //!< Set to true to use diameters as gammas
    float T;                //!< Current temperature
    unsigned int timestep;  //!< Current timestep
    unsigned int seed;      //!< User chosen random number seed
    };

//! Kernel driver for the second part of the BDNVT update called by TwoStepBDNVTGPU
cudaError_t gpu_bdnvt_step_two(const gpu_pdata_arrays &pdata,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               float4 *d_net_force,
                               const bdnvt_step_two_args& bdnvt_args,
                               float deltaT,
                               float D,
                               bool limit,
                               float limit_val,
                               float* d_sum_bdenergy,
                               float* d_partial_sum_bdenergy,
                               unsigned int block_size, 
                               unsigned int num_blocks);
                               
#endif //__TWO_STEP_BDNVT_GPU_CUH__

