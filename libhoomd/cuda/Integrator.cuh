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

/*! \file Integrator.cuh
    \brief Declares methods and data structures used by the Integrator class on the GPU
*/

#ifndef __INTEGRATOR_CUH__
#define __INTEGRATOR_CUH__

#include "ParticleData.cuh"

//! struct to pack up several force and virial arrays for addition
/*! To keep the argument count down to gpu_integrator_sum_accel, up to 6 force/virial array pairs are packed up in this 
    struct for addition to the net force/virial in a single kernel call. If there is not a multiple of 5 forces to sum, 
    set some of the pointers to NULL and they will be ignored.
*/
struct gpu_force_list
    {
    //! Initializes to NULL
    gpu_force_list() 
        : f0(NULL), f1(NULL), f2(NULL), f3(NULL), f4(NULL), f5(NULL),
          v0(NULL), v1(NULL), v2(NULL), v3(NULL), v4(NULL), v5(NULL)
          {
          }
          
    float4 *f0; //!< Pointer to force array 0
    float4 *f1; //!< Pointer to force array 1
    float4 *f2; //!< Pointer to force array 2
    float4 *f3; //!< Pointer to force array 3
    float4 *f4; //!< Pointer to force array 4
    float4 *f5; //!< Pointer to force array 5
    float *v0;  //!< Pointer to virial array 0
    float *v1;  //!< Pointer to virial array 1
    float *v2;  //!< Pointer to virial array 2
    float *v3;  //!< Pointer to virial array 3
    float *v4;  //!< Pointer to virial array 4
    float *v5;  //!< Pointer to virial array 5
    };

//! Sums up the net force and virial on the GPU for Integrator
cudaError_t gpu_integrator_sum_net_force(float4 *d_net_force,
                                         float *d_net_virial,
                                         const gpu_force_list& force_list,
                                         unsigned int nparticles,
                                         bool clear);

#endif

