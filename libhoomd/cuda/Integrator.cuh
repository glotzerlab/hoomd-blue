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
          t0(NULL), t1(NULL), t2(NULL), t3(NULL), t4(NULL), t5(NULL),
          v0(NULL), v1(NULL), v2(NULL), v3(NULL), v4(NULL), v5(NULL),
          vpitch0(0), vpitch1(0), vpitch2(0), vpitch3(0), vpitch4(0), vpitch5(0)
          {
          }
          
    Scalar4 *f0; //!< Pointer to force array 0
    Scalar4 *f1; //!< Pointer to force array 1
    Scalar4 *f2; //!< Pointer to force array 2
    Scalar4 *f3; //!< Pointer to force array 3
    Scalar4 *f4; //!< Pointer to force array 4
    Scalar4 *f5; //!< Pointer to force array 5
    
    Scalar4 *t0; //!< Pointer to torque array 0
    Scalar4 *t1; //!< Pointer to torque array 1
    Scalar4 *t2; //!< Pointer to torque array 2
    Scalar4 *t3; //!< Pointer to torque array 3
    Scalar4 *t4; //!< Pointer to torque array 4
    Scalar4 *t5; //!< Pointer to torque array 5

    Scalar *v0;  //!< Pointer to virial array 0
    Scalar *v1;  //!< Pointer to virial array 1
    Scalar *v2;  //!< Pointer to virial array 2
    Scalar *v3;  //!< Pointer to virial array 3
    Scalar *v4;  //!< Pointer to virial array 4
    Scalar *v5;  //!< Pointer to virial array 5

    unsigned int vpitch0; //!< Pitch of virial array 0
    unsigned int vpitch1; //!< Pitch of virial array 1
    unsigned int vpitch2; //!< Pitch of virial array 2
    unsigned int vpitch3; //!< Pitch of virial array 3
    unsigned int vpitch4; //!< Pitch of virial array 4
    unsigned int vpitch5; //!< Pitch of virial array 5
 };

//! Driver for gpu_integrator_sum_net_force_kernel()
cudaError_t gpu_integrator_sum_net_force(Scalar4 *d_net_force,
                                         Scalar *d_net_virial,
                                         const unsigned int virial_pitch,
                                         Scalar4 *d_net_torque,
                                         const gpu_force_list& force_list,
                                         unsigned int nparticles,
                                         bool clear,
                                         bool compute_virial);

#endif

