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

#ifndef _COMPUTE_THERMO_GPU_CUH_
#define _COMPUTE_THERMO_GPU_CUH_

#include <cuda_runtime.h>

#include "ParticleData.cuh"
#include "ComputeThermoTypes.h"
#include "HOOMDMath.h"

/*! \file ComputeThermoGPU.cuh
    \brief Kernel driver function declarations for ComputeThermoGPU
    */

//! Holder for arguments to gpu_compute_thermo
struct compute_thermo_args
    {
    Scalar4 *d_net_force;    //!< Net force / pe array to sum
    Scalar *d_net_virial;    //!< Net virial array to sum
    unsigned int virial_pitch; //!< Pitch of 2D net_virial array
    unsigned int ndof;      //!< Number of degrees of freedom for T calculation
    unsigned int D;         //!< Dimensionality of the system
    Scalar4 *d_scratch;      //!< n_blocks elements of scratch space for partial sums
    Scalar *d_scratch_pressure_tensor; //!< n_blocks*6 elements of scratch spaace for partial sums of the pressure tensor
    unsigned int block_size;    //!< Block size to execute on the GPU
    unsigned int n_blocks;      //!< Number of blocks to execute / n_blocks * block_size >= group_size
    Scalar external_virial_xx;  //!< xx component of the external virial
    Scalar external_virial_xy;  //!< xy component of the external virial
    Scalar external_virial_xz;  //!< xz component of the external virial
    Scalar external_virial_yy;  //!< yy component of the external virial
    Scalar external_virial_yz;  //!< yz component of the external virial
    Scalar external_virial_zz;  //!< zz component of the external virial
    };

//! Computes the thermodynamic properties for ComputeThermo
cudaError_t gpu_compute_thermo(Scalar *d_properties,
                               Scalar4 *d_vel,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               const BoxDim& box,
                               const compute_thermo_args& args,
                               const bool compute_pressure_tensor
                               );

#endif

