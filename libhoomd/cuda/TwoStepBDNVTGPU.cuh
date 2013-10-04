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

/*! \file TwoStepBDNVTGPU.cuh
    \brief Declares GPU kernel code for BDNVT integration on the GPU. Used by TwoStepBDNVTGPU.
*/

#include "ParticleData.cuh"
#include "HOOMDMath.h"

#ifndef __TWO_STEP_BDNVT_GPU_CUH__
#define __TWO_STEP_BDNVT_GPU_CUH__

//! Temporary holder struct to limit the number of arguments passed to gpu_bdnvt_step_two()
struct bdnvt_step_two_args
    {
    Scalar *d_gamma;         //!< Device array listing per-type gammas
    unsigned int n_types;   //!< Number of types in \a d_gamma
    bool gamma_diam;        //!< Set to true to use diameters as gammas
    Scalar T;                //!< Current temperature
    unsigned int timestep;  //!< Current timestep
    unsigned int seed;      //!< User chosen random number seed
    Scalar *d_sum_bdenergy;   //!< Energy transfer sum from bd thermal reservoir
    Scalar *d_partial_sum_bdenergy;  //!< Array used for summation
    unsigned int block_size;  //!<  Block size
    unsigned int num_blocks;  //!<  Number of blocks
    bool tally;               //!< Set to true is bd thermal reservoir energy tally is to be performed
    };

//! Kernel driver for the second part of the BDNVT update called by TwoStepBDNVTGPU
cudaError_t gpu_bdnvt_step_two(const Scalar4 *d_pos,
                               Scalar4 *d_vel,
                               Scalar3 *d_accel,
                               const Scalar *d_diameter,
                               const unsigned int *d_tag,
                               unsigned int *d_group_members,
                               unsigned int group_size,
                               Scalar4 *d_net_force,
                               const bdnvt_step_two_args& bdnvt_args,
                               Scalar deltaT,
                               Scalar D,
                               bool limit,
                               Scalar limit_val);

#endif //__TWO_STEP_BDNVT_GPU_CUH__
