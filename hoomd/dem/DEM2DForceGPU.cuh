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

// Maintainer: mspells

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "DEMEvaluator.h"

/*! \file DEM2DForceGPU.cuh
  \brief Declares GPU kernel code for calculating the Lennard-Jones pair forces. Used by DEM2DForceComputeGPU.
*/

#ifndef __DEM2DFORCEGPU_CUH__
#define __DEM2DFORCEGPU_CUH__

#ifdef ENABLE_CUDA

//! Kernel driver that computes 2D DEM forces on the GPU for DEM2DForceComputeGPU
template<typename Real, typename Real2, typename Real4, typename Evaluator>
cudaError_t gpu_compute_dem2d_forces(
        Scalar4* d_force,
        Scalar4* d_torque,
        Scalar* d_virial,
        const unsigned int virial_pitch,
        const unsigned int N,
        const unsigned int n_ghosts,
        const Scalar4 *d_pos,
        const Scalar4 *d_quat,
        const Real2 *d_vertices,
        const unsigned int *d_num_shape_verts,
        const Scalar *d_diameter,
        const Scalar4 *d_velocity,
        const unsigned int vertexCount,
        const BoxDim& box,
        const unsigned int *d_n_neigh,
        const unsigned int *d_nlist,
        const unsigned int *d_head_list,
        const Evaluator evaluator,
        const Real r_cutsq,
        const unsigned int n_shapes,
        const unsigned int particlesPerBlock,
        const unsigned int maxVerts);

#endif

#endif
